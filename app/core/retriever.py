"""混合检索器：BM25(jieba) + GPU 向量召回 + RRF 融合 + GPU 重排 + 约束过滤。

性能设计（评分项④）：
- 菜谱向量离线预计算（scripts/build_index.py），启动时整体载入显存/内存
- 查询向量实时编码（GPU，单条 <20ms），BM25 内存索引零延迟
- 重排器常驻显存，仅对 top 候选打分
- 全链路无网络请求，目标 <300ms
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass

import numpy as np

from app import config
from app.core.recipe_store import Recipe, get_store

import jieba
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text.lower()) if t.strip()]


def tags_pool(r: Recipe) -> str:
    return " ".join(r.tag_list)


@dataclass
class RetrievedRecipe:
    recipe: Recipe
    score: float
    bm25_rank: int = -1
    vec_rank: int = -1
    rerank_score: float | None = None


def _pick_device(pref: str) -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class RecipeRetriever:
    def __init__(self):
        self.store = get_store()
        self._embedder = None
        self._reranker = None
        self._embed_device = _pick_device(config.EMBEDDING_DEVICE)
        self._rerank_device = _pick_device(config.RERANKER_DEVICE)
        self._lock = threading.Lock()
        self.last_latency_ms = 0.0
        self._init_texts()
        self._init_indexes()

    # ---------- 初始化 ----------
    def _init_texts(self):
        """缓存检索/重排两套文本（重排用截断版降低 GPU 耗时）。"""
        self.doc_texts: list[str] = []
        self.rerank_texts: list[str] = []
        for r in self.store.recipes:
            ings = "、".join(dict.fromkeys(i.name for i in r.ingredients))
            tags = "、".join(r.tag_list)
            full = f"{r.name}。食材：{ings}。标签：{tags}"
            self.doc_texts.append(full)
            short_ings = "、".join(dict.fromkeys(i.name for i in r.ingredients[:10]))
            self.rerank_texts.append(f"{r.name} 食材:{short_ings} 标签:{tags[:40]}")
        self.corpus_tokens = []
        for r in self.store.recipes:
            text = r.name + " " + " ".join(i.name for i in r.ingredients) \
                + " " + tags_pool(r) + " " + "".join(r.steps)[:120]
            self.corpus_tokens.append(_tokenize(text))

    def _init_indexes(self):
        t0 = time.perf_counter()
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.doc_vecs = self._load_or_build_vectors()
        self._embed_ready = self.doc_vecs is not None
        print(f"[retriever] 索引就绪: {len(self.corpus_tokens)} 菜谱, "
              f"embed_device={self._embed_device}, {time.perf_counter()-t0:.1f}s")

    def _embed_model(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                config.EMBEDDING_MODEL, device=self._embed_device
            )
        return self._embedder

    def _query_instruction(self) -> str:
        name = config.EMBEDDING_MODEL.lower()
        # bge-*-zh 系列检索需要指令前缀；bge-m3 无需
        if "zh" in name and "m3" not in name:
            return config.EMBED_QUERY_INSTRUCTION
        return ""

    def _load_or_build_vectors(self) -> np.ndarray | None:
        index_file = config.INDEX_DIR / "embeddings.npy"
        meta_file = config.INDEX_DIR / "meta.json"
        try:
            if index_file.exists() and meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if meta.get("model") == config.EMBEDDING_MODEL and meta.get("count") == len(self.store.recipes):
                    vecs = np.load(index_file)
                    print(f"[retriever] 载入预计算向量 {vecs.shape}")
                    return vecs
        except Exception as e:
            print(f"[retriever] 向量索引载入失败，将重建: {e}")
        # 构建
        try:
            model = self._embed_model()
            vecs = model.encode(
                self.doc_texts, batch_size=64, normalize_embeddings=True,
                show_progress_bar=False, convert_to_numpy=True,
            ).astype(np.float32)
            config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
            np.save(index_file, vecs)
            meta_file.write_text(
                json.dumps({"model": config.EMBEDDING_MODEL, "count": len(self.doc_texts)}, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[retriever] 已构建并缓存向量索引 {vecs.shape} -> {index_file}")
            return vecs
        except Exception as e:
            print(f"[retriever] 向量编码不可用（{e}），退化为纯 BM25")
            return None

    def _rerank_model(self):
        if self._reranker is None and config.RERANKER_MODEL:
            try:
                from sentence_transformers import CrossEncoder
                ce = CrossEncoder(
                    config.RERANKER_MODEL, device=self._rerank_device,
                    max_length=256,
                )
                if self._rerank_device == "cuda":
                    try:
                        ce.model.half()  # fp16 提速，A100/消费卡均安全
                    except Exception:
                        pass
                self._reranker = ce
            except Exception as e:
                print(f"[retriever] 重排器加载失败（跳过重排）: {e}")
                self._reranker = False  # 标记不可用
        return self._reranker or None

    # ---------- 检索 ----------
    def encode_query(self, query: str) -> np.ndarray | None:
        if not self._embed_ready:
            return None
        with self._lock:
            model = self._embed_model()
            text = self._query_instruction() + query
            vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return vec.astype(np.float32)

    def search(
        self,
        query: str,
        ok_ids: set[int] | None = None,
        label_filter: dict[str, list[str]] | None = None,
        keyword_boost: list[str] | None = None,
        topk: int | None = None,
    ) -> list[RetrievedRecipe]:
        """混合检索主入口。

        query: 自然语言需求
        ok_ids: 约束引擎允许的菜谱 id（过敏/忌口已过滤）
        label_filter: 标签过滤，如 {"餐次": ["晚餐"], "口味": ["清淡"]}
        keyword_boost: 软偏好关键词（健康需求食材），命中加权
        """
        t0 = time.perf_counter()
        topk = topk or config.RERANK_TOPK
        n = len(self.store.recipes)

        # 标签过滤
        allowed = ok_ids if ok_ids is not None else set(self.store.by_id.keys())
        if label_filter:
            for dim, want in label_filter.items():
                if not want:
                    continue
                allowed = {
                    rid for rid in allowed
                    if any(w in self.store.by_id[rid].tags.get(dim, []) for w in want)
                }
        if not allowed:
            return []

        idx_map = [r.id for r in self.store.recipes]
        pos_of = {rid: i for i, rid in enumerate(idx_map)}

        # BM25
        q_tokens = _tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens) if q_tokens else np.zeros(n)
        bm25_order = np.argsort(-bm25_scores)

        # 向量
        vec_scores = None
        qvec = self.encode_query(query)
        if qvec is not None:
            vec_scores = self.doc_vecs @ qvec.T
            vec_scores = vec_scores.reshape(-1)
        vec_order = np.argsort(-vec_scores) if vec_scores is not None else None

        # RRF 融合
        K = 60.0
        rrf = np.zeros(n)
        for rank, pos in enumerate(bm25_order[: config.RECALL_TOPK * 3]):
            rrf[pos] += 1.0 / (K + rank + 1)
        if vec_order is not None:
            for rank, pos in enumerate(vec_order[: config.RECALL_TOPK * 3]):
                rrf[pos] += 1.0 / (K + rank + 1)

        # 软偏好加权：关键词命中食材/菜名，功效标签命中（如减脂/助眠）
        if keyword_boost:
            for pos, rid in enumerate(idx_map):
                if rid not in allowed:
                    continue
                r = self.store.recipes[pos]
                text = r.name + "".join(i.name for i in r.ingredients)
                hits = sum(1 for kw in keyword_boost if kw and kw in text)
                tag_str = "、".join(r.tag_list)
                hits += sum(1 for kw in keyword_boost if kw and kw in tag_str)
                rrf[pos] += 0.02 * hits

        # 过滤 + 截断到重排候选
        recall_n = config.RECALL_TOPK
        cand_pos = [pos for pos in np.argsort(-rrf) if idx_map[pos] in allowed][:recall_n]

        results = [
            RetrievedRecipe(
                recipe=self.store.recipes[pos],
                score=float(rrf[pos]),
                bm25_rank=int(np.where(bm25_order == pos)[0][0]) + 1 if pos in bm25_order[:500] else -1,
                vec_rank=int(np.where(vec_order == pos)[0][0]) + 1 if vec_order is not None and pos in vec_order[:500] else -1,
            )
            for pos in cand_pos
        ]

        # GPU 重排（截断文本 + 小批次，控制延迟）
        reranker = self._rerank_model()
        if reranker and results:
            try:
                rerank_n = min(24, len(results))
                sub = results[:rerank_n]
                pairs = [[query, self.rerank_texts[pos_of[r.recipe.id]]] for r in sub]
                with self._lock:
                    scores = reranker.predict(pairs, show_progress_bar=False, batch_size=24)
                for r, s in zip(sub, scores):
                    r.rerank_score = float(s)
                sub.sort(key=lambda x: x.rerank_score or 0, reverse=True)
                results = sub + results[rerank_n:]
            except Exception as e:
                print(f"[retriever] 重排失败，使用 RRF 序: {e}")

        elapsed = (time.perf_counter() - t0) * 1000
        self.last_latency_ms = elapsed
        return results[:topk]


_retriever: RecipeRetriever | None = None


def get_retriever() -> RecipeRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RecipeRetriever()
    return _retriever
