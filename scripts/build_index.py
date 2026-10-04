"""离线构建菜谱向量索引。

- dashscope（默认）：阿里云百炼 text-embedding-v4（10条/批，约200次调用）
- local：本地 GPU sentence-transformers
用法：
  py -3 scripts/build_index.py                       # 按 RAG_PROVIDER 环境变量
  set DASHSCOPE_API_KEY=sk-xxx && py -3 scripts/build_index.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")


def main():
    t0 = time.perf_counter()
    from app import config
    from app.core.recipe_store import get_store

    store = get_store()
    print(f"菜谱 {len(store.recipes)} 条载入完成")

    texts = []
    for r in store.recipes:
        texts.append(
            r.name + "。食材：" + "、".join(dict.fromkeys(i.name for i in r.ingredients))
            + "。标签：" + "、".join(r.tag_list)
        )

    if config.RAG_PROVIDER == "dashscope":
        from app.core.retriever import DashscopeEmbedder
        emb = DashscopeEmbedder()
        print(f"嵌入模型 {emb.index_key}（阿里云百炼 API，{emb.BATCH} 条/批）")
    else:
        from app.core.retriever import LocalEmbedder
        emb = LocalEmbedder()
        print(f"嵌入模型 {emb.index_key}（本地推理）")

    vecs = emb.encode(texts)

    import numpy as np
    import json
    vecs = np.asarray(vecs, dtype=np.float32)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.INDEX_DIR / "embeddings.npy", vecs)
    (config.INDEX_DIR / "meta.json").write_text(
        json.dumps({"index_key": emb.index_key, "count": len(texts)}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"完成: {vecs.shape} -> {config.INDEX_DIR / 'embeddings.npy'}，"
          f"耗时 {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
