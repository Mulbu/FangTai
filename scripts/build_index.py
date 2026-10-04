"""离线构建菜谱向量索引（GPU 批量编码，结果缓存在 index/ 目录）。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.core.recipe_store import get_store  # noqa: E402


def main():
    t0 = time.perf_counter()
    store = get_store()
    print(f"菜谱 {len(store.recipes)} 条载入完成")
    from sentence_transformers import SentenceTransformer
    device = "cuda" if _cuda_ok() else "cpu"
    print(f"加载嵌入模型 {config.EMBEDDING_MODEL} → {device}")
    model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    texts = []
    for r in store.recipes:
        texts.append(
            r.name + "。食材：" + "、".join(dict.fromkeys(i.name for i in r.ingredients))
            + "。标签：" + "、".join(r.tag_list)
        )
    vecs = model.encode(
        texts, batch_size=64, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype("float32")
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import json
    np.save(config.INDEX_DIR / "embeddings.npy", vecs)
    (config.INDEX_DIR / "meta.json").write_text(
        json.dumps({"model": config.EMBEDDING_MODEL, "count": len(texts)}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"完成: {vecs.shape} → {config.INDEX_DIR / 'embeddings.npy'}，耗时 {time.perf_counter()-t0:.1f}s")


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
