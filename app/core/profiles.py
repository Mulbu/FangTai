"""用户健康档案加载。"""
from __future__ import annotations

import json
from typing import Optional

from app import config

_cache: dict[str, dict[int, dict]] = {}


def load_profiles(kind: str = "detailed") -> dict[int, dict]:
    """kind: detailed(详细版含体检指标) | simple(脱敏版)"""
    if kind in _cache:
        return _cache[kind]
    name = "50个用户健康档案_详细版7.13.json" if kind == "detailed" else "50个用户健康档案（脱敏）.json"
    with open(config.DATA_DIR / name, encoding="utf-8") as f:
        rows = json.load(f)
    _cache[kind] = {int(r["id"]): r for r in rows}
    return _cache[kind]


def get_profile(pid: int, kind: str = "detailed") -> Optional[dict]:
    return load_profiles(kind).get(pid)


def profile_text(pid: int, kind: str = "detailed") -> str:
    """档案的自然语言描述，供 prompt 注入。"""
    p = get_profile(pid, kind)
    if not p:
        return ""
    lines = [
        f"用户{p['id']}：{p['性别']}，{p['年龄']}岁，劳动强度{p['劳动强度']}",
    ]
    if p.get("身高_cm"):
        lines[0] += f"，身高{p['身高_cm']}cm，体重{p['体重_kg']}kg，BMI {p['BMI']}"
    crowd = p.get("特殊人群") or []
    if crowd:
        extra = f"，孕{p['孕周期']}" if p.get("孕周期") else ""
        lines.append(f"特殊人群/慢病：{'、'.join(crowd)}{extra}")
    if p.get("口味偏好"):
        lines.append(f"口味偏好：{p['口味偏好']}")
    if p.get("过敏食材"):
        lines.append(f"过敏食材：{'、'.join(p['过敏食材'])}")
    if p.get("健康需求"):
        lines.append(f"健康需求：{'、'.join(p['健康需求'])}")
    m = p.get("体检指标") or {}
    if m:
        lines.append("体检指标：" + "，".join(f"{k}={v}" for k, v in m.items()))
    return "\n".join(lines)
