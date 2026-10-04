# AGENTS.md

方太个性化膳食规划 Agent（浙江省 AI 竞赛"方太"专项赛）：FastAPI + LLM Agent（GLM）+ 云端 RAG（阿里云百炼）的膳食推荐系统。代码注释、文档、数据均为中文。

## 目录

- `app/config.py` — 全部配置，仅来自环境变量（**无 .env 文件**，docker-compose 注入）
- `app/main.py` — FastAPI 入口，lifespan 内预热菜谱库/索引
- `app/api/routes.py` — REST/SSE 路由（`/api/*`），会话存于进程内 `_sessions` 字典
- `app/core/` — 核心逻辑：`agent.py`（编排）、`constraint_engine.py`、`retriever.py`（BM25+向量混合检索）、`recipe_store.py`、`nutrition.py`、`planner.py`、`dialog.py`、`llm.py`、`profiles.py`
- `app/web/index.html` — 单文件聊天界面
- `data/` — 运行数据副本；`资料/` — 赛题细则（docx）与原始数据（只读，勿改）
- `docs/` — 技术方案、部署文档；改检索/约束/编排前先读 `docs/技术方案文档.md`

## 常用命令（Linux；仓库注释里的 `py -3`/`set` 是 Windows 写法）

```bash
export LLM_API_KEY=... DASHSCOPE_API_KEY=sk-...   # 运行前必须
python -m uvicorn app.main:app --port 8000        # 本地起服务
docker compose up -d --build                      # Docker 一键部署

python -m pytest tests/ -v                        # 单元测试（也可直接 python tests/test_core.py）
python scripts/eval.py --api http://127.0.0.1:8000    # 20 组对话用例全量评测
python scripts/score_test.py --api http://127.0.0.1:8000  # 赛题评分表自测（100 分制）
bash scripts/test_deepseek.sh                     # DeepSeek(deepseek-flash) 全量评测，数据取自 资料/（密钥经环境变量或 scripts/.keys.env 注入，该文件已 gitignore）
python scripts/dev_run.py                         # 进程内调试单条对话（不走 HTTP）
python scripts/build_index.py                     # 离线重建向量索引（调云端 API 约 200 次）
```

改动 `app/core/` 后必须跑单元测试 + 评测脚本验证；`eval_report.json`/`score_report.json` 是已提交的评测产物。

## 架构规则

- 数据流：意图/槽位抽取 → 约束过滤 → 混合检索 → LLM 选菜 → 组合校验 → 流式生成，全在 `app/core/agent.py` 编排，勿把业务逻辑写进 routes。
- SSE 事件协议：`intent / plan / delta / clarify / done`，`plan` 事件携带结构化方案供评测机判，改字段名会破坏 `scripts/eval.py` 与 `score_test.py`。
- **零幻觉**：推荐菜名必须来自菜谱库（`store.exists()` 校验）；**零违反**：过敏/忌口经约束引擎双遍校验。这两条是赛题硬性指标，不得放宽。
- 核心模块均为模块级单例工厂：`get_store() / get_retriever() / get_llm() / get_agent()`，直接复用，勿自建实例。

## 注意事项

- 菜谱 CSV 必须以 `gb18030` 读取（`recipe_store.py`），用户档案/对话用例为 UTF-8。
- RAG 默认云端 dashscope 模式（text-embedding-v4 + gte-rerank-v2）；未配置 `DASHSCOPE_API_KEY` 时检索**直接报错**（不降级）；仅运行期云 API 异常才降级纯 BM25。本地 GPU 模式（`RAG_PROVIDER=local`）需 `requirements-local.txt`，见部署文档。
- `numpy` 必须 <2.0.0（BM25 兼容）。
- `LLM_THINKING=high` 质量优先但性能分 0/30；调性能问题先看该配置。
- 首次启动会调云端 API 构建索引（约 300s），Docker healthcheck 已设 `start_period: 300s`；索引缓存在 `index-cache` 卷。
