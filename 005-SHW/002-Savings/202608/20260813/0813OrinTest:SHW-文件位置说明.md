# Sentrix Home Web — 文件位置说明

> 生成时间:2026-08-13
> 目标主机:`orin@192.168.0.118`
> 解压来源:`sentrix-home-web-main.zip`(实际未加密,`unzip` 直接解出)

## 0. 文件在哪儿(一句话)

完整内容位于远程主机:

```
orin@192.168.0.118:~/sentrix-home-web-main/sentrix-home-web-main/
```

⚠️ **注意双层嵌套**:压缩包内自带一层 `sentrix-home-web-main/`,解压时又套了一层同名目录,所以真实根目录是上面那个**两层**路径。若想拍平成单层 `~/sentrix-home-web-main/`,见文末「附:拍平双层目录」。

总大小约 **5.6 MB**(纯代码 + 文档 + 配置,不含媒体数据)。

---

## 1. 顶层目录结构

```
~/sentrix-home-web-main/sentrix-home-web-main/
├── backend/        # FastAPI 后端 + SQLite 记忆库 + 媒体流水线(2.3M,核心)
├── src/            # 浏览器前端代码与样式(284K)
├── services/       # 独立推理服务:e2b_server / vllm_manager(100K)
├── scripts/        # 运行/维护/基准/夹具脚本(908K)
├── docs/           # 项目记忆 + 设计/实现记录 + 基准报告(2.0M)
├── configs/        # 检索、vLLM 注册表、工具能力矩阵等 JSON 配置(32K)
├── test/           # Node 前端与仓库结构回归测试(36K)
├── index.html      # 前端入口(16 行,挂载点)
├── server.js       # Web 网关 / 代理(107 行)
├── package.json    # 前端依赖与脚本
├── package-lock.json
├── start.sh        # 一键启动脚本(169 行)
├── clear.sh        # 清理脚本(158 行)
├── requirements-text.txt
├── README.md       # 项目说明
└── .gitignore
```

---

## 2. 各目录详解

### 2.1 `backend/` — 后端核心(2.3M)

FastAPI 应用、SQLite 记忆库、媒体处理流水线、模型适配器、身份聚类与 Python 回归测试。

**关键入口**
- `backend/app.py` — FastAPI 应用入口(`backend.app:app`)
- `backend/pipeline.py` — 媒体处理流水线
- `backend/router.py` / `backend/routing_rules.py` — 请求路由
- `backend/db.py` — SQLite 存储层
- `backend/agent.py` / `backend/thin_agent.py` — Agent 主/精简实现

**子包**
| 子目录 | 作用 |
|---|---|
| `backend/agent_runtime/` | Agent 运行时:预算、裁判、守卫、工具注册、结果集等(`runtime.py`、`judge.py`、`budget_manager.py`、`final_guard.py`…) |
| `backend/embeddings/` | 多模态嵌入:文本(BGE/CLIP-text)、视觉(CLIP/Chinese-CLIP),`router.py` 负责分发 |
| `backend/retrieval/` | 检索引擎:词法、ANN(文本/视觉)、邻接、实体、融合、排序、近重复 |
| `backend/validation/` | 校验:断言、全链路 profile、模型调用账本 |

**其它后端模块**(部分)
记忆相关:`core_memory.py`、`structured_memory.py`、`advanced_memory_tools.py`、`memory_gate.py`、`memory_corrections.py`、`narrative_context.py`
回答生成:`answer_brief.py`、`answer_composer.py`、`complex_answer.py`、`response_plan.py`、`response_writer.py`、`response_validator.py`
身份/视觉:`face_clustering.py`、`face_embeddings.py`、`person_appearance.py`、`image_io.py`
检索/证据:`evidence_retrieval.py`、`retrieval_indexes.py`、`retrieval_strategy.py`
模型:`model_clients.py`、`model_routing.py`、`hardware.py`

**测试**:`backend/tests/` 约 90 个 `test_*.py`,覆盖 Agent、检索、嵌入、记忆、路由、各 Phase(R/R9)、faithfulness 等。运行:
```bash
.venv/bin/python -m unittest discover -s backend/tests -v
```

依赖见 `backend/requirements.txt`。

### 2.2 `src/` — 前端(284K)

浏览器端代码。
- `src/app.js` — 主应用逻辑
- `src/api.js` — 与后端通信
- `src/normalizers.js` — 数据归一化
- `src/image-metadata.js` — 图片元数据处理
- `src/styles.css` — 样式

前端测试在仓库根 `test/`,运行 `npm test`。

### 2.3 `services/` — 独立推理服务(100K)

- `services/e2b_server/` — E2B 模型服务(`app.py`、`model.py`、`ollama_shape.py`),自带 README 与 tests
- `services/vllm_manager/` — vLLM 管理服务(`app.py`、`manager.py`),自带 README

### 2.4 `scripts/` — 脚本(908K)

| 子目录 | 作用 | 代表脚本 |
|---|---|---|
| `scripts/runtime/` | 启动各端口实例(8091~8097、judge、face smoke、e2b、ollama) | `start_sentrix_api.sh`、`start_sentrix_ollama.sh`、`start_sentrix_e2b.sh` 等 |
| `scripts/maintenance/` | 破坏性/长耗时维护:重建记忆、回填 GPS/封面/场景、ANN 重建、模型健康探测 | `rebuild_memory.py`、`rebuild_ann_indices.py`、`backfill_gps.py`、`probe_model_health.py` |
| `scripts/benchmarks/` | 受控评测与基准(约 70 个脚本 + JSON 用例) | `evaluate_*.py`、`run_qa_benchmark.sh`、`measure_latency.py` |
| `scripts/fixtures/` | 可复现公开测试数据与元数据生成 | `build_virtual_family_album.py`、`download_test_data.py` |

⚠️ maintenance 类命令会**替换派生记忆数据**,执行前请确认(README 中明确标注「intentionally explicit」)。

### 2.5 `docs/` — 文档与基准(2.0M)

- `docs/PROJECT_MEMORY.md` — **产品定义 / 架构 / 数据契约 / 验收结果 / 当前工作队列的权威记录**(最重要的总纲)
- `docs/baseline/` — 基准与阶段报告(约 90 个文件):`runtime-v2-phase*`、`thin-agent-phase-R*`、`sentrix-12b-*`、`retrieval_R7/R8_*` 等
- `docs/phaseb/` — Phase B 制品:agent profile、structured memory coverage、tool readiness matrix
- `docs/plans/` — 路线规划(digital-memory-steward、semantic-entity-roadmap)
- `docs/superpowers/plans/` 与 `docs/superpowers/specs/` — 按日期组织的设计/实现计划与规格(2026-07 ~ 2026-08)
- `docs/IMAGE-TO-MEMORY-REPORT.md` — 图像转记忆报告

### 2.6 `configs/`(32K)

- `configs/retrieval/defaults.json` — 检索默认参数
- `configs/sentrix_vllm_registry_192_168_0_153.json` / `..._secondary_192_168_0_153.json` — vLLM 模型注册表(指向 153)
- `configs/tool_capability_matrix.json` — 工具能力矩阵

### 2.7 `test/`(36K)

Node 端前端与仓库结构回归测试:`image-metadata`、`normalizers`、`no-demo-data`、`phase-c-agent-ux`、`project-structure`。运行 `npm test`。

---

## 3. 运行入口与端口

按 README,部署目标是 **153**(`/home/asus/Github/Sentrix-Home-Web`),Web 界面 `http://192.168.0.153:4174`。当前文件解压在 **118(Orin)** 上,如需在 118 上运行,需相应调整模型路径与端口。

**启动后端(FastAPI)**
```bash
scripts/runtime/start_sentrix_ollama.sh   # 或其它 start_sentrix_api_*.sh
.venv/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8090
```

**启动前端网关**
```bash
SENTRIX_BACKEND_URL=http://127.0.0.1:8090 PORT=4174 npm run dev
```

**端口对照**(来自 `scripts/runtime/`)
| 脚本 | 端口 | 用途 |
|---|---|---|
| `start_sentrix_api.sh` | 8090(默认) | 主 API |
| `start_sentrix_api_8091.sh` | 8091 | API 实例 |
| `start_sentrix_api_8092_rx.sh` / `_validation.sh` | 8092 | RX / 校验 |
| `start_sentrix_api_8094_parser_timeout.sh` | 8094 | parser 超时场景 |
| `start_sentrix_api_8095_model_mismatch.sh` | 8095 | 模型不匹配场景 |
| `start_sentrix_api_8096_ollama.sh` | 8096 | Ollama 后端 |
| `start_sentrix_api_8097_phaseb.sh` | 8097 | Phase B |
| `start_sentrix_e2b.sh` | — | E2B 服务 |
| `start_sentrix_face_smoke_11003.sh` | 11003 | 人脸冒烟测试 |
| `start_judge_100.sh` | — | LLM judge |

---

## 4. 常用命令速查

```bash
# 进入项目(注意双层路径)
cd ~/sentrix-home-web-main/sentrix-home-web-main

# 后端测试
.venv/bin/python -m unittest discover -s backend/tests -v

# 前端测试
npm test

# 语法检查
node --check src/app.js
node --check src/api.js
.venv/bin/python -m compileall -q backend scripts

# 重建派生记忆(破坏性,需指定源相册)
.venv/bin/python scripts/maintenance/rebuild_memory.py --root . --source /path/to/source-album
```

---

## 5. 注意事项

1. **双层目录**:`~/sentrix-home-web-main/sentrix-home-web-main/`(见文末拍平方法)。
2. **原 zip 未加密**:解压不需要密码(用任意 `-P` 都能成功)。
3. **Ollama 端口归属**:Sentrix 占用 `11435`;`11434` 为其它项目共享,**不可停止或改动**。
4. **maintenance 脚本**会覆盖派生记忆数据,执行前务必确认。
5. README 中的模型路径(AdaFace、face-models、数据目录)是针对 **153** 的环境,在 118 上运行需自行替换。

---

## 附:拍平双层目录

如需把内容上移成单层 `~/sentrix-home-web-main/`:

```bash
ssh orin@192.168.0.118 'cd ~ && \
  mv sentrix-home-web-main/sentrix-home-web-main sentrix-home-web-main_inner && \
  rmdir sentrix-home-web-main && \
  mv sentrix-home-web-main_inner sentrix-home-web-main'
```

执行后项目根目录变为单层:`~/sentrix-home-web-main/`。
