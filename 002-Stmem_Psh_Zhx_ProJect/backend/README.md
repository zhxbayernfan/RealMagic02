# LingBot-MAP 空间记忆 — 后端模块

## 系统要求

- Python >= 3.10
- CUDA >= 12.0（或 CPU 推理，慢）
- 操作系统：Ubuntu 22.04+ arm64/x86_64
- Git、wget/curl
- 网络：HuggingFace 镜像访问（直连不可用时需配置 HF_ENDPOINT）

## 网络准备

在部分网络环境（如中国大陆）中，GitHub 和部分模型 CDN 可能无法直连。

### GitHub 不可达时的替代方案

```bash
# 方案 A：使用 SSH clone（需先在 GitHub 配置 SSH key）
git clone git@github.com:ysh12304124/stmem.git -b psh /home/orin/lingbot-map-stmem

# 方案 B：配置 Git 代理
git config --global http.proxy http://your-proxy:port

# 方案 C：从可访问的机器 scp 整个仓库
# 参考后续「离线部署」章节
```

### HuggingFace 直连不可达

```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com
```

## 快速部署

### 1. 克隆仓库

```bash
git clone https://github.com/ysh12304124/stmem.git -b psh /home/orin/lingbot-map-stmem
cd /home/orin/lingbot-map-stmem
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，设置 STMEM_HOME 为你的部署路径
```

```bash
export STMEM_HOME=/home/orin/lingbot-map-stmem
# 其余路径自动推导，也可独立覆盖
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. 创建/复用 conda 环境

**方案 A：复用已有 dgsg 环境（推荐）**

```bash
conda activate dgsg
pip install einops  # 唯一缺失的依赖
```

**方案 B：从零创建**

```bash
conda create -n lingbot-map python=3.10 -y
conda activate lingbot-map
pip install -r backend/requirements.txt
```

### 3.5 安装 lingbot_map Python 包

`batch_service.py` 依赖 `lingbot_map` 包（lingbot-map 模型推理库）。需要从源码安装：

```bash
# 克隆 lingbot-map 开源仓库
git clone https://github.com/prclibo/lingbot-map /tmp/lingbot-map-src
# 或从 HuggingFace/ModelScope 下载

# 安装为可编辑包
pip install -e /tmp/lingbot-map-src

# 验证
python3 -c "import lingbot_map; print(lingbot_map.__file__)"
```

> **注意**：`lingbot_map` 包与 `lingbot-map-long.pt` 模型权重是两个独立组件，都需要安装。

### 4. 下载模型文件

#### lingbot-map 权重（4.4GB）

```bash
mkdir -p checkpoints
# 方式 A: HuggingFace
huggingface-cli download prclibo/lingbot-map lingbot-map-long.pt --local-dir checkpoints/
# 方式 B: ModelScope（国内）
pip install modelscope
modelscope download --model prclibo/lingbot-map --local_dir checkpoints/
```

#### DGSG 模型

```bash
mkdir -p models

# YOLOE (76MB + 78MB) — ultralytics 首次运行时自动下载
# 也可手动下载放到 models/

# SAM2 (155MB) — Facebook CDN 在国内可能不可达，优先用 HF 镜像
# huggingface-cli download facebook/sam2.1-b --local-dir models/
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_b.pt -P models/ 2>/dev/null || \
  wget https://hf-mirror.com/facebook/sam2.1-b/resolve/main/sam2.1_b.pt -P models/

# OpenCLIP (578MB)
huggingface-cli download laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K --local-dir models/

# Moondream2 (7.2GB)
huggingface-cli download vikhyatk/moondream2 --local-dir models/moondream2

# YOLOv8s-world (26MB) — ultralytics 首次运行时自动下载
```

#### DAv2 模型（约 1.3GB）

首次运行 `scale_calibrate.py` 时自动通过 HuggingFace 下载，无需手动操作。

### 5. 启动服务

```bash
conda activate dgsg
cd /home/orin/lingbot-map-stmem
uvicorn backend.batch_service:app --host 0.0.0.0 --port 8000
```

### 6. 验证

浏览器访问 `http://<Orin-IP>:8000`，应返回：
```json
{"status": "running", "service": "LingBot-MAP Streaming Service"}
```

## 目录结构

```
STMEM_HOME/
├── backend/
│   ├── batch_service.py        # FastAPI 推理主入口
│   ├── requirements.txt        # pip 依赖清单
│   ├── scripts/
│   │   ├── scale_calibrate.py   # DAv2 米制尺度校准
│   │   ├── scale_data.py        # 数据缩放 + 存档
│   │   ├── convert_memory_pc.py # npz -> 前端 binary
│   │   └── run_dgsg_pipeline.sh # DGSG 建图胶水脚本
│   └── dgsg/
│       ├── lingbot.sh           # 建图入口脚本
│       ├── scripts/
│       │   ├── dgsg_refactor.py         # 主建图管线
│       │   └── construct_scene_graph.py # 场景图构建
│       └── configs/mydata/      # 建图配置
├── checkpoints/
│   └── lingbot-map-long.pt     # lingbot-map 模型权重
├── models/                      # DGSG 模型文件
│   ├── yoloe-26l-seg.pt
│   ├── yoloe-26l-seg-pf.pt
│   ├── sam2.1_b.pt
│   ├── open_clip_pytorch_model.bin
│   └── moondream2/
├── src/web/assets/
│   └── spatial.js               # 前端空间记忆可视化
├── data/                        # [运行时] 推理数据
├── experiments/                 # [运行时] DGSG 输出
├── datasave/                    # [运行时] 数据存档
├── .env.example
└── README.md
```

## 环境变量参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `STMEM_HOME` | (必须) | 项目根目录 |
| `STMEM_DATA_DIR` | `$STMEM_HOME/data` | 帧/深度/位姿数据 |
| `STMEM_MODEL_PATH` | `$STMEM_HOME/checkpoints/lingbot-map-long.pt` | 模型权重 |
| `STMEM_CONDA_PYTHON` | `python3` | Python 解释器 |
| `STMEM_DGSG_DIR` | `$STMEM_HOME/backend/dgsg` | DGSG 管线目录 |
| `STMEM_DGSG_EXP_DIR` | `$STMEM_HOME/backend/dgsg/experiments/mydata` | 实验输出 |
| `STMEM_DGSG_MODEL_DIR` | `$STMEM_HOME/models` | DGSG 模型目录 |
| `STMEM_DATASAVE_DIR` | `$STMEM_HOME/datasave` | 数据存档 |
| `STMEM_DAV2_MODEL` | `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` | DAv2 模型 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像 |

## 端到端流程

1. 前端上传帧 → `POST /batch/{id}/frames`
2. 开始推理 → `POST /batch/{id}/start_inference`（Scale frames 初始化 + 逐帧推理）
3. 结束推理 → `POST /batch/{id}/finish_inference`（自动触发后续管线）
4. 后台自动执行：
   a. DAv2 尺度校准 (`scale_calibrate.py`)
   b. 数据缩放 (`scale_data.py`)
   c. DGSG 建图管线 (`run_dgsg_pipeline.sh`)
   d. 前端数据转换 (`convert_memory_pc.py`)
5. 前端自动切换至空间记忆可视化模式
