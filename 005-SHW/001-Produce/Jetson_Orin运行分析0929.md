# Jetson Orin AGX(123) 部署复盘汇总

> 对象: Sentrix-Home-Web + Photobench 端侧部署全流程
> 对照: 118(AGX)/46(NX)/M4 部署顺利 vs 123(AGX) 多轮失败
> 日期: 2026.09.29 | 整理: 部署组(基于20+轮run记录与三方独立诊断综合)
> 定位: 同一套代码,为什么不同机器命运不同——环境问题的完整解剖与部署SOP

---

## 一、五平台部署对照

| 平台 | 芯片 | 内存 | 部署结果 | 全流程耗时 | 资产成功率 | 备注 |
|------|------|------|----------|-----------|-----------|------|
| 118 | Orin AGX | 64G | ✅ 一次配置成功 | 100qa ~6h | 62.5% | 连续3天稳定运行 |
| 46 | Orin NX | 16G | ✅ 顺利 | 487qa ~4.5h | — | 并发压到2解决 |
| M4 | Apple Silicon | — | ✅ 顺利 | — | — | 同样踩过抄历史参数的坑 |
| **123** | Orin AGX | 64G | ❌ 20+轮失败后修复 | 单轮16h+未完(修复前) | 修复前2.2%→**修复后98.9%**(终值359/363) | 见下文 |

**核心事实:123与118硬件完全相同(同为AGX 64G),代码主干一致(backend/pipeline.py 两机md5相同),失败全部来自运行环境。**

---

## 二、llama-server 参数错位(头号杀手)

### 2.1 参数对照

| 参数 | 正确(118) | 错误(123) | 毒性 |
|------|-----------|-----------|------|
| --n-gpu-layers | **999**(全部上GPU) | **99**(大半模型在CPU) | ★★★★★ 单条即可致3倍慢 |
| --parallel | **1**(单槽独占) | **4**(四槽分算力) | ★★★★ |
| --reasoning | **off** | **缺失**(思考模式开启) | ★★★★ 小模型边答边想,输出预算腰斩 |
| --ctx-size | 8192 | 16384 | ★★ 显存翻倍 |
| --ctx-checkpoints 0 / --cache-ram 0 / --no-cache-prompt / --no-cache-idle-slots / --slot-prompt-similarity 0 | 齐全 | **全缺** | ★★★ |

### 2.2 正确启动命令

```bash
llama-server -m <qwen35-q4km.gguf路径> \
  --mmproj <mmproj路径> --host 0.0.0.0 --port 8100 \
  --ctx-size 8192 --parallel 1 --n-gpu-layers 999 \
  --flash-attn on --jinja --reasoning off \
  --ctx-checkpoints 0 --cache-ram 0 --no-cache-prompt \
  --no-cache-idle-slots --slot-prompt-similarity 0
```

### 2.3 教训

- **跨机器部署,VLM启动参数必须从成功机器逐字复制**。123 的 --n-gpu-layers 99 直观看像"全部层数",实际导致大半模型留在CPU——数字不能靠直觉,必须对照成功机器原值
- 同类坑跨平台复发:46 的正确 ctx-size 是 24576(每槽6144),在Mac上部署时抄了旧的历史参数再次踩雷。**每个平台的参数都是独立的,不存在"通用默认值"**

---

## 三、并发超订的数学约束

### 3.1 一条公式

```
n_ctx_slot = ctx_size / parallel  必须 ≥  单次请求最大tokens
```

实证:--ctx-size 8192 --parallel 8 → 每槽1024 tokens,而 image analysis 请求需要约1338 tokens → 400 Bad Request。

### 3.2 并发链条的整订

pipeline_workers(按内存自动算,64G→8) × event_summary_workers(8) = 16 路并发,灌向只有 1-4 个槽位的 VLM;同时 face/CLIP 各自再抢GPU——**多层超订叠加,任何一层单独看都"合理"**。

### 3.3 推荐并发配置(弱VLM环境)

```bash
SENTRIX_PIPELINE_MAX_WORKERS=2
SENTRIX_EVENT_SUMMARY_MAX_WORKERS=2
PHOTOBENCH_QA_CONCURRENCY=2
PHOTOBENCH_JUDGE_CONCURRENCY=1
OLLAMA_TIMEOUT_SECONDS=600          # 默认180s不够,VLM单请求~67s+排队
SENTRIX_PIPELINE_MAX_RETRIES=3
```

---

## 四、bge 文本嵌入 sidecar 超时(根因链)

现象:semantic 阶段全部超时,failed_stage=semantic,error=timed out。

```
pipeline.py:87 硬编码20s超时(httpx.post → 8101/embed)
  ↓
sidecar: CPU运行 + 全局锁串行(所有请求排队)
  ↓
16个worker并发灌入 → sidecar线程111个,全部futex_wait等锁
  ↓
248次BrokenPipeError —— 算完了,客户端20s前已经断开
  ↓
20s超时线在CPU串行模式下不可能通过(单次encode实测120s)
```

**隔离对照**:face.detect 空闲时0.78秒,run中465-608秒——检测逻辑本身无罪,纯粹被并发超订拖死。

**修复**:
```bash
SENTRIX_TEXT_EMBEDDER_DEVICE=cuda   # bge-m3 跑CPU纯属浪费,单次encode 120s→秒级
# 治本(需改代码):拆全局锁 + _embed_text退避重试 + 20s超时改可配
```

---

## 五、幽灵进程生态学(Jetson边缘设备"不死族"图鉴)

部署期发现四类"杀不死"的进程,每种需要不同的根除方式:

| 类型 | 实例 | 根除方式 |
|------|------|----------|
| systemd user service | gemma4-q4-8081.service(enabled),kill后自动复活 | `systemctl --user stop` + `systemctl --user disable` |
| 看门狗脚本 | start_q4_8081.sh 每30秒检查,死了就重启 | 先杀看门狗,再杀目标进程 |
| crontab 自愈 | auto_restart_8091.sh 每5分钟检查,CPU>400%杀重 | `crontab -l` 清查 |
| 长寿重复进程 | 11001 旧实例运行37天,856% CPU,4300个zombie assets | `pgrep -f` 确认单实例后清理 |

**排查口诀**:
```bash
ps aux | grep <进程名> | grep -v grep    # 第一步永远是看现场
kill 后再 ps 一次——复活即查 systemd/crontab/看门狗三件套
pgrep -f <名> | wc -l 确认唯一实例
```

**生态学规律**:部署失败的机器,几乎都有前人留下的自愈机制。**每个"杀不死"背后都有一位"不想让它死"的前人。**

---

## 六、配置漂移(跨机器部署头号杀手)

| # | 案例 | 症状 |
|---|------|------|
| 1 | runtime_providers.py 硬编码 192.168.0.118 | VLM 请求发往**别人的机器** |
| 2 | DEFAULT_SENTRIX_URL 兜底指向 http://192.168.0.153:8091 | 配置缺一个字段,静默连到错误后端 |
| 3 | .env 过期:SENTRIX_LLM_BACKEND=ollama,指向11434死端口 | 该文件"随时会坑到下一个人" |
| 4 | endpoint_model 填 "llm" 而非完整模型路径 | VLM 报无效模型 |
| 5 | 123 缺少"4xx不重试"逻辑(118有) | 必然失败的请求白白重试,浪费全部并发 |
| 6 | 双 API 实例(8091+11001)共用 SQLite/qdrant 目录锁 | 锁冲突 → vector search 降级为全表扫描 |

**排查法(部署前必做)**:
```bash
grep -rn "192.168\|10.0\|/home/asus\|11434" <代码目录>
env | grep -E 'SENTRIX|CLIP|FACE|OLLAMA|PHOTOBENCH'
```

**注意区分"配置文件"与"运行实况"**:.env 里写的模型与实际运行的进程可以完全不同——验证以 `ps -eo args | grep llama-server` 和 `/props` 为准,不以配置文件为准。

---

## 七、环境与数据层

### 7.1 环境项
- **GPU温度**:123 run中71°C vs 118的44°C(差27°C;Orin降频阈值约80°C,未实证降频,列待查)
- **时间漂移**:Jetson 无RTC电池,断电后时间偏差会导致 pip/apt 证书验证失败——操作前 `date` 确认
- **磁盘**:93%满(剩3.7G)导致I/O性能劣化;部署前确保≥10G可用
- **依赖缺失**:insightface 未装 → identity_seed 8人全部"no detectable faces"。部署前 `python3 -c "import insightface, ..."` 验证全部依赖

### 7.2 数据层
- **sentrix.db 膨胀**:每次full run创建新scope,旧scope的assets不自动清理(6000+旧failed);处理:清理旧scope + VACUUM
- **zombie workers**:先杀orchestrator后,其pipeline workers继续运行(856% CPU无active run)——**杀orchestrator前必须先停run**
- **连接堆积**:httpx 单次请求创建新连接,90+ TIME-WAIT;高频调用必须用连接池复用

---

## 八、部署SOP(在新Jetson设备上部署本系统的标准流程)

### 8.1 部署前检查

```bash
date                                        # 系统时间
df -h /                                     # ≥10G可用
cat /sys/devices/virtual/thermal/thermal_zone*/temp   # GPU温度<60°C
ps aux | grep llama-server | grep -v grep   # 无冲突进程
crontab -l                                  # 无自动重启脚本
systemctl --user list-units                 # 无冲突service
grep -rn "192.168\|/home/asus\|11434" <代码目录>       # 无硬编码残留
python3 -c "import insightface"             # 依赖齐全
```

### 8.2 启动顺序

```bash
# 1. 清场
pkill -f sentrix; pkill -f orchestrator; pkill -f sidecar

# 2. sidecar(CUDA)
nohup env SENTRIX_TEXT_EMBEDDER_DEVICE=cuda \
  python scripts/maintenance/text_embedder_sidecar.py --port 8101 &

# 3. API(2worker,CLIP/Face让CPU)
nohup env CLIP_DEVICE=cpu FACE_PROVIDERS=CPUExecutionProvider \
  SENTRIX_PIPELINE_MAX_WORKERS=2 \
  python -m uvicorn backend.app:app --port 11001 --workers 2 &

# 4. orchestrator(低并发+长超时)
nohup env PHOTOBENCH_QA_CONCURRENCY=2 PHOTOBENCH_JUDGE_CONCURRENCY=1 \
  SENTRIX_PIPELINE_MAX_RETRIES=3 OLLAMA_TIMEOUT_SECONDS=600 \
  python services/photobench/backend/benchmark_orchestrator.py --port 8771 &
```

### 8.3 部署中监控

```bash
curl -s http://127.0.0.1:11001/api/assets?scope_id=<scope>&limit=2000   # 资产状态
tail -f <orchestrator日志> | grep -E 'phase|finished'                    # 阶段进展
nvidia-smi; jtop                                                        # GPU/温度
ss -tan | grep 8100 | grep -c TIME-WAIT                                 # 连接堆积
```

### 8.4 部署后纪律
- **VLM 启动后不再杀启**——重启清零 KV cache 与 graphs_reused,性能从最优逐渐衰退
- 稳定运行1小时后再开始评测
- 全程不重启服务;异常先诊断后动手

---

## 九、遗留问题(诚实清单)

1. **VLM 响应衰退**:启动30-50分钟后从400ms衰退到10s+,与处理量无关;杀启重启仅短期有效;118同配置三天无衰退。原因未明,候选:温度、内存碎片、驱动版本。**未解**。
2. **GPU 温差 27°C**(71 vs 44)未找到确定性根因(散热配置?负载历史?)。**待查**。
3. **sidecar 全局锁**未拆(治本需改代码:允许多请求并行encode + 退避重试 + 超时可配)。**待重构**。
4. **orchestrator 分支差异**(123=zhx-123分支,与118主线差722行;含嵌套目录残留backend/backend/等223文件)未合并。**待合并**。

---

## 十、本次部署最终结果

**run aa931e** | 2026-09-29 11:29 起跑 | album3-max 全量(363资产) | 全程约6.5小时

### 10.1 Pipeline 层(修复成效终验)

| 指标 | 118(参考基线) | 123 修复前 | **123 修复后(本轮)** |
|------|--------------|-----------|---------------------|
| 资产成功率 | 62.5%(227/363) | 2.2%(8/363) | **98.9%(359/363)** |
| failed | 136 | 303+ | **4** |
| 单轮状态 | completed_with_errors | 多轮16h+未完 | completed_with_errors |

五层修复逐项生效:llama参数(999/parallel/reasoning off)→参数错位消除;gemma4幽灵systemd根除→GPU独占;硬编码IP改127.0.0.1→请求归位;并发压到2+超时600s→排队消失;db清理→I/O恢复。**修复前2.2%→90.1%(中程)→98.9%(终值),每一层修复都有对应的数据阶跃。**

### 10.2 QA 层(新病根第一层,已定位未手术)

100 样本全部失败:HTTP 404 "turn not found"——**双 API 实例(8091旧+11001新)共用一份 SQLite,turn 在一侧创建、在另一侧查询必然 404**。judge 因此 skip 100/100,quality 未产出。前端所见的"驴唇不对马嘴"即 404 占位,非模型输出。

病灶已在 8091 旧实例(自0925运行至今),清除后单后端重跑 QA 即可——pipeline 成果已在库(scope 复用),QA 重跑约50分钟。

### 10.3 定性

100qa 卷面:**pipeline 达标并创新高(98.9% vs 基线62.5%),QA 层被新病根拦截,本轮不计分**。这是一份迟到了许久、且只答完一半的卷子——答完的一半证明修复路线全部正确;没答完的一半,病灶已定位到具体进程与端口。

完整存档:02仓库 `005-SHW/002-Savings/202609/20260929/`(aa931e 全量JSON,含各阶段耗时与逐资产状态)。

---

## 十一、结论

同一套代码,五个平台,两种命运——**代码不是变量,环境才是**。本复盘的全部内容可以压缩成三句话:

1. **逐字抄成功机器的启动参数,一个flag不许改,并以进程实况(而非配置文件)为准。**
2. **并发必须用数学算**(n_ctx_slot ≥ 请求tokens;GPU 任务分配矩阵;workers 对齐槽位),不许按内存拍脑袋。
3. **部署前先清场**——Jetson 边缘设备上,每一个"杀不死"的进程背后都有一位"不想让它死"的前人;幽灵不清,参数全对也白搭。

---

*本文档由 123 部署全程 20+ 轮 run 的失败记录、三方独立诊断与 118/46/M4 成功基线对照综合而成。*
