# PhotoBench 2.0 需要对齐 1.0 的 8 项

> 2026-10-20 ｜ 只读分析，未改任何代码  
> 范围：`Sentrix-Home-Web`（1.0，`psh`）→ `sentrix-home-web2.0`（2.0，`psh2.0`）的 PhotoBench 与后端接口  
> **一句话：功能都搬过去了，但 1.0 在 10 月 13–16 日修的那批「硬件数字算得准不准」的问题，2.0 没对齐。**

---

## 速查表

| #                               | 缺什么                     | 影响                           | 补哪里（1.0）                                                           | 优先级    |
| ------------------------------- | ----------------------- | ---------------------------- | ------------------------------------------------------------------ | ------ |
| 1                               | Mac 没算模型权重映射            | 内存数字只剩 39%                   | `backend/apple_telemetry.py:519-554`、`:581`                        | **P0** |
| 2                               | Jetson 的 CPU/GPU 拆分没存下来 | 46 上总数少 79%                  | `benchmark_orchestrator.py:1929-1931`、`:2005-2006`、`:2426`、`:5994` | **P0** |
| 3                               | 读不到账本报 0，而不是「不可用」       | 权限一掉静默漏 79%                  | `backend/jetson_telemetry.py:149-165`                              | P1     |
| 4                               | Mac 功耗没有任何校验            | 325 W 脏值入库，无法修正              | `backend/apple_telemetry.py:84/112/162/170/264/277`                | P1     |
| 5                               | 没有算子线程上限                | 重负载下评测可能被杀                   | `backend/thread_limits.py` + `deploy/thread-limits.env`            | P1     |
| 6                               | 健康检查没拆轻量版               | 配看门狗时可能超时误判                  | `backend/app.py` 的 `/api/health` 与 `/api/diagnostics`              | P1     |
| 7                               | 曲线少 25 个指标              | 只影响 vLLM 场景                  | `benchmark_orchestrator.py:1996-2016`                              | P2（待定） |
| 8                               | 没有平台参数与装环境脚本            | 153 的 `CLIP_DEVICE=cpu` 无处承载 | `scripts/deploy/platform-profiles.sh`、`install_env.sh`             | P2     |
| 第 7 项需先确认产品是否保留 vLLM 支持，不要直接照抄。 |                         |                              |                                                                    |        |

---

## 背景：Jetson 的内存为什么要特殊算

Jetson 没有独立显存，模型用的「显存」是**从系统内存里切走的一块**。所以：

```
模型真实占用 = CPU 侧 + 切给 GPU 的那侧
```

后半部分有两种量法，**两台机器表现完全不同**：

|       | 账本（nvmap）   | 差额（VmRSS − 明细合计）   |
| ----- | ----------- | ------------------ |
| 量的什么  | 划走了多少（分配量）  | 实际落地了多少（驻留量）       |
| 要权限吗  | 要 root      | 不要                 |
| 118 上 | 准（4889 MiB） | 准（4849 MiB，差 0.8%） |
| 46 上  | 准（2617 MiB） | **恒为 0，不能用**       |

**为什么 46 上差额是 0**：两台机器系统版本不同（118 是 L4T R36.4.7，46 是 R36.4.3），旧版的内存分配器 `nvmap.ko` 不把 GPU 占用记进进程内存账，新版会记。实测在 46 上绕开模型直接调驱动申请 2 GiB，账本显示拿到 2113 MiB，而进程内存 25.86 MiB **一动不动**。

**所以统一方案**：两台都用账本（唯一两边都准的渠道），差额只在账本读不到时兜底，**且算出 0 就报「不可用」**——因为 46 上那个 0 代表「没记账」，不是「没占用」。

**如果只看 VmRSS 会怎样**：118 报 6776 MiB（几乎正确，因为含 GPU），46 报 714 MiB（真实 3332，**漏 79%**）。同一份模型差 4.7 倍，纯属记账方式不同。

---

## 1. Mac 上少算了模型权重（P0）

**2.0 现状**：只取 `phys_footprint`（`apple_telemetry.py:398`），全仓没有 `PROC_PIDREGIONPATHINFO`。

**问题**：Mac 上 llama.cpp 用 mmap 加载 `.gguf`，**这些页不计入 `phys_footprint`**。1.0 实测：footprint 1612.8 MiB，权重映射 2572.9 MiB，真实约 4186 —— **2.0 只报 39%，且比例随模型变化，无法用系数校正**。

**补哪里**：`backend/apple_telemetry.py` 的 `_weight_map_bytes()`（`:581`）与 `_process_bytes()`（`:519-554`），结构体在 `:409`。  
**必须带上的坑**：macOS 27 上结构体 1272 字节，`vip_path` 在偏移 248，给小了返回 0 + `ENOMEM`。

---

## 2. Jetson 的 CPU/GPU 拆分算出来了，但没存下来（P0）

**2.0 现状**：`jetson_telemetry.py:192-194` 已产出三个字段：

```
model_process_host_memory_mib     CPU 侧
model_process_gpu_memory_mib      GPU 侧
model_process_gpu_memory_source   来源渠道
```

但 `application/run_execution/gpu_sampler.py` 的字段映射表没有它们，`domain/runs/analysis.py:1629` 的曲线清单也没有 → **存盘时被丢弃**。

**影响**：118 上总数歪打正着是对的（VmRSS 本含 GPU），但看不出 GPU 的显存占用；实测而 **46 上总数直接少 2617 MiB（79%）**。

**补哪里**：1.0 有**三处**清单，必须同时补，少一处就静默丢字段：  
`benchmark_orchestrator.py:1929-1931`（采样装配）、`:2005-2006`（曲线清单）、`:2426-2427` 与 `:5994-5995`（实时面板/历史）。

---

## 3. 「读不到」被当成「占用是 0」（P1）

**2.0 现状**：`jetson_telemetry.py:163-166` 是「两个都读、取大的」。

**注意：正常运行时不会出错**（118 取 4889 ✓、46 取 2617 ✓）。问题只在**账本读不到时**：46 的差额恒为 0，于是报 **0 并当成真实值**，漏 79% 且界面无任何异常。

**补哪里**：`backend/jetson_telemetry.py:149-165`，改成：

```
账本可读 → 用账本
账本不可读 → 用差额
差额 ≤ 0 → 报 unavailable，不当真
```

两块板日常都走账本，**改完数字不变**。

---

## 4. Mac 功耗没有任何校验（P1）

**2.0 现状**：`apple_telemetry.py:109` 读到什么存什么。

**问题**：macmon 0.8.2 在 macOS 27 上会把积压的功耗增量一次性吐出，实测峰值 325 W（历史 8842 W）。而 `gpu_samples.jsonl` **只追加不改写**，脏值进去就无法修正。

**补哪里**：`backend/apple_telemetry.py` 的 `MACMON_MIN_VERSION`（`:112`）、`POWER_LIMIT_W=150`（`:84`）、`macmon_power_is_trusted()`（`:264`）、`_plausible_power/_temperature`（`:162/170`）、`_drop_power_readings`（`:277`）；画曲线时的二次过滤在 `benchmark_orchestrator.py:6025`。

---

## 5. 没有限制算子线程数（P1）

**2.0 现状**： `set_num_threads` / `OMP_NUM_THREADS` / `intra_op_num_threads` 零命中。

**问题**：torch / onnxruntime / opencv 默认按核数开满线程。1.0 在 153 上实测跑出 **241 个线程抢 32 核** → 健康检查超过 6 秒 → 被判死 → 重启 → **正在跑的评测被 SIGTERM 杀掉**。2.0 进程结构相同，问题不会自己消失。

**补哪里**：`backend/thread_limits.py`（含 `lib_threads()`、`ort_session_options()`）、`deploy/thread-limits.env`；调用点在 `backend/app.py:62` 与 `backend/model_clients.py:17`。  
**坑**：`OMP_NUM_THREADS` 对 opencv / onnxruntime 无效，必须在代码里设。

---

## 6. 健康检查接口耗时长，没拆轻量版（P1）

**2.0 现状**：`apps/api/src/sentrix/api/routes/health.py` 的 `/api/v1/health` 同时查数据库和嵌入侧车。

**说明**：这个设计本身有价值（能发现「进程活着但嵌入器挂了」，1.0 吃过 354/370 次静默降级的亏）。**风险**在于 1.0 正是因为「完整诊断太重、被高频轮询会超时」才拆成两个端点：

|    | 1.0                        | 2.0                  |
| -- | -------------------------- | -------------------- |
| 轻量 | `/api/health`，0.15–0.21 ms | 无                    |
| 完整 | `/api/diagnostics`，7–11 ms | `/api/v1/health`（合一） |

**补哪里**：`backend/app.py` 的两个端点。或者至少确认监控侧超时阈值够宽。

---

## 7. 曲线少 25 个指标（P2，待定）

**2.0 现状**：`analysis.py:1629` 的清单 21 项，1.0 是 46 项。

缺的是：vLLM 的 KV cache 细分 13 项、推理吞吐 `runtime_*` 13 项（1.0 还有「所有 `runtime_` 开头字段自动进曲线」的兜底）、第 2 项那 3 个 Jetson 字段。

**先确认**：2.0 只服务 llama.cpp → 属有意裁剪，不用补；还要接 vLLM（1.0 的 8100 现在就是 vLLM）→ 真缺口。

**补哪里**：`benchmark_orchestrator.py:1996-2016`。

---

## 8. 没有平台参数与装环境脚本（P2）

**2.0 现状**：`scripts/deploy/` 只有一个 `start_legacy_mac.sh`；1.0 有 8 个（`start_all.sh`、`install_env.sh`、`platform-profiles.sh`、`restart_photobench.sh` 等）。

**公平地说**：2.0 用 systemd 替代了一部分（`deploy/systemd/` 3 个 service，均带 `Restart=on-failure`），比脚本更可靠。

**真正无替代的两个**：

- `platform-profiles.sh` —— 承载「153 主动选 `CLIP_DEVICE=cpu`」这类**探测不出来的人工知识**（CUDA 明明可用，但会让分配器崩）
- `install_env.sh` —— 新机器一键建三个运行环境

---

## 附：三处注释与代码不符

不是缺陷，但会误导人。**2.0 自身跑起来没问题**。

| 位置                                       | 注释说               | 实际                                                                |
| ---------------------------------------- | ----------------- | ----------------------------------------------------------------- |
| `api/routes/suites.py` 开头                | 续跑接口没接            | 文件末尾有，`application/runs/suites.py:276` 完整实现                       |
| `config/service.py` 开头                   | `.env.local` 读写没搬 | `persist_runtime_connection()` 412–419 行就在写 `BENCH_JUDGE_API_KEY` |
| `persist_runtime_connection()` docstring | 只落非密钥参数           | 紧接着就处理 `judge_api_key`                                            |

---

## 建议处理顺序

**按「评测会不会出事」排**：5（线程）→ 6（健康检查）→ 2（GPU 拆分）。

**按「数字准不准」排**：1（Mac 权重）→ 2（GPU 拆分）→ 3（报 0）。

前两项改动都不大：第 1 项是新增一个函数（约 80 行，1.0 有现成实现），第 2 项只是补三处字段清单，不动逻辑。

---

## 取证说明

- 1.0 路由：`benchmark_orchestrator.py:8137-8607` 三个 handler 全文
- 2.0 路由：运行时 `GET http://127.0.0.1:8772/openapi.json`（34 条，权威）
- 2.0 字段：`gpu_sampler.py` 全文、`analysis.py:1629-1650`
- 遥测：两边 `jetson_telemetry.py` / `apple_telemetry.py` 逐段比对
- 判定「不存在」均以 grep 零命中为依据，未做推测
- 已核对 GitHub `ysh12304124/sentrix-home-web-2.0`（main `497bfcf`）：四个关键文件与 153 本地**逐字节一致**，本文结论对 GitHub 版本同样成立
