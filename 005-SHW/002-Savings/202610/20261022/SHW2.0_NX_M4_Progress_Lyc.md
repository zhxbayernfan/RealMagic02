# Sentrix Home Web 2\.0 NX与M4验证进展

# Mac Mini M4 部署进展汇总

## 一、项目概况

将 Sentrix Home Web 2\.0 部署到本地 Mac Mini M4（16GB 统一内存），完成全链路跑通。

## 二、遇到的问题与修复

### 问题 1：模型名显示为 `__current__`

**现象**：PhotoBench run 列表显示 `__current__ · PhotoBench-xxx · compact-10q`，看不出用的什么模型。

**原因**：

- 后端 `model_name` 字段直接存了 `request.model_profile`（即 `__current__`）

- 前端 `modelName()` 函数优先读 `model_profile` 而不是 `model_name`

**修复**：

- **后端** `service.py`：新增 `_probe_model_name()` 方法，从 VLM 端点 `GET /v1/models` 探测实际模型名，去掉路径前缀和后缀（如 `/path/to/qwen35-q4km.gguf` → `qwen35-q4km`）

- **前端** `format.ts`：`modelName()` 优先读 `model_name`，fallback 到 `model_profile`

```TypeScript
// 修改前
return str(r.model_profile) || str(r.model_name) || ...
// 修改后
return str(r.model_name) || str(r.model_profile) || ...
```

---

### 问题 2：导入阶段卡住 / 被取消



**现象**：PhotoBench 的 `photo_import` 阶段长时间显示 `导入 0/? 个媒体`，最终报 `cancelled before phase pipeline_processing`。

**原因**：

- 2\.0 的导入是**同步长请求**（一个 HTTP POST 包含：文件导入 \+ VLM 观察 \+ 嵌入 \+ 人脸 \+ 事件归因 \+ 总结），14 张照片需要 \~12 分钟

- HTTP 连接在长时间等待过程中可能断开，PhotoBench 收不到响应

- `httpx.ReadTimeout` 不在重试列表中

**现状**：全链路最终可以跑通（full 模式），但耗时较长。

---

### 问题 3：`build` 模式不支持

**现象**：前端选"构建相册"报错 `mode must be one of ['full', 'reuse'], got: 'build'`。

**原因**：2\.0 后端只实现了 `full` 和 `reuse` 两种模式，`build`（只导入不评测）未实现。

**修复**：

- `suites.py`：`RUN_MODES` 加入 `"build"`，build 模式不要求 `qa_set`

- `service.py`：build 模式跑 `model_deploy → scope_setup → identity_seed → photo_import → pipeline_processing → gpu_metrics → aggregate`，跳过 `qa_eval`

---

### 问题 4：onnxruntime\-gpu 不兼容 macOS

**现象**：`make install` 时 `onnxruntime-gpu` 在 Mac 上安装失败。

**修复**：`apps/api/pyproject.toml` 改为 `onnxruntime==1.30.0`（CPU 版本）。

---





# NX部署进展进展汇总

## 一、项目概况

将 Sentrix Home Web 2\.0 部署到NX，导入阶段报错，解决中。

## 二、遇到的问题与修复

### 问题 1：PyTorch CUDA 兼容性问题

- **问题**：

    - PyTorch 2\.14\.1 是用 CUDA 13\.0 编译的

    - 系统 CUDA 版本是 12\.6

    - CLIP 侧车无法使用 CUDA 加速

- **解决**：

    1. 卸载当前 PyTorch（CUDA 13\.0 版本）

    2. 安装 PyTorch 2\.14\.1\+cu126（CUDA 12\.6 版本）

    3. 安装过程耗时约 35 分钟（下载 2GB\+）

    4. 验证 CUDA 可用：CUDA available: True, CUDA version: 12\.6

### 问题 2：CUDA Provider 错误调试

#### 2\.1 错误现象

- **错误信息**：

    ```Plain Text
    CudaProviderMissingError: buffalo_l w600k_r50 请求了 CUDAExecutionProvider，
    实际拿到的 provider 链是 ['CPUExecutionProvider']
    ```

- **影响**：照片导入失败，人脸检测无法工作

- **错误来源**：PhotoBench 导入照片时调用 buffalo\_l 模型进行人脸检测

#### 2\.2 初步诊断

- **验证 onnxruntime\-gpu 已安装**：

    ```Bash
    .venv/bin/python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
    # 输出：['CUDAExecutionProvider', 'CPUExecutionProvider']
    ```

- **问题**：虽然 onnxruntime 报告支持 CUDA，但实际创建 session 时仍然退化到 CPU

#### 2\.3 根本原因分析

- **代码分析**：查看 apps/api/src/sentrix/infrastructure/models/onnx\_runtime\.py

- **关键发现**：

    - 代码中有 `preload_cuda_runtime()` 函数，通过 import torch 预加载 CUDA 库

    - 但动态链接器只在进程启动时读取 LD\_LIBRARY\_PATH

    - 运行时设置 LD\_LIBRARY\_PATH 对已启动的进程无效

    - 必须在进程启动前就设置好 LD\_LIBRARY\_PATH

#### 2\.4 解决方案尝试

##### 尝试 1：设置 LD\_LIBRARY\_PATH（失败）

- **操作**：在 systemd 服务文件中添加 Environment="LD\_LIBRARY\_PATH=\.\.\."

- **结果**：API 服务可以正常启动，但 PhotoBench 服务启动失败（exit code 3）

- **原因**：环境变量配置位置错误（放在了 \[Install\] 部分而不是 \[Service\] 部分）

##### 尝试 2：修正 systemd 配置（失败）

- **操作**：

    1. 修正 sentrix\-api\.service，将 Environment 移到 \[Service\] 部分

    2. 重启 API 服务

- **结果**：API 服务正常，但 PhotoBench 仍然失败

##### 尝试 3：为 PhotoBench 添加配置（失败）

- **操作**：

    1. 在 sentrix\-photobench\.service 中添加相同的环境变量

    2. 第一次放错位置（\[Install\] 部分）

    3. 修正到 \[Service\] 部分

    4. 重启服务

- **结果**：服务启动失败，不断重启

