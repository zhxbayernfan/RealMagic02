# Jetson Orin端侧VLM推理方案技术对比:Ollama / llama-server / TensorRT

## 1. 问题背景:Ollama管线的失败根因

Sentrix Home Web评测管线的Ollama方案曾于153上成功跑通并合入主干;但迁移至123(Orin,aarch64)后,部署gemma4:12b出现管线整体失效.根因如下:

- gemma4:12b是**思考型模型**:经/v1/chat/completions调用时,先输出reasoning(思考过程),再输出正式content
- Ollama的模型模板(Modelfile)在构建时即决定了思考行为,**调用方没有统一的剥离/关闭开关**——是否思考、思考多长,完全由模板决定,不可控
- 管线设置num_predict=2048,思考过程消耗1500+ tokens,预算耗尽后content返回**空字符串**
- 空描述导致FTS索引无内容可建,Agent检索0步,QA评测全链路失效
- 该缺陷是**结构性**的:加大预算无法根治(思考内容依旧混入/挤占输出,且耗时不可控),参数调优无从下手

**结论:问题不在模型,而在Ollama这层封装对输出路径没有控制权.Ollama在153的成功证明其方案本身可行,而123(Orin,aarch64)上gemma4:12b的思考行为经Ollama封装无法关闭——同一套封装,换环境后失去了输出控制权.选型必须换封装,而不是换参数.**

注:该缺陷的触发取决于**模型版本×模板行为**的组合——153的部署组合未触发,123(gemma4:12b,aarch64构建)的组合触发;两者间的具体差异(模型版本/平台构建)列为后续考证项.

## 2. 三方案定位总览

| 项 | Ollama | llama-server | TensorRT(Edge-LLM) |
|---|---|---|---|
| 本质 | llama.cpp的模型管理器(高度封装) | llama.cpp的**原生server**(裸暴露) | NVIDIA官方推理优化引擎 |
| 后端 | llama.cpp | llama.cpp(同源) | TensorRT自研kernel |
| 模型格式 | GGUF | GGUF(**与Ollama同源同格式**) | 编译后的engine文件 |
| 控制粒度 | 模板焊死,调用方不可控 | 模板可覆盖/参数全显式/可绕过模板 | 构建时定义,运行时无reasoning |
| 部署形态 | 服务+模型管理一体化 | 单二进制+模型文件 | engine编译产物+配套server |
| 目标场景 | 个人/快速原型 | 生产环境/需要精确控制的管线 | 极致性能/大规模生产 |
| 118实战 | - | **已验证跑通(lyc)** | 未在118使用 |

关键事实:**llama-server与Ollama同出llama.cpp一脉,吃同一份GGUF模型**——差异不在引擎本身,而在封装层把控制权交给了谁.

## 3. 推理机制对比(输出路径/思考模式)

### 3.1 三条输出路径

```
Ollama路径:
  prompt → [模板强制:reasoning思考过程] → content
           ↑ 思考与content共用num_predict,模板不可改,路径不可绕

llama-server路径:
  prompt → [思考行为可控制:覆盖chat-template/调整参数] → content
           ↑ --chat-template显式指定,或直接使用/completion端点绕过模板,路径由调用方定义

TensorRT路径:
  prompt → content
           ↑ 构建时即无reasoning分支,输出即结果
```

### 3.2 控制权对比

| 控制项 | Ollama | llama-server | TensorRT |
|---|---|---|---|
| 思考模式关闭 | ❌ 模板决定,不可关 | ✅ 模板覆盖/参数控制 | ✅ 天生无思考分支 |
| chat-template | ❌ 模型内置 | ✅ --chat-template任意指定 | ✅ 构建时定义 |
| 输出预算(num_predict) | 可设但被思考稀释 | ✅ 全额作用于content | ✅ 全额作用于content |
| 绕过模板直连 | ❌ | ✅ /completion端点 | ✅ 原生 |
| 采样参数显式控制 | 部分 | ✅ 全量暴露 | ✅ 构建时+运行时 |

Ollama的失败,本质上是一次**控制权缺失事故**:管线需要的只是"不要思考、直接给content"这一个开关,而Ollama的封装把这个开关藏死了.llama-server对同一份模型文件提供这个开关;TensorRT则在引擎构建阶段就消灭了这个问题.

## 4. 部署与维护成本对比

### 4.1 Ollama:极低

- 单服务+模型文件,ollama pull即完成部署
- 无编译、无版本矩阵、升级模型重pull即可
- 代价:上述控制权缺失,黑盒行为不可调

### 4.2 llama-server:低

- 单二进制+GGUF模型文件,**无编译环节**
- 启动参数(--chat-template/--port/--n-predict等)全显式,即起即用
- 与Ollama同源同格式,模型文件可复用,迁移成本近零
- 维护:升级=换二进制/换模型文件,不涉及构建链

### 4.3 TensorRT:重

TensorRT(含Edge-LLM分支)的部署是一条**重型工程链**,成本集中在以下几点:

| 成本项 | 具体内容 |
|---|---|
| **版本矩阵锁死** | JetPack↔CUDA↔cuDNN↔TensorRT↔驱动,五层版本必须精确匹配;任一层升级即全链重装 |
| **engine编译** | engine非通用产物,与GPU架构(compute capability)一一绑定;编译本身在aarch64上资源受限、耗时长 |
| **量化校准** | INT4量化需要准备校准数据集(calibration set),校准质量直接影响精度 |
| **多模态配方** | VLM(视觉语言模型)需要专门的TRT-LLM模型配方支持,并非所有模型都有现成配方;无配方则需自行适配模型代码 |
| **draft模型链** | MTP/推测解码需额外编译draft engine,链路再翻一倍 |
| **升级=重编译** | 换模型版本/换推理参数(如draft-topk/draft-step)→重新编译;换JetPack版本→全部重编译 |
| **排障门槛** | 编译错误需要NVIDIA生态深度知识;engine构建失败的可查资料远少于Ollama/llama.cpp社区 |

一句话:**TensorRT的engine是"一次编译,绑死一套硬件+一套软件栈"的产物——部署它不是装软件,是建一条小型编译生产线.**

| 部署维度 | Ollama | llama-server | TensorRT |
|---|---|---|---|
| 首次部署耗时 | 分钟级 | 分钟级 | 天级(含编译/校准/排障) |
| 版本变更成本 | 重pull | 换文件 | 全链重编译 |
| 依赖复杂度 | 自包含 | 自包含 | 五层版本矩阵 |
| 排障可查资料 | 多(社区) | 多(llama.cpp社区) | 少(企业级文档+源码) |

## 5. 性能与资源对比

| 项 | Ollama(GGUF) | llama-server(GGUF) | TensorRT(INT4+MTP) |
|---|---|---|---|
| 量化 | GGUF通用量化 | GGUF通用量化(与Ollama同源) | INT4+NVIDIA定制优化 |
| 推理性能 | llama.cpp通用水平 | **与Ollama同源同水平** | 算子融合+MTP,典型提升2-3倍 |
| 显存 | FP16约8GB级 | FP16约8GB级 | INT4约3GB级 |
| Jetson适配 | 社区级 | 社区级(与Ollama同) | 官方 |

性能上TensorRT占优;但注意:**llama-server与Ollama性能完全同源**——选llama-server放弃的不是性能,是Ollama的错误封装;选TensorRT获得的性能提升,要以第4节的部署链为代价.在123当前"复刻118验证路线"的目标下,性能不是第一优先级,**跑通并复现管线行为才是**.

## 6. 管线适配性与选型结论

管线结构:photo_import → VLM生成caption(JSON) → FTS索引 → Agent检索 → QA评测.

| 管线需求 | Ollama | llama-server | TensorRT |
|---|---|---|---|
| caption稳定产出 | ❌ 思考挤占预算,content空 | ✅ 模板可控,content稳定 | ✅ 无思考分支 |
| 与118行为对齐 | ❌ 封装不同,行为无法对齐 | ✅ **与118同款server,行为可对齐** | ⚠️ 全新引擎,行为需从头验证 |
| 显存压力 | 8GB级 | 8GB级 | 3GB级 |
| 复刻周期 | 已失败 | **短(同格式同引擎,迁移近零)** | 长(编译链+配方+联调) |
| 排障社区支持 | 多 | 多(同llama.cpp生态) | 少 |

**选型结论**:

1. **决定性因素是输出可控性**:管线每一环都需要可解析的结构化输出,Ollama因思考模式不可关而结构性出局
2. **llama-server是当前最优解**:与Ollama同源同格式(模型文件直接复用),补上了控制权缺口;更关键的是——**118已用同款server验证跑通,选择llama-server即选择与118行为对齐,复刻路径有实战先例**
3. **TensorRT是性能天花板,不是当前解**:其引擎级优化(INT4+MTP)确有价值,但部署链 heavyweight,适合作为管线稳定后的远期优化项,而非复刻阶段的选型

---
**当前进展**:选型论证完成(本文档).118同款llama-server启动参数由200C实地抄录中,123将按同参数部署,评测数据另行提交.

> 通俗一句:123的问题从来不是教室不够高级——是钥匙不在自己手里.llama-server要做的,就是把钥匙拿回来,关上那扇吵闹的窗.
