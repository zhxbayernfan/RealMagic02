# Progress And Plan 0921
## SHW Photobench端侧评测进展与计划

**报告人**: L10'(CEA)  
**日期**: 2026.09.21  
**审核**: 兵部(LinuxC)  

---

## 一、执行摘要

**目标**: AGX端侧对齐NX基准(0.781@487qa),伪NX模拟验证端侧可用性  
**现状**: 16轮迭代找到NX成功配置,正在复现  
**关键认知**: 
- 1.1分是幻象,真实锚点是NX基准0.781
- FTS分离法是幻觉,正确路径是抄NX配置
- 伪NX模拟是主线任务,不是附加项  

---

## 二、迭代历史(16轮)

| 轮次 | 配置 | quality | retrieval | 结果 |
|------|------|---------|-----------|------|
| R3 | gemma4-bf16, reuse | 0.57 | 0.086 | 历史最佳(但非全链路) |
| R5 | gemma4-bf16, full | 0.26 | 0.010 | FTS连错DB |
| R6 | gemma4-bf16, full | 0.31 | 0.062 | 8091重启,tool_rejected修复 |
| R7 | qwen35-q4km本地, full | 0.335 | 0.179 | planner blocked |
| R8-R10 | 混合架构尝试 | 0.31 | 0.057 | 代码改不动,__pycache__坑 |
| R11 | gemma4-bf16, full | 0.31 | 0.071 | 确定性模型,同配置同结果 |
| R12 | qwen3.5-4b-lora@153, reuse | 0.48 | 0.016 | 3090远程,非端侧 |
| R13 | gemma4-bf16, MULTI_RETRIEVER | 0.33 | 0.067 | 环境变量未生效 |
| R14 | 查embedding_router | - | - | orchestrator缺embedding |
| R15 | qwen35, full | cancelled | - | pipeline卡死 |
| R16 | qwen35, reuse | 0.567 | 0.142 | 被cancelled |

**核心教训**:
1. FTS分离法是幻觉,不存在捷径
2. MULTI_RETRIEVER=1是错误方向,NX成功配置没有此变量
3. 全链路full模式在AGX上易卡死,需查根因
4. **正确路径:抄NX成功配置,不自己发明**

---

## 三、NX成功配置(抄作业)

**来源**: NX(46) run 20260918-170458, quality=0.781(487qa)

**配置详情**:
- **模型**: qwen35-q4km.gguf @ 8100端口
- **模式**: reuse(复用已有scope)
- **scope**: album_ced044ece245
- **sentrix_url**: http://127.0.0.1:11001
- **judge**: doubao-seed-2.0-lite(火山引擎)
- **环境变量**: 无特殊变量(干净启动)
- **耗时**: 4h22m(487qa)

**123复现状态**:
- ✅ qwen35-q4km已上8100端口
- ✅ gemma4已关
- ✅ MULTI_RETRIEVER已去掉
- ⏳ scope用album_a509d459d173(NX的scope不存在)
- ⏳ sentrix_url保持8091(123无11001服务)

---

## 四、当前进展

**正在执行**: R17(NX配置复现)
- run_id: 待启动
- 配置: qwen35-q4km@8100, reuse, album_a509d459d173
- 目标: 100qa quality>1.1, 2-3h完成

**123资源状态**:
- 负载: 0.57(空闲)
- 内存: 21G/61G(释放24G)
- 磁盘: /home 163G可用

---

## 五、三阶段计划(重锚NX基准)

### 阶段1: 对齐NX基准(本周)
- **目标**: 100qa reuse quality≥0.78(NX 487qa基准0.781)
- **方法**: NX配置复现,**judge烟测前置**
- **验收**: 
  1. ✅ doubao订阅有效(curl烟测)
  2. 100qa reuse≥0.78
  3. 全链路full模式跑通
- **陷阱警示**: 不追1.1幻象,不在reuse舒适区刷分

### 阶段2: 伪NX模拟(下周,主线)
- **目标**: AGX上模拟NX(16G内存限制),跑487qa
- **方法**: cgroups限制内存/CPU,复现NX性能
- **并行策略**: R17跑着时搭cgroups环境,不串行等
- **验收**: 伪NX环境下487qa quality≥0.78
- **意义**: 回答"端侧盒子能不能用"

### 阶段3: 端侧优化(下下周)
- **目标**: 伪NX环境下优化质量
- **方法**: 调prompt/换模型/优化检索
- **验收**: 伪NX 487qa quality>0.85

---

## 六、风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| judge订阅失效 | 中 | 分数无效 | **R17前烟测** |
| scope不匹配 | 中 | quality低 | 用NX相同scope或重建 |
| full模式卡死 | 高 | 超时 | 先用reuse验证,再修full |
| 模型端口冲突 | 低 | 启动失败 | 已确认8100可用 |
| 伪NX模拟失真 | 中 | 结论不可靠 | cgroups精确限制,对比验证 |
| judge服务不稳 | 中 | 评分失败 | 备用judge配置 |

---

## 七、需兵部支持

1. **确认scope策略**: 用album_a509d459d173还是重建NX的scope?
2. **full模式卡死根因**: 需深入查pipeline阶段为何hang
3. **judge服务烟测**: doubao-seed-2.0-lite订阅是否有效?(必须前置验证)
4. **资源协调**: 487qa跑时可能需独占123,协调其他用户

---

## 八、下一步行动

1. **立即**: 
   - judge烟测:验证doubao-seed-2.0-lite订阅有效
   - 启动R17(NX配置100qa reuse),等出分
2. **R17达标后**(quality≥0.78): 跑100qa full模式验证全链路
3. **100qa full达标后**: 搭伪NX环境(cgroups限制16G内存)
4. **伪NX环境就绪**: 跑487qa全链路,目标≥0.78
5. **487qa达标后**: 优化伪NX质量>0.85

**交付物**: 本周内100qa对齐NX基准,伪NX环境搭建方案

---

*报告结束*  
*下次更新: R17出分后*
