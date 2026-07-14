---
name: 001Stmem-future-plans
description: Stmem未完成任务与后续计划
metadata:
  node_type: doc
  type: reference
  originSessionId: 1e2ce9cd-714c-4c88-bedd-8f8ed149b07b
---

# 未来Todos

## 第0点(最重要):逻辑整理
梳理整个stmem.html+memory.js的数据流:
-每个tab调用了哪些API.
-后端每个路由返回什么数据.
-前端哪些变量/函数是共享的,哪些是独立的.
-画出数据流图(文字版).
## 第1点:HTML+JS分离
不只是JS，HTML也想抽离到独立文件.
-创建`memory-report/`文件夹.
-`memory-report/index.html`-报告页独立HTML.
-`memory-report/report.js`-报告页JS逻辑.
-stmem.html的tab3改为iframe或跳转引用.
## 第2点:后续
-其他tab(记忆检索/片段事件/设置)同样独立.
-需要分离的内容:
-`/api/memory/mood`和`/api/memory/report`的fetch.
-叙事生成`/api/memory/narrative`的异步插入.
-情绪节律渲染.
-关键词/地点/人物列表渲染.
-事件计数逻辑.
