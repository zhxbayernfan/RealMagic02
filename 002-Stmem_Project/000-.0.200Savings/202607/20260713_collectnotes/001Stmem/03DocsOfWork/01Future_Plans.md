# 未来Todos

## 第0点（最重要）：逻辑整理
梳理整个 stmem.html + memory.js 的数据流：
- 每个 tab 调用了哪些 API
- 后端每个路由返回什么数据
- 前端哪些变量/函数是共享的，哪些是独立的
- 画出数据流图（文字版）

## 第1点：HTML + JS 分离
不只是 JS，HTML 也想抽离到独立文件。
- 创建 `memory-report/` 文件夹
- `memory-report/index.html` — 报告页独立 HTML
- `memory-report/report.js` — 报告页 JS 逻辑
- stmem.html 的 tab3 改为 iframe 或跳转引用

先问组长意见再动手。

## 第2点：后续
- 其他 tab（记忆检索/片段事件/设置）同样独立
- 需要分离的内容：
  - `/api/memory/mood` 和 `/api/memory/report` 的 fetch
  - 叙事生成 `/api/memory/narrative` 的异步插入
  - 情绪节律渲染
  - 关键词/地点/人物列表渲染
  - 事件计数逻辑
