# 数据流完整对照（memory.js ↔ stmem.html）

> 行号基于 latest commit（43479c1 07101830）

## 1. mood 标题：`2026年07月，生活平淡如水`

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 425 | `/api/memory/mood` 路由入口，`sessions` 按时段数分档：≤10忙碌 / ≤30平淡 / ≤60有趣 / >60狂热 |
| 426 | `moodText` 同一阈值，完整版文案 |
| 432 | `sendJson({..., mood, moodText, ...})` |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 350 | `const m = await fetch('/api/memory/mood')` |
| 392 | `var mood = m.mood \|\| ''` |
| 447 | `<div class="rep-title">' + y + '年' + mo + '月，<br>' + (m.mood\|\|'')` |

---

## 2. 事件数：`28个事件`

**前端 stmem.html**（纯前端计算，不依赖后端）
| 行 | 内容 |
|----|------|
| 358-391 | 全部事件计数逻辑 |
| 362-376 | `getTheme(desc)` 按描述第一句关键词打主题标签（工作/宠物/街景/美食/走廊/特写/深夜） |
| 378-383 | 按时间排序→相邻帧主题不同=新事件 |
| 448 | `eventCount` 填入副标题 `'... 由 AI 根据 ' + eventCount + ' 个事件...'` |

---

## 3. 帧数：`108帧`

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 425 | `rows = db.prepare("SELECT capture_time, description FROM memories...").all()` → `frameList` |
| 343 | `frames: frameList \|\| []` 在 sendJson 中返回 |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 351 | `r.frames` 取 `report` API 的 frames 数组 |
| 448 | `r.frames.length` 填入副标题 |

---

## 4. 天数：`这个月你的相机记录了8天的生活`

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 425 | `activeDays = days.size`（去重的有记录的天数） |
| 432 | `sendJson({..., activeDays, ...})` |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 391 | `var activeDays = m.activeDays \|\| '--'` |
| 451 | `<p class="narrative">...<b>' + activeDays + '</b> 天的生活...` |

---

## 5. 叙事正文

**后端 memory.js** `/api/memory/narrative` 路由
| 行 | 内容 |
|----|------|
| 466-490 | 从 SQLite 取所有描述第一句→每50帧一组→写`/tmp/narr_input.txt`→调`gen_narr_50f.py`→gemma4:e2b（借100的Ollama，`192.168.0.100:11434`）→各组生成→融合→返回`{narrative}` |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 443 | `fetch("/api/memory/narrative")` 异步获取 |
| 451 | `(d.narrative\|\|'')` 插入段落 |

---

## 6. topPlace：`走廊`

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 425 | 遍历`locCounts`（办公室/走廊/…11个地点词），出现最多的= `topPlace` |
| 432 | `sendJson({..., topPlace, ...})` |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 389 | `var topPlace = m.topPlace \|\| ''` |
| 451 | `出现最多的地方是 <b style="color:#5E7A18;">' + (r.topPlaces&&r.topPlaces.length?r.topPlaces[0].name:'---')` |

---

## 7. topPerson：`一只白猫`

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 425 | 遍历`personList`（绿衣男子/橘猫/黑衣男子/…/白猫/机器狗），匹配最多的= `topPerson`，加量词（一只/一位） |
| 432 | `sendJson({..., topPerson, ...})` |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 390 | `var topPerson = m.topPerson \|\| ''` |
| 451 | `陪伴你最久的是 <b style="color:#C7700E;">' + (topPerson\|\|'---')` |

---

## 8. moodText：`你这个月的生活平淡如水`

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 426 | `moodText` 五档文案 |
| 432 | `sendJson({..., moodText, ...})` |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 393 | `var moodText = m.moodText \|\| mood` |
| 451 | `整体而言，<b>' + moodText + '</b>` |

---

## 9-12. 四张统计卡片

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 300 | `locWords` 定义（~60个地点词）|
| 303 | `spaceCount` = 本月第一句匹配到的不同地点词数 |
| 305 | `recordDuration` = 首帧到末帧的有效拍摄时长 |
| 307 | `weekNewCount` = 本周新增地点类型（同词表+同提取逻辑）|
| 301 | `realMemCount` = `memories` 表总行数（=总物件数）|
| 347-364 | `sendJson({..., stats:[{...},{...},{...},{...}], ...})` |

| 卡片 | backend 数据源 | 显示值 |
|------|---------------|--------|
| 本月记录场景数 | `stats[0].value` = `spaceCount` | 7 |
| 记录时长 | `stats[1].value` = `recordDuration` | 1h21m8s |
| 本周新场景 | `stats[2].value` = `'+'+weekNewCount` | +4 |
| 物件数 | `stats[3].value` = `realMemCount` | 108 |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 351 | `const r = await fetch('/api/memory/report')` |
| 403-405 | 遍历 `r.stats` 拼 `statsHtml` |
| 453 | `<div class="num-grid">' + statsHtml` 渲染在 divider 下方 |

---

## 13. 情绪节律

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 425 | `emoDict_c` 四类关键词：平静🔵/温暖🟠/热烈🔴/自然🟢 |
| 425 | 每时段遍历描述，匹配四类关键词→命中最多=该段情绪；兜底标签`沉静` |
| 432 | `sendJson({..., sessionEmotions, ...})` |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 350 | `m = await fetch('/api/memory/mood')`（`m.sessionEmotions` 即情绪数据） |
| 458 | 情绪节律渲染区域 |
| 458 | `.slice(-10)` 取最后10段 |
| 458 | `emoMap` 颜色映射，柱高 `Math.min(60, max(12, charCount*3))px` |
| 460 | 图例四色 |

---

## 14. 本月关键词

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 310-314 | 从全量 descriptions 统计 `kwList` 词频，公式 `cnt/totalFrames*40`，上限28px下限8px，<20%过滤 |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 411-416 | 遍历 `r.keywords`，解析 `font-size`，限制 8-28px |
| 462 | `<div class="kw-cloud">' + kwHtml` |

---

## 15. 最常去的地方

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 310-313 | `topPlaces` 按 `sceneWords` 匹配描述第一句→按帧数排序取前8，含比例柱状图 |

**已知 bug：**
1. **单位错误** — `count: cnt+' 段'` 应该是 `cnt+' 帧'`。`cnt` 是按帧计数的（每帧最多计1次），不是按时段
2. **"办公"和"办公室"重叠匹配** — 如果某帧第一句出现"办公室"，会同时命中"办公"和"办公室"，导致同一帧被计两次。但当前数据里没有帧第一句包含"办公室"（都是"办公桌""办公环境"），所以暂不影响

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 420-424 | 遍历 `r.topPlaces`，拼柱状图 |
| 464 | `<div class="places-grid"><div>最常去的地方</div>...` |

---

## 16. 陪伴你的人

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 320-332 | `charList` 正则匹配（绿衣男子/橘猫/白猫/黑衣男子/白衣男子/蓝衣男子/游戏角色/机器狗）按次数排序取前6 |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 428-432 | 遍历 `r.people` 渲染人物卡片 |
| 464 | people 列表在 places-grid 右侧 |
