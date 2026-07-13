# 数据流完整对照（memory.js ↔ stmem.html）

> 使用 `// ST-N:` 标记在源码中定位，搜索 `ST-` 即可
> 基于 commit `d15819e`（main 首合）

## 1. mood 标题：`2026年07月，生活平淡如水`

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 486 | `/api/memory/mood` 路由入口，`sessions` 按时段数分档：≤10忙碌 / ≤30平淡 / ≤60有趣 / >60狂热 |
| 488 | `moodText` + `topPlace` / `topPerson` / `activeDays` 同路由计算 |
| 495 | `sendJson({..., mood, moodText, ...})` |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 351 | `const m = await fetch('/api/memory/mood')` |
| 394 | `var mood = m.mood \|\| ''` |
| 455 | `<div class="rep-title">' + y + '年' + mo + '月，<br>' + (m.mood\|\|'')` |

---

## 2. 事件数：`28个事件`

**前端 stmem.html**（纯前端计算，不依赖后端）
| 行 | 内容 |
|----|------|
| 359-391 | 全部事件计数逻辑 |
| 362-376 | `getTheme(desc)` 按描述第一句关键词打主题标签（工作/宠物/街景/美食/走廊/特写/深夜） |
| 378-383 | 按时间排序→相邻帧主题不同=新事件 |
| 448 | `eventCount` 填入副标题 `'... 由 AI 根据 ' + eventCount + ' 个事件...'` |

---

## 3. 帧数：`108帧`

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 488 | `rows = db.prepare("SELECT capture_time, description FROM memories...").all()` |
| 402 | `frameRows` → `frameList`（ST-3标记） |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 352 | `r.frames` 取 `report` API 的 frames 数组 |
| 448 | `r.frames.length` 填入副标题 |

---

## 4. 天数：`这个月你的相机记录了8天的生活`

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 488 | `activeDays = days.size`（去重的有记录的天数） |
| 495 | `sendJson({..., activeDays, ...})` |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 392 | `var activeDays = m.activeDays \|\| '--'` |
| 458 | `<p class="narrative">...<b>' + activeDays + ' 天</b>的生活...` |

---

## 5. 叙事正文

**后端 memory.js** `/api/memory/narrative` 路由
| 行 | 内容 |
|----|------|
| 529-550 | 从 SQLite 取所有描述第一句→每50帧一组→写`/tmp/narr_input.txt`→调`reportsupports/scripts/NarrFifty.py`→gemma4:e2b（借100的Ollama，`192.168.0.100:11434`）→各组生成→融合→返回`{narrative}` |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 450 | `fetch("/api/memory/narrative")` 异步获取 |
| 458 | `(d.narrative\|\|'')` 插入段落 |

---

## 6. topPlace：`走廊`

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 488 | mood 路由内，用11个地点词正则匹配全部描述文本，出现最多的= `topPlace` |
| 495 | `sendJson({..., topPlace, ...})` |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 390 | `var topPlace = m.topPlace \|\| ''` |
| 458 | `出现最多的地方是 <b style="color:#5E7A18;">' + (r.topPlaces&&r.topPlaces.length?r.topPlaces[0].name:'---')` |

---

## 7. topPerson：`一位白衣男士`

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 488 | mood 路由内，动态提取人/动物（颜色+衣物+性别），出现最多的= `topPerson`，加量词（一只/一位/一个） |
| 495 | `sendJson({..., topPerson, ...})` |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 391 | `var topPerson = m.topPerson \|\| ''` |
| 458 | `陪伴你最久的是 <b style="color:#C7700E;">' + (topPerson\|\|'---')` |

---

## 8. moodText：`你这个月的生活平淡如水`

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 488 | `moodText` 五档文案 |
| 495 | `sendJson({..., moodText, ...})` |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 394 | `var moodText = m.moodText \|\| mood` |
| 458 | `整体而言，<b>' + moodText + '</b>` |

---

## 9-12. 四张统计卡片

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 300 | `locWords` 定义（~60个地点词）|
| 303 | `spaceCount` = 本月第一句匹配到的不同地点词数（ST-9）|
| 305 | `recordDuration` = 首帧到末帧的有效拍摄时长（ST-10）|
| 307 | `weekNewCount` = 本周新增地点类型（ST-11）|
| 301 | `realMemCount` = `memories` 表总行数（ST-12）|
| 395-419 | `sendJson({..., stats:[...], ...})` |

| 卡片 | backend 数据源 | 显示值 |
|------|---------------|--------|
| 本月记录场景数 | `stats[0].value` = `spaceCount` | 7 |
| 记录时长 | `stats[1].value` = `recordDuration` | 1h21m8s |
| 本周新场景 | `stats[2].value` = `'+'+weekNewCount` | +4 |
| 物件数 | `stats[3].value` = `realMemCount` | 108 |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 352 | `const r = await fetch('/api/memory/report')` |
| 404-406 | 遍历 `r.stats` 拼 `statsHtml` |
| 460 | `<div class="num-grid">' + statsHtml` |

---

## 13. 情绪节律

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 488 | `emoDict_c` 四类关键词：平静🔵/温暖🟠/热烈🔴/自然🟢 |
| 494 | 每时段遍历描述，匹配四类关键词→命中最多=该段情绪；兜底标签`沉静` |
| 495 | `sendJson({..., sessionEmotions, ...})` |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 351 | `m = await fetch('/api/memory/mood')`（`m.sessionEmotions` 即情绪数据） |
| 465 | 情绪节律渲染，`.slice(-10)` 取最后10段 |
| 465 | `emoMap` 颜色映射，柱高 `Math.min(60, max(12, charCount*3))px` |
| 466 | 图例四色 |

---

## 14. 本月关键词

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 322 | 从全量 descriptions 统计 `kwList` 词频，公式 `cnt/totalFrames*40`，上限28px下限8px，<20%过滤 |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 412-417 | 遍历 `r.keywords`，解析 `font-size`，限制 8-28px |
| 468 | `<div class="kw-cloud">' + kwHtml` |

---

## 15. 最常去的地方

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 315-318 | `topPlaces` 按 `sceneWords` 匹配描述第一句→按帧数排序取前8，含比例柱状图。合并"办公"→"办公室" |
| 315 | 已知问题：词表方式，LLM 描述可能匹配不到新地点词汇 |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 421-425 | 遍历 `r.topPlaces`，拼柱状图 |
| 471 | `<div class="places-grid"><div>最常去的地方</div>...` |

---

## 16. 陪伴你的人

**后端 memory.js**
| 行 | 内容 |
|----|------|
| 323-388 | **动态提取**：颜色+衣物→性别→标识，或颜色+动物名。不依赖预设词表。含最后一次出现日期 |
| 385 | `people = chars.map(...)` → `{initial, name, count, sub:'最近 X月X日', bg}` |

**前端 stmem.html**
| 行 | 内容 |
|----|------|
| 429-433 | 遍历 `r.people` 渲染人物卡片（MW风格：40px头像+名字+sub+计数） |
| 471 | people 列表在 places-grid 右侧 |
