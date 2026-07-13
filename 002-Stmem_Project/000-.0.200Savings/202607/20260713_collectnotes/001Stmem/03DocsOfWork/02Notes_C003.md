# C003 工作笔记

## 环境
- **200** = 192.168.0.200（本机，服务端）
- **100** = 192.168.0.100（LinuxC 系列，Ollama 在此：`192.168.0.100:11434`）
- **220** = MacC（zhx 的 Mac，原 244）

## Git 信息
- 远程：`git@github.com:ysh12304124/stmem`
- 分支：`zhx`（我们的分支）
- 提交格式：`MMDDHHmm描述`（如 `07130930lifecycle_90d`）
- 组长：ysh（main 分支）
- 推远程：`git push origin zhx --force`

---

## 002 留下的教训（已消化）

### 技术教训

**1. 叙事生成：不要硬拼，交给模型**
- 错误：从描述里随机选句拼接
- 正确：把所有描述第一句喂 gemma4:e2b，让模型自己组织语言
- VLM 描述已经有诗意第一句了，不要额外加工

**2. 场景数：不要造分类，不要预设词表，不要正则提取**
- 三次翻车：getTheme 分类→"宠物不是场景"；正则提取→噪声太大；预设词表→被否
- 最终：接受词表，但统一提取逻辑（只看描述第一句）

**3. 关键词公式：用比值不用绝对值**
- `count/totalFrames * 40` 优于 `12 + cnt * 2`
- 70% 以上 = 28px，20% 以下过滤

**4. 情绪检测**
- 4 色系统（平静/温暖/热烈/自然）+ 沉静兜底
- 词库缩减后才有效（去掉泛词）
- "温暖的光线"类通杀→需主动平衡

**5. 事件计数：简单主题聚类即可**
- 相邻帧主题不同 = 新事件，不回头合并
- 不考虑时间间距（时间戳可能伪造）

**6. gemma4:e2b 的特性**
- 纯文本输入可用（传 options 会导致空返回 ⚠️）
- 图生文时 options 正常 ✅
- 上下文 ~8K tokens，50 帧/组分批生成→融合
- 调用 100 机器的 Ollama（`192.168.0.100:11434`）

**7. 路由清理**
- 同路径路由，先注册的赢
- 删代码注意 try-catch 范围

### 流程教训
- **先问再动手** — 定义让用户定，不要猜
- **改动前先读代码** — 看完整上下文，尤其 try-catch 范围
- **重启确认** — 改完确认进程确实重启了
- **commit 格式** — `MMDDHHmm描述`，每个功能点独立 commit

---

## 已完成的改动

### 0710（周五）
- Git author 修复：`zhxbayernfan@gmail.com` → `1303425363@qq.com`（26 commits）
- 情绪节律：删除"每日"二字
- 底部备注：加圆角框、锁图标、"仅你本人可见"加粗
- 000Notes 目录整理：01Scripts / 02Produces / 03DocsOfWork / 99UnimportantScripts
- 文档行号更新

### 0711（周六）
- 叙事路径修复（memory.js → reportsupports/scripts/NarrFifty.py）
- Rebase main（29 commits 无冲突）

### 0713（周一）
- 生命周期归档阈值 7 天→90 天，数据恢复
- MiSans 字体部署 + reportsupports 目录
- 加载动画：iOS 菊花 spinner
- 搜索框自适应：fixed→flex 布局
- 动态人/动物识别（替换硬编码 charList）
- topPerson 同步修复（mood 路由与 report 路由一致）
- 陪伴你的人：MW 风格显示
- 备份文件清理
- ST-N 代码注释标记
- 00Program_Logic.md 文档同步
- PR 合入 main（首版）

---

## 文件索引
| 文件 | 说明 |
|------|------|
| `../02Produces/01Events_Noted.txt` | zhx 手动标注的 24 时段 37 事件 |
| `../02Produces/02Memories_*.txt` | 04 脚本导出的描述数据 |
| `../02Produces/03Memories_*.sqlite` | SQLite 备份 |
| `00MWReference.html` | 组长给的参考页面（19M） |
| `01Future_Plans.md` | TODO 列表 |
| `00Program_Logic.md` | 16 个数据流对照 |
| `../99UnimportantScripts/NarrFifty.py` | 叙事生成脚本 |

## 命名规范
- 6 位帧号（frame_001.jpg ~ frame_108.jpg）
- 脚本文件名：首字母大写驼峰
- commit message：`MMDDHHmm描述`
