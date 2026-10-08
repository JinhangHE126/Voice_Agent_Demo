# AGENTS.md — Voice Agent Demo 上手指南

> 给其他智能体 / 新协作者：读完本文件即可直接改代码、跑服务、做测试。  
> 工作目录默认：本仓库根目录 `Voice_Agent_Demo/`。

---

## 1. 项目是什么

实时语音维修客服 Demo：

```text
浏览器麦克风
→ WebSocket PCM16
→ Qwen ASR（DashScope）
→ DialogManager（维修工单状态机）
→ 本地预制 wav（固定话术）或 MiniMax TTS（动态句）
→ 浏览器播放
```

业务目标：多轮收集并落库：

1. `repair_description`（维修内容）
2. `customer_phone`（电话）
3. `customer_address`（地址）
4. `customer_name`（姓名）

然后整句确认 → 写 SQLite → 结束语。

语言：粤语为主，支持普通话 / 英文跟随。

---

## 2. 完整端到端流程

从浏览器开麦到工单落库的完整路径（改实时链路或业务时对照此节）。

### 2.1 总览

```text
点「开始」
→ 建 WebSocket + 开麦克风
→ 服务端播开场白（不可打断）
→ 循环：听 → 识别 → 抽字段 → 追问/确认 → 写库
→ 播「工单已保存」→ 结束
```

一轮用户话的时序：

```text
浏览器麦克风
  → WebSocket PCM16
  → VAD 判说完
  → Qwen ASR final
  → DialogManager + 抽取器更新草稿
  → （可选）filler wav
  → 本地 wav 或 MiniMax TTS
  → 浏览器播放
  → playback.done
  → 再听下一句
  …
  → 用户确认
  → SQLite 工单
  → order_saved 结束语
```

两层状态不要混用：


| 层                  | 状态                                                            | 职责          |
| ------------------ | ------------------------------------------------------------- | ----------- |
| 语音会话（orchestrator） | `LISTENING → ENDPOINTING → THINKING → SPEAKING → LISTENING …` | 何时听 / 何时说   |
| 工单（DialogManager）  | `COLLECTING → CONFIRMING → SAVING → COMPLETED`                | 槽位齐不齐、确认、写库 |




### 2.2 会话启动

1. 浏览器打开页面，点开始。
2. 连上 `ws://host/ws`（HTTPS 用 `wss`）；入口：`app/web.py` → `CallSession`。
3. `CallSession.start()`：
  - 建 ASR（`AGENT_MODE=cloud` 用 Qwen，否则 mock）
  - 预热 TTS
  - 维修模式下立刻播 greeting，并设 `_greeting_uninterruptible = True`（开场期间麦克风音频直接丢弃）
4. 浏览器同时 `getUserMedia`：回声消除、降噪、AGC；PCM 降到 16k，二进制帧发到服务端。
5. 前端收到 `tts.audio` 就播；播完发 `playback.done`，服务端回到 `LISTENING`，开始真正听用户。



### 2.3 检测声音（听一轮）

状态：`LISTENING`。每个 PCM 包同时：

1. 推给 ASR（流式 partial，前端可显示中间结果）
2. 进本地 VAD（`app/vad.py`：WebRTC + 自适应能量门限）

VAD 事件：

- `speech_start`：用户开始说
- 说话中持续推音频
- `speech_end`：静音约 `vad_silence_ms`（默认 650ms）/ 达 `vad_max_speech_ms` / 滑动窗口有声比例过低 → 状态变 `ENDPOINTING`，调用 `asr.end_utterance()` 要 final

若 ASR 长时间无进度，`asr_no_progress_timeout_ms` watchdog 可强制结束本轮，避免卡死。

### 2.4 识别出文字

1. Qwen（或 mock）回 `asr.final`（文本 + 语言）。
2. **空识别**：回到 `LISTENING`，不推进工单（避免状态机饿死）。
3. **有文字**：
  - 发 `asr.final` 给前端展示
  - 状态 → `THINKING`
  - 进入 `_run_reply`



### 2.5 业务处理（抽槽 + 状态机）

`DialogManager.process_user_text_async`（`repair-order-agent`）：

1. **抽取**：规则 +（可选）LLM → 结构化 JSON
  `name / phone / address / repair / intent / confirm|deny ...`  
   抽取器只出 JSON，**不负责聊天**。
2. **写入草稿**（置信度够才写）：四槽按优先级追问缺项。
3. **电话无效**：有号码候选但校验失败 → `invalid_phone_length` / `invalid_phone_generic`，不要当没听到。
4. **四槽齐全** → `CONFIRMING`，生成整句确认（动态 TTS，含变量）。

等待抽取时，orchestrator 可并行播短 filler（`ack_*.wav`）。  
前端收 `order.draft`，右侧草稿实时更新。

### 2.6 开口回复

根据 `ResponsePlan`：


| 类型                            | 做法                                   |
| ----------------------------- | ------------------------------------ |
| 固定话术（`ask_phone`、`greeting`…） | 优先播本地 `audio/<lang>/<prompt_id>.wav` |
| 动态句（确认句）                      | MiniMax 流式 TTS                       |


流程：

1. 状态 → `SPEAKING`
2. 音频块以 `tts.audio` 推前端播放
3. **SPEAKING** 中用户大声说话可 barge-in（时长 + 能量阈值）；**THINKING** 期间故意不打断
4. 前端 `playback.done`（服务端有 `playback_done_grace_ms` 防过早）→ 回 `LISTENING`
5. 下一轮重复「听 → 识别 → 抽取 → 说」



### 2.7 确认与落库

四槽齐全后助手读确认句。用户回应：


| 用户意图        | 结果                                                                          |
| ----------- | --------------------------------------------------------------------------- |
| 啱 / 对 / yes | `CONFIRMING` → `SAVING` → `repository.save()` → `COMPLETED`，播 `order_saved` |
| 唔啱 + 指定字段   | 只清该槽，回 `COLLECTING`，再问那个字段                                                  |
| 纯否认未说改哪     | 问 `ask_correct_field`                                                       |
| 确认阶段又改了字段   | 更新后回收集                                                                      |


落库：`call_id` + 姓名 / 电话 / 地址 / 维修描述 / 状态 →  
默认 `voice-agent-demo/output/repair_orders.sqlite`（`REPAIR_ORDER_SQLITE_PATH` 可改）。





### 2.8 循环各阶段：技术栈与优化

对应「听 → 识别 → 抽字段 → 追问/确认 → 写库」。此处多为语音客服工程优化（延迟 / 鲁棒 / 可控），不是论文级新算法。

#### 听（端点检测）

**技术**

- 前端：`getUserMedia` + Web Audio，PCM16 @ 16 kHz，经 WebSocket 二进制帧上传
- 浏览器侧：回声消除 / 降噪 / AGC
- 服务端：`EnergyVAD`（`app/vad.py`）

**算法 / 策略**


| 手段                              | 作用                                                          |
| ------------------------------- | ----------------------------------------------------------- |
| WebRTC VAD（mode 0–3）            | 逐 20ms 帧判「像不像语音」                                            |
| 自适应能量门限                         | `noise_floor` 用 EMA 学环境底噪；阈值 = `max(固定门限, 底噪 × multiplier)` |
| 最短语音时长 `min_speech_ms`          | 过滤短脉冲噪声，避免误开麦                                               |
| 静音结束 `vad_silence_ms`（默认 650ms） | 句间逗号停顿仍拼成一句                                                 |
| 最长说话 `vad_max_speech_ms`        | 防一轮说太久卡死                                                    |
| 滑动窗口有声比例                        | 窗口内有声帧过少 → 提前 `speech_end`，处理「越说越轻」                         |
| ASR progress watchdog           | 长时间无 partial → 强制 `end_utterance`                           |


**为何本地 VAD、不用云端 VAD**：云端静音阈值短，容易在逗号处切半句；本地管边界，ASR 设 `turn_detection: null` 做手动 commit。

#### 识别（ASR）

**技术**

- DashScope WebSocket：`qwen3-asr-flash-realtime`
- 音频：`input_audio_buffer.append`（base64 PCM）→ `commit` → 等 `transcription.completed`
- 语言：空配置 = 粤语 / 普通话 / 英文自动检测

**优化**


| 手段                  | 作用                                                      |
| ------------------- | ------------------------------------------------------- |
| 流式 partial          | 前端边说边显示，降低「没反应」感                                        |
| Final 超时兜底（默认 2.5s） | 云端不回 completed 时，用最后 partial 当 final，避免卡在 `ENDPOINTING` |
| 断线重连 + cooldown     | 发送失败自动重连，不拆浏览器会话                                        |
| 空 final 仍回调         | 回 `LISTENING`，状态机不饿死                                    |
| 语言跟随                | ASR 语言 + 粤语助词启发式 → TTS `language_boost`                 |




#### 抽字段

**技术**

- `info_extractor.py`：**规则优先，LLM 补全**
- LLM：DashScope OpenAI 兼容接口（如 `qwen-plus`），异步调用，超时约 3s
- 输出：结构化 JSON（name/phone/address/repair/intent/confirm…），不生成客服话术

**规则侧启发式与校验**


| 模块    | 做法                                                            |
| ----- | ------------------------------------------------------------- |
| 电话    | 正则抓数字；中文数字映射（一二三…）；口述纠错（幺→1、拐→7）；长度校验（港 8 / 内地 11 / +852/+86） |
| 姓名    | 「我叫…」「陈先生」等模式；去掉口语尾巴（就可以了）                                    |
| 地址    | 关键词 + 长度/有效性校验                                                |
| 维修    | 维修/坏/整等 marker + 清洗、摘要                                        |
| 确认/否认 | 词表：啱/对/yes vs 唔啱/错/no                                         |
| 语言    | 粤语粒子、拉丁字母比例等启发式                                               |
| 置信度门控 | 字段 confidence ≥ 约 0.55–0.8 才写草稿，防误写                           |


**优化**

1. **规则够用就不调 LLM**（确认、否认、纯数字电话等），省延迟。
2. **expected_field 上下文**：正在问电话时才开 fuzzy 映射，避免「公司」被误成数字。
3. **invalid phone 单独话术**：有候选但非法 → 针对性追问，不当「没听到」。
4. **与 filler 并行**：抽取跑着时先播 `ack_*.wav`，掩盖等待。



#### 追问 / 确认（对话策略）

**技术**

- 确定性状态机：`DialogManager`（不是自由聊天 LLM）
- 状态：`COLLECTING → CONFIRMING → SAVING → COMPLETED`
- 槽位优先级：`repair → phone → address → name`

**策略逻辑**

- 缺哪个问哪个；一句话可填多槽
- 四槽齐 → 整句确认（动态 TTS，含变量，不拆碎片播）
- 确认：啱 → 写库；唔啱 + 字段 → 只清该槽再问；纯否认 → `ask_correct_field`

**开口侧优化（体感「快」）**


| 手段                  | 作用                                       |
| ------------------- | ---------------------------------------- |
| 固定话术本地 wav          | `greeting`/`ask_*`/`order_saved` 几乎零合成延迟 |
| MiniMax 流式 TTS      | 确认句边生成边播；会话预热 WebSocket                  |
| filler 轮换           | 长句才播；短答（「陈先生」「啱」）不播                      |
| barge-in            | SPEAKING 时能量 + 最短时长；THINKING 不打断防误取消     |
| generation_id       | 打断后旧音频/旧回复作废                             |
| playback.done grace | 防浏览器过早报播完                                |
| 本地 wav 按实时节奏切块推送    | 保留可打断窗口                                  |


目标：EoU → 首包可播音频尽量 ≤ 1s（云路径；见 `CallSession` 注释）。

#### 写库

**技术**

- `OrderRepository`：内存列表 + SQLite（`repair_orders.sqlite`）
- 触发：`CONFIRMING` + 用户确认 + `draft.is_complete()` → `SAVING` → `save()` → `COMPLETED`

**约束**：四槽不齐不写；确认后才落库；纠错只清指定槽，不清整单。

#### 整环技术栈一览

```text
浏览器 WebAudio / WebSocket
        ↓ PCM16
本地 VAD（WebRTC + 自适应能量 + 静音/窗口策略）
        ↓ end_utterance
Qwen 实时 ASR（流式 + final 超时兜底）
        ↓ 文本
规则抽取 ± LLM JSON 抽取（置信度门控）
        ↓ 更新草稿
DialogManager 状态机（追问优先级 / 确认纠错）
        ↓ ResponsePlan
本地 wav 或 MiniMax 流式 TTS（+ filler）
        ↓ 用户确认
SQLite 工单
```

优化维度归纳：

1. **延迟**：本地 VAD、规则短路、filler、预录音、TTS 预热、流式播
2. **鲁棒**：自适应噪声门、ASR 超时、空 final、电话 fuzzy、置信度
3. **可控**：状态机 + 槽位优先级，不用大模型直接决定是否写库
4. **体验**：打断、开场不可打断、语言跟随、确认句整句读

---



## 3. 目录结构（必须保持同级）

```text
Voice_Agent_Demo/
  AGENTS.md                 ← 本文件
  README.md
  voice-agent-demo/         ← 实时语音 + Web UI（启动入口）
  repair-order-agent/       ← 工单领域逻辑（被 voice-agent-demo 通过 sys.path 引用）
```

**硬依赖：** `voice-agent-demo/app/orchestrator.py` 会把同级 `repair-order-agent` 加入 `sys.path`。  
两个目录必须始终同级，不要拆开只部署其中一个。

### voice-agent-demo 关键文件


| 路径                             | 作用                                |
| ------------------------------ | --------------------------------- |
| `scripts/run_web.py`           | 启动 FastAPI/Uvicorn                |
| `app/web.py`                   | HTTP + WebSocket 入口               |
| `app/orchestrator.py`          | 会话编排：VAD、ASR、打断、filler、TTS        |
| `app/vad.py`                   | WebRTC + 能量门限 + 强制结束/滑动窗口         |
| `app/services/asr_qwen.py`     | Qwen 实时 ASR（含 final 超时兜底）         |
| `app/services/tts_minimax.py`  | MiniMax 流式 TTS                    |
| `app/services/prompt_audio.py` | 本地 `audio/<lang>/<prompt>.wav` 路由 |
| `app/config.py`                | 环境变量 / Settings                   |
| `static/index.html`            | 前端听筒页 + 订单草稿展示                    |
| `audio/yue/`                   | 粤语预制录音（mono 16k PCM wav）          |
| `DEPLOY_EC2.md`                | EC2 + Cloudflare Tunnel 演示步骤      |
| `.env` / `.env.example`        | 配置                                |




### repair-order-agent 关键文件


| 路径                                              | 作用                       |
| ----------------------------------------------- | ------------------------ |
| `repair_order/domain/repair_order.py`           | `RepairOrderDraft` 槽位与状态 |
| `repair_order/services/dialog_manager.py`       | 状态机、多语言话术、追问优先级          |
| `repair_order/services/info_extractor.py`       | 规则 + LLM JSON 抽取         |
| `repair_order/repositories/order_repository.py` | 内存 + SQLite 持久化          |
| `开发文档.md`                                       | 产品/架构设计（中文长文）            |
| `tests/`                                        | 业务单测                     |


---



## 4. 一键本地运行

```powershell
cd voice-agent-demo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\run_web.py
```

打开：`http://127.0.0.1:7862/`

关键配置：

- `AGENT_MODE=cloud`：真实 ASR/TTS
- `AGENT_MODE=mock`：无云密钥架构演示
- `BUSINESS_MODE=repair_order`：维修工单模式（默认）
- `HOST=0.0.0.0`：允许外网/隧道访问；本地浏览器用 `127.0.0.1` 即可麦克风

公网 `http://公网IP` **不能用麦克风**（浏览器安全策略）。手机演示需 HTTPS（见 `DEPLOY_EC2.md` Cloudflare Tunnel）。

---



## 5. 核心业务逻辑（改代码必读）



### 字段优先级（死规则）

```text
repair_description → customer_phone → customer_address → customer_name
```

实现：`DialogManager._FIELD_PRIORITY`。

### 状态

```text
COLLECTING → CONFIRMING → SAVING → COMPLETED
```



### 抽取 vs 开口（必须分开）

- **抽取器** `info_extractor.py`：只输出结构化 JSON（name/phone/address/repair/intent/...），不聊天。
- **开口**：固定 `prompt_id` 本地 wav，或动态 TTS（确认句含变量，禁止拆成多段碎片拼接）。



### 确认纠错

- `啱/对/yes` → 写库
- `唔啱` + 指定字段 → 只清该槽，回 COLLECTING
- 纯否认未指定 → `ask_correct_field`



### 电话校验

- 支持香港 8 位、内地 11 位、+852/+86 变体
- 无效号码走 `invalid_phone_length` / `invalid_phone_generic`，不要直接当没听到



### 语言

- `session_language`: `yue | cmn | en`
- 录音目录：`audio/yue/`、`audio/zh/`、`audio/en/`（目前主要 yue）
- TTS `language_boost`：粤语 `Chinese,Yue`，普通话 `Chinese`，英文 `English`

---



## 6. 语音编排注意点（orchestrator）

已实现、改时不要回退：

1. **开场不可打断**：`_greeting_uninterruptible`，从 session start 到 greeting 播完忽略麦克风。
2. **THINKING 禁止 barge-in**：避免环境噪音取消待回复。
3. **SPEAKING barge-in** 有时长 + 能量阈值。
4. **ASR final 超时**：`dashscope_asr_final_timeout_ms`，防止卡在 `endpointing`。
5. **空 ASR final 也回调**：避免状态机饿死。
6. **playback.done 过早忽略**：`playback_done_grace_ms`。
7. **filler**：等待抽取时播 `ack_*.wav`；长句曾优先 `ack_checking`，文案宜短、勿每轮“确认一下”。

Prefabricated wav 要求：

- mono / 16kHz / 16-bit PCM
- 文件名 = `prompt_id`，如 `ask_address.wav`
- 放 `voice-agent-demo/audio/yue/`

推荐优先预制：`greeting`、`ask_repair`、`ask_phone`、`ask_address`、`ask_name`、`order_saved`、短 filler。  
`confirm_order` 继续动态 TTS（含变量）。

---



## 7. 测试

```powershell
cd repair-order-agent
pytest -q

cd ..\voice-agent-demo
pytest -q
```

改对话/抽取后务必跑 `repair-order-agent` 测试。  
改 VAD/编排后跑 `voice-agent-demo` 测试。

---



## 8. 改动约定

1. **最小改动**：只改任务相关文件；不顺手大重构。
2. **业务规则写在** `repair-order-agent`；语音实时链路写在 `voice-agent-demo`。
3. **不要把抽取和开口 LLM 混成一个模型调用。**
4. **没有齐四槽不得 completed**（含 address）。
5. **不要提交真实密钥到公开仓库**；本包若含 `.env`，仅限私有使用。
6. 用户未要求时不要主动 `git commit` / `push`。
7. Windows 环境：PowerShell；路径注意 `repair-order-agent` 同级。

---



## 9. 常见故障速查


| 现象                     | 原因                | 处理                                           |
| ---------------------- | ----------------- | -------------------------------------------- |
| 页面有草稿 UI 但行为像旧版        | 服务未重启             | Ctrl+C 后重跑 `run_web.py`，浏览器强刷                |
| 一直 `endpointing`       | ASR final 未回      | 查超时兜底 / DashScope 网络                         |
| 说完变 listening 不说话      | THINKING 被误打断或空识别 | 查 barge-in / asr.final empty                 |
| 公网/手机无麦克风              | 非 HTTPS           | Cloudflare Tunnel 或正式域名证书                    |
| 502 via Tunnel         | 7862 服务没开         | 先 `python scripts/run_web.py` 再开 cloudflared |
| Import repair_order 失败 | 目录不同级             | 保持两个包同级                                      |


---



## 10. 建议的下一步（产品）

按优先级：

1. 补齐 `audio/yue/*.wav` 固定话术与短 filler
2. 地址必填硬校验（演示防漏）
3. 沉默超时 / 单槽重试上限 / escalate 事件
4. 多语言录音目录 `zh/`、`en/`
5. 合并为单一可部署包（去掉 sys.path 旁路）
6. 正式域名 HTTPS（替换临时 trycloudflare）

更完整的状态机与设计背景：`repair-order-agent/开发文档.md`。

---



## 11. 给智能体的工作方式

接到任务时：

1. 先读本文件（含 §2 完整流程）+ 相关模块（上表）
2. 改代码后跑对应 `pytest`
3. 需要用户可见行为变更时，提示**重启** `run_web.py` **+ 强刷浏览器**
4. 回复用户用简洁中文；代码注释/提交信息按用户要求
5. Ask 模式只分析不改文件；Agent 模式才落地修改

**入口命令备忘：**

```powershell
cd <repo>/voice-agent-demo
.\.venv\Scripts\Activate.ps1
python scripts\run_web.py
```

