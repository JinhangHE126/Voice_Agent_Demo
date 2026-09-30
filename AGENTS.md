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

## 2. 目录结构（必须保持同级）

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

| 路径 | 作用 |
|---|---|
| `scripts/run_web.py` | 启动 FastAPI/Uvicorn |
| `app/web.py` | HTTP + WebSocket 入口 |
| `app/orchestrator.py` | 会话编排：VAD、ASR、打断、filler、TTS |
| `app/vad.py` | WebRTC + 能量门限 + 强制结束/滑动窗口 |
| `app/services/asr_qwen.py` | Qwen 实时 ASR（含 final 超时兜底） |
| `app/services/tts_minimax.py` | MiniMax 流式 TTS |
| `app/services/prompt_audio.py` | 本地 `audio/<lang>/<prompt>.wav` 路由 |
| `app/config.py` | 环境变量 / Settings |
| `static/index.html` | 前端听筒页 + 订单草稿展示 |
| `audio/yue/` | 粤语预制录音（mono 16k PCM wav） |
| `DEPLOY_EC2.md` | EC2 + Cloudflare Tunnel 演示步骤 |
| `.env` / `.env.example` | 配置 |

### repair-order-agent 关键文件

| 路径 | 作用 |
|---|---|
| `repair_order/domain/repair_order.py` | `RepairOrderDraft` 槽位与状态 |
| `repair_order/services/dialog_manager.py` | 状态机、多语言话术、追问优先级 |
| `repair_order/services/info_extractor.py` | 规则 + LLM JSON 抽取 |
| `repair_order/repositories/order_repository.py` | 内存 + SQLite 持久化 |
| `开发文档.md` | 产品/架构设计（中文长文） |
| `tests/` | 业务单测 |

---

## 3. 一键本地运行

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

## 4. 核心业务逻辑（改代码必读）

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

## 5. 语音编排注意点（orchestrator）

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

## 6. 测试

```powershell
cd repair-order-agent
pytest -q

cd ..\voice-agent-demo
pytest -q
```

改对话/抽取后务必跑 `repair-order-agent` 测试。  
改 VAD/编排后跑 `voice-agent-demo` 测试。

---

## 7. 改动约定

1. **最小改动**：只改任务相关文件；不顺手大重构。
2. **业务规则写在 `repair-order-agent`**；语音实时链路写在 `voice-agent-demo`。
3. **不要把抽取和开口 LLM 混成一个模型调用。**
4. **没有齐四槽不得 completed**（含 address）。
5. **不要提交真实密钥到公开仓库**；本包若含 `.env`，仅限私有使用。
6. 用户未要求时不要主动 `git commit` / `push`。
7. Windows 环境：PowerShell；路径注意 `repair-order-agent` 同级。

---

## 8. 常见故障速查

| 现象 | 原因 | 处理 |
|---|---|---|
| 页面有草稿 UI 但行为像旧版 | 服务未重启 | Ctrl+C 后重跑 `run_web.py`，浏览器强刷 |
| 一直 `endpointing` | ASR final 未回 | 查超时兜底 / DashScope 网络 |
| 说完变 listening 不说话 | THINKING 被误打断或空识别 | 查 barge-in / asr.final empty |
| 公网/手机无麦克风 | 非 HTTPS | Cloudflare Tunnel 或正式域名证书 |
| 502 via Tunnel | 7862 服务没开 | 先 `python scripts/run_web.py` 再开 cloudflared |
| Import repair_order 失败 | 目录不同级 | 保持两个包同级 |

---

## 9. 建议的下一步（产品）

按优先级：

1. 补齐 `audio/yue/*.wav` 固定话术与短 filler
2. 地址必填硬校验（演示防漏）
3. 沉默超时 / 单槽重试上限 / escalate 事件
4. 多语言录音目录 `zh/`、`en/`
5. 合并为单一可部署包（去掉 sys.path 旁路）
6. 正式域名 HTTPS（替换临时 trycloudflare）

更完整的状态机与设计背景：`repair-order-agent/开发文档.md`。

---

## 10. 给智能体的工作方式

接到任务时：

1. 先读本文件 + 相关模块（上表）
2. 改代码后跑对应 `pytest`
3. 需要用户可见行为变更时，提示**重启 `run_web.py` + 强刷浏览器**
4. 回复用户用简洁中文；代码注释/提交信息按用户要求
5. Ask 模式只分析不改文件；Agent 模式才落地修改

**入口命令备忘：**

```powershell
cd <repo>/voice-agent-demo
.\.venv\Scripts\Activate.ps1
python scripts\run_web.py
```
