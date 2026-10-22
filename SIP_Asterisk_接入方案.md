# SIP 接入方案：Asterisk + AudioSocket 对接现有 AI 流水线

> 目标：把现有「浏览器麦克风 → 实时语音维修客服」的 AI 流水线，通过 SIP 接入客户 PBX。
> 本文档描述**目标架构**与**实现顺序**，供逐步落地参考。

---

## 1. 背景与目标

现有系统是浏览器直连的实时语音链路：

```text
浏览器麦克风
  → WebSocket PCM16 @ 16kHz
  → Qwen ASR
  → DialogManager（维修工单状态机）
  → 本地预制 wav / MiniMax TTS
  → 浏览器播放
```

要改成客户 PBX 通过 SIP 打进来，复用同一套 `CallSession` 状态机与 `repair-order-agent` 业务逻辑。

**核心结论**：业务/状态机/ASR/TTS 全部复用，只新增一层「SIP 媒体传输」，不做整体重写。

---

## 2. 目标架构

```text
客户 PBX
   │
   │ SIP Trunk（INVITE + RTP）
   │ codec：G.711 / G.722 / Opus（由客户侧协商）
   ▼
Asterisk（Linux）
   │  chan_pjsip：SIP / RTP / 转码
   │  dialplan：Answer() → AudioSocket(<call_uuid>, host, port) → Hangup()
   │  内部：任意电话 codec → SLIN（8kHz mono PCM）
   │
   │ AudioSocket（TCP，Asterisk 主动连出做客户端）
   │  帧类型 0x10 = 8kHz / 16bit / mono PCM（每 20ms 一帧 320 字节）
   ▼
AI 媒体桥（Python asyncio TCP server，新增）
   │  每来电 = 一条 TCP 连接 = 一个 CallSession(call_id = uuid)
   │
   ├─ 收 AudioSocket 0x10（8k）→ 重采样 16k → 喂 VAD
   ├─ VAD → Qwen ASR（16k）→ DialogManager + 抽取器
   ├─ 本地 wav（16k）或 MiniMax TTS（16k）
   └─ 出音频 16k → 重采样 8k → 写回 AudioSocket 0x10
   │
   ▼
Asterisk 转回 RTP → 客户 PBX → 用户听筒
```

**关键点**：AudioSocket 默认走 8kHz（见 §4），所以桥内负责 8k↔16k 重采样；
现有 AI 流水线内部保持 16kHz 不变。

---



## 3. 分层职责


| 层    | 组件                          | 职责                                 |
| ---- | --------------------------- | ---------------------------------- |
| 电话接入 | 客户 PBX + Asterisk           | SIP、RTP、codec、振铃/应答/挂断、可选 DTMF     |
| 媒体传输 | AudioSocket + Python 桥      | 把「电话 PCM」变成「会话 PCM」，双向写帧           |
| 会话编排 | `CallSession`               | VAD / ASR / 打断策略 / filler / TTS 路由 |
| 业务   | `DialogManager` + 抽取器       | 四槽收集 → 确认 → SQLite                 |
| 语音能力 | Qwen ASR / 本地 wav / MiniMax | 识别与合成（均为 16k）                      |


Asterisk 只管电话域；Python 只管 AI 流水线。两者只通过 **8k PCM + call_uuid** 耦合。

---



## 4. 采样率约定（重要）



### 4.1 结论：生产默认 8k，桥内重采样

- `AudioSocket()` dialplan 应用**固定输出 8kHz**（16bit/8kHz/mono PCM，帧类型 `0x10`，20ms = 320 字节）。
- 官方应用层文档明确写：应用接口是 8k；更高采样率需走 **channel driver** 接口（`Dial(AudioSocket/.../c(slin16))`），且 16kHz 支持较新、踩坑多、有版本门槛。
- 因此：**生产先按 8k 默认接入，桥内做 8k↔16k 重采样**，最稳。



### 4.2 采样率由「消息类型」决定，不是 ID 协商

AudioSocket 协议里：

- `0x01` 载荷是 **UUID**（16 字节二进制），不是采样率协商。
- 采样率由**帧类型**决定：`0x10` = 8k slin，`0x12` = 16k slin16。

所以桥内**按帧类型解析**，不要写死「ID 消息带 sample_rate」。

### 4.3 帧格式

```text
每帧 = 3 字节头 + 载荷
  头：1 字节 type + 2 字节 big-endian 长度
```


| type   | 含义              | 载荷                     |
| ------ | --------------- | ---------------------- |
| `0x00` | 挂断（或直接关 socket） | 空（`0x00 0x00 0x00`）    |
| `0x01` | UUID            | 16 字节二进制               |
| `0x03` | DTMF            | 1 字节 ASCII             |
| `0x10` | 音频 8k slin      | 16bit LE mono PCM      |
| `0x12` | 音频 16k slin16   | 16bit LE mono PCM（暂不用） |
| `0xff` | 错误              | 可选错误码                  |




### 4.4 后续可选：原生 16k

等确认 Asterisk 版本含高采样率补丁（PR #1492，20/21/22/23 系），且用
`Dial(AudioSocket/.../c(slin16))` 验证通过后，再切原生 16k、去掉重采样。
在此之前不要依赖 `AudioSocket(...,slin16)`——它不是正规入口。

---



## 5. Python 侧目录结构

**不新建顶层 sibling 包**，先在 `voice-agent-demo` 内新增传输层，与现有 WebSocket 并列：

```text
voice-agent-demo/
  app/
    web.py                 # 现有浏览器入口（不动业务）
    orchestrator.py        # CallSession（小改：解耦输出回调）
    transports/            # 新增
      __init__.py
      base.py              # 传输抽象：入 PCM / 出 PCM / 播完 / 挂断
      resample.py          # 8k ↔ 16k 重采样
      audiosocket.py       # AudioSocket 帧编解码 + asyncio TCP server
      audiosocket_session.py  # 会话适配：连接 CallSession 与 AudioSocket
    scripts/
      run_audiosocket.py   # 启动桥（独立于 run_web.py）
```

浏览器 Demo 与 SIP 接入共用同一套 `CallSession`，只换传输实现。

### 5.1 传输抽象（`base.py`）

`CallSession` 目前与浏览器绑定：输出走 `emit("tts.audio", …)`，播完靠浏览器 `playback.done`。
SIP 没有浏览器，需要抽象出：

```text
CallSession
   · 入：on_pcm16(bytes)              ← WebSocket / AudioSocket 共用
   · 出：on_audio_chunk(pcm16_16k)    ← 抽象回调（不再绑死 emit tts.audio）
   · 播完：on_playback_complete()     ← 替代浏览器 playback.done
   · 挂断：close()
```

浏览器路径继续走原 `emit`；SIP 路径只挂这三个回调。

---



## 6. 一通电话时序

```text
1. 客户拨入 → PBX INVITE → Asterisk Answer
2. Asterisk: AudioSocket(uuid, 127.0.0.1, port)
3. Python accept → 读 0x01 UUID → new CallSession(call_id=uuid)
4. 播 greeting（本地 16k wav）→ 重采样 8k → 写 AudioSocket 0x10 → RTP 回客户
5. 循环：
     收 0x10(8k) → 重采样 16k → VAD → ASR final → DialogManager
     → 本地 wav / MiniMax(16k) → 重采样 8k → 按 20ms 切块写 0x10 → 客户听
6. 四槽确认 → SQLite → order_saved → 播完挂断
7. Asterisk BYE → 客户侧挂断
```

---



## 7. 实现顺序（分阶段）



### 阶段 0：不动业务，先立接口（最小改动）✅ 已完成

在 `CallSession` 增加可选回调，浏览器路径行为不变：

| 参数 | 默认 | 作用 |
|---|---|---|
| `on_audio_out` | `None` | 出 PCM（16k）时回调；SIP 侧写 AudioSocket |
| `auto_complete_playback` | `False` | 为 `True` 时本轮音频写完后自动 `on_playback_done`（按 PCM 时长等待），无需浏览器 `playback.done` |

实现要点：

- 所有 `tts.audio` 出口收拢到 `_deliver_audio`
- `app/transports/base.py`：传输抽象（Stage 1+ 实现）
- 验收：`pytest tests/test_orchestrator.py -q`（含 `test_auto_complete_playback_advances_without_browser_done`）

**不碰 Asterisk、不碰 DialogManager。**

### 阶段 1：重采样（纯函数，先测）✅ 已完成

`voice-agent-demo/app/transports/resample.py`：

| API | 作用 |
|---|---|
| `resample_up(pcm8k)` | 一次性 8k → 16k（测试 / 整段缓冲） |
| `resample_down(pcm16k)` | 一次性 16k → 8k |
| `make_upsampler()` / `make_downsampler()` | 返回带状态的 `RateConverter`（每通电话一对，跨 20ms 帧复用） |
| `AUDIOSOCKET_FRAME_BYTES_8K` | 320（20ms @ 8k PCM16） |

实现：`audioop.ratecv`（CPython 3.12 stdlib；3.13+ 需 `audioop-lts`）。

验收：`pytest tests/test_resample.py -q`（空输入、奇数字节对齐、长度约 2x/0.5x、流式≈一次性、往返能量）。

### 阶段 2：AudioSocket 协议层（不接业务）✅ 已完成

`voice-agent-demo/app/transports/audiosocket.py`：

| 组件 | 作用 |
|---|---|
| `encode_*` / `read_frame` / `Frame` | 协议编解码（`0x00` hangup / `0x01` UUID / `0x03` DTMF / `0x10` 8k 音频 / `0xff` error） |
| `chunk_pcm` / `send_audio_frames` | 按 320 字节切帧（末帧零填充） |
| `AudioSocketConnection` | 单连接读写 |
| `AudioSocketServer` | `asyncio.start_server`，Asterisk 主动连入 |
| `echo_handler` | 阶段 2 回环（收 `0x10` 原样回写） |

验收：`pytest tests/test_audiosocket.py -q`（编解码、echo 回环、UUID 捕获、对端直接关连接）。

**尚未**接 `CallSession` / 重采样（阶段 3）。

### 阶段 3：粘合（会话适配器）✅ 已完成

`voice-agent-demo/app/transports/audiosocket_session.py`：

```text
收 0x10(8k) → upsample 16k → CallSession.on_pcm16
CallSession 出 16k → downsample 8k → 320-byte 0x10（可选实时 pacing）
Hangup / session.completed → session.close() + 挂断
```

| 组件 | 作用 |
|---|---|
| `AudioSocketCallBridge` | 单连接粘合：重采样 + `CallSession(on_audio_out, auto_complete_playback)` |
| `handle_audiosocket_call` | `AudioSocketServer` 的 handler 入口 |
| `scripts/run_audiosocket.py` | 启动桥：`python scripts/run_audiosocket.py` |
| 配置 | `AUDIOSOCKET_HOST` / `AUDIOSOCKET_PORT`（默认 9092）/ `AUDIOSOCKET_PACE_AUDIO` |

`call_id` = AudioSocket UUID。`CallSession.on_audio_flush` 在自动播完前冲掉末帧（零填充到 320）。

验收：`pytest tests/test_audiosocket_session.py -q`（greeting+ask_name 出音频；用户语音 → mock ASR → repair_order 回复音频）。

本地 mock 试跑：

```powershell
cd voice-agent-demo
python scripts\run_audiosocket.py
```

### 阶段 4：接 Asterisk

Linux 上 dialplan（故意用应用，吃默认 8k）：

```text
exten => _X.,1,Answer()
 same => n,Set(UUID=${UNIQUEID})
 same => n,AudioSocket(${UUID},127.0.0.1,9092)
 same => n,Hangup()
```

Python 桥与 Asterisk **同机或同低延迟内网**（AudioSocket 走 TCP，勿出公网）。
真机拨入后再开 `AGENT_MODE=cloud`。

### 阶段 5：电话细节最后调

按优先级：

1. 半双工：SPEAKING 期间不处理上行（现有逻辑已接近，电话无浏览器 AEC 时更稳）
2. 挂断处理：`0x00` / socket close → 正常收尾
3. DTMF：可选透传（打断 / 按键确认）
4. 静音抑制：关闭，避免句首被切
5. 延迟：EoU → 首包音频尽量 ≤ 1s（Asterisk 与桥同机可达）

AEC / barge-in 先不碰。

---



## 8. 电话场景注意事项


| 事项        | 说明                         | 对策                              |
| --------- | -------------------------- | ------------------------------- |
| 回声消除（AEC） | 浏览器有 AEC，SIP 裸音频没有         | 优先 PBX/话机/SBC 侧 AEC；否则半双工兜底     |
| 半双工       | 播放时不听，天然防回声                | 沿用现有 SPEAKING 不听逻辑              |
| DTMF      | 打断/确认比能量检测可靠               | 让 Asterisk 透传 RFC2833，桥内检测      |
| 静音抑制      | 会吃掉句首                      | 关闭 `SILENCE_SUPPRESSION`        |
| 延迟        | SIP+RTP+TCP 比浏览器多 50–150ms | Asterisk 与桥同机/同内网               |
| 会话关联      | 用 UUID 做 call_id           | `CallSession(call_id=uuid)` 已支持 |
| 并发        | 每来电一条 TCP                  | 一连接一协程，独立 `CallSession`         |


---



## 9. 与现状对照


| 项目  | 浏览器 Demo（现有）           | SIP + Asterisk（目标）                |
| --- | ---------------------- | --------------------------------- |
| 入局  | WebSocket PCM16 @ 16k  | AudioSocket 0x10 @ 8k → 重采样 16k   |
| 出局  | `tts.audio` + 浏览器播     | 16k → 8k → AudioSocket 0x10 → RTP |
| 播完  | `playback.done`        | 本轮 PCM 写完 + 排空                    |
| 采样率 | 16k                    | 桥内 8k↔16k 重采样                     |
| 回声  | 浏览器 AEC                | PBX AEC 或半双工                      |
| 业务  | DialogManager + SQLite | 不变                                |


---



## 10. 明确不做（现阶段）

- 不新建 `repair-order-agent` 式第三业务包
- 不改 DialogManager / 抽取器
- 不先冲 16k / `Dial(.../c(slin16))`
- 不写复杂 dialplan（转接、队列、IVR）
- 不先做 barge-in / AEC

---

**一句话总结**：客户 PBX 只看到标准 SIP Trunk；Asterisk 把任意电话 codec 转成 8k PCM；
桥内重采样到 16k 复用现有 `CallSession`；回程反向重采样写回。业务零改动，只新增传输层。