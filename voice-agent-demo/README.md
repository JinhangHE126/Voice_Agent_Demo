# Voice Agent Demo

粤语优先电话语音助手原型：**Qwen ASR → 维修订单 DialogManager → MiniMax TTS**。

默认业务模式 `BUSINESS_MODE=repair_order`：多轮收集姓名、电话、维修内容，确认后写入 SQLite。

等待抽取时会并行播放本地承接语（若 `audio/yue/*.wav` 存在）：
- `ack_checking.wav` / `ack_got_it.wav` / `ack_ok.wav` / `ack_writing.wav`
- 固定问句也可预制：`greeting.wav`、`ask_name.wav`、`ask_phone.wav` 等

目标：说完（EoU）到首包可播音频 **≤ 1000ms**。

本目录是独立工程，业务逻辑来自同级目录 `repair-order-agent/repair_order`。

## 架构

```
Mic PCM16 16k
  → 本地能量 VAD（打断）+ Qwen3-ASR-Flash-Realtime（server_vad / yue）
  → 分級 LLM
       L0 FAQ（faq.json，即时）
       L1 快模型（DashScope OpenAI 兼容，默认 qwen-plus）
       L2 Bonsai（长问/复杂才走，关闭 thinking）
  → MiniMax speech-2.8-turbo WebSocket（language_boost=Chinese,Yue，PCM 流）
  → 浏览器播放 PCM
```

## 快速开始

```bash
cd voice-agent-demo
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python scripts/run_web.py
```

打开 http://127.0.0.1:7862/

默认 `AGENT_MODE=mock`：不调云端，用示意音验证整条链路与延迟埋点。

## Cloud 模式

在 `.env` 填写：

- `DASHSCOPE_API_KEY` — Qwen ASR +（可选）L1 快 LLM
- `MINIMAX_API_KEY` — MiniMax TTS
- `FAST_LLM_API_KEY` — 可留空，回退用 DashScope key
- `BONSAI_API_KEY` / `BONSAI_BASE_URL` — L2（可选）

然后：

```
AGENT_MODE=cloud
```

大陆 MiniMax 可把 `MINIMAX_WS_URL` 改成 `wss://api.minimaxi.com/ws/v1/t2a_v2`。

## 测试

```bash
pytest -q
```

## 端口

| 工程 | 端口 |
|------|------|
| voice-pipeline-demo | 7860 |
| voice-realtime-demo | 7861 |
| voice-agent-demo | **7862** |
