# Voice Agent Demo

实时语音维修客服 Demo：多轮收集维修内容、电话、地址、姓名 → 确认 → 写入 SQLite。

```text
浏览器麦克风
  → WebSocket PCM16
  → Qwen ASR
  → DialogManager（维修工单状态机）
  → 本地预制 wav / MiniMax TTS
  → 浏览器播放
```

语言：粤语为主，支持普通话 / 英文跟随。

## 目录

```text
Voice_Agent_Demo/
  AGENTS.md              ← 完整上手说明（改代码优先读）
  voice-agent-demo/      ← 实时语音 + Web UI（启动入口）
  repair-order-agent/    ← 工单业务逻辑（必须与上面同级）
```

两个子目录必须始终同级，不要拆开只部署其中一个。

## 本地运行

```powershell
cd voice-agent-demo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# 若尚无 .env：copy .env.example .env
python scripts\run_web.py
```

浏览器打开：http://127.0.0.1:7862/

| 配置 | 说明 |
|---|---|
| `AGENT_MODE=mock` | 默认；不调云端，验证链路 |
| `AGENT_MODE=cloud` | 真实 ASR / TTS；需在 `.env` 填写密钥 |
| `BUSINESS_MODE=repair_order` | 默认业务模式 |

Cloud 模式至少配置：

- `DASHSCOPE_API_KEY` — Qwen ASR（及可选抽取 LLM）
- `MINIMAX_API_KEY` — MiniMax TTS

大陆 MiniMax 可将 `MINIMAX_WS_URL` 设为 `wss://api.minimaxi.com/ws/v1/t2a_v2`。

## 注意

- 本地麦克风请用 `127.0.0.1`；公网 `http://` 浏览器通常不给麦克风，手机演示需 HTTPS
- 改代码后重启 `run_web.py`，并强刷浏览器
- `.venv` 体积大且与机器相关，需本机重建
- 本目录若含 `.env`，勿推送到公开仓库

## 测试

```powershell
cd repair-order-agent
pytest -q

cd ..\voice-agent-demo
pytest -q
```

## 更多文档

| 文档 | 内容 |
|---|---|
| [AGENTS.md](./AGENTS.md) | 架构、状态机、改代码约定、故障速查 |
| [SIP_Asterisk_接入方案.md](./SIP_Asterisk_接入方案.md) | Asterisk + AudioSocket SIP 接入（阶段 0–3 已完成） |
| [voice-agent-demo/README.md](./voice-agent-demo/README.md) | 语音链路细节 |
| [voice-agent-demo/DEPLOY_EC2.md](./voice-agent-demo/DEPLOY_EC2.md) | EC2 + Cloudflare Tunnel |
| [repair-order-agent/开发文档.md](./repair-order-agent/开发文档.md) | 产品与业务设计 |

