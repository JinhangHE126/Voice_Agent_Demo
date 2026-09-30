# Progress — voice-agent-demo

## Done

- [x] 独立工程脚手架（不改动现有 demo）
- [x] Config / VAD / models
- [x] Qwen3-ASR-Flash-Realtime 客户端（+ mock）
- [x] 分級 LLM：L0 FAQ → L1 fast → L2 Bonsai（关 thinking）
- [x] MiniMax speech-2.8-turbo WebSocket TTS（+ mock PCM）
- [x] Orchestrator：EoU 延迟埋点、barge-in、generation_id
- [x] WebSocket UI（PCM 上行 + PCM 播放）@ :7862
- [x] pytest（mock 首音频 ≤1s、FAQ 路由）

## Next

- [ ] 填 `DASHSCOPE_API_KEY` + `MINIMAX_API_KEY`，`AGENT_MODE=cloud` 实机压延迟
- [ ] 选粤语更稳的 MiniMax `voice_id`
- [ ] FreeSWITCH/Asterisk 侧挂 PCM 桥（电话入线）
- [ ] FAQ 热更新 / 业务意图扩展
