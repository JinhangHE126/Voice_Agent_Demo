# repair-order-agent

Minimal business-core scaffold for a maintenance-call voice agent.

## Purpose

Collect and confirm three fields from a caller:

- customer name
- customer phone
- repair description

Then persist a confirmed order through a repository layer.

## Structure

- `repair_order/`: importable business package (used by `voice-agent-demo`)
- `scripts/text_chat.py`: multi-turn text dialog (no microphone)
- `scripts/replay_wav_case.py`: run one real wav through ASR + business flow
- `tests/test_dialog_manager.py`: multi-turn and extraction tests

## Prefill Audio (for voice-agent-demo)

Place mono 16k PCM wav files in:

`../voice-agent-demo/audio/yue/`

Recommended first:
- `ack_ok.wav`
- `ack_got_it.wav`
- `ack_checking.wav`
- `greeting.wav`

The voice demo plays fillers in parallel with async LLM extraction.

## Replay a Real WAV Case

```bash
cd repair-order-agent
python scripts/replay_wav_case.py --wav "C:\Users\pg9\Desktop\Epro\repair-order-agent\data\audio\2026080400078_I_CPD10124_34991845_37520836_Tel.wav"
```

To enable LLM structured extraction:

```bash
python scripts/replay_wav_case.py --use-llm-extractor --wav "C:\Users\pg9\Desktop\Epro\repair-order-agent\data\audio\2026080400078_I_CPD10124_34991845_37520836_Tel.wav"
```

The script prints:

- ASR transcript/language
- extracted order fields
- next business reply (prompt id + text)
- expected-case compare (if `data/expected/<wav_stem>.json` exists)
