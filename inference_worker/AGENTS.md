# inference_worker — worker package (transport + backend bridge)

## Purpose

The worker runtime: connect to the grid, pop/receive text jobs, transform them to OpenAI
`/chat/completions`, run them against the local backend, and return generations. Plus the
launcher (CLI/GUI), backend detection, config, and cross-platform service install.

## Ownership

- **Grid transport:**
  - `api_client.py` — `APIClient`: HTTP polling against `/v2/generate/text/{pop,submit}` +
    `find_user`. The default mode.
  - `ws_client.py` — `StreamingWorker`: persistent WebSocket to `/v1/workers/ws`. Registers,
    receives pushed jobs, streams tokens back live, awaits a `done` ack with `den` reward.
    Enabled by `GRID_STREAMING=true`. Reasoning models stream `<think>…</think>` live.
  - `p2p_client.py` — experimental libp2p/gossipsub transport (`P2P_ENABLED`, runs trio).
- **Backend bridge:** `worker.py` — `TextWorker` (polling loop, payload transform, stale
  detection, retries, `WorkerStats`) and shared helpers (`strip_thinking_tags`,
  `ENLISTMENT_PROMPT`). `detect_backends.py` — port scan + model/context probes + Ollama install.
- **Config / launch:** `config.py` (`Settings`, per-machine default worker name, stable config
  dir), `env_utils.py` (.env read/write + dashboard token), `cli.py` (argparse entry, GUI vs
  console), `gui.py` (Tkinter window), `headless.py` (terminal quick-setup), `service.py`
  (systemd / launchd / Windows-startup install).
- `web/` — browser setup wizard + dashboard. Owned in its own AGENTS.md.

## Local Contracts

- **Grid model name** is the advertised id (`grid/<model>` or `openai/<model>`); the
  **backend model name** (`MODEL_NAME`) is what the local engine serves. Do not conflate.
- **Always submit a result** (even empty/faulted) so a job never hangs in the grid; mark
  `state="faulted"` on backend validation errors.
- **Thinking tags:** polling mode strips `<think>` from final text; streaming mode surfaces
  reasoning live wrapped in `<think>…</think>` and always closes an open block.
- Transport errors back off with bounded exponential delay; 401 surfaces as `api_auth_error`.
- `service.py` uses `sys.executable` (not pip wrappers), `shlex.quote`s runtime paths, and
  writes units via secure temp files — keep these invariants.

## Work Guidance

- Job-shape changes must land in BOTH `worker.py` (poll) and `ws_client.py` (stream).
- New backend engine → add a probe entry in `detect_backends.KNOWN_ENGINES`.

## Verification

- `pytest` from repo root (smoke tests).
- Manual: point at a local Ollama, run `grid-text-worker`, confirm jobs complete in the dashboard.

## Child DOX Index

- [web/AGENTS.md](web/AGENTS.md) — FastAPI setup wizard + control dashboard.
