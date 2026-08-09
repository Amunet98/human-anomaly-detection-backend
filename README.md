# Human Anomaly Detection — Backend

[![Live Demo](https://img.shields.io/badge/Live%20Demo-bimeshpoudel.com.np-facc15)](https://www.bimeshpoudel.com.np/human-anomaly-live-demo)
[![Node.js](https://img.shields.io/badge/Node.js-5fa04e?logo=nodedotjs&logoColor=white)](https://nodejs.org)
[![Express](https://img.shields.io/badge/Express-000)](https://expressjs.com)
[![Socket.IO](https://img.shields.io/badge/Socket.IO-010101?logo=socketdotio)](https://socket.io)
[![Prisma](https://img.shields.io/badge/Prisma-2d3748?logo=prisma)](https://www.prisma.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ONNX-8A2BE2)](https://docs.ultralytics.com)

Express + Socket.IO + Prisma/Postgres API. This is the hub of the system: it
receives live camera frames from [server-opencv](https://github.com/Amunet98/server-opencv),
runs a self-hosted YOLOv8 ONNX model (`best.onnx`) against them, broadcasts
detections to the [frontend](https://github.com/Amunet98/human-anomaly-detection-frontend)
over a socket, and persists them to Postgres.

Model provenance, class order, known failure modes and (once measured) accuracy
live in [`training/MODEL_CARD.md`](training/MODEL_CARD.md). Worth reading before
quoting any accuracy figure for this system — there isn't a measured one yet.

## How it fits together

```
server-opencv (webcam or sample video)
        │  socket.io client, emits 'data' (base64 jpeg frames)
        │  authenticated as the producer via PRODUCER_TOKEN
        ▲  backend replies with 'stream-control' {active} so capture
        │  only runs while somebody is actually watching
        ▼
   backend (this repo)
        │  re-emits 'frame' to viewers (throttled to ~6.7 fps)
        │  samples frames for inference (~2/sec, throttled)
        │  on detection: emits 'detection' (every box) + 'detected'
        │  (legacy label string), writes raw_data row
        ▼
   frontend (socket.io client + REST calls to /category, /item/:id, /analyze)
```

`/analyze` runs the same model against a single uploaded image or image URL —
what the frontend's upload/URL-check features use instead of a third-party
detection API.

## Setup

```bash
npm install
npx prisma generate      # regenerates the Prisma client from schema.prisma
npx prisma db push        # creates tables in the database from DATABASE_URL
npm run prisma:seed       # optional: seeds demo category/item/class data
npm start
```

### Environment variables (`.env`)

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres connection string |
| `PORT` | defaults to `8081` locally; hosting platforms set this automatically |
| `CLASS_NAMES` | comma-separated posture labels (default: `fall,sit,stand`). No longer `best.onnx`'s own class list — the model detects `person` + keypoints and posture is derived in `posture.js`. Must match `CLASS_NAMES` in the frontend's `constants.js` |
| `DETECTION_CONFIDENCE` | minimum confidence to report a detection (default `0.4`) |
| `PRODUCER_TOKEN` | shared secret authenticating the server-opencv capture service — set the **same value** on both services; without it, incoming `data` frames are ignored |

## Hardening

- The socket `data` handler (shared live feed + detection persistence) only
  accepts frames from the authenticated producer (`?role=producer` +
  matching `PRODUCER_TOKEN`) — arbitrary clients can't spoof frames.
- CORS (Express and Socket.IO) is an allowlist — the portfolio domain,
  the frontend's Vercel deployments, and localhost — not `*`.
- `POST /analyze` is rate-limited (20/min/IP), and its `imageUrl` path is
  SSRF-guarded: http(s) only, private/loopback/link-local and cloud-metadata
  IPs rejected, DNS re-validated at connection time (anti-rebinding), and
  redirects disabled.
- `GET /detected` returns only the latest 200 detections and never includes
  raw camera frames; frame images are no longer persisted at all.

### Note on Prisma version

This project pins `prisma`/`@prisma/client` to `5.22.0`. Prisma 6.x/7.x's
newer WASM-based query engine segfaulted reliably in local testing here; 5.x
uses the older native-binary engine and has been solid. Worth retrying a
newer major once this is resolved upstream.

## Deploying

Needs a long-lived Node process (Socket.IO) and a Postgres database — Railway
or Render both work well. Set `DATABASE_URL`, `PRODUCER_TOKEN`, and (if the
platform doesn't set it automatically) `PORT`. The Render build command runs
`prisma db push` and the seed on every deploy, so a fresh empty database
provisions itself.

Postgres is on **Neon's free plan**, not a Render managed add-on: Render's
free databases expire 30 days after creation and are then deleted, and a
workspace gets only one, so they can't back a long-lived demo. Neon's free
tier doesn't expire.

Use Neon's **direct (unpooled)** connection string — this backend is a single
long-lived process, so pooling gains nothing, and `prisma db push` can fail
through pgbouncer in transaction mode. Keep only `?sslmode=require` on the
string; Prisma 5.22's query engine doesn't recognise `channel_binding`.

The production instance runs on Render (GitHub-connected — push to `main`
auto-deploys).

### Keeping the demo warm

Render's free tier spins a service down after 15 idle minutes, so a cold
visitor waits ~50s. Two pieces keep this service up, and the split between
them is the whole point:

1. **A daily Claude routine at 09:50 Asia/Kathmandu *starts* it** — one
   `curl --max-time 180` with retries, which sits through the full cold start.
2. **cron-job.org *holds* it up** — `/` every 10 minutes, 10:00–20:00
   Asia/Kathmandu.

**A pinger cannot cold-start this service, only keep it warm.** cron-job.org
caps its request timeout at 30s on the free plan, and the cold start takes
~50s, so the first ping of the day always aborts the wake partway through.
Render's router then refuses the retries — `x-render-routing:
hibernate-rate-limited` (a 429 whose body is the plain string `Too Many
Requests`) or `x-render-routing: no-deploy` (an HTML error page, which
cron-job.org rejects as "output too large"). Neither error comes from this
app; successful responses carry `x-render-origin-server: Render` and the
failures don't. Once it starts, every ping that day fails and cron-job.org
auto-disables the job after 26 consecutive failures. That took the demo down
on 2026-08-08 and again on 2026-08-09, and the only reason it ever worked was
that stray traffic happened to wake the service before 10:00.

If the demo is cold some morning, check `x-render-routing` on a request before
theorising — it names the cause directly.

Three constraints shape all of this, and they are easy to trip over:

- **Render allows 750 free instance-hours per month across the whole
  workspace**, and the penalty for exceeding it is suspension until the month
  resets. Keeping one service awake 24/7 is ~744 h — 99% of the quota — so
  round-the-clock pinging is out. A ~10-hour window for the backend alone
  projects to ~316 h/month, which leaves real room: **measured 81.72 / 750 h
  on 2026-08-09, nine days into the month**, so the window could be widened to
  ~16–18 h/day (~500–560 h) if the demo needs longer availability. Check the
  live figure at https://dashboard.render.com/billing before widening — the
  earlier ~610 h/month estimate in this file was guesswork and ran about 2x
  high.
- **Only the backend is kept warm.** `server-opencv`'s ping was deleted on
  2026-08-09 — it is not in the live demo path (see the `SAMPLE_CLIP_URL`
  comment in the frontend's `LiveStream.js`) and it was costing ~305 h/month
  of that quota. It still responds if hit directly, just slowly.
- **The pings must target `/`, which never touches Prisma.** Hitting a
  DB-backed route instead would wake Neon's compute on every ping and burn
  its 100 compute-hours/month.

Don't reach for GitHub Actions here — it was tried and removed
(`.github/workflows/keep-warm.yml`). Scheduled workflows on free public repos
are heavily deprioritized (a `*/10` cron actually fired about hourly), and
runs were intermittently cancelled with "The job was not acquired by Runner of
type hosted" — a GitHub capacity failure with no workflow-side fix.

## API

- `GET /category`, `GET /item/:id`, `GET /item/classes/:id`, `GET /detected` — read the seeded category/item/class data and detection history (latest 200, without raw frames).
- `POST /analyze` — body is `{ "image": "<base64>" }` or `{ "imageUrl": "<url>" }`; returns `{ detections: [...], top: {...} }`. Rate-limited per IP.
- Socket events: listens for `data` (producer-authenticated incoming frames), emits `frame` (throttled relayed video), `detected` (label string), and `stream-control` (tells the producer to start/stop capturing based on viewer count).
