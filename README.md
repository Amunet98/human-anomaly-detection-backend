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

Model provenance, class order, known failure modes and measured accuracy live
in [`training/MODEL_CARD.md`](training/MODEL_CARD.md). Read it before quoting
any accuracy figure for this system.

The short version, measured 2026-08-12 via the frontend's `npm run eval:robust`:
**84.2% clean (16/19), 86.0% perturbed (98/114)**, macro-F1 0.838 / 0.859 over
all four classes, with
`posture.js`'s thresholds calibrated over 3,106 detections from 4,924
deduplicated images. Note `best.onnx` is **COCO-pretrained yolov8n-pose,
unmodified** — posture comes from keypoint geometry, not from a trained
classification head.

That was a decision, not a default. The first version *was* a trained
classifier: yolov8n fine-tuned on an 8,340-image Roboflow dataset to predict
`fall`/`sit`/`stand` directly. Evaluating it is what killed it — under
perturbation it scored 76.7%, and its `sit` class had precision 0.545 against
recall 1.000, the signature of a model keyed on scene appearance (bench, floor,
indoor room) rather than body configuration. More data would not have fixed a
shortcut that the task framing itself made available. See
[`training/MODEL_CARD.md`](training/MODEL_CARD.md) for the full comparison.

**That 86.0% is not a deployment accuracy.** The fixture set was selected for
cases the classifier was expected to get wrong, so it is adversarial rather
than representative; reading it as a field figure would be wrong in both
directions. The model card spells out why.

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
| `CLASS_NAMES` | comma-separated posture labels (default: `fall,sit,squat,stand`). No longer `best.onnx`'s own class list — the model detects `person` + keypoints and posture is derived in `posture.js`. Must match `CLASS_NAMES` in the frontend's `constants.js`, and note `postprocess.js` keeps a separate frozen `LEGACY_CLASS_NAMES` for the archived 3-class weights — do not merge them |
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

### Cold starts, and why there is no keep-warm

Render's free tier spins a service down after 15 idle minutes, so the first
visitor of the day waits ~50s. **That is the accepted behaviour — there is
deliberately nothing keeping this service warm.** The frontend already renders
the uploaded image immediately, before the request resolves, precisely because
the host may be spinning up, so the wait reads as loading rather than as a
broken page.

Keep-warm was tried and retired on 2026-08-10. The write-up below is what that
cost to learn; the point of keeping it is that every alternative here has
already been ruled out.

**A pinger can hold this service up but can never start it.** cron-job.org
caps its request timeout at 30s on the free plan, and the cold start takes
~50s, so the first ping of the day always aborted the wake partway through.
Render's router then refused the retries — `x-render-routing:
hibernate-rate-limited` (a 429 whose body is the plain string `Too Many
Requests`) or `x-render-routing: no-deploy` (an HTML error page, which
cron-job.org rejects as "output too large"). Neither error came from this app;
successful responses carry `x-render-origin-server: Render` and the failures
don't. Once it began, every ping that day failed and cron-job.org auto-disabled
the job after 26 consecutive failures — on 2026-08-08, 2026-08-09 and again on
2026-08-10. It only ever appeared to work because stray traffic happened to
wake the service before the window opened.

Splitting the job in two didn't save it: a daily Claude routine at 09:50
Asia/Kathmandu *could* cold-start the service with `curl --max-time 180`, but
it only bought the ten hours cron-job.org was supposed to hold, and when the
holding half auto-disabled again the wake was burning instance-hours for
nothing. Both are now disabled.

**One earlier session blamed Cloudflare rate-limiting cron-job.org's shared
egress IPs. That was wrong — don't revive it.** The stored failure bodies are
plain Render router responses, and `x-render-routing` names the cause outright.
Check it before theorising:

```
curl -sD - -o /dev/null --max-time 180 <url>/ | grep -i 'x-render\|http/'
```

cron-job.org's HISTORY view stores full failed response bodies and headers;
that is where the answer was.

Constraints worth knowing before reaching for any of this again:

- **Render allows 750 free instance-hours per month across the whole
  workspace**, and the penalty for exceeding it is suspension until the month
  resets. Keeping one service awake 24/7 is ~744 h — 99% of the quota — so
  round-the-clock pinging was never on the table. Measured **81.72 / 750 h on
  2026-08-09**, nine days into the month; the live figure is at
  https://dashboard.render.com/billing. (An earlier ~610 h/month estimate in
  this file was guesswork and ran about 2x high.)
- **`server-opencv` was never worth warming.** Its ping was deleted on
  2026-08-09 — it is not in the live demo path (see the `SAMPLE_CLIP_URL`
  comment in the frontend's `LiveStream.js`) and it was costing ~305 h/month of
  that quota. It still responds if hit directly, just cold.
- **Any ping must target `/`, which never touches Prisma.** Hitting a DB-backed
  route instead wakes Neon's compute on every request and burns its 100
  compute-hours/month.

Don't reach for GitHub Actions here — it was tried and removed
(`.github/workflows/keep-warm.yml`). Scheduled workflows on free public repos
are heavily deprioritized (a `*/10` cron actually fired about hourly), and
runs were intermittently cancelled with "The job was not acquired by Runner of
type hosted" — a GitHub capacity failure with no workflow-side fix.

The real fix, if the demo ever needs to be instant, is a paid Render instance
(no hibernation) — not another scheduler.

## API

- `GET /category`, `GET /item/:id`, `GET /item/classes/:id`, `GET /detected` — read the seeded category/item/class data and detection history (latest 200, without raw frames).
- `POST /analyze` — body is `{ "image": "<base64>" }` or `{ "imageUrl": "<url>" }`; returns `{ detections: [...], top: {...} }`. Rate-limited per IP.
- Socket events: listens for `data` (producer-authenticated incoming frames), emits `frame` (throttled relayed video), `detected` (label string), and `stream-control` (tells the producer to start/stop capturing based on viewer count).
