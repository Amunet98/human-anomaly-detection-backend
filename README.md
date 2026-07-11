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

## How it fits together

```
server-opencv (webcam or sample video)
        │  socket.io client, emits 'data' (base64 jpeg frames)
        ▼
   backend (this repo)
        │  re-emits 'frame' immediately (smooth video)
        │  samples frames for inference (~2/sec, throttled)
        │  on detection: emits 'detected', writes raw_data row
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
| `CLASS_NAMES` | comma-separated class labels matching `best.onnx`'s output order (default: `Fall Detected,No Fall`) |
| `DETECTION_CONFIDENCE` | minimum confidence to report a detection (default `0.4`) |

### Note on Prisma version

This project pins `prisma`/`@prisma/client` to `5.22.0`. Prisma 6.x/7.x's
newer WASM-based query engine segfaulted reliably in local testing here; 5.x
uses the older native-binary engine and has been solid. Worth retrying a
newer major once this is resolved upstream.

## Deploying

Needs a long-lived Node process (Socket.IO) and a Postgres database —
Railway or Render both work well, either with a managed Postgres add-on or
an external one (Supabase/Neon). Set `DATABASE_URL` and (if the platform
doesn't set it automatically) `PORT`. Run `npx prisma db push` once against
the production database before first boot.

## API

- `GET /category`, `GET /item/:id`, `GET /item/classes/:id`, `GET /detected` — read the seeded category/item/class data and detection history.
- `POST /analyze` — body is `{ "image": "<base64>" }` or `{ "imageUrl": "<url>" }`; returns `{ detections: [...], top: {...} }`.
- Socket events: listens for `data` (incoming frames), emits `frame` (relayed video) and `detected` (label string).
