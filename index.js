require('dotenv').config();
const express = require('express');
const cors = require('cors');
const app = express();

// Only the real frontend (production custom domain + its Vercel deployments)
// and local dev may call this API / open a socket - closes off the "anyone
// on the internet can drive this backend from their own page" gap. Requests
// with no Origin header (curl, the opencv producer's Node socket.io-client)
// are left alone since CORS can't meaningfully restrict non-browser callers
// anyway; those are gated by PRODUCER_TOKEN instead where it matters.
const ALLOWED_ORIGIN_PATTERNS = [
	/^https:\/\/(www\.)?bimeshpoudel\.com\.np$/,
	/^https:\/\/frontend-new-[a-z0-9-]+\.vercel\.app$/,
	/^http:\/\/localhost:\d+$/,
];
function isAllowedOrigin(origin) {
	if (!origin) return true;
	return ALLOWED_ORIGIN_PATTERNS.some((re) => re.test(origin));
}
const corsOptions = { origin: (origin, callback) => callback(null, isAllowedOrigin(origin)) };

app.use(cors(corsOptions));
app.use(express.json({ limit: '15mb' }));
const server = require('http').Server(app);

const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

const { analyzeBase64, analyzeUrl } = require('./inference');

/// configuring socket
const io = require('socket.io')(server, {
	cors: {
		origin: (origin, callback) => callback(null, isAllowedOrigin(origin)),
	},
	transports: ["websocket", "polling"]
});

// Persists a detection so /detected has real history, and broadcasts it to
// connected clients as the simple label string LiveStream/Home.js expect.
async function recordDetection(label, frameBase64) {
	io.emit('detected', label);
	try {
		await prisma.raw_data.create({
			data: { name: label, image_frame: frameBase64 },
		});
	} catch (error) {
		console.log('Failed to persist detection:', error.message);
	}
}

// Running full YOLO inference on every incoming frame (~25fps from the
// opencv service) would peg the CPU, so we only sample a frame at most
// every INFERENCE_INTERVAL_MS and skip while a run is still in flight.
const INFERENCE_INTERVAL_MS = 500;
let lastInferenceAt = 0;
let inferenceInFlight = false;

// Relaying every incoming frame (~25fps) to every connected viewer chews
// through bandwidth fast (Render free tier is 5GB/mo egress) for a live
// preview that doesn't need full frame rate. Throttle the broadcast
// independently of inference.
const FRAME_BROADCAST_INTERVAL_MS = 150;
let lastFrameBroadcastAt = 0;

// The opencv capture service used to stream 24/7 regardless of whether
// anyone was watching. It now connects with ?role=producer and waits for a
// 'stream-control' signal - we count real (non-producer) connections and
// tell it to start/stop capturing accordingly.
let viewerCount = 0;

// Optional shared secret gating who counts as the real opencv producer.
// `role=producer` alone is just a client-supplied query string - anyone can
// send it - so without this set, isProducer below stays spoofable by design
// (kept permissive so this doesn't break an existing deployment that hasn't
// set the var yet). Set PRODUCER_TOKEN to the same random value here and in
// server-opencv-main's env to actually close that gap.
const PRODUCER_TOKEN = process.env.PRODUCER_TOKEN;

// Per-IP sliding window for /analyze - it's a full ONNX inference run per
// request, cheap to hammer and expensive to serve, so an unthrottled public
// endpoint is an easy CPU-exhaustion DoS. In-memory/per-instance, resets on
// restart - fine for this scale.
const ANALYZE_WINDOW_MS = 60_000;
const ANALYZE_MAX_PER_WINDOW = 20;
const analyzeHits = new Map();
function analyzeRateLimited(ip) {
	const now = Date.now();
	const entry = analyzeHits.get(ip);
	if (!entry || now > entry.resetAt) {
		analyzeHits.set(ip, { count: 1, resetAt: now + ANALYZE_WINDOW_MS });
		return false;
	}
	entry.count += 1;
	return entry.count > ANALYZE_MAX_PER_WINDOW;
}
// Entries are never removed above, so without this sweep analyzeHits would
// grow for as long as the process stays up (a new IP hitting /analyze once
// adds a permanent entry). Clear out anything past its window every few
// minutes instead.
setInterval(() => {
	const now = Date.now();
	for (const [ip, entry] of analyzeHits) {
		if (now > entry.resetAt) analyzeHits.delete(ip);
	}
}, 5 * 60_000).unref();

function maybeRunInference(frameBase64) {
	const now = Date.now();
	if (inferenceInFlight || now - lastInferenceAt < INFERENCE_INTERVAL_MS) return;
	lastInferenceAt = now;
	inferenceInFlight = true;

	analyzeBase64(frameBase64)
		.then((result) => {
			if (result.top) {
				console.log(`Detected: ${result.top.className} (${result.top.confidence})`);
				return recordDetection(result.top.className, frameBase64);
			}
		})
		.catch((error) => console.log('Inference failed:', error.message))
		.finally(() => {
			inferenceInFlight = false;
		});
}

// Runs inference on one visitor's own camera frames (see LiveStream.js) and
// reports the result back to just that connection via 'own-detected' - never
// rebroadcast to other visitors, and not persisted to raw_data (unlike the
// shared demo camera/sample video above, these are a stranger's own webcam
// frames). Throttled per-connection so concurrent visitors don't starve each
// other the way a single shared lastInferenceAt/inferenceInFlight would.
function makeOwnCameraInference(client) {
	let lastInferenceAt = 0;
	let inferenceInFlight = false;
	return function (frameBase64) {
		const now = Date.now();
		if (inferenceInFlight || now - lastInferenceAt < INFERENCE_INTERVAL_MS) return;
		lastInferenceAt = now;
		inferenceInFlight = true;

		analyzeBase64(frameBase64)
			.then((result) => {
				if (result.top) client.emit('own-detected', result.top.className);
			})
			.catch((error) => console.log('Own-camera inference failed:', error.message))
			.finally(() => {
				inferenceInFlight = false;
			});
	};
}

// socket server listening
io.on('connection', client => {
	const isProducer = PRODUCER_TOKEN
		? client.handshake.query.role === 'producer' && client.handshake.query.token === PRODUCER_TOKEN
		: client.handshake.query.role === 'producer';

	if (isProducer) {
		console.log('opencv producer connected');
		client.join('producers');
		// Covers a producer reconnect (e.g. after a Render restart) while
		// viewers are already watching - without this it'd stay paused.
		client.emit('stream-control', { active: viewerCount > 0 });
	} else {
		console.log('connection established');
		viewerCount++;
		if (viewerCount === 1) io.to('producers').emit('stream-control', { active: true });
	}

	client.on('data', data => {
		// Only the real opencv capture service should be able to push frames
		// into the shared live feed / detection history - without this check
		// any connected client could spoof 'data' events (fake live feed
		// content, junk written to raw_data via recordDetection).
		if (!isProducer) return;
		const now = Date.now();
		if (now - lastFrameBroadcastAt >= FRAME_BROADCAST_INTERVAL_MS) {
			lastFrameBroadcastAt = now;
			io.emit('frame', data);
		}
		maybeRunInference(data);
	});

	const runOwnCameraInference = makeOwnCameraInference(client);
	client.on('camera-frame', data => {
		runOwnCameraInference(data);
	});

	client.on('disconnect', () => {
		console.log('client disconnected');
		if (!isProducer) {
			viewerCount = Math.max(0, viewerCount - 1);
			if (viewerCount === 0) io.to('producers').emit('stream-control', { active: false });
		}
	});
});

// Creating get request simple route
app.get('/', (req, res) => {
	res.send('Detection system')
});

// Runs the same ONNX model used for the live stream against a single
// uploaded/linked image - what the frontend's upload and URL-check
// features call instead of a third-party API.
app.post('/analyze', async (req, res) => {
	const ip = (req.headers['x-forwarded-for'] || '').split(',')[0].trim() || req.socket.remoteAddress;
	if (analyzeRateLimited(ip)) {
		return res.status(429).send('Too many requests - give it a minute.');
	}
	try {
		const { image, imageUrl } = req.body;
		if (!image && !imageUrl) {
			return res.status(400).send('Pass either "image" (base64) or "imageUrl" in the request body.');
		}
		const result = image ? await analyzeBase64(image) : await analyzeUrl(imageUrl);
		res.status(200).send(result);
	} catch (error) {
		console.log(error);
		// Validation errors (bad/disallowed URL, e.g. SSRF guard) from analyzeUrl.
		if (error.statusCode) {
			return res.status(error.statusCode).send(error.message);
		}
		// axios errors carry the failed HTTP response - surface that (e.g. a
		// host blocking the fetch with 403/400) instead of a generic 500 that
		// looks like our own server broke.
		if (error.response) {
			return res
				.status(502)
				.send(`Could not fetch that image URL (host responded ${error.response.status}). Try a different URL or upload the image instead.`);
		}
		// Sharp throws this when the fetched URL returned something that
		// isn't actually image data - e.g. a webpage/search-results link
		// instead of a direct link to a .jpg/.png file.
		if (error.message?.includes('unsupported image format')) {
			return res
				.status(400)
				.send('That URL doesn\'t point directly to an image file. Right-click the image itself and copy its address (should end in .jpg/.png/etc), not a link to the page it\'s on.');
		}
		res.status(500).send('internal Server error');
	}
});

app.get('/category', async (req, res) => {
	try {
		const output = await prisma.category.findMany();
		res.status(200).send(output);
	} catch (error) {
		console.log(error);
		res.status(500).send('internal Server error');
	}
});

app.get('/item/:id', async (req, res) => {
	try {
		const categoryId = parseInt(req.params.id, 10);
		if (Number.isNaN(categoryId)) {
			return res.status(400).send('please pass a valid category id in request params !');
		}
		const output = await prisma.items.findMany({
			where: {
				category_id: categoryId
			}
		});
		res.status(200).send(output);
	} catch (error) {
		console.log(error);
		res.status(500).send('internal Server error');
	}
});

app.get('/item/classes/:id', async (req, res) => {
	try {
		const itemId = parseInt(req.params.id, 10);
		if (Number.isNaN(itemId)) {
			return res.status(400).send('please pass a valid item id in request params !');
		}
		const output = await prisma.item_class_assign.findMany({
			where: {
				item_id: itemId
			}
		});
		res.status(200).send(output);
	} catch (error) {
		console.log(error);
		res.status(500).send('internal Server error');
	}
});

// Deliberately excludes image_frame - this is unauthenticated, and the raw
// captured camera frames shouldn't be bulk-downloadable by anyone who finds
// the URL. Capped to the most recent 200 so the response can't grow
// unbounded as raw_data accumulates.
app.get('/detected', async (req, res) => {
	try {
		const output = await prisma.raw_data.findMany({
			orderBy: {
				time: 'desc'
			},
			take: 200,
			select: {
				id: true,
				item_class_assign_id: true,
				category_id: true,
				item_id: true,
				class_id: true,
				time: true,
				gps_coordinates_lat: true,
				gps_coordinates_lng: true,
				name: true,
			},
		});
		res.status(200).send(output);
	} catch (error) {
		console.log(error);
		res.status(500).send('internal Server error');
	}
});

// Port comes from the hosting platform in production (Railway/Render set process.env.PORT),
// falls back to 81 for local dev to match the frontend's existing localhost:81 socket URL.
const PORT = process.env.PORT || 81;
server.listen(PORT, () => {
  console.log(`🚀 Main Socket Backend active on port ${PORT}`);
});
