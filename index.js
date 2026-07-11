require('dotenv').config();
const express = require('express');
const cors = require('cors');
const app = express();
app.use(cors());
app.use(express.json({ limit: '15mb' }));
const server = require('http').Server(app);

const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

const { analyzeBase64, analyzeUrl } = require('./inference');

/// configuring socket
const io = require('socket.io')(server, {
	cors: {
		origin: "*"
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

// socket server listening
io.on('connection', client => {
	console.log('connection established')
	client.on('data', data => {
		io.emit('frame', data);
		maybeRunInference(data);
	});
	client.on('disconnect', () => {
		console.log('client disconnected');
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
	try {
		const { image, imageUrl } = req.body;
		if (!image && !imageUrl) {
			return res.status(400).send('Pass either "image" (base64) or "imageUrl" in the request body.');
		}
		const result = image ? await analyzeBase64(image) : await analyzeUrl(imageUrl);
		res.status(200).send(result);
	} catch (error) {
		console.log(error);
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
		const categoryId = req.params.id
		if (categoryId == null || undefined) {
			res.status(500);
			res.send('please pass category id in request params !');
		}
		const output = await prisma.items.findMany({
			where: {
				category_id: parseInt(categoryId)
			}
		});
		res.send(output);
		res.status(200);
	} catch (error) {
		console.log(error);
		res.status(500);
		res.send('internal Server error')
	}
});

app.get('/item/classes/:id', async (req, res) => {
	try {
		const itemId = req.params.id
		if (itemId == null || undefined) {
			res.status(500);
			res.send('please pass item id in request params !');
		}
		const output = await prisma.item_class_assign.findMany({
			where: {
				item_id: parseInt(itemId)
			}
		});
		res.send(output);
		res.status(200);
	} catch (error) {
		console.log(error);
		res.status(500);
		res.send('internal Server error')
	}
});

app.get('/detected', async (req, res) => {
	try {
		const output = await prisma.raw_data.findMany({
			orderBy: {
				time: 'desc'
			},
		});
		res.send(output);
		res.status(200);
	} catch (error) {
		console.log(error);
		res.status(500);
		res.send('internal Server error')
	}
});

// Port comes from the hosting platform in production (Railway/Render set process.env.PORT),
// falls back to 81 for local dev to match the frontend's existing localhost:81 socket URL.
const PORT = process.env.PORT || 81;
server.listen(PORT, () => {
  console.log(`🚀 Main Socket Backend active on port ${PORT}`);
});
