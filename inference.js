const path = require('path');
const dns = require('dns');
const dnsPromises = require('dns').promises;
const net = require('net');
const http = require('http');
const https = require('https');
const ort = require('onnxruntime-node');
const sharp = require('sharp');
const axios = require('axios');

// Sharp's libvips backend keeps a decode cache and a worker pool by default,
// both of which cost real memory that a 512MB-RAM host doesn't have to
// spare alongside a loaded ONNX model. Trade a bit of throughput for a
// smaller footprint.
sharp.cache(false);
sharp.concurrency(1);

const MODEL_PATH = path.join(__dirname, 'best.onnx');
const INPUT_SIZE = 640;

// best.onnx does carry ultralytics metadata, and it is the authority on class
// order: names = {0: 'fall', 1: 'sit', 2: 'stand'} (trained from
// stand-sit-fall-2/data.yaml, imgsz 640, opset 12, nms=False). onnxruntime-node
// doesn't expose custom metadata_props, so the names are restated here rather
// than read back from the file - keep this default in sync with the model, and
// with CLASS_NAMES in render.yaml / .env. Retraining with a different class
// order means changing both.
const CLASS_NAMES = (process.env.CLASS_NAMES || 'fall,sit,stand')
	.split(',')
	.map((s) => s.trim());

const CONFIDENCE_THRESHOLD = parseFloat(process.env.DETECTION_CONFIDENCE || '0.4');
const IOU_THRESHOLD = 0.45;

// Minimum box area as a fraction of the *largest* surviving box, applied after
// NMS. Mirrored in the browser's constants.js MIN_BOX_AREA_RATIO - keep the two
// in step or npm run parity (frontend-new) fails.
//
// Kills sub-limb false positives: the model emits a `stand 0.43` box on a
// subject's boot at 3.1% of the true person box's area. Relative rather than an
// absolute fraction of frame area, because absolute is not scale-invariant - a
// floor tight enough to catch that boot (1.15% of frame) would also delete
// genuinely distant people in far-field CCTV.
const MIN_BOX_AREA_RATIO = 0.15;

let sessionPromise = null;
function getSession() {
	if (!sessionPromise) {
		// onnxruntime-node defaults to spawning a thread per visible CPU core.
		// On a heavily CPU-throttled container (e.g. a free hosting tier) that
		// thrashes badly - way more threads than actual CPU time available,
		// so runs take tens of seconds instead of under one. Single-threaded
		// is faster in practice on constrained hosts; override via env if a
		// given host actually has real cores to spare.
		const threads = parseInt(process.env.ONNX_NUM_THREADS, 10) || 1;
		sessionPromise = ort.InferenceSession.create(MODEL_PATH, {
			intraOpNumThreads: threads,
			interOpNumThreads: threads,
			// Both trade a bit of speed for a smaller memory footprint - the
			// arena/pattern allocators pre-reserve memory for reuse, which is
			// the wrong tradeoff on a RAM-constrained host.
			enableCpuMemArena: false,
			enableMemPattern: false,
			graphOptimizationLevel: 'basic',
		});
	}
	return sessionPromise;
}

// Letterbox-resize into INPUT_SIZE x INPUT_SIZE (matches ultralytics' own
// preprocessing), padding with mid-gray so the aspect ratio isn't distorted.
async function preprocess(buffer) {
	const metadata = await sharp(buffer).metadata();
	const scale = Math.min(INPUT_SIZE / metadata.width, INPUT_SIZE / metadata.height);
	const newW = Math.round(metadata.width * scale);
	const newH = Math.round(metadata.height * scale);
	const padLeft = Math.floor((INPUT_SIZE - newW) / 2);
	const padTop = Math.floor((INPUT_SIZE - newH) / 2);

	const { data } = await sharp(buffer)
		.resize(newW, newH)
		.extend({
			top: padTop,
			bottom: INPUT_SIZE - newH - padTop,
			left: padLeft,
			right: INPUT_SIZE - newW - padLeft,
			background: { r: 114, g: 114, b: 114 },
		})
		.removeAlpha()
		.raw()
		.toBuffer({ resolveWithObject: true });

	// sharp's raw output is HWC, RGB. Convert to CHW float32 in [0, 1].
	const area = INPUT_SIZE * INPUT_SIZE;
	const tensorData = new Float32Array(3 * area);
	for (let i = 0; i < area; i++) {
		tensorData[i] = data[i * 3] / 255;
		tensorData[area + i] = data[i * 3 + 1] / 255;
		tensorData[2 * area + i] = data[i * 3 + 2] / 255;
	}

	return { tensorData, scale, padLeft, padTop };
}

function iou(a, b) {
	const x1 = Math.max(a.x1, b.x1);
	const y1 = Math.max(a.y1, b.y1);
	const x2 = Math.min(a.x2, b.x2);
	const y2 = Math.min(a.y2, b.y2);
	const interW = Math.max(0, x2 - x1);
	const interH = Math.max(0, y2 - y1);
	const inter = interW * interH;
	const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
	const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
	return inter / (areaA + areaB - inter);
}

// Greedy, class-AGNOSTIC NMS - a box is suppressed by any higher-scoring box it
// overlaps, regardless of class. Mirrored in the browser's postprocess.js nms().
//
// Class-agnostic is forced by the label set: fall/sit/stand are mutually
// exclusive *postures of one person*, not objects that can coexist in the same
// place. The previous class-aware rule let the runner-up posture survive on the
// same box, so a standing subject came back as both `stand 0.72` and `sit 0.44`
// at IoU ~0.99. Restore the classId guard only if the model is ever retrained on
// classes that can genuinely overlap.
function nms(boxes) {
	const sorted = [...boxes].sort((a, b) => b.score - a.score);
	const kept = [];
	for (const box of sorted) {
		if (kept.some((k) => iou(k, box) > IOU_THRESHOLD)) continue;
		kept.push(box);
	}
	return kept;
}

// Drops boxes tiny relative to the largest survivor in the same frame. Runs
// after NMS. Mirrored in the browser's postprocess.js filterTinyBoxes().
//
// The largest box is compared against itself and so is always kept - a lone
// distant person is never self-filtered.
function filterTinyBoxes(boxes) {
	if (boxes.length < 2) return boxes;
	const area = (b) => Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
	const maxArea = Math.max(...boxes.map(area));
	if (maxArea <= 0) return boxes;
	return boxes.filter((b) => area(b) >= MIN_BOX_AREA_RATIO * maxArea);
}

// Parses YOLOv8's raw export output: [1, 4 + numClasses, numAnchors],
// laid out channel-major (all cx, then all cy, then all w, then all h,
// then per-class scores), and maps boxes back to original image pixels.
//
// numClasses and numAnchors come from the tensor's own dims rather than from
// constants: driving the class loop off CLASS_NAMES.length means a misconfigured
// CLASS_NAMES silently drops the model's trailing classes instead of failing
// loudly, and hardcoding 8400 anchors bakes in imgsz=640.
function postprocess(output, { scale, padLeft, padTop }) {
	const outputData = output.data;
	const [, channels, numAnchors] = output.dims;
	const numClasses = channels - 4;
	const boxes = [];
	for (let a = 0; a < numAnchors; a++) {
		let bestScore = -Infinity;
		let bestClass = -1;
		for (let c = 0; c < numClasses; c++) {
			const score = outputData[(4 + c) * numAnchors + a];
			if (score > bestScore) {
				bestScore = score;
				bestClass = c;
			}
		}
		if (bestScore < CONFIDENCE_THRESHOLD) continue;

		const cx = outputData[0 * numAnchors + a];
		const cy = outputData[1 * numAnchors + a];
		const w = outputData[2 * numAnchors + a];
		const h = outputData[3 * numAnchors + a];

		boxes.push({
			x1: (cx - w / 2 - padLeft) / scale,
			y1: (cy - h / 2 - padTop) / scale,
			x2: (cx + w / 2 - padLeft) / scale,
			y2: (cy + h / 2 - padTop) / scale,
			score: bestScore,
			classId: bestClass,
		});
	}

	return filterTinyBoxes(nms(boxes)).map((b) => ({
		// Falls back to the raw index rather than `undefined` if CLASS_NAMES is
		// shorter than the model's class count - a visibly wrong label beats a
		// silently missing one.
		className: CLASS_NAMES[b.classId] ?? `class_${b.classId}`,
		confidence: Math.round(b.score * 1000) / 1000,
		box: [Math.round(b.x1), Math.round(b.y1), Math.round(b.x2), Math.round(b.y2)],
	}));
}

async function analyzeBuffer(buffer) {
	const session = await getSession();
	const { tensorData, scale, padLeft, padTop } = await preprocess(buffer);
	const tensor = new ort.Tensor('float32', tensorData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
	const results = await session.run({ images: tensor });
	const detections = postprocess(results.output0, { scale, padLeft, padTop });
	const top = detections.reduce(
		(best, d) => (!best || d.confidence > best.confidence ? d : best),
		null
	);
	return { detections, top };
}

// Accepts a raw base64 string or a data: URL.
async function analyzeBase64(base64Image) {
	const raw = base64Image.includes(',') ? base64Image.split(',').pop() : base64Image;
	return analyzeBuffer(Buffer.from(raw, 'base64'));
}

// Blocks loopback/private/link-local ranges - notably 169.254.169.254, the
// cloud metadata address that SSRF exploits target on AWS/GCP/Azure/etc.
function isDisallowedIp(ip) {
	const type = net.isIP(ip);
	if (!type) return true;
	if (type === 4) {
		const [a, b] = ip.split('.').map(Number);
		if (a === 127 || a === 10 || a === 0 || a >= 224) return true;
		if (a === 172 && b >= 16 && b <= 31) return true;
		if (a === 192 && b === 168) return true;
		if (a === 169 && b === 254) return true;
		return false;
	}
	const lower = ip.toLowerCase();
	if (lower === '::1' || lower === '::') return true;
	if (lower.startsWith('fe8') || lower.startsWith('fe9') || lower.startsWith('fea') || lower.startsWith('feb')) return true;
	if (lower.startsWith('fc') || lower.startsWith('fd')) return true;
	if (lower.startsWith('::ffff:')) return isDisallowedIp(lower.slice(7));
	return false;
}

// Rejects anything that isn't a plain public http(s) URL before we let axios
// fetch it - without this, `imageUrl` from the request body lets a caller
// make this server request internal/cloud-metadata addresses (SSRF).
async function assertPublicHttpUrl(rawUrl) {
	let parsed;
	try {
		parsed = new URL(rawUrl);
	} catch {
		throw Object.assign(new Error('That is not a valid URL.'), { statusCode: 400 });
	}
	if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
		throw Object.assign(new Error('Only http/https image URLs are allowed.'), { statusCode: 400 });
	}
	if (parsed.hostname === 'localhost') {
		throw Object.assign(new Error('That URL is not allowed.'), { statusCode: 400 });
	}
	let addresses;
	try {
		addresses = await dnsPromises.lookup(parsed.hostname, { all: true });
	} catch {
		throw Object.assign(new Error("Could not resolve that URL's host."), { statusCode: 400 });
	}
	if (!addresses.length || addresses.some((a) => isDisallowedIp(a.address))) {
		throw Object.assign(new Error('That URL is not allowed.'), { statusCode: 400 });
	}
}

// assertPublicHttpUrl above resolves+validates once for a fast, friendly
// error message - but axios resolves the hostname again on its own when it
// actually opens the connection, moments later. A malicious host with a
// short DNS TTL could pass the first check while pointing at a public IP,
// then re-resolve to an internal one by the time axios connects (DNS
// rebinding / TOCTOU). This custom `lookup` re-validates at the exact
// moment the TCP connection is opened, so there's no gap between "checked"
// and "used".
function validatingLookup(hostname, options, callback) {
	if (typeof options === 'function') {
		callback = options;
		options = {};
	}
	dns.lookup(hostname, { all: true }, (err, addresses) => {
		if (err) return callback(err);
		if (!addresses.length || addresses.some((a) => isDisallowedIp(a.address))) {
			return callback(new Error('That URL is not allowed.'));
		}
		if (options.all) return callback(null, addresses);
		const { address, family } = addresses[0];
		callback(null, address, family);
	});
}
const validatingHttpAgent = new http.Agent({ lookup: validatingLookup });
const validatingHttpsAgent = new https.Agent({ lookup: validatingLookup });

async function analyzeUrl(imageUrl) {
	await assertPublicHttpUrl(imageUrl);
	// Many image hosts (Wikimedia, Roboflow, etc.) reject requests with no
	// User-Agent as bot traffic (403), even for a plain public image fetch.
	const response = await axios.get(imageUrl, {
		responseType: 'arraybuffer',
		timeout: 8000,
		// A public URL could still redirect to an internal one - don't follow.
		maxRedirects: 0,
		httpAgent: validatingHttpAgent,
		httpsAgent: validatingHttpsAgent,
		headers: { 'User-Agent': 'Mozilla/5.0 (compatible; human-anomaly-detection-bot/1.0)' },
	});
	return analyzeBuffer(Buffer.from(response.data));
}

module.exports = { analyzeBuffer, analyzeBase64, analyzeUrl, CLASS_NAMES };
