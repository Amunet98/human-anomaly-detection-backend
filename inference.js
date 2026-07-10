const path = require('path');
const ort = require('onnxruntime-node');
const sharp = require('sharp');
const axios = require('axios');

const MODEL_PATH = path.join(__dirname, 'best.onnx');
const INPUT_SIZE = 640;
const NUM_ANCHORS = 8400;

// The exported best.onnx has no embedded class-name metadata (verified by
// inspecting the file directly), so these are inferred from context (the
// project's Roboflow model is "fall-detection") rather than read from the
// model. Confirm/relabel via env if the index order turns out to be swapped.
const CLASS_NAMES = (process.env.CLASS_NAMES || 'Fall Detected,No Fall')
	.split(',')
	.map((s) => s.trim());

const CONFIDENCE_THRESHOLD = parseFloat(process.env.DETECTION_CONFIDENCE || '0.4');
const IOU_THRESHOLD = 0.45;

let sessionPromise = null;
function getSession() {
	if (!sessionPromise) sessionPromise = ort.InferenceSession.create(MODEL_PATH);
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

function nms(boxes) {
	const sorted = [...boxes].sort((a, b) => b.score - a.score);
	const kept = [];
	for (const box of sorted) {
		if (kept.some((k) => k.classId === box.classId && iou(k, box) > IOU_THRESHOLD)) continue;
		kept.push(box);
	}
	return kept;
}

// Parses YOLOv8's raw export output: [1, 4 + numClasses, numAnchors],
// laid out channel-major (all cx, then all cy, then all w, then all h,
// then per-class scores), and maps boxes back to original image pixels.
function postprocess(outputData, { scale, padLeft, padTop }) {
	const boxes = [];
	for (let a = 0; a < NUM_ANCHORS; a++) {
		let bestScore = -Infinity;
		let bestClass = -1;
		for (let c = 0; c < CLASS_NAMES.length; c++) {
			const score = outputData[(4 + c) * NUM_ANCHORS + a];
			if (score > bestScore) {
				bestScore = score;
				bestClass = c;
			}
		}
		if (bestScore < CONFIDENCE_THRESHOLD) continue;

		const cx = outputData[0 * NUM_ANCHORS + a];
		const cy = outputData[1 * NUM_ANCHORS + a];
		const w = outputData[2 * NUM_ANCHORS + a];
		const h = outputData[3 * NUM_ANCHORS + a];

		boxes.push({
			x1: (cx - w / 2 - padLeft) / scale,
			y1: (cy - h / 2 - padTop) / scale,
			x2: (cx + w / 2 - padLeft) / scale,
			y2: (cy + h / 2 - padTop) / scale,
			score: bestScore,
			classId: bestClass,
		});
	}

	return nms(boxes).map((b) => ({
		className: CLASS_NAMES[b.classId],
		confidence: Math.round(b.score * 1000) / 1000,
		box: [Math.round(b.x1), Math.round(b.y1), Math.round(b.x2), Math.round(b.y2)],
	}));
}

async function analyzeBuffer(buffer) {
	const session = await getSession();
	const { tensorData, scale, padLeft, padTop } = await preprocess(buffer);
	const tensor = new ort.Tensor('float32', tensorData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
	const results = await session.run({ images: tensor });
	const detections = postprocess(results.output0.data, { scale, padLeft, padTop });
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

async function analyzeUrl(imageUrl) {
	const response = await axios.get(imageUrl, { responseType: 'arraybuffer' });
	return analyzeBuffer(Buffer.from(response.data));
}

module.exports = { analyzeBuffer, analyzeBase64, analyzeUrl, CLASS_NAMES };
