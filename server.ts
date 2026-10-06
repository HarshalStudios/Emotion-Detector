import express, { Request, Response } from 'express';
import path from 'path';
import dotenv from 'dotenv';
import multer from 'multer';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const app = express();
const PORT = 3000;
const isProd = process.env.NODE_ENV === 'production';

// Multipart upload handler for webcam / sample frames
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 15 * 1024 * 1024, // 15 MB
  },
});

const CANONICAL_CLASSES = [
  'Neutral',
  'Happy',
  'Sad',
  'Surprise',
  'Fear',
  'Disgust',
  'Angry',
] as const;

type ExpressionClass = (typeof CANONICAL_CLASSES)[number];

// Optional Gemini client for deep multimodal inspection
const geminiApiKey = process.env.GEMINI_API_KEY;
let aiClient: GoogleGenAI | null = null;
if (geminiApiKey) {
  try {
    aiClient = new GoogleGenAI();
  } catch (err) {
    console.warn('[Server] Could not initialize GoogleGenAI:', err);
  }
}

// Upstream ML inference backend URL (FastAPI)
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://127.0.0.1:8001';

// 1. Authoritative Health Endpoint: Proxies directly to FastAPI /health
app.get('/health', async (_req: Request, res: Response) => {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000);

    const upstream = await fetch(`${FASTAPI_URL}/health`, {
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (upstream.ok) {
      const data = await upstream.json();
      return res.json({
        ...data,
        backend_connected: true,
      });
    }

    return res.status(upstream.status).json({
      status: 'disconnected',
      backend_connected: false,
      error: `FastAPI responded with HTTP ${upstream.status}`,
    });
  } catch (err: unknown) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    return res.status(503).json({
      status: 'disconnected',
      backend_connected: false,
      error: `FastAPI backend unavailable at ${FASTAPI_URL}: ${errorMsg}`,
    });
  }
});

// 2. Service Info Endpoint
app.get('/api/info', async (_req: Request, res: Response) => {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000);

    const upstream = await fetch(`${FASTAPI_URL}/api/info`, {
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (upstream.ok) {
      const data = await upstream.json();
      return res.json({ ...data, backend_connected: true });
    }
  } catch {
    // Upstream info unreachable
  }

  return res.json({
    service: 'Emotion Detector (A4 Spatial-Frequency-Geometry)',
    status: 'online',
    fastapi_target: FASTAPI_URL,
    architecture: 'MobileNetV3-Large (Spatial) + 2D-FFT (Frequency) + 62-D (Geometry)',
    classes: CANONICAL_CLASSES,
  });
});

// 3. Authoritative Predict Endpoint: Pure proxy to FastAPI /predict
// Strictly passes through real ONNX inference results with ZERO local calculation or logits synthesis.
app.post('/predict', upload.single('file') as any, async (req: Request, res: Response) => {
  try {
    const file = req.file;
    if (!file || !file.buffer) {
      return res.status(400).json({
        status: 'error',
        face_detected: false,
        error: 'No image file uploaded in multipart form under field "file"',
        backend_connected: true,
      });
    }

    // Build standard multipart request to forward to FastAPI
    const formData = new FormData();
    const blob = new Blob([new Uint8Array(file.buffer)], { type: file.mimetype || 'image/jpeg' });
    formData.append('file', blob, file.originalname || 'frame.jpg');

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 6000);

    const upstreamResponse = await fetch(`${FASTAPI_URL}/predict`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (upstreamResponse.ok) {
      const predictionData = await upstreamResponse.json();
      return res.json({
        ...predictionData,
        backend_connected: true,
      });
    }

    // Upstream returned HTTP error
    const errText = await upstreamResponse.text();
    return res.status(upstreamResponse.status).json({
      status: 'UNAVAILABLE',
      face_detected: false,
      partial_face: false,
      prediction: 'UNAVAILABLE',
      prediction_index: -1,
      confidence: 0,
      probabilities: {
        Neutral: 0,
        Happy: 0,
        Sad: 0,
        Surprise: 0,
        Fear: 0,
        Disgust: 0,
        Angry: 0,
      },
      geometry_valid: false,
      detection_confidence: 0,
      head_pose: { pitch: 0, yaw: 0, roll: 0 },
      bbox: [0, 0, 0, 0],
      backend_connected: false,
      error: `FastAPI error: ${errText.slice(0, 200)}`,
    });
  } catch (err: unknown) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.warn(`[Proxy Error] FastAPI backend at ${FASTAPI_URL} unreachable:`, errorMsg);

    // Return honest UNAVAILABLE status. Never generate fake emotions or fallback to Happy/Neutral.
    return res.status(503).json({
      status: 'UNAVAILABLE',
      face_detected: false,
      partial_face: false,
      prediction: 'UNAVAILABLE',
      prediction_index: -1,
      confidence: 0,
      probabilities: {
        Neutral: 0,
        Happy: 0,
        Sad: 0,
        Surprise: 0,
        Fear: 0,
        Disgust: 0,
        Angry: 0,
      },
      geometry_valid: false,
      detection_confidence: 0,
      head_pose: { pitch: 0, yaw: 0, roll: 0 },
      bbox: [0, 0, 0, 0],
      backend_connected: false,
      error: `FastAPI backend unavailable at ${FASTAPI_URL}: ${errorMsg}`,
    });
  }
});

// 4. Optional Gemini Deep Inspection Endpoint
app.post('/api/deep-analyze', upload.single('file') as any, async (req: Request, res: Response) => {
  if (!aiClient || !req.file?.buffer) {
    return res.status(400).json({
      error: aiClient ? 'No image buffer provided' : 'GEMINI_API_KEY is not configured on server',
    });
  }

  try {
    const base64Image = req.file.buffer.toString('base64');
    const mimeType = req.file.mimetype || 'image/jpeg';

    const response = await aiClient.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: [
        {
          role: 'user',
          parts: [
            {
              inlineData: {
                data: base64Image,
                mimeType,
              },
            },
            {
              text: 'Analyze the facial expression in this image. Classify into one of: Neutral, Happy, Sad, Surprise, Fear, Disgust, Angry. Return a valid JSON object with keys: prediction, confidence (0 to 1), reasoning (short 1-sentence), actionUnits (list of observed facial movements). Return ONLY JSON.',
            },
          ],
        },
      ],
    });

    const text = response.text || '';
    return res.json({ result: text });
  } catch (err: unknown) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    return res.status(500).json({ error: errorMsg });
  }
});

// Body parsers for JSON routes
app.use(express.json({ limit: '25mb' }));
app.use(express.urlencoded({ extended: true, limit: '25mb' }));

async function main() {
  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(process.cwd(), 'dist')));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.resolve(process.cwd(), 'dist', 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Emotion Detector unified server running on http://0.0.0.0:${PORT}`);
  });
}

main().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
