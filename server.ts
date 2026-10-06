import express, { Request, Response } from 'express';
import path from 'path';
import dotenv from 'dotenv';
import multer from 'multer';
import { spawn, ChildProcess } from 'child_process';

dotenv.config();

const app = express();
const PORT = 3000;
const isProd = process.env.NODE_ENV === 'production';

// Target URL for the authoritative FastAPI ML backend
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://127.0.0.1:8001';

// Multipart upload handler for webcam / sample frames to forward as proxy
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 15 * 1024 * 1024, // 15 MB
  },
});

let fastapiProcess: ChildProcess | null = null;

// Ensure FastAPI is running on 127.0.0.1:8001
async function ensureFastapiBackend() {
  try {
    const res = await fetch(`${FASTAPI_URL}/health`, { signal: AbortSignal.timeout(1000) });
    if (res.ok) {
      console.log(`[FastAPI Proxy] Detected active FastAPI backend at ${FASTAPI_URL}`);
      return;
    }
  } catch {
    // Backend not running yet, spawn it
  }

  console.log(`[FastAPI Proxy] Starting local FastAPI backend on ${FASTAPI_URL}...`);
  fastapiProcess = spawn('python3', ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', '8001'], {
    stdio: 'inherit',
    detached: false,
  });

  fastapiProcess.on('error', (err) => {
    console.error('[FastAPI Subprocess Error]:', err);
  });

  fastapiProcess.on('exit', (code, signal) => {
    console.log(`[FastAPI Subprocess Exit] code: ${code}, signal: ${signal}`);
  });

  // Poll until ready (max 10s)
  for (let i = 0; i < 20; i++) {
    await new Promise((r) => setTimeout(r, 500));
    try {
      const res = await fetch(`${FASTAPI_URL}/health`, { signal: AbortSignal.timeout(1000) });
      if (res.ok) {
        console.log(`[FastAPI Proxy] FastAPI backend ready at ${FASTAPI_URL}`);
        return;
      }
    } catch {
      // Retry
    }
  }
  console.warn(`[FastAPI Proxy] Timeout waiting for FastAPI backend at ${FASTAPI_URL}`);
}

// 1. Authoritative Health Endpoint: Pure reverse proxy to FastAPI /health
app.get('/health', async (_req: Request, res: Response) => {
  try {
    const upstream = await fetch(`${FASTAPI_URL}/health`, {
      signal: AbortSignal.timeout(3000),
    });

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

// 2. Service Info Endpoint: Pure reverse proxy to FastAPI /api/info
app.get('/api/info', async (_req: Request, res: Response) => {
  try {
    const upstream = await fetch(`${FASTAPI_URL}/api/info`, {
      signal: AbortSignal.timeout(3000),
    });

    if (upstream.ok) {
      const data = await upstream.json();
      return res.json({
        ...data,
        backend_connected: true,
      });
    }
  } catch {
    // Upstream unreachable
  }

  return res.json({
    service: 'Emotion Detector (A4 Spatial-Frequency-Geometry Proxy)',
    status: 'online',
    fastapi_target: FASTAPI_URL,
    architecture: 'MobileNetV3-Large (Spatial) + 2D-FFT (Frequency) + 62-D (Geometry)',
    classes: ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry'],
    backend_connected: false,
  });
});

// 3. Authoritative Predict Endpoint: Pure zero-computation reverse proxy to FastAPI /predict
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

    const upstreamResponse = await fetch(`${FASTAPI_URL}/predict`, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(10000),
    });

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

// Clean shutdown handler
function handleShutdown() {
  if (fastapiProcess) {
    console.log('[FastAPI Proxy] Stopping FastAPI backend subprocess...');
    try {
      fastapiProcess.kill('SIGTERM');
    } catch {
      // Ignore
    }
  }
  process.exit(0);
}

process.on('SIGINT', handleShutdown);
process.on('SIGTERM', handleShutdown);

async function main() {
  await ensureFastapiBackend();

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

