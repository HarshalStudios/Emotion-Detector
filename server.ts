import express, { Request, Response } from 'express';
import path from 'path';
import http from 'http';
import { spawn } from 'child_process';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = 3000;
const FASTAPI_PORT = 8001;
const isProd = process.env.NODE_ENV === 'production';

// Ensure Python FastAPI server with trained A4 ONNX model is running
let fastApiProcess: any = null;

async function checkFastApiHealth(): Promise<boolean> {
  try {
    const res = await fetch(`http://127.0.0.1:${FASTAPI_PORT}/health`, { signal: AbortSignal.timeout(1000) });
    return res.ok;
  } catch {
    return false;
  }
}

async function ensureFastApiRunning() {
  const isHealthy = await checkFastApiHealth();
  if (isHealthy) {
    console.log(`[Server] FastAPI daemon already running on port ${FASTAPI_PORT}`);
    return;
  }

  console.log(`[Server] Starting FastAPI daemon (uvicorn backend.main:app) on port ${FASTAPI_PORT}...`);
  fastApiProcess = spawn(
    'python3',
    ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', String(FASTAPI_PORT)],
    {
      cwd: process.cwd(),
      stdio: ['ignore', 'inherit', 'inherit'],
    }
  );

  fastApiProcess.on('exit', (code: number | null) => {
    console.warn(`[Server] FastAPI process exited with code ${code}. Auto-restarting...`);
    fastApiProcess = null;
    setTimeout(() => {
      ensureFastApiRunning().catch(console.error);
    }, 1000);
  });

  // Wait for health
  for (let i = 0; i < 20; i++) {
    await new Promise((r) => setTimeout(r, 500));
    if (await checkFastApiHealth()) {
      console.log(`[Server] FastAPI daemon successfully initialized and healthy on port ${FASTAPI_PORT}`);
      return;
    }
  }
  console.warn(`[Server] FastAPI daemon did not report healthy within 10s`);
}

// 1. Authoritative Predict Endpoint: Pipe multipart stream directly to FastAPI on 8001
app.post('/predict', (req: Request, res: Response) => {
  const proxyReq = http.request(
    {
      hostname: '127.0.0.1',
      port: FASTAPI_PORT,
      path: '/predict',
      method: 'POST',
      headers: req.headers,
    },
    (proxyRes) => {
      res.writeHead(proxyRes.statusCode || 200, proxyRes.headers);
      proxyRes.pipe(res);
    }
  );

  proxyReq.on('error', (err) => {
    console.error('[Proxy /predict Error]', err.message);
    res.status(502).json({
      status: 'error',
      face_detected: false,
      error: 'FastAPI inference service unreachable',
      backend_connected: false,
    });
  });

  req.pipe(proxyReq);
});

// 2. Authoritative Health Endpoint: Forward to FastAPI on 8001
app.get('/health', async (_req: Request, res: Response) => {
  try {
    const fastApiResponse = await fetch(`http://127.0.0.1:${FASTAPI_PORT}/health`, {
      signal: AbortSignal.timeout(2000),
    });
    const data = await fastApiResponse.json();
    return res.status(fastApiResponse.status).json(data);
  } catch (err: unknown) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    console.warn('[Proxy /health Warning]', errorMsg);
    return res.status(503).json({
      status: 'error',
      model_loaded: false,
      pipeline_ready: false,
      error: errorMsg,
    });
  }
});

// 3. Service Info
app.get('/api/info', async (_req: Request, res: Response) => {
  try {
    const r = await fetch(`http://127.0.0.1:${FASTAPI_PORT}/`);
    const data = await r.json();
    return res.json(data);
  } catch {
    return res.json({
      service: 'Emotion Detector Backend',
      status: 'online',
      model: 'A4_spatial_frequency_geometry',
    });
  }
});

// Body parsers for any other routes
app.use(express.json({ limit: '25mb' }));
app.use(express.urlencoded({ extended: true, limit: '25mb' }));

async function main() {
  await ensureFastApiRunning();

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
    console.log(`Emotion Detector server running on http://0.0.0.0:${PORT}`);
  });
}

main().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
