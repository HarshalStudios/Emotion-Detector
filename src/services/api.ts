/**
 * Emotion Detector - FastAPI ML Backend Client
 * Connects to POST /predict using multipart/form-data.
 * Schema adheres to the authoritative FastAPI backend contract:
 * {
 *   "status": "OK",
 *   "face_detected": true,
 *   "partial_face": false,
 *   "prediction": "Happy",
 *   "prediction_index": 1,
 *   "confidence": 0.4234,
 *   "probabilities": { ... },
 *   "geometry_valid": true,
 *   "detection_confidence": 0.8659,
 *   "head_pose": { "pitch": 11.84, "yaw": 20.14, "roll": -3.43 },
 *   "bbox": [200, 212, 152, 175]
 * }
 */

export const CANONICAL_EXPRESSIONS = [
  'Neutral',
  'Happy',
  'Sad',
  'Surprise',
  'Fear',
  'Disgust',
  'Angry',
] as const;

export type ExpressionType = typeof CANONICAL_EXPRESSIONS[number];

export const EXPRESSION_EMOJIS: Record<string, string> = {
  Neutral: '😐',
  Happy: '😊',
  Sad: '🙁',
  Surprise: '😲',
  Fear: '😨',
  Disgust: '🤢',
  Angry: '😠',
};

export interface PredictResponse {
  status: string;
  face_detected: boolean;
  partial_face: boolean;
  prediction: string;
  prediction_index: number;
  confidence: number;
  probabilities: Record<string, number>;
  geometry_valid: boolean;
  detection_confidence: number;
  head_pose: {
    pitch: number;
    yaw: number;
    roll: number;
  };
  bbox: [number, number, number, number]; // [x, y, w, h]
  raw_logits?: Record<string, number>;
  backend_connected?: boolean;
  debug_info?: {
    blendshape_count: number;
    ratio_count: number;
    vector_dim: number;
    landmark_count: number;
    filtering: {
      min_width: number;
      min_height: number;
      max_abs_yaw: number;
      max_abs_pitch: number;
      max_abs_roll: number;
      min_detection_confidence: number;
    };
    blendshapes?: { name: string; value: number }[];
    ratios?: { name: string; value: number }[];
  };
}

export interface HealthCheckResult {
  connected: boolean;
  url: string;
  httpStatus: number | null;
  responseBody: string;
  error?: string;
  corsBlocked?: boolean;
}

export class ApiService {
  private static instance: ApiService;
  private backendReachable: boolean = false;
  private lastHealthResult: HealthCheckResult | null = null;

  public static getInstance(): ApiService {
    if (!ApiService.instance) {
      ApiService.instance = new ApiService();
    }
    return ApiService.instance;
  }

  /**
   * Verified base URL: environment variable VITE_API_URL or relative proxy fallback
   */
  public getBaseUrl(): string {
    const envApiUrl = import.meta.env.VITE_API_URL;
    if (envApiUrl && typeof envApiUrl === 'string' && envApiUrl.trim() !== '') {
      return envApiUrl.trim().replace(/\/+$/, '');
    }
    return '';
  }

  /**
   * Real GET /health connectivity verification
   */
  public async checkHealth(): Promise<HealthCheckResult> {
    const baseUrl = this.getBaseUrl();
    const targetUrl = baseUrl ? `${baseUrl}/health` : '/health';
    console.log(`[FastAPI Health] Requesting: GET ${targetUrl}`);

    try {
      const response = await fetch(targetUrl, {
        method: 'GET',
        headers: {
          Accept: 'application/json',
        },
      });

      const contentType = response.headers.get('content-type') || '';
      const text = await response.text();

      console.log(`[FastAPI Health] HTTP Status: ${response.status}, Content-Type: ${contentType}`);
      console.log(`[FastAPI Health] Body preview: ${text.slice(0, 160)}`);

      // Vite returns HTML index.html for non-existent API routes
      const isJson = contentType.includes('application/json');

      if (response.ok && isJson) {
        try {
          const json = JSON.parse(text);
          if (json.status === 'ok' || json.status === 'healthy') {
            this.backendReachable = true;
            const res: HealthCheckResult = {
              connected: true,
              url: targetUrl,
              httpStatus: response.status,
              responseBody: text,
            };
            this.lastHealthResult = res;
            return res;
          }
        } catch {
          // not valid JSON
        }
      }

      this.backendReachable = false;
      const res: HealthCheckResult = {
        connected: false,
        url: targetUrl,
        httpStatus: response.status,
        responseBody: text.slice(0, 200),
        error: isJson
          ? `Unexpected JSON payload: ${text}`
          : `HTTP ${response.status}: Server returned HTML (${contentType}), not JSON. Vite SPA fallback active; FastAPI backend is not serving /health at this endpoint.`,
      };
      this.lastHealthResult = res;
      return res;
    } catch (err: unknown) {
      this.backendReachable = false;
      const errorMsg = err instanceof Error ? err.message : String(err);
      console.warn(`[FastAPI Health] Connection failed:`, errorMsg);
      const res: HealthCheckResult = {
        connected: false,
        url: targetUrl,
        httpStatus: null,
        responseBody: '',
        error: errorMsg,
        corsBlocked: errorMsg.toLowerCase().includes('failed to fetch') || errorMsg.toLowerCase().includes('networkerror'),
      };
      this.lastHealthResult = res;
      return res;
    }
  }

  /**
   * Send captured video frame to POST /predict as multipart/form-data
   */
  public async predictImage(imageBlob: Blob): Promise<PredictResponse> {
    const baseUrl = this.getBaseUrl();
    const targetUrl = baseUrl ? `${baseUrl}/predict` : '/predict';
    const formData = new FormData();
    formData.append('file', imageBlob, 'frame.jpg');

    try {
      const response = await fetch(targetUrl, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data: PredictResponse = await response.json();
        this.backendReachable = true;
        return {
          ...data,
          backend_connected: true,
        };
      } else {
        throw new Error(`POST ${targetUrl} returned HTTP ${response.status}`);
      }
    } catch (err) {
      this.backendReachable = false;
      console.warn(`[FastAPI Predict] Prediction request failed:`, err);
      // Return honest UNAVAILABLE response - NEVER return a fake prediction or Happy/Neutral fallback
      return {
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
      };
    }
  }

  public isBackendOnline(): boolean {
    return this.backendReachable;
  }

  public getLastHealthResult(): HealthCheckResult | null {
    return this.lastHealthResult;
  }
}

export const apiService = ApiService.getInstance();
