// Browser port of the ISL pipeline: keypoints.py (MediaPipe hand + pose landmarkers),
// features.py (signer-invariant features, 64-frame resample, velocities) and the BiLSTM,
// exported to ONNX by `python -m isl.export_onnx` and run with onnxruntime-web.
//
// Feature arithmetic follows numpy's float32 rules step by step: every operation numpy does in
// float32 is rounded with Math.fround here too, and resampling uses np.interp's float64 formula.
// The same landmarks therefore give bit-identical model inputs (checked by web/check_features.mjs).
//
// The pure feature functions have no imports, so Node can load this file; the MediaPipe and
// onnxruntime-web runtimes are imported from jsDelivr only when loadRuntimes() is called.

export const SEQ_LEN = 64;
export const N_HAND = 21;
export const POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24];   // nose, shoulders, elbows, wrists, hips
export const F_BASE = 239;                                     // 27 pose + 84 global hand + 126 local hand + 2 flags
export const F_TOTAL = F_BASE * 2;                             // + frame-to-frame velocity
export const MIN_CONF = 0.4;                                   // keypoints.py LandmarkExtractor default
export const STRIDE = 3;                                       // dataset clips: every 3rd frame (README, extract)

export const CDN = {
  mediapipe: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0',   // same release as the Python mediapipe 1.1.0
  ort: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
};

const f = Math.fround;
const L_SHOULDER = 11, R_SHOULDER = 12;
const TINY = f(1e-3);    // numpy compares float32 arrays with a float32 copy of the Python 1e-3

// ------------------------------------------------------------------ features.py
const norm2 = (a, b) => f(Math.sqrt(f(f(a * a) + f(b * b))));    // np.linalg.norm over (x, y), float32

/** frames[t] = { pose: Float32Array(33*4), left: Float32Array(21*3), right: Float32Array(21*3),
 *  mask: Float32Array(2) }, the arrays keypoints.py returns per frame. Returns (T, 239) row-major. */
export function buildFrameFeatures(frames) {
  const T = frames.length, out = new Float32Array(T * F_BASE);
  for (let t = 0; t < T; t++) {
    const { pose, left, right, mask } = frames[t];
    let k = t * F_BASE;
    // _body_frame: shoulder-centred origin and shoulder-width scale
    const lsx = pose[L_SHOULDER * 4], lsy = pose[L_SHOULDER * 4 + 1];
    const rsx = pose[R_SHOULDER * 4], rsy = pose[R_SHOULDER * 4 + 1];
    const ox = f(f(lsx + rsx) / 2), oy = f(f(lsy + rsy) / 2);
    let s = norm2(f(lsx - rsx), f(lsy - rsy));
    if (s < TINY) s = 1;
    for (const i of POSE_IDX) {
      out[k++] = f(f(pose[i * 4] - ox) / s);
      out[k++] = f(f(pose[i * 4 + 1] - oy) / s);
      out[k++] = pose[i * 4 + 3];                                  // visibility
    }
    const hands = [[left, mask[0]], [right, mask[1]]];
    // global: where each hand is relative to the body
    for (const [hand, m] of hands) {
      for (let j = 0; j < N_HAND; j++) {
        out[k++] = f(f(f(hand[j * 3] - ox) / s) * m);
        out[k++] = f(f(f(hand[j * 3 + 1] - oy) / s) * m);
      }
    }
    // local: hand shape only, wrist-centred and scaled by wrist -> middle-finger MCP (landmark 9)
    for (const [hand, m] of hands) {
      const wx = hand[0], wy = hand[1], wz = hand[2];
      let hs = norm2(f(hand[27] - wx), f(hand[28] - wy));
      if (hs < TINY) hs = 1;
      for (let j = 0; j < N_HAND; j++) {
        out[k++] = f(f(f(hand[j * 3] - wx) / hs) * m);
        out[k++] = f(f(f(hand[j * 3 + 1] - wy) / hs) * m);
        out[k++] = f(f(f(hand[j * 3 + 2] - wz) / hs) * m);
      }
    }
    out[k++] = mask[0];
    out[k++] = mask[1];
  }
  for (let i = 0; i < out.length; i++) if (!Number.isFinite(out[i])) out[i] = 0;   // np.nan_to_num
  return out;
}

// numpy's np.interp computes slope * (x - xp[j]) + fp[j] in C, and on Apple Silicon the compiler
// fuses that into one multiply-add (a single rounding). fma() reproduces it exactly: an error-free
// product and sum, then the low parts rounded to odd before the final add (Boldo & Melquiond, 2008).
const bits = new Float64Array(1), words = new Uint32Array(bits.buffer);   // little-endian: words[0] = low
function split(a) { const c = 134217729 * a, hi = c - (c - a); return [hi, a - hi]; }   // Veltkamp, 2^27 + 1
function roundToOdd(s, err) {
  if (err === 0) return s;
  bits[0] = s;
  if (words[0] & 1) return s;                        // already odd
  const up = (s > 0) === (err > 0);                  // step away from zero when the error points that way
  if (up) { words[0] = (words[0] + 1) >>> 0; if (words[0] === 0) words[1]++; }
  else { if (words[0] === 0) words[1]--; words[0] = (words[0] - 1) >>> 0; }
  return bits[0];
}
export function fma(a, b, c) {
  const p = a * b, [ah, al] = split(a), [bh, bl] = split(b);
  const pe = ((ah * bh - p) + ah * bl + al * bh) + al * bl;          // a*b = p + pe exactly
  const th = c + p, tb = th - c, tl = (c - (th - tb)) + (p - tb);       // c + p = th + tl exactly
  const v = tl + pe, vb = v - tl, ve = (tl - (v - vb)) + (pe - vb);    // tl + pe = v + ve exactly
  return th + roundToOdd(v, ve);
}

/** np.linspace(0.0, 1.0, n): i * step, with the last point set exactly to 1. */
function linspace01(n) {
  const out = new Float64Array(n), step = 1 / (n - 1);
  for (let i = 0; i < n; i++) out[i] = i * step;
  out[n - 1] = 1;
  return out;
}

/** features.resample: linear interpolation of a (T, C) clip onto `length` frames (np.interp). */
export function resample(seq, T, C, length = SEQ_LEN) {
  if (T === length) return Float32Array.from(seq);
  const out = new Float32Array(length * C);
  if (T === 1) {
    for (let i = 0; i < length; i++) out.set(seq.subarray(0, C), i * C);
    return out;
  }
  const xp = linspace01(T), x = linspace01(length);
  let j = 0;
  for (let i = 0; i < length; i++) {
    const xv = x[i];
    while (j < T - 1 && xp[j + 1] <= xv) j++;                    // largest j with xp[j] <= xv
    const row = i * C;
    if (j === T - 1 || xp[j] === xv) {
      for (let c = 0; c < C; c++) out[row + c] = seq[j * C + c];
      continue;
    }
    const dx = xp[j + 1] - xp[j];
    for (let c = 0; c < C; c++) {
      const y0 = seq[j * C + c], y1 = seq[(j + 1) * C + c];
      const slope = (y1 - y0) / dx;
      let v = fma(slope, xv - xp[j], y0);
      if (Number.isNaN(v)) {                                       // np.interp's fallback order
        v = fma(slope, xv - xp[j + 1], y1);
        if (Number.isNaN(v) && y0 === y1) v = y0;
      }
      out[row + c] = v;
    }
  }
  return out;
}

/** features.with_deltas: append frame-to-frame velocity (first frame's velocity is zero). */
export function withDeltas(seq, T, C) {
  const out = new Float32Array(T * C * 2);
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < C; c++) {
      const v = seq[t * C + c];
      out[t * 2 * C + c] = v;
      out[t * 2 * C + C + c] = t === 0 ? f(v - v) : f(v - seq[(t - 1) * C + c]);
    }
  }
  return out;
}

/** Raw landmark frames -> the (64, 478) tensor the model reads, exactly as dataset.py builds it. */
export function framesToInput(frames) {
  const feats = buildFrameFeatures(frames);
  return withDeltas(resample(feats, frames.length, F_BASE), SEQ_LEN, F_BASE);
}

export function topK(probs, classes, k = 5) {
  return Array.from(probs, (p, i) => [classes[i], p]).sort((a, b) => b[1] - a[1]).slice(0, k);
}

export const prettyWord = w => w.replace(/_/g, ' ');

// ------------------------------------------------------------------ runtimes
let runtimes = null;

/** Import MediaPipe Tasks and onnxruntime-web (wasm backend) once. */
export function loadRuntimes() {
  runtimes ??= (async () => {
    const [mp, ort] = await Promise.all([
      import(`${CDN.mediapipe}/vision_bundle.mjs`),
      import(`${CDN.ort}ort.wasm.min.mjs`),
    ]);
    ort.env.wasm.wasmPaths = CDN.ort;
    ort.env.wasm.numThreads = 1;          // threads need cross-origin isolation, which static hosting can't set
    const vision = await mp.FilesetResolver.forVisionTasks(`${CDN.mediapipe}/wasm`);
    return { mp, ort, vision };
  })();
  return runtimes;
}

/** fetch() as bytes, reporting (loaded, total) as it streams. */
export async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = +res.headers.get('content-length') || 0;
  if (!res.body || !onProgress) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader(), parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); loaded += value.length; onProgress(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

// ------------------------------------------------------------------ keypoints.py
/** MediaPipe hand (2 hands) + pose landmarkers in VIDEO mode, CPU inference, with the Python
 *  confidence thresholds. One instance per clip, like extract.py, so tracking never carries over. */
export class LandmarkExtractor {
  static async create(vision, mp, handModel, poseModel, minConf = MIN_CONF) {
    const base = buf => ({ modelAssetBuffer: buf, delegate: 'CPU' });
    const [hands, pose] = await Promise.all([
      mp.HandLandmarker.createFromOptions(vision, {
        baseOptions: base(handModel), runningMode: 'VIDEO', numHands: 2,
        minHandDetectionConfidence: minConf, minHandPresenceConfidence: minConf, minTrackingConfidence: minConf,
      }),
      mp.PoseLandmarker.createFromOptions(vision, {
        baseOptions: base(poseModel), runningMode: 'VIDEO', numPoses: 1,
        minPoseDetectionConfidence: minConf, minPosePresenceConfidence: minConf, minTrackingConfidence: minConf,
      }),
    ]);
    return new LandmarkExtractor(hands, pose);
  }

  constructor(hands, pose) { this.hands = hands; this.pose = pose; this.lastTs = -1; }

  /** keypoints.LandmarkExtractor._frame: one RGB image -> pose (33,4), left (21,3), right (21,3), mask (2). */
  frame(image, tsMs) {
    const ts = Math.max(Math.trunc(tsMs), this.lastTs + 1);      // VIDEO mode needs strictly increasing stamps
    this.lastTs = ts;
    const hres = this.hands.detectForVideo(image, ts), pres = this.pose.detectForVideo(image, ts);
    const pose = new Float32Array(33 * 4);
    if (pres.landmarks.length) {
      pres.landmarks[0].forEach((lm, i) => { pose[i * 4] = lm.x; pose[i * 4 + 1] = lm.y; pose[i * 4 + 2] = lm.z; pose[i * 4 + 3] = lm.visibility || 0; });
    }
    const hands = [new Float32Array(N_HAND * 3), new Float32Array(N_HAND * 3)], mask = new Float32Array(2);
    hres.landmarks.forEach((lms, n) => {
      // MediaPipe labels handedness from the camera's view; "Left" is the signer's right hand
      const slot = hres.handedness[n][0].categoryName === 'Left' ? 1 : 0;
      if (mask[slot]) return;                                       // keep the first (more confident) duplicate
      lms.forEach((lm, j) => { hands[slot][j * 3] = lm.x; hands[slot][j * 3 + 1] = lm.y; hands[slot][j * 3 + 2] = lm.z; });
      mask[slot] = 1;
    });
    return { pose, left: hands[0], right: hands[1], mask };
  }

  close() { this.hands.close(); this.pose.close(); }
}

// ------------------------------------------------------------------ video files
function once(el, ev) { return new Promise((res, rej) => { el.addEventListener(ev, res, { once: true }); el.addEventListener('error', () => rej(el.error || new Error(`video ${ev} failed`)), { once: true }); }); }

/** Resolve when the frame at the current time is ready to draw after a seek. */
async function seek(video, t) {
  const done = once(video, 'seeked');
  video.currentTime = t;
  await done;
  // 'seeked' means the new frame is decoded; also wait for it to be presented when the page is visible
  // (hidden pages throttle timers to ~1 s, so the safety timeout would stall a long run)
  if (video.requestVideoFrameCallback && document.visibilityState === 'visible') {
    await new Promise(res => { const id = video.requestVideoFrameCallback(() => res()); setTimeout(() => { video.cancelVideoFrameCallback?.(id); res(); }, 120); });
  }
}

/** keypoints.LandmarkExtractor.run_video: every `stride`-th frame of a clip, with the same
 *  timestamps (int(idx / fps * 1000)). The browser can't step through frames like cv2.read(),
 *  so each kept frame is reached by seeking to its midpoint; `fps` and `frameCount` are the
 *  values OpenCV reports for the file. Calls onFrame(frame, idx, canvas) as it goes. */
export async function runVideo(extractor, video, { fps, frameCount, stride = STRIDE, canvas, onFrame, signal } = {}) {
  if (video.readyState < 1) await once(video, 'loadedmetadata');
  fps ||= 30;
  frameCount ||= Math.round(video.duration * fps);
  canvas ||= document.createElement('canvas');
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d', { willReadFrequently: false });
  const frames = [];
  for (let idx = 0; idx < frameCount; idx += stride) {
    if (signal?.aborted) throw new DOMException('stopped', 'AbortError');
    await seek(video, Math.min((idx + 0.5) / fps, video.duration));
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const fr = extractor.frame(canvas, idx / fps * 1000);
    frames.push(fr);
    onFrame?.(fr, idx, canvas);
  }
  return frames;
}

// ------------------------------------------------------------------ the model
export class Classifier {
  static async create(ort, modelBytes) {
    const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ['wasm'] });
    return new Classifier(ort, session);
  }

  constructor(ort, session) { this.ort = ort; this.session = session; }

  /** Raw landmark frames -> softmax probabilities over the 61 words. */
  async predict(frames) {
    const x = new this.ort.Tensor('float32', framesToInput(frames), [1, SEQ_LEN, F_TOTAL]);
    const out = await this.session.run({ features: x });
    return Float32Array.from(out.probs.data);
  }
}

// ------------------------------------------------------------------ drawing
const HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
const POSE_EDGES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24]];

/** Draw one frame's skeleton (the points the model uses) over an image of size w x h. */
export function drawFrame(ctx, fr, w, h, { pose = '#c6cddc', hand = '#d9b15c', mirror = false } = {}) {
  const X = x => (mirror ? 1 - x : x) * w, Y = y => y * h;
  ctx.lineWidth = Math.max(1.5, w / 320); ctx.lineCap = 'round';
  if (fr.pose[L_SHOULDER * 4 + 3] || fr.pose[L_SHOULDER * 4]) {
    ctx.strokeStyle = pose; ctx.beginPath();
    for (const [a, b] of POSE_EDGES) { ctx.moveTo(X(fr.pose[a * 4]), Y(fr.pose[a * 4 + 1])); ctx.lineTo(X(fr.pose[b * 4]), Y(fr.pose[b * 4 + 1])); }
    ctx.stroke();
  }
  [[fr.left, fr.mask[0]], [fr.right, fr.mask[1]]].forEach(([p, m]) => {
    if (!m) return;
    ctx.strokeStyle = hand; ctx.fillStyle = hand; ctx.beginPath();
    for (const [a, b] of HAND_EDGES) { ctx.moveTo(X(p[a * 3]), Y(p[a * 3 + 1])); ctx.lineTo(X(p[b * 3]), Y(p[b * 3 + 1])); }
    ctx.stroke();
    for (let j = 0; j < N_HAND; j++) { ctx.beginPath(); ctx.arc(X(p[j * 3]), Y(p[j * 3 + 1]), ctx.lineWidth * 1.2, 0, 7); ctx.fill(); }
  });
}
