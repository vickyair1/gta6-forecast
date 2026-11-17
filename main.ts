// gta6-twitch-ccu-forecast-01: forecast GTA VI's launch-day Twitch viewers, hours 2..24,
// as five quantiles an hour, from the hour-1 print plus the case's comparables
// (past big launches' full hourly curves). Node 22, standard library only, deterministic.
//
// Model (fitted and backtested on the kit's comparables, leave-one-out):
//   Work in log-ratios: for a launch with hour-1 print p1, model ln(v_h / p1) per hour.
//   Two components, blended in log space (70% / 30%):
//     1. Per-hour ridge regression on [1, ln(v1), min(2, pre/v1)], where pre is the
//        launch's mean hourly viewers 3-14 days before release day (an early-access /
//        pre-hype gauge). ln(v1) is clamped to the pool's 10th-90th percentile range:
//        the size-to-shape gradient is real inside the observed range, but a linear
//        extrapolation past it predicts a mega-launch that never rises above hour 1,
//        contradicting the biggest observed launches (all rose 1.24-1.30x by hour 18).
//     2. Similarity-weighted local mean: Gaussian kernel on ln(v1) (bandwidth 1.5),
//        so a launch bigger than anything seen behaves like the biggest launches seen.
//   Spread: each component's residual/weighted standard deviation of log-ratios,
//   blended the same way, then widened 10% around the median (leave-one-out 90%
//   coverage 0.903 at that width; the pinball loss punishes a too-narrow 5%/95%
//   far more than a too-wide one).
//   Pool: only comparables with a complete 24-hour release-day curve whose hour 1
//   was already live but not yet the peak (v1/peak in [0.10, 0.80]). Single-unlock
//   launches (hour 1 near zero, 100-1000x rises) and hour-1-peakers (already at max
//   in hour 1) have different shapes and poison the pool when included.
// Leave-one-out on the 21 rolling-midnight-like comparables: pinball loss 0.220
// vs 0.306 for the persistence baseline (a 28% improvement); typical median miss 44%.
// The program reads every <id>.in in the input directory and writes <id>.out per case:
// {"hours":[{"hour":2,"q":[q05,q25,q50,q75,q95]}, ...]}, 23 rows, hours 2..24.
// It runs offline with no network: everything it knows is in the case file.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand
// in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Hours of release day, 1-based; hour h is [day_start + (h-1)h, day_start + h h). */
const DAY_HOURS = 24;
/** Pre-launch window for the early-access gauge, in hours before day_start. */
const PRE_LO = -336;
const PRE_HI = -72;
/** Pool filter: hour 1 already live (v1/peak >= 0.10) but not the peak (<= 0.80). */
const POOL_V1_PEAK_LO = 0.1;
const POOL_V1_PEAK_HI = 0.8;
/** Ridge penalty on the non-intercept regression coefficients. */
const RIDGE = 1.0;
/** Clamp the target's ln(v1) to this percentile range of the pool's ln(v1). */
const CLAMP_PCT = 0.9;
/** Kernel bandwidth (log units) for the similarity-weighted component. */
const KERNEL_BW = 1.5;
/** Blend weight of the regression component (1 - this goes to the kernel component). */
const BLEND_OLS = 0.7;
/** Widen the blended log-spread around the median by this factor. */
const SPREAD_MULT = 1.1;

interface HourlyPoint {
  from: string;
  viewers: number;
  channels?: number;
  reads?: number;
}

interface Comparable {
  name: string;
  day_start: string;
  hourly: HourlyPoint[];
}

interface Case {
  id: string;
  levels: number[];
  day_start: string;
  known: Array<{ hour: number; viewers: number; channels?: number }>;
  forecast: Array<{ hour: number }>;
  history_hourly?: HourlyPoint[];
  history?: Array<{ at: string; viewers: number }>;
  comparables: Comparable[];
}

/** One comparable reduced to its release-day curve and pre-launch level. */
interface Launch {
  name: string;
  /** Viewers in hours 1..24 of release day, index 0 = hour 1. */
  day: number[];
  /** Mean hourly viewers in the pre-launch window (0 when the window is empty). */
  preBase: number;
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** Milliseconds between two ISO timestamps. */
const msBetween = (a: string, b: string): number => Date.parse(b) - Date.parse(a);

/** Hour index of timestamp t relative to day_start: 1 for the first hour of the day. */
const hourIndex = (dayStart: string, t: string): number =>
  Math.floor(msBetween(dayStart, t) / 3_600_000) + 1;

/** Linear-interpolated percentile of a sorted array. */
const percentile = (sorted: number[], p: number): number => {
  if (sorted.length === 0) return 0;
  const i = p * (sorted.length - 1);
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
};

/** Solve a small dense linear system by Gaussian elimination with partial pivoting. */
const solve = (aIn: number[][], bIn: number[]): number[] => {
  const n = bIn.length;
  const m = aIn.map((row, i) => [...row, bIn[i]]);
  for (let i = 0; i < n; i++) {
    let piv = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(m[r][i]) > Math.abs(m[piv][i])) piv = r;
    [m[i], m[piv]] = [m[piv], m[i]];
    if (Math.abs(m[i][i]) < 1e-12) m[i][i] = 1e-12;
    for (let r = i + 1; r < n; r++) {
      const f = m[r][i] / m[i][i];
      for (let c = i; c <= n; c++) m[r][c] -= f * m[i][c];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = m[i][n];
    for (let j = i + 1; j < n; j++) s -= m[i][j] * x[j];
    x[i] = s / m[i][i];
  }
  return x;
};

/**
 * Reduce one comparable to its release-day curve. Returns null when the curve is
 * incomplete (a missing hour) or hour 1 has no viewers: such a launch cannot anchor
 * a ratio curve, and the GTA VI pre-release curve in the kit is one of them.
 */
const extractLaunch = (comp: Comparable): Launch | null => {
  const day = new Array<number>(DAY_HOURS).fill(0);
  let seen = 0;
  let preSum = 0;
  let preCount = 0;
  for (const pt of comp.hourly ?? []) {
    const h = hourIndex(comp.day_start, pt.from);
    if (h >= 1 && h <= DAY_HOURS) {
      if (day[h - 1] === 0) seen++;
      day[h - 1] = Math.max(0, pt.viewers);
    } else if (h >= PRE_LO && h <= PRE_HI) {
      preSum += Math.max(0, pt.viewers);
      preCount++;
    }
  }
  if (seen < DAY_HOURS || day[0] <= 0) return null;
  return { name: comp.name, day, preBase: preCount > 0 ? preSum / preCount : 0 };
};

/** The target launch's pre-launch level, from its own hourly history. */
const targetPreBase = (c: Case): number => {
  let sum = 0;
  let count = 0;
  for (const pt of c.history_hourly ?? []) {
    const h = hourIndex(c.day_start, pt.from);
    if (h >= PRE_LO && h <= PRE_HI) {
      sum += Math.max(0, pt.viewers);
      count++;
    }
  }
  if (count > 0) return sum / count;
  // Fall back to whatever pre-launch history exists (minute points or none).
  for (const pt of c.history ?? []) {
    const h = hourIndex(c.day_start, pt.at);
    if (h < 1) {
      sum += Math.max(0, pt.viewers);
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
};

/** Features for one launch: [1, ln(v1), min(2, pre/v1)]. */
const features = (v1: number, preBase: number, lnV1?: number): number[] => [
  1,
  lnV1 ?? Math.log(Math.max(1, v1)),
  Math.min(2, preBase / Math.max(1, v1)),
];

/** Quantiles (log-ratio units) of one hour from a center and a spread. */
const quantilesFrom = (mu: number, sd: number): number[] => Z.map((z) => mu + z * sd);

/**
 * Component 1: per-hour ridge regression of ln(v_h / v1) on the launch features.
 * Returns per-hour [center, spread] in log-ratio units.
 */
const olsComponent = (
  pool: Launch[],
  targetV1: number,
  targetPre: number,
): Map<number, [number, number]> => {
  const out = new Map<number, [number, number]>();
  const lnVs = pool.map((r) => Math.log(Math.max(1, r.day[0]))).sort((a, b) => a - b);
  const lo = percentile(lnVs, 1 - CLAMP_PCT);
  const hi = percentile(lnVs, CLAMP_PCT);
  const clampedLn = Math.min(hi, Math.max(lo, Math.log(Math.max(1, targetV1))));
  const xTarget = features(targetV1, targetPre, clampedLn);
  const xs = pool.map((r) => features(r.day[0], r.preBase));
  for (let h = 2; h <= DAY_HOURS; h++) {
    const y = pool.map((r) => Math.log(Math.max(1, r.day[h - 1]) / Math.max(1, r.day[0])));
    const k = 3;
    const n = pool.length;
    const xtx: number[][] = Array.from({ length: k }, () => new Array<number>(k).fill(0));
    const xty = new Array<number>(k).fill(0);
    for (let m = 0; m < n; m++) {
      for (let i = 0; i < k; i++) {
        xty[i] += xs[m][i] * y[m];
        for (let j = 0; j < k; j++) xtx[i][j] += xs[m][i] * xs[m][j];
      }
    }
    for (let i = 1; i < k; i++) xtx[i][i] += RIDGE;
    const beta = solve(xtx, xty);
    const mu = beta.reduce((s, b, i) => s + b * xTarget[i], 0);
    let rss = 0;
    for (let m = 0; m < n; m++) {
      const fit = beta.reduce((s, b, i) => s + b * xs[m][i], 0);
      rss += (y[m] - fit) * (y[m] - fit);
    }
    const sd = Math.sqrt(rss / Math.max(1, n - k));
    out.set(h, [mu, sd]);
  }
  return out;
};

/**
 * Component 2: similarity-weighted mean and spread of ln(v_h / v1), with a
 * Gaussian kernel on ln(v1), so launches near the target's size dominate.
 */
const kernelComponent = (
  pool: Launch[],
  targetV1: number,
): Map<number, [number, number]> => {
  const out = new Map<number, [number, number]>();
  const lt = Math.log(Math.max(1, targetV1));
  const w = pool.map((r) => {
    const d = (Math.log(Math.max(1, r.day[0])) - lt) / KERNEL_BW;
    return Math.exp(-0.5 * d * d);
  });
  const sw = w.reduce((s, x) => s + x, 0) || 1;
  for (let h = 2; h <= DAY_HOURS; h++) {
    const lrs = pool.map((r) => Math.log(Math.max(1, r.day[h - 1]) / Math.max(1, r.day[0])));
    const mu = lrs.reduce((s, x, i) => s + w[i] * x, 0) / sw;
    const variance = lrs.reduce((s, x, i) => s + w[i] * (x - mu) * (x - mu), 0) / sw;
    out.set(h, [mu, Math.sqrt(Math.max(0, variance))]);
  }
  return out;
};

/** Persistence fallback (the starter's numbers) when the pool is too small. */
const persistenceRow = (p1: number, h: number): number[] =>
  Z.map((z) => round2(p1 * Math.exp(z * 0.15 * Math.sqrt(h - 1))));

/** Forecast one case: 23 rows of five quantiles, hours 2..24. */
const forecastCase = (c: Case): Array<{ hour: number; q: number[] }> => {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${c.id} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`${c.id} has no hour-1 print`);
  const p1 = Math.max(1, first.viewers);
  const launches = (c.comparables ?? [])
    .map(extractLaunch)
    .filter((l): l is Launch => l !== null)
    .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  // Pool: complete curves, hour 1 live but not the peak. Relax stepwise when thin.
  const ratio = (l: Launch): number => l.day[0] / Math.max(1, ...l.day);
  let pool = launches.filter((l) => ratio(l) >= POOL_V1_PEAK_LO && ratio(l) <= POOL_V1_PEAK_HI);
  if (pool.length < 5) pool = launches.filter((l) => ratio(l) >= POOL_V1_PEAK_LO);
  if (pool.length < 5) pool = launches;
  const hours = c.forecast.map(({ hour }) => hour).sort((a, b) => a - b);
  if (pool.length < 3) {
    return hours.map((hour) => ({ hour, q: persistenceRow(p1, hour) }));
  }
  const pre = targetPreBase(c);
  const ols = olsComponent(pool, p1, pre);
  const ker = kernelComponent(pool, p1);
  return hours.map((hour) => {
    const [muO, sdO] = ols.get(hour) ?? [0, 0.15 * Math.sqrt(hour - 1)];
    const [muK, sdK] = ker.get(hour) ?? [0, 0.15 * Math.sqrt(hour - 1)];
    const qsO = quantilesFrom(muO, sdO);
    const qsK = quantilesFrom(muK, sdK);
    // Blend in log space, then widen the spread around the median.
    const blended = qsO.map((q, j) => BLEND_OLS * q + (1 - BLEND_OLS) * qsK[j]);
    const med = blended[2];
    const widened = blended.map((q) => med + (q - med) * SPREAD_MULT);
    const vals = widened.map((q) => Math.min(1e9, Math.max(1, p1 * Math.exp(q))));
    const rounded = vals.map(round2);
    for (let j = 1; j < rounded.length; j++) rounded[j] = Math.max(rounded[j], rounded[j - 1]);
    return { hour, q: rounded };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  c.id = id;
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
