[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/vickyair1/gta6-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/vickyair1/gta6-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch forecast

A program that forecasts how many people watch Grand Theft Auto VI live on
Twitch in each hour of its release day (2026-11-19, UTC), hours 2-24, as five
quantiles an hour — from the first hour's print plus the launch-day curves of
past big launches. Built for the [make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).

## How it works

The forecast is built in log-ratios, `ln(viewers_h / viewers_1)`, from the
case's comparables — the full hourly Twitch curves of past launches:

- **Pool.** Only comparables with a complete 24-hour release-day curve whose
  hour 1 was already live but not yet the peak (`v1/peak` in [0.10, 0.80]).
  Single-unlock launches (hour 1 near zero, 100-1000x rises) and launches
  that peaked in hour 1 have different shapes and poison the pool.
- **Component 1 (70%): per-hour ridge regression** on `[1, ln(v1),
  min(2, pre/v1)]`, where `pre` is the launch's mean hourly viewers 3-14
  days before release day (an early-access / pre-hype gauge). `ln(v1)` is
  clamped to the pool's 10th-90th percentile range: the size-to-shape
  gradient is real inside the observed range, but a linear extrapolation
  past it predicts a mega-launch that never rises above hour 1,
  contradicting the biggest observed launches (all rose 1.24-1.30x by
  hour 18).
- **Component 2 (30%): similarity-weighted local mean**, Gaussian kernel on
  `ln(v1)` (bandwidth 1.5), so a launch bigger than anything seen behaves
  like the biggest launches seen.
- **Spread.** Each component's residual/weighted standard deviation of
  log-ratios, blended the same way, widened 10% around the median. The
  pinball loss punishes a too-narrow 5%/95% far more than a too-wide one.

Leave-one-out on the 21 rolling-midnight-like comparables: pinball loss
**0.220** vs **0.306** for the persistence baseline (a 28% improvement),
90% coverage 0.903, typical median miss 44%.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `<id>.in` in the input directory, writes `<id>.out` per case:
`{"hours":[{"hour":2,"q":[q05,q25,q50,q75,q95]}, ...]}`, 23 rows, hours
2-24. Node 22, standard library only, no network, deterministic — two
runs on the same case write the same bytes, in well under a second.
