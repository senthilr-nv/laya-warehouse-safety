# Benchmark evidence

This directory preserves the benchmark history, including unfavorable development results. The
development manifest informed training changes, so only the separately frozen final manifest is
reported as one-shot final evidence.

All runs used a DGX Spark with Laya 0.3.3 and the container pinned by this repository. The final
implementation commit was `55f60f2`.

## Evaluation chronology

The first development run used the natural, advance-dominant rollout loss. Its report digest is
`4b1b38bf4bc992187fc632089edc6dda8875a97092fae83d653cbdc279def9cd`. Laya completed 6.25% of
policy-only episodes with an 87.5% collision rate. With the shield, it completed 43.75%, timed out
56.25%, and made unsafe requests on 76.26% of decisions.

The second development run added inverse-frequency loss weighting. Its report digest is
`d279728d408273da9464efb88dcad160ed2d13a43554675efa1b3bda7015cb15`. Laya completed 31.25% of
policy-only episodes with a 62.5% collision rate. With the shield, it completed 50%, timed out 50%,
and made unsafe requests on 70.83% of decisions.

An eight-epoch development diagnostic exposed unstable lateral-label recall: validation contained
only three `shift_left` and three `shift_right` cases. No benchmark was run from that diagnostic.
The final recipe therefore added horizontal mirror augmentation, reported exact label supports, and
selected checkpoints by minimum action-label accuracy before macro or aggregate metrics.

These development results are retained under [`development/`](development/) and must not be cited
as untouched generalization evidence.

## One-shot final result

The final manifest was frozen before model execution. It has 32 physical states disjoint from the
development manifest and this pinned digest:

```text
98c7ac272381a3f6c5d8db26f90960d84055c7bc84eeb3d23425125520790d6b
```

The final report's deterministic result digest is:

```text
6d0a6c6f12a4e72a848d8a2f62373e2a93589b302c99349251e45498c23afc63
```

### Overall outcomes

| Controller | Track | Completion | Collision | Timeout | Unsafe request rate | Shield interventions |
|---|---|---:|---:|---:|---:|---:|
| Heuristic | Policy only | 84.38% | 0% | 15.62% | 0% | 0 |
| Heuristic | Layered | 84.38% | 0% | 15.62% | 0% | 0 |
| Laya | Policy only | 34.38% | 65.62% | 0% | 15.56% | 0 |
| Laya | Layered | 62.5% | 0% | 37.5% | 68.39% | 476 |
| Seeded random | Policy only | 12.5% | 59.38% | 28.12% | 8.91% | 0 |
| Seeded random | Layered | 31.25% | 0% | 68.75% | 11.7% | 73 |

### Laya by scenario family

| Family | Track | Completion | Collision | Timeout | `path_blocked` accuracy |
|---|---|---:|---:|---:|---:|
| Stationary pallet | Policy only | 100% | 0% | 0% | 88.89% |
| Stationary pallet | Layered | 100% | 0% | 0% | 88.89% |
| Crossing worker | Policy only | 37.5% | 62.5% | 0% | 78.05% |
| Crossing worker | Layered | 100% | 0% | 0% | 87.5% |
| Combined pallet and worker | Policy only | 0% | 100% | 0% | 18.18% |
| Combined pallet and worker | Layered | 25% | 0% | 75% | 20.4% |

The checkpoint was perfect on 114 validation states, but validation support was 96 `advance`, 3
`shift_left`, 3 `shift_right`, and 12 `wait`. Those figures did not predict combined-family
behavior. The final result does not establish compositional generalization: it shows strong
stationary-pallet behavior, a shield-dependent crossing-worker result, and a clear combined-family
failure.

The measured Laya runtime was 7,671.8053 ms model load, 707.5028 ms first inference, 27.1623 ms warm
p50, and 28.3339 ms warm p95 over 830 warm decisions. These measurements describe this container
run only.

## Artifacts

- [`final/report.json`](final/report.json) — versioned final report with all aggregate and
  per-episode metrics.
- [`final/training_metrics.json`](final/training_metrics.json) — train/validation history and exact
  label supports.
- [`final/episodes/`](final/episodes/) — 18 replay-verifiable records, one per controller, track,
  and family.
- [`development/unweighted-report.json`](development/unweighted-report.json) and
  [`development/unweighted-training.json`](development/unweighted-training.json) — initial
  development evidence.
- [`development/weighted-report.json`](development/weighted-report.json) and
  [`development/weighted-training.json`](development/weighted-training.json) — weighted
  development evidence.
- [`development/eight-epoch-training.json`](development/eight-epoch-training.json) — extended
  validation diagnostic.

This is simulator evidence, not a robotics safety claim.
