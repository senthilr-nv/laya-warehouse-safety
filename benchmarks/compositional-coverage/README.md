# Compositional-coverage diagnostic

The protocol and manifests were committed as `61413e8` before training. At that freeze point, no
new model had been trained and the final manifest had not been evaluated. One train/dev run and one
final comparison run were subsequently completed from that commit.

This experiment tests one narrow hypothesis: the prior combined pallet-and-worker failure was
primarily caused by missing combined-family training coverage rather than an inherent Laya
limitation. It does not test real warehouse safety or broad generalization.

The prior evidence at commit `272c124` remains unchanged under [`../final/`](../final/). Its exact
fine-tuned checkpoint remains on the DGX Spark with weights SHA-256
`a4a3bd6746d5a01b6147cf673f1d7a8909406cf72dcc0f285981f738c6be44a8` and will be evaluated as the
unchanged baseline in the same final run as the new checkpoint.

## Frozen split

The checked-in JSON files contain every scenario field. IDs and seeds do not establish separation.
The leakage boundary uses two signatures:

- Geometry: grid, robot start, goal columns, stationary actor positions, and moving-actor aisle
  rows. Scenario and actor IDs are excluded.
- Worker trajectory: grid width, actor kind, initial column, aisle row, and horizontal velocity.
  This identifies the phase and direction of the deterministic bounce trajectory; IDs and episode
  seeds are excluded.

Every final geometry signature and every final worker trajectory signature is absent from both
train and dev. Tests pin the files, recompute all three digest types, and enforce those set
intersections as empty.

| Role | Stationary | Crossing | Combined | Scenario digest |
|---|---:|---:|---:|---|
| Train | 20 | 20 | 8 | `175534e88d06d9fd2a2b97613ea8e07c5256bff50966937660a06a25f4d65430` |
| Dev | 6 | 6 | 8 | `162edfce1683c848657271eac2020924e505cd53e50d5c163b2c6f1dbf396293` |
| Final | 8 | 8 | 16 | `bb40999de39c6f921c5a960de4e9dee4be9a13aab351f2719441d9d0f1ae4bfe` |

| Role | Geometry-signature digest | Worker-trajectory-signature digest |
|---|---|---|
| Train | `c49f249ecb17fd2c06db6594ae73ab55f96d99e82a2fcdbaffb50a7e5dd7dd40` | `3429cf915f9f8072e8e355f4327019e22e212c59a33d79a68a0b9afc46459eb8` |
| Dev | `4e706d11fab7460e743cddaeeb4d701c56afcb06dbaaf77971a898de146ca77c` | `3b3f165095c1240cfc888162a5eb35538ddeddaf18fd3c0cc90a57ee9a3b8544` |
| Final | `ec2746f5ec06c44f96649a463a9fa19811d7be206e2d9675bdf57f710d55b6b8` | `9ab0d9f08b6e21cac3a19e950e31d0761b2c4b9f59025ddb045d1800a7e2b773` |

The train manifest copies the prior 20 stationary and 20 crossing physical layouts, then adds eight
combined layouts with pallet rows 2-3 and worker rows 4-5. The dev manifest copies the prior six
stationary and six crossing validation layouts, then adds eight combined layouts with pallet rows
4-5 and worker rows 2-3. The final manifest is separately enumerated: stationary pallets occupy
rows 1, 6, or 7; moving workers occupy rows 1 or 7; combined pallets occupy rows 1, 6, or 7. One
stationary final scenario starts the robot in column 2 so all eight stationary physical layouts are
distinct from train and dev.

## Fixed protocol

Only train and dev may affect loss weighting or checkpoint selection. The final manifest must not
be loaded by training and must be executed once after the checkpoint is fixed.

The new checkpoint keeps the prior recipe: Laya 0.3.3, base snapshot
`1c5edc17a7acd8701df6fc341c0d179f1c62c982`, seed 17, eight epochs, batch size 16, encoder learning
rate `2.5e-5`, head learning rate `1e-4`, horizontal mirror augmentation, inverse-frequency weight
per question and label, and the same lexicographic dev selection order. The only planned data
change is the eight combined train scenarios and eight combined dev scenarios.

The final command evaluates four controllers over the same scenarios in one process: heuristic,
seeded random, the preserved baseline checkpoint, and the new coverage checkpoint. Both policy-only
and layered tracks retain the existing 40-tick limit, observation schema, typed questions, oracle,
path-blocked horizon, safety shield, controller semantics, and metrics. The hardware and pinned
container remain the same DGX Spark setup.

Training command, to be run only from the committed manifest implementation:

```sh
python scripts/train_laya_policy.py \
  --model convaiinnovations/laya \
  --model-revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982 \
  --train-manifest benchmarks/compositional-coverage/manifests/train.json \
  --dev-manifest benchmarks/compositional-coverage/manifests/dev.json \
  --epochs 8 \
  --seed 17 \
  --implementation-commit COMMITTED_SHA \
  --output /results/compositional-coverage-v1/laya-combined-coverage
```

The final runner refuses to overwrite its output and records portable selected replays, controller
weight hashes, training provenance, runtime evidence, and a deterministic result digest. Exact
model inference will not be reproducible from this repository because neither checkpoint weights
will be published. The manifests, aggregate metrics, raw selected records, hashes, and deterministic
replay verification will remain independently checkable.

Final command, executed once after checkpoint selection:

```sh
python scripts/run_coverage_benchmark.py \
  --baseline-model /results/laya-final-model-55f60f2 \
  --coverage-model /results/compositional-coverage-v1/laya-combined-coverage-61413e8 \
  --final-manifest benchmarks/compositional-coverage/manifests/final.json \
  --implementation-commit 61413e8d4a42a4ff72e7b180f831ecc3fd9b455e \
  --device cuda \
  --max-ticks 40 \
  --output /results/compositional-coverage-v1/final/report.json \
  --episodes-dir /results/compositional-coverage-v1/final/episodes
```

## Training evidence

Training produced 461 source states and 922 horizontally mirrored states. The dev set had 193
states. Epoch 7 won the registered selection order with 0.6667 minimum action-label accuracy,
0.8341 action macro accuracy, 0.9326 `path_blocked` accuracy, and 0.0624 Brier score. Training took
535.16 seconds on the DGX Spark.

- Coverage checkpoint weights SHA-256:
  `f48d28b16208ecabedd6d165ccc65c7fa7a387fc7c0751a5c2d43995099b96a5`
- Base checkpoint snapshot: `1c5edc17a7acd8701df6fc341c0d179f1c62c982`
- Base weights SHA-256:
  `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`
- Training metrics SHA-256:
  `90583751e797af6190c0918263882689d0be3e83c8bde515c4078dd3ad99e147`

## Single final result

The final manifest was executed once after epoch selection. The report's deterministic result
digest is:

```text
76c46855490e0e6a5ea5314462575a4b503162cbae6ff5b310b13aa5ba87615d
```

### Overall outcomes

| Controller | Track | Completion | Collision | Timeout | Unsafe request | Shield interventions | `path_blocked` accuracy | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Heuristic | Policy only | 87.5% | 0% | 12.5% | 0% | 0 | — | — |
| Heuristic | Layered | 87.5% | 0% | 12.5% | 0% | 0 | — | — |
| Preserved Laya baseline | Policy only | 25% | 71.88% | 3.12% | 36% | 0 | 56% | 0.3355 |
| Coverage-trained Laya | Policy only | 18.75% | 81.25% | 0% | 27.27% | 0 | 75.76% | 0.1840 |
| Preserved Laya baseline | Layered | 50% | 0% | 50% | 76.25% | 579 | 27.62% | 0.4848 |
| Coverage-trained Laya | Layered | 87.5% | 0% | 12.5% | 8.27% | 46 | 53.6% | 0.3391 |
| Seeded random | Policy only | 12.5% | 71.88% | 15.62% | 11.26% | 0 | — | — |
| Seeded random | Layered | 46.88% | 0% | 53.12% | 9.72% | 46 | — | — |

### Laya outcomes by family

| Checkpoint | Family | Track | Completion | Collision | Timeout | Unsafe request | Shield interventions | `path_blocked` accuracy | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | Stationary | Policy only | 62.5% | 25% | 12.5% | 37.93% | 0 | 54.02% | 0.3554 |
| Coverage | Stationary | Policy only | 37.5% | 62.5% | 0% | 14.29% | 0 | 74.29% | 0.2056 |
| Baseline | Crossing | Policy only | 37.5% | 62.5% | 0% | 11.9% | 0 | 80.95% | 0.1729 |
| Coverage | Crossing | Policy only | 37.5% | 62.5% | 0% | 13.95% | 0 | 65.12% | 0.2494 |
| Baseline | Combined | Policy only | 0% | 100% | 0% | 76.19% | 0 | 14.29% | 0.5779 |
| Coverage | Combined | Policy only | 0% | 100% | 0% | 76.19% | 0 | 100% | 0.0141 |
| Baseline | Stationary | Layered | 87.5% | 0% | 12.5% | 33.33% | 5 | 62.96% | 0.2908 |
| Coverage | Stationary | Layered | 100% | 0% | 0% | 9.88% | 8 | 88.89% | 0.0968 |
| Baseline | Crossing | Layered | 100% | 0% | 0% | 8.64% | 7 | 87.65% | 0.1175 |
| Coverage | Crossing | Layered | 100% | 0% | 0% | 8.33% | 7 | 79.76% | 0.1504 |
| Baseline | Combined | Layered | 6.25% | 0% | 93.75% | 92.8% | 567 | 13.42% | 0.5678 |
| Coverage | Combined | Layered | 75% | 0% | 25% | 7.93% | 31 | 40.66% | 0.4298 |

`path_blocked` is measured on each controller's visited states. Different policies can visit
different state distributions, especially after shield intervention, so cross-controller Brier and
accuracy comparisons are descriptive rather than a shared fixed-state calibration test.

### Runtime

| Checkpoint | Model load | First inference | Warm count | Warm p50 | Warm p95 |
|---|---:|---:|---:|---:|---:|
| Baseline | 7,784.3661 ms | 535.2003 ms | 949 | 27.2160 ms | 28.4222 ms |
| Coverage | 5,784.5387 ms | 28.1142 ms | 654 | 27.0143 ms | 28.0897 ms |

The baseline was loaded and evaluated first, so the first-inference figures include different GPU
warm-up states and are not a controlled cold-latency comparison. Warm latency is nearly unchanged.

## Evidence

- [`development/training_metrics.json`](development/training_metrics.json) — complete train/dev
  history and checkpoint selection.
- [`final/report.json`](final/report.json) — aggregate and per-episode final metrics, provenance,
  model hashes, and runtime evidence.
- [`final/episodes/`](final/episodes/) — 24 portable, deterministic replay records covering every
  controller, track, and family.

The report file SHA-256 is
`474f85e2b03682cfae12141967df178e74758359a90970e77e6937fd2b1b7061`. All 24 selected records
passed deterministic replay verification after transfer from the DGX Spark.

## Interpretation rule

- A large improvement by the coverage checkpoint on held-out combined geometries supports the
  training-coverage hypothesis for this simulator.
- Little or no improvement points toward representation, objective, or planning limitations.
- Either outcome remains synthetic evidence and cannot support a warehouse-safety, broad
  generalization, or fundamental Laya-capability claim.

## Interpretation

The simple form of the hypothesis is not supported for the policy acting alone. On held-out
combined scenarios, both checkpoints completed 0% and collided in 100% of policy-only episodes.
The coverage checkpoint made `path_blocked` much more accurate on those short trajectories, but
that signal did not produce a successful action policy. Policy-only performance also regressed on
the stationary family.

Training coverage did materially change the layered system. On combined scenarios with the
unchanged shield, completion increased from 6.25% to 75%, timeout fell from 93.75% to 25%, and
shield interventions fell from 567 to 31. This supports a narrower conclusion: combined examples
helped the model produce behavior that the deterministic shield could recover into successful
routes. Those completions remain shield-dependent and do not demonstrate a collision-safe learned
policy.

The mixed outcome points beyond data coverage alone toward action representation, training
objective, sequential planning, or interference between scenario families. This single synthetic
diagnostic cannot distinguish those explanations and does not establish an inherent limitation or
fundamental capability of Laya.
