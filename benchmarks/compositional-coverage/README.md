# Compositional-coverage diagnostic

Status at manifest freeze: protocol registered; no new model had been trained and the final manifest
had not been evaluated.

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

## Interpretation rule

- A large improvement by the coverage checkpoint on held-out combined geometries supports the
  training-coverage hypothesis for this simulator.
- Little or no improvement points toward representation, objective, or planning limitations.
- Either outcome remains synthetic evidence and cannot support a warehouse-safety, broad
  generalization, or fundamental Laya-capability claim.
