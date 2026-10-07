# Experiment code

Synchronized aerial and ground RGB-D histories are encoded into a shared target-state distribution. A common future sample conditions both generated views, while surface-depth ordering determines view-specific visibility.

| Package | Purpose |
| --- | --- |
| `paper_experiments/agcr_exp` | Data indexing, observation head, historical association/filtering, dynamics, surface geometry, frontend training |
| `paired_interventions` | Four-cell native family construction/audit, evidence and camera objectives, ordered appearance gating, Wan adapters and RGB-D decoding |
| `base48` | Registered state/depth/video adaptation, receiver-only/paired conditions, shared sampling and independent evaluation |
| `research_extensions` | Matched evidence-loss repeats, ConcatGRU/informed CV controls, history/future overlays and decoded-video feedback |
| `external_baselines` | SynCamMaster and ShareVerse task adapters with pinned upstream revisions |
| `formal_evaluation` | State/depth/visibility, image/tracking metrics, paired statistics and calibration |
| `downstream_tracking` | Independent readers, identity heads and fixed-budget training-data regimes |
| `performance`, `generation_exports`, `experiment_runner` | Ordered prefetch, window export and local artifact-aware execution |

## Environment and assets

Run commands from the repository root. CPU contracts require no model download:

```bash
python -m pip install -r requirements-experiments.txt
python -m experiments.run_cpu_tests
```

Recorded full-video runtime: **Python 3.12, PyTorch 2.12.1 with CUDA 13.0**. GPU API versions are in `requirements-experiments-gpu.txt`; independent metric dependencies are in `requirements-experiments-metrics.txt`. Install a suitable CUDA PyTorch build before the GPU dependencies. CPU contracts also pass on Python 3.8/PyTorch 2.0.1; this does not establish full-video compatibility for that older environment.

Pretrained weights and dataset arrays are obtained separately. The source package includes no fitted checkpoints or private run products. Keep official source revisions and third-party licenses when obtaining DINOv2, Wan2.2, SynCamMaster, ShareVerse, RT-DETR and metric assets.

Copy `experiments/paths.example.json` to a local configuration and fill these paths:

| Key | Content |
| --- | --- |
| `upstream` | Accepted learned historical/dynamics checkpoint |
| `statistics` | Training-only state/acceleration normalization |
| `dino_code`, `dino_weights` | Local DINOv2 source and official ViT-B/14 weights |
| `wan` | Official `Wan-AI/Wan2.2-TI2V-5B-Diffusers` directory with `transformer/` and `vae/` |
| `empty_text` | Official empty-prompt tensor prepared by `paired_interventions.prepare_text` |
| `families` | Audited four-cell family index (`agcr-intervention-index-1`) |
| `ordinary` | Ordinary collection window index (`agcr-window-index-1`) |
| `pause_marker` | Optional local persistent pause-file path |

Relative paths resolve from the working directory. Asset preparation CLIs are `paired_interventions.prepare_text`, `downstream_tracking.prepare_assets` and `formal_evaluation.prepare_assets`.

## Data

Native samples use **6 historical + 16 future frames, 5 Hz, two 720 × 480 views**. Optical-axis depth is in **mm**, positions in **m**, and velocities in **m/s**. `inputs` contain histories, calibration and planned camera poses; future imagery and target states remain in a separate `supervision` dictionary.

A family crosses two world-motion branches with two passive future-camera paths. Informative pairs keep receiver history identical while changing partner-visible motion. Ambiguous pairs keep both histories identical. Camera pairs change future camera poses while retaining history and world motion. The audit verifies native arrays, target identities and visibility labels. Whole scene parents, worlds and camera variants stay in one split.

```bash
python -m experiments.paired_interventions.plan_families --help
python -m experiments.paired_interventions.render_family --help
python -m experiments.paired_interventions.collect --help
python -m experiments.paired_interventions.audit --help
python -m experiments.paper_experiments.agcr_exp.cli index --help
```

## Training and sampling

Observation/dynamics entrypoints are `paper_experiments.agcr_exp.train_observation` and `paper_experiments.agcr_exp.train_dynamics_geometry_ddp`. Standard adaptation uses `base48/protocol.json`, variants `receiver_only`, `paired`, `evidence`, and phases `state`, `depth`, `video`:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module experiments.base48.train \
  --paths experiments/paths.local.json \
  --protocol experiments/base48/protocol.json \
  --phase state --variant paired --seed 17 \
  --output artifacts/runs/paired_state_seed17 \
  --budget-ledger artifacts/budgets/paired_state_seed17.json --gpu-hours 2
```

The resource cap is an example, not a completion-time estimate. Stage prerequisites and data identities are checked. Later phases use accepted `--init` checkpoints; `--resume` restores the same run. Paired RGB-D sampling requires both completed renderer and state checkpoints:

```bash
python -m experiments.base48.sample generate --help
python -m experiments.base48.state_eval --help
python -m experiments.paired_interventions.train --help
python -m experiments.research_extensions.train_joint_feedback_four_device --help
```

The ordered gate preserves frame-specific visibility and time codes. Its mean-control averages both features and masks before the slot encoding. Hidden target state retains an ungated prediction path. Full feedback and stagewise generation are distinct settings.

## Matched studies and controls

`research_extensions.train_evidence_repeats` changes the evidence coefficient under matched histories, initialization, data and update budgets. `train_concat_gru_evidence` repeats the loss comparison on ConcatGRU without adding new base objectives. Informed CV shares the learned association/filtering posterior.

Matched repeat inputs are frozen once with `research_extensions.input_lock`; pass the same `--input-lock` to both coefficients. The local budget ledger supplies resource limits and reservations, rather than expired deployment dates. Checkpoint/cache/lock receipts preserve the scientific input boundary and completion evidence.

```bash
python -m experiments.research_extensions.input_lock --help
python -m experiments.research_extensions.train_evidence_repeats --help
python -m experiments.research_extensions.train_concat_gru_evidence --help
python -m experiments.research_extensions.evaluate_concat_gru_evidence --help
python -m experiments.research_extensions.evaluate_state_prediction_baselines --help
```

Source hashes bind runs to this public source tree. Legacy checkpoints require an explicit validated migration; merely replacing stored hashes is not a resume procedure.

## Baselines and evaluation

`external_baselines/official_sources.json` pins upstream revisions. Clone upstream implementations locally and obtain their official backbone assets. `base48.external` retains the registered SynCamMaster adaptation; `external_baselines.run` supports SynCamMaster and ShareVerse. RGB-only methods retain RGB-only outputs.

```bash
python -m experiments.external_baselines.run --help
python -m experiments.base48.external --help
python -m experiments.formal_evaluation.evaluate --help
python -m experiments.formal_evaluation.compare --help
python -m experiments.downstream_tracking.train --help
```

Evaluation retains missing predictions and reference denominators, averages eight paired samples, and resamples complete families rather than individual frames. Independent readers measure generated content; internal predicted states are not substituted for image measurements. Metric backends include LPIPS, I3D/FVD, COCO AP and TrackEval. Depth is scored on reference-valid pixels. Downstream regimes distinguish original renderings, generated clips and mixtures with matched budgets. An implemented experiment interface does not imply that every configured matrix entry has been completed or produces positive gains.
