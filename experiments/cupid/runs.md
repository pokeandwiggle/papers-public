# CUPID vs random filtering — run log

Claim tested: CUPID performance-influence filtering of existing demonstrations
improves real-robot task success over random filtering (filter-k, 50%).

| Task | Dataset (excellent+ok) | Base (all demos) | Random 50% | CUPID 50% |
|---|---|---|---|---|
| pour_screws_into_funnel (public) | dataset:1435 (308: 284 exc / 24 ok) | train:921 | train:924 | v2: pending approval |
| place_loaded_boxes_in_container_bimanual | dataset:1436 (352: 299 / 53) | train:922 | train:925 | v2: pending approval |
| clamp_electrical_component | dataset:1437 (458: 276 / 182) | train:923 | train:926 | v2: pending approval |

Shared training: preset `diffusion`, `--gpus 2` (64/GPU, effective batch 128),
`steps=100000` (= 200k x 64 samples of the preset/paper budget), JPEG85 default.
Random selection: seed 0, `random_keep` (selection_random_*.json).
Base/random submitted from f6670b6 / 2921309 (identical training code).

## Models and evals (valid)
Serving revision for all station evals: 6a8bf66a38c406c7e6f84a4ec395db2b565148dd.

| Task | Base model / eval (CUPID rollouts) | Random model / eval |
|---|---|---|
| pour | model:678 / eval:2490 (9/30) | model:697 / eval:2508 (16/30) |
| boxes | model:677 / eval:2489 (22/30) | model:690 / eval:2503 (18/30) |
| clamp | model:692 / eval:2504 (1/30) | model:693 / eval:2505 (0/30) |

## v1 CUPID arm — INVALID (scoring bug)
score.py passed the accumulated X^T X to `gauss_newton_inverse`, which formed
X^T X again, so v1 scores used (X^T X)^-2 instead of TRAK's (X^T X)^-1
(verified: saved v1 trajectory matrices match rollout_sums @ inv(xtx @ xtx) @ sums.T
to <=5e-5 relative error). v1 selections (selection_cupid_*_v1_invalid.json),
models (pour model:734 via train:1058 failed/train:1061, boxes model:733 via
train:1060, clamp model:732 via train:1059), evals (2617, 2581, 2578) and
results_v1_invalid.json do not test CUPID and are excluded from the comparison.

## v2 CUPID scoring (fix: scoring_version 2)
- Gram matrix accumulated once in `run_train`; `gauss_newton_inverse` now takes
  X^T X, rejects non-square input; `cupid_scores` is the single entry point and is
  regression-tested against brute-force pairwise influence from raw features.
- Rescored from the saved per-demo and per-rollout feature sums (`score.py rescore`);
  these sums do not depend on the inverse, so no re-featurization was needed.
- v2 vs v1: Spearman 0.62 / 0.67 / 0.71 (pour / boxes / clamp); kept-set overlap
  115/154, 131/176, 175/229.
- v2 diagnostics (split-half Spearman over outcome-stratified rollout halves;
  kept excellent/ok; AUC excellent>ok): pour 0.68, 141/13, 0.49; boxes 0.46,
  153/23, 0.53; clamp 0.75, 142/87, 0.52.

## v2 reviews (fresh-context; correctness + fidelity)
- No method-breaking defect. Both reviewers independently reproduced v2 traj/net
  scores from the saved sums with inv(X^T X) (rel. err <= 2e-11), confirmed the
  kept sets are exactly the top 50%, the export->source-ID mapping equals the base
  checkpoints' training manifests, rollout labels match the eval manifests, and
  rollout sums are identical between v1 and v2 (rescoring valid).
- Ridge sensitivity: Spearman >= 0.998 for ridge <= 1e-4 * mean diag.
- Fixed: stale v1 params in tmp/ (would have retrained v1). v2 params are now built
  by `build_params.py` from the committed selection (asserts scoring_version 2).
- Open protocol issue (user decision): v2 CUPID evals would run days after the
  random-arm evals (Oct 4-5); re-evaluating the random models in the same session
  removes station/operator drift as a confound (3 extra evals).

## Deviations from paper / official code (highest impact first)
1. Eval timing: CUPID-v2 evals not concurrent with random evals (see above).
2. Scoring rollouts and final evals use the same 30 eval placements (favours CUPID;
   the reference also overlaps seeds). Claim limited to these placements.
3. One training seed and one random subset per arm, n=30 per task (paper's real-world
   protocol); observed subset/seed variance is large (pour: random 16/30 vs base 9/30).
4. Data: naturally collected excellent+ok pools (92/85/60% excellent), not the paper's
   constructed mixed-quality/strategy/spurious sets; base trained on all demos (paper:
   random ~9/10 subset); failed demos excluded (paper demos are all successes).
   Less headroom; biases toward a null result.
5. Clamp base at floor (1/30): scores are almost entirely failure-driven; SR cannot
   separate arms there (AP can).
6. Training pairs on a frame stride (5/15/20; N/d 6-7) instead of every sample, for the
   local GPU budget. Adds noise, no ranking bias.
7. CountSketch (random signs, bucket i mod d) instead of TRAK's dense JL projection
   (287M params; no fast_jl). Unbiased; noise only.
8. Policy/training: LeRobot DP preset (3 cameras, ImageNet ResNet-18 with BatchNorm,
   no crop aug, n_obs 1, horizon/action 32/32, DDIM 16, no EMA, JPEG85), 2 GPUs x 64 x
   100k steps (= 200k x 64 samples) vs Chi et al. DP 200k steps. Same in all arms.
9. Execution with the platform `blend` runtime (overlapping chunks, crossfade) rather
   than receding-horizon execution; Prop. 1 holds approximately. Same everywhere.
10. Test pairs reconstructed from recordings: AV1 re-decode + cover crop + JPEG85; the
    emitted (post-hygiene) chunk re-encoded; cold-start/unloaded chunks dropped;
    post-success frames may be included (reference stops at success).
11. Numerics: TF32 gradients; float64 Gram/inverse (TRAK float32 + rescale); batched
    draws; mean-form l_square. Negligible.
12. Constants: 50% filter only (paper 66%/50%), m=30 rollouts (paper 25), filter-k
    only, equal-weight paired analysis.
13. Disclosure: the invalid v1 CUPID arm and its result predate this fix.

## v2 CUPID arm — launched after user approval (2026-10-07)
User decisions: relaunch training (yes); re-evaluate random models concurrently
(no — evaluation treated as consistent); extra random seed / full-frame rescoring
(no — tested separately). After retraining, verify v2 vs v1 (`verify_v2_runs.py`).

| Task | v2 training | v2 model / eval |
|---|---|---|
| pour | train:1108 (4b03c95) | model:783 / eval:2754 (13/30) |
| boxes | train:1109 (4b03c95) | model:784 / eval:2756 (7/30) |
| clamp | train:1110 (4b03c95) | model:785 / eval:2757 (1/30) |

## v2 result (results.json; CUPID-v2 - random, paired by placement)
| Task | SR diff (pp) | AP diff (pp) | discordant SR (cupid/random) | McNemar exact p |
|---|---|---|---|---|
| pour | -10.0 | -1.1 | 4 / 7 | 0.55 |
| boxes | -36.7 | -13.3 | 3 / 14 | 0.013 |
| clamp | +3.3 | +10.0 | 1 / 0 | 1.0 |
| Overall (equal weights) | -14.4 [-25.7, -3.2], sign-flip p=0.024 | -1.5 [-9.5, +6.5], p=0.74 | | |
Bootstrap CIs: SR [-25.6, -3.3], AP [-9.2, +6.4].
compare_selections_v2.txt: kept sets overlap random at chance and match it on rating
mix, placement coverage and duration; boxes CUPID failures concentrate at
place_box_in_container (22/30 vs 8/30 random).
score_components.py: Spearman(net, outcome-independent total influence) = -0.95 / +0.77
/ -0.97 (pour / boxes / clamp); boxes kept set = top-total for 139/176 demos.

v2-vs-v1 verification (`verify_v2_runs.py`, report in verify_v2_runs.txt): all checks
pass for all three tasks — submitted IDs and the checkpoints' training manifests equal
the v2 selections (not v1); normalization refit on the v2 subset; weights differ;
policy config, JPEG recipe, preset, steps, dataset and 2-GPU shape identical to v1;
v2 runs not resumed (v1 boxes had resumed from step 50k after a Spot preemption,
which is the only config.json difference).

Submission check (`verify_v2_runs.py`): params IDs == v2 selection != v1 selection;
preset/overrides/dataset/world size identical to v1 and random runs.

## Plan for the v2 CUPID arm (as approved)
1. Train 3 CUPID-v2 policies with params from `build_params.py cupid <task>`
   (`tmp/params_cupid_v2_<task>.json`), identical to the random arm except IDs:
   preset `diffusion`, `--gpus 2`, `steps=100000`, same dataset handles, training
   code unchanged since f6670b6. Spot first; STANDARD only after sustained
   preemption (as for v1 pour).
2. `serve prepare` at 6a8bf66 (same serving revision as all other arms).
3. Station evals with `--same-setup-as` the base eval of each task
   (eval:2490 / 2489 / 2504): same 30 placements, 1 repeat, `blend` runtime.
4. Re-run `analyze.py` with the v2 CUPID evals vs the existing random evals;
   report v1 numbers only as an invalid run.
Budget: 3 training credits (6 of 15 remain available before this) and 3 evals.

## CUPID scoring settings (unchanged, fixed before seeing any result)
- Rollouts: one station eval of each base policy (default `blend` runtime, task's
  30 eval placements, 1 repeat); counted episodes with R=+1 (excellent/ok) / -1.
- Features: l_square, B=64, all parameters, d=4000 sign sketch, lambda=0, TF32.
- Training-pair stride (frames): pour 5, boxes 15, clamp 20 (~25-28k pairs each,
  N/d ~6-7). Every chunk request of every counted rollout is a test pair.
- Selection: keep top 50% of net (sum-of-sums) performance influence.
- Analysis: `analyze.py` (paired by placement, equal task weights,
  Welch-Satterthwaite CI, stratified sign-flip p); `diagnose.py` for score
  reliability.
