# ManiFlow vs Diffusion Policy — comparison checklist (round 2)

Claim under test: RGB ManiFlow (DiT-X + consistency flow training, arXiv 2509.01819)
reaches higher real-robot task success than RGB Diffusion Policy when data, backbone,
augmentation, action representation, horizon, budget and evaluation are matched.

Sources:
- **Paper**: arXiv 2509.01819v1 (Sec. 2, 3; Appendix A.1–A.3, Alg. 1–2, Table 4).
- **MF code**: geyan21/ManiFlow_Policy@ef2f116 (`maniflow_image_policy.py`, `ditx*.py`,
  `transformer_obs_encoder.py`, `ema_model.py`, `maniflow_image_transformer_policy_robotwin.yaml`,
  `train_maniflow_robotwin_workspace.py`). Parity-tested files are vendored in
  `baseline/tests/reference/maniflow_ef2f116`.
- **DP code**: real-stanford/diffusion_policy@5ba07ac (`train_diffusion_unet_real_hybrid_workspace.yaml`,
  `ema_model.py`, byte-identical to ManiFlow's).
- **Resolved**: presets `baseline/configs/maniflow.json` (ManiFlow, "MF") and
  `baseline/configs/diffusion_ema.json` (Diffusion Policy, "DP"); training on 4 GPUs × 16 per GPU.

"Shared" means one setting applied identically to both arms.

Scope note: the paper's RGB ManiFlow vs RGB DP comparison is simulation only (Table 1/7:
Adroit, DexArt, RoboTwin 1.0). Its real-robot results (Fig. 9, Table 2), the 64-step horizon
(A.2) and the few-step result (Table 4) come from 3D (point-cloud) ManiFlow. This study
therefore extrapolates the 2D simulation comparison to real robots, borrowing the 3D
real-world horizon and the few-step setting as instructed.

Launch contract: every cloud run uses `--gpus 4` with `overrides.batch_size = 16`
(preset value 64 is per GPU), so both arms train with an effective batch of 64.

## 1. Tasks, data and actions

| Parameter | Paper | Official code | Resolved MF | Resolved DP | Reason for differences |
|---|---|---|---|---|---|
| Tasks | RGB comparison: sim only (Adroit, DexArt, RoboTwin 1.0); real robots: 8 tasks with 3D ManiFlow vs DP3 | sim benchmarks | pour_screws_into_funnel, place_loaded_boxes_in_container_bimanual, put_tools_in_standing_toolbox | same | Local robots; toolbox replaces spray handover (DP 29/30, no headroom). Pour is public |
| Demonstrations | sim 10/100/50; real ~50 (bimanual), 70–80 (single arm) | 50 (RoboTwin) | 284 / 299 / 296 excellent episodes in the exports of datasets 1456 / 1458 / 1459 | same exports | Existing reviewed data; shared |
| Train/holdout split | — | 2% val | all episodes train, none held out | same | Platform preset; shared |
| Observation | 2D RGB (+proprio) for the 2D comparison | 1 cam 224² + agent_pos | 3 cams (static + 2 wrist) 256², 20-D TCP state | same | Station sensors; shared |
| Obs history `n_obs_steps` | 2 | MF 2; original DP real config 2 | 1 | 1 | Serving emits on chunk requests only, one observation per request; shared |
| Action space | absolute joint/hand targets | absolute, min–max to [-1,1] | absolute TCP xyz + rot-6D rows + gripper width (20-D), MIN_MAX [-1,1], rotation rows fixed range | same | Platform action contract; shared |
| Image normalization | — | none in TransformerObsEncoder | ImageNet mean/std | same | Pretrained ImageNet backbone; shared |
| Augmentation | crop 0.95, rot ±5°, color jitter (b .3, c .4, s .5, h .08) | same, applied at train *and* eval (crop + resize back) | random crop 243/256 at train, center crop at eval, no resize back; rot ±5°; jitter (.3,.4,.5,.08) train only; platform JPEG-85 round trip | same | LeRobot crop semantics; eval-time jitter in the code is a quirk not applied; shared |
| Padding near episode end | — | MF: repeat last action, `pad_after = n_action_steps − 1` (≤ 31 padded steps at 64/32); original DP: drops last frames | repeat last action, loss unmasked, `drop_n_last_frames = 0` (anchors up to the last frame, ≤ 63 padded steps) | same | Platform preset; shared. Slightly more padded targets than the MF code at horizon 64 |

## 2. Observation encoder

| Parameter | Paper | Official code | Resolved MF | Resolved DP | Reason for differences |
|---|---|---|---|---|---|
| Backbone | ResNet-18 trained from scratch (2D) | transformer path: CLIP ViT-B/16 (config) or timm ResNet; GroupNorm when not pretrained | torchvision ResNet-18, ImageNet weights, BatchNorm, one per camera, fine-tuned | same backbone type, separate instance | Keeps the platform DP backbone (paper compares methods with the same ResNet-18); BatchNorm kept because weights are pretrained (code: GroupNorm only `if not pretrained`) |
| Feature head | visual tokens to DiT-X cross-attention | TransformerObsEncoder, ResNet, `feature_aggregation=None`: layer-4 map → h·w tokens, `Linear(512→n_emb)`; low-dim `Linear(D→n_emb)` per obs step | 8×8 layer-4 map (243 crop) → 64 tokens/camera, `Linear(512→768)`; state `Linear(20→768)` → **193 tokens × 768** (cameras sorted by key, then state) | SpatialSoftmax (32 keypoints) → 64-D per camera, concatenated with state → 212-D global condition | Each arm uses its method's encoder head on the shared backbone; MF follows the official multi-token path (round 1 used one concatenated token) |
| Token positions | — | DiT-X learnable `vis_cond_pos_embed`, zero-init, sliced to L | learnable, zero-init, exactly L = 193 positions | n/a | Same parameters actually used |
| Proprioception | 2-layer MLP; progressive masking with probability p (A.2); Fig. 4 caption: low-dim inputs via AdaLN-Zero | one `Linear(D→n_emb)` state token in cross-attention; no masking | `Linear(20→768)` state token; no masking | state concatenated into the global condition (FiLM) | MF follows the official code; masking is not in the official code |

## 3. Policy network

| Parameter | Paper | Official code | Resolved MF | Resolved DP | Reason for differences |
|---|---|---|---|---|---|
| Architecture | DiT-X (AdaLN-Zero on self-attn, cross-attn, MLP) | DiTX, 12 layers, 8 heads, 768, MLP ×4, attn dropout 0.1, qkv_bias + qk_norm (robotwin) | identical (parity-tested) | LeRobot U-Net, down_dims (512, 1024, 2048), kernel 5, FiLM scale, step embed 128 | DP: platform preset (original real-robot DP used (256, 512, 1024)) |
| Timestep conditioning | t and Δt | sinusoidal(128) → MLP, concat → linear | identical | diffusion step sinusoidal(128) | Method-specific |
| Prediction horizon | 64 (real world, 3D ManiFlow, A.2); 16 RoboTwin, 4 Adroit/DexArt | MF 16 (RoboTwin); original DP real config 16 | **64** steps | **64** steps | PI instruction: paper real-world value for both arms (also LeRobot DP's default); 64 steps at 30 Hz = 2.13 s (paper does not state its control rate: assumption) |

## 4. Training objective and optimisation

| Parameter | Paper | Official code | Resolved MF | Resolved DP | Reason for differences |
|---|---|---|---|---|---|
| Objective | flow matching + continuous-time consistency (Alg. 2) | 75% flow / 25% consistency; flow t ~ 0.999·Beta(1, 1.5), Δt = 0; consistency t ~ U{0,…,0.9} (K = 10), Δt ~ U[0,1], Δt′ = Δt, relative target time; target (x̃₁ − x_t)/(1 − t) | identical (parity-tested, same RNG order) | DDPM ε-prediction, 100 train steps, squaredcos_cap_v2, clip sample | Method-specific; DP matches original DP |
| Consistency target network | EMA θ⁻ | EMA DiT, in train mode (dropout) after epoch 0, online encoder features | EMA DiT, train mode during training, online encoder features (detached) | n/a | Matches code |
| Optimiser | — | AdamW lr 1e-4, betas (0.9, 0.95), wd 1e-3, all params | same | Adam lr 1e-4, betas (0.95, 0.999), wd 1e-6 | Each method's reference optimiser (DP: original DP / LeRobot) |
| LR schedule | — | cosine, 500 warmup | same | same | Shared |
| Effective batch | — | 64 (transformer config), 128 (timm config) | 64 (4 GPUs × 16) | 64 (4 GPUs × 16) | Platform preset; shared |
| Budget | sim: RoboTwin 2000 epochs, Adroit/DexArt 3000 (B.1) | 501 epochs (RoboTwin configs); original DP real 600 epochs | max(200k, ⌈25·frames/64⌉) steps: pour 200,000; boxes 200,000; toolbox 222,723 | same | Platform formula; shared |
| Seeds | 3 (sim) | — | 1 training seed per task | 1 | Budget; shared |
| Checkpoint used | final (RoboTwin), top-5 mean (Adroit/DexArt) | top-1 by validation loss | final checkpoint | final checkpoint | No held-out validation; shared |
| Gradient clipping | — | none | norm 10 | norm 10 | LeRobot trainer default; shared |
| Matmul precision | — | FP32 (PyTorch default) | FP32 highest | FP32 highest | Shared |
| EMA | "EMA model provides essential stabilization" | EMAModel: update_after_step 0, inv_gamma 1, power 0.75, min 0, max 0.9999; whole policy | same, whole network (encoder + DiT) | same (original DP), whole network (encoders + U-Net) | Shared; BatchNorm running stats copied from the online net (references use GroupNorm and never needed this) |
| Training compile | — | none | none | U-Net compiled (EMA copy eager) | Speed only |

## 5. Inference and execution

| Parameter | Paper | Official code | Resolved MF | Resolved DP | Reason for differences |
|---|---|---|---|---|---|
| Served weights | — | EMA policy | **EMA** | **EMA** | Both references evaluate EMA weights |
| Sampler steps | 1–2 headline (Table 4, 3D ManiFlow on RoboTwin: 64.5% at 2 vs 61.9% at 10) | 10 (config default) | **2** Euler steps, relative Δt = 0.5 | DDIM 16 steps | MF paper headline (PI instruction, checked offline on held-out episodes); DP keeps platform preset (original real DP: DDIM 8) |
| Actions emitted per plan `n_action_steps` | temporal ensembling over 64-step predictions | MF 16 (RoboTwin); original DP real config 8 | 32 | 32 | Shared; see execution row |
| Execution | temporal ensembling | — | station `blend`: request when ≤ 800 ms of actions remain, 600 ms cross-fade | same | No temporal ensembling on the station. With 32 emitted steps (1.07 s) a new plan is requested after ~8 steps + latency (~0.3 s), so plans overlap and are blended like ensembling while the queue never runs dry. Shorter chunks (≤ 24 steps) would replan at inference latency, making replan rate differ between arms (DP slower) |
| Control rate | not stated | sim step | 30 Hz | 30 Hz | Platform; horizon 64 interpreted as 64 steps at 30 Hz (assumption) |
| Serving compile / latency | — | — | eager | compiled EMA U-Net | Speed only |

### Offline 2- vs 10-step check (before training)

Local pilot of the `maniflow` preset on pour (12,000 steps, 256 training episodes, 28
evenly spaced episodes held out; checkpoint served with EMA weights), evaluated with
`python -m paw_baseline.maniflow.offline_steps` on 280 held-out anchors × 3 noise seeds,
identical initial noise across step counts. Pre-specified criterion: 2 steps is "clearly
worse" if executed-chunk normalized MSE rises > 10% with a 95% CI excluding zero.

| Steps | Exec. MSE (×10⁻³) | Left / right position error (mm) | Left / right rotation (°) | Jerk (mm/step², ground truth 0.11) |
|---|---|---|---|---|
| 1 | 5.39 | 26.6 / 17.6 | 4.11 / 3.28 | 6.43 |
| 2 | 5.51 | 25.7 / 16.0 | 3.85 / 2.99 | 3.52 |
| 4 | 5.79 | 26.1 / 15.8 | 3.88 / 2.93 | 2.32 |
| 10 | 5.99 | 26.9 / 16.0 | 3.97 / 2.96 | 1.60 |

2 vs 10 steps: −8.0% MSE (episode-bootstrap 95% CI −9.3% to −6.7%), 2 steps better in
27/28 held-out episodes → not clearly worse. Fewer steps give rougher chunks at this
early pilot stage (2 steps ≈ 2.2× the 10-step jerk).

## 6. Evaluation

| Parameter | Paper | Official code | Resolved MF | Resolved DP | Reason for differences |
|---|---|---|---|---|---|
| Trials | real: ~10 per object (2.8–5 for sorting and single-arm tasks), 305 rollouts per method (Table 9) | — | 30 placements per task (Eval_Nominal assignment), 1 repetition | same placements (`--same-setup-as`) | Platform protocol; shared |
| Metrics | success rate | — | SR (excellent/ok), AP from subtask annotations | same | Shared |
| Analysis | — | — | paired by placement, equal task weights, t-interval (Welch df) + sign-flip test | same | Pre-specified |
