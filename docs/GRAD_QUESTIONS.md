# Graduate-Level Questions — IQL Robustness Analysis

**Course:** CMPE 260 — Reinforcement Learning | San José State University
**Paper:** *Robustness of Implicit Q-Learning Under Controlled Distribution Shift*
**Format:** In-class presentation Q&A — 35 questions across 7 categories

---

## Category 1: IQL Foundations (6 questions)

### Q1. Expectile Regression for Value Learning

**Why does IQL use expectile regression instead of standard mean-squared Bellman error for learning the value function V(s)?**

<details>
<summary>Suggested Answer</summary>

Standard Bellman backups require evaluating `max_a Q(s,a)`, which queries out-of-distribution (OOD) actions not present in the offline dataset — a major source of overestimation. IQL sidesteps this by learning V(s) via expectile regression over dataset actions: `V(s) = argmin_v E[L_τ(Q(s,a) - v)]`, where the asymmetric loss `L_τ(u) = |τ - 𝟙(u<0)| · u²` biases V toward higher-return actions without ever maximizing over unseen actions. This keeps all value learning within the support of the data.

</details>

---

### Q2. Role of the Expectile τ

**Explain the role of the expectile hyperparameter τ in IQL. What happens at the extremes τ → 0.5 and τ → 1.0, and why is the default set to 0.7?**

<details>
<summary>Suggested Answer</summary>

At τ = 0.5, the expectile loss is symmetric and V(s) converges to the conditional mean of Q-values — a pessimistic estimate. As τ → 1.0, V(s) approaches the maximum Q-value in the dataset, recovering something closer to the standard Bellman optimality backup. The default τ = 0.7 balances optimism (extracting good actions) with conservatism (not overestimating). This paper's ablation confirms: higher τ yields higher baseline returns but worse robustness and higher cross-seed variance.

</details>

---

### Q3. Advantage-Weighted Policy Extraction

**How does IQL extract a policy from the learned Q and V functions, and why is this approach preferable to direct policy gradient methods in the offline setting?**

<details>
<summary>Suggested Answer</summary>

IQL uses advantage-weighted behavioral cloning: `max_π E_{(s,a)~D}[exp(β(Q(s,a) - V(s))) · log π(a|s)]`. Actions with positive advantage (Q > V) receive exponentially higher weight, while low-advantage actions are downweighted. This avoids querying the policy on OOD actions during training — unlike actor-critic methods that would evaluate `Q(s, π(s))` for potentially unseen actions. The temperature β controls how sharply the policy concentrates on high-advantage actions.

</details>

---

### Q4. Temperature β vs Expectile τ

**Both the temperature β and the expectile τ control the degree of optimism in IQL. How do their roles differ, and could adjusting β achieve the same robustness effects as lowering τ?**

<details>
<summary>Suggested Answer</summary>

τ controls optimism in the *value function* — how much V(s) focuses on the upper tail of Q-values. β controls optimism in *policy extraction* — how sharply the policy concentrates on high-advantage actions. Lowering τ makes V more pessimistic, which propagates through the TD backup to affect all Q-values. Lowering β makes the policy more uniform over dataset actions but doesn't change the value estimates themselves. They operate at different stages of the pipeline, so adjusting β alone cannot replicate the robustness benefits of lower τ observed in this paper.

</details>

---

### Q5. IQL vs CQL Under Distribution Shift

**Compare IQL and CQL in terms of their mechanisms for handling distribution shift. Based on the theoretical differences, which would you hypothesize to be more robust under environment-level perturbations, and why?**

<details>
<summary>Suggested Answer</summary>

CQL explicitly penalizes Q-values for OOD actions via a regularization term, producing a pessimistic lower bound on the true Q-function. IQL avoids OOD actions implicitly by never querying them during value learning. Under environment-level perturbations (gravity, friction), the state-action distribution at test time diverges from training — CQL's explicit pessimism might provide stronger guarantees since it directly penalizes overestimation, while IQL's implicit approach only ensures in-distribution accuracy. However, CQL's conservatism can also hurt baseline performance more severely.

</details>

---

### Q6. IQL vs TD3+BC

**TD3+BC adds a behavior cloning term `λQ(s,π(s)) - ||π(s) - a||²` to constrain the policy. How does this differ from IQL's approach, and what are the implications for robustness under dynamics shift?**

<details>
<summary>Suggested Answer</summary>

TD3+BC constrains the *policy* to stay close to dataset actions via an explicit BC penalty, while IQL constrains the *value function* to stay within dataset support via expectile regression. Under dynamics shift, TD3+BC's policy constraint means the agent will continue executing dataset-like actions even when the environment changes — which could be beneficial (staying in known territory) or harmful (the "right" action under shifted dynamics may differ from the dataset). IQL's value-level constraint is more indirect but allows the policy to adapt within the advantage-weighted framework.

</details>

---

## Category 2: Q-Ensemble Extension (5 questions)

### Q7. Theoretical Motivation for TripleCritic

**The paper extends IQL's DoubleCritic (min(Q₁, Q₂)) to a TripleCritic (min(Q₁, Q₂, Q₃)). What is the theoretical motivation for taking the minimum over more Q-networks?**

<details>
<summary>Suggested Answer</summary>

Taking the minimum over independently initialized Q-networks produces a lower (more conservative) estimate of the true Q-value. With 2 networks, the minimum is an order statistic that underestimates the mean; with 3 networks, the minimum of 3 i.i.d. estimates is stochastically smaller than the minimum of 2. This additional conservatism reduces overestimation of OOD state-action pairs, which is particularly valuable under distribution shift where the agent encounters states not well-represented in the training data.

</details>

---

### Q8. Why min(Q₁,Q₂,Q₃) Is More Conservative

**Formally, if Q₁, Q₂, Q₃ are i.i.d. random variables with CDF F, what is the CDF of min(Q₁,Q₂,Q₃) and how does it compare to min(Q₁,Q₂)?**

<details>
<summary>Suggested Answer</summary>

The CDF of the minimum of n i.i.d. variables is `F_min(x) = 1 - (1-F(x))^n`. For n=2: `F_min(x) = 1 - (1-F(x))²`; for n=3: `F_min(x) = 1 - (1-F(x))³`. Since `(1-F(x))³ ≤ (1-F(x))²` for all x, the 3-network minimum has a higher CDF everywhere — meaning it assigns more probability to lower values. In expectation, `E[min(Q₁,Q₂,Q₃)] < E[min(Q₁,Q₂)]`, producing a strictly more conservative value estimate.

</details>

---

### Q9. When Does 3Q Help vs Hurt?

**The results show 3Q improves robustness on Hopper but not on Walker2d or HalfCheetah. What environment-specific properties might explain this asymmetry?**

<details>
<summary>Suggested Answer</summary>

Hopper is a single-legged robot with 3 actuators and 11 observations — it's inherently unstable and sensitive to overestimation (a single bad action causes falling). The extra conservatism of 3Q prevents the policy from taking risky actions that might work in training but fail under shift. Walker2d (2 legs, 6 actuators) has more redundancy and stability, so excessive conservatism may prevent it from taking necessary corrective actions. HalfCheetah is inherently robust (low AUDC everywhere), so the additional conservatism provides no benefit. The optimal level of pessimism is environment-dependent.

</details>

---

### Q10. Computational Cost of TripleCritic

**What is the computational overhead of using 3 Q-networks instead of 2, and is it justified by the robustness gains observed?**

<details>
<summary>Suggested Answer</summary>

The TripleCritic adds one additional MLP (256×256×1) to the critic, increasing critic parameters by ~50% and critic forward/backward pass time proportionally. However, the critic is only one of three components (actor, critic, value), so total training time increases by roughly 15-20%. Given that 3Q only consistently helps on Hopper (1 of 3 environments), the cost-benefit ratio is questionable for general use. A more adaptive approach — selecting critic count based on environment properties — would be more efficient.

</details>

---

### Q11. Beyond 3 Critics — Diminishing Returns

**Would extending to 4 or 5 Q-networks (QuadCritic, QuintCritic) further improve robustness? What theoretical and practical considerations limit the ensemble size?**

<details>
<summary>Suggested Answer</summary>

Theoretically, E[min(Q₁,...,Qₙ)] decreases monotonically with n, so more critics always produce more conservative estimates. However, diminishing returns set in: the gap between min-of-3 and min-of-4 is smaller than between min-of-2 and min-of-3. Practically, excessive conservatism can cause the policy to become overly cautious, refusing to take any action confidently — this is the "pessimism collapse" problem. The Walker2d results already hint at this: 3Q hurts gravity robustness compared to 2Q, suggesting the environment is already at the conservatism frontier.

</details>

---

## Category 3: Distribution Shift & Robustness (6 questions)

### Q12. Types of Distribution Shift

**The paper tests four types of distribution shift. Classify each as either a dynamics shift, observation shift, or reward shift, and explain why this taxonomy matters for understanding robustness.**

<details>
<summary>Suggested Answer</summary>

Gravity and friction are *dynamics shifts* — they change the transition function P(s'|s,a). Observation noise is an *observation shift* — it changes what the agent perceives but not the underlying dynamics. Reward perturbation is a *reward shift* — it changes R(s,a) but not transitions or observations. This taxonomy matters because each type stresses different components: dynamics shifts test whether the learned Q-values generalize to new physics, observation shifts test policy robustness to perceptual noise, and reward shifts test whether the agent adapts at test time (it shouldn't, in offline RL).

</details>

---

### Q13. AUDC Metric — Interpretation

**The paper uses AUDC (Area Under Degradation Curve) as the primary robustness metric. Why is AUDC preferred over simply reporting worst-case performance or performance at a single shift level?**

<details>
<summary>Suggested Answer</summary>

AUDC integrates the normalized performance drop `|Δ(δ)|` over all shift levels, capturing the *entire degradation profile* rather than a single point. Worst-case performance only reflects the most extreme shift level and ignores graceful vs. catastrophic degradation patterns. A single shift level might not be representative — an agent could be robust at 1.5× gravity but collapse at 2.0×. AUDC captures both the onset and severity of degradation, providing a single scalar that summarizes robustness across the full perturbation range (lower AUDC = more robust).

</details>

---

### Q14. Near-Zero Reward Perturbation AUDC

**The reward perturbation AUDC is < 0.002 across all configurations. Why is this expected, and what would it mean if this value were significantly higher?**

<details>
<summary>Suggested Answer</summary>

IQL is an offline algorithm — the policy is frozen after training and never updates from rewards during evaluation. Adding noise to the reward signal at test time therefore cannot affect the agent's behavior, since the policy `π(a|s)` depends only on observations, not on rewards received. If reward perturbation showed significant degradation, it would indicate a bug: either the agent is performing online updates at test time, or the reward noise is somehow leaking into the observation space. This serves as a critical sanity check for the experimental protocol.

</details>

---

### Q15. Gravity vs Friction Sensitivity

**Hopper shows gravity AUDC of 0.616 and friction AUDC of 0.713 (2Q, τ=0.7), while HalfCheetah shows 0.255 and 0.016 respectively. Why is HalfCheetah so much more robust to friction changes than Hopper?**

<details>
<summary>Suggested Answer</summary>

HalfCheetah is a running robot with a low center of mass and continuous ground contact across multiple body segments — friction changes affect all contact points somewhat uniformly, and the running gait is inherently stable. Hopper is a single-legged robot that relies on precise ground contact at one foot — friction changes dramatically affect the push-off phase and landing stability. Additionally, HalfCheetah's 6 actuators provide more redundancy to compensate for friction changes, while Hopper's 3 actuators have less margin for adaptation.

</details>

---

### Q16. Sim-to-Real Implications

**How do the distribution shifts tested in this paper relate to real-world sim-to-real transfer challenges? Which of the four shift types is most representative of actual deployment conditions?**

<details>
<summary>Suggested Answer</summary>

Gravity shift simulates deploying a robot in a different gravitational environment (e.g., different payload weight). Friction shift simulates surface changes (indoor tile vs. outdoor terrain). Observation noise simulates sensor degradation, calibration drift, or environmental interference — this is arguably the most representative of real deployment, since sensors always have noise. Reward perturbation is least relevant since real robots don't receive explicit reward signals. In practice, real-world shifts are often *compositional* (multiple shifts simultaneously), which this paper does not test but identifies as future work.

</details>

---

### Q17. Asymmetric Gravity Degradation

**Under gravity shift, both 0.5× (reduced) and 2.0× (increased) gravity cause ~70% performance drop on Hopper. Why is the degradation roughly symmetric around the training gravity of 1.0×?**

<details>
<summary>Suggested Answer</summary>

The policy learned a control strategy calibrated for Earth gravity — the timing of joint torques, the magnitude of push-offs, and the balance corrections are all tuned to 9.81 m/s². Under 0.5× gravity, the robot over-compensates (jumps too high, loses balance on landing). Under 2.0× gravity, it under-compensates (can't generate enough force to hop). Both directions represent equal deviations from the training distribution in terms of the mismatch between learned torque profiles and required torque profiles, explaining the roughly symmetric degradation.

</details>

---

## Category 4: Expectile Ablation (4 questions)

### Q18. τ and the Robustness-Performance Trade-off

**The ablation shows that lower τ improves robustness (lower AUDC) but reduces baseline performance. Is this trade-off fundamental to IQL, or could it be mitigated?**

<details>
<summary>Suggested Answer</summary>

The trade-off is fundamental to expectile regression: lower τ produces more pessimistic value estimates, which leads to a more conservative policy that avoids high-risk/high-reward actions. Under nominal conditions, this conservatism costs performance; under shift, it provides a safety margin. Potential mitigations include: (1) adaptive τ that starts high during training and decreases for deployment, (2) separate τ values for different state regions based on uncertainty, or (3) combining IQL with explicit robustness objectives. However, some form of this trade-off is inherent to any pessimism-based approach.

</details>

---

### Q19. Environment-Dependent τ Sensitivity

**On HalfCheetah, gravity AUDC varies only from 0.255 to 0.277 across all τ values, while on Hopper it ranges from 0.529 to 0.730. Why is HalfCheetah insensitive to τ?**

<details>
<summary>Suggested Answer</summary>

HalfCheetah's running gait is inherently stable — the robot maintains ground contact and forward momentum regardless of the value function's optimism level. The policy doesn't need to make high-stakes decisions (like balancing on one leg), so the difference between a conservative and optimistic policy is small. Hopper, by contrast, requires precise timing and force calibration for each hop — an optimistic policy (high τ) may attempt aggressive hops that fail catastrophically under shifted gravity, while a conservative policy (low τ) takes safer, smaller hops that degrade more gracefully.

</details>

---

### Q20. High τ and Cross-Seed Variance

**The paper reports that τ = 0.9 consistently produces higher cross-seed variance (e.g., Hopper 2Q gravity AUDC std = 0.097 vs 0.027 at τ = 0.7). What mechanism causes this instability?**

<details>
<summary>Suggested Answer</summary>

At τ = 0.9, the value function targets the 90th percentile of Q-values, making it highly sensitive to the upper tail of the Q-distribution. Small differences in random initialization (different seeds) lead to different Q-value distributions, and the 90th percentile amplifies these differences. Additionally, high τ creates a positive feedback loop: optimistic V estimates → aggressive policy → high-variance returns → unstable Q-targets. At τ = 0.7, the value function targets a more central tendency, which is less sensitive to initialization and produces more stable training dynamics.

</details>

---

### Q21. Optimal τ Selection Strategy

**Given that the optimal τ is environment-dependent, propose a principled method for selecting τ without exhaustive ablation. What information would you need?**

<details>
<summary>Suggested Answer</summary>

One approach is to use the offline dataset itself to estimate environment stability: compute the variance of returns in the dataset, the diversity of states visited, and the sensitivity of rewards to small action perturbations. Environments with high variance and sensitivity (like Hopper) would benefit from lower τ, while stable environments (like HalfCheetah) can tolerate higher τ. Alternatively, one could use a validation set with mild perturbations to select τ via cross-validation, or employ meta-learning across environments to learn a τ-selection policy. The key information needed is some measure of the environment's sensitivity to policy changes.

</details>

---

## Category 5: Experimental Design & Methodology (5 questions)

### Q22. Multi-Seed Evaluation

**The experiment uses 4 seeds (42, 43, 44, 45). Is this sufficient for statistical significance? What are the trade-offs of using more seeds?**

<details>
<summary>Suggested Answer</summary>

Four seeds is modest by statistical standards — with n=4, confidence intervals are wide and outlier seeds can significantly affect the mean (e.g., Hopper 2Q seed 44 baseline is 1418 vs. mean 1571). However, each seed involves training for 300k steps and evaluating across 64 conditions (4 shifts × 4 levels × 4 τ values), so the computational cost scales linearly. The paper partially compensates by reporting mean ± std and identifying cases where differences exceed one standard deviation. Ideally, 10+ seeds would provide tighter confidence intervals, but the 1,536 total evaluations already represent substantial compute.

</details>

---

### Q23. D4RL Medium Dataset Choice

**Why did the authors choose "medium" quality datasets rather than "expert" or "random"? How might results differ on these other dataset qualities?**

<details>
<summary>Suggested Answer</summary>

Medium datasets contain data from a partially-trained policy — realistic but imperfect. Expert datasets would make the baseline too strong (less room for degradation analysis), while random datasets would make the baseline too weak (hard to distinguish shift-induced degradation from poor policy quality). On expert data, the policy would likely be more robust (better coverage of good actions) but also more brittle (optimized for a narrow operating regime). On random data, the policy would be inherently conservative (low baseline) but potentially more robust (trained on diverse, low-quality trajectories that cover more of the state space).

</details>

---

### Q24. Evaluation Protocol — 10 Episodes per Condition

**Each shift-level evaluation averages over 10 episodes. How does episode count affect the reliability of AUDC estimates, and what is the trade-off?**

<details>
<summary>Suggested Answer</summary>

With 10 episodes, the standard error of the mean return is `σ_episode / √10`. For high-variance environments like Hopper (where individual episode returns can range from 300 to 2000), this may not be sufficient to distinguish small AUDC differences. However, increasing to 100 episodes would multiply evaluation time by 10× across all 1,536 conditions. The paper mitigates this by aggregating across 4 seeds (effectively 40 episodes per condition) and focusing on AUDC differences that exceed one standard deviation. For the key finding (Hopper 3Q gravity AUDC 0.529 vs 2Q 0.616), the gap is ~3× the std, providing reasonable confidence.

</details>

---

### Q25. Shift Levels — Why These Specific Values?

**The gravity shift uses levels {0.5×, 1.0×, 1.5×, 2.0×}. How would the conclusions change if finer-grained levels (e.g., 0.5×, 0.75×, 1.0×, 1.25×, 1.5×, 1.75×, 2.0×) were used?**

<details>
<summary>Suggested Answer</summary>

Finer-grained levels would produce a smoother degradation curve and a more accurate AUDC estimate. With only 4 levels, the trapezoidal integration used for AUDC may miss non-linear degradation patterns (e.g., a sharp cliff between 1.25× and 1.5× gravity). However, the current 4-level design captures the key qualitative pattern (symmetric degradation around 1.0×) and keeps the evaluation matrix manageable. Finer levels would be most valuable for identifying critical thresholds — the exact shift magnitude where performance collapses — which could inform deployment safety margins.

</details>

---

### Q26. Baseline Measurement Design

**The paper embeds baseline measurement within the shift evaluation (gravity=1.0× is one of the 4 levels). What is the advantage of this design over running a separate baseline evaluation?**

<details>
<summary>Suggested Answer</summary>

Embedding the baseline ensures that baseline and shifted evaluations use the exact same trained model checkpoint, random seed state, and evaluation protocol — eliminating any confounds from separate runs. It also avoids redundant computation: instead of running baseline + shift separately, the baseline is simply one row in the shift CSV. This design guarantees that the robustness drop `Δ(δ) = (J(E₀) - J(E_δ)) / J(E₀)` uses a perfectly matched denominator, reducing noise in the AUDC calculation.

</details>

---

## Category 6: Broader Context & Future Work (4 questions)

### Q27. Compositional Distribution Shifts

**This paper tests each shift type independently. In real-world deployment, multiple shifts often occur simultaneously (e.g., changed gravity AND noisy sensors). How would you extend this evaluation framework to handle compositional shifts?**

<details>
<summary>Suggested Answer</summary>

The wrapper-based design already supports composition — one could stack `GravityShift(ObservationNoise(env))` to apply multiple shifts simultaneously. The challenge is combinatorial explosion: 4 shift types × 4 levels each = 256 combinations per model. A practical approach would be to test a representative subset: (1) pairwise combinations of the two most damaging shifts (gravity + friction), (2) a "worst-case" combination of all shifts at moderate levels, and (3) random sampling from the shift space. The AUDC metric would need to be extended to a multi-dimensional integral over the joint shift space.

</details>

---

### Q28. Real-World Deployment — When to Use 2Q vs 3Q

**A robotics engineer asks: "Should I use 2Q or 3Q for my deployment?" Based on this paper's findings, what practical guidance would you give?**

<details>
<summary>Suggested Answer</summary>

The answer depends on the robot's morphology and expected deployment conditions. For inherently unstable systems (like single-legged hoppers or bipeds with narrow stability margins), 3Q provides meaningful robustness gains — use it. For stable, multi-actuated systems (like wheeled robots or quadrupeds), 2Q is sufficient and avoids the computational overhead. If the deployment environment is well-characterized (small expected shifts), use 2Q with default τ. If the environment is uncertain (large potential shifts), use 3Q with lower τ (0.5–0.7). The key insight is that there is no universal answer — the optimal configuration must be matched to the deployment scenario.

</details>

---

### Q29. Meta-Learning for Hyperparameter Selection

**The paper concludes that the optimal (critic count, τ) pair is environment-dependent. Could meta-learning be used to automatically select these hyperparameters? What would the meta-learning setup look like?**

<details>
<summary>Suggested Answer</summary>

A meta-learning approach would train across a distribution of environments (varying morphologies, dynamics parameters) to learn a mapping from environment features to optimal (critic count, τ). The meta-features could include: observation/action dimensionality, dataset return statistics, state-space coverage metrics, and dynamics sensitivity estimates. The meta-objective would minimize AUDC across a held-out set of shift conditions. This is related to AutoRL and hyperparameter transfer, but the key challenge is that the "environment distribution" for meta-training must be representative of deployment scenarios — a chicken-and-egg problem if the deployment environment is truly unknown.

</details>

---

### Q30. Online Fine-Tuning After Distribution Shift Detection

**If we could detect that distribution shift has occurred during deployment, could we fine-tune the offline-trained policy online? What challenges would arise, and how does IQL's architecture facilitate or hinder this?**

<details>
<summary>Suggested Answer</summary>

Online fine-tuning after shift detection is a natural extension. IQL's architecture is relatively amenable to this because the expectile regression framework can incorporate new data by simply adding shifted-environment transitions to the replay buffer. However, challenges include: (1) catastrophic forgetting of the original policy, (2) the need for safe exploration in the shifted environment (the policy may take dangerous actions while adapting), and (3) sample efficiency — fine-tuning may require many interactions in the new environment. A practical approach would be to use the offline-trained policy as a warm start and apply conservative fine-tuning (e.g., with a KL constraint to the original policy), similar to the offline-to-online RL paradigm.

</details>

---

## Category 7: Presentation & Slide-Specific Questions (5 questions)

*These questions are based on the content presented in the [slide deck](../slides/Slides_Robustness-of-Implicit-Q-Learning-Under-Controlled-Distribution-Shift.pptx).*

### Q31. Three Competing Strategies (Slide 2)

**Slide 2 lists CQL, IQL, and TD3+BC as "three competing strategies" for handling distribution shift. The slide states CQL uses pessimism, IQL stays within dataset support, and TD3+BC uses behavior cloning. If you had to deploy an offline RL agent in a safety-critical medical dosing scenario where the patient population at deployment differs from the training data, which strategy would you choose and why?**

<details>
<summary>Suggested Answer</summary>

CQL would be the strongest candidate for safety-critical deployment because it provides explicit pessimistic lower bounds on Q-values — meaning it systematically underestimates the value of untested actions. In medical dosing, overestimating the benefit of an unseen dosage could be fatal, so CQL's conservative bias is a feature, not a bug. IQL's implicit approach (staying within dataset support) is less transparent about its safety guarantees — it avoids OOD actions during training but doesn't explicitly penalize them. TD3+BC would keep dosages close to historical prescriptions, which is safe but may fail to improve on suboptimal historical practices.

</details>

---

### Q32. Wrapper-Only Shift Design (Slide 3)

**Slide 3 emphasizes that wrappers are applied "only at test time" and the policy is trained once on standard D4RL data. Why is it critical that the policy is NOT retrained under the shifted conditions? What would change about the experimental conclusions if the policy were fine-tuned under each shift?**

<details>
<summary>Suggested Answer</summary>

Not retraining isolates the question "how robust is the *original* policy?" from "how well can the algorithm *adapt* to new conditions?" If we fine-tuned under each shift, we would be measuring IQL's transfer learning or online adaptation capability rather than its zero-shot robustness. The AUDC metric would become meaningless because a fine-tuned policy would trivially achieve low degradation. The experimental design specifically tests the realistic deployment scenario where you cannot retrain — e.g., a robot shipped to a customer site with different floor surfaces. This is the gap identified in the literature: prior work assumes identical training and testing environments.

</details>

---

### Q33. Hopper AUDC Table Interpretation (Slide 4)

**Slide 4 shows that 3Q wins on ALL four shift types for Hopper, yet the paper's conclusion states the benefit is "environment-dependent." Looking at the Hopper AUDC table, the gravity AUDC for 3Q (0.529 ± 0.069) has a larger standard deviation than 2Q (0.616 ± 0.027). Does this undermine the claim that 3Q is more robust? How would you determine statistical significance here?**

<details>
<summary>Suggested Answer</summary>

The larger std for 3Q (0.069 vs 0.027) means individual seeds vary more, but the mean difference (0.087) exceeds both standard deviations, suggesting the effect is real. To formally test significance with n=4 seeds, a paired t-test or Welch's t-test would be appropriate — pairing by seed to control for seed-specific variation. With only 4 data points, statistical power is limited; a bootstrap confidence interval on the mean difference would be more robust. The overlapping error bars (0.529+0.069=0.598 vs 0.616-0.027=0.589) do overlap slightly, so the claim is suggestive but not conclusive at conventional significance levels (p<0.05). More seeds would resolve this.

</details>

---

### Q34. The τ=0.9 Anomaly (Slide 5)

**Slide 5 shows that Hopper 2Q gravity AUDC at τ=0.9 is 0.687, which is *better* than τ=0.8 (0.730), breaking the monotonic trend of "higher τ = worse robustness." What could explain this non-monotonicity, and does it invalidate the general conclusion?**

<details>
<summary>Suggested Answer</summary>

The non-monotonicity at τ=0.9 is likely a statistical artifact driven by high cross-seed variance — the std at τ=0.9 is 0.097 (the highest of any configuration), meaning individual seeds range from 0.563 to 0.793. One lucky seed can pull the mean down. This doesn't invalidate the general trend because: (1) the overall pattern from τ=0.5 (0.538) to τ=0.8 (0.730) is clearly monotonic, (2) the τ=0.9 mean is still worse than τ=0.5 and τ=0.7, and (3) the high variance itself is a finding — τ=0.9 is unreliable. The general conclusion "lower τ improves robustness" holds as a trend, with the caveat that very high τ introduces instability that can produce outlier results in either direction.

</details>

---

### Q35. "No Single Config Dominates" (Slide 6)

**Slide 6 concludes that the "optimal (critic count, τ) pair is environment-dependent — no single config dominates across all tasks." From a practical engineering perspective, is this a satisfying conclusion? What would a practitioner actually do when deploying to a new environment?**

<details>
<summary>Suggested Answer</summary>

This is an honest but unsatisfying conclusion for practitioners — it essentially says "you need to tune per environment." A practical deployment strategy would be: (1) start with the conservative default (2Q, τ=0.7) which performs reasonably across all environments, (2) if the deployment environment is known to be unstable or shift-prone, lower τ to 0.5 and consider 3Q, (3) if computational budget allows, run a small ablation on a proxy environment that approximates deployment conditions. The deeper issue is that offline RL currently lacks a principled method for robustness-aware hyperparameter selection without environment interaction — this is exactly the "shift-aware τ selection" future direction proposed on the slide. Until such methods exist, the conservative default is the safest bet.

</details>

---

## Quick Reference — Questions by Difficulty

| Difficulty | Questions |
|---|---|
| **Foundational** (verify understanding) | Q1, Q2, Q3, Q12, Q14, Q32 |
| **Analytical** (connect concepts) | Q4, Q5, Q6, Q7, Q13, Q15, Q22, Q23, Q33 |
| **Critical** (evaluate & challenge) | Q8, Q9, Q10, Q17, Q18, Q19, Q20, Q24, Q25, Q26, Q34 |
| **Research-level** (extend & propose) | Q11, Q16, Q21, Q27, Q28, Q29, Q30, Q31, Q35 |
