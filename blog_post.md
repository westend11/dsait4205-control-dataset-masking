# How can we measure pixel-level invariance in visual models?
Arda Bulbul, 5811821


## 1 Introduction & Motivation
A fundamental goal of machine learning (and thereby deep learning) is **generalization**, and there are two ways in which a model might achieve it:

* **Interpolation** to nearby samples, bounding $\lVert f(\mathbf{x}) - f(\mathbf{y})\rVert \leq l\,\lVert \mathbf{x} - \mathbf{y}\rVert_2$, where one might regularize to lower the Lipschitz constant $l$ and make $f$ smoother. This is particularly important, especially in the context of reinforcement learning, as we want samples that are close in input space to map onto similar outputs;
* **Extrapolation** by learning in/equivariances: $\varphi(f(\mathbf{x})) = f(\phi(\mathbf{x})),\ \forall (\phi, \varphi) \in \Phi,\ \forall \mathbf{x}$. This is significantly more complicated to achieve relative to interpolation, requiring large amounts of training data to work reliably. Selecting the right architecture can both reduce the data hunger of the problem and help with learning in/equivariances—in most cases architecture determines whether learning is possible in the first place.

Now, why is this relevant for RL? To understand this, let's first briefly mention how learning is defined in RL. You have two main components: the agent (e.g. a robot arm in a manufacturing plant) and the environment (e.g. the manufacturing plant). The environment is defined by a state space (which can be either discrete or continuous), where the agent interacts with the environment—and based on the outcomes of these interactions (e.g. moving from state $s_0$ to $s_{10}$), the agent is either rewarded or punished. Based on the complexity of the task at hand and the environment, the agent must be trained on a considerably large amount of data. However, this simply is not enough. RL algorithms are deployed on physical agents who take real, physical actions; therefore the agent is not really allowed to take the 'wrong' actions, as they will have physical, possibly dangerous repercussions.

The large bulk of RL research is focused on in-distribution generalization, where training and test contexts are drawn from the same distribution. Hence, by training the model on a rich dataset of different contexts, we try to ensure that the model's queries are never OOD, as extrapolation is usually not a strength of most ML/DL algorithms. For real-life deployment this problem gets more significant as the environment becomes non-stationary, complex, concerning multiple actors (e.g. humans, other robots, etc.), and full of distractions. We can take action against this by expanding our dataset to achieve distraction-robustness, but to what end? Will we be able to anticipate everything? No.

A model that overfits to background/irrelevant pixels (e.g. a manufacturing robot taking the wrong action when a human is in sight, simply because this was never in the training dataset) breaks when the background or certain components of the environment change (out-of-distribution distractions). Therefore, a good practice in robust RL algorithms is ensuring that predictions depend **only on task-relevant pixels**, and are **invariant to task-irrelevant ones**. Otherwise, it would mean that the model is attributing meaning to everything within its input space.

This problem is a prevalent one in the RL research space, and a multitude of methods—namely data augmentation, data-augmented representation learning, and more—are already in active development. Among them, one of the most notable, high-performing techniques is **masking**. MaDi (Grooten et al., AAMAS 2024) [1] introduces the technique of learning a mask that suppresses distractions, trained from the reward signal alone. In a related vein, SGQN (Bertoin et al., NeurIPS 2022) [2] forces the agent's saliency onto decision-relevant pixels. Both share the same thesis: *identify which pixels matter, and ignore the rest.*

These methods *propose* a way to achieve the property. This project asks the complementary question: how do we **measure** whether a model has it? The goal of this control-dataset project is to build a simple dataset of 84×84 samples that turns the slogan "ignore irrelevant pixels" into a number—one that distinguishes a model that has the property from one that does not. We define the recurring terms once, up front:

| term | meaning |
|---|---|
| **task-relevant pixels** | the pixels genuinely needed to solve the task (here: the agent and the target) |
| **distractor** | task-irrelevant content added to a frame to try to fool the model |
| **invariance** | the model's output does *not* change when only irrelevant pixels change |
| **mask / saliency** | a per-pixel map of which pixels to keep (mask) or where the model looks (saliency) |
| **OOD** | out-of-distribution: test conditions not seen during training |


## 2 Dataset Design

### 2.1 Synthetic Scene (Task Definition)
We have an agent (green circle) with a defined visual target (red square) on a grey 84×84 frame (84×84 = DMControl convention, which matches the anchor papers [1, 2, 9]). The operational goal of the agent is to successfully label the relative directional position of the target, quantised into 8 classes (e.g. "NW" for "North-West"). Synthetically generating the dataset specifically enables maximal control, provides perfect ground-truth references, and is simply cheap and fast.

A task label is not a decoration here—it is what makes invariance *definable*. "The output should not change" is meaningless without an output, so every frame is tied to a concrete task (predicting the direction). The relevant pixels are then exactly the two sprites, and the label is a pure function of their coordinates, so the ground truth is exact and free.

> **Figure 1.** A clean scene `x` (left), its relevant-region mask `r` overlaid (middle), and the direction label `y` (e.g. NW). Source: `examples/track_a_figure.png`, bottom-left tile.

### 2.2 Input Configuration (Sample Definition)
Each sample carries a five-tuple, with the following elements:

* `x`: the clean frame without any added distractions,
* `x'`: the frame with an added distraction (corruption/distraction types discussed in 2.3),
* `m`: the binary mask specifying which pixel is a distraction (1) and which is not (0),
* `y`: the directional position of the target (one of 8 classes),
* `r`: the binary mask specifying which pixel is task-relevant (1) and which is not (0).

The pair `(m, r)` is what upgrades this from a segmentation dataset to a *control* dataset for invariance: `r` says where the model *should* look, and `m` says exactly where the distraction is—so we can later score both "did the prediction stay invariant?" and "did the model's mask cover the distractor and avoid the relevant region?"

### 2.3 Distractor Types
Each sample carries an `x'` with a distinct distraction type, testing only one axis at a time per sample with everything else frozen—textbook control design. The swept intensity axis is what makes each type a *control* rather than a one-off corruption. Here is the selection of distraction types and their swept axes:

* **shape** — a filled rectangle/ellipse/triangle; axis = area fraction of the frame `{1, 2, 4, 8, 16}%`,
* **noise** — an additive Gaussian patch; axis = standard deviation $\sigma \in \{10, 25, 50, 75, 100\}$ (on the 0–255 scale),
* **overlay** — an alpha-blended colour tint; axis = opacity $\alpha \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$,
* **texture** — a repeating checkerboard; axis = contrast $\in \{0.1, 0.3, 0.5, 0.7, 0.9\}$.

> **Figure 2.** For each distractor type, the triplet `x | x' | m` using the *same* base scene across rows, so only the distractor changes. Bottom row: the `r` overlay and one occluding example. Source: `examples/track_a_figure.png`.

### 2.4 Controlled Confounding Factor: Occlusion
There is one confound that, if left unhandled, silently invalidates the whole experiment. If a distractor lands **on** the agent or target, it hides task-relevant pixels—and then a *correct* model **should** change its output. Labelling such a sample as "expected to be invariant" would be wrong: we would be penalising a model for doing exactly the right thing.

We control this explicitly by constraining distractor placement and shipping two splits:

* **`non_occluding/`** — the distractor is placed strictly in the `r == 0` region, so it never touches the agent or target. Here the expected behaviour is **invariance**: a robust model's prediction should not move.
* **`occluding/`** — the distractor deliberately covers part of `r`. Here the expected behaviour is that the model **may** (and should) change its output. This split is a **negative control**: it proves the dataset is not trivially "always invariant", and it gives us a way to detect the opposite failure (a model that ignores *everything*, including pixels it needs).

Concretely, placement samples a candidate patch and accepts it only if it avoids `r` (non-occluding) or is centred on a relevant pixel (occluding). The invariant—"`m` overlaps `r` if and only if the sample is in the occluding split"—holds by construction for every sample.


## 3 Dataset Generation
The generator is pure `numpy` + `Pillow` (no deep-learning dependencies), and its two non-negotiable properties are **reproducibility** and **balance**.

**Reproducibility — one seed → one byte-identical dataset.** A single `np.random.default_rng(seed)` is created once and threaded through every random call; the global `np.random` is never touched. Determinism has two requirements, both satisfied: (i) one source of randomness, and (ii) one fixed order of consumption—the nested loops `for scene → for split → for type → for intensity` never change order. Re-running `python generate.py --track A --seed 0` therefore reproduces the dataset bit-for-bit (verified by hashing every output file across two runs: zero differences).

**Balance.** Because the loops are a full cross-product, every `(split × type × intensity)` cell receives the same number of samples; `dataset_info.json` asserts this. The default run of 30 scenes yields 1200 samples (30 scenes × 4 types × 5 intensities × 2 splits), perfectly balanced across the control axes.

**Provenance.** Every sample writes one JSON row (`metadata.jsonl`) recording its type, intensity level and value, seed, occluding flag, label, sprite coordinates, and the paths to its four images—so any sample is fully reconstructible from its row alone. The on-disk layout stores the clean frame `x` and relevant mask `r` once per scene (shared by all of that scene's distractors) and the distracted pair `(x', m)` per sample:

```
dataset/
├── scenes/          scene_NNNNN_x.png, scene_NNNNN_r.png   (shared per scene)
├── non_occluding/   sample_NNNNNN_xp.png, sample_NNNNNN_m.png
├── occluding/       sample_NNNNNN_xp.png, sample_NNNNNN_m.png
├── metadata.jsonl   one provenance row per sample
└── dataset_info.json  seed, counts, balance flag, intensity grids
```

The whole dataset regenerates from a seed in one command, and `requirements.txt` pins `numpy`/`Pillow`/`scipy` so the byte-identical guarantee holds across machines.


## 4 Measurement Protocol
The dataset is an instrument; here is how a reader uses it to score a model `f`. Two metrics, mirroring the two halves of the property.

**Invariance gap.** For a model `f`, define
$$\Delta = d\big(f(\mathbf{x}),\, f(\mathbf{x'})\big),$$
the distance between the model's response to the clean and distracted frame. We use the total-variation distance between predicted class distributions ($\Delta = \tfrac{1}{2}\sum_i |p_i(\mathbf{x}) - p_i(\mathbf{x'})|$), which is $0$ for identical predictions and $1$ for disjoint ones. A model with the property should satisfy, on the **non-occluding** split, $\Delta \approx 0$ *and* a flat $\Delta$ as intensity rises. On the **occluding** split, $\Delta$ should instead **rise**—a sanity check that the model reacts when it genuinely should.

**Mask alignment.** If the model exposes a mask or saliency $\hat{\mathbf{s}}$, we report its intersection-over-union with the ground-truth distractor mask, $\text{IoU}(\hat{\mathbf{s}}, \mathbf{m})$. A *masking* model (MaDi-style) should **cover** the distractor, so $\text{IoU}$ is high; a *robust* model should **exclude** the distractor from its saliency, so $\text{IoU}(\text{saliency}, \mathbf{m}) \approx 0$.

To show the protocol runs end-to-end, the code also evaluates a trivial stand-in `f = `raw pixels. By construction this is *not* invariant—if pixels change, `f` changes—so its $\Delta$ rises with intensity. That is precisely the baseline curve a real, robust model is expected to **flatten**, which motivates the experiment in Section 5.


## 5 Validation Experiment
A control dataset is only useful if it can actually *separate* a model that has the property from one that does not. To validate this, we train two small CNNs on the 8-class direction task and let the dataset grade them:

* **vanilla** — trained on clean frames `x` only; never taught to ignore distractions.
* **robust** — trained on clean frames *and* their **non-occluding** distracted frames `x'` (a label-preserving augmentation, in the spirit of MaDi/SODA [1, 3]); taught to ignore off-target junk.

Both models reach well above the 0.125 chance accuracy on held-out scenes (vanilla 0.82, robust 0.71), so their predictions—and therefore the $\Delta$ comparison—are meaningful. *(Implementation note: the task depends on the sprites' positions, so the network must preserve spatial location; an early version using global average pooling collapsed to chance, and was fixed by flattening the feature map.)* We then sweep distractor intensity and measure $\Delta$ on each split.

> **Figure 3.** Invariance gap $\Delta$ vs. distractor intensity. Left (`non_occluding`): vanilla rises, robust stays flat near zero. Right (`occluding`): both rise. Source: `verify_dl_curves.png`.

The results confirm three claims, each corresponding to one thing a valid instrument must do:

1. **It detects non-invariance.** On `non_occluding`, the vanilla model's gap climbs monotonically with intensity ($\Delta$: 0.25 → 0.61), even though the distractor never touches the sprites. The model keyed on irrelevant pixels, and the dataset exposes exactly that.
2. **It confirms invariance.** On the *same* `non_occluding` frames, the robust model's gap stays flat and near zero ($\Delta$: 0.006 → 0.027, roughly 20× smaller). Same scenes, opposite verdict—so the dataset is *discriminative*.
3. **Its negative control is valid.** On `occluding`, **both** models' gaps rise ($\Delta$ up to ≈0.56), including the robust one. Covering the agent/target genuinely destroys task information, so a correct model *should* react. The dataset is therefore not rigged to always reward "invariant".

The cleanest single observation: the robust model **ignores distractors that miss the target ($\Delta \approx 0$) and reacts to ones that hit it ($\Delta \approx 0.5$)**—its invariance tracks the relevant-region mask `r`, which is exactly the property we set out to measure. The effect holds across all four distractor types, not just one easy case.


## 6 Limitations
A few honest caveats bound the scope of these claims:

* **Synthetic simplicity.** The scenes are deliberately minimal; the property is cleanly *isolated*, but the dataset does not capture the difficulty of natural images. It is a probe, not a benchmark of absolute robustness.
* **The validation model is tiny.** The two CNNs exist only to demonstrate that the dataset *discriminates*; the dataset measures whatever `f` you plug in and does not certify any single architecture. Stronger models would sharpen, not change, the conclusions.
* **Approximate class balance.** The dataset is *exactly* balanced across the control axes (type × intensity × split), which is what the experiment needs, but the 8 direction classes are only *approximately* uniform at small scene counts due to random placement.
* **One property, one track.** This work covers the masking/distractor axis only. A natural extension (Track B) sweeps *augmentation strength* and labels the crossover point at which the task signal is destroyed—reusing the same scene renderer—to probe the "augmentation strength is a double-edged knob" tension raised by MaDi [1]. It is left as future work.


## References
1. B. Grooten, T. Tomilin, G. Vasan, M. E. Taylor, A. R. Mahmood, M. Fang, M. Pechenizkiy, D. C. Mocanu. *MaDi: Learning to Mask Distractions for Generalization in Visual Deep Reinforcement Learning.* AAMAS 2024, pp. 733–742. arXiv:2312.15339.
2. D. Bertoin, A. Zouitine, M. Zouitine, E. Rachelson. *Look where you look! Saliency-guided Q-networks for generalization in visual Reinforcement Learning.* NeurIPS 2022. arXiv:2209.09203.
3. N. Hansen, X. Wang. *Generalization in Reinforcement Learning by Soft Data Augmentation (SODA).* ICRA 2021. arXiv:2011.13389.
4. N. Hansen, H. Su, X. Wang. *Stabilizing Deep Q-Learning with ConvNets and Vision Transformers under Data Augmentation (SVEA).* NeurIPS 2021. arXiv:2107.00644.
5. M. Laskin, K. Lee, A. Stooke, L. Pinto, P. Abbeel, A. Srinivas. *Reinforcement Learning with Augmented Data (RAD).* NeurIPS 2020. arXiv:2004.14990.
6. I. Kostrikov, D. Yarats, R. Fergus. *Image Augmentation Is All You Need (DrQ).* ICLR 2021. arXiv:2004.13649.
7. A. Stone, O. Ramirez, K. Konolige, R. Jonschkowski. *The Distracting Control Suite – A Challenging Benchmark for Reinforcement Learning from Pixels.* 2021. arXiv:2101.02722.
8. Y. Tassa et al. *DeepMind Control Suite.* 2018. arXiv:1801.00690.
