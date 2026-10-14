# Your Diffusion Model is Just a Distribution Matching Model

Read the diffusion papers in order, and the same object keeps coming back: the training data, and the direction that points back toward that data after you add a little noise. This note writes that idea in plain language. Then it checks the idea on a few small experiments, small enough that every mode fits on one screen.

---

## What the papers have been fitting

The usual story is easy to tell. You add noise to a picture. You train a network to remove that noise. At test time you start from pure noise and walk backward, step by step, until a picture appears. The papers that justify this procedure are more specific. They say what the network is fitting.

Hyvärinen’s score matching [1] estimates the slope of the log density, written \(\nabla_x \log p(x)\). That slope is called the score. The method matches this slope to the data. It does this without building a sampler, and without computing the partition function. Vincent [2] then noticed a useful fact. If you train a denoising autoencoder only to undo Gaussian noise, you are estimating that same score for the *noisy* density. Undo the noise at one scale, and you have matched \(p_\sigma\): the data after a Gaussian blur, in probability space.

Song and Ermon [3] made the noise scale an input of the network. The network sees many noise levels, and it fits \(\nabla \log p_{\sigma}(x)\) at each level. Ho et al. [4] wrote the form that most people train today. The network predicts the noise \(\varepsilon\) that was added. That prediction is the same score, only rescaled. Sohl-Dickstein et al. [5] had already described generation as running a diffusion backward in time. The later papers estimate the scores that this backward pass needs. Song et al. [6] wrote the whole ladder of noise levels as one stochastic differential equation. Sampling is then an integral of those estimated scores, along the smoothed densities \(\{p_t\}\). Karras et al. [7] showed that many of these samplers are the same probability-flow equation. They differ in the noise schedule, and in how the network output is scaled. In every case, the thing you integrate is still a field of the training distribution.

Later losses say the same thing more directly. Flow matching [8] trains a vector field. Its paths carry a simple starting distribution onto the data distribution. Rectified flow [9] learns a similar transport. The way it pairs points makes the paths straighter. Distribution-matching distillation [10] shortens a long sampler by matching the student’s distribution to the teacher’s. Each of these losses names the distribution it wants to match.

Another group of papers asks what happens when the training set is finite. Replication studies [11] and training-data extraction [12] find a clear pattern. When the model is large, and when the data contain many copies, new samples land on training examples, or very near them. Pidstrigach [13] gives conditions for two outcomes. In one case the model recovers the support of the data distribution, meaning the set of points that can actually occur. In the other case that support is just the finite list of training points. Follow the exact score of the empirical distribution all the way down to zero noise, and the backward process breaks into small lumps around those points [14]. The empirical distribution here is \(\widehat{p} = \frac1N \sum_i \delta_{x^{(i)}}\): one spike on each training point. Its score behaves like a soft nearest neighbor. It pulls a sample toward points that were actually seen [2], [6].

The same loss also has a large-data reading. Oko et al. [15] show that careful empirical score matching estimates smooth densities at nearly the best possible rate. The method also adapts when the data live on a lower-dimensional set. Kadkhodaie et al. [16] train denoisers on separate pieces of a large dataset. Once each piece is big enough, the learned scores agree with each other, and the generated images are new images, distinct from the training set. Both readings ask one question: which distribution is the denoiser matching? With a small set, or a set full of copies, that distribution looks like the sample itself. With a large set, and with the preferences built into the network, it looks like a smoother density that the architecture can represent.

Read together, these papers describe a training procedure that agrees with the empirical distribution — the distribution of the data you actually trained on — and with the parts of that distribution that the condition \(c\) points to.

---

## What the loss is averaging

Start from a clean example \(x_0\) and a condition \(c\). Add Gaussian noise \(\varepsilon\) on a schedule \(\bar\alpha_t\):

\[
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \varepsilon.
\]

The usual training loss asks the network to guess that noise:

\[
\mathcal{L}(\theta)=\mathbb{E}\big[
\|\varepsilon-\varepsilon_\theta(x_t,t,c)\|_2^2
\big].
\]

In words: the network sees a noisy picture, a time, and a condition. It tries to name the noise that was added.

The best possible guess for \(\varepsilon\) is fixed by the smoothed conditional density \(p_t(x_t \mid c)\). “Smoothed” means the clean examples have been blurred by the noise at time \(t\). “Conditional” means we only use the examples that came with this \(c\). In score form,

\[
\nabla_{x_t}\log p_t(x_t \mid c)
= \frac{1}{\sqrt{1-\bar\alpha_t}}\,
\mathbb{E}[-\varepsilon \mid x_t, c].
\]

So predicting \(\varepsilon\) is the same job as fitting the score of that smoothed conditional. On a finite dataset, the conditional is simply the subset of training pairs that used this \(c\). When many examples share \(c\), the score is clear, and it points back at those examples. When \(c\) never showed up in training, the loss never provided a target for it.

The rest of this note tries to see that fact directly. A short text string is a useful probe. Nearby strings are easy to write, and easy to tell apart. Many real training setups already pass a string into the network: a template, a caption, or a short code.

![Five fields — subject, color, layout, quality, and style — are joined into one exact string. That string is the key for one drawer of training images, written \(p(x \mid c)\). The same string at test time opens the same drawer. A paraphrase or a typo is a different string, so it looks up a different drawer, often an empty one.](figures/12_pe_schema.png){paper}

Real text-to-image systems put a richer string in this same place. Classifier-free guidance [17] trains one network in two ways: on pairs \((x, c)\), and on \(x\) with the condition left blank. At sampling time it follows the gap between those two scores, which makes the condition stronger. Latent diffusion [18] gets \(c\) from a caption encoder. The loss still matches the noise for the pairs that carried that particular string. A paraphrase stays a different vector until some other training step has tied the two strings together.

A new word can be added as a key on purpose. Textual inversion [19] learns one embedding from a few photos of one subject. DreamBooth [20] ties a rare name to that subject with a short fine-tune. After that, the new name brings the subject back. What you see is co-occurrence: the new string was paired with those images, so the string retrieves them.

On this reading, training a string next to a visual mode is enough for that string to retrieve the mode later. A paraphrase, a typo, a code that never appeared beside the mode, or a combination left out of the training set, all point somewhere else. Sometimes that place is empty. Sometimes it is a neighbor that did appear in training. Brittle wording is the condition channel doing the job the loss asked for. The same pattern shows up in full-size images. Prompts drop objects, or attach the wrong color, when they ask for a combination the training pairs did not reliably contain [21], [22].

---

## Three places to look

The experiments below keep the dataset small, so you can see the training mass. We use rigid schema strings because a fixed key makes a hit and a miss easy to score. The string plays the same role that a class id or a caption would play: it selects one part of the training distribution. The picture-level version of this comparison — exact string, paraphrase, typo, and a held-out combination — is in the last section, next to the other wordings. It is the same test, with the strings written out.

| Step | What we look at | What shows up |
|------|-----------------|---------------|
| 1 · 2D ring | How many points, which modes are missing, which words were swapped | Samples follow the counts, and they follow the pairing |
| 2 · Lattice | The exact key of a cell that was left out of training | Samples fall on the nearest cell that was trained |
| 3 · Schema and language | The wording, moved from a template toward ordinary language | The picture stays with the string that was actually trained |

### 1 · 2D — mass, holes, and labels

Eight Gaussian islands sit on a ring, like a compass. Each island has its own string. We train a small conditional diffusion model on these points. The geometry is the same conditional story, drawn so you can see it.

The exact training string for an island retrieves that island. The hit rate is about 1. A paraphrase of the request leaves the ring. A typo inside the string also leaves the ring. Both have hit rate 0. The number in the corner of each panel is that hit rate.

![(a) The eight training islands, with N, E, S, and W marked. (b) The exact key for N puts every sample on the north island: hit 1.00. (c) A paraphrase of that request and (d) the key with one typo leave all islands empty: hit 0.00. The number above each panel is the hit rate, the fraction of samples that land on the island we asked for.](figures/20_2d_pe_board.png){paper}

Three more cuts, still on the training distribution. Delete the northeast island and ask with its old string: the island stays empty. Give the north island eight times as many points, and sample with no condition: the samples follow the counts. Train the word “north” on the south island and “south” on the north island: at test time each string lands where it was trained.

![(a) Training with the northeast island removed. The dashed ring marks the hole. (b) Asking with the old northeast key still finds nothing there: hit 0.00. (c) North has eight times as many training points as each other island. Gray bars are the training fractions, blue bars are unconditional samples, and the blue bars follow the gray ones. (d, e) After the words are swapped in training, “north” lands on the south island and “south” lands on the north island, both with hit 1.00.](figures/21_2d_ablation_board.png){paper}

Put points on the midpoints between islands and denoise them. They snap onto islands that have mass. There is no smooth bridge left between the islands. The exact northeast string retrieves the northeast island. The average of the north embedding and the east embedding does not. The model saw whole keys paired with whole modes.

![(a) Orange diamonds mark the midpoints between neighboring islands, where we start the samples. (b) After denoising, those points have moved onto the islands. Nothing stays in between. (c) The exact northeast key lands on the northeast island: hit 1.00. (d) The average of the north key and the east key misses that island: hit 0.00.](figures/22_2d_geometry_board.png){paper}

### 2 · Lattice — in-between and outside both land on trained cells

A 5×5 grid makes “in between” and “outside” easy to point at. Blue cells were in the training set. Gray rings were held out. We train on a partial grid — a checkerboard, an inner block, or only the corners — and then ask with the exact key of a cell that was held out. We score two things: did the sample hit the cell we asked for, and did it hit the nearest cell that was trained?

Cells that were trained still render in the right place. Held-out cells miss the cell we asked for. They land on a nearby trained cell instead. This is true for holes sitting between neighbors, and for cells outside the trained block. It looks like retrieval of nearby training mass, under a key the loss has seen. That is the same agreement the score papers describe.

![Blue disks are trained cells, and white rings were held out. (a) Asking for a trained cell puts the samples (dark blue) on that cell. (b–d) Asking for held-out cells, in three training layouts: a checkerboard, an inner block with the outer ring missing, and only the four corners. The orange samples gather on or near trained cells. (e) Two scores per setting. Gray is the hit rate on the cell we asked for, and blue is the hit rate on the nearest trained cell. For held-out cells the gray bar is 0.00 every time, while the blue bar is 0.45 for the checkerboard and 0.63 for the outer ring. Paraphrase and typo score 0.00 on both.](figures/30_render_matching_board.png){paper}

![Four held-out patterns. Under each panel, “asked” is the hit rate on the cell we asked for, and “nearest” is the hit rate on the closest trained cell. Asked is 0.00 in all four. Nearest is 0.45 for the checkerboard, 0.10 when only the corners were trained, 0.77 when the right columns were held out, and 0.63 when the outer ring was held out. When most samples miss the nearest cell, they still stay close to trained ones, as in panel (b).](figures/31_interp_extrap_grid.png){paper}

### 3 · Schema and language — how far the string can drift

The last suite uses the tiny pictures: one shape, one color, one position. It keeps that catalog fixed and changes only the condition string. We walk the string away from the exact template, one step at a time:

exact string, then a reordering, then a typo, then an opaque code, then a natural-language template, then free English, then Chinese.

Training is still conditional denoising. The string is a continuous vector, so an unseen wording is a new vector. It is not a special empty slot reserved in advance. Eight settings run in parallel. The table is the full run. “Exact key” is the hit rate for the training string. “NL template” is the hit rate for a fixed English sentence that describes the same picture.

| Regime | Exact key | NL template | What it shows |
|--------|-----------|-------------|----------------|
| `pe_schema` | 1.00 | 0.00 | The schema string works. Free English misses. |
| `pe_opaque` | 1.00 | 0.00 | The code `c1` retrieves the red pixels. The word “red” opens a different drawer. |
| `pe_nl_paired` | 0.97 | 0.94 | English works when it was trained together with the schema. |
| `nl_only` | 0.97 | 1.00 | English can be the key, when English is what was trained. |
| `holdout_combo` | — | — | The held-out combination is hit 0.19 of the time. The nearest trained mode is hit 0.38 of the time. |
| `remap` | 1.00 | 0.00 | The model keeps the training drawer. A human remapping of the code does not open it. |

The schema-only model, asked for a red circle on the left. The hit rate is the fraction of samples that land on that picture. Only the exact training string is reliable. A one-character typo still picks up a few samples. Every other wording misses.

| Probe | What changed | The string we asked with | Hit |
|-------|----------------|--------------------------|-----|
| D0 exact | Nothing. This is the training key. | `[PE] subject=circle \| color=red \| layout=left \| quality=hq \| style=flat` | 1.00 |
| D1 reorder | The same fields, written in a different order. | `[PE] color=red\|subject=circle\|layout=left\|quality=hq\|style=flat` | 0.00 |
| D2 typo | `layout=left` became `layout=lefx`. | `[PE] subject=circle \| color=red \| layout=lefx \| quality=hq \| style=flat` | 0.06 |
| D3 opaque | The words became codes: `s1`, `c1`, `p1`. | `[PE] subject=s1 \| color=c1 \| layout=p1 \| quality=hq \| style=flat` | 0.00 |
| D4 English template | A fixed English sentence for the same picture. | a red circle on the left, high quality, flat style | 0.00 |
| D5 free English | A looser English sentence, never used in training. | please show circle in red sitting toward the left side, still camera | 0.00 |
| D6 Chinese | The same request in Chinese. | 左侧的红色圆形，高质量，扁平风格 | 0.00 |
| D7 wrong color | `red` replaced by `crimson`, a color word the model never trained. | a crimson circle on the left, high quality, flat style | 0.00 |

![The same probes, across training setups. Rows are setups: schema only, opaque codes, schema paired with English, English only, and a paraphrase battery. Columns D0 through D7 are the strings from the list above, from the exact key out to a wrong string. Dark blue cells are hits near 1. Pale cells are misses. The dark column on the left is the exact key. A second dark cell appears only when that probe was part of training: the opaque code in the opaque row, and the English template in the paired and English-only rows.](figures/41_distance_heatmap.png){paper}

![Two training setups, same pictures. Each panel shows one picked sample, and the hit rate under it is measured over all samples for that string. Top row: the model was trained with schema strings such as color=red. The exact key returns the red circle (hit 1.00). The opaque code, English “red”, and Chinese all miss (hit 0.00). Bottom row: the color in training was the code c1, and the pixels were still red. Now the exact key and the opaque code both return the red circle (hit 1.00), and English “red” misses. The code retrieves the pixels it was paired with. The color word retrieves them only when that word was in the training pairs.](figures/42_opaque_vs_semantic.png){paper}

![What happens when English is added to training. Top row, schema only: the exact key returns the red circle (hit 1.00), and the English template returns a blue shape instead (hit 0.00). Bottom row, schema plus the paired English sentence: the exact key (hit 0.97) and the English template (hit 0.94) both return the red circle. Free English was never a training string, and it still misses (hit 0.00). The drawer opens for a sentence when that sentence appeared with the picture in training.](figures/43_nl_requires_cooccurrence.png){paper}

The same run, written as four comparisons. These are the numbers behind the pictures above.

| Check | Comparison | Hit |
|-------|------------|-----|
| Wording | Exact schema string | 1.00 |
| Wording | One-character typo (`lefx`) | 0.06 |
| Wording | English template, schema-only training | 0.00 |
| Wording | Free English, schema-only training | 0.00 |
| Co-occurrence | English template, schema only | 0.00 |
| Co-occurrence | English template, after that sentence was trained too | 0.94 |
| Codes | Exact code, and the opaque probe, in the opaque-code model | 1.00 |
| Codes | The English word “red”, in that same model | 0.00 |
| Composition | Held-out combination, intended mode | 0.19 |
| Composition | Held-out combination, nearest trained mode | 0.38 |

> **Readout.** An opaque code brings back the pixels it was paired with, even when the code is not a color word. An English sentence brings a mode back when that sentence was in the training pairs. A combination that was held out stays near modes that were trained. The condition channel is following co-occurrence.

---

## What this reading leaves open

A new sample can sit far from any single training image and still belong to a rich empirical distribution [15], [16]. The size of the dataset, the way it was filtered, and the augmentations all change which distribution is being matched, and how densely that distribution covers the pictures you care about. Extra structure can also come from the architecture, from a condition encoder trained with some other objective, or from a loss written for a job beyond \(\mathcal{L}\). Three examples are guidance that strengthens a condition score [17], a made-up word fit to a handful of images [19], and a reward that prefers samples with the right composition [22].

The claim in this note is narrower. Under the ordinary denoising loss, what the model has been shown is the training distribution, and the condition values that appeared next to it. Treating a paraphrase as the same request, or building a combination that was never seen, asks for something further. In the lattice experiment, a lot of what people call interpolation or extrapolation is the sampler landing on the nearest region that had training mass.

---

## Conclusion

Put the score-matching papers in a line, and the target is the data distribution [1]–[10], [15]. Put the experiments in a line, and the same target is visible. Exact keys that were trained with a mode bring that mode back. Strings that never appeared next to it usually miss. Missing modes stay missing. Held-out locations fall toward locations that were trained.

If a system should treat a paraphrase as the same request, or should build a combination it never saw, that behavior has to enter through something the matching loss can see: the data, the way \(c\) is written, or an objective written for that behavior.

---

## References

1. A. Hyvärinen. *Estimation of Non-Normalized Statistical Models by Score Matching.* JMLR, 2005.
2. P. Vincent. *A Connection Between Score Matching and Denoising Autoencoders.* Neural Computation, 2011.
3. Y. Song, S. Ermon. *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS, 2019. https://arxiv.org/abs/1907.05600
4. J. Ho, A. Jain, P. Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS, 2020. https://arxiv.org/abs/2006.11239
5. J. Sohl-Dickstein, E. A. Weiss, N. Maheswaranathan, S. Ganguli. *Deep Unsupervised Learning using Nonequilibrium Thermodynamics.* ICML, 2015. https://arxiv.org/abs/1503.03585
6. Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR, 2021. https://arxiv.org/abs/2011.13456
7. T. Karras, M. Aittala, T. Aila, S. Laine. *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS, 2022. https://arxiv.org/abs/2206.00364
8. Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le. *Flow Matching for Generative Modeling.* ICLR, 2023. https://arxiv.org/abs/2210.02747
9. X. Liu, C. Gong, Q. Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR, 2023. https://arxiv.org/abs/2209.03003
10. T. Yin, M. Gharbi, R. Zhang, E. Shechtman, F. Durand, W. T. Freeman, T. Park. *One-step Diffusion with Distribution Matching Distillation.* CVPR, 2024. https://arxiv.org/abs/2311.18828
11. G. Somepalli, V. Singla, M. Goldblum, J. Geiping, T. Goldstein. *Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models.* CVPR, 2023. https://arxiv.org/abs/2212.03860
12. N. Carlini et al. *Extracting Training Data from Diffusion Models.* USENIX Security, 2023. https://arxiv.org/abs/2301.13188
13. J. Pidstrigach. *Score-Based Generative Models Detect Manifolds.* NeurIPS, 2022. https://arxiv.org/abs/2206.01018
14. G. Biroli, T. Bonnaire, V. de Bortoli, M. Mézard. *Dynamical Regimes of Diffusion Models.* Nature Communications, 2024. https://arxiv.org/abs/2402.18491
15. K. Oko, S. Akiyama, T. Suzuki. *Diffusion Models are Minimax Optimal Distribution Estimators.* ICML, 2023. https://arxiv.org/abs/2303.01861
16. Z. Kadkhodaie, F. Guth, E. P. Simoncelli, S. Mallat. *Generalization in Diffusion Models Arises from Geometry-Adaptive Harmonic Representations.* ICLR, 2024. https://arxiv.org/abs/2310.02557
17. J. Ho, T. Salimans. *Classifier-Free Diffusion Guidance.* arXiv, 2022. https://arxiv.org/abs/2207.12598
18. R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR, 2022. https://arxiv.org/abs/2112.10752
19. R. Gal, Y. Alaluf, Y. Atzmon, O. Patashnik, A. H. Bermano, G. Chechik, D. Cohen-Or. *An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion.* ICLR, 2023. https://arxiv.org/abs/2208.01618
20. N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, K. Aberman. *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation.* CVPR, 2023. https://arxiv.org/abs/2208.12242
21. H. Chefer, Y. Alaluf, Y. Vinker, L. Wolf, D. Cohen-Or. *Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models.* SIGGRAPH, 2023. https://arxiv.org/abs/2301.13826
22. K. Huang, K. Sun, E. Xie, Z. Li, X. Liu. *T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-Image Generation.* NeurIPS, 2023. https://arxiv.org/abs/2307.06350
