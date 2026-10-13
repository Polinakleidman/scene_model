# The mathematics of SCENE and multimodal SCENE (`scene_mm`)

Full derivation: the model, all likelihoods, the loss, gradients, optimisation and metrics. Line numbers refer to [scene_mm/model.py](../scene_mm/model.py), [scene_mm/train.py](../scene_mm/train.py) and [scene_mm/data.py](../scene_mm/data.py).

---

## 0. Notation

| Symbol | Meaning | Size |
|---|---|---|
| $x_{ij}\in\{0,1,2,\dots\}$ | RNA count: molecules of gene $j$ detected in cell $i$ | $n$ cells × $G$ genes |
| $y_{ik}$ | count of protein $k$ in cell $i$ (CITE-seq ADT) | $n$ × $K$ proteins |
| $z_i\in\mathbb R^d$ | position of a cell, **shared** by RNA and proteins | $n\times d$ |
| $w_j\in\mathbb R^d$ | position of a gene | $G\times d$ |
| $v_k\in\mathbb R^d$ | position of a protein | $K\times d$ |
| $\psi^{R}_i,\ \psi^{P}_i$ | log depth of a cell, separately for RNA and proteins | scalars |
| $\rho_j,\ \rho_k$ | log baseline level of a gene or protein | scalars |
| $g^{(\ell)}_{b,j}$ | batch effect of level $\ell$ for batch $b$ and feature $j$ | see §1.2 |
| $\alpha>0$ | slope of the detection sigmoid (ZIP only) | 1 scalar |
| $\theta_j>0$ | NB inverse dispersion (size) of feature $j$ | one per feature |
| $\beta_k$ | log background of protein $k$ | one per protein |
| $d_{ij}=\lVert z_i-w_j\rVert_2$ | Euclidean distance, **not squared** | |

Only $x$ and $y$ are observed; everything else is a parameter fitted during training. There is no neural network: every cell and every feature has its own free parameters.

---

## 1. Cell–feature affinity

### 1.1 Main formula

$$
\eta_{ij} \;=\; \psi_i + \rho_j + \sum_{\ell} g^{(\ell)}_{b_\ell(i),\,j} \;-\; \lVert z_i - w_j\rVert_2
$$

Code: `Modality.params`, lines 112–116: `eta = cell_re[cells] + feat_re.T + b - dist`, with `dist = torch.cdist(z, W)`.

- $\eta$ is the log of the expected count, as in a generalised linear model (GLM) with a log link.
- $\psi_i$ accounts for the total number of molecules in the cell, $\rho_j$ for how active the gene is overall.
- **Geometry enters only through $-d_{ij}$**: the closer the points, the larger $\eta$.
- The distance is not squared, as in the authors' code. With $d$ instead of $d^2$, distant pairs are not penalised quadratically, so the periphery of the map is not squashed.

The log link makes effects multiplicative: $e^{\eta} = e^{\psi_i}\, e^{\rho_j}\, e^{-d_{ij}}$ = depth × level × specificity. Each unit of distance multiplies the expected count by $e^{-1}\approx0.37$.

### 1.2 Batch effects (a hierarchy of levels)

Each batch level $\ell$ (for example "donor", or "lab → sub-batch") has its own term:

- **full** (`full`): $g^{(\ell)}_{b,j}$ is a free number for every batch × feature pair;
- **low rank** (`lowrank`, rank $r$): $g^{(\ell)}_{b,j} = u_b^\top s_j$ with $u_b, s_j\in\mathbb R^r$. Fewer parameters and less risk of absorbing biology.

Code: `BatchEffects.forward`, lines 56–63. Terms of different levels **add up**, so nested batches are described by $g^{(1)}+g^{(2)}$.

**Where batch terms are used in our runs** (`scripts/run_experiment.py`): PBMC trained on all donors, **no batch term**; PBMC leave-one-donor-out, donor, `lowrank`, rank 2; NeurIPS, `batch` (site × donor), `lowrank`, rank 2. When a batch term is on, it enters both RNA and proteins (for proteins it multiplies both background and signal, §3).

Why it works: $g$ depends only on the batch and the feature, not on the biological state of the cell. A shift in a gene's expression common to a whole batch goes into $g$, while differences between cells within a batch stay in $z_i$.

---

## 2. RNA likelihoods

Common pattern: $\eta_{ij}$ gives the parameters of a distribution, and the distribution gives the probability of the observed count $x_{ij}$. Only this step changes; the geometry (§1) is the same.

### 2.1 Poisson (base case, variant `poisson`)

$\lambda_{ij} = e^{\eta_{ij}}$, $\quad P(x\mid\lambda) = \dfrac{\lambda^x e^{-\lambda}}{x!}$.

$$
\log P(x) = x\log\lambda - \lambda - \log x! = x\,\eta - e^{\eta} - \log\Gamma(x+1).
$$

Properties: $\mathbb E[x]=\lambda$, $\operatorname{Var}[x]=\lambda$, so the variance equals the mean. Here $\lambda$ is the Poisson rate, the expected count. Code: lines 142–144.

### 2.2 The authors' ZIP, a hurdle model (variant `zip`)

The name ZIP ("zero-inflated Poisson") is historical. Strictly, this is a **hurdle** model: zeros come only from a separate probability, and the Poisson part never produces them.

**Parameters:**
$$
\pi_{ij} = \sigma(\alpha\,\eta_{ij}) = \frac{1}{1+e^{-\alpha\eta_{ij}}},\qquad \lambda_{ij} = e^{\eta_{ij}}.
$$
One $\eta$ gives two numbers: the detection probability $\pi$ and the rate $\lambda$. The parameter $\alpha$ sets how sharply $\pi$ goes from 0 to 1.

**Generative story:**
1. With probability $1-\pi$ the gene is not detected: $x=0$.
2. With probability $\pi$ it is detected, and $x$ is drawn from a **zero-truncated Poisson**.

**Normalising the truncated Poisson.** We need a distribution on $\{1,2,\dots\}$. The Poisson probabilities for $x\ge1$ sum to
$$
\sum_{x\ge 1}\frac{\lambda^x e^{-\lambda}}{x!} = 1 - P(0) = 1-e^{-\lambda}.
$$
Hence
$$
P(x\mid x\ge1) = \frac{\lambda^x e^{-\lambda}}{x!\,(1-e^{-\lambda})} = \frac{\lambda^x}{x!\,(e^{\lambda}-1)}.
$$

**Full distribution:**
$$
P(x) = \begin{cases} 1-\pi, & x=0,\\[4pt] \pi\cdot\dfrac{\lambda^x}{x!\,(e^{\lambda}-1)}, & x\ge1.\end{cases}
$$

**Log-likelihood:**
$$
\log P(x) = \begin{cases} \log(1-\pi), & x=0,\\[4pt] \log\pi + x\log\lambda - \log\Gamma(x+1) - \log(e^{\lambda}-1), & x\ge1.\end{cases}
$$

**Numerical stability.** For small $\lambda$, $e^\lambda-1$ loses precision, so the code uses
$$
\log(e^{\lambda}-1) = \lambda + \log(1-e^{-\lambda}) = \lambda + \log\big(-\operatorname{expm1}(-\lambda)\big).
$$
Code: lines 137–141 and 160–165, `log_expm1 = lam + log(-expm1(-lam))`.

**Moments:**
$$
\mathbb E[x\mid x>0] = \frac{\lambda}{1-e^{-\lambda}},\qquad \mathbb E[x] = \pi\,\frac{\lambda}{1-e^{-\lambda}}.
$$
Code: `mean_given_positive`, `mean`, lines 187–200.

**Main limitation.** The non-zero part is Poisson, so its spread is fixed by $\lambda$: $\operatorname{Var}[x\mid x>0] \approx \mathbb E[x\mid x>0]$ for large $\lambda$. There is nothing to describe extra spread (over-dispersion).

### 2.3 ZIP-decoupled (variant `zip_decoupled`)

The only difference: detection has its own cell and gene effects,
$$
\pi_{ij} = \sigma\big(\alpha(\psi^{\pi}_i + \rho^{\pi}_j + g - d_{ij})\big),\qquad \lambda_{ij} = e^{\psi_i + \rho_j + g - d_{ij}}.
$$
The distance is shared. The idea: "is the gene detected" and "how many copies" may depend on the gene differently. Result (H1): detection did not improve. Code: lines 89–91 and 120–122.

### 2.4 Negative binomial (variant `nb`)

**Parameters:** $\mu_{ij}=e^{\eta_{ij}}$ (the mean) and $\theta_j>0$ (the gene's inverse dispersion). There is no $\alpha$.

**Derivation as a Poisson–gamma mixture.** Let the true rate of a gene in a cell be random, because genes are transcribed in bursts and cells differ:
$$
\ell \sim \operatorname{Gamma}(\text{shape}=\theta,\ \text{rate}=\theta/\mu),\qquad x\mid\ell \sim \operatorname{Poisson}(\ell).
$$
Then $\mathbb E[\ell]=\mu$ and $\operatorname{Var}[\ell]=\mu^2/\theta$. Integrating out $\ell$:
$$
P(x) = \int_0^\infty \frac{\ell^x e^{-\ell}}{x!}\cdot\frac{(\theta/\mu)^\theta}{\Gamma(\theta)}\ell^{\theta-1}e^{-\ell\theta/\mu}\,d\ell
= \frac{(\theta/\mu)^\theta}{x!\,\Gamma(\theta)}\int_0^\infty \ell^{x+\theta-1}e^{-\ell(1+\theta/\mu)}\,d\ell .
$$
The integral is a gamma function, $\int_0^\infty \ell^{a-1}e^{-c\ell}d\ell=\Gamma(a)/c^a$, with $a=x+\theta$ and $c=(\theta+\mu)/\mu$. This gives
$$
P(x) = \frac{\Gamma(x+\theta)}{\Gamma(\theta)\,x!}\left(\frac{\theta}{\theta+\mu}\right)^{\theta}\left(\frac{\mu}{\theta+\mu}\right)^{x}.
$$

**Moments** (law of total variance):
$$
\mathbb E[x]=\mu,\qquad \operatorname{Var}[x] = \mathbb E[\ell] + \operatorname{Var}[\ell] = \mu + \frac{\mu^2}{\theta}.
$$
The first term is Poisson sampling noise, the second is biological variation of the rate between cells. As $\theta\to\infty$ this becomes Poisson; a small $\theta$ means strong over-dispersion. In our data 81% of genes have $\theta<1$.

**Log-likelihood** (code: lines 145–148 and 169–175):
$$
\log P(x) = \theta\big(\log\theta - \log(\theta+\mu)\big) + x\big(\log\mu - \log(\theta+\mu)\big) + \log\Gamma(x+\theta) - \log\Gamma(\theta) - \log\Gamma(x+1).
$$

**Zeros and detection.** There is no separate parameter for zeros; the zero probability follows from $\mu$ and $\theta$:
$$
P(x=0) = \left(\frac{\theta}{\theta+\mu}\right)^{\theta},\qquad P(x>0)=1-\left(\frac{\theta}{\theta+\mu}\right)^{\theta}.
$$
Code: `prob_detect`, lines 177–185. This is used for the AUC.

### 2.5 Comparison

| | Poisson | ZIP (hurdle) | NB |
|---|---|---|---|
| From $\eta$ we get | $\lambda=e^\eta$ | $\pi=\sigma(\alpha\eta)$, $\lambda=e^\eta$ | $\mu=e^\eta$ |
| $P(0)$ | $e^{-\lambda}$ | $1-\pi$ | $(\theta/(\theta+\mu))^\theta$ |
| Variance | $\lambda$ | Poisson for non-zeros | $\mu+\mu^2/\theta$ |
| Extra parameters | — | $\alpha$ | $\theta_j$ per gene |

---

## 3. Protein likelihood: NB with background (`nb_bg`)

$$
\mu_{ik} = \underbrace{e^{\psi^P_i + g_{ik} + \beta_k}}_{\text{background}} \;+\; \underbrace{e^{\psi^P_i + g_{ik} + \rho_k - \lVert z_i - v_k\rVert}}_{\text{signal}}
= e^{\psi^P_i + g_{ik}}\Big(e^{\beta_k} + e^{\rho_k - \lVert z_i - v_k\rVert}\Big),
$$
$$
y_{ik}\sim \operatorname{NB}(\mu_{ik},\theta_k).
$$
Code: lines 127–130 (`fg + bgm`). The log-likelihood is the formula of §2.4 with $\mu_{ik}$ in place of $\mu_{ij}$.

In words: expected protein count = depth × (background + level × $e^{-\text{distance}}$), with depth $=e^{\psi^P}$, background $=e^{\beta}$, level $=e^{\rho}$. Because of the sum, $\log\mu_{ik} = \psi^P_i + g_{ik} + \log\big(e^{\beta_k} + e^{\rho_k-d}\big)$ is not linear: near the protein it behaves like the RNA formula, far away it floors at the background.

**What the terms mean:**
- **Background** does not depend on the cell's position. It is antibody that does not come from real binding: non-specific sticking, ambient antibody left in the droplet after washing, binding to dead cells or debris. When a cell is far from $v_k$, the signal $\to0$ and $\mu_{ik}\to$ background, not 0. Without this term the model would have to place every cell close to the protein to explain non-zero counts everywhere.
- **Signal** has the same geometry as genes.
- One can read this as the sum of two independent sources of antibody counts: if background $\sim\operatorname{Pois}(a)$ and signal $\sim\operatorname{Pois}(b)$, the sum $\sim\operatorname{Pois}(a+b)$; NB over-dispersion is added on top. totalVI (Gayoso et al. 2021) models protein background similarly, as a mixture.

**Signal share** (used for H3 and for isotype controls):
$$
s_k = \frac{1}{n}\sum_i \frac{e^{\rho_k - \lVert z_i-v_k\rVert}}{e^{\beta_k}+e^{\rho_k - \lVert z_i-v_k\rVert}}.
$$
Batch and $\psi$ cancel out. Isotype controls get $s_k\approx0.003$–$0.007$; real proteins have a median of $0.42$.

**Why not ZIP for proteins:** only ~8–10% of protein entries are zero and the mean count is 25–50. Zeros are rare and the spread is large, so NB fits and a separate zero model is unnecessary.

**Separate depth for proteins:** $\psi^P_i$ is separate from $\psi^R_i$ because the total protein count depends on the antibody panel and staining. After training, $\psi^R_i$ correlates 0.83 with the log of the cell's total RNA count, and $\psi^P_i$ 0.82 with the log of its total protein count (PBMC, NB, 16D).

---

## 4. The loss

### 4.1 Objective

$$
\mathcal L(\Theta) = -\frac{1}{N}\Bigg[\sum_{(i,j)} \log P_{\text{RNA}}\big(\tilde x_{ij}\mid\eta_{ij}\big) \;+\; w\!\!\sum_{(i,k)\notin\mathcal M_P}\!\! \log P_{\text{NB}}\big(y_{ik}\mid\mu_{ik},\theta_k\big)\Bigg]
$$

- $\Theta$ is all parameters: $z, w, v, \psi, \rho, g, \beta, \theta, \alpha$.
- $\tilde x$ is the RNA matrix used for training. **10% of the non-zero counts are held out: in $\tilde x$ they are set to zero** (the authors' protocol, `data.py: mask_rna`). So the RNA sum runs over all pairs.
- $\mathcal M_P$ is the 10% of held-out protein entries. They are **excluded** from the sum (`mask_dense`), because for proteins a zero is rare and cannot be substituted.
- $w$ is the protein weight: 1 by default, 3 was best in the sweep.
- $N$ is the total number of entries (RNA + protein), so the scale of the loss does not depend on data size.

Code: `MultiSCENE.block_log_lik` (lines 221–234: `total += spec.weight * s`) and `train.fit` (lines 73–92: `(-ll / n_entries).backward()`).

### 4.2 Regularisation

AdamW with weight decay $\lambda_{wd}=10^{-3}$ on **the positions $z, w, v$ and the batch effects $g$**. The effects $\psi,\rho$ and $\theta,\alpha,\beta$ are not regularised, as in the authors' code. Code: `param_groups`, lines 209–219.

Purpose: a weak pull of all points toward the origin. Without it the model is invariant to shifting all points and the scales can drift. In AdamW this is decoupled decay rather than an exact L2 penalty in $\mathcal L$, but the interpretation is close to a Gaussian prior on the positions.

### 4.3 Parameterising positive quantities

$\alpha=\operatorname{softplus}(a)$, $\theta=\operatorname{softplus}(t)+10^{-4}$, where $\operatorname{softplus}(u)=\log(1+e^u)>0$. The optimiser updates the unconstrained raw parameter. (The sigmoid is used only for the detection probability $\pi$ in ZIP, which must lie in $(0,1)$.) Initial values: $\alpha=1$, $\theta=10$, $\beta=-1$. Positions are initialised from $\mathcal N(0,1)$, effects from $\mathcal N(0,0.01^2)$ (lines 82–96).

---

## 5. Gradients: why points attract and repel

Chain rule: $\dfrac{\partial \log P}{\partial z_i} = \dfrac{\partial\log P}{\partial\eta}\cdot\dfrac{\partial\eta}{\partial z_i}$, with
$$
\frac{\partial\eta_{ij}}{\partial z_i} = -\frac{z_i - w_j}{\lVert z_i-w_j\rVert}\quad(\text{minus the unit vector from } w_j \text{ to } z_i).
$$
So **if $\partial\log P/\partial\eta>0$, gradient ascent moves $z_i$ toward $w_j$**: the model under-predicted the count and the points should come closer. If it is $<0$, they move apart.

**Derivatives with respect to $\eta$** (obtained by differentiating the formulas of §2):

| Model | $\partial\log P/\partial\eta$ | Meaning |
|---|---|---|
| Poisson | $x - \lambda$ | more observed than expected → move closer |
| NB | $\dfrac{\theta\,(x-\mu)}{\theta+\mu}$ | the same, but **damped** for large $\mu$ or small $\theta$: an outlier is absorbed by the tail |
| ZIP, $x=0$ | $-\alpha\pi$ | a zero pushes away, more strongly if the model expected detection |
| ZIP, $x\ge1$ | $\alpha(1-\pi) + x - \dfrac{\lambda}{1-e^{-\lambda}}$ | detection pulls toward the gene + a correction for the count size |

Derivation for NB:
$\log P = \theta\log\theta - (\theta+x)\log(\theta+\mu) + x\log\mu + \text{const}$ and $\frac{d\mu}{d\eta}=\mu$, so
$\frac{\partial}{\partial\eta} = -\frac{(\theta+x)\mu}{\theta+\mu} + x = \frac{\theta(x-\mu)}{\theta+\mu}$.

**The key difference between NB and ZIP in the gradient.** Suppose $x=40$ and the model expected 10. The Poisson part of ZIP gives a force of $40-10=30$ and pulls the cell strongly toward the gene. NB with $\theta=1$ gives $\frac{1\cdot 30}{11}\approx2.7$: the outlier is explained by the heavy tail and the geometry barely moves.

Gradients with respect to $w_j$ are symmetric (opposite sign). For $\psi_i$, $\rho_j$, $g$ the derivative is $\partial\log P/\partial\eta$ without the geometric factor.

---

## 6. Optimisation

- **Algorithm:** AdamW, learning rate $0.05$, 600 epochs (`train.Config`).
- **Full batch:** one epoch is one gradient step on **all** the data.
- **In blocks:** the $n\times G$ matrix (30,000 × 20,000) does not fit into GPU memory at once. Cells are split into blocks; for each, $-\log P/N$ is computed and `backward()` is called. Gradients **accumulate**, and one `opt.step()` follows all blocks. This gives exactly the full-batch gradient.
- **Savings:** ~90% of RNA entries are zero. The expensive $\log\Gamma$ and $\log(e^\lambda-1)$ are evaluated only at non-zero entries; zeros use closed forms: ZIP $\log(1-\pi)$, Poisson $-\lambda$, NB $\theta(\log\theta-\log(\theta+\mu))$ (`log_lik_sum`, lines 150–175).
- **Gradients** come from PyTorch automatic differentiation, including through `torch.cdist`.

---

## 7. Fold-in: new cells without retraining

A new donor $D$ with cells $i\in D$. We freeze $w, v, \rho, \beta, \theta, \alpha$ and the batch effects of the old batches, add a new batch row $g_{D,\cdot}$ (initialised at zero, `BatchEffects.add_batch`) and solve
$$
\min_{\{z_i,\psi_i\}_{i\in D},\ g_{D,\cdot}}\; -\sum_{i\in D}\Big[\sum_j \log P(x_{ij}) + w\sum_k\log P(y_{ik})\Big]
$$
with the same optimiser for 200 epochs. If the new donor has only RNA, the protein term is absent; this is the setting of the "predict proteins from RNA" task.

**Protein prediction** for such a cell is the NB mean from §3:
$$
\hat y_{ik} = e^{\hat\psi^P_i}\Big(e^{\beta_k} + e^{\rho_k - \lVert z_i - v_k\rVert}\Big).
$$
The protein depth $\psi^P_i$ of a new cell is not observed; it is predicted by a linear regression $\hat\psi^P = a\,\psi^R + b$ fitted on the reference cells. The protein batch term is not used for the new donor. Code: [scripts/run_protein_prediction.py](../scripts/run_protein_prediction.py), lines 49–59.

**Aggregation** (no-training baseline): $z_i = \sum_j \omega_{ij} w_j / \sum_j\omega_{ij}$ with $\omega_{ij} = x_{ij}/\bar x_j$, the count relative to the gene's mean count.

---

## 8. Held-out metrics

Held out: $P$ = 10% of the non-zero RNA counts and the same number of **random true zeros** $Z$ (`mask_rna`).

- **Held-out NLL** (`train.evaluate_heldout`, lines 138–160):
  $$
  \text{NLL} = \tfrac12\Big(\overline{-\log P(x)}_{P} + \overline{-\log P(0)}_{Z}\Big),
  $$
  i.e. the mean over positives and the mean over zeros, weighted equally. This is the only metric that compares distribution families fairly.
- **AUC-ROC, PR-AUC, Max-F1:** classifying positives versus zeros by $P(x>0)$: $\pi$ for ZIP, §2.4 for NB.
- **Poisson deviance** (the authors' metric) on the positives:
  $$
  D = 2\Big[x\log\frac{x}{m} - (x-m)\Big],\qquad m=\mathbb E[x\mid x>0].
  $$
  This is the **Poisson** deviance: it assumes the variance equals the mean. NB explains large counts with its tail rather than its mean ($m$ is smaller), so deviance penalises it unfairly. Counts ≥ 20 (3% of positives) carry ~48% of NB's deviance.
- **Proteins:** NLL on the held-out entries $\mathcal M_P$, and Pearson correlation between $y$ and $\mu$.

---

## 9. Where things are in the code

| What | Where |
|---|---|
| $\eta$, distribution parameters | `Modality.params`, model.py:112 |
| Log-likelihoods (with a mask) | `Modality.log_prob`, model.py:134 |
| Fast sum without a mask | `Modality.log_lik_sum`, model.py:150 |
| $P(x>0)$, $\mathbb E[x]$, $\mathbb E[x\mid x>0]$ | model.py:177–200 |
| Batch effects | `BatchEffects`, model.py:36 |
| Sum over modalities with weight $w$ | `MultiSCENE.block_log_lik`, model.py:221 |
| Weight-decay groups | `MultiSCENE.param_groups`, model.py:209 |
| Training loop | `train.fit`, train.py:60–95 |
| Held-out data | `data.mask_rna`, `data.mask_dense` |
| Metrics | `train.evaluate_heldout`, train.py:138 |
| Fold-in | [scene_mm/foldin.py](../scene_mm/foldin.py) |
