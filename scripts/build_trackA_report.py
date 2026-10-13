"""Build the ML-track report: trackA_report.ipynb -> trackA_report.html (single file, code hidden).

Needs: results_tables/*.csv (scripts/collect_results.py, analyze_likelihood.py, analyze_proteins.py)
and results/pbmc_full_rna+prot_nb_d16_s0/embedding.h5ad for the joint-space picture.
Run from scene_project/:  python scripts/build_trackA_report.py
"""
import subprocess
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PROJ = Path(__file__).resolve().parents[1]
NB = PROJ / "trackA_report.ipynb"
HTML = PROJ / "trackA_report.html"
cells = []


def md(t):
    cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))


def code(t):
    cells.append(nbf.v4.new_code_cell(t.strip("\n")))


md(r"""
# Трек A: мультимодальная SCENE — результаты

**Проект:** *Joint cell–gene–protein embeddings for interpreting drug responses: a stress test of SCENE* (курс CBIO7215).
**Трек A (ML):**
1. Добавить в SCENE белки (CITE-seq) как третий тип точек.
2. Проверить другие правдоподобия для РНК.
3. Научиться вкладывать новых доноров без переобучения.
4. Сравнить с totalVI и scVI.

Код — `scene_mm/`, описание модели и **гипотезы, записанные до получения результатов**, — в `scene_mm/README.md`.

### Главное
- **Модель работает и совпадает с авторами:** RNA-only ZIP даёт те же AUC, PR-AUC и F1, что код авторов, до 3–4-го знака.
- **Белки встают рядом со своими генами:** медианный перцентиль своего гена 0.4–3% среди ~14–21 тыс. генов; при случайном расположении было бы 50%. Лучше всего встают маркеры линий (CD16 ↔ *FCGR3A*, CD20 ↔ *MS4A1*, CD127 ↔ *IL7R*). Хуже всего — белки без сигнала в крови. Доля сигнала в счёте белка объясняет это на обоих датасетах (Spearman ≈ −0.65).
- **Белки улучшают кластеризацию тонких типов клеток** умеренно, и **заметно улучшают перенос меток на новых доноров:** fold-in 0.83 → 0.90.
- **NB не хуже ZIP:** её «плохой» deviance — артефакт метрики, которая предполагает дисперсию, равную среднему. По отложенному правдоподобию, правильной метрике для разных семейств, NB лучше.
- **Против totalVI:** по точности сравнимо, но не лучше (0.90 против 0.91). Выигрыш SCENE в интерпретируемости: гены и белки в одном пространстве с клетками.
""")

code(r"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, anndata as ad
from scipy.stats import spearmanr
from IPython.display import display, Markdown
T = Path("results_tables")
full = pd.read_csv(T / "full.csv"); bfull = pd.read_csv(T / "baseline_full.csv"); lodo = pd.read_csv(T / "lodo.csv")
pairs = pd.read_csv(T / "protein_pairs.csv"); calib = pd.read_csv(T / "likelihood_calibration.csv")
theta = pd.read_csv(T / "likelihood_theta.csv"); psig = pd.read_csv(T / "protein_signal.csv")
LAB = {"pbmc": "celltype.l2", "neurips": "cell_type"}
LIKS = ["zip", "nb", "zip_decoupled"]; LIKN = {"zip": "ZIP", "nb": "NB", "zip_decoupled": "ZIP-decoupled"}
DSN = {"pbmc": "PBMC (30k клеток, 217 белков)", "neurips": "NeurIPS BMMC (30k клеток, 134 белка)"}
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
def ms(s): return f"{s.mean():.3f} ± {s.std():.3f}"
""")

md(r"""
## 1. Что построено

**Модель.** У каждой клетки одна точка $z_i$, общая для всех модальностей. У генов свои точки $w_j$, у белков — $v_k$. Сродство клетки и гена:
$$\eta_{ij} = \psi_i + \rho_j + \textstyle\sum_\ell g^{(\ell)}_{b_\ell(i),j} - \lVert z_i - w_j\rVert$$
Здесь $\psi_i$ и $\rho_j$ — числа: эффекты клетки и гена (в статье $\omega_j$); расстояние евклидово, без квадрата, как в коде авторов.

| Модальность | Правдоподобие |
|---|---|
| РНК `zip` | как у авторов: $P(x>0)=\sigma(\alpha\eta)$, положительные счёты — Пуассон без нуля с $\lambda = e^\eta$ |
| РНК `nb` | отрицательное биномиальное со средним $e^\eta$ и своей дисперсией $\theta_j$ на ген |
| РНК `zip_decoupled` | как `zip`, но у вероятности детекции свои эффекты клетки и гена |
| **Белки** `nb_bg` | NB со средним $\mu_{ik} = e^{\psi_i}\big(e^{\beta_k} + e^{\rho_k - \lVert z_i - v_k\rVert}\big)$: фон + сигнал |

Фон $\beta_k$ — это «эмбиентные» антитела, которые прилипают ко всем клеткам. Белок далеко от клетки остаётся на уровне фона.

**Новые доноры:**
- **агрегация** (без обучения): клетка = взвешенное среднее точек её генов и белков, вес = счёт / средний счёт признака;
- **fold-in**: всё, что относится к признакам, заморожено; учатся только новые клетки и строка батч-эффекта нового донора (200 эпох).

**Данные:**
- PBMC CITE-seq (Hao 2021): та же подвыборка 30k клеток, что в воспроизведении, 8 доноров, 30 тонких типов `celltype.l2`;
- NeurIPS 2021 BMMC CITE-seq: 30k случайных клеток, 45 типов, 12 батчей.

**Сетка:** 132 прогона на двух RTX 5000 Ada, ~7 ч.
- Полные прогоны: {РНК, РНК+белок} × 3 правдоподобия × {3D, 16D} × 2 датасета × 3 сида.
- LODO на PBMC: 8 доноров × 4 модели.
- scVI и totalVI.

**Проверка совпадения с авторами.** RNA-only ZIP, 2000 клеток PBMC, 16D, 600 эпох:

| | AUC | PR-AUC | F1 | deviance |
|---|---|---|---|---|
| код авторов, сиды 0/1 | 0.9236 / 0.9236 | 0.9194 / 0.9192 | 0.8477 / 0.8481 | 0.538 / 0.575 |
| `scene_mm`, сиды 0/1 | 0.9235 / 0.9235 | 0.9193 / 0.9192 | 0.8477 / 0.8479 | 0.583 / 0.546 |
""")

# ---------------------------------------------------------------- H1
md(r"""
## 2. H1 — Правдоподобие для РНК

*Гипотеза (до результатов):* NB даст лучшее отложенное лог-правдоподобие, AUC у всех будет похожий, а decoupled-вариант немного улучшит AUC.

Из обучения скрыто 10% ненулевых счётов, в ту же долю добавлено столько же настоящих нулей. Метрики:
- **NLL** — отрицательный лог-правдоподобие скрытых значений. Меньше = лучше. **Единственная метрика, честно сравнимая между семействами распределений.**
- **AUC** — отличает ли модель скрытые ненулевые от нулей по $P(x > 0)$.
- **Poisson deviance** — метрика авторов для величины счёта, по $E[x \mid x > 0]$.

Среднее ± стандартное отклонение по 3 сидам.
""")
code(r"""
t = full.assign(lik=full.likelihood.map(LIKN)).groupby(["dataset", "dim", "modality", "lik"]).agg(
    NLL=("rna_heldout_nll", ms), AUC=("rna_auc_roc", ms), deviance=("rna_poisson_deviance", ms))
t
""")
code(r"""
fig, axes = plt.subplots(2, 3, figsize=(11, 5.2))
for r, ds in enumerate(["pbmc", "neurips"]):
    d = full[(full.dataset == ds) & (full.dim == 16)]
    for c, (col, name) in enumerate([("rna_heldout_nll", "отложенный NLL (меньше = лучше)"), ("rna_auc_roc", "AUC детекции"),
                                     ("rna_poisson_deviance", "Poisson deviance (меньше = лучше)")]):
        ax = axes[r, c]
        lows, highs = [], []
        for k, mod in enumerate(["rna", "rna+prot"]):
            g = d[d.modality == mod].groupby("likelihood")[col]
            m_, s_ = np.array([g.mean()[l] for l in LIKS]), np.nan_to_num(np.array([g.std()[l] for l in LIKS]))
            ax.bar(np.arange(3) + (k - 0.5) * 0.35, m_, 0.35, yerr=s_,
                   label={"rna": "РНК", "rna+prot": "РНК + белок"}[mod], color=["#6c8ebf", "#d79b00"][k])
            lows.append((m_ - s_).min()); highs.append((m_ + s_).max())
        lo, hi = min(lows), max(highs); pad = 0.25 * (hi - lo) + 1e-4
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(range(3)); ax.set_xticklabels([LIKN[l] for l in LIKS]); ax.set_title(f"{ds.upper()}, 16D — {name}", fontsize=8)
axes[0, 0].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.show()
""")
md(r"""
**Итог по метрикам:**
- **NB лучше по NLL** на обоих датасетах: PBMC 1.173 против 1.190 у ZIP; NeurIPS 1.226 против 1.307.
- **AUC одинаковый** у всех трёх: ~0.926 и ~0.897.
- **Deviance у NB заметно хуже.**
- Decoupled не улучшил AUC. Эта часть гипотезы **не подтвердилась**.

### Почему у NB хуже deviance при лучшем NLL

Обученные модели (PBMC, только РНК, 16D, seed 0) пересчитаны точно на тех же скрытых значениях, и метрики разложены по величине наблюдаемого счёта:
""")
code(r"""
c = calib[calib.bin != "zeros (true negatives)"].copy()
show = c.pivot(index="bin", columns="likelihood", values=["n", "mean_obs", "mean_pred_given_pos", "mean_deviance", "share_of_total_deviance", "mean_nll"])
show = show.reindex(["1", "2", "3-4", "5-9", "10-19", "20+"])
show.columns = [f"{a} [{b}]" for a, b in show.columns]
display(show[["n [zip]", "mean_obs [zip]", "mean_pred_given_pos [zip]", "mean_pred_given_pos [nb]",
              "mean_deviance [zip]", "mean_deviance [nb]", "share_of_total_deviance [nb]", "mean_nll [zip]", "mean_nll [nb]"]].round(3))
fig, axes = plt.subplots(1, 3, figsize=(11, 3))
order = ["1", "2", "3-4", "5-9", "10-19", "20+"]
for lik, col in [("zip", "#6c8ebf"), ("nb", "#b85450")]:
    d = c[c.likelihood == lik].set_index("bin").reindex(order)
    axes[0].plot(order, d.mean_pred_given_pos / d.mean_obs, "o-", color=col, label=LIKN[lik])
    axes[1].plot(order, d.mean_deviance, "o-", color=col, label=LIKN[lik])
    axes[2].plot(order, d.mean_nll, "o-", color=col, label=LIKN[lik])
axes[0].axhline(1, color="grey", lw=0.8); axes[0].set_title("предсказано / наблюдено (1 = идеально)", fontsize=8)
axes[1].set_title("Poisson deviance по корзинам", fontsize=8); axes[2].set_title("NLL по корзинам", fontsize=8)
for a in axes: a.set_xlabel("наблюдаемый счёт")
axes[0].legend(frameon=False); plt.tight_layout(); plt.show()
display(Markdown(f"Дисперсия NB по генам: медиана $\\theta$ = {theta.theta.median():.1f}; у {100 * (theta.theta < 1).mean():.0f}% генов $\\theta < 1$ (сильная сверхдисперсия)."))
""")
md(r"""
**Вывод по H1**
- **На единицах** (62% всех скрытых ненулевых значений) NB лучше ZIP по обеим метрикам: NLL 1.87 против 1.93. На истинных нулях модели одинаковы (NLL 0.069).
- **На счётах 2–9** NB немного хуже по NLL (например, 2.56 против 2.50 на двойках), а начиная с 5 — и по deviance.
- Основной проигрыш NB по deviance — **большие счёты (≥ 20, 3.4% ненулевых), на них приходится ~48% всего её deviance.** Там NB занижает среднее (предсказывает ~39 при наблюдаемых 51; ZIP — 48). Большой счёт она объясняет **широким хвостом** распределения, а не поднятием среднего.
- Poisson deviance предполагает, что дисперсия равна среднему, поэтому сильно штрафует за заниженное среднее.
- Лог-правдоподобие учитывает хвост, и по нему **NB на тех же больших счётах лучше** (4.32 против 4.66).
- В сумме по всем скрытым значениям NB лучше по NLL за счёт самых частых единиц и больших счётов.

**Следствие для статьи и отчёта:** Poisson deviance авторов пригодна для сравнения внутри пуассоновского семейства, но не между ZIP и NB. **H1 подтверждена** для NB, **не подтверждена** для decoupled-варианта.
""")

# ---------------------------------------------------------------- H2
md(r"""
## 3. H2 — Помогают ли белки различать типы клеток

*Гипотеза:* белки улучшат кластеризацию тонких типов клеток (NMI и ARI).

KMeans с $k$ = числу типов на вложении клеток против разметки: PBMC `celltype.l2` (30 типов), NeurIPS `cell_type` (45 типов). Среднее ± стандартное отклонение по 3 сидам, 16D. Для сравнения — scVI и totalVI (16D, 3 сида).
""")
code(r"""
rows = []
for ds, lab in LAB.items():
    d = full[(full.dataset == ds) & (full.dim == 16)]
    for mod in ["rna", "rna+prot"]:
        for lik in LIKS:
            s = d[(d.modality == mod) & (d.likelihood == lik)]
            rows.append({"датасет": ds, "модель": f"SCENE {'РНК' if mod == 'rna' else 'РНК+белок'} {LIKN[lik]}",
                         "NMI": ms(s[f"{lab}_nmi"]), "ARI": ms(s[f"{lab}_ari"])})
    for m in ["scvi", "totalvi"]:
        s = bfull[(bfull.dataset == ds) & (bfull.model == m) & (bfull.label == lab)]
        rows.append({"датасет": ds, "модель": {"scvi": "scVI (РНК)", "totalvi": "totalVI (РНК+белок)"}[m], "NMI": ms(s.nmi), "ARI": ms(s.ari)})
pd.DataFrame(rows).set_index(["датасет", "модель"])
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
for ax, (ds, lab) in zip(axes, LAB.items()):
    d = full[(full.dataset == ds) & (full.dim == 16)]
    for k, mod in enumerate(["rna", "rna+prot"]):
        g = d[d.modality == mod].groupby("likelihood")[f"{lab}_nmi"]
        ax.bar(np.arange(3) + (k - 0.5) * 0.35, [g.mean()[l] for l in LIKS], 0.35, yerr=[g.std()[l] for l in LIKS],
               color=["#6c8ebf", "#d79b00"][k], label={"rna": "SCENE РНК", "rna+prot": "SCENE РНК + белок"}[mod])
    for j, (m, col) in enumerate([("scvi", "#82b366"), ("totalvi", "#9673a6")]):
        v = bfull[(bfull.dataset == ds) & (bfull.model == m) & (bfull.label == lab)].nmi.mean()
        ax.axhline(v, color=col, ls="--", lw=1.2, label={"scvi": "scVI", "totalvi": "totalVI"}[m])
    ax.set_xticks(range(3)); ax.set_xticklabels([LIKN[l] for l in LIKS]); ax.set_ylim(0.55, 0.73)
    ax.set_title(f"{ds.upper()}: NMI по тонким типам ({lab})", fontsize=8)
axes[0].legend(frameon=False, fontsize=7, loc="lower left"); plt.tight_layout(); plt.show()
""")
md(r"""
**Вывод по H2 — подтверждена, эффект умеренный.**
- Белки повышают NMI при всех трёх правдоподобиях на обоих датасетах. PBMC: ZIP 0.645 → 0.667, NB 0.617 → 0.679. NeurIPS: +0.01–0.02.
- Лучшая мультимодальная SCENE на уровне scVI и немного ниже totalVI.
- Больше всего белки помогают в PBMC — там 217 белков, и подтипы T- и NK-клеток определяются именно поверхностными маркерами.
""")

# ---------------------------------------------------------------- H3
md(r"""
## 4. H3 — Белок рядом со своим геном

*Гипотеза:* в общем пространстве белок ближе к своему гену, чем случайный ген.

Для каждого белка с известным геном (словарь ADT → ген, ~110 пар на датасет) берутся все гены и сортируются по расстоянию до белка. Метрика — **перцентиль своего гена** в этом списке: 0 = свой ген ближайший, 0.5 = как у случайного.
""")
code(r"""
s = full[full.modality == "rna+prot"].assign(lik=lambda d: d.likelihood.map(LIKN)).groupby(["dataset", "dim", "lik"]).agg(
    пар=("pg_n_pairs", "first"), медианный_перцентиль=("pg_median_rank", ms), доля_в_топ1=("pg_top1", ms), доля_в_топ5=("pg_top5", ms))
s
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
for ax, ds in zip(axes, ["pbmc", "neurips"]):
    for dim, ls in [(16, "-"), (3, ":")]:
        for lik, col in [("zip", "#6c8ebf"), ("nb", "#b85450")]:
            r = np.sort(pairs[(pairs.dataset == ds) & (pairs.dim == dim) & (pairs.likelihood == lik)].rank_pct)
            ax.plot(r, np.arange(1, len(r) + 1) / len(r), ls, color=col, label=f"{LIKN[lik]} {dim}D")
    xs = np.r_[np.linspace(0, 0.01, 200), np.logspace(-2, 0, 400)]; ax.plot(xs, xs, color="grey", lw=0.8, label="случайно")
    ax.set_xscale("symlog", linthresh=0.01); ax.set_xlim(0, 1)
    ax.set_xlabel("перцентиль своего гена (0 = ближайший)"); ax.set_ylabel("доля белков"); ax.set_title(DSN[ds], fontsize=8)
axes[0].legend(frameon=False, fontsize=7); plt.tight_layout(); plt.show()
""")
md(r"""
Кумулятивные кривые всех пар (все сиды). Кривая, прижатая к левому верхнему углу, значит, что почти все белки лежат у своего гена. Диагональ — случайное расположение.

**Какие пары встают лучше и хуже всего** (PBMC, NB, 16D, среднее по 3 сидам):
""")
code(r"""
q = pairs[(pairs.dataset == "pbmc") & (pairs.dim == 16) & (pairs.likelihood == "nb")].groupby(["protein", "gene"]).rank_pct.mean().sort_values()
best, worst = q.head(12).reset_index(), q.tail(12).reset_index()
display(pd.concat([best.rename(columns=lambda c: f"лучшие: {c}"), worst.rename(columns=lambda c: f"худшие: {c}")], axis=1).round(4))
""")
md(r"""
**Лучшие** — классические маркеры линий: CD16 (NK, моноциты), CD20 (B-клетки), CD127 (T-клетки), CD3, HLA-DR, CD123 (pDC). У них свой ген ближайший из ~20 000.

**Худшие** — белки, которых в покоящейся крови почти нет:
- маркеры активации: CTLA4, OX40/*TNFRSF4*, 4-1BB/*TNFRSF9*, GITR/*TNFRSF18*, PD-L1/L2;
- нетипичные для крови: HER2/*ERBB2*, MCAM, ENG.

### Почему: сигнал против фона
Для каждого белка по модели посчитана **доля сигнала** — средняя по клеткам доля предсказанного счёта, приходящаяся на слагаемое $e^{\rho_k - d}$, а не на фон $e^{\beta_k}$.
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, ds in zip(axes, ["pbmc", "neurips"]):
    a = psig[psig.dataset == ds].groupby(["protein", "gene"]).mean(numeric_only=True).reset_index()
    rho, p = spearmanr(a.fg_share, a.rank_pct)
    ax.scatter(a.fg_share, a.rank_pct, s=12, alpha=0.7)
    for _, r in a[(a.rank_pct < 0.0005) | (a.rank_pct > 0.45)].iterrows():
        ax.annotate(r.protein, (r.fg_share, r.rank_pct), fontsize=6)
    ax.set_yscale("symlog", linthresh=0.001); ax.set_xlabel("доля сигнала в счёте белка (1 − фон)")
    ax.set_ylabel("перцентиль своего гена"); ax.set_title(f"{ds.upper()}: Spearman ρ = {rho:+.2f} (p = {p:.0e})", fontsize=8)
plt.tight_layout(); plt.show()
out = []
for ds in ["pbmc", "neurips"]:
    a = psig[psig.dataset == ds].groupby(["protein", "gene"]).mean(numeric_only=True)
    out.append({"датасет": ds, **{f"ρ с {n}": round(spearmanr(a[c], a.rank_pct)[0], 2) for c, n in
                [("fg_share", "долей сигнала"), ("gene_mean", "экспрессией гена"), ("protein_mean", "средним счётом белка")]}})
display(pd.DataFrame(out).set_index("датасет"))
""")
md(r"""
**Вывод по H3 — подтверждена, и сильно.**
- Медианный перцентиль своего гена в 16D: **0.4–3%**.
- Чем большая доля счёта белка — сигнал, тем ближе он к своему гену. Связь одинаково сильная на обоих датасетах (ρ ≈ −0.65).
- Экспрессия гена объясняет это только на PBMC, а доля сигнала — на обоих.

Иначе говоря, модель **не ошибается** на части белков: белки без биологического сигнала в данных просто не к чему привязать. Модель честно относит их счёт к фону.

**В 3D** связь слабее: медиана 10–14% на PBMC, 2–7% на NeurIPS. Трёх измерений мало, чтобы разместить ~20 тыс. генов, 200 белков и десятки типов клеток без наложений.

### Как это выглядит в общем пространстве
PBMC, РНК + белок, NB, 16D. Для картинки 16-мерное пространство сжато в 2D методом UMAP, который сохраняет соседство точек. UMAP построен по клеткам (подвыборка 6000, цвет — `celltype.l1`), генам (подвыборка 3000, серые) и белкам-маркерам (чёрные) вместе. Красные — гены этих белков; линия соединяет пару. Под картинкой — точный ранг своего гена в исходном 16-мерном пространстве, без искажений UMAP.
""")
code(r"""
emb = ad.read_h5ad("results/pbmc_full_rna+prot_nb_d16_s0/embedding.h5ad")
import re, umap
Z = emb.obsm["X_scene"]; W = emb.varm["W_rna"]; P = emb.uns["protein_positions"]
V = P[[f"w{i}" for i in range(Z.shape[1])]].to_numpy()
show_pairs = [("CD3", "CD3E"), ("CD4", "CD4"), ("CD8", "CD8A"), ("CD19", "CD19"), ("CD20", "MS4A1"), ("CD14", "CD14"),
              ("CD16", "FCGR3A"), ("CD56", "NCAM1"), ("CD127", "IL7R"), ("HLA-DR", "HLA-DRA"), ("CD123", "IL3RA"), ("CD1c", "CD1C")]
genes = list(emb.var_names); pnames = [re.sub(r"-\d+$", "", str(n)) for n in P.index]
show_pairs = [(p_, g) for p_, g in show_pairs if p_ in pnames and g in genes]
rng = np.random.default_rng(0)
ci = rng.choice(len(Z), 6000, replace=False)
gi = np.unique(np.r_[rng.choice(len(W), 3000, replace=False), [genes.index(g) for _, g in show_pairs]])
pi_ = [pnames.index(p_) for p_, _ in show_pairs]
X = np.vstack([Z[ci], W[gi], V[pi_]])
E = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(X)
ec, eg, ep = E[:len(ci)], E[len(ci):len(ci) + len(gi)], E[len(ci) + len(gi):]
lab = emb.obs["celltype.l1"].astype(str).to_numpy()[ci]
fig, ax = plt.subplots(figsize=(8.5, 6.5))
ax.scatter(eg[:, 0], eg[:, 1], s=1, color="lightgrey", alpha=0.5, label="гены (подвыборка)")
for i, l in enumerate(sorted(set(lab))):
    m = lab == l; ax.scatter(ec[m, 0], ec[m, 1], s=2, alpha=0.4, label=l, color=plt.cm.tab10(i))
pair_rank = {}
for k, (p_, g) in enumerate(show_pairs):
    gv = eg[list(gi).index(genes.index(g))]; pv = ep[k]
    ax.plot([pv[0], gv[0]], [pv[1], gv[1]], color="k", lw=0.7)
    ax.scatter(*pv, color="k", s=22, zorder=5); ax.scatter(*gv, color="red", s=16, zorder=5)
    ax.annotate(p_, pv, fontsize=7, fontweight="bold"); ax.annotate(g, gv, fontsize=6, color="red", xytext=(3, -8), textcoords="offset points")
    d = np.linalg.norm(W - V[pnames.index(p_)], axis=1); pair_rank[f"{p_} → {g}"] = int((d < d[genes.index(g)]).sum()) + 1
ax.legend(frameon=False, fontsize=7, markerscale=4, loc="upper right"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("UMAP общего пространства: клетки, гены (серые), белки (чёрные) и их гены (красные)", fontsize=9); plt.show()
display(Markdown("**Место своего гена среди всех ~20 700 генов по расстоянию до белка (в 16D, 1 = ближайший):** " +
                 ", ".join(f"{k}: **{v}**" for k, v in pair_rank.items())))

""")
md(r"""
Как читать картинку:
- **Белки** (чёрные) встают **внутрь «своих» кластеров клеток**: CD19, CD20 и CD1c у B-клеток, CD14 у моноцитов, CD16 и CD56 у NK, CD3, CD4 и CD127 у T-клеток.
- **Их гены** (красные) остаются в облаке генов, на его краю, обращённом к тому же типу клеток.

Значит, белок ближе к клеткам, чем к генам. Это ожидаемо: у белков плотные и большие счёты, и они сильно определяют положение клетки. При этом **среди генов** свой ген — один из ближайших к белку:
- у CD16, CD127 и CD20 — первый или второй из ~20 700;
- у HLA-DR, CD123, CD3 — в первой десятке;
- у CD4, CD8 и CD56 — хуже (ранг 250–1100, то есть топ-1.5–5%).

UMAP передаёт соседство приблизительно, поэтому длина линий — лишь иллюстрация. Точная проверка — ранги, посчитанные в 16D.
""")

# ---------------------------------------------------------------- H4 / H5
md(r"""
## 5. H4 и H5 — Новые доноры (leave-one-donor-out, PBMC)

*Гипотезы:*
- **H4:** fold-in переносит метки на нового донора почти так же хорошо, как референс, и лучше агрегации;
- **H5:** totalVI — сильный бейзлайн, мультимодальная SCENE будет сравнимой, но не лучше.

Для каждого из 8 доноров модель обучается на остальных семи (донор — уровень батча). Затем клетки отложенного донора вкладываются одним из способов, и типы клеток переносятся с референса kNN (k = 15). Для scVI и totalVI используется стандартный путь scvi-tools для новых данных — **scArches**: веса референса заморожены, дообучаются только параметры новых данных. Среднее ± стандартное отклонение по 8 донорам.
""")
code(r"""
def lodo_table(label):
    d = lodo[lodo.label == label].copy()
    d["способ"] = d.embedding.map({"aggregate": "агрегация", "foldin": "fold-in", "scArches": "scArches"})
    d["модель"] = d.model.str.replace("SCENE rna+prot", "SCENE РНК+белок").str.replace("SCENE rna", "SCENE РНК") \
        .str.replace(" zip", " ZIP").str.replace(" nb", " NB").replace({"scvi": "scVI (РНК)", "totalvi": "totalVI (РНК+белок)"})
    return d.groupby(["модель", "способ"]).agg(точность=("accuracy", ms), macro_F1=("macro_f1", ms), доноров=("accuracy", "size"))
display(Markdown("**Тонкие типы (`celltype.l2`, 30 типов)**")); display(lodo_table("celltype.l2"))
display(Markdown("**Крупные типы (`celltype.l1`, 8 типов)**")); display(lodo_table("celltype.l1"))
""")
code(r"""
d = lodo[lodo.label == "celltype.l2"].copy()
d["key"] = d.model + " / " + d.embedding
keys = ["SCENE rna zip / aggregate", "SCENE rna zip / foldin", "SCENE rna nb / foldin", "SCENE rna+prot zip / foldin",
        "SCENE rna+prot nb / aggregate", "SCENE rna+prot nb / foldin", "scvi / scArches", "totalvi / scArches"]
names = ["SCENE РНК ZIP, агрегация", "SCENE РНК ZIP, fold-in", "SCENE РНК NB, fold-in", "SCENE РНК+белок ZIP, fold-in",
         "SCENE РНК+белок NB, агрегация", "SCENE РНК+белок NB, fold-in", "scVI, scArches", "totalVI, scArches"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, metric, title in [(axes[0], "accuracy", "точность"), (axes[1], "macro_f1", "macro-F1")]:
    for i, k in enumerate(keys):
        v = d[d.key == k].sort_values("donor")[metric].to_numpy()
        ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=12, alpha=0.7)
        ax.hlines(v.mean(), i - 0.3, i + 0.3, color="k", lw=1.5)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax.set_title(f"Перенос тонких типов на отложенного донора — {title}", fontsize=8)
plt.tight_layout(); plt.show()
""")
md(r"""
Точки — 8 отложенных доноров, чёрная черта — среднее.

**Вывод по H4 — подтверждена.**
- Fold-in намного лучше агрегации: на тонких типах 0.83–0.90 против 0.48–0.53.
- **Белки заметно помогают fold-in:** 0.83 → **0.90** по точности, 0.63 → **0.73** по macro-F1.
- Агрегация без обучения на 30 тонких типах слабая. Усреднение генов «стягивает» клетку к центру пространства, и тонкие различия теряются. Для 8 крупных типов она работает лучше.
- Fold-in одного донора (~4 тыс. клеток, 200 эпох) занимает секунды.

**Вывод по H5 — подтверждена.**
- totalVI остаётся лучшим: 0.914 / 0.785. Мультимодальная SCENE близко: 0.896 / 0.726. Обе обходят scVI (0.879 / 0.712).
- SCENE на уровне лучшего метода по точности и слегка уступает по macro-F1, то есть на редких типах.
- Её преимущество — **интерпретируемость**: гены и белки лежат в одном пространстве с клетками (раздел 4). У totalVI и scVI такого нет.
""")

# ---------------------------------------------------------------- 3D vs 16D, summary
md(r"""
## 6. 3D против 16D
""")
code(r"""
rows = []
for ds, lab in LAB.items():
    for dim in [3, 16]:
        d = full[(full.dataset == ds) & (full.dim == dim) & (full.likelihood == "nb")]
        rows.append({"датасет": ds, "размерность": f"{dim}D",
                     "NMI (РНК)": d[d.modality == "rna"][f"{lab}_nmi"].mean(), "NMI (РНК+белок)": d[d.modality == "rna+prot"][f"{lab}_nmi"].mean(),
                     "AUC": d.rna_auc_roc.mean(), "NLL": d.rna_heldout_nll.mean(),
                     "белок↔ген: медиана": d[d.modality == "rna+prot"].pg_median_rank.mean(),
                     "белок↔ген: в топ-5%": d[d.modality == "rna+prot"].pg_top5.mean()})
pd.DataFrame(rows).set_index(["датасет", "размерность"]).round(3)
""")
md(r"""
В 3D всё хуже, но не катастрофически:
- AUC теряет ~0.005;
- NMI теряет 0.05–0.10;
- сильнее всего страдает соответствие «белок ↔ ген».

Это согласуется с воспроизведением: для предсказания детекции хватает очень малой размерности, а тонкая структура (типы, пары белок–ген) требует 8–16 измерений. **3D хорош для картинок, 16D — для анализа.**

## 7. Итог по гипотезам

| | Гипотеза (записана до результатов) | Результат |
|---|---|---|
| H1 | NB лучше ZIP по отложенному правдоподобию; AUC похож; decoupled чуть лучше по AUC | ✅ NB лучше по NLL, AUC одинаковый. ❌ decoupled ничего не дал. Худший deviance у NB — артефакт метрики, а не модели |
| H2 | белки улучшают кластеризацию тонких типов | ✅ умеренно (+0.01–0.06 NMI), на всех правдоподобиях и обоих датасетах |
| H3 | белок ближе к своему гену, чем случайный ген | ✅ сильно: медиана 0.4–3%. Неудачи объясняются отсутствием сигнала (ρ ≈ −0.65) |
| H4 | fold-in лучше агрегации и близок к референсу | ✅ 0.90 против 0.48 (тонкие типы); белки помогают fold-in (+0.07) |
| H5 | totalVI сильный; SCENE сравнима, но не лучше | ✅ 0.896 против 0.914 по точности; SCENE лучше scVI |

## 8. Ограничения
- **Подвыборки:** 30k клеток на датасет, LODO только на PBMC, с одним сидом на донора.
- **Кластеризация через KMeans** с $k$ = числу типов — грубая метрика: на неё влияет дисбаланс типов.
- **Бейзлайны** (scVI, totalVI) с настройками по умолчанию (16D, 400 эпох), без подбора гиперпараметров. Для SCENE мы тоже ничего не подбирали, взяли настройки авторов.
- **Словарь ADT → ген** составлен вручную (~110 однозначных пар на датасет). Белки без однозначного гена, например изотипические контроли и TCR-сегменты, в тест H3 не входят.
- **Вес белков** в правдоподобии = 1 (как у РНК). Мы его не подбирали, а он влияет на баланс модальностей.

## 9. Как воспроизвести
```
python scripts/prepare_data.py all                   # данные
python scripts/make_grid.py cuda                     # сетка из 132 прогонов
sbatch --export=ALL,WORKER=0 scripts/run_grid.sbatch # и WORKER=1 (две станции ws-ia)
python scripts/collect_results.py                    # таблицы
python scripts/analyze_likelihood.py mps             # калибровка ZIP против NB
python scripts/analyze_proteins.py mps               # сигнал белков
python scripts/build_trackA_report.py                # этот отчёт
```
Все эмбеддинги лежат в `results/<run_id>/embedding.h5ad`: клетки в `obsm["X_scene"]`, гены в `varm["W_rna"]`, белки в `uns["protein_positions"]`. Трек B может использовать их напрямую.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(PROJ)}}).execute()
nbf.write(nb, NB)
subprocess.run([str(Path(sys.executable).parent / "jupyter"), "nbconvert", "--to", "html", "--no-input", "--output", str(HTML), str(NB)], check=True)
print(f"wrote {NB.name}, {HTML.name} ({HTML.stat().st_size / 1e6:.1f} MB)")
