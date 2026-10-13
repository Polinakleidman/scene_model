"""Build SCENE_reproduction_report.ipynb: paper summary, architecture, and our reproduced figures.

Run from reproduction/:  .venv/bin/python build_report.py
The notebook is executed, so all figures and tables are embedded and it opens anywhere.
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "SCENE_reproduction_report.ipynb"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


def fig(*paths, width=None):
    """Show one figure, or several side by side (paths relative to the figure output folder)."""
    args = ", ".join(repr(p) for p in paths)
    code(f"row({args}{'' if width is None else f', width={width}'})")


# ---------------------------------------------------------------------------------------------
md(r"""
# SCENE: пересказ статьи и наше воспроизведение

**Статья:** Møberg et al., *Setting the SCENE for Interpretable Cell–Gene Embeddings in Single-Cell RNA-seq*, bioRxiv (2026). [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.09.12.750699v1) · код: [SCENE](https://github.com/oscarmoeberg/SCENE), [SCENE-reproducibility](https://github.com/oscarmoeberg/SCENE-reproducibility)

**Что в ноутбуке**
1. О чём статья
2. Архитектура модели
3. Данные
4. Как мы воспроизводили и чем это отличается от статьи
5. Результаты по рисункам: что показано и совпадает ли со статьёй
6. Сводка «утверждение статьи → наш результат»
7. Что не воспроизвели

Все рисунки ниже построены **нашим прогоном** с помощью ноутбуков авторов. Картинки из самой статьи здесь не используются.
""")

code(r"""
import base64
from pathlib import Path

import pandas as pd
from IPython.display import HTML, display

FIGS = Path("SCENE-reproducibility/notebooks/figures/output")
pd.set_option("display.precision", 3)


def _uri(path):
    p = FIGS / path
    mime = "image/svg+xml" if p.suffix == ".svg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def row(*paths, width=None):
    # Side-by-side figures, embedded so the notebook is self-contained.
    # Legends get a narrow slot; the other panels share the rest of the row.
    legends = [p for p in paths if "legend" in p]
    n_main = max(len(paths) - len(legends), 1)
    main_w = width or ((96 - 12 * len(legends)) // n_main)
    imgs = "".join(
        f'<figure style="display:inline-block;width:{12 if "legend" in p else main_w}%;margin:0 0.5%;vertical-align:top">'
        f'<img src="{_uri(p)}" style="width:100%;background:white"/>'
        f'<figcaption style="font-size:10px;color:#777">{p}</figcaption></figure>'
        for p in paths
    )
    display(HTML(f"<div>{imgs}</div>"))
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 1. О чём статья

**Задача.** Данные scRNA-seq — это матрица счётов «клетки × гены»: много нулей, большой шум, десятки тысяч генов. Стандартный пайплайн выглядит так: нормализация → логарифм → отбор ~2000 вариабельных генов (HVG) → PCA → кластеризация/UMAP. Каждый шаг необратимо что-то выбрасывает. К тому же на выходе получается только представление **клеток**, а гены приходится интерпретировать отдельно.

**Идея SCENE** (*Single-Cell Euclidean Network Embedding*):
- матрица счётов рассматривается как **двудольный взвешенный граф**: клетки с одной стороны, гены с другой, вес ребра — число UMI;
- клетки и гены вкладываются **в одно евклидово пространство**. Чем ближе ген к клетке, тем выше вероятность, что он в ней экспрессирован, и тем больше счёт;
- модель обучается прямо на **сырых счётах всех генов**, без нормализации и отбора HVG, с явным вероятностным правдоподобием.

**Что это даёт.** Расстояния «клетка–клетка», «клетка–ген» и «ген–ген» имеют смысл внутри одного пространства. Маркеры лежат рядом со своими типами клеток, гены общей программы (например, ответа на лекарство) собираются в компактные группы. По утверждению авторов, это работает даже в **нативных 2D/3D**, без UMAP.

**Главные утверждения статьи**
1. Качество кластеризации и батч-интеграции на уровне или лучше Harmony, scVI и SIMBA. Особенно на симуляции с вложенными батчами (SIM2), где scVI «схлопывает» группы.
2. Осмысленные вложения даже в 2D/3D. Предсказание связей насыщается уже при малой размерности.
3. Геометрия генов отражает биологию: модули ответа на глюкокортикоиды (GR) и TCR-стимуляцию компактнее, чем у SIMBA.
4. Импутация пропущенных счётов лучше, чем у scVI.
""")

md(r"""
## 2. Архитектура

SCENE — это не нейросеть с энкодером, а **латентная модель расстояний** (*latent distance model*, Hoff et al. 2002). У каждой клетки $i$ есть точка $z_i \in \mathbb{R}^d$, у каждого гена $j$ — точка $w_j \in \mathbb{R}^d$. Все параметры — «свободные» таблицы, которые оптимизируются напрямую.

**Правдоподобие (zero-inflated / hurdle Poisson).** Отдельно моделируется, *детектирован ли* ген, и *сколько* его, если детектирован:

$$\eta_{ij} = \psi_i + \rho_j + \sum_{\ell} g^{(\ell)}_{b_\ell(i),\,j} \;-\; \lVert z_i - w_j \rVert$$

(расстояние — обычная евклидова норма, **без квадрата**: в коде авторов `torch.cdist(p=2)`)

$$\pi_{ij} = P(x_{ij} > 0) = \sigma(\alpha\,\eta_{ij}), \qquad x_{ij} \mid x_{ij} > 0 \;\sim\; \text{Poisson}_{>0}\big(\lambda_{ij}\big),\quad \lambda_{ij} = e^{\eta_{ij}}$$

| Компонент | Что означает |
|---|---|
| $\lVert z_i - w_j\rVert$ | «сродство» клетки и гена: чем ближе, тем выше и вероятность детекции, и ожидаемый счёт |
| $\psi_i$ | эффект клетки: глубина секвенирования, размер библиотеки |
| $\rho_j$ | эффект гена: общий уровень экспрессии (в статье $\omega_j$; переименован, чтобы не путать с координатой гена $w_j$) |
| $g^{(\ell)}_{b,j}$ | поправка на батч $b$ для гена $j$ в слое батчей $\ell$ (донор, лаборатория…). Полная $\Gamma \in \mathbb{R}^{B\times G}$ или низкоранговая $UV^\top$ |
| $\alpha$ | обучаемый масштаб, связывающий расстояние с вероятностью детекции |

**Обучение**
- Максимум правдоподобия по **всей** матрице $N \times G$, стоимость $O(NG)$ за эпоху.
- AdamW, lr = 0.05, weight decay $10^{-3}$ на $z$ и $w$, 600 эпох.
- Инициализация — спектральное вложение (лапласиан) двудольного графа.

**Импутация.** Ожидаемый счёт $E[x_{ij}] = \pi_{ij}\,\lambda_{ij}/(1 - e^{-\lambda_{ij}})$.

**Чем отличается от соседей**

| Метод | Сырые счёты | Совместное пространство клеток и генов | Явное правдоподобие | Энкодер |
|---|---|---|---|---|
| PCA / Harmony | нет (нормализация + HVG) | нет | нет | нет |
| scVI | да | нет (только клетки) | да (ZINB) | да, VAE |
| SIMBA | нет (дискретизация) | да (граф клетки–гены–…) | нет (граф-эмбеддинг PyTorch-BigGraph) | нет |
| **SCENE** | **да, все гены** | **да** | **да (ZIP)** | **нет** — поэтому новые клетки нельзя вложить без переобучения |

Ниже схема из рис. 1 статьи, построенная ноутбуком авторов: матрица счётов → двудольный граф → общее пространство.
""")
fig("fig_1/panel_a_matrix_thinbars.svg", "fig_1/panel_b_bipartite.svg", "fig_1/panel_c_embedding.svg")

# ---------------------------------------------------------------------------------------------
md(r"""
## 3. Данные

| Датасет | Размер (клетки × гены) | Зачем в статье |
|---|---|---|
| Cortex (мышь, Zeisel) | 3 005 × 19 972 | бенчмарк кластеризации, перебор размерности |
| PBMC CITE-seq (Hao 2021) | 152 094 × 20 729 (у нас подвыборка 30 000) | кластеризация (рис. 2), бенчмарк, импутация |
| IFN-β, Kang 2018 | 24 673 × 15 706 | ответ на интерферон (рис. 3) |
| HCA Nuclei (сердце) | 13 733 × 26 662 | батч-интеграция Harvard/Sanger (рис. 4) |
| SIM1 / SIM2 (scIB) | 12 097 / 19 318 клеток | симуляции батчей; SIM2 — вложенные батчи |
| NeurIPS 2021 BMMC | 90 261 × 13 953; подвыборка стволовых клеток 39 860 | интеграция; ветвление эритроидной и моноцитарной линий (рис. 5) |
| GR time course (дексаметазон) | 6 × 400 клеток | модули генов ответа на глюкокортикоиды (рис. 6) |
| TCR stimulation | 9 492 × 17 840 | регулоны TF при активации T-клеток (рис. 6) |
""")

md(r"""
## 4. Как мы воспроизводили

- **Код:** скрипты авторов (`bash/paper/*.sh`, 350 команд, включая SIMBA и TCR), запущенные нашей очередью `run_queue.py`. Все выполнены, все 10 ноутбуков с рисунками авторов отработали без ошибок.
- **Где считалось:**

| Часть | Где |
|---|---|
| 74 команды: GR, cortex, Kang, SIM, HCA, полный NeurIPS… | MacBook M5 Pro (MPS) |
| весь scVI, PCA/Harmony и оценки для PBMC | кластер, общий GPU-узел (RTX 5000 Ada) |
| остальное: перебор размерностей, SCENE на PBMC, импутация, оценки | кластер, выделенные рабочие станции (RTX 5000 Ada, 32 ГБ) |
| SIMBA (39 обучений, CPU) и TCR | кластер, выделенные рабочие станции, две параллельно |

Все итоговые обучения SCENE — **на всей матрице, как в статье**. Исключение — полный NeurIPS (90k клеток): мини-батчи по 8192 клеток, иначе не хватает памяти. Пробные прогоны на общем GPU с мини-батчами потом пересчитаны на выделенных станциях.

**Отличия от статьи** (подробно в `README.md`):
1. **PBMC — подвыборка 30 000 клеток** из 152 094. Полная матрица требует ~130 ГБ памяти при обучении на всей матрице.
2. **Перебор размерностей (рис. 5e) — 2 сида** (0, 1) вместо 5.
3. **SIMBA и TCR — технические обходы:**
   - `dgl==2.2.0` поставлен с сервера DGL: на PyPI этой версии нет. torch 2.3.0 в CPU-сборке, SIMBA обучается на CPU. Остальные версии как в `environment_simba.yaml`.
   - Конвертация TCR из Seurat сделана напрямую (`prepare_tcr_fix.R`): `as.SingleCellExperiment` не работает с текущим SeuratObject. Размер данных совпал со справкой авторов (9 492 × 17 840).
   - Оценку регулонов для SCENE-TCR мы считали в 12 потоков. При `permutation_rng=independent` у каждого модуля свой генератор, поэтому результат от числа потоков не зависит.
4. MPS/CUDA вместо CPU авторов — мелкие численные отличия ожидаемы.
5. В `imputation.py` добавлены необязательные мини-батчи через переменные окружения. В итоговом прогоне импутации они не использовались: он шёл на всей матрице.
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 5. Результаты

### Рис. 2 — Кластеры в совместном пространстве (PBMC CITE-seq)

**Что показано.** Клетки и гены из пространства SCENE (16D) кластеризованы методом Уорда вместе. Затем кластеры сравниваются с разметкой типов клеток.
- **Дендрограмма:** иерархия кластеров клеток и генов.
- **UMAP, лапласиан против обученной модели:** уже спектральная инициализация даёт грубую структуру, обучение её уточняет.
- **Тепловая карта:** доля каждого типа клеток в каждом кластере; гены — отдельный «тип».

**Совпадает ли со статьёй.** В статье NMI (кластеры против типов, без генов) = **0.63**. У нас **0.660** (на подвыборке 30k). Основная путаница, как и у авторов, внутри близких T-клеток и моноцитов, а не между далёкими линиями.
""")
fig("fig_2/pbmc_cite_seq_dendrogram.svg", width=98)
fig("fig_2/pbmc_cite_seq_umap_laplacian_celltype.svg", "fig_2/pbmc_cite_seq_umap_latent_celltype.svg", "fig_2/pbmc_cite_seq_umap_celltype_legend.svg")
fig("fig_2/pbmc_cite_seq_ward_x_celltype_heatmap_with_gene.svg", width=45)

md(r"""
### Рис. 3 — Ответ на интерферон-β (Kang 2018)

**Что показано.** PBMC до и после стимуляции IFN-β (6 ч) в одном пространстве SCENE.
- **UMAP по типам клеток и по условию** (контроль/стимуляция): тип клетки и стимуляция — две оси вариации одновременно.
- **UMAP, раскрашенный расстоянием до гена** (MX1, ISG15, IFIT1, CXCL10, OAS1, RSAD2 — классические гены ответа на интерферон): клетки ближе всего к этим генам там, где находятся стимулированные клетки. Ген «помечает» свою область пространства.
- **π от расстояния для маркеров** (CD74, FCER1A…): вероятность детекции падает с расстоянием, причём быстрее для «своих» типов клеток.

**Совпадает ли со статьёй.** Качественно да: IFN-гены локализуются у стимулированных клеток, маркеры — у своих типов.
""")
fig("fig_3/kang_umap_cell_type.svg", "fig_3/kang_umap_label.svg", "fig_3/celltype_legend.svg")
fig("fig_3/kang_umap_latent_scLDM_dist_MX1.svg", "fig_3/kang_umap_latent_scLDM_dist_ISG15.svg", "fig_3/kang_umap_latent_scLDM_dist_IFIT1.svg", "fig_3/kang_umap_latent_scLDM_dist_CXCL10.svg")
code(r"""pis = sorted((FIGS / "fig_3/pi_vs_dist").glob("*.svg"))[:4]
row(*[str(p.relative_to(FIGS)) for p in pis])""")

md(r"""
### Рис. 4 — Бенчмарк: кластеризация и батч-интеграция

**Что показано.** Метрики scIB, среднее по 5 сидам:
- **сохранение биологии:** KMeans NMI и ARI с разметкой, силуэт по меткам;
- **удаление батча:** силуэт по батчу, связность графа (graph connectivity).

Везде «больше — лучше». SCENE подписан как `scLDM` — так называется модель в коде авторов.
""")
code(r"""
bench = pd.read_csv(FIGS / "fig_4/benchmark_summary_combined.csv")
cols = {"KMeans NMI_mean": "NMI", "KMeans ARI_mean": "ARI", "Silhouette label_mean": "Sil. label",
        "Silhouette batch_mean": "Sil. batch", "Graph connectivity_mean": "Graph conn."}
table = bench.set_index(["dataset", "display_name"])[list(cols)].rename(columns=cols)
table.index.names = ["датасет", "метод"]
table.round(3)
""")
fig("fig_4/nmi_grouped_bar.svg", "fig_4/ari_grouped_bar.svg", "fig_4/legend_methods.svg")
fig("fig_4/sil_labels_grouped_bar.svg", "fig_4/sil_batch_grouped_bar.svg", "fig_4/graph_conn_grouped_bar.svg")
md(r"""
**SIM2 — вложенные батчи.** Это главный аргумент статьи: scVI (с учётом батча и подбатча) смешивает 4 биологические группы, а SCENE их разделяет. Слева направо: scVI по группам, scVI по батчам, SCENE по группам, SCENE по батчам.
""")
fig("fig_4/sim2_umap_scvi_batch_subbatch_Group.svg", "fig_4/sim2_umap_scvi_batch_subbatch_Batch.svg",
    "fig_4/sim2_umap_scldm_batch_subbatch_Group.svg", "fig_4/sim2_umap_scldm_batch_subbatch_Batch.svg")
md(r"""
**HCA Nuclei (сердце).** Две лаборатории (Harvard/Sanger). SCENE с учётом батчей перемешивает источники и сохраняет типы клеток. Дальше — **нативное 2D-пространство без UMAP**: клетки и маркерные гены сердца в одних координатах.
""")
fig("fig_4/hca_umap_scldm_cell_source.svg", "fig_4/hca_umap_scldm_batch_cell_source.svg", "fig_4/hca_umap_scldm_batch_cell_type.svg")
fig("fig_4/hca_joint_2D_latent_cell_type.png", width=70)
md(r"""
**Совпадает ли со статьёй — в целом да, но картина тоньше, чем «SCENE лучше всех».**
- **Cortex, SIM1, SIM2:** SCENE явно лучший.
- **SIM2 (вложенные батчи)** — главный аргумент статьи, и он воспроизводится сильно: NMI SCENE 0.99, scVI 0.29, SIMBA и Harmony 0.05. У SIMBA и Harmony NMI и ARI совпали до последнего знака: KMeans дал одинаковое разбиение (скорее всего по батчам), хотя сами эмбеддинги разные — силуэт и связность у них не совпадают.
- **HCA, NeurIPS, PBMC:** SIMBA — сильный конкурент и по NMI чуть впереди SCENE (HCA 0.853 против 0.849, NeurIPS 0.689 против 0.673, PBMC 0.663 против 0.643). Разница небольшая, но формулировка «competitive or superior» здесь скорее «competitive».
- **PBMC:** лучший Harmony (0.68). У нас подвыборка 30k вместо 152k, это может влиять.
""")

md(r"""
### Рис. 5 — Ультранизкая размерность

**Что показано.**
- **Нативные 2D/3D-вложения** стволовых клеток NeurIPS: ветки эритроидной и моноцитарной линий от HSC видны **без UMAP**. Раскраска — относительная близость клетки к генам-концам линий: HBB (эритроциты) и FCGR3A (моноциты).
- **Зависимость от размерности $d \in \{2,3,8,16,32,64\}$:** из обучения скрыто 10 % ненулевых счётов, затем измеряется, как модель их восстанавливает.
  - *Link prediction* (AUC-ROC, PR-AUC, max F1): угадывает ли модель, что ген детектирован.
  - *Poisson deviance*: насколько точно предсказан сам счёт, меньше — лучше.

(В именах файлов ниже стоит `kang_` — это префикс из ноутбука авторов, данные — стволовые клетки NeurIPS.)
""")
fig("fig_5/kang_scLDM_2d_cell_type.svg", "fig_5/kang_scLDM_3d_cell_type_angle1.svg", "fig_5/kang_scLDM_3d_double_score_rawdiff_FCGR3A_vs_HBB_angle1.svg", "fig_5/celltype_legend.svg")
fig("fig_5/auc_roc_vs_dimension.svg", "fig_5/micro_f1_vs_dimension.svg", "fig_5/poisson_deviance_vs_dimension.svg", "fig_5/legend_datasets.svg")
code(r"""
dims = pd.read_csv(FIGS / "fig_5/metric_summary_by_dataset_dim.csv")
dims.pivot(index="Dataset", columns="dim", values="Poisson Deviance_mean").round(3)
""")
md(r"""
**Совпадает ли со статьёй — да, по всем трём пунктам.**
1. Link prediction **насыщается сразу**: AUC почти не растёт после 2–3D (0.90 → 0.91).
2. Deviance продолжает **улучшаться с размерностью** на cortex и PBMC.
3. На стволовых клетках **после 8D становится хуже** (0.79 при 8D → 0.95 при 64D). Ровно это и пишут авторы.

Оговорка: у нас 2 сида вместо 5.
""")

md(r"""
### Рис. 6 — Модули генов ответа на глюкокортикоиды (GR)

**Что показано.** Клетки рака груди T47D обработаны дексаметазоном: контроль, 1, 2, 4, 8, 18 ч. На каждой временной точке обучается 3D-SCENE, и проверяется, насколько **компактны** в пространстве генов 4 заранее заданных модуля:
- прямые мишени GR (GRE);
- поздние TF;
- гликолиз;
- цикл Кребса (TCA).

Компактность = (среднее расстояние внутри модуля) / (среднее до остальных генов). Значимость оценивается против 10 000 случайных модулей того же размера; q — после поправки Бенджамини–Хохберга.

**Ожидание:**
- прямые мишени GR становятся компактнее со временем стимуляции;
- метаболические модули компактны всегда.

Ниже:
- тепловая карта −log10(q): априорное ожидание, SIMBA и SCENE;
- нативное 2D-пространство SCENE в контроле и через 18 ч с выделенными генами модуля GR Direct.
""")
fig("fig_6/gr_heatmap.svg", width=95)
code(r"""
mods = ["GR_direct_GRE_core", "GR_secondary_TF_core", "Glycolysis_core", "TCA_cycle_core"]
res = Path("SCENE-reproducibility/results/geometric_biology")
rows = {}
for t in ["00", "01", "02", "04", "08", "18"]:
    s = pd.read_csv(res / f"scLDM_GR_{t}_3D/gene_module_eval.csv", index_col="module")["qval_fdr_bh"]
    b = pd.read_csv(res / f"SIMBA_GR_{t}/gene_module_eval.csv", index_col="module")["qval_fdr_bh"]
    rows["control" if t == "00" else f"{int(t)} h"] = {m: f"{s[m]:.1e} / {b[m]:.1e}" for m in mods}
print("q-value: SCENE / SIMBA  (q < 0.05 = module is compact)")
pd.DataFrame(rows).T
""")
fig("fig_6/kang_scLDM_2d_time_hr_adata_00.svg", "fig_6/kang_scLDM_2d_time_hr_adata_18.svg")
md(r"""
**Совпадает ли со статьёй — да.**
- **Прямые мишени GR:** у SCENE значимы с первого часа (q ≈ 10⁻⁴–10⁻³), у SIMBA слабее (q ≈ 0.006–0.03). В контроле незначимы у обоих, как и должно быть.
- **Поздние TF:** у SCENE становятся значимыми к 4–18 ч, как ожидается по биологии. У SIMBA не значимы ни в одной точке.
- **Метаболические модули** (гликолиз, TCA) компактны у обоих во всех точках. Это «положительный контроль».

### Рис. 6 (продолжение) — Регулоны TF при активации T-клеток (TCR)

**Что показано.** CD4⁺ T-клетки без стимуляции и после стимуляции anti-CD3/CD28 обучены в одном пространстве. Для каждого TF берётся его регулон (мишени из DoRothEA, классы A/B) и проверяется, компактен ли он в пространстве генов. Всего 174 регулона.
- **Тепловая карта:** −log10(q) для 26 TF, важных при активации T-клеток: ранняя транскрипция (NFAT, NF-κB, EGR, ATF), ответ на цитокины (STAT/IRF), анаболический метаболизм (MYC, HIF1A, SREBF2).
- **Точечный график:** q-значения всех регулонов у SCENE против SIMBA.
""")
fig("fig_6/tf_heatmap.svg", width=98)
fig("fig_6/qval_jitter.svg", width=40)
code(r"""
s = pd.read_csv(res / "scLDM_TCR/gene_module_eval.csv", index_col="module")["qval_fdr_bh"]
b = pd.read_csv(res / "SIMBA_TCR/gene_module_eval.csv", index_col="module")["qval_fdr_bh"]
print(f"regulons: {len(s)};  q < 0.05: SCENE {(s < 0.05).sum()}, SIMBA {(b < 0.05).sum()}, both {((s < 0.05) & (b < 0.05)).sum()}")
tfs = ["NFATC1", "NFATC2", "RELA", "REL", "NFKB1", "NFKB2", "EGR1", "ATF1", "ATF2", "ELK1",
       "STAT1", "IRF1", "MYC", "MAX", "HIF1A", "SREBF2"]
pd.DataFrame({"SCENE q": s.reindex(tfs), "SIMBA q": b.reindex(tfs)}).map(lambda q: f"{q:.1e}")
""")
md(r"""
**Совпадает ли со статьёй — да.** SCENE находит **58 значимых регулонов из 174, SIMBA — 42** (у обоих — 39). Ключевые регуляторы активации T-клеток значимы у SCENE, но не у SIMBA: NFATC1, NFATC2, NFKB2, ATF1, ATF2, HIF1A. Это ровно тот пример, который приводят авторы.
""")

md(r"""
### Приложение — Нативные 2D-пространства разных методов

Стволовые клетки NeurIPS, **2D без UMAP**: PCA, scVI, SIMBA и SCENE. В статье этим показывают, что ветвление эритроидной и моноцитарной линий у SCENE видно прямо в координатах модели.
""")
fig("app/low_dim/kang_PCA_2d.svg", "app/low_dim/kang_scVI_2d.svg", "app/low_dim/kang_SIMBA_2d.png", "fig_5/kang_scLDM_2d_cell_type.svg")
md(r"""
### Приложение — Импутация (PBMC)

**Что показано.** 10 % ненулевых счётов скрыто, модели обучены на остальном и предсказывают скрытые значения. Сравниваются SCENE с ZIP-правдоподобием, SCENE с обычным Poisson и scVI.
""")
code(r"""pd.read_csv(FIGS / "app/imputation_metrics.csv", index_col=0).drop(columns=["Spearman"]).round(3)""")
fig("app/pbmc_imputation_scvi_vs_scene.svg", width=80)
md(r"""
**Совпадает ли со статьёй — да.** SCENE-ZIP точнее scVI:
- Pearson 0.85 против 0.65;
- RMSE 8.6 против 12.6.

Zero-inflated вариант лучше обычного Poisson. Колонку Spearman скрипт авторов выдал нулевой для всех моделей, поэтому в таблице её нет. Причину мы не разбирали.
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 6. Сводка: утверждения статьи и наш результат

| Утверждение статьи | Наш результат | Статус |
|---|---|---|
| Кластеры в совместном пространстве совпадают с типами клеток (PBMC, NMI 0.63) | NMI 0.660 | ✅ |
| Маркеры и IFN-гены локализуются у своих клеток (Kang) | видно на рис. 3 | ✅ качественно |
| SCENE на уровне или лучше Harmony/scVI/SIMBA по метрикам scIB | лучший на Cortex, SIM1, SIM2; на HCA, NeurIPS, PBMC SIMBA или Harmony чуть впереди | ⚠️ «на уровне», не «лучше» везде |
| На SIM2 (вложенные батчи) scVI схлопывает группы, SCENE — нет | NMI 0.99 против 0.29 | ✅ сильно |
| Link prediction насыщается при малой d, deviance улучшается с d | так и есть | ✅ |
| Стволовые клетки: >8D ухудшает восстановление счётов | 0.79 (8D) → 0.95 (64D) | ✅ |
| Модули GR компактнее у SCENE, чем у SIMBA | GR Direct: q 10⁻⁴–10⁻³ против 0.006–0.03; поздние TF значимы только у SCENE | ✅ |
| Регулоны TCR лучше, чем у SIMBA | 58 против 42 значимых из 174; NFAT, NFKB2, ATF1/2, HIF1A — только у SCENE | ✅ |
| Импутация лучше scVI | Pearson 0.85 против 0.65 | ✅ |

## 7. Что не воспроизвели
- **Полный PBMC** (152k клеток): мы использовали подвыборку 30k.
- **5 сидов в переборе размерностей:** у нас 2.

Всё остальное из `bash/paper/*.sh`, включая SIMBA и TCR, выполнено.

## Выводы для нашего проекта
- Все главные утверждения воспроизводятся: вложенные батчи, низкие размерности, импутация, модули GR и регулоны TCR лучше, чем у SIMBA.
- **Нюанс для обсуждения:** на реальных данных (HCA, NeurIPS, PBMC) SIMBA по кластеризации не хуже SCENE. Сильное преимущество SCENE видно на симуляциях с батчами и в биологической интерпретации генов.
- **PBMC** стоит проверить на полном датасете: там SCENE не лидирует.
- Модель **трансдуктивна** (нет энкодера): новых клеток без переобучения не вложить. Это прямое направление для улучшения (fold-in или агрегация вложений генов).
- Тест компактности модулей сравнивает с **равномерно** случайными генами. Нулевой тест с учётом уровня экспрессии — открытый вопрос и хорошее продолжение.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
nbf.write(nb, OUT)
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
