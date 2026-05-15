# Applied Machine Learning — Assignment 2
## Clustering from Scratch: Prototype Adjustment vs DBSCAN

**Student:** Rishikesh Mahendra Dharane  
**Student ID:** R00277319  
**Module:** COMP Applied Machine Learning  
**Institution:** Munster Technological University  

---

## Overview

This project implements two clustering algorithms **entirely from scratch** using only `numpy` and `matplotlib` — no sklearn, no scipy, no pre-built clustering tools. The goal is to contrast a prototype-based method (custom online K-means variant) against a density-based method (DBSCAN) on two synthetic datasets with different geometric structures.

---

## Personalised Parameters

Derived from student ID `R00277319`:

| Parameter | Calculation | Value |
|-----------|-------------|-------|
| `a` (last digit) | 9 | 9 |
| `b` (second-last digit) | 1 | 1 |
| `seed` | 100 + 10×9 + 1 | **191** |
| Rare-word threshold `t` | — | — |
| Smoothing `α` set | seed mod 3 = 0 | {0, 0.1, 1} |

**Dataset A cluster centers:**
- Group 1: `(-4 - 0.1×9, 4 + 0.1×1)` = `(-4.9, 4.1)`
- Group 2: `(0, -4 - 0.1×9)` = `(0.0, -4.9)`
- Group 3: `(4 + 0.1×1, 4)` = `(4.1, 4.0)`

**Dataset B blob center:** `(5 + 0.1×9, 0)` = `(5.9, 0)`

---

## Project Structure

```
assignment2/
│
├── assignment2_complete.ipynb   ← Main notebook (all 4 tasks)
├── README.md                    ← This file
│
└── output plots (generated on run)
    ├── task1_scatter.png        ← Raw data scatter plots
    ├── task2_final_clusters.png ← Final clustering results
    ├── task2_centroid_paths.png ← Centroid movement every 50 epochs
    ├── task2_sse.png            ← SSE convergence curve
    ├── task2_lr_comparison.png  ← Learning rate comparison
    ├── task3_4nn_plot.png       ← 4-NN elbow plot for ε selection
    └── task3_dbscan.png         ← DBSCAN clustering results
```

---

## Tasks Summary

### Task 1 — Synthetic Data Generation *(6 marks)*

Generates two 2D datasets using `numpy.random` seeded at 191:

- **Dataset A** — 120 points across three Gaussian clusters (40 points each). Well-separated, convex shapes.
- **Dataset B** — 120 points combining:
  - 70 ring points (polar coordinates, radius ≈ 4.5)
  - 30 blob points (Gaussian, centre = (5.9, 0))
  - 20 uniform noise points in [−7, 7] × [−7, 7]

Outputs a scatter plot and mean/std summary statistics for both datasets.

---

### Task 2 — Manual Prototype Adjustment Clustering *(18 marks)*

A custom **online K-means variant** — the largest task in the assignment.

**Key differences from standard K-means:**

| Feature | Standard K-means | This method |
|---------|-----------------|-------------|
| Initialisation | Random | Deterministic (min x+y, then farthest) |
| Update timing | After full epoch (batch) | After each point (online) |
| Update rule | Move to mean | Fixed step `α` toward point |
| Empty clusters | Reinitialise randomly | Nudge by small random offset |

**Update rule:**
```
if point_x > centroid_x  →  centroid_x += α
if point_x < centroid_x  →  centroid_x -= α
(same for y; no change if exactly equal)
```

**Parameters:**
- K = 3 for Dataset A, K = 2 for Dataset B
- 200 epochs
- Learning rates compared: α ∈ {0.01, 0.05, 0.2}

**Outputs:** final cluster plots, centroid path plots (every 50 epochs), SSE vs epoch curve, learning rate comparison plot, final centroid coordinates.

---

### Task 3 — DBSCAN *(8 marks)*

Full DBSCAN implementation from scratch using Euclidean distance.

**Fixed parameter:** `minPts = 4`  
**ε chosen via:** 4-nearest-neighbour distance plot (elbow method)

| Dataset | Chosen ε | Reason |
|---------|----------|--------|
| A | 1.2 | Elbow in 4-NN curve around 1.0–1.3 |
| B | 0.6 | Elbow in 4-NN curve around 0.5–0.7 |

**Algorithm steps:**
1. For each unvisited point, find all points within distance ε
2. If fewer than `minPts` neighbours → mark as noise (−1)
3. Otherwise → start new cluster, expand by visiting all reachable neighbours
4. Border points assigned to first core point that reaches them

**Outputs:** 4-NN elbow plots, DBSCAN cluster plots, cluster count and noise count.

---

### Task 4 — Comparison and Analysis *(8 marks)*

**Manual silhouette score** implemented from scratch:

```
a(i) = average distance from point i to all other points in same cluster
b(i) = minimum average distance from point i to points in any other cluster
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Noise points (label = −1) are excluded from DBSCAN silhouette computation.  
Single-point clusters get silhouette score = 0.

**Final comparison table** columns: Dataset | Algorithm | Clusters | Noise | Final SSE | Silhouette Score

**Key findings:**
- Dataset A suits the prototype method (convex, well-separated Gaussian clusters)
- Dataset B suits DBSCAN (ring shape + noise — density-based detection handles both naturally)
- Higher α converges faster but oscillates; lower α is stable but slow to converge

---

## How to Run

### Requirements

```
python >= 3.8
numpy
matplotlib
jupyter
```

No other packages needed — everything is implemented from scratch.

### Install dependencies

```bash
pip install numpy matplotlib jupyter
```

### Run the notebook

```bash
jupyter notebook assignment2_complete.ipynb
```

Then run all cells top to bottom using **Kernel → Restart & Run All**.

> ⚠️ **Important:** Always run from the top. Tasks 2, 3, and 4 depend on variables (`dataset_a`, `dataset_b`, `assign_a`, `labels_b`, etc.) defined in earlier cells. Running cells out of order will cause `NameError`.

### Expected runtime

| Task | Approximate time |
|------|-----------------|
| Task 1 — Data generation | < 1 second |
| Task 2 — Prototype clustering (all 6 runs) | ~5–10 seconds |
| Task 3 — DBSCAN + 4-NN plots | ~15–30 seconds |
| Task 4 — Silhouette scores | ~30–60 seconds |

Task 4 is slowest because silhouette requires computing distances between every pair of points (O(n²)).

---

## Variable Reference

Key variables available after running each task:

| Variable | Type | Description |
|----------|------|-------------|
| `dataset_a` | `ndarray (120, 2)` | Dataset A points |
| `dataset_b` | `ndarray (120, 2)` | Dataset B points |
| `assign_a` | `ndarray (120,)` | Prototype cluster labels for A (0/1/2) |
| `assign_b` | `ndarray (120,)` | Prototype cluster labels for B (0/1) |
| `sse_a_hist` | `list[float]` | SSE per epoch for Dataset A (α=0.05) |
| `sse_b_hist` | `list[float]` | SSE per epoch for Dataset B (α=0.05) |
| `all_results` | `dict` | All prototype results keyed by `(dataset, alpha)` |
| `labels_a` | `ndarray (120,)` | DBSCAN labels for A (−1=noise) |
| `labels_b` | `ndarray (120,)` | DBSCAN labels for B (−1=noise) |
| `eps_a` | `float` | Chosen ε for Dataset A |
| `eps_b` | `float` | Chosen ε for Dataset B |

---

## Constraints Followed

- ✅ Only `numpy` and `matplotlib` used
- ✅ No `sklearn`, `scipy`, or any pre-built clustering tools
- ✅ Centroid initialisation implemented manually
- ✅ Point assignment uses manual Euclidean distance
- ✅ Centroid update rule matches spec exactly (fixed step, not gradient)
- ✅ SSE computed after full epoch using final centroid positions
- ✅ DBSCAN implemented from scratch
- ✅ Silhouette score implemented from scratch
- ✅ 4-nearest-neighbour distance computed manually
- ✅ All results use seed = 191 (from student ID R00277319)

---

## AI Usage Disclosure

As per MTU Academic Integrity Policy, this project used AI assistance (Claude, Anthropic) for:
- Code structure guidance and debugging
- Explanation of algorithm concepts
- Report and README formatting

All algorithmic implementations, parameter choices, analysis, and conclusions are the student's own work.
