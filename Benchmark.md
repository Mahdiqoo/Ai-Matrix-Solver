# Benchmarks — Ai-Matrix-Solver

**Summary**

This file collects the benchmark results for the Ai-Matrix-Solver across multiple model presets (Flash, Balanced, Accurate, Extreme, Real). All runs use **batch size = 1024**. Times are wall-clock per batch (milliseconds) and throughput is reported as systems/s (1024 systems per batch divided by time).

**Columns / units**

* `N` — matrix dimension.
* `Batch` — systems in the batch (1024 throughout).
* `Model time (ms)` — wall-clock time for the model to solve the full batch.
* `Model (systems/s)` — throughput computed from the measured time.
* `torch.linalg.solve time (ms)` — wall-clock time for the exact solver for the full batch.
* `torch.linalg.solve (systems/s)` — throughput for the exact solver.
* `mean/max rel error in x` — mean and max relative error of the model solution vs ground-truth x_true.
* `mean/max rel residual` — mean and max of `||Ax-b||/||b||` for the model solutions.
* `Exact solver mean relative error (sanity)` — sanity-check error for the exact solver.

---

## Flash

|   N | Batch | Model time (ms) | Model (systems/s) | torch.linalg.solve time (ms) | torch.linalg.solve (systems/s) | mean rel err x | max rel err x | mean rel residual | max rel residual | Exact solver mean rel err |
| --: | ----: | --------------: | ----------------: | ---------------------------: | -----------------------------: | -------------: | ------------: | ----------------: | ---------------: | ------------------------: |
|   4 |  1024 |            2.58 |          396818.3 |                         0.20 |                      5165638.4 |      5.335e-02 |     4.251e-01 |         2.201e-02 |        3.199e-01 |                 9.823e-08 |
|   8 |  1024 |            1.87 |          546348.1 |                         0.19 |                      5277099.2 |      5.571e-02 |     3.614e-01 |         1.953e-02 |        7.129e-02 |                 1.600e-07 |
|  16 |  1024 |            1.79 |          570819.8 |                         0.26 |                      3876115.7 |      6.601e-02 |     3.122e-01 |         2.037e-02 |        8.673e-02 |                 2.445e-07 |
|  32 |  1024 |            1.78 |          574370.3 |                         0.73 |                      1397280.8 |      7.078e-02 |     2.942e-01 |         2.009e-02 |        4.441e-02 |                 2.968e-07 |
|  64 |  1024 |            1.81 |          565090.2 |                         3.60 |                       284536.8 |      7.507e-02 |     1.863e-01 |         2.015e-02 |        3.321e-02 |                 3.203e-07 |
| 128 |  1024 |            2.15 |          475369.0 |                        13.45 |                        76115.3 |      7.805e-02 |     1.553e-01 |         2.024e-02 |        2.740e-02 |                 3.457e-07 |
| 256 |  1024 |            5.93 |          172672.2 |                        42.33 |                        24192.4 |      7.924e-02 |     1.405e-01 |         2.028e-02 |        2.632e-02 |                 3.767e-07 |
| 512 |  1024 |           21.52 |           47585.6 |                       148.70 |                         6886.4 |      7.944e-02 |     1.204e-01 |         2.028e-02 |        2.404e-02 |                 4.247e-07 |

---

## Balanced

|   N | Batch | Model time (ms) | Model (systems/s) | torch.linalg.solve time (ms) | torch.linalg.solve (systems/s) | mean rel err x | max rel err x | mean rel residual | max rel residual | Exact solver mean rel err |
| --: | ----: | --------------: | ----------------: | ---------------------------: | -----------------------------: | -------------: | ------------: | ----------------: | ---------------: | ------------------------: |
|   4 |  1024 |            3.97 |          258205.5 |                         0.21 |                      4943993.8 |      2.841e-02 |     3.163e-01 |         9.244e-03 |        1.157e-01 |                 9.411e-08 |
|   8 |  1024 |            3.54 |          289068.8 |                         0.22 |                      4718741.8 |      3.307e-02 |     3.007e-01 |         8.557e-03 |        8.374e-02 |                 1.556e-07 |
|  16 |  1024 |            3.45 |          296566.3 |                         0.25 |                      4058354.7 |      4.252e-02 |     2.636e-01 |         9.088e-03 |        3.656e-02 |                 2.480e-07 |
|  32 |  1024 |            3.69 |          277314.5 |                         0.79 |                      1293082.8 |      4.613e-02 |     1.835e-01 |         8.978e-03 |        2.606e-02 |                 2.998e-07 |
|  64 |  1024 |            3.57 |          286896.7 |                         3.61 |                       283724.1 |      5.136e-02 |     1.674e-01 |         9.180e-03 |        2.280e-02 |                 3.187e-07 |
| 128 |  1024 |            4.15 |          246455.3 |                        13.36 |                        76628.2 |      5.429e-02 |     1.289e-01 |         9.245e-03 |        1.844e-02 |                 3.465e-07 |
| 256 |  1024 |           11.73 |           87317.3 |                        51.13 |                        20028.1 |      5.610e-02 |     1.109e-01 |         9.302e-03 |        1.517e-02 |                 3.774e-07 |
| 512 |  1024 |           42.84 |           23904.0 |                       156.24 |                         6554.2 |      5.640e-02 |     9.344e-02 |         9.293e-03 |        1.234e-02 |                 4.264e-07 |

---

## Accurate

|   N | Batch | Model time (ms) | Model (systems/s) | torch.linalg.solve time (ms) | torch.linalg.solve (systems/s) | mean rel err x | max rel err x | mean rel residual | max rel residual | Exact solver mean rel err |
| --: | ----: | --------------: | ----------------: | ---------------------------: | -----------------------------: | -------------: | ------------: | ----------------: | ---------------: | ------------------------: |
|   4 |  1024 |            5.25 |          194949.3 |                         0.20 |                      5054493.8 |      1.176e-02 |     1.785e-01 |         3.582e-03 |        5.912e-02 |                 9.411e-08 |
|   8 |  1024 |            5.21 |          196488.9 |                         0.19 |                      5289037.6 |      1.406e-02 |     1.760e-01 |         3.414e-03 |        3.936e-02 |                 1.556e-07 |
|  16 |  1024 |            5.88 |          174028.5 |                         0.26 |                      3957549.1 |      1.882e-02 |     1.565e-01 |         3.896e-03 |        2.004e-02 |                 2.480e-07 |
|  32 |  1024 |            6.07 |          168669.4 |                         0.74 |                      1375715.4 |      2.061e-02 |     1.012e-01 |         3.857e-03 |        1.252e-02 |                 2.998e-07 |
|  64 |  1024 |            5.24 |          195276.4 |                         3.68 |                       277902.2 |      2.381e-02 |     9.545e-02 |         4.014e-03 |        1.137e-02 |                 3.187e-07 |
| 128 |  1024 |            6.03 |          169688.1 |                        13.18 |                        77719.9 |      2.584e-02 |     7.222e-02 |         4.086e-03 |        8.499e-03 |                 3.465e-07 |
| 256 |  1024 |           17.41 |           58809.3 |                        39.45 |                        25953.6 |      2.708e-02 |     6.129e-02 |         4.146e-03 |        7.289e-03 |                 3.774e-07 |
| 512 |  1024 |           64.17 |           15957.8 |                       156.07 |                         6561.0 |      2.733e-02 |     5.121e-02 |         4.150e-03 |        5.886e-03 |                 4.264e-07 |

---

## Extreme

|   N | Batch | Model time (ms) | Model (systems/s) | torch.linalg.solve time (ms) | torch.linalg.solve (systems/s) | mean rel err x | max rel err x | mean rel residual | max rel residual | Exact solver mean rel err |
| --: | ----: | --------------: | ----------------: | ---------------------------: | -----------------------------: | -------------: | ------------: | ----------------: | ---------------: | ------------------------: |
|   4 |  1024 |            6.69 |          153157.1 |                         0.25 |                      4174395.1 |      5.370e-03 |     1.012e-01 |         1.560e-03 |        3.185e-02 |                 9.411e-08 |
|   8 |  1024 |            6.59 |          155337.0 |                         0.19 |                      5363615.4 |      6.623e-03 |     1.035e-01 |         1.544e-03 |        1.867e-02 |                 1.556e-07 |
|  16 |  1024 |            6.64 |          154203.4 |                         0.24 |                      4202163.4 |      9.033e-03 |     9.328e-02 |         1.828e-03 |        1.176e-02 |                 2.480e-07 |
|  32 |  1024 |            6.72 |          152374.3 |                         0.72 |                      1422303.2 |      9.945e-03 |     5.640e-02 |         1.841e-03 |        6.545e-03 |                 2.998e-07 |
|  64 |  1024 |            6.66 |          153733.2 |                         3.55 |                       288714.2 |      1.182e-02 |     5.558e-02 |         1.945e-03 |        6.115e-03 |                 3.187e-07 |
| 128 |  1024 |            7.98 |          128343.3 |                        13.08 |                        78311.5 |      1.312e-02 |     4.150e-02 |         1.999e-03 |        4.676e-03 |                 3.465e-07 |
| 256 |  1024 |           23.28 |           43989.1 |                        43.97 |                        23288.0 |      1.392e-02 |     3.450e-02 |         2.040e-03 |        3.854e-03 |                 3.774e-07 |
| 512 |  1024 |           85.49 |           11977.5 |                       156.78 |                         6531.6 |      1.412e-02 |     2.886e-02 |         2.044e-03 |        3.174e-03 |                 4.264e-07 |

---

## Real

|   N | Batch | Model time (ms) | Model (systems/s) | torch.linalg.solve time (ms) | torch.linalg.solve (systems/s) | mean rel err x | max rel err x | mean rel residual | max rel residual | Exact solver mean rel err |
| --: | ----: | --------------: | ----------------: | ---------------------------: | -----------------------------: | -------------: | ------------: | ----------------: | ---------------: | ------------------------: |
|   4 |  1024 |            8.44 |          121261.0 |                         0.23 |                      4396264.9 |      2.098e-03 |     5.756e-02 |         5.162e-04 |        3.438e-02 |                 9.394e-08 |
|   8 |  1024 |            8.60 |          119082.1 |                         0.21 |                      4978147.7 |      2.903e-03 |     4.780e-02 |         5.606e-04 |        1.829e-02 |                 1.547e-07 |
|  16 |  1024 |           14.12 |           72507.9 |                         0.32 |                      3212538.9 |      3.784e-03 |     4.376e-02 |         6.565e-04 |        5.255e-03 |                 2.420e-07 |
|  32 |  1024 |            9.54 |          107318.3 |                         0.78 |                      1305912.6 |      4.876e-03 |     3.245e-02 |         7.957e-04 |        4.101e-03 |                 2.945e-07 |
|  64 |  1024 |            8.61 |          118971.1 |                         3.71 |                       275908.5 |      5.859e-03 |     2.695e-02 |         8.924e-04 |        2.753e-03 |                 3.179e-07 |
| 128 |  1024 |           10.20 |          100379.7 |                        13.58 |                        75427.0 |      6.700e-03 |     2.426e-02 |         9.582e-04 |        2.453e-03 |                 3.466e-07 |
| 256 |  1024 |           28.75 |           35615.7 |                        42.09 |                        24327.1 |      7.177e-03 |     1.820e-02 |         9.909e-04 |        1.859e-03 |                 3.778e-07 |
| 512 |  1024 |          106.98 |            9571.7 |                       158.90 |                         6444.4 |      7.469e-03 |     1.765e-02 |         1.014e-03 |        1.785e-03 |                 4.253e-07 |

---

**Notes & suggestions**

* All runs use batch size = 1024; consider adding a small table (or script) that shows performance vs batch size for a single N to illustrate scaling.
* Please add hardware/runtime details (GPU model, CUDA/cuDNN, PyTorch version) and random seeds to make these results fully reproducible.
* If you'd like, I can also generate a compact CSV or a plotted figure (PNG) of throughput and error vs N for one or more presets.
