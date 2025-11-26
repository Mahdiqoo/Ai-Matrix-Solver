# Ai-Matrix-Solver
This repository contains a family of **neural iterative solvers** for linear systems $Ax = b$, trained in PyTorch.

These solvers are designed to work for **any matrix size** $N \times N$ on GPU. They utilize a learned gradient-descent approach to approximate solutions significantly faster than `torch.linalg.solve` for large dimensions and batch sizes.

> **⚠️ Note:** These models are **approximate solvers**. They aim for small residuals (e.g., $10^{-3}$ to $10^{-2}$), not machine‑precision solutions. For exact results, always use `torch.linalg.solve`.

---

## 🚀 Key Features

* **Dimension Agnostic:** The same trained model weights work for any matrix size $N$.
* **Batch Processing:** Optimized for solving batches of systems simultaneously ($A \in \mathbb{R}^{B \times N \times N}$).
* **Controllable Trade-off:** Choose between speed and accuracy by selecting a model variant with a specific number of iterations.
* **Low Complexity:** Inference cost is roughly $O(T \cdot B \cdot N^2)$ (where $T$ is iterations), compared to the cubic complexity $O(N^3)$ of direct solvers.

---

## 🛠️ Installation

**Requirements:**
* Python 3.9+
* PyTorch ≥ 2.0 (CUDA recommended)

1.  **Install PyTorch:**
    Follow instructions at [pytorch.org](https://pytorch.org), or run:
    ```bash
    pip install torch --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```

2.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/learned-linear-solver.git](https://github.com/yourusername/learned-linear-solver.git)
    cd learned-linear-solver
    ```

3.  **Setup Weights:**
    Place your pretrained weights in a `models/` directory:
    ```text
    models/
    ├── flash.pth
    ├── balanced.pth
    ├── accurate.pth
    ├── extreme.pth
    └── real.pth
    ```

---

## 📊 Model Variants

All variants share the same internal architecture (Hidden Dim = 20). The only difference is the **number of iterations** ($T$) the solver runs, which dictates the speed/accuracy trade-off.

| Variant | Filename | Iterations ($T$) | Speed | Accuracy | Recommended Use Case |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Flash** | `models/flash.pth` | 5 | ⭐⭐⭐⭐⭐ | ⭐ | Initializers, rough approximations |
| **Balanced** | `models/balanced.pth` | 10 | ⭐⭐⭐⭐ | ⭐⭐ | Good general compromise |
| **Accurate** | `models/accurate.pth` | 15 | ⭐⭐⭐ | ⭐⭐⭐ | General purpose correctness |
| **Extreme** | `models/extreme.pth` | 20 | ⭐⭐ | ⭐⭐⭐⭐ | High precision requirements |
| **Real** | `models/real.pth` | 25 | ⭐ | ⭐⭐⭐⭐⭐ | Highest accuracy, closest to exact |

---

## 🧠 How It Works

Each solver implements a **fixed‑step learned gradient descent**. For a system $Ax=b$, the update rule at step $t$ is:

$$
\begin{aligned}
x_0 &= 0 \\
r_t &= b - A x_t \\
\alpha_t &= f_\theta \left( \frac{\|r_t\|}{\|b\|}, \frac{t+1}{T}, \alpha_{t-1} \right) \\
x_{t+1} &= x_t + \alpha_t r_t
\end{aligned}
$$

**Where:**
* $f_\theta$ is a small MLP (shared parameters $\theta$ for any matrix size $N$).
* The inputs to the MLP are the **relative residual**, **progress fraction**, and the **previous step size**.
* This requires exactly **one matrix–vector product** per iteration ($A @ x$).

---

## ⚡ Quick Start

Save the following code as `demo.py`. This script defines the model architecture, loads a variant (e.g., Balanced), generates a random SPD batch, and benchmarks it against the exact solver.

```python
import time
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.backends.cudnn.benchmark = True

# ----------------------------
# 1. Solver Architecture
# ----------------------------
class FastLearnedGradientSolver(nn.Module):
    """
    Learned gradient-descent solver for A x = b.
    - Same parameters for any N.
    - Uses T iterations of: x_{t+1} = x_t + alpha_t * r_t
    """
    def __init__(self, num_iters: int, hidden_dim: int = 20, alpha_max: float = 1.0):
        super().__init__()
        self.num_iters = num_iters
        self.alpha_max = alpha_max
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, A, b, num_iters=None):
        T = self.num_iters if num_iters is None else num_iters
        B, N, _ = A.shape
        x = torch.zeros(B, N, device=A.device, dtype=A.dtype)
        b_norm = torch.norm(b, dim=1, keepdim=True) + 1e-12
        alpha_prev = torch.zeros(B, 1, device=A.device, dtype=A.dtype)

        for t in range(T):
            Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
            r = b - Ax
            r_norm = torch.norm(r, dim=1, keepdim=True) + 1e-12
            rel_res = r_norm / b_norm
            step_frac = torch.full_like(rel_res, float(t + 1) / T)
            
            # MLP inputs: [relative_residual, progress, prev_alpha]
            feats = torch.cat([rel_res, step_frac, alpha_prev], dim=1)
            alpha = self.alpha_max * torch.sigmoid(self.mlp(feats))
            
            x = x + alpha * r
            alpha_prev = alpha
        return x

# ----------------------------
# 2. Helpers: Generator & Loader
# ----------------------------
def generate_spd_batch(batch_size, n, device=device, lam_min=0.1, lam_max=2.0):
    """Generates random SPD systems A x = b."""
    M = torch.randn(n, n, device=device)
    Q, _ = torch.linalg.qr(M)
    lam = torch.rand(batch_size, n, device=device) * (lam_max - lam_min) + lam_min
    AQ = lam.unsqueeze(2) * Q
    A = torch.einsum("ij,bjk->bik", Q.t(), AQ)
    x_true = torch.randn(batch_size, n, device=device)
    b = torch.bmm(A, x_true.unsqueeze(-1)).squeeze(-1)
    return A, b, x_true

def load_variant(name, checkpoint_dir="models"):
    configs = {
        "flash":    {"iters": 5,  "ckpt": "flash.pth"},
        "balanced": {"iters": 10, "ckpt": "balanced.pth"},
        "accurate": {"iters": 15, "ckpt": "accurate.pth"},
        "extreme":  {"iters": 20, "ckpt": "extreme.pth"},
        "real":     {"iters": 25, "ckpt": "real.pth"},
    }
    name = name.lower()
    if name not in configs: raise ValueError(f"Unknown variant: {name}")
    
    cfg = configs[name]
    model = FastLearnedGradientSolver(num_iters=cfg["iters"]).to(device)
    try:
        model.load_state_dict(torch.load(f"{checkpoint_dir}/{cfg['ckpt']}", map_location=device))
    except FileNotFoundError:
        print(f"Warning: Checkpoint not found at {checkpoint_dir}/{cfg['ckpt']}. Initializing random weights for demo.")
    
    model.eval()
    return model

# ----------------------------
# 3. Benchmark Main
# ----------------------------
if __name__ == "__main__":
    VARIANT = "Balanced" 
    N, B = 512, 1024  # Matrix size 512, Batch size 1024

    print(f"--- Benchmarking '{VARIANT}' on {B} systems of size {N}x{N} ---")
    model = load_variant(VARIANT)
    A, b, x_true = generate_spd_batch(B, N, device=device)

    # Warm-up
    with torch.no_grad():
        _ = model(A, b)
        _ = torch.linalg.solve(A, b.unsqueeze(-1))
    if device.type == "cuda": torch.cuda.synchronize()

    # 1. Neural Solver
    t0 = time.perf_counter()
    with torch.no_grad(): x_pred = model(A, b)
    if device.type == "cuda": torch.cuda.synchronize()
    t_model = time.perf_counter() - t0

    # 2. Exact Solver
    t0 = time.perf_counter()
    with torch.no_grad(): x_exact = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
    if device.type == "cuda": torch.cuda.synchronize()
    t_exact = time.perf_counter() - t0

    # Metrics
    res_model = torch.norm(torch.bmm(A, x_pred.unsqueeze(-1)).squeeze(-1) - b, dim=1) / torch.norm(b, dim=1)
    
    print(f"\nSpeed:")
    print(f"  Neural Model       : {t_model*1e3:.2f} ms ({B/t_model:.0f} systems/sec)")
    print(f"  torch.linalg.solve : {t_exact*1e3:.2f} ms ({B/t_exact:.0f} systems/sec)")
    print(f"  Speedup            : {t_exact/t_model:.2f}x")
    print(f"\nAccuracy:")
    print(f"  Mean Relative Residual: {res_model.mean().item():.3e}")
```
⚠️ Limitations & Notes
1. SPD Requirement: The models are trained on Symmetric Positive Definite (SPD) matrices. Performance may degrade on non-SPD or highly ill-conditioned matrices.
2. Approximation: These are suitable for large-scale simulations, inner loops, preconditioners, or initializers. They are not suitable for applications requiring strict numerical guarantees.
3. Hardware: Benchmarks are GPU-based. CPU performance characteristics will differ.
