# Navier–Stokes using Fourier Neural Operators (FNO)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%F0%9F%A7%AA-red)](https://pytorch.org/)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-brightgreen)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains experiments on applying **Fourier Neural Operators (FNOs)** to learn and predict the dynamics of the **2D Navier–Stokes equations**.
The project explores architectural tweaks, custom loss functions, and training strategies (such as **noise injection**) to achieve robust long-term forecasting.

---

## 📌 Repository Structure

* **`2d-ns-family-fno.ipynb`**
  Baseline and family experiments with different Fourier Neural Operator architectures.
* **`2d-ns-superres.ipynb`**
  Super-resolution experiments and custom loss exploration (`SobolevLoss`, `FrequencyLoss`, `LpLoss`).
* **`2d-ns-tweeks.ipynb`**
  Critical head-to-head comparisons of loss functions, training strategies, and benchmarking against the original FNO paper.

---

## 🚀 Key Objectives

1. Investigate **loss functions** for PDE learning (`LpLoss`, `SobolevLoss`, `FrequencyLoss`).
2. Explore the impact of **noise injection** as a form of curriculum learning for long-term stability.
3. Compare custom FNO variants with the **official implementation** and author-reported benchmarks.
4. Evaluate both **single-step accuracy** and **autoregressive rollouts**.

---

## 📊 Results

### 🔹 Super-Resolution Experiments (`2d-ns-superres.ipynb`)

| Model       | Architecture                        | Training Loss   | Noise | 1-Step Error (t=10) | **Autoregressive Error (t=29)** |
| :---------- | :---------------------------------- | :-------------- | :---- | :------------------ | :------------------------------ |
| **Model 1** | Custom (`W=72, M=20`)               | `SobolevLoss`   | No    | 5.20%               | 74.80%                          |
| **Model 2** | Custom (`W=72, M=20`)               | `SobolevLoss`   | Yes   | 5.16%               | 71.02%                          |
| **Model 3** | Custom (`W=72, M=20`)               | `FrequencyLoss` | Yes   | 5.83%               | 70.00%                          |
| **Model 4** | Official (`W=64, M=16`)             | `SobolevLoss`   | Yes   | 7.37%               | 72.31%                          |
| **Model 5** | **Author's Replica** (`W=64, M=16`) | **`LpLoss`**    | Yes   | **2.28%**           | **56.60%**                      |

**Takeaway**: The author’s replica with **LpLoss + noise injection** outperformed all custom variants in both single-step and long-term error.

---

### 🔹 Tweaks & Loss Comparisons (`2d-ns-tweeks.ipynb`)

| Model / Method    | Training Loss         | Best 1-Step Error (Validation) | Autoregressive Error (t=29) |
| :---------------- | :-------------------- | :----------------------------- | :-------------------------- |
| **Model 1**       | **`LpLoss` + Noise**  | **1.04%**                      | **57.0%**                   |
| **Model 2**       | `SobolevLoss` + Noise | 2.67%                          | 71.1%                       |
| *Paper Benchmark* | *`LpLoss`*            | 3.8%                           | *N/A (Visual)*              |

#### Analysis

1. **Most Effective Strategy** → Standard `LpLoss` + **Noise Injection** yielded **1.04% validation error**, improving significantly over the paper’s reported **3.8%**.
2. **SobolevLoss Trade-off** → Effective in clean settings but struggles under noise injection, leading to higher autoregressive error.
3. **Noise Injection as Curriculum Learning** → Training with perturbed inputs made the model resilient to compounding errors, ensuring better stability in rollouts.

---

## 🧠 Learnings

* **LpLoss remains the most reliable** for FNO-based PDE prediction, especially when combined with training tricks.
* **SobolevLoss and FrequencyLoss** provide insights but are less robust in noisy autoregressive setups.
* **Noise Injection** is crucial for **long-term stability** in PDE rollouts.

---

## ⚙️ Setup & Dependencies

Clone the repo:

```bash
git clone https://github.com/harshu0117/Navier-Stokes-using-FNO.git
cd Navier-Stokes-using-FNO
```

Install dependencies:

```bash
pip install torch neuraloperator  
```

Run experiments:

```bash
jupyter notebook 2d-ns-tweeks.ipynb
```

---

## 📖 References

* Zongyi Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations* (NeurIPS 2020).
* [Original FNO Repository](https://github.com/zongyi-li/fourier_neural_operator)



