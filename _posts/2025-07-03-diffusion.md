---
layout: post
title: "diffusion"
hidden: true
date: 2025-07-03
---

# From Noise to Art: The Complete Guide to Diffusion Models

*July 3, 2025*

If you've been anywhere near the AI space recently, you've seen the stunning images produced by models like DALL-E 3, Midjourney, and Stable Diffusion. The technology behind many of these state-of-the-art generators is a class of models known as **Diffusion Models**.

At first glance, their inner workings seem almost magical. They start with pure random noise and meticulously sculpt it into a coherent, often beautiful, image. But how? This post offers a comprehensive guide, from the foundational theory of **score matching** to the practical implementations in **DDPMs**, and finally to the unified, high-performance framework of **EDM** (Elucidating Diffusion Models).

---

## Part 1: The Core Theory - Score-Based Generation

Before we can denoise an image, we must first understand what our model is fundamentally trying to learn. The goal of any generative model is to learn the probability distribution of the training data, $p(x)$. If we could perfectly model this distribution, we could sample from it to create new data that is indistinguishable from the real thing.

### The Score Function: A Guiding Field for Creativity

The challenge is that for high-dimensional data like images, the function $p(x)$ is intractably complex. A more accessible quantity to work with is its **score function**, which is the gradient of the log-probability of the data with respect to the input itself:

$$\text{score} = \nabla_x \log p(x)$$

Think of the data distribution as a landscape with mountains where the data is dense (e.g., images of cats) and valleys where it's sparse. The **score function at any point `x` is a vector that points in the direction of the steepest ascent up the nearest mountain**. It's a guiding field that always points toward regions of higher data density.

If we had this magical guiding field, we could start from a random point (pure noise) and take small steps in the direction of the score, eventually climbing a "probability mountain" and arriving at a plausible data sample (a realistic image). This is the core idea behind sampling with methods like **Langevin dynamics**.

### The Problem: Why The True Score is Unattainable

The problem is, we can't get the score function for the true data distribution $p(x)$ directly. The function $p(x)$ has an unknown and intractable normalization constant (also called the partition function), which would require integrating over the entire, astronomically large space of all possible images. Without this constant, we can't compute $\log p(x)$ and therefore cannot compute its gradient, the score.

### The Solution: Denoising Score Matching 💡

This is where a series of brilliant insights comes together. The original concept of **Score Matching** (Hyvärinen, 2005) provided a way to learn a score function without knowing the normalization constant. However, it required calculating a term (the trace of the Hessian) that was still too computationally expensive for large neural networks.

The crucial breakthrough was **Denoising Score Matching** (Vincent, 2011). This work revealed an incredible connection: the complex score matching objective is mathematically equivalent to a much simpler task—**training a model to denoise corrupted data**.

Here's the idea: instead of trying to model the score of clean, perfect data, what if we model the score of data corrupted by various levels of Gaussian noise?

It turns out that for a clean data point $x_0$ and a noisy version $x_t$ (created by adding Gaussian noise $\epsilon$), there’s a beautiful relationship: the score of the noised data distribution, $\nabla_{x_t} \log p(x_t)$, is directly proportional to the negative of the noise that was added.

$$\nabla_{x_t} \log p(x_t) \propto -\epsilon$$

#### An Intuitive Look: The "Center of Gravity" Analogy 🌌

This relationship is the most important concept to grasp. Let's build an intuition for it.

Imagine the space of all possible images. Within this space, there's a complex, beautifully structured shape called the **"manifold of real images."** Think of this as a galaxy. Every point on this galaxy (`x₀`) is a perfect, clean image of a cat, a dog, a car, etc.

1.  **The Setup:** This galaxy has a "gravitational pull." Points on the galaxy are stable and "highly probable." Points far away are "improbable." The **score function** is the vector field that describes this pull at every point in space.

2.  **The Perturbation (Adding Noise):** Now, we take a specific star (`x₀`) in our galaxy. We give it a single, random "kick" with a rocket. The path of this rocket is the noise vector `ϵ`. The star is pushed off the galaxy to a new, isolated position in empty space, `x_t`.

    ![Analogy Diagram](https://i.imgur.com/r62sK6l.png)

3.  **The Question:** We are now at the location `x_t`. We've forgotten where we came from, but we can still feel the galaxy's gravitational pull. Which direction does the gravity pull us? It pulls us back towards the galaxy. **What is the most direct path back?**

4.  **The Intuitive Answer:** The most direct way to undo the random kick from our rocket (`ϵ`) is to fire an identical rocket in the exact opposite direction (`-ϵ`). This `-ϵ` vector points straight back toward where we started.

This is the core of the intuition. The score `∇_{x_t} log p(x_t)` represents the "gravitational pull" of the entire data manifold. For any specific noisy point `x_t`, the single most defining reason it is "improbable" is the specific random noise `ϵ` that was added to it. Therefore, the most effective way to make it more probable is simply to remove that noise. The direction for that is `-ϵ`.

This intuition holds up mathematically because `x_t` is a sample from a Gaussian distribution whose mean is based on `x_0`. The score of *any* Gaussian distribution always points from a sample point directly back towards the distribution's mean. In our case, the vector from the mean to the sample `x_t` is exactly the noise vector `ϵ`, so the vector pointing *back* to the mean is `-ϵ`.

This insight is powerful because it transforms an impossibly abstract problem ("calculate the gradient of a log probability") into a concrete engineering task ("predict the noise").

The pioneering work that applied this to create large-scale generative models was **Noise Conditional Score Networks (NCSN)** (Song & Ermon, 2019). They trained a single deep neural network conditioned on the noise level `σ` to estimate the score, laying the direct theoretical and practical foundation for the diffusion models that followed.

---

## Part 2: The Practical Recipe - DDPMs

**Denoising Diffusion Probabilistic Models (DDPMs)** can be understood as a practical, discrete-time recipe built on these score-based principles.

### The Forward Process: Creating the Noisy Data

DDPM defines a fixed forward process over `T` discrete timesteps to generate the noisy images needed for training. This is a **Variance Preserving (VP)** process defined by a noise schedule `βt`.

* The single-step transition is:
    $$
    x_t = \sqrt{1 - \beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon_{t-1} \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, I)
    $$
* Crucially, we can jump to any step `t` directly from the original image `x₀`. Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$:
    $$
    x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon
    $$

### The Reverse Process: Learning the Score by Predicting Noise

The DDPM U-Net model, $\epsilon_\theta(x_t, t)$, is trained to predict the noise `ϵ` from the noisy image `x_t`. This act of predicting the noise is **implicitly learning the score function**. The simple and effective loss function is:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ || \epsilon - \epsilon_\theta(x_t, t) ||^2 \right]
$$

To generate an image, DDPM uses an **ancestral sampler**. It starts with pure noise $x_T$ and iterates backwards. Each step `t` involves:
1.  Using the model $\epsilon_\theta$ to predict the noise.
2.  Using this prediction to estimate the direction towards the cleaner image $x_{t-1}$.
3.  Adding a small amount of random noise back in, inspired by Langevin dynamics.

This process is inherently **stochastic** and **slow**, as it requires all `T` steps (e.g., 1000) to produce an image.

---

## Part 3: The Unified Framework - EDM's Breakthroughs

The **EDM (Elucidating Diffusion Models)** paper brilliantly unified the score-based perspective of NCSN and the probabilistic view of DDPM into a single, high-performance framework.

### Unifying Theory: The Denoiser IS the Score Estimator

EDM makes the connection between denoising and score matching explicit. It defines a **denoiser function** `D(x; σ)` that aims to predict the clean image `x₀` from a noisy input `x` with noise level `σ`. The estimated score can then be derived directly from this denoiser's output.

To make the denoiser `D(x; σ)` work reliably across all noise levels, EDM introduces **principled preconditioning**. The final denoiser is a wrapper around the U-Net `F_θ`:

```
    D(x; σ) = c_skip(σ) * x + c_out(σ) * F_θ(c_in(σ) * x; σ)
```
This structure makes the U-Net's job vastly simpler and more stable, allowing it to learn a much more accurate score representation.

### The Reverse Process: Precision Engineering with ODE Solvers

Instead of the slow, stochastic Langevin-style sampling, EDM advocates for treating generation as solving a **probability flow ODE**. This is a deterministic path from noise to image, guided by the score field. To walk this path efficiently, we can use advanced numerical solvers.

Here is a step-by-step of the **deterministic Heun method** used in EDM:

1.  **Setup:** Start with pure noise `x_0` at noise level `σ_0`. Choose `N` evaluation steps, giving a schedule of noise levels `σ_0, σ_1, ..., σ_N=0`.

2.  **Iterate for i = 0 to N-1:** To get from `x_i` to `x_{i+1}`:

    a. **Calculate the Score (Derivative) at the current point:** First, get the denoised estimate using your model, then calculate the score `d_i`.
    $$
    d_i = \frac{x_i - D(x_i; \sigma_i)}{\sigma_i}
    $$

    b. **Predictor Step:** Take a simple "Euler" step to a temporary point `x_hat`. This is your first guess for the next location.
    $$
    x_{\text{hat}} = x_i + d_i \cdot (\sigma_{i+1} - \sigma_i)
    $$

    c. **Calculate the Score at the predicted point:** Now, evaluate the score `d_hat` at this new temporary location.
    $$
    d_{\text{hat}} = \frac{x_{\text{hat}} - D(x_{\text{hat}}; \sigma_{i+1})}{\sigma_{i+1}}
    $$

    d. **Corrector Step:** Average the two scores and take a much more accurate, final step from your original position `x_i` to get `x_{i+1}`.
    $$
    x_{i+1} = x_i + \frac{1}{2} (d_i + d_{\text{hat}}) \cdot (\sigma_{i+1} - \sigma_i)
    $$

3.  **Final Image:** After `N` steps, `x_N` is the final, clean image. This 2nd-order method is far more accurate than DDPM's 1st-order sampler, enabling high-quality results in as few as 20-40 steps.

---

## Part 4: The Complete Comparison & Practical Guide

Here is a final summary of the evolution from DDPM to EDM:

| Aspect | DDPM | EDM |
| :--- | :--- | :--- |
| **Core Theory** | Probabilistic model, discrete time. | Unified score-based and probabilistic model, continuous time. |
| **Training Model** | Simple U-Net `ϵ_θ(x_t, t)` predicts `ϵ`, implicitly learning the score. | Preconditioned denoiser `D(x; σ)` explicitly enables score calculation. |
| **Forward Params** | `T`, `beta_start`, `beta_end`, `scheduler`. | `σ_min`, `σ_max`, `ρ`. |
| **Sampling Logic** | Stochastic ancestral sampling (1st-order Langevin-style). | Solving an ODE (typically 2nd-order Heun). |
| **Determinism** | Inherently **stochastic**. | Inherently **deterministic** (but can be made stochastic). |
| **Speed/Steps** | Slow, requires many steps (`T`=1000+). | Fast, requires few steps (`N`=20-80). |

### Practical Guide to the EDM Toolkit

* **The Solver:** Start with the **deterministic Heun solver**. It offers the best balance of speed, quality, and reproducibility.
* **Number of Steps (`N`):** For fast previews, try `N = 20-25`. For high quality, `N = 40-80` is often sufficient.
* **The Timestep Schedule (`ρ`):** `ρ = 7` is an excellent default. It concentrates steps at low `σ` values, which is crucial for fine details.
* **Stochasticity (`S_churn`):** Start with `S_churn = 0` (fully deterministic). Only introduce churn for advanced fine-tuning or troubleshooting.
