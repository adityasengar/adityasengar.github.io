---
layout: post
title: "Diffusion models"
date: 2025-07-03
---
# From Noise to Art: The Complete Guide to Diffusion Models

*July 3, 2025*

If you've been anywhere near the AI space recently, you've seen the stunning images produced by models like DALL-E 3, Midjourney, and Stable Diffusion. The technology behind many of these state-of-the-art generators is a class of models known as **Diffusion Models**.

At first glance, their inner workings seem almost magical. They start with pure random noise and meticulously sculpt it into a coherent, often beautiful, image. But how? This post offers a comprehensive guide, from the foundational theory of **score matching** to the practical implementations in **DDPMs**, and finally to the unified, high-performance framework of **EDM** (Elucidating Diffusion Models). While many excellent and more polished blogs on this topic exist, this post is primarily intended as a comprehensive personal reference.

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

This is where a series of brilliant insights comes together. The original concept of **Score Matching** [^1] provided a way to learn a score function without knowing the normalization constant. However, it required calculating a term (the trace of the Hessian) that was still too computationally expensive for large neural networks.

The crucial breakthrough was **Denoising Score Matching** [^2]. This work revealed an incredible connection: the complex score matching objective is mathematically equivalent to a much simpler task—**training a model to denoise corrupted data**.

Here's the idea: instead of trying to model the score of clean, perfect data, what if we model the score of data corrupted by various levels of Gaussian noise?

It turns out that for a clean data point $x_0$ and a noisy version $x_t$, there is a direct analytical expression for the score of the *conditional* data distribution, $p(x_t|x_0)$. 

Given that $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$, the score is exactly:

$$
\nabla_{x_t} \log p(x_t|x_0) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}
$$

The key insight of denoising score matching is that this easily computed conditional score is a great proxy for the score of the full marginal distribution, $\nabla_{x_t} \log p(x_t)$, which is what we need for generation. This transforms the problem: to learn the score, we just need to predict the noise $\epsilon$ that was added.


#### An Intuitive Look: The "Center of Gravity" Analogy 🌌

This relationship is the most important concept to grasp. Let's build an intuition for it.

Imagine the space of all possible images. Within this space, there's a complex, beautifully structured shape called the **"manifold of real images."** Think of this as a galaxy. Every point on this galaxy (`x₀`) is a perfect, clean image of a cat, a dog, a car, etc.

1.  **The Setup:** This galaxy has a "gravitational pull." Points on the galaxy are stable and "highly probable." Points far away are "improbable." The **score function** is the vector field that describes this pull at every point in space.

2.  **The Perturbation (Adding Noise):** Now, we take a specific star (`x₀`) in our galaxy. We give it a single, random "kick" with a rocket. The path of this rocket is the noise vector `ϵ`. The star is pushed off the galaxy to a new, isolated position in empty space, `x_t`.

3.  **The Question:** We are now at the location `x_t`. We've forgotten where we came from, but we can still feel the galaxy's gravitational pull. Which direction does the gravity pull us? It pulls us back towards the galaxy. **What is the most direct path back?**

4.  **The Intuitive Answer:** The most direct way to undo the random kick from our rocket (`ϵ`) is to fire an identical rocket in the exact opposite direction (`-ϵ`). This `-ϵ` vector points straight back toward where we started.

This is the core of the intuition. The score `∇_{x_t} \log p(x_t)` represents the "gravitational pull" of the entire data manifold. For any specific noisy point `x_t`, the single most defining reason it is "improbable" is the specific random noise `ϵ` that was added to it. Therefore, the most effective way to make it more probable is simply to remove that noise. The direction for that is `-ϵ`.

This insight is powerful because it transforms an impossibly abstract problem ("calculate the gradient of a log probability") into a concrete engineering task ("predict the noise").

The pioneering work that applied this to create large-scale generative models was **Noise Conditional Score Networks (NCSN)** [^3], which laid the direct foundation for what followed.

---

## Part 2: The Practical Recipe - DDPMs

**Denoising Diffusion Probabilistic Models (DDPMs)** can be understood as a practical, discrete-time recipe built on these score-based principles.

### The Forward Process: Creating the Noisy Data

DDPM defines a fixed forward process over `T` discrete timesteps. This is a **Variance Preserving (VP)** process defined by a noise schedule `βt`.
* The single-step transition is:
    $$
    x_t = \sqrt{1 - \beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon_{t-1} \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, I)
    $$
* We can jump to any step `t` directly from the original image `x₀`. Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$:
    $$
    x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon
    $$
This formulation is a crucial mechanism known as the **reparameterization trick**. It allows us to generate a sample $x_t$ in a way that is differentiable. Instead of sampling from a distribution whose parameters are learned (a stochastic step that blocks gradients), we sample a standard normal noise $\epsilon$ and deterministically transform it. This allows gradients to flow back through the sampling process, which is essential for training with gradient descent.

### The Reverse Process & Loss Function: Three Roads to the Same Destination

The goal is to learn a model $p_\theta(x_{t-1} | x_t)$ that approximates the true (but intractable) posterior $q(x_{t-1} | x_t, x_0)$. DDPMs achieve this by defining the reverse transition as a Gaussian whose variance is fixed ($\sigma_t^2 = \beta_t$) and whose mean $\mu_\theta(x_t, t)$ is learned by a neural network.

By deriving the mean from the predicted noise $\epsilon_\theta(x_t, t)$, we get:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$
This equation reveals that predicting the noise $\epsilon$ with a network $\epsilon_\theta$ directly provides the parameters for the reverse diffusion step. This insight dramatically simplifies the optimization objective, as seen in the perspectives below.

#### Perspective 1: The Probabilistic View (`L_vlb`)
The DDPM paper [^4] started from a rigorous probabilistic view, treating the model as a **Variational Autoencoder**. The goal is to maximize the Evidence Lower Bound (or Variational Lower Bound, `L_vlb`). This loss is a sum of KL Divergence terms that, for each step, measure how well the model’s predicted reverse step matches the true reverse step. This is the most complex but most theoretically pure formulation.

The full loss is composed of a term for the final step ($L_0$), a term for the noise prior ($L_T$), and a sum of terms for all intermediate steps ($L_{t-1}$):

$$\mathcal{L}_{vlb} = L_0 + L_T + \sum_{t=1}^{T} L_{t-1}$$

Each intermediate term $L_{t-1}$ is a KL divergence comparing the model's prediction to the true posterior:

$$L_{t-1} = D_{KL}(q(x_{t-1} | x_t, x_0) || p_\theta(x_{t-1} | x_t))$$

#### Perspective 2: The Simple & Practical View (`L_simple`)

The DDPM paper’s crucial finding was that optimizing the full `L_vlb` was less stable and produced worse samples than optimizing a simplified, re-weighted version. This simplified loss function is the one most commonly associated with DDPMs: a simple Mean Squared Error between the true and predicted noise.

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ || \epsilon - \epsilon_\theta(x_t, t) ||^2 \right]$$

This is the practical implementation of **Denoising Score Matching**: by learning to predict the noise `ϵ`, the model is implicitly learning the score function.

#### Perspective 3: The Explicit Score Matching View (`L_score`)

If we were to train a model `s_θ` to **explicitly** predict the score, the loss function would directly compare the true score with the model’s predicted score. This comes from the foundational theory of score-based models.

$$\mathcal{L}_{\text{score}} = \mathbb{E}_t \left[ \lambda(t) || \nabla_{x_t} \log p(x_t) - s_\theta(x_t, t) ||^2 \right]$$

This looks different, but it’s mathematically equivalent to `L_simple`. Given that the true score is:

$$\nabla_{x_t} \log p(x_t) = -\frac{\epsilon}{\sigma_t}$$

and we can define our model’s score prediction in terms of its noise prediction as:

$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sigma_t}$$

Substituting these into `L_score` with an appropriate weighting `λ(t)` recovers the simple noise prediction loss.

---

## Part 3: The Unified Framework - EDM's Breakthroughs

The **EDM (Elucidating Diffusion Models)** paper [^5] brilliantly unified these perspectives into a single, high-performance framework.

### Unifying Theory: The Denoiser IS the Score Estimator

EDM makes the connection between denoising and score matching explicit. It defines a **denoiser function** `D(x; σ)` that aims to predict the clean image `x₀` from a noisy input `x` with noise level `σ`. To make this denoiser work reliably, EDM introduces **principled preconditioning**:

```
    D(x; σ) = c_skip(σ) * x + c_out(σ) * F_θ(c_in(σ) * x; σ)
```
This structure makes the U-Net's (`F_θ`) job vastly simpler and more stable.

### The Reverse Process: Precision Engineering with ODE Solvers

EDM treats generation as solving an **ODE**, using advanced numerical solvers. Here is the step-by-step **deterministic Heun method**:

1.  **Setup:** Start with pure noise `x_0` at `σ_0`. Choose `N` steps, giving a schedule `σ_0, σ_1, ..., σ_N=0`.

2.  **Iterate for i = 0 to N-1:**
    a.  **Calculate Score at current point (`d_i`):**
        $$
        d_i = \frac{x_i - D(x_i; \sigma_i)}{\sigma_i}
        $$
    b.  **Predictor Step to a temporary point (`x_hat`):**
        $$
        x_{\text{hat}} = x_i + d_i \cdot (\sigma_{i+1} - \sigma_i)
        $$
    c.  **Calculate Score at the predicted point (`d_hat`):**
        $$
        d_{\text{hat}} = \frac{x_{\text{hat}} - D(x_{\text{hat}}; \sigma_{i+1})}{\sigma_{i+1}}
        $$
    d.  **Corrector Step to find the final `x_{i+1}`:**
        $$
        x_{i+1} = x_i + \frac{1}{2} (d_i + d_{\text{hat}}) \cdot (\sigma_{i+1} - \sigma_i)
        $$
3.  **Final Image:** After `N` steps, `x_N` is the final image.

---

## Part 4: The Complete Comparison & Practical Guide

Here is a final summary of the evolution from DDPM to EDM:

| Aspect                 | DDPM                                                       | EDM                                                                  |
| :--------------------- | :--------------------------------------------------------- | :------------------------------------------------------------------- |
| **Core Theory** | Probabilistic model, discrete time.                        | Unified score-based and probabilistic model, continuous time.        |
| **Training Model** | Simple U-Net `ϵ_θ(x_t, t)` predicts `ϵ`, implicitly learning the score. | Preconditioned denoiser `D(x; σ)` explicitly enables score calculation. |
| **Forward Params** | `T`, `beta_start`, `beta_end`, `scheduler`.                | `σ_min`, `σ_max`, `ρ`.                                                 |
| **Sampling Logic** | Stochastic ancestral sampling (1st-order Langevin-style).   | Solving an ODE (typically 2nd-order Heun).                           |
| **Determinism** | Inherently **stochastic**.                                 | Inherently **deterministic** (but can be made stochastic).           |
| **Speed/Steps** | Slow, requires many steps (`T`=1000+).                     | Fast, requires few steps (`N`=20-80).                                  |

### Practical Guide to the EDM Toolkit

* **The Solver:** Start with the **deterministic Heun solver**.
* **Number of Steps (`N`):** For previews, try `N = 20-25`. For high quality, `N = 40-80`.
* **The Timestep Schedule (`ρ`):** `ρ = 7` is an excellent default.
* **Stochasticity (`S_churn`):** Start with `S_churn = 0`.

---

## Part 5: Beyond Generation - What Your Trained Model Knows

A common question is: if I trained my model to just predict noise, can I use it for more complex tasks without retraining? The answer is a resounding **yes**. Your trained noise prediction model `ϵ_θ` is more powerful than it seems.

### Getting the Score Function for Free

You do not need to retrain your model to get the score function. The connection is direct: since your model learned to predict the noise, you can use its output to calculate the score at any timestep `t`. For a standard DDPM, the formula is:

$$
\text{score} = \nabla_{x_t} \log p(x_t) \approx -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

### The Compass vs. The Altimeter: Score vs. Log-Probability

It's crucial to distinguish between what's easy to get (the score) and what's hard (the log-probability value itself).

* **What you get (The Compass):** The score, `∇_{x_t} \log p(x_t)`, is the **gradient** of the log-probability. Your model gives you a perfect compass that, at any location `x_t` on the probability landscape, tells you exactly which way is "uphill".

* **What is hard to get (The Altimeter):** The log-probability value, `log p(x_t)`, is your exact altitude. Calculating this value is generally **intractable**.

For most practical applications like guiding generation, having the compass (the score) is exactly what you need.

### Application: Simulating Physical Systems with Learned Forces ⚛️

This connection between the score function and a guiding field is not just an analogy—it has profound implications in the physical sciences. As you noted, score-based models are now being used to simulate complex molecular systems.

The core idea is to draw a parallel between physics and statistics:

* In physics, a **force field** can often be described as the negative gradient of a potential energy function: $F = -\nabla U(x)$. The force pushes particles towards states of lower energy.
* In our model, the **score function** is the gradient of the log-probability function: $\text{score} = \nabla \log p(x)$. The score pushes samples towards states of higher probability.

Therefore, we can establish an equivalence where high probability corresponds to low energy: $\log p(x) \Leftrightarrow -U(x)$. This means our learned score function is a direct proxy for a physical force field!

$$
\text{Force} \quad F(x) = -\nabla U(x) \Leftrightarrow \nabla \log p(x) = \text{score}_\theta(x)
$$

This allows scientists to train a diffusion model on a dataset of known stable molecular conformations. The trained model learns the "score" or the implicit "force field" that holds those molecules together. They can then plug this learned force field into a **Langevin dynamics simulation**—a method for simulating how particles move under forces and random thermal fluctuations [^6].

The update rule for Langevin dynamics is essentially:
`next_position = current_position + force_field_drift + random_noise_kick`

This is precisely what the original NCSN and DDPM samplers do! They are a form of Langevin dynamics where the "force field" is provided by the learned score network. This powerful connection allows researchers to simulate and generate new, stable protein structures, design drugs, and explore molecular dynamics in ways that were previously computationally prohibitive.

---

### References

[^1]: Hyvärinen, A. (2005). *Estimation of Non-Normalized Statistical Models by Score Matching.* Journal of Machine Learning Research.
[^2]: Vincent, P. (2011). *A Connection Between Score Matching and Denoising Autoencoders.* Neural Computation.
[^3]: Song, Y., & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution.* Advances in Neural Information Processing Systems 32 (NeurIPS).
[^4]: Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* Advances in Neural Information Processing Systems 33 (NeurIPS).
[^5]: Karras, T., Aittala, M., Aila,T., & Laine, S. (2022). *Elucidating the Design Space of Diffusion-Based Generative Models.* Advances in Neural Information Processing Systems 35 (NeurIPS).
[^6]: Arts, M., Garcia Satorras, V., Huang, C. W., Zugner, D., Federici, M., Clementi, C., ... & van den Berg, R. (2023). Two for one: Diffusion models and force fields for coarse-grained molecular dynamics. Journal of Chemical Theory and Computation, 19(18), 6151-6159.
