---
title: "Appendix (From Alphafold 2 to Boltz-2)"
series: "From AlphaFold2 to Boltz-2: The Protein Prediction Revolution"
author: "Aditya Sengar"
part: 4
date: 2025-06-26
---

# A. A Deeper Look at Self-Attention

The core computational engine of the Transformer architecture is the **self-attention mechanism**. Its purpose is to generate a new, contextually-aware representation for each element in an input sequence. It achieves this by systematically weighing the influence of every other element in the sequence, allowing the model to dynamically capture relationships between them.

To illustrate this, we will use a relevant example from protein biology. Imagine the model is processing a peptide sequence and needs to learn the electrostatic interaction between a positively charged Lysine (K) and a negatively charged Aspartate (D), which can form a salt bridge. We will walk through how self-attention enables the model to learn this relationship, focusing on updating the representation for Lysine.

1. **Input Embeddings:**  
   We begin with an input matrix $X$, where each row $x_i$ is a vector (an "embedding") representing a single amino acid residue. These initial embeddings are learned and encode fundamental properties. For our simplified 3-residue sequence (Alanine, Lysine, Aspartate), let's assume a 4-dimensional embedding ($d_{model}=4$). The dimensions could conceptually represent properties like size, hydrophobicity, positive charge, and negative charge.

   $$
   X =
   \begin{bmatrix}
   x_{\text{Ala}} \\\\
   x_{\text{Lys}} \\\\
   x_{\text{Asp}}
   \end{bmatrix}
   =
   \begin{bmatrix}
   0.2 & 0.1 & 0.0 & 0.0 \\\\
   0.8 & 0.4 & 1.0 & 0.0 \\\\
   0.5 & 0.3 & 0.0 & 1.0
   \end{bmatrix}
   $$
   *(Ala: small, neutral; Lys: large, strong positive charge; Asp: medium, strong negative charge)*

2. **Projecting to Query, Key, and Value Spaces:**  
   The model learns three distinct weight matrices ($W_Q$, $W_K$, $W_V$) to project each input embedding into three different roles. This projection allows the model to extract specific features relevant for establishing relationships.

   - A **Query** vector ($q$): Asks, "Given my properties, who should I pay attention to?" For Lysine, its query vector will be trained to effectively ask, "I have a positive charge; where is a corresponding negative charge?"
   - A **Key** vector ($k$): States, "Here are the properties I offer for others to query." For Aspartate, its key vector will be trained to state, "I possess a negative charge."
   - A **Value** vector ($v$): Contains the information of that residue to be passed on to others. If a residue receives high attention, it will heavily incorporate this value vector.

   Let's define our learned weight matrices, which project from $d_{model}=4$ to a smaller dimension $d_k=d_v=2$:

   $$
   W_Q = \begin{bmatrix} 0 & 1 \\\\ 0 & 1 \\\\ 1 & 0 \\\\ 0 & 0 \end{bmatrix},\quad
   W_K = \begin{bmatrix} 1 & 0 \\\\ 0 & 1 \\\\ 0 & 0 \\\\ 1 & 0 \end{bmatrix},\quad
   W_V = \begin{bmatrix} 0 & 1 \\\\ 1 & 0 \\\\ 1 & 1 \\\\ 0 & 1 \end{bmatrix}
   $$

   We calculate the $Q$, $K$, and $V$ matrices by multiplying our input $X$ with these weight matrices:

   $$
   \begin{align*}
   Q &= X W_Q = \begin{bmatrix} 0.2 & 0.3 \\\\ 0.8 & 1.2 \\\\ 0.5 & 0.8 \end{bmatrix} \\\\
   K &= X W_K = \begin{bmatrix} 0.2 & 0.1 \\\\ 0.8 & 0.4 \\\\ 1.5 & 0.3 \end{bmatrix} \\\\
   V &= X W_V = \begin{bmatrix} 0.1 & 0.4 \\\\ 1.2 & 1.8 \\\\ 0.3 & 1.8 \end{bmatrix}
   \end{align*}
   $$

3. **Calculate Raw Scores (Attention Logits):**  
   To determine how much attention Lysine should pay to other residues, we take its query vector, $q_{\text{Lys}} = [0.8, 1.2]$, and compute its dot product with the key vector of every residue in the sequence. The dot product is a measure of similarity; a higher value signifies greater relevance.

   $$
   \begin{align*}
   \text{score}(\text{Lys, Ala}) &= [0.8, 1.2] \cdot [0.2, 0.1] = (0.8)(0.2) + (1.2)(0.1) = 0.28 \\\\
   \text{score}(\text{Lys, Lys}) &= [0.8, 1.2] \cdot [0.8, 0.4] = (0.8)(0.8) + (1.2)(0.4) = 1.12 \\\\
   \text{score}(\text{Lys, Asp}) &= [0.8, 1.2] \cdot [1.5, 0.3] = (0.8)(1.5) + (1.2)(0.3) = 1.56
   \end{align*}
   $$

   As designed, the score for the interacting Lys-Asp pair is the highest, indicating a strong potential relationship.

4. **Scale and Normalize (Softmax):**  
   The scores $[0.28, 1.12, 1.56]$ are first scaled. This is a critical step for stabilizing training. As the dimension of the key vectors ($d_k$) increases, the variance of the dot products also increases, which can push the softmax function into saturated regions with extremely small gradients. To counteract this, we scale the scores by dividing by $\sqrt{d_k}$. Here, $d_k=2$, so we scale by $\sqrt{2} \approx 1.414$.

   $$
   \text{Scaled Scores} = \left[ \frac{0.28}{\sqrt{2}}, \frac{1.12}{\sqrt{2}}, \frac{1.56}{\sqrt{2}} \right] \approx [0.198, 0.792, 1.103]
   $$

   Next, these scaled scores are passed through a softmax function, which converts them into a probability distribution of positive weights that sum to 1. These are the final **attention weights** ($\alpha$):

   $$
   \begin{align*}
   \alpha_{\text{Lys}} &= \text{softmax}([0.198, 0.792, 1.103]) \\\\
   &= \left[ \frac{e^{0.198}}{e^{0.198} + e^{0.792} + e^{1.103}}, \frac{e^{0.792}}{e^{0.198} + e^{0.792} + e^{1.103}}, \frac{e^{1.103}}{e^{0.198} + e^{0.792} + e^{1.103}} \right] \\\\
   &= \left[ \frac{1.22}{1.22 + 2.21 + 3.01}, \frac{2.21}{1.22 + 2.21 + 3.01}, \frac{3.01}{1.22 + 2.21 + 3.01} \right] \\\\
   &\approx [0.19, 0.34, \mathbf{0.47}]
   \end{align*}
   $$

   The weights clearly show that to update its representation, Lysine should draw 47% of its new information from Aspartate.

5. **Produce the Final Output:**  
   The new, context-aware representation for Lysine, denoted $z_{\text{Lys}}$, is the weighted sum of all the **Value** vectors in the sequence, using the attention weights we just calculated.

   $$
   \begin{align*}
   z_{\text{Lys}} &= (\alpha_{\text{Lys,Ala}} \times v_{\text{Ala}}) + (\alpha_{\text{Lys,Lys}} \times v_{\text{Lys}}) + (\alpha_{\text{Lys,Asp}} \times v_{\text{Asp}}) \\\\
   &= (0.19 \times [0.1, 0.4]) + (0.34 \times [1.2, 1.8]) + (0.47 \times [0.3, 1.8]) \\\\
   &= [0.019, 0.076] + [0.408, 0.612] + [0.141, 0.846] \\\\
   &= [0.568, 1.534]
   \end{align*}
   $$

The final output vector for Lysine, $z_{\text{Lys}} = [0.568, 1.534]$, has now powerfully incorporated information from Aspartate, effectively encoding their likely interaction directly into its features. This new vector is then passed to the next layer of the Transformer.

---

### Multi-Head Attention

A single attention calculation (as shown above) allows the model to focus on one type of relationship at a time (e.g., electrostatic interactions). However, protein structures are governed by many concurrent interactions (hydrophobic interactions, hydrogen bonds, steric constraints, etc.).

**Multi-Head Attention** addresses this limitation by performing the entire self-attention process multiple times in parallel. Each parallel run is called a "head," and each head has its own independently learned weight matrices ($W_Q^{(i)}$, $W_K^{(i)}$, $W_V^{(i)}$). This is analogous to having multiple specialist "heads" analyzing the sequence simultaneously:

- **Head 1** might learn to identify salt bridges.
- **Head 2** might focus on hydrophobic interactions.
- **Head 3** might track backbone hydrogen bond patterns.
- etc.

The resulting output vectors from all heads ($z^{(1)}, z^{(2)}, \dots$) are concatenated and then passed through a final linear projection matrix, $W_O$, to produce the final, unified output. This creates a representation that is simultaneously rich with information about many different kinds of structural and chemical relationships.

---



# B. The Physics and Mathematics of Conditional Diffusion

The Diffusion Module at the heart of AlphaFold 3 is a concept of profound elegance, drawing its power from the principles of non-equilibrium thermodynamics and statistical physics. The core idea is to master the creation of order by first mastering its destruction. Imagine a drop of ink in water—a low-entropy, high-information state. Over time, random molecular motion (Brownian motion) causes it to spread into a uniform, high-entropy state of maximum disorder. The diffusion model asks: can we learn the exact, time-reversed path to command every single ink molecule to re-form into the original, complex droplet?

AlphaFold 3 applies this very principle to molecular structures. It learns to construct a perfectly folded biomolecular complex by first understanding, with mathematical precision, the process of its dissolution into a featureless, random cloud of atoms.

## The Forward Process: A Controlled Descent into Chaos

The “forward process” is not simply about adding noise; it’s a carefully calibrated Markov chain that systematically degrades a known structure. We start with a structure from the training data, $\mathbf{x}_0$. At each of many successive timesteps ($t = 1, 2, \dots, T$), we add a small amount of Gaussian noise, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$
\mathbf{x}_t = \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1} \;+\; \sqrt{\beta_t}\,\boldsymbol{\epsilon}.
$$

Here, the **variance schedule**, $\{\beta_t\}$, is crucial. It’s a set of small, pre-defined constants that typically increase with $t$. This means the process starts by adding very subtle noise and becomes progressively more aggressive, ensuring a smooth and stable transition into an “atomic fog,” $\mathbf{x}_T$, which is indistinguishable from pure Gaussian noise.

A key mathematical property of this process is that we can sample the state at any arbitrary timestep $t$ directly from the original structure $\mathbf{x}_0$, without iterating. By defining

$$
\alpha_t = 1 - \beta_t
\quad\text{and}\quad
\bar{\alpha}_t = \prod_{i=1}^t \alpha_i,
$$

we get the closed-form expression:

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 \;+\; \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}.
$$

This formula is the bedrock of the training process, allowing for efficient, parallelized learning.

---

## The Reverse Process: The Sculptor’s Learned Intuition

This is where the learning occurs. The goal of the reverse process is to undo the forward process, taking a step back from a noisy state $x_{t}$ to a slightly cleaner state $x_{t-1}$. This requires learning the posterior distribution $p(x_{t-1} \mid x_t)$. While this distribution is intractable to compute directly, it becomes tractable when conditioned on the original data $\mathbf{x}_0$.

The breakthrough of diffusion models was to train a powerful neural network, $\boldsymbol{\epsilon}_\theta$, to approximate this reverse step. Instead of directly predicting the cleaned-up coordinates, the network is trained on a simpler, more stable objective: **predicting the noise vector $\boldsymbol{\epsilon}$ that was added at that timestep**.

The training objective is:

$$
L_{\text{diffusion}}
= \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}
\bigl[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t,\,t)\|^2\bigr].
$$

If the network can perfectly predict the noise that was added to create $x_t$, we can use this prediction to mathematically reverse the step and estimate the cleaner structure $x_{t-1}$. This is the sculptor learning their craft—by repeatedly observing how a masterpiece is turned into a raw block, they develop a perfect intuition for which piece of “marble dust” ($\boldsymbol{\epsilon}$) to remove at any stage of the carving process.

---

## Conditional Diffusion: The All-Important Blueprint

Without guidance, the reverse process would simply generate a random, chemically plausible molecule. To build a *specific* complex, the process must be **conditioned**. The denoising network doesn’t just receive the noisy atom cloud $\mathbf{x}_t$; it receives a second, crucial input: a “blueprint” that defines the target structure. In AlphaFold 3, this blueprint is the final, incredibly rich **pair representation** ($\mathbf{z}_{\text{trunk}}$) generated by the Pairformer.

It’s helpful to think of this blueprint not just as instructions, but as a **guiding force field** or a **potential energy landscape**. The Pairformer’s output defines the “valleys” of low energy (representing stable hydrogen bonds, favorable hydrophobic contacts) and the “hills” of high energy (representing steric clashes or improper geometry). The diffusion process is then a simulation of a particle navigating this landscape, simultaneously trying to find its way “downhill” into an energy minimum while also shedding its random, high-energy thermal motion (the noise).

Let’s revisit our sculptor with this deeper understanding. Imagine the model is predicting the binding of a drug to a protein:

- **The Scene:**  
  The Diffusion Module observes a jumbled cloud of protein and drug atoms ($\mathbf{x}_t$). In the context of the energy landscape defined by the blueprint, this is a high-energy, unfavorable conformation. The atoms are sitting on “hills” and in awkward positions.

- **The Blueprint’s Guidance:**  
  The blueprint ($\mathbf{z}_{\text{trunk}}$) provides the gradient of the landscape. It indicates the direction of “steepest descent” towards a better structure. Its encoded relationships effectively say:  
  - “There is a deep energy valley if this drug’s oxygen atom moves 2 Å closer to that protein’s nitrogen atom. The gradient points this way.”  
  - “There is a high-energy hill of steric repulsion if this bulky ring gets any closer to that protein side chain. The gradient points away from there.”

- **The Guided Chisel Stroke:**  
  The neural network, $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,\,t,\,\mathbf{z}_{\text{trunk}})$, now performs a dual task. It looks at the jumbled atoms and the guiding landscape. It predicts a noise vector that, when subtracted, achieves two goals at once:  
  1. It removes a component of the random, non-directional noise.  
  2. It nudges the atoms in a direction that follows the gradient, moving the entire system “downhill” into a more energetically favorable state.

By repeating this **guided denoising step** hundreds of times, the model doesn’t just create a plausible structure; it meticulously settles the atoms into a deep minimum on the energy landscape defined by the Pairformer. The final result is a single, coherent 3D structure that precipitates out of the initial chaos, perfectly sculpted according to the laws of physics and chemistry encoded in its instructions.  

