
---
title: "Appendix"
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


