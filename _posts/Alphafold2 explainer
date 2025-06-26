# From AlphaFold2 to Boltz-2, Part 1: Deconstructing the AlphaFold 2 Revolution

**Author:** Aditya Sengar  
**Date:** Today

## In This Post: A Deep Dive into the Foundation

This post is the first in a multi-part series charting the rapid evolution of AI in structural biology. Before we can appreciate the paradigm shifts of diffusion models like AlphaFold 3 or the new frontiers of affinity prediction, we must first build a deep, foundational understanding of the model that started it all: **AlphaFold 2**.

In this deep dive, we will move beyond high-level diagrams to deconstruct the specific mechanisms that enabled the CASP14 breakthrough. A thorough grasp of these details is essential to truly appreciate the architectural choices and trade-offs made in the models that followed. We will explore its two-part engine:

- The **Evoformer** trunk, which established a rich, *bidirectional dialogue* between evolutionary data (MSA) and a geometric hypothesis (the pair representation).
- The deterministic **Structure Module**, which used the intricate, purpose-built Invariant Point Attention (IPA) to translate this blueprint into a 3D structure with hard-coded rotational physics.

**In short: We'll see how AlphaFold 2 introduced bidirectional reasoning between sequence and structure to achieve unprecedented accuracy, setting the stage for everything to come.**

## AlphaFold 2: A New Architecture for Biological Structure

When DeepMind presented the AlphaFold 2 results at the CASP14 competition in late 2020, the results were so astounding that it left the field of computational biology wondering how it was possible. When the paper and code were released, the answer was revealed: a brilliant deep learning architecture that fundamentally reinvented how to reason about protein structure.

The computational heart of this system consists of two main engines working in concert: the **Evoformer** and the **Structure Module**.

![AlphaFold 2 Architecture](AF2.png)
*Figure 1: High-level overview of the AlphaFold 2 architecture. The model takes Multiple Sequence Alignments (MSAs) and optional structural templates as input. The **Evoformer** block iteratively refines an MSA representation and a pair representation. The final pair representation guides the **Structure Module**, which uses Invariant Point Attention (IPA) to produce the 3D coordinates. (Alt-text: A flowchart showing inputs (MSA, templates) feeding into a central Evoformer block, which outputs to a Structure Module that generates the final 3D protein structure.)*

### A Primer on Self-Attention (The Transformer's Engine)

Before we can understand the Evoformer, we must understand its core computational tool: the **attention mechanism**. In essence, self-attention allows the model to look at other words in a sentence to understand any given word. For the phrase "the cat sat on the mat," the model learns that "cat" and "mat" are highly relevant to understanding "sat." For those interested in the mechanics, let's walk through a simplified numerical example.

> **📘 The Math of Self-Attention**
> 
> 1. **Input Embeddings:** We start with word vectors, represented as a matrix **X**.
> 
>    ```
>    X = [1  0]  % cat
>        [0  1]  % sat
>        [1  1]  % mat
>    ```
> 
> 2. **Project to Query, Key, and Value spaces:** The model learns matrices **W_Q, W_K, W_V** to project inputs into three roles: a **Query (Q)** ("What am I looking for?"), a **Key (K)** ("What info do I have?"), and a **Value (V)** ("What info I will share.").
> 
> 3. **Calculate Scores:** To update "sat", its Query vector **q_sat** is compared to all Key vectors via dot product to get a relevance score.
> 
>    ```
>    score(sat, cat) = [0, 1] · [0.5, 0] = 0
>    score(sat, sat) = [0, 1] · [0, 0.5] = 0.5
>    score(sat, mat) = [0, 1] · [0.5, 0.5] = 0.5
>    ```
> 
> 4. **Normalize:** The scores are converted to weights that sum to 1 using a softmax function.
> 
>    ```
>    Attention Weights ≈ [0.27, 0.365, 0.365]
>    ```
> 
> 5. **Produce Output:** The new representation for "sat" is the weighted sum of all **Value** vectors.
> 
>    ```
>    output_sat = (0.27 × [1, 0]) + (0.365 × [0, 1]) + (0.365 × [1, 1])
>               = [0.635, 0.73]
>    ```
> 
> The result, [0.635, 0.73], is a new vector for "sat" that has absorbed context from its neighbors.

### Building the Blueprint: Inputs and Representations

The classical approach to using a **Multiple Sequence Alignment (MSA)** was to distill it into a 2D contact map before analysis. AlphaFold 2's paradigm shift was to **avoid this premature distillation**. Instead, it processes two powerful representations simultaneously, allowing them to have a rich "dialogue."

#### A Detailed Evolutionary Profile (MSA Representation)

The model processes a representation of size (N_seq × N_res), where N_seq is the number of sequences and N_res is the number of residues. For each residue, the model creates a rich input vector answering several questions: What is this residue? What is its evolutionary context (i.e., the statistical profile of amino acids seen at this position in its cluster)? And what deletion information is present? This vector is then processed to produce the internal MSA representation (c_m=256 channels) that is refined within the Evoformer.

#### The Geometric Blueprint (Pair Representation)

This is a square matrix of size (N_res × N_res) with 128 channels (c_z=128) that evolves into a detailed geometric blueprint. It is initialized by combining two pieces of information.

First, it encodes basic information about which amino acids are at positions i and j.

```
z_ij^pairs = Linear(one_hot(AA_i)) + Linear(one_hot(AA_j))
```

Second, it adds an embedding that represents the 1D sequence separation between them, d_ij = i - j.

```
z_ij^initial = z_ij^pairs + Embedding(d_ij)
```

This initial grid is like digital graph paper with basic relationships sketched out; the Evoformer's job is to fill it with sophisticated 3D information.

#### Structural Templates: An Editable Head Start

If homologous structures exist, AlphaFold 2 leverages their geometry not as a rigid scaffold, but as a set of editable hints integrated in two ways:

1. **A 2D Geometric Hint:** A distogram (map of Cβ-Cβ distances) is extracted from each template. The model then uses attention to look at all available templates for each residue pair (i, j) and takes a weighted average of the geometric hints, adding the result to its own pair representation.

2. **A 1D "Advisor":** In an elegant move, the known backbone torsion angles from the templates are embedded and **concatenated directly to the MSA representation**, making N_seq = N_clust + N_templ. This treats the template not as a static map, but as an expert participant in the evolutionary dialogue, allowing its structural information to directly bias the MSA attention mechanisms.

### The Evoformer: A Reasoning Engine for Co-evolution and Geometry

The Evoformer's power comes from a block of operations repeated 48 times. Each block forces the MSA and Pair representations to communicate and refine each other.

#### Part 1: The MSA Stack and Axial Attention

The block first focuses on the MSA representation. To handle the large (N_seq × N_res) matrix, it "factorizes" attention into two cheaper, sequential steps: row-wise attention (finding relationships within one sequence) and column-wise attention (comparing evolutionary evidence across sequences for one position). This is followed by a standard MLP transition layer.

#### Part 2: The Pair Stack and Triangular Updates

Next, the block turns to the pair representation (z) to enforce geometric consistency. It uses novel "triangular" operations based on the logic that if you know about pairs (i, k) and (k, j), you can infer properties of the "closing" pair (i, j).

![Triangular Operations](triangle_update.png)
*Figure 2: The triangular operations at the heart of the Evoformer. A triplet of residues (i, j, k) corresponds to a set of edges. Information on edge (i,j) is updated by incorporating information from the other two edges in the triangle, first via gated multiplicative updates and then via attention. (Alt-text: A diagram showing three nodes i, j, k forming a triangle. An arrow indicates that information from edges (i,k) and (j,k) are used to update the representation for edge (i,j).)*

**Triangular Multiplicative Update.** This is a fast, non-attention method of passing information. For each triangle of residues (i, j, k), information from edges z_ik and z_jk is combined to update edge z_ij. A crucial **gating mechanism** is used, which acts like a faucet, allowing the network to dynamically control how much of the new information flows into the existing representation based on its current state.

**Triangular Self-Attention.** This is a true attention mechanism where an edge (i, j) acts as a "query" to gather information from other edges that share a node with it. The key innovation is how it scores relevance. The score includes a learned **bias** that comes directly from the triangle's closing edge, z_jk.

> **📘 Triangular Attention Score**
> 
> The score for how edge (i,j) attends to edge (i,k) is calculated as:
> 
> ```
> score_ijk = (q_ij · k_ik) / √d_k + Linear(z_jk)
> ```
> 
> Here, q_ij and k_ik are query and key vectors derived from their respective pair representations. The addition of the bias term from z_jk is the critical step.

This allows the model to ask a sophisticated question: *"How relevant is edge (i,k) to my query (i,j), given my current belief about the geometry of the third edge (j,k)?"*

#### Part 3: The Communication Hub

The true innovation is how the two stacks communicate within each block.

- **Path 1: From MSA to Pairs:** Co-evolutionary signals are injected into the geometry via an **Outer Product Mean**. This operation averages correlations found between columns i and j across all sequences in the MSA and adds the result to the pair representation z_ij.

- **Path 2: From Pairs to MSA:** The current geometric hypothesis guides the MSA analysis via an **Attention Bias**. During MSA row-wise attention, the score between residues i and j is biased by a value derived from z_ij. If the model believes i and j are close in 3D space, it forces the MSA attention to focus on their relationship, guiding the search for confirming evolutionary signals.

### The Structure Module: From Blueprint to 3D Coordinates

After 48 Evoformer iterations, the refined single (**s_i**) and pair (**z_ij**) representations are passed to the **Structure Module**. This module translates the abstract blueprint into a physical 3D structure using a custom transformer block called **Invariant Point Attention (IPA)**.

The core idea is to give each residue its own local reference frame T_i = (R_i, **t_i**): a rotation and an origin. Think of this as giving each residue its own *personal compass and position tracker*. All reasoning is done relative to these local frames.

What makes IPA "invariant"? Each residue learns to place several "query" points in its local frame—like *virtual attachment points*. The attention score between two residues is calculated based on the 3D distances between these points in global space. Because physical distance is invariant to rotation and translation, the score naturally respects 3D physics. The final attention score combines three signals:

> **📘 Invariant Point Attention Score**
> 
> For head h, the score for residue i attending to residue j is:
> 
> ```
> score^(h)_ij = q^(h)_i^T k^(h)_j + b^(h)_ij - (1/σ_h²) Σ_{k,ℓ} w^(h)_{kℓ} d_{ij,kℓ}²
>                ︸─────Abstract Match─────︸   ︸Blueprint Bias︸   ︸──────3D Proximity──────︸
> ```

Conceptually, these terms represent:

1. **Abstract Match:** A "chemical compatibility" check derived from the single representation.
2. **Blueprint Bias:** A direct bias from the final pair representation **z_ij**, importing the Evoformer's wisdom.
3. **3D Proximity:** A penalty for large distances between the virtual attachment points based on the current, partially folded structure.

Based on these scores, each residue "votes" on how every other residue should move. These votes are aggregated into a single movement command—a small rotation and translation "nudge". Applying these nudges iteratively over 8 cycles allows the protein to fold gradually and resolve complex geometric constraints.

> **💡 A Pocket Glossary of AlphaFold Metrics**
> 
> AlphaFold 2 doesn't just predict a structure; it predicts its own accuracy with several key metrics.
> 
> - **pLDDT (predicted Local Distance Difference Test):** A per-residue score from 0-100 indicating confidence in the local backbone prediction. High pLDDT (>90) means high accuracy.
> - **PAE (Predicted Aligned Error):** A 2D plot showing the expected error in the position of residue Y if the structure is aligned on residue X. Low PAE between domains means the model is confident in their relative orientation.
> - **FAPE (Frame Aligned Point Error):** The main loss function used for training the structure module. It is a torsion-invariant measure of the error between the predicted and true structures.

### Conclusion and Next Steps

AlphaFold 2's architecture was a landmark achievement, introducing a new way of thinking about biological structure prediction. Its core innovations—the bidirectional reasoning of the Evoformer, the enforcement of geometric constraints with triangular updates, and the physically-aware IPA module—created a system that could solve protein structures to near-experimental accuracy.

However, its complexity and purpose-built modules paved the way for a radical simplification and generalization. In **Part 2 of this series**, we will explore how DeepMind rebuilt its own revolution with AlphaFold 3, replacing the rigid Structure Module with the flexible, generative power of diffusion models and shifting the core philosophy of the entire architecture.
