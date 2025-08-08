## Major Achievements

### 1. Developing a Generative AI Framework for All-Atom Protein Dynamics

![LD-FPG Figure](/images/LDFPG.png)

A significant challenge in computational biology is modeling the complete range of motion of proteins, as their function is intrinsically linked to their dynamics. To address this, I co-conceived and led the development of a novel generative AI framework, **Latent Diffusion for Full Protein Generation (LD-FPG)** [1].  

My specific contribution was central to this project: I designed and implemented the core machine learning pipeline, which integrates a **spectral graph neural network autoencoder** with a **latent diffusion model**. I also performed the rigorous validation of our method on a complex and medically relevant drug target — the **human dopamine D2 receptor (D2R)**, a G-protein-coupled receptor (GPCR).

Our work demonstrated, for the first time, that a latent diffusion model can generate complete, all-atom conformational ensembles directly from molecular dynamics data with high fidelity. The framework successfully reproduced not only the **global backbone architecture** but also the **distributions of side-chain dihedral angles** — dynamics essential for molecular recognition.

This achievement provides the scientific community with a powerful and computationally efficient tool to study the dynamics of challenging proteins, opening new avenues for structure-based drug design against previously intractable targets. In the spirit of open and reproducible science, I also curated and publicly released the extensive D2R molecular dynamics dataset and the complete LD-FPG codebase to facilitate further innovation by other researchers [1].

**Resources**  
- 📄 [Read the Paper](https://doi.org/10.48550/arXiv.2506.17064)  
- 💻 [View the Code on GitHub](https://github.com/adityasengar/LD-FPG/tree/main)  

---

**Cited Works**  
[1] Sengar, A., Hariri, A., Probst, D., Barth, P., & Vandergheynst, P. (2025). *Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings*. arXiv preprint [arXiv:2506.17064](https://doi.org/10.48550/arXiv.2506.17064).
