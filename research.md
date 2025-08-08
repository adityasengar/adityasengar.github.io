## Major Achievements

---

## . Developing a Generative AI Framework for All-Atom Protein Dynamics

![LD-FPG Figure](/images/LDFPG.png)

A significant challenge in computational biology is modeling the complete range of motion of proteins, as their function is intrinsically linked to their dynamics. To address this, I co-conceived and led the development of a novel generative AI framework, **Latent Diffusion for Full Protein Generation (LD-FPG)** [1].  

My specific contribution was central to this project: I designed and implemented the core machine learning pipeline, which integrates a **spectral graph neural network autoencoder** with a **latent diffusion model**. I also performed the rigorous validation of our method on a complex and medically relevant drug target — the **human dopamine D2 receptor (D2R)**, a G-protein-coupled receptor (GPCR).

Our work demonstrated, for the first time, that a latent diffusion model can generate complete, all-atom conformational ensembles directly from molecular dynamics data with high fidelity. The framework successfully reproduced not only the **global backbone architecture** but also the **distributions of side-chain dihedral angles** — dynamics essential for molecular recognition.

This achievement provides the scientific community with a powerful and computationally efficient tool to study the dynamics of challenging proteins, opening new avenues for structure-based drug design against previously intractable targets. In the spirit of open and reproducible science, I also curated and publicly released the extensive D2R molecular dynamics dataset and the complete LD-FPG codebase to facilitate further innovation by other researchers [1].

**Resources**  
- 📄 [Read the Paper](https://doi.org/10.48550/arXiv.2506.17064)  
- 💻 [View the Code on GitHub](https://github.com/adityasengar/LD-FPG/tree/main)  
- 📊 [View the Dataset on Zenodo](https://zenodo.org/records/15479781)


---
## Major Achievements at Imperial — DNA Nanotechnology & Biomolecular Systems

![LD-FPG Figure](/images/oxdna.jpg)


I combined method development, simulations, and theoretical analysis to advance our understanding of DNA nanotechnology using the oxDNA framework. This work, showcased at five international conferences and through several posters, tackled fundamental challenges in DNA design and function.

### 1. Free-Energy Mapping of Four-Way DNA Junctions (with Bulges)

In [1], I developed enhanced sampling techniques based on oxDNA to map the free-energy landscape of four-way DNA junctions—crucial intermediates in strand displacement reactions. My simulations revealed a surprising result: introducing two bulges actually **destabilized** the structure due to increased entropy penalties. This finding challenges conventional design assumptions and provides valuable insights for improving DNA nanostructure robustness.

### 2. Kinetic Proofreading in Nonenzymatic DNA Strand Displacement

In [2], I applied oxDNA to model a **kinetic proofreading (KP)** mechanism in nonenzymatic DNA strand-displacement systems. This work provided quantitative reaction-rate estimates and showed that operating under out-of-equilibrium conditions significantly enhances molecular recognition—especially in discriminating single-nucleotide mismatches. These insights can inform highly specific applications such as SNP detection and DNA-based diagnostics.

### 3) oxDNA primer: when to use it, how to simulate, how to interpret

In [3], we produced a comprehensive **primer and tutorial** on the oxDNA coarse-grained DNA model. The paper explains model variants (oxDNA1/oxDNA2), force-field details, sequence-dependent parameterization, and mapping to experimental units. It also walks through simulation protocols—**Langevin dynamics**, **Monte Carlo**, and advanced accelerated sampling methods such as **Virtual Move Monte Carlo (VMMC)**—demonstrating how these can speed up equilibration of large, strongly interacting DNA structures.  

We included worked examples where VMMC efficiently explores conformations of DNA origami and multi-strand assemblies, and showed how to combine it with umbrella sampling to obtain free-energy profiles. The paper also details analysis workflows for computing structural observables, thermodynamic quantities, and reaction pathways, providing a benchmark reference for reproducible oxDNA studies.

#### Additional method (independent of [3])  
I developed a **fast-kinetics sampling workflow** in oxDNA to study dynamic events such as DNA bubble formation; this approach has since been adopted by groups at the University of Cambridge and MIT (submitted).

---

### Resources  

- 📄 [Paper 1: “Overcoming the speed limit of four-way DNA branch migration with bulges in toeholds” (bioRxiv, 2023)](https://www.biorxiv.org/content/10.1101/2023.05.15.540824v1)  
- 📄 [Paper 2: “Kinetic Proofreading Can Enhance Specificity in a Nonenzymatic DNA Strand Displacement Network” (J. Am. Chem. Soc., 2024)](https://pubs.acs.org/doi/full/10.1021/jacs.3c14673)  
- 📄 [Paper 3: “A Primer on the oxDNA Model of DNA…” (Frontiers in Molecular Biosciences, 2021)](https://www.frontiersin.org/journals/molecular-biosciences/articles/10.3389/fmolb.2021.693710/full)  


---


## Developing a "Molecule-to-Reactor" Computational Pipeline to Advance Catalytic Engineering

![Multiscale Catalyst Modeling](/images/reactor.jpg)

My research has focused on overcoming a critical barrier in chemical engineering: the inability of traditional models to bridge the vast gap between molecular-level surface events and macroscopic reactor performance. To address this, I established a foundational, particle-based computational fluid dynamics (CFD) framework that provides a unified, multiscale view of catalytic processes. The core innovation of this framework was its ability to explicitly resolve essential small-scale physics—such as diffusion and surface reactions—while seamlessly operating at the larger scales of industrial reactors [2]. This model was rigorously designed and validated to switch between reaction, diffusion, and convection-dominated regimes, a versatility that was previously out of reach for real-time simulations. I further advanced this platform by integrating complex, nonlinear surface reaction kinetics, enabling highly accurate and scalable simulations of multicomponent systems and their mass transfer fluxes under non-equilibrium conditions [1].

Building on this robust modeling platform, I directed its predictive power toward solving a pressing industrial problem: **catalyst deactivation in the production of sustainable aviation fuels**, a primary cause of economic loss and process inefficiency. I developed and compared two distinct theoretical models that quantitatively connect molecular-scale deactivation mechanisms with observable reactor-scale performance degradation [3]. By applying these models to analyze industrial alkylation reaction data, my investigation yielded a crucial breakthrough: the identification of a previously unknown molecular compound that acts as a potent deactivating agent. Furthermore, the models revealed that the deactivation rate is highly sensitive to proton mobility on the catalyst surface. This analysis provided a concrete, physics-based strategy for extending catalyst lifetime by optimizing these proton interactions, offering a clear path to minimize operational downtime and improve the economic viability of sustainable fuel production [4].

Collectively, this work constitutes a complete "molecule-to-reactor" predictive pipeline, demonstrating a clear progression from **fundamental method development** to **high-impact industrial application**. The multiscale model was selected as the **cover article of Chemical Engineering Science** [2], and the deactivation research earned me an invitation to present at the **Faraday Discussions** [5], a leading international forum for groundbreaking research. These results have been shared at multiple international conferences and directly with industry leaders, including a presentation at **Albemarle**, fostering collaborations that bridge academic theory and industrial practice.


**Resources**  

[1] [Sengar, A., Kuipers, J. A. M., Van Santen, R. A., & Padding, J. T. (2017). *Particle-based modeling of heterogeneous chemical kinetics including mass transfer*. Physical Review E, 96(2), 022115.](https://doi.org/10.1103/PhysRevE.96.022115)  

[2] [Sengar, A., Kuipers, J. A. M., van Santen, R. A., & Padding, J. T. (2019). *Towards a particle based approach for multiscale modeling of heterogeneous catalytic reactors*. Chemical Engineering Science, 198, 184-197.](https://www.sciencedirect.com/science/article/pii/S0009250918307607)

[3] [Sengar, A., Van Santen, R. A., Steur, E., Kuipers, J. A., & Padding, J. (2018). *Deactivation kinetics of solid acid catalyst with laterally interacting protons*. ACS Catalysis, 8(10), 9016-9033.](https://pubs.acs.org/doi/10.1021/acscatal.8b01511) 

[4] [Sengar, A., Van Santen, R. A., & Kuipers, J. A. (2020). *Deactivation kinetics of the catalytic alkylation reaction*. ACS Catalysis, 10(13), 6988-7006.](https://pubs.acs.org/doi/10.1021/acscatal.0c00932)

[5] [Van Santen, R. A., Sengar, A., & Steur, E. (2018). *The challenge of catalyst prediction*. Faraday Discussions, 208, 35-52.](https://pubs.rsc.org/en/content/articlelanding/2018/fd/c7fd00208d)

