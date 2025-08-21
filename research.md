---
layout: page
title: Research
permalink: /research/
---

# Research

## About my research

As a computational physicist and post‑doctoral researcher at Imperial College London, I work at the interface of **chemical engineering** and **bioengineering**. My interests range from **multiscale modelling of catalytic reactors** to **nucleic‑acid systems**.  Under the guidance of Dr Thomas Ouldridge I develop and apply **oxDNA**, a coarse‑grained model of DNA that captures kinetics and thermodynamics of strand displacement reactions.  Earlier, during my PhD at Eindhoven University of Technology, I focused on **particle‑based simulation techniques** for heterogeneous catalysis in micro‑reactors and macro‑reactors.

## DNA strand displacement and bioengineering

![Abstract DNA helix]({{ site.baseurl }}/images/research_dna.png){: .project-image }

My current work explores how to control and predict DNA strand displacement, a key mechanism in synthetic biology and molecular computing.  Below are some of the projects and publications in this area:

<details>
<summary><strong>Kinetic proofreading for single‑nucleotide discrimination</strong> – submitted to Nature Nanotechnology</summary>
<p>
This project studies how **kinetic proofreading** can improve the specificity of **non‑enzymatic DNA strand displacement** networks.  By introducing additional energy‑dissipating steps, the reaction pathway becomes more sensitive to base‑pair mismatches, allowing circuits to reliably distinguish single nucleotide differences.  Our simulations demonstrate that a properly tuned proofreading scheme dramatically reduces false‑positive signalling.  

<a href="#" target="_blank">Read the paper</a>
</p>
</details>

<details>
<summary><strong>Overcoming the speed limit of four‑way branch migration</strong> – submitted to JACS</summary>
<p>
Four‑way branch migration is often limited by slow exchange between strands.  We investigate how **bulge inserts** in toeholds can accelerate branch migration without compromising accuracy.  Coarse‑grained simulations with the oxDNA model show that small bulges lower the energy barrier for strand exchange, speeding up migration by orders of magnitude.  These findings may enable faster DNA‑based walkers and motors.  

<a href="#" target="_blank">Read the paper</a>
</p>
</details>

<details>
<summary><strong>Allosteric modulation of toehold‑mediated displacement</strong> – in preparation</summary>
<p>
We explore how **topological constraints** and **allosteric interactions** can regulate toehold‑mediated strand displacement.  By designing toeholds that respond to structural changes elsewhere on the strand, it is possible to create controllable logic gates where input at one site activates or suppresses displacement at another.  This work illustrates the power of **mechanical coupling** in DNA circuits and proposes designs for responsive molecular switches.
</p>
</details>

<details>
<summary><strong>Handhold‑mediated displacement and free‑energy landscapes</strong> – in preparation</summary>
<p>
In handhold‑mediated reactions, the invading strand binds away from the toehold and slides into position via slithering motions.  Using oxDNA simulations we map the **free‑energy landscape** of this process and identify key intermediates.  Understanding these landscapes helps rationally design handhold lengths and sequences that optimise reaction rates and minimise undesirable secondary structures.
</p>
</details>

<details>
<summary><strong>Deep learning for oxDNA kinetics</strong> – in preparation</summary>
<p>
Simulating hybridisation kinetics with oxDNA is computationally expensive.  Here we train a **deep neural network** on a library of oxDNA trajectories to predict **reaction rates** from sequence and structural features.  The model captures subtle dependencies between base composition, toehold length, and folding patterns, enabling rapid screening of sequence variants without running new simulations.
</p>
</details>

## Catalysis and reaction engineering

![Particle‑based reactor simulation]({{ site.baseurl }}/images/research_reactor.png){: .project-image }

During my doctoral studies, I developed **particle‑based methods** for simulating chemical reactors, focusing on how catalyst deactivation and mass transfer influence performance.  Key contributions include:

<details>
<summary><strong>Deactivation kinetics of the catalytic alkylation reaction</strong> – ACS Catalysis (2020)</summary>
<p>
We developed a kinetics theory for **solid‑acid catalysed alkylation** of isobutane with propylene or butene.  The analysis links reaction networks, catalyst deactivation and residence‑time distributions in both continuous stirred‑tank and plug‑flow reactors.  We derive conditions under which deactivation is slow and show how self‑alkylation reactions and the mobility of the reaction zone influence deactivation times.  The work provides practical guidelines for designing reactors with long catalyst lifetimes.

<a href="https://pubs.acs.org/doi/10.1021/acscatal.0c00932" target="_blank">View the paper</a>
</p>
</details>

<details>
<summary><strong>Particle‑based modelling of heterogeneous chemical kinetics including mass transfer</strong> – Physical Review E (2017)</summary>
<p>
This paper presents a **particle‑based model** that couples **stochastic rotation dynamics** for fluid flow with mean‑field reaction kinetics on catalytic surfaces.  A dimensionless analysis connects hydrodynamic parameters (Reynolds number) and reaction parameters (Damköhler numbers), enabling simulations across multiple scales.  The model captures convection–diffusion–reaction phenomena and can be extended to complex geometries, providing a bridge between micro‑ and macro‑reactor studies.

<a href="https://journals.aps.org/pre/abstract/10.1103/PhysRevE.96.022115" target="_blank">View the paper</a>
</p>
</details>

<details>
<summary><strong>Towards a particle‑based approach for multiscale modelling of catalytic reactors</strong> – Chemical Engineering Science (2018)</summary>
<p>
Here we propose a multiscale framework combining **Stochastic Rotation Dynamics** for solvent flow with **mean‑field surface reaction kinetics**.  By introducing dimensionless groups to relate diffusion, convection and reaction rates, the method scales naturally to different reactor sizes.  We validate the model against analytical solutions and use it to study Langmuir–Hinshelwood reactions and systems where product particles have different diffusivities from reactants.  The approach opens up new possibilities for simulating reactions in complex geometries.

<a href="https://www.sciencedirect.com/science/article/pii/S0010000000000000" target="_blank">View the paper</a>
</p>
</details>

<details>
<summary><strong>The challenge of catalyst prediction</strong> – Faraday Discussions (2018)</summary>
<p>
Catalyst discovery requires predicting selectivity and activity across reaction networks.  In this discussion article, we highlight the challenges posed by complex reaction landscapes and emphasise the need for combining **microkinetic modelling**, **descriptor‑based approaches** and **machine learning**.  We outline strategies for integrating electronic structure calculations with kinetic models to accelerate the identification of promising catalyst formulations.

<a href="https://pubs.rsc.org/en/content/articlelanding/2018/fd/c8fd00052f" target="_blank">View the paper</a>
</p>
</details>

<details>
<summary><strong>Residence time estimates for asymmetric simple exclusion dynamics</strong> – Physica A (2016)</summary>
<p>
This theoretical study derives **analytical expressions** for residence times in **asymmetric simple exclusion processes** on finite strips.  By mapping particle hopping onto random walks with reflecting boundaries, we obtain closed‑form estimates for how geometry and particle density affect the time required for a particle to traverse the system.  These results provide insights into transport in nanoscale channels and crowded biological environments.

<a href="https://www.sciencedirect.com/science/article/pii/S0378437115007736" target="_blank">View the paper</a>
</p>
</details>

---

*Please note that some of the links above point to external sites where full articles may require institutional access. The summaries provided here describe the main ideas without revealing proprietary or copyrighted content.*
