---
layout: page
title: Freelancing Projects
permalink: /freelancing/
---

# Freelancing Projects

Below is a collection of consulting and research projects carried out by Aditya Sengar.  Each entry expands to reveal a short technical summary, an illustration and a link placeholder for the corresponding code or report.

## Machine learning projects

<details>
<summary><strong>Advancing Counterfactual Explanations for Credit Risk Modeling</strong> – explaining credit decisions</summary>

Credit‑risk models are often criticised for being opaque.  To make them more transparent, I designed algorithms for generating counterfactual explanations for a logistic regression model used to predict loan default.  A logistic classifier has the form

\[
P(y=1\mid\mathbf{x}) = \frac{1}{1 + \exp(-\mathbf{w}^\top \mathbf{x})}
\]

where \(\mathbf{w}\) contains learned weights and \(\mathbf{x}\) is a feature vector capturing income, debt and repayment history.  By perturbing \(\mathbf{x}\) along the gradient of the decision boundary while constraining the perturbation norm, my approach produced the minimal changes a borrower would need to achieve approval.  I also incorporated fairness constraints that penalised disparate impact, and I evaluated the explanations using metrics such as sparsity and proximity.

<p><img src="{{ site.baseurl }}/images/credit_risk.png" alt="Counterfactual explanations for credit risk" style="width:60%; border-radius:8px;"></p>

<p><a href="#">GitHub link (to be added)</a></p>
</details>

<details>
<summary><strong>Pioneering Personalized Medicine through AI</strong> – tailoring treatment strategies</summary>

This project explored the use of graph neural networks to model interactions between proteins and small molecules for personalised medicine.  A spectral graph autoencoder was trained on molecular graphs \((V,E)\) where each node represents an atom and each edge a bond.  The update rule for the \(k\)‑th message passing layer is

\[
h_v^{(k+1)} = \sigma\Bigl(W^{(k)} h_v^{(k)} + \sum_{u\in \mathcal{N}(v)} U^{(k)} h_u^{(k)}\Bigr),
\]

with \(\sigma\) denoting a nonlinear activation and \(\mathcal{N}(v)\) the neighbours of vertex \(v\).  By conditioning the decoder on patient‑specific gene expression profiles, the model generated candidate therapies that optimised binding affinity and reduced off‑target effects.  Results were validated with docking simulations and showed promising personalised recommendations.

<p><img src="{{ site.baseurl }}/images/personalized_medicine.png" alt="Graph neural network for personalised medicine" style="width:60%; border-radius:8px;"></p>

<p><a href="#">GitHub link (to be added)</a></p>
</details>

<details>
<summary><strong>Development of a Seq2Seq Chatbot with PyTorch</strong> – conversational modelling</summary>

Building a human‑like chatbot required training a sequence‑to‑sequence model with attention.  The encoder and decoder were implemented with gated recurrent units.  At each time step \(t\) the encoder updates its hidden state via

\[
h_t = \mathrm{GRU}(x_t, h_{t-1}),
\]

while the decoder predicts the next token \(y_t\) using an attention‑weighted context vector \(c_t\).  Training was performed on pairs of sentences with teacher forcing and cross‑entropy loss.  To reduce over‑fitting I used dropout and gradient clipping.  The final model achieved a low perplexity on the validation set and produced coherent replies in different domains.

<p><img src="{{ site.baseurl }}/images/seq2seq_chatbot.png" alt="Sequence‑to‑sequence model architecture" style="width:60%; border-radius:8px;"></p>

<p><a href="#">GitHub link (to be added)</a></p>
</details>

<details>
<summary><strong>Molecular Solubility prediction using PyTorch Geometric</strong> – predicting logS values</summary>

Predicting aqueous solubility is essential in drug discovery.  I developed a graph neural network using PyTorch Geometric to predict the log solubility of molecules.  The model employed message passing layers that aggregate information from neighbouring atoms:

\[
h_v^{(k+1)} = \phi\Bigl(h_v^{(k)}, \square_{u\in \mathcal{N}(v)} \psi(h_u^{(k)}, h_v^{(k)}, e_{uv})\Bigr),
\]

where \(\psi\) is a message function and \(\phi\) an update function.  Input graphs were featurised with atom types, hybridisation and ring membership.  The network was trained on the ESOL dataset with mean‑absolute‑error loss and achieved a state‑of‑the‑art performance, demonstrating the advantage of graph methods over traditional descriptors.

<p><img src="{{ site.baseurl }}/images/molecular_solubility.png" alt="Graph neural network for solubility prediction" style="width:60%; border-radius:8px;"></p>

<p><a href="#">GitHub link (to be added)</a></p>
</details>

<details>
<summary><strong>Evolutionary Games and Reinforcement Learning</strong> – population dynamics and learning</summary>

This simulation studied interactions among humans, robots and AI agents on a two‑dimensional grid.  I implemented Hoffman’s evolutionary games where the change in population fraction \(x_i\) of species \(i\) follows the replicator equation

\[
\dot{x}_i = x_i \bigl(f_i(\mathbf{x}) - \bar{f}(\mathbf{x})\bigr),
\]

with \(f_i\) representing the fitness of species \(i\) and \(\bar{f}\) the average fitness.  I also implemented a Q‑learning agent with update rule

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \bigl[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\bigr],
\]

where \(r\) is the reward and \(\gamma\) the discount factor.  Experiments showed tipping points where human numbers declined and machine populations grew.  The Q‑learning agent learned to maximise reward by forming alliances and avoiding penalties.

<p><img src="{{ site.baseurl }}/images/evolutionary_games.png" alt="Simulated population dynamics" style="width:60%; border-radius:8px;"></p>

<p><a href="#">GitHub link (to be added)</a></p>
</details>

<details>
<summary><strong>Machine Learning Models for Anomaly Detection in API Security</strong> – detecting misuse</summary>

Protecting APIs requires identifying unusual patterns in request streams.  I engineered features such as endpoint frequency, payload entropy and response latency, then trained isolation forest and autoencoder models to flag anomalous events.  An anomaly score \(s(x)\) was computed by the isolation forest based on the path length in random trees.  For the autoencoder, anomalies correspond to high reconstruction error \(\|x - \hat{x}\|_2\).  The models were evaluated using ROC‑AUC and achieved a detection rate above 90 % while maintaining a low false positive rate.

<p><img src="{{ site.baseurl }}/images/anomaly_detection.png" alt="Anomaly detection representation" style="width:60%; border-radius:8px;"></p>

<p><a href="#">GitHub link (to be added)</a></p>
</details>

<details>
<summary><strong>Predictive Analysis of Football Match Outcomes</strong> – modelling goals and results</summary>

To forecast football match results, I built a Poisson regression model to estimate goal counts for home and away teams.  The expected goals for a team were modelled as

\[
\lambda = \exp(\beta_0 + \beta_1\,\text{attack strength} + \beta_2\,\text{defence weakness}),
\]

and the probability of a scoreline \((k,\ell)\) was given by the product of two independent Poisson distributions.  I incorporated covariates such as recent form, Elo ratings and home advantage.  Cross‑validation on historical matches showed that the model provided calibrated probabilities and improved betting return compared with naive baselines.

<p><img src="{{ site.baseurl }}/images/football_prediction.png" alt="Football outcome prediction" style="width:60%; border-radius:8px;"></p>

<p><a href="#">GitHub link (to be added)</a></p>
</details>

<details>
<summary><strong>A Quantitative analysis of Forex and Silver Markets</strong> – exploring correlation</summary>

This study examined the relationship between currency pairs (e.g., EUR/USD) and silver prices.  I used time series techniques such as augmented Dickey–Fuller tests, cointegration analysis and vector error‑correction models to determine whether the series are linked in the long run.  Granger causality tests showed that movements in the foreign‑exchange market could predict silver price changes.  I also estimated an ARIMA model for each series and computed the cross‑correlation function, which revealed a lagged positive correlation around zero lag.

<p><img src="{{ site.baseurl }}/images/forex_silver.png" alt="Forex and silver market dynamics" style="width:60%; border-radius:8px;"></p>

<p><a href="#">GitHub link (to be added)</a></p>
</details>

<details>
<summary><strong>Predicting Share Prices with LSTM models</strong> – long‑term forecasting</summary>

Stock prices exhibit temporal dependencies and nonlinear patterns.  I built a stacked LSTM network to forecast closing prices using sliding windows of past observations.  The LSTM cell computes gating signals:

\[
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f),\quad i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i),\quad o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o),
\]

with memory state \(c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t\).  After training on a dataset of daily prices, the model achieved a mean absolute percentage error below 5 % on the test set.  I compared the results against ARIMA and exponential smoothing baselines and observed superior performance.

<p><img src="{{ site.baseurl }}/images/stock_prediction.png" alt="LSTM stock price prediction" style="width:60%; border-radius:8px;"></p>

<p><a href="#">GitHub link (to be added)</a></p>
</details>

## Algorithm design projects

<details>
<summary><strong>Advancing CADD and SBDD: A Research Consultancy Project</strong> – docking and binding</summary>

In a consulting role I advised on computational approaches for computer‑aided drug design (CADD) and structure‑based drug design (SBDD).  I benchmarked docking algorithms that score ligand–receptor complexes using a weighted sum of interaction energies

\[
E = \sum_i w_i E_i,
\]

where \(E_i\) includes van der Waals, electrostatics and solvation terms.  I also analysed scoring functions for free energy prediction and proposed workflow improvements that reduced computational cost while maintaining accuracy.

<p><img src="{{ site.baseurl }}/images/drug_discovery.png" alt="Drug design simulation" style="width:60%; border-radius:8px;"></p>

<p><a href="#">GitHub link (to be added)</a></p>
</details>

<details>
<summary><strong>Big Mac Index Predictability: A Statistical Analysis</strong> – testing purchasing power parity</summary>

The Big Mac Index is often used to gauge exchange‑rate misalignment.  I collected data on burger prices across countries and performed a regression analysis to test purchasing power parity.  The fitted model took the form

\[
\text{Price}_{\text{local}} = \beta_0 + \beta_1 \times \text{Price}_{\text{US}} + \varepsilon,
\]

with \(\varepsilon\) representing random error.  Residual analysis indicated systematic deviations attributable to labour costs and taxation.  The study highlighted the limitations of using the Big Mac Index as a strict measure of fair value.

<p><img src="{{ site.baseurl }}/images/stock_prediction.png" alt="Big Mac index analysis" style="width:60%; border-radius:8px;"></p>

<p><a href="#">GitHub link (to be added)</a></p>
</details>

<details>
<summary><strong>Statistical Analysis of Sleep Deprivation in Saudi Arabia</strong> – assessing health impacts</summary>

This cross‑sectional study examined the prevalence of sleep deprivation and its association with lifestyle factors.  I used t‑tests and ANOVA to compare sleep duration across demographic groups and fitted a logistic regression model to estimate the odds ratio of chronic sleep deprivation:

\[
\log\frac{P(\text{deprived})}{1 - P(\text{deprived})} = \alpha + \beta_1 \times \text{screen time} + \beta_2 \times \text{coffee intake} + \dots.
\]

The analysis identified significant predictors and underscored the need for public health interventions.

<p><img src="{{ site.baseurl }}/images/sleep_study.png" alt="Sleep deprivation analysis" style="width:60%; border-radius:8px;"></p>

<p><a href="#">GitHub link (to be added)</a></p>
</details>

<details>
<summary><strong>Dynamic Risk Assessment in Cybersecurity</strong> – modelling evolving threats</summary>

Cyber threats evolve over time, and static risk models quickly become outdated.  I constructed a Bayesian network to represent dependencies among system vulnerabilities, threat actors and countermeasures.  The posterior risk given evidence \(E\) is computed using Bayes’ rule:

\[
P(R \mid E) \propto P(E \mid R) P(R),
\]

where \(R\) is the event of a security breach.  The network was updated with data on incident reports, enabling dynamic risk scoring and prioritisation of mitigation efforts.

<p><img src="{{ site.baseurl }}/images/cybersecurity.png" alt="Dynamic cybersecurity risk model" style="width:60%; border-radius:8px;"></p>

<p><a href="#">GitHub link (to be added)</a></p>
</details>

<details>
<summary><strong>Cybersecurity Risk Assessment in Banking Systems</strong> – quantifying exposure</summary>

For a banking client I developed a quantitative risk assessment framework that aggregated vulnerabilities across multiple systems.  Each asset was assigned a vulnerability score \(V_i\) and breach probability \(P_i\).  The overall risk index was computed as

\[
\text{Risk} = \sum_i w_i \times V_i \times P_i,
\]

with weights \(w_i\) reflecting asset importance.  The framework helped the client prioritise investments in security controls and satisfy regulatory requirements.

<p><img src="{{ site.baseurl }}/images/cybersecurity.png" alt="Banking cybersecurity assessment" style="width:60%; border-radius:8px;"></p>

<p><a href="#">GitHub link (to be added)</a></p>
</details>

## Market research projects

<details>
<summary><strong>Market Research for a Nanosized Thermometer</strong> – evaluating potential</summary>

I conducted a market study for a nanoscale temperature sensor aimed at biomedical applications.  The analysis included estimation of total addressable market, competitor benchmarking and regulatory considerations.  Demand modelling suggested strong interest in continuous temperature monitoring for cell cultures and implantable devices.

<p><img src="{{ site.baseurl }}/images/thermometer.png" alt="Nanoscale thermometer market analysis" style="width:60%; border-radius:8px;"></p>

<p><a href="#">Link to report (to be added)</a></p>
</details>

<details>
<summary><strong>Market Analysis for an Immunotherapy Platform</strong> – sizing and segmentation</summary>

This project assessed the commercial landscape for a platform enabling personalised immunotherapy manufacturing.  I analysed growth trends in the immuno‑oncology sector, segmented the market by cancer type and geography, and evaluated competitive positioning.  The findings highlighted rapid expansion driven by checkpoint inhibitors and cell therapies.

<p><img src="{{ site.baseurl }}/images/immunotherapy.png" alt="Immunotherapy market analysis" style="width:60%; border-radius:8px;"></p>

<p><a href="#">Link to report (to be added)</a></p>
</details>

<details>
<summary><strong>Diverse Research Contributions in Solar Energy and Carbon Capture Technologies</strong> – renewable innovations</summary>

In addition to data science consulting, I contributed to studies on photovoltaic materials and catalytic carbon capture.  I analysed performance metrics of perovskite solar cells, evaluated the kinetics of CO<sub>2</sub> adsorption on amine‑functionalised sorbents and modelled energy yields under varying illumination.  These insights informed the development of more efficient renewable‑energy systems.

<p><img src="{{ site.baseurl }}/images/solar_energy.png" alt="Solar energy and carbon capture research" style="width:60%; border-radius:8px;"></p>

<p><a href="#">Link to report (to be added)</a></p>
</details>