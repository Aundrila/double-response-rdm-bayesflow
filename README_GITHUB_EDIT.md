# Double Responses in Racing Diffusion Models with BayesFlow

**Simulation-based parameter recovery for double responses in a Racing Diffusion Model using BayesFlow.**

The implementation is inspired by:

> Evans, N. J., Dutilh, G., Wagenmakers, E.-J., & van der Maas, H. L. J. (2020).  
> *Double responding: A new constraint for models of speeded decision making.*  
> Cognitive Psychology, 121, 101292.

---

## Research Question

The central question is:

> **Does adding double-response information improve recovery of hidden parameters in a Racing Diffusion Model (RDM)?**

Two BayesFlow inference systems are trained on the same simulated datasets.

### Single-response estimator

Uses:

$$
(\mathrm{correct}, RT)
$$

### Double-response estimator

Uses:

$$
(\mathrm{correct}, RT, DR, DRT)
$$

where:

- `RT` = first-response time
- `DR` = double-response indicator
- `DRT` = double-response latency

The two estimators use the same prior, simulator, parameter draws, target parameters, and neural-network architecture. They differ only in the observed information supplied to BayesFlow.

---

# Scientific Model and Simulator

## Simplified Racing Diffusion Model

The model contains two independent evidence accumulators:

- accumulator 0 supports the **correct** response
- accumulator 1 supports the **error** response

For accumulator $i$, the evidence evolves according to:

$$
dx_i(t) = v_i\,dt + dW_i(t)
$$

where $v_i$ is the drift rate and $dW_i(t)$ represents independent Gaussian diffusion noise with scale 1.

For each trial, the initial evidence is sampled as:

$$
x_i(0) \sim \mathrm{Uniform}(0, A)
$$

Both accumulators race toward the common response threshold:

$$
B = A + b
$$

Let $T_c$ and $T_e$ denote the threshold-crossing times of the correct and error accumulators.

The first decision time is:

$$
T_{\mathrm{decision}} = \min(T_c, T_e)
$$

The observed first-response time is:

$$
RT = \min(T_c, T_e) + t_0
$$

where $t_0$ represents non-decision time.

A double response occurs when the losing accumulator also reaches the threshold within the 250 ms post-response window:

$$
DR = \mathbb{1}\left(0 < |T_c - T_e| \leq 0.250\right)
$$

For double-response trials, the double-response latency is:

$$
DRT = |T_c - T_e|
$$

---

## Model Parameters

The latent parameter vector is:

$$
\theta = (A, b, t_0, v_c, v_e)
$$

| Parameter | Meaning |
|---|---|
| $A$ | Width of the starting-point distribution |
| $b$ | Additional distance from the top of the starting range to the threshold |
| $t_0$ | Non-decision time |
| $v_c$ | Correct-accumulator drift rate |
| $v_e$ | Error-accumulator drift rate |

The diffusion-noise scale is fixed at:

$$
\sigma = 1
$$

---

## Prior Distributions

The notebook uses the following priors:

$$
A \sim \mathrm{Uniform}(0.20, 0.90)
$$

$$
b \sim \mathrm{Uniform}(1.00, 2.80)
$$

$$
t_0 \sim \mathrm{Uniform}(0.15, 0.35)
$$

$$
v_e \sim \mathrm{Uniform}(0.05, 1.10)
$$

and

$$
v_c - v_e \sim \mathrm{Uniform}(0.70, 3.20)
$$

Therefore:

$$
v_c = v_e + (v_c - v_e)
$$

which guarantees:

$$
v_c > v_e
$$

These priors are used for the simulation-based parameter-recovery study and should not be interpreted as empirical parameter estimates.

---

## Vectorized RDM Simulator

For a Brownian accumulator with positive drift $v$, diffusion scale 1, and remaining threshold distance $d$, the first-passage time follows an inverse Gaussian distribution:

$$
T \sim IG\left(\mu = \frac{d}{v}, \lambda = d^2\right)
$$

NumPy provides this distribution through:

```python
rng.wald(mean, scale)
```

The original paper approximated the evidence trajectories using **5 ms Euler steps**.

In this notebook, the threshold-crossing time is instead sampled directly from the **exact continuous-time first-passage-time distribution** for each independent accumulator.

For the simplified independent RDM implemented here, this represents the same underlying drift-diffusion process without introducing time discretization. This substantially reduces the computational cost of generating the large simulation bank.

The double-response latency is:

$$
DRT = |T_c - T_e|
$$

A double response is recorded when the losing accumulator reaches the threshold within the predefined 250 ms post-response window.

---

# Observation Representations

Each simulated dataset is represented in two matched forms.

## Single-Response Representation

Each trial contains:

```text
[correct, first_rt]
```

with shape:

```text
(n_datasets, n_trials, 2)
```

## Double-Response Representation

Each trial contains:

```text
[correct, first_rt, double_response, double_response_latency]
```

with shape:

```text
(n_datasets, n_trials, 4)
```

For trials without a double response, the double-response latency is stored as `0.0`. The double-response indicator distinguishes these cases from observed latencies.

---

# Controlled Comparison

The same simulated parameter-dataset pairs are used for both inference systems.

```text
                  Sample latent parameters
                           |
                           v
                    Simulate RDM trials
                           |
              +------------+------------+
              |                         |
              v                         v
      Single-response view      Double-response view
        [correct, RT]          [correct, RT, DR, DRT]
              |                         |
              v                         v
        DeepSet summary           DeepSet summary
              |                         |
              v                         v
        CouplingFlow               CouplingFlow
        posterior                  posterior
              |                         |
              +------------+------------+
                           |
                           v
              Compare parameter recovery
```

Both estimators use the same:

- prior
- scientific simulator
- latent parameter targets
- training parameter draws
- test parameter draws
- number of trials
- neural-network architecture

They differ only in the observed trial-level information passed to the summary network.

---

# BayesFlow Architecture

Each simulated dataset is an exchangeable set of trials. A **DeepSet** is therefore used to learn a permutation-invariant representation of the dataset.

## Summary Network

```python
bf.networks.DeepSet(
    summary_dim=16,
    embed_dim=32,
    depth=2,
    mlp_widths=(32,),
    dropout=0.0,
)
```

## Posterior Network

The posterior estimator uses an affine coupling flow:

```python
bf.networks.CouplingFlow(
    transform="affine",
    depth=4,
    subnet_kwargs={
        "widths": (64, 64),
        "dropout": 0.0,
    },
)
```

BayesFlow standardizes both:

```text
inference_variables
summary_variables
```

The implementation uses **learned trial-level summaries** rather than manually specified summary-statistic vectors.

---

# Computation Profiles

The notebook provides two computation profiles.

## Fast Mode

`FAST_MODE=True` is intended for debugging and verifying that the complete workflow runs successfully.

## Full Mode

`FAST_MODE=False` uses a larger simulation bank and longer neural-network training.

The full configuration is substantially more computationally expensive.

---

# Training

## Train the Single-Response Estimator

The target is the five-dimensional RDM parameter vector:

$$
\theta = (A, b, t_0, v_c, v_e)
$$

The input contains only the first-response information from each trial:

$$
(\mathrm{correct}, RT)
$$

where `correct` indicates whether the first response was correct and `RT` denotes first-response time.

---

## Train the Double-Response Estimator

The target parameters and neural-network architecture remain unchanged.

The input contains the full trial-level observation:

$$
(\mathrm{correct}, RT, DR, DRT)
$$

where:

- `RT` = first-response time
- `DR` = double-response indicator
- `DRT` = double-response latency

---

# Parameter-Recovery Evaluation

For every held-out synthetic dataset, BayesFlow produces posterior samples.

The posterior mean is used as a point estimate:

$$
\hat{\theta}^{(s)} = E_q\left[\theta \mid y^{(s)}\right]
$$

## RMSE

Parameter-recovery error is measured using:

$$
RMSE =
\sqrt{
\frac{1}{S}
\sum_{s=1}^{S}
\left(
\hat{\theta}^{(s)} - \theta_{\mathrm{true}}^{(s)}
\right)^2
}
$$

Lower RMSE indicates better parameter recovery.

## Normalized RMSE

RMSE is normalized by the empirical standard deviation of the corresponding parameter in the training prior sample.

This makes recovery performance more comparable across parameters with different numerical scales.

## MAE

Mean absolute error is also reported as an additional point-recovery metric.

## 95% Posterior Coverage

Coverage measures the fraction of held-out datasets for which the true generating parameter lies inside the estimated 95% posterior credible interval.

## Posterior Interval Width

The mean width of the 95% credible interval provides a measure of posterior uncertainty.

Recovery performance should be interpreted jointly using:

- RMSE / normalized RMSE
- MAE
- posterior coverage
- posterior interval width

Narrower intervals alone do not imply better inference if coverage becomes poor.

---

# Results

The saved notebook output currently corresponds to the small `FAST_MODE=True` development configuration:

```text
300 training datasets
300 trials per dataset
40 held-out test datasets
5 training epochs
250 posterior samples per test dataset
```

## Overall Recovery

| Model | Mean normalized RMSE | Mean 95% coverage | Mean normalized CI width |
|---|---:|---:|---:|
| Single response | 0.924 | 0.955 | 3.228 |
| Double response | 0.856 | 0.960 | 3.191 |

In this saved run, the double-response estimator achieved a lower average normalized RMSE while maintaining similar overall posterior coverage.

## Parameter-Wise RMSE

| Parameter | Single RMSE | Double RMSE | RMSE change |
|---|---:|---:|---:|
| $A$ | 0.2201 | 0.1976 | +10.2% |
| $b$ | 0.3483 | 0.3226 | +7.4% |
| $t_0$ | 0.0584 | 0.0596 | -1.9% |
| $v_c$ | 0.6289 | 0.4921 | +21.7% |
| $v_e$ | 0.2903 | 0.2841 | +2.1% |

Positive percentages indicate a reduction in RMSE after adding double-response information.

Because these numbers come from a small development run, they should not be interpreted as stable final scientific estimates. A stronger comparison should use the full computation profile and repeated random seeds.

---

# Optional Empirical-Data Illustration

The notebook can optionally read:

```text
parsedData_doubleResp.Rdata
```

using the `rdata` package.

The empirical data contain:

```text
Correct
Time1
DoubleResp
Time2
```

where:

- `Correct` indicates whether the first response was correct
- `Time1` is the first-response time
- `DoubleResp` indicates whether an opposite second response occurred
- `Time2` is the latency between the first and second responses

The optional section applies the already-trained amortized estimators to a selected participant.

This section should be interpreted as an **illustration of amortized inference**, not as a complete empirical model-fitting analysis.

---

# Installation

The notebook uses **BayesFlow 2** with the **JAX Keras backend**.

```bash
pip install \
    "bayesflow==2.0.12" \
    "keras>=3.12,<4" \
    "jax[cpu]>=0.5" \
    "rdata>=1.0" \
    "numpy>=2.2,<3" \
    "pandas>=2.3,<4" \
    "scipy>=1.15,<2" \
    "matplotlib>=3.10,<4"
```

The backend is configured before importing Keras or BayesFlow:

```python
import os

os.environ["KERAS_BACKEND"] = "jax"
```

---

# Running the Notebook

1. Clone the repository.
2. Create and activate a Python environment.
3. Install the dependencies.
4. Open `bayesflow_double_response_parameter_recovery.ipynb`.
5. Set `FAST_MODE=True` for a quick development run or `FAST_MODE=False` for the larger recovery experiment.
6. Run the notebook from top to bottom.
7. Use the optional empirical-data section only if the RData file is available.

---

# Reproducibility

The notebook uses:

```python
SEED = 20260717
```

Exact neural-network results may still vary across hardware and software versions.

For a stronger recovery study, repeated training with several random seeds is recommended.

---

# Repository Structure

```text
double-response-rdm-bayesflow/
│
├── bayesflow_double_response_parameter_recovery.ipynb
├── README.md
├── requirements.txt
│
└── data/
    └── parsedData_doubleResp.Rdata    # optional
```

The RData file is not required for the synthetic parameter-recovery experiment.

Only include the empirical dataset in a public repository if redistribution is permitted.

---

# Scope and Limitations

This repository implements a focused subset of the broader course project.

The current notebook does **not** implement:

- lateral inhibition
- leakage
- simulation-based calibration
- hierarchical participant modeling
- the complete set of evidence-accumulation models considered in Evans et al. (2020)
- the handcrafted Base/Full summary-statistic setup used in later versions of the broader project

Additional limitations include:

- the within-trial diffusion scale is fixed
- no between-trial drift variability is modeled
- no between-trial non-decision-time variability is modeled
- the priors are intended for the simulation study rather than participant-level empirical estimation
- the saved fast-mode run is too small for strong scientific conclusions

---

# Possible Extensions

Possible next steps include:

1. increase the number of simulations, trials, epochs, and posterior draws
2. repeat neural-network training using multiple random seeds
3. compare parameter recovery across different trial counts
4. perform simulation-based calibration
5. add posterior predictive checks
6. use participant-calibrated priors
7. include between-trial drift or non-decision-time variability
8. extend the independent RDM with a competitive mechanism such as lateral inhibition
9. compare multiple evidence-accumulation models
10. perform a more complete empirical analysis

---

# Final Project Statement

This notebook evaluates the following hypothesis:

$$
\text{Adding double-response observations}
\quad \stackrel{?}{\Longrightarrow} \quad
\text{improved recovery of RDM parameters}
$$

The comparison is controlled because the two BayesFlow systems use:

- the same prior
- the same scientific simulator
- the same latent parameter targets
- the same training and test parameter draws
- the same neural-network architecture

and differ only in the **observed trial-level information** provided to the summary network.

The single-response estimator receives:

$$
(\mathrm{correct}, RT)
$$

whereas the double-response estimator receives:

$$
(\mathrm{correct}, RT, DR, DRT)
$$

Parameter recovery on held-out synthetic datasets is then used to determine whether including double-response information improves inference of the underlying RDM parameters.

---

# Reference

Evans, N. J., Dutilh, G., Wagenmakers, E.-J., & van der Maas, H. L. J. (2020).  
**Double responding: A new constraint for models of speeded decision making.**  
*Cognitive Psychology, 121*, 101292.  
https://doi.org/10.1016/j.cogpsych.2020.101292

BayesFlow documentation:  
https://bayesflow.org/

---

# Contributors

- Ahmed Arian Sajid
- Aundrila Acharjee
- Sabrina Sultana

**TU Dortmund University — Simulation-Based Inference**
