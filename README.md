# Bounded-KAN

**Physics-constrained Kolmogorov-Arnold Networks for stable system identification**

This repository provides a structural adaptation of the B-spline Kolmogorov-Arnold Network (KAN) architecture, designed for physical system identification, digital twins, and robust regression. 

It introduces forward uncertainty propagation using interval arithmetic to track out-of-distribution (OOD) states natively.
While standard KANs perform well at function approximation in purely mathematical domains, applying them to physical telemetry often requires interventions, like dynamic grid updates or statistical normalization such as LayerNorm, to handle out-of-bounds (OOB) anomalies. In this context, OOB refers to any data point that exceeds the nominal operational range of the system, whether caused by a real but long-tail phenomenon (e.g., unseen weather regimes) or a transient sensor failure (e.g., signal spikes). Unfortunately, these standard deep learning techniques remove the spatial meaning of the network's internal variables. 

This architecture addresses this by freezing the spatial grid and enforcing strict physical bounds natively, prioritizing metric stability and OOB safety over localized curve-fitting flexibility.

## Core design philosophy

Bounded-KAN is built on three central ideas, meant to bridge the gap between theoretical non-linear mapping and the robust fail-safes required for physical engineering:

1. **Progressive Koopman-style unbending:** Rather than relying on black-box MLP node activations, the model acts as a structural filter. It uses constrained B-splines to progressively unbend non-linear physical inputs layer-by-layer, lifting them into a linearized latent space (analogous to finding "observables" in Koopman Operator Theory).

2. **Embrace out-of-bounds (OOB) values:** Real-world physics do not stay neatly within standardized grids. Instead of arbitrarily squashing long-tail events or sensor glitches with global activations (`tanh`, `SiLU`), the architecture encourages grid overflow. OOB states are safely clamped on the non-linear spline track and routed unclamped through a parallel linear track, ensuring mathematically stable extrapolation.

3. **Epistemic uncertainty tracking:** The network computes a continuous dual property alongside the physical prediction. Using interval arithmetic, this signal strictly forward-propagates the mathematical severity of any out-of-bounds state, providing a deterministic measure of when the network is forced to extrapolate.

---

## Under the hood: the OOB routing mechanism

To safely execute this philosophy, the network requires a specific mental model for how it routes data—especially during the backward pass.

In standard implementations, out-of-bounds data either "falls off" the spline grid entirely (dropping to zero) or requires the input to be naively clamped. However, if clamped *without* gradient detachment, the model forces the boundary knot to absorb the training loss for all out-of-bounds states. The boundary knot becomes a wastebasket for outlying values, compressing the long-tail distribution into a single coordinate and warping predictions for nominal operations.

This architecture acts as a traffic cop for physical regimes:
* **The nominal regime (non-linear track):** Dense, expected data operates inside the grid, shaping the non-linear B-splines.
* **The out-of-bounds regime (linear track):** OOB data are clamped on the non-linear track (with detached gradients to protect the nominal knots). The excess signal flows entirely through the linear track. 

This ensures the non-linear splines strictly learn the nominal physics, while the linear track safely catches long-tail events.

## Architectural constraints

To maintain the absolute physical meaning of these latent observables during deployment, the model relies on two structural constraints:

### 1. Static grid boundaries
Original KAN architectures often rely on dynamic grid updates (knot insertion or movement) during training. This architecture strictly disables this. Dynamic updates shift the underlying coordinate system of the network mid-training, causing downstream layers to lose their physical calibration. By enforcing a static grid, the model sacrifices some theoretical curve-fitting capacity to guarantee that a specific latent state retains its exact metric meaning from initialization to deployment.

#### Inter-layer out-of-bounds preservation

To guarantee that OOB retains its physical magnitude through deep layers, the network relies on a standard residual macro-architecture rather than dynamic normalization:

**Global residual skip connections**
Dynamic scaling (e.g., layer normalization) is disabled, as batch-dependent variance adjustments remove both the absolute magnitude of physical metrics and the linear OOB safety valve. Instead, layers are wrapped in standard residual blocks ($y = x + \text{KAN}(x)$). 

Standard L2 weight decay (via AdamW) gently pressures the internal linear weights toward zero. Consequently, during an anomaly, the saturated layer's contribution minimizes, allowing the global identity skip connection to act as a physical passthrough. This preserves the scale of OOB events across arbitrary depths without requiring custom loss functions, matrix adjustments, or normalization layers.

### 2. Linear skip connections as safety valves
Because the spline gradients are detached for OOB values, the network routes the excess gradients entirely through the parallel linear skip connection. This serves as a vital safety valve: it protects the non-linear splines from gradient pollution, and it ensures that OOB inputs extrapolate linearly and predictably. This limits the downstream impact of anomalies, making system filtering more reliable.

#### Justification for linear extrapolation (physical basis functions)

While real-world OOB events often exhibit higher-order scaling (e.g., cubic wave resistance), the model enforces a linear default for OOB extrapolation. This is a deliberate design choice to prevent mathematical instability caused by sensor faults. 

To safely capture higher-order OOB physics, domain knowledge should be embedded directly via feature engineering. As long as the input features form a sufficient physical basis, particularly for asymptotic behaviours, the linear skip connection will naturally capture higher-order OOB phenomena without compromising the nominal operating region.

Applying a post-summation node activation (such as `SiLU` or `Tanh`) fundamentally sabotages this mechanism. A non-linear activation will warp the magnitude of the OOB event, rendering the linear skip connection unable to model it. For this reason, activations are disabled by default (using `Identity`). 

## Feature engineering and explicit interactions

While deep architectures can theoretically learn multiplicative interactions (such as computing `x * y` by combining multiple layers), forcing the network to deduce these relationships from scratch consumes capacity and degrades poorly when out-of-bounds. 

To capture known physical behaviors, domain knowledge should be embedded directly via feature engineering. Providing the network with a dictionary of physical basis functions (e.g., `x^2` or `cos(θ)`) allows the linear skip connection to latch onto these engineered features as a stable baseline. This leaves the splines to map the local residuals, ensuring safe extrapolation when the splines saturate.

### The multiplicative suppression problem and OOB dual

Standard neural networks can inadvertently mask out-of-bounds anomalies when features are combined. If you manually pre-compute an interaction like `wave_height * cos(wind_dir)` and pass it to the network as a raw input, the anomaly signal is suppressed.  For instance, if `wave_height` is OOB (e.g., twice nominal range) but `cos(wave_dir)` is near zero, their product is within nominal bounds. A standard model outputs a regular in-bounds prediction, ignoring the underlying OOB wave height. 

To prevent this suppression, the network requires interaction terms to be defined internally via an `interaction_map` rather than expanded manually beforehand. 

The network computes a continuous dual property ($D$) alongside the standard physical prediction. This dual represents the mathematical severity of the out-of-bounds state. 
* The physical prediction is computed using the non-linear splines and the linear track.
* The dual severity strictly bypasses the splines and propagates via the absolute values of the linear weights, ensuring that uncertainties compound and never cancel out.

By defining interactions explicitly through the `interaction_map`, the model correctly applies the uncertainty product rule to the input features before they enter the network. If a large wave anomaly interacts with a nominal-range cosine, the resulting interaction term inherits a proportional severity score. This results in a deterministic distress signal that persists through the entire depth of the network, ensuring that underlying OOB is routed through the linear track and optionally providing downstream consumer with an indicator of when the model is operating on mathematically contaminated data.

### Defining the nominal range: data density vs. physical limits

When defining the `grid_range` and normalizing inputs, the boundaries should reflect the density of the training data rather than the theoretical limits of the physical system. 

B-splines require consistent data distribution across their internal grid to form a stable curve. If a physical feature (such as wave height) has a theoretical operational limit of 5.0 meters, but the training dataset becomes sparse above 2.0 meters, setting the spline boundary to 5.0 meters forces the model to fit curves in an under-constrained region. This often causes the splines to oscillate or overfit to a handful of isolated data points.

Instead, the grid boundary should be placed where the data density noticeably drops off (e.g., at 2.0 meters). By treating the sparse region as out-of-bounds, the network safely clamps the splines in the dense region and relies on the linear track to extrapolate smoothly through the sparse tail. The working principle is to treat the nominal range strictly as the bounds of the dense training data.

## Installation

You can install the package directly from GitHub:

```bash
pip install git+[https://github.com/simula/bounded-kan.git](https://github.com/simula/bounded-kan.git)

## Usage example

The model handles explicit feature expansion and interval arithmetic internally. A standard linear layer should be used as the final readout.

```python
import torch
import torch.nn as nn
from bounded_kan import KAN

# Define explicit cross-terms using indices
# e.g., for features [wave, wind, cos_dir]:
# [0, 0] adds wave^2
# [0, 2] adds wave * cos_dir
interactions = [[0, 0], [0, 2]]

# The KAN model automatically expands the initial input dimension
# and sets up the continuous dual routing.
kan_encoder = KAN(
    layers_hidden=[3, 16, 8], # Input dim is 3 (wave, wind, cos_dir)
    grid_range=(0.0, 1.0),
    interaction_map=interactions
)

# The readout: Strictly linear combination of the final observables
linear_mixer = nn.Linear(in_features=8, out_features=1)

model = nn.Sequential(
    kan_encoder,
    linear_mixer
)

# Nominal physical data
x_nominal = torch.tensor([[0.5, 0.8, 0.1]])

# Pass data through the encoder, requesting the dual distress signal
latent_features, severity_signal = kan_encoder(x_nominal, return_dual=True)
prediction = linear_mixer(latent_features)

# For an out-of-bounds event (e.g., wave height sensor reads 5.0)
x_oob = torch.tensor([[5.0, 0.8, 0.1]])
latent_oob, severity_oob = kan_encoder(x_oob, return_dual=True)

# severity_oob > 0 indicates the prediction relies on mathematically 
# extrapolated values, allowing downstream logic to trigger heuristics.
if severity_oob.mean() > 0.0:
    print("Warning: operating in uncharted physical regime.")
```

## Attribution

This repository is an adaptation of the excellent **[efficient-kan](https://github.com/Blealtan/efficient-kan)** library by Blealtan. 

The core B-spline evaluation mechanics, memory-efficient tensor formulation, and foundational matrix operations are directly derived from `efficient-kan`. The modifications introduced here are strictly architectural (specifically the detached routing, strict boundary clamping, interval arithmetic dual, and default identity activations) designed to constrain the network for physical system identification. Full credit for the underlying efficiency and base implementation belongs to the original author.
