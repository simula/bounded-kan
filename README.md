# Bounded-KAN

**Physics-constrained Kolmogorov-Arnold Networks for stable system identification**

Bounded-KAN is a structural adaptation of the Kolmogorov-Arnold Network (KAN) architecture, specifically designed for physical system identification, digital twins, and robust regression. 

While standard KANs excel at function approximation in purely mathematical domains, applying them to physical telemetry often requires interventions, like dynamic grid updates or statistical normalization (LayerNorm), to handle out-of-bounds (OOB) anomalies. In this context, OOB refers to any data point that exceeds the nominal operational range of the system, whether caused by a real but long-tail phenomenon (e.g., severe weather) or a transient sensor failure (e.g., signal spikes). Unfortunately, these standard deep learning techniques destroy the spatial meaning of the network's internal variables. 

Bounded-KAN addresses this by freezing the spatial grid and enforcing strict physical bounds natively within the architecture, prioritizing metric stability and OOB safety over localized curve-fitting flexibility.

## The mental model: Progressive Koopman-inspired latents

Bounded-KAN is conceptually intended to progressively untangle non-linear system dynamics layer by layer.

Analogous to the concept of observables in Koopman Operator Theory, the architecture aims to lift raw physical inputs into a linearized latent space. It achieves this by applying constrained, univariate B-spline transformations to isolate and straighten specific physical phenomena.

By making layer-level activation functions opt-in (defaulting to the Identity function), the splines are forced to perform the non-linear transformations. This guarantees that internal transitions remain pure compositions of univariate functions. The resulting forward pass can be understood as a progressive linearization pipeline: raw physics enter, non-linearities are un-bent locally by the splines, and the final output is safely projected through a linear combination of stable, Koopman-like observables.

## Core architectural mechanisms

To maintain the absolute physical meaning of these latent observables during extreme weather events or sensor failures, Bounded-KAN introduces three structural constraints:

### 1. Static grid boundaries
Dynamic grid updates (knot insertion or movement) fundamentally shift the underlying coordinate system of a network mid-training, causing downstream layers to lose their physical calibration. Bounded-KAN relies on a strictly static grid. While this sacrifices some adaptive curve-fitting capacity, it guarantees that a specific latent state retains its exact metric meaning from initialization to deployment.

### 2. Detached OOB routing (preventing knot saturation)
When a physical variable exceeds its nominal operating range, evaluating it on a standard spline forces extrapolation or grid distortion. Bounded-KAN handles extreme states via strict forward-pass clamping. 

Crucially, during the backward pass, gradients for these clamped OOB values are explicitly **detached** from the spline weights. This prevents the boundary knots from absorbing gradient penalties caused by transient anomalies, preserving the topological integrity of the boundary for nominal operations.

For example, consider a physical system experiencing a rare extreme operating condition—such as a vessel navigating a severe storm or a system recording a massive pressure spike. In standard implementations, out-of-bounds data either "falls off" the spline grid entirely (dropping the non-linear output abruptly to zero), or requires the input to be clamped to the boundary. However, if naively clamped *without* gradient detachment, the model forces the boundary knot to absorb the training loss for all out-of-bounds states, effectively compressing the entire tail of the extreme distribution into a single spline coordinate. This "saturation" warps the boundary and degrades predictions for normal operations. By explicitly detaching the gradient for out-of-bounds values, these extreme events—which are often too sparse to properly learn non-linearly anyway—leave the operational-regime spline untouched. Instead, the excess signal is safely routed exclusively through the linear base scaling (described below).

### 3. Linear skip connections as OOB safety valves
If the spline gradients are detached for OOB values, the network still requires a mechanism to penalize extreme errors. Bounded-KAN routes these excess gradients entirely through a parallel linear skip connection. This serves two purposes:
* It protects the non-linear splines from extreme gradient pollution.
* It ensures that out-of-bounds inputs extrapolate linearly and predictably, limiting the "blast radius" of anomalies and making downstream filtering significantly more reliable.

#### Justification for linear extrapolation (physical basis functions)

While real-world OOB events often exhibit higher-order scaling (e.g., cubic wave resistance in severe storms), Bounded-KAN strictly enforces a *linear* default for OOB extrapolation. This is a deliberate design choice to prevent mathematical explosions caused by arbitrary sensor faults. 

To safely capture higher-order OOB physics, domain knowledge should be embedded directly via feature engineering: providing the network with a rich dictionary of physical basis functions (e.g., `x^2`, `x^3`). If the input features form a sufficient physical basis, in particular for asymptotic behaviours, the linear skip connection will naturally and safely capture higher-order OOB phenomena without compromising the architectural fail-safe. This approach aligns directly with the formulation of "observables" in applied Koopman Operator Theory.

Note that applying a post-summation node activation (such as `SiLU` or `Tanh`, or even `ReLU`) fundamentally sabotages this mechanism. A non-linear activation will warp or squash the magnitude of the OOB event, rendering the linear skip connection unable to model it. For this reason, Bounded-KAN disables activations by default (using `Identity`). This ensures that once an extreme event enters the linear extrapolation track, its physical magnitude remains strictly proportional to its basis features.

## Installation

You can install Bounded-KAN directly from GitHub:

```bash
pip install git+[https://github.com/simula/bounded-kan.git](https://github.com/simula/bounded-kan.git)
```

## Usage example

To fully realize the Koopman-inspired architecture, Bounded-KAN should act as the non-linear "encoder" (discovering and straightening the observables), while a standard linear layer acts as the final readout.

```python
import torch
from bounded_kan import KANLinear

# The Koopman encoder: Unbends non-linear physics into 8 stable observables
kan_encoder = KANLinear(
    in_features=4,
    out_features=8,
    grid_range=(-1.0, 1.0),
    grid_size=10
)

# The readout: Strictly linear combination of the final observables
linear_mixer = nn.Linear(in_features=8, out_features=1)

# Construct the full digital twin pipeline
model = nn.Sequential(
    kan_encoder,
    linear_mixer
)

# Nominal physical data (e.g., draft, speed, wind, trim)
x_nominal = torch.tensor([[0.5, -0.2, 0.8, 0.1]])
prediction = model(x_nominal)

# Extreme OOB data (e.g., a massive sensor spike to 5.0)
# The spike is clamped by the KAN splines, and its gradient is detached.
# The excess magnitude safely bypasses the splines via the linear skip connection,
# preserving the predictable, linear behavior of the final readout.
x_extreme = torch.tensor([[5.0, -0.2, 0.8, 0.1]])
prediction_extreme = model(x_extreme)
```

## Attribution

This repository is an adaptation of the excellent
**[efficient-kan](https://github.com/Blealtan/efficient-kan)** library by Blealtan. 

The core B-spline evaluation mechanics, memory optimizations, and foundational
matrix operations in `Bounded-KAN` are directly derived from `efficient-kan`.
The modifications introduced here are strictly architectural (specifically the
detached routing, strict boundary clamping, and default identity activations) 
designed to constrain the network for physical system identification. Full
credit for the underlying efficiency and base implementation belongs to the
original author.
