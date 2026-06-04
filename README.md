# Bounded-KAN

**Physics-constrained Kolmogorov-Arnold Networks for stable system identification**

Bounded-KAN is a structural adaptation of the B-spline Kolmogorov-Arnold Network (KAN) architecture, specifically designed for physical system identification, digital twins, and robust regression. 

While standard KANs excel at function approximation in purely mathematical domains, applying them to physical telemetry often requires interventions, like dynamic grid updates or statistical normalization (LayerNorm), to handle out-of-bounds (OOB) anomalies. In this context, OOB refers to any data point that exceeds the nominal operational range of the system, whether caused by a real but long-tail phenomenon (e.g., severe weather) or a transient sensor failure (e.g., signal spikes). Unfortunately, these standard deep learning techniques destroy the spatial meaning of the network's internal variables. 

Bounded-KAN addresses this by freezing the spatial grid and enforcing strict physical bounds natively within the architecture, prioritizing metric stability and OOB safety over localized curve-fitting flexibility.

## Core design philosophy

Bounded-KAN is built on two central ideas, meant to bridge the gap between theoretical non-linear mapping and the robust fail-safes required for physical engineering:

1. **Progressive Koopman-style unbending:** Rather than relying on black-box MLP node activations, Bounded-KAN acts as a structural filter. It uses constrained B-splines to progressively unbend non-linear physical inputs layer-by-layer, lifting them into a linearized latent space (analogous to finding "observables" in Koopman Operator Theory).

2. **Embrace out-of-bounds (OOB) values:** Real-world physics do not stay neatly within standardized grids. Instead of arbitrarily squashing extreme weather events or sensor glitches with global activations (`tanh`, `SiLU`), Bounded-KAN explicitly encourages grid overflow. OOB states are safely clamped on the non-linear spline track and routed unclamped through a parallel linear track, ensuring mathematically stable extrapolation.

---

## Under the hood: The OOB routing mechanism

To safely execute this philosophy, Bounded-KAN requires a specific mental model for how it routes data—especially during the backward pass.

In standard implementations, out-of-bounds data either "falls off" the spline grid entirely (dropping to zero) or requires the input to be naively clamped. However, if clamped *without* gradient detachment, the model forces the boundary knot to absorb the training loss for all extreme out-of-bounds states. The boundary knot becomes an "extreme-weather wastebasket," compressing the entire long-tail distribution into a single coordinate and warping predictions for nominal operations.

Bounded-KAN acts as a traffic cop for physical regimes:
* **The nominal regime (non-linear track):** Dense, expected data operates inside the grid, shaping the non-linear B-splines.
* **The extreme regime (linear track):** Rare, OOB events are clamped on the non-linear track (with detached gradients to protect the nominal knots). The excess signal flows entirely through the linear track. 

This ensures the non-linear splines strictly learn the *nominal* physics, while the linear track safely catches the *normal but extreme* long-tail events.

## Architectural constraints

To maintain the absolute physical meaning of these latent observables during deployment, Bounded-KAN relies on two structural constraints:

### 1. Static grid boundaries
Original KAN architectures often rely on dynamic grid updates (knot insertion or movement) during training. Bounded-KAN strictly disables this. Dynamic updates fundamentally shift the underlying coordinate system of the network mid-training, causing downstream layers to lose their physical calibration. By enforcing a static grid, Bounded-KAN sacrifices some theoretical curve-fitting capacity to guarantee that a specific latent state retains its exact metric meaning from initialization to deployment.

#### Inter-layer OOB preservation

To guarantee that OOB retains its physical magnitude through deep layers, Bounded-KAN relies on a standard residual macro-architecture rather than dynamic normalization:

**Global residual skip connections**
Dynamic scaling (e.g., layer normalization) is disabled, as batch-dependent variance adjustments destroy both the absolute magnitude of physical metrics and the linear OOB safety valve. Instead, layers are wrapped in standard residual blocks ($y = x + \text{KAN}(x)$). 

Standard L2 weight decay (via AdamW) gently pressures the KAN's internal linear weights toward zero. Consequently, during an extreme anomaly, the saturated KAN layer's contribution safely minimizes, allowing the global identity skip connection to act as a perfect 1-to-1 physical passthrough. This preserves the exact scale of OOB events across arbitrary depths without requiring custom loss functions, matrix hacking, or normalization layers.

### 2. Linear Skip connections as safety valves
Because the spline gradients are deliberately detached for OOB values, the network routes the excess gradients entirely through the parallel linear skip connection. This serves as a vital safety valve: it protects the non-linear splines from extreme gradient pollution, and it ensures that OOB inputs extrapolate linearly and predictably. This strictly limits the "blast radius" of severe anomalies, making downstream system filtering significantly more reliable.

#### Justification for linear extrapolation (physical basis functions)

While real-world OOB events often exhibit higher-order scaling (e.g., cubic wave resistance in severe storms), Bounded-KAN strictly enforces a *linear* default for OOB extrapolation. This is a deliberate design choice to prevent mathematical explosions caused by arbitrary sensor faults. 

To safely capture higher-order OOB physics, domain knowledge should be embedded directly via feature engineering: providing the network with a rich dictionary of physical basis functions (e.g., `x^2`, `x^3`).
As long as the input features form a sufficient physical basis, particularly for asymptotic behaviours, the linear skip connection will naturally and safely capture higher-order OOB phenomena without compromising the nominal operating region.

Note that applying a post-summation node activation (such as `SiLU` or `Tanh`, or even `ReLU`) fundamentally sabotages this mechanism. A non-linear activation will warp or squash the magnitude of the OOB event, rendering the linear skip connection unable to model it. For this reason, Bounded-KAN disables activations by default (using `Identity`). This ensures that once an extreme event enters the linear extrapolation track, its physical magnitude remains strictly proportional to its basis features.

## Feature engineering and inclusive boundaries

While some recent KAN variants use "mixed bases" (e.g., assigning Fourier edge-functions to periodic variables and splines to others), Bounded-KAN deliberately sticks to a uniform B-spline architecture. 

The justification is straightforward: codebase simplicity and practical necessity. Maintaining a heterogeneous architecture introduces complexity that is largely unnecessary for applied physical modeling. To capture specific periodic or asymptotic physics, domain knowledge should instead be embedded directly via feature engineering (e.g., passing `sin(θ)` or `x^2 * cos(θ)` as inputs). The linear skip connection will naturally lock onto these engineered features as the global baseline, leaving the splines to map the complex local residuals.

**The Inclusive Boundary Rule:**
Because engineered features like `sin(θ)` natively peak at ±1.0, the spline grid boundaries are **strictly inclusive**. They define the absolute mathematical domain, not an anomaly safe-zone. OOB gradient detachment and linear routing trigger strictly *outside* this range (`|x| > 1.0`), ensuring the network learns the true peaks of bounded physical functions without prematurely dropping gradients.

## Deep composition and graceful linear degradation

Because individual KAN layers are fundamentally additive, the network must rely on deep layer composition to learn multiplicative interactions (e.g., computing `x * y` via the algebraic identity `x * y = 1/4 * ((x+y)^2 - (x-y)^2)`). Forcing the network to deduce the interactions from scratch consumes significant capacity, requiring multiple layers just to construct, square by spline fit, and subtract the intermediate terms.
Therefore, known multiplicative relationships should ideally be provided as engineered features. Nevertheless, a multi-layer KAN is capable of mapping non-linear compositions within the nominal operating envelope.

Reasoning about how these deep compositions behave with OOB data reveals Bounded-KAN's most vital safety feature: **Graceful Linear Degradation**.

### The danger of polynomial extrapolation
In unconstrained neural networks, if a model learns a steep 4th-order interaction deep in its layers based on calm weather data, it will attempt to extrapolate that 4th-order polynomial into an extreme storm. This results in physically impossible outputs that destroy downstream control systems.

Bounded-KAN prevents this through progressive spline clamping. When an extreme OOB signal propagates through multiple layers:
1. The deep-layer splines reach their boundary limits and saturate, clamping to constant maximums.
2. The entire network collapses down into a stable compound linear matrix multiplication.

### Latent basis latching
Rather than hallucinating deep interactions, the collapsing network will search for and latch onto the most stable latent basis available in the linear track.

This reinforces the dual purpose of robust feature engineering. Not only does it save network capacity in the nominal regime, but if domain experts provide a rich physical basis (e.g., explicitly passing `x^2` or `x^2 * cos(θ)` alongside `x` and `cos(θ)`), the network does not need to rely on deep composition to survive an anomaly. When the splines saturate during a storm, the linear track may find the explicit `x^2` feature to be the best available explanation, scaling the magnitude predictably.

Ultimately, Bounded-KAN delivers the expressivity of a deep Kolmogorov-Arnold Network where there is data, and the mathematical safety of a linear projection where there isn't.

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
