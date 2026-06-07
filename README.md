# PhysKAN

**Physics-constrained Kolmogorov-Arnold Networks for stable system identification**

This repository provides a structural adaptation of the B-spline Kolmogorov-Arnold Network (KAN) architecture, designed for physical system identification, digital twins, and robust regression.
While standard KANs perform well at function approximation in purely mathematical domains, applying them to physical telemetry often requires interventions, like dynamic grid updates or statistical normalization, to handle out-of-bounds (OOB) anomalies.
In this context, OOB refers to any data point that exceeds the nominal operational range of the system, whether caused by a real but long-tail phenomenon (e.g., unseen weather regimes) or a transient sensor failure (e.g., signal spikes).
Unfortunately, these standard deep learning techniques remove the spatial meaning of the network's internal variables.
This architecture addresses this by freezing the spatial grid and enforcing physical bounds natively, prioritizing metric stability and OOB safety over localized curve-fitting flexibility.
It also uses forward uncertainty propagation with interval arithmetic to track the OOB state through the network.

## Core design philosophy

PhysKAN is built on three central ideas, meant to bridge the gap between theoretical non-linear mapping and the fail-safes required for physical engineering:

1. **Progressive Koopman-style unbending:** Rather than relying on black-box MLP node activations, the model acts as a structural filter.
It uses constrained B-splines to progressively unbend non-linear physical inputs layer-by-layer, lifting them into a linearized latent space.

2. **Embrace out-of-bounds (OOB) values:** Real-world physics do not stay neatly within standardized grids.
Instead of arbitrarily squashing long-tail events or sensor glitches with clamps or global activations, the architecture uses the grid range to explicitly define the boundary between the dense, well-modeled operational regime and the sparse tail.
OOB states are clamped on the non-linear spline track and routed unclamped through a parallel linear track, ensuring stable extrapolation.

3. **Epistemic uncertainty tracking:** The network computes a continuous dual property alongside the physical prediction.
This signal forward-propagates the mathematical severity of any out-of-bounds state, providing a deterministic measure of when the network is forced to extrapolate.

## Under the hood: the out-of-bounds routing mechanism

To execute this philosophy, the network requires a specific mental model for how it routes data, especially during the backward pass.
In standard implementations, out-of-bounds data either falls off the spline grid entirely or requires the input to be clamped.
However, if clamped without gradient detachment, the boundary knot absorbs the training loss for all out-of-bounds states.
It becomes a wastebasket for outlying values, compressing the long-tail distribution into a single coordinate and warping predictions for nominal operations.
The PhysKAN architecture routes data based on physical regimes.
Dense, expected data operates inside the grid, shaping the non-linear B-splines.
OOB data are clamped on the non-linear track, with detached gradients to protect the nominal-range knots.
The excess signal flows entirely through the linear track.
This ensures the non-linear splines learn the nominal physics, while the linear track catches long-tail events.

## Architectural constraints

To maintain the physical meaning of these latent observables during deployment, the model relies on structural constraints.

### 1. Static grid boundaries

KAN architectures often rely on dynamic grid updates during training.
This architecture disables this feature.
Dynamic updates shift the underlying coordinate system of the network mid-training, causing downstream layers to lose their physical calibration.
By enforcing a static grid, the model sacrifices some curve-fitting capacity to guarantee that a specific latent state retains its metric meaning from initialization to deployment.

### 2. Linear skip connections as safety valves

Because the spline gradients are detached for OOB values, the network routes the excess gradients entirely through the parallel linear skip connection.
This serves as a safety valve.
It protects the non-linear splines from gradient pollution, and it ensures that OOB inputs extrapolate linearly and predictably.
This limits the downstream impact of anomalies, making system filtering more reliable.

#### Justification for linear extrapolation

While real-world OOB events often exhibit higher-order scaling, the model enforces a linear default for OOB extrapolation.
This is a deliberate design choice to prevent mathematical instability caused by sensor faults.
To capture higher-order OOB physics, domain knowledge should be embedded directly via feature engineering.
As long as the input features form a sufficient physical basis, the linear skip connection will naturally capture higher-order OOB phenomena as a linear combination of features.
Applying a post-summation node activation (such as SiLU or tanh) sabotages this mechanism.
A non-linear activation will warp the magnitude of the OOB event, rendering the linear skip connection unable to model it.
For this reason, activations are disabled by default.

### 3. Target isolation in multi-output networks

When deploying multi-output configurations, using deep hidden layers can introduce cancellation entanglement.
The network may fit a zero-impact relationship by balancing opposing weights across internal nodes rather than setting the weights to zero.
While the physical output remains correct within nominal ranges, this hidden entanglement is unstable out-of-bounds.
The absolute-weighted path of the dual severity tracker exposes this state by signaling high severity even if the physical prediction appears unaffected.
To achieve decoupled causal isolation between targets, a shallow architecture with no hidden layers should be used.

## Feature engineering and explicit interactions

Deep architectures can deduce multiplicative interactions, such as computing the product of two inputs by combining multiple layers.
Forcing the network to learn these relationships from scratch consumes capacity and degrades when out-of-bounds.
To capture physical behaviors, domain knowledge should be embedded directly via feature engineering.
Providing the network with a dictionary of physical basis functions allows the linear skip connection to latch onto these features as a baseline.
This leaves the splines to map the local residuals.
However, combining features naively outside the network can mask out-of-bounds anomalies.
If a large wave anomaly interacts with a nominal-range cosine feature, their product may fall within nominal bounds, suppressing the anomaly signal.
To prevent this suppression, PhysKAN provides two orthogonal paths to define interactions internally.
For known asymptotic behaviors, the preferred method is passing an explicit `interaction_map` to the constructor.
For unknown interactions, setting `symbolic_order` greater than zero allows the network to automatically set up an internal polynomial expansion directly from the raw inputs.
The network computes a continuous dual property alongside the standard physical prediction.
This dual represents the mathematical severity of the out-of-bounds state.
The physical prediction is computed using the non-linear splines and the linear tracks.
The dual severity bypasses the splines and propagates via the absolute values of the linear weights, ensuring that uncertainties compound.
By defining interactions internally, the model applies the uncertainty product rule to the input features before they enter the network layers.
This deterministic distress signal persists through the entire depth of the network.
It ensures that the non-linear splines are firewalled from learning from the anomaly, while the linear track handles the extrapolated magnitude.

### Defining the nominal range: data density vs. physical limits

When defining the grid boundaries and normalizing inputs, the ranges should reflect the density of the training data rather than the theoretical limits of the physical system.
B-splines require consistent data distribution across their internal grid to form a stable curve.
If the training dataset becomes sparse well before the theoretical operational limit, setting the spline boundary to the theoretical limit forces the model to fit curves in an under-constrained region.
This can cause the splines to oscillate or overfit to isolated data points.
Instead, the grid boundary should be placed where the data density drops off.
By treating the sparse region as out-of-bounds, the network clamps the splines in the dense region and relies on the linear track to extrapolate through the sparse tail.

## Installation

You can install the package from pip:

```bash
pip install physkan
```

## Usage example

The model handles dual interval arithmetic and OOB routing internally.
A minimal shallow setup is recommended to start, as it provides direct mapping without hidden entanglement.

```python
import torch
import torch.nn as nn
from physkan import KAN

# Initialize a minimal shallow model.
# layer_dims defines a direct mapping from 2 inputs to 1 output.
model = KAN(
    layer_dims=[2, 1],
    grid_size=5,
    spline_order=3
)

# Standard training optimization loop setup.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Nominal physical data
x_nominal = torch.tensor([[0.5, 0.1]])

# Pass data through the model
prediction, severity_signal = model(x_nominal)

# For an out-of-bounds event
x_oob = torch.tensor([[4.0, 0.1]])
prediction_oob, severity_oob = model(x_oob)

# Severity greater than zero indicates the prediction relies on extrapolated values.
if severity_oob.mean() > 0.0:
    print("Warning: operating in uncharted physical regime.")
```

## Demos

To see the out-of-bounds routing, cancellation entanglement, and hybrid symbolic architecture in action, please review the demonstrator notebooks provided in the `demos/` directory of this repository.

## Attribution

This repository is an adaptation of the `efficient-kan` library by Blealtan.

The B-spline evaluation mechanics, memory-efficient tensor formulation, and matrix operations are derived from that work.

The modifications introduced here are architectural, designed to constrain the network for physical system identification.

Credit for the underlying efficiency and base implementation belongs to the original author.
