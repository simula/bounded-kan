# %% [markdown]
# # PhysKAN: Architecture demonstrations
#
# This suite demonstrates the uncertainty-forwarding and gradient firewall mechanics of the PhysKAN architecture.

# %%
import torch
import torch.nn as nn

from physkan import KAN
from physkan.demonstrator import KANDemonstrator, TradKAN

torch.manual_seed(42)

# Helper: Generate raw physical state (x)
def generate_x_data(x_min, x_max, steps=200):
    return torch.linspace(x_min, x_max, steps).unsqueeze(1)


# Helper: Generate raw physical state (x) and angle (theta)
def generate_x_theta_train(steps=400):
    x = torch.rand(steps, 1) * 2 - 1
    # Full phase [-pi, pi] to break collinearity and ensure cos(theta) spans [-1, 1]
    theta = torch.rand(steps, 1) * 2 * torch.pi - torch.pi
    return torch.cat([x, theta], dim=1)


def generate_x_theta_eval(x_min, x_max, steps=200):
    x = torch.linspace(x_min, x_max, steps).unsqueeze(1)
    # Lock theta at 1.5 rad (~85 deg) so cos(theta) is near 0.07.
    # This exposes the naive multiplication trap for evaluation.
    theta = torch.full((steps, 1), 1.5)
    return torch.cat([x, theta], dim=1)

nominal_data = generate_x_data(-1.0, 1.0, 100)
dense_data = generate_x_data(-4.0, 4.0, 100)
sparse_data = torch.cat([nominal_data, dense_data[torch.randperm(dense_data.size(0))[:10]]])

nominal_x_theta = generate_x_theta_train()
eval_x_theta = generate_x_theta_eval(-4.0, 4.0)

# %%
# %matplotlib inline

# %% [markdown]
# # 0a. Standard KAN vulnerability (arbitrary extrapolation)
#
# **Goal:** Mimic an unprotected KAN using narrow nominal bounds `(-1.0, 1.0)` and the default `SiLU` base activation.
# We train on the nominal range and extrapolate.
#
# **Result:** While our native clamp turns the out-of-bounds discontinuity into a plateau, the asymmetric nature of `SiLU` makes extrapolation unpredictable.
# It grows on the right but flatlines on the left.

# %%
model_0a = TradKAN(
    layer_dims=[1, 1],
    grid_size=5,
    spline_order=3,
    grid_range=(-1.0, 1.0),
)
demo_0a = KANDemonstrator(model=model_0a, target_fn=lambda x: x**2)

demo_0a.train(nominal_data)
demo_0a.plot(dense_data, "0a. Trained with only nominal-range data")


# %% [markdown]
# # 0b. The data sparsity vulnerability (almost arbitrary extrapolation)
#
# Of course the extrapolation wasn't active: it never trained on any points outside the spline grid. Let's do that!

# %%
model_0b = TradKAN(
    layer_dims=[1, 1],
    grid_size=5,
    spline_order=3,
    grid_range=(-1.0, 1.0),
)
demo_0b = KANDemonstrator(model=model_0b, target_fn=lambda x: x**2)

demo_0b.train(sparse_data)
demo_0b.plot(dense_data, "0b. Trained with sparse OOB data")

# %% [markdown]
# # 0c. The wide grid fallacy (untrained knot collapse)
#
# **Goal:** A practitioner might try to fix 0a/0b by expanding the bounds to cover the extrapolation limits `(-4.0, 4.0)`.
# We increase `grid_size` proportionally to maintain resolution.
#
# **Result:** B-splines have strictly local support.
# The knots in the `(1.0, 4.0)` region receive very sparse gradient updates during training.
# The prediction detaches from the physics and outputs rubbish.
# This shows why expanding bounds without data is unsafe.

# %%
model_0c = TradKAN(
    layer_dims=[1, 1],
    grid_size=20,
    spline_order=3,
    grid_range=(-4.0, 4.0),
)
demo_0c = KANDemonstrator(model=model_0c, target_fn=lambda x: x**2)

demo_0c.train(sparse_data)
demo_0c.plot(generate_x_data(-4.0, 4.0, 200), "0c. Wide bounds fallacy (untrained extrapolation)", nominal_range=(-4.0, 4.0))

# %% [markdown]
# # 1. Spline plateau (symmetric linear track)
#
# **Goal:** Show how PhysKAN behaves with the strict `Identity` linear baseline.
#
# **Result:** Inside the bounds, the splines fit the curve.
# Out of bounds, the mechanical clamp freezes the splines to prevent oscillation.
# Notice that the left-side extrapolation looks slightly worse than the naive SiLU in case 0a.
#
# **Why?** The symmetry exists in the training data, but we enforced an asymmetric linear asymptote by using the strict `Identity` base track.
# The splines fit the symmetric parabola locally, while the base track absorbs a residual slope.
# When the splines clamp out of bounds, that raw linear slope is exposed.
# We intentionally trade the flatlining of SiLU for predictable linear extrapolation.
#
# **Note:** Explicit feature engineering demonstrated in step 2 is the preferred approach for enforcing symmetric bounds.

# %%
model_1 = KAN(layer_dims=[1, 1], grid_size=5, spline_order=3)
demo_1 = KANDemonstrator(model=model_1, target_fn=lambda x: x**2)

demo_1.train(sparse_data)
demo_1.plot(dense_data, "1. Spline plateau (symmetric linear track)")

# %% [markdown]
# # 2a. Linear recovery via feature engineering
#
# **Goal:** Provide $x^2$ as an engineered feature.
# Show that extrapolation works because the unbroken linear track carries the out-of-bounds scaling.

# %%
model_2a = KAN(layer_dims=[2, 1], grid_size=5, spline_order=3)
demo_2a = KANDemonstrator(
    model=model_2a, target_fn=lambda x: x**2, feature_fn=lambda x: torch.cat([x, x**2], dim=1)
)

demo_2a.train(nominal_data)
demo_2a.plot(dense_data, "2a. Linear recovery (engineered $x^2$)")

# %% [markdown]
# # 2b. Interval protection (the collinearity trap)
#
# **Goal:** Use the `interaction_map` to compute the $x^2$ interaction internally.
#
# **Result:** The firewall operates as expected.
# It recognizes the out-of-bounds variance of $x$ and raises the severity $D$.
# However, the physical prediction overshoots.
# During training, instead of relying purely on the interaction feature, the network put weight on the raw $x$ feature and used the splines to cancel out the error.
# When extrapolated, the splines clamped, the cancellation stopped, and the raw error emerged.
# This shows why severity tracking is necessary.

# %%
model_2b = KAN(layer_dims=[1, 1], interaction_map=[[0, 0]], grid_size=5, spline_order=3)
demo_2b = KANDemonstrator(model=model_2b, target_fn=lambda x: x**2, feature_fn=lambda x: x)

demo_2b.train(nominal_data)
demo_2b.plot(dense_data, "2b. Interval protection (collinearity trap)")

# %% [markdown]
# # 2c. The dropout fix (forcing physical isolation)
#
# **Goal:** Prevent the network from using splines to hide linear weights.
# We introduce spline dropout.
# By randomly zeroing out the splines during training, the linear track is forced to explain the physical features.
#
# **Result:** The linear track sets the weight of pure $x$ to zero, and the weight of the interaction feature to 1.0.
# The physical prediction is now flat, matching the true physics.
# The severity firewall remains active.

# %%
model_2c = KAN(
    layer_dims=[1, 1], interaction_map=[[0, 0]], grid_size=5, spline_order=3, spline_dropout=0.05
)
demo_2c = KANDemonstrator(model=model_2c, target_fn=lambda x: x**2, feature_fn=lambda x: x)

demo_2c.train(nominal_data)
demo_2c.plot(dense_data, "2c. The dropout fix (physical isolation)")

# %% [markdown]
# # 3a. Protected interaction layer
#
# **Goal:** Use the `interaction_map` to compute the product of $x^2$ and $\cos(\theta)$.
# The interval arithmetic assesses the variance, raises the dual severity $D$, and activates the gradient firewall.
# Extrapolation plateaus safely.

# %%
model_3a = KAN(
    layer_dims=[2, 1], interaction_map=[[0, 1]], grid_size=5, spline_order=3, spline_dropout=0.05
)
demo_3a = KANDemonstrator(
    model=model_3a,
    target_fn=lambda x: (x[:, 0:1] ** 2) * torch.cos(x[:, 1:2]),
    feature_fn=lambda x: torch.cat([x[:, 0:1] ** 2, torch.cos(x[:, 1:2])], dim=1),
)

demo_3a.train(nominal_x_theta)
demo_3a.plot(eval_x_theta, "3a. Protected interaction layer")

# %% [markdown]
# # 3b. Deep network feature discovery
#
# **Goal:** Remove explicit interaction mapping.
# Provide just $x$ and $\cos(\theta)$ to a deeper network (`[2, 4, 1]`) to let it learn the interaction $x^2 \cos(\theta)$.
# Show that the dual compounds through the linear matrices, protecting the entire depth.

# %%
torch.manual_seed(42)
model_3b = KAN(layer_dims=[2, 4, 1], grid_size=5, spline_order=3, spline_dropout=0.1)
demo_3b = KANDemonstrator(
    model=model_3b,
    target_fn=lambda x: (x[:, 0:1] ** 2) * torch.cos(x[:, 1:2]),
    feature_fn=lambda x: torch.cat([x[:, 0:1], torch.cos(x[:, 1:2])], dim=1),
)

demo_3b.train(nominal_x_theta, epochs=1000)
demo_3b.plot(eval_x_theta, "3b. Deep discovery (matrix dual routing)")

# %% [markdown]
# **The Takeaway:** The severity firewall ($D$) spikes, alerting us that we have left the data-driven regime.
# However, because the deep network relies on unconstrained spline combinations to approximate multiplication, the physical extrapolation shape becomes erratic.
#
# High severity ($D$) means the model is now relying entirely on its structural priors.
# If those priors are unconstrained deep networks, extrapolation is unpredictable.
# If we engineer those priors correctly, we can extrapolate safely.
#
# To see how we constrain automated feature discovery using the symbolic track, proceed to `demo_deep.py`. Or, if you'd rather see how the "nominal regime" splines are improved by PhysKAN, proceed to `demo_splines.py`. Indeed, why not both!
# %%
