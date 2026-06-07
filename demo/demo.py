# %% [markdown]
# # PhysKAN: Architecture demonstrations
#
# This suite demonstrates the uncertainty-forwarding and gradient firewall mechanics of the PhysKAN architecture.

# %%
import torch
import torch.nn as nn

from physkan import KAN, KANDemonstrator, KANLinear

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


# %%
# %matplotlib inline

# %% [markdown]
# # 0a. Standard KAN vulnerability (arbitrary OOB)
#
# **Goal:** Mimic an unprotected KAN using narrow nominal bounds `(-1.0, 1.0)` and the default `SiLU` base activation.
# We train on the nominal range and extrapolate.
#
# **Result:** While our native clamp turns the out-of-bounds discontinuity into a plateau, the asymmetric nature of `SiLU` makes extrapolation unpredictable.
# It grows on the right but flatlines on the left.

# %%
model_0a = KANLinear(
    in_features=1,
    out_features=1,
    grid_size=5,
    spline_order=3,
    grid_range=(-1.0, 1.0),
    base_activation=nn.SiLU,
)
demo_0a = KANDemonstrator(model=model_0a, target_fn=lambda x: x**2)

demo_0a.train(generate_x_data(-1.0, 1.0, 100))
demo_0a.plot(generate_x_data(-4.0, 4.0, 200), "0a. Narrow bounds (arbitrary SiLU asymmetry)")

# %% [markdown]
# # 0b. The wide grid fallacy (untrained knot collapse)
#
# **Goal:** Mimic a practitioner trying to fix 0a by expanding the bounds to cover the extrapolation limits `(-4.0, 4.0)`.
# We increase `grid_size` proportionally to maintain resolution.
#
# **Result:** B-splines have strictly local support.
# The knots in the `(1.0, 4.0)` region receive zero gradient updates during training.
# The prediction detaches from the physics and outputs initialization noise.
# This shows why expanding bounds without data is unsafe.

# %%
model_0b = KANLinear(
    in_features=1,
    out_features=1,
    grid_size=20,
    spline_order=3,
    grid_range=(-4.0, 4.0),
    base_activation=nn.SiLU,
)
demo_0b = KANDemonstrator(model=model_0b, target_fn=lambda x: x**2)

demo_0b.train(generate_x_data(-1.0, 1.0, 100))
demo_0b.plot(generate_x_data(-4.0, 4.0, 200), "0b. Wide bounds fallacy (untrained extrapolation)")

# %% [markdown]
# # 0c. The data sparsity vulnerability (the interpolation hole)
#
# **Goal:** Train across the full wide grid `(-1.0, 4.0)`.
# Real physical data has gaps, so we filter out all training data between `2.0` and `3.5` to simulate a sparse transition regime.
#
# **Result:** Even though the bounds enclose all the data, the knots inside the hole receive zero gradient updates.
# Instead of bridging the gap smoothly, the prediction outputs untrained noise.
# This shows that relying purely on splines across sparse datasets degrades physical identification.

# %%
model_0c = KANLinear(
    in_features=1,
    out_features=1,
    grid_size=20,
    spline_order=3,
    grid_range=(-1.0, 4.0),
    base_activation=nn.SiLU,
)
demo_0c = KANDemonstrator(model=model_0c, target_fn=lambda x: x**2)

x_train_0c = generate_x_data(-1.0, 4.0, steps=200)
x_train_sparse = x_train_0c[(x_train_0c[:, 0] < 1.5) | (x_train_0c[:, 0] > 3.5)]

demo_0c.train(x_train_sparse)
demo_0c.plot(generate_x_data(-4.0, 4.0, 200), "0c. Data sparsity (interpolation hole collapse)")

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
# **Note:** Explicit feature engineering demonstrated in step 2 is the preferred approach for enforcing
