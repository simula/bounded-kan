# %% [markdown]
# # PhysKAN: Architecture Demonstrations
# This suite systematically proves the uncertainty-forwarding and gradient
# firewall mechanics of PhysKAN architecture.

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
    # This specifically exposes the naive multiplication trap for evaluation.
    theta = torch.full((steps, 1), 1.5)
    return torch.cat([x, theta], dim=1)


# %%
# %matplotlib inline

# %% [markdown]
# # 0a. Standard KAN Vulnerability (Arbitrary OOB)
# **Goal:** Mimic an unprotected KAN using narrow nominal bounds `(-1.0, 1.0)` and
# the default `SiLU` base activation. We train on the nominal range and extrapolate.
#
# **Result:** While our native clamp turns the violent out-of-bounds discontinuity
# into a plateau, the asymmetric nature of `SiLU` (linear for positive, zero for
# negative) makes extrapolation arbitrary. It grows on the right but flatlines on
# the left.

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
demo_0a.plot(generate_x_data(-4.0, 4.0, 200), "0a. Narrow Bounds (Arbitrary SiLU Asymmetry)")

# %% [markdown]
# # 0b. The "Wide Grid" Fallacy (Untrained Knot Collapse)
# **Goal:** Mimic a practitioner trying to fix 0a by expanding the bounds to cover
# the extrapolation limits `(-4.0, 4.0)`. We increase `grid_size` proportionally
# to maintain resolution.
#
# **Result:** B-splines have strictly local support. The knots in the `(1.0, 4.0)`
# region receive absolutely zero gradient updates during training. The prediction
# completely detaches from the physics and outputs chaotic initialization noise,
# proving why expanding bounds without data is mathematically unsafe.

# %%
model_0b = KANLinear(
    in_features=1,
    out_features=1,
    grid_size=20,  # Increased to maintain resolution over wider bounds
    spline_order=3,
    grid_range=(-4.0, 4.0),  # The practitioner's "fix"
    base_activation=nn.SiLU,
)
demo_0b = KANDemonstrator(model=model_0b, target_fn=lambda x: x**2)

demo_0b.train(generate_x_data(-1.0, 1.0, 100))
demo_0b.plot(generate_x_data(-4.0, 4.0, 200), "0b. Wide Bounds Fallacy (Untrained Extrapolation)")

# %% [markdown]
# # 0c. The Data Sparsity Vulnerability (The Interpolation Hole)
# **Goal:** The practitioner now tries to train across the full wide grid `(-1.0, 4.0)`.
# However, real physical data has gaps. We filter out all training data between `2.0`
# and `3.5` to simulate a sparse transition regime (e.g., ships avoiding marginal weather).
#
# **Result:** Even though the bounds enclose all the data, the knots *inside the hole* # receive zero gradient updates. Instead of bridging the gap smoothly, the prediction
# violently collapses into the void, outputting untrained noise. This proves that
# relying purely on splines across sparse datasets destroys physical identification.

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

# Generate full data, then explicitly mask out the (2.0 to 3.5) transition regime
x_train_0c = generate_x_data(-1.0, 4.0, steps=200)
x_train_sparse = x_train_0c[(x_train_0c[:, 0] < 1.5) | (x_train_0c[:, 0] > 3.5)]

demo_0c.train(x_train_sparse)
demo_0c.plot(generate_x_data(-4.0, 4.0, 200), "0c. Data Sparsity (Interpolation Hole Collapse)")

# %% [markdown]
# # 1. Spline Plateau (Symmetric Linear Track)
# **Goal:** Show how PhysKAN behaves with the strict `Identity` linear baseline.
#
# **Result:** Inside the bounds, the splines perfectly fit the curve. Out of bounds,
# the mechanical clamp safely freezes the splines to prevent chaotic oscillation.
# However, notice that the left-side extrapolation actually looks slightly *worse* # than the naive SiLU in Case 0a!
#
# **Why?** The symmetry exists in the training data, but *we structurally
# enforced* an asymmetric linear asymptote by using the strict `Identity` base track.
# The splines easily fit the symmetric parabola locally, while the base track absorbs
# a slight residual slope. When the splines clamp out of bounds, that raw linear slope
# is exposed. We intentionally trade the arbitrary, "lucky" flatlining of SiLU for
# strict, predictable linear extrapolation.
#
# **Try this:** If you know the physical domain is symmetric, you can pass `base_activation=torch.abs` when initializing the model
# to structurally enforce a symmetric V-shape out of bounds. While this makes the baseline extrapolation look slightly better,
# it is still just a linear approximation. In general, the explicit feature engineering demonstrated in step 2 is the preferred
# approach.

# %%
model_1 = KAN(layer_dims=[1, 1], grid_size=5, spline_order=3)
demo_1 = KANDemonstrator(model=model_1, target_fn=lambda x: x**2)

demo_1.train(generate_x_data(-1.0, 1.0, 100))
demo_1.plot(generate_x_data(-4.0, 4.0, 200), "1. Spline Plateau (Symmetric Linear Track)")

# %% [markdown]
# # 2a. Linear Recovery via Feature Engineering
# **Goal:** Provide $x^2$ as an engineered feature. Show that extrapolation now
# works perfectly because the unbroken linear track carries the out-of-bounds scaling.

# %%
model_2a = KAN(layer_dims=[2, 1], grid_size=5, spline_order=3)
demo_2a = KANDemonstrator(
    model=model_2a, target_fn=lambda x: x**2, feature_fn=lambda x: torch.cat([x, x**2], dim=1)
)

demo_2a.train(generate_x_data(-1.0, 1.0, 100))
demo_2a.plot(generate_x_data(-4.0, 4.0, 200), "2. Linear Recovery (Engineered $x^2$)")

# %% [markdown]
# # 2b. Interval Protection (The Collinearity Fix)
# **Goal:** Use the `KANInteraction` module to compute the product using interval
# arithmetic. We feed the network $x^2$, $\cos(\theta)$, and their interaction.
#
# **Result:** Look at the bottom plot—the firewall worked perfectly! It recognized
# the massive out-of-bounds variance of $x$ and slammed the severity $D$ up to 6.0.
# However, the physical prediction (top plot) overshoots. Why? Spurious correlation.
# During training, the network got lazy. Instead of relying purely on the interaction
# feature, it put weight on the raw $x^2$ feature, and used the splines to cancel
# out the error. When extrapolated, the splines clamped, the cancellation stopped,
# and the raw $x^2$ error shot up. This proves why severity tracking is non-negotiable!

# %%
model_2b = KAN(layer_dims=[1, 1], interaction_map=[[0, 0]], grid_size=5, spline_order=3)
demo_2b = KANDemonstrator(model=model_2b, target_fn=lambda x: x**2, feature_fn=lambda x: x)

demo_2b.train(generate_x_data(-1.0, 1.0, 100))
demo_2b.plot(generate_x_data(-4.0, 4.0, 200), "2. Linear Recovery (Engineered $x^2$)")

# %% [markdown]
# # 2c. The Dropout Fix (Forcing Physical Isolation)
# **Goal:** How do we stop the network from using splines as a crutch to hide bad
# linear weights? We introduce **Spline Dropout**. By randomly zeroing out the
# splines during training, the linear track is forced to explain as much as possible of the physical features.
#
# **Result:** The linear track sets the weight of pure $x^2$ to zero, and the weight
# of the interaction feature to 1.0. The physical prediction is now perfectly flat
# (matching the true physics), AND the severity firewall remains fully active.

# %%
model_2c = KAN(
    layer_dims=[1, 1], interaction_map=[[0, 0]], grid_size=5, spline_order=3, spline_dropout=0.05
)
demo_2c = KANDemonstrator(model=model_2c, target_fn=lambda x: x**2, feature_fn=lambda x: x)

demo_2c.train(generate_x_data(-1.0, 1.0, 100))
demo_2c.plot(generate_x_data(-4.0, 4.0, 200), "2c. The Dropout Fix (Perfect Physics + Firewall)")

# %% [markdown]
# # 3a. Protected Interaction Layer
# **Goal:** Use the `KANInteraction` module to compute the product.
# The interval arithmetic accurately assesses the high variance, raises a severe dual $D$,
# and slams the gradient firewall shut. Extrapolation plateaus safely.

# %%
model_3a = KAN(
    layer_dims=[2, 1], interaction_map=[[0, 1]], grid_size=5, spline_order=3, spline_dropout=0.05
)
demo_3a = KANDemonstrator(
    model=model_3a,
    target_fn=lambda x: (x[:, 0:1] ** 2) * torch.cos(x[:, 1:2]),
    feature_fn=lambda x: torch.cat([x[:, 0:1] ** 2, torch.cos(x[:, 1:2])], dim=1),
)

demo_3a.train(generate_x_theta_train())
demo_3a.plot(generate_x_theta_eval(-4.0, 4.0), "3a. Interval Protection (Interaction Firewall)")

# %% [markdown]
# # 3b. Deep Network Feature Discovery
# **Goal:** Remove explicit interaction mapping. Provide just $x^2$ and $\cos(\theta)$
# to a deeper network (`[2, 4, 1]`) to let it learn the interaction. Show that
# the dual mathematically compounds through the linear matrices, protecting the entire depth.

# %%
torch.manual_seed(42)
model_3b = KAN(layer_dims=[2, 4, 1], grid_size=5, spline_order=3, spline_dropout=0.1)
demo_3b = KANDemonstrator(
    model=model_3b,
    target_fn=lambda x: (x[:, 0:1] ** 2) * torch.cos(x[:, 1:2]),
    feature_fn=lambda x: torch.cat([x[:, 0:1] ** 2, torch.cos(x[:, 1:2])], dim=1),
)

demo_3b.train(generate_x_theta_train(steps=800), epochs=1000)
demo_3b.plot(generate_x_theta_eval(-4.0, 4.0), "3b. Deep Discovery (Matrix Dual Routing)")

# %% [markdown]
# **The Takeaway:** The severity firewall ($D$) still spikes perfectly, alerting us that
# we have left the data-driven regime. However, because the deep network relies on
# fragile, unconstrained spline combinations to fake multiplication, the physical
# extrapolation shape becomes highly erratic - different random seeds will give very different extrapolations.
#
# This highlights a crucial philosophical point: high severity ($D$) doesn't inherently
# mean "danger"—it simply means the model is now relying entirely on its structural priors.
# If those priors are unconstrained deep networks, extrapolation is chaotic. But if we
# engineer those priors correctly, we can extrapolate safely and indefinitely.
#
# If deep KANs and automated feature discovery are part of your plans, please proceed
# to `demo_deep.py` to see how we leash the beast!
# %%
