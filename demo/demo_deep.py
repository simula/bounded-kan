# %% [markdown]
# # PhysKAN: Architecture demonstrations
#
# This suite systematically proves the uncertainty-forwarding and gradient firewall mechanics of the PhysKAN architecture.

# %%
import torch

from physkan import KAN, KANDemonstrator

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
# # 3b. Deep network feature discovery
#
# We repeat this final example from `demo.py` for context.
# The takeaway is that high severity ($D$) does not inherently mean danger.
# It simply means the model is now relying entirely on its structural priors.
# If those priors are unconstrained deep networks, extrapolation is unpredictable.
# If we engineer those priors correctly, we can extrapolate safely.

# %%
torch.manual_seed(42)
model_3b = KAN(layer_dims=[2, 4, 1], grid_size=5, spline_order=3, spline_dropout=0.1)
demo_3b = KANDemonstrator(
    model=model_3b,
    target_fn=lambda x: (x[:, 0:1] ** 2) * torch.cos(x[:, 1:2]),
    feature_fn=lambda x: torch.cat([x[:, 0:1] ** 2, torch.cos(x[:, 1:2])], dim=1),
)

demo_3b.train(generate_x_theta_train(steps=800), epochs=1000)
demo_3b.plot(generate_x_theta_eval(-4.0, 4.0), "3b. Deep discovery (matrix dual routing)")

# %% [markdown]
# # 3c. Deep network feature discovery, a hybrid approach
#
# To fix the unpredictable extrapolation of the deep network, we introduce the symbolic track.
# By setting `symbolic_order=2`, the model automatically builds a polynomial expansion of the inputs.
# We apply heavy spline dropout (`0.8`) to starve the deep splines, forcing the symbolic track to learn the macro-physics.
# The splines only activate to map local residuals.
# Because the symbolic track provides a predictable structural prior, extrapolation remains stable even when the dual severity ($D$) indicates we have left the training data.

# %%
torch.manual_seed(42)
model_3c = KAN(
    layer_dims=[2, 4, 1], grid_size=5, spline_order=3, symbolic_order=2, spline_dropout=0.8
)
demo_3c = KANDemonstrator(
    model=model_3c,
    target_fn=lambda x: (x[:, 0:1] ** 2) * torch.cos(x[:, 1:2]),
    feature_fn=lambda x: torch.cat([x[:, 0:1] ** 2, torch.cos(x[:, 1:2])], dim=1),
)

demo_3c.train(generate_x_theta_train(steps=800), epochs=1000)
demo_3c.plot(generate_x_theta_eval(-4.0, 4.0), "3c. Hybrid deep discovery")

# %% [markdown]
# # 4a. Multi-target surgical detachment
#
# **Goal:** Demonstrate that the dual severity tracker is a specific diagnostic tool rather than a global error flag.
#
# We will map a system with two outputs:
# * $y_1$ relies on an $x^2$ anomaly.
# * $y_2$ is insulated, relying on a stable $\cos(\theta)$ feature and a fractional coefficient of $x$.
#
# **The Expectation:** When $x$ goes out of bounds, the network should firewall $y_1$ (high severity) while leaving $y_2$ untouched.
# The severity is quarantined because the underlying linear weights strictly dictate the localized interval routing ($|W| \cdot D$).

# %%
torch.manual_seed(42)


def target_multi(x):
    # y1 is sensitive to the out-of-bounds anomaly
    y1 = x[:, 0:1] ** 2
    # y2 is insulated, relying mostly on bounded cos(theta)
    y2 = (1e-3 * x[:, 0:1]) + torch.cos(x[:, 1:2])
    return torch.cat([y1, y2], dim=1)


def feature_multi(x):
    # Provide the exact bases so the symbolic track can perfectly map the weights
    return torch.cat(
        [
            x[:, 0:1],  # Raw x
            x[:, 0:1] ** 2,  # The x^2 anomaly
            torch.cos(x[:, 1:2]),  # The bounded periodic feature
        ],
        dim=1,
    )


# We use 3 inputs for the 3 explicit features.
# symbolic_order=1 lets the global skip-connection effortlessly lock onto the correct features.
model_4a = KAN(
    layer_dims=[3, 4, 2], grid_size=5, spline_order=3, symbolic_order=1, spline_dropout=0.8
)

demo_4a = KANDemonstrator(model=model_4a, target_fn=target_multi, feature_fn=feature_multi)

demo_4a.train(generate_x_theta_train(steps=800), epochs=1000)
demo_4a.plot(generate_x_theta_eval(-4.0, 4.0), "4a. Multi-target surgical detachment")

# %% [markdown]
# # 4b. Perfect quarantine (the shallow solution)
#
# **The trap of deep routing:** In the previous plot, the physical extrapolation for $y_1$ (the orange line in the top panel) looked flat and safe.
# However, the severity tracker in the bottom panel indicated that $y_1$ was compromised.
#
# Even though the extrapolations were good, the dual estimate caught the network balancing opposing weights.
#
# This is a phenomenon called cancellation entanglement.
# Because we used a dense deep network (`layer_dims=[3, 4, 2]`), the optimizer did not cleanly sever the connection by setting the weight to exactly zero.
# Instead, it routed the anomaly through multiple hidden nodes using opposing weights that physically canceled each other out (e.g., computing $+5.0x^2$ on one node and $-5.0x^2$ on another).
#
# Because our dual severity firewall mathematically compounds through absolute weights to guarantee safety boundaries, it sees right through the cancellation: $|5.0| + |-5.0| = 10.0$.
# The tracker correctly warned us that $y_1$ was balancing opposing out-of-bounds errors.
#
# **The reality:** The only solution is to avoid deep interactions if you require guaranteed causal isolation.
# To achieve perfect surgical detachment, we must remove the dense hidden layers and force the model to directly map inputs to outputs.
#
