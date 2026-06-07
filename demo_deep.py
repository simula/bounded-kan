# %% [markdown]
# # PhysKAN: Architecture Demonstrations
# This suite systematically proves the uncertainty-forwarding and gradient
# firewall mechanics of the PhysKAN architecture.

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
# # 3b. Deep Network Feature Discovery
# We repeat this final example from `demo.py` for context, with takeaway
# "... high severity ($D$) doesn't inherently
# mean "danger"—it simply means the model is now relying entirely on its structural priors.
# If those priors are unconstrained deep networks, extrapolation is chaotic. But if we
# engineer those priors correctly, we can extrapolate safely and indefinitely.".

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
# # 3c. Deep Network Feature Discovery, a hybrid approach
# ...hybrid polynomial/kan...

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
demo_3c.plot(generate_x_theta_eval(-4.0, 4.0), "3b. Hybrid Deep Discovery")

# %% [markdown]
# # 4a. Multi-target Surgical Detachment
#
# **Goal:** Demonstrate that the Dual Severity Tracker is not a global panic button, but a surgical, node-specific diagnostic tool.
#
# We will map a system with two outputs:
# * $y_1$ relies heavily on an $x^2$ anomaly.
# * $y_2$ is highly insulated, relying on a stable $\cos(\theta)$ feature and a tiny fractional coefficient of $x$.
#
# **The Expectation:** When $x$ goes violently out of bounds, the network should aggressively firewall $y_1$ (high severity) while leaving $y_2$ almost completely untouched. The severity is quarantined because the underlying linear weights strictly dictate the localized interval routing ($|W| \cdot D$).

# %%
torch.manual_seed(42)


def target_multi(x):
    # y1 is highly sensitive to the out-of-bounds explosion
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
demo_4a.plot(generate_x_theta_eval(-4.0, 4.0), "4a. Multi-target Surgical Detachment")

# %% [markdown]
# # 4b. Perfect Quarantine (The Shallow Solution)
#
# **The Trap of Deep Routing:** In the previous plot, the physical extrapolation for $y_1$ (the orange line in the top panel) looked perfectly flat and safe. However, the severity tracker in the bottom panel screamed that $y_1$ was heavily compromised.
#
# Even though the extrapolations were good, the dual estimate seemed flawed. But it wasn't flawed—it caught the network cheating.
#
# This is a phenomenon called **Cancellation Entanglement**. Because we used a dense deep network (`layer_dims=[3, 4, 2]`), the optimizer didn't cleanly sever the connection by setting the weight to exactly zero. Instead, it routed the anomaly through multiple hidden nodes using massive opposing weights that physically canceled each other out (e.g., computing $+5.0x^2$ on one node and $-5.0x^2$ on another).
#
# Because our Dual Severity firewall mathematically compounds through absolute weights to guarantee safety boundaries, it sees right through the cancellation: $|5.0| + |-5.0| = 10.0$. The tracker correctly warned us that $y_1$ was balancing on a fragile knife's edge of opposing out-of-bounds errors.
#
# **The Harsh Reality:**
# The only real solution, unfortunately, is to avoid deep interactions if you require mathematically guaranteed causal isolation. To achieve perfect surgical detachment, we must remove the dense "mixing pot" and force the model to directly map inputs to outputs.
#
# Let's drop the hidden layers (`[3, 2]`) and watch the firewall perform a flawless quarantine.


# %%
# 4b. Perfect Surgical Detachment (Shallow Architecture)
def target_multi(x):
    # y1 is highly sensitive to the out-of-bounds explosion
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


model_4b = KAN(
    layer_dims=[3, 2],  # NO hidden layers. Pure direct mapping.
    grid_size=5,
    spline_order=3,
    symbolic_order=1,
    spline_dropout=0.8,
)

demo_4b = KANDemonstrator(model=model_4b, target_fn=target_multi, feature_fn=feature_multi)
demo_4b.train(generate_x_theta_train(steps=800), epochs=1000)
demo_4b.plot(generate_x_theta_eval(-4.0, 4.0), "4b. Perfect Quarantine (Shallow)")

# %%
