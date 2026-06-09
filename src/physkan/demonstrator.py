import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .kan import KAN

class TradKAN(KAN):
    """Mimic a traditional KAN implementation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **(dict(base_activation=nn.SiLU, transition_overlap=1.0) | kwargs))

    def forward(self, x, return_dual=False):
        x, dual = self.interactor(x)
        return super().forward((x, 0.0 * dual), return_dual=return_dual)


class KANDemonstrator:
    """A utility class for training and visualizing PhysKAN models,
    specifically designed to analyze out-of-bounds (OOB) dual severity tracking.
    """

    def __init__(self, model, target_fn, feature_fn=None):
        self.model = model
        self.target_fn = target_fn
        # If no feature engineering is provided, pass raw features through
        self.feature_fn = feature_fn if feature_fn else lambda x: x

    def train(self, x_raw_train, epochs=500, lr=0.05, weight_decay=1e-4, hidden_loss=0.0, stiffness_loss=0.0, sobolev_loss=0.0, l1_l2=0.5):
        """Trains the model using the provided raw input tensors."""
        y_train = self.target_fn(x_raw_train)
        features = self.feature_fn(x_raw_train)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            # Expecting the network to return (primal_prediction, dual_severity)
            y_pred, dual_pred, damages = self.model(features, return_dual=True)
            loss = criterion(y_pred, y_train)
            loss += hidden_loss * self.model.get_deep_loss(lambda_l1=l1_l2, lambda_l2=1.0 - l1_l2)
            loss += stiffness_loss * self.model.get_stiffness_loss(lambda_l1=l1_l2, lambda_l2=1.0 - l1_l2)
            if sobolev_loss > 0.0:
                loss += sobolev_loss * self.model.get_sobolev_loss(lambda_l1=l1_l2, lambda_l2=1.0 - l1_l2)
            loss.backward()
            optimizer.step()

        return loss.item()

    def predict(self, x_raw):
        """Evaluates the model without tracking gradients."""
        self.model.eval()
        features = self.feature_fn(x_raw)
        with torch.no_grad():
            return self.model(features, return_dual=True)

    def plot(self, x_raw_eval, title="PhysKAN Demonstration", x_axis_idx: int = 0, plot_dual: bool | None = None, nominal_range: tuple[float, float] = (-1.0, 1.0)):
        """Plots the physical prediction (primal) and severity tracking (dual).
        x_axis_idx dictates which raw feature column to plot on the x-axis.
        """
        # Sort by the primary plotting axis for clean lines
        sort_idx = torch.argsort(x_raw_eval[:, x_axis_idx])
        x_raw_eval = x_raw_eval[sort_idx]

        y_true = self.target_fn(x_raw_eval)
        y_pred, dual_pred, damages_pred = self.predict(x_raw_eval)
        if plot_dual is None:
            plot_dual = (dual_pred != 0.0).any()

        x_plot = x_raw_eval[:, x_axis_idx].numpy()

        if plot_dual:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 2.8))
        fig.suptitle(title, fontsize=14)

        model_type = "TradKAN" if isinstance(self.model, TradKAN) else "PhysKan"

        # 1. Primal Plot (Physics)
        for i in range(y_true.shape[1]):
            label_true = f"True $y_{i}$" if y_true.shape[1] > 1 else "True physics"
            label_pred = f"Pred $y_{i}$" if y_true.shape[1] > 1 else f"{model_type} prediction"
            ax1.plot(x_plot, y_pred[:, i].numpy(), "-", linewidth=2, label=label_pred)
            ax1.plot(x_plot, y_true[:, i].numpy(), "k--", alpha=0.7, label=label_true)

        ax1.axvspan(*nominal_range, color="gray", alpha=0.1, label="Nominal range")
        ax1.set_ylabel("Target value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if plot_dual:
            # 2. Dual Plot (Severity)
            num_targets = dual_pred.shape[1]

            # Use a color palette that stands out for warnings (reds, oranges, purples)
            severity_colors = ["#ff0000", "#ff7f0e", "#800080", "#d62728"]

            for i in range(num_targets):
                color = severity_colors[i % len(severity_colors)]
                label_str = "" if num_targets == 1 else f" $y_{i}$"
                severity = dual_pred[:, i].numpy()
                ax2.plot(x_plot, severity, color=color, linestyle="-", linewidth=2, label=f"Dual severity{label_str}")
                if len(damages_pred) > 1:
                    damage = sum(d_local.amax(dim=-1) for d_local in damages_pred[1:])
                    ax2.plot(x_plot, damage, color=color, linestyle=":", linewidth=1, label=f"Hidden-layer OOB{label_str}")

            ax2.axvspan(*nominal_range, color="gray", alpha=0.1)
            ax2.set_ylabel("OOB severity")
            ax2.set_xlabel(f"Input (feature {x_axis_idx})")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax1.set_xlabel(f"Input (feature {x_axis_idx})")

        plt.tight_layout()
        plt.show()
