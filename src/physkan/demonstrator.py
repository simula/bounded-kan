import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class KANDemonstrator:
    """A utility class for training and visualizing PhysKAN models,
    specifically designed to analyze out-of-bounds (OOB) dual severity tracking.
    """

    def __init__(self, model, target_fn, feature_fn=None):
        self.model = model
        self.target_fn = target_fn
        # If no feature engineering is provided, pass raw features through
        self.feature_fn = feature_fn if feature_fn else lambda x: x

    def train(self, x_raw_train, epochs=500, lr=0.05, weight_decay=1e-4):
        """Trains the model using the provided raw input tensors."""
        y_train = self.target_fn(x_raw_train)
        features = self.feature_fn(x_raw_train)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            # Expecting the network to return (primal_prediction, dual_severity)
            y_pred, d_pred = self.model(features, return_dual=True)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

        return loss.item()

    def predict(self, x_raw):
        """Evaluates the model without tracking gradients."""
        self.model.eval()
        features = self.feature_fn(x_raw)
        with torch.no_grad():
            y_pred, d_pred = self.model(features, return_dual=True)
        return y_pred, d_pred

    def plot(self, x_raw_eval, title="KAN Demonstration", x_axis_idx=0):
        """Plots the physical prediction (primal) and severity tracking (dual).
        x_axis_idx dictates which raw feature column to plot on the x-axis.
        """
        # Sort by the primary plotting axis for clean lines
        sort_idx = torch.argsort(x_raw_eval[:, x_axis_idx])
        x_raw_eval = x_raw_eval[sort_idx]

        y_true = self.target_fn(x_raw_eval)
        y_pred, d_pred = self.predict(x_raw_eval)

        x_plot = x_raw_eval[:, x_axis_idx].numpy()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(title, fontsize=14)

        # 1. Primal Plot (Physics)
        for i in range(y_true.shape[1]):
            label_true = f"True $y_{i}$" if y_true.shape[1] > 1 else "True Physics"
            label_pred = f"Pred $y_{i}$" if y_true.shape[1] > 1 else "KAN Prediction"
            ax1.plot(x_plot, y_pred[:, i].numpy(), "-", linewidth=2, label=label_pred)
            ax1.plot(x_plot, y_true[:, i].numpy(), "k--", alpha=0.7, label=label_true)

        ax1.axvspan(-1.0, 1.0, color="gray", alpha=0.1, label="Nominal Range")
        ax1.set_ylabel("Physical Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Dual Plot (Severity)
        num_targets = d_pred.shape[1]

        # Use a color palette that stands out for warnings (reds, oranges, purples)
        severity_colors = ["#ff0000", "#ff7f0e", "#800080", "#d62728"]

        for i in range(num_targets):
            severity = d_pred[:, i].numpy()
            label_str = "Dual Severity ($D$)" if num_targets == 1 else f"Severity $y_{i}$"
            color = severity_colors[i % len(severity_colors)]

            ax2.plot(x_plot, severity, color=color, linestyle="-", linewidth=2, label=label_str)

        ax2.axvspan(-1.0, 1.0, color="gray", alpha=0.1)
        ax2.set_ylabel("OOB Severity")
        ax2.set_xlabel(f"Raw Input (Feature index {x_axis_idx})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
