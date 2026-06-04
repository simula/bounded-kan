import math

import torch
import torch.nn.functional as F


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        base_activation=torch.nn.Identity,
        grid_range=(-1.0, 1.0),
        pure_spline_mode=False,
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.pure_spline_mode = pure_spline_mode

        # Static grid formulation
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # The two parallel tracks
        if not self.pure_spline_mode:
            self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        self.base_activation = base_activation()

        self.reset_parameters()

    def reset_parameters(self):
        if not self.pure_spline_mode:
            torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
            # Force zero intercept error by mean-centering each row.
            # This guarantees that the row-sum of base_weight is EXACTLY 0.0,
            # meaning the effective weight row-sum is EXACTLY 1.0 across all channels.
            with torch.no_grad():
                self.base_weight -= self.base_weight.mean(dim=1, keepdim=True)
        torch.nn.init.zeros_(self.spline_weight)

    def b_splines(self, x: torch.Tensor):
        """Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)

        # Determine active spline basis
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
            ) + ((grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:])

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        # ---------------------------------------------------------
        # 1. Linear Track (Takes the raw, un-clamped input)
        # ---------------------------------------------------------
        if self.pure_spline_mode:
            base_output = 0.0
        else:
            effective_weight = self.base_weight + (1.0 / self.in_features)
            base_output = F.linear(self.base_activation(x), effective_weight)

        # ---------------------------------------------------------
        # 2. Spline Track (Plateau and Detach)
        # ---------------------------------------------------------
        lower_bound, upper_bound = self.grid_range

        # Identify features to detach from spline track
        is_oob = (x < lower_bound) | (x > upper_bound)
        any_oob = is_oob.any()

        # Evaluate splines with clamped inputs
        x_clamped = x.clamp(lower_bound, upper_bound) if any_oob else x
        spline_output = F.linear(
            self.b_splines(x_clamped).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        if any_oob:
            spline_output = torch.where(is_oob, spline_output.detach(), spline_output)

        output = (base_output + spline_output).reshape(*original_shape[:-1], self.out_features)
        return output


class KAN(torch.nn.Module):
    """Kolmogorov-Arnold Network (KAN) macro-architecture composed of sequentially stacked KANLinear layers.

    Coordinates deep layer propagation by chaining self-contained Bounded KAN blocks. If
    `pure_spline_mode` is False, the underlying layers maintain an internal scale-preserving
    uniform baseline that seamlessly conserves signal magnitude across dimensional changes
    (expansions/contractions) during extreme out-of-bounds anomalies.

    Args:
        layers_hidden (list[int]): Architectural dimensions mapping from input to output
            features (e.g., [input_dim, hidden_dim, output_dim]).
        grid_size (int, optional): Number of inner intervals partitioning the spline domain
        spline_order (int, optional): Polynomial degree of the local B-spline bases.
        base_activation (torch.nn.Module, optional): Activation function applied exclusively
            to the linear track. Use with caution - see README!
        grid_range (tuple[float, float], optional): Physical bounds `(lower, upper)` defining
            the spline evaluation domain.
        pure_spline_mode (bool, optional): If True, completely disables the linear track
            across all child layers, forcing hard saturation/clipping at boundaries instead
            of proportional linear extrapolation.
    """

    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        base_activation=torch.nn.Identity,
        grid_range=(-1.0, 1.0),
        pure_spline_mode=False,
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_activation=base_activation,
                    grid_range=grid_range,
                    pure_spline_mode=pure_spline_mode,
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
