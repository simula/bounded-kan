import copy
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interaction import KANInteraction, PolynomialSkip


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        base_activation=torch.nn.Identity,
        grid_range=(-1.0, 1.0),
        spline_dropout=0.0,
        pure_spline_mode=False,
        _quiet_init=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.spline_dropout = spline_dropout
        self.pure_spline_mode = pure_spline_mode

        if inspect.isclass(base_activation):
            self.base_activation = base_activation()
        elif isinstance(base_activation, nn.Module):
            # Deepcopy guarantees isolation if passed to multiple layers!
            self.base_activation = copy.deepcopy(base_activation)
        elif callable(base_activation):
            self.base_activation = base_activation
        else:
            raise ValueError("base_activation must be a class, module instance, or callable.")

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

        self.reset_parameters(_quiet_init)

    def reset_parameters(self, quiet=False):
        if not self.pure_spline_mode:
            torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
            # Force zero intercept error by mean-centering each row.
            # This guarantees that the row-sum of base_weight is EXACTLY 0.0,
            # meaning the effective weight row-sum is EXACTLY 1.0 across all channels.
            with torch.no_grad():
                self.base_weight -= self.base_weight.mean(dim=1, keepdim=True)
                if quiet:
                    self.base_weight /= 100.0
        torch.nn.init.zeros_(self.spline_weight)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"grid_size={self.grid_size}, "
            f"spline_order={self.spline_order}, "
            f"grid_range={self.grid_range}"
        )

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

    def forward_internal(self, x: torch.Tensor, d: torch.Tensor):
        """Internal forward pass computing both primal (physics) and dual (severity)."""
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        d = d.reshape(-1, self.in_features)

        # 1. Linear track (primal and dual)
        if self.pure_spline_mode:
            base_output = 0.0
            dual_output = 0.0
        else:
            effective_weight = self.base_weight + (1.0 / self.in_features)
            base_output = F.linear(self.base_activation(x), effective_weight)
            # Dual severity propagates purely through absolute weights
            dual_output = F.linear(d, effective_weight.abs())

        # 2. Spline track (*clamped* primal only)
        lower_bound, upper_bound = self.grid_range
        if self.spline_order == 0:
            # Deg-0 splines lack padding knots, so force the interval open
            upper_bound -= 1e-6
        if torch.jit.is_tracing() or x.min() < lower_bound or x.max() > upper_bound:
            x = x.clamp(lower_bound, upper_bound)

        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        if self.spline_dropout > 0.0:
            spline_output = F.dropout(spline_output, p=self.spline_dropout, training=self.training)

        if torch.is_grad_enabled() and dual_output.max() > 1e-6:
            # Gaussian drop-off: exp(-(x * 2.5)^2)
            # 0.01 -> 99.9% trust (noise is ignored)
            # 0.10 -> 93.9% trust (minor leakage allowed)
            # 0.50 -> 20.9% trust (aggressively pinching off)
            # 1.00 ->  0.1% trust (hard detach)
            trust = torch.exp(-((dual_output.detach() * 2.5) ** 2))
            spline_output = trust * spline_output + (1.0 - trust) * spline_output.detach()

        # 3. Combine and return tuple
        x_final = (base_output + spline_output).reshape(*original_shape[:-1], self.out_features)
        d_final = dual_output.reshape(*original_shape[:-1], self.out_features)
        return x_final, d_final

    def forward(self, x: torch.Tensor, return_dual: bool = False):
        """Public API. Assumes zero incoming severity if used as a standalone layer."""
        d = torch.zeros_like(x)
        x_out, d_out = self.forward_internal(x, d)
        return (x_out, d_out) if return_dual else x_out


class KAN(torch.nn.Module):
    """Kolmogorov-Arnold Network (KAN) macro-architecture composed of sequentially stacked KANLinear layers.

    Coordinates deep layer propagation by chaining self-contained Bounded KAN blocks. If
    `pure_spline_mode` is False, the underlying layers maintain an internal scale-preserving
    uniform baseline that seamlessly conserves signal magnitude across dimensional changes
    (expansions/contractions) during extreme out-of-bounds anomalies.

    Args:
        layer_dims: Architectural dimensions mapping from input to output features (e.g., [input_dim, hidden_dim, output_dim]).
        grid_size: Number of inner intervals partitioning the spline domain
        spline_order: Polynomial degree of the local B-spline bases.
        base_activation: Activation function applied exclusively to the linear track. Change with caution - see README!
        grid_range: Physical bounds `(lower, upper)` defining the spline evaluation domain.
        pure_spline_mode: If True, completely disables the linear track across all child layers, forcing hard
            saturation/clipping at boundaries instead of proportional linear extrapolation.
        spline_dropout: Dropout probability, to encourage asymptote learning.
        interaction_map: Multiplicative feature interaction indices. Use to define explicit cross-terms while preserving
            strict OOB propagation.
    """

    def __init__(
        self,
        layer_dims: list[int],
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: torch.nn.Module = torch.nn.Identity,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        pure_spline_mode: bool = False,
        spline_dropout: float = 0.0,
        interaction_map: list[list[int]] = [],
        symbolic_order: int = 0,
    ):
        super().__init__()
        self.interactor = KANInteraction(interaction_map, grid_range)
        self.layer_dims = layer_dims
        eff_layer_dims = list(layer_dims)
        eff_layer_dims[0] += len(interaction_map)

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(eff_layer_dims, eff_layer_dims[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_activation=base_activation,
                    grid_range=grid_range,
                    spline_dropout=spline_dropout,
                    pure_spline_mode=pure_spline_mode,
                    _quiet_init=symbolic_order > 0,
                )
            )

        # 2. The Shallow Polynomial Skip Connection
        self.symbolic_order = symbolic_order
        if self.symbolic_order > 0:
            self.poly_skip = PolynomialSkip(
                in_features=eff_layer_dims[0], out_features=eff_layer_dims[-1], order=symbolic_order
            )

    def extra_repr(self) -> str:
        # Most information is already in the layers, just add the pre-interactions dim
        return f"in_features={self.layer_dims[0]}"

    def forward(
        self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor], return_dual: bool = False
    ):
        if isinstance(x, tuple):
            if self.interactor.interaction_map:
                raise ValueError(
                    "Both explicit and implicit interactions supplied. This is not supported."
                )
            x, d = x
            if x.shape[1] != self.layers[0].in_features:
                raise ValueError(
                    f"Wrong input dimension {x.shape[1]}, expected {self.layers[0].in_features}"
                )
        else:
            if x.shape[1] != self.layer_dims[0]:
                raise ValueError(
                    f"Wrong input dimension {x.shape[1]}, expected {self.layer_dims[0]}"
                )
            x, d = self.interactor(x)

        if self.symbolic_order > 0:
            poly_out, poly_dual = self.poly_skip(x, d)

        # Route features along with dual through the layers
        for layer in self.layers:
            x, d = layer.forward_internal(x, d)
        if self.symbolic_order > 0:
            x = x + poly_out
            d = d + poly_dual

        return (x, d) if return_dual else x
