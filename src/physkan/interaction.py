import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANInteraction(torch.nn.Module):
    """Computes explicit feature interactions and their OOB duals.

    Args:
        interaction_map (list[list[int]]): List of index lists defining the multiplicative terms.
            Example: [[0, 0], [0, 1]] means add (feats[:,0]^2 and (feats[:,0] * feats[:,1]).
        grid_range (tuple): The nominal boundaries to compute the dual.
    """

    def __init__(self, interaction_map: list[list[int]], grid_range=(-1.0, 1.0)):
        super().__init__()
        self.interaction_map = interaction_map
        self.grid_range = grid_range

    def extra_repr(self) -> str:
        return f"interaction_map={self.interaction_map}, grid_range={self.grid_range}"

    def forward(self, x: torch.Tensor):
        # 1. Genesis: Calculate initial OOB severity of raw inputs
        lower_bound, upper_bound = self.grid_range
        d = F.relu(x - upper_bound) + F.relu(lower_bound - x)
        if not self.interaction_map:
            return x, d

        out_x = [x]
        out_d = [d]
        for term_indices in self.interaction_map:
            # Gather the columns for the current interaction (e.g., [0, 0, 1])
            x_gather = x[:, term_indices]
            d_gather = d[:, term_indices]
            x_abs_gather = x_gather.abs()
            # --- Primal Physics ---
            # Multiply the physical features together
            x_interact = torch.prod(x_gather, dim=1, keepdim=True)
            # --- Dual Severity ---
            # Total Uncertainty Volume minus Nominal Volume
            total_vol = torch.prod(x_abs_gather + d_gather, dim=1, keepdim=True)
            nominal_vol = torch.prod(x_abs_gather, dim=1, keepdim=True)
            d_interact = total_vol - nominal_vol

            out_x.append(x_interact)
            out_d.append(d_interact)

        # Concatenate and return the augmented (primal, dual) tuple
        return torch.cat(out_x, dim=1), torch.cat(out_d, dim=1)


class PolynomialSkip(nn.Module):
    def __init__(self, in_features, out_features, order=2):
        super().__init__()

        # 1. Generate all multi-indices for polynomials up to 'order'
        # e.g., for inputs [0, 1] order 2: (0,), (1,), (0,0), (0,1), (1,1)
        self.combinations = []
        for d in range(1, order + 1):
            combos = itertools.combinations_with_replacement(range(in_features), d)
            self.combinations.extend(list(combos))

        feature_degrees = [len(c) for c in self.combinations]
        degrees_tensor = torch.tensor(feature_degrees, dtype=torch.float32)
        degree_penalty = 2.0 + 2.0 * degrees_tensor
        self.register_buffer("degree_penalty", degree_penalty.unsqueeze(0))

        num_features = len(self.combinations)

        # 2. The standard linear weights
        self.weights = nn.Parameter(torch.randn(out_features, num_features) * 0.1)

        # 3. The Probationary Gates
        self.gates = nn.Parameter(1.0 * torch.ones(out_features, num_features))

    def forward(self, x, dual_input=None):
        poly_features = []
        poly_duals = []

        # --- A. Compute Polynomials & Interval Duals ---
        for combo in self.combinations:
            term = torch.ones_like(x[:, 0:1])
            term_dual = torch.zeros_like(x[:, 0:1]) if dual_input is not None else None

            for idx in combo:
                x_i = x[:, idx : idx + 1]

                # Interval Multiplication for the Dual Severity
                if dual_input is not None:
                    d_i = dual_input[:, idx : idx + 1]
                    # Severity of a product: (|A| + D_A) * (|B| + D_B) - |A * B|
                    new_dual = (torch.abs(term) + term_dual) * (torch.abs(x_i) + d_i) - torch.abs(
                        term * x_i
                    )
                    term_dual = new_dual

                term = term * x_i

            poly_features.append(term)
            if dual_input is not None:
                poly_duals.append(term_dual)

        P_x = torch.cat(poly_features, dim=1)

        # --- B. Apply the -5.0 Sigmoid Gate ---
        active_weights = self.weights * torch.sigmoid(self.gates - self.degree_penalty)

        # --- C. Route the Physical Prediction ---
        out = F.linear(P_x, active_weights)

        # --- D. Route the Dual Severity (The Abs-Weighted Path) ---
        if dual_input is not None:
            D_x = torch.cat(poly_duals, dim=1)
            # You called it: the dual routes through the absolute value of the active weights!
            out_dual = F.linear(D_x, torch.abs(active_weights))
            return out, out_dual

        return out


class LearnedPower(nn.Module):
    def __init__(self, bounds: tuple[float, float], init_val=None):
        super().__init__()
        self.min_val, self.max_val = bounds
        if init_power is None:
            init_power = (self.min_val + self.max_val) / 2.0

        init_logit = math.log((init_power - self.min_val) / (self.max_val - init_power))
        self.raw_alpha = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

    def get_order(self):
        return self.min_val + (self.max_val - self.min_val) * torch.sigmoid(self.raw_alpha)

    def forward(self, x):
        alpha = self.get_order()
        # Epsilon prevents NaN gradients from the derivative of the exponent: x^a * ln(x)
        safe_x = torch.abs(x) + 1e-6
        return torch.sign(x) * (safe_x ** alpha)
