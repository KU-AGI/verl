import torch
import torch.distributed as dist

class AdaptiveEntropyCoefficient:
    """
    Learns an entropy coefficient that can go negative, parameterized via
    ψ on the real line, with α = sinh(ψ) (so α ∈ ℝ and |ψ| ≈ ln|α| for large α).
    """
    def __init__(self, initial_alpha=0.0, target_entropy=1.0, lr=1e-5, device="cpu", max_coeff=1e-3, min_coeff=-1e-3):
        # initialize ψ = arcsinh(initial_alpha)
        init_psi = initial_alpha  # if you want exact arcsinh: torch.asinh(torch.tensor(initial_alpha))
        self.psi = torch.nn.Parameter(torch.asinh(torch.tensor(init_psi, device=device)))
        self.target_entropy = target_entropy
        self.opt = torch.optim.Adam([self.psi], lr=lr)
        self.max_coeff = max_coeff
        self.min_coeff = min_coeff

    @property
    def alpha(self):
        # α = sinh(ψ), can be negative
        alpha = torch.sinh(self.psi)
        return alpha

    def get_alpha(self):
        # α = sinh(ψ), can be negative
        alpha = torch.sinh(self.psi)
        alpha = torch.clamp(alpha, min=self.min_coeff, max=self.max_coeff)
        return alpha.detach()

    def state_dict(self):
        return {
            "psi": self.psi.detach().cpu(),
            "optimizer": self.opt.state_dict(),
            "target_entropy": self.target_entropy,
            "max_coeff": self.max_coeff,
            "min_coeff": self.min_coeff,
        }

    def load_state_dict(self, state):
        device = self.psi.device
        if "psi" in state:
            self.psi.data.copy_(state["psi"].to(device=device, dtype=self.psi.dtype))
        self.target_entropy = state.get("target_entropy", self.target_entropy)
        self.max_coeff = state.get("max_coeff", self.max_coeff)
        self.min_coeff = state.get("min_coeff", self.min_coeff)
        if "optimizer" in state:
            self.opt.load_state_dict(state["optimizer"])
            for opt_state in self.opt.state.values():
                for key, value in opt_state.items():
                    if torch.is_tensor(value):
                        opt_state[key] = value.to(device=device)

    def update(self, entropy):
        ent = entropy.detach().to(device=self.psi.device, dtype=self.psi.dtype)
        # loss = -α * (ent - target)  (so pushes α > 0 when ent < target, α < 0 when ent > target)
        loss = - (self.alpha * (ent - self.target_entropy)).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # compute psi based on clipped alpha
        self.psi.data = torch.asinh(self.get_alpha())
        return loss.item()

    def update_distributed(self, entropy_sum, entropy_count):
        """Update from global token-mean entropy on every actor rank.

        The caller accumulates a complete mini-batch before entering this
        collective, so dynamic micro-batching cannot desynchronize calls.
        """
        stats = torch.stack(
            [
                entropy_sum.detach().to(dtype=torch.float64),
                entropy_count.detach().to(device=entropy_sum.device, dtype=torch.float64),
            ]
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if stats[1].item() <= 0:
            return None
        return self.update(stats[0] / stats[1])
