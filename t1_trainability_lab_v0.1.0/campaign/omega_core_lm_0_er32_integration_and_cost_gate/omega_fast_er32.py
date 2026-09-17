"""Phase 1 ER32 adapter around the real CPU fastpath F implementation."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


_FASTPATH_DIR = Path(__file__).resolve().parent.parent / "omega_core_lm_0_r1_cpu_fastpath_validation"
if str(_FASTPATH_DIR) not in sys.path:
    sys.path.insert(0, str(_FASTPATH_DIR))

from omega_fast_candidate import OmegaCoreLMFast  # noqa: E402


ER32_RANK = 32


class FactorizedVocabulary(nn.Module):
    """Linked low-rank input embedding and output vocabulary projection."""

    def __init__(
        self,
        vocab_size: int,
        rank: int = ER32_RANK,
        dimension: int = 128,
        implementation: str = "efficient",
        *,
        initialize: bool = True,
    ) -> None:
        super().__init__()
        if implementation not in {"explicit", "efficient"}:
            raise ValueError(f"unknown implementation: {implementation}")
        self.vocab_size = vocab_size
        self.rank = rank
        self.dimension = dimension
        self.implementation = implementation
        self.C = nn.Parameter(torch.empty(vocab_size, rank))
        self.U = nn.Parameter(torch.empty(rank, dimension))
        if initialize:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.C, mean=0.0, std=1.0)
        nn.init.normal_(self.U, mean=0.0, std=1.0 / math.sqrt(self.rank))

    def forward(self, tokens: Tensor) -> Tensor:
        if self.implementation == "explicit":
            return F.embedding(tokens, self.C @ self.U)
        return self.C[tokens] @ self.U


class OmegaCoreLMFastER32(OmegaCoreLMFast):
    """Real F core with only its tied lexical interface replaced by C/U."""

    def __init__(
        self,
        *,
        vocab_size: int,
        rank: int = ER32_RANK,
        dimension: int = 128,
        slots: int = 8,
        rounds: int = 4,
        variant: str = "shared",
        implementation: str = "efficient",
        _initialize_factors: bool = True,
    ) -> None:
        if rank != ER32_RANK:
            raise ValueError(f"ER32 rank must be {ER32_RANK}")

        # A one-row placeholder avoids constructing a dense [V,D] parameter.
        super().__init__(
            vocab_size=1,
            dimension=dimension,
            slots=slots,
            rounds=rounds,
            variant=variant,
        )
        self.vocab_size = vocab_size
        self.rank = rank
        self.embedding = FactorizedVocabulary(
            vocab_size,
            rank,
            dimension,
            implementation,
            initialize=_initialize_factors,
        )
        self.implementation = implementation

    @classmethod
    def from_f_reference(
        cls,
        f_reference: OmegaCoreLMFast,
        *,
        experimental_seed: int,
        implementation: str = "efficient",
    ) -> "OmegaCoreLMFastER32":
        """Create ER32 from one fresh F reference without copying its embedding."""

        if not isinstance(f_reference, OmegaCoreLMFast):
            raise TypeError("f_reference must be an OmegaCoreLMFast instance")
        device = next(f_reference.parameters()).device
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = cls(
                vocab_size=f_reference.vocab_size,
                dimension=f_reference.dimension,
                slots=f_reference.slots,
                rounds=f_reference.rounds,
                variant=f_reference.variant,
                implementation=implementation,
                _initialize_factors=False,
            ).to(device)

        # Keep factor pairing independent of K and of all ambient RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(experimental_seed)
            model.embedding.reset_parameters()

        source = f_reference.state_dict()
        destination = model.state_dict()
        with torch.no_grad():
            for name, value in source.items():
                if name == "embedding.weight":
                    continue
                if name not in destination:
                    raise RuntimeError(f"missing ER32 destination state: {name}")
                destination[name].copy_(value)
        return model

    def logits_from_projected(self, projected: Tensor) -> Tensor:
        scale = 1.0 / math.sqrt(self.dimension)
        C, U = self.embedding.C, self.embedding.U
        if self.embedding.implementation == "explicit":
            return projected @ (C @ U).transpose(-2, -1) * scale
        return (projected @ U.transpose(-2, -1)) @ C.transpose(-2, -1) * scale
