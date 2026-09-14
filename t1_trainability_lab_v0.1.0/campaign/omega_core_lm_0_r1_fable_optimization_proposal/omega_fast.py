"""omega_fast.py — núcleo recurrente OMEGA CORE-LM-0 R1 optimizado (CPU y GPU).

Misma matemática que `OmegaCoreLM0R1Technical` + `WorkspaceUpdateBlock` del repo
(t1_trainability_lab_v0.1.0), reorganizada para que el bucle secuencial por token
contenga SOLO lo que es verdaderamente secuencial.

Qué cambia respecto al original (todo es equivalente en exactitud fp32, ~1e-6):

  1. Embedding + parte "token" del prelude se calculan para los T tokens de golpe
     (un GEMM) fuera del bucle.  Dentro del bucle sólo queda  W_sum · mean(state).
  2. Readout (RMSNorm + proyección + matmul con el vocabulario 50 257) sale del
     bucle: se apilan los estados [B,T,S,D] y se hace UN solo GEMM grande.
     En el original esto ocurría 256 veces por ventana con matrices [B,128]x[128,50257].
  3. Q/K/V fusionados en un Linear(D, 3D)   → 1 GEMM en vez de 3 por ronda.
  4. Atención sobre slots vía F.scaled_dot_product_attention → 1 kernel en vez de
     matmul + softmax + matmul.
  5. RMSNorm vía F.rms_norm (kernel fusionado) en vez de square/mean/rsqrt/mul/mul.
  6. La depth-embedding de cada ronda se pliega en el bias de fc1
     (fc1(x + d) == fc1(x) + W1·d), precomputado una vez por ventana.
  7. gate * update + state  → torch.addcmul (un kernel).
  8. sigmoid(gate_logits) y F.linear(W_sum) precomputados una vez por ventana.
  9. Pérdida de destilación por chunks de tokens con checkpoint: nunca se
     materializan simultáneamente varias copias de [B·T, 50257] en fp32.

Uso:
    fast = OmegaCoreLMFast.from_reference(ref_model)     # copia pesos
    state, logits = fast.forward_window(tokens, state)   # misma firma
    loss = distillation_loss_chunked(fast, state_seq, teacher_logits, targets, mask)

Para GPU añade además (ver README/benchmark):
    fast.forward_window = torch.compile(fast.forward_window, mode="reduce-overhead")
o bien torch.cuda.make_graphed_callables sobre el paso por token.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

EPS = 1e-6


class FastWorkspaceUpdateBlock(nn.Module):
    """Equivalente a WorkspaceUpdateBlock (SlotMix + CoreMLP + RMSNorm), fusionado."""

    def __init__(self, dimension: int, slots: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.slots = slots
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.out = nn.Linear(dimension, dimension)
        self.fc1 = nn.Linear(dimension, 4 * dimension)
        self.fc2 = nn.Linear(4 * dimension, dimension)
        self.norm_weight = nn.Parameter(torch.ones(dimension))

    @torch.no_grad()
    def load_from_reference(self, block: nn.Module) -> None:
        sm, core = block.slot_mix, block.core.network
        self.qkv.weight.copy_(torch.cat((sm.query.weight, sm.key.weight, sm.value.weight), 0))
        self.qkv.bias.copy_(torch.cat((sm.query.bias, sm.key.bias, sm.value.bias), 0))
        self.out.weight.copy_(sm.output.weight); self.out.bias.copy_(sm.output.bias)
        self.fc1.weight.copy_(core[0].weight); self.fc1.bias.copy_(core[0].bias)
        self.fc2.weight.copy_(core[2].weight); self.fc2.bias.copy_(core[2].bias)
        self.norm_weight.copy_(block.rms_norm.weight)

    def forward(self, state: Tensor, anchor: Tensor, fc1_bias_r: Tensor, gate_r: Tensor) -> Tensor:
        # state, anchor: [B, S, D]; fc1_bias_r: [4D] (bias + W1·depth_r); gate_r: [D]
        d = self.dimension
        q, k, v = self.qkv(state + anchor).split(d, dim=-1)
        # SDPA usa scale = 1/sqrt(D) por defecto: idéntico a SlotMix.scale
        mixed = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)).squeeze(1)
        mixed = self.out(mixed)
        update = self.fc2(F.gelu(F.linear(mixed, self.fc1.weight, fc1_bias_r)))
        return F.rms_norm(torch.addcmul(state, gate_r, update), (d,), self.norm_weight, EPS)


class OmegaCoreLMFast(nn.Module):
    def __init__(self, *, vocab_size: int, dimension: int = 128, slots: int = 8, rounds: int = 4, variant: str = "shared") -> None:
        super().__init__()
        if variant not in {"shared", "untied"}:
            raise ValueError(variant)
        self.vocab_size, self.dimension, self.slots, self.rounds, self.variant = vocab_size, dimension, slots, rounds, variant
        self.embedding = nn.Embedding(vocab_size, dimension)
        self.prelude = nn.Linear(2 * dimension, slots * dimension)
        self.prelude_norm_weight = nn.Parameter(torch.ones(slots * dimension))
        self.blocks = nn.ModuleList(FastWorkspaceUpdateBlock(dimension, slots) for _ in range(1 if variant == "shared" else rounds))
        self.depth_embedding = nn.Embedding(rounds, dimension)
        self.gate_logits = nn.Parameter(torch.full((rounds, dimension), -2.1972245773362196))
        self.readout_norm_weight = nn.Parameter(torch.ones(slots * dimension))
        self.output_projection = nn.Linear(slots * dimension, dimension, bias=False)

    # ------------------------------------------------------------------ pesos
    @classmethod
    def from_reference(cls, ref: nn.Module) -> "OmegaCoreLMFast":
        fast = cls(vocab_size=ref.vocab_size, dimension=ref.dimension, slots=ref.slots, rounds=ref.rounds, variant=ref.variant)
        with torch.no_grad():
            fast.embedding.weight.copy_(ref.embedding.weight)
            fast.prelude.weight.copy_(ref.prelude.weight); fast.prelude.bias.copy_(ref.prelude.bias)
            fast.prelude_norm_weight.copy_(ref.prelude_norm.weight)
            for fb, rb in zip(fast.blocks, ref.blocks):
                fb.load_from_reference(rb)
            fast.depth_embedding.weight.copy_(ref.depth_embedding.weight)
            fast.gate_logits.copy_(ref.gate_logits)
            fast.readout_norm_weight.copy_(ref.readout_norm.weight)
            fast.output_projection.weight.copy_(ref.output_projection.weight)
        return fast.to(next(ref.parameters()).device)

    def initial_state(self, batch_size: int, *, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.slots, self.dimension, device=device)

    # ------------------------------------------------------------------ núcleo
    def _round_constants(self) -> tuple[list[Tensor], Tensor]:
        """Constantes por ronda, calculadas UNA vez por ventana (no por token)."""
        gates = torch.sigmoid(self.gate_logits)                       # [K, D]
        depths = self.depth_embedding.weight                          # [K, D]
        biases = []
        for r in range(self.rounds):
            blk = self.blocks[0] if self.variant == "shared" else self.blocks[r]
            biases.append(blk.fc1.bias + F.linear(depths[r], blk.fc1.weight))   # b1 + W1·depth_r
        return biases, gates

    def recur_states(self, tokens: Tensor, previous_state: Tensor, valid_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Sólo la parte secuencial.  Devuelve (estado final [B,S,D], estados [B,T,S,D])."""
        B, T = tokens.shape
        S, D, SD = self.slots, self.dimension, self.slots * self.dimension
        W = self.prelude.weight                                       # [S·D, 2D]
        # (1) parte de token del prelude para los T tokens: un GEMM
        tok_part = F.linear(self.embedding(tokens), W[:, :D], self.prelude.bias)  # [B, T, S·D]
        W_sum = W[:, D:]                                              # [S·D, D]
        biases, gates = self._round_constants()
        blocks = [self.blocks[0] if self.variant == "shared" else self.blocks[r] for r in range(self.rounds)]

        state = previous_state
        outs = []
        for t in range(T):
            write = tok_part[:, t] + F.linear(state.mean(dim=1), W_sum)
            anchor = F.rms_norm(state.flatten(1) + write, (SD,), self.prelude_norm_weight, EPS).view(B, S, D)
            s = anchor
            for r in range(self.rounds):
                s = blocks[r](s, anchor, biases[r], gates[r])
            if valid_mask is not None:
                s = torch.where(valid_mask[:, t].view(-1, 1, 1), s, state)
            state = s
            outs.append(state)
        return state, torch.stack(outs, dim=1)

    def project(self, states: Tensor) -> Tensor:
        """RMSNorm + proyección S·D→D para todos los tokens de golpe.  [B,T,S,D] → [B,T,D]"""
        flat = states.flatten(2)
        return self.output_projection(F.rms_norm(flat, (flat.shape[-1],), self.readout_norm_weight, EPS))

    def logits_from_projected(self, projected: Tensor) -> Tensor:
        # F.linear con el peso [V, D] evita la transposición no contigua del original
        return F.linear(projected, self.embedding.weight) * (self.dimension ** -0.5)

    def forward_window(self, tokens: Tensor, previous_state: Tensor, valid_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Misma firma/salida que el original: (estado final, logits [B,T,V])."""
        state, states = self.recur_states(tokens, previous_state, valid_mask)
        return state, self.logits_from_projected(self.project(states))


# ---------------------------------------------------------------------- pérdida
def _chunk_loss(projected: Tensor, emb_weight: Tensor, teacher_logits: Tensor, targets: Tensor, weights: Tensor, scale: float, temperature: float) -> Tensor:
    logits = F.linear(projected, emb_weight) * scale                  # [N, V]
    ce = F.cross_entropy(logits, targets, reduction="none")
    s_logp = F.log_softmax(logits / temperature, dim=-1)
    t_p = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(s_logp, t_p, reduction="none").sum(-1) * temperature ** 2
    return torch.stack(((ce * weights).sum(), (kl * weights).sum()))


def distillation_loss_chunked(model: OmegaCoreLMFast, states: Tensor, teacher_logits: Tensor, targets: Tensor, valid_mask: Tensor,
                              *, temperature: float = 2.0, chunk_tokens: int = 512, use_checkpoint: bool = False) -> dict[str, Tensor]:
    """0.5·CE + 0.5·T²·KL, medias enmascaradas: mismo resultado que distillation_loss del repo.

    Los logits [B·T, V] se producen por chunks (y con checkpoint se recomputan en
    backward), así el pico de memoria y el tráfico de RAM caen ~chunk/T veces.
    """
    projected = model.project(states).flatten(0, 1)                   # [N, D]
    V = model.vocab_size
    tl = teacher_logits.reshape(-1, V)
    tg = targets.reshape(-1)
    w = valid_mask.reshape(-1).to(projected.dtype)
    scale = model.dimension ** -0.5
    total = projected.new_zeros(2)
    for i in range(0, projected.shape[0], chunk_tokens):
        args = (projected[i:i + chunk_tokens], model.embedding.weight, tl[i:i + chunk_tokens], tg[i:i + chunk_tokens], w[i:i + chunk_tokens], scale, temperature)
        total = total + (checkpoint(_chunk_loss, *args, use_reentrant=False) if use_checkpoint else _chunk_loss(*args))
    denom = w.sum().clamp_min(1.0)
    ce, kl = total[0] / denom, total[1] / denom
    return {"ce": ce, "kl": kl, "total": 0.5 * ce + 0.5 * kl}


def teacher_targets(teacher_logits: Tensor, temperature: float = 2.0) -> tuple[Tensor, Tensor]:
    """Precalcula (una vez, sin gradiente) lo que la KL necesita del profesor:
    p_t = softmax(z_t/T)  y  su entropía negativa  Σ p_t·log p_t.
    Si el profesor está cacheado, cachea directamente estos dos tensores."""
    with torch.no_grad():
        logp = F.log_softmax(teacher_logits / temperature, dim=-1)
        p = logp.exp()
        return p, (p * logp).sum(-1)


def distillation_loss_lse(model: OmegaCoreLMFast, states: Tensor, teacher_probs: Tensor, teacher_neg_entropy: Tensor,
                          targets: Tensor, valid_mask: Tensor, *, temperature: float = 2.0) -> dict[str, Tensor]:
    """Misma pérdida (0.5·CE + 0.5·T²·KL) escrita con logsumexp:

        CE_i      = lse(z_i) − z_i[y_i]
        T²·KL_i   = T²·[ Σ p_t log p_t  −  (p_t · z_i)/T  +  lse(z_i/T) ]

    Sobre [N, V] sólo hay: 1 GEMM, 2 logsumexp, 1 producto punto y 1 gather.
    El original materializa log_softmax(z), log_softmax(z/T), softmax(z_t) y el
    tensor elementwise de kl_div: ~2–3× más tráfico de memoria en fp32.
    """
    projected = model.project(states).flatten(0, 1)
    logits = model.logits_from_projected(projected)                   # [N, V]
    V = logits.shape[-1]
    tg = targets.reshape(-1)
    w = valid_mask.reshape(-1).to(logits.dtype)
    p_t = teacher_probs.reshape(-1, V)
    lse1 = torch.logsumexp(logits, dim=-1)
    ce = lse1 - logits.gather(1, tg.unsqueeze(1)).squeeze(1)
    lseT = torch.logsumexp(logits / temperature, dim=-1)
    cross = (p_t * logits).sum(-1) / temperature
    kl = (teacher_neg_entropy.reshape(-1) - cross + lseT) * temperature ** 2
    denom = w.sum().clamp_min(1.0)
    ce_m, kl_m = (ce * w).sum() / denom, (kl * w).sum() / denom
    return {"ce": ce_m, "kl": kl_m, "total": 0.5 * ce_m + 0.5 * kl_m}


# ---------------------------------------------------------------------- utilidades
def teacher_logits_once(teacher: nn.Module, source: Tensor, window_tokens: int = 256) -> tuple[Tensor, Tensor]:
    """Un solo forward del profesor con contexto 0:2W y corte para ambas ventanas.

    Por la máscara causal, las posiciones 0:W con contexto 0:2W son idénticas a
    las obtenidas con contexto 0:W (el original las recalculaba dos veces).
    """
    with torch.inference_mode():
        logits = teacher(input_ids=source[:, : 2 * window_tokens]).logits
    return logits[:, :window_tokens].clone(), logits[:, window_tokens: 2 * window_tokens].clone()


def build_teacher_cache(teacher: nn.Module, documents: list[dict], path: str, *, window_tokens: int = 256, temperature: float = 2.0,
                        batch: int = 8, device: str = "cpu", dtype: torch.dtype = torch.float16) -> None:
    """Calcula UNA vez, para todo el corpus, lo que la KL necesita del profesor y lo
    guarda en un memmap:  probs [N_docs, 2W, V] (fp16) y neg_entropy [N_docs, 2W] (fp32).

    Con el corpus fijo (WikiText-2, 513 tokens/doc) el profesor es una función
    determinista de cada documento: recalcularlo en cada update es el 30 % del
    tiempo por update medido en tu campaña CPU.  Lectura por update de 8 docs:
    8·512·50257·2 B ≈ 411 MB, ~0.1–0.2 s desde NVMe vs ~3 s de forward.
    """
    import numpy as np
    n = len(documents); V = teacher.config.vocab_size; L = 2 * window_tokens
    probs = np.lib.format.open_memmap(path + ".probs.npy", mode="w+", dtype=np.float16 if dtype == torch.float16 else np.float32, shape=(n, L, V))
    negent = np.lib.format.open_memmap(path + ".negent.npy", mode="w+", dtype=np.float32, shape=(n, L))
    teacher = teacher.to(device).eval()
    for i in range(0, n, batch):
        src = torch.tensor([d["tokens"][:L] for d in documents[i:i + batch]], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = teacher(input_ids=src).logits.float()
            p, ne = teacher_targets(logits, temperature)
        probs[i:i + batch] = p.to(dtype).cpu().numpy(); negent[i:i + batch] = ne.cpu().numpy()
    probs.flush(); negent.flush()


def cuda_graphed_recurrence(model: OmegaCoreLMFast, tokens_example: Tensor, state_example: Tensor):
    """GPU: captura forward+backward de `recur_states` en un CUDA Graph (shapes estáticas).

    Alternativa a torch.compile(mode="reduce-overhead") cuando no hay Triton
    (p. ej. Windows nativo).  Elimina el coste de lanzamiento de los ~70 kernels
    por token × 256 tokens × (fwd+bwd) que hoy dominan en GPU con batch pequeño.
    """
    class _Recur(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, tokens, state):
            final, states = self.m.recur_states(tokens, state)
            return final, states
    wrapped = _Recur(model)
    return torch.cuda.make_graphed_callables(wrapped, (tokens_example, state_example))
