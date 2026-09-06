"""T1-U0 shared READ -> COMPUTE -> COMMIT primitives.

The module deliberately keeps semantic memory, round control, recurrent
computation, and typed writes separate.  It is an API/contract layer for U0-A;
training orchestration is intentionally left to a later change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import CoreMLP, RMSNorm


OPCODES: tuple[str, ...] = (
    "READ_P",
    "READ_E",
    "ALU_ADD",
    "ALU_SUB",
    "ALU_MUL",
    "ACCUM_W",
    "EMIT",
)
OPCODE_IDS: Mapping[str, int] = {name: index for index, name in enumerate(OPCODES)}

SLOT_P = 0
SLOT_R = 1
SLOT_E = 2
SLOT_W = 3
SLOT_COUNT = 4

ROW_REL = 0
ROW_PAIR = 1
ROW_ASSIGN = 2
ROW_ATTR = 3
ROW_VEC = 4
ROW_COUNT = 5

READ_OPCODE_IDS = frozenset(
    (OPCODE_IDS["READ_P"], OPCODE_IDS["READ_E"], OPCODE_IDS["ACCUM_W"])
)
READ_MODE_BLEND = 0
READ_MODE_SELECT = 1


def _select_payload(
    attention: Tensor,
    values: Tensor,
    selected_index: Tensor,
    valid: Tensor,
    *,
    training: bool,
) -> Tensor:
    """Materialize SELECT payload, optionally with straight-through weights."""
    batch = attention.shape[0]
    hard = torch.zeros_like(attention).scatter(1, selected_index.unsqueeze(-1), 1.0)
    hard = hard * valid.unsqueeze(-1).to(dtype=attention.dtype)
    if training:
        weights = hard + (attention - attention.detach())
        weights = torch.where(valid.unsqueeze(-1), weights, torch.zeros_like(weights))
        return torch.einsum("bm,bmd->bd", weights, values)
    return values.gather(1, selected_index.view(batch, 1, 1).expand(-1, 1, values.shape[-1])).squeeze(1)


@dataclass(frozen=True, init=False)
class ReadResult:
    """Single shared-reader result for one recurrent round."""

    payload: Tensor
    selected_index: Tensor
    attention_soft: Tensor
    selection_margin: Tensor
    valid: Tensor
    read_mode: str | Tensor

    def __init__(
        self,
        payload: Tensor,
        attention_soft: Tensor | None = None,
        selection_margin: Tensor | None = None,
        valid: Tensor | None = None,
        selected_index: Tensor | None = None,
        read_mode: str | Tensor = "BLEND",
        *,
        attention: Tensor | None = None,
        margin: Tensor | None = None,
    ) -> None:
        """Build result, retaining aliases for legacy ablation scripts."""
        if attention_soft is None:
            attention_soft = attention
        if selection_margin is None:
            selection_margin = margin
        if attention_soft is None or selection_margin is None or valid is None:
            raise TypeError("ReadResult requires attention_soft, selection_margin, and valid")
        if selected_index is None:
            selected_index = torch.full((payload.shape[0],), -1, dtype=torch.long, device=payload.device)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "selected_index", selected_index)
        object.__setattr__(self, "attention_soft", attention_soft)
        object.__setattr__(self, "selection_margin", selection_margin)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "read_mode", read_mode)

    @property
    def attention(self) -> Tensor:
        """Legacy alias for diagnostic soft attention."""
        return self.attention_soft

    @property
    def margin(self) -> Tensor:
        """Legacy alias for selection margin."""
        return self.selection_margin


@dataclass(frozen=True)
class CandidateState:
    """Candidates emitted by the shared recurrent core."""

    values: Tensor
    alu_logits: Tensor | None = None

    @property
    def state(self) -> Tensor:
        return self.values


class SharedMemoryReader(nn.Module):
    """One masked, key-addressed reader shared by every task.

    ``memory_keys`` and ``memory_values`` are already canonical D-dimensional
    row projections.  The projections below are shared φ_K/φ_V maps, not
    task-specific readers.  ``row_mask`` is the generator-provided legal-row
    mask μ_r.  A non-memory opcode returns an empty result and performs no
    lookup.
    """

    def __init__(
        self,
        dimension: int = 64,
        *,
        opcode_count: int = len(OPCODES),
        immediate_count: int = 512,
        source_slot_count: int = SLOT_COUNT,
        row_type_count: int = ROW_COUNT,
        attention_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if dimension < 1:
            raise ValueError("dimension must be positive")
        if attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive")
        self.dimension = dimension
        self.attention_temperature = attention_temperature
        self.query = nn.Linear(dimension, dimension, bias=False)
        # Canonical codecs arrive in the shared D-dimensional space.  Keep
        # their identity paths exact; learned codec transforms belong to a
        # later typed-adapter experiment, not U0-A.
        self.condition_projection = nn.Linear(3 * dimension, dimension)
        self.opcode_embedding = nn.Embedding(opcode_count, dimension)
        self.immediate_embedding = nn.Embedding(immediate_count, dimension)
        self.source_slot_embedding = nn.Embedding(source_slot_count, dimension)
        self.row_type_embedding = nn.Embedding(row_type_count, dimension)
        self.input_norm = RMSNorm(dimension)
        self.call_count = 0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.query,):
            nn.init.eye_(layer.weight)
        # Preserve exact canonical payload/lookup at initialization.  These
        # contextual paths remain trainable and can be activated by U0-A.
        nn.init.zeros_(self.condition_projection.weight)
        nn.init.zeros_(self.condition_projection.bias)
        nn.init.zeros_(self.row_type_embedding.weight)

    def reset_call_count(self) -> None:
        self.call_count = 0

    def _batch_vector(self, value: Tensor, *, dimension: int, embedding: nn.Embedding) -> Tensor:
        if value.ndim == 1:
            vector = embedding(value)
            if embedding is self.immediate_embedding:
                vector = torch.where((value == 511).unsqueeze(-1), torch.zeros_like(vector), vector)
            return vector
        if value.ndim == 2 and value.shape[-1] == dimension:
            return value
        raise ValueError("control value must have shape [B] or [B, D]")

    @staticmethod
    def _allowed_types(opcode: Tensor, memory_types: Tensor) -> Tensor:
        allowed = torch.zeros_like(memory_types, dtype=torch.bool)
        allowed |= ((opcode == OPCODE_IDS["READ_P"]).unsqueeze(1) & ((memory_types == ROW_REL) | (memory_types == ROW_ASSIGN)))
        allowed |= ((opcode == OPCODE_IDS["READ_E"]).unsqueeze(1) & ((memory_types == ROW_PAIR) | (memory_types == ROW_ATTR)))
        allowed |= (opcode == OPCODE_IDS["ACCUM_W"]).unsqueeze(1) & (memory_types == ROW_VEC)
        return allowed

    def forward(
        self,
        state: Tensor,
        memory_keys: Tensor,
        memory_values: Tensor,
        memory_types: Tensor,
        row_mask: Tensor,
        opcode: Tensor,
        immediate: Tensor,
        source_slot: Tensor,
        read_mode: str | Tensor = "BLEND",
        diagnostic_read_e_select: bool = False,
    ) -> ReadResult:
        self.call_count += 1
        if state.ndim != 3 or state.shape[1] != SLOT_COUNT or state.shape[-1] != self.dimension:
            raise ValueError("state must have shape [B, 4, D]")
        batch = state.shape[0]
        if memory_keys.ndim == 2:
            memory_keys = memory_keys.unsqueeze(0).expand(batch, -1, -1)
        if memory_values.ndim == 2:
            memory_values = memory_values.unsqueeze(0).expand(batch, -1, -1)
        if memory_keys.ndim != 3 or memory_values.shape != memory_keys.shape:
            raise ValueError("memory keys and values must have shape [B, M, D]")
        if memory_keys.shape[0] != batch or memory_keys.shape[-1] != self.dimension:
            raise ValueError("memory tensors must match state batch and dimension")
        if memory_types.shape != row_mask.shape or memory_types.shape != memory_keys.shape[:2]:
            raise ValueError("memory_types and row_mask must have shape [B, M]")
        if opcode.shape != (batch,) or source_slot.shape != (batch,):
            raise ValueError("opcode and source_slot must have shape [B]")

        opcode_long = opcode.to(dtype=torch.long)
        if isinstance(read_mode, str):
            if read_mode not in {"BLEND", "SELECT"}:
                raise ValueError("read_mode must be SELECT or BLEND")
            mode_ids = torch.full((batch,), READ_MODE_SELECT if read_mode == "SELECT" else READ_MODE_BLEND, dtype=torch.long, device=opcode.device)
        else:
            if read_mode.shape != (batch,):
                raise ValueError("read_mode tensor must have shape [B]")
            mode_ids = read_mode.to(device=opcode.device, dtype=torch.long)
            if torch.any((mode_ids != READ_MODE_BLEND) & (mode_ids != READ_MODE_SELECT)):
                raise ValueError("read_mode values must be READ_MODE_BLEND or READ_MODE_SELECT")
        read_opcodes = torch.isin(opcode_long, torch.tensor(tuple(READ_OPCODE_IDS), device=opcode.device))
        # Narrow diagnostic escape hatch: READ_E may use SELECT only when the
        # caller explicitly labels this intervention. Normal policy remains
        # BLEND for READ_E, and SELECT remains unchanged for ACCUM_W.
        unsupported_select = read_opcodes & (mode_ids == READ_MODE_SELECT) & (opcode_long != OPCODE_IDS["ACCUM_W"])
        allowed_diagnostic_read_e = diagnostic_read_e_select & (opcode_long == OPCODE_IDS["READ_E"])
        if torch.any(unsupported_select & ~allowed_diagnostic_read_e):
            raise ValueError("SELECT is currently supported only for ACCUM_W or diagnostic READ_E")

        source_slot = source_slot.to(dtype=torch.long)
        source = state.gather(1, source_slot.view(batch, 1, 1).expand(-1, 1, self.dimension)).squeeze(1)
        opcode_vector = self._batch_vector(opcode_long, dimension=self.dimension, embedding=self.opcode_embedding)
        immediate_vector = self._batch_vector(immediate, dimension=self.dimension, embedding=self.immediate_embedding)
        source_vector = self.source_slot_embedding(source_slot)
        condition = self.condition_projection(torch.cat((opcode_vector, immediate_vector, source_vector), dim=-1))
        query = self.query(self.input_norm(source) + condition)

        row_type_vectors = self.row_type_embedding(memory_types.to(dtype=torch.long))
        keys = memory_keys + row_type_vectors
        values = memory_values
        logits = torch.einsum("bd,bmd->bm", query, keys)
        logits = logits * (self.attention_temperature / (self.dimension**0.5))
        legal = row_mask.to(dtype=torch.bool) & self._allowed_types(opcode.to(dtype=torch.long), memory_types)
        valid = legal.any(dim=1) & read_opcodes
        safe_logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
        attention = torch.softmax(safe_logits, dim=-1)
        attention = torch.where(legal, attention, torch.zeros_like(attention))
        attention = torch.nan_to_num(attention)
        selected_index = safe_logits.argmax(dim=-1)
        payload_blend = torch.einsum("bm,bmd->bd", attention, values)
        payload_select = _select_payload(attention, values, selected_index, valid, training=self.training)
        payload = torch.where((mode_ids == READ_MODE_SELECT).unsqueeze(-1), payload_select, payload_blend)
        payload = torch.where(valid.unsqueeze(-1), payload, torch.zeros_like(payload))
        selected_index = torch.where(valid, selected_index, torch.full_like(selected_index, -1))

        sorted_logits = safe_logits.sort(dim=-1, descending=True).values
        top1 = sorted_logits[:, 0]
        top2 = sorted_logits[:, 1] if sorted_logits.shape[1] > 1 else torch.zeros_like(top1)
        row_counts = legal.sum(dim=1)
        margin = torch.where(row_counts > 1, top1 - top2, top1)
        margin = torch.where(valid, margin, torch.zeros_like(margin))
        return ReadResult(
            payload=payload,
            selected_index=selected_index,
            attention_soft=attention,
            selection_margin=margin,
            valid=valid,
            read_mode=read_mode,
        )


class SharedRecurrentCore(nn.Module):
    """Single heavy trunk shared by all slots, tasks, and rounds."""

    def __init__(self, dimension: int = 64, *, alu_feature_width: int | None = None, alu_rank: int | None = None) -> None:
        super().__init__()
        self.dimension = dimension
        self.alu_feature_width = alu_feature_width or max(1, min(16, dimension // 4))
        self.alu_rank = alu_rank or max(1, min(8, self.alu_feature_width))
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)
        self.condition = nn.Linear(3 * dimension, dimension)
        self.alu_left_projection = nn.Linear(dimension, self.alu_feature_width, bias=False)
        self.alu_right_projection = nn.Linear(dimension, self.alu_feature_width, bias=False)
        self.alu_opcode_projection = nn.Linear(dimension, self.alu_feature_width, bias=False)
        alu_input_width = 4 * self.alu_feature_width
        self.alu_adapters = nn.ModuleDict(
            {
                name: nn.ModuleDict(
                    {
                        "down": nn.Linear(alu_input_width, self.alu_rank, bias=False),
                        "up": nn.Linear(self.alu_rank, dimension, bias=False),
                    }
                )
                for name in ("ALU_ADD", "ALU_SUB", "ALU_MUL")
            }
        )
        self.mlp = CoreMLP(dimension)
        # U0-A keeps workspace's exact transport path free of learned
        # correction.  U0-C may explicitly unfreeze this branch.
        self.workspace_correction = nn.Linear(dimension, dimension)
        nn.init.zeros_(self.workspace_correction.weight)
        nn.init.zeros_(self.workspace_correction.bias)
        for parameter in self.workspace_correction.parameters():
            parameter.requires_grad_(False)
        self.norm = RMSNorm(dimension)
        self.scale = dimension**-0.5
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)
        for adapter in self.alu_adapters.values():
            nn.init.xavier_uniform_(adapter["down"].weight)
            nn.init.zeros_(adapter["up"].weight)
        nn.init.zeros_(self.mlp.network[-1].weight)
        nn.init.zeros_(self.mlp.network[-1].bias)

    def forward(
        self,
        normalized_state: Tensor,
        opcode_embedding: Tensor,
        immediate_embedding: Tensor,
        read_payload: Tensor,
        slot_type_embeddings: Tensor,
        presence_mask: Tensor,
        opcode: Tensor | None = None,
        slot_read_mask: Tensor | None = None,
    ) -> CandidateState:
        if normalized_state.ndim != 3 or normalized_state.shape[1] != SLOT_COUNT or normalized_state.shape[-1] != self.dimension:
            raise ValueError("normalized_state must have shape [B, 4, D]")
        batch = normalized_state.shape[0]
        if opcode_embedding.shape != (batch, self.dimension) or immediate_embedding.shape != (batch, self.dimension) or read_payload.shape != (batch, self.dimension):
            raise ValueError("round conditioning tensors must have shape [B, D]")
        if slot_type_embeddings.ndim == 2:
            if slot_type_embeddings.shape != (SLOT_COUNT, self.dimension):
                raise ValueError("slot_type_embeddings must have shape [4, D] or [B, 4, D]")
            slot_type_embeddings = slot_type_embeddings.unsqueeze(0).expand(batch, -1, -1)
        if slot_type_embeddings.shape != normalized_state.shape or presence_mask.shape != normalized_state.shape[:2]:
            raise ValueError("slot types/presence mask must match state")
        present = presence_mask.to(dtype=normalized_state.dtype).unsqueeze(-1)
        state = normalized_state * present
        condition = self.condition(torch.cat((opcode_embedding, immediate_embedding, read_payload), dim=-1)).unsqueeze(1)
        conditioned = (state + slot_type_embeddings + condition) * present

        # ALU needs explicit operand/register asymmetry from SU-4.  Keep this
        # adapter local to the register channel; retrieval opcodes remain on
        # the unchanged shared path.
        alu_delta = torch.zeros_like(conditioned[:, SLOT_R, :])
        if opcode is not None:
            if opcode.shape != (batch,):
                raise ValueError("opcode must have shape [B]")
            left = self.alu_left_projection(normalized_state[:, SLOT_R, :])
            right = self.alu_right_projection(immediate_embedding)
            op_state = self.alu_opcode_projection(opcode_embedding)
            alu_features = torch.cat((left, right, op_state, left - right), dim=-1)
            for name, adapter in self.alu_adapters.items():
                indices = (opcode == OPCODE_IDS[name]).nonzero(as_tuple=False).flatten()
                if indices.numel():
                    selected_features = alu_features.index_select(0, indices)
                    selected_delta = adapter["up"](adapter["down"](selected_features))
                    alu_delta = alu_delta.index_copy(0, indices, selected_delta)
        register_mask = conditioned.new_zeros(SLOT_COUNT)
        register_mask[SLOT_R] = 1
        conditioned = conditioned + alu_delta.unsqueeze(1) * register_mask.view(1, SLOT_COUNT, 1)

        query = self.query(conditioned)
        key = self.key(conditioned)
        value = self.value(conditioned)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if slot_read_mask is None:
            legal_slots = presence_mask.unsqueeze(1).expand(-1, SLOT_COUNT, -1)
        else:
            if slot_read_mask.shape != (batch, SLOT_COUNT, SLOT_COUNT):
                raise ValueError("slot_read_mask must have shape [B, 4, 4]")
            legal_slots = presence_mask.unsqueeze(1) & slot_read_mask.to(dtype=torch.bool)
            active_rows = presence_mask.to(dtype=torch.bool)
            if torch.any(active_rows & ~legal_slots.any(dim=-1)):
                raise ValueError("active attention row has no legal source slot")
        scores = scores.masked_fill(~legal_slots, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = torch.nan_to_num(attention)
        mixed = self.output(torch.matmul(attention, value)) * present
        candidate = self.mlp(self.norm(conditioned + mixed)) * present
        workspace_candidate = self.workspace_correction(self.norm(conditioned[:, SLOT_W, :] + mixed[:, SLOT_W, :]))
        workspace_candidate = workspace_candidate * present[:, SLOT_W, :]
        candidate = torch.cat((candidate[:, :SLOT_W, :], workspace_candidate.unsqueeze(1)), dim=1)
        return CandidateState(values=candidate)


class TypedCommit(nn.Module):
    """Apply exact per-opcode writes while preserving inactive slots."""

    def __init__(self, dimension: int = 64) -> None:
        super().__init__()
        self.dimension = dimension
        self.operation_heads = nn.ModuleDict(
            {
                "ALU_ADD": nn.Linear(dimension, 32),
                "ALU_SUB": nn.Linear(dimension, 32),
                "ALU_MUL": nn.Linear(dimension, 32),
            }
        )
        for head in self.operation_heads.values():
            nn.init.eye_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        state: Tensor,
        candidates: CandidateState | Tensor,
        read_result: ReadResult,
        opcode: Tensor,
        destination_slot: Tensor,
        presence_mask: Tensor,
        register_codebook: Tensor | None = None,
        alu_logits: Tensor | None = None,
        workspace_correction: Tensor | None = None,
    ) -> Tensor:
        values = candidates.values if isinstance(candidates, CandidateState) else candidates
        if state.ndim != 3 or state.shape[1:] != (SLOT_COUNT, self.dimension) or values.shape != state.shape:
            raise ValueError("state and candidates must have shape [B, 4, D]")
        batch = state.shape[0]
        if read_result.payload.shape != (batch, self.dimension):
            raise ValueError("read_result payload must have shape [B, D]")
        if opcode.shape != (batch,) or destination_slot.shape != (batch,) or presence_mask.shape != state.shape[:2]:
            raise ValueError("opcode, destination_slot, and presence_mask have invalid shapes")
        if workspace_correction is not None and workspace_correction.shape != (batch, self.dimension):
            raise ValueError("workspace_correction must have shape [B, D]")
        destination = F.one_hot(destination_slot.to(dtype=torch.long), num_classes=SLOT_COUNT).to(dtype=torch.bool).unsqueeze(-1)
        next_state = state

        def write(mask: Tensor, value: Tensor) -> None:
            nonlocal next_state
            selected = mask.view(batch, 1, 1) & destination
            next_state = torch.where(selected, value.unsqueeze(1), next_state)

        write(opcode == OPCODE_IDS["READ_P"], read_result.payload)
        write(opcode == OPCODE_IDS["READ_E"], read_result.payload)
        workspace = state[:, SLOT_W, :] + read_result.payload + values[:, SLOT_W, :]
        if workspace_correction is not None:
            workspace = workspace + workspace_correction
        write(opcode == OPCODE_IDS["ACCUM_W"], workspace)

        is_alu = torch.zeros(batch, dtype=torch.bool, device=opcode.device)
        selected_logits = torch.zeros(batch, 32, dtype=values.dtype, device=values.device)
        if alu_logits is None:
            alu_logits = self.select_alu_logits(values[:, SLOT_R, :], opcode)
        for name in ("ALU_ADD", "ALU_SUB", "ALU_MUL"):
            is_operation = opcode == OPCODE_IDS[name]
            is_alu |= is_operation
            selected_logits = torch.where(is_operation.unsqueeze(-1), alu_logits, selected_logits)
        if is_alu.any():
            if register_codebook is None or register_codebook.shape != (32, self.dimension):
                raise ValueError("register_codebook must have shape [32, D] for ALU commit")
            register = torch.softmax(selected_logits, dim=-1) @ register_codebook
            write(is_alu, register)

        # EMIT has no write. Presence is reapplied after every round, including
        # slots that were initialized to zero and slots left untouched.
        return next_state * presence_mask.to(dtype=next_state.dtype).unsqueeze(-1)

    def select_alu_logits(self, register: Tensor, opcode: Tensor) -> Tensor:
        """Dispatch only active operation heads, leaving inactive heads untouched."""

        if register.ndim != 2 or register.shape[-1] != self.dimension or opcode.shape != (register.shape[0],):
            raise ValueError("register/opcode shapes invalid")
        selected = torch.zeros(register.shape[0], 32, dtype=register.dtype, device=register.device)
        for name in ("ALU_ADD", "ALU_SUB", "ALU_MUL"):
            indices = (opcode == OPCODE_IDS[name]).nonzero(as_tuple=False).flatten()
            if indices.numel():
                selected = selected.index_copy(0, indices, self.operation_heads[name](register.index_select(0, indices)))
        return selected


class _TypedDecoder(nn.Module):
    """Direct tied cosine decoder over caller-supplied canonical codebook."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, state: Tensor, codebook: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[-1] != self.dimension:
            raise ValueError("state must have shape [B, D]")
        if codebook.ndim != 2 or codebook.shape[-1] != self.dimension:
            raise ValueError("codebook must have shape [K, D]")
        return F.normalize(state, dim=-1) @ F.normalize(codebook, dim=-1).transpose(0, 1) * 20.0


class UnifiedT1U0(nn.Module):
    """Composition shell exposing shared objects for U0-A tests/orchestration."""

    TASKS = (
        "pointer_chasing",
        "multi_hop",
        "associative_recall",
        "variable_binding",
        "sequential_update",
        "workspace_accumulation",
    )

    def __init__(self, dimension: int = 64) -> None:
        super().__init__()
        self.dimension = dimension
        self.memory_reader = SharedMemoryReader(dimension)
        self.core = SharedRecurrentCore(dimension)
        self.commit = TypedCommit(dimension)
        self.pointer_decoder = _TypedDecoder(dimension)
        self.evidence_decoder = _TypedDecoder(dimension)
        self.register_decoder = _TypedDecoder(dimension)
        self.workspace_decoder = _TypedDecoder(dimension)
        self.opcode_embedding = nn.Embedding(len(OPCODES), dimension)
        self.immediate_embedding = nn.Embedding(512, dimension)
        # Shared namespaced representation for canonical memory keys, values,
        # initial slots, and typed output decoders.
        self.token_embedding = nn.Embedding(512, dimension)
        self.slot_type_embeddings = nn.Parameter(torch.zeros(SLOT_COUNT, dimension))

    def components_for_task(self, task: str) -> dict[str, nn.Module]:
        if task not in self.TASKS:
            raise ValueError(f"unknown U0 task: {task}")
        decoder: nn.Module
        if task in {"pointer_chasing", "multi_hop"}:
            decoder = self.pointer_decoder
        elif task in {"associative_recall", "variable_binding"}:
            decoder = self.evidence_decoder
        elif task == "sequential_update":
            decoder = self.register_decoder
        else:
            decoder = self.workspace_decoder
        return {"memory_reader": self.memory_reader, "core": self.core, "commit": self.commit, "decoder": decoder}

    @staticmethod
    def normalize_state(state: Tensor, presence_mask: Tensor) -> Tensor:
        if state.ndim != 3 or state.shape[1] != SLOT_COUNT:
            raise ValueError("state must have shape [B, 4, D]")
        present = presence_mask.to(dtype=state.dtype).unsqueeze(-1)
        scale = torch.rsqrt(state.square().mean(dim=-1, keepdim=True) + 1e-6)
        return state * scale * present

    @staticmethod
    def slot_read_mask_for_opcode(opcode: Tensor) -> Tensor:
        """Materialize explicit internal read-set policy per instruction/sample."""
        if opcode.ndim != 1:
            raise ValueError("opcode must have shape [B]")
        batch = opcode.shape[0]
        mask = torch.ones((batch, SLOT_COUNT, SLOT_COUNT), dtype=torch.bool, device=opcode.device)
        is_alu = torch.isin(opcode.to(dtype=torch.long), torch.tensor((OPCODE_IDS["ALU_ADD"], OPCODE_IDS["ALU_SUB"], OPCODE_IDS["ALU_MUL"]), device=opcode.device))
        mask[is_alu, SLOT_R, :] = False
        mask[is_alu, SLOT_R, SLOT_R] = True
        return mask

    def step(
        self,
        state: Tensor,
        memory_keys: Tensor,
        memory_values: Tensor,
        memory_types: Tensor,
        row_mask: Tensor,
        opcode: Tensor,
        immediate: Tensor,
        source_slot: Tensor,
        destination_slot: Tensor,
        presence_mask: Tensor,
        read_mode: str | Tensor = "BLEND",
        diagnostic_read_e_select: bool = False,
        transform_id: Tensor | None = None,
        correction_module: nn.Module | None = None,
        slot_read_mask: Tensor | None = None,
        read_set: str = "explicit",
    ) -> tuple[Tensor, CandidateState, ReadResult]:
        """Execute one auditable READ → COMPUTE → COMMIT round."""

        batch = state.shape[0]
        if opcode.shape != (batch,):
            raise ValueError("opcode must have shape [B]")
        read_required = torch.isin(opcode.to(dtype=torch.long), torch.tensor(tuple(READ_OPCODE_IDS), device=opcode.device)).any()
        if read_required:
            read_result = self.memory_reader(
                state,
                memory_keys,
                memory_values,
                memory_types,
                row_mask,
                opcode,
                immediate,
                source_slot,
                read_mode=read_mode,
                diagnostic_read_e_select=diagnostic_read_e_select,
            )
        else:
            if immediate.ndim == 2:
                payload = torch.zeros((batch, self.dimension), dtype=state.dtype, device=state.device)
            else:
                payload = torch.zeros_like(state[:, 0, :])
            memory_width = memory_types.shape[-1]
            read_result = ReadResult(
                payload=payload,
                attention=torch.zeros((batch, memory_width), dtype=state.dtype, device=state.device),
                margin=torch.zeros(batch, dtype=state.dtype, device=state.device),
                valid=torch.zeros(batch, dtype=torch.bool, device=state.device),
                read_mode=read_mode,
            )
        if (transform_id is None) != (correction_module is None):
            raise ValueError("transform_id and correction_module must be provided together")
        if read_set not in {"legacy", "explicit"}:
            raise ValueError("read_set must be legacy or explicit")
        if read_set == "explicit":
            if slot_read_mask is not None:
                raise ValueError("slot_read_mask is materialized by explicit read_set")
            slot_read_mask = self.slot_read_mask_for_opcode(opcode)
        workspace_correction = None
        if transform_id is not None and correction_module is not None:
            if transform_id.shape != (batch,):
                raise ValueError("transform_id must have shape [B]")
            workspace_correction = correction_module(read_result.payload, state[:, SLOT_W, :], transform_id)
        opcode_embedding = self.opcode_embedding(opcode.to(dtype=torch.long))
        immediate_embedding = self.memory_reader._batch_vector(
            immediate,
            dimension=self.dimension,
            embedding=self.immediate_embedding,
        )
        normalized = self.normalize_state(state, presence_mask)
        candidates = self.core(
            normalized,
            opcode_embedding,
            immediate_embedding,
            read_result.payload,
            self.slot_type_embeddings,
            presence_mask,
            opcode=opcode,
            slot_read_mask=slot_read_mask,
        )
        alu_logits = self.commit.select_alu_logits(candidates.values[:, SLOT_R, :], opcode)
        register_codebook = self.token_embedding(torch.arange(288, 320, device=state.device))
        next_state = self.commit(
            state,
            candidates,
            read_result,
            opcode,
            destination_slot,
            presence_mask,
            register_codebook=register_codebook,
            alu_logits=alu_logits,
            workspace_correction=workspace_correction,
        )
        return next_state, CandidateState(candidates.values, alu_logits), read_result
