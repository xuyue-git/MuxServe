"""Internal control identities and storage descriptors; never raw GPU pointers."""

from array import array
from dataclasses import dataclass, field
from typing import Any


class IntegrationError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, order=True)
class RequestIdentity:
    model_id: str
    request_id: str
    attempt: int = 0

    def __post_init__(self):
        if not self.model_id or not self.request_id or self.attempt < 0:
            raise IntegrationError("INVALID_ARGUMENT", "Invalid request identity")


@dataclass(frozen=True)
class ModelLayout:
    model_id: str
    num_layers: int
    kv_heads_per_rank: int
    ranks: tuple[int, ...]
    head_dim: int = 128
    block_size: int = 16
    dtype: str = "float16"

    def __post_init__(self):
        if (not self.model_id or self.num_layers < 1 or self.kv_heads_per_rank < 1
                or not self.ranks or len(set(self.ranks)) != len(self.ranks)
                or any(r < 0 for r in self.ranks)):
            raise IntegrationError("INVALID_ARGUMENT", "Invalid model layout")
        if self.head_dim != 128 or self.block_size != 16 or self.dtype != "float16":
            raise IntegrationError("LAYOUT_MISMATCH", "Initial Head-wise layout is FP16/H128/B16")

    @property
    def blocks_per_token_group(self) -> int:
        return self.num_layers * self.kv_heads_per_rank


@dataclass(frozen=True)
class PackedBlocks:
    """One rank's (layer, head, page) physical block table in control messages.

    The owner's store keeps nested lists. Messages carry this immutable form:
    copy and pickle move one bytes object, equality is a byte compare, and a
    rank reads it as one numpy view instead of walking ~10^4 Python ints per
    copy. Packing validates the rectangular shape and int32 range.
    """
    layers: int
    heads: int
    pages: int
    data: bytes

    @classmethod
    def pack(cls, table: Any) -> "PackedBlocks":
        if isinstance(table, PackedBlocks):
            return table
        try:
            layers = len(table)
            heads = len(table[0]) if layers else 0
            pages = len(table[0][0]) if heads else 0
            flat = array("i")
            for layer in table:
                if len(layer) != heads:
                    raise ValueError
                for blocks in layer:
                    if len(blocks) != pages or (pages and set(map(type, blocks)) != _INT_ONLY):
                        raise ValueError
                    flat.extend(blocks)
        except (TypeError, ValueError, IndexError, OverflowError) as exc:
            raise IntegrationError("LAYOUT_MISMATCH", "Block table must be a rectangular int32 mapping") from exc
        if flat.itemsize != 4:
            raise IntegrationError("LAYOUT_MISMATCH", "Platform int is not 32-bit")
        return cls(layers, heads, pages, flat.tobytes())

    def tolist(self) -> list[list[list[int]]]:
        flat = array("i")
        flat.frombytes(self.data)
        values = flat.tolist()
        width = self.pages
        return [[values[(layer * self.heads + head) * width:(layer * self.heads + head + 1) * width]
                 for head in range(self.heads)] for layer in range(self.layers)]

    def array(self):
        import numpy as np
        return np.frombuffer(self.data, dtype=np.int32).reshape(self.layers, self.heads, self.pages)

    def __len__(self) -> int:
        return self.layers


_INT_ONLY = {int}


def pack_tables(tables: dict[int, Any]) -> dict[int, PackedBlocks]:
    return {rank: PackedBlocks.pack(table) for rank, table in tables.items()}


@dataclass(frozen=True)
class KVTarget:
    identity: RequestIdentity
    target_tokens: int
    allocation_id: str | None = None


@dataclass(frozen=True)
class KVView:
    allocation_id: str
    identity: RequestIdentity
    capacity_tokens: int
    valid_tokens: int
    block_size: int
    # Nested lists in the store; PackedBlocks in owner->rank control messages.
    block_tables: dict[int, Any]
    version: int = 0


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    batch_id: str
    views: tuple[KVView, ...]


@dataclass(frozen=True)
class RequestSpec:
    model_id: str
    request_id: str
    input_ids: tuple[int, ...]
    max_new_tokens: int
    timeout_ms: int = 120000
    sampling: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestRef:
    request_id: str
    attempt: int
    expected_output_count: int


@dataclass(frozen=True)
class BatchPlan:
    group_id: str
    batch_id: str
    model_id: str
    stage: str
    requests: tuple[RequestRef, ...]
    purpose: str = "GENERATE"
    prepare_timeout_ms: int = 30000
    # D only: number of consecutive decode forwards granted by one reservation.
    # 1 keeps the original one-token grant protocol unchanged.
    steps: int = 1


@dataclass(frozen=True)
class RuntimeEvent:
    group_id: str
    event_seq: int
    kind: str
    model_id: str
    request_id: str | None = None
    attempt: int | None = None
    batch_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResumeRecord:
    resume_record_id: str
    identity: RequestIdentity
    input_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    max_new_tokens: int
    sampling: dict[str, Any]
    sampling_state: Any
    deadline: float
    config_fingerprint: str
