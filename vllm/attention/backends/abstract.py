# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import Generic, Optional, Protocol, TypeVar, cast

import torch

from vllm.config.cache import BlockSize
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms.interface import DeviceCapability


class AttentionType:
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """

    DECODER = "decoder"
    """Decoder attention between previous layer Q/K/V."""
    ENCODER = "encoder"
    """Encoder attention between previous layer Q/K/V for encoder-decoder."""
    ENCODER_ONLY = "encoder_only"
    """Encoder attention between previous layer Q/K/V."""
    ENCODER_DECODER = "encoder_decoder"
    """Attention between dec. Q and enc. K/V for encoder-decoder."""


class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    # For some attention backends, we allocate an output tensor before
    # calling the custom op. When piecewise cudagraph is enabled, this
    # makes sure the output tensor is allocated inside the cudagraph.
    accept_output_buffer: bool = False

    # Whether this backend supports receiving pre-quantized query input.
    # If True, the attention layer will handle query quantization instead
    # of the backend, allowing torch.compile to fuse quantization with
    # previous operations.
    # Needs to be worked through for all backends
    # https://github.com/vllm-project/vllm/issues/25584
    supports_quant_query_input: bool = False

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @classmethod
    def make_metadata(cls, *args, **kwargs) -> "AttentionMetadata":
        return cls.get_metadata_cls()(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def get_builder_cls():  # -> Type["AttentionMetadataBuilder"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        raise NotImplementedError

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        raise NotImplementedError

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        supported_head_sizes = cls.get_supported_head_sizes()
        return (not supported_head_sizes) or head_size in supported_head_sizes

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        supported_dtypes = cls.get_supported_dtypes()
        return (not supported_dtypes) or dtype in supported_dtypes

    @classmethod
    def get_supported_kv_cache_dtypes(cls) -> list[Optional[str]]:
        return ["auto"]

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: Optional[str]) -> bool:
        supported_kv_cache_dtypes = cls.get_supported_kv_cache_dtypes()
        return (not supported_kv_cache_dtypes) or (
            kv_cache_dtype is not None and kv_cache_dtype in supported_kv_cache_dtypes
        )

    @classmethod
    def get_supported_block_sizes(cls) -> list[BlockSize]:
        return []

    @classmethod
    def supports_block_size(cls, block_size: int) -> bool:
        try:
            block_size_literal = cast(BlockSize, block_size)
        except ValueError:
            return False
        supported_block_sizes = cls.get_supported_block_sizes()
        return (
            not supported_block_sizes
        ) or block_size_literal in supported_block_sizes

    @classmethod
    def is_mla(cls) -> bool:
        raise NotImplementedError

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return False

    @classmethod
    def get_min_compute_capability(cls) -> Optional[DeviceCapability]:
        return None

    @classmethod
    def get_max_compute_capability(cls) -> Optional[DeviceCapability]:
        return None

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        min_capability = cls.get_min_compute_capability()
        max_capability = cls.get_max_compute_capability()
        return ((min_capability is None) or (capability >= min_capability)) and (
            (max_capability is None) or (capability <= max_capability)
        )

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: Optional[str],
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> Optional[str]:
        return None

    @classmethod
    def validate_configuration(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: Optional[str],
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> list[str]:
        invalid_reasons = []
        if not cls.supports_head_size(head_size):
            invalid_reasons.append("head_size not supported")
        if not cls.supports_dtype(dtype):
            invalid_reasons.append("dtype not supported")
        if not cls.supports_kv_cache_dtype(kv_cache_dtype):
            invalid_reasons.append("kv_cache_dtype not supported")
        if not cls.supports_block_size(block_size):
            invalid_reasons.append("block_size not supported")
        if use_mla != cls.is_mla():
            if use_mla:
                invalid_reasons.append("MLA not supported")
            else:
                invalid_reasons.append("non-MLA not supported")
        if has_sink and not cls.supports_sink():
            invalid_reasons.append("sink setting not supported")
        if use_sparse != cls.is_sparse():
            if use_sparse:
                invalid_reasons.append("sparse not supported")
            else:
                invalid_reasons.append("non-sparse not supported")
        if not cls.supports_compute_capability(device_capability):
            invalid_reasons.append("compute capability not supported")
        combination_reason = cls.supports_combination(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            device_capability,
        )
        if combination_reason is not None:
            invalid_reasons.append(combination_reason)
        return invalid_reasons


class AttentionMetadata:
    pass


T = TypeVar("T", bound=AttentionMetadata)


class AttentionLayer(Protocol):
    _q_scale: torch.Tensor
    _k_scale: torch.Tensor
    _v_scale: torch.Tensor
    _q_scale_float: float
    _k_scale_float: float
    _v_scale_float: float
    _prob_scale: torch.Tensor

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor: ...


class AttentionImpl(ABC, Generic[T]):
    # Whether the attention impl can return the softmax lse for decode.
    # Some features like decode context parallelism require the softmax lse.
    can_return_lse_for_decode: bool = False

    # some attention backends might not always want to return lse
    # even if they can return lse (for efficiency reasons)
    need_to_return_lse_for_decode: bool = False

    dcp_world_size: int
    dcp_rank: int

    def __new__(cls, *args, **kwargs):
        # use __new__ so that all subclasses will call this
        self = super().__new__(cls)
        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.need_to_return_lse_for_decode = (
            self.dcp_world_size > 1 and self.can_return_lse_for_decode
        )
        return self

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[list[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def fused_output_quant_supported(self, quant_key: QuantKey):
        """
        Does this attention implementation support fused output quantization.
        This is used by the AttnFusionPass to only fuse output quantization
        onto implementations that support it.

        :param quant_key: QuantKey object that describes the quantization op
        :return: is fusion supported for this type of quantization
        """
        return False


class MLAAttentionImpl(AttentionImpl[T], Generic[T]):
    @abstractmethod
    def forward(
        self,
        layer: AttentionLayer,
        hidden_states_or_cq: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype != "auto"
