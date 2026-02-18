# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Track whether upstream flash-attn is available on ROCm.
# Set lazily during _load_fa_ops and never modified afterwards.
_ROCM_FLASH_ATTN_AVAILABLE = False

# Lazy-load C extension symbols to avoid initializing CUDA at import time.
# The actual imports are deferred to first access via module __getattr__.
_fa_ops_loaded = False


def _load_fa_ops() -> None:
    global _fa_ops_loaded, _ROCM_FLASH_ATTN_AVAILABLE
    if _fa_ops_loaded:
        return
    _fa_ops_loaded = True

    if current_platform.is_cuda():
        from vllm._custom_ops import reshape_and_cache_flash
        from vllm.vllm_flash_attn import (  # type: ignore[attr-defined]
            flash_attn_varlen_func,
            get_scheduler_metadata,
        )

        globals()["reshape_and_cache_flash"] = reshape_and_cache_flash
        globals()["flash_attn_varlen_func"] = flash_attn_varlen_func
        globals()["get_scheduler_metadata"] = get_scheduler_metadata

    elif current_platform.is_xpu():
        from vllm import _custom_ops as ops
        from vllm._xpu_ops import xpu_ops

        globals()["reshape_and_cache_flash"] = ops.reshape_and_cache_flash
        globals()["flash_attn_varlen_func"] = xpu_ops.flash_attn_varlen_func
        globals()["get_scheduler_metadata"] = xpu_ops.get_scheduler_metadata

    elif current_platform.is_rocm():
        try:
            from flash_attn import (
                flash_attn_varlen_func,  # type: ignore[no-redef]
            )

            _ROCM_FLASH_ATTN_AVAILABLE = True
            globals()["flash_attn_varlen_func"] = flash_attn_varlen_func
        except ImportError:

            def _flash_attn_varlen_func_stub(*args: Any, **kwargs: Any) -> Any:
                raise ImportError(
                    "ROCm platform requires upstream flash-attn "
                    "to be installed. Please install flash-attn first."
                )

            globals()["flash_attn_varlen_func"] = _flash_attn_varlen_func_stub

        def _get_scheduler_metadata_stub(*args: Any, **kwargs: Any) -> None:
            return None

        globals()["get_scheduler_metadata"] = _get_scheduler_metadata_stub

        from vllm import _custom_ops as ops

        globals()["reshape_and_cache_flash"] = ops.reshape_and_cache_flash


_LAZY_FA_OPS = frozenset(
    {"reshape_and_cache_flash", "flash_attn_varlen_func", "get_scheduler_metadata"}
)


def __getattr__(name: str):
    if name in _LAZY_FA_OPS:
        _load_fa_ops()
        if name in globals():
            return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_flash_attn_version(requires_alibi: bool = False) -> int | None:
    # import here to avoid circular dependencies
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        return 2
    if current_platform.is_rocm():
        # ROCm doesn't use vllm_flash_attn; return None to skip fa_version arg
        return None
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            fa_version_unsupported_reason,
            is_fa_version_supported,
        )

        device_capability = current_platform.get_device_capability()

        assert device_capability is not None

        # 1. default version depending on platform
        fa_version = (
            3 if (device_capability.major == 9 and is_fa_version_supported(3)) else 2
        )

        # 2. override if passed by environment or config
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
        if (
            vllm_config is not None
            and vllm_config.attention_config.flash_attn_version is not None
        ):
            fa_version = vllm_config.attention_config.flash_attn_version

        # 3. fallback for unsupported combinations
        if device_capability.major == 10 and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 on Blackwell platform, "
                "defaulting to FA version 2."
            )
            fa_version = 2

        if requires_alibi and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        if not is_fa_version_supported(fa_version):
            logger.error(
                "Cannot use FA version %d is not supported due to %s",
                fa_version,
                fa_version_unsupported_reason(fa_version),
            )

        assert is_fa_version_supported(fa_version)
        return fa_version
    except (ImportError, AssertionError):
        return None


def flash_attn_supports_fp8() -> bool:
    return (
        get_flash_attn_version() == 3
        and current_platform.is_device_capability_family(90)
    )


def flash_attn_supports_sinks() -> bool:
    if current_platform.is_xpu():
        return True
    else:
        return get_flash_attn_version() == 3


def flash_attn_supports_mla():
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import (
                is_fa_version_supported,
            )

            return is_fa_version_supported(
                3
            ) and current_platform.is_device_capability_family(90)
        except (ImportError, AssertionError):
            pass
    return False


def is_flash_attn_varlen_func_available() -> bool:
    """Check if flash_attn_varlen_func is available.

    This function determines whether the flash_attn_varlen_func imported at module
    level is a working implementation or a stub.

    Platform-specific sources:
    - CUDA: vllm.vllm_flash_attn.flash_attn_varlen_func
    - XPU: xpu_ops.flash_attn_varlen_func
    - ROCm: upstream flash_attn.flash_attn_varlen_func (if available)

    Note: This is separate from the AITER flash attention backend (rocm_aiter_fa.py)
    which uses rocm_aiter_ops.flash_attn_varlen_func. The condition to use AITER is
    handled separately via _aiter_ops.is_aiter_found_and_supported().

    Returns:
        bool: True if a working flash_attn_varlen_func implementation is available.
    """
    if current_platform.is_cuda() or current_platform.is_xpu():
        # CUDA and XPU always have flash_attn_varlen_func available
        return True

    if current_platform.is_rocm():
        # Use the flag set during module import to check if
        # upstream flash-attn was successfully imported
        return _ROCM_FLASH_ATTN_AVAILABLE

    return False
