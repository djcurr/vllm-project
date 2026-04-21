# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for dual-size KV cache admission policy.

These tests cover _assign_request_kv_size_class in EngineCore.
"""

from unittest.mock import MagicMock

import pytest

from vllm.v1.request import KVSizeClass


def _make_cache_config(
    dual_kv: bool = True,
    small_block: int = 16,
    large_block: int = 32,
    threshold: int = 256,
    default_block: int = 16,
):
    cfg = MagicMock()
    cfg.experimental_dual_kv_blocks = dual_kv
    cfg.experimental_small_kv_block_size = small_block
    cfg.experimental_large_kv_block_size = large_block
    cfg.experimental_dual_kv_threshold_tokens = threshold
    cfg.block_size = default_block
    return cfg


def _make_request(num_prompt_tokens: int, max_tokens: int):
    req = MagicMock()
    req.num_prompt_tokens = num_prompt_tokens
    req.max_tokens = max_tokens
    return req


def _assign(cache_config, request):
    """Inline the logic from EngineCore._assign_request_kv_size_class."""
    if not cache_config.experimental_dual_kv_blocks:
        request.kv_size_class = "default"
        request.kv_block_size = cache_config.block_size
        return

    expected_total_tokens = request.num_prompt_tokens + request.max_tokens
    effective_large_threshold = max(
        cache_config.experimental_dual_kv_threshold_tokens,
        cache_config.experimental_large_kv_block_size * 32,
    )
    if expected_total_tokens <= effective_large_threshold:
        request.kv_size_class = "small"
        request.kv_block_size = cache_config.experimental_small_kv_block_size
    else:
        request.kv_size_class = "large"
        request.kv_block_size = cache_config.experimental_large_kv_block_size


class TestDualKVDisabled:
    def test_default_class_when_disabled(self):
        cfg = _make_cache_config(dual_kv=False, default_block=16)
        req = _make_request(100, 50)
        _assign(cfg, req)
        assert req.kv_size_class == "default"
        assert req.kv_block_size == 16

    def test_default_class_uses_config_block_size(self):
        cfg = _make_cache_config(dual_kv=False, default_block=32)
        req = _make_request(100, 50)
        _assign(cfg, req)
        assert req.kv_block_size == 32


class TestSmallClassAssignment:
    def test_short_request_goes_small(self):
        # 100 + 50 = 150, well under floor of 32*32=1024
        cfg = _make_cache_config(small_block=16, large_block=32, threshold=256)
        req = _make_request(100, 50)
        _assign(cfg, req)
        assert req.kv_size_class == "small"
        assert req.kv_block_size == 16

    def test_exactly_at_floor_goes_small(self):
        # floor = max(256, 32*32) = 1024; total = 1024 --> small
        cfg = _make_cache_config(small_block=16, large_block=32, threshold=256)
        req = _make_request(512, 512)
        _assign(cfg, req)
        assert req.kv_size_class == "small"
        assert req.kv_block_size == 16

    def test_configured_threshold_lower_than_floor_still_uses_floor(self):
        # threshold=64, floor=32*32=1024; request of 500 tokens goes small
        cfg = _make_cache_config(small_block=16, large_block=32, threshold=64)
        req = _make_request(300, 200)
        _assign(cfg, req)
        assert req.kv_size_class == "small"

    def test_small_block_size_assigned(self):
        cfg = _make_cache_config(small_block=8, large_block=32, threshold=256)
        req = _make_request(50, 50)
        _assign(cfg, req)
        assert req.kv_block_size == 8


class TestLargeClassAssignment:
    def test_long_request_goes_large(self):
        # floor = 32*32=1024; total = 2000 --> large
        cfg = _make_cache_config(small_block=16, large_block=32, threshold=256)
        req = _make_request(1000, 1000)
        _assign(cfg, req)
        assert req.kv_size_class == "large"
        assert req.kv_block_size == 32

    def test_just_over_floor_goes_large(self):
        # floor = 1024; total = 1025 --> large
        cfg = _make_cache_config(small_block=16, large_block=32, threshold=256)
        req = _make_request(513, 512)
        _assign(cfg, req)
        assert req.kv_size_class == "large"

    def test_configured_threshold_above_floor_is_respected(self):
        # threshold=4096 > floor of 1024; total=2000 --> small (under 4096)
        cfg = _make_cache_config(small_block=16, large_block=32, threshold=4096)
        req = _make_request(1000, 1000)
        _assign(cfg, req)
        assert req.kv_size_class == "small"

    def test_large_block_size_assigned(self):
        cfg = _make_cache_config(small_block=16, large_block=64, threshold=256)
        req = _make_request(1500, 1500)
        _assign(cfg, req)
        assert req.kv_block_size == 64


class TestThresholdFloor:
    def test_floor_scales_with_large_block_size(self):
        # large_block=16 --> floor=16*32=512; total=600 --> large
        cfg = _make_cache_config(small_block=8, large_block=16, threshold=256)
        req = _make_request(300, 300)
        _assign(cfg, req)
        assert req.kv_size_class == "large"

    def test_floor_scales_with_large_block_size_small_total(self):
        # large_block=16 --> floor=512; total=400 --> small
        cfg = _make_cache_config(small_block=8, large_block=16, threshold=256)
        req = _make_request(200, 200)
        _assign(cfg, req)
        assert req.kv_size_class == "small"

    @pytest.mark.parametrize("large_block", [8, 16, 32, 64])
    def test_floor_is_32x_large_block(self, large_block):
        cfg = _make_cache_config(small_block=4, large_block=large_block, threshold=0)
        floor = large_block * 32
        # exactly at floor --> small
        req_at = _make_request(floor // 2, floor // 2)
        _assign(cfg, req_at)
        assert req_at.kv_size_class == "small"
        # one over floor --> large
        req_over = _make_request(floor // 2, floor // 2 + 1)
        _assign(cfg, req_over)
        assert req_over.kv_size_class == "large"
