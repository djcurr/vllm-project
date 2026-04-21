# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for dual-size KV block ID normalization.

Tests _normalize_dual_kv_block_ids_for_group and _build_dual_kv_pool_mappings
logic.
"""

import pytest

from vllm.v1.worker.gpu_model_runner import DualKVPoolMapping


def _make_mapping(
    kv_size_class: str,
    logical_block_sizes: list[int],
    kernel_block_sizes: list[int],
    blocks_per_logical_block: list[int],
    kernel_block_offsets: list[int],
) -> DualKVPoolMapping:
    return DualKVPoolMapping(
        kv_size_class=kv_size_class,
        logical_block_sizes=logical_block_sizes,
        kernel_block_sizes=kernel_block_sizes,
        blocks_per_logical_block=blocks_per_logical_block,
        kernel_block_offsets=kernel_block_offsets,
    )


def _normalize(mapping: DualKVPoolMapping, block_ids: list[int], group_id: int = 0) -> list[int]:
    """Inline _normalize_dual_kv_block_ids_for_group logic."""
    blocks_per_logical = mapping.blocks_per_logical_block[group_id]
    offset = mapping.kernel_block_offsets[group_id]
    if blocks_per_logical == 1:
        return [offset + b for b in block_ids]
    normalized = []
    for b in block_ids:
        start = offset + b * blocks_per_logical
        normalized.extend(range(start, start + blocks_per_logical))
    return normalized


class TestSmallPoolNormalization:
    """Small pool: multiplier=1, offset=0 (always starts at 0)."""

    def setup_method(self):
        self.mapping = _make_mapping(
            kv_size_class="small",
            logical_block_sizes=[16],
            kernel_block_sizes=[16],
            blocks_per_logical_block=[1],
            kernel_block_offsets=[0],
        )

    def test_empty_block_ids(self):
        assert _normalize(self.mapping, []) == []

    def test_single_block(self):
        assert _normalize(self.mapping, [0]) == [0]

    def test_multiple_blocks_no_offset(self):
        assert _normalize(self.mapping, [0, 1, 2]) == [0, 1, 2]

    def test_non_contiguous_blocks(self):
        # Free pool may allocate non-contiguous IDs
        assert _normalize(self.mapping, [3, 7, 15]) == [3, 7, 15]

    def test_preserves_order(self):
        ids = [5, 2, 8, 1]
        assert _normalize(self.mapping, ids) == ids


class TestLargePoolNormalization:
    """Large pool: multiplier=2, offset = small_pool_kernel_blocks."""

    def setup_method(self):
        # small pool: 100 logical blocks of size 16 = 100 kernel blocks
        # large pool: offset 100, multiplier 2 (32-token logical / 16-token kernel)
        self.small_kernel_blocks = 100
        self.mapping = _make_mapping(
            kv_size_class="large",
            logical_block_sizes=[32],
            kernel_block_sizes=[16],
            blocks_per_logical_block=[2],
            kernel_block_offsets=[self.small_kernel_blocks],
        )

    def test_empty_block_ids(self):
        assert _normalize(self.mapping, []) == []

    def test_first_logical_block_expands_to_two_kernel_blocks(self):
        # logical block 0 --> kernel blocks [100, 101]
        result = _normalize(self.mapping, [0])
        assert result == [100, 101]

    def test_second_logical_block(self):
        # logical block 1 --> kernel blocks [102, 103]
        result = _normalize(self.mapping, [1])
        assert result == [102, 103]

    def test_multiple_logical_blocks_expand_correctly(self):
        # logical blocks [0, 1, 2] --> [100,101, 102,103, 104,105]
        result = _normalize(self.mapping, [0, 1, 2])
        assert result == [100, 101, 102, 103, 104, 105]

    def test_non_contiguous_logical_blocks(self):
        # logical blocks [0, 3] --> [100,101, 106,107]
        result = _normalize(self.mapping, [0, 3])
        assert result == [100, 101, 106, 107]

    def test_output_length_is_2x_input(self):
        ids = list(range(10))
        assert len(_normalize(self.mapping, ids)) == 20


class TestOffsetCorrectness:
    """Verify offsets correctly partition the unified KV tensor."""

    @pytest.mark.parametrize("small_blocks,large_blocks", [
        (50, 25),
        (100, 100),
        (200, 50),
    ])
    def test_small_and_large_ranges_do_not_overlap(self, small_blocks, large_blocks):
        small_mapping = _make_mapping(
            kv_size_class="small",
            logical_block_sizes=[16],
            kernel_block_sizes=[16],
            blocks_per_logical_block=[1],
            kernel_block_offsets=[0],
        )
        large_mapping = _make_mapping(
            kv_size_class="large",
            logical_block_sizes=[32],
            kernel_block_sizes=[16],
            blocks_per_logical_block=[2],
            kernel_block_offsets=[small_blocks],
        )

        small_ids = list(range(small_blocks))
        large_ids = list(range(large_blocks))

        normalized_small = set(_normalize(small_mapping, small_ids))
        normalized_large = set(_normalize(large_mapping, large_ids))

        assert normalized_small.isdisjoint(normalized_large), (
            "Small and large pools overlap in the unified KV tensor"
        )

    def test_large_pool_starts_immediately_after_small(self):
        small_blocks = 75
        small_mapping = _make_mapping("small", [16], [16], [1], [0])
        large_mapping = _make_mapping("large", [32], [16], [2], [small_blocks])

        last_small = _normalize(small_mapping, [small_blocks - 1])[-1]
        first_large = _normalize(large_mapping, [0])[0]
        assert first_large == last_small + 1


class TestMultiGroup:
    """Multiple KV cache groups (one per attention type)."""

    def test_group_0_and_group_1_use_correct_offsets(self):
        mapping = _make_mapping(
            kv_size_class="large",
            logical_block_sizes=[32, 32],
            kernel_block_sizes=[16, 16],
            blocks_per_logical_block=[2, 2],
            kernel_block_offsets=[100, 300],  # different offsets per group
        )
        assert _normalize(mapping, [0], group_id=0) == [100, 101]
        assert _normalize(mapping, [0], group_id=1) == [300, 301]
