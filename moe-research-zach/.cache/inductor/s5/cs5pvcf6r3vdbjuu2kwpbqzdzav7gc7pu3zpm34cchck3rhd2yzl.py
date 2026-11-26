# AOT ID: ['0_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch._inductor.kernel.flex_attention

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /home/zapan4/moe-research/.cache/inductor/6q/c6qaegpfitwmhxxg6hnnsq4ca5qzaweylg4qwptg7no6rllzg7ju.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 8, 16384], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %flex_attention_backward : [num_users=3] = call_function[target=torch.ops.higher_order.flex_attention_backward](args = (%primals_1, %primals_2, %primals_3, %getitem, %getitem_1, %tangents_1, %full_default, %fw_graph, %joint_graph, (%primals_5, %primals_6, %primals_7, %primals_8, %primals_9, %primals_10, %primals_11, %primals_12, 128, 128, %mask_graph), 0.20412414523193154, {ROWS_GUARANTEED_SAFE: False, PRESCALE_QK: False, OUTPUT_LOGSUMEXP: True}, (), (%primals_4,)), kwargs = {})
triton_per_fused_zeros_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[131072, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_zeros_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (32*x0)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (32*x0)), None).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.0
    tmp8 = tmp6 - tmp7
    tl.store(out_ptr1 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/jp/cjpu7vm7tuyjot4qlku5a5ylksfxcbwvfmetkmpsadald75g7fyy.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 8, 16384], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %flex_attention_backward : [num_users=3] = call_function[target=torch.ops.higher_order.flex_attention_backward](args = (%primals_1, %primals_2, %primals_3, %getitem, %getitem_1, %tangents_1, %full_default, %fw_graph, %joint_graph, (%primals_5, %primals_6, %primals_7, %primals_8, %primals_9, %primals_10, %primals_11, %primals_12, 128, 128, %mask_graph), 0.20412414523193154, {ROWS_GUARANTEED_SAFE: False, PRESCALE_QK: False, OUTPUT_LOGSUMEXP: True}, (), (%primals_4,)), kwargs = {})
triton_tem_fused_zeros_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=2,
    num_warps=4,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*i32', 9: '*i32', 10: '*i32', 11: '*i32', 12: '*i32', 13: '*i32', 14: '*i32', 15: '*i32', 16: '*i64', 17: '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_zeros_1', 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_(arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0):
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    PRESCALE_QK : tl.constexpr = False
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    FLOAT32_PRECISION : tl.constexpr = 'ieee'
    IS_DIVISIBLE : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.20412414523193154
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 32
    V_HEAD_DIM : tl.constexpr = 32
    BLOCK_M1 : tl.constexpr = 64
    BLOCK_N1 : tl.constexpr = 64
    BLOCK_M2 : tl.constexpr = 64
    BLOCK_N2 : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    Q = arg_Q
    K = arg_K
    V = arg_V
    LSE = arg_LSE
    DELTA = arg_DELTA
    DO = arg_DO
    DQ = arg_DQ
    DV = arg_DV
    KV_NUM_BLKS = arg_KV_NUM_BLKS
    KV_IDX = arg_KV_IDX
    Q_NUM_BLKS = arg_Q_NUM_BLKS
    Q_IDX = arg_Q_IDX
    FULL_KV_NUM_BLKS = arg_FULL_KV_NUM_BLKS
    FULL_KV_IDX = arg_FULL_KV_IDX
    FULL_Q_NUM_BLKS = arg_FULL_Q_NUM_BLKS
    FULL_Q_IDX = arg_FULL_Q_IDX

    # Sub notation for this kernel:
    #
    # Q: Query, K: Key, V: Value
    # LSE: logsumexp (logsumexp is always stored in fp32 regardless of the input dtype)
    # DELTA: Precomputed sum(OUT*DO, axis=-1)
    # DO: Derivative of Output, DQ: Derivative of Query, DV: Derivative of Value
    # DK: Derivative of Key, is the written to via the store_output call due to some limitations with
    # inductor codegen
    # M: Number of queries, N: Number of keys/values
    # QK_HEAD_DIM: The dimension of the query and key embeddings
    # V_HEAD_DIM: The dimension of the value embeddings
    # z: Batch size, h: Number of heads, m: Number of queries or keys/values, d: Head dim
    # GQA_SHARED_HEADS: number of query heads sharing one kv head in GQA setups.
    # (Modifiable) Performance tuning options
    # BLOCK_M1: when calculating DK & DV, iterate over BLOCK_M1 across the seqlen dim of Q in each thread block.
    # BLOCK_N1: when calculating DK & DV, the thread block size across the seqlen dim of K/V.
    # BLOCK_M2: when calculating DQ, the thread block size across the seqlen dim of Q.
    # BLOCK_N2: when calculating DQ, iterate over BLOCK_N2 across the seqlen dim of K/V in each thread block.
    #
    # The following FULL_* and PARTIAL_* is defined in the block sparse mask grid, rather than the thread block grid.
    # KV_NUM_BLKS: The number of KV blocks (that may or may not require masking) for each query.
    # KV_IDX: The indices of KV blocks (that may or may not require masking) for each query.
    # Q_NUM_BLKS: The number of Q blocks (that may or may not require masking) for each query.
    # Q_IDX: The indices of Q blocks (that may or may not require masking) for each query.
    # FULL_KV_NUM_BLKS: The number of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_KV_IDX: The indices of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_Q_NUM_BLKS: The number of fully unmasked Q blocks (so we don't need masking) for each query.
    # FULL_Q_IDX: The indices of fully unmasked Q blocks (so we don't need masking) for each query.

    # The below are kernel options that can be applied for certain score_mods,
    # or involve a numerics vs. perf tradeoff
    # PRESCALE_QK: Whether to pre-scale QK by 1/sqrt(d) and change of base. Has
    # about 20% more numerical error, but slightly faster.

    # Define strides of inputs
    stride_qz, stride_qh, stride_qm, stride_qd = 4194304, 524288, 32, 1
    stride_kz, stride_kh, stride_kn, stride_kd = 4194304, 524288, 32, 1
    stride_vz, stride_vh, stride_vn, stride_vd = 4194304, 524288, 32, 1
    stride_doz, stride_doh, stride_dom, stride_dod = 4194304, 524288, 32, 1

    stride_dqz, stride_dqh, stride_dqm, stride_dqd = 4194304, 524288, 32, 1
    stride_dvz, stride_dvh, stride_dvm, stride_dvd = 4194304, 524288, 32, 1

    Z = 1
    HQ = 8
    HKV = 8
    Q_LEN = 16384
    KV_LEN = 16384

    MATMUL_PRECISION = Q.dtype.element_ty

    pid = tl.program_id(0)
    NUM_KV_BLOCKS = tl.cdiv(KV_LEN, BLOCK_N1)
    NUM_Q_BLOCKS = tl.cdiv(Q_LEN, BLOCK_M2)

    off_hz = tl.program_id(2)
    off_z = off_hz // HKV # batch idx
    off_hkv = off_hz % HKV # kv head idx

    SPARSE_Z = 1
    SPARSE_HQ = 1

    sparse_idx_z = off_z % SPARSE_Z

    k_adj = (stride_kh * off_hkv + stride_kz * off_z).to(tl.int64)
    v_adj = (stride_vh * off_hkv + stride_vz * off_z).to(tl.int64)
    dv_adj = (stride_dvh * off_hkv + stride_dvz * off_z).to(tl.int64)

    # offset K, V, DV pointers for batch/kv-head
    K += k_adj
    V += v_adj
    DV += dv_adj

    RCP_LN2 = 1.44269504
    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    if pid >= NUM_KV_BLOCKS:
        off_pid = pid - NUM_KV_BLOCKS
        # THIS BLOCK DOES DQ
        SPARSE_Q_MULTIPLE = (SPARSE_Q_BLOCK_SIZE // BLOCK_M2)
        SPARSE_KV_MULTIPLE = (SPARSE_KV_BLOCK_SIZE // BLOCK_N2)
        off_hq2 = off_pid // NUM_Q_BLOCKS + off_hkv * GQA_SHARED_HEADS
        start_m2_block = off_pid % NUM_Q_BLOCKS
        off_pid_mask = start_m2_block // SPARSE_Q_MULTIPLE
        stride_kv_num_blks_h = 128
        stride_kv_idx_h = 16384
        stride_kv_idx_m = 128

        sparse_idx_hq2 = off_hq2 % SPARSE_HQ
        sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq2

        sparse_kv_num_blks_offset = sparse_hz_offset * stride_kv_num_blks_h + off_pid_mask
        sparse_kv_idx_offset = sparse_hz_offset * stride_kv_idx_h + off_pid_mask * stride_kv_idx_m  # noqa: B950

        # Offset Q, DQ, DO, DELTA & LSE. These inputs are offseted by query heads.
        q_adj2 = (stride_qh * off_hq2 + stride_qz * off_z).to(tl.int64)
        do_adj2 = (stride_doh * off_hq2 + stride_doz * off_z).to(tl.int64)
        dq_adj2 = (stride_dqh * off_hq2 + stride_dqz * off_z).to(tl.int64)
        off_chz2 = ((off_z * HQ + off_hq2) * Q_LEN).to(tl.int64)

        Q2 = Q + q_adj2
        DO2 = DO + do_adj2
        # TODO: This does not work if DQ is not the same layout as Q (for example,
        # if Q is broadcasted)
        DQ2 = DQ + dq_adj2
        LSE2 = LSE + off_chz2
        DELTA2 = DELTA + off_chz2

        dq = tl.zeros([BLOCK_M2, QK_HEAD_DIM], dtype=tl.float32)

        start_m2 = start_m2_block * BLOCK_M2
        offs_m2 = start_m2 + tl.arange(0, BLOCK_M2)

        # load Q and do: they stay in SRAM throughout the inner loop.
        if IS_DIVISIBLE:
            q = tl.load(Q2 + offs_m2[:, None] * stride_qm + offs_k[None, :] * stride_qd)
            do = tl.load(DO2 + offs_m2[:, None] * stride_dom + offs_v[None, :] * stride_dod)
        else:
            q = tl.load(Q2 + offs_m2[:, None] * stride_qm + offs_k[None, :] * stride_qd, mask=offs_m2[:, None] < Q_LEN)
            do = tl.load(DO2 + offs_m2[:, None] * stride_dom + offs_v[None, :] * stride_dod, mask=offs_m2[:, None] < Q_LEN)

        if PRESCALE_QK:
            q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

        if IS_DIVISIBLE:
            Di = tl.load(DELTA2 + offs_m2)
            lse = tl.load(LSE2 + offs_m2)
        else:
            Di = tl.load(DELTA2 + offs_m2, mask=offs_m2 < Q_LEN)
            lse = tl.load(LSE2 + offs_m2, mask=offs_m2 < Q_LEN)
        lse = tl.where(lse == -float("inf"), 0.0, lse)
        lse = lse[:, None]

        # ~~~~~~~~~~~ fully unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # KV_IDX and KV_NUM_BLKS are always contiguous.
        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
        sparse_kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)

        offs_n2 = kv_start + tl.arange(0, BLOCK_N2)
        dq = bwd_dq_inner(
            arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
            K, V,
            dq, q, do, Di, lse,
            off_z, off_hq2, offs_m2, offs_n2,
            stride_kn, stride_kd, stride_vn, stride_vd,
            kv_indices, sparse_kv_num_blocks,
            MATMUL_PRECISION,
            IS_FULL_BLOCKS=False,
        )

        if HAS_FULL_BLOCKS:
            # ~~~~~~~~~~~ partial unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # FULL_KV_IDX and FULL_KV_NUM_BLKS are always contiguous.
            kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
            kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
            sparse_kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)

            offs_n2 = kv_start + tl.arange(0, BLOCK_N2)
            dq = bwd_dq_inner(
                arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
                K, V,
                dq, q, do, Di, lse,
                off_z, off_hq2, offs_m2, offs_n2,
                stride_kn, stride_kd, stride_vn, stride_vd,
                kv_indices, sparse_kv_num_blocks,
                MATMUL_PRECISION,
                IS_FULL_BLOCKS=True,
            )

        # Write back dQ.
        dq_ptrs = DQ2 + offs_m2[:, None] * stride_dqm + offs_k[None, :] * stride_dqd
        dq *= SM_SCALE
        if IS_DIVISIBLE:
            tl.store(dq_ptrs, dq)
        else:
            tl.store(dq_ptrs, dq, mask=offs_m2[:, None] < Q_LEN)
    else:
        # THIS BLOCK DOES DK & DV
        SPARSE_Q_MULTIPLE = (SPARSE_Q_BLOCK_SIZE // BLOCK_M1)
        SPARSE_KV_MULTIPLE = (SPARSE_KV_BLOCK_SIZE // BLOCK_N1)

        pid_mask = pid // SPARSE_KV_MULTIPLE

        stride_q_num_blks_h = 128
        stride_q_idx_h = 16384
        stride_q_idx_n = 128

        dv = tl.zeros([BLOCK_N1, V_HEAD_DIM], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N1, QK_HEAD_DIM], dtype=tl.float32)

        start_n1 = pid * BLOCK_N1
        offs_n1 = start_n1 + tl.arange(0, BLOCK_N1)

        # load K and V: they stay in SRAM throughout the inner loop.
        if IS_DIVISIBLE:
            k = tl.load(K + offs_n1[:, None] * stride_kn + offs_k[None, :] * stride_kd)
            v = tl.load(V + offs_n1[:, None] * stride_vn + offs_v[None, :] * stride_vd)
        else:
            k = tl.load(K + offs_n1[:, None] * stride_kn + offs_k[None, :] * stride_kd, mask=offs_n1[:, None] < KV_LEN)
            v = tl.load(V + offs_n1[:, None] * stride_vn + offs_v[None, :] * stride_vd, mask=offs_n1[:, None] < KV_LEN)
        if PRESCALE_QK:
            k = (k * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

        for off_g in range(0, GQA_SHARED_HEADS):
            off_hq1 = off_hkv * GQA_SHARED_HEADS + off_g

            # Offset Q, DQ, DO, DELTA & LSE. These inputs are offseted by query heads.
            q_adj1 = (stride_qh * off_hq1 + stride_qz * off_z).to(tl.int64)
            do_adj1 = (stride_doh * off_hq1 + stride_doz * off_z).to(tl.int64)
            dq_adj1 = (stride_dqh * off_hq1 + stride_dqz * off_z).to(tl.int64)
            off_chz1 = ((off_z * HQ + off_hq1) * Q_LEN).to(tl.int64)

            Q1 = Q + q_adj1
            DO1 = DO + do_adj1
            # TODO: This does not work if DQ is not the same layout as Q (for example,
            # if Q is broadcasted)
            LSE1 = LSE + off_chz1
            DELTA1 = DELTA + off_chz1

            sparse_idx_hq1 = off_hq1 % SPARSE_HQ
            sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq1

            sparse_q_num_blks_offset = sparse_hz_offset * stride_q_num_blks_h + pid_mask
            sparse_q_idx_offset = sparse_hz_offset * stride_q_idx_h + pid_mask * stride_q_idx_n  # noqa: B950

            # ~~~~~~~~~~~~~~~ fully unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Q_IDX and Q_NUM_BLKS are always contiguous.
            q_indices = Q_IDX + sparse_q_idx_offset
            q_start = tl.load(q_indices) * SPARSE_Q_BLOCK_SIZE # first q block we're loading
            sparse_q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)

            offs_m1 = q_start + tl.arange(0, BLOCK_M1)
            dk, dv = bwd_dkdv_inner(
                arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
                Q1, DO1, DELTA1, LSE1,
                dk, dv, k, v,
                off_z, off_hq1, offs_n1, offs_m1,
                stride_qm, stride_qd, stride_dom, stride_dod,
                q_indices, sparse_q_num_blocks,
                MATMUL_PRECISION,
                IS_FULL_BLOCKS=False,
            )


            if HAS_FULL_BLOCKS:
                # ~~~~~~~~~~~~~~~ fully unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # FULL_Q_IDX and FULL_Q_NUM_BLKS are always contiguous.
                q_indices = FULL_Q_IDX + sparse_q_idx_offset
                q_start = tl.load(q_indices) * SPARSE_Q_BLOCK_SIZE # first q block we're loading
                sparse_q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)

                offs_m1 = q_start + tl.arange(0, BLOCK_M1)
                dk, dv = bwd_dkdv_inner(
                    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
                    Q1, DO1, DELTA1, LSE1,
                    dk, dv, k, v,
                    off_z, off_hq1, offs_n1, offs_m1,
                    stride_qm, stride_qd, stride_dom, stride_dod,
                    q_indices, sparse_q_num_blocks,
                    MATMUL_PRECISION,
                    IS_FULL_BLOCKS=True,
                )

        # Write back dV and dK.
        dv_ptrs = DV + offs_n1[:, None] * stride_dvm + offs_v[None, :] * stride_dvd

        index_n = offs_n1[:, None]
        index_k = offs_k[None, :]

        if IS_DIVISIBLE:
            tl.store(dv_ptrs, dv)
        else:
            tl.store(dv_ptrs, dv, mask=index_n < KV_LEN)

        dk *= SM_SCALE
        mask = index_n < KV_LEN
        xindex = index_k + (32*index_n) + (524288*off_hkv) + (4194304*off_z)
        tl.store(out_ptr0 + (tl.broadcast_to(index_k + (32*index_n) + (524288*off_hkv), dk.shape)), dk, mask)

@triton.jit
def bwd_dq_inner(
    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
    K, V,  # pointers
    dq, q, do, Di, lse,
    off_z, off_hq, offs_m2, offs_n2,
    stride_kn, stride_kd, stride_vn, stride_vd,
    kv_indices, sparse_kv_num_blocks,
    MATMUL_PRECISION,
    IS_FULL_BLOCKS,
):
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    PRESCALE_QK : tl.constexpr = False
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    FLOAT32_PRECISION : tl.constexpr = 'ieee'
    IS_DIVISIBLE : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.20412414523193154
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 32
    V_HEAD_DIM : tl.constexpr = 32
    BLOCK_M1 : tl.constexpr = 64
    BLOCK_N1 : tl.constexpr = 64
    BLOCK_M2 : tl.constexpr = 64
    BLOCK_N2 : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128

    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N2)
    RCP_LN2: tl.constexpr = 1.44269504
    Q_LEN = 16384
    KV_LEN = 16384

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    kT_ptrs = K + offs_n2[None, :] * stride_kn + offs_k[:, None] * stride_kd
    vT_ptrs = V + offs_n2[None, :] * stride_vn + offs_v[:, None] * stride_vd
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)

    hi = tl.minimum(sparse_kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N2), 1))
    if not IS_DIVISIBLE:
        if hi >= 1:
            for start_n in range(0, hi - 1):
                dq = bwd_dq_block_mn(
                    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
                    dq, q, kT_ptrs, vT_ptrs, do, Di, lse, Q_LEN, KV_LEN,
                    off_z, off_hq, offs_m2, offs_n2,
                    stride_kn, stride_kd, stride_vn, stride_vd,
                    kv_indices, sparse_kv_num_blocks,
                    MATMUL_PRECISION, RCP_LN2,
                    IS_FULL_BLOCKS,
                )

                # Increment pointers.
                offset = get_offset_for_next_block(
                    start_n, kv_indices, sparse_kv_num_blocks,
                    SPARSE_KV_BLOCK_SIZE, SPARSE_KV_MULTIPLE, BLOCK_N2
                )

                kT_ptrs += offset * stride_kn
                vT_ptrs += offset * stride_vn

                offs_n2 += offset

            dq = bwd_dq_block_mn(
                arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
                dq, q, kT_ptrs, vT_ptrs, do, Di, lse, Q_LEN, KV_LEN,
                off_z, off_hq, offs_m2, offs_n2,
                stride_kn, stride_kd, stride_vn, stride_vd,
                kv_indices, sparse_kv_num_blocks,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
            )
    else:
        for start_n in range(0, hi):
            dq = bwd_dq_block_mn(
                arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
                dq, q, kT_ptrs, vT_ptrs, do, Di, lse, Q_LEN, KV_LEN,
                off_z, off_hq, offs_m2, offs_n2,
                stride_kn, stride_kd, stride_vn, stride_vd,
                kv_indices, sparse_kv_num_blocks,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS,
            )

            # Increment pointers.
            offset = get_offset_for_next_block(
                start_n, kv_indices, sparse_kv_num_blocks,
                SPARSE_KV_BLOCK_SIZE, SPARSE_KV_MULTIPLE, BLOCK_N2
            )

            kT_ptrs += offset * stride_kn
            vT_ptrs += offset * stride_vn

            offs_n2 += offset

    return dq


@triton.jit
def bwd_dq_block_mn(
    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
    dq, q, kT_ptrs, vT_ptrs, do, Di, lse, Q_LEN, KV_LEN,
    off_z, off_hq, offs_m2, offs_n2,
    stride_kn, stride_kd, stride_vn, stride_vd,
    kv_indices, sparse_kv_num_blocks,
    MATMUL_PRECISION, RCP_LN2,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,
):
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    PRESCALE_QK : tl.constexpr = False
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    FLOAT32_PRECISION : tl.constexpr = 'ieee'
    IS_DIVISIBLE : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.20412414523193154
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 32
    V_HEAD_DIM : tl.constexpr = 32
    BLOCK_M1 : tl.constexpr = 64
    BLOCK_N1 : tl.constexpr = 64
    BLOCK_M2 : tl.constexpr = 64
    BLOCK_N2 : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128


    if IS_DIVISIBLE:
        kT = tl.load(kT_ptrs)
    else:
        kT = tl.load(kT_ptrs, mask=offs_n2[None, :] < KV_LEN)
    qk = tl.dot(q, kT, input_precision=FLOAT32_PRECISION)
    if not PRESCALE_QK:
        qk *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    pre_mod_scores = qk
    if CHECK_BLOCK_BOUNDARY:
        m = offs_m2[:, None] % Q_LEN
        n = offs_n2[None, :] % KV_LEN
    else:
        m = offs_m2[:, None]
        n = offs_n2[None, :]
    post_mod_scores = (qk)


    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        post_mod_scores = tl.where(offs_n2[None, :] < KV_LEN, post_mod_scores, float("-inf"))

    if not IS_FULL_BLOCKS:
        tmp0 = tl.load(in_ptr16 + (m)) == tl.load(in_ptr16 + (n))
        tmp1 = (n) <= (m)
        tmp2 = tmp0 & tmp1
        tmp3 = (m) - (n)
        tmp4 = tl.full([1], 128, tl.int32)
        tmp5 = tmp3 < tmp4
        tmp6 = tmp2 & tmp5
        mask_mod_output = tmp6


        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n2[None, :] < KV_LEN, mask_mod_output, float("-inf"))
        # apply mask for partial masked block
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    p = tl.math.exp2(post_mod_scores - lse)
    # Compute dP and dS.
    if IS_DIVISIBLE:
        vT = tl.load(vT_ptrs)
    else:
        vT = tl.load(vT_ptrs, mask=offs_n2[None, :] < KV_LEN)
    dp = tl.dot(do, vT, input_precision=FLOAT32_PRECISION)
    ds = p * (dp - Di[:, None])
    # ~~~~~~~~~~~~~~~~~~~ Apply joint modification  ~~~~~~~~~~~~~~~~~~~
    grad_scores = (ds)

    if CHECK_BLOCK_BOUNDARY:
        grad_scores = tl.where(offs_n2[None, :] < KV_LEN, grad_scores, 0.0)

    ds = grad_scores

    if not IS_FULL_BLOCKS:
        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n2[None, :] < KV_LEN, mask_mod_output, float("-inf"))
        # (grads) apply mask for partially unmasked block
        ds = tl.where(mask_mod_output, ds, 0.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ds = ds.to(MATMUL_PRECISION)
    # Compute dQ.
    dq += tl.dot(ds, tl.trans(kT), input_precision=FLOAT32_PRECISION)

    return dq


@triton.jit
def bwd_dkdv_inner(
    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
    Q, DO, DELTA, LSE, # pointers
    dk, dv, k, v,
    off_z, off_hq, offs_n1, offs_m1,
    stride_qm, stride_qd, stride_dom, stride_dod,
    q_indices, sparse_q_num_blocks,
    MATMUL_PRECISION,
    IS_FULL_BLOCKS,
):
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    PRESCALE_QK : tl.constexpr = False
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    FLOAT32_PRECISION : tl.constexpr = 'ieee'
    IS_DIVISIBLE : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.20412414523193154
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 32
    V_HEAD_DIM : tl.constexpr = 32
    BLOCK_M1 : tl.constexpr = 64
    BLOCK_N1 : tl.constexpr = 64
    BLOCK_M2 : tl.constexpr = 64
    BLOCK_N2 : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128

    SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M1)
    RCP_LN2: tl.constexpr = 1.44269504
    Q_LEN = 16384
    KV_LEN = 16384

    offs_k = tl.arange(0, QK_HEAD_DIM)
    offs_v = tl.arange(0, V_HEAD_DIM)

    qT_ptrs = Q + offs_m1[None, :] * stride_qm + offs_k[:, None] * stride_qd
    do_ptrs = DO + offs_m1[:, None] * stride_dom + offs_v[None, :] * stride_dod
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    hi = tl.minimum(sparse_q_num_blocks * SPARSE_Q_MULTIPLE, tl.maximum(tl.cdiv(Q_LEN, BLOCK_M1), 1))

    if not IS_DIVISIBLE:
        if hi >= 1:
            for start_m in range(0, hi - 1):
                dk, dv = bwd_dkdv_block_mn(
                    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
                    dk, dv, qT_ptrs, k, v, do_ptrs, DELTA, LSE, Q_LEN, KV_LEN,
                    off_z, off_hq, offs_n1, offs_m1,
                    stride_qm, stride_qd, stride_dom, stride_dod,
                    q_indices, sparse_q_num_blocks,
                    MATMUL_PRECISION, RCP_LN2,
                    IS_FULL_BLOCKS,
                )
                # Increment pointers.
                offset = get_offset_for_next_block(
                    start_m, q_indices, sparse_q_num_blocks,
                    SPARSE_Q_BLOCK_SIZE, SPARSE_Q_MULTIPLE, BLOCK_M1
                )

                qT_ptrs += offset * stride_qm
                do_ptrs += offset * stride_dom

                offs_m1 += offset

            dk, dv = bwd_dkdv_block_mn(
                arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
                dk, dv, qT_ptrs, k, v, do_ptrs, DELTA, LSE, Q_LEN, KV_LEN,
                off_z, off_hq, offs_n1, offs_m1,
                stride_qm, stride_qd, stride_dom, stride_dod,
                q_indices, sparse_q_num_blocks,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
            )
    else:
        for start_m in range(0, hi):
            dk, dv = bwd_dkdv_block_mn(
                arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
                dk, dv, qT_ptrs, k, v, do_ptrs, DELTA, LSE, Q_LEN, KV_LEN,
                off_z, off_hq, offs_n1, offs_m1,
                stride_qm, stride_qd, stride_dom, stride_dod,
                q_indices, sparse_q_num_blocks,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS,
            )
            # Increment pointers.
            offset = get_offset_for_next_block(
                start_m, q_indices, sparse_q_num_blocks,
                SPARSE_Q_BLOCK_SIZE, SPARSE_Q_MULTIPLE, BLOCK_M1
            )

            qT_ptrs += offset * stride_qm
            do_ptrs += offset * stride_dom

            offs_m1 += offset

    return dk, dv


@triton.jit
def bwd_dkdv_block_mn(
    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
    dk, dv, qT_ptrs, k, v, do_ptrs, DELTA, LSE, Q_LEN, KV_LEN,
    off_z, off_hq, offs_n1, offs_m1,
    stride_qm, stride_qd, stride_dom, stride_dod,
    q_indices, sparse_q_num_blocks,
    MATMUL_PRECISION, RCP_LN2,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,
):
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    PRESCALE_QK : tl.constexpr = False
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    FLOAT32_PRECISION : tl.constexpr = 'ieee'
    IS_DIVISIBLE : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.20412414523193154
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 32
    V_HEAD_DIM : tl.constexpr = 32
    BLOCK_M1 : tl.constexpr = 64
    BLOCK_N1 : tl.constexpr = 64
    BLOCK_M2 : tl.constexpr = 64
    BLOCK_N2 : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128


    # Load LSE before computing qk to reduce pipeline stall.
    if IS_DIVISIBLE:
        qT = tl.load(qT_ptrs)
        lse = tl.load(LSE + offs_m1)
    else:
        qT = tl.load(qT_ptrs, mask=offs_m1[None, :] < Q_LEN)
        lse = tl.load(LSE + offs_m1, mask=offs_m1 < Q_LEN)
    lse = tl.where(lse == -float("inf"), 0.0, lse)
    qkT = tl.dot(k, qT, input_precision=FLOAT32_PRECISION)
    if not PRESCALE_QK:
        qkT *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    if CHECK_BLOCK_BOUNDARY:
        m = offs_m1[None, :] % Q_LEN
        n = offs_n1[:, None] % KV_LEN
    else:
        m = offs_m1[None, :]
        n = offs_n1[:, None]
    pre_mod_scores = qkT
    post_mod_scores = (qkT)


    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        post_mod_scores = tl.where(offs_n1[:, None] < KV_LEN, post_mod_scores, float("-inf"))

    if not IS_FULL_BLOCKS:
        tmp7 = tl.load(in_ptr16 + (m)) == tl.load(in_ptr16 + (n))
        tmp8 = (n) <= (m)
        tmp9 = tmp7 & tmp8
        tmp10 = (m) - (n)
        tmp11 = tl.full([1], 128, tl.int32)
        tmp12 = tmp10 < tmp11
        tmp13 = tmp9 & tmp12
        mask_mod_output = tmp13

        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n1[:, None] < KV_LEN, mask_mod_output, float("-inf"))
        # (grads) apply mask for fully masked block
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    pT = tl.math.exp2(post_mod_scores - lse[None, :])
    if IS_DIVISIBLE:
        do = tl.load(do_ptrs)
    else:
        do = tl.load(do_ptrs, mask=offs_m1[:, None] < Q_LEN)
    # Compute dV.
    ppT = pT
    dv += tl.dot(ppT.to(MATMUL_PRECISION), do, input_precision=FLOAT32_PRECISION)
    if IS_DIVISIBLE:
        Di = tl.load(DELTA + offs_m1)
    else:
        Di = tl.load(DELTA + offs_m1, mask=offs_m1 < Q_LEN)
    # Compute dP and dS.
    dpT = tl.dot(v, tl.trans(do), input_precision=FLOAT32_PRECISION)
    dsT = pT * (dpT - Di[None, :])
    # ~~~~~~~~~~~~~~~~~~~ Apply joint modification  ~~~~~~~~~~~~~~~~~~~
    grad_scores = (dsT)

    if CHECK_BLOCK_BOUNDARY:
        grad_scores = tl.where(offs_n1[:, None] < KV_LEN, grad_scores, 0.0)

    dsT = grad_scores
    if not IS_FULL_BLOCKS:
        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n1[:, None] < KV_LEN, mask_mod_output, float("-inf"))
        # (grads) apply mask for partially unmasked block
        dsT = tl.where(mask_mod_output, dsT, 0.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dk += tl.dot(dsT.to(MATMUL_PRECISION), tl.trans(qT), input_precision=FLOAT32_PRECISION)

    return dk, dv

@triton.jit
def get_offset_for_next_block(loop_iter, col_indices, total_blocks, SPARSE_BLOCK, SPARSE_BLOCK_MULTIPLE, BLOCK):
    cur_block_idx = loop_iter // SPARSE_BLOCK_MULTIPLE
    cur_block = tl.load(col_indices + cur_block_idx, eviction_policy="evict_last")
    next_block = tl.load(col_indices + cur_block_idx + 1, eviction_policy="evict_last", mask=cur_block_idx + 1 < total_blocks)
    needs_jump = (loop_iter + 1) % SPARSE_BLOCK_MULTIPLE == 0
    jump_to_block = (next_block - cur_block ) * SPARSE_BLOCK - (SPARSE_BLOCK_MULTIPLE - 1) * BLOCK

    offset = jump_to_block * needs_jump + (1 - needs_jump) * BLOCK
    return offset
''', device_str='cuda')
meta0 = {'ROWS_GUARANTEED_SAFE': False, 'PRESCALE_QK': False, 'OUTPUT_LOGSUMEXP': True, 'FLOAT32_PRECISION': "'ieee'", 'IS_DIVISIBLE': True, 'SM_SCALE': 0.20412414523193154, 'GQA_SHARED_HEADS': 1, 'HAS_FULL_BLOCKS': True, 'QK_HEAD_DIM': 32, 'V_HEAD_DIM': 32, 'BLOCK_M1': 64, 'BLOCK_N1': 64, 'BLOCK_M2': 64, 'BLOCK_N2': 64, 'SPARSE_Q_BLOCK_SIZE': 128, 'SPARSE_KV_BLOCK_SIZE': 128}


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, getitem, getitem_1, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (1, 8, 16384, 32), (4194304, 524288, 32, 1))
    assert_size_stride(primals_2, (1, 8, 16384, 32), (4194304, 524288, 32, 1))
    assert_size_stride(primals_3, (1, 8, 16384, 32), (4194304, 524288, 32, 1))
    assert_size_stride(primals_4, (16384, ), (1, ))
    assert_size_stride(primals_5, (1, 1, 128), (128, 128, 1))
    assert_size_stride(primals_6, (1, 1, 128, 128), (16384, 16384, 128, 1))
    assert_size_stride(primals_7, (1, 1, 128), (128, 128, 1))
    assert_size_stride(primals_8, (1, 1, 128, 128), (16384, 16384, 128, 1))
    assert_size_stride(primals_9, (1, 1, 128), (128, 128, 1))
    assert_size_stride(primals_10, (1, 1, 128, 128), (16384, 16384, 128, 1))
    assert_size_stride(primals_11, (1, 1, 128), (128, 128, 1))
    assert_size_stride(primals_12, (1, 1, 128, 128), (16384, 16384, 128, 1))
    assert_size_stride(getitem, (1, 8, 16384, 32), (4194304, 524288, 32, 1))
    assert_size_stride(getitem_1, (1, 8, 16384), (131072, 16384, 1))
    assert_size_stride(tangents_1, (1, 8, 16384, 32), (4194304, 524288, 32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, 8, 16384), (131072, 16384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_per_fused_zeros_0.run(getitem, tangents_1, buf1, 131072, 32, grid=grid(131072), stream=stream0)
        del getitem
        buf3 = empty_strided_cuda((1, 8, 16384, 32), (4194304, 524288, 32, 1), torch.bfloat16)
        buf4 = empty_strided_cuda((1, 8, 16384, 32), (4194304, 524288, 32, 1), torch.bfloat16)
        buf5 = empty_strided_cuda((1, 8, 16384, 32), (4194304, 524288, 32, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.zeros]
        triton_tem_fused_zeros_1.run(primals_1, primals_2, primals_3, getitem_1, buf1, tangents_1, buf3, buf4, primals_5, primals_6, primals_9, primals_10, primals_7, primals_8, primals_11, primals_12, primals_4, buf5, grid=torch._inductor.kernel.flex_attention.flex_attention_backward_grid(1, 8, 16384, 32, 8, 16384, meta0), stream=stream0)
        del buf1
        del getitem_1
        del primals_1
        del primals_10
        del primals_11
        del primals_12
        del primals_2
        del primals_3
        del primals_4
        del primals_5
        del primals_6
        del primals_7
        del primals_8
        del primals_9
        del tangents_1
    return (buf3, buf5, buf4, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 8, 16384, 32), (4194304, 524288, 32, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((1, 8, 16384, 32), (4194304, 524288, 32, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((1, 8, 16384, 32), (4194304, 524288, 32, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((16384, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((1, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.int32)
    primals_6 = rand_strided((1, 1, 128, 128), (16384, 16384, 128, 1), device='cuda:0', dtype=torch.int32)
    primals_7 = rand_strided((1, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.int32)
    primals_8 = rand_strided((1, 1, 128, 128), (16384, 16384, 128, 1), device='cuda:0', dtype=torch.int32)
    primals_9 = rand_strided((1, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.int32)
    primals_10 = rand_strided((1, 1, 128, 128), (16384, 16384, 128, 1), device='cuda:0', dtype=torch.int32)
    primals_11 = rand_strided((1, 1, 128), (128, 128, 1), device='cuda:0', dtype=torch.int32)
    primals_12 = rand_strided((1, 1, 128, 128), (16384, 16384, 128, 1), device='cuda:0', dtype=torch.int32)
    getitem = rand_strided((1, 8, 16384, 32), (4194304, 524288, 32, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_1 = rand_strided((1, 8, 16384), (131072, 16384, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((1, 8, 16384, 32), (4194304, 524288, 32, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, getitem, getitem_1, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
