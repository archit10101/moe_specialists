# AOT ID: ['4_backward']
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/kb/ckbkhwe6enpnecsm6kof2rz4p7yxtqbr76rgobxy7l7ihi27c3c3.py
# Topologically Sorted Source Nodes: [mul_29, mul_27, mul_25, mul_23, mul_21, mul_19, mul_17, mul_15, mul_13, mul_11, mul_9, mul_7, mul_5, mul_3], Original ATen: [aten.mul, aten.sum, aten.select_backward, aten.add]
# Source node to ATen node mapping:
#   mul_11 => mul_11
#   mul_13 => mul_13
#   mul_15 => mul_15
#   mul_17 => mul_17
#   mul_19 => mul_19
#   mul_21 => mul_21
#   mul_23 => mul_23
#   mul_25 => mul_25
#   mul_27 => mul_27
#   mul_29 => mul_29
#   mul_3 => mul_3
#   mul_5 => mul_5
#   mul_7 => mul_7
#   mul_9 => mul_9
# Graph fragment:
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_108, %mul_31), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_35, [1], True), kwargs = {})
#   %full_default_8 : [num_users=15] = call_function[target=torch.ops.aten.full.default](args = ([32, 64, 64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze, 1, -1), kwargs = {})
#   %add_18 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_107, %select_scatter_default_1), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_31, %view_9), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, %mul_29), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_36, [1], True), kwargs = {})
#   %select_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_1, 1, -1), kwargs = {})
#   %add_20 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_106, %select_scatter_default_3), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_29, %view_9), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_20, %mul_27), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_37, [1], True), kwargs = {})
#   %select_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_2, 1, -1), kwargs = {})
#   %add_22 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_105, %select_scatter_default_5), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_27, %view_9), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %mul_25), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_38, [1], True), kwargs = {})
#   %select_scatter_default_7 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_3, 1, -1), kwargs = {})
#   %add_24 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_104, %select_scatter_default_7), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_25, %view_9), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %mul_23), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_39, [1], True), kwargs = {})
#   %select_scatter_default_9 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_4, 1, -1), kwargs = {})
#   %add_26 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_103, %select_scatter_default_9), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_23, %view_9), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, %mul_21), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_40, [1], True), kwargs = {})
#   %select_scatter_default_11 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_5, 1, -1), kwargs = {})
#   %add_28 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_102, %select_scatter_default_11), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_21, %view_9), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_28, %mul_19), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_41, [1], True), kwargs = {})
#   %select_scatter_default_13 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_6, 1, -1), kwargs = {})
#   %add_30 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_101, %select_scatter_default_13), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_19, %view_9), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_30, %mul_17), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_42, [1], True), kwargs = {})
#   %select_scatter_default_15 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_7, 1, -1), kwargs = {})
#   %add_32 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_100, %select_scatter_default_15), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_17, %view_9), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_32, %mul_15), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_43, [1], True), kwargs = {})
#   %select_scatter_default_17 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_8, 1, -1), kwargs = {})
#   %add_34 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_99, %select_scatter_default_17), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_15, %view_9), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_34, %mul_13), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_44, [1], True), kwargs = {})
#   %select_scatter_default_19 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_9, 1, -1), kwargs = {})
#   %add_36 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_98, %select_scatter_default_19), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_13, %view_9), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_36, %mul_11), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_45, [1], True), kwargs = {})
#   %select_scatter_default_21 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_10, 1, -1), kwargs = {})
#   %add_38 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_97, %select_scatter_default_21), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_11, %view_9), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %mul_9), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_46, [1], True), kwargs = {})
#   %select_scatter_default_23 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_11, 1, -1), kwargs = {})
#   %add_40 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_96, %select_scatter_default_23), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_9, %view_9), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_40, %mul_7), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_47, [1], True), kwargs = {})
#   %select_scatter_default_25 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_12, 1, -1), kwargs = {})
#   %add_42 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_95, %select_scatter_default_25), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_7, %view_9), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_42, %mul_5), kwargs = {})
#   %sum_14 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_48, [1], True), kwargs = {})
#   %select_scatter_default_27 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_13, 1, -1), kwargs = {})
#   %add_44 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_94, %select_scatter_default_27), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_5, %view_9), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_44, %mul_3), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_49, [1], True), kwargs = {})
triton_per_fused_add_mul_select_backward_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*i1', 7: '*i1', 8: '*i1', 9: '*i1', 10: '*i1', 11: '*i1', 12: '*i1', 13: '*i1', 14: '*i1', 15: '*i1', 16: '*i1', 17: '*i1', 18: '*i1', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: 'i32', 35: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_select_backward_sum_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 61, 'num_reduction': 15, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (61440 + x0 + (64*r2) + (65536*x1)), None)
    tmp1 = tl.load(in_ptr1 + (960 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (960 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (r2 + (64*x1)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (57344 + x0 + (64*r2) + (65536*x1)), None)
    tmp12 = tl.load(in_ptr1 + (896 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp15 = tl.load(in_ptr2 + (896 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp25 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (53248 + x0 + (64*r2) + (65536*x1)), None)
    tmp32 = tl.load(in_ptr1 + (832 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp35 = tl.load(in_ptr2 + (832 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr6 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp46 = tl.load(in_ptr0 + (49152 + x0 + (64*r2) + (65536*x1)), None)
    tmp47 = tl.load(in_ptr1 + (768 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp50 = tl.load(in_ptr2 + (768 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr7 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp61 = tl.load(in_ptr0 + (45056 + x0 + (64*r2) + (65536*x1)), None)
    tmp62 = tl.load(in_ptr1 + (704 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp65 = tl.load(in_ptr2 + (704 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr8 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp76 = tl.load(in_ptr0 + (40960 + x0 + (64*r2) + (65536*x1)), None)
    tmp77 = tl.load(in_ptr1 + (640 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp80 = tl.load(in_ptr2 + (640 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp84 = tl.load(in_ptr9 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp91 = tl.load(in_ptr0 + (36864 + x0 + (64*r2) + (65536*x1)), None)
    tmp92 = tl.load(in_ptr1 + (576 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp95 = tl.load(in_ptr2 + (576 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp99 = tl.load(in_ptr10 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp106 = tl.load(in_ptr0 + (32768 + x0 + (64*r2) + (65536*x1)), None)
    tmp107 = tl.load(in_ptr1 + (512 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp110 = tl.load(in_ptr2 + (512 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp114 = tl.load(in_ptr11 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp121 = tl.load(in_ptr0 + (28672 + x0 + (64*r2) + (65536*x1)), None)
    tmp122 = tl.load(in_ptr1 + (448 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp125 = tl.load(in_ptr2 + (448 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp129 = tl.load(in_ptr12 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp136 = tl.load(in_ptr0 + (24576 + x0 + (64*r2) + (65536*x1)), None)
    tmp137 = tl.load(in_ptr1 + (384 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp140 = tl.load(in_ptr2 + (384 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp144 = tl.load(in_ptr13 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp151 = tl.load(in_ptr0 + (20480 + x0 + (64*r2) + (65536*x1)), None)
    tmp152 = tl.load(in_ptr1 + (320 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp155 = tl.load(in_ptr2 + (320 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp159 = tl.load(in_ptr14 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp166 = tl.load(in_ptr0 + (16384 + x0 + (64*r2) + (65536*x1)), None)
    tmp167 = tl.load(in_ptr1 + (256 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp170 = tl.load(in_ptr2 + (256 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp174 = tl.load(in_ptr15 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp181 = tl.load(in_ptr0 + (12288 + x0 + (64*r2) + (65536*x1)), None)
    tmp182 = tl.load(in_ptr1 + (192 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp185 = tl.load(in_ptr2 + (192 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp189 = tl.load(in_ptr16 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp196 = tl.load(in_ptr0 + (8192 + x0 + (64*r2) + (65536*x1)), None)
    tmp197 = tl.load(in_ptr1 + (128 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp200 = tl.load(in_ptr2 + (128 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp204 = tl.load(in_ptr17 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp211 = tl.load(in_ptr0 + (4096 + x0 + (64*r2) + (65536*x1)), None)
    tmp212 = tl.load(in_ptr1 + (64 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp215 = tl.load(in_ptr2 + (64 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp219 = tl.load(in_ptr18 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tmp5 = tmp3 / tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.sum(tmp8, 1)[:, None]
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 * tmp13
    tmp16 = tmp14 / tmp15
    tmp17 = r2
    tmp18 = tl.full([1, 1], 63, tl.int32)
    tmp19 = tmp17 == tmp18
    tmp20 = 0.0
    tmp21 = tl.where(tmp19, tmp10, tmp20)
    tmp22 = tmp16 + tmp21
    tmp24 = tmp23.to(tl.float32)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp22 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.sum(tmp28, 1)[:, None]
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 * tmp33
    tmp36 = tmp34 / tmp35
    tmp37 = tl.where(tmp19, tmp30, tmp20)
    tmp38 = tmp36 + tmp37
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp40 * tmp25
    tmp42 = tmp38 * tmp41
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, RBLOCK])
    tmp45 = tl.sum(tmp43, 1)[:, None]
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp46 * tmp48
    tmp51 = tmp49 / tmp50
    tmp52 = tl.where(tmp19, tmp45, tmp20)
    tmp53 = tmp51 + tmp52
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tmp55 * tmp25
    tmp57 = tmp53 * tmp56
    tmp58 = tl.broadcast_to(tmp57, [XBLOCK, RBLOCK])
    tmp60 = tl.sum(tmp58, 1)[:, None]
    tmp63 = tmp62.to(tl.float32)
    tmp64 = tmp61 * tmp63
    tmp66 = tmp64 / tmp65
    tmp67 = tl.where(tmp19, tmp60, tmp20)
    tmp68 = tmp66 + tmp67
    tmp70 = tmp69.to(tl.float32)
    tmp71 = tmp70 * tmp25
    tmp72 = tmp68 * tmp71
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp75 = tl.sum(tmp73, 1)[:, None]
    tmp78 = tmp77.to(tl.float32)
    tmp79 = tmp76 * tmp78
    tmp81 = tmp79 / tmp80
    tmp82 = tl.where(tmp19, tmp75, tmp20)
    tmp83 = tmp81 + tmp82
    tmp85 = tmp84.to(tl.float32)
    tmp86 = tmp85 * tmp25
    tmp87 = tmp83 * tmp86
    tmp88 = tl.broadcast_to(tmp87, [XBLOCK, RBLOCK])
    tmp90 = tl.sum(tmp88, 1)[:, None]
    tmp93 = tmp92.to(tl.float32)
    tmp94 = tmp91 * tmp93
    tmp96 = tmp94 / tmp95
    tmp97 = tl.where(tmp19, tmp90, tmp20)
    tmp98 = tmp96 + tmp97
    tmp100 = tmp99.to(tl.float32)
    tmp101 = tmp100 * tmp25
    tmp102 = tmp98 * tmp101
    tmp103 = tl.broadcast_to(tmp102, [XBLOCK, RBLOCK])
    tmp105 = tl.sum(tmp103, 1)[:, None]
    tmp108 = tmp107.to(tl.float32)
    tmp109 = tmp106 * tmp108
    tmp111 = tmp109 / tmp110
    tmp112 = tl.where(tmp19, tmp105, tmp20)
    tmp113 = tmp111 + tmp112
    tmp115 = tmp114.to(tl.float32)
    tmp116 = tmp115 * tmp25
    tmp117 = tmp113 * tmp116
    tmp118 = tl.broadcast_to(tmp117, [XBLOCK, RBLOCK])
    tmp120 = tl.sum(tmp118, 1)[:, None]
    tmp123 = tmp122.to(tl.float32)
    tmp124 = tmp121 * tmp123
    tmp126 = tmp124 / tmp125
    tmp127 = tl.where(tmp19, tmp120, tmp20)
    tmp128 = tmp126 + tmp127
    tmp130 = tmp129.to(tl.float32)
    tmp131 = tmp130 * tmp25
    tmp132 = tmp128 * tmp131
    tmp133 = tl.broadcast_to(tmp132, [XBLOCK, RBLOCK])
    tmp135 = tl.sum(tmp133, 1)[:, None]
    tmp138 = tmp137.to(tl.float32)
    tmp139 = tmp136 * tmp138
    tmp141 = tmp139 / tmp140
    tmp142 = tl.where(tmp19, tmp135, tmp20)
    tmp143 = tmp141 + tmp142
    tmp145 = tmp144.to(tl.float32)
    tmp146 = tmp145 * tmp25
    tmp147 = tmp143 * tmp146
    tmp148 = tl.broadcast_to(tmp147, [XBLOCK, RBLOCK])
    tmp150 = tl.sum(tmp148, 1)[:, None]
    tmp153 = tmp152.to(tl.float32)
    tmp154 = tmp151 * tmp153
    tmp156 = tmp154 / tmp155
    tmp157 = tl.where(tmp19, tmp150, tmp20)
    tmp158 = tmp156 + tmp157
    tmp160 = tmp159.to(tl.float32)
    tmp161 = tmp160 * tmp25
    tmp162 = tmp158 * tmp161
    tmp163 = tl.broadcast_to(tmp162, [XBLOCK, RBLOCK])
    tmp165 = tl.sum(tmp163, 1)[:, None]
    tmp168 = tmp167.to(tl.float32)
    tmp169 = tmp166 * tmp168
    tmp171 = tmp169 / tmp170
    tmp172 = tl.where(tmp19, tmp165, tmp20)
    tmp173 = tmp171 + tmp172
    tmp175 = tmp174.to(tl.float32)
    tmp176 = tmp175 * tmp25
    tmp177 = tmp173 * tmp176
    tmp178 = tl.broadcast_to(tmp177, [XBLOCK, RBLOCK])
    tmp180 = tl.sum(tmp178, 1)[:, None]
    tmp183 = tmp182.to(tl.float32)
    tmp184 = tmp181 * tmp183
    tmp186 = tmp184 / tmp185
    tmp187 = tl.where(tmp19, tmp180, tmp20)
    tmp188 = tmp186 + tmp187
    tmp190 = tmp189.to(tl.float32)
    tmp191 = tmp190 * tmp25
    tmp192 = tmp188 * tmp191
    tmp193 = tl.broadcast_to(tmp192, [XBLOCK, RBLOCK])
    tmp195 = tl.sum(tmp193, 1)[:, None]
    tmp198 = tmp197.to(tl.float32)
    tmp199 = tmp196 * tmp198
    tmp201 = tmp199 / tmp200
    tmp202 = tl.where(tmp19, tmp195, tmp20)
    tmp203 = tmp201 + tmp202
    tmp205 = tmp204.to(tl.float32)
    tmp206 = tmp205 * tmp25
    tmp207 = tmp203 * tmp206
    tmp208 = tl.broadcast_to(tmp207, [XBLOCK, RBLOCK])
    tmp210 = tl.sum(tmp208, 1)[:, None]
    tmp213 = tmp212.to(tl.float32)
    tmp214 = tmp211 * tmp213
    tmp216 = tmp214 / tmp215
    tmp217 = tl.where(tmp19, tmp210, tmp20)
    tmp218 = tmp216 + tmp217
    tmp220 = tmp219.to(tl.float32)
    tmp221 = tmp220 * tmp25
    tmp222 = tmp218 * tmp221
    tmp223 = tl.broadcast_to(tmp222, [XBLOCK, RBLOCK])
    tmp225 = tl.sum(tmp223, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp30, None)
    tl.store(out_ptr2 + (x3), tmp45, None)
    tl.store(out_ptr3 + (x3), tmp60, None)
    tl.store(out_ptr4 + (x3), tmp75, None)
    tl.store(out_ptr5 + (x3), tmp90, None)
    tl.store(out_ptr6 + (x3), tmp105, None)
    tl.store(out_ptr7 + (x3), tmp120, None)
    tl.store(out_ptr8 + (x3), tmp135, None)
    tl.store(out_ptr9 + (x3), tmp150, None)
    tl.store(out_ptr10 + (x3), tmp165, None)
    tl.store(out_ptr11 + (x3), tmp180, None)
    tl.store(out_ptr12 + (x3), tmp195, None)
    tl.store(out_ptr13 + (x3), tmp210, None)
    tl.store(out_ptr14 + (x3), tmp225, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/sl/cslvob6fp256utd5d7ypymwkgwpj6qn7no3r3rpw5jm26m7opcur.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_6 : [num_users=16] = call_function[target=torch.ops.aten.full.default](args = ([32, 16, 64, 64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %select_108, 1, 15), kwargs = {})
#   %full_default_8 : [num_users=15] = call_function[target=torch.ops.aten.full.default](args = ([32, 64, 64], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze, 1, -1), kwargs = {})
#   %add_18 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_107, %select_scatter_default_1), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %add_18, 1, 14), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_2), kwargs = {})
#   %select_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_1, 1, -1), kwargs = {})
#   %add_20 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_106, %select_scatter_default_3), kwargs = {})
#   %select_scatter_default_4 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %add_20, 1, 13), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_19, %select_scatter_default_4), kwargs = {})
#   %select_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_2, 1, -1), kwargs = {})
#   %add_22 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_105, %select_scatter_default_5), kwargs = {})
#   %select_scatter_default_6 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %add_22, 1, 12), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %select_scatter_default_6), kwargs = {})
#   %select_scatter_default_7 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_3, 1, -1), kwargs = {})
#   %add_24 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_104, %select_scatter_default_7), kwargs = {})
#   %select_scatter_default_8 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %add_24, 1, 11), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %select_scatter_default_8), kwargs = {})
#   %select_scatter_default_9 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_4, 1, -1), kwargs = {})
#   %add_26 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_103, %select_scatter_default_9), kwargs = {})
#   %select_scatter_default_10 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %add_26, 1, 10), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_25, %select_scatter_default_10), kwargs = {})
#   %select_scatter_default_11 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_5, 1, -1), kwargs = {})
#   %add_28 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_102, %select_scatter_default_11), kwargs = {})
#   %select_scatter_default_12 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %add_28, 1, 9), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %select_scatter_default_12), kwargs = {})
#   %select_scatter_default_13 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_6, 1, -1), kwargs = {})
#   %add_30 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_101, %select_scatter_default_13), kwargs = {})
#   %select_scatter_default_14 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %add_30, 1, 8), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_29, %select_scatter_default_14), kwargs = {})
#   %select_scatter_default_15 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_7, 1, -1), kwargs = {})
#   %add_32 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_100, %select_scatter_default_15), kwargs = {})
#   %select_scatter_default_16 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %add_32, 1, 7), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_31, %select_scatter_default_16), kwargs = {})
#   %select_scatter_default_17 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_8, 1, -1), kwargs = {})
#   %add_34 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_99, %select_scatter_default_17), kwargs = {})
#   %select_scatter_default_18 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %add_34, 1, 6), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %select_scatter_default_18), kwargs = {})
#   %select_scatter_default_19 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_9, 1, -1), kwargs = {})
#   %add_36 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_98, %select_scatter_default_19), kwargs = {})
#   %select_scatter_default_20 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %add_36, 1, 5), kwargs = {})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_35, %select_scatter_default_20), kwargs = {})
#   %select_scatter_default_21 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_10, 1, -1), kwargs = {})
#   %add_38 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_97, %select_scatter_default_21), kwargs = {})
#   %select_scatter_default_22 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %add_38, 1, 4), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_37, %select_scatter_default_22), kwargs = {})
#   %select_scatter_default_23 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_11, 1, -1), kwargs = {})
#   %add_40 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_96, %select_scatter_default_23), kwargs = {})
#   %select_scatter_default_24 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %add_40, 1, 3), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_39, %select_scatter_default_24), kwargs = {})
#   %select_scatter_default_25 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_12, 1, -1), kwargs = {})
#   %add_42 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_95, %select_scatter_default_25), kwargs = {})
#   %select_scatter_default_26 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %add_42, 1, 2), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_41, %select_scatter_default_26), kwargs = {})
#   %select_scatter_default_27 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_13, 1, -1), kwargs = {})
#   %add_44 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_94, %select_scatter_default_27), kwargs = {})
#   %select_scatter_default_28 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %add_44, 1, 1), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_43, %select_scatter_default_28), kwargs = {})
#   %select_scatter_default_29 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_8, %squeeze_14, 1, -1), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_93, %select_scatter_default_29), kwargs = {})
#   %select_scatter_default_30 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_6, %add_46, 1, 0), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_45, %select_scatter_default_30), kwargs = {})
triton_poi_fused_add_select_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_select_backward_1', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 63, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 4096) % 16
    x3 = (xindex // 65536)
    x4 = xindex % 4096
    x1 = (xindex // 64) % 64
    x0 = xindex % 64
    x6 = xindex
    tmp3 = tl.load(in_ptr0 + (61440 + x4 + (65536*x3)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp7 = tl.load(in_ptr2 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (57344 + x4 + (65536*x3)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp17 = tl.load(in_ptr2 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (x0 + (64*x3)), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (53248 + x4 + (65536*x3)), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr1 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp33 = tl.load(in_ptr2 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr4 + (x0 + (64*x3)), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (49152 + x4 + (65536*x3)), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr1 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp46 = tl.load(in_ptr2 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr5 + (x0 + (64*x3)), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (45056 + x4 + (65536*x3)), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr1 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp59 = tl.load(in_ptr2 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr6 + (x0 + (64*x3)), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr0 + (40960 + x4 + (65536*x3)), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr1 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp72 = tl.load(in_ptr2 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr7 + (x0 + (64*x3)), None, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr0 + (36864 + x4 + (65536*x3)), None, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr1 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp85 = tl.load(in_ptr2 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr8 + (x0 + (64*x3)), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr0 + (32768 + x4 + (65536*x3)), None, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr1 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp98 = tl.load(in_ptr2 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp100 = tl.load(in_ptr9 + (x0 + (64*x3)), None, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr0 + (28672 + x4 + (65536*x3)), None, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr1 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp111 = tl.load(in_ptr2 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp113 = tl.load(in_ptr10 + (x0 + (64*x3)), None, eviction_policy='evict_last')
    tmp120 = tl.load(in_ptr0 + (24576 + x4 + (65536*x3)), None, eviction_policy='evict_last')
    tmp121 = tl.load(in_ptr1 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp124 = tl.load(in_ptr2 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp126 = tl.load(in_ptr11 + (x0 + (64*x3)), None, eviction_policy='evict_last')
    tmp133 = tl.load(in_ptr0 + (20480 + x4 + (65536*x3)), None, eviction_policy='evict_last')
    tmp134 = tl.load(in_ptr1 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp137 = tl.load(in_ptr2 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp139 = tl.load(in_ptr12 + (x0 + (64*x3)), None, eviction_policy='evict_last')
    tmp146 = tl.load(in_ptr0 + (16384 + x4 + (65536*x3)), None, eviction_policy='evict_last')
    tmp147 = tl.load(in_ptr1 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp150 = tl.load(in_ptr2 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp152 = tl.load(in_ptr13 + (x0 + (64*x3)), None, eviction_policy='evict_last')
    tmp159 = tl.load(in_ptr0 + (12288 + x4 + (65536*x3)), None, eviction_policy='evict_last')
    tmp160 = tl.load(in_ptr1 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp163 = tl.load(in_ptr2 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp165 = tl.load(in_ptr14 + (x0 + (64*x3)), None, eviction_policy='evict_last')
    tmp172 = tl.load(in_ptr0 + (8192 + x4 + (65536*x3)), None, eviction_policy='evict_last')
    tmp173 = tl.load(in_ptr1 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp176 = tl.load(in_ptr2 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp178 = tl.load(in_ptr15 + (x0 + (64*x3)), None, eviction_policy='evict_last')
    tmp185 = tl.load(in_ptr0 + (4096 + x4 + (65536*x3)), None, eviction_policy='evict_last')
    tmp186 = tl.load(in_ptr1 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp189 = tl.load(in_ptr2 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp191 = tl.load(in_ptr16 + (x0 + (64*x3)), None, eviction_policy='evict_last')
    tmp198 = tl.load(in_ptr0 + (x4 + (65536*x3)), None, eviction_policy='evict_last')
    tmp199 = tl.load(in_ptr1 + (x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp202 = tl.load(in_ptr2 + (x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp204 = tl.load(in_ptr17 + (x0 + (64*x3)), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 15, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp8 = tmp6 / tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp2, tmp8, tmp9)
    tmp11 = tl.full([1], 14, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 * tmp15
    tmp18 = tmp16 / tmp17
    tmp19 = x1
    tmp20 = tl.full([1], 63, tl.int32)
    tmp21 = tmp19 == tmp20
    tmp23 = tl.where(tmp21, tmp22, tmp9)
    tmp24 = tmp18 + tmp23
    tmp25 = tl.where(tmp12, tmp24, tmp9)
    tmp26 = tmp10 + tmp25
    tmp27 = tl.full([1], 13, tl.int32)
    tmp28 = tmp0 == tmp27
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 * tmp31
    tmp34 = tmp32 / tmp33
    tmp36 = tl.where(tmp21, tmp35, tmp9)
    tmp37 = tmp34 + tmp36
    tmp38 = tl.where(tmp28, tmp37, tmp9)
    tmp39 = tmp26 + tmp38
    tmp40 = tl.full([1], 12, tl.int32)
    tmp41 = tmp0 == tmp40
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp42 * tmp44
    tmp47 = tmp45 / tmp46
    tmp49 = tl.where(tmp21, tmp48, tmp9)
    tmp50 = tmp47 + tmp49
    tmp51 = tl.where(tmp41, tmp50, tmp9)
    tmp52 = tmp39 + tmp51
    tmp53 = tl.full([1], 11, tl.int32)
    tmp54 = tmp0 == tmp53
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tmp55 * tmp57
    tmp60 = tmp58 / tmp59
    tmp62 = tl.where(tmp21, tmp61, tmp9)
    tmp63 = tmp60 + tmp62
    tmp64 = tl.where(tmp54, tmp63, tmp9)
    tmp65 = tmp52 + tmp64
    tmp66 = tl.full([1], 10, tl.int32)
    tmp67 = tmp0 == tmp66
    tmp70 = tmp69.to(tl.float32)
    tmp71 = tmp68 * tmp70
    tmp73 = tmp71 / tmp72
    tmp75 = tl.where(tmp21, tmp74, tmp9)
    tmp76 = tmp73 + tmp75
    tmp77 = tl.where(tmp67, tmp76, tmp9)
    tmp78 = tmp65 + tmp77
    tmp79 = tl.full([1], 9, tl.int32)
    tmp80 = tmp0 == tmp79
    tmp83 = tmp82.to(tl.float32)
    tmp84 = tmp81 * tmp83
    tmp86 = tmp84 / tmp85
    tmp88 = tl.where(tmp21, tmp87, tmp9)
    tmp89 = tmp86 + tmp88
    tmp90 = tl.where(tmp80, tmp89, tmp9)
    tmp91 = tmp78 + tmp90
    tmp92 = tl.full([1], 8, tl.int32)
    tmp93 = tmp0 == tmp92
    tmp96 = tmp95.to(tl.float32)
    tmp97 = tmp94 * tmp96
    tmp99 = tmp97 / tmp98
    tmp101 = tl.where(tmp21, tmp100, tmp9)
    tmp102 = tmp99 + tmp101
    tmp103 = tl.where(tmp93, tmp102, tmp9)
    tmp104 = tmp91 + tmp103
    tmp105 = tl.full([1], 7, tl.int32)
    tmp106 = tmp0 == tmp105
    tmp109 = tmp108.to(tl.float32)
    tmp110 = tmp107 * tmp109
    tmp112 = tmp110 / tmp111
    tmp114 = tl.where(tmp21, tmp113, tmp9)
    tmp115 = tmp112 + tmp114
    tmp116 = tl.where(tmp106, tmp115, tmp9)
    tmp117 = tmp104 + tmp116
    tmp118 = tl.full([1], 6, tl.int32)
    tmp119 = tmp0 == tmp118
    tmp122 = tmp121.to(tl.float32)
    tmp123 = tmp120 * tmp122
    tmp125 = tmp123 / tmp124
    tmp127 = tl.where(tmp21, tmp126, tmp9)
    tmp128 = tmp125 + tmp127
    tmp129 = tl.where(tmp119, tmp128, tmp9)
    tmp130 = tmp117 + tmp129
    tmp131 = tl.full([1], 5, tl.int32)
    tmp132 = tmp0 == tmp131
    tmp135 = tmp134.to(tl.float32)
    tmp136 = tmp133 * tmp135
    tmp138 = tmp136 / tmp137
    tmp140 = tl.where(tmp21, tmp139, tmp9)
    tmp141 = tmp138 + tmp140
    tmp142 = tl.where(tmp132, tmp141, tmp9)
    tmp143 = tmp130 + tmp142
    tmp144 = tl.full([1], 4, tl.int32)
    tmp145 = tmp0 == tmp144
    tmp148 = tmp147.to(tl.float32)
    tmp149 = tmp146 * tmp148
    tmp151 = tmp149 / tmp150
    tmp153 = tl.where(tmp21, tmp152, tmp9)
    tmp154 = tmp151 + tmp153
    tmp155 = tl.where(tmp145, tmp154, tmp9)
    tmp156 = tmp143 + tmp155
    tmp157 = tl.full([1], 3, tl.int32)
    tmp158 = tmp0 == tmp157
    tmp161 = tmp160.to(tl.float32)
    tmp162 = tmp159 * tmp161
    tmp164 = tmp162 / tmp163
    tmp166 = tl.where(tmp21, tmp165, tmp9)
    tmp167 = tmp164 + tmp166
    tmp168 = tl.where(tmp158, tmp167, tmp9)
    tmp169 = tmp156 + tmp168
    tmp170 = tl.full([1], 2, tl.int32)
    tmp171 = tmp0 == tmp170
    tmp174 = tmp173.to(tl.float32)
    tmp175 = tmp172 * tmp174
    tmp177 = tmp175 / tmp176
    tmp179 = tl.where(tmp21, tmp178, tmp9)
    tmp180 = tmp177 + tmp179
    tmp181 = tl.where(tmp171, tmp180, tmp9)
    tmp182 = tmp169 + tmp181
    tmp183 = tl.full([1], 1, tl.int32)
    tmp184 = tmp0 == tmp183
    tmp187 = tmp186.to(tl.float32)
    tmp188 = tmp185 * tmp187
    tmp190 = tmp188 / tmp189
    tmp192 = tl.where(tmp21, tmp191, tmp9)
    tmp193 = tmp190 + tmp192
    tmp194 = tl.where(tmp184, tmp193, tmp9)
    tmp195 = tmp182 + tmp194
    tmp196 = tl.full([1], 0, tl.int32)
    tmp197 = tmp0 == tmp196
    tmp200 = tmp199.to(tl.float32)
    tmp201 = tmp198 * tmp200
    tmp203 = tmp201 / tmp202
    tmp205 = tl.where(tmp21, tmp204, tmp9)
    tmp206 = tmp203 + tmp205
    tmp207 = tl.where(tmp197, tmp206, tmp9)
    tmp208 = tmp195 + tmp207
    tl.store(in_out_ptr0 + (x6), tmp208, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_3, view_9, bitwise_and_2, bitwise_and_3, bitwise_and_4, bitwise_and_5, bitwise_and_6, bitwise_and_7, bitwise_and_8, bitwise_and_9, bitwise_and_10, bitwise_and_11, bitwise_and_12, bitwise_and_13, bitwise_and_14, bitwise_and_15, mul_31, unsqueeze_35, unsqueeze_36, permute_1, permute_4, tangents_1 = args
    args.clear()
    assert_size_stride(primals_3, (32768, 384), (384, 1))
    assert_size_stride(view_9, (1, 64, 1), (64, 1, 1))
    assert_size_stride(bitwise_and_2, (32, 64), (64, 1))
    assert_size_stride(bitwise_and_3, (32, 64), (64, 1))
    assert_size_stride(bitwise_and_4, (32, 64), (64, 1))
    assert_size_stride(bitwise_and_5, (32, 64), (64, 1))
    assert_size_stride(bitwise_and_6, (32, 64), (64, 1))
    assert_size_stride(bitwise_and_7, (32, 64), (64, 1))
    assert_size_stride(bitwise_and_8, (32, 64), (64, 1))
    assert_size_stride(bitwise_and_9, (32, 64), (64, 1))
    assert_size_stride(bitwise_and_10, (32, 64), (64, 1))
    assert_size_stride(bitwise_and_11, (32, 64), (64, 1))
    assert_size_stride(bitwise_and_12, (32, 64), (64, 1))
    assert_size_stride(bitwise_and_13, (32, 64), (64, 1))
    assert_size_stride(bitwise_and_14, (32, 64), (64, 1))
    assert_size_stride(bitwise_and_15, (32, 64), (64, 1))
    assert_size_stride(mul_31, (32, 64, 1), (64, 1, 1))
    assert_size_stride(unsqueeze_35, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(unsqueeze_36, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_1, (512, 64, 64), (4096, 1, 64))
    assert_size_stride(permute_4, (64, 384), (384, 1))
    assert_size_stride(tangents_1, (32768, 64), (64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 1, 64), (64, 64, 1), torch.float32)
        buf1 = empty_strided_cuda((32, 1, 64), (64, 64, 1), torch.float32)
        buf2 = empty_strided_cuda((32, 1, 64), (64, 64, 1), torch.float32)
        buf4 = empty_strided_cuda((32, 1, 64), (64, 64, 1), torch.float32)
        buf5 = empty_strided_cuda((32, 1, 64), (64, 64, 1), torch.float32)
        buf7 = empty_strided_cuda((32, 1, 64), (64, 64, 1), torch.float32)
        buf8 = empty_strided_cuda((32, 1, 64), (64, 64, 1), torch.float32)
        buf10 = empty_strided_cuda((32, 1, 64), (64, 64, 1), torch.float32)
        buf11 = empty_strided_cuda((32, 1, 64), (64, 64, 1), torch.float32)
        buf13 = empty_strided_cuda((32, 1, 64), (64, 64, 1), torch.float32)
        buf14 = empty_strided_cuda((32, 1, 64), (64, 64, 1), torch.float32)
        buf16 = empty_strided_cuda((32, 1, 64), (64, 64, 1), torch.float32)
        buf17 = empty_strided_cuda((32, 1, 64), (64, 64, 1), torch.float32)
        buf19 = empty_strided_cuda((32, 1, 64), (64, 64, 1), torch.float32)
        buf20 = empty_strided_cuda((32, 1, 64), (64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_29, mul_27, mul_25, mul_23, mul_21, mul_19, mul_17, mul_15, mul_13, mul_11, mul_9, mul_7, mul_5, mul_3], Original ATen: [aten.mul, aten.sum, aten.select_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mul_select_backward_sum_0.run(tangents_1, unsqueeze_36, unsqueeze_35, mul_31, bitwise_and_15, view_9, bitwise_and_14, bitwise_and_13, bitwise_and_12, bitwise_and_11, bitwise_and_10, bitwise_and_9, bitwise_and_8, bitwise_and_7, bitwise_and_6, bitwise_and_5, bitwise_and_4, bitwise_and_3, bitwise_and_2, buf0, buf1, buf2, buf4, buf5, buf7, buf8, buf10, buf11, buf13, buf14, buf16, buf17, buf19, buf20, 2048, 64, grid=grid(2048), stream=stream0)
        del bitwise_and_10
        del bitwise_and_11
        del bitwise_and_12
        del bitwise_and_13
        del bitwise_and_14
        del bitwise_and_15
        del bitwise_and_2
        del bitwise_and_3
        del bitwise_and_4
        del bitwise_and_5
        del bitwise_and_6
        del bitwise_and_7
        del bitwise_and_8
        del bitwise_and_9
        del mul_31
        del view_9
        buf3 = empty_strided_cuda((32, 16, 64, 64), (65536, 4096, 64, 1), torch.float32)
        buf6 = buf3; del buf3  # reuse
        buf9 = buf6; del buf6  # reuse
        buf12 = buf9; del buf9  # reuse
        buf15 = buf12; del buf12  # reuse
        buf18 = buf15; del buf15  # reuse
        buf21 = buf18; del buf18  # reuse
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward, aten.add]
        triton_poi_fused_add_select_backward_1.run(buf22, tangents_1, unsqueeze_36, unsqueeze_35, buf0, buf1, buf2, buf4, buf5, buf7, buf8, buf10, buf11, buf13, buf14, buf16, buf17, buf19, buf20, 2097152, grid=grid(2097152), stream=stream0)
        del buf0
        del buf1
        del buf10
        del buf11
        del buf13
        del buf14
        del buf16
        del buf17
        del buf19
        del buf2
        del buf20
        del buf4
        del buf5
        del buf7
        del buf8
        del tangents_1
        del unsqueeze_35
        del unsqueeze_36
        buf23 = empty_strided_cuda((512, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1, reinterpret_tensor(buf22, (512, 64, 64), (4096, 64, 1), 0), out=buf23)
        del buf22
        del permute_1
        buf24 = empty_strided_cuda((64, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (64, 32768), (1, 64), 0), primals_3, out=buf24)
        del primals_3
        buf25 = empty_strided_cuda((32768, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (32768, 64), (64, 1), 0), permute_4, out=buf25)
        del buf23
        del permute_4
    return (None, buf24, buf25, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((32768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    view_9 = rand_strided((1, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    bitwise_and_2 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.bool)
    bitwise_and_3 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.bool)
    bitwise_and_4 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.bool)
    bitwise_and_5 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.bool)
    bitwise_and_6 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.bool)
    bitwise_and_7 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.bool)
    bitwise_and_8 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.bool)
    bitwise_and_9 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.bool)
    bitwise_and_10 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.bool)
    bitwise_and_11 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.bool)
    bitwise_and_12 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.bool)
    bitwise_and_13 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.bool)
    bitwise_and_14 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.bool)
    bitwise_and_15 = rand_strided((32, 64), (64, 1), device='cuda:0', dtype=torch.bool)
    mul_31 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_35 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_36 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.bool)
    permute_1 = rand_strided((512, 64, 64), (4096, 1, 64), device='cuda:0', dtype=torch.float32)
    permute_4 = rand_strided((64, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((32768, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_3, view_9, bitwise_and_2, bitwise_and_3, bitwise_and_4, bitwise_and_5, bitwise_and_6, bitwise_and_7, bitwise_and_8, bitwise_and_9, bitwise_and_10, bitwise_and_11, bitwise_and_12, bitwise_and_13, bitwise_and_14, bitwise_and_15, mul_31, unsqueeze_35, unsqueeze_36, permute_1, permute_4, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
