# AOT ID: ['5_backward']
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/o7/co7t7ab4arrgouggfiz7buq7bznkowol7jbyu7uirtpsfkdfymon.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.select_backward, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_105 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_294, %mul_101), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_105, [1], True), kwargs = {dtype: torch.float32})
#   %full_default_20 : [num_users=45] = call_function[target=torch.ops.aten.full.default](args = ([32, 64, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze, 1, -1), kwargs = {})
#   %add_55 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_293, %select_scatter_default_1), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, %mul_99), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_106, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_1, 1, -1), kwargs = {})
#   %add_57 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_292, %select_scatter_default_3), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %mul_97), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_107, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_2, 1, -1), kwargs = {})
#   %add_59 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_291, %select_scatter_default_5), kwargs = {})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_59, %mul_95), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_108, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_7 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_3, 1, -1), kwargs = {})
#   %add_61 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_290, %select_scatter_default_7), kwargs = {})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_61, %mul_93), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_109, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_9 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_4, 1, -1), kwargs = {})
#   %add_63 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_289, %select_scatter_default_9), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_63, %mul_91), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_110, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_11 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_5, 1, -1), kwargs = {})
#   %add_65 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_288, %select_scatter_default_11), kwargs = {})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_65, %mul_89), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_111, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_13 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_6, 1, -1), kwargs = {})
#   %add_67 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_287, %select_scatter_default_13), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_67, %mul_87), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_112, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_15 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_7, 1, -1), kwargs = {})
#   %add_69 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_286, %select_scatter_default_15), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_69, %mul_85), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_113, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_17 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_8, 1, -1), kwargs = {})
#   %add_71 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_285, %select_scatter_default_17), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_71, %mul_83), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_114, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_19 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_9, 1, -1), kwargs = {})
#   %add_73 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_284, %select_scatter_default_19), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_73, %mul_81), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_115, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_21 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_10, 1, -1), kwargs = {})
#   %add_75 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_283, %select_scatter_default_21), kwargs = {})
#   %mul_116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_75, %mul_79), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_116, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_23 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_11, 1, -1), kwargs = {})
#   %add_77 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_282, %select_scatter_default_23), kwargs = {})
#   %mul_117 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_77, %mul_77), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_117, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_25 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_12, 1, -1), kwargs = {})
#   %add_79 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_281, %select_scatter_default_25), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_79, %mul_75), kwargs = {})
#   %sum_14 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_118, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_27 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_13, 1, -1), kwargs = {})
#   %add_81 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_280, %select_scatter_default_27), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %mul_73), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_119, [1], True), kwargs = {dtype: torch.float32})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_310, %mul_67), kwargs = {})
#   %sum_16 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_121, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_32 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_15, 1, -1), kwargs = {})
#   %add_85 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_309, %select_scatter_default_32), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_85, %mul_65), kwargs = {})
#   %sum_17 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_122, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_34 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_16, 1, -1), kwargs = {})
#   %add_87 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_308, %select_scatter_default_34), kwargs = {})
#   %mul_123 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_87, %mul_63), kwargs = {})
#   %sum_18 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_123, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_36 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_17, 1, -1), kwargs = {})
#   %add_89 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_307, %select_scatter_default_36), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %mul_61), kwargs = {})
#   %sum_19 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_124, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_38 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_18, 1, -1), kwargs = {})
#   %add_91 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_306, %select_scatter_default_38), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_91, %mul_59), kwargs = {})
#   %sum_20 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_125, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_40 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_19, 1, -1), kwargs = {})
#   %add_93 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_305, %select_scatter_default_40), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_93, %mul_57), kwargs = {})
#   %sum_21 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_126, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_42 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_20, 1, -1), kwargs = {})
#   %add_95 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_304, %select_scatter_default_42), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_95, %mul_55), kwargs = {})
#   %sum_22 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_127, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_44 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_21, 1, -1), kwargs = {})
#   %add_97 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_303, %select_scatter_default_44), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_97, %mul_53), kwargs = {})
#   %sum_23 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_128, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_46 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_22, 1, -1), kwargs = {})
#   %add_99 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_302, %select_scatter_default_46), kwargs = {})
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_99, %mul_51), kwargs = {})
#   %sum_24 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_129, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_48 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_23, 1, -1), kwargs = {})
#   %add_101 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_301, %select_scatter_default_48), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_101, %mul_49), kwargs = {})
#   %sum_25 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_130, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_50 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_24, 1, -1), kwargs = {})
#   %add_103 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_300, %select_scatter_default_50), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_103, %mul_47), kwargs = {})
#   %sum_26 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_131, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_52 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_25, 1, -1), kwargs = {})
#   %add_105 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_299, %select_scatter_default_52), kwargs = {})
#   %mul_132 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_105, %mul_45), kwargs = {})
#   %sum_27 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_132, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_54 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_26, 1, -1), kwargs = {})
#   %add_107 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_298, %select_scatter_default_54), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_107, %mul_43), kwargs = {})
#   %sum_28 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_133, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_56 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_27, 1, -1), kwargs = {})
#   %add_109 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_297, %select_scatter_default_56), kwargs = {})
#   %mul_134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_109, %mul_41), kwargs = {})
#   %sum_29 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_134, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_58 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_28, 1, -1), kwargs = {})
#   %add_111 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_296, %select_scatter_default_58), kwargs = {})
#   %mul_135 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_111, %mul_39), kwargs = {})
#   %sum_30 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_135, [1], True), kwargs = {dtype: torch.float32})
triton_per_fused_add_mul_select_backward_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: 'i32', 65: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_select_backward_sum_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 105, 'num_reduction': 30, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 32
    x1 = (xindex // 32)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (92224 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (960 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (960 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + (86080 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp13 = tl.load(in_ptr1 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp16 = tl.load(in_ptr2 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr4 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr0 + (79936 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp31 = tl.load(in_ptr1 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp34 = tl.load(in_ptr2 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr5 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr0 + (73792 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp45 = tl.load(in_ptr1 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp48 = tl.load(in_ptr2 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tl.load(in_ptr6 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr0 + (67648 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp59 = tl.load(in_ptr1 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp62 = tl.load(in_ptr2 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tl.load(in_ptr7 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp72 = tl.load(in_ptr0 + (61504 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp73 = tl.load(in_ptr1 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp76 = tl.load(in_ptr2 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp80 = tl.load(in_ptr8 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp86 = tl.load(in_ptr0 + (55360 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp87 = tl.load(in_ptr1 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp90 = tl.load(in_ptr2 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp94 = tl.load(in_ptr9 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp100 = tl.load(in_ptr0 + (49216 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp101 = tl.load(in_ptr1 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp104 = tl.load(in_ptr2 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp108 = tl.load(in_ptr10 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp114 = tl.load(in_ptr0 + (43072 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp115 = tl.load(in_ptr1 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp118 = tl.load(in_ptr2 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp122 = tl.load(in_ptr11 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp128 = tl.load(in_ptr0 + (36928 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp129 = tl.load(in_ptr1 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp132 = tl.load(in_ptr2 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp136 = tl.load(in_ptr12 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp142 = tl.load(in_ptr0 + (30784 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp143 = tl.load(in_ptr1 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp146 = tl.load(in_ptr2 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp150 = tl.load(in_ptr13 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp156 = tl.load(in_ptr0 + (24640 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp157 = tl.load(in_ptr1 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp160 = tl.load(in_ptr2 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp164 = tl.load(in_ptr14 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp170 = tl.load(in_ptr0 + (18496 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp171 = tl.load(in_ptr1 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp174 = tl.load(in_ptr2 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp178 = tl.load(in_ptr15 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp184 = tl.load(in_ptr0 + (12352 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp185 = tl.load(in_ptr1 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp188 = tl.load(in_ptr2 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp192 = tl.load(in_ptr16 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp198 = tl.load(in_ptr0 + (6208 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp199 = tl.load(in_ptr1 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp202 = tl.load(in_ptr2 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp206 = tl.load(in_ptr17 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp212 = tl.load(in_ptr0 + (92192 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp214 = tl.load(in_ptr18 + (960 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp216 = tl.load(in_ptr19 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp222 = tl.load(in_ptr0 + (86048 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp224 = tl.load(in_ptr18 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp228 = tl.load(in_ptr20 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp234 = tl.load(in_ptr0 + (79904 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp236 = tl.load(in_ptr18 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp240 = tl.load(in_ptr21 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp246 = tl.load(in_ptr0 + (73760 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp248 = tl.load(in_ptr18 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp252 = tl.load(in_ptr22 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp258 = tl.load(in_ptr0 + (67616 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp260 = tl.load(in_ptr18 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp264 = tl.load(in_ptr23 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp270 = tl.load(in_ptr0 + (61472 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp272 = tl.load(in_ptr18 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp276 = tl.load(in_ptr24 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp282 = tl.load(in_ptr0 + (55328 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp284 = tl.load(in_ptr18 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp288 = tl.load(in_ptr25 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp294 = tl.load(in_ptr0 + (49184 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp296 = tl.load(in_ptr18 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp300 = tl.load(in_ptr26 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp306 = tl.load(in_ptr0 + (43040 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp308 = tl.load(in_ptr18 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp312 = tl.load(in_ptr27 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp318 = tl.load(in_ptr0 + (36896 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp320 = tl.load(in_ptr18 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp324 = tl.load(in_ptr28 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp330 = tl.load(in_ptr0 + (30752 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp332 = tl.load(in_ptr18 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp336 = tl.load(in_ptr29 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp342 = tl.load(in_ptr0 + (24608 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp344 = tl.load(in_ptr18 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp348 = tl.load(in_ptr30 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp354 = tl.load(in_ptr0 + (18464 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp356 = tl.load(in_ptr18 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp360 = tl.load(in_ptr31 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp366 = tl.load(in_ptr0 + (12320 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp368 = tl.load(in_ptr18 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp372 = tl.load(in_ptr32 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp378 = tl.load(in_ptr0 + (6176 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp380 = tl.load(in_ptr18 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp384 = tl.load(in_ptr33 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tmp5 = tmp3 / tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp15 / tmp16
    tmp18 = r2
    tmp19 = tl.full([1, 1], 63, tl.int32)
    tmp20 = tmp18 == tmp19
    tmp21 = 0.0
    tmp22 = tl.where(tmp20, tmp11, tmp21)
    tmp23 = tmp17 + tmp22
    tmp25 = tmp23 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
    tmp28 = tl.where(xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp30 * tmp32
    tmp35 = tmp33 / tmp34
    tmp36 = tl.where(tmp20, tmp29, tmp21)
    tmp37 = tmp35 + tmp36
    tmp39 = tmp37 * tmp38
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp42 = tl.where(xmask, tmp40, 0)
    tmp43 = tl.sum(tmp42, 1)[:, None]
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp44 * tmp46
    tmp49 = tmp47 / tmp48
    tmp50 = tl.where(tmp20, tmp43, tmp21)
    tmp51 = tmp49 + tmp50
    tmp53 = tmp51 * tmp52
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK, RBLOCK])
    tmp56 = tl.where(xmask, tmp54, 0)
    tmp57 = tl.sum(tmp56, 1)[:, None]
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp58 * tmp60
    tmp63 = tmp61 / tmp62
    tmp64 = tl.where(tmp20, tmp57, tmp21)
    tmp65 = tmp63 + tmp64
    tmp67 = tmp65 * tmp66
    tmp68 = tl.broadcast_to(tmp67, [XBLOCK, RBLOCK])
    tmp70 = tl.where(xmask, tmp68, 0)
    tmp71 = tl.sum(tmp70, 1)[:, None]
    tmp74 = tmp73.to(tl.float32)
    tmp75 = tmp72 * tmp74
    tmp77 = tmp75 / tmp76
    tmp78 = tl.where(tmp20, tmp71, tmp21)
    tmp79 = tmp77 + tmp78
    tmp81 = tmp79 * tmp80
    tmp82 = tl.broadcast_to(tmp81, [XBLOCK, RBLOCK])
    tmp84 = tl.where(xmask, tmp82, 0)
    tmp85 = tl.sum(tmp84, 1)[:, None]
    tmp88 = tmp87.to(tl.float32)
    tmp89 = tmp86 * tmp88
    tmp91 = tmp89 / tmp90
    tmp92 = tl.where(tmp20, tmp85, tmp21)
    tmp93 = tmp91 + tmp92
    tmp95 = tmp93 * tmp94
    tmp96 = tl.broadcast_to(tmp95, [XBLOCK, RBLOCK])
    tmp98 = tl.where(xmask, tmp96, 0)
    tmp99 = tl.sum(tmp98, 1)[:, None]
    tmp102 = tmp101.to(tl.float32)
    tmp103 = tmp100 * tmp102
    tmp105 = tmp103 / tmp104
    tmp106 = tl.where(tmp20, tmp99, tmp21)
    tmp107 = tmp105 + tmp106
    tmp109 = tmp107 * tmp108
    tmp110 = tl.broadcast_to(tmp109, [XBLOCK, RBLOCK])
    tmp112 = tl.where(xmask, tmp110, 0)
    tmp113 = tl.sum(tmp112, 1)[:, None]
    tmp116 = tmp115.to(tl.float32)
    tmp117 = tmp114 * tmp116
    tmp119 = tmp117 / tmp118
    tmp120 = tl.where(tmp20, tmp113, tmp21)
    tmp121 = tmp119 + tmp120
    tmp123 = tmp121 * tmp122
    tmp124 = tl.broadcast_to(tmp123, [XBLOCK, RBLOCK])
    tmp126 = tl.where(xmask, tmp124, 0)
    tmp127 = tl.sum(tmp126, 1)[:, None]
    tmp130 = tmp129.to(tl.float32)
    tmp131 = tmp128 * tmp130
    tmp133 = tmp131 / tmp132
    tmp134 = tl.where(tmp20, tmp127, tmp21)
    tmp135 = tmp133 + tmp134
    tmp137 = tmp135 * tmp136
    tmp138 = tl.broadcast_to(tmp137, [XBLOCK, RBLOCK])
    tmp140 = tl.where(xmask, tmp138, 0)
    tmp141 = tl.sum(tmp140, 1)[:, None]
    tmp144 = tmp143.to(tl.float32)
    tmp145 = tmp142 * tmp144
    tmp147 = tmp145 / tmp146
    tmp148 = tl.where(tmp20, tmp141, tmp21)
    tmp149 = tmp147 + tmp148
    tmp151 = tmp149 * tmp150
    tmp152 = tl.broadcast_to(tmp151, [XBLOCK, RBLOCK])
    tmp154 = tl.where(xmask, tmp152, 0)
    tmp155 = tl.sum(tmp154, 1)[:, None]
    tmp158 = tmp157.to(tl.float32)
    tmp159 = tmp156 * tmp158
    tmp161 = tmp159 / tmp160
    tmp162 = tl.where(tmp20, tmp155, tmp21)
    tmp163 = tmp161 + tmp162
    tmp165 = tmp163 * tmp164
    tmp166 = tl.broadcast_to(tmp165, [XBLOCK, RBLOCK])
    tmp168 = tl.where(xmask, tmp166, 0)
    tmp169 = tl.sum(tmp168, 1)[:, None]
    tmp172 = tmp171.to(tl.float32)
    tmp173 = tmp170 * tmp172
    tmp175 = tmp173 / tmp174
    tmp176 = tl.where(tmp20, tmp169, tmp21)
    tmp177 = tmp175 + tmp176
    tmp179 = tmp177 * tmp178
    tmp180 = tl.broadcast_to(tmp179, [XBLOCK, RBLOCK])
    tmp182 = tl.where(xmask, tmp180, 0)
    tmp183 = tl.sum(tmp182, 1)[:, None]
    tmp186 = tmp185.to(tl.float32)
    tmp187 = tmp184 * tmp186
    tmp189 = tmp187 / tmp188
    tmp190 = tl.where(tmp20, tmp183, tmp21)
    tmp191 = tmp189 + tmp190
    tmp193 = tmp191 * tmp192
    tmp194 = tl.broadcast_to(tmp193, [XBLOCK, RBLOCK])
    tmp196 = tl.where(xmask, tmp194, 0)
    tmp197 = tl.sum(tmp196, 1)[:, None]
    tmp200 = tmp199.to(tl.float32)
    tmp201 = tmp198 * tmp200
    tmp203 = tmp201 / tmp202
    tmp204 = tl.where(tmp20, tmp197, tmp21)
    tmp205 = tmp203 + tmp204
    tmp207 = tmp205 * tmp206
    tmp208 = tl.broadcast_to(tmp207, [XBLOCK, RBLOCK])
    tmp210 = tl.where(xmask, tmp208, 0)
    tmp211 = tl.sum(tmp210, 1)[:, None]
    tmp213 = tmp212 * tmp2
    tmp215 = tmp213 / tmp214
    tmp217 = tmp215 * tmp216
    tmp218 = tl.broadcast_to(tmp217, [XBLOCK, RBLOCK])
    tmp220 = tl.where(xmask, tmp218, 0)
    tmp221 = tl.sum(tmp220, 1)[:, None]
    tmp223 = tmp222 * tmp14
    tmp225 = tmp223 / tmp224
    tmp226 = tl.where(tmp20, tmp221, tmp21)
    tmp227 = tmp225 + tmp226
    tmp229 = tmp227 * tmp228
    tmp230 = tl.broadcast_to(tmp229, [XBLOCK, RBLOCK])
    tmp232 = tl.where(xmask, tmp230, 0)
    tmp233 = tl.sum(tmp232, 1)[:, None]
    tmp235 = tmp234 * tmp32
    tmp237 = tmp235 / tmp236
    tmp238 = tl.where(tmp20, tmp233, tmp21)
    tmp239 = tmp237 + tmp238
    tmp241 = tmp239 * tmp240
    tmp242 = tl.broadcast_to(tmp241, [XBLOCK, RBLOCK])
    tmp244 = tl.where(xmask, tmp242, 0)
    tmp245 = tl.sum(tmp244, 1)[:, None]
    tmp247 = tmp246 * tmp46
    tmp249 = tmp247 / tmp248
    tmp250 = tl.where(tmp20, tmp245, tmp21)
    tmp251 = tmp249 + tmp250
    tmp253 = tmp251 * tmp252
    tmp254 = tl.broadcast_to(tmp253, [XBLOCK, RBLOCK])
    tmp256 = tl.where(xmask, tmp254, 0)
    tmp257 = tl.sum(tmp256, 1)[:, None]
    tmp259 = tmp258 * tmp60
    tmp261 = tmp259 / tmp260
    tmp262 = tl.where(tmp20, tmp257, tmp21)
    tmp263 = tmp261 + tmp262
    tmp265 = tmp263 * tmp264
    tmp266 = tl.broadcast_to(tmp265, [XBLOCK, RBLOCK])
    tmp268 = tl.where(xmask, tmp266, 0)
    tmp269 = tl.sum(tmp268, 1)[:, None]
    tmp271 = tmp270 * tmp74
    tmp273 = tmp271 / tmp272
    tmp274 = tl.where(tmp20, tmp269, tmp21)
    tmp275 = tmp273 + tmp274
    tmp277 = tmp275 * tmp276
    tmp278 = tl.broadcast_to(tmp277, [XBLOCK, RBLOCK])
    tmp280 = tl.where(xmask, tmp278, 0)
    tmp281 = tl.sum(tmp280, 1)[:, None]
    tmp283 = tmp282 * tmp88
    tmp285 = tmp283 / tmp284
    tmp286 = tl.where(tmp20, tmp281, tmp21)
    tmp287 = tmp285 + tmp286
    tmp289 = tmp287 * tmp288
    tmp290 = tl.broadcast_to(tmp289, [XBLOCK, RBLOCK])
    tmp292 = tl.where(xmask, tmp290, 0)
    tmp293 = tl.sum(tmp292, 1)[:, None]
    tmp295 = tmp294 * tmp102
    tmp297 = tmp295 / tmp296
    tmp298 = tl.where(tmp20, tmp293, tmp21)
    tmp299 = tmp297 + tmp298
    tmp301 = tmp299 * tmp300
    tmp302 = tl.broadcast_to(tmp301, [XBLOCK, RBLOCK])
    tmp304 = tl.where(xmask, tmp302, 0)
    tmp305 = tl.sum(tmp304, 1)[:, None]
    tmp307 = tmp306 * tmp116
    tmp309 = tmp307 / tmp308
    tmp310 = tl.where(tmp20, tmp305, tmp21)
    tmp311 = tmp309 + tmp310
    tmp313 = tmp311 * tmp312
    tmp314 = tl.broadcast_to(tmp313, [XBLOCK, RBLOCK])
    tmp316 = tl.where(xmask, tmp314, 0)
    tmp317 = tl.sum(tmp316, 1)[:, None]
    tmp319 = tmp318 * tmp130
    tmp321 = tmp319 / tmp320
    tmp322 = tl.where(tmp20, tmp317, tmp21)
    tmp323 = tmp321 + tmp322
    tmp325 = tmp323 * tmp324
    tmp326 = tl.broadcast_to(tmp325, [XBLOCK, RBLOCK])
    tmp328 = tl.where(xmask, tmp326, 0)
    tmp329 = tl.sum(tmp328, 1)[:, None]
    tmp331 = tmp330 * tmp144
    tmp333 = tmp331 / tmp332
    tmp334 = tl.where(tmp20, tmp329, tmp21)
    tmp335 = tmp333 + tmp334
    tmp337 = tmp335 * tmp336
    tmp338 = tl.broadcast_to(tmp337, [XBLOCK, RBLOCK])
    tmp340 = tl.where(xmask, tmp338, 0)
    tmp341 = tl.sum(tmp340, 1)[:, None]
    tmp343 = tmp342 * tmp158
    tmp345 = tmp343 / tmp344
    tmp346 = tl.where(tmp20, tmp341, tmp21)
    tmp347 = tmp345 + tmp346
    tmp349 = tmp347 * tmp348
    tmp350 = tl.broadcast_to(tmp349, [XBLOCK, RBLOCK])
    tmp352 = tl.where(xmask, tmp350, 0)
    tmp353 = tl.sum(tmp352, 1)[:, None]
    tmp355 = tmp354 * tmp172
    tmp357 = tmp355 / tmp356
    tmp358 = tl.where(tmp20, tmp353, tmp21)
    tmp359 = tmp357 + tmp358
    tmp361 = tmp359 * tmp360
    tmp362 = tl.broadcast_to(tmp361, [XBLOCK, RBLOCK])
    tmp364 = tl.where(xmask, tmp362, 0)
    tmp365 = tl.sum(tmp364, 1)[:, None]
    tmp367 = tmp366 * tmp186
    tmp369 = tmp367 / tmp368
    tmp370 = tl.where(tmp20, tmp365, tmp21)
    tmp371 = tmp369 + tmp370
    tmp373 = tmp371 * tmp372
    tmp374 = tl.broadcast_to(tmp373, [XBLOCK, RBLOCK])
    tmp376 = tl.where(xmask, tmp374, 0)
    tmp377 = tl.sum(tmp376, 1)[:, None]
    tmp379 = tmp378 * tmp200
    tmp381 = tmp379 / tmp380
    tmp382 = tl.where(tmp20, tmp377, tmp21)
    tmp383 = tmp381 + tmp382
    tmp385 = tmp383 * tmp384
    tmp386 = tl.broadcast_to(tmp385, [XBLOCK, RBLOCK])
    tmp388 = tl.where(xmask, tmp386, 0)
    tmp389 = tl.sum(tmp388, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
    tl.store(out_ptr1 + (x3), tmp29, xmask)
    tl.store(out_ptr2 + (x3), tmp43, xmask)
    tl.store(out_ptr3 + (x3), tmp57, xmask)
    tl.store(out_ptr4 + (x3), tmp71, xmask)
    tl.store(out_ptr5 + (x3), tmp85, xmask)
    tl.store(out_ptr6 + (x3), tmp99, xmask)
    tl.store(out_ptr7 + (x3), tmp113, xmask)
    tl.store(out_ptr8 + (x3), tmp127, xmask)
    tl.store(out_ptr9 + (x3), tmp141, xmask)
    tl.store(out_ptr10 + (x3), tmp155, xmask)
    tl.store(out_ptr11 + (x3), tmp169, xmask)
    tl.store(out_ptr12 + (x3), tmp183, xmask)
    tl.store(out_ptr13 + (x3), tmp197, xmask)
    tl.store(out_ptr14 + (x3), tmp211, xmask)
    tl.store(out_ptr15 + (x3), tmp221, xmask)
    tl.store(out_ptr16 + (x3), tmp233, xmask)
    tl.store(out_ptr17 + (x3), tmp245, xmask)
    tl.store(out_ptr18 + (x3), tmp257, xmask)
    tl.store(out_ptr19 + (x3), tmp269, xmask)
    tl.store(out_ptr20 + (x3), tmp281, xmask)
    tl.store(out_ptr21 + (x3), tmp293, xmask)
    tl.store(out_ptr22 + (x3), tmp305, xmask)
    tl.store(out_ptr23 + (x3), tmp317, xmask)
    tl.store(out_ptr24 + (x3), tmp329, xmask)
    tl.store(out_ptr25 + (x3), tmp341, xmask)
    tl.store(out_ptr26 + (x3), tmp353, xmask)
    tl.store(out_ptr27 + (x3), tmp365, xmask)
    tl.store(out_ptr28 + (x3), tmp377, xmask)
    tl.store(out_ptr29 + (x3), tmp389, xmask)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/7t/c7trsvowntelwuphyrdqitaekn3fqa5q75uodbq2snphvaq3oo5l.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward, aten.add, aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_18 : [num_users=48] = call_function[target=torch.ops.aten.full.default](args = ([32, 16, 64, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %select_294, 1, 15), kwargs = {})
#   %full_default_20 : [num_users=45] = call_function[target=torch.ops.aten.full.default](args = ([32, 64, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze, 1, -1), kwargs = {})
#   %add_55 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_293, %select_scatter_default_1), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_55, 1, 14), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_2), kwargs = {})
#   %select_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_1, 1, -1), kwargs = {})
#   %add_57 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_292, %select_scatter_default_3), kwargs = {})
#   %select_scatter_default_4 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_57, 1, 13), kwargs = {})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_56, %select_scatter_default_4), kwargs = {})
#   %select_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_2, 1, -1), kwargs = {})
#   %add_59 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_291, %select_scatter_default_5), kwargs = {})
#   %select_scatter_default_6 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_59, 1, 12), kwargs = {})
#   %add_60 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_58, %select_scatter_default_6), kwargs = {})
#   %select_scatter_default_7 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_3, 1, -1), kwargs = {})
#   %add_61 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_290, %select_scatter_default_7), kwargs = {})
#   %select_scatter_default_8 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_61, 1, 11), kwargs = {})
#   %add_62 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_60, %select_scatter_default_8), kwargs = {})
#   %select_scatter_default_9 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_4, 1, -1), kwargs = {})
#   %add_63 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_289, %select_scatter_default_9), kwargs = {})
#   %select_scatter_default_10 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_63, 1, 10), kwargs = {})
#   %add_64 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_62, %select_scatter_default_10), kwargs = {})
#   %select_scatter_default_11 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_5, 1, -1), kwargs = {})
#   %add_65 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_288, %select_scatter_default_11), kwargs = {})
#   %select_scatter_default_12 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_65, 1, 9), kwargs = {})
#   %add_66 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_64, %select_scatter_default_12), kwargs = {})
#   %select_scatter_default_13 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_6, 1, -1), kwargs = {})
#   %add_67 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_287, %select_scatter_default_13), kwargs = {})
#   %select_scatter_default_14 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_67, 1, 8), kwargs = {})
#   %add_68 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_66, %select_scatter_default_14), kwargs = {})
#   %select_scatter_default_15 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_7, 1, -1), kwargs = {})
#   %add_69 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_286, %select_scatter_default_15), kwargs = {})
#   %select_scatter_default_16 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_69, 1, 7), kwargs = {})
#   %add_70 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_68, %select_scatter_default_16), kwargs = {})
#   %select_scatter_default_17 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_8, 1, -1), kwargs = {})
#   %add_71 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_285, %select_scatter_default_17), kwargs = {})
#   %select_scatter_default_18 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_71, 1, 6), kwargs = {})
#   %add_72 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_70, %select_scatter_default_18), kwargs = {})
#   %select_scatter_default_19 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_9, 1, -1), kwargs = {})
#   %add_73 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_284, %select_scatter_default_19), kwargs = {})
#   %select_scatter_default_20 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_73, 1, 5), kwargs = {})
#   %add_74 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_72, %select_scatter_default_20), kwargs = {})
#   %select_scatter_default_21 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_10, 1, -1), kwargs = {})
#   %add_75 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_283, %select_scatter_default_21), kwargs = {})
#   %select_scatter_default_22 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_75, 1, 4), kwargs = {})
#   %add_76 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_74, %select_scatter_default_22), kwargs = {})
#   %select_scatter_default_23 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_11, 1, -1), kwargs = {})
#   %add_77 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_282, %select_scatter_default_23), kwargs = {})
#   %select_scatter_default_24 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_77, 1, 3), kwargs = {})
#   %add_78 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_76, %select_scatter_default_24), kwargs = {})
#   %select_scatter_default_25 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_12, 1, -1), kwargs = {})
#   %add_79 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_281, %select_scatter_default_25), kwargs = {})
#   %select_scatter_default_26 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_79, 1, 2), kwargs = {})
#   %add_80 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_78, %select_scatter_default_26), kwargs = {})
#   %select_scatter_default_27 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_13, 1, -1), kwargs = {})
#   %add_81 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_280, %select_scatter_default_27), kwargs = {})
#   %select_scatter_default_28 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_81, 1, 1), kwargs = {})
#   %add_82 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_80, %select_scatter_default_28), kwargs = {})
#   %convert_element_type_63 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_126, torch.bfloat16), kwargs = {})
#   %select_scatter_default_31 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %select_310, 1, 15), kwargs = {})
#   %select_scatter_default_32 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_15, 1, -1), kwargs = {})
#   %add_85 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_309, %select_scatter_default_32), kwargs = {})
#   %select_scatter_default_33 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_85, 1, 14), kwargs = {})
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default_31, %select_scatter_default_33), kwargs = {})
#   %select_scatter_default_34 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_16, 1, -1), kwargs = {})
#   %add_87 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_308, %select_scatter_default_34), kwargs = {})
#   %select_scatter_default_35 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_87, 1, 13), kwargs = {})
#   %add_88 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_86, %select_scatter_default_35), kwargs = {})
#   %select_scatter_default_36 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_17, 1, -1), kwargs = {})
#   %add_89 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_307, %select_scatter_default_36), kwargs = {})
#   %select_scatter_default_37 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_89, 1, 12), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_88, %select_scatter_default_37), kwargs = {})
#   %select_scatter_default_38 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_18, 1, -1), kwargs = {})
#   %add_91 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_306, %select_scatter_default_38), kwargs = {})
#   %select_scatter_default_39 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_91, 1, 11), kwargs = {})
#   %add_92 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_90, %select_scatter_default_39), kwargs = {})
#   %select_scatter_default_40 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_19, 1, -1), kwargs = {})
#   %add_93 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_305, %select_scatter_default_40), kwargs = {})
#   %select_scatter_default_41 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_93, 1, 10), kwargs = {})
#   %add_94 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_92, %select_scatter_default_41), kwargs = {})
#   %select_scatter_default_42 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_20, 1, -1), kwargs = {})
#   %add_95 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_304, %select_scatter_default_42), kwargs = {})
#   %select_scatter_default_43 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_95, 1, 9), kwargs = {})
#   %add_96 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_94, %select_scatter_default_43), kwargs = {})
#   %select_scatter_default_44 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_21, 1, -1), kwargs = {})
#   %add_97 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_303, %select_scatter_default_44), kwargs = {})
#   %select_scatter_default_45 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_97, 1, 8), kwargs = {})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_96, %select_scatter_default_45), kwargs = {})
#   %select_scatter_default_46 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_22, 1, -1), kwargs = {})
#   %add_99 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_302, %select_scatter_default_46), kwargs = {})
#   %select_scatter_default_47 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_99, 1, 7), kwargs = {})
#   %add_100 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_98, %select_scatter_default_47), kwargs = {})
#   %select_scatter_default_48 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_23, 1, -1), kwargs = {})
#   %add_101 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_301, %select_scatter_default_48), kwargs = {})
#   %select_scatter_default_49 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_101, 1, 6), kwargs = {})
#   %add_102 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_100, %select_scatter_default_49), kwargs = {})
#   %select_scatter_default_50 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_24, 1, -1), kwargs = {})
#   %add_103 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_300, %select_scatter_default_50), kwargs = {})
#   %select_scatter_default_51 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_103, 1, 5), kwargs = {})
#   %add_104 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_102, %select_scatter_default_51), kwargs = {})
#   %select_scatter_default_52 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_25, 1, -1), kwargs = {})
#   %add_105 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_299, %select_scatter_default_52), kwargs = {})
#   %select_scatter_default_53 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_105, 1, 4), kwargs = {})
#   %add_106 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_104, %select_scatter_default_53), kwargs = {})
#   %select_scatter_default_54 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_26, 1, -1), kwargs = {})
#   %add_107 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_298, %select_scatter_default_54), kwargs = {})
#   %select_scatter_default_55 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_107, 1, 3), kwargs = {})
#   %add_108 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_106, %select_scatter_default_55), kwargs = {})
#   %select_scatter_default_56 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_27, 1, -1), kwargs = {})
#   %add_109 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_297, %select_scatter_default_56), kwargs = {})
#   %select_scatter_default_57 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_109, 1, 2), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_108, %select_scatter_default_57), kwargs = {})
#   %select_scatter_default_58 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_28, 1, -1), kwargs = {})
#   %add_111 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_296, %select_scatter_default_58), kwargs = {})
#   %select_scatter_default_59 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_111, 1, 1), kwargs = {})
#   %add_112 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_110, %select_scatter_default_59), kwargs = {})
#   %convert_element_type_68 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_130, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_add_select_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*bf16', 35: '*bf16', 36: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_select_backward_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 110, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048) % 16
    x0 = xindex % 32
    x1 = (xindex // 32) % 64
    x3 = (xindex // 32768)
    x6 = xindex
    x4 = (xindex // 2048)
    tmp3 = tl.load(in_ptr0 + (92224 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp7 = tl.load(in_ptr2 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (86080 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp17 = tl.load(in_ptr2 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (79936 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr1 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp33 = tl.load(in_ptr2 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr4 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (73792 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr1 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp46 = tl.load(in_ptr2 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr5 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (67648 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr1 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp59 = tl.load(in_ptr2 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr6 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr0 + (61504 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr1 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp72 = tl.load(in_ptr2 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr7 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr0 + (55360 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr1 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp85 = tl.load(in_ptr2 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr8 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr0 + (49216 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr1 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp98 = tl.load(in_ptr2 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp100 = tl.load(in_ptr9 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr0 + (43072 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr1 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp111 = tl.load(in_ptr2 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp113 = tl.load(in_ptr10 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp120 = tl.load(in_ptr0 + (36928 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp121 = tl.load(in_ptr1 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp124 = tl.load(in_ptr2 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp126 = tl.load(in_ptr11 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp133 = tl.load(in_ptr0 + (30784 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp134 = tl.load(in_ptr1 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp137 = tl.load(in_ptr2 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp139 = tl.load(in_ptr12 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp146 = tl.load(in_ptr0 + (24640 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp147 = tl.load(in_ptr1 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp150 = tl.load(in_ptr2 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp152 = tl.load(in_ptr13 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp159 = tl.load(in_ptr0 + (18496 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp160 = tl.load(in_ptr1 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp163 = tl.load(in_ptr2 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp165 = tl.load(in_ptr14 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp172 = tl.load(in_ptr0 + (12352 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp173 = tl.load(in_ptr1 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp176 = tl.load(in_ptr2 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp178 = tl.load(in_ptr15 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp185 = tl.load(in_ptr0 + (6208 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp186 = tl.load(in_ptr1 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp189 = tl.load(in_ptr2 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp191 = tl.load(in_ptr16 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp199 = tl.load(in_ptr0 + (64 + x0 + (96*x1) + (98304*(x4 // 16))), None)
    tmp200 = tl.load(in_ptr1 + (x1 + (1024*(x4 // 16))), None, eviction_policy='evict_last').to(tl.int1)
    tmp203 = tl.load(in_ptr2 + (x1 + (1024*(x4 // 16))), None, eviction_policy='evict_last')
    tmp205 = tl.load(in_ptr17 + (x0 + (32*(x4 // 16))), None, eviction_policy='evict_last')
    tmp211 = tl.load(in_ptr0 + (92192 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp213 = tl.load(in_ptr18 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp216 = tl.load(in_ptr0 + (86048 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp218 = tl.load(in_ptr18 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp220 = tl.load(in_ptr19 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp225 = tl.load(in_ptr0 + (79904 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp227 = tl.load(in_ptr18 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp229 = tl.load(in_ptr20 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp234 = tl.load(in_ptr0 + (73760 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp236 = tl.load(in_ptr18 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp238 = tl.load(in_ptr21 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp243 = tl.load(in_ptr0 + (67616 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp245 = tl.load(in_ptr18 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp247 = tl.load(in_ptr22 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp252 = tl.load(in_ptr0 + (61472 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp254 = tl.load(in_ptr18 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp256 = tl.load(in_ptr23 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp261 = tl.load(in_ptr0 + (55328 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp263 = tl.load(in_ptr18 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp265 = tl.load(in_ptr24 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp270 = tl.load(in_ptr0 + (49184 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp272 = tl.load(in_ptr18 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp274 = tl.load(in_ptr25 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp279 = tl.load(in_ptr0 + (43040 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp281 = tl.load(in_ptr18 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp283 = tl.load(in_ptr26 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp288 = tl.load(in_ptr0 + (36896 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp290 = tl.load(in_ptr18 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp292 = tl.load(in_ptr27 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp297 = tl.load(in_ptr0 + (30752 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp299 = tl.load(in_ptr18 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp301 = tl.load(in_ptr28 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp306 = tl.load(in_ptr0 + (24608 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp308 = tl.load(in_ptr18 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp310 = tl.load(in_ptr29 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp315 = tl.load(in_ptr0 + (18464 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp317 = tl.load(in_ptr18 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp319 = tl.load(in_ptr30 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp324 = tl.load(in_ptr0 + (12320 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp326 = tl.load(in_ptr18 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp328 = tl.load(in_ptr31 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp333 = tl.load(in_ptr0 + (6176 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp335 = tl.load(in_ptr18 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp337 = tl.load(in_ptr32 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp342 = tl.load(in_ptr0 + (32 + x0 + (96*x1) + (98304*(x4 // 16))), None)
    tmp344 = tl.load(in_ptr18 + (x1 + (1024*(x4 // 16))), None, eviction_policy='evict_last')
    tmp346 = tl.load(in_ptr33 + (x0 + (32*(x4 // 16))), None, eviction_policy='evict_last')
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
    tmp196 = (x6 // 2048) % 16
    tmp197 = tl.full([1], 0, tl.int32)
    tmp198 = tmp196 == tmp197
    tmp201 = tmp200.to(tl.float32)
    tmp202 = tmp199 * tmp201
    tmp204 = tmp202 / tmp203
    tmp206 = tl.where(tmp21, tmp205, tmp9)
    tmp207 = tmp204 + tmp206
    tmp208 = tl.where(tmp198, tmp207, tmp9)
    tmp209 = tmp195 + tmp208
    tmp210 = tmp209.to(tl.float32)
    tmp212 = tmp211 * tmp5
    tmp214 = tmp212 / tmp213
    tmp215 = tl.where(tmp2, tmp214, tmp9)
    tmp217 = tmp216 * tmp15
    tmp219 = tmp217 / tmp218
    tmp221 = tl.where(tmp21, tmp220, tmp9)
    tmp222 = tmp219 + tmp221
    tmp223 = tl.where(tmp12, tmp222, tmp9)
    tmp224 = tmp215 + tmp223
    tmp226 = tmp225 * tmp31
    tmp228 = tmp226 / tmp227
    tmp230 = tl.where(tmp21, tmp229, tmp9)
    tmp231 = tmp228 + tmp230
    tmp232 = tl.where(tmp28, tmp231, tmp9)
    tmp233 = tmp224 + tmp232
    tmp235 = tmp234 * tmp44
    tmp237 = tmp235 / tmp236
    tmp239 = tl.where(tmp21, tmp238, tmp9)
    tmp240 = tmp237 + tmp239
    tmp241 = tl.where(tmp41, tmp240, tmp9)
    tmp242 = tmp233 + tmp241
    tmp244 = tmp243 * tmp57
    tmp246 = tmp244 / tmp245
    tmp248 = tl.where(tmp21, tmp247, tmp9)
    tmp249 = tmp246 + tmp248
    tmp250 = tl.where(tmp54, tmp249, tmp9)
    tmp251 = tmp242 + tmp250
    tmp253 = tmp252 * tmp70
    tmp255 = tmp253 / tmp254
    tmp257 = tl.where(tmp21, tmp256, tmp9)
    tmp258 = tmp255 + tmp257
    tmp259 = tl.where(tmp67, tmp258, tmp9)
    tmp260 = tmp251 + tmp259
    tmp262 = tmp261 * tmp83
    tmp264 = tmp262 / tmp263
    tmp266 = tl.where(tmp21, tmp265, tmp9)
    tmp267 = tmp264 + tmp266
    tmp268 = tl.where(tmp80, tmp267, tmp9)
    tmp269 = tmp260 + tmp268
    tmp271 = tmp270 * tmp96
    tmp273 = tmp271 / tmp272
    tmp275 = tl.where(tmp21, tmp274, tmp9)
    tmp276 = tmp273 + tmp275
    tmp277 = tl.where(tmp93, tmp276, tmp9)
    tmp278 = tmp269 + tmp277
    tmp280 = tmp279 * tmp109
    tmp282 = tmp280 / tmp281
    tmp284 = tl.where(tmp21, tmp283, tmp9)
    tmp285 = tmp282 + tmp284
    tmp286 = tl.where(tmp106, tmp285, tmp9)
    tmp287 = tmp278 + tmp286
    tmp289 = tmp288 * tmp122
    tmp291 = tmp289 / tmp290
    tmp293 = tl.where(tmp21, tmp292, tmp9)
    tmp294 = tmp291 + tmp293
    tmp295 = tl.where(tmp119, tmp294, tmp9)
    tmp296 = tmp287 + tmp295
    tmp298 = tmp297 * tmp135
    tmp300 = tmp298 / tmp299
    tmp302 = tl.where(tmp21, tmp301, tmp9)
    tmp303 = tmp300 + tmp302
    tmp304 = tl.where(tmp132, tmp303, tmp9)
    tmp305 = tmp296 + tmp304
    tmp307 = tmp306 * tmp148
    tmp309 = tmp307 / tmp308
    tmp311 = tl.where(tmp21, tmp310, tmp9)
    tmp312 = tmp309 + tmp311
    tmp313 = tl.where(tmp145, tmp312, tmp9)
    tmp314 = tmp305 + tmp313
    tmp316 = tmp315 * tmp161
    tmp318 = tmp316 / tmp317
    tmp320 = tl.where(tmp21, tmp319, tmp9)
    tmp321 = tmp318 + tmp320
    tmp322 = tl.where(tmp158, tmp321, tmp9)
    tmp323 = tmp314 + tmp322
    tmp325 = tmp324 * tmp174
    tmp327 = tmp325 / tmp326
    tmp329 = tl.where(tmp21, tmp328, tmp9)
    tmp330 = tmp327 + tmp329
    tmp331 = tl.where(tmp171, tmp330, tmp9)
    tmp332 = tmp323 + tmp331
    tmp334 = tmp333 * tmp187
    tmp336 = tmp334 / tmp335
    tmp338 = tl.where(tmp21, tmp337, tmp9)
    tmp339 = tmp336 + tmp338
    tmp340 = tl.where(tmp184, tmp339, tmp9)
    tmp341 = tmp332 + tmp340
    tmp343 = tmp342 * tmp201
    tmp345 = tmp343 / tmp344
    tmp347 = tl.where(tmp21, tmp346, tmp9)
    tmp348 = tmp345 + tmp347
    tmp349 = tl.where(tmp198, tmp348, tmp9)
    tmp350 = tmp341 + tmp349
    tmp351 = tmp350.to(tl.float32)
    tl.store(out_ptr0 + (x6), tmp210, None)
    tl.store(out_ptr1 + (x6), tmp351, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/gy/cgynny47dfdukbs2zaqnarchas633kizdqd5zshr3dlkt4v2eau4.py
# Topologically Sorted Source Nodes: [mul_31, mul_29, mul_27, mul_25, mul_23, mul_21, mul_19, mul_17, mul_15, mul_13, mul_11, mul_9, mul_7, mul_5], Original ATen: [aten.select_backward, aten.mul, aten.sum, aten.add]
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
#   mul_31 => mul_31
#   mul_5 => mul_5
#   mul_7 => mul_7
#   mul_9 => mul_9
# Graph fragment:
#   %full_default_20 : [num_users=45] = call_function[target=torch.ops.aten.full.default](args = ([32, 64, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_326, %mul_33), kwargs = {})
#   %sum_31 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_137, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_63 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_30, 1, -1), kwargs = {})
#   %add_116 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_325, %select_scatter_default_63), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_31, %view_10), kwargs = {})
#   %mul_138 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_116, %mul_31), kwargs = {})
#   %sum_32 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_138, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_65 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_31, 1, -1), kwargs = {})
#   %add_118 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_324, %select_scatter_default_65), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_29, %view_10), kwargs = {})
#   %mul_139 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_118, %mul_29), kwargs = {})
#   %sum_33 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_139, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_67 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_32, 1, -1), kwargs = {})
#   %add_120 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_323, %select_scatter_default_67), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_27, %view_10), kwargs = {})
#   %mul_140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_120, %mul_27), kwargs = {})
#   %sum_34 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_140, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_69 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_33, 1, -1), kwargs = {})
#   %add_122 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_322, %select_scatter_default_69), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_25, %view_10), kwargs = {})
#   %mul_141 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_122, %mul_25), kwargs = {})
#   %sum_35 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_141, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_71 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_34, 1, -1), kwargs = {})
#   %add_124 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_321, %select_scatter_default_71), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_23, %view_10), kwargs = {})
#   %mul_142 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_124, %mul_23), kwargs = {})
#   %sum_36 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_142, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_73 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_35, 1, -1), kwargs = {})
#   %add_126 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_320, %select_scatter_default_73), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_21, %view_10), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_126, %mul_21), kwargs = {})
#   %sum_37 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_143, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_75 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_36, 1, -1), kwargs = {})
#   %add_128 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_319, %select_scatter_default_75), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_19, %view_10), kwargs = {})
#   %mul_144 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_128, %mul_19), kwargs = {})
#   %sum_38 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_144, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_77 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_37, 1, -1), kwargs = {})
#   %add_130 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_318, %select_scatter_default_77), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_17, %view_10), kwargs = {})
#   %mul_145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_130, %mul_17), kwargs = {})
#   %sum_39 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_145, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_79 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_38, 1, -1), kwargs = {})
#   %add_132 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_317, %select_scatter_default_79), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_15, %view_10), kwargs = {})
#   %mul_146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_132, %mul_15), kwargs = {})
#   %sum_40 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_146, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_81 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_39, 1, -1), kwargs = {})
#   %add_134 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_316, %select_scatter_default_81), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_13, %view_10), kwargs = {})
#   %mul_147 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_134, %mul_13), kwargs = {})
#   %sum_41 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_147, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_83 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_40, 1, -1), kwargs = {})
#   %add_136 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_315, %select_scatter_default_83), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_11, %view_10), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_136, %mul_11), kwargs = {})
#   %sum_42 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_148, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_85 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_41, 1, -1), kwargs = {})
#   %add_138 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_314, %select_scatter_default_85), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_9, %view_10), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_138, %mul_9), kwargs = {})
#   %sum_43 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_149, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_87 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_42, 1, -1), kwargs = {})
#   %add_140 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_313, %select_scatter_default_87), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_7, %view_10), kwargs = {})
#   %mul_150 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_140, %mul_7), kwargs = {})
#   %sum_44 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_150, [1], True), kwargs = {dtype: torch.float32})
#   %select_scatter_default_89 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_43, 1, -1), kwargs = {})
#   %add_142 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_312, %select_scatter_default_89), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_5, %view_10), kwargs = {})
#   %mul_151 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_142, %mul_5), kwargs = {})
#   %sum_45 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_151, [1], True), kwargs = {dtype: torch.float32})
triton_per_fused_add_mul_select_backward_sum_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*i1', 7: '*i1', 8: '*i1', 9: '*i1', 10: '*i1', 11: '*i1', 12: '*i1', 13: '*i1', 14: '*i1', 15: '*i1', 16: '*i1', 17: '*i1', 18: '*i1', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: 'i32', 35: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_select_backward_sum_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 61, 'num_reduction': 15, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 32
    x1 = (xindex // 32)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (92160 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (960 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (960 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + (86016 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp13 = tl.load(in_ptr1 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp16 = tl.load(in_ptr2 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr4 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp26 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (79872 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp34 = tl.load(in_ptr1 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp37 = tl.load(in_ptr2 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.load(in_ptr6 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp49 = tl.load(in_ptr0 + (73728 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp50 = tl.load(in_ptr1 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp53 = tl.load(in_ptr2 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.load(in_ptr7 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp65 = tl.load(in_ptr0 + (67584 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp66 = tl.load(in_ptr1 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp69 = tl.load(in_ptr2 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp73 = tl.load(in_ptr8 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp81 = tl.load(in_ptr0 + (61440 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp82 = tl.load(in_ptr1 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp85 = tl.load(in_ptr2 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp89 = tl.load(in_ptr9 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp97 = tl.load(in_ptr0 + (55296 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp98 = tl.load(in_ptr1 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp101 = tl.load(in_ptr2 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp105 = tl.load(in_ptr10 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp113 = tl.load(in_ptr0 + (49152 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp114 = tl.load(in_ptr1 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp117 = tl.load(in_ptr2 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp121 = tl.load(in_ptr11 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp129 = tl.load(in_ptr0 + (43008 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp130 = tl.load(in_ptr1 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp133 = tl.load(in_ptr2 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp137 = tl.load(in_ptr12 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp145 = tl.load(in_ptr0 + (36864 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp146 = tl.load(in_ptr1 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp149 = tl.load(in_ptr2 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp153 = tl.load(in_ptr13 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp161 = tl.load(in_ptr0 + (30720 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp162 = tl.load(in_ptr1 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp165 = tl.load(in_ptr2 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp169 = tl.load(in_ptr14 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp177 = tl.load(in_ptr0 + (24576 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp178 = tl.load(in_ptr1 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp181 = tl.load(in_ptr2 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp185 = tl.load(in_ptr15 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp193 = tl.load(in_ptr0 + (18432 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp194 = tl.load(in_ptr1 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp197 = tl.load(in_ptr2 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp201 = tl.load(in_ptr16 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp209 = tl.load(in_ptr0 + (12288 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp210 = tl.load(in_ptr1 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp213 = tl.load(in_ptr2 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp217 = tl.load(in_ptr17 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp225 = tl.load(in_ptr0 + (6144 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp226 = tl.load(in_ptr1 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp229 = tl.load(in_ptr2 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp233 = tl.load(in_ptr18 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tmp5 = tmp3 / tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp15 / tmp16
    tmp18 = r2
    tmp19 = tl.full([1, 1], 63, tl.int32)
    tmp20 = tmp18 == tmp19
    tmp21 = 0.0
    tmp22 = tl.where(tmp20, tmp11, tmp21)
    tmp23 = tmp17 + tmp22
    tmp25 = tmp24.to(tl.float32)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp23 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp33 * tmp35
    tmp38 = tmp36 / tmp37
    tmp39 = tl.where(tmp20, tmp32, tmp21)
    tmp40 = tmp38 + tmp39
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp42 * tmp26
    tmp44 = tmp40 * tmp43
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, RBLOCK])
    tmp47 = tl.where(xmask, tmp45, 0)
    tmp48 = tl.sum(tmp47, 1)[:, None]
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp49 * tmp51
    tmp54 = tmp52 / tmp53
    tmp55 = tl.where(tmp20, tmp48, tmp21)
    tmp56 = tmp54 + tmp55
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tmp58 * tmp26
    tmp60 = tmp56 * tmp59
    tmp61 = tl.broadcast_to(tmp60, [XBLOCK, RBLOCK])
    tmp63 = tl.where(xmask, tmp61, 0)
    tmp64 = tl.sum(tmp63, 1)[:, None]
    tmp67 = tmp66.to(tl.float32)
    tmp68 = tmp65 * tmp67
    tmp70 = tmp68 / tmp69
    tmp71 = tl.where(tmp20, tmp64, tmp21)
    tmp72 = tmp70 + tmp71
    tmp74 = tmp73.to(tl.float32)
    tmp75 = tmp74 * tmp26
    tmp76 = tmp72 * tmp75
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK, RBLOCK])
    tmp79 = tl.where(xmask, tmp77, 0)
    tmp80 = tl.sum(tmp79, 1)[:, None]
    tmp83 = tmp82.to(tl.float32)
    tmp84 = tmp81 * tmp83
    tmp86 = tmp84 / tmp85
    tmp87 = tl.where(tmp20, tmp80, tmp21)
    tmp88 = tmp86 + tmp87
    tmp90 = tmp89.to(tl.float32)
    tmp91 = tmp90 * tmp26
    tmp92 = tmp88 * tmp91
    tmp93 = tl.broadcast_to(tmp92, [XBLOCK, RBLOCK])
    tmp95 = tl.where(xmask, tmp93, 0)
    tmp96 = tl.sum(tmp95, 1)[:, None]
    tmp99 = tmp98.to(tl.float32)
    tmp100 = tmp97 * tmp99
    tmp102 = tmp100 / tmp101
    tmp103 = tl.where(tmp20, tmp96, tmp21)
    tmp104 = tmp102 + tmp103
    tmp106 = tmp105.to(tl.float32)
    tmp107 = tmp106 * tmp26
    tmp108 = tmp104 * tmp107
    tmp109 = tl.broadcast_to(tmp108, [XBLOCK, RBLOCK])
    tmp111 = tl.where(xmask, tmp109, 0)
    tmp112 = tl.sum(tmp111, 1)[:, None]
    tmp115 = tmp114.to(tl.float32)
    tmp116 = tmp113 * tmp115
    tmp118 = tmp116 / tmp117
    tmp119 = tl.where(tmp20, tmp112, tmp21)
    tmp120 = tmp118 + tmp119
    tmp122 = tmp121.to(tl.float32)
    tmp123 = tmp122 * tmp26
    tmp124 = tmp120 * tmp123
    tmp125 = tl.broadcast_to(tmp124, [XBLOCK, RBLOCK])
    tmp127 = tl.where(xmask, tmp125, 0)
    tmp128 = tl.sum(tmp127, 1)[:, None]
    tmp131 = tmp130.to(tl.float32)
    tmp132 = tmp129 * tmp131
    tmp134 = tmp132 / tmp133
    tmp135 = tl.where(tmp20, tmp128, tmp21)
    tmp136 = tmp134 + tmp135
    tmp138 = tmp137.to(tl.float32)
    tmp139 = tmp138 * tmp26
    tmp140 = tmp136 * tmp139
    tmp141 = tl.broadcast_to(tmp140, [XBLOCK, RBLOCK])
    tmp143 = tl.where(xmask, tmp141, 0)
    tmp144 = tl.sum(tmp143, 1)[:, None]
    tmp147 = tmp146.to(tl.float32)
    tmp148 = tmp145 * tmp147
    tmp150 = tmp148 / tmp149
    tmp151 = tl.where(tmp20, tmp144, tmp21)
    tmp152 = tmp150 + tmp151
    tmp154 = tmp153.to(tl.float32)
    tmp155 = tmp154 * tmp26
    tmp156 = tmp152 * tmp155
    tmp157 = tl.broadcast_to(tmp156, [XBLOCK, RBLOCK])
    tmp159 = tl.where(xmask, tmp157, 0)
    tmp160 = tl.sum(tmp159, 1)[:, None]
    tmp163 = tmp162.to(tl.float32)
    tmp164 = tmp161 * tmp163
    tmp166 = tmp164 / tmp165
    tmp167 = tl.where(tmp20, tmp160, tmp21)
    tmp168 = tmp166 + tmp167
    tmp170 = tmp169.to(tl.float32)
    tmp171 = tmp170 * tmp26
    tmp172 = tmp168 * tmp171
    tmp173 = tl.broadcast_to(tmp172, [XBLOCK, RBLOCK])
    tmp175 = tl.where(xmask, tmp173, 0)
    tmp176 = tl.sum(tmp175, 1)[:, None]
    tmp179 = tmp178.to(tl.float32)
    tmp180 = tmp177 * tmp179
    tmp182 = tmp180 / tmp181
    tmp183 = tl.where(tmp20, tmp176, tmp21)
    tmp184 = tmp182 + tmp183
    tmp186 = tmp185.to(tl.float32)
    tmp187 = tmp186 * tmp26
    tmp188 = tmp184 * tmp187
    tmp189 = tl.broadcast_to(tmp188, [XBLOCK, RBLOCK])
    tmp191 = tl.where(xmask, tmp189, 0)
    tmp192 = tl.sum(tmp191, 1)[:, None]
    tmp195 = tmp194.to(tl.float32)
    tmp196 = tmp193 * tmp195
    tmp198 = tmp196 / tmp197
    tmp199 = tl.where(tmp20, tmp192, tmp21)
    tmp200 = tmp198 + tmp199
    tmp202 = tmp201.to(tl.float32)
    tmp203 = tmp202 * tmp26
    tmp204 = tmp200 * tmp203
    tmp205 = tl.broadcast_to(tmp204, [XBLOCK, RBLOCK])
    tmp207 = tl.where(xmask, tmp205, 0)
    tmp208 = tl.sum(tmp207, 1)[:, None]
    tmp211 = tmp210.to(tl.float32)
    tmp212 = tmp209 * tmp211
    tmp214 = tmp212 / tmp213
    tmp215 = tl.where(tmp20, tmp208, tmp21)
    tmp216 = tmp214 + tmp215
    tmp218 = tmp217.to(tl.float32)
    tmp219 = tmp218 * tmp26
    tmp220 = tmp216 * tmp219
    tmp221 = tl.broadcast_to(tmp220, [XBLOCK, RBLOCK])
    tmp223 = tl.where(xmask, tmp221, 0)
    tmp224 = tl.sum(tmp223, 1)[:, None]
    tmp227 = tmp226.to(tl.float32)
    tmp228 = tmp225 * tmp227
    tmp230 = tmp228 / tmp229
    tmp231 = tl.where(tmp20, tmp224, tmp21)
    tmp232 = tmp230 + tmp231
    tmp234 = tmp233.to(tl.float32)
    tmp235 = tmp234 * tmp26
    tmp236 = tmp232 * tmp235
    tmp237 = tl.broadcast_to(tmp236, [XBLOCK, RBLOCK])
    tmp239 = tl.where(xmask, tmp237, 0)
    tmp240 = tl.sum(tmp239, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
    tl.store(out_ptr1 + (x3), tmp32, xmask)
    tl.store(out_ptr2 + (x3), tmp48, xmask)
    tl.store(out_ptr3 + (x3), tmp64, xmask)
    tl.store(out_ptr4 + (x3), tmp80, xmask)
    tl.store(out_ptr5 + (x3), tmp96, xmask)
    tl.store(out_ptr6 + (x3), tmp112, xmask)
    tl.store(out_ptr7 + (x3), tmp128, xmask)
    tl.store(out_ptr8 + (x3), tmp144, xmask)
    tl.store(out_ptr9 + (x3), tmp160, xmask)
    tl.store(out_ptr10 + (x3), tmp176, xmask)
    tl.store(out_ptr11 + (x3), tmp192, xmask)
    tl.store(out_ptr12 + (x3), tmp208, xmask)
    tl.store(out_ptr13 + (x3), tmp224, xmask)
    tl.store(out_ptr14 + (x3), tmp240, xmask)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/hq/chqkvfi27f5sj4vbh6ci35bvvu23mcg3gzxl4ocveumgc5cn5x3e.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward, aten.add, aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_18 : [num_users=48] = call_function[target=torch.ops.aten.full.default](args = ([32, 16, 64, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_20 : [num_users=45] = call_function[target=torch.ops.aten.full.default](args = ([32, 64, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default_62 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %select_326, 1, 15), kwargs = {})
#   %select_scatter_default_63 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_30, 1, -1), kwargs = {})
#   %add_116 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_325, %select_scatter_default_63), kwargs = {})
#   %select_scatter_default_64 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_116, 1, 14), kwargs = {})
#   %add_117 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default_62, %select_scatter_default_64), kwargs = {})
#   %select_scatter_default_65 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_31, 1, -1), kwargs = {})
#   %add_118 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_324, %select_scatter_default_65), kwargs = {})
#   %select_scatter_default_66 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_118, 1, 13), kwargs = {})
#   %add_119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_117, %select_scatter_default_66), kwargs = {})
#   %select_scatter_default_67 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_32, 1, -1), kwargs = {})
#   %add_120 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_323, %select_scatter_default_67), kwargs = {})
#   %select_scatter_default_68 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_120, 1, 12), kwargs = {})
#   %add_121 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_119, %select_scatter_default_68), kwargs = {})
#   %select_scatter_default_69 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_33, 1, -1), kwargs = {})
#   %add_122 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_322, %select_scatter_default_69), kwargs = {})
#   %select_scatter_default_70 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_122, 1, 11), kwargs = {})
#   %add_123 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_121, %select_scatter_default_70), kwargs = {})
#   %select_scatter_default_71 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_34, 1, -1), kwargs = {})
#   %add_124 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_321, %select_scatter_default_71), kwargs = {})
#   %select_scatter_default_72 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_124, 1, 10), kwargs = {})
#   %add_125 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_123, %select_scatter_default_72), kwargs = {})
#   %select_scatter_default_73 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_35, 1, -1), kwargs = {})
#   %add_126 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_320, %select_scatter_default_73), kwargs = {})
#   %select_scatter_default_74 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_126, 1, 9), kwargs = {})
#   %add_127 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_125, %select_scatter_default_74), kwargs = {})
#   %select_scatter_default_75 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_36, 1, -1), kwargs = {})
#   %add_128 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_319, %select_scatter_default_75), kwargs = {})
#   %select_scatter_default_76 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_128, 1, 8), kwargs = {})
#   %add_129 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_127, %select_scatter_default_76), kwargs = {})
#   %select_scatter_default_77 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_37, 1, -1), kwargs = {})
#   %add_130 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_318, %select_scatter_default_77), kwargs = {})
#   %select_scatter_default_78 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_130, 1, 7), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_129, %select_scatter_default_78), kwargs = {})
#   %select_scatter_default_79 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_38, 1, -1), kwargs = {})
#   %add_132 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_317, %select_scatter_default_79), kwargs = {})
#   %select_scatter_default_80 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_132, 1, 6), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_131, %select_scatter_default_80), kwargs = {})
#   %select_scatter_default_81 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_39, 1, -1), kwargs = {})
#   %add_134 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_316, %select_scatter_default_81), kwargs = {})
#   %select_scatter_default_82 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_134, 1, 5), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_133, %select_scatter_default_82), kwargs = {})
#   %select_scatter_default_83 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_40, 1, -1), kwargs = {})
#   %add_136 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_315, %select_scatter_default_83), kwargs = {})
#   %select_scatter_default_84 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_136, 1, 4), kwargs = {})
#   %add_137 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_135, %select_scatter_default_84), kwargs = {})
#   %select_scatter_default_85 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_41, 1, -1), kwargs = {})
#   %add_138 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_314, %select_scatter_default_85), kwargs = {})
#   %select_scatter_default_86 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_138, 1, 3), kwargs = {})
#   %add_139 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_137, %select_scatter_default_86), kwargs = {})
#   %select_scatter_default_87 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_42, 1, -1), kwargs = {})
#   %add_140 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_313, %select_scatter_default_87), kwargs = {})
#   %select_scatter_default_88 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_140, 1, 2), kwargs = {})
#   %add_141 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_139, %select_scatter_default_88), kwargs = {})
#   %select_scatter_default_89 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_43, 1, -1), kwargs = {})
#   %add_142 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_312, %select_scatter_default_89), kwargs = {})
#   %select_scatter_default_90 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %add_142, 1, 1), kwargs = {})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_141, %select_scatter_default_90), kwargs = {})
#   %convert_element_type_73 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_134, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_add_select_backward_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*bf16', 19: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_select_backward_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 63, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048) % 16
    x0 = xindex % 32
    x1 = (xindex // 32) % 64
    x3 = (xindex // 32768)
    x6 = xindex
    x4 = (xindex // 2048)
    tmp3 = tl.load(in_ptr0 + (92160 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp7 = tl.load(in_ptr2 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (86016 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp17 = tl.load(in_ptr2 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (79872 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr1 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp33 = tl.load(in_ptr2 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr4 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (73728 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr1 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp46 = tl.load(in_ptr2 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr5 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (67584 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr1 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp59 = tl.load(in_ptr2 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr6 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr0 + (61440 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr1 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp72 = tl.load(in_ptr2 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr7 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr0 + (55296 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr1 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp85 = tl.load(in_ptr2 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr8 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr0 + (49152 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr1 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp98 = tl.load(in_ptr2 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp100 = tl.load(in_ptr9 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr0 + (43008 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr1 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp111 = tl.load(in_ptr2 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp113 = tl.load(in_ptr10 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp120 = tl.load(in_ptr0 + (36864 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp121 = tl.load(in_ptr1 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp124 = tl.load(in_ptr2 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp126 = tl.load(in_ptr11 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp133 = tl.load(in_ptr0 + (30720 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp134 = tl.load(in_ptr1 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp137 = tl.load(in_ptr2 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp139 = tl.load(in_ptr12 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp146 = tl.load(in_ptr0 + (24576 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp147 = tl.load(in_ptr1 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp150 = tl.load(in_ptr2 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp152 = tl.load(in_ptr13 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp159 = tl.load(in_ptr0 + (18432 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp160 = tl.load(in_ptr1 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp163 = tl.load(in_ptr2 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp165 = tl.load(in_ptr14 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp172 = tl.load(in_ptr0 + (12288 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp173 = tl.load(in_ptr1 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp176 = tl.load(in_ptr2 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp178 = tl.load(in_ptr15 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp185 = tl.load(in_ptr0 + (6144 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp186 = tl.load(in_ptr1 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp189 = tl.load(in_ptr2 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp191 = tl.load(in_ptr16 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp199 = tl.load(in_ptr0 + (x0 + (96*x1) + (98304*(x4 // 16))), None)
    tmp200 = tl.load(in_ptr1 + (x1 + (1024*(x4 // 16))), None, eviction_policy='evict_last').to(tl.int1)
    tmp203 = tl.load(in_ptr2 + (x1 + (1024*(x4 // 16))), None, eviction_policy='evict_last')
    tmp205 = tl.load(in_ptr17 + (x0 + (32*(x4 // 16))), None, eviction_policy='evict_last')
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
    tmp196 = (x6 // 2048) % 16
    tmp197 = tl.full([1], 0, tl.int32)
    tmp198 = tmp196 == tmp197
    tmp201 = tmp200.to(tl.float32)
    tmp202 = tmp199 * tmp201
    tmp204 = tmp202 / tmp203
    tmp206 = tl.where(tmp21, tmp205, tmp9)
    tmp207 = tmp204 + tmp206
    tmp208 = tl.where(tmp198, tmp207, tmp9)
    tmp209 = tmp195 + tmp208
    tmp210 = tmp209.to(tl.float32)
    tl.store(out_ptr0 + (x6), tmp210, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/r6/cr6zs4wnrvxorclpffoci6es5oswlcxsolddiskjwaz7z427gwuy.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_74 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_3, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*bf16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 64
    x2 = (xindex // 4096)
    x3 = (xindex // 64)
    x4 = xindex
    tmp9 = tl.load(in_ptr0 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp0 = x0 + ((-1)*x1)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.0
    tmp3 = tmp1 >= tmp2
    tmp4 = 0.9576032757759094
    tmp5 = ((0.0) * ((0.0) >= (x0 + ((-1)*x1))) + (x0 + ((-1)*x1)) * ((x0 + ((-1)*x1)) > (0.0)))
    tmp6 = tmp5.to(tl.float32)
    tmp7 = libdevice.pow(tmp4, tmp6)
    tmp8 = tl.where(tmp3, tmp7, tmp2)
    tmp11 = tmp9 == tmp10
    tmp13 = tmp11 & tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp8 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp16, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/cn/ccn6klxwznm3fq6d2p4mb6rrfdzkq6rvkoqohde6f5hqgkd7u4nx.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_79 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_4, torch.bfloat16), kwargs = {})
#   %convert_element_type_83 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_137, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 96
    x1 = (xindex // 96)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 64, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-64) + x0 + (32*x1)), tmp2, other=0.0).to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.full(tmp4.shape, 0.0, tmp4.dtype)
    tmp6 = tl.where(tmp2, tmp4, tmp5)
    tmp7 = 0.0
    tmp8 = tl.where(tmp2, tmp6, tmp7)
    tmp9 = tl.full([1], 32, tl.int64)
    tmp10 = tmp0 >= tmp9
    tmp11 = tmp0 < tmp1
    tmp12 = tmp10 & tmp11
    tmp13 = tl.load(in_ptr1 + ((-32) + x0 + (32*x1)), tmp12, other=0.0).to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp12, tmp14, tmp15)
    tmp17 = tl.where(tmp12, tmp16, tmp7)
    tmp18 = tmp8 + tmp17
    tmp19 = tmp0 < tmp9
    tmp20 = tl.load(in_ptr2 + (x0 + (32*x1)), tmp19, other=0.0).to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp19, tmp21, tmp22)
    tmp24 = tl.where(tmp19, tmp23, tmp7)
    tmp25 = tmp18 + tmp24
    tmp26 = tmp25.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp26, None)
    tl.store(out_ptr1 + (x2), tmp26, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ra/cra2h3wun7k2j5fimrntvoqdxm3rxvzfadgaqw4imc6uzqz4of2z.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.div, aten.pow, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_78 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.bfloat16), kwargs = {})
#   %mul_153 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_138, %primals_2), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_153, %primals_1), kwargs = {})
#   %mul_155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_153, %rsqrt), kwargs = {})
#   %sum_47 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_154, [2], True), kwargs = {dtype: torch.float32})
#   %div_9 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_6, 384), kwargs = {})
#   %pow_12 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_1, 1.0), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_12, 2.0), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_9, %mul_158), kwargs = {})
#   %add_147 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_155, %mul_159), kwargs = {})
triton_per_fused__to_copy_add_div_mul_pow_sum_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_pow_sum_6', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 32768
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1 + (384*x0)), rmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp11 = tmp4 * tmp10
    tmp12 = tmp11 * tmp2
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp3 * tmp10
    tmp15 = -0.5
    tmp16 = tmp9 * tmp15
    tmp17 = tmp10 * tmp10
    tmp18 = tmp17 * tmp10
    tmp19 = tmp16 * tmp18
    tmp20 = 0.0026041666666666665
    tmp21 = tmp19 * tmp20
    tmp22 = 2.0
    tmp23 = tmp4 * tmp22
    tmp24 = tmp21 * tmp23
    tmp25 = tmp14 + tmp24
    tl.store(out_ptr1 + (r1 + (384*x0)), tmp13, rmask)
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp25, rmask)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/7b/c7boawfuhftep5crza3uu2vervwueuri6lhkan57rro3d2tpkpb4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_87 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_5, torch.float32), kwargs = {})
triton_poi_fused__to_copy_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[512, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 96
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (384*x1)), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x1 + (96*y0)), tmp1, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/o6/co6ut6fsbqlcdnmxo32ykwtdwh7xax6w7i7ho7op3v2gfu27ljha.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_87 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_5, torch.float32), kwargs = {})
#   %permute_7 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_87, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[128, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (96*x1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (384*y0)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/uj/cujcauayveyqeta5casa5zym5fvwt55xh5jc5o7xbeexnpav4wp6.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   y => mul
# Graph fragment:
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %rsqrt), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_138, %mul), kwargs = {})
#   %sum_46 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_152, [0, 1], True), kwargs = {dtype: torch.float32})
triton_red_fused_mul_sum_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    x1 = (xindex // 384)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp4 = tmp2 * tmp3
        tmp5 = tmp1 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/dy/cdyvgff7ufrccqzkou3qxk2llmgjdrbdiln55ad35if4i2l5leso.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   y => mul
# Graph fragment:
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %rsqrt), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_138, %mul), kwargs = {})
#   %sum_46 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_152, [0, 1], True), kwargs = {dtype: torch.float32})
triton_red_fused_mul_sum_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, rsqrt, unsqueeze_2, view_10, bitwise_and_2, bitwise_and_3, bitwise_and_4, bitwise_and_5, bitwise_and_6, bitwise_and_7, bitwise_and_8, bitwise_and_9, bitwise_and_10, bitwise_and_11, bitwise_and_12, bitwise_and_13, bitwise_and_14, bitwise_and_15, mul_33, unsqueeze_35, unsqueeze_36, mul_39, mul_41, mul_43, mul_45, mul_47, mul_49, mul_51, mul_53, mul_55, mul_57, mul_59, mul_61, mul_63, mul_65, mul_67, unsqueeze_72, mul_73, mul_75, mul_77, mul_79, mul_81, mul_83, mul_85, mul_87, mul_89, mul_91, mul_93, mul_95, mul_97, mul_99, mul_101, unsqueeze_109, convert_element_type_64, convert_element_type_69, convert_element_type_82, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, 1024, 384), (393216, 384, 1))
    assert_size_stride(primals_2, (384, ), (1, ))
    assert_size_stride(primals_4, (32, 1024), (1024, 1))
    assert_size_stride(rsqrt, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(unsqueeze_2, (32, 16, 64, 1), (1024, 64, 1, 1))
    assert_size_stride(view_10, (1, 64, 1), (64, 1, 1))
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
    assert_size_stride(mul_33, (32, 64, 1), (64, 1, 1))
    assert_size_stride(unsqueeze_35, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(unsqueeze_36, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(mul_39, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_41, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_43, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_45, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_47, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_49, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_51, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_53, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_55, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_57, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_59, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_61, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_63, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_65, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_67, (32, 64, 1), (64, 1, 1))
    assert_size_stride(unsqueeze_72, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(mul_73, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_75, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_77, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_79, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_81, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_83, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_85, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_87, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_89, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_91, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_93, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_95, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_97, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_99, (32, 64, 1), (64, 1, 1))
    assert_size_stride(mul_101, (32, 64, 1), (64, 1, 1))
    assert_size_stride(unsqueeze_109, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(convert_element_type_64, (512, 64, 64), (4096, 1, 64))
    assert_size_stride(convert_element_type_69, (512, 64, 64), (4096, 1, 64))
    assert_size_stride(convert_element_type_82, (96, 384), (384, 1))
    assert_size_stride(tangents_1, (32, 1024, 96), (98304, 96, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf1 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf2 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf4 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf5 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf7 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf8 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf10 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf11 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf13 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf14 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf16 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf17 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf19 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf20 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf24 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf25 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf26 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf28 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf29 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf31 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf32 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf34 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf35 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf37 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf38 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf40 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf41 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf43 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf44 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.select_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mul_select_backward_sum_0.run(tangents_1, unsqueeze_36, unsqueeze_109, mul_101, mul_99, mul_97, mul_95, mul_93, mul_91, mul_89, mul_87, mul_85, mul_83, mul_81, mul_79, mul_77, mul_75, mul_73, unsqueeze_72, mul_67, mul_65, mul_63, mul_61, mul_59, mul_57, mul_55, mul_53, mul_51, mul_49, mul_47, mul_45, mul_43, mul_41, mul_39, buf0, buf1, buf2, buf4, buf5, buf7, buf8, buf10, buf11, buf13, buf14, buf16, buf17, buf19, buf20, buf24, buf25, buf26, buf28, buf29, buf31, buf32, buf34, buf35, buf37, buf38, buf40, buf41, buf43, buf44, 1024, 64, grid=grid(1024), stream=stream0)
        del mul_101
        del mul_39
        del mul_41
        del mul_43
        del mul_45
        del mul_47
        del mul_49
        del mul_51
        del mul_53
        del mul_55
        del mul_57
        del mul_59
        del mul_61
        del mul_63
        del mul_65
        del mul_67
        del mul_73
        del mul_75
        del mul_77
        del mul_79
        del mul_81
        del mul_83
        del mul_85
        del mul_87
        del mul_89
        del mul_91
        del mul_93
        del mul_95
        del mul_97
        del mul_99
        buf22 = empty_strided_cuda((512, 64, 32), (2048, 32, 1), torch.bfloat16)
        buf46 = empty_strided_cuda((512, 64, 32), (2048, 32, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward, aten.add, aten._to_copy]
        triton_poi_fused__to_copy_add_select_backward_1.run(tangents_1, unsqueeze_36, unsqueeze_109, buf0, buf1, buf2, buf4, buf5, buf7, buf8, buf10, buf11, buf13, buf14, buf16, buf17, buf19, buf20, unsqueeze_72, buf24, buf25, buf26, buf28, buf29, buf31, buf32, buf34, buf35, buf37, buf38, buf40, buf41, buf43, buf44, buf22, buf46, 1048576, grid=grid(1048576), stream=stream0)
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
        del buf24
        del buf25
        del buf26
        del buf28
        del unsqueeze_109
        del unsqueeze_72
        buf23 = empty_strided_cuda((512, 64, 32), (2048, 32, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
        extern_kernels.bmm(convert_element_type_64, buf22, out=buf23)
        del convert_element_type_64
        buf47 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
        extern_kernels.bmm(convert_element_type_69, buf46, out=buf47)
        del convert_element_type_69
        buf48 = buf8; del buf8  # reuse
        buf49 = buf7; del buf7  # reuse
        buf50 = buf5; del buf5  # reuse
        buf52 = buf44; del buf44  # reuse
        buf53 = buf43; del buf43  # reuse
        buf55 = buf41; del buf41  # reuse
        buf56 = buf40; del buf40  # reuse
        buf58 = buf4; del buf4  # reuse
        buf59 = buf38; del buf38  # reuse
        buf61 = buf37; del buf37  # reuse
        buf62 = buf35; del buf35  # reuse
        buf64 = buf34; del buf34  # reuse
        buf65 = buf32; del buf32  # reuse
        buf67 = buf31; del buf31  # reuse
        buf68 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [mul_31, mul_29, mul_27, mul_25, mul_23, mul_21, mul_19, mul_17, mul_15, mul_13, mul_11, mul_9, mul_7, mul_5], Original ATen: [aten.select_backward, aten.mul, aten.sum, aten.add]
        triton_per_fused_add_mul_select_backward_sum_2.run(tangents_1, unsqueeze_36, unsqueeze_35, mul_33, bitwise_and_15, view_10, bitwise_and_14, bitwise_and_13, bitwise_and_12, bitwise_and_11, bitwise_and_10, bitwise_and_9, bitwise_and_8, bitwise_and_7, bitwise_and_6, bitwise_and_5, bitwise_and_4, bitwise_and_3, bitwise_and_2, buf48, buf49, buf50, buf52, buf53, buf55, buf56, buf58, buf59, buf61, buf62, buf64, buf65, buf67, buf68, 1024, 64, grid=grid(1024), stream=stream0)
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
        del mul_33
        del view_10
        buf71 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward, aten.add, aten._to_copy]
        triton_poi_fused__to_copy_add_select_backward_3.run(tangents_1, unsqueeze_36, unsqueeze_35, buf48, buf49, buf50, buf52, buf53, buf55, buf56, buf58, buf59, buf61, buf62, buf64, buf65, buf67, buf68, buf71, 1048576, grid=grid(1048576), stream=stream0)
        del buf48
        del buf49
        del buf50
        del buf52
        del buf53
        del buf55
        del buf56
        del buf58
        del buf59
        del buf61
        del buf62
        del buf64
        del buf65
        del buf67
        del buf68
        del tangents_1
        del unsqueeze_35
        del unsqueeze_36
        buf70 = empty_strided_cuda((512, 64, 64), (4096, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(primals_4, unsqueeze_2, buf70, 2097152, grid=grid(2097152), stream=stream0)
        del primals_4
        del unsqueeze_2
        buf72 = empty_strided_cuda((512, 64, 32), (2048, 32, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.bmm]
        extern_kernels.bmm(buf70, buf71, out=buf72)
        del buf70
        del buf71
        buf73 = empty_strided_cuda((96, 32768), (1, 96), torch.bfloat16)
        buf76 = empty_strided_cuda((32768, 96), (96, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_5.run(buf23, buf47, buf72, buf73, buf76, 3145728, grid=grid(3145728), stream=stream0)
        del buf23
        del buf47
        del buf72
        buf77 = empty_strided_cuda((32768, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mm]
        extern_kernels.mm(buf76, convert_element_type_82, out=buf77)
        del buf76
        del convert_element_type_82
        buf74 = empty_strided_cuda((32768, 384), (384, 1), torch.bfloat16)
        buf83 = empty_strided_cuda((32, 1024, 384), (393216, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.div, aten.pow, aten.add]
        triton_per_fused__to_copy_add_div_mul_pow_sum_6.run(buf77, primals_2, primals_1, rsqrt, buf74, buf83, 32768, 384, grid=grid(32768), stream=stream0)
        del primals_2
        buf75 = empty_strided_cuda((96, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mm]
        extern_kernels.mm(buf73, buf74, out=buf75)
        del buf73
        del buf74
        buf78 = empty_strided_cuda((384, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7.run(buf75, buf78, 384, 96, grid=grid(384, 96), stream=stream0)
        del buf75
        buf79 = empty_strided_cuda((96, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_8.run(buf78, buf79, 96, 384, grid=grid(96, 384), stream=stream0)
        del buf78
        buf80 = empty_strided_cuda((1, 1, 384, 256), (98304, 98304, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_9.run(buf77, primals_1, rsqrt, buf80, 98304, 128, grid=grid(98304), stream=stream0)
        del buf77
        del primals_1
        del rsqrt
        buf81 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_10.run(buf80, buf81, 384, 256, grid=grid(384), stream=stream0)
        del buf80
    return (buf83, reinterpret_tensor(buf81, (384, ), (1, ), 0), buf79, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 1024, 384), (393216, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    rsqrt = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_2 = rand_strided((32, 16, 64, 1), (1024, 64, 1, 1), device='cuda:0', dtype=torch.bool)
    view_10 = rand_strided((1, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
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
    mul_33 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_35 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_36 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.bool)
    mul_39 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_41 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_43 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_45 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_47 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_49 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_51 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_53 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_55 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_57 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_59 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_61 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_63 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_65 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_67 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_72 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_73 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_75 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_77 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_79 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_81 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_83 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_85 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_87 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_89 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_91 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_93 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_95 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_97 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_99 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    mul_101 = rand_strided((32, 64, 1), (64, 1, 1), device='cuda:0', dtype=torch.float32)
    unsqueeze_109 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_64 = rand_strided((512, 64, 64), (4096, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_69 = rand_strided((512, 64, 64), (4096, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_82 = rand_strided((96, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((32, 1024, 96), (98304, 96, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_4, rsqrt, unsqueeze_2, view_10, bitwise_and_2, bitwise_and_3, bitwise_and_4, bitwise_and_5, bitwise_and_6, bitwise_and_7, bitwise_and_8, bitwise_and_9, bitwise_and_10, bitwise_and_11, bitwise_and_12, bitwise_and_13, bitwise_and_14, bitwise_and_15, mul_33, unsqueeze_35, unsqueeze_36, mul_39, mul_41, mul_43, mul_45, mul_47, mul_49, mul_51, mul_53, mul_55, mul_57, mul_59, mul_61, mul_63, mul_65, mul_67, unsqueeze_72, mul_73, mul_75, mul_77, mul_79, mul_81, mul_83, mul_85, mul_87, mul_89, mul_91, mul_93, mul_95, mul_97, mul_99, mul_101, unsqueeze_109, convert_element_type_64, convert_element_type_69, convert_element_type_82, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
