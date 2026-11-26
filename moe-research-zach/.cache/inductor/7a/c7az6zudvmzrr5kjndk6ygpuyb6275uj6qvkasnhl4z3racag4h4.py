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


# kernel path: /home/zapan4/moe-research/.cache/inductor/h4/ch4wwrlxidyokszfomtrdbmoyb4q4imll267fzuy3hjqkuz6hdzp.py
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
    tmp5 = tl.load(in_ptr2 + (960 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr0 + (86080 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp14 = tl.load(in_ptr1 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp18 = tl.load(in_ptr2 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr4 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr0 + (79936 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp33 = tl.load(in_ptr1 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp37 = tl.load(in_ptr2 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.load(in_ptr5 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.load(in_ptr0 + (73792 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp48 = tl.load(in_ptr1 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp52 = tl.load(in_ptr2 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr6 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.load(in_ptr0 + (67648 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp63 = tl.load(in_ptr1 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp67 = tl.load(in_ptr2 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp71 = tl.load(in_ptr7 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp77 = tl.load(in_ptr0 + (61504 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp78 = tl.load(in_ptr1 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp82 = tl.load(in_ptr2 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp86 = tl.load(in_ptr8 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp92 = tl.load(in_ptr0 + (55360 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp93 = tl.load(in_ptr1 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp97 = tl.load(in_ptr2 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp101 = tl.load(in_ptr9 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp107 = tl.load(in_ptr0 + (49216 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp108 = tl.load(in_ptr1 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp112 = tl.load(in_ptr2 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp116 = tl.load(in_ptr10 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp122 = tl.load(in_ptr0 + (43072 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp123 = tl.load(in_ptr1 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp127 = tl.load(in_ptr2 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp131 = tl.load(in_ptr11 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp137 = tl.load(in_ptr0 + (36928 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp138 = tl.load(in_ptr1 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp142 = tl.load(in_ptr2 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp146 = tl.load(in_ptr12 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp152 = tl.load(in_ptr0 + (30784 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp153 = tl.load(in_ptr1 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp157 = tl.load(in_ptr2 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp161 = tl.load(in_ptr13 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp167 = tl.load(in_ptr0 + (24640 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp168 = tl.load(in_ptr1 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp172 = tl.load(in_ptr2 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp176 = tl.load(in_ptr14 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp182 = tl.load(in_ptr0 + (18496 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp183 = tl.load(in_ptr1 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp187 = tl.load(in_ptr2 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp191 = tl.load(in_ptr15 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp197 = tl.load(in_ptr0 + (12352 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp198 = tl.load(in_ptr1 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp202 = tl.load(in_ptr2 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp206 = tl.load(in_ptr16 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp212 = tl.load(in_ptr0 + (6208 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp213 = tl.load(in_ptr1 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp217 = tl.load(in_ptr2 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp221 = tl.load(in_ptr17 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp227 = tl.load(in_ptr0 + (92192 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp229 = tl.load(in_ptr18 + (960 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp231 = tl.load(in_ptr19 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp237 = tl.load(in_ptr0 + (86048 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp239 = tl.load(in_ptr18 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp243 = tl.load(in_ptr20 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp249 = tl.load(in_ptr0 + (79904 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp251 = tl.load(in_ptr18 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp255 = tl.load(in_ptr21 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp261 = tl.load(in_ptr0 + (73760 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp263 = tl.load(in_ptr18 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp267 = tl.load(in_ptr22 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp273 = tl.load(in_ptr0 + (67616 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp275 = tl.load(in_ptr18 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp279 = tl.load(in_ptr23 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp285 = tl.load(in_ptr0 + (61472 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp287 = tl.load(in_ptr18 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp291 = tl.load(in_ptr24 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp297 = tl.load(in_ptr0 + (55328 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp299 = tl.load(in_ptr18 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp303 = tl.load(in_ptr25 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp309 = tl.load(in_ptr0 + (49184 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp311 = tl.load(in_ptr18 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp315 = tl.load(in_ptr26 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp321 = tl.load(in_ptr0 + (43040 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp323 = tl.load(in_ptr18 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp327 = tl.load(in_ptr27 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp333 = tl.load(in_ptr0 + (36896 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp335 = tl.load(in_ptr18 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp339 = tl.load(in_ptr28 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp345 = tl.load(in_ptr0 + (30752 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp347 = tl.load(in_ptr18 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp351 = tl.load(in_ptr29 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp357 = tl.load(in_ptr0 + (24608 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp359 = tl.load(in_ptr18 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp363 = tl.load(in_ptr30 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp369 = tl.load(in_ptr0 + (18464 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp371 = tl.load(in_ptr18 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp375 = tl.load(in_ptr31 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp381 = tl.load(in_ptr0 + (12320 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp383 = tl.load(in_ptr18 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp387 = tl.load(in_ptr32 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp393 = tl.load(in_ptr0 + (6176 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp395 = tl.load(in_ptr18 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp399 = tl.load(in_ptr33 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp0 * tmp3
    tmp6 = tmp4 / tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp13 * tmp16
    tmp19 = tmp17 / tmp18
    tmp20 = r2
    tmp21 = tl.full([1, 1], 63, tl.int32)
    tmp22 = tmp20 == tmp21
    tmp23 = 0.0
    tmp24 = tl.where(tmp22, tmp12, tmp23)
    tmp25 = tmp19 + tmp24
    tmp27 = tmp25 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp32 * tmp35
    tmp38 = tmp36 / tmp37
    tmp39 = tl.where(tmp22, tmp31, tmp23)
    tmp40 = tmp38 + tmp39
    tmp42 = tmp40 * tmp41
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, RBLOCK])
    tmp45 = tl.where(xmask, tmp43, 0)
    tmp46 = tl.sum(tmp45, 1)[:, None]
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp47 * tmp50
    tmp53 = tmp51 / tmp52
    tmp54 = tl.where(tmp22, tmp46, tmp23)
    tmp55 = tmp53 + tmp54
    tmp57 = tmp55 * tmp56
    tmp58 = tl.broadcast_to(tmp57, [XBLOCK, RBLOCK])
    tmp60 = tl.where(xmask, tmp58, 0)
    tmp61 = tl.sum(tmp60, 1)[:, None]
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tmp62 * tmp65
    tmp68 = tmp66 / tmp67
    tmp69 = tl.where(tmp22, tmp61, tmp23)
    tmp70 = tmp68 + tmp69
    tmp72 = tmp70 * tmp71
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp75 = tl.where(xmask, tmp73, 0)
    tmp76 = tl.sum(tmp75, 1)[:, None]
    tmp79 = tmp78.to(tl.float32)
    tmp80 = tmp79.to(tl.float32)
    tmp81 = tmp77 * tmp80
    tmp83 = tmp81 / tmp82
    tmp84 = tl.where(tmp22, tmp76, tmp23)
    tmp85 = tmp83 + tmp84
    tmp87 = tmp85 * tmp86
    tmp88 = tl.broadcast_to(tmp87, [XBLOCK, RBLOCK])
    tmp90 = tl.where(xmask, tmp88, 0)
    tmp91 = tl.sum(tmp90, 1)[:, None]
    tmp94 = tmp93.to(tl.float32)
    tmp95 = tmp94.to(tl.float32)
    tmp96 = tmp92 * tmp95
    tmp98 = tmp96 / tmp97
    tmp99 = tl.where(tmp22, tmp91, tmp23)
    tmp100 = tmp98 + tmp99
    tmp102 = tmp100 * tmp101
    tmp103 = tl.broadcast_to(tmp102, [XBLOCK, RBLOCK])
    tmp105 = tl.where(xmask, tmp103, 0)
    tmp106 = tl.sum(tmp105, 1)[:, None]
    tmp109 = tmp108.to(tl.float32)
    tmp110 = tmp109.to(tl.float32)
    tmp111 = tmp107 * tmp110
    tmp113 = tmp111 / tmp112
    tmp114 = tl.where(tmp22, tmp106, tmp23)
    tmp115 = tmp113 + tmp114
    tmp117 = tmp115 * tmp116
    tmp118 = tl.broadcast_to(tmp117, [XBLOCK, RBLOCK])
    tmp120 = tl.where(xmask, tmp118, 0)
    tmp121 = tl.sum(tmp120, 1)[:, None]
    tmp124 = tmp123.to(tl.float32)
    tmp125 = tmp124.to(tl.float32)
    tmp126 = tmp122 * tmp125
    tmp128 = tmp126 / tmp127
    tmp129 = tl.where(tmp22, tmp121, tmp23)
    tmp130 = tmp128 + tmp129
    tmp132 = tmp130 * tmp131
    tmp133 = tl.broadcast_to(tmp132, [XBLOCK, RBLOCK])
    tmp135 = tl.where(xmask, tmp133, 0)
    tmp136 = tl.sum(tmp135, 1)[:, None]
    tmp139 = tmp138.to(tl.float32)
    tmp140 = tmp139.to(tl.float32)
    tmp141 = tmp137 * tmp140
    tmp143 = tmp141 / tmp142
    tmp144 = tl.where(tmp22, tmp136, tmp23)
    tmp145 = tmp143 + tmp144
    tmp147 = tmp145 * tmp146
    tmp148 = tl.broadcast_to(tmp147, [XBLOCK, RBLOCK])
    tmp150 = tl.where(xmask, tmp148, 0)
    tmp151 = tl.sum(tmp150, 1)[:, None]
    tmp154 = tmp153.to(tl.float32)
    tmp155 = tmp154.to(tl.float32)
    tmp156 = tmp152 * tmp155
    tmp158 = tmp156 / tmp157
    tmp159 = tl.where(tmp22, tmp151, tmp23)
    tmp160 = tmp158 + tmp159
    tmp162 = tmp160 * tmp161
    tmp163 = tl.broadcast_to(tmp162, [XBLOCK, RBLOCK])
    tmp165 = tl.where(xmask, tmp163, 0)
    tmp166 = tl.sum(tmp165, 1)[:, None]
    tmp169 = tmp168.to(tl.float32)
    tmp170 = tmp169.to(tl.float32)
    tmp171 = tmp167 * tmp170
    tmp173 = tmp171 / tmp172
    tmp174 = tl.where(tmp22, tmp166, tmp23)
    tmp175 = tmp173 + tmp174
    tmp177 = tmp175 * tmp176
    tmp178 = tl.broadcast_to(tmp177, [XBLOCK, RBLOCK])
    tmp180 = tl.where(xmask, tmp178, 0)
    tmp181 = tl.sum(tmp180, 1)[:, None]
    tmp184 = tmp183.to(tl.float32)
    tmp185 = tmp184.to(tl.float32)
    tmp186 = tmp182 * tmp185
    tmp188 = tmp186 / tmp187
    tmp189 = tl.where(tmp22, tmp181, tmp23)
    tmp190 = tmp188 + tmp189
    tmp192 = tmp190 * tmp191
    tmp193 = tl.broadcast_to(tmp192, [XBLOCK, RBLOCK])
    tmp195 = tl.where(xmask, tmp193, 0)
    tmp196 = tl.sum(tmp195, 1)[:, None]
    tmp199 = tmp198.to(tl.float32)
    tmp200 = tmp199.to(tl.float32)
    tmp201 = tmp197 * tmp200
    tmp203 = tmp201 / tmp202
    tmp204 = tl.where(tmp22, tmp196, tmp23)
    tmp205 = tmp203 + tmp204
    tmp207 = tmp205 * tmp206
    tmp208 = tl.broadcast_to(tmp207, [XBLOCK, RBLOCK])
    tmp210 = tl.where(xmask, tmp208, 0)
    tmp211 = tl.sum(tmp210, 1)[:, None]
    tmp214 = tmp213.to(tl.float32)
    tmp215 = tmp214.to(tl.float32)
    tmp216 = tmp212 * tmp215
    tmp218 = tmp216 / tmp217
    tmp219 = tl.where(tmp22, tmp211, tmp23)
    tmp220 = tmp218 + tmp219
    tmp222 = tmp220 * tmp221
    tmp223 = tl.broadcast_to(tmp222, [XBLOCK, RBLOCK])
    tmp225 = tl.where(xmask, tmp223, 0)
    tmp226 = tl.sum(tmp225, 1)[:, None]
    tmp228 = tmp227 * tmp3
    tmp230 = tmp228 / tmp229
    tmp232 = tmp230 * tmp231
    tmp233 = tl.broadcast_to(tmp232, [XBLOCK, RBLOCK])
    tmp235 = tl.where(xmask, tmp233, 0)
    tmp236 = tl.sum(tmp235, 1)[:, None]
    tmp238 = tmp237 * tmp16
    tmp240 = tmp238 / tmp239
    tmp241 = tl.where(tmp22, tmp236, tmp23)
    tmp242 = tmp240 + tmp241
    tmp244 = tmp242 * tmp243
    tmp245 = tl.broadcast_to(tmp244, [XBLOCK, RBLOCK])
    tmp247 = tl.where(xmask, tmp245, 0)
    tmp248 = tl.sum(tmp247, 1)[:, None]
    tmp250 = tmp249 * tmp35
    tmp252 = tmp250 / tmp251
    tmp253 = tl.where(tmp22, tmp248, tmp23)
    tmp254 = tmp252 + tmp253
    tmp256 = tmp254 * tmp255
    tmp257 = tl.broadcast_to(tmp256, [XBLOCK, RBLOCK])
    tmp259 = tl.where(xmask, tmp257, 0)
    tmp260 = tl.sum(tmp259, 1)[:, None]
    tmp262 = tmp261 * tmp50
    tmp264 = tmp262 / tmp263
    tmp265 = tl.where(tmp22, tmp260, tmp23)
    tmp266 = tmp264 + tmp265
    tmp268 = tmp266 * tmp267
    tmp269 = tl.broadcast_to(tmp268, [XBLOCK, RBLOCK])
    tmp271 = tl.where(xmask, tmp269, 0)
    tmp272 = tl.sum(tmp271, 1)[:, None]
    tmp274 = tmp273 * tmp65
    tmp276 = tmp274 / tmp275
    tmp277 = tl.where(tmp22, tmp272, tmp23)
    tmp278 = tmp276 + tmp277
    tmp280 = tmp278 * tmp279
    tmp281 = tl.broadcast_to(tmp280, [XBLOCK, RBLOCK])
    tmp283 = tl.where(xmask, tmp281, 0)
    tmp284 = tl.sum(tmp283, 1)[:, None]
    tmp286 = tmp285 * tmp80
    tmp288 = tmp286 / tmp287
    tmp289 = tl.where(tmp22, tmp284, tmp23)
    tmp290 = tmp288 + tmp289
    tmp292 = tmp290 * tmp291
    tmp293 = tl.broadcast_to(tmp292, [XBLOCK, RBLOCK])
    tmp295 = tl.where(xmask, tmp293, 0)
    tmp296 = tl.sum(tmp295, 1)[:, None]
    tmp298 = tmp297 * tmp95
    tmp300 = tmp298 / tmp299
    tmp301 = tl.where(tmp22, tmp296, tmp23)
    tmp302 = tmp300 + tmp301
    tmp304 = tmp302 * tmp303
    tmp305 = tl.broadcast_to(tmp304, [XBLOCK, RBLOCK])
    tmp307 = tl.where(xmask, tmp305, 0)
    tmp308 = tl.sum(tmp307, 1)[:, None]
    tmp310 = tmp309 * tmp110
    tmp312 = tmp310 / tmp311
    tmp313 = tl.where(tmp22, tmp308, tmp23)
    tmp314 = tmp312 + tmp313
    tmp316 = tmp314 * tmp315
    tmp317 = tl.broadcast_to(tmp316, [XBLOCK, RBLOCK])
    tmp319 = tl.where(xmask, tmp317, 0)
    tmp320 = tl.sum(tmp319, 1)[:, None]
    tmp322 = tmp321 * tmp125
    tmp324 = tmp322 / tmp323
    tmp325 = tl.where(tmp22, tmp320, tmp23)
    tmp326 = tmp324 + tmp325
    tmp328 = tmp326 * tmp327
    tmp329 = tl.broadcast_to(tmp328, [XBLOCK, RBLOCK])
    tmp331 = tl.where(xmask, tmp329, 0)
    tmp332 = tl.sum(tmp331, 1)[:, None]
    tmp334 = tmp333 * tmp140
    tmp336 = tmp334 / tmp335
    tmp337 = tl.where(tmp22, tmp332, tmp23)
    tmp338 = tmp336 + tmp337
    tmp340 = tmp338 * tmp339
    tmp341 = tl.broadcast_to(tmp340, [XBLOCK, RBLOCK])
    tmp343 = tl.where(xmask, tmp341, 0)
    tmp344 = tl.sum(tmp343, 1)[:, None]
    tmp346 = tmp345 * tmp155
    tmp348 = tmp346 / tmp347
    tmp349 = tl.where(tmp22, tmp344, tmp23)
    tmp350 = tmp348 + tmp349
    tmp352 = tmp350 * tmp351
    tmp353 = tl.broadcast_to(tmp352, [XBLOCK, RBLOCK])
    tmp355 = tl.where(xmask, tmp353, 0)
    tmp356 = tl.sum(tmp355, 1)[:, None]
    tmp358 = tmp357 * tmp170
    tmp360 = tmp358 / tmp359
    tmp361 = tl.where(tmp22, tmp356, tmp23)
    tmp362 = tmp360 + tmp361
    tmp364 = tmp362 * tmp363
    tmp365 = tl.broadcast_to(tmp364, [XBLOCK, RBLOCK])
    tmp367 = tl.where(xmask, tmp365, 0)
    tmp368 = tl.sum(tmp367, 1)[:, None]
    tmp370 = tmp369 * tmp185
    tmp372 = tmp370 / tmp371
    tmp373 = tl.where(tmp22, tmp368, tmp23)
    tmp374 = tmp372 + tmp373
    tmp376 = tmp374 * tmp375
    tmp377 = tl.broadcast_to(tmp376, [XBLOCK, RBLOCK])
    tmp379 = tl.where(xmask, tmp377, 0)
    tmp380 = tl.sum(tmp379, 1)[:, None]
    tmp382 = tmp381 * tmp200
    tmp384 = tmp382 / tmp383
    tmp385 = tl.where(tmp22, tmp380, tmp23)
    tmp386 = tmp384 + tmp385
    tmp388 = tmp386 * tmp387
    tmp389 = tl.broadcast_to(tmp388, [XBLOCK, RBLOCK])
    tmp391 = tl.where(xmask, tmp389, 0)
    tmp392 = tl.sum(tmp391, 1)[:, None]
    tmp394 = tmp393 * tmp215
    tmp396 = tmp394 / tmp395
    tmp397 = tl.where(tmp22, tmp392, tmp23)
    tmp398 = tmp396 + tmp397
    tmp400 = tmp398 * tmp399
    tmp401 = tl.broadcast_to(tmp400, [XBLOCK, RBLOCK])
    tmp403 = tl.where(xmask, tmp401, 0)
    tmp404 = tl.sum(tmp403, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tl.store(out_ptr1 + (x3), tmp31, xmask)
    tl.store(out_ptr2 + (x3), tmp46, xmask)
    tl.store(out_ptr3 + (x3), tmp61, xmask)
    tl.store(out_ptr4 + (x3), tmp76, xmask)
    tl.store(out_ptr5 + (x3), tmp91, xmask)
    tl.store(out_ptr6 + (x3), tmp106, xmask)
    tl.store(out_ptr7 + (x3), tmp121, xmask)
    tl.store(out_ptr8 + (x3), tmp136, xmask)
    tl.store(out_ptr9 + (x3), tmp151, xmask)
    tl.store(out_ptr10 + (x3), tmp166, xmask)
    tl.store(out_ptr11 + (x3), tmp181, xmask)
    tl.store(out_ptr12 + (x3), tmp196, xmask)
    tl.store(out_ptr13 + (x3), tmp211, xmask)
    tl.store(out_ptr14 + (x3), tmp226, xmask)
    tl.store(out_ptr15 + (x3), tmp236, xmask)
    tl.store(out_ptr16 + (x3), tmp248, xmask)
    tl.store(out_ptr17 + (x3), tmp260, xmask)
    tl.store(out_ptr18 + (x3), tmp272, xmask)
    tl.store(out_ptr19 + (x3), tmp284, xmask)
    tl.store(out_ptr20 + (x3), tmp296, xmask)
    tl.store(out_ptr21 + (x3), tmp308, xmask)
    tl.store(out_ptr22 + (x3), tmp320, xmask)
    tl.store(out_ptr23 + (x3), tmp332, xmask)
    tl.store(out_ptr24 + (x3), tmp344, xmask)
    tl.store(out_ptr25 + (x3), tmp356, xmask)
    tl.store(out_ptr26 + (x3), tmp368, xmask)
    tl.store(out_ptr27 + (x3), tmp380, xmask)
    tl.store(out_ptr28 + (x3), tmp392, xmask)
    tl.store(out_ptr29 + (x3), tmp404, xmask)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/wy/cwy3ob25cvqzc65kfjpf77x4pxvnwqghglzbig77n3wm7b4dgzy3.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.select_backward, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_100 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select_294, torch.bfloat16), kwargs = {})
#   %full_default_18 : [num_users=48] = call_function[target=torch.ops.aten.full.default](args = ([32, 16, 64, 32], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_100, 1, 15), kwargs = {})
#   %full_default_20 : [num_users=45] = call_function[target=torch.ops.aten.full.default](args = ([32, 64, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze, 1, -1), kwargs = {})
#   %add_55 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_293, %select_scatter_default_1), kwargs = {})
#   %convert_element_type_101 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_55, torch.bfloat16), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_101, 1, 14), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_2), kwargs = {})
#   %select_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_1, 1, -1), kwargs = {})
#   %add_57 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_292, %select_scatter_default_3), kwargs = {})
#   %convert_element_type_102 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_57, torch.bfloat16), kwargs = {})
#   %select_scatter_default_4 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_102, 1, 13), kwargs = {})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_56, %select_scatter_default_4), kwargs = {})
#   %select_scatter_default_5 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_2, 1, -1), kwargs = {})
#   %add_59 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_291, %select_scatter_default_5), kwargs = {})
#   %convert_element_type_103 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_59, torch.bfloat16), kwargs = {})
#   %select_scatter_default_6 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_103, 1, 12), kwargs = {})
#   %add_60 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_58, %select_scatter_default_6), kwargs = {})
#   %select_scatter_default_7 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_3, 1, -1), kwargs = {})
#   %add_61 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_290, %select_scatter_default_7), kwargs = {})
#   %convert_element_type_104 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_61, torch.bfloat16), kwargs = {})
#   %select_scatter_default_8 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_104, 1, 11), kwargs = {})
#   %add_62 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_60, %select_scatter_default_8), kwargs = {})
#   %select_scatter_default_9 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_4, 1, -1), kwargs = {})
#   %add_63 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_289, %select_scatter_default_9), kwargs = {})
#   %convert_element_type_105 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_63, torch.bfloat16), kwargs = {})
#   %select_scatter_default_10 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_105, 1, 10), kwargs = {})
#   %add_64 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_62, %select_scatter_default_10), kwargs = {})
#   %select_scatter_default_11 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_5, 1, -1), kwargs = {})
#   %add_65 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_288, %select_scatter_default_11), kwargs = {})
#   %convert_element_type_106 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_65, torch.bfloat16), kwargs = {})
#   %select_scatter_default_12 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_106, 1, 9), kwargs = {})
#   %add_66 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_64, %select_scatter_default_12), kwargs = {})
#   %select_scatter_default_13 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_6, 1, -1), kwargs = {})
#   %add_67 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_287, %select_scatter_default_13), kwargs = {})
#   %convert_element_type_107 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_67, torch.bfloat16), kwargs = {})
#   %select_scatter_default_14 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_107, 1, 8), kwargs = {})
#   %add_68 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_66, %select_scatter_default_14), kwargs = {})
#   %select_scatter_default_15 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_7, 1, -1), kwargs = {})
#   %add_69 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_286, %select_scatter_default_15), kwargs = {})
#   %convert_element_type_108 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_69, torch.bfloat16), kwargs = {})
#   %select_scatter_default_16 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_108, 1, 7), kwargs = {})
#   %add_70 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_68, %select_scatter_default_16), kwargs = {})
#   %select_scatter_default_17 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_8, 1, -1), kwargs = {})
#   %add_71 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_285, %select_scatter_default_17), kwargs = {})
#   %convert_element_type_109 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_71, torch.bfloat16), kwargs = {})
#   %select_scatter_default_18 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_109, 1, 6), kwargs = {})
#   %add_72 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_70, %select_scatter_default_18), kwargs = {})
#   %select_scatter_default_19 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_9, 1, -1), kwargs = {})
#   %add_73 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_284, %select_scatter_default_19), kwargs = {})
#   %convert_element_type_110 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.bfloat16), kwargs = {})
#   %select_scatter_default_20 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_110, 1, 5), kwargs = {})
#   %add_74 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_72, %select_scatter_default_20), kwargs = {})
#   %select_scatter_default_21 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_10, 1, -1), kwargs = {})
#   %add_75 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_283, %select_scatter_default_21), kwargs = {})
#   %convert_element_type_111 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_75, torch.bfloat16), kwargs = {})
#   %select_scatter_default_22 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_111, 1, 4), kwargs = {})
#   %add_76 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_74, %select_scatter_default_22), kwargs = {})
#   %select_scatter_default_23 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_11, 1, -1), kwargs = {})
#   %add_77 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_282, %select_scatter_default_23), kwargs = {})
#   %convert_element_type_112 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_77, torch.bfloat16), kwargs = {})
#   %select_scatter_default_24 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_112, 1, 3), kwargs = {})
#   %add_78 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_76, %select_scatter_default_24), kwargs = {})
#   %select_scatter_default_25 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_12, 1, -1), kwargs = {})
#   %add_79 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_281, %select_scatter_default_25), kwargs = {})
#   %convert_element_type_113 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_79, torch.bfloat16), kwargs = {})
#   %select_scatter_default_26 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_113, 1, 2), kwargs = {})
#   %add_80 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_78, %select_scatter_default_26), kwargs = {})
#   %select_scatter_default_27 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_13, 1, -1), kwargs = {})
#   %add_81 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_280, %select_scatter_default_27), kwargs = {})
#   %convert_element_type_114 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_81, torch.bfloat16), kwargs = {})
#   %select_scatter_default_28 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_114, 1, 1), kwargs = {})
#   %add_82 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_80, %select_scatter_default_28), kwargs = {})
#   %select_scatter_default_29 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_14, 1, -1), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_279, %select_scatter_default_29), kwargs = {})
#   %convert_element_type_115 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_83, torch.bfloat16), kwargs = {})
#   %select_scatter_default_30 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_115, 1, 0), kwargs = {})
#   %add_84 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_82, %select_scatter_default_30), kwargs = {})
#   %convert_element_type_118 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select_310, torch.bfloat16), kwargs = {})
#   %select_scatter_default_31 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_118, 1, 15), kwargs = {})
#   %select_scatter_default_32 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_15, 1, -1), kwargs = {})
#   %add_85 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_309, %select_scatter_default_32), kwargs = {})
#   %convert_element_type_119 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_85, torch.bfloat16), kwargs = {})
#   %select_scatter_default_33 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_119, 1, 14), kwargs = {})
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default_31, %select_scatter_default_33), kwargs = {})
#   %select_scatter_default_34 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_16, 1, -1), kwargs = {})
#   %add_87 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_308, %select_scatter_default_34), kwargs = {})
#   %convert_element_type_120 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_87, torch.bfloat16), kwargs = {})
#   %select_scatter_default_35 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_120, 1, 13), kwargs = {})
#   %add_88 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_86, %select_scatter_default_35), kwargs = {})
#   %select_scatter_default_36 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_17, 1, -1), kwargs = {})
#   %add_89 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_307, %select_scatter_default_36), kwargs = {})
#   %convert_element_type_121 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_89, torch.bfloat16), kwargs = {})
#   %select_scatter_default_37 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_121, 1, 12), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_88, %select_scatter_default_37), kwargs = {})
#   %select_scatter_default_38 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_18, 1, -1), kwargs = {})
#   %add_91 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_306, %select_scatter_default_38), kwargs = {})
#   %convert_element_type_122 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_91, torch.bfloat16), kwargs = {})
#   %select_scatter_default_39 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_122, 1, 11), kwargs = {})
#   %add_92 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_90, %select_scatter_default_39), kwargs = {})
#   %select_scatter_default_40 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_19, 1, -1), kwargs = {})
#   %add_93 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_305, %select_scatter_default_40), kwargs = {})
#   %convert_element_type_123 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_93, torch.bfloat16), kwargs = {})
#   %select_scatter_default_41 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_123, 1, 10), kwargs = {})
#   %add_94 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_92, %select_scatter_default_41), kwargs = {})
#   %select_scatter_default_42 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_20, 1, -1), kwargs = {})
#   %add_95 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_304, %select_scatter_default_42), kwargs = {})
#   %convert_element_type_124 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_95, torch.bfloat16), kwargs = {})
#   %select_scatter_default_43 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_124, 1, 9), kwargs = {})
#   %add_96 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_94, %select_scatter_default_43), kwargs = {})
#   %select_scatter_default_44 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_21, 1, -1), kwargs = {})
#   %add_97 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_303, %select_scatter_default_44), kwargs = {})
#   %convert_element_type_125 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_97, torch.bfloat16), kwargs = {})
#   %select_scatter_default_45 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_125, 1, 8), kwargs = {})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_96, %select_scatter_default_45), kwargs = {})
#   %select_scatter_default_46 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_22, 1, -1), kwargs = {})
#   %add_99 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_302, %select_scatter_default_46), kwargs = {})
#   %convert_element_type_126 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_99, torch.bfloat16), kwargs = {})
#   %select_scatter_default_47 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_126, 1, 7), kwargs = {})
#   %add_100 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_98, %select_scatter_default_47), kwargs = {})
#   %select_scatter_default_48 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_23, 1, -1), kwargs = {})
#   %add_101 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_301, %select_scatter_default_48), kwargs = {})
#   %convert_element_type_127 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_101, torch.bfloat16), kwargs = {})
#   %select_scatter_default_49 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_127, 1, 6), kwargs = {})
#   %add_102 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_100, %select_scatter_default_49), kwargs = {})
#   %select_scatter_default_50 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_24, 1, -1), kwargs = {})
#   %add_103 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_300, %select_scatter_default_50), kwargs = {})
#   %convert_element_type_128 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_103, torch.bfloat16), kwargs = {})
#   %select_scatter_default_51 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_128, 1, 5), kwargs = {})
#   %add_104 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_102, %select_scatter_default_51), kwargs = {})
#   %select_scatter_default_52 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_25, 1, -1), kwargs = {})
#   %add_105 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_299, %select_scatter_default_52), kwargs = {})
#   %convert_element_type_129 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_105, torch.bfloat16), kwargs = {})
#   %select_scatter_default_53 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_129, 1, 4), kwargs = {})
#   %add_106 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_104, %select_scatter_default_53), kwargs = {})
#   %select_scatter_default_54 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_26, 1, -1), kwargs = {})
#   %add_107 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_298, %select_scatter_default_54), kwargs = {})
#   %convert_element_type_130 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_107, torch.bfloat16), kwargs = {})
#   %select_scatter_default_55 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_130, 1, 3), kwargs = {})
#   %add_108 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_106, %select_scatter_default_55), kwargs = {})
#   %select_scatter_default_56 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_27, 1, -1), kwargs = {})
#   %add_109 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_297, %select_scatter_default_56), kwargs = {})
#   %convert_element_type_131 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_109, torch.bfloat16), kwargs = {})
#   %select_scatter_default_57 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_131, 1, 2), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_108, %select_scatter_default_57), kwargs = {})
#   %select_scatter_default_58 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_28, 1, -1), kwargs = {})
#   %add_111 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_296, %select_scatter_default_58), kwargs = {})
#   %convert_element_type_132 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_111, torch.bfloat16), kwargs = {})
#   %select_scatter_default_59 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_132, 1, 1), kwargs = {})
#   %add_112 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_110, %select_scatter_default_59), kwargs = {})
#   %select_scatter_default_60 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_29, 1, -1), kwargs = {})
#   %add_113 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_295, %select_scatter_default_60), kwargs = {})
#   %convert_element_type_133 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_113, torch.bfloat16), kwargs = {})
#   %select_scatter_default_61 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_133, 1, 0), kwargs = {})
#   %add_114 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_112, %select_scatter_default_61), kwargs = {})
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
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_select_backward_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'no_x_dim': False, 'num_load': 110, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048) % 16
    x0 = xindex % 32
    x1 = (xindex // 32) % 64
    x3 = (xindex // 32768)
    x5 = xindex
    tmp3 = tl.load(in_ptr0 + (92224 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (86080 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp20 = tl.load(in_ptr2 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (79936 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr1 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp38 = tl.load(in_ptr2 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr4 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr0 + (73792 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr1 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp53 = tl.load(in_ptr2 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr5 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr0 + (67648 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr1 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp68 = tl.load(in_ptr2 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr6 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr0 + (61504 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr1 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp83 = tl.load(in_ptr2 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr7 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr0 + (55360 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr1 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp98 = tl.load(in_ptr2 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp100 = tl.load(in_ptr8 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr0 + (49216 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp109 = tl.load(in_ptr1 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp113 = tl.load(in_ptr2 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp115 = tl.load(in_ptr9 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp123 = tl.load(in_ptr0 + (43072 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp124 = tl.load(in_ptr1 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp128 = tl.load(in_ptr2 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp130 = tl.load(in_ptr10 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp138 = tl.load(in_ptr0 + (36928 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp139 = tl.load(in_ptr1 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp143 = tl.load(in_ptr2 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp145 = tl.load(in_ptr11 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp153 = tl.load(in_ptr0 + (30784 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp154 = tl.load(in_ptr1 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp158 = tl.load(in_ptr2 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp160 = tl.load(in_ptr12 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp168 = tl.load(in_ptr0 + (24640 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp169 = tl.load(in_ptr1 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp173 = tl.load(in_ptr2 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp175 = tl.load(in_ptr13 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp183 = tl.load(in_ptr0 + (18496 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp184 = tl.load(in_ptr1 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp188 = tl.load(in_ptr2 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp190 = tl.load(in_ptr14 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp198 = tl.load(in_ptr0 + (12352 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp199 = tl.load(in_ptr1 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp203 = tl.load(in_ptr2 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp205 = tl.load(in_ptr15 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp213 = tl.load(in_ptr0 + (6208 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp214 = tl.load(in_ptr1 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp218 = tl.load(in_ptr2 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp220 = tl.load(in_ptr16 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp228 = tl.load(in_ptr0 + (64 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp229 = tl.load(in_ptr1 + (x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp233 = tl.load(in_ptr2 + (x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp235 = tl.load(in_ptr17 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp241 = tl.load(in_ptr0 + (92192 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp243 = tl.load(in_ptr18 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp247 = tl.load(in_ptr0 + (86048 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp249 = tl.load(in_ptr18 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp251 = tl.load(in_ptr19 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp257 = tl.load(in_ptr0 + (79904 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp259 = tl.load(in_ptr18 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp261 = tl.load(in_ptr20 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp267 = tl.load(in_ptr0 + (73760 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp269 = tl.load(in_ptr18 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp271 = tl.load(in_ptr21 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp277 = tl.load(in_ptr0 + (67616 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp279 = tl.load(in_ptr18 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp281 = tl.load(in_ptr22 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp287 = tl.load(in_ptr0 + (61472 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp289 = tl.load(in_ptr18 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp291 = tl.load(in_ptr23 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp297 = tl.load(in_ptr0 + (55328 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp299 = tl.load(in_ptr18 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp301 = tl.load(in_ptr24 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp307 = tl.load(in_ptr0 + (49184 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp309 = tl.load(in_ptr18 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp311 = tl.load(in_ptr25 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp317 = tl.load(in_ptr0 + (43040 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp319 = tl.load(in_ptr18 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp321 = tl.load(in_ptr26 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp327 = tl.load(in_ptr0 + (36896 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp329 = tl.load(in_ptr18 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp331 = tl.load(in_ptr27 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp337 = tl.load(in_ptr0 + (30752 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp339 = tl.load(in_ptr18 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp341 = tl.load(in_ptr28 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp347 = tl.load(in_ptr0 + (24608 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp349 = tl.load(in_ptr18 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp351 = tl.load(in_ptr29 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp357 = tl.load(in_ptr0 + (18464 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp359 = tl.load(in_ptr18 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp361 = tl.load(in_ptr30 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp367 = tl.load(in_ptr0 + (12320 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp369 = tl.load(in_ptr18 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp371 = tl.load(in_ptr31 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp377 = tl.load(in_ptr0 + (6176 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp379 = tl.load(in_ptr18 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp381 = tl.load(in_ptr32 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp387 = tl.load(in_ptr0 + (32 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp389 = tl.load(in_ptr18 + (x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp391 = tl.load(in_ptr33 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 15, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp3 * tmp6
    tmp9 = tmp7 / tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = tl.full([1], 14, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp15 * tmp18
    tmp21 = tmp19 / tmp20
    tmp22 = x1
    tmp23 = tl.full([1], 63, tl.int32)
    tmp24 = tmp22 == tmp23
    tmp26 = tl.where(tmp24, tmp25, tmp11)
    tmp27 = tmp21 + tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.where(tmp14, tmp28, tmp11)
    tmp30 = tmp12 + tmp29
    tmp31 = tl.full([1], 13, tl.int32)
    tmp32 = tmp0 == tmp31
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp33 * tmp36
    tmp39 = tmp37 / tmp38
    tmp41 = tl.where(tmp24, tmp40, tmp11)
    tmp42 = tmp39 + tmp41
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tl.where(tmp32, tmp43, tmp11)
    tmp45 = tmp30 + tmp44
    tmp46 = tl.full([1], 12, tl.int32)
    tmp47 = tmp0 == tmp46
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp48 * tmp51
    tmp54 = tmp52 / tmp53
    tmp56 = tl.where(tmp24, tmp55, tmp11)
    tmp57 = tmp54 + tmp56
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tl.where(tmp47, tmp58, tmp11)
    tmp60 = tmp45 + tmp59
    tmp61 = tl.full([1], 11, tl.int32)
    tmp62 = tmp0 == tmp61
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tmp65.to(tl.float32)
    tmp67 = tmp63 * tmp66
    tmp69 = tmp67 / tmp68
    tmp71 = tl.where(tmp24, tmp70, tmp11)
    tmp72 = tmp69 + tmp71
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tl.where(tmp62, tmp73, tmp11)
    tmp75 = tmp60 + tmp74
    tmp76 = tl.full([1], 10, tl.int32)
    tmp77 = tmp0 == tmp76
    tmp80 = tmp79.to(tl.float32)
    tmp81 = tmp80.to(tl.float32)
    tmp82 = tmp78 * tmp81
    tmp84 = tmp82 / tmp83
    tmp86 = tl.where(tmp24, tmp85, tmp11)
    tmp87 = tmp84 + tmp86
    tmp88 = tmp87.to(tl.float32)
    tmp89 = tl.where(tmp77, tmp88, tmp11)
    tmp90 = tmp75 + tmp89
    tmp91 = tl.full([1], 9, tl.int32)
    tmp92 = tmp0 == tmp91
    tmp95 = tmp94.to(tl.float32)
    tmp96 = tmp95.to(tl.float32)
    tmp97 = tmp93 * tmp96
    tmp99 = tmp97 / tmp98
    tmp101 = tl.where(tmp24, tmp100, tmp11)
    tmp102 = tmp99 + tmp101
    tmp103 = tmp102.to(tl.float32)
    tmp104 = tl.where(tmp92, tmp103, tmp11)
    tmp105 = tmp90 + tmp104
    tmp106 = tl.full([1], 8, tl.int32)
    tmp107 = tmp0 == tmp106
    tmp110 = tmp109.to(tl.float32)
    tmp111 = tmp110.to(tl.float32)
    tmp112 = tmp108 * tmp111
    tmp114 = tmp112 / tmp113
    tmp116 = tl.where(tmp24, tmp115, tmp11)
    tmp117 = tmp114 + tmp116
    tmp118 = tmp117.to(tl.float32)
    tmp119 = tl.where(tmp107, tmp118, tmp11)
    tmp120 = tmp105 + tmp119
    tmp121 = tl.full([1], 7, tl.int32)
    tmp122 = tmp0 == tmp121
    tmp125 = tmp124.to(tl.float32)
    tmp126 = tmp125.to(tl.float32)
    tmp127 = tmp123 * tmp126
    tmp129 = tmp127 / tmp128
    tmp131 = tl.where(tmp24, tmp130, tmp11)
    tmp132 = tmp129 + tmp131
    tmp133 = tmp132.to(tl.float32)
    tmp134 = tl.where(tmp122, tmp133, tmp11)
    tmp135 = tmp120 + tmp134
    tmp136 = tl.full([1], 6, tl.int32)
    tmp137 = tmp0 == tmp136
    tmp140 = tmp139.to(tl.float32)
    tmp141 = tmp140.to(tl.float32)
    tmp142 = tmp138 * tmp141
    tmp144 = tmp142 / tmp143
    tmp146 = tl.where(tmp24, tmp145, tmp11)
    tmp147 = tmp144 + tmp146
    tmp148 = tmp147.to(tl.float32)
    tmp149 = tl.where(tmp137, tmp148, tmp11)
    tmp150 = tmp135 + tmp149
    tmp151 = tl.full([1], 5, tl.int32)
    tmp152 = tmp0 == tmp151
    tmp155 = tmp154.to(tl.float32)
    tmp156 = tmp155.to(tl.float32)
    tmp157 = tmp153 * tmp156
    tmp159 = tmp157 / tmp158
    tmp161 = tl.where(tmp24, tmp160, tmp11)
    tmp162 = tmp159 + tmp161
    tmp163 = tmp162.to(tl.float32)
    tmp164 = tl.where(tmp152, tmp163, tmp11)
    tmp165 = tmp150 + tmp164
    tmp166 = tl.full([1], 4, tl.int32)
    tmp167 = tmp0 == tmp166
    tmp170 = tmp169.to(tl.float32)
    tmp171 = tmp170.to(tl.float32)
    tmp172 = tmp168 * tmp171
    tmp174 = tmp172 / tmp173
    tmp176 = tl.where(tmp24, tmp175, tmp11)
    tmp177 = tmp174 + tmp176
    tmp178 = tmp177.to(tl.float32)
    tmp179 = tl.where(tmp167, tmp178, tmp11)
    tmp180 = tmp165 + tmp179
    tmp181 = tl.full([1], 3, tl.int32)
    tmp182 = tmp0 == tmp181
    tmp185 = tmp184.to(tl.float32)
    tmp186 = tmp185.to(tl.float32)
    tmp187 = tmp183 * tmp186
    tmp189 = tmp187 / tmp188
    tmp191 = tl.where(tmp24, tmp190, tmp11)
    tmp192 = tmp189 + tmp191
    tmp193 = tmp192.to(tl.float32)
    tmp194 = tl.where(tmp182, tmp193, tmp11)
    tmp195 = tmp180 + tmp194
    tmp196 = tl.full([1], 2, tl.int32)
    tmp197 = tmp0 == tmp196
    tmp200 = tmp199.to(tl.float32)
    tmp201 = tmp200.to(tl.float32)
    tmp202 = tmp198 * tmp201
    tmp204 = tmp202 / tmp203
    tmp206 = tl.where(tmp24, tmp205, tmp11)
    tmp207 = tmp204 + tmp206
    tmp208 = tmp207.to(tl.float32)
    tmp209 = tl.where(tmp197, tmp208, tmp11)
    tmp210 = tmp195 + tmp209
    tmp211 = tl.full([1], 1, tl.int32)
    tmp212 = tmp0 == tmp211
    tmp215 = tmp214.to(tl.float32)
    tmp216 = tmp215.to(tl.float32)
    tmp217 = tmp213 * tmp216
    tmp219 = tmp217 / tmp218
    tmp221 = tl.where(tmp24, tmp220, tmp11)
    tmp222 = tmp219 + tmp221
    tmp223 = tmp222.to(tl.float32)
    tmp224 = tl.where(tmp212, tmp223, tmp11)
    tmp225 = tmp210 + tmp224
    tmp226 = tl.full([1], 0, tl.int32)
    tmp227 = tmp0 == tmp226
    tmp230 = tmp229.to(tl.float32)
    tmp231 = tmp230.to(tl.float32)
    tmp232 = tmp228 * tmp231
    tmp234 = tmp232 / tmp233
    tmp236 = tl.where(tmp24, tmp235, tmp11)
    tmp237 = tmp234 + tmp236
    tmp238 = tmp237.to(tl.float32)
    tmp239 = tl.where(tmp227, tmp238, tmp11)
    tmp240 = tmp225 + tmp239
    tmp242 = tmp241 * tmp6
    tmp244 = tmp242 / tmp243
    tmp245 = tmp244.to(tl.float32)
    tmp246 = tl.where(tmp2, tmp245, tmp11)
    tmp248 = tmp247 * tmp18
    tmp250 = tmp248 / tmp249
    tmp252 = tl.where(tmp24, tmp251, tmp11)
    tmp253 = tmp250 + tmp252
    tmp254 = tmp253.to(tl.float32)
    tmp255 = tl.where(tmp14, tmp254, tmp11)
    tmp256 = tmp246 + tmp255
    tmp258 = tmp257 * tmp36
    tmp260 = tmp258 / tmp259
    tmp262 = tl.where(tmp24, tmp261, tmp11)
    tmp263 = tmp260 + tmp262
    tmp264 = tmp263.to(tl.float32)
    tmp265 = tl.where(tmp32, tmp264, tmp11)
    tmp266 = tmp256 + tmp265
    tmp268 = tmp267 * tmp51
    tmp270 = tmp268 / tmp269
    tmp272 = tl.where(tmp24, tmp271, tmp11)
    tmp273 = tmp270 + tmp272
    tmp274 = tmp273.to(tl.float32)
    tmp275 = tl.where(tmp47, tmp274, tmp11)
    tmp276 = tmp266 + tmp275
    tmp278 = tmp277 * tmp66
    tmp280 = tmp278 / tmp279
    tmp282 = tl.where(tmp24, tmp281, tmp11)
    tmp283 = tmp280 + tmp282
    tmp284 = tmp283.to(tl.float32)
    tmp285 = tl.where(tmp62, tmp284, tmp11)
    tmp286 = tmp276 + tmp285
    tmp288 = tmp287 * tmp81
    tmp290 = tmp288 / tmp289
    tmp292 = tl.where(tmp24, tmp291, tmp11)
    tmp293 = tmp290 + tmp292
    tmp294 = tmp293.to(tl.float32)
    tmp295 = tl.where(tmp77, tmp294, tmp11)
    tmp296 = tmp286 + tmp295
    tmp298 = tmp297 * tmp96
    tmp300 = tmp298 / tmp299
    tmp302 = tl.where(tmp24, tmp301, tmp11)
    tmp303 = tmp300 + tmp302
    tmp304 = tmp303.to(tl.float32)
    tmp305 = tl.where(tmp92, tmp304, tmp11)
    tmp306 = tmp296 + tmp305
    tmp308 = tmp307 * tmp111
    tmp310 = tmp308 / tmp309
    tmp312 = tl.where(tmp24, tmp311, tmp11)
    tmp313 = tmp310 + tmp312
    tmp314 = tmp313.to(tl.float32)
    tmp315 = tl.where(tmp107, tmp314, tmp11)
    tmp316 = tmp306 + tmp315
    tmp318 = tmp317 * tmp126
    tmp320 = tmp318 / tmp319
    tmp322 = tl.where(tmp24, tmp321, tmp11)
    tmp323 = tmp320 + tmp322
    tmp324 = tmp323.to(tl.float32)
    tmp325 = tl.where(tmp122, tmp324, tmp11)
    tmp326 = tmp316 + tmp325
    tmp328 = tmp327 * tmp141
    tmp330 = tmp328 / tmp329
    tmp332 = tl.where(tmp24, tmp331, tmp11)
    tmp333 = tmp330 + tmp332
    tmp334 = tmp333.to(tl.float32)
    tmp335 = tl.where(tmp137, tmp334, tmp11)
    tmp336 = tmp326 + tmp335
    tmp338 = tmp337 * tmp156
    tmp340 = tmp338 / tmp339
    tmp342 = tl.where(tmp24, tmp341, tmp11)
    tmp343 = tmp340 + tmp342
    tmp344 = tmp343.to(tl.float32)
    tmp345 = tl.where(tmp152, tmp344, tmp11)
    tmp346 = tmp336 + tmp345
    tmp348 = tmp347 * tmp171
    tmp350 = tmp348 / tmp349
    tmp352 = tl.where(tmp24, tmp351, tmp11)
    tmp353 = tmp350 + tmp352
    tmp354 = tmp353.to(tl.float32)
    tmp355 = tl.where(tmp167, tmp354, tmp11)
    tmp356 = tmp346 + tmp355
    tmp358 = tmp357 * tmp186
    tmp360 = tmp358 / tmp359
    tmp362 = tl.where(tmp24, tmp361, tmp11)
    tmp363 = tmp360 + tmp362
    tmp364 = tmp363.to(tl.float32)
    tmp365 = tl.where(tmp182, tmp364, tmp11)
    tmp366 = tmp356 + tmp365
    tmp368 = tmp367 * tmp201
    tmp370 = tmp368 / tmp369
    tmp372 = tl.where(tmp24, tmp371, tmp11)
    tmp373 = tmp370 + tmp372
    tmp374 = tmp373.to(tl.float32)
    tmp375 = tl.where(tmp197, tmp374, tmp11)
    tmp376 = tmp366 + tmp375
    tmp378 = tmp377 * tmp216
    tmp380 = tmp378 / tmp379
    tmp382 = tl.where(tmp24, tmp381, tmp11)
    tmp383 = tmp380 + tmp382
    tmp384 = tmp383.to(tl.float32)
    tmp385 = tl.where(tmp212, tmp384, tmp11)
    tmp386 = tmp376 + tmp385
    tmp388 = tmp387 * tmp231
    tmp390 = tmp388 / tmp389
    tmp392 = tl.where(tmp24, tmp391, tmp11)
    tmp393 = tmp390 + tmp392
    tmp394 = tmp393.to(tl.float32)
    tmp395 = tl.where(tmp227, tmp394, tmp11)
    tmp396 = tmp386 + tmp395
    tl.store(in_out_ptr0 + (x5), tmp240, None)
    tl.store(in_out_ptr1 + (x5), tmp396, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/67/c67nqp6v3aohrnjnrguyz4ouh45vph5xa2vguazcgzq7btnuk2fa.py
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
    tmp5 = tl.load(in_ptr2 + (960 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr0 + (86016 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp14 = tl.load(in_ptr1 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp18 = tl.load(in_ptr2 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr4 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp29 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr0 + (79872 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp37 = tl.load(in_ptr1 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp41 = tl.load(in_ptr2 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr6 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp54 = tl.load(in_ptr0 + (73728 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp55 = tl.load(in_ptr1 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp59 = tl.load(in_ptr2 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp63 = tl.load(in_ptr7 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp72 = tl.load(in_ptr0 + (67584 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp73 = tl.load(in_ptr1 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp77 = tl.load(in_ptr2 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp81 = tl.load(in_ptr8 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp90 = tl.load(in_ptr0 + (61440 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp91 = tl.load(in_ptr1 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp95 = tl.load(in_ptr2 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp99 = tl.load(in_ptr9 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp108 = tl.load(in_ptr0 + (55296 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp109 = tl.load(in_ptr1 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp113 = tl.load(in_ptr2 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp117 = tl.load(in_ptr10 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp126 = tl.load(in_ptr0 + (49152 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp127 = tl.load(in_ptr1 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp131 = tl.load(in_ptr2 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp135 = tl.load(in_ptr11 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp144 = tl.load(in_ptr0 + (43008 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp145 = tl.load(in_ptr1 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp149 = tl.load(in_ptr2 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp153 = tl.load(in_ptr12 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp162 = tl.load(in_ptr0 + (36864 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp163 = tl.load(in_ptr1 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp167 = tl.load(in_ptr2 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp171 = tl.load(in_ptr13 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp180 = tl.load(in_ptr0 + (30720 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp181 = tl.load(in_ptr1 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp185 = tl.load(in_ptr2 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp189 = tl.load(in_ptr14 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp198 = tl.load(in_ptr0 + (24576 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp199 = tl.load(in_ptr1 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp203 = tl.load(in_ptr2 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp207 = tl.load(in_ptr15 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp216 = tl.load(in_ptr0 + (18432 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp217 = tl.load(in_ptr1 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp221 = tl.load(in_ptr2 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp225 = tl.load(in_ptr16 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp234 = tl.load(in_ptr0 + (12288 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp235 = tl.load(in_ptr1 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp239 = tl.load(in_ptr2 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp243 = tl.load(in_ptr17 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp252 = tl.load(in_ptr0 + (6144 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp253 = tl.load(in_ptr1 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp257 = tl.load(in_ptr2 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp261 = tl.load(in_ptr18 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp0 * tmp3
    tmp6 = tmp4 / tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp13 * tmp16
    tmp19 = tmp17 / tmp18
    tmp20 = r2
    tmp21 = tl.full([1, 1], 63, tl.int32)
    tmp22 = tmp20 == tmp21
    tmp23 = 0.0
    tmp24 = tl.where(tmp22, tmp12, tmp23)
    tmp25 = tmp19 + tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27.to(tl.float32)
    tmp30 = tmp28 * tmp29
    tmp31 = tmp25 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
    tmp34 = tl.where(xmask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None]
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp36 * tmp39
    tmp42 = tmp40 / tmp41
    tmp43 = tl.where(tmp22, tmp35, tmp23)
    tmp44 = tmp42 + tmp43
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp47 * tmp29
    tmp49 = tmp44 * tmp48
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK, RBLOCK])
    tmp52 = tl.where(xmask, tmp50, 0)
    tmp53 = tl.sum(tmp52, 1)[:, None]
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tmp54 * tmp57
    tmp60 = tmp58 / tmp59
    tmp61 = tl.where(tmp22, tmp53, tmp23)
    tmp62 = tmp60 + tmp61
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tmp65 * tmp29
    tmp67 = tmp62 * tmp66
    tmp68 = tl.broadcast_to(tmp67, [XBLOCK, RBLOCK])
    tmp70 = tl.where(xmask, tmp68, 0)
    tmp71 = tl.sum(tmp70, 1)[:, None]
    tmp74 = tmp73.to(tl.float32)
    tmp75 = tmp74.to(tl.float32)
    tmp76 = tmp72 * tmp75
    tmp78 = tmp76 / tmp77
    tmp79 = tl.where(tmp22, tmp71, tmp23)
    tmp80 = tmp78 + tmp79
    tmp82 = tmp81.to(tl.float32)
    tmp83 = tmp82.to(tl.float32)
    tmp84 = tmp83 * tmp29
    tmp85 = tmp80 * tmp84
    tmp86 = tl.broadcast_to(tmp85, [XBLOCK, RBLOCK])
    tmp88 = tl.where(xmask, tmp86, 0)
    tmp89 = tl.sum(tmp88, 1)[:, None]
    tmp92 = tmp91.to(tl.float32)
    tmp93 = tmp92.to(tl.float32)
    tmp94 = tmp90 * tmp93
    tmp96 = tmp94 / tmp95
    tmp97 = tl.where(tmp22, tmp89, tmp23)
    tmp98 = tmp96 + tmp97
    tmp100 = tmp99.to(tl.float32)
    tmp101 = tmp100.to(tl.float32)
    tmp102 = tmp101 * tmp29
    tmp103 = tmp98 * tmp102
    tmp104 = tl.broadcast_to(tmp103, [XBLOCK, RBLOCK])
    tmp106 = tl.where(xmask, tmp104, 0)
    tmp107 = tl.sum(tmp106, 1)[:, None]
    tmp110 = tmp109.to(tl.float32)
    tmp111 = tmp110.to(tl.float32)
    tmp112 = tmp108 * tmp111
    tmp114 = tmp112 / tmp113
    tmp115 = tl.where(tmp22, tmp107, tmp23)
    tmp116 = tmp114 + tmp115
    tmp118 = tmp117.to(tl.float32)
    tmp119 = tmp118.to(tl.float32)
    tmp120 = tmp119 * tmp29
    tmp121 = tmp116 * tmp120
    tmp122 = tl.broadcast_to(tmp121, [XBLOCK, RBLOCK])
    tmp124 = tl.where(xmask, tmp122, 0)
    tmp125 = tl.sum(tmp124, 1)[:, None]
    tmp128 = tmp127.to(tl.float32)
    tmp129 = tmp128.to(tl.float32)
    tmp130 = tmp126 * tmp129
    tmp132 = tmp130 / tmp131
    tmp133 = tl.where(tmp22, tmp125, tmp23)
    tmp134 = tmp132 + tmp133
    tmp136 = tmp135.to(tl.float32)
    tmp137 = tmp136.to(tl.float32)
    tmp138 = tmp137 * tmp29
    tmp139 = tmp134 * tmp138
    tmp140 = tl.broadcast_to(tmp139, [XBLOCK, RBLOCK])
    tmp142 = tl.where(xmask, tmp140, 0)
    tmp143 = tl.sum(tmp142, 1)[:, None]
    tmp146 = tmp145.to(tl.float32)
    tmp147 = tmp146.to(tl.float32)
    tmp148 = tmp144 * tmp147
    tmp150 = tmp148 / tmp149
    tmp151 = tl.where(tmp22, tmp143, tmp23)
    tmp152 = tmp150 + tmp151
    tmp154 = tmp153.to(tl.float32)
    tmp155 = tmp154.to(tl.float32)
    tmp156 = tmp155 * tmp29
    tmp157 = tmp152 * tmp156
    tmp158 = tl.broadcast_to(tmp157, [XBLOCK, RBLOCK])
    tmp160 = tl.where(xmask, tmp158, 0)
    tmp161 = tl.sum(tmp160, 1)[:, None]
    tmp164 = tmp163.to(tl.float32)
    tmp165 = tmp164.to(tl.float32)
    tmp166 = tmp162 * tmp165
    tmp168 = tmp166 / tmp167
    tmp169 = tl.where(tmp22, tmp161, tmp23)
    tmp170 = tmp168 + tmp169
    tmp172 = tmp171.to(tl.float32)
    tmp173 = tmp172.to(tl.float32)
    tmp174 = tmp173 * tmp29
    tmp175 = tmp170 * tmp174
    tmp176 = tl.broadcast_to(tmp175, [XBLOCK, RBLOCK])
    tmp178 = tl.where(xmask, tmp176, 0)
    tmp179 = tl.sum(tmp178, 1)[:, None]
    tmp182 = tmp181.to(tl.float32)
    tmp183 = tmp182.to(tl.float32)
    tmp184 = tmp180 * tmp183
    tmp186 = tmp184 / tmp185
    tmp187 = tl.where(tmp22, tmp179, tmp23)
    tmp188 = tmp186 + tmp187
    tmp190 = tmp189.to(tl.float32)
    tmp191 = tmp190.to(tl.float32)
    tmp192 = tmp191 * tmp29
    tmp193 = tmp188 * tmp192
    tmp194 = tl.broadcast_to(tmp193, [XBLOCK, RBLOCK])
    tmp196 = tl.where(xmask, tmp194, 0)
    tmp197 = tl.sum(tmp196, 1)[:, None]
    tmp200 = tmp199.to(tl.float32)
    tmp201 = tmp200.to(tl.float32)
    tmp202 = tmp198 * tmp201
    tmp204 = tmp202 / tmp203
    tmp205 = tl.where(tmp22, tmp197, tmp23)
    tmp206 = tmp204 + tmp205
    tmp208 = tmp207.to(tl.float32)
    tmp209 = tmp208.to(tl.float32)
    tmp210 = tmp209 * tmp29
    tmp211 = tmp206 * tmp210
    tmp212 = tl.broadcast_to(tmp211, [XBLOCK, RBLOCK])
    tmp214 = tl.where(xmask, tmp212, 0)
    tmp215 = tl.sum(tmp214, 1)[:, None]
    tmp218 = tmp217.to(tl.float32)
    tmp219 = tmp218.to(tl.float32)
    tmp220 = tmp216 * tmp219
    tmp222 = tmp220 / tmp221
    tmp223 = tl.where(tmp22, tmp215, tmp23)
    tmp224 = tmp222 + tmp223
    tmp226 = tmp225.to(tl.float32)
    tmp227 = tmp226.to(tl.float32)
    tmp228 = tmp227 * tmp29
    tmp229 = tmp224 * tmp228
    tmp230 = tl.broadcast_to(tmp229, [XBLOCK, RBLOCK])
    tmp232 = tl.where(xmask, tmp230, 0)
    tmp233 = tl.sum(tmp232, 1)[:, None]
    tmp236 = tmp235.to(tl.float32)
    tmp237 = tmp236.to(tl.float32)
    tmp238 = tmp234 * tmp237
    tmp240 = tmp238 / tmp239
    tmp241 = tl.where(tmp22, tmp233, tmp23)
    tmp242 = tmp240 + tmp241
    tmp244 = tmp243.to(tl.float32)
    tmp245 = tmp244.to(tl.float32)
    tmp246 = tmp245 * tmp29
    tmp247 = tmp242 * tmp246
    tmp248 = tl.broadcast_to(tmp247, [XBLOCK, RBLOCK])
    tmp250 = tl.where(xmask, tmp248, 0)
    tmp251 = tl.sum(tmp250, 1)[:, None]
    tmp254 = tmp253.to(tl.float32)
    tmp255 = tmp254.to(tl.float32)
    tmp256 = tmp252 * tmp255
    tmp258 = tmp256 / tmp257
    tmp259 = tl.where(tmp22, tmp251, tmp23)
    tmp260 = tmp258 + tmp259
    tmp262 = tmp261.to(tl.float32)
    tmp263 = tmp262.to(tl.float32)
    tmp264 = tmp263 * tmp29
    tmp265 = tmp260 * tmp264
    tmp266 = tl.broadcast_to(tmp265, [XBLOCK, RBLOCK])
    tmp268 = tl.where(xmask, tmp266, 0)
    tmp269 = tl.sum(tmp268, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tl.store(out_ptr1 + (x3), tmp35, xmask)
    tl.store(out_ptr2 + (x3), tmp53, xmask)
    tl.store(out_ptr3 + (x3), tmp71, xmask)
    tl.store(out_ptr4 + (x3), tmp89, xmask)
    tl.store(out_ptr5 + (x3), tmp107, xmask)
    tl.store(out_ptr6 + (x3), tmp125, xmask)
    tl.store(out_ptr7 + (x3), tmp143, xmask)
    tl.store(out_ptr8 + (x3), tmp161, xmask)
    tl.store(out_ptr9 + (x3), tmp179, xmask)
    tl.store(out_ptr10 + (x3), tmp197, xmask)
    tl.store(out_ptr11 + (x3), tmp215, xmask)
    tl.store(out_ptr12 + (x3), tmp233, xmask)
    tl.store(out_ptr13 + (x3), tmp251, xmask)
    tl.store(out_ptr14 + (x3), tmp269, xmask)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/tw/ctwkgmzbc6p2jo7ciqjnr7oa2cykave4thg2mg2n5pyh2eq7ile5.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward, aten._to_copy, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_18 : [num_users=48] = call_function[target=torch.ops.aten.full.default](args = ([32, 16, 64, 32], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_20 : [num_users=45] = call_function[target=torch.ops.aten.full.default](args = ([32, 64, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_136 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%select_326, torch.bfloat16), kwargs = {})
#   %select_scatter_default_62 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_136, 1, 15), kwargs = {})
#   %select_scatter_default_63 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_30, 1, -1), kwargs = {})
#   %add_116 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_325, %select_scatter_default_63), kwargs = {})
#   %convert_element_type_137 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_116, torch.bfloat16), kwargs = {})
#   %select_scatter_default_64 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_137, 1, 14), kwargs = {})
#   %add_117 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default_62, %select_scatter_default_64), kwargs = {})
#   %select_scatter_default_65 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_31, 1, -1), kwargs = {})
#   %add_118 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_324, %select_scatter_default_65), kwargs = {})
#   %convert_element_type_138 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_118, torch.bfloat16), kwargs = {})
#   %select_scatter_default_66 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_138, 1, 13), kwargs = {})
#   %add_119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_117, %select_scatter_default_66), kwargs = {})
#   %select_scatter_default_67 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_32, 1, -1), kwargs = {})
#   %add_120 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_323, %select_scatter_default_67), kwargs = {})
#   %convert_element_type_139 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_120, torch.bfloat16), kwargs = {})
#   %select_scatter_default_68 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_139, 1, 12), kwargs = {})
#   %add_121 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_119, %select_scatter_default_68), kwargs = {})
#   %select_scatter_default_69 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_33, 1, -1), kwargs = {})
#   %add_122 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_322, %select_scatter_default_69), kwargs = {})
#   %convert_element_type_140 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_122, torch.bfloat16), kwargs = {})
#   %select_scatter_default_70 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_140, 1, 11), kwargs = {})
#   %add_123 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_121, %select_scatter_default_70), kwargs = {})
#   %select_scatter_default_71 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_34, 1, -1), kwargs = {})
#   %add_124 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_321, %select_scatter_default_71), kwargs = {})
#   %convert_element_type_141 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_124, torch.bfloat16), kwargs = {})
#   %select_scatter_default_72 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_141, 1, 10), kwargs = {})
#   %add_125 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_123, %select_scatter_default_72), kwargs = {})
#   %select_scatter_default_73 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_35, 1, -1), kwargs = {})
#   %add_126 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_320, %select_scatter_default_73), kwargs = {})
#   %convert_element_type_142 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_126, torch.bfloat16), kwargs = {})
#   %select_scatter_default_74 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_142, 1, 9), kwargs = {})
#   %add_127 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_125, %select_scatter_default_74), kwargs = {})
#   %select_scatter_default_75 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_36, 1, -1), kwargs = {})
#   %add_128 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_319, %select_scatter_default_75), kwargs = {})
#   %convert_element_type_143 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_128, torch.bfloat16), kwargs = {})
#   %select_scatter_default_76 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_143, 1, 8), kwargs = {})
#   %add_129 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_127, %select_scatter_default_76), kwargs = {})
#   %select_scatter_default_77 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_37, 1, -1), kwargs = {})
#   %add_130 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_318, %select_scatter_default_77), kwargs = {})
#   %convert_element_type_144 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_130, torch.bfloat16), kwargs = {})
#   %select_scatter_default_78 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_144, 1, 7), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_129, %select_scatter_default_78), kwargs = {})
#   %select_scatter_default_79 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_38, 1, -1), kwargs = {})
#   %add_132 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_317, %select_scatter_default_79), kwargs = {})
#   %convert_element_type_145 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_132, torch.bfloat16), kwargs = {})
#   %select_scatter_default_80 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_145, 1, 6), kwargs = {})
#   %add_133 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_131, %select_scatter_default_80), kwargs = {})
#   %select_scatter_default_81 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_39, 1, -1), kwargs = {})
#   %add_134 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_316, %select_scatter_default_81), kwargs = {})
#   %convert_element_type_146 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_134, torch.bfloat16), kwargs = {})
#   %select_scatter_default_82 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_146, 1, 5), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_133, %select_scatter_default_82), kwargs = {})
#   %select_scatter_default_83 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_40, 1, -1), kwargs = {})
#   %add_136 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_315, %select_scatter_default_83), kwargs = {})
#   %convert_element_type_147 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_136, torch.bfloat16), kwargs = {})
#   %select_scatter_default_84 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_147, 1, 4), kwargs = {})
#   %add_137 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_135, %select_scatter_default_84), kwargs = {})
#   %select_scatter_default_85 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_41, 1, -1), kwargs = {})
#   %add_138 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_314, %select_scatter_default_85), kwargs = {})
#   %convert_element_type_148 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_138, torch.bfloat16), kwargs = {})
#   %select_scatter_default_86 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_148, 1, 3), kwargs = {})
#   %add_139 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_137, %select_scatter_default_86), kwargs = {})
#   %select_scatter_default_87 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_42, 1, -1), kwargs = {})
#   %add_140 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_313, %select_scatter_default_87), kwargs = {})
#   %convert_element_type_149 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_140, torch.bfloat16), kwargs = {})
#   %select_scatter_default_88 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_149, 1, 2), kwargs = {})
#   %add_141 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_139, %select_scatter_default_88), kwargs = {})
#   %select_scatter_default_89 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_43, 1, -1), kwargs = {})
#   %add_142 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_312, %select_scatter_default_89), kwargs = {})
#   %convert_element_type_150 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_142, torch.bfloat16), kwargs = {})
#   %select_scatter_default_90 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_150, 1, 1), kwargs = {})
#   %add_143 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_141, %select_scatter_default_90), kwargs = {})
#   %select_scatter_default_91 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_20, %squeeze_44, 1, -1), kwargs = {})
#   %add_144 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_311, %select_scatter_default_91), kwargs = {})
#   %convert_element_type_151 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_144, torch.bfloat16), kwargs = {})
#   %select_scatter_default_92 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_18, %convert_element_type_151, 1, 0), kwargs = {})
#   %add_145 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_143, %select_scatter_default_92), kwargs = {})
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
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_select_backward_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 63, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048) % 16
    x0 = xindex % 32
    x1 = (xindex // 32) % 64
    x3 = (xindex // 32768)
    x5 = xindex
    tmp3 = tl.load(in_ptr0 + (92160 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (86016 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp20 = tl.load(in_ptr2 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (79872 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr1 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp38 = tl.load(in_ptr2 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr4 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr0 + (73728 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr1 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp53 = tl.load(in_ptr2 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr5 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr0 + (67584 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr1 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp68 = tl.load(in_ptr2 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr6 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr0 + (61440 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr1 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp83 = tl.load(in_ptr2 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr7 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr0 + (55296 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr1 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp98 = tl.load(in_ptr2 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp100 = tl.load(in_ptr8 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr0 + (49152 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp109 = tl.load(in_ptr1 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp113 = tl.load(in_ptr2 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp115 = tl.load(in_ptr9 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp123 = tl.load(in_ptr0 + (43008 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp124 = tl.load(in_ptr1 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp128 = tl.load(in_ptr2 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp130 = tl.load(in_ptr10 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp138 = tl.load(in_ptr0 + (36864 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp139 = tl.load(in_ptr1 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp143 = tl.load(in_ptr2 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp145 = tl.load(in_ptr11 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp153 = tl.load(in_ptr0 + (30720 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp154 = tl.load(in_ptr1 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp158 = tl.load(in_ptr2 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp160 = tl.load(in_ptr12 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp168 = tl.load(in_ptr0 + (24576 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp169 = tl.load(in_ptr1 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp173 = tl.load(in_ptr2 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp175 = tl.load(in_ptr13 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp183 = tl.load(in_ptr0 + (18432 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp184 = tl.load(in_ptr1 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp188 = tl.load(in_ptr2 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp190 = tl.load(in_ptr14 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp198 = tl.load(in_ptr0 + (12288 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp199 = tl.load(in_ptr1 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp203 = tl.load(in_ptr2 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp205 = tl.load(in_ptr15 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp213 = tl.load(in_ptr0 + (6144 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp214 = tl.load(in_ptr1 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp218 = tl.load(in_ptr2 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp220 = tl.load(in_ptr16 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp228 = tl.load(in_ptr0 + (x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp229 = tl.load(in_ptr1 + (x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp233 = tl.load(in_ptr2 + (x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp235 = tl.load(in_ptr17 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 15, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp3 * tmp6
    tmp9 = tmp7 / tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = tl.full([1], 14, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp15 * tmp18
    tmp21 = tmp19 / tmp20
    tmp22 = x1
    tmp23 = tl.full([1], 63, tl.int32)
    tmp24 = tmp22 == tmp23
    tmp26 = tl.where(tmp24, tmp25, tmp11)
    tmp27 = tmp21 + tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.where(tmp14, tmp28, tmp11)
    tmp30 = tmp12 + tmp29
    tmp31 = tl.full([1], 13, tl.int32)
    tmp32 = tmp0 == tmp31
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp33 * tmp36
    tmp39 = tmp37 / tmp38
    tmp41 = tl.where(tmp24, tmp40, tmp11)
    tmp42 = tmp39 + tmp41
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tl.where(tmp32, tmp43, tmp11)
    tmp45 = tmp30 + tmp44
    tmp46 = tl.full([1], 12, tl.int32)
    tmp47 = tmp0 == tmp46
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp48 * tmp51
    tmp54 = tmp52 / tmp53
    tmp56 = tl.where(tmp24, tmp55, tmp11)
    tmp57 = tmp54 + tmp56
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tl.where(tmp47, tmp58, tmp11)
    tmp60 = tmp45 + tmp59
    tmp61 = tl.full([1], 11, tl.int32)
    tmp62 = tmp0 == tmp61
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tmp65.to(tl.float32)
    tmp67 = tmp63 * tmp66
    tmp69 = tmp67 / tmp68
    tmp71 = tl.where(tmp24, tmp70, tmp11)
    tmp72 = tmp69 + tmp71
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tl.where(tmp62, tmp73, tmp11)
    tmp75 = tmp60 + tmp74
    tmp76 = tl.full([1], 10, tl.int32)
    tmp77 = tmp0 == tmp76
    tmp80 = tmp79.to(tl.float32)
    tmp81 = tmp80.to(tl.float32)
    tmp82 = tmp78 * tmp81
    tmp84 = tmp82 / tmp83
    tmp86 = tl.where(tmp24, tmp85, tmp11)
    tmp87 = tmp84 + tmp86
    tmp88 = tmp87.to(tl.float32)
    tmp89 = tl.where(tmp77, tmp88, tmp11)
    tmp90 = tmp75 + tmp89
    tmp91 = tl.full([1], 9, tl.int32)
    tmp92 = tmp0 == tmp91
    tmp95 = tmp94.to(tl.float32)
    tmp96 = tmp95.to(tl.float32)
    tmp97 = tmp93 * tmp96
    tmp99 = tmp97 / tmp98
    tmp101 = tl.where(tmp24, tmp100, tmp11)
    tmp102 = tmp99 + tmp101
    tmp103 = tmp102.to(tl.float32)
    tmp104 = tl.where(tmp92, tmp103, tmp11)
    tmp105 = tmp90 + tmp104
    tmp106 = tl.full([1], 8, tl.int32)
    tmp107 = tmp0 == tmp106
    tmp110 = tmp109.to(tl.float32)
    tmp111 = tmp110.to(tl.float32)
    tmp112 = tmp108 * tmp111
    tmp114 = tmp112 / tmp113
    tmp116 = tl.where(tmp24, tmp115, tmp11)
    tmp117 = tmp114 + tmp116
    tmp118 = tmp117.to(tl.float32)
    tmp119 = tl.where(tmp107, tmp118, tmp11)
    tmp120 = tmp105 + tmp119
    tmp121 = tl.full([1], 7, tl.int32)
    tmp122 = tmp0 == tmp121
    tmp125 = tmp124.to(tl.float32)
    tmp126 = tmp125.to(tl.float32)
    tmp127 = tmp123 * tmp126
    tmp129 = tmp127 / tmp128
    tmp131 = tl.where(tmp24, tmp130, tmp11)
    tmp132 = tmp129 + tmp131
    tmp133 = tmp132.to(tl.float32)
    tmp134 = tl.where(tmp122, tmp133, tmp11)
    tmp135 = tmp120 + tmp134
    tmp136 = tl.full([1], 6, tl.int32)
    tmp137 = tmp0 == tmp136
    tmp140 = tmp139.to(tl.float32)
    tmp141 = tmp140.to(tl.float32)
    tmp142 = tmp138 * tmp141
    tmp144 = tmp142 / tmp143
    tmp146 = tl.where(tmp24, tmp145, tmp11)
    tmp147 = tmp144 + tmp146
    tmp148 = tmp147.to(tl.float32)
    tmp149 = tl.where(tmp137, tmp148, tmp11)
    tmp150 = tmp135 + tmp149
    tmp151 = tl.full([1], 5, tl.int32)
    tmp152 = tmp0 == tmp151
    tmp155 = tmp154.to(tl.float32)
    tmp156 = tmp155.to(tl.float32)
    tmp157 = tmp153 * tmp156
    tmp159 = tmp157 / tmp158
    tmp161 = tl.where(tmp24, tmp160, tmp11)
    tmp162 = tmp159 + tmp161
    tmp163 = tmp162.to(tl.float32)
    tmp164 = tl.where(tmp152, tmp163, tmp11)
    tmp165 = tmp150 + tmp164
    tmp166 = tl.full([1], 4, tl.int32)
    tmp167 = tmp0 == tmp166
    tmp170 = tmp169.to(tl.float32)
    tmp171 = tmp170.to(tl.float32)
    tmp172 = tmp168 * tmp171
    tmp174 = tmp172 / tmp173
    tmp176 = tl.where(tmp24, tmp175, tmp11)
    tmp177 = tmp174 + tmp176
    tmp178 = tmp177.to(tl.float32)
    tmp179 = tl.where(tmp167, tmp178, tmp11)
    tmp180 = tmp165 + tmp179
    tmp181 = tl.full([1], 3, tl.int32)
    tmp182 = tmp0 == tmp181
    tmp185 = tmp184.to(tl.float32)
    tmp186 = tmp185.to(tl.float32)
    tmp187 = tmp183 * tmp186
    tmp189 = tmp187 / tmp188
    tmp191 = tl.where(tmp24, tmp190, tmp11)
    tmp192 = tmp189 + tmp191
    tmp193 = tmp192.to(tl.float32)
    tmp194 = tl.where(tmp182, tmp193, tmp11)
    tmp195 = tmp180 + tmp194
    tmp196 = tl.full([1], 2, tl.int32)
    tmp197 = tmp0 == tmp196
    tmp200 = tmp199.to(tl.float32)
    tmp201 = tmp200.to(tl.float32)
    tmp202 = tmp198 * tmp201
    tmp204 = tmp202 / tmp203
    tmp206 = tl.where(tmp24, tmp205, tmp11)
    tmp207 = tmp204 + tmp206
    tmp208 = tmp207.to(tl.float32)
    tmp209 = tl.where(tmp197, tmp208, tmp11)
    tmp210 = tmp195 + tmp209
    tmp211 = tl.full([1], 1, tl.int32)
    tmp212 = tmp0 == tmp211
    tmp215 = tmp214.to(tl.float32)
    tmp216 = tmp215.to(tl.float32)
    tmp217 = tmp213 * tmp216
    tmp219 = tmp217 / tmp218
    tmp221 = tl.where(tmp24, tmp220, tmp11)
    tmp222 = tmp219 + tmp221
    tmp223 = tmp222.to(tl.float32)
    tmp224 = tl.where(tmp212, tmp223, tmp11)
    tmp225 = tmp210 + tmp224
    tmp226 = tl.full([1], 0, tl.int32)
    tmp227 = tmp0 == tmp226
    tmp230 = tmp229.to(tl.float32)
    tmp231 = tmp230.to(tl.float32)
    tmp232 = tmp228 * tmp231
    tmp234 = tmp232 / tmp233
    tmp236 = tl.where(tmp24, tmp235, tmp11)
    tmp237 = tmp234 + tmp236
    tmp238 = tmp237.to(tl.float32)
    tmp239 = tl.where(tmp227, tmp238, tmp11)
    tmp240 = tmp225 + tmp239
    tl.store(in_out_ptr0 + (x5), tmp240, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/fd/cfdzxtsm4fhdzweeh2lkz6ohtvynthwp3cc45dk64gxpt33bg3jp.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_80 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([32, 1024, 96], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_80, %view_128, 2, 64, 96), kwargs = {})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_80, %view_132, 2, 32, 64), kwargs = {})
#   %add_115 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default, %slice_scatter_default_1), kwargs = {})
#   %slice_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_80, %view_136, 2, 0, 32), kwargs = {})
#   %add_146 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_115, %slice_scatter_default_2), kwargs = {})
triton_poi_fused_add_slice_backward_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.full([1], 32, tl.int64)
    tmp7 = tmp0 >= tmp6
    tmp8 = tmp0 < tmp1
    tmp9 = tmp7 & tmp8
    tmp10 = tl.load(in_ptr1 + ((-32) + x0 + (32*x1)), tmp9, other=0.0).to(tl.float32)
    tmp11 = tl.where(tmp9, tmp10, tmp4)
    tmp12 = tmp5 + tmp11
    tmp13 = tmp0 < tmp6
    tmp14 = tl.load(in_ptr2 + (x0 + (32*x1)), tmp13, other=0.0).to(tl.float32)
    tmp15 = tl.where(tmp13, tmp14, tmp4)
    tmp16 = tmp12 + tmp15
    tl.store(out_ptr0 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ay/caymerbsyi37o5u7ynmbdwvqn6gczkdp5arnl4ychzmqxomflsx5.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_159 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_7, torch.float32), kwargs = {})
triton_poi_fused__to_copy_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/uy/cuyvuo23hrxekf7rxzb4pchny2rc4o6hplnfu4dyndzsk5lktc6a.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   y => mul
# Graph fragment:
#   %convert_element_type_158 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_138, torch.float32), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %rsqrt), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_158, %mul), kwargs = {})
#   %sum_46 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_152, [0, 1], True), kwargs = {dtype: torch.float32})
triton_red_fused__to_copy_mul_sum_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mul_sum_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + (384*r2) + (49152*x1)), rmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/cx/ccxr3p3uyheybql4bb3xlc263t6zvtc6slvqluc45szeeif4ctqi.py
# Topologically Sorted Source Nodes: [y], Original ATen: [aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   y => mul
# Graph fragment:
#   %convert_element_type_158 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_138, torch.float32), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %rsqrt), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_158, %mul), kwargs = {})
#   %sum_46 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_152, [0, 1], True), kwargs = {dtype: torch.float32})
triton_red_fused__to_copy_mul_sum_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mul_sum_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/ta/ctaw5quyr4zmtewdwnvt3pls5pypz4lfcwjwj3ez5vmt7ot5ke3a.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.div, aten.pow, aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_158 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_138, torch.float32), kwargs = {})
#   %mul_153 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_158, %primals_2), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_153, %primals_1), kwargs = {})
#   %mul_155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_153, %rsqrt), kwargs = {})
#   %sum_47 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_154, [2], True), kwargs = {dtype: torch.float32})
#   %div_9 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_6, 384), kwargs = {})
#   %pow_12 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_1, 1.0), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_12, 2.0), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_9, %mul_158), kwargs = {})
#   %add_147 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_155, %mul_159), kwargs = {})
triton_per_fused__to_copy_add_div_mul_pow_sum_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_pow_sum_8', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
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
    tmp11 = tmp3 * tmp10
    tmp12 = -0.5
    tmp13 = tmp9 * tmp12
    tmp14 = tmp10 * tmp10
    tmp15 = tmp14 * tmp10
    tmp16 = tmp13 * tmp15
    tmp17 = 0.0026041666666666665
    tmp18 = tmp16 * tmp17
    tmp19 = 2.0
    tmp20 = tmp4 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp11 + tmp21
    tl.store(out_ptr1 + (r1 + (384*x0)), tmp22, rmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, rsqrt, view, view_10, bitwise_and_2, bitwise_and_3, bitwise_and_4, bitwise_and_5, bitwise_and_6, bitwise_and_7, bitwise_and_8, bitwise_and_9, bitwise_and_10, bitwise_and_11, bitwise_and_12, bitwise_and_13, bitwise_and_14, bitwise_and_15, mul_33, unsqueeze_35, unsqueeze_36, mul_39, mul_41, mul_43, mul_45, mul_47, mul_49, mul_51, mul_53, mul_55, mul_57, mul_59, mul_61, mul_63, mul_65, mul_67, unsqueeze_72, mul_73, mul_75, mul_77, mul_79, mul_81, mul_83, mul_85, mul_87, mul_89, mul_91, mul_93, mul_95, mul_97, mul_99, mul_101, unsqueeze_109, permute_1, permute_2, permute_3, permute_6, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, 1024, 384), (393216, 384, 1))
    assert_size_stride(primals_2, (384, ), (1, ))
    assert_size_stride(rsqrt, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view, (32768, 384), (384, 1))
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
    assert_size_stride(permute_1, (512, 64, 64), (4096, 1, 64))
    assert_size_stride(permute_2, (512, 64, 64), (4096, 1, 64))
    assert_size_stride(permute_3, (512, 64, 64), (4096, 1, 64))
    assert_size_stride(permute_6, (96, 384), (384, 1))
    assert_size_stride(tangents_1, (32, 1024, 96), (98304, 96, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf1 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf3 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf4 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf6 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf7 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf9 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf10 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf12 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf13 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf15 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf16 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf18 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf19 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf21 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf24 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf25 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf27 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf28 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf30 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf31 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf33 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf34 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf36 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf37 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf39 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf40 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf42 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf43 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        buf45 = empty_strided_cuda((32, 1, 32), (32, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.select_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mul_select_backward_sum_0.run(tangents_1, unsqueeze_36, unsqueeze_109, mul_101, mul_99, mul_97, mul_95, mul_93, mul_91, mul_89, mul_87, mul_85, mul_83, mul_81, mul_79, mul_77, mul_75, mul_73, unsqueeze_72, mul_67, mul_65, mul_63, mul_61, mul_59, mul_57, mul_55, mul_53, mul_51, mul_49, mul_47, mul_45, mul_43, mul_41, mul_39, buf0, buf1, buf3, buf4, buf6, buf7, buf9, buf10, buf12, buf13, buf15, buf16, buf18, buf19, buf21, buf24, buf25, buf27, buf28, buf30, buf31, buf33, buf34, buf36, buf37, buf39, buf40, buf42, buf43, buf45, 1024, 64, grid=grid(1024), stream=stream0)
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
        buf2 = empty_strided_cuda((32, 16, 64, 32), (32768, 2048, 32, 1), torch.bfloat16)
        buf5 = buf2; del buf2  # reuse
        buf8 = buf5; del buf5  # reuse
        buf11 = buf8; del buf8  # reuse
        buf14 = buf11; del buf11  # reuse
        buf17 = buf14; del buf14  # reuse
        buf20 = buf17; del buf17  # reuse
        buf22 = buf20; del buf20  # reuse
        buf26 = empty_strided_cuda((32, 16, 64, 32), (32768, 2048, 32, 1), torch.bfloat16)
        buf29 = buf26; del buf26  # reuse
        buf32 = buf29; del buf29  # reuse
        buf35 = buf32; del buf32  # reuse
        buf38 = buf35; del buf35  # reuse
        buf41 = buf38; del buf38  # reuse
        buf44 = buf41; del buf41  # reuse
        buf46 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.select_backward, aten.add]
        triton_poi_fused__to_copy_add_select_backward_1.run(buf22, buf46, tangents_1, unsqueeze_36, unsqueeze_109, buf0, buf1, buf3, buf4, buf6, buf7, buf9, buf10, buf12, buf13, buf15, buf16, buf18, buf19, buf21, unsqueeze_72, buf24, buf25, buf27, buf28, buf30, buf31, buf33, buf34, buf36, buf37, buf39, buf40, buf42, buf43, buf45, 1048576, grid=grid(1048576), stream=stream0)
        del buf0
        del buf1
        del buf10
        del buf12
        del buf13
        del buf15
        del buf16
        del buf18
        del buf19
        del buf21
        del buf24
        del buf25
        del buf27
        del buf28
        del buf3
        del unsqueeze_109
        del unsqueeze_72
        buf23 = empty_strided_cuda((512, 64, 32), (2048, 32, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_1, reinterpret_tensor(buf22, (512, 64, 32), (2048, 32, 1), 0), out=buf23)
        del permute_1
        buf47 = reinterpret_tensor(buf22, (512, 64, 32), (2048, 32, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_2, reinterpret_tensor(buf46, (512, 64, 32), (2048, 32, 1), 0), out=buf47)
        del permute_2
        buf48 = buf9; del buf9  # reuse
        buf49 = buf7; del buf7  # reuse
        buf51 = buf6; del buf6  # reuse
        buf52 = buf45; del buf45  # reuse
        buf54 = buf43; del buf43  # reuse
        buf55 = buf42; del buf42  # reuse
        buf57 = buf40; del buf40  # reuse
        buf58 = buf4; del buf4  # reuse
        buf60 = buf39; del buf39  # reuse
        buf61 = buf37; del buf37  # reuse
        buf63 = buf36; del buf36  # reuse
        buf64 = buf34; del buf34  # reuse
        buf66 = buf33; del buf33  # reuse
        buf67 = buf31; del buf31  # reuse
        buf69 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [mul_31, mul_29, mul_27, mul_25, mul_23, mul_21, mul_19, mul_17, mul_15, mul_13, mul_11, mul_9, mul_7, mul_5], Original ATen: [aten.select_backward, aten.mul, aten.sum, aten.add]
        triton_per_fused_add_mul_select_backward_sum_2.run(tangents_1, unsqueeze_36, unsqueeze_35, mul_33, bitwise_and_15, view_10, bitwise_and_14, bitwise_and_13, bitwise_and_12, bitwise_and_11, bitwise_and_10, bitwise_and_9, bitwise_and_8, bitwise_and_7, bitwise_and_6, bitwise_and_5, bitwise_and_4, bitwise_and_3, bitwise_and_2, buf48, buf49, buf51, buf52, buf54, buf55, buf57, buf58, buf60, buf61, buf63, buf64, buf66, buf67, buf69, 1024, 64, grid=grid(1024), stream=stream0)
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
        buf50 = buf46; del buf46  # reuse
        buf53 = buf50; del buf50  # reuse
        buf56 = buf53; del buf53  # reuse
        buf59 = buf56; del buf56  # reuse
        buf62 = buf59; del buf59  # reuse
        buf65 = buf62; del buf62  # reuse
        buf68 = buf65; del buf65  # reuse
        buf70 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward, aten._to_copy, aten.add]
        triton_poi_fused__to_copy_add_select_backward_3.run(buf70, tangents_1, unsqueeze_36, unsqueeze_35, buf48, buf49, buf51, buf52, buf54, buf55, buf57, buf58, buf60, buf61, buf63, buf64, buf66, buf67, buf69, 1048576, grid=grid(1048576), stream=stream0)
        del buf48
        del buf49
        del buf51
        del buf52
        del buf54
        del buf55
        del buf57
        del buf58
        del buf60
        del buf61
        del buf63
        del buf64
        del buf66
        del buf67
        del buf69
        del tangents_1
        del unsqueeze_35
        del unsqueeze_36
        buf71 = empty_strided_cuda((512, 64, 32), (2048, 32, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_3, reinterpret_tensor(buf70, (512, 64, 32), (2048, 32, 1), 0), out=buf71)
        del buf70
        del permute_3
        buf72 = empty_strided_cuda((32, 1024, 96), (98304, 96, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten.add]
        triton_poi_fused_add_slice_backward_4.run(buf23, buf47, buf71, buf72, 3145728, grid=grid(3145728), stream=stream0)
        del buf23
        del buf47
        del buf71
        buf73 = empty_strided_cuda((96, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (96, 32768), (1, 96), 0), view, out=buf73)
        del view
        buf74 = empty_strided_cuda((32768, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (32768, 96), (96, 1), 0), permute_6, out=buf74)
        del buf72
        del permute_6
        buf75 = empty_strided_cuda((96, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_5.run(buf73, buf75, 36864, grid=grid(36864), stream=stream0)
        del buf73
        buf76 = empty_strided_cuda((1, 1, 384, 256), (98304, 98304, 1, 384), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused__to_copy_mul_sum_6.run(buf74, primals_1, rsqrt, buf76, 98304, 128, grid=grid(98304), stream=stream0)
        buf77 = empty_strided_cuda((1, 1, 384), (384, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused__to_copy_mul_sum_7.run(buf76, buf77, 384, 256, grid=grid(384), stream=stream0)
        del buf76
        buf79 = empty_strided_cuda((32, 1024, 384), (393216, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.div, aten.pow, aten.add]
        triton_per_fused__to_copy_add_div_mul_pow_sum_8.run(buf74, primals_2, primals_1, rsqrt, buf79, 32768, 384, grid=grid(32768), stream=stream0)
        del buf74
        del primals_1
        del primals_2
        del rsqrt
    return (buf79, reinterpret_tensor(buf77, (384, ), (1, ), 0), buf75, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 1024, 384), (393216, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((32768, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
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
    permute_1 = rand_strided((512, 64, 64), (4096, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_2 = rand_strided((512, 64, 64), (4096, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_3 = rand_strided((512, 64, 64), (4096, 1, 64), device='cuda:0', dtype=torch.bfloat16)
    permute_6 = rand_strided((96, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((32, 1024, 96), (98304, 96, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, rsqrt, view, view_10, bitwise_and_2, bitwise_and_3, bitwise_and_4, bitwise_and_5, bitwise_and_6, bitwise_and_7, bitwise_and_8, bitwise_and_9, bitwise_and_10, bitwise_and_11, bitwise_and_12, bitwise_and_13, bitwise_and_14, bitwise_and_15, mul_33, unsqueeze_35, unsqueeze_36, mul_39, mul_41, mul_43, mul_45, mul_47, mul_49, mul_51, mul_53, mul_55, mul_57, mul_59, mul_61, mul_63, mul_65, mul_67, unsqueeze_72, mul_73, mul_75, mul_77, mul_79, mul_81, mul_83, mul_85, mul_87, mul_89, mul_91, mul_93, mul_95, mul_97, mul_99, mul_101, unsqueeze_109, permute_1, permute_2, permute_3, permute_6, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
