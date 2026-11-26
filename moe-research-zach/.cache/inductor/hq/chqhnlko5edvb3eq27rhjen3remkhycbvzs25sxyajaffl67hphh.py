# AOT ID: ['6_inference']
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/je/cjep42ar4lz2o5fn2hp72eyhtcwuxu252h6zcx2gbrt4wp6wv5vr.py
# Topologically Sorted Source Nodes: [sub, lag, ge, tensor, clamp_1, pow_1, zeros, base, eq, same, to_1, mul, sub_2, lag_1, ge_19, tensor_3, clamp_2, pow_4, zeros_2, base_1, eq_17, same_1, to_22, mul_34], Original ATen: [aten.sub, aten._to_copy, aten.ge, aten.lift_fresh, aten.clamp, aten.pow, aten.zeros, aten.where, aten.eq, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   base => where
#   base_1 => where_1
#   clamp_1 => clamp_min_1
#   clamp_2 => clamp_min_2
#   eq => eq
#   eq_17 => eq_17
#   ge => ge
#   ge_19 => ge_19
#   lag => convert_element_type
#   lag_1 => convert_element_type_21
#   mul => mul
#   mul_34 => mul_34
#   pow_1 => pow_1
#   pow_4 => pow_4
#   same => bitwise_and
#   same_1 => bitwise_and_17
#   sub => sub
#   sub_2 => sub_2
#   tensor => full_default
#   tensor_3 => full_default_6
#   to_1 => convert_element_type_1
#   to_22 => convert_element_type_22
#   zeros => full_default_1
#   zeros_2 => full_default_7
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_2, %view_3), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub, torch.float32), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convert_element_type, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.8408964276313782), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type, 0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%full_default, %clamp_min_1), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %pow_1, %full_default_1), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %bitwise_and : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq, %unsqueeze_2), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bitwise_and, torch.float32), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, %convert_element_type_1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_43, %view_44), kwargs = {})
#   %convert_element_type_21 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_2, torch.float32), kwargs = {})
#   %ge_19 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convert_element_type_21, 0), kwargs = {})
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.9892280101776123), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_21, 0), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%full_default_6, %clamp_min_2), kwargs = {})
#   %full_default_7 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge_19, %pow_4, %full_default_7), kwargs = {})
#   %eq_17 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze_37, %unsqueeze_38), kwargs = {})
#   %bitwise_and_17 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_17, %unsqueeze_39), kwargs = {})
#   %convert_element_type_22 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bitwise_and_17, torch.float32), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_1, %convert_element_type_22), kwargs = {})
triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_lift_fresh_mul_pow_sub_where_zeros_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_lift_fresh_mul_pow_sub_where_zeros_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 64
    x3 = (xindex // 64)
    x2 = (xindex // 4096)
    x4 = xindex
    tmp9 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp0 = x1 + ((-1)*x0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.0
    tmp3 = tmp1 >= tmp2
    tmp4 = 0.8408964276313782
    tmp5 = ((0.0) * ((0.0) >= (x1 + ((-1)*x0))) + (x1 + ((-1)*x0)) * ((x1 + ((-1)*x0)) > (0.0)))
    tmp6 = tmp5.to(tl.float32)
    tmp7 = libdevice.pow(tmp4, tmp6)
    tmp8 = tl.where(tmp3, tmp7, tmp2)
    tmp11 = tmp9 == tmp10
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tmp9 >= tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp8 * tmp15
    tmp17 = 0.9892280101776123
    tmp18 = libdevice.pow(tmp17, tmp6)
    tmp19 = tl.where(tmp3, tmp18, tmp2)
    tmp20 = tmp19 * tmp15
    tl.store(out_ptr0 + (x4), tmp16, None)
    tl.store(out_ptr1 + (x4), tmp20, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/v4/cv4dwznmgm3jpwzzg7x7ugm3yixf2q4qoh6tgx7q3lrxjavgj3jp.py
# Topologically Sorted Source Nodes: [getitem], Original ATen: [aten.index]
# Source node to ATen node mapping:
#   getitem => index
# Graph fragment:
#   %index : [num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg1_1, [%clamp_min]), kwargs = {})
triton_poi_fused_index_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 32)
    x0 = xindex % 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = tl.full([XBLOCK], 50304, tl.int32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert((0 <= tmp6) & (tmp6 < 50304), "index out of bounds: 0 <= tmp6 < 50304")
    tmp8 = tl.load(in_ptr1 + (x0 + (32*tmp6)), None)
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/qw/cqwhhebhj3lq2qjnxifeofvxajpxtvkdhsaxri5eieawyexo6m5g.py
# Topologically Sorted Source Nodes: [mul_3, mul_4, S_1, stack, mul_37, mul_38, S_18, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_1 => add_2
#   S_18 => add_20
#   mul_3 => mul_3
#   mul_37 => mul_37
#   mul_38 => mul_38
#   mul_4 => mul_4
#   stack => cat
#   stack_1 => cat_1
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_5, %view_10), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %unsqueeze_6), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_8, %mul_4), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_1, %add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16], 1), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_42, %view_51), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_43), kwargs = {})
#   %add_20 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_101, %mul_38), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_19, %add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34], 1), kwargs = {})
triton_poi_fused_add_mul_stack_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x1 = (xindex // 32) % 64
    x0 = xindex % 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2048 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (64 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (63 + (1024*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (2016 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr2 + (2048 + x3 + (32768*x2)), None)
    tmp31 = tl.load(in_ptr2 + (2016 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp3 = tmp1 == tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp1 >= tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 0.8408964276313782
    tmp9 = 1 + x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.pow(tmp8, tmp10)
    tmp12 = tmp7 * tmp11
    tmp14 = tl.full([1], -2, tl.int64)
    tmp15 = tmp2 == tmp14
    tmp16 = tmp2 >= tmp4
    tmp17 = tmp15 & tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 64.0
    tmp20 = libdevice.pow(tmp8, tmp19)
    tmp21 = tmp18 * tmp20
    tmp22 = 0.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp13 + tmp23
    tmp25 = tmp12 * tmp24
    tmp26 = tmp0 + tmp25
    tmp28 = 0.9892280101776123
    tmp29 = libdevice.pow(tmp28, tmp10)
    tmp30 = tmp7 * tmp29
    tmp32 = libdevice.pow(tmp28, tmp19)
    tmp33 = tmp18 * tmp32
    tmp34 = tmp33 * tmp22
    tmp35 = tmp31 + tmp34
    tmp36 = tmp30 * tmp35
    tmp37 = tmp27 + tmp36
    tl.store(out_ptr0 + (x4), tmp26, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp26, None)
    tl.store(out_ptr2 + (x4), tmp37, None)
    tl.store(out_ptr3 + (x3 + (32768*x2)), tmp37, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/rc/crcoab4upjpvp7llbrdmghsqbmkgf4d5jdokb5oqfxauz6m3pifa.py
# Topologically Sorted Source Nodes: [mul_7, mul_5, mul_6, S_2, mul_8, S_3, stack, mul_41, mul_39, mul_40, S_19, mul_42, S_20, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_19 => add_21
#   S_2 => add_3
#   S_20 => add_22
#   S_3 => add_4
#   mul_39 => mul_39
#   mul_40 => mul_40
#   mul_41 => mul_41
#   mul_42 => mul_42
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   stack => cat
#   stack_1 => cat_1
# Graph fragment:
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_9, %view_14), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_7, %view_12), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %unsqueeze_8), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_14, %mul_6), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_10), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_20, %mul_8), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_1, %add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16], 1), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_46, %view_55), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_44, %view_53), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_39, %unsqueeze_45), kwargs = {})
#   %add_21 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_107, %mul_40), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_41, %unsqueeze_47), kwargs = {})
#   %add_22 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_113, %mul_42), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_19, %add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34], 1), kwargs = {})
triton_poi_fused_add_mul_stack_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x1 = (xindex // 32) % 64
    x0 = xindex % 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (6144 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (192 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (191 + (1024*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (6112 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (127 + (1024*x2)), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr3 + (6144 + x3 + (32768*x2)), None)
    tmp31 = tl.load(in_ptr3 + (6112 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (4096 + x3 + (32768*x2)), None)
    tmp40 = tl.load(in_ptr1 + (128 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr3 + (4096 + x3 + (32768*x2)), None)
    tmp3 = tmp1 == tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp1 >= tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 0.8408964276313782
    tmp9 = 1 + x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.pow(tmp8, tmp10)
    tmp12 = tmp7 * tmp11
    tmp15 = tmp2 == tmp14
    tmp16 = tmp2 >= tmp4
    tmp17 = tmp15 & tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 64.0
    tmp20 = libdevice.pow(tmp8, tmp19)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp21 * tmp22
    tmp24 = tmp13 + tmp23
    tmp25 = tmp12 * tmp24
    tmp26 = tmp0 + tmp25
    tmp28 = 0.9892280101776123
    tmp29 = libdevice.pow(tmp28, tmp10)
    tmp30 = tmp7 * tmp29
    tmp32 = libdevice.pow(tmp28, tmp19)
    tmp33 = tmp18 * tmp32
    tmp35 = tmp33 * tmp34
    tmp36 = tmp31 + tmp35
    tmp37 = tmp30 * tmp36
    tmp38 = tmp27 + tmp37
    tmp41 = tmp40 == tmp14
    tmp42 = tmp40 >= tmp4
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp44 * tmp11
    tmp46 = tmp45 * tmp22
    tmp47 = tmp39 + tmp46
    tmp49 = tmp44 * tmp29
    tmp50 = tmp49 * tmp34
    tmp51 = tmp48 + tmp50
    tl.store(out_ptr0 + (x4), tmp26, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp26, None)
    tl.store(out_ptr2 + (x4), tmp38, None)
    tl.store(out_ptr3 + (x3 + (32768*x2)), tmp38, None)
    tl.store(out_ptr4 + (x3 + (32768*x2)), tmp47, None)
    tl.store(out_ptr5 + (x3 + (32768*x2)), tmp51, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/rh/crhusemuo2cfu2lqu6zhdfa5pf4zy57cblioupft5nddynenuwwd.py
# Topologically Sorted Source Nodes: [mul_11, mul_9, mul_10, S_4, mul_12, S_5, stack, mul_45, mul_43, mul_44, S_21, mul_46, S_22, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_21 => add_23
#   S_22 => add_24
#   S_4 => add_5
#   S_5 => add_6
#   mul_10 => mul_10
#   mul_11 => mul_11
#   mul_12 => mul_12
#   mul_43 => mul_43
#   mul_44 => mul_44
#   mul_45 => mul_45
#   mul_46 => mul_46
#   mul_9 => mul_9
#   stack => cat
#   stack_1 => cat_1
# Graph fragment:
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_13, %view_18), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_11, %view_16), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %unsqueeze_12), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_26, %mul_10), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %unsqueeze_14), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mul_12), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_1, %add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16], 1), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_50, %view_59), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_48, %view_57), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_49), kwargs = {})
#   %add_23 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_119, %mul_44), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_45, %unsqueeze_51), kwargs = {})
#   %add_24 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_125, %mul_46), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_19, %add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34], 1), kwargs = {})
triton_poi_fused_add_mul_stack_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x1 = (xindex // 32) % 64
    x0 = xindex % 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (10240 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (320 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (319 + (1024*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (10208 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (255 + (1024*x2)), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr3 + (10240 + x3 + (32768*x2)), None)
    tmp31 = tl.load(in_ptr3 + (10208 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (8192 + x3 + (32768*x2)), None)
    tmp40 = tl.load(in_ptr1 + (256 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr3 + (8192 + x3 + (32768*x2)), None)
    tmp3 = tmp1 == tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp1 >= tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 0.8408964276313782
    tmp9 = 1 + x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.pow(tmp8, tmp10)
    tmp12 = tmp7 * tmp11
    tmp15 = tmp2 == tmp14
    tmp16 = tmp2 >= tmp4
    tmp17 = tmp15 & tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 64.0
    tmp20 = libdevice.pow(tmp8, tmp19)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp21 * tmp22
    tmp24 = tmp13 + tmp23
    tmp25 = tmp12 * tmp24
    tmp26 = tmp0 + tmp25
    tmp28 = 0.9892280101776123
    tmp29 = libdevice.pow(tmp28, tmp10)
    tmp30 = tmp7 * tmp29
    tmp32 = libdevice.pow(tmp28, tmp19)
    tmp33 = tmp18 * tmp32
    tmp35 = tmp33 * tmp34
    tmp36 = tmp31 + tmp35
    tmp37 = tmp30 * tmp36
    tmp38 = tmp27 + tmp37
    tmp41 = tmp40 == tmp14
    tmp42 = tmp40 >= tmp4
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp44 * tmp11
    tmp46 = tmp45 * tmp22
    tmp47 = tmp39 + tmp46
    tmp49 = tmp44 * tmp29
    tmp50 = tmp49 * tmp34
    tmp51 = tmp48 + tmp50
    tl.store(out_ptr0 + (x4), tmp26, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp26, None)
    tl.store(out_ptr2 + (x4), tmp38, None)
    tl.store(out_ptr3 + (x3 + (32768*x2)), tmp38, None)
    tl.store(out_ptr4 + (x3 + (32768*x2)), tmp47, None)
    tl.store(out_ptr5 + (x3 + (32768*x2)), tmp51, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/b6/cb6tb2buc5emtzfpnmuazngy6d7ygt5ykzlpby7toszqatzle4lv.py
# Topologically Sorted Source Nodes: [mul_15, mul_13, mul_14, S_6, mul_16, S_7, stack, mul_49, mul_47, mul_48, S_23, mul_50, S_24, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_23 => add_25
#   S_24 => add_26
#   S_6 => add_7
#   S_7 => add_8
#   mul_13 => mul_13
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_16 => mul_16
#   mul_47 => mul_47
#   mul_48 => mul_48
#   mul_49 => mul_49
#   mul_50 => mul_50
#   stack => cat
#   stack_1 => cat_1
# Graph fragment:
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_17, %view_22), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_15, %view_20), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_16), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_38, %mul_14), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %unsqueeze_18), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_44, %mul_16), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_1, %add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16], 1), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_54, %view_63), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_52, %view_61), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_47, %unsqueeze_53), kwargs = {})
#   %add_25 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_131, %mul_48), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_55), kwargs = {})
#   %add_26 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_137, %mul_50), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_19, %add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34], 1), kwargs = {})
triton_poi_fused_add_mul_stack_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x1 = (xindex // 32) % 64
    x0 = xindex % 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (14336 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (448 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (447 + (1024*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (14304 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (383 + (1024*x2)), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr3 + (14336 + x3 + (32768*x2)), None)
    tmp31 = tl.load(in_ptr3 + (14304 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (12288 + x3 + (32768*x2)), None)
    tmp40 = tl.load(in_ptr1 + (384 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr3 + (12288 + x3 + (32768*x2)), None)
    tmp3 = tmp1 == tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp1 >= tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 0.8408964276313782
    tmp9 = 1 + x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.pow(tmp8, tmp10)
    tmp12 = tmp7 * tmp11
    tmp15 = tmp2 == tmp14
    tmp16 = tmp2 >= tmp4
    tmp17 = tmp15 & tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 64.0
    tmp20 = libdevice.pow(tmp8, tmp19)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp21 * tmp22
    tmp24 = tmp13 + tmp23
    tmp25 = tmp12 * tmp24
    tmp26 = tmp0 + tmp25
    tmp28 = 0.9892280101776123
    tmp29 = libdevice.pow(tmp28, tmp10)
    tmp30 = tmp7 * tmp29
    tmp32 = libdevice.pow(tmp28, tmp19)
    tmp33 = tmp18 * tmp32
    tmp35 = tmp33 * tmp34
    tmp36 = tmp31 + tmp35
    tmp37 = tmp30 * tmp36
    tmp38 = tmp27 + tmp37
    tmp41 = tmp40 == tmp14
    tmp42 = tmp40 >= tmp4
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp44 * tmp11
    tmp46 = tmp45 * tmp22
    tmp47 = tmp39 + tmp46
    tmp49 = tmp44 * tmp29
    tmp50 = tmp49 * tmp34
    tmp51 = tmp48 + tmp50
    tl.store(out_ptr0 + (x4), tmp26, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp26, None)
    tl.store(out_ptr2 + (x4), tmp38, None)
    tl.store(out_ptr3 + (x3 + (32768*x2)), tmp38, None)
    tl.store(out_ptr4 + (x3 + (32768*x2)), tmp47, None)
    tl.store(out_ptr5 + (x3 + (32768*x2)), tmp51, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/qg/cqgarrpjyhqi54kkcqcenoj3amwhlog5mkihh7mtmku6ptxutqfa.py
# Topologically Sorted Source Nodes: [mul_19, mul_17, mul_18, S_8, mul_20, S_9, stack, mul_53, mul_51, mul_52, S_25, mul_54, S_26, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_25 => add_27
#   S_26 => add_28
#   S_8 => add_9
#   S_9 => add_10
#   mul_17 => mul_17
#   mul_18 => mul_18
#   mul_19 => mul_19
#   mul_20 => mul_20
#   mul_51 => mul_51
#   mul_52 => mul_52
#   mul_53 => mul_53
#   mul_54 => mul_54
#   stack => cat
#   stack_1 => cat_1
# Graph fragment:
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_21, %view_26), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_19, %view_24), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %unsqueeze_20), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_50, %mul_18), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_22), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_56, %mul_20), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_1, %add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16], 1), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_58, %view_67), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_56, %view_65), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_51, %unsqueeze_57), kwargs = {})
#   %add_27 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_143, %mul_52), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_53, %unsqueeze_59), kwargs = {})
#   %add_28 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_149, %mul_54), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_19, %add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34], 1), kwargs = {})
triton_poi_fused_add_mul_stack_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x1 = (xindex // 32) % 64
    x0 = xindex % 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (18432 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (576 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (575 + (1024*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (18400 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (511 + (1024*x2)), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr3 + (18432 + x3 + (32768*x2)), None)
    tmp31 = tl.load(in_ptr3 + (18400 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (16384 + x3 + (32768*x2)), None)
    tmp40 = tl.load(in_ptr1 + (512 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr3 + (16384 + x3 + (32768*x2)), None)
    tmp3 = tmp1 == tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp1 >= tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 0.8408964276313782
    tmp9 = 1 + x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.pow(tmp8, tmp10)
    tmp12 = tmp7 * tmp11
    tmp15 = tmp2 == tmp14
    tmp16 = tmp2 >= tmp4
    tmp17 = tmp15 & tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 64.0
    tmp20 = libdevice.pow(tmp8, tmp19)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp21 * tmp22
    tmp24 = tmp13 + tmp23
    tmp25 = tmp12 * tmp24
    tmp26 = tmp0 + tmp25
    tmp28 = 0.9892280101776123
    tmp29 = libdevice.pow(tmp28, tmp10)
    tmp30 = tmp7 * tmp29
    tmp32 = libdevice.pow(tmp28, tmp19)
    tmp33 = tmp18 * tmp32
    tmp35 = tmp33 * tmp34
    tmp36 = tmp31 + tmp35
    tmp37 = tmp30 * tmp36
    tmp38 = tmp27 + tmp37
    tmp41 = tmp40 == tmp14
    tmp42 = tmp40 >= tmp4
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp44 * tmp11
    tmp46 = tmp45 * tmp22
    tmp47 = tmp39 + tmp46
    tmp49 = tmp44 * tmp29
    tmp50 = tmp49 * tmp34
    tmp51 = tmp48 + tmp50
    tl.store(out_ptr0 + (x4), tmp26, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp26, None)
    tl.store(out_ptr2 + (x4), tmp38, None)
    tl.store(out_ptr3 + (x3 + (32768*x2)), tmp38, None)
    tl.store(out_ptr4 + (x3 + (32768*x2)), tmp47, None)
    tl.store(out_ptr5 + (x3 + (32768*x2)), tmp51, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/gw/cgwvtqzu624wn3xwnka6rjtpbv3chyab5ocoeqo4juegt63xrtfs.py
# Topologically Sorted Source Nodes: [mul_23, mul_21, mul_22, S_10, mul_24, S_11, stack, mul_57, mul_55, mul_56, S_27, mul_58, S_28, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_10 => add_11
#   S_11 => add_12
#   S_27 => add_29
#   S_28 => add_30
#   mul_21 => mul_21
#   mul_22 => mul_22
#   mul_23 => mul_23
#   mul_24 => mul_24
#   mul_55 => mul_55
#   mul_56 => mul_56
#   mul_57 => mul_57
#   mul_58 => mul_58
#   stack => cat
#   stack_1 => cat_1
# Graph fragment:
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_25, %view_30), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_23, %view_28), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %unsqueeze_24), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_62, %mul_22), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %unsqueeze_26), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_68, %mul_24), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_1, %add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16], 1), kwargs = {})
#   %mul_57 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_62, %view_71), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_60, %view_69), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %unsqueeze_61), kwargs = {})
#   %add_29 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_155, %mul_56), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_57, %unsqueeze_63), kwargs = {})
#   %add_30 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_161, %mul_58), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_19, %add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34], 1), kwargs = {})
triton_poi_fused_add_mul_stack_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x1 = (xindex // 32) % 64
    x0 = xindex % 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (22528 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (704 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (703 + (1024*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (22496 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (639 + (1024*x2)), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr3 + (22528 + x3 + (32768*x2)), None)
    tmp31 = tl.load(in_ptr3 + (22496 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (20480 + x3 + (32768*x2)), None)
    tmp40 = tl.load(in_ptr1 + (640 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr3 + (20480 + x3 + (32768*x2)), None)
    tmp3 = tmp1 == tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp1 >= tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 0.8408964276313782
    tmp9 = 1 + x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.pow(tmp8, tmp10)
    tmp12 = tmp7 * tmp11
    tmp15 = tmp2 == tmp14
    tmp16 = tmp2 >= tmp4
    tmp17 = tmp15 & tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 64.0
    tmp20 = libdevice.pow(tmp8, tmp19)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp21 * tmp22
    tmp24 = tmp13 + tmp23
    tmp25 = tmp12 * tmp24
    tmp26 = tmp0 + tmp25
    tmp28 = 0.9892280101776123
    tmp29 = libdevice.pow(tmp28, tmp10)
    tmp30 = tmp7 * tmp29
    tmp32 = libdevice.pow(tmp28, tmp19)
    tmp33 = tmp18 * tmp32
    tmp35 = tmp33 * tmp34
    tmp36 = tmp31 + tmp35
    tmp37 = tmp30 * tmp36
    tmp38 = tmp27 + tmp37
    tmp41 = tmp40 == tmp14
    tmp42 = tmp40 >= tmp4
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp44 * tmp11
    tmp46 = tmp45 * tmp22
    tmp47 = tmp39 + tmp46
    tmp49 = tmp44 * tmp29
    tmp50 = tmp49 * tmp34
    tmp51 = tmp48 + tmp50
    tl.store(out_ptr0 + (x4), tmp26, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp26, None)
    tl.store(out_ptr2 + (x4), tmp38, None)
    tl.store(out_ptr3 + (x3 + (32768*x2)), tmp38, None)
    tl.store(out_ptr4 + (x3 + (32768*x2)), tmp47, None)
    tl.store(out_ptr5 + (x3 + (32768*x2)), tmp51, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/qq/cqqcxyiwfvfhmf7hugstt2xf3gpnqdpk5b4ygshrcz4fiir6waia.py
# Topologically Sorted Source Nodes: [mul_27, mul_25, mul_26, S_12, mul_28, S_13, stack, mul_61, mul_59, mul_60, S_29, mul_62, S_30, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_12 => add_13
#   S_13 => add_14
#   S_29 => add_31
#   S_30 => add_32
#   mul_25 => mul_25
#   mul_26 => mul_26
#   mul_27 => mul_27
#   mul_28 => mul_28
#   mul_59 => mul_59
#   mul_60 => mul_60
#   mul_61 => mul_61
#   mul_62 => mul_62
#   stack => cat
#   stack_1 => cat_1
# Graph fragment:
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_29, %view_34), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_27, %view_32), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_28), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_74, %mul_26), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, %unsqueeze_30), kwargs = {})
#   %add_14 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_80, %mul_28), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_1, %add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16], 1), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_66, %view_75), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_64, %view_73), kwargs = {})
#   %mul_60 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_59, %unsqueeze_65), kwargs = {})
#   %add_31 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_167, %mul_60), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_67), kwargs = {})
#   %add_32 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_173, %mul_62), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_19, %add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34], 1), kwargs = {})
triton_poi_fused_add_mul_stack_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x1 = (xindex // 32) % 64
    x0 = xindex % 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (26624 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (832 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (831 + (1024*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (26592 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (767 + (1024*x2)), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr3 + (26624 + x3 + (32768*x2)), None)
    tmp31 = tl.load(in_ptr3 + (26592 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (24576 + x3 + (32768*x2)), None)
    tmp40 = tl.load(in_ptr1 + (768 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr3 + (24576 + x3 + (32768*x2)), None)
    tmp3 = tmp1 == tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp1 >= tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 0.8408964276313782
    tmp9 = 1 + x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.pow(tmp8, tmp10)
    tmp12 = tmp7 * tmp11
    tmp15 = tmp2 == tmp14
    tmp16 = tmp2 >= tmp4
    tmp17 = tmp15 & tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 64.0
    tmp20 = libdevice.pow(tmp8, tmp19)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp21 * tmp22
    tmp24 = tmp13 + tmp23
    tmp25 = tmp12 * tmp24
    tmp26 = tmp0 + tmp25
    tmp28 = 0.9892280101776123
    tmp29 = libdevice.pow(tmp28, tmp10)
    tmp30 = tmp7 * tmp29
    tmp32 = libdevice.pow(tmp28, tmp19)
    tmp33 = tmp18 * tmp32
    tmp35 = tmp33 * tmp34
    tmp36 = tmp31 + tmp35
    tmp37 = tmp30 * tmp36
    tmp38 = tmp27 + tmp37
    tmp41 = tmp40 == tmp14
    tmp42 = tmp40 >= tmp4
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp44 * tmp11
    tmp46 = tmp45 * tmp22
    tmp47 = tmp39 + tmp46
    tmp49 = tmp44 * tmp29
    tmp50 = tmp49 * tmp34
    tmp51 = tmp48 + tmp50
    tl.store(out_ptr0 + (x4), tmp26, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp26, None)
    tl.store(out_ptr2 + (x4), tmp38, None)
    tl.store(out_ptr3 + (x3 + (32768*x2)), tmp38, None)
    tl.store(out_ptr4 + (x3 + (32768*x2)), tmp47, None)
    tl.store(out_ptr5 + (x3 + (32768*x2)), tmp51, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/fz/cfzsxon6yuqcne3n7gggojie7jhu3d2np4o2fjzfsxgi4zjek44q.py
# Topologically Sorted Source Nodes: [mul_31, mul_29, mul_30, S_14, mul_32, S_15, mul_65, mul_63, mul_64, S_31, mul_66, S_32], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   S_14 => add_15
#   S_15 => add_16
#   S_31 => add_33
#   S_32 => add_34
#   mul_29 => mul_29
#   mul_30 => mul_30
#   mul_31 => mul_31
#   mul_32 => mul_32
#   mul_63 => mul_63
#   mul_64 => mul_64
#   mul_65 => mul_65
#   mul_66 => mul_66
# Graph fragment:
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_33, %view_38), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_31, %view_36), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %unsqueeze_32), kwargs = {})
#   %add_15 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_86, %mul_30), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_34), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_92, %mul_32), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_70, %view_79), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_68, %view_77), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_63, %unsqueeze_69), kwargs = {})
#   %add_33 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_179, %mul_64), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_65, %unsqueeze_71), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_185, %mul_66), kwargs = {})
triton_poi_fused_add_mul_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x1 = (xindex // 32) % 64
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (30720 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (960 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (959 + (1024*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (30688 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (895 + (1024*x2)), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr3 + (30720 + x3 + (32768*x2)), None)
    tmp31 = tl.load(in_ptr3 + (30688 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (28672 + x3 + (32768*x2)), None)
    tmp40 = tl.load(in_ptr1 + (896 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr3 + (28672 + x3 + (32768*x2)), None)
    tmp3 = tmp1 == tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp1 >= tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 0.8408964276313782
    tmp9 = 1 + x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.pow(tmp8, tmp10)
    tmp12 = tmp7 * tmp11
    tmp15 = tmp2 == tmp14
    tmp16 = tmp2 >= tmp4
    tmp17 = tmp15 & tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 64.0
    tmp20 = libdevice.pow(tmp8, tmp19)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp21 * tmp22
    tmp24 = tmp13 + tmp23
    tmp25 = tmp12 * tmp24
    tmp26 = tmp0 + tmp25
    tmp28 = 0.9892280101776123
    tmp29 = libdevice.pow(tmp28, tmp10)
    tmp30 = tmp7 * tmp29
    tmp32 = libdevice.pow(tmp28, tmp19)
    tmp33 = tmp18 * tmp32
    tmp35 = tmp33 * tmp34
    tmp36 = tmp31 + tmp35
    tmp37 = tmp30 * tmp36
    tmp38 = tmp27 + tmp37
    tmp41 = tmp40 == tmp14
    tmp42 = tmp40 >= tmp4
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp44 * tmp11
    tmp46 = tmp45 * tmp22
    tmp47 = tmp39 + tmp46
    tmp49 = tmp44 * tmp29
    tmp50 = tmp49 * tmp34
    tmp51 = tmp48 + tmp50
    tl.store(out_ptr0 + (x3 + (32768*x2)), tmp26, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp38, None)
    tl.store(out_ptr2 + (x3 + (32768*x2)), tmp47, None)
    tl.store(out_ptr3 + (x3 + (32768*x2)), tmp51, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/bf/cbf4uvyyerfj4ntf44rvynu4isngb4p44fsp53yxecpxtspyud4q.py
# Topologically Sorted Source Nodes: [mul_1, unsqueeze_4, mul_2, S, mul_35, unsqueeze_41, mul_36, S_17], Original ATen: [aten.mul, aten.unsqueeze, aten.add]
# Source node to ATen node mapping:
#   S => add_1
#   S_17 => add_19
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_35 => mul_35
#   mul_36 => mul_36
#   unsqueeze_4 => full_default_4
#   unsqueeze_41 => full_default_10
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_3, %view_8), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 1, 32], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %full_default_4), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_2, %mul_2), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_40, %view_49), kwargs = {})
#   %full_default_10 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 1, 32], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_35, %full_default_10), kwargs = {})
#   %add_19 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_95, %mul_36), kwargs = {})
triton_poi_fused_add_mul_unsqueeze_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_unsqueeze_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x1 = (xindex // 32) % 64
    tmp0 = tl.load(in_ptr0 + (x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (x3 + (32768*x2)), None)
    tmp2 = tl.full([1], -2, tl.int64)
    tmp3 = tmp1 == tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp1 >= tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 0.8408964276313782
    tmp9 = 1 + x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.pow(tmp8, tmp10)
    tmp12 = tmp7 * tmp11
    tmp13 = 0.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp0 + tmp14
    tmp17 = 0.9892280101776123
    tmp18 = libdevice.pow(tmp17, tmp10)
    tmp19 = tmp7 * tmp18
    tmp20 = tmp19 * tmp13
    tmp21 = tmp16 + tmp20
    tl.store(out_ptr0 + (x3 + (32768*x2)), tmp15, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/fp/cfpyfn6t4n7wya7ctkhvdwvjtww6mkknrgu665la4n7i5mdtcgqo.py
# Topologically Sorted Source Nodes: [truediv_1, to_20, f, cosine_similarity, truediv_3, to_41, g], Original ATen: [aten.div, aten._to_copy, aten.mul, aten.linalg_vector_norm, aten.clamp_min, aten.sum]
# Source node to ATen node mapping:
#   cosine_similarity => clamp_min_3, clamp_min_4, div_4, div_5, mul_68, pow_10, pow_7, pow_8, pow_9, sum_1, sum_2, sum_3
#   f => mul_33
#   g => mul_67
#   to_20 => convert_element_type_20
#   to_41 => convert_element_type_41
#   truediv_1 => div_1
#   truediv_3 => div_3
# Graph fragment:
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_40, %unsqueeze_35), kwargs = {})
#   %convert_element_type_20 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_36, torch.float32), kwargs = {})
#   %mul_33 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %convert_element_type_20), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_33, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_7, [-1], True), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_8, 1e-06), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_33, %clamp_min_3), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_81, %unsqueeze_72), kwargs = {})
#   %convert_element_type_41 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_73, torch.float32), kwargs = {})
#   %mul_67 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %convert_element_type_41), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%mul_67, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_9, [-1], True), kwargs = {})
#   %pow_10 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %clamp_min_4 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_10, 1e-06), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_67, %clamp_min_4), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_5, %div_4), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_68, [-1]), kwargs = {})
triton_per_fused__to_copy_clamp_min_div_linalg_vector_norm_mul_sum_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_clamp_min_div_linalg_vector_norm_mul_sum_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
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
    tmp0 = tl.load(in_ptr0 + (r1 + (32*x0)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr3 + (r1 + (32*x0)), None)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.8408964276313782
    tmp6 = libdevice.pow(tmp5, tmp4)
    tmp7 = tmp3 - tmp6
    tmp8 = 6.285213507883244
    tmp9 = tmp7 * tmp8
    tmp10 = tmp0 / tmp9
    tmp12 = tl.full([1, 1], 0, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp10 * tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.sum(tmp17, 1)[:, None]
    tmp21 = 0.9892280101776123
    tmp22 = libdevice.pow(tmp21, tmp4)
    tmp23 = tmp3 - tmp22
    tmp24 = 92.83338515052041
    tmp25 = tmp23 * tmp24
    tmp26 = tmp20 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.sum(tmp29, 1)[:, None]
    tmp32 = libdevice.sqrt(tmp19)
    tmp33 = 1e-06
    tmp34 = triton_helpers.maximum(tmp32, tmp33)
    tmp35 = tmp15 / tmp34
    tmp36 = libdevice.sqrt(tmp31)
    tmp37 = triton_helpers.maximum(tmp36, tmp33)
    tmp38 = tmp27 / tmp37
    tmp39 = tmp35 * tmp38
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp42 = tl.sum(tmp40, 1)[:, None]
    tl.store(out_ptr3 + (x0), tmp42, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/5v/c5vyfcvyts4ouxda7bhbj5t3mkwgphxm47nazijwuuo6gwaaiqvd.py
# Topologically Sorted Source Nodes: [prev, d, above, gt_1, prev_1, invert, and__35, mod, eq_34, or_], Original ATen: [aten.zeros_like, aten.rsub, aten.gt, aten.bitwise_and, aten.bitwise_not, aten.remainder, aten.eq, aten.bitwise_or]
# Source node to ATen node mapping:
#   above => gt
#   and__35 => bitwise_and_35
#   d => sub_4
#   eq_34 => eq_34
#   gt_1 => gt_1
#   invert => bitwise_not
#   mod => remainder
#   or_ => bitwise_or
#   prev => full_6
#   prev_1 => bitwise_and_34
# Graph fragment:
#   %full_6 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32, 1024], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %sum_3), kwargs = {})
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%sub_4, 0.8335774540901184), kwargs = {})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_6, %slice_158, 1, 1, 9223372036854775807), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%arg3_1, 0), kwargs = {})
#   %bitwise_and_34 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%slice_scatter_default, %gt_1), kwargs = {})
#   %bitwise_not : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%bitwise_and_34,), kwargs = {})
#   %bitwise_and_35 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%gt, %bitwise_not), kwargs = {})
#   %remainder : [num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%arg3_1, 256), kwargs = {})
#   %eq_34 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%remainder, 0), kwargs = {})
#   %bitwise_or : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_and_35, %eq_34), kwargs = {})
triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_eq_gt_remainder_rsub_zeros_like_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*i1', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_eq_gt_remainder_rsub_zeros_like_12', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp15 = tl.load(in_ptr1 + (x2), None)
    tmp1 = 1.0
    tmp2 = tmp1 - tmp0
    tmp3 = 0.8335774540901184
    tmp4 = tmp2 > tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 1, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.load(in_ptr0 + ((-1) + x2), tmp7, other=0.0)
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 > tmp3
    tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
    tmp12 = tl.where(tmp7, tmp10, tmp11)
    tmp13 = tl.full([1], False, tl.int1)
    tmp14 = tl.where(tmp7, tmp12, tmp13)
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 > tmp16
    tmp18 = tmp14 & tmp17
    tmp19 = tmp18 == 0
    tmp20 = tmp4 & tmp19
    tmp21 = tl.full([1], 256, tl.int64)
    tmp22 = tmp15 % tmp21
    tmp23 = tl.full([1], 0, tl.int32)
    tmp24 = tmp22 != tmp23
    tmp25 = libdevice.signbit(tmp22) if (tmp22).dtype is tl.float32 else tmp22 < 0
    tmp26 = libdevice.signbit(tmp21) if (tmp21).dtype is tl.float32 else tmp21 < 0
    tmp27 = tmp25 != tmp26
    tmp28 = tmp24 & tmp27
    tmp29 = tmp22 + tmp21
    tmp30 = tl.where(tmp28, tmp29, tmp22)
    tmp31 = tmp30 == tmp16
    tmp32 = tmp20 | tmp31
    tl.store(out_ptr0 + (x2), tmp32, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 1024), (1024, 1))
    assert_size_stride(arg1_1, (50304, 32), (32, 1))
    assert_size_stride(arg2_1, (32, 1024), (1024, 1))
    assert_size_stride(arg3_1, (32, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 16, 64, 64), (65536, 4096, 64, 1), torch.float32)
        buf28 = empty_strided_cuda((32, 16, 64, 64), (65536, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sub, lag, ge, tensor, clamp_1, pow_1, zeros, base, eq, same, to_1, mul, sub_2, lag_1, ge_19, tensor_3, clamp_2, pow_4, zeros_2, base_1, eq_17, same_1, to_22, mul_34], Original ATen: [aten.sub, aten._to_copy, aten.ge, aten.lift_fresh, aten.clamp, aten.pow, aten.zeros, aten.where, aten.eq, aten.bitwise_and, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_lift_fresh_mul_pow_sub_where_zeros_0.run(arg2_1, buf0, buf28, 2097152, grid=grid(2097152), stream=stream0)
        buf1 = empty_strided_cuda((32, 1024, 32), (32768, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [getitem], Original ATen: [aten.index]
        triton_poi_fused_index_1.run(arg0_1, arg1_1, buf1, 1048576, grid=grid(1048576), stream=stream0)
        del arg0_1
        del arg1_1
        buf2 = empty_strided_cuda((512, 64, 32), (2048, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [S_in], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (512, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf1, (512, 64, 32), (2048, 32, 1), 0), out=buf2)
        del buf0
        buf29 = empty_strided_cuda((512, 64, 32), (2048, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [S_in_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf28, (512, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf1, (512, 64, 32), (2048, 32, 1), 0), out=buf29)
        del buf28
        buf3 = empty_strided_cuda((32, 64, 32), (2048, 32, 1), torch.float32)
        buf26 = buf1; del buf1  # reuse
        buf12 = reinterpret_tensor(buf26, (32, 64, 32), (32768, 32, 1), 2048)  # alias
        buf30 = empty_strided_cuda((32, 64, 32), (2048, 32, 1), torch.float32)
        buf53 = empty_strided_cuda((32, 1024, 32), (32768, 32, 1), torch.float32)
        buf39 = reinterpret_tensor(buf53, (32, 64, 32), (32768, 32, 1), 2048)  # alias
        # Topologically Sorted Source Nodes: [mul_3, mul_4, S_1, stack, mul_37, mul_38, S_18, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_2.run(buf2, arg2_1, buf29, buf3, buf12, buf30, buf39, 65536, grid=grid(65536), stream=stream0)
        buf4 = empty_strided_cuda((32, 64, 32), (2048, 32, 1), torch.float32)
        buf14 = reinterpret_tensor(buf26, (32, 64, 32), (32768, 32, 1), 6144)  # alias
        buf31 = empty_strided_cuda((32, 64, 32), (2048, 32, 1), torch.float32)
        buf41 = reinterpret_tensor(buf53, (32, 64, 32), (32768, 32, 1), 6144)  # alias
        buf13 = reinterpret_tensor(buf26, (32, 64, 32), (32768, 32, 1), 4096)  # alias
        buf40 = reinterpret_tensor(buf53, (32, 64, 32), (32768, 32, 1), 4096)  # alias
        # Topologically Sorted Source Nodes: [mul_7, mul_5, mul_6, S_2, mul_8, S_3, stack, mul_41, mul_39, mul_40, S_19, mul_42, S_20, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_3.run(buf2, arg2_1, buf3, buf29, buf30, buf4, buf14, buf31, buf41, buf13, buf40, 65536, grid=grid(65536), stream=stream0)
        buf5 = buf30; del buf30  # reuse
        buf16 = reinterpret_tensor(buf26, (32, 64, 32), (32768, 32, 1), 10240)  # alias
        buf32 = buf3; del buf3  # reuse
        buf43 = reinterpret_tensor(buf53, (32, 64, 32), (32768, 32, 1), 10240)  # alias
        buf15 = reinterpret_tensor(buf26, (32, 64, 32), (32768, 32, 1), 8192)  # alias
        buf42 = reinterpret_tensor(buf53, (32, 64, 32), (32768, 32, 1), 8192)  # alias
        # Topologically Sorted Source Nodes: [mul_11, mul_9, mul_10, S_4, mul_12, S_5, stack, mul_45, mul_43, mul_44, S_21, mul_46, S_22, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_4.run(buf2, arg2_1, buf4, buf29, buf31, buf5, buf16, buf32, buf43, buf15, buf42, 65536, grid=grid(65536), stream=stream0)
        buf6 = buf4; del buf4  # reuse
        buf18 = reinterpret_tensor(buf26, (32, 64, 32), (32768, 32, 1), 14336)  # alias
        buf33 = buf31; del buf31  # reuse
        buf45 = reinterpret_tensor(buf53, (32, 64, 32), (32768, 32, 1), 14336)  # alias
        buf17 = reinterpret_tensor(buf26, (32, 64, 32), (32768, 32, 1), 12288)  # alias
        buf44 = reinterpret_tensor(buf53, (32, 64, 32), (32768, 32, 1), 12288)  # alias
        # Topologically Sorted Source Nodes: [mul_15, mul_13, mul_14, S_6, mul_16, S_7, stack, mul_49, mul_47, mul_48, S_23, mul_50, S_24, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_5.run(buf2, arg2_1, buf5, buf29, buf32, buf6, buf18, buf33, buf45, buf17, buf44, 65536, grid=grid(65536), stream=stream0)
        buf7 = buf5; del buf5  # reuse
        buf20 = reinterpret_tensor(buf26, (32, 64, 32), (32768, 32, 1), 18432)  # alias
        buf34 = buf32; del buf32  # reuse
        buf47 = reinterpret_tensor(buf53, (32, 64, 32), (32768, 32, 1), 18432)  # alias
        buf19 = reinterpret_tensor(buf26, (32, 64, 32), (32768, 32, 1), 16384)  # alias
        buf46 = reinterpret_tensor(buf53, (32, 64, 32), (32768, 32, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [mul_19, mul_17, mul_18, S_8, mul_20, S_9, stack, mul_53, mul_51, mul_52, S_25, mul_54, S_26, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_6.run(buf2, arg2_1, buf6, buf29, buf33, buf7, buf20, buf34, buf47, buf19, buf46, 65536, grid=grid(65536), stream=stream0)
        buf8 = buf6; del buf6  # reuse
        buf22 = reinterpret_tensor(buf26, (32, 64, 32), (32768, 32, 1), 22528)  # alias
        buf35 = buf33; del buf33  # reuse
        buf49 = reinterpret_tensor(buf53, (32, 64, 32), (32768, 32, 1), 22528)  # alias
        buf21 = reinterpret_tensor(buf26, (32, 64, 32), (32768, 32, 1), 20480)  # alias
        buf48 = reinterpret_tensor(buf53, (32, 64, 32), (32768, 32, 1), 20480)  # alias
        # Topologically Sorted Source Nodes: [mul_23, mul_21, mul_22, S_10, mul_24, S_11, stack, mul_57, mul_55, mul_56, S_27, mul_58, S_28, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_7.run(buf2, arg2_1, buf7, buf29, buf34, buf8, buf22, buf35, buf49, buf21, buf48, 65536, grid=grid(65536), stream=stream0)
        buf9 = buf7; del buf7  # reuse
        buf24 = reinterpret_tensor(buf26, (32, 64, 32), (32768, 32, 1), 26624)  # alias
        buf36 = buf34; del buf34  # reuse
        buf51 = reinterpret_tensor(buf53, (32, 64, 32), (32768, 32, 1), 26624)  # alias
        buf23 = reinterpret_tensor(buf26, (32, 64, 32), (32768, 32, 1), 24576)  # alias
        buf50 = reinterpret_tensor(buf53, (32, 64, 32), (32768, 32, 1), 24576)  # alias
        # Topologically Sorted Source Nodes: [mul_27, mul_25, mul_26, S_12, mul_28, S_13, stack, mul_61, mul_59, mul_60, S_29, mul_62, S_30, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_8.run(buf2, arg2_1, buf8, buf29, buf35, buf9, buf24, buf36, buf51, buf23, buf50, 65536, grid=grid(65536), stream=stream0)
        del buf35
        del buf8
        buf10 = reinterpret_tensor(buf26, (32, 64, 32), (32768, 32, 1), 30720)  # alias
        buf37 = reinterpret_tensor(buf53, (32, 64, 32), (32768, 32, 1), 30720)  # alias
        buf25 = reinterpret_tensor(buf26, (32, 64, 32), (32768, 32, 1), 28672)  # alias
        buf52 = reinterpret_tensor(buf53, (32, 64, 32), (32768, 32, 1), 28672)  # alias
        # Topologically Sorted Source Nodes: [mul_31, mul_29, mul_30, S_14, mul_32, S_15, mul_65, mul_63, mul_64, S_31, mul_66, S_32], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_9.run(buf2, arg2_1, buf9, buf29, buf36, buf10, buf37, buf25, buf52, 65536, grid=grid(65536), stream=stream0)
        del buf36
        del buf9
        buf11 = reinterpret_tensor(buf26, (32, 64, 32), (32768, 32, 1), 0)  # alias
        buf38 = reinterpret_tensor(buf53, (32, 64, 32), (32768, 32, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [mul_1, unsqueeze_4, mul_2, S, mul_35, unsqueeze_41, mul_36, S_17], Original ATen: [aten.mul, aten.unsqueeze, aten.add]
        triton_poi_fused_add_mul_unsqueeze_10.run(buf2, arg2_1, buf29, buf11, buf38, 65536, grid=grid(65536), stream=stream0)
        del buf2
        del buf29
        buf56 = empty_strided_cuda((32, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [truediv_1, to_20, f, cosine_similarity, truediv_3, to_41, g], Original ATen: [aten.div, aten._to_copy, aten.mul, aten.linalg_vector_norm, aten.clamp_min, aten.sum]
        triton_per_fused__to_copy_clamp_min_div_linalg_vector_norm_mul_sum_11.run(buf26, arg3_1, arg2_1, buf53, buf56, 32768, 32, grid=grid(32768), stream=stream0)
        del arg2_1
        del buf10
        del buf11
        del buf12
        del buf13
        del buf14
        del buf15
        del buf16
        del buf17
        del buf18
        del buf19
        del buf20
        del buf21
        del buf22
        del buf23
        del buf24
        del buf25
        del buf26
        del buf37
        del buf38
        del buf39
        del buf40
        del buf41
        del buf42
        del buf43
        del buf44
        del buf45
        del buf46
        del buf47
        del buf48
        del buf49
        del buf50
        del buf51
        del buf52
        del buf53
        buf57 = empty_strided_cuda((32, 1024), (1024, 1), torch.bool)
        # Topologically Sorted Source Nodes: [prev, d, above, gt_1, prev_1, invert, and__35, mod, eq_34, or_], Original ATen: [aten.zeros_like, aten.rsub, aten.gt, aten.bitwise_and, aten.bitwise_not, aten.remainder, aten.eq, aten.bitwise_or]
        triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_eq_gt_remainder_rsub_zeros_like_12.run(buf56, arg3_1, buf57, 32768, grid=grid(32768), stream=stream0)
        del arg3_1
        del buf56
    return (buf57, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((50304, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    arg3_1 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
