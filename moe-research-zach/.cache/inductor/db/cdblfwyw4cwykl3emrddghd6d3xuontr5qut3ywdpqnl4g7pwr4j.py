# AOT ID: ['5_forward']
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/v5/cv5gocb54v5t4tgetdmeuxpcdf4ic4orfyy4odxeqd4x6njsk2io.py
# Topologically Sorted Source Nodes: [pow_1, mean, add, rsqrt, y, mul_1], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   mean => mean
#   mul_1 => mul_1
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   y => mul
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_1, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_2), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask, other=0.0)
    tmp12 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.where(rmask, tmp2, 0)
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp6 = 384.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-06
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp0 * tmp10
    tmp13 = tmp11 * tmp12
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr0 + (r1 + (384*x0)), tmp13, rmask)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/qp/cqprdlyihogkxhgtyqygmtypsf2v6o7y7qtdhap7fupz6v2nxjk6.py
# Topologically Sorted Source Nodes: [ge_1, ge_18], Original ATen: [aten.ge]
# Source node to ATen node mapping:
#   ge_1 => ge_1
#   ge_18 => ge_18
# Graph fragment:
#   %ge_1 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%view_3, 0), kwargs = {})
#   %ge_18 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%primals_4, 0), kwargs = {})
triton_poi_fused_ge_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*i1', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ge_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
    tl.store(out_ptr1 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/wr/cwrtkjcftfwho36em5fk7dwlq5lzx4fvvvno4qtfn2ndw5ewgxg3.py
# Topologically Sorted Source Nodes: [sub, lag, ge, tensor, clamp, pow_2, zeros, base, eq, same, to_3, mul_2], Original ATen: [aten.sub, aten._to_copy, aten.ge, aten.lift_fresh, aten.clamp, aten.pow, aten.zeros, aten.where, aten.eq, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   base => where
#   clamp => clamp_min
#   eq => eq
#   ge => ge
#   lag => convert_element_type
#   mul_2 => mul_2
#   pow_2 => pow_2
#   same => bitwise_and
#   sub => sub
#   tensor => full_default
#   to_3 => convert_element_type_1
#   zeros => full_default_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_4, %view_5), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub, torch.float32), kwargs = {})
#   %ge : [num_users=3] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convert_element_type, 0), kwargs = {})
#   %full_default : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.9945994019508362), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type, 0), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%full_default, %clamp_min), kwargs = {})
#   %full_default_1 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %pow_2, %full_default_1), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %bitwise_and : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq, %unsqueeze_2), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bitwise_and, torch.float32), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, %convert_element_type_1), kwargs = {})
triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_lift_fresh_mul_pow_sub_where_zeros_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_lift_fresh_mul_pow_sub_where_zeros_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    x3 = (xindex // 64)
    x2 = (xindex // 4096)
    x4 = xindex
    tmp9 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last').to(tl.int1)
    tmp0 = x1 + ((-1)*x0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.0
    tmp3 = tmp1 >= tmp2
    tmp4 = 0.9945994019508362
    tmp5 = ((0.0) * ((0.0) >= (x1 + ((-1)*x0))) + (x1 + ((-1)*x0)) * ((x1 + ((-1)*x0)) > (0.0)))
    tmp6 = tmp5.to(tl.float32)
    tmp7 = libdevice.pow(tmp4, tmp6)
    tmp8 = tl.where(tmp3, tmp7, tmp2)
    tmp11 = tmp9 == tmp10
    tmp13 = tmp11 & tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp8 * tmp14
    tl.store(out_ptr0 + (x4), tmp15, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/wy/cwyl6vrhm27qf6anlm4fj7lmi4kma7aolo2qsvynl3dwcg4lebyf.py
# Topologically Sorted Source Nodes: [i, tensor, add_1, to_4, dec], Original ATen: [aten.arange, aten.lift_fresh, aten.add, aten._to_copy, aten.pow]
# Source node to ATen node mapping:
#   add_1 => add_1
#   dec => pow_3
#   i => iota
#   tensor => full_default
#   to_4 => convert_element_type_2
# Graph fragment:
#   %iota : [num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %full_default : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.9945994019508362), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, 1), kwargs = {})
#   %convert_element_type_2 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float32), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%full_default, %convert_element_type_2), kwargs = {})
triton_poi_fused__to_copy_add_arange_lift_fresh_pow_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_lift_fresh_pow_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.9945994019508362
    tmp1 = 1 + x0
    tmp2 = tmp1.to(tl.float32)
    tmp3 = libdevice.pow(tmp0, tmp2)
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/xv/cxv3qt67ddk2ylac6w3exs5bbslarhnkunhc3wn7fl4vnrspdopx.py
# Topologically Sorted Source Nodes: [eq_2, ge_3, and__2, mul_39, mul_73], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__2 => bitwise_and_2
#   eq_2 => eq_2
#   ge_3 => ge_3
#   mul_39 => mul_39
#   mul_73 => mul_73
# Graph fragment:
#   %eq_2 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_6, %view_11), kwargs = {})
#   %ge_3 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_6, 0), kwargs = {})
#   %bitwise_and_2 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_2, %ge_3), kwargs = {})
#   %mul_39 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_5, %view_51), kwargs = {})
#   %mul_73 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_5, %view_92), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (64 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (63 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9986470937728882
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.9996616244316101
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/it/citkdqkovfb5rvhbm3nncd4fdufvvqjy57lwhbbeo3v7mthiqasw.py
# Topologically Sorted Source Nodes: [mul_5, mul_6, S_1, stack], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_1 => add_3
#   mul_5 => mul_5
#   mul_6 => mul_6
#   stack => cat
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_5, %view_10), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %unsqueeze_6), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_8, %mul_6), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16, %add_17], 1), kwargs = {})
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x4 = (xindex // 32)
    x1 = (xindex // 32) % 64
    x0 = xindex % 32
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (2048 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2016 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (63 + (1024*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (63))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tl.full([1], -2, tl.int64)
    tmp8 = tmp6 == tmp7
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = tmp6 >= tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp16 = 0.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp5 + tmp17
    tmp19 = tmp4 * tmp18
    tmp20 = tmp0 + tmp19
    tl.store(out_ptr0 + (x5), tmp20, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp20, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/os/cosbr7uhyz2bdp7shwj44xll2ey3cc4r5wfhccoq67bdgu4xardh.py
# Topologically Sorted Source Nodes: [eq_3, ge_4, and__3, mul_41, mul_75], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__3 => bitwise_and_3
#   eq_3 => eq_3
#   ge_4 => ge_4
#   mul_41 => mul_41
#   mul_75 => mul_75
# Graph fragment:
#   %eq_3 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_12, %view_13), kwargs = {})
#   %ge_4 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_12, 0), kwargs = {})
#   %bitwise_and_3 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_3, %ge_4), kwargs = {})
#   %mul_41 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_7, %view_51), kwargs = {})
#   %mul_75 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_7, %view_92), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (127 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9986470937728882
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.9996616244316101
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/mr/cmre4dfn7qqazkqwfr456eq7hbs2qiuuy2glrwwsqlgtzvuzathj.py
# Topologically Sorted Source Nodes: [eq_4, ge_5, and__4, mul_43, mul_77], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__4 => bitwise_and_4
#   eq_4 => eq_4
#   ge_5 => ge_5
#   mul_43 => mul_43
#   mul_77 => mul_77
# Graph fragment:
#   %eq_4 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_18, %view_15), kwargs = {})
#   %ge_5 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_18, 0), kwargs = {})
#   %bitwise_and_4 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_4, %ge_5), kwargs = {})
#   %mul_43 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_9, %view_51), kwargs = {})
#   %mul_77 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_9, %view_92), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (192 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (191 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9986470937728882
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.9996616244316101
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/zw/czwi4ce5wfslvoosb2twu4jfm42ip66gsmpsl6fsru4wmezso7vc.py
# Topologically Sorted Source Nodes: [mul_3, unsqueeze_4, mul_4, S, mul_7, mul_8, S_2, mul_9, mul_10, S_3, stack], Original ATen: [aten.mul, aten.unsqueeze, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S => add_2
#   S_2 => add_4
#   S_3 => add_5
#   mul_10 => mul_10
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   stack => cat
#   unsqueeze_4 => full_default_4
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_3, %view_10), kwargs = {})
#   %full_default_4 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([32, 1, 32], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %full_default_4), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_2, %mul_4), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_7, %view_10), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_8), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_14, %mul_8), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_9, %view_10), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %unsqueeze_10), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_20, %mul_10), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16, %add_17], 1), kwargs = {})
triton_poi_fused_add_mul_stack_unsqueeze_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i1', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_unsqueeze_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x1 = (xindex // 32) % 64
    x4 = (xindex // 32)
    x0 = xindex % 32
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (6144 + x3 + (32768*x2)), None)
    tmp14 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp17 = tl.load(in_ptr0 + (6112 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp20 = tl.load(in_ptr2 + (63))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp23 = tl.load(in_ptr5 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (4096 + x3 + (32768*x2)), None)
    tmp29 = tl.load(in_ptr4 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.full([1], -2, tl.int64)
    tmp3 = tmp1 == tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp1 >= tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 0.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp0 + tmp11
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 * tmp8
    tmp19 = tmp18.to(tl.float32)
    tmp22 = tmp19 * tmp21
    tmp24 = tmp22 * tmp23
    tmp25 = tmp17 + tmp24
    tmp26 = tmp16 * tmp25
    tmp27 = tmp13 + tmp26
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp8
    tmp32 = tmp31 * tmp23
    tmp33 = tmp28 + tmp32
    tl.store(out_ptr0 + (x3 + (32768*x2)), tmp12, None)
    tl.store(out_ptr1 + (x5), tmp27, None)
    tl.store(out_ptr2 + (x3 + (32768*x2)), tmp27, None)
    tl.store(out_ptr3 + (x3 + (32768*x2)), tmp33, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/6n/c6nbthzc3sgnjjqeetr2ltuekf7azgq22uhi37pwh3n5tfj6tlud.py
# Topologically Sorted Source Nodes: [eq_5, ge_6, and__5, mul_45, mul_79], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__5 => bitwise_and_5
#   eq_5 => eq_5
#   ge_6 => ge_6
#   mul_45 => mul_45
#   mul_79 => mul_79
# Graph fragment:
#   %eq_5 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_24, %view_17), kwargs = {})
#   %ge_6 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_24, 0), kwargs = {})
#   %bitwise_and_5 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_5, %ge_6), kwargs = {})
#   %mul_45 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_11, %view_51), kwargs = {})
#   %mul_79 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_11, %view_92), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (255 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9986470937728882
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.9996616244316101
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/an/canue4lncmcqfeexcxtec342yltedomio7a5jenxr54d5t7syhfh.py
# Topologically Sorted Source Nodes: [eq_6, ge_7, and__6, mul_47, mul_81], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__6 => bitwise_and_6
#   eq_6 => eq_6
#   ge_7 => ge_7
#   mul_47 => mul_47
#   mul_81 => mul_81
# Graph fragment:
#   %eq_6 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_30, %view_19), kwargs = {})
#   %ge_7 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_30, 0), kwargs = {})
#   %bitwise_and_6 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_6, %ge_7), kwargs = {})
#   %mul_47 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_13, %view_51), kwargs = {})
#   %mul_81 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_13, %view_92), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (320 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (319 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9986470937728882
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.9996616244316101
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/wt/cwtjz24o3czvage26og2xcwt7sevesvcqpqkguurfccmiyqkfarc.py
# Topologically Sorted Source Nodes: [mul_11, mul_12, S_4, mul_13, mul_14, S_5, stack], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_4 => add_6
#   S_5 => add_7
#   mul_11 => mul_11
#   mul_12 => mul_12
#   mul_13 => mul_13
#   mul_14 => mul_14
#   stack => cat
# Graph fragment:
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_11, %view_10), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %unsqueeze_12), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_26, %mul_12), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_13, %view_10), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_14), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mul_14), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16, %add_17], 1), kwargs = {})
triton_poi_fused_add_mul_stack_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x4 = (xindex // 32)
    x1 = (xindex // 32) % 64
    x0 = xindex % 32
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (10240 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (10208 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (63))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr4 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (8192 + x3 + (32768*x2)), None)
    tmp17 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 + tmp12
    tmp14 = tmp4 * tmp13
    tmp15 = tmp0 + tmp14
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp3
    tmp20 = tmp19 * tmp11
    tmp21 = tmp16 + tmp20
    tl.store(out_ptr0 + (x5), tmp15, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp15, None)
    tl.store(out_ptr2 + (x3 + (32768*x2)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/xm/cxm4v73nste4l437dttr4kn6bmuydcptuyoee2547jy7i3dg44lo.py
# Topologically Sorted Source Nodes: [eq_7, ge_8, and__7, mul_49, mul_83], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__7 => bitwise_and_7
#   eq_7 => eq_7
#   ge_8 => ge_8
#   mul_49 => mul_49
#   mul_83 => mul_83
# Graph fragment:
#   %eq_7 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_36, %view_21), kwargs = {})
#   %ge_8 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_36, 0), kwargs = {})
#   %bitwise_and_7 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_7, %ge_8), kwargs = {})
#   %mul_49 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_15, %view_51), kwargs = {})
#   %mul_83 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_15, %view_92), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_12', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (384 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (383 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9986470937728882
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.9996616244316101
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/cc/cccwtay57bo4p5nxlz37awpldnem354mi6fg2avuzsns3g4he54m.py
# Topologically Sorted Source Nodes: [eq_8, ge_9, and__8, mul_51, mul_85], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__8 => bitwise_and_8
#   eq_8 => eq_8
#   ge_9 => ge_9
#   mul_51 => mul_51
#   mul_85 => mul_85
# Graph fragment:
#   %eq_8 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_42, %view_23), kwargs = {})
#   %ge_9 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_42, 0), kwargs = {})
#   %bitwise_and_8 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_8, %ge_9), kwargs = {})
#   %mul_51 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_17, %view_51), kwargs = {})
#   %mul_85 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_17, %view_92), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_13', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (448 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (447 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9986470937728882
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.9996616244316101
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/45/c45rnd5bhiv37jsz5psifkyt7cghgtxzv44hv3iuglvlfrixawlc.py
# Topologically Sorted Source Nodes: [mul_15, mul_16, S_6, mul_17, mul_18, S_7, stack], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_6 => add_8
#   S_7 => add_9
#   mul_15 => mul_15
#   mul_16 => mul_16
#   mul_17 => mul_17
#   mul_18 => mul_18
#   stack => cat
# Graph fragment:
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_15, %view_10), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %unsqueeze_16), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_38, %mul_16), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_17, %view_10), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %unsqueeze_18), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_44, %mul_18), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16, %add_17], 1), kwargs = {})
triton_poi_fused_add_mul_stack_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_14', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x4 = (xindex // 32)
    x1 = (xindex // 32) % 64
    x0 = xindex % 32
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (14336 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (14304 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (63))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr4 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (12288 + x3 + (32768*x2)), None)
    tmp17 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 + tmp12
    tmp14 = tmp4 * tmp13
    tmp15 = tmp0 + tmp14
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp3
    tmp20 = tmp19 * tmp11
    tmp21 = tmp16 + tmp20
    tl.store(out_ptr0 + (x5), tmp15, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp15, None)
    tl.store(out_ptr2 + (x3 + (32768*x2)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/r3/cr3vydeifnch4qi2z7qbgxtzv4epbm7e5tjmlyjylp6t23i5zvrm.py
# Topologically Sorted Source Nodes: [eq_9, ge_10, and__9, mul_53, mul_87], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__9 => bitwise_and_9
#   eq_9 => eq_9
#   ge_10 => ge_10
#   mul_53 => mul_53
#   mul_87 => mul_87
# Graph fragment:
#   %eq_9 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_48, %view_25), kwargs = {})
#   %ge_10 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_48, 0), kwargs = {})
#   %bitwise_and_9 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_9, %ge_10), kwargs = {})
#   %mul_53 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_19, %view_51), kwargs = {})
#   %mul_87 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_19, %view_92), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_15', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (511 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9986470937728882
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.9996616244316101
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ox/coxzaraedotl5ztgxln6vnoaencel7sc4ldfrpp2wnymrb77s7c3.py
# Topologically Sorted Source Nodes: [eq_10, ge_11, and__10, mul_55, mul_89], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__10 => bitwise_and_10
#   eq_10 => eq_10
#   ge_11 => ge_11
#   mul_55 => mul_55
#   mul_89 => mul_89
# Graph fragment:
#   %eq_10 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_54, %view_27), kwargs = {})
#   %ge_11 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_54, 0), kwargs = {})
#   %bitwise_and_10 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_10, %ge_11), kwargs = {})
#   %mul_55 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_21, %view_51), kwargs = {})
#   %mul_89 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_21, %view_92), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_16', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (576 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (575 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9986470937728882
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.9996616244316101
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/vg/cvgs3cpmsgcodwd5uahsjdkzp77z2rxmkrtprtpnm3abbojwdvku.py
# Topologically Sorted Source Nodes: [mul_19, mul_20, S_8, mul_21, mul_22, S_9, stack], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_8 => add_10
#   S_9 => add_11
#   mul_19 => mul_19
#   mul_20 => mul_20
#   mul_21 => mul_21
#   mul_22 => mul_22
#   stack => cat
# Graph fragment:
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_19, %view_10), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_20), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_50, %mul_20), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_21, %view_10), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %unsqueeze_22), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_56, %mul_22), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16, %add_17], 1), kwargs = {})
triton_poi_fused_add_mul_stack_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_17', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x4 = (xindex // 32)
    x1 = (xindex // 32) % 64
    x0 = xindex % 32
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (18432 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (18400 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (63))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr4 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (16384 + x3 + (32768*x2)), None)
    tmp17 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 + tmp12
    tmp14 = tmp4 * tmp13
    tmp15 = tmp0 + tmp14
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp3
    tmp20 = tmp19 * tmp11
    tmp21 = tmp16 + tmp20
    tl.store(out_ptr0 + (x5), tmp15, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp15, None)
    tl.store(out_ptr2 + (x3 + (32768*x2)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/bc/cbccr5xkm5icso7qvadhfbybnrc6rugky5ow4kvotg2ao4wmhcvg.py
# Topologically Sorted Source Nodes: [eq_11, ge_12, and__11, mul_57, mul_91], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__11 => bitwise_and_11
#   eq_11 => eq_11
#   ge_12 => ge_12
#   mul_57 => mul_57
#   mul_91 => mul_91
# Graph fragment:
#   %eq_11 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_60, %view_29), kwargs = {})
#   %ge_12 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_60, 0), kwargs = {})
#   %bitwise_and_11 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_11, %ge_12), kwargs = {})
#   %mul_57 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_23, %view_51), kwargs = {})
#   %mul_91 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_23, %view_92), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_18', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (640 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (639 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9986470937728882
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.9996616244316101
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/xn/cxn5y3dnftmtlmkcw2eqiuhupwnsxe3irk5m3tr75ll5pkllljhn.py
# Topologically Sorted Source Nodes: [eq_12, ge_13, and__12, mul_59, mul_93], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__12 => bitwise_and_12
#   eq_12 => eq_12
#   ge_13 => ge_13
#   mul_59 => mul_59
#   mul_93 => mul_93
# Graph fragment:
#   %eq_12 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_66, %view_31), kwargs = {})
#   %ge_13 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_66, 0), kwargs = {})
#   %bitwise_and_12 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_12, %ge_13), kwargs = {})
#   %mul_59 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_25, %view_51), kwargs = {})
#   %mul_93 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_25, %view_92), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_19', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (704 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (703 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9986470937728882
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.9996616244316101
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/pk/cpkdehuaoxe2bacryo4bzptltnmmwnhniwpngwhdinmzbwyu43bb.py
# Topologically Sorted Source Nodes: [mul_23, mul_24, S_10, mul_25, mul_26, S_11, stack], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_10 => add_12
#   S_11 => add_13
#   mul_23 => mul_23
#   mul_24 => mul_24
#   mul_25 => mul_25
#   mul_26 => mul_26
#   stack => cat
# Graph fragment:
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_23, %view_10), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %unsqueeze_24), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_62, %mul_24), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_25, %view_10), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_26), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_68, %mul_26), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16, %add_17], 1), kwargs = {})
triton_poi_fused_add_mul_stack_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_20', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x4 = (xindex // 32)
    x1 = (xindex // 32) % 64
    x0 = xindex % 32
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (22528 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (22496 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (63))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr4 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (20480 + x3 + (32768*x2)), None)
    tmp17 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 + tmp12
    tmp14 = tmp4 * tmp13
    tmp15 = tmp0 + tmp14
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp3
    tmp20 = tmp19 * tmp11
    tmp21 = tmp16 + tmp20
    tl.store(out_ptr0 + (x5), tmp15, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp15, None)
    tl.store(out_ptr2 + (x3 + (32768*x2)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/c3/cc3hczbvgwkovvhhupjgfqwpv636ub7kwj4ui3izo6eboa3kcgkg.py
# Topologically Sorted Source Nodes: [eq_13, ge_14, and__13, mul_61, mul_95], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__13 => bitwise_and_13
#   eq_13 => eq_13
#   ge_14 => ge_14
#   mul_61 => mul_61
#   mul_95 => mul_95
# Graph fragment:
#   %eq_13 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_72, %view_33), kwargs = {})
#   %ge_14 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_72, 0), kwargs = {})
#   %bitwise_and_13 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_13, %ge_14), kwargs = {})
#   %mul_61 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_27, %view_51), kwargs = {})
#   %mul_95 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_27, %view_92), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_21', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (767 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9986470937728882
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.9996616244316101
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/22/c22tavaqkzh5nmtpod45bxy33qxky5r2gzjpuj3n3fpfh77wycju.py
# Topologically Sorted Source Nodes: [eq_14, ge_15, and__14, mul_63, mul_97], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__14 => bitwise_and_14
#   eq_14 => eq_14
#   ge_15 => ge_15
#   mul_63 => mul_63
#   mul_97 => mul_97
# Graph fragment:
#   %eq_14 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_78, %view_35), kwargs = {})
#   %ge_15 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_78, 0), kwargs = {})
#   %bitwise_and_14 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_14, %ge_15), kwargs = {})
#   %mul_63 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_29, %view_51), kwargs = {})
#   %mul_97 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_29, %view_92), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_22', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (832 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (831 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9986470937728882
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.9996616244316101
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/x2/cx2z44akcq3d4mh65ea2kqjsrgx42uzpfecncy6n5kjnqrvz3z3j.py
# Topologically Sorted Source Nodes: [mul_27, mul_28, S_12, mul_29, mul_30, S_13, stack], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_12 => add_14
#   S_13 => add_15
#   mul_27 => mul_27
#   mul_28 => mul_28
#   mul_29 => mul_29
#   mul_30 => mul_30
#   stack => cat
# Graph fragment:
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_27, %view_10), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, %unsqueeze_28), kwargs = {})
#   %add_14 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_74, %mul_28), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_29, %view_10), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %unsqueeze_30), kwargs = {})
#   %add_15 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_80, %mul_30), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16, %add_17], 1), kwargs = {})
triton_poi_fused_add_mul_stack_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_23', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x4 = (xindex // 32)
    x1 = (xindex // 32) % 64
    x0 = xindex % 32
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (26624 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (26592 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (63))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr4 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (24576 + x3 + (32768*x2)), None)
    tmp17 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 + tmp12
    tmp14 = tmp4 * tmp13
    tmp15 = tmp0 + tmp14
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp3
    tmp20 = tmp19 * tmp11
    tmp21 = tmp16 + tmp20
    tl.store(out_ptr0 + (x5), tmp15, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp15, None)
    tl.store(out_ptr2 + (x3 + (32768*x2)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/tf/ctfsftlrqqo6euysbu4psdumjyy5ly3454jufwzl57uinnr6dbhs.py
# Topologically Sorted Source Nodes: [eq_15, ge_16, and__15, mul_65, mul_99], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__15 => bitwise_and_15
#   eq_15 => eq_15
#   ge_16 => ge_16
#   mul_65 => mul_65
#   mul_99 => mul_99
# Graph fragment:
#   %eq_15 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_84, %view_37), kwargs = {})
#   %ge_16 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_84, 0), kwargs = {})
#   %bitwise_and_15 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_15, %ge_16), kwargs = {})
#   %mul_65 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_31, %view_51), kwargs = {})
#   %mul_99 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_31, %view_92), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_24', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (896 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (895 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9986470937728882
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.9996616244316101
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/b4/cb4mtcwjjicgbefpgo37eo5snpg44nenfcnrfpq2fbcx6womsdpe.py
# Topologically Sorted Source Nodes: [mul_33], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_33 => mul_33
# Graph fragment:
#   %mul_33 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_33, %view_10), kwargs = {})
triton_poi_fused_mul_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_25', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (960 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (959 + (1024*x1)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/fd/cfdskk7hsrjylsy2cn6lxw7gqmn2nz4qyw3lbu5pntp24jptlopx.py
# Topologically Sorted Source Nodes: [mul_31, mul_32, S_14, mul_34, S_15], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   S_14 => add_16
#   S_15 => add_17
#   mul_31 => mul_31
#   mul_32 => mul_32
#   mul_34 => mul_34
# Graph fragment:
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_31, %view_10), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_32), kwargs = {})
#   %add_16 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_86, %mul_32), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_33, %unsqueeze_34), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_92, %mul_34), kwargs = {})
triton_poi_fused_add_mul_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_26', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x4 = (xindex // 32)
    x1 = (xindex // 32) % 64
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (28672 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (30720 + x3 + (32768*x2)), None)
    tmp9 = tl.load(in_ptr4 + (x4), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (30688 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp13 = tl.load(in_ptr2 + (63))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 + tmp6
    tmp12 = tmp11.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp16 = tmp15 * tmp5
    tmp17 = tmp10 + tmp16
    tmp18 = tmp9 * tmp17
    tmp19 = tmp8 + tmp18
    tl.store(out_ptr0 + (x3 + (32768*x2)), tmp7, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp19, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/m3/cm3cah6v5senrt4gfyayhu6jfiuqgqbnzc32osruc6n3axi7rtoc.py
# Topologically Sorted Source Nodes: [tensor, to_21, add_18, pow_4, sub_1, Z], Original ATen: [aten.lift_fresh, aten._to_copy, aten.add, aten.pow, aten.rsub, aten.div]
# Source node to ATen node mapping:
#   Z => div
#   add_18 => add_18
#   pow_4 => pow_4
#   sub_1 => sub_1
#   tensor => full_default
#   to_21 => convert_element_type_19
# Graph fragment:
#   %full_default : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.9945994019508362), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_19 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_5, torch.float32), kwargs = {})
#   %add_18 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_19, 1.0), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%full_default, %add_18), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %pow_4), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, 0.0054005765163668285), kwargs = {})
triton_poi_fused__to_copy_add_div_lift_fresh_pow_rsub_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_lift_fresh_pow_rsub_27', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.9945994019508362
    tmp5 = libdevice.pow(tmp4, tmp3)
    tmp6 = tmp2 - tmp5
    tmp7 = 185.16541650126229
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ee/cee7mvlkuajoapudrdwal7ogmxdjhh3hcix2kf7g6kg37humr3ui.py
# Topologically Sorted Source Nodes: [sub, lag, ge, clamp, zeros, tensor_3, pow_5, base_1, eq_17, same_1, to_24, mul_36], Original ATen: [aten.sub, aten._to_copy, aten.ge, aten.clamp, aten.zeros, aten.lift_fresh, aten.pow, aten.where, aten.eq, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   base_1 => where_1
#   clamp => clamp_min
#   eq_17 => eq_17
#   ge => ge
#   lag => convert_element_type
#   mul_36 => mul_36
#   pow_5 => pow_5
#   same_1 => bitwise_and_17
#   sub => sub
#   tensor_3 => full_default_6
#   to_24 => convert_element_type_22
#   zeros => full_default_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_4, %view_5), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub, torch.float32), kwargs = {})
#   %ge : [num_users=3] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convert_element_type, 0), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type, 0), kwargs = {})
#   %full_default_1 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_6 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.9986470937728882), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%full_default_6, %clamp_min), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %pow_5, %full_default_1), kwargs = {})
#   %eq_17 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze, %unsqueeze_38), kwargs = {})
#   %bitwise_and_17 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_17, %unsqueeze_2), kwargs = {})
#   %convert_element_type_22 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bitwise_and_17, torch.float32), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_1, %convert_element_type_22), kwargs = {})
#   %convert_element_type_69 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_2, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_lift_fresh_mul_pow_sub_where_zeros_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp32', 3: '*bf16', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_lift_fresh_mul_pow_sub_where_zeros_28', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 64
    x4 = (xindex // 64)
    x2 = (xindex // 4096)
    x3 = xindex
    tmp9 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp0 = x1 + ((-1)*x0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.0
    tmp3 = tmp1 >= tmp2
    tmp4 = 0.9986470937728882
    tmp5 = ((0.0) * ((0.0) >= (x1 + ((-1)*x0))) + (x1 + ((-1)*x0)) * ((x1 + ((-1)*x0)) > (0.0)))
    tmp6 = tmp5.to(tl.float32)
    tmp7 = libdevice.pow(tmp4, tmp6)
    tmp8 = tl.where(tmp3, tmp7, tmp2)
    tmp11 = tmp9 == tmp10
    tmp13 = tmp11 & tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp8 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/gk/cgkwvu5jzljd5k6kskesgpudcwtxeydmd2aurlxpcfgrnobrhea4.py
# Topologically Sorted Source Nodes: [sub, lag, ge, clamp, zeros, tensor_6, pow_8, base_2, eq_34, same_2, to_45, mul_70], Original ATen: [aten.sub, aten._to_copy, aten.ge, aten.clamp, aten.zeros, aten.lift_fresh, aten.pow, aten.where, aten.eq, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   base_2 => where_2
#   clamp => clamp_min
#   eq_34 => eq_34
#   ge => ge
#   lag => convert_element_type
#   mul_70 => mul_70
#   pow_8 => pow_8
#   same_2 => bitwise_and_34
#   sub => sub
#   tensor_6 => full_default_12
#   to_45 => convert_element_type_43
#   zeros => full_default_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_4, %view_5), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub, torch.float32), kwargs = {})
#   %ge : [num_users=3] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convert_element_type, 0), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type, 0), kwargs = {})
#   %full_default_1 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_12 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.9996616244316101), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%full_default_12, %clamp_min), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %pow_8, %full_default_1), kwargs = {})
#   %eq_34 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze, %unsqueeze_75), kwargs = {})
#   %bitwise_and_34 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_34, %unsqueeze_2), kwargs = {})
#   %convert_element_type_43 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bitwise_and_34, torch.float32), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_2, %convert_element_type_43), kwargs = {})
#   %convert_element_type_64 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_lift_fresh_mul_pow_sub_where_zeros_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp32', 3: '*bf16', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_lift_fresh_mul_pow_sub_where_zeros_29', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 64
    x4 = (xindex // 64)
    x2 = (xindex // 4096)
    x3 = xindex
    tmp9 = tl.load(in_ptr0 + (x4), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp0 = x1 + ((-1)*x0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.0
    tmp3 = tmp1 >= tmp2
    tmp4 = 0.9996616244316101
    tmp5 = ((0.0) * ((0.0) >= (x1 + ((-1)*x0))) + (x1 + ((-1)*x0)) * ((x1 + ((-1)*x0)) > (0.0)))
    tmp6 = tmp5.to(tl.float32)
    tmp7 = libdevice.pow(tmp4, tmp6)
    tmp8 = tl.where(tmp3, tmp7, tmp2)
    tmp11 = tmp9 == tmp10
    tmp13 = tmp11 & tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp8 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/qf/cqfr4euibcghhky6pvpnegzbquo7nlcvzeg4gwwlxbql5adesci4.py
# Topologically Sorted Source Nodes: [mul_40, S_18, mul_42, S_19, stack_1, mul_74, S_35, mul_76, S_36, stack_2], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_18 => add_21
#   S_19 => add_22
#   S_35 => add_39
#   S_36 => add_40
#   mul_40 => mul_40
#   mul_42 => mul_42
#   mul_74 => mul_74
#   mul_76 => mul_76
#   stack_1 => cat_1
#   stack_2 => cat_2
# Graph fragment:
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_39, %unsqueeze_43), kwargs = {})
#   %add_21 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_101, %mul_40), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_41, %unsqueeze_45), kwargs = {})
#   %add_22 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_107, %mul_42), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34, %add_35], 1), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_80), kwargs = {})
#   %add_39 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_194, %mul_74), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_75, %unsqueeze_82), kwargs = {})
#   %add_40 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_200, %mul_76), kwargs = {})
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_38, %add_39, %add_40, %add_41, %add_42, %add_43, %add_44, %add_45, %add_46, %add_47, %add_48, %add_49, %add_50, %add_51, %add_52, %add_53], 1), kwargs = {})
triton_poi_fused_add_mul_stack_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_30', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x4 = (xindex // 32)
    x0 = xindex % 32
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (4096 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (4064 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (63 + (64*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2016 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (63 + (1024*x2)), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (2048 + x3 + (32768*x2)), None)
    tmp24 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (4096 + x3 + (32768*x2)), None)
    tmp28 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr4 + (4064 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (63 + (64*x2)), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr4 + (2016 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (2048 + x3 + (32768*x2)), None)
    tmp42 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.full([1], -2, tl.int64)
    tmp7 = tmp5 == tmp6
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tmp5 >= tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 0.9986470937728882
    tmp13 = 64.0
    tmp14 = libdevice.pow(tmp12, tmp13)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp4 + tmp17
    tmp19 = tmp3 * tmp18
    tmp20 = tmp2 + tmp19
    tmp21 = tmp1 * tmp20
    tmp22 = tmp0 + tmp21
    tmp25 = tmp24 * tmp18
    tmp26 = tmp23 + tmp25
    tmp32 = 0.9996616244316101
    tmp33 = libdevice.pow(tmp32, tmp13)
    tmp34 = tmp11 * tmp33
    tmp35 = tmp34 * tmp16
    tmp36 = tmp31 + tmp35
    tmp37 = tmp30 * tmp36
    tmp38 = tmp29 + tmp37
    tmp39 = tmp28 * tmp38
    tmp40 = tmp27 + tmp39
    tmp43 = tmp42 * tmp36
    tmp44 = tmp41 + tmp43
    tl.store(out_ptr0 + (x5), tmp22, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp22, None)
    tl.store(out_ptr2 + (x3 + (32768*x2)), tmp26, None)
    tl.store(out_ptr3 + (x5), tmp40, None)
    tl.store(out_ptr4 + (x3 + (32768*x2)), tmp40, None)
    tl.store(out_ptr5 + (x3 + (32768*x2)), tmp44, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/hc/chcpkpafdcvjssjnyoo5carf5q4kqmiqg6qlljyxx2dbgnluo6zu.py
# Topologically Sorted Source Nodes: [mul_44, S_20, mul_46, S_21, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_20 => add_23
#   S_21 => add_24
#   mul_44 => mul_44
#   mul_46 => mul_46
#   stack_1 => cat_1
# Graph fragment:
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_47), kwargs = {})
#   %add_23 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_113, %mul_44), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_45, %unsqueeze_49), kwargs = {})
#   %add_24 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_119, %mul_46), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34, %add_35], 1), kwargs = {})
triton_poi_fused_add_mul_stack_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_31', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x4 = (xindex // 32)
    x0 = xindex % 32
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (8192 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (8160 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (63 + (64*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (6144 + x3 + (32768*x2)), None)
    tmp10 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tmp1 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tl.store(out_ptr0 + (x5), tmp8, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp8, None)
    tl.store(out_ptr2 + (x3 + (32768*x2)), tmp12, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/dt/cdttqubpalanhs4o5vu7mycl76o3olz63l45ibbrwqbsx63bmmmv.py
# Topologically Sorted Source Nodes: [mul_48, S_22, mul_50, S_23, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_22 => add_25
#   S_23 => add_26
#   mul_48 => mul_48
#   mul_50 => mul_50
#   stack_1 => cat_1
# Graph fragment:
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_47, %unsqueeze_51), kwargs = {})
#   %add_25 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_125, %mul_48), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_53), kwargs = {})
#   %add_26 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_131, %mul_50), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34, %add_35], 1), kwargs = {})
triton_poi_fused_add_mul_stack_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_32', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x4 = (xindex // 32)
    x0 = xindex % 32
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (12288 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (12256 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (63 + (64*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (10240 + x3 + (32768*x2)), None)
    tmp10 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tmp1 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tl.store(out_ptr0 + (x5), tmp8, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp8, None)
    tl.store(out_ptr2 + (x3 + (32768*x2)), tmp12, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/3x/c3xwlfjmzleojhrgwjcg3careufelyuaocauvfs2t5epbqoghl7n.py
# Topologically Sorted Source Nodes: [mul_52, S_24, mul_54, S_25, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_24 => add_27
#   S_25 => add_28
#   mul_52 => mul_52
#   mul_54 => mul_54
#   stack_1 => cat_1
# Graph fragment:
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_51, %unsqueeze_55), kwargs = {})
#   %add_27 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_137, %mul_52), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_53, %unsqueeze_57), kwargs = {})
#   %add_28 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_143, %mul_54), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34, %add_35], 1), kwargs = {})
triton_poi_fused_add_mul_stack_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_33', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x4 = (xindex // 32)
    x0 = xindex % 32
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (16384 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (16352 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (63 + (64*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (14336 + x3 + (32768*x2)), None)
    tmp10 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tmp1 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tl.store(out_ptr0 + (x5), tmp8, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp8, None)
    tl.store(out_ptr2 + (x3 + (32768*x2)), tmp12, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/vb/cvbvgarfipnrlkdsygruivte75sf5j57twdzoxpwscamzz5op6qm.py
# Topologically Sorted Source Nodes: [mul_56, S_26, mul_58, S_27, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_26 => add_29
#   S_27 => add_30
#   mul_56 => mul_56
#   mul_58 => mul_58
#   stack_1 => cat_1
# Graph fragment:
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %unsqueeze_59), kwargs = {})
#   %add_29 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_149, %mul_56), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_57, %unsqueeze_61), kwargs = {})
#   %add_30 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_155, %mul_58), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34, %add_35], 1), kwargs = {})
triton_poi_fused_add_mul_stack_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_34', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x4 = (xindex // 32)
    x0 = xindex % 32
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (20480 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (20448 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (63 + (64*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (18432 + x3 + (32768*x2)), None)
    tmp10 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tmp1 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tl.store(out_ptr0 + (x5), tmp8, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp8, None)
    tl.store(out_ptr2 + (x3 + (32768*x2)), tmp12, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/vp/cvp5ofh7wvenvt7sub2nkxzeldqzumfj3dc5fcfst4kpukayjcb2.py
# Topologically Sorted Source Nodes: [mul_60, S_28, mul_62, S_29, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_28 => add_31
#   S_29 => add_32
#   mul_60 => mul_60
#   mul_62 => mul_62
#   stack_1 => cat_1
# Graph fragment:
#   %mul_60 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_59, %unsqueeze_63), kwargs = {})
#   %add_31 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_161, %mul_60), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_65), kwargs = {})
#   %add_32 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_167, %mul_62), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34, %add_35], 1), kwargs = {})
triton_poi_fused_add_mul_stack_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_35', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x4 = (xindex // 32)
    x0 = xindex % 32
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (24576 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (24544 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (63 + (64*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (22528 + x3 + (32768*x2)), None)
    tmp10 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tmp1 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tl.store(out_ptr0 + (x5), tmp8, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp8, None)
    tl.store(out_ptr2 + (x3 + (32768*x2)), tmp12, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/zf/czfbfwwlgx6wixa67bjxmfzv6kwqyhg4sj4i6ohe77yr5emw4ijv.py
# Topologically Sorted Source Nodes: [mul_64, S_30, mul_66, S_31, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_30 => add_33
#   S_31 => add_34
#   mul_64 => mul_64
#   mul_66 => mul_66
#   stack_1 => cat_1
# Graph fragment:
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_63, %unsqueeze_67), kwargs = {})
#   %add_33 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_173, %mul_64), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_65, %unsqueeze_69), kwargs = {})
#   %add_34 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_179, %mul_66), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34, %add_35], 1), kwargs = {})
triton_poi_fused_add_mul_stack_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_36', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x4 = (xindex // 32)
    x0 = xindex % 32
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (28672 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (28640 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (63 + (64*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (26624 + x3 + (32768*x2)), None)
    tmp10 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tmp1 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tl.store(out_ptr0 + (x5), tmp8, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp8, None)
    tl.store(out_ptr2 + (x3 + (32768*x2)), tmp12, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/vd/cvdyysn33mstt3iyzz6eosoyhw2e4voj26ihgixqy5ybhqe3hsqu.py
# Topologically Sorted Source Nodes: [mul_67, mul_101], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_101 => mul_101
#   mul_67 => mul_67
# Graph fragment:
#   %mul_67 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_33, %view_51), kwargs = {})
#   %mul_101 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_33, %view_92), kwargs = {})
triton_poi_fused_mul_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_37', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (960 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (959 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9986470937728882
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.9996616244316101
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp11, None)
    tl.store(out_ptr1 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/nc/cncn4bcypapacnyki7otvjt3ec7drkze4i2kfs73pamckstuiykn.py
# Topologically Sorted Source Nodes: [unsqueeze_4, mul_37, mul_38, S_17, mul_71, mul_72, S_34], Original ATen: [aten.unsqueeze, aten.mul, aten.add]
# Source node to ATen node mapping:
#   S_17 => add_20
#   S_34 => add_38
#   mul_37 => mul_37
#   mul_38 => mul_38
#   mul_71 => mul_71
#   mul_72 => mul_72
#   unsqueeze_4 => full_default_4
# Graph fragment:
#   %full_default_4 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([32, 1, 32], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_3, %view_51), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %full_default_4), kwargs = {})
#   %add_20 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_95, %mul_38), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_3, %view_92), kwargs = {})
#   %mul_72 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_71, %full_default_4), kwargs = {})
#   %add_38 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_188, %mul_72), kwargs = {})
triton_poi_fused_add_mul_unsqueeze_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_unsqueeze_38', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp8 = 0.9986470937728882
    tmp9 = 1 + x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.pow(tmp8, tmp10)
    tmp12 = tmp7 * tmp11
    tmp13 = 0.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp0 + tmp14
    tmp17 = 0.9996616244316101
    tmp18 = libdevice.pow(tmp17, tmp10)
    tmp19 = tmp7 * tmp18
    tmp20 = tmp19 * tmp13
    tmp21 = tmp16 + tmp20
    tl.store(out_ptr0 + (x3 + (32768*x2)), tmp15, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/o2/co2uozvddgz6nxg32gt4ghxdcx6ufldcyqgomayair4rf3cp3h6m.py
# Topologically Sorted Source Nodes: [mul_68, S_32], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   S_32 => add_35
#   mul_68 => mul_68
# Graph fragment:
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_67, %unsqueeze_71), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_185, %mul_68), kwargs = {})
triton_poi_fused_add_mul_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_39', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x4 = (xindex // 32)
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (30720 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x3 + (32768*x2)), tmp4, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/h5/ch5u2jbusgldqjtsqu6z3kbdgavdfgd7ydrrhuwf4ozcshawtuq2.py
# Topologically Sorted Source Nodes: [to_21, add_18, tensor_3, pow_7, sub_3, Z_1, tensor_6, pow_10, sub_5, Z_2], Original ATen: [aten._to_copy, aten.add, aten.lift_fresh, aten.pow, aten.rsub, aten.div]
# Source node to ATen node mapping:
#   Z_1 => div_2
#   Z_2 => div_4
#   add_18 => add_18
#   pow_10 => pow_10
#   pow_7 => pow_7
#   sub_3 => sub_3
#   sub_5 => sub_5
#   tensor_3 => full_default_6
#   tensor_6 => full_default_12
#   to_21 => convert_element_type_19
# Graph fragment:
#   %convert_element_type_19 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_5, torch.float32), kwargs = {})
#   %add_18 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_19, 1.0), kwargs = {})
#   %full_default_6 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.9986470937728882), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%full_default_6, %add_18), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %pow_7), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_3, 0.00135288710902981), kwargs = {})
#   %full_default_12 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.9996616244316101), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_10 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%full_default_12, %add_18), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %pow_10), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_5, 0.00033839350375630683), kwargs = {})
triton_poi_fused__to_copy_add_div_lift_fresh_pow_rsub_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_lift_fresh_pow_rsub_40', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = 0.9986470937728882
    tmp5 = libdevice.pow(tmp4, tmp3)
    tmp6 = tmp2 - tmp5
    tmp7 = 739.1599737520787
    tmp8 = tmp6 * tmp7
    tmp9 = 0.9996616244316101
    tmp10 = libdevice.pow(tmp9, tmp3)
    tmp11 = tmp2 - tmp10
    tmp12 = 2955.1394719449086
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/dh/cdhbkcuyw3o7kxwby4s6d3cd5rxaye52iqcqvzs3hpkjqs2q2pni.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat_3
# Graph fragment:
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_35, %mul_69, %mul_103], -1), kwargs = {})
triton_poi_fused_cat_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_41', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 96
    x1 = (xindex // 96)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((32*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 / tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp4, tmp10, tmp11)
    tmp13 = tmp0 >= tmp3
    tmp14 = tl.full([1], 64, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tmp13 & tmp15
    tmp17 = tl.load(in_ptr3 + ((32*x1) + ((-32) + x0)), tmp16, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr4 + (x1), tmp16, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 / tmp18
    tmp20 = tl.load(in_ptr2 + (x1), tmp16, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 * tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp16, tmp22, tmp23)
    tmp25 = tmp0 >= tmp14
    tmp26 = tl.full([1], 96, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + ((32*x1) + ((-64) + x0)), tmp25, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr6 + (x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 / tmp29
    tmp31 = tl.load(in_ptr2 + (x1), tmp25, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp30 * tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp25, tmp33, tmp34)
    tmp36 = tl.where(tmp16, tmp24, tmp35)
    tmp37 = tl.where(tmp4, tmp12, tmp36)
    tl.store(out_ptr0 + (x2), tmp37, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/dz/cdztp6bk2uqykx4ycbsyryhxqxmj36cemj7i3ovyjer3ntnaq3ae.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_82 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_6, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_42 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_42', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (32, 1024, 384), (393216, 384, 1))
    assert_size_stride(primals_2, (384, ), (1, ))
    assert_size_stride(primals_3, (96, 384), (384, 1))
    assert_size_stride(primals_4, (32, 1024), (1024, 1))
    assert_size_stride(primals_5, (32, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf1 = reinterpret_tensor(buf0, (32, 1024, 1), (1024, 1, 1), 0); del buf0  # reuse
        buf2 = empty_strided_cuda((32, 1024, 384), (393216, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pow_1, mean, add, rsqrt, y, mul_1], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_0.run(buf1, primals_1, primals_2, buf2, 32768, 384, grid=grid(32768), stream=stream0)
        buf3 = empty_strided_cuda((32768, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (32768, 384), (384, 1), 0), reinterpret_tensor(primals_3, (384, 96), (1, 384), 0), out=buf3)
        del buf2
        buf4 = empty_strided_cuda((32, 16, 64), (1024, 64, 1), torch.bool)
        buf48 = empty_strided_cuda((32, 1024), (1024, 1), torch.bool)
        # Topologically Sorted Source Nodes: [ge_1, ge_18], Original ATen: [aten.ge]
        triton_poi_fused_ge_1.run(primals_4, buf4, buf48, 32768, grid=grid(32768), stream=stream0)
        buf5 = empty_strided_cuda((32, 16, 64, 64), (65536, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sub, lag, ge, tensor, clamp, pow_2, zeros, base, eq, same, to_3, mul_2], Original ATen: [aten.sub, aten._to_copy, aten.ge, aten.lift_fresh, aten.clamp, aten.pow, aten.zeros, aten.where, aten.eq, aten.bitwise_and, aten.mul]
        triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_lift_fresh_mul_pow_sub_where_zeros_2.run(primals_4, buf4, buf5, 2097152, grid=grid(2097152), stream=stream0)
        buf6 = empty_strided_cuda((512, 64, 32), (2048, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [S_in], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf5, (512, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf3, (512, 64, 32), (6144, 96, 1), 0), out=buf6)
        buf7 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [i, tensor, add_1, to_4, dec], Original ATen: [aten.arange, aten.lift_fresh, aten.add, aten._to_copy, aten.pow]
        triton_poi_fused__to_copy_add_arange_lift_fresh_pow_3.run(buf7, 64, grid=grid(64), stream=stream0)
        buf8 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf51 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf93 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_2, ge_3, and__2, mul_39, mul_73], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_4.run(primals_4, buf8, buf51, buf93, 2048, grid=grid(2048), stream=stream0)
        buf9 = empty_strided_cuda((32, 64, 32), (2048, 32, 1), torch.float32)
        buf46 = empty_strided_cuda((32, 1024, 32), (32768, 32, 1), torch.float32)
        buf31 = reinterpret_tensor(buf46, (32, 64, 32), (32768, 32, 1), 2048)  # alias
        # Topologically Sorted Source Nodes: [mul_5, mul_6, S_1, stack], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_5.run(buf6, buf8, buf7, primals_4, buf9, buf31, 65536, grid=grid(65536), stream=stream0)
        buf10 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf52 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf94 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_3, ge_4, and__3, mul_41, mul_75], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_6.run(primals_4, buf10, buf52, buf94, 2048, grid=grid(2048), stream=stream0)
        buf11 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf54 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf96 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_4, ge_5, and__4, mul_43, mul_77], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_7.run(primals_4, buf11, buf54, buf96, 2048, grid=grid(2048), stream=stream0)
        buf30 = reinterpret_tensor(buf46, (32, 64, 32), (32768, 32, 1), 0)  # alias
        buf12 = empty_strided_cuda((32, 64, 32), (2048, 32, 1), torch.float32)
        buf33 = reinterpret_tensor(buf46, (32, 64, 32), (32768, 32, 1), 6144)  # alias
        buf32 = reinterpret_tensor(buf46, (32, 64, 32), (32768, 32, 1), 4096)  # alias
        # Topologically Sorted Source Nodes: [mul_3, unsqueeze_4, mul_4, S, mul_7, mul_8, S_2, mul_9, mul_10, S_3, stack], Original ATen: [aten.mul, aten.unsqueeze, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_unsqueeze_8.run(buf6, primals_4, buf7, buf11, buf10, buf9, buf30, buf12, buf33, buf32, 65536, grid=grid(65536), stream=stream0)
        buf13 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf55 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf97 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_5, ge_6, and__5, mul_45, mul_79], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_9.run(primals_4, buf13, buf55, buf97, 2048, grid=grid(2048), stream=stream0)
        buf14 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf57 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf99 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_6, ge_7, and__6, mul_47, mul_81], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_10.run(primals_4, buf14, buf57, buf99, 2048, grid=grid(2048), stream=stream0)
        buf15 = buf9; del buf9  # reuse
        buf35 = reinterpret_tensor(buf46, (32, 64, 32), (32768, 32, 1), 10240)  # alias
        buf34 = reinterpret_tensor(buf46, (32, 64, 32), (32768, 32, 1), 8192)  # alias
        # Topologically Sorted Source Nodes: [mul_11, mul_12, S_4, mul_13, mul_14, S_5, stack], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_11.run(buf6, buf14, buf7, buf13, buf12, buf15, buf35, buf34, 65536, grid=grid(65536), stream=stream0)
        buf16 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf58 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf100 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_7, ge_8, and__7, mul_49, mul_83], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_12.run(primals_4, buf16, buf58, buf100, 2048, grid=grid(2048), stream=stream0)
        buf17 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf60 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf102 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_8, ge_9, and__8, mul_51, mul_85], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_13.run(primals_4, buf17, buf60, buf102, 2048, grid=grid(2048), stream=stream0)
        buf18 = buf12; del buf12  # reuse
        buf37 = reinterpret_tensor(buf46, (32, 64, 32), (32768, 32, 1), 14336)  # alias
        buf36 = reinterpret_tensor(buf46, (32, 64, 32), (32768, 32, 1), 12288)  # alias
        # Topologically Sorted Source Nodes: [mul_15, mul_16, S_6, mul_17, mul_18, S_7, stack], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_14.run(buf6, buf17, buf7, buf16, buf15, buf18, buf37, buf36, 65536, grid=grid(65536), stream=stream0)
        buf19 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf61 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf103 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_9, ge_10, and__9, mul_53, mul_87], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_15.run(primals_4, buf19, buf61, buf103, 2048, grid=grid(2048), stream=stream0)
        buf20 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf63 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf105 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_10, ge_11, and__10, mul_55, mul_89], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_16.run(primals_4, buf20, buf63, buf105, 2048, grid=grid(2048), stream=stream0)
        buf21 = buf15; del buf15  # reuse
        buf39 = reinterpret_tensor(buf46, (32, 64, 32), (32768, 32, 1), 18432)  # alias
        buf38 = reinterpret_tensor(buf46, (32, 64, 32), (32768, 32, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [mul_19, mul_20, S_8, mul_21, mul_22, S_9, stack], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_17.run(buf6, buf20, buf7, buf19, buf18, buf21, buf39, buf38, 65536, grid=grid(65536), stream=stream0)
        buf22 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf64 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf106 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_11, ge_12, and__11, mul_57, mul_91], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_18.run(primals_4, buf22, buf64, buf106, 2048, grid=grid(2048), stream=stream0)
        buf23 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf66 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf108 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_12, ge_13, and__12, mul_59, mul_93], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_19.run(primals_4, buf23, buf66, buf108, 2048, grid=grid(2048), stream=stream0)
        buf24 = buf18; del buf18  # reuse
        buf41 = reinterpret_tensor(buf46, (32, 64, 32), (32768, 32, 1), 22528)  # alias
        buf40 = reinterpret_tensor(buf46, (32, 64, 32), (32768, 32, 1), 20480)  # alias
        # Topologically Sorted Source Nodes: [mul_23, mul_24, S_10, mul_25, mul_26, S_11, stack], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_20.run(buf6, buf23, buf7, buf22, buf21, buf24, buf41, buf40, 65536, grid=grid(65536), stream=stream0)
        buf25 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf67 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf109 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_13, ge_14, and__13, mul_61, mul_95], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_21.run(primals_4, buf25, buf67, buf109, 2048, grid=grid(2048), stream=stream0)
        buf26 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf69 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf111 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_14, ge_15, and__14, mul_63, mul_97], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_22.run(primals_4, buf26, buf69, buf111, 2048, grid=grid(2048), stream=stream0)
        buf27 = buf21; del buf21  # reuse
        buf43 = reinterpret_tensor(buf46, (32, 64, 32), (32768, 32, 1), 26624)  # alias
        buf42 = reinterpret_tensor(buf46, (32, 64, 32), (32768, 32, 1), 24576)  # alias
        # Topologically Sorted Source Nodes: [mul_27, mul_28, S_12, mul_29, mul_30, S_13, stack], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_23.run(buf6, buf26, buf7, buf25, buf24, buf27, buf43, buf42, 65536, grid=grid(65536), stream=stream0)
        buf28 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf70 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf112 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_15, ge_16, and__15, mul_65, mul_99], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_24.run(primals_4, buf28, buf70, buf112, 2048, grid=grid(2048), stream=stream0)
        buf29 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_33], Original ATen: [aten.mul]
        triton_poi_fused_mul_25.run(primals_4, buf7, buf29, 2048, grid=grid(2048), stream=stream0)
        buf44 = reinterpret_tensor(buf46, (32, 64, 32), (32768, 32, 1), 28672)  # alias
        buf45 = reinterpret_tensor(buf46, (32, 64, 32), (32768, 32, 1), 30720)  # alias
        # Topologically Sorted Source Nodes: [mul_31, mul_32, S_14, mul_34, S_15], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_26.run(buf6, buf28, buf7, buf27, buf29, buf44, buf45, 65536, grid=grid(65536), stream=stream0)
        buf47 = empty_strided_cuda((32, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [tensor, to_21, add_18, pow_4, sub_1, Z], Original ATen: [aten.lift_fresh, aten._to_copy, aten.add, aten.pow, aten.rsub, aten.div]
        triton_poi_fused__to_copy_add_div_lift_fresh_pow_rsub_27.run(primals_5, buf47, 32768, grid=grid(32768), stream=stream0)
        del buf30
        del buf31
        del buf32
        del buf33
        del buf34
        del buf35
        del buf36
        del buf37
        del buf38
        del buf39
        del buf40
        del buf41
        del buf42
        del buf43
        del buf44
        del buf45
        buf49 = buf5; del buf5  # reuse
        buf135 = empty_strided_cuda((512, 64, 64), (4096, 1, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [sub, lag, ge, clamp, zeros, tensor_3, pow_5, base_1, eq_17, same_1, to_24, mul_36], Original ATen: [aten.sub, aten._to_copy, aten.ge, aten.clamp, aten.zeros, aten.lift_fresh, aten.pow, aten.where, aten.eq, aten.bitwise_and, aten.mul]
        triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_lift_fresh_mul_pow_sub_where_zeros_28.run(primals_4, buf4, buf49, buf135, 2097152, grid=grid(2097152), stream=stream0)
        buf50 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [S_in_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf49, (512, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf3, (512, 64, 32), (6144, 96, 1), 32), out=buf50)
        buf91 = buf49; del buf49  # reuse
        buf134 = empty_strided_cuda((512, 64, 64), (4096, 1, 64), torch.bfloat16)
        # Topologically Sorted Source Nodes: [sub, lag, ge, clamp, zeros, tensor_6, pow_8, base_2, eq_34, same_2, to_45, mul_70], Original ATen: [aten.sub, aten._to_copy, aten.ge, aten.clamp, aten.zeros, aten.lift_fresh, aten.pow, aten.where, aten.eq, aten.bitwise_and, aten.mul]
        triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_lift_fresh_mul_pow_sub_where_zeros_29.run(primals_4, buf4, buf91, buf134, 2097152, grid=grid(2097152), stream=stream0)
        buf92 = empty_strided_cuda((512, 64, 32), (2048, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [S_in_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf91, (512, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf3, (512, 64, 32), (6144, 96, 1), 64), out=buf92)
        del buf91
        buf53 = buf27; del buf27  # reuse
        buf89 = empty_strided_cuda((32, 1024, 32), (32768, 32, 1), torch.float32)
        buf75 = reinterpret_tensor(buf89, (32, 64, 32), (32768, 32, 1), 4096)  # alias
        buf74 = reinterpret_tensor(buf89, (32, 64, 32), (32768, 32, 1), 2048)  # alias
        buf95 = buf24; del buf24  # reuse
        buf131 = empty_strided_cuda((32, 1024, 32), (32768, 32, 1), torch.float32)
        buf117 = reinterpret_tensor(buf131, (32, 64, 32), (32768, 32, 1), 4096)  # alias
        buf116 = reinterpret_tensor(buf131, (32, 64, 32), (32768, 32, 1), 2048)  # alias
        # Topologically Sorted Source Nodes: [mul_40, S_18, mul_42, S_19, stack_1, mul_74, S_35, mul_76, S_36, stack_2], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_30.run(buf50, buf52, buf51, primals_4, buf92, buf94, buf93, buf53, buf75, buf74, buf95, buf117, buf116, 65536, grid=grid(65536), stream=stream0)
        buf56 = empty_strided_cuda((32, 64, 32), (2048, 32, 1), torch.float32)
        buf77 = reinterpret_tensor(buf89, (32, 64, 32), (32768, 32, 1), 8192)  # alias
        buf76 = reinterpret_tensor(buf89, (32, 64, 32), (32768, 32, 1), 6144)  # alias
        # Topologically Sorted Source Nodes: [mul_44, S_20, mul_46, S_21, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_31.run(buf50, buf55, buf54, buf53, buf56, buf77, buf76, 65536, grid=grid(65536), stream=stream0)
        buf59 = buf53; del buf53  # reuse
        buf79 = reinterpret_tensor(buf89, (32, 64, 32), (32768, 32, 1), 12288)  # alias
        buf78 = reinterpret_tensor(buf89, (32, 64, 32), (32768, 32, 1), 10240)  # alias
        # Topologically Sorted Source Nodes: [mul_48, S_22, mul_50, S_23, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_32.run(buf50, buf58, buf57, buf56, buf59, buf79, buf78, 65536, grid=grid(65536), stream=stream0)
        buf62 = buf56; del buf56  # reuse
        buf81 = reinterpret_tensor(buf89, (32, 64, 32), (32768, 32, 1), 16384)  # alias
        buf80 = reinterpret_tensor(buf89, (32, 64, 32), (32768, 32, 1), 14336)  # alias
        # Topologically Sorted Source Nodes: [mul_52, S_24, mul_54, S_25, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_33.run(buf50, buf61, buf60, buf59, buf62, buf81, buf80, 65536, grid=grid(65536), stream=stream0)
        buf65 = buf59; del buf59  # reuse
        buf83 = reinterpret_tensor(buf89, (32, 64, 32), (32768, 32, 1), 20480)  # alias
        buf82 = reinterpret_tensor(buf89, (32, 64, 32), (32768, 32, 1), 18432)  # alias
        # Topologically Sorted Source Nodes: [mul_56, S_26, mul_58, S_27, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_34.run(buf50, buf64, buf63, buf62, buf65, buf83, buf82, 65536, grid=grid(65536), stream=stream0)
        buf68 = buf62; del buf62  # reuse
        buf85 = reinterpret_tensor(buf89, (32, 64, 32), (32768, 32, 1), 24576)  # alias
        buf84 = reinterpret_tensor(buf89, (32, 64, 32), (32768, 32, 1), 22528)  # alias
        # Topologically Sorted Source Nodes: [mul_60, S_28, mul_62, S_29, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_35.run(buf50, buf67, buf66, buf65, buf68, buf85, buf84, 65536, grid=grid(65536), stream=stream0)
        buf71 = buf65; del buf65  # reuse
        buf87 = reinterpret_tensor(buf89, (32, 64, 32), (32768, 32, 1), 28672)  # alias
        buf86 = reinterpret_tensor(buf89, (32, 64, 32), (32768, 32, 1), 26624)  # alias
        # Topologically Sorted Source Nodes: [mul_64, S_30, mul_66, S_31, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_36.run(buf50, buf70, buf69, buf68, buf71, buf87, buf86, 65536, grid=grid(65536), stream=stream0)
        del buf68
        buf72 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf114 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_67, mul_101], Original ATen: [aten.mul]
        triton_poi_fused_mul_37.run(primals_4, buf72, buf114, 2048, grid=grid(2048), stream=stream0)
        buf73 = reinterpret_tensor(buf89, (32, 64, 32), (32768, 32, 1), 0)  # alias
        buf115 = reinterpret_tensor(buf131, (32, 64, 32), (32768, 32, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [unsqueeze_4, mul_37, mul_38, S_17, mul_71, mul_72, S_34], Original ATen: [aten.unsqueeze, aten.mul, aten.add]
        triton_poi_fused_add_mul_unsqueeze_38.run(buf50, primals_4, buf92, buf73, buf115, 65536, grid=grid(65536), stream=stream0)
        buf88 = reinterpret_tensor(buf89, (32, 64, 32), (32768, 32, 1), 30720)  # alias
        # Topologically Sorted Source Nodes: [mul_68, S_32], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_39.run(buf50, buf72, buf71, buf88, 65536, grid=grid(65536), stream=stream0)
        del buf50
        buf90 = empty_strided_cuda((32, 1024), (1024, 1), torch.float32)
        buf132 = empty_strided_cuda((32, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [to_21, add_18, tensor_3, pow_7, sub_3, Z_1, tensor_6, pow_10, sub_5, Z_2], Original ATen: [aten._to_copy, aten.add, aten.lift_fresh, aten.pow, aten.rsub, aten.div]
        triton_poi_fused__to_copy_add_div_lift_fresh_pow_rsub_40.run(primals_5, buf90, buf132, 32768, grid=grid(32768), stream=stream0)
        del buf73
        del buf74
        del buf75
        del buf76
        del buf77
        del buf78
        del buf79
        del buf80
        del buf81
        del buf82
        del buf83
        del buf84
        del buf85
        del buf86
        del buf87
        del buf88
        del primals_5
        buf98 = buf71; del buf71  # reuse
        buf119 = reinterpret_tensor(buf131, (32, 64, 32), (32768, 32, 1), 8192)  # alias
        buf118 = reinterpret_tensor(buf131, (32, 64, 32), (32768, 32, 1), 6144)  # alias
        # Topologically Sorted Source Nodes: [mul_78, S_37, mul_80, S_38, stack_2], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_31.run(buf92, buf97, buf96, buf95, buf98, buf119, buf118, 65536, grid=grid(65536), stream=stream0)
        buf101 = buf95; del buf95  # reuse
        buf121 = reinterpret_tensor(buf131, (32, 64, 32), (32768, 32, 1), 12288)  # alias
        buf120 = reinterpret_tensor(buf131, (32, 64, 32), (32768, 32, 1), 10240)  # alias
        # Topologically Sorted Source Nodes: [mul_82, S_39, mul_84, S_40, stack_2], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_32.run(buf92, buf100, buf99, buf98, buf101, buf121, buf120, 65536, grid=grid(65536), stream=stream0)
        buf104 = buf98; del buf98  # reuse
        buf123 = reinterpret_tensor(buf131, (32, 64, 32), (32768, 32, 1), 16384)  # alias
        buf122 = reinterpret_tensor(buf131, (32, 64, 32), (32768, 32, 1), 14336)  # alias
        # Topologically Sorted Source Nodes: [mul_86, S_41, mul_88, S_42, stack_2], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_33.run(buf92, buf103, buf102, buf101, buf104, buf123, buf122, 65536, grid=grid(65536), stream=stream0)
        buf107 = buf101; del buf101  # reuse
        buf125 = reinterpret_tensor(buf131, (32, 64, 32), (32768, 32, 1), 20480)  # alias
        buf124 = reinterpret_tensor(buf131, (32, 64, 32), (32768, 32, 1), 18432)  # alias
        # Topologically Sorted Source Nodes: [mul_90, S_43, mul_92, S_44, stack_2], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_34.run(buf92, buf106, buf105, buf104, buf107, buf125, buf124, 65536, grid=grid(65536), stream=stream0)
        buf110 = buf104; del buf104  # reuse
        buf127 = reinterpret_tensor(buf131, (32, 64, 32), (32768, 32, 1), 24576)  # alias
        buf126 = reinterpret_tensor(buf131, (32, 64, 32), (32768, 32, 1), 22528)  # alias
        # Topologically Sorted Source Nodes: [mul_94, S_45, mul_96, S_46, stack_2], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_35.run(buf92, buf109, buf108, buf107, buf110, buf127, buf126, 65536, grid=grid(65536), stream=stream0)
        buf113 = buf107; del buf107  # reuse
        buf129 = reinterpret_tensor(buf131, (32, 64, 32), (32768, 32, 1), 28672)  # alias
        buf128 = reinterpret_tensor(buf131, (32, 64, 32), (32768, 32, 1), 26624)  # alias
        # Topologically Sorted Source Nodes: [mul_98, S_47, mul_100, S_48, stack_2], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_36.run(buf92, buf112, buf111, buf110, buf113, buf129, buf128, 65536, grid=grid(65536), stream=stream0)
        del buf110
        buf130 = reinterpret_tensor(buf131, (32, 64, 32), (32768, 32, 1), 30720)  # alias
        # Topologically Sorted Source Nodes: [mul_102, S_49], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_39.run(buf92, buf114, buf113, buf130, 65536, grid=grid(65536), stream=stream0)
        del buf113
        del buf92
        buf133 = reinterpret_tensor(buf3, (32, 1024, 96), (98304, 96, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        triton_poi_fused_cat_41.run(buf46, buf47, buf48, buf89, buf90, buf131, buf132, buf133, 3145728, grid=grid(3145728), stream=stream0)
        del buf115
        del buf116
        del buf117
        del buf118
        del buf119
        del buf120
        del buf121
        del buf122
        del buf123
        del buf124
        del buf125
        del buf126
        del buf127
        del buf128
        del buf129
        del buf130
        del buf131
        del buf46
        del buf89
        buf136 = empty_strided_cuda((96, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_42.run(primals_3, buf136, 36864, grid=grid(36864), stream=stream0)
        del primals_3
    return (buf133, primals_1, primals_2, primals_4, buf1, reinterpret_tensor(buf4, (32, 16, 64, 1), (1024, 64, 1, 1), 0), reinterpret_tensor(buf7, (1, 64, 1), (64, 1, 1), 0), buf8, buf10, buf11, buf13, buf14, buf16, buf17, buf19, buf20, buf22, buf23, buf25, buf26, buf28, buf29, reinterpret_tensor(buf47, (32, 1024, 1), (1024, 1, 1), 0), reinterpret_tensor(buf48, (32, 1024, 1), (1024, 1, 1), 0), buf51, buf52, buf54, buf55, buf57, buf58, buf60, buf61, buf63, buf64, buf66, buf67, buf69, buf70, buf72, reinterpret_tensor(buf90, (32, 1024, 1), (1024, 1, 1), 0), buf93, buf94, buf96, buf97, buf99, buf100, buf102, buf103, buf105, buf106, buf108, buf109, buf111, buf112, buf114, reinterpret_tensor(buf132, (32, 1024, 1), (1024, 1, 1), 0), buf134, buf135, buf136, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 1024, 384), (393216, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((96, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
