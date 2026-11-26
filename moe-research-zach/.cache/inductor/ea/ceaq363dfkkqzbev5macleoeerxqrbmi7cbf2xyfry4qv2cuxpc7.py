# AOT ID: ['4_forward']
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/z7/cz7mbemdaiu4gmg77hlnkspryv7oqjfw4k72amoz4ytqshnr7jka.py
# Topologically Sorted Source Nodes: [sub, lag, ge, tensor, clamp, pow_1, zeros, base, eq, same, to_2, mul], Original ATen: [aten.sub, aten._to_copy, aten.ge, aten.lift_fresh, aten.clamp, aten.pow, aten.zeros, aten.where, aten.eq, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   base => where
#   clamp => clamp_min
#   eq => eq
#   ge => ge
#   lag => convert_element_type
#   mul => mul
#   pow_1 => pow_1
#   same => bitwise_and
#   sub => sub
#   tensor => full_default
#   to_2 => convert_element_type_1
#   zeros => full_default_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_3, %view_4), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub, torch.float32), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convert_element_type, 0), kwargs = {})
#   %full_default : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.9945994019508362), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type, 0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%full_default, %clamp_min), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %pow_1, %full_default_1), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %bitwise_and : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq, %unsqueeze_2), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bitwise_and, torch.float32), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, %convert_element_type_1), kwargs = {})
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
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_lift_fresh_mul_pow_sub_where_zeros_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp4 = 0.9945994019508362
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
    tl.store(out_ptr0 + (x4), tmp16, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ep/cepeqaqunhjhtetcicr2hyq772kazksdenv63qw2j5vbolajsu7n.py
# Topologically Sorted Source Nodes: [i, tensor, add, to_3, dec], Original ATen: [aten.arange, aten.lift_fresh, aten.add, aten._to_copy, aten.pow]
# Source node to ATen node mapping:
#   add => add
#   dec => pow_2
#   i => iota
#   tensor => full_default
#   to_3 => convert_element_type_2
# Graph fragment:
#   %iota : [num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %full_default : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.9945994019508362), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, 1), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.float32), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%full_default, %convert_element_type_2), kwargs = {})
triton_poi_fused__to_copy_add_arange_lift_fresh_pow_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_lift_fresh_pow_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/sg/csg3r3bkjfyskyd4fybamqgv26fst3vomfozw5ox57zczokwgini.py
# Topologically Sorted Source Nodes: [eq_2, ge_3, and__2], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
# Source node to ATen node mapping:
#   and__2 => bitwise_and_2
#   eq_2 => eq_2
#   ge_3 => ge_3
# Graph fragment:
#   %eq_2 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_6, %view_10), kwargs = {})
#   %ge_3 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_6, 0), kwargs = {})
#   %bitwise_and_2 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_2, %ge_3), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/xg/cxguy4lbqt5k7gi3436i4iavpkhnibw3zcojq4kxdm4huwb4uwjd.py
# Topologically Sorted Source Nodes: [mul_3, mul_4, S_1, stack], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_1 => add_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   stack => cat
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_5, %view_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %unsqueeze_6), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_8, %mul_4), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_1, %add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16], 1), kwargs = {})
triton_poi_fused_add_mul_stack_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 4096)
    x3 = xindex % 4096
    x4 = (xindex // 64)
    x1 = (xindex // 64) % 64
    x0 = xindex % 64
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (4096 + x3 + (65536*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (4032 + x0 + (65536*x2)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr1 + (x3 + (65536*x2)), tmp20, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ie/ciepsjejbf6jekba5ax762bdwks5sdvmosurkrackkcfsin4stp5.py
# Topologically Sorted Source Nodes: [eq_3, ge_4, and__3], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
# Source node to ATen node mapping:
#   and__3 => bitwise_and_3
#   eq_3 => eq_3
#   ge_4 => ge_4
# Graph fragment:
#   %eq_3 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_12, %view_12), kwargs = {})
#   %ge_4 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_12, 0), kwargs = {})
#   %bitwise_and_3 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_3, %ge_4), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/6i/c6ihcugidvjenwxrxi5idoegjsf6pbqzgtfkjpgbinp3ltwdhow6.py
# Topologically Sorted Source Nodes: [eq_4, ge_5, and__4], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
# Source node to ATen node mapping:
#   and__4 => bitwise_and_4
#   eq_4 => eq_4
#   ge_5 => ge_5
# Graph fragment:
#   %eq_4 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_18, %view_14), kwargs = {})
#   %ge_5 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_18, 0), kwargs = {})
#   %bitwise_and_4 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_4, %ge_5), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ak/cakfftdqb63gzuhxcfrqj6wqilfa5xcuft2q3esvty6h7chvctnj.py
# Topologically Sorted Source Nodes: [mul_1, unsqueeze_4, mul_2, S, mul_5, mul_6, S_2, mul_7, mul_8, S_3, stack], Original ATen: [aten.mul, aten.unsqueeze, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S => add_1
#   S_2 => add_3
#   S_3 => add_4
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   stack => cat
#   unsqueeze_4 => full_default_4
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_3, %view_9), kwargs = {})
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 1, 64], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %full_default_4), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_2, %mul_2), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_7, %view_9), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %unsqueeze_8), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_14, %mul_6), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_9, %view_9), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_10), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_20, %mul_8), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_1, %add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16], 1), kwargs = {})
triton_poi_fused_add_mul_stack_unsqueeze_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i1', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_unsqueeze_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 4096)
    x3 = xindex % 4096
    x1 = (xindex // 64) % 64
    x4 = (xindex // 64)
    x0 = xindex % 64
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (65536*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (12288 + x3 + (65536*x2)), None)
    tmp14 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp17 = tl.load(in_ptr0 + (12224 + x0 + (65536*x2)), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp20 = tl.load(in_ptr2 + (63))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp23 = tl.load(in_ptr5 + (4032 + x0 + (4096*x2)), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (8192 + x3 + (65536*x2)), None)
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
    tl.store(out_ptr0 + (x3 + (65536*x2)), tmp12, None)
    tl.store(out_ptr1 + (x5), tmp27, None)
    tl.store(out_ptr2 + (x3 + (65536*x2)), tmp27, None)
    tl.store(out_ptr3 + (x3 + (65536*x2)), tmp33, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/tl/ctlqdgqx2kzumpxdl4jv2ftqwqjodp3a34j6xa65rska47ncjxx2.py
# Topologically Sorted Source Nodes: [eq_5, ge_6, and__5], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
# Source node to ATen node mapping:
#   and__5 => bitwise_and_5
#   eq_5 => eq_5
#   ge_6 => ge_6
# Graph fragment:
#   %eq_5 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_24, %view_16), kwargs = {})
#   %ge_6 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_24, 0), kwargs = {})
#   %bitwise_and_5 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_5, %ge_6), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/zh/czhd2kqryiftjluqwiqud5sqzvddshznhmyc3733tjoqkc4la5w5.py
# Topologically Sorted Source Nodes: [eq_6, ge_7, and__6], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
# Source node to ATen node mapping:
#   and__6 => bitwise_and_6
#   eq_6 => eq_6
#   ge_7 => ge_7
# Graph fragment:
#   %eq_6 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_30, %view_18), kwargs = {})
#   %ge_7 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_30, 0), kwargs = {})
#   %bitwise_and_6 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_6, %ge_7), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/qm/cqmxekxrd7iwxz2ja6rtqr24t44kj2wg7o7m66v76cdmnu22gghb.py
# Topologically Sorted Source Nodes: [mul_9, mul_10, S_4, mul_11, mul_12, S_5, stack], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_4 => add_5
#   S_5 => add_6
#   mul_10 => mul_10
#   mul_11 => mul_11
#   mul_12 => mul_12
#   mul_9 => mul_9
#   stack => cat
# Graph fragment:
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_11, %view_9), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %unsqueeze_12), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_26, %mul_10), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_13, %view_9), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %unsqueeze_14), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %mul_12), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_1, %add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16], 1), kwargs = {})
triton_poi_fused_add_mul_stack_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 4096)
    x3 = xindex % 4096
    x4 = (xindex // 64)
    x1 = (xindex // 64) % 64
    x0 = xindex % 64
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (20480 + x3 + (65536*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (20416 + x0 + (65536*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (63))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr4 + (4032 + x0 + (4096*x2)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (16384 + x3 + (65536*x2)), None)
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
    tl.store(out_ptr1 + (x3 + (65536*x2)), tmp15, None)
    tl.store(out_ptr2 + (x3 + (65536*x2)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/dg/cdgw26fujhh7ytcg7446xesff6raz6xh5oovee2pdy57kxhmg7u5.py
# Topologically Sorted Source Nodes: [eq_7, ge_8, and__7], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
# Source node to ATen node mapping:
#   and__7 => bitwise_and_7
#   eq_7 => eq_7
#   ge_8 => ge_8
# Graph fragment:
#   %eq_7 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_36, %view_20), kwargs = {})
#   %ge_8 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_36, 0), kwargs = {})
#   %bitwise_and_7 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_7, %ge_8), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/5o/c5owtwwqigigiiszduw4wmw3hm3mewl5bbvnxeypu77rjtnnjrid.py
# Topologically Sorted Source Nodes: [eq_8, ge_9, and__8], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
# Source node to ATen node mapping:
#   and__8 => bitwise_and_8
#   eq_8 => eq_8
#   ge_9 => ge_9
# Graph fragment:
#   %eq_8 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_42, %view_22), kwargs = {})
#   %ge_9 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_42, 0), kwargs = {})
#   %bitwise_and_8 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_8, %ge_9), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ef/cef27nck53avhicizl4ljekvliujf5t3nk7ki73tqy22psdspewy.py
# Topologically Sorted Source Nodes: [mul_13, mul_14, S_6, mul_15, mul_16, S_7, stack], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_6 => add_7
#   S_7 => add_8
#   mul_13 => mul_13
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_16 => mul_16
#   stack => cat
# Graph fragment:
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_15, %view_9), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_16), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_38, %mul_14), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_17, %view_9), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %unsqueeze_18), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_44, %mul_16), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_1, %add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16], 1), kwargs = {})
triton_poi_fused_add_mul_stack_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_12', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 4096)
    x3 = xindex % 4096
    x4 = (xindex // 64)
    x1 = (xindex // 64) % 64
    x0 = xindex % 64
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (28672 + x3 + (65536*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (28608 + x0 + (65536*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (63))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr4 + (4032 + x0 + (4096*x2)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (24576 + x3 + (65536*x2)), None)
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
    tl.store(out_ptr1 + (x3 + (65536*x2)), tmp15, None)
    tl.store(out_ptr2 + (x3 + (65536*x2)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/zh/czhytujhtanqnk3zyrvbiift7juohdqj5iumzfn4i7cpgdhros53.py
# Topologically Sorted Source Nodes: [eq_9, ge_10, and__9], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
# Source node to ATen node mapping:
#   and__9 => bitwise_and_9
#   eq_9 => eq_9
#   ge_10 => ge_10
# Graph fragment:
#   %eq_9 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_48, %view_24), kwargs = {})
#   %ge_10 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_48, 0), kwargs = {})
#   %bitwise_and_9 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_9, %ge_10), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_13', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/4w/c4w6gglcd2pjrrmxzrwbsr4kk7rogjcj3ezdx5psx2lcs2x4qw5e.py
# Topologically Sorted Source Nodes: [eq_10, ge_11, and__10], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
# Source node to ATen node mapping:
#   and__10 => bitwise_and_10
#   eq_10 => eq_10
#   ge_11 => ge_11
# Graph fragment:
#   %eq_10 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_54, %view_26), kwargs = {})
#   %ge_11 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_54, 0), kwargs = {})
#   %bitwise_and_10 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_10, %ge_11), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_14', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/m6/cm6sqt4b4aq2trztcr53pfz5vtfhg6uaijocoys4gqxld3lc6hnt.py
# Topologically Sorted Source Nodes: [mul_17, mul_18, S_8, mul_19, mul_20, S_9, stack], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_8 => add_9
#   S_9 => add_10
#   mul_17 => mul_17
#   mul_18 => mul_18
#   mul_19 => mul_19
#   mul_20 => mul_20
#   stack => cat
# Graph fragment:
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_19, %view_9), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %unsqueeze_20), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_50, %mul_18), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_21, %view_9), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_22), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_56, %mul_20), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_1, %add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16], 1), kwargs = {})
triton_poi_fused_add_mul_stack_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_15', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 4096)
    x3 = xindex % 4096
    x4 = (xindex // 64)
    x1 = (xindex // 64) % 64
    x0 = xindex % 64
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (36864 + x3 + (65536*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (36800 + x0 + (65536*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (63))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr4 + (4032 + x0 + (4096*x2)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (32768 + x3 + (65536*x2)), None)
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
    tl.store(out_ptr1 + (x3 + (65536*x2)), tmp15, None)
    tl.store(out_ptr2 + (x3 + (65536*x2)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/c2/cc26yvyyfmyoh2h45jwm4fykzqkfgli6ib47zuq4ujg5qzsoa2xo.py
# Topologically Sorted Source Nodes: [eq_11, ge_12, and__11], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
# Source node to ATen node mapping:
#   and__11 => bitwise_and_11
#   eq_11 => eq_11
#   ge_12 => ge_12
# Graph fragment:
#   %eq_11 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_60, %view_28), kwargs = {})
#   %ge_12 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_60, 0), kwargs = {})
#   %bitwise_and_11 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_11, %ge_12), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_16', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ck/cckjx5mnc4nfngwcngirfcyyi2itypiycx3ut53lfo3bpsbjlnjp.py
# Topologically Sorted Source Nodes: [eq_12, ge_13, and__12], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
# Source node to ATen node mapping:
#   and__12 => bitwise_and_12
#   eq_12 => eq_12
#   ge_13 => ge_13
# Graph fragment:
#   %eq_12 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_66, %view_30), kwargs = {})
#   %ge_13 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_66, 0), kwargs = {})
#   %bitwise_and_12 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_12, %ge_13), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_17', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/gr/cgr3hqppladqmh2sqrcc4pjhoeg5idi3y6424v6rbxkha2x3v4ev.py
# Topologically Sorted Source Nodes: [mul_21, mul_22, S_10, mul_23, mul_24, S_11, stack], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_10 => add_11
#   S_11 => add_12
#   mul_21 => mul_21
#   mul_22 => mul_22
#   mul_23 => mul_23
#   mul_24 => mul_24
#   stack => cat
# Graph fragment:
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_23, %view_9), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %unsqueeze_24), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_62, %mul_22), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_25, %view_9), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %unsqueeze_26), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_68, %mul_24), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_1, %add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16], 1), kwargs = {})
triton_poi_fused_add_mul_stack_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_18', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 4096)
    x3 = xindex % 4096
    x4 = (xindex // 64)
    x1 = (xindex // 64) % 64
    x0 = xindex % 64
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (45056 + x3 + (65536*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (44992 + x0 + (65536*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (63))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr4 + (4032 + x0 + (4096*x2)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (40960 + x3 + (65536*x2)), None)
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
    tl.store(out_ptr1 + (x3 + (65536*x2)), tmp15, None)
    tl.store(out_ptr2 + (x3 + (65536*x2)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/lp/clpw4xzptyi4mg2so72nxtswok7x4t4g7zi2ouid4fitcupvqxwy.py
# Topologically Sorted Source Nodes: [eq_13, ge_14, and__13], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
# Source node to ATen node mapping:
#   and__13 => bitwise_and_13
#   eq_13 => eq_13
#   ge_14 => ge_14
# Graph fragment:
#   %eq_13 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_72, %view_32), kwargs = {})
#   %ge_14 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_72, 0), kwargs = {})
#   %bitwise_and_13 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_13, %ge_14), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_19', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/oh/coh3whsqjqo6kailjkngole73h3ivdp7ifscttx2isc4te2bb5yx.py
# Topologically Sorted Source Nodes: [eq_14, ge_15, and__14], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
# Source node to ATen node mapping:
#   and__14 => bitwise_and_14
#   eq_14 => eq_14
#   ge_15 => ge_15
# Graph fragment:
#   %eq_14 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_78, %view_34), kwargs = {})
#   %ge_15 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_78, 0), kwargs = {})
#   %bitwise_and_14 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_14, %ge_15), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_20', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/xr/cxrqyluhh7vbfgrvqpuifjt4bs7edibnnwbmblqfwxxdeymgumy3.py
# Topologically Sorted Source Nodes: [mul_25, mul_26, S_12, mul_27, mul_28, S_13, stack], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_12 => add_13
#   S_13 => add_14
#   mul_25 => mul_25
#   mul_26 => mul_26
#   mul_27 => mul_27
#   mul_28 => mul_28
#   stack => cat
# Graph fragment:
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_27, %view_9), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_28), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_74, %mul_26), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_29, %view_9), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, %unsqueeze_30), kwargs = {})
#   %add_14 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_80, %mul_28), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_1, %add_2, %add_3, %add_4, %add_5, %add_6, %add_7, %add_8, %add_9, %add_10, %add_11, %add_12, %add_13, %add_14, %add_15, %add_16], 1), kwargs = {})
triton_poi_fused_add_mul_stack_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_21', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 4096)
    x3 = xindex % 4096
    x4 = (xindex // 64)
    x1 = (xindex // 64) % 64
    x0 = xindex % 64
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (53248 + x3 + (65536*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (53184 + x0 + (65536*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (63))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr4 + (4032 + x0 + (4096*x2)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (49152 + x3 + (65536*x2)), None)
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
    tl.store(out_ptr1 + (x3 + (65536*x2)), tmp15, None)
    tl.store(out_ptr2 + (x3 + (65536*x2)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/jm/cjmcttmrbv4oognf545yxisz5x5qsqgphsxs5wtfqv3nzdjd2qxv.py
# Topologically Sorted Source Nodes: [eq_15, ge_16, and__15], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
# Source node to ATen node mapping:
#   and__15 => bitwise_and_15
#   eq_15 => eq_15
#   ge_16 => ge_16
# Graph fragment:
#   %eq_15 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_84, %view_36), kwargs = {})
#   %ge_16 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_84, 0), kwargs = {})
#   %bitwise_and_15 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_15, %ge_16), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_22', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ou/cou7qmedxu3m7mrixfgskpns7dthzzqhqfjgl4wdwmeiumrqcls6.py
# Topologically Sorted Source Nodes: [mul_31], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_31 => mul_31
# Graph fragment:
#   %mul_31 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_33, %view_9), kwargs = {})
triton_poi_fused_mul_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_23', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/iu/ciuqai3oe3goruf4mnsnj7moyzus62ogus7w3ka3zwedmzeab7wa.py
# Topologically Sorted Source Nodes: [mul_29, mul_30, S_14, mul_32, S_15], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   S_14 => add_15
#   S_15 => add_16
#   mul_29 => mul_29
#   mul_30 => mul_30
#   mul_32 => mul_32
# Graph fragment:
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_31, %view_9), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %unsqueeze_32), kwargs = {})
#   %add_15 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_86, %mul_30), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_34), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_92, %mul_32), kwargs = {})
triton_poi_fused_add_mul_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_24', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 4096)
    x3 = xindex % 4096
    x4 = (xindex // 64)
    x1 = (xindex // 64) % 64
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (57344 + x3 + (65536*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (4032 + x0 + (4096*x2)), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (61440 + x3 + (65536*x2)), None)
    tmp9 = tl.load(in_ptr4 + (x4), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (61376 + x0 + (65536*x2)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3 + (65536*x2)), tmp7, None)
    tl.store(out_ptr1 + (x3 + (65536*x2)), tmp19, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/hf/chfs3hfz5n3wmp33d6gr5x4pbcanbk5vgozje2wd2hrr2xjxtjiq.py
# Topologically Sorted Source Nodes: [tensor, to_20, add_17, pow_3, sub_1, Z], Original ATen: [aten.lift_fresh, aten._to_copy, aten.add, aten.pow, aten.rsub, aten.div]
# Source node to ATen node mapping:
#   Z => div
#   add_17 => add_17
#   pow_3 => pow_3
#   sub_1 => sub_1
#   tensor => full_default
#   to_20 => convert_element_type_19
# Graph fragment:
#   %full_default : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.9945994019508362), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_19 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_4, torch.float32), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_19, 1.0), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%full_default, %add_17), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %pow_3), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, 0.0054005765163668285), kwargs = {})
triton_poi_fused__to_copy_add_div_lift_fresh_pow_rsub_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_lift_fresh_pow_rsub_25', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/3a/c3azgrh5cmekimwqtnsbexyi2zbbnru5biupuadw6ztvapjqvcc3.py
# Topologically Sorted Source Nodes: [ge_18], Original ATen: [aten.ge]
# Source node to ATen node mapping:
#   ge_18 => ge_18
# Graph fragment:
#   %ge_18 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%primals_1, 0), kwargs = {})
triton_poi_fused_ge_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ge_26', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/2d/c2docx73sx4jyukhjsemy36uxdqmpagwrwsns2vu2gxht55shri4.py
# Topologically Sorted Source Nodes: [truediv_1, to_21, mul_33], Original ATen: [aten.div, aten._to_copy, aten.mul]
# Source node to ATen node mapping:
#   mul_33 => mul_33
#   to_21 => convert_element_type_20
#   truediv_1 => div_1
# Graph fragment:
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_41, %unsqueeze_35), kwargs = {})
#   %convert_element_type_20 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_36, torch.float32), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %convert_element_type_20), kwargs = {})
triton_poi_fused__to_copy_div_mul_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_div_mul_27', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp0 / tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (32, 1024), (1024, 1))
    assert_size_stride(primals_2, (64, 384), (384, 1))
    assert_size_stride(primals_3, (32768, 384), (384, 1))
    assert_size_stride(primals_4, (32, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(primals_3, reinterpret_tensor(primals_2, (384, 64), (1, 384), 0), out=buf0)
        buf1 = empty_strided_cuda((32, 16, 64, 64), (65536, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sub, lag, ge, tensor, clamp, pow_1, zeros, base, eq, same, to_2, mul], Original ATen: [aten.sub, aten._to_copy, aten.ge, aten.lift_fresh, aten.clamp, aten.pow, aten.zeros, aten.where, aten.eq, aten.bitwise_and, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_lift_fresh_mul_pow_sub_where_zeros_0.run(primals_1, buf1, 2097152, grid=grid(2097152), stream=stream0)
        buf2 = empty_strided_cuda((512, 64, 64), (4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [S_in], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1, (512, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf0, (512, 64, 64), (4096, 64, 1), 0), out=buf2)
        buf3 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [i, tensor, add, to_3, dec], Original ATen: [aten.arange, aten.lift_fresh, aten.add, aten._to_copy, aten.pow]
        triton_poi_fused__to_copy_add_arange_lift_fresh_pow_1.run(buf3, 64, grid=grid(64), stream=stream0)
        buf4 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [eq_2, ge_3, and__2], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
        triton_poi_fused_bitwise_and_eq_ge_2.run(primals_1, buf4, 2048, grid=grid(2048), stream=stream0)
        buf5 = empty_strided_cuda((32, 64, 64), (4096, 64, 1), torch.float32)
        buf42 = reinterpret_tensor(buf0, (32, 1024, 64), (65536, 64, 1), 0); del buf0  # reuse
        buf27 = reinterpret_tensor(buf42, (32, 64, 64), (65536, 64, 1), 4096)  # alias
        # Topologically Sorted Source Nodes: [mul_3, mul_4, S_1, stack], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_3.run(buf2, buf4, buf3, primals_1, buf5, buf27, 131072, grid=grid(131072), stream=stream0)
        buf6 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [eq_3, ge_4, and__3], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
        triton_poi_fused_bitwise_and_eq_ge_4.run(primals_1, buf6, 2048, grid=grid(2048), stream=stream0)
        buf7 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [eq_4, ge_5, and__4], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
        triton_poi_fused_bitwise_and_eq_ge_5.run(primals_1, buf7, 2048, grid=grid(2048), stream=stream0)
        buf26 = reinterpret_tensor(buf42, (32, 64, 64), (65536, 64, 1), 0)  # alias
        buf8 = empty_strided_cuda((32, 64, 64), (4096, 64, 1), torch.float32)
        buf29 = reinterpret_tensor(buf42, (32, 64, 64), (65536, 64, 1), 12288)  # alias
        buf28 = reinterpret_tensor(buf42, (32, 64, 64), (65536, 64, 1), 8192)  # alias
        # Topologically Sorted Source Nodes: [mul_1, unsqueeze_4, mul_2, S, mul_5, mul_6, S_2, mul_7, mul_8, S_3, stack], Original ATen: [aten.mul, aten.unsqueeze, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_unsqueeze_6.run(buf2, primals_1, buf3, buf7, buf6, buf5, buf26, buf8, buf29, buf28, 131072, grid=grid(131072), stream=stream0)
        buf9 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [eq_5, ge_6, and__5], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
        triton_poi_fused_bitwise_and_eq_ge_7.run(primals_1, buf9, 2048, grid=grid(2048), stream=stream0)
        buf10 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [eq_6, ge_7, and__6], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
        triton_poi_fused_bitwise_and_eq_ge_8.run(primals_1, buf10, 2048, grid=grid(2048), stream=stream0)
        buf11 = buf5; del buf5  # reuse
        buf31 = reinterpret_tensor(buf42, (32, 64, 64), (65536, 64, 1), 20480)  # alias
        buf30 = reinterpret_tensor(buf42, (32, 64, 64), (65536, 64, 1), 16384)  # alias
        # Topologically Sorted Source Nodes: [mul_9, mul_10, S_4, mul_11, mul_12, S_5, stack], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_9.run(buf2, buf10, buf3, buf9, buf8, buf11, buf31, buf30, 131072, grid=grid(131072), stream=stream0)
        buf12 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [eq_7, ge_8, and__7], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
        triton_poi_fused_bitwise_and_eq_ge_10.run(primals_1, buf12, 2048, grid=grid(2048), stream=stream0)
        buf13 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [eq_8, ge_9, and__8], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
        triton_poi_fused_bitwise_and_eq_ge_11.run(primals_1, buf13, 2048, grid=grid(2048), stream=stream0)
        buf14 = buf8; del buf8  # reuse
        buf33 = reinterpret_tensor(buf42, (32, 64, 64), (65536, 64, 1), 28672)  # alias
        buf32 = reinterpret_tensor(buf42, (32, 64, 64), (65536, 64, 1), 24576)  # alias
        # Topologically Sorted Source Nodes: [mul_13, mul_14, S_6, mul_15, mul_16, S_7, stack], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_12.run(buf2, buf13, buf3, buf12, buf11, buf14, buf33, buf32, 131072, grid=grid(131072), stream=stream0)
        buf15 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [eq_9, ge_10, and__9], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
        triton_poi_fused_bitwise_and_eq_ge_13.run(primals_1, buf15, 2048, grid=grid(2048), stream=stream0)
        buf16 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [eq_10, ge_11, and__10], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
        triton_poi_fused_bitwise_and_eq_ge_14.run(primals_1, buf16, 2048, grid=grid(2048), stream=stream0)
        buf17 = buf11; del buf11  # reuse
        buf35 = reinterpret_tensor(buf42, (32, 64, 64), (65536, 64, 1), 36864)  # alias
        buf34 = reinterpret_tensor(buf42, (32, 64, 64), (65536, 64, 1), 32768)  # alias
        # Topologically Sorted Source Nodes: [mul_17, mul_18, S_8, mul_19, mul_20, S_9, stack], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_15.run(buf2, buf16, buf3, buf15, buf14, buf17, buf35, buf34, 131072, grid=grid(131072), stream=stream0)
        buf18 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [eq_11, ge_12, and__11], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
        triton_poi_fused_bitwise_and_eq_ge_16.run(primals_1, buf18, 2048, grid=grid(2048), stream=stream0)
        buf19 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [eq_12, ge_13, and__12], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
        triton_poi_fused_bitwise_and_eq_ge_17.run(primals_1, buf19, 2048, grid=grid(2048), stream=stream0)
        buf20 = buf14; del buf14  # reuse
        buf37 = reinterpret_tensor(buf42, (32, 64, 64), (65536, 64, 1), 45056)  # alias
        buf36 = reinterpret_tensor(buf42, (32, 64, 64), (65536, 64, 1), 40960)  # alias
        # Topologically Sorted Source Nodes: [mul_21, mul_22, S_10, mul_23, mul_24, S_11, stack], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_18.run(buf2, buf19, buf3, buf18, buf17, buf20, buf37, buf36, 131072, grid=grid(131072), stream=stream0)
        buf21 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [eq_13, ge_14, and__13], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
        triton_poi_fused_bitwise_and_eq_ge_19.run(primals_1, buf21, 2048, grid=grid(2048), stream=stream0)
        buf22 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [eq_14, ge_15, and__14], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
        triton_poi_fused_bitwise_and_eq_ge_20.run(primals_1, buf22, 2048, grid=grid(2048), stream=stream0)
        buf23 = buf17; del buf17  # reuse
        buf39 = reinterpret_tensor(buf42, (32, 64, 64), (65536, 64, 1), 53248)  # alias
        buf38 = reinterpret_tensor(buf42, (32, 64, 64), (65536, 64, 1), 49152)  # alias
        # Topologically Sorted Source Nodes: [mul_25, mul_26, S_12, mul_27, mul_28, S_13, stack], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_21.run(buf2, buf22, buf3, buf21, buf20, buf23, buf39, buf38, 131072, grid=grid(131072), stream=stream0)
        del buf20
        buf24 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [eq_15, ge_16, and__15], Original ATen: [aten.eq, aten.ge, aten.bitwise_and]
        triton_poi_fused_bitwise_and_eq_ge_22.run(primals_1, buf24, 2048, grid=grid(2048), stream=stream0)
        buf25 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_31], Original ATen: [aten.mul]
        triton_poi_fused_mul_23.run(primals_1, buf3, buf25, 2048, grid=grid(2048), stream=stream0)
        buf40 = reinterpret_tensor(buf42, (32, 64, 64), (65536, 64, 1), 57344)  # alias
        buf41 = reinterpret_tensor(buf42, (32, 64, 64), (65536, 64, 1), 61440)  # alias
        # Topologically Sorted Source Nodes: [mul_29, mul_30, S_14, mul_32, S_15], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_24.run(buf2, buf24, buf3, buf23, buf25, buf40, buf41, 131072, grid=grid(131072), stream=stream0)
        del buf23
        buf43 = empty_strided_cuda((32, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [tensor, to_20, add_17, pow_3, sub_1, Z], Original ATen: [aten.lift_fresh, aten._to_copy, aten.add, aten.pow, aten.rsub, aten.div]
        triton_poi_fused__to_copy_add_div_lift_fresh_pow_rsub_25.run(primals_4, buf43, 32768, grid=grid(32768), stream=stream0)
        del buf26
        del buf27
        del buf28
        del buf29
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
        del primals_4
        buf44 = empty_strided_cuda((32, 1024), (1024, 1), torch.bool)
        # Topologically Sorted Source Nodes: [ge_18], Original ATen: [aten.ge]
        triton_poi_fused_ge_26.run(primals_1, buf44, 32768, grid=grid(32768), stream=stream0)
        del primals_1
        buf45 = reinterpret_tensor(buf2, (32, 1024, 64), (65536, 64, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [truediv_1, to_21, mul_33], Original ATen: [aten.div, aten._to_copy, aten.mul]
        triton_poi_fused__to_copy_div_mul_27.run(buf42, buf43, buf44, buf45, 2097152, grid=grid(2097152), stream=stream0)
        del buf42
    return (reinterpret_tensor(buf45, (32768, 64), (64, 1), 0), primals_3, reinterpret_tensor(buf3, (1, 64, 1), (64, 1, 1), 0), buf4, buf6, buf7, buf9, buf10, buf12, buf13, buf15, buf16, buf18, buf19, buf21, buf22, buf24, buf25, reinterpret_tensor(buf43, (32, 1024, 1), (1024, 1, 1), 0), reinterpret_tensor(buf44, (32, 1024, 1), (1024, 1, 1), 0), reinterpret_tensor(buf1, (512, 64, 64), (4096, 1, 64), 0), primals_2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    primals_2 = rand_strided((64, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
