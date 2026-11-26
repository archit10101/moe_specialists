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


# kernel path: /home/zapan4/moe-research/.cache/inductor/5r/c5rcnufco3ozwj5vhadmffrbfrfvd4wie6ekcr2kssmviqz5w3ma.py
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
#   %ge : [num_users=3] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convert_element_type, 0), kwargs = {})
#   %full_default : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.9576032757759094), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type, 0), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%full_default, %clamp_min), kwargs = {})
#   %full_default_1 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
    tmp4 = 0.9576032757759094
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/ef/cefudwa234zuggweikhruaxy2urznjyzp3dwqeruk2cflu44xddt.py
# Topologically Sorted Source Nodes: [i, tensor, add, to_3, dec], Original ATen: [aten.arange, aten.lift_fresh, aten.add, aten._to_copy, aten.pow]
# Source node to ATen node mapping:
#   add => add
#   dec => pow_2
#   i => iota
#   tensor => full_default
#   to_3 => convert_element_type_2
# Graph fragment:
#   %iota : [num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %full_default : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.9576032757759094), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, 1), kwargs = {})
#   %convert_element_type_2 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.float32), kwargs = {})
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
    tmp0 = 0.9576032757759094
    tmp1 = 1 + x0
    tmp2 = tmp1.to(tl.float32)
    tmp3 = libdevice.pow(tmp0, tmp2)
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/qn/cqngdmxxc2xs7afcijxmdgpqmwqt77twbx3rsldavx2gd6ipqjov.py
# Topologically Sorted Source Nodes: [eq_2, ge_3, and__2, mul_37, mul_71], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__2 => bitwise_and_2
#   eq_2 => eq_2
#   ge_3 => ge_3
#   mul_37 => mul_37
#   mul_71 => mul_71
# Graph fragment:
#   %eq_2 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_6, %view_10), kwargs = {})
#   %ge_3 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_6, 0), kwargs = {})
#   %bitwise_and_2 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_2, %ge_3), kwargs = {})
#   %mul_37 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_5, %view_50), kwargs = {})
#   %mul_71 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_5, %view_91), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp7 = 0.9945994019508362
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.999323308467865
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/qh/cqh5tsvlcwga72opnpdtnkbpjp23ek2cb4xxsrc2x3a72lv6xlxw.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x4 = (xindex // 21)
    x1 = (xindex // 21) % 64
    x0 = xindex % 21
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (1344 + x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1323 + x0 + (21504*x2)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr1 + (x3 + (21504*x2)), tmp20, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ib/cibth7mljhba7wcebeqvu6hiuhkbp6in7loxon7tahh27s75dbz4.py
# Topologically Sorted Source Nodes: [eq_3, ge_4, and__3, mul_39, mul_73], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__3 => bitwise_and_3
#   eq_3 => eq_3
#   ge_4 => ge_4
#   mul_39 => mul_39
#   mul_73 => mul_73
# Graph fragment:
#   %eq_3 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_12, %view_12), kwargs = {})
#   %ge_4 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_12, 0), kwargs = {})
#   %bitwise_and_3 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_3, %ge_4), kwargs = {})
#   %mul_39 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_7, %view_50), kwargs = {})
#   %mul_73 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_7, %view_91), kwargs = {})
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
    tmp0 = tl.load(in_ptr0 + (128 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (127 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9945994019508362
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.999323308467865
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/iu/ciu3fb7hd4mndgh52gn63ocnzjbxrjw2tftuyssqj7y3onfg7lxf.py
# Topologically Sorted Source Nodes: [eq_4, ge_5, and__4, mul_41, mul_75], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__4 => bitwise_and_4
#   eq_4 => eq_4
#   ge_5 => ge_5
#   mul_41 => mul_41
#   mul_75 => mul_75
# Graph fragment:
#   %eq_4 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_18, %view_14), kwargs = {})
#   %ge_5 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_18, 0), kwargs = {})
#   %bitwise_and_4 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_4, %ge_5), kwargs = {})
#   %mul_41 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_9, %view_50), kwargs = {})
#   %mul_75 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_9, %view_91), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp7 = 0.9945994019508362
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.999323308467865
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/om/comzomdnif7vnymg5wz4jrhiabwups4p5nqd3skfmxak3bs7di3r.py
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
#   %full_default_4 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([32, 1, 21], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i1', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_unsqueeze_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x1 = (xindex // 21) % 64
    x4 = (xindex // 21)
    x0 = xindex % 21
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (4032 + x3 + (21504*x2)), None)
    tmp14 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp17 = tl.load(in_ptr0 + (4011 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp20 = tl.load(in_ptr2 + (63))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp23 = tl.load(in_ptr5 + (1323 + x0 + (1344*x2)), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (2688 + x3 + (21504*x2)), None)
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
    tl.store(out_ptr0 + (x3 + (21504*x2)), tmp12, None)
    tl.store(out_ptr1 + (x5), tmp27, None)
    tl.store(out_ptr2 + (x3 + (21504*x2)), tmp27, None)
    tl.store(out_ptr3 + (x3 + (21504*x2)), tmp33, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/uc/cuckhd4rckyaouloyw2faoaor6covlwtdnufnxucfyavgkqusrzu.py
# Topologically Sorted Source Nodes: [eq_5, ge_6, and__5, mul_43, mul_77], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__5 => bitwise_and_5
#   eq_5 => eq_5
#   ge_6 => ge_6
#   mul_43 => mul_43
#   mul_77 => mul_77
# Graph fragment:
#   %eq_5 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_24, %view_16), kwargs = {})
#   %ge_6 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_24, 0), kwargs = {})
#   %bitwise_and_5 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_5, %ge_6), kwargs = {})
#   %mul_43 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_11, %view_50), kwargs = {})
#   %mul_77 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_11, %view_91), kwargs = {})
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
    tmp0 = tl.load(in_ptr0 + (256 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (255 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9945994019508362
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.999323308467865
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/3a/c3arhcfxycm637dlsrb6xs2xzryv5rejssdgwx7c5b2r7fgpmvix.py
# Topologically Sorted Source Nodes: [eq_6, ge_7, and__6, mul_45, mul_79], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__6 => bitwise_and_6
#   eq_6 => eq_6
#   ge_7 => ge_7
#   mul_45 => mul_45
#   mul_79 => mul_79
# Graph fragment:
#   %eq_6 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_30, %view_18), kwargs = {})
#   %ge_7 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_30, 0), kwargs = {})
#   %bitwise_and_6 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_6, %ge_7), kwargs = {})
#   %mul_45 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_13, %view_50), kwargs = {})
#   %mul_79 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_13, %view_91), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp7 = 0.9945994019508362
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.999323308467865
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/s7/cs7ublzlvuxf5nlaetxjibs2jppwhgtlmb6yg55uvql3juinzyps.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x4 = (xindex // 21)
    x1 = (xindex // 21) % 64
    x0 = xindex % 21
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (6720 + x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (6699 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (63))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr4 + (1323 + x0 + (1344*x2)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (5376 + x3 + (21504*x2)), None)
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
    tl.store(out_ptr1 + (x3 + (21504*x2)), tmp15, None)
    tl.store(out_ptr2 + (x3 + (21504*x2)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ei/cei7i2aoycgsx73zur5vamdtjh7mlbjyfijttqgddurqcdb6pfli.py
# Topologically Sorted Source Nodes: [eq_7, ge_8, and__7, mul_47, mul_81], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__7 => bitwise_and_7
#   eq_7 => eq_7
#   ge_8 => ge_8
#   mul_47 => mul_47
#   mul_81 => mul_81
# Graph fragment:
#   %eq_7 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_36, %view_20), kwargs = {})
#   %ge_8 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_36, 0), kwargs = {})
#   %bitwise_and_7 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_7, %ge_8), kwargs = {})
#   %mul_47 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_15, %view_50), kwargs = {})
#   %mul_81 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_15, %view_91), kwargs = {})
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
    tmp0 = tl.load(in_ptr0 + (384 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (383 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9945994019508362
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.999323308467865
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/y4/cy4pvsspzj6erfe3lvds3tfl5gb5qlsfjp74sfebasodxjpx22vk.py
# Topologically Sorted Source Nodes: [eq_8, ge_9, and__8, mul_49, mul_83], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__8 => bitwise_and_8
#   eq_8 => eq_8
#   ge_9 => ge_9
#   mul_49 => mul_49
#   mul_83 => mul_83
# Graph fragment:
#   %eq_8 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_42, %view_22), kwargs = {})
#   %ge_9 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_42, 0), kwargs = {})
#   %bitwise_and_8 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_8, %ge_9), kwargs = {})
#   %mul_49 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_17, %view_50), kwargs = {})
#   %mul_83 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_17, %view_91), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp7 = 0.9945994019508362
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.999323308467865
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/z6/cz6hytekww2mq4grdmqypjtrjidce7pwrh3mgqovggova5idtuov.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_12', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x4 = (xindex // 21)
    x1 = (xindex // 21) % 64
    x0 = xindex % 21
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (9408 + x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (9387 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (63))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr4 + (1323 + x0 + (1344*x2)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (8064 + x3 + (21504*x2)), None)
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
    tl.store(out_ptr1 + (x3 + (21504*x2)), tmp15, None)
    tl.store(out_ptr2 + (x3 + (21504*x2)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/a5/ca5teuwttwndmlqvxwnnrgktiv3iswczlwlvg7t2yh7ybwvwd6p2.py
# Topologically Sorted Source Nodes: [eq_9, ge_10, and__9, mul_51, mul_85], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__9 => bitwise_and_9
#   eq_9 => eq_9
#   ge_10 => ge_10
#   mul_51 => mul_51
#   mul_85 => mul_85
# Graph fragment:
#   %eq_9 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_48, %view_24), kwargs = {})
#   %ge_10 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_48, 0), kwargs = {})
#   %bitwise_and_9 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_9, %ge_10), kwargs = {})
#   %mul_51 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_19, %view_50), kwargs = {})
#   %mul_85 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_19, %view_91), kwargs = {})
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
    tmp0 = tl.load(in_ptr0 + (512 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (511 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9945994019508362
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.999323308467865
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/h4/ch4rv552g76q4jhveslp2smswoh5oant6gmvv7udv7zs6of5s7vr.py
# Topologically Sorted Source Nodes: [eq_10, ge_11, and__10, mul_53, mul_87], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__10 => bitwise_and_10
#   eq_10 => eq_10
#   ge_11 => ge_11
#   mul_53 => mul_53
#   mul_87 => mul_87
# Graph fragment:
#   %eq_10 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_54, %view_26), kwargs = {})
#   %ge_11 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_54, 0), kwargs = {})
#   %bitwise_and_10 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_10, %ge_11), kwargs = {})
#   %mul_53 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_21, %view_50), kwargs = {})
#   %mul_87 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_21, %view_91), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_14', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp7 = 0.9945994019508362
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.999323308467865
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/2t/c2t5tw7thvd7fxzz75at4zpkgczrqoo5iuggi7bxlcrrem4qzbv6.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_15', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x4 = (xindex // 21)
    x1 = (xindex // 21) % 64
    x0 = xindex % 21
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (12096 + x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (12075 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (63))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr4 + (1323 + x0 + (1344*x2)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (10752 + x3 + (21504*x2)), None)
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
    tl.store(out_ptr1 + (x3 + (21504*x2)), tmp15, None)
    tl.store(out_ptr2 + (x3 + (21504*x2)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ep/ceplfa5cd4aay7ruvlahlf3g55p2ioywlj6w5amyjobi2mulzuqj.py
# Topologically Sorted Source Nodes: [eq_11, ge_12, and__11, mul_55, mul_89], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__11 => bitwise_and_11
#   eq_11 => eq_11
#   ge_12 => ge_12
#   mul_55 => mul_55
#   mul_89 => mul_89
# Graph fragment:
#   %eq_11 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_60, %view_28), kwargs = {})
#   %ge_12 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_60, 0), kwargs = {})
#   %bitwise_and_11 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_11, %ge_12), kwargs = {})
#   %mul_55 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_23, %view_50), kwargs = {})
#   %mul_89 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_23, %view_91), kwargs = {})
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
    tmp0 = tl.load(in_ptr0 + (640 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (639 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9945994019508362
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.999323308467865
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/nc/cncb6snv7w25sno2jbt2j2g4ioqufb4pdwgtkpqe3nwfya6dcltu.py
# Topologically Sorted Source Nodes: [eq_12, ge_13, and__12, mul_57, mul_91], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__12 => bitwise_and_12
#   eq_12 => eq_12
#   ge_13 => ge_13
#   mul_57 => mul_57
#   mul_91 => mul_91
# Graph fragment:
#   %eq_12 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_66, %view_30), kwargs = {})
#   %ge_13 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_66, 0), kwargs = {})
#   %bitwise_and_12 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_12, %ge_13), kwargs = {})
#   %mul_57 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_25, %view_50), kwargs = {})
#   %mul_91 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_25, %view_91), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_17', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp7 = 0.9945994019508362
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.999323308467865
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ux/cuxrfwxf66b444i4eqqhs7abn3zrkq3udvkieghozzlwpicaljqa.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_18', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x4 = (xindex // 21)
    x1 = (xindex // 21) % 64
    x0 = xindex % 21
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (14784 + x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (14763 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (63))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr4 + (1323 + x0 + (1344*x2)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (13440 + x3 + (21504*x2)), None)
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
    tl.store(out_ptr1 + (x3 + (21504*x2)), tmp15, None)
    tl.store(out_ptr2 + (x3 + (21504*x2)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/o4/co44hhiyuftytkg2e2ycgjv32cv45kat2eu2mah2fgklmjpanonn.py
# Topologically Sorted Source Nodes: [eq_13, ge_14, and__13, mul_59, mul_93], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__13 => bitwise_and_13
#   eq_13 => eq_13
#   ge_14 => ge_14
#   mul_59 => mul_59
#   mul_93 => mul_93
# Graph fragment:
#   %eq_13 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_72, %view_32), kwargs = {})
#   %ge_14 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_72, 0), kwargs = {})
#   %bitwise_and_13 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_13, %ge_14), kwargs = {})
#   %mul_59 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_27, %view_50), kwargs = {})
#   %mul_93 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_27, %view_91), kwargs = {})
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
    tmp0 = tl.load(in_ptr0 + (768 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (767 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9945994019508362
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.999323308467865
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/qi/cqi7v77cp2kf4s6vju5yjhra7bvugxxo4li7tcvst6hcd5auxtlt.py
# Topologically Sorted Source Nodes: [eq_14, ge_15, and__14, mul_61, mul_95], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__14 => bitwise_and_14
#   eq_14 => eq_14
#   ge_15 => ge_15
#   mul_61 => mul_61
#   mul_95 => mul_95
# Graph fragment:
#   %eq_14 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_78, %view_34), kwargs = {})
#   %ge_15 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_78, 0), kwargs = {})
#   %bitwise_and_14 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_14, %ge_15), kwargs = {})
#   %mul_61 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_29, %view_50), kwargs = {})
#   %mul_95 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_29, %view_91), kwargs = {})
triton_poi_fused_bitwise_and_eq_ge_mul_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_eq_ge_mul_20', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp7 = 0.9945994019508362
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.999323308467865
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/sh/cshbkigga4gvhzqggakkxtqz3bhkptrwe3us36btdxiksyj5b7yl.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_21', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x4 = (xindex // 21)
    x1 = (xindex // 21) % 64
    x0 = xindex % 21
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (17472 + x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17451 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (63))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr4 + (1323 + x0 + (1344*x2)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (16128 + x3 + (21504*x2)), None)
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
    tl.store(out_ptr1 + (x3 + (21504*x2)), tmp15, None)
    tl.store(out_ptr2 + (x3 + (21504*x2)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/6o/c6oz3mf5sgebnoq4v7jt7ikyph3xnfq6b6hkbme6ygvdspi4nxwg.py
# Topologically Sorted Source Nodes: [eq_15, ge_16, and__15, mul_63, mul_97], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   and__15 => bitwise_and_15
#   eq_15 => eq_15
#   ge_16 => ge_16
#   mul_63 => mul_63
#   mul_97 => mul_97
# Graph fragment:
#   %eq_15 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%select_84, %view_36), kwargs = {})
#   %ge_16 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%select_84, 0), kwargs = {})
#   %bitwise_and_15 : [num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_15, %ge_16), kwargs = {})
#   %mul_63 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_31, %view_50), kwargs = {})
#   %mul_97 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_31, %view_91), kwargs = {})
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
    tmp0 = tl.load(in_ptr0 + (896 + x0 + (1024*x1)), None)
    tmp1 = tl.load(in_ptr0 + (895 + (1024*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp0 >= tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 0.9945994019508362
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.999323308467865
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp5, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp14, None)
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/oc/cocndeqvl6nozsga5teernihpi6oozghinqz2vdykakcnxp2rftk.py
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
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_24', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x4 = (xindex // 21)
    x1 = (xindex // 21) % 64
    x0 = xindex % 21
    tmp0 = tl.load(in_ptr0 + (18816 + x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (1323 + x0 + (1344*x2)), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (20160 + x3 + (21504*x2)), None)
    tmp9 = tl.load(in_ptr4 + (x4), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (20139 + x0 + (21504*x2)), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3 + (21504*x2)), tmp7, None)
    tl.store(out_ptr1 + (x3 + (21504*x2)), tmp19, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/an/canfwtqoz2m2ove3cy4civnc2sbew56ajgqka5q2ym22wohxm5ek.py
# Topologically Sorted Source Nodes: [tensor, to_20, add_17, pow_3, sub_1, Z], Original ATen: [aten.lift_fresh, aten._to_copy, aten.add, aten.pow, aten.rsub, aten.div]
# Source node to ATen node mapping:
#   Z => div
#   add_17 => add_17
#   pow_3 => pow_3
#   sub_1 => sub_1
#   tensor => full_default
#   to_20 => convert_element_type_19
# Graph fragment:
#   %full_default : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.9576032757759094), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_19 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_4, torch.float32), kwargs = {})
#   %add_17 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_19, 1.0), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%full_default, %add_17), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %pow_3), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, 0.0423967193014263), kwargs = {})
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
    tmp4 = 0.9576032757759094
    tmp5 = libdevice.pow(tmp4, tmp3)
    tmp6 = tmp2 - tmp5
    tmp7 = 23.58673068287051
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/tf/ctftwgpwjrbzpyoj2u3l3mkfov5vbcl64lxujeqqibd2yc4xlpjb.py
# Topologically Sorted Source Nodes: [sub, lag, ge, clamp, zeros, tensor_3, pow_4, base_1, eq_17, same_1, to_23, mul_34, tensor_6, pow_7, base_2, eq_34, same_2, to_44, mul_68], Original ATen: [aten.sub, aten._to_copy, aten.ge, aten.clamp, aten.zeros, aten.lift_fresh, aten.pow, aten.where, aten.eq, aten.bitwise_and, aten.mul]
# Source node to ATen node mapping:
#   base_1 => where_1
#   base_2 => where_2
#   clamp => clamp_min
#   eq_17 => eq_17
#   eq_34 => eq_34
#   ge => ge
#   lag => convert_element_type
#   mul_34 => mul_34
#   mul_68 => mul_68
#   pow_4 => pow_4
#   pow_7 => pow_7
#   same_1 => bitwise_and_17
#   same_2 => bitwise_and_34
#   sub => sub
#   tensor_3 => full_default_6
#   tensor_6 => full_default_12
#   to_23 => convert_element_type_22
#   to_44 => convert_element_type_43
#   zeros => full_default_1
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_3, %view_4), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub, torch.float32), kwargs = {})
#   %ge : [num_users=3] = call_function[target=torch.ops.aten.ge.Scalar](args = (%convert_element_type, 0), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type, 0), kwargs = {})
#   %full_default_1 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_6 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.9945994019508362), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%full_default_6, %clamp_min), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %pow_4, %full_default_1), kwargs = {})
#   %eq_17 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze, %unsqueeze_38), kwargs = {})
#   %bitwise_and_17 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_17, %unsqueeze_2), kwargs = {})
#   %convert_element_type_22 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bitwise_and_17, torch.float32), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_1, %convert_element_type_22), kwargs = {})
#   %full_default_12 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.999323308467865), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%full_default_12, %clamp_min), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %pow_7, %full_default_1), kwargs = {})
#   %eq_34 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze, %unsqueeze_75), kwargs = {})
#   %bitwise_and_34 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq_34, %unsqueeze_2), kwargs = {})
#   %convert_element_type_43 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bitwise_and_34, torch.float32), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_2, %convert_element_type_43), kwargs = {})
triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_lift_fresh_mul_pow_sub_where_zeros_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_lift_fresh_mul_pow_sub_where_zeros_27', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp17 = 0.999323308467865
    tmp18 = libdevice.pow(tmp17, tmp6)
    tmp19 = tl.where(tmp3, tmp18, tmp2)
    tmp20 = tmp19 * tmp15
    tl.store(out_ptr0 + (x4), tmp16, None)
    tl.store(out_ptr1 + (x4), tmp20, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/6x/c6xiyuzqw2h2jidw6osldl3krac64te6uokchmy5oq2ovw243pm4.py
# Topologically Sorted Source Nodes: [mul_38, S_18, mul_40, S_19, stack_1, mul_72, S_35, mul_74, S_36, stack_2], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_18 => add_20
#   S_19 => add_21
#   S_35 => add_38
#   S_36 => add_39
#   mul_38 => mul_38
#   mul_40 => mul_40
#   mul_72 => mul_72
#   mul_74 => mul_74
#   stack_1 => cat_1
#   stack_2 => cat_2
# Graph fragment:
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_43), kwargs = {})
#   %add_20 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_101, %mul_38), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_39, %unsqueeze_45), kwargs = {})
#   %add_21 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_107, %mul_40), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_19, %add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34], 1), kwargs = {})
#   %mul_72 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_71, %unsqueeze_80), kwargs = {})
#   %add_38 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_194, %mul_72), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_82), kwargs = {})
#   %add_39 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_200, %mul_74), kwargs = {})
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_37, %add_38, %add_39, %add_40, %add_41, %add_42, %add_43, %add_44, %add_45, %add_46, %add_47, %add_48, %add_49, %add_50, %add_51, %add_52], 1), kwargs = {})
triton_poi_fused_add_mul_stack_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_28', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x4 = (xindex // 21)
    x0 = xindex % 21
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (2688 + x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2667 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (63 + (64*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1323 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (63 + (1024*x2)), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (1344 + x3 + (21504*x2)), None)
    tmp24 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (2688 + x3 + (21504*x2)), None)
    tmp28 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr4 + (2667 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (63 + (64*x2)), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr4 + (1323 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (1344 + x3 + (21504*x2)), None)
    tmp42 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.full([1], -2, tl.int64)
    tmp7 = tmp5 == tmp6
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tmp5 >= tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 0.9945994019508362
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
    tmp32 = 0.999323308467865
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
    tl.store(out_ptr1 + (x3 + (21504*x2)), tmp22, None)
    tl.store(out_ptr2 + (x3 + (21504*x2)), tmp26, None)
    tl.store(out_ptr3 + (x5), tmp40, None)
    tl.store(out_ptr4 + (x3 + (21504*x2)), tmp40, None)
    tl.store(out_ptr5 + (x3 + (21504*x2)), tmp44, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/e5/ce5w7ellc6totnnymt6zw4g2ijkyyjwyii3sh5phzi75lw2oeydo.py
# Topologically Sorted Source Nodes: [mul_42, S_20, mul_44, S_21, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_20 => add_22
#   S_21 => add_23
#   mul_42 => mul_42
#   mul_44 => mul_44
#   stack_1 => cat_1
# Graph fragment:
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_41, %unsqueeze_47), kwargs = {})
#   %add_22 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_113, %mul_42), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_49), kwargs = {})
#   %add_23 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_119, %mul_44), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_19, %add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34], 1), kwargs = {})
triton_poi_fused_add_mul_stack_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_29', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x4 = (xindex // 21)
    x0 = xindex % 21
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (5376 + x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (5355 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (63 + (64*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (1323 + x0 + (1344*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (4032 + x3 + (21504*x2)), None)
    tmp10 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tmp1 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tl.store(out_ptr0 + (x5), tmp8, None)
    tl.store(out_ptr1 + (x3 + (21504*x2)), tmp8, None)
    tl.store(out_ptr2 + (x3 + (21504*x2)), tmp12, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/l3/cl3udnd4v7trsebeestrdswi6xe5e3cztw7uzqeqwpanavay2554.py
# Topologically Sorted Source Nodes: [mul_46, S_22, mul_48, S_23, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_22 => add_24
#   S_23 => add_25
#   mul_46 => mul_46
#   mul_48 => mul_48
#   stack_1 => cat_1
# Graph fragment:
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_45, %unsqueeze_51), kwargs = {})
#   %add_24 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_125, %mul_46), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_47, %unsqueeze_53), kwargs = {})
#   %add_25 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_131, %mul_48), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_19, %add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34], 1), kwargs = {})
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_30', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x4 = (xindex // 21)
    x0 = xindex % 21
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (8064 + x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (8043 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (63 + (64*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (1323 + x0 + (1344*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (6720 + x3 + (21504*x2)), None)
    tmp10 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tmp1 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tl.store(out_ptr0 + (x5), tmp8, None)
    tl.store(out_ptr1 + (x3 + (21504*x2)), tmp8, None)
    tl.store(out_ptr2 + (x3 + (21504*x2)), tmp12, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/4y/c4yaomv2njlukabnq3rkjymmncuksnzvfy6czyvx3ytfercmp2jx.py
# Topologically Sorted Source Nodes: [mul_50, S_24, mul_52, S_25, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_24 => add_26
#   S_25 => add_27
#   mul_50 => mul_50
#   mul_52 => mul_52
#   stack_1 => cat_1
# Graph fragment:
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_55), kwargs = {})
#   %add_26 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_137, %mul_50), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_51, %unsqueeze_57), kwargs = {})
#   %add_27 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_143, %mul_52), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_19, %add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34], 1), kwargs = {})
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
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x4 = (xindex // 21)
    x0 = xindex % 21
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (10752 + x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (10731 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (63 + (64*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (1323 + x0 + (1344*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (9408 + x3 + (21504*x2)), None)
    tmp10 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tmp1 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tl.store(out_ptr0 + (x5), tmp8, None)
    tl.store(out_ptr1 + (x3 + (21504*x2)), tmp8, None)
    tl.store(out_ptr2 + (x3 + (21504*x2)), tmp12, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/hk/chk52ocumtjpwh4o5vwzmlrftkvskwnff2preclolhanctdmb2ex.py
# Topologically Sorted Source Nodes: [mul_54, S_26, mul_56, S_27, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_26 => add_28
#   S_27 => add_29
#   mul_54 => mul_54
#   mul_56 => mul_56
#   stack_1 => cat_1
# Graph fragment:
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_53, %unsqueeze_59), kwargs = {})
#   %add_28 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_149, %mul_54), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %unsqueeze_61), kwargs = {})
#   %add_29 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_155, %mul_56), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_19, %add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34], 1), kwargs = {})
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
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x4 = (xindex // 21)
    x0 = xindex % 21
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (13440 + x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (13419 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (63 + (64*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (1323 + x0 + (1344*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (12096 + x3 + (21504*x2)), None)
    tmp10 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tmp1 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tl.store(out_ptr0 + (x5), tmp8, None)
    tl.store(out_ptr1 + (x3 + (21504*x2)), tmp8, None)
    tl.store(out_ptr2 + (x3 + (21504*x2)), tmp12, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/6k/c6k6spuq3g4qfeoyb2auiu2grc44vuvkludey7j7w2xybmfyvf65.py
# Topologically Sorted Source Nodes: [mul_58, S_28, mul_60, S_29, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_28 => add_30
#   S_29 => add_31
#   mul_58 => mul_58
#   mul_60 => mul_60
#   stack_1 => cat_1
# Graph fragment:
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_57, %unsqueeze_63), kwargs = {})
#   %add_30 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_161, %mul_58), kwargs = {})
#   %mul_60 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_59, %unsqueeze_65), kwargs = {})
#   %add_31 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_167, %mul_60), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_19, %add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34], 1), kwargs = {})
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
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x4 = (xindex // 21)
    x0 = xindex % 21
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (16128 + x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (16107 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (63 + (64*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (1323 + x0 + (1344*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (14784 + x3 + (21504*x2)), None)
    tmp10 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tmp1 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tl.store(out_ptr0 + (x5), tmp8, None)
    tl.store(out_ptr1 + (x3 + (21504*x2)), tmp8, None)
    tl.store(out_ptr2 + (x3 + (21504*x2)), tmp12, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/gl/cglnjgozhyooq46nytnnwwnp6syaityjjanoxzp73hyuda2l3qma.py
# Topologically Sorted Source Nodes: [mul_62, S_30, mul_64, S_31, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
# Source node to ATen node mapping:
#   S_30 => add_32
#   S_31 => add_33
#   mul_62 => mul_62
#   mul_64 => mul_64
#   stack_1 => cat_1
# Graph fragment:
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_67), kwargs = {})
#   %add_32 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_173, %mul_62), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_63, %unsqueeze_69), kwargs = {})
#   %add_33 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_179, %mul_64), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_19, %add_20, %add_21, %add_22, %add_23, %add_24, %add_25, %add_26, %add_27, %add_28, %add_29, %add_30, %add_31, %add_32, %add_33, %add_34], 1), kwargs = {})
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
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x4 = (xindex // 21)
    x0 = xindex % 21
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (18816 + x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (18795 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (63 + (64*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (1323 + x0 + (1344*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (17472 + x3 + (21504*x2)), None)
    tmp10 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tmp1 * tmp6
    tmp8 = tmp0 + tmp7
    tmp11 = tmp10 * tmp4
    tmp12 = tmp9 + tmp11
    tl.store(out_ptr0 + (x5), tmp8, None)
    tl.store(out_ptr1 + (x3 + (21504*x2)), tmp8, None)
    tl.store(out_ptr2 + (x3 + (21504*x2)), tmp12, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/hd/chdeukclstccv5frr25lb7wyikzvmdsidvozf77eteyv7ck6mjli.py
# Topologically Sorted Source Nodes: [mul_65, mul_99], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_65 => mul_65
#   mul_99 => mul_99
# Graph fragment:
#   %mul_65 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_33, %view_50), kwargs = {})
#   %mul_99 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_33, %view_91), kwargs = {})
triton_poi_fused_mul_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_35', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp7 = 0.9945994019508362
    tmp8 = 1 + x0
    tmp9 = tmp8.to(tl.float32)
    tmp10 = libdevice.pow(tmp7, tmp9)
    tmp11 = tmp6 * tmp10
    tmp12 = 0.999323308467865
    tmp13 = libdevice.pow(tmp12, tmp9)
    tmp14 = tmp6 * tmp13
    tl.store(out_ptr0 + (x2), tmp11, None)
    tl.store(out_ptr1 + (x2), tmp14, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/76/c76qlbwj2whxyi42i6oh7qyjtlhhh6jjx5vf4nythr4ax5a777wy.py
# Topologically Sorted Source Nodes: [unsqueeze_4, mul_35, mul_36, S_17, mul_69, mul_70, S_34], Original ATen: [aten.unsqueeze, aten.mul, aten.add]
# Source node to ATen node mapping:
#   S_17 => add_19
#   S_34 => add_37
#   mul_35 => mul_35
#   mul_36 => mul_36
#   mul_69 => mul_69
#   mul_70 => mul_70
#   unsqueeze_4 => full_default_4
# Graph fragment:
#   %full_default_4 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([32, 1, 21], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_3, %view_50), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_35, %full_default_4), kwargs = {})
#   %add_19 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_95, %mul_36), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_3, %view_91), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_69, %full_default_4), kwargs = {})
#   %add_37 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_188, %mul_70), kwargs = {})
triton_poi_fused_add_mul_unsqueeze_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_unsqueeze_36', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x1 = (xindex // 21) % 64
    tmp0 = tl.load(in_ptr0 + (x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (x3 + (21504*x2)), None)
    tmp2 = tl.full([1], -2, tl.int64)
    tmp3 = tmp1 == tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp1 >= tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 0.9945994019508362
    tmp9 = 1 + x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.pow(tmp8, tmp10)
    tmp12 = tmp7 * tmp11
    tmp13 = 0.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp0 + tmp14
    tmp17 = 0.999323308467865
    tmp18 = libdevice.pow(tmp17, tmp10)
    tmp19 = tmp7 * tmp18
    tmp20 = tmp19 * tmp13
    tmp21 = tmp16 + tmp20
    tl.store(out_ptr0 + (x3 + (21504*x2)), tmp15, None)
    tl.store(out_ptr1 + (x3 + (21504*x2)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/mh/cmhsmgsdusu2boap76oij2jd26q7skz5vfyaf6b2td5yc3gwk3x6.py
# Topologically Sorted Source Nodes: [mul_66, S_32], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   S_32 => add_34
#   mul_66 => mul_66
# Graph fragment:
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_65, %unsqueeze_71), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_185, %mul_66), kwargs = {})
triton_poi_fused_add_mul_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_37', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x4 = (xindex // 21)
    x0 = xindex % 21
    tmp0 = tl.load(in_ptr0 + (20160 + x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (1323 + x0 + (1344*x2)), None, eviction_policy='evict_last')
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x3 + (21504*x2)), tmp4, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/kl/cklzyezdjisp4ucnokvnhy6uy6zgackxxerxrm3qlck656diaxjb.py
# Topologically Sorted Source Nodes: [to_20, add_17, tensor_3, pow_6, sub_3, Z_1, tensor_6, pow_9, sub_5, Z_2], Original ATen: [aten._to_copy, aten.add, aten.lift_fresh, aten.pow, aten.rsub, aten.div]
# Source node to ATen node mapping:
#   Z_1 => div_2
#   Z_2 => div_4
#   add_17 => add_17
#   pow_6 => pow_6
#   pow_9 => pow_9
#   sub_3 => sub_3
#   sub_5 => sub_5
#   tensor_3 => full_default_6
#   tensor_6 => full_default_12
#   to_20 => convert_element_type_19
# Graph fragment:
#   %convert_element_type_19 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_4, torch.float32), kwargs = {})
#   %add_17 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_19, 1.0), kwargs = {})
#   %full_default_6 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.9945994019508362), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%full_default_6, %add_17), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %pow_6), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_3, 0.0054005765163668285), kwargs = {})
#   %full_default_12 : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.999323308467865), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Tensor](args = (%full_default_12, %add_17), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %pow_9), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_5, 0.0006766724973492622), kwargs = {})
triton_poi_fused__to_copy_add_div_lift_fresh_pow_rsub_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_lift_fresh_pow_rsub_38', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp4 = 0.9945994019508362
    tmp5 = libdevice.pow(tmp4, tmp3)
    tmp6 = tmp2 - tmp5
    tmp7 = 185.16541650126229
    tmp8 = tmp6 * tmp7
    tmp9 = 0.999323308467865
    tmp10 = libdevice.pow(tmp9, tmp3)
    tmp11 = tmp2 - tmp10
    tmp12 = 1477.8197782787283
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/n5/cn5x2cjs7oerncfk4d7gk5haxa43smo56wrgkbaezbutogix3cul.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat_3
# Graph fragment:
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_33, %mul_67, %mul_101], -1), kwargs = {})
triton_poi_fused_cat_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_39', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2064384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 63
    x1 = (xindex // 63)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((21*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 / tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp4, tmp10, tmp11)
    tmp13 = tmp0 >= tmp3
    tmp14 = tl.full([1], 42, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tmp13 & tmp15
    tmp17 = tl.load(in_ptr3 + ((21*x1) + ((-21) + x0)), tmp16, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr4 + (x1), tmp16, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 / tmp18
    tmp20 = tl.load(in_ptr2 + (x1), tmp16, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 * tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp16, tmp22, tmp23)
    tmp25 = tmp0 >= tmp14
    tmp26 = tl.full([1], 63, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr5 + ((21*x1) + ((-42) + x0)), tmp25, eviction_policy='evict_last', other=0.0)
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (32, 1024), (1024, 1))
    assert_size_stride(primals_2, (63, 384), (384, 1))
    assert_size_stride(primals_3, (32768, 384), (384, 1))
    assert_size_stride(primals_4, (32, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768, 63), (63, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(primals_3, reinterpret_tensor(primals_2, (384, 63), (1, 384), 0), out=buf0)
        buf1 = empty_strided_cuda((32, 16, 64, 64), (65536, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sub, lag, ge, tensor, clamp, pow_1, zeros, base, eq, same, to_2, mul], Original ATen: [aten.sub, aten._to_copy, aten.ge, aten.lift_fresh, aten.clamp, aten.pow, aten.zeros, aten.where, aten.eq, aten.bitwise_and, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_lift_fresh_mul_pow_sub_where_zeros_0.run(primals_1, buf1, 2097152, grid=grid(2097152), stream=stream0)
        buf2 = empty_strided_cuda((512, 64, 21), (1344, 21, 1), torch.float32)
        # Topologically Sorted Source Nodes: [S_in], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1, (512, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf0, (512, 64, 21), (4032, 63, 1), 0), out=buf2)
        buf3 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [i, tensor, add, to_3, dec], Original ATen: [aten.arange, aten.lift_fresh, aten.add, aten._to_copy, aten.pow]
        triton_poi_fused__to_copy_add_arange_lift_fresh_pow_1.run(buf3, 64, grid=grid(64), stream=stream0)
        buf4 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf47 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf89 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_2, ge_3, and__2, mul_37, mul_71], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_2.run(primals_1, buf4, buf47, buf89, 2048, grid=grid(2048), stream=stream0)
        buf5 = empty_strided_cuda((32, 64, 21), (1344, 21, 1), torch.float32)
        buf42 = empty_strided_cuda((32, 1024, 21), (21504, 21, 1), torch.float32)
        buf27 = reinterpret_tensor(buf42, (32, 64, 21), (21504, 21, 1), 1344)  # alias
        # Topologically Sorted Source Nodes: [mul_3, mul_4, S_1, stack], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_3.run(buf2, buf4, buf3, primals_1, buf5, buf27, 43008, grid=grid(43008), stream=stream0)
        buf6 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf48 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf90 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_3, ge_4, and__3, mul_39, mul_73], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_4.run(primals_1, buf6, buf48, buf90, 2048, grid=grid(2048), stream=stream0)
        buf7 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf50 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf92 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_4, ge_5, and__4, mul_41, mul_75], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_5.run(primals_1, buf7, buf50, buf92, 2048, grid=grid(2048), stream=stream0)
        buf26 = reinterpret_tensor(buf42, (32, 64, 21), (21504, 21, 1), 0)  # alias
        buf8 = empty_strided_cuda((32, 64, 21), (1344, 21, 1), torch.float32)
        buf29 = reinterpret_tensor(buf42, (32, 64, 21), (21504, 21, 1), 4032)  # alias
        buf28 = reinterpret_tensor(buf42, (32, 64, 21), (21504, 21, 1), 2688)  # alias
        # Topologically Sorted Source Nodes: [mul_1, unsqueeze_4, mul_2, S, mul_5, mul_6, S_2, mul_7, mul_8, S_3, stack], Original ATen: [aten.mul, aten.unsqueeze, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_unsqueeze_6.run(buf2, primals_1, buf3, buf7, buf6, buf5, buf26, buf8, buf29, buf28, 43008, grid=grid(43008), stream=stream0)
        buf9 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf51 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf93 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_5, ge_6, and__5, mul_43, mul_77], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_7.run(primals_1, buf9, buf51, buf93, 2048, grid=grid(2048), stream=stream0)
        buf10 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf53 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf95 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_6, ge_7, and__6, mul_45, mul_79], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_8.run(primals_1, buf10, buf53, buf95, 2048, grid=grid(2048), stream=stream0)
        buf11 = buf5; del buf5  # reuse
        buf31 = reinterpret_tensor(buf42, (32, 64, 21), (21504, 21, 1), 6720)  # alias
        buf30 = reinterpret_tensor(buf42, (32, 64, 21), (21504, 21, 1), 5376)  # alias
        # Topologically Sorted Source Nodes: [mul_9, mul_10, S_4, mul_11, mul_12, S_5, stack], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_9.run(buf2, buf10, buf3, buf9, buf8, buf11, buf31, buf30, 43008, grid=grid(43008), stream=stream0)
        buf12 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf54 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf96 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_7, ge_8, and__7, mul_47, mul_81], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_10.run(primals_1, buf12, buf54, buf96, 2048, grid=grid(2048), stream=stream0)
        buf13 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf56 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf98 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_8, ge_9, and__8, mul_49, mul_83], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_11.run(primals_1, buf13, buf56, buf98, 2048, grid=grid(2048), stream=stream0)
        buf14 = buf8; del buf8  # reuse
        buf33 = reinterpret_tensor(buf42, (32, 64, 21), (21504, 21, 1), 9408)  # alias
        buf32 = reinterpret_tensor(buf42, (32, 64, 21), (21504, 21, 1), 8064)  # alias
        # Topologically Sorted Source Nodes: [mul_13, mul_14, S_6, mul_15, mul_16, S_7, stack], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_12.run(buf2, buf13, buf3, buf12, buf11, buf14, buf33, buf32, 43008, grid=grid(43008), stream=stream0)
        buf15 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf57 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf99 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_9, ge_10, and__9, mul_51, mul_85], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_13.run(primals_1, buf15, buf57, buf99, 2048, grid=grid(2048), stream=stream0)
        buf16 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf59 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf101 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_10, ge_11, and__10, mul_53, mul_87], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_14.run(primals_1, buf16, buf59, buf101, 2048, grid=grid(2048), stream=stream0)
        buf17 = buf11; del buf11  # reuse
        buf35 = reinterpret_tensor(buf42, (32, 64, 21), (21504, 21, 1), 12096)  # alias
        buf34 = reinterpret_tensor(buf42, (32, 64, 21), (21504, 21, 1), 10752)  # alias
        # Topologically Sorted Source Nodes: [mul_17, mul_18, S_8, mul_19, mul_20, S_9, stack], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_15.run(buf2, buf16, buf3, buf15, buf14, buf17, buf35, buf34, 43008, grid=grid(43008), stream=stream0)
        buf18 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf60 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf102 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_11, ge_12, and__11, mul_55, mul_89], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_16.run(primals_1, buf18, buf60, buf102, 2048, grid=grid(2048), stream=stream0)
        buf19 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf62 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf104 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_12, ge_13, and__12, mul_57, mul_91], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_17.run(primals_1, buf19, buf62, buf104, 2048, grid=grid(2048), stream=stream0)
        buf20 = buf14; del buf14  # reuse
        buf37 = reinterpret_tensor(buf42, (32, 64, 21), (21504, 21, 1), 14784)  # alias
        buf36 = reinterpret_tensor(buf42, (32, 64, 21), (21504, 21, 1), 13440)  # alias
        # Topologically Sorted Source Nodes: [mul_21, mul_22, S_10, mul_23, mul_24, S_11, stack], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_18.run(buf2, buf19, buf3, buf18, buf17, buf20, buf37, buf36, 43008, grid=grid(43008), stream=stream0)
        buf21 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf63 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf105 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_13, ge_14, and__13, mul_59, mul_93], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_19.run(primals_1, buf21, buf63, buf105, 2048, grid=grid(2048), stream=stream0)
        buf22 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf65 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf107 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_14, ge_15, and__14, mul_61, mul_95], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_20.run(primals_1, buf22, buf65, buf107, 2048, grid=grid(2048), stream=stream0)
        buf23 = buf17; del buf17  # reuse
        buf39 = reinterpret_tensor(buf42, (32, 64, 21), (21504, 21, 1), 17472)  # alias
        buf38 = reinterpret_tensor(buf42, (32, 64, 21), (21504, 21, 1), 16128)  # alias
        # Topologically Sorted Source Nodes: [mul_25, mul_26, S_12, mul_27, mul_28, S_13, stack], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_21.run(buf2, buf22, buf3, buf21, buf20, buf23, buf39, buf38, 43008, grid=grid(43008), stream=stream0)
        buf24 = empty_strided_cuda((32, 64), (64, 1), torch.bool)
        buf66 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf108 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [eq_15, ge_16, and__15, mul_63, mul_97], Original ATen: [aten.eq, aten.ge, aten.bitwise_and, aten.mul]
        triton_poi_fused_bitwise_and_eq_ge_mul_22.run(primals_1, buf24, buf66, buf108, 2048, grid=grid(2048), stream=stream0)
        buf25 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_31], Original ATen: [aten.mul]
        triton_poi_fused_mul_23.run(primals_1, buf3, buf25, 2048, grid=grid(2048), stream=stream0)
        buf40 = reinterpret_tensor(buf42, (32, 64, 21), (21504, 21, 1), 18816)  # alias
        buf41 = reinterpret_tensor(buf42, (32, 64, 21), (21504, 21, 1), 20160)  # alias
        # Topologically Sorted Source Nodes: [mul_29, mul_30, S_14, mul_32, S_15], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_24.run(buf2, buf24, buf3, buf23, buf25, buf40, buf41, 43008, grid=grid(43008), stream=stream0)
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
        buf44 = empty_strided_cuda((32, 1024), (1024, 1), torch.bool)
        # Topologically Sorted Source Nodes: [ge_18], Original ATen: [aten.ge]
        triton_poi_fused_ge_26.run(primals_1, buf44, 32768, grid=grid(32768), stream=stream0)
        buf45 = empty_strided_cuda((32, 16, 64, 64), (65536, 4096, 64, 1), torch.float32)
        buf87 = empty_strided_cuda((32, 16, 64, 64), (65536, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sub, lag, ge, clamp, zeros, tensor_3, pow_4, base_1, eq_17, same_1, to_23, mul_34, tensor_6, pow_7, base_2, eq_34, same_2, to_44, mul_68], Original ATen: [aten.sub, aten._to_copy, aten.ge, aten.clamp, aten.zeros, aten.lift_fresh, aten.pow, aten.where, aten.eq, aten.bitwise_and, aten.mul]
        triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_lift_fresh_mul_pow_sub_where_zeros_27.run(primals_1, buf45, buf87, 2097152, grid=grid(2097152), stream=stream0)
        buf46 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [S_in_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf45, (512, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf0, (512, 64, 21), (4032, 63, 1), 21), out=buf46)
        buf88 = empty_strided_cuda((512, 64, 21), (1344, 21, 1), torch.float32)
        # Topologically Sorted Source Nodes: [S_in_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf87, (512, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf0, (512, 64, 21), (4032, 63, 1), 42), out=buf88)
        buf49 = buf23; del buf23  # reuse
        buf85 = empty_strided_cuda((32, 1024, 21), (21504, 21, 1), torch.float32)
        buf71 = reinterpret_tensor(buf85, (32, 64, 21), (21504, 21, 1), 2688)  # alias
        buf70 = reinterpret_tensor(buf85, (32, 64, 21), (21504, 21, 1), 1344)  # alias
        buf91 = buf20; del buf20  # reuse
        buf127 = empty_strided_cuda((32, 1024, 21), (21504, 21, 1), torch.float32)
        buf113 = reinterpret_tensor(buf127, (32, 64, 21), (21504, 21, 1), 2688)  # alias
        buf112 = reinterpret_tensor(buf127, (32, 64, 21), (21504, 21, 1), 1344)  # alias
        # Topologically Sorted Source Nodes: [mul_38, S_18, mul_40, S_19, stack_1, mul_72, S_35, mul_74, S_36, stack_2], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_28.run(buf46, buf48, buf47, primals_1, buf88, buf90, buf89, buf49, buf71, buf70, buf91, buf113, buf112, 43008, grid=grid(43008), stream=stream0)
        buf52 = empty_strided_cuda((32, 64, 21), (1344, 21, 1), torch.float32)
        buf73 = reinterpret_tensor(buf85, (32, 64, 21), (21504, 21, 1), 5376)  # alias
        buf72 = reinterpret_tensor(buf85, (32, 64, 21), (21504, 21, 1), 4032)  # alias
        # Topologically Sorted Source Nodes: [mul_42, S_20, mul_44, S_21, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_29.run(buf46, buf51, buf50, buf49, buf52, buf73, buf72, 43008, grid=grid(43008), stream=stream0)
        buf55 = buf49; del buf49  # reuse
        buf75 = reinterpret_tensor(buf85, (32, 64, 21), (21504, 21, 1), 8064)  # alias
        buf74 = reinterpret_tensor(buf85, (32, 64, 21), (21504, 21, 1), 6720)  # alias
        # Topologically Sorted Source Nodes: [mul_46, S_22, mul_48, S_23, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_30.run(buf46, buf54, buf53, buf52, buf55, buf75, buf74, 43008, grid=grid(43008), stream=stream0)
        buf58 = buf52; del buf52  # reuse
        buf77 = reinterpret_tensor(buf85, (32, 64, 21), (21504, 21, 1), 10752)  # alias
        buf76 = reinterpret_tensor(buf85, (32, 64, 21), (21504, 21, 1), 9408)  # alias
        # Topologically Sorted Source Nodes: [mul_50, S_24, mul_52, S_25, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_31.run(buf46, buf57, buf56, buf55, buf58, buf77, buf76, 43008, grid=grid(43008), stream=stream0)
        buf61 = buf55; del buf55  # reuse
        buf79 = reinterpret_tensor(buf85, (32, 64, 21), (21504, 21, 1), 13440)  # alias
        buf78 = reinterpret_tensor(buf85, (32, 64, 21), (21504, 21, 1), 12096)  # alias
        # Topologically Sorted Source Nodes: [mul_54, S_26, mul_56, S_27, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_32.run(buf46, buf60, buf59, buf58, buf61, buf79, buf78, 43008, grid=grid(43008), stream=stream0)
        buf64 = buf58; del buf58  # reuse
        buf81 = reinterpret_tensor(buf85, (32, 64, 21), (21504, 21, 1), 16128)  # alias
        buf80 = reinterpret_tensor(buf85, (32, 64, 21), (21504, 21, 1), 14784)  # alias
        # Topologically Sorted Source Nodes: [mul_58, S_28, mul_60, S_29, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_33.run(buf46, buf63, buf62, buf61, buf64, buf81, buf80, 43008, grid=grid(43008), stream=stream0)
        buf67 = buf61; del buf61  # reuse
        buf83 = reinterpret_tensor(buf85, (32, 64, 21), (21504, 21, 1), 18816)  # alias
        buf82 = reinterpret_tensor(buf85, (32, 64, 21), (21504, 21, 1), 17472)  # alias
        # Topologically Sorted Source Nodes: [mul_62, S_30, mul_64, S_31, stack_1], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_34.run(buf46, buf66, buf65, buf64, buf67, buf83, buf82, 43008, grid=grid(43008), stream=stream0)
        del buf64
        buf68 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        buf110 = empty_strided_cuda((32, 64, 1), (64, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_65, mul_99], Original ATen: [aten.mul]
        triton_poi_fused_mul_35.run(primals_1, buf68, buf110, 2048, grid=grid(2048), stream=stream0)
        buf69 = reinterpret_tensor(buf85, (32, 64, 21), (21504, 21, 1), 0)  # alias
        buf111 = reinterpret_tensor(buf127, (32, 64, 21), (21504, 21, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [unsqueeze_4, mul_35, mul_36, S_17, mul_69, mul_70, S_34], Original ATen: [aten.unsqueeze, aten.mul, aten.add]
        triton_poi_fused_add_mul_unsqueeze_36.run(buf46, primals_1, buf88, buf69, buf111, 43008, grid=grid(43008), stream=stream0)
        del primals_1
        buf84 = reinterpret_tensor(buf85, (32, 64, 21), (21504, 21, 1), 20160)  # alias
        # Topologically Sorted Source Nodes: [mul_66, S_32], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_37.run(buf46, buf68, buf67, buf84, 43008, grid=grid(43008), stream=stream0)
        del buf46
        buf86 = empty_strided_cuda((32, 1024), (1024, 1), torch.float32)
        buf128 = empty_strided_cuda((32, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [to_20, add_17, tensor_3, pow_6, sub_3, Z_1, tensor_6, pow_9, sub_5, Z_2], Original ATen: [aten._to_copy, aten.add, aten.lift_fresh, aten.pow, aten.rsub, aten.div]
        triton_poi_fused__to_copy_add_div_lift_fresh_pow_rsub_38.run(primals_4, buf86, buf128, 32768, grid=grid(32768), stream=stream0)
        del buf69
        del buf70
        del buf71
        del buf72
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
        del primals_4
        buf94 = buf67; del buf67  # reuse
        buf115 = reinterpret_tensor(buf127, (32, 64, 21), (21504, 21, 1), 5376)  # alias
        buf114 = reinterpret_tensor(buf127, (32, 64, 21), (21504, 21, 1), 4032)  # alias
        # Topologically Sorted Source Nodes: [mul_76, S_37, mul_78, S_38, stack_2], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_29.run(buf88, buf93, buf92, buf91, buf94, buf115, buf114, 43008, grid=grid(43008), stream=stream0)
        buf97 = buf91; del buf91  # reuse
        buf117 = reinterpret_tensor(buf127, (32, 64, 21), (21504, 21, 1), 8064)  # alias
        buf116 = reinterpret_tensor(buf127, (32, 64, 21), (21504, 21, 1), 6720)  # alias
        # Topologically Sorted Source Nodes: [mul_80, S_39, mul_82, S_40, stack_2], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_30.run(buf88, buf96, buf95, buf94, buf97, buf117, buf116, 43008, grid=grid(43008), stream=stream0)
        buf100 = buf94; del buf94  # reuse
        buf119 = reinterpret_tensor(buf127, (32, 64, 21), (21504, 21, 1), 10752)  # alias
        buf118 = reinterpret_tensor(buf127, (32, 64, 21), (21504, 21, 1), 9408)  # alias
        # Topologically Sorted Source Nodes: [mul_84, S_41, mul_86, S_42, stack_2], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_31.run(buf88, buf99, buf98, buf97, buf100, buf119, buf118, 43008, grid=grid(43008), stream=stream0)
        buf103 = buf97; del buf97  # reuse
        buf121 = reinterpret_tensor(buf127, (32, 64, 21), (21504, 21, 1), 13440)  # alias
        buf120 = reinterpret_tensor(buf127, (32, 64, 21), (21504, 21, 1), 12096)  # alias
        # Topologically Sorted Source Nodes: [mul_88, S_43, mul_90, S_44, stack_2], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_32.run(buf88, buf102, buf101, buf100, buf103, buf121, buf120, 43008, grid=grid(43008), stream=stream0)
        buf106 = buf100; del buf100  # reuse
        buf123 = reinterpret_tensor(buf127, (32, 64, 21), (21504, 21, 1), 16128)  # alias
        buf122 = reinterpret_tensor(buf127, (32, 64, 21), (21504, 21, 1), 14784)  # alias
        # Topologically Sorted Source Nodes: [mul_92, S_45, mul_94, S_46, stack_2], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_33.run(buf88, buf105, buf104, buf103, buf106, buf123, buf122, 43008, grid=grid(43008), stream=stream0)
        buf109 = buf103; del buf103  # reuse
        buf125 = reinterpret_tensor(buf127, (32, 64, 21), (21504, 21, 1), 18816)  # alias
        buf124 = reinterpret_tensor(buf127, (32, 64, 21), (21504, 21, 1), 17472)  # alias
        # Topologically Sorted Source Nodes: [mul_96, S_47, mul_98, S_48, stack_2], Original ATen: [aten.mul, aten.add, aten.stack]
        triton_poi_fused_add_mul_stack_34.run(buf88, buf108, buf107, buf106, buf109, buf125, buf124, 43008, grid=grid(43008), stream=stream0)
        del buf106
        buf126 = reinterpret_tensor(buf127, (32, 64, 21), (21504, 21, 1), 20160)  # alias
        # Topologically Sorted Source Nodes: [mul_100, S_49], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_37.run(buf88, buf110, buf109, buf126, 43008, grid=grid(43008), stream=stream0)
        del buf109
        del buf88
        buf129 = reinterpret_tensor(buf0, (32, 1024, 63), (64512, 63, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        triton_poi_fused_cat_39.run(buf42, buf43, buf44, buf85, buf86, buf127, buf128, buf129, 2064384, grid=grid(2064384), stream=stream0)
        del buf111
        del buf112
        del buf113
        del buf114
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
        del buf42
        del buf85
    return (reinterpret_tensor(buf129, (32768, 63), (63, 1), 0), primals_3, reinterpret_tensor(buf3, (1, 64, 1), (64, 1, 1), 0), buf4, buf6, buf7, buf9, buf10, buf12, buf13, buf15, buf16, buf18, buf19, buf21, buf22, buf24, buf25, reinterpret_tensor(buf43, (32, 1024, 1), (1024, 1, 1), 0), reinterpret_tensor(buf44, (32, 1024, 1), (1024, 1, 1), 0), buf47, buf48, buf50, buf51, buf53, buf54, buf56, buf57, buf59, buf60, buf62, buf63, buf65, buf66, buf68, reinterpret_tensor(buf86, (32, 1024, 1), (1024, 1, 1), 0), buf89, buf90, buf92, buf93, buf95, buf96, buf98, buf99, buf101, buf102, buf104, buf105, buf107, buf108, buf110, reinterpret_tensor(buf128, (32, 1024, 1), (1024, 1, 1), 0), reinterpret_tensor(buf87, (512, 64, 64), (4096, 1, 64), 0), reinterpret_tensor(buf45, (512, 64, 64), (4096, 1, 64), 0), reinterpret_tensor(buf1, (512, 64, 64), (4096, 1, 64), 0), primals_2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    primals_2 = rand_strided((63, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
