# AOT ID: ['5_inference']
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/eq/ceqycbs3hdccutrlrnsihb6r2bbnck2hdr5c6l27bbj3f7yjrkuo.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/tw/ctwhgglhzxk25ypkfazaidvgbfsnztqiuwrn3iuil4xkdi4zhvjh.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg1_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 442368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/7o/c7onp5xxupo3cn25pb6lsso2wnheeyay56toyqmixhxh2omkzmss.py
# Topologically Sorted Source Nodes: [mul, rot, mul_1, q_1, y, mul_2, rot_1, mul_3, k_1], Original ATen: [aten.mul, aten.cat, aten.add, aten._to_copy, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   k_1 => add_1
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   q_1 => add
#   rot => cat
#   rot_1 => cat_1
#   y => _scaled_dot_product_efficient_attention, convert_element_type_4, convert_element_type_5
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %arg2_1), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_2], -1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %arg3_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.bfloat16), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %arg2_1), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_1, %slice_4], -1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_1, %arg3_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %mul_3), kwargs = {})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%convert_element_type_5, %convert_element_type_4, %select_2, %expand, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_cat_mul_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: '*bf16', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_cat_mul_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 384)
    x3 = xindex % 384
    x0 = xindex % 64
    x1 = (xindex // 64) % 6
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (1152*x2)), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (384 + x3 + (1152*x2)), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = x0
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tl.full([1], 32, tl.int64)
    tmp8 = tmp4 < tmp7
    tmp9 = tl.load(in_ptr0 + (32 + (64*x1) + (1152*x2) + x0), tmp8, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = -tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp8, tmp10, tmp11)
    tmp13 = tmp4 >= tmp7
    tmp14 = tl.full([1], 64, tl.int64)
    tmp15 = tmp4 < tmp14
    tmp16 = tl.load(in_ptr0 + ((64*x1) + (1152*x2) + ((-32) + x0)), tmp13, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.where(tmp8, tmp12, tmp16)
    tmp18 = tmp17.to(tl.float32)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp3 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp24 * tmp2
    tmp26 = tl.load(in_ptr0 + (416 + (64*x1) + (1152*x2) + x0), tmp8, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp27 = -tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp8, tmp27, tmp28)
    tmp30 = tl.load(in_ptr0 + (384 + (64*x1) + (1152*x2) + ((-32) + x0)), tmp13, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tl.where(tmp8, tmp29, tmp30)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp32 * tmp19
    tmp34 = tmp25 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp22, None)
    tl.store(out_ptr1 + (x4), tmp35, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/we/cwegwepiunptqvc6mhrm5jivrrzgpgs737a7mvl2teckn5anziq2.py
# Topologically Sorted Source Nodes: [mul, rot, mul_1, q_1, y, mul_2, rot_1, mul_3, k_1], Original ATen: [aten.mul, aten.cat, aten.add, aten._to_copy, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   k_1 => add_1
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   q_1 => add
#   rot => cat
#   rot_1 => cat_1
#   y => _scaled_dot_product_efficient_attention, convert_element_type_4, convert_element_type_5
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %arg2_1), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_2], -1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %arg3_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.bfloat16), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %arg2_1), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_1, %slice_4], -1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_1, %arg3_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %mul_3), kwargs = {})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%convert_element_type_5, %convert_element_type_4, %select_2, %expand, False), kwargs = {})
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_cat_mul_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*i1', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_cat_mul_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 1048576
    x2 = (xindex // 6291456)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1048576*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tmp0 == 0
    tmp2 = float("-inf")
    tmp3 = 0.0
    tmp4 = tl.where(tmp1, tmp2, tmp3)
    tl.store(out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/4p/c4pb4sxq5y6mabilobycc4tom3sxe63mits7sxkxymbfb6y7u7ii.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_1 => convert_element_type_6
# Graph fragment:
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg5_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 1024, 384), (393216, 384, 1))
    assert_size_stride(arg1_1, (1152, 384), (384, 1))
    assert_size_stride(arg2_1, (4, 1, 1024, 64), (65536, 65536, 64, 1))
    assert_size_stride(arg3_1, (4, 1, 1024, 64), (65536, 65536, 64, 1))
    assert_size_stride(arg4_1, (4, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg5_1, (384, 384), (384, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 1024, 384), (393216, 384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(arg0_1, buf0, 1572864, grid=grid(1572864), stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((1152, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_1.run(arg1_1, buf1, 442368, grid=grid(442368), stream=stream0)
        del arg1_1
        buf2 = empty_strided_cuda((4096, 1152), (1152, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (4096, 384), (384, 1), 0), reinterpret_tensor(buf1, (384, 1152), (1, 384), 0), out=buf2)
        del buf1
        buf3 = reinterpret_tensor(buf0, (4, 6, 1024, 64), (393216, 64, 384, 1), 0); del buf0  # reuse
        buf4 = empty_strided_cuda((4, 6, 1024, 64), (393216, 64, 384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [mul, rot, mul_1, q_1, y, mul_2, rot_1, mul_3, k_1], Original ATen: [aten.mul, aten.cat, aten.add, aten._to_copy, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_cat_mul_2.run(buf2, arg2_1, arg3_1, buf3, buf4, 1572864, grid=grid(1572864), stream=stream0)
        del arg2_1
        del arg3_1
        buf5 = empty_strided_cuda((4, 6, 1024, 1024), (6291456, 1048576, 1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [mul, rot, mul_1, q_1, y, mul_2, rot_1, mul_3, k_1], Original ATen: [aten.mul, aten.cat, aten.add, aten._to_copy, aten._scaled_dot_product_efficient_attention]
        triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_add_cat_mul_3.run(arg4_1, buf5, 25165824, grid=grid(25165824), stream=stream0)
        del arg4_1
        # Topologically Sorted Source Nodes: [mul, rot, mul_1, q_1, y, mul_2, rot_1, mul_3, k_1], Original ATen: [aten.mul, aten.cat, aten.add, aten._to_copy, aten._scaled_dot_product_efficient_attention]
        buf6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf3, buf4, reinterpret_tensor(buf2, (4, 6, 1024, 64), (1179648, 64, 1152, 1), 768), buf5, False)
        del buf2
        del buf3
        del buf5
        buf7 = buf6[0]
        del buf6
        buf11 = empty_strided_cuda((384, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(arg5_1, buf11, 147456, grid=grid(147456), stream=stream0)
        del arg5_1
        buf12 = reinterpret_tensor(buf4, (4096, 384), (384, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (4096, 384), (384, 1), 0), reinterpret_tensor(buf11, (384, 384), (1, 384), 0), out=buf12)
        del buf11
        del buf7
    return (reinterpret_tensor(buf12, (4, 1024, 384), (393216, 384, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 1024, 384), (393216, 384, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 1, 1024, 64), (65536, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((4, 1, 1024, 64), (65536, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((4, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg5_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
