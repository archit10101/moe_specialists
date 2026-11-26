# AOT ID: ['2_backward']
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/qz/cqzmkt5macsxvoatfgg2gn3qqya4m4xdqh4lxlzjhfegss7rsvlr.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_13 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_7, torch.float32), kwargs = {})
triton_poi_fused__to_copy_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/7e/c7e6mmag2nzgty5nx4r2nqkp37ijxtq6o4barq4k5ddvximwk2og.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.slice_backward, aten.add, aten.neg]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_14 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_9, torch.float32), kwargs = {})
#   %convert_element_type_15 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_10, torch.float32), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_15, %primals_3), kwargs = {})
#   %convert_element_type_17 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5, torch.bfloat16), kwargs = {})
#   %full_default : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([32, 6, 1024, 64], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %slice_6, 3, 0, 32), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_17, %slice_scatter_default), kwargs = {})
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_5,), kwargs = {})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %neg_2, 3, 32, 9223372036854775807), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %slice_scatter_default_1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_14, %primals_3), kwargs = {})
#   %convert_element_type_19 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7, torch.bfloat16), kwargs = {})
#   %slice_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %slice_8, 3, 0, 32), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_19, %slice_scatter_default_2), kwargs = {})
#   %neg_3 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_7,), kwargs = {})
#   %slice_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %neg_3, 3, 32, 9223372036854775807), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %slice_scatter_default_3), kwargs = {})
triton_poi_fused__to_copy_add_mul_neg_slice_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_neg_slice_backward_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 64
    x2 = (xindex // 384)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr3 + (x3), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = x0
    tmp6 = tl.full([1], 32, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = tl.load(in_ptr0 + (32 + x3), tmp7, other=0.0).to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (32 + x0 + (64*x2)), tmp7, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp7, tmp12, tmp13)
    tmp15 = 0.0
    tmp16 = tl.where(tmp7, tmp14, tmp15)
    tmp17 = tmp4 + tmp16
    tmp18 = tmp5 >= tmp6
    tmp19 = tl.load(in_ptr0 + ((-32) + x3), tmp18, other=0.0).to(tl.float32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.load(in_ptr2 + ((-32) + x0 + (64*x2)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = -tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp18, tmp24, tmp25)
    tmp27 = tl.where(tmp18, tmp26, tmp15)
    tmp28 = tmp17 + tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp2
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.load(in_ptr3 + (32 + x3), tmp7, other=0.0).to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp34 * tmp10
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp7, tmp36, tmp37)
    tmp39 = tl.where(tmp7, tmp38, tmp15)
    tmp40 = tmp32 + tmp39
    tmp41 = tl.load(in_ptr3 + ((-32) + x3), tmp18, other=0.0).to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp42 * tmp21
    tmp44 = tmp43.to(tl.float32)
    tmp45 = -tmp44
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp18, tmp45, tmp46)
    tmp48 = tl.where(tmp18, tmp47, tmp15)
    tmp49 = tmp40 + tmp48
    tl.store(out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr1 + (x3), tmp49, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/m4/cm4rhuoeztmv2vavdr4g6y7krnxylhschxqfjyb25tgh5eqvmkik.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37748736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 384) % 3
    x0 = xindex % 384
    x2 = (xindex // 1152)
    x3 = xindex
    tmp3 = tl.load(in_ptr0 + (x0 + (384*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (x0 + (384*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (x0 + (384*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tl.full([1], 1, tl.int32)
    tmp7 = tmp0 == tmp6
    tmp9 = tl.where(tmp7, tmp8, tmp4)
    tmp10 = tmp5 + tmp9
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp14 = tl.where(tmp12, tmp13, tmp4)
    tmp15 = tmp10 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/6s/c6s2cayzfzjhu2iyf276ur7iu2df6eqksumyxjcj56yi3vfh5fm4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_11, torch.float32), kwargs = {})
triton_poi_fused__to_copy_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/qe/cqe7q5dskegiyd6og44nctxhq5av32g6ydlccvnjneudi5vkx5on.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_25 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_13, torch.float32), kwargs = {})
triton_poi_fused__to_copy_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 442368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_3, primals_4, view, select_2, convert_element_type_4, convert_element_type_5, getitem, getitem_1, getitem_6, getitem_7, permute_6, permute_12, tangents_1 = args
    args.clear()
    assert_size_stride(primals_3, (32, 1, 1024, 64), (65536, 65536, 64, 1))
    assert_size_stride(primals_4, (32, 1, 1024, 64), (65536, 65536, 64, 1))
    assert_size_stride(view, (32768, 384), (384, 1))
    assert_size_stride(select_2, (32, 6, 1024, 64), (1179648, 64, 1152, 1))
    assert_size_stride(convert_element_type_4, (32, 6, 1024, 64), (393216, 64, 384, 1))
    assert_size_stride(convert_element_type_5, (32, 6, 1024, 64), (393216, 64, 384, 1))
    assert_size_stride(getitem, (32, 6, 1024, 64), (393216, 64, 384, 1))
    assert_size_stride(getitem_1, (32, 6, 1024), (6144, 1024, 1))
    assert_size_stride(getitem_6, (), ())
    assert_size_stride(getitem_7, (), ())
    assert_size_stride(permute_6, (384, 384), (384, 1))
    assert_size_stride(permute_12, (1152, 384), (384, 1))
    assert_size_stride(tangents_1, (32, 1024, 384), (393216, 384, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((384, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (384, 32768), (1, 384), 0), reinterpret_tensor(getitem, (32768, 384), (384, 1), 0), out=buf0)
        buf1 = empty_strided_cuda((32768, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (32768, 384), (384, 1), 0), permute_6, out=buf1)
        del permute_6
        del tangents_1
        buf2 = empty_strided_cuda((384, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(buf0, buf2, 147456, grid=grid(147456), stream=stream0)
        del buf0
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf3 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf1, (32, 6, 1024, 64), (393216, 64, 384, 1), 0), convert_element_type_5, convert_element_type_4, select_2, getitem, getitem_1, None, None, 1024, 1024, 0.0, True, getitem_6, getitem_7, scale=0.125)
        del convert_element_type_4
        del convert_element_type_5
        del getitem
        del getitem_1
        del getitem_6
        del getitem_7
        del select_2
        buf4 = buf3[0]
        buf5 = buf3[1]
        buf6 = buf3[2]
        del buf3
        buf7 = reinterpret_tensor(buf1, (32, 6, 1024, 64), (393216, 64, 384, 1), 0); del buf1  # reuse
        buf8 = empty_strided_cuda((32, 6, 1024, 64), (393216, 64, 384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.slice_backward, aten.add, aten.neg]
        triton_poi_fused__to_copy_add_mul_neg_slice_backward_1.run(buf5, primals_3, primals_4, buf4, buf7, buf8, 12582912, grid=grid(12582912), stream=stream0)
        del buf4
        del buf5
        del primals_3
        del primals_4
        buf9 = empty_strided_cuda((32, 1024, 3, 6, 64), (1179648, 1152, 384, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_2.run(buf6, buf7, buf8, buf9, 37748736, grid=grid(37748736), stream=stream0)
        del buf6
        del buf7
        buf10 = empty_strided_cuda((1152, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf9, (1152, 32768), (1, 1152), 0), view, out=buf10)
        del view
        buf11 = reinterpret_tensor(buf8, (32768, 384), (384, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf9, (32768, 1152), (1152, 1), 0), permute_12, out=buf11)
        del buf9
        del permute_12
        buf12 = empty_strided_cuda((32, 1024, 384), (393216, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf11, buf12, 12582912, grid=grid(12582912), stream=stream0)
        del buf11
        buf13 = empty_strided_cuda((1152, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf10, buf13, 442368, grid=grid(442368), stream=stream0)
        del buf10
    return (buf12, buf13, None, None, buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((32, 1, 1024, 64), (65536, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 1, 1024, 64), (65536, 65536, 64, 1), device='cuda:0', dtype=torch.float32)
    view = rand_strided((32768, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    select_2 = rand_strided((32, 6, 1024, 64), (1179648, 64, 1152, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_4 = rand_strided((32, 6, 1024, 64), (393216, 64, 384, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_5 = rand_strided((32, 6, 1024, 64), (393216, 64, 384, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem = rand_strided((32, 6, 1024, 64), (393216, 64, 384, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_1 = rand_strided((32, 6, 1024), (6144, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_7 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    permute_6 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_12 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((32, 1024, 384), (393216, 384, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_3, primals_4, view, select_2, convert_element_type_4, convert_element_type_5, getitem, getitem_1, getitem_6, getitem_7, permute_6, permute_12, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
