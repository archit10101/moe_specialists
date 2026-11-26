# AOT ID: ['0_forward']
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/6p/c6pqodg3aqrll3nw37xkajgwd2kfuwu3d6cmkgntpyxpbp2i6l5w.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.embedding]
# Source node to ATen node mapping:
#   x => embedding
# Graph fragment:
#   %embedding : [num_users=1] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_4, %primals_3), kwargs = {})
triton_poi_fused_embedding_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 384)
    x0 = xindex % 384
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 50304, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 50304), "index out of bounds: 0 <= tmp4 < 50304")
    tmp6 = tl.load(in_ptr1 + (x0 + (384*tmp4)), None)
    tl.store(out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/4y/c4yw345rqnab7et42pjbtes5qo7k7givyo7wcitveoa2wqzbq5om.py
# Topologically Sorted Source Nodes: [cos, cos_1, sin, sin_1], Original ATen: [aten.cos, aten._to_copy, aten.sin]
# Source node to ATen node mapping:
#   cos => cos
#   cos_1 => convert_element_type_2
#   sin => sin
#   sin_1 => convert_element_type_3
# Graph fragment:
#   %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view,), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cos, torch.float32), kwargs = {})
#   %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view,), kwargs = {})
#   %convert_element_type_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sin, torch.float32), kwargs = {})
triton_poi_fused__to_copy_cos_sin_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cos_sin_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 64)
    x0 = xindex % 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float64)
    tmp2 = 2*(x0 % 32)
    tmp3 = tmp2.to(tl.float64)
    tmp4 = tl.full([1], 0.015625, tl.float64)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1], 10000.0, tl.float64)
    tmp7 = libdevice.pow(tmp6, tmp5)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = tl.full([1], 1.0, tl.float64)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp1 * tmp11
    tmp13 = libdevice.cos(tmp12)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = libdevice.sin(tmp12)
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp14, None)
    tl.store(out_ptr1 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ao/caoymtuzunlvt5hj7ny2dtgieeqnvv7frsf77hebrphgubxx4t5x.py
# Topologically Sorted Source Nodes: [same, ones, causal, and_, m, eye, pad_self, or_], Original ATen: [aten.eq, aten.ones, aten.tril, aten.bitwise_and, aten.eye, aten.bitwise_or]
# Source node to ATen node mapping:
#   and_ => bitwise_and
#   causal => iota_1, le, logical_and, sub
#   eye => eq_1
#   m => bitwise_and_1
#   ones => full_default
#   or_ => bitwise_or
#   pad_self => bitwise_and_2
#   same => eq
# Graph fragment:
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze_4, %unsqueeze_5), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1024, 1024], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %iota_1 : [num_users=4] = call_function[target=torch.ops.prims.iota.default](args = (1024,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_6, %unsqueeze_7), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub, 0), kwargs = {})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le, %full_default), kwargs = {})
#   %bitwise_and : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq, %logical_and), kwargs = {})
#   %bitwise_and_1 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %unsqueeze_8), kwargs = {})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze_10, %iota_1), kwargs = {})
#   %bitwise_and_2 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%unsqueeze_9, %eq_1), kwargs = {})
#   %bitwise_or : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_and_1, %bitwise_and_2), kwargs = {})
triton_poi_fused_bitwise_and_bitwise_or_eq_eye_ones_tril_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_bitwise_or_eq_eye_ones_tril_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = (xindex // 1024)
    x0 = xindex % 1024
    x2 = (xindex // 1048576)
    x1 = (xindex // 1024) % 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 == tmp1
    tmp3 = x0 + ((-1)*x1)
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 <= tmp4
    tmp6 = tl.full([1], True, tl.int1)
    tmp7 = tmp5 & tmp6
    tmp8 = tmp2 & tmp7
    tmp9 = tmp0 >= tmp4
    tmp10 = tmp8 & tmp9
    tmp11 = tmp0 < tmp4
    tmp12 = x1
    tmp13 = x0
    tmp14 = tmp12 == tmp13
    tmp15 = tmp11 & tmp14
    tmp16 = tmp10 | tmp15
    tl.store(out_ptr0 + (x4), tmp16, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/6l/c6lctqwbcnxibgmowd2mau7elenbcbchkv2rid4xvz3aymdngkf6.py
# Topologically Sorted Source Nodes: [ge_1, npf, sum_1], Original ATen: [aten.ge, aten._to_copy, aten.sum]
# Source node to ATen node mapping:
#   ge_1 => ge_1
#   npf => convert_element_type_4
#   sum_1 => sum_1
# Graph fragment:
#   %ge_1 : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%primals_2, 0), kwargs = {})
#   %convert_element_type_4 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%convert_element_type_4,), kwargs = {dtype: torch.float32})
triton_red_fused__to_copy_ge_sum_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[4, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_ge_sum_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (8192*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tl.store(out_ptr0 + (r1 + (8192*x0)), tmp2, rmask & xmask)
        tl.store(out_ptr1 + (r1 + (8192*x0)), tmp3, rmask & xmask)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/qx/cqxszdetjvvo7ykl54r6piwag3nbisxl7jod74xur7xue3rhsiam.py
# Topologically Sorted Source Nodes: [sum_1, n_valid], Original ATen: [aten.sum, aten.clamp]
# Source node to ATen node mapping:
#   n_valid => clamp_min
#   sum_1 => sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%convert_element_type_4,), kwargs = {dtype: torch.float32})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sum_1, 1), kwargs = {})
triton_per_fused_clamp_sum_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {2: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_clamp_sum_4', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 1.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp5, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (32, 1024), (1024, 1))
    assert_size_stride(primals_2, (32, 1024), (1024, 1))
    assert_size_stride(primals_3, (32, 1024), (1024, 1))
    assert_size_stride(primals_4, (50304, 384), (384, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 1024, 384), (393216, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.embedding]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_0.run(primals_3, primals_4, buf0, 12582912, grid=grid(12582912), stream=stream0)
        del primals_4
        buf1 = empty_strided_cuda((32, 1024, 64), (65536, 64, 1), torch.float32)
        buf2 = empty_strided_cuda((32, 1024, 64), (65536, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cos, cos_1, sin, sin_1], Original ATen: [aten.cos, aten._to_copy, aten.sin]
        triton_poi_fused__to_copy_cos_sin_1.run(primals_1, buf1, buf2, 2097152, grid=grid(2097152), stream=stream0)
        del primals_1
        buf3 = empty_strided_cuda((32, 1024, 1024), (1048576, 1024, 1), torch.bool)
        # Topologically Sorted Source Nodes: [same, ones, causal, and_, m, eye, pad_self, or_], Original ATen: [aten.eq, aten.ones, aten.tril, aten.bitwise_and, aten.eye, aten.bitwise_or]
        triton_poi_fused_bitwise_and_bitwise_or_eq_eye_ones_tril_2.run(primals_2, buf3, 33554432, grid=grid(33554432), stream=stream0)
        buf4 = empty_strided_cuda((32, 1024), (1024, 1), torch.bool)
        buf5 = empty_strided_cuda((32768, ), (1, ), torch.float32)
        buf6 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [ge_1, npf, sum_1], Original ATen: [aten.ge, aten._to_copy, aten.sum]
        triton_red_fused__to_copy_ge_sum_3.run(primals_2, buf4, buf5, buf6, 4, 8192, grid=grid(4), stream=stream0)
        del primals_2
        buf7 = empty_strided_cuda((), (), torch.float32)
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [sum_1, n_valid], Original ATen: [aten.sum, aten.clamp]
        triton_per_fused_clamp_sum_4.run(buf8, buf6, 1, 4, grid=grid(1), stream=stream0)
        del buf6
    return (buf0, reinterpret_tensor(buf1, (32, 1, 1024, 64), (65536, 65536, 64, 1), 0), reinterpret_tensor(buf2, (32, 1, 1024, 64), (65536, 65536, 64, 1), 0), reinterpret_tensor(buf3, (32, 1, 1024, 1024), (1048576, 1048576, 1024, 1), 0), reinterpret_tensor(buf4, (32768, ), (1, ), 0), buf5, buf8, primals_3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    primals_2 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    primals_3 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    primals_4 = rand_strided((50304, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
