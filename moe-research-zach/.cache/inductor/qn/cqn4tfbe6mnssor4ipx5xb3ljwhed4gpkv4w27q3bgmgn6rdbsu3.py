# AOT ID: ['0_backward']
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/fi/cfiz7jcthxmwmvat4c3mj22pyxr72yo5hfbr5ct7yemp3bxoa5dh.py
# Topologically Sorted Source Nodes: [y_30], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   y_30 => mul_103
# Graph fragment:
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_60, %rsqrt_20), kwargs = {})
#   %mul_105 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %mul_103), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_105, [0, 1], True), kwargs = {dtype: torch.float32})
triton_red_fused_mul_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/nb/cnb5dzmwwfhwghmvuiu5n3cbarybqbvq65z5spkcaoky3dw3dyke.py
# Topologically Sorted Source Nodes: [y_30], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   y_30 => mul_103
# Graph fragment:
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_60, %rsqrt_20), kwargs = {})
#   %mul_105 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %mul_103), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_105, [0, 1], True), kwargs = {dtype: torch.float32})
triton_red_fused_mul_sum_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
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
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/tf/ctf6itama2j5cttck7h74oa4fm7qw4ez62rwh254vfdz7xopob6m.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.div, aten.pow, aten.add, aten._to_copy, aten.view]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_106 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %primals_75), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_106, %add_60), kwargs = {})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_106, %rsqrt_20), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_107, [2], True), kwargs = {dtype: torch.float32})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_11, 256), kwargs = {})
#   %pow_24 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_60, 1.0), kwargs = {})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_24, 2.0), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %mul_111), kwargs = {})
#   %add_62 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_108, %mul_112), kwargs = {})
#   %convert_element_type_224 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_62, torch.bfloat16), kwargs = {})
#   %view_122 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_224, [32768, 256]), kwargs = {})
triton_per_fused__to_copy_add_div_mul_pow_sum_view_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_pow_sum_view_2', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 32768
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), None)
    tmp1 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r1 + (256*x0)), None)
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp9 = tmp2 * tmp8
    tmp10 = -0.5
    tmp11 = tmp7 * tmp10
    tmp12 = tmp8 * tmp8
    tmp13 = tmp12 * tmp8
    tmp14 = tmp11 * tmp13
    tmp15 = 0.00390625
    tmp16 = tmp14 * tmp15
    tmp17 = 2.0
    tmp18 = tmp3 * tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = tmp9 + tmp19
    tmp21 = tmp20.to(tl.float32)
    tl.store(out_ptr1 + (r1 + (256*x0)), tmp20, None)
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ad/cadnxuvyj27t4spkgy6ot74a7alg5szwrfmujyqbc4uuzsaydkn4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_229 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_73, torch.float32), kwargs = {})
triton_poi_fused__to_copy_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/mp/cmp6j3gfhqrn6oil6as624ytgfiidzw6tpcqae5zcwojrpnfb4fp.py
# Topologically Sorted Source Nodes: [silu_9], Original ATen: [aten.silu, aten.mul, aten.sigmoid, aten.fill, aten.sub, aten.add]
# Source node to ATen node mapping:
#   silu_9 => convert_element_type_215, convert_element_type_216, mul_101, sigmoid_9
# Graph fragment:
#   %convert_element_type_215 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_116, torch.float32), kwargs = {})
#   %sigmoid_9 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_215,), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_215, %sigmoid_9), kwargs = {})
#   %convert_element_type_216 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_101, torch.bfloat16), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_123, %convert_element_type_216), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_123, %view_118), kwargs = {})
#   %sigmoid_10 : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_116,), kwargs = {})
#   %full_default_21 : [num_users=10] = call_function[target=torch.ops.aten.full.default](args = ([32, 1024, 768], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_21, %sigmoid_10), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_116, %sub_1), kwargs = {})
#   %add_63 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_115, 1), kwargs = {})
#   %mul_116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_10, %add_63), kwargs = {})
#   %mul_117 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_114, %mul_116), kwargs = {})
triton_poi_fused_add_fill_mul_sigmoid_silu_sub_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_silu_sub_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp0 * tmp5
    tmp8 = tmp0 * tmp7
    tmp9 = tl.sigmoid(tmp1)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp1 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp8 * tmp14
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr1 + (x0), tmp15, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/cv/ccvf5e3h4xqg6evidoe3irzsz7yxditvxfsh5hd7jrcmpwcxwmvs.py
# Topologically Sorted Source Nodes: [y_29], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   y_29 => mul_99
# Graph fragment:
#   %convert_element_type_234 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_125, torch.float32), kwargs = {})
#   %convert_element_type_240 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_127, torch.float32), kwargs = {})
#   %add_64 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_234, %convert_element_type_240), kwargs = {})
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_58, %rsqrt_19), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_64, %mul_99), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_118, [0, 1], True), kwargs = {dtype: torch.float32})
triton_red_fused__to_copy_add_mul_sum_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mul_sum_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp7 = tmp5 * tmp6
        tmp8 = tmp4 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/hd/chdafjgdglxjojvxrovhtxi4colsuu3sgq63jahrln24monpq52n.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum, aten.div, aten.pow, aten.view]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_234 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_125, torch.float32), kwargs = {})
#   %convert_element_type_240 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_127, torch.float32), kwargs = {})
#   %add_64 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_234, %convert_element_type_240), kwargs = {})
#   %mul_119 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_64, %primals_71), kwargs = {})
#   %mul_120 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_119, %add_58), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_119, %rsqrt_19), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_120, [2], True), kwargs = {dtype: torch.float32})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_62, %mul_121), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_12, 256), kwargs = {})
#   %pow_26 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_58, 1.0), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_26, 2.0), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %mul_124), kwargs = {})
#   %add_66 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_65, %mul_125), kwargs = {})
#   %convert_element_type_242 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_66, torch.bfloat16), kwargs = {})
#   %view_129 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_242, [32768, 256]), kwargs = {})
triton_per_fused__to_copy_add_div_mul_pow_sum_view_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_pow_sum_view_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 32768
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1 + (256*x0)), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (r1 + (256*x0)), None)
    tmp12 = tl.load(in_out_ptr0 + (r1 + (256*x0)), None)
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp14 = tmp6 * tmp13
    tmp15 = tmp12 + tmp14
    tmp16 = -0.5
    tmp17 = tmp11 * tmp16
    tmp18 = tmp13 * tmp13
    tmp19 = tmp18 * tmp13
    tmp20 = tmp17 * tmp19
    tmp21 = 0.00390625
    tmp22 = tmp20 * tmp21
    tmp23 = 2.0
    tmp24 = tmp7 * tmp23
    tmp25 = tmp22 * tmp24
    tmp26 = tmp15 + tmp25
    tmp27 = tmp26.to(tl.float32)
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp26, None)
    tl.store(out_ptr1 + (r1 + (256*x0)), tmp27, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ss/csskolrjykwdfwv6npuntci3szrkiw3gfc3cz7f3ugy4bg3ontlk.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_247 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_86, torch.float32), kwargs = {})
triton_poi_fused__to_copy_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_7', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/wj/cwjy6jwbmymc7v7h6rnnleoqxqhrudngfssnuvomos2kg2anaaiv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.slice_backward, aten.add, aten.neg]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_248 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_40, torch.float32), kwargs = {})
#   %convert_element_type_249 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_41, torch.float32), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_249, %unsqueeze_2), kwargs = {})
#   %convert_element_type_251 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_127, torch.bfloat16), kwargs = {})
#   %full_default_22 : [num_users=40] = call_function[target=torch.ops.aten.full.default](args = ([32, 4, 1024, 64], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_22, %slice_42, 3, 0, 32), kwargs = {})
#   %add_67 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_251, %slice_scatter_default), kwargs = {})
#   %neg_20 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_41,), kwargs = {})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_22, %neg_20, 3, 32, 9223372036854775807), kwargs = {})
#   %add_68 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_67, %slice_scatter_default_1), kwargs = {})
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_248, %unsqueeze_2), kwargs = {})
#   %convert_element_type_253 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_129, torch.bfloat16), kwargs = {})
#   %slice_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_22, %slice_44, 3, 0, 32), kwargs = {})
#   %add_69 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_253, %slice_scatter_default_2), kwargs = {})
#   %neg_21 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_43,), kwargs = {})
#   %slice_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_22, %neg_21, 3, 32, 9223372036854775807), kwargs = {})
#   %add_70 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_69, %slice_scatter_default_3), kwargs = {})
triton_poi_fused__to_copy_add_mul_neg_slice_backward_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*i64', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_neg_slice_backward_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x2 = (xindex // 256)
    x0 = xindex % 64
    x1 = (xindex // 64) % 4
    x3 = (xindex // 256) % 1024
    x4 = (xindex // 262144)
    tmp0 = tl.load(in_ptr0 + (x5), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr2 + (x5), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float64)
    tmp4 = 2*(x0 % 32)
    tmp5 = tmp4.to(tl.float64)
    tmp6 = tl.full([1], 0.015625, tl.float64)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full([1], 10000.0, tl.float64)
    tmp9 = libdevice.pow(tmp8, tmp7)
    tmp10 = tl.full([1], 1, tl.int32)
    tmp11 = tmp10 / tmp9
    tmp12 = tl.full([1], 1.0, tl.float64)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp3 * tmp13
    tmp15 = libdevice.cos(tmp14)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp1 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = x0
    tmp20 = tl.full([1], 32, tl.int64)
    tmp21 = tmp19 < tmp20
    tmp22 = tl.load(in_ptr0 + (32 + x5), tmp21, other=0.0).to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl.load(in_ptr1 + (x2), tmp21, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24.to(tl.float64)
    tmp26 = tmp25 * tmp13
    tmp27 = libdevice.sin(tmp26)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp23 * tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp21, tmp30, tmp31)
    tmp33 = 0.0
    tmp34 = tl.where(tmp21, tmp32, tmp33)
    tmp35 = tmp18 + tmp34
    tmp36 = tmp19 >= tmp20
    tmp37 = tl.load(in_ptr0 + ((-32) + x5), tmp36, other=0.0).to(tl.float32)
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.load(in_ptr1 + (x2), tmp36, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp39.to(tl.float64)
    tmp41 = tmp40 * tmp13
    tmp42 = libdevice.sin(tmp41)
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp38 * tmp43
    tmp45 = tmp44.to(tl.float32)
    tmp46 = -tmp45
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp36, tmp46, tmp47)
    tmp49 = tl.where(tmp36, tmp48, tmp33)
    tmp50 = tmp35 + tmp49
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tmp52 * tmp16
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tl.load(in_ptr2 + (32 + x5), tmp21, other=0.0).to(tl.float32)
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tmp56 * tmp28
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tl.full(tmp58.shape, 0.0, tmp58.dtype)
    tmp60 = tl.where(tmp21, tmp58, tmp59)
    tmp61 = tl.where(tmp21, tmp60, tmp33)
    tmp62 = tmp54 + tmp61
    tmp63 = tl.load(in_ptr2 + ((-32) + x5), tmp36, other=0.0).to(tl.float32)
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp64 * tmp43
    tmp66 = tmp65.to(tl.float32)
    tmp67 = -tmp66
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp36, tmp67, tmp68)
    tmp70 = tl.where(tmp36, tmp69, tmp33)
    tmp71 = tmp62 + tmp70
    tl.store(out_ptr1 + (x0 + (64*x3) + (65536*x1) + (262144*x4)), tmp50, None)
    tl.store(out_ptr3 + (x0 + (64*x3) + (65536*x1) + (262144*x4)), tmp71, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/2h/c2hmg6jkkya75tldd3ofxrha344fncv2pvdhe6qohhs5dcd2xbwu.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_88,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 256) % 3
    x5 = xindex % 256
    x6 = (xindex // 768)
    x0 = xindex % 64
    x1 = (xindex // 64) % 4
    x3 = (xindex // 768) % 1024
    x4 = (xindex // 786432)
    x7 = xindex
    tmp3 = tl.load(in_ptr0 + (x5 + (256*x6)), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (x0 + (64*x3) + (65536*x1) + (262144*x4)), None, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (x0 + (64*x3) + (65536*x1) + (262144*x4)), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x2
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
    tl.store(out_ptr0 + (x7), tmp15, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/bx/cbx3ieqdahw6aqp2rv3wum77qyh7podd6a7r5ooypxxfcpdbwgsz.py
# Topologically Sorted Source Nodes: [y_27], Original ATen: [aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   y_27 => mul_93
# Graph fragment:
#   %convert_element_type_258 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_134, torch.float32), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_54, %rsqrt_18), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_258, %mul_93), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_130, [0, 1], True), kwargs = {dtype: torch.float32})
triton_red_fused__to_copy_mul_sum_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mul_sum_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/rk/crkcsmoeokun44bozuhg5nzucz3bmcq6qetlytylure2o3ygj2ea.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.add, aten.div, aten.pow, aten.view]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_258 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_134, torch.float32), kwargs = {})
#   %mul_131 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_258, %primals_68), kwargs = {})
#   %mul_132 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_131, %add_54), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_131, %rsqrt_18), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_132, [2], True), kwargs = {dtype: torch.float32})
#   %add_73 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_66, %mul_133), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_13, 256), kwargs = {})
#   %pow_28 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_54, 1.0), kwargs = {})
#   %mul_136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_28, 2.0), kwargs = {})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %mul_136), kwargs = {})
#   %add_74 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_73, %mul_137), kwargs = {})
#   %convert_element_type_260 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_74, torch.bfloat16), kwargs = {})
#   %view_136 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_260, [32768, 256]), kwargs = {})
triton_per_fused__to_copy_add_div_mul_pow_sum_view_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_pow_sum_view_11', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
    xnumel = 32768
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (r1 + (256*x0)), None)
    tmp9 = tl.load(in_out_ptr0 + (r1 + (256*x0)), None)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp11 = tmp3 * tmp10
    tmp12 = tmp9 + tmp11
    tmp13 = -0.5
    tmp14 = tmp8 * tmp13
    tmp15 = tmp10 * tmp10
    tmp16 = tmp15 * tmp10
    tmp17 = tmp14 * tmp16
    tmp18 = 0.00390625
    tmp19 = tmp17 * tmp18
    tmp20 = 2.0
    tmp21 = tmp4 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp12 + tmp22
    tmp24 = tmp23.to(tl.float32)
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp23, None)
    tl.store(out_ptr1 + (r1 + (256*x0)), tmp24, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/bt/cbtrselfnf4rqowoc3jzvxese3uzm7gd6gcsthcbcdm6irfgdkc5.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum, aten.div, aten.pow, aten.view]
# Source node to ATen node mapping:
#   x_1 => add_4
# Graph fragment:
#   %convert_element_type_558 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_251, torch.float32), kwargs = {})
#   %convert_element_type_564 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_253, torch.float32), kwargs = {})
#   %add_172 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_558, %convert_element_type_564), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_6), kwargs = {})
#   %mul_344 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_172, %primals_8), kwargs = {})
#   %mul_345 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_344, %add_4), kwargs = {})
#   %mul_346 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_344, %rsqrt_1), kwargs = {})
#   %sum_40 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_345, [2], True), kwargs = {dtype: torch.float32})
#   %add_173 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_170, %mul_346), kwargs = {})
#   %div_20 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_30, 256), kwargs = {})
#   %pow_62 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_4, 1.0), kwargs = {})
#   %mul_349 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_62, 2.0), kwargs = {})
#   %mul_350 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_20, %mul_349), kwargs = {})
#   %add_174 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_173, %mul_350), kwargs = {})
#   %convert_element_type_566 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_174, torch.bfloat16), kwargs = {})
#   %view_255 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_566, [32768, 256]), kwargs = {})
triton_per_fused__to_copy_add_div_mul_pow_sum_view_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: '*fp32', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_pow_sum_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel):
    xnumel = 32768
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1 + (256*x0)), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (r1 + (256*x0)), None)
    tmp8 = tl.load(in_ptr4 + (r1 + (256*x0)), None).to(tl.float32)
    tmp15 = tl.load(in_out_ptr0 + (r1 + (256*x0)), None)
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp4 * tmp5
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp11 = tmp6 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp17 = tmp6 * tmp16
    tmp18 = tmp15 + tmp17
    tmp19 = -0.5
    tmp20 = tmp14 * tmp19
    tmp21 = tmp16 * tmp16
    tmp22 = tmp21 * tmp16
    tmp23 = tmp20 * tmp22
    tmp24 = 0.00390625
    tmp25 = tmp23 * tmp24
    tmp26 = 2.0
    tmp27 = tmp10 * tmp26
    tmp28 = tmp25 * tmp27
    tmp29 = tmp18 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tl.store(in_out_ptr0 + (r1 + (256*x0)), tmp29, None)
    tl.store(out_ptr1 + (r1 + (256*x0)), tmp30, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/zu/czujriyvdmy3uuzslkhebdyfof7nxiu2r7b6yq7iskvzn7gm3fm4.py
# Topologically Sorted Source Nodes: [x_1, y_2, y], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   x_1 => add_4
#   y => mul_3
#   y_2 => mul_9
# Graph fragment:
#   %convert_element_type_558 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_251, torch.float32), kwargs = {})
#   %convert_element_type_564 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_253, torch.float32), kwargs = {})
#   %add_172 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_558, %convert_element_type_564), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_6), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, %rsqrt_1), kwargs = {})
#   %mul_343 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_172, %mul_9), kwargs = {})
#   %sum_39 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_343, [0, 1], True), kwargs = {dtype: torch.float32})
#   %convert_element_type_582 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_260, torch.float32), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, %rsqrt), kwargs = {})
#   %mul_355 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_582, %mul_3), kwargs = {})
#   %sum_41 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_355, [0, 1], True), kwargs = {dtype: torch.float32})
triton_red_fused__to_copy_add_mul_sum_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*bf16', 4: '*fp32', 5: '*bf16', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mul_sum_13', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr4 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr5 + (x0 + (256*r2) + (32768*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr6 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp5 + tmp7
        tmp10 = tmp8 * tmp9
        tmp11 = tmp4 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp5 * tmp17
        tmp19 = tmp16 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, None)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/d7/cd7rzjzoyso6ot5lskrbullpd32gtutk5dridptmmznia6zttulw.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_102 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([50304, 256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_embedding_dense_backward_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_14', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12877824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/d5/cd5w7ovocjvyqwizbzz3r3bgpjkhfgibzjppvos2t2npdbtv6ntf.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.add, aten.div, aten.pow, aten.embedding_dense_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_582 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_260, torch.float32), kwargs = {})
#   %mul_356 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_582, %primals_5), kwargs = {})
#   %mul_357 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_356, %embedding), kwargs = {})
#   %mul_358 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_356, %rsqrt), kwargs = {})
#   %sum_42 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_357, [2], True), kwargs = {dtype: torch.float32})
#   %add_181 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_174, %mul_358), kwargs = {})
#   %div_21 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_31, 256), kwargs = {})
#   %pow_64 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%embedding, 1.0), kwargs = {})
#   %mul_361 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_64, 2.0), kwargs = {})
#   %mul_362 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_21, %mul_361), kwargs = {})
#   %add_182 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_181, %mul_362), kwargs = {})
#   %full_default_101 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_10 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_12, %full_default_101, %add_182), kwargs = {})
#   %full_default_102 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([50304, 256], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_102, [%primals_2], %where_10, True), kwargs = {})
triton_per_fused__to_copy_add_div_embedding_dense_backward_mul_pow_sum_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_embedding_dense_backward_mul_pow_sum_15', 'mutated_arg_names': ['out_ptr1'], 'no_x_dim': True, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel):
    xnumel = 32768
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (256*x0)), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (r1 + (256*x0)), None)
    tmp9 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (r1 + (256*x0)), None)
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp10 = tl.full([RBLOCK], 50304, tl.int32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tl.device_assert((0 <= tmp13) & (tmp13 < 50304), "index out of bounds: 0 <= tmp13 < 50304")
    tmp15 = tl.full([1], -1, tl.int64)
    tmp16 = tmp9 == tmp15
    tmp19 = tmp3 * tmp18
    tmp20 = tmp17 + tmp19
    tmp21 = -0.5
    tmp22 = tmp8 * tmp21
    tmp23 = tmp18 * tmp18
    tmp24 = tmp23 * tmp18
    tmp25 = tmp22 * tmp24
    tmp26 = 0.00390625
    tmp27 = tmp25 * tmp26
    tmp28 = 2.0
    tmp29 = tmp4 * tmp28
    tmp30 = tmp27 * tmp29
    tmp31 = tmp20 + tmp30
    tmp32 = 0.0
    tmp33 = tl.where(tmp16, tmp32, tmp31)
    tl.atomic_add(out_ptr1 + (tl.broadcast_to(r1 + (256*tmp13), [RBLOCK])), tmp33, None, sem='relaxed')
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_2, primals_3, primals_5, primals_8, primals_12, primals_15, primals_19, primals_22, primals_26, primals_29, primals_33, primals_36, primals_40, primals_43, primals_47, primals_50, primals_54, primals_57, primals_61, primals_64, primals_68, primals_71, primals_75, embedding, rsqrt, view_1, select_2, convert_element_type_8, convert_element_type_9, where, getitem, getitem_1, getitem_2, getitem_3, mm_1, rsqrt_1, view_7, mm_2, mm_3, view_11, add_6, rsqrt_2, view_13, select_5, convert_element_type_30, convert_element_type_31, getitem_4, getitem_5, getitem_6, getitem_7, add_10, rsqrt_3, view_19, mm_7, mm_8, view_23, add_12, rsqrt_4, view_25, select_8, convert_element_type_52, convert_element_type_53, getitem_8, getitem_9, getitem_10, getitem_11, add_16, rsqrt_5, view_31, mm_12, mm_13, view_35, add_18, rsqrt_6, view_37, select_11, convert_element_type_74, convert_element_type_75, getitem_12, getitem_13, getitem_14, getitem_15, add_22, rsqrt_7, view_43, mm_17, mm_18, view_47, add_24, rsqrt_8, view_49, select_14, convert_element_type_96, convert_element_type_97, getitem_16, getitem_17, getitem_18, getitem_19, add_28, rsqrt_9, view_55, mm_22, mm_23, view_59, add_30, rsqrt_10, view_61, select_17, convert_element_type_118, convert_element_type_119, getitem_20, getitem_21, getitem_22, getitem_23, add_34, rsqrt_11, view_67, mm_27, mm_28, view_71, add_36, rsqrt_12, view_73, select_20, convert_element_type_140, convert_element_type_141, getitem_24, getitem_25, getitem_26, getitem_27, add_40, rsqrt_13, view_79, mm_32, mm_33, view_83, add_42, rsqrt_14, view_85, select_23, convert_element_type_162, convert_element_type_163, getitem_28, getitem_29, getitem_30, getitem_31, add_46, rsqrt_15, view_91, mm_37, mm_38, view_95, add_48, rsqrt_16, view_97, select_26, convert_element_type_184, convert_element_type_185, getitem_32, getitem_33, getitem_34, getitem_35, add_52, rsqrt_17, view_103, mm_42, mm_43, view_107, add_54, rsqrt_18, view_109, select_29, convert_element_type_206, convert_element_type_207, getitem_36, getitem_37, getitem_38, getitem_39, add_58, rsqrt_19, view_115, mm_47, mm_48, view_119, add_60, rsqrt_20, permute_72, permute_76, permute_81, permute_85, permute_91, permute_95, permute_99, permute_104, permute_108, permute_114, permute_118, permute_122, permute_127, permute_131, permute_137, permute_141, permute_145, permute_150, permute_154, permute_160, permute_164, permute_168, permute_173, permute_177, permute_183, permute_187, permute_191, permute_196, permute_200, permute_206, permute_210, permute_214, permute_219, permute_223, permute_229, permute_233, permute_237, permute_242, permute_246, permute_252, permute_256, permute_260, permute_265, permute_269, permute_275, permute_279, permute_283, permute_288, permute_292, permute_298, tangents_1 = args
    args.clear()
    assert_size_stride(primals_2, (32, 1024), (1024, 1))
    assert_size_stride(primals_3, (32, 1024), (1024, 1))
    assert_size_stride(primals_5, (256, ), (1, ))
    assert_size_stride(primals_8, (256, ), (1, ))
    assert_size_stride(primals_12, (256, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_26, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_36, (256, ), (1, ))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_54, (256, ), (1, ))
    assert_size_stride(primals_57, (256, ), (1, ))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_64, (256, ), (1, ))
    assert_size_stride(primals_68, (256, ), (1, ))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(embedding, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_1, (32768, 256), (256, 1))
    assert_size_stride(select_2, (32, 4, 1024, 64), (786432, 64, 768, 1))
    assert_size_stride(convert_element_type_8, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(convert_element_type_9, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(where, (32, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(getitem, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_1, (32, 4, 1024), (4096, 1024, 1))
    assert_size_stride(getitem_2, (), ())
    assert_size_stride(getitem_3, (), ())
    assert_size_stride(mm_1, (32768, 256), (256, 1))
    assert_size_stride(rsqrt_1, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_7, (32768, 256), (256, 1))
    assert_size_stride(mm_2, (32768, 768), (768, 1))
    assert_size_stride(mm_3, (32768, 768), (768, 1))
    assert_size_stride(view_11, (32768, 768), (768, 1))
    assert_size_stride(add_6, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_2, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_13, (32768, 256), (256, 1))
    assert_size_stride(select_5, (32, 4, 1024, 64), (786432, 64, 768, 1))
    assert_size_stride(convert_element_type_30, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(convert_element_type_31, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_4, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_5, (32, 4, 1024), (4096, 1024, 1))
    assert_size_stride(getitem_6, (), ())
    assert_size_stride(getitem_7, (), ())
    assert_size_stride(add_10, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_3, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_19, (32768, 256), (256, 1))
    assert_size_stride(mm_7, (32768, 768), (768, 1))
    assert_size_stride(mm_8, (32768, 768), (768, 1))
    assert_size_stride(view_23, (32768, 768), (768, 1))
    assert_size_stride(add_12, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_4, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_25, (32768, 256), (256, 1))
    assert_size_stride(select_8, (32, 4, 1024, 64), (786432, 64, 768, 1))
    assert_size_stride(convert_element_type_52, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(convert_element_type_53, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_8, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_9, (32, 4, 1024), (4096, 1024, 1))
    assert_size_stride(getitem_10, (), ())
    assert_size_stride(getitem_11, (), ())
    assert_size_stride(add_16, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_5, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_31, (32768, 256), (256, 1))
    assert_size_stride(mm_12, (32768, 768), (768, 1))
    assert_size_stride(mm_13, (32768, 768), (768, 1))
    assert_size_stride(view_35, (32768, 768), (768, 1))
    assert_size_stride(add_18, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_6, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_37, (32768, 256), (256, 1))
    assert_size_stride(select_11, (32, 4, 1024, 64), (786432, 64, 768, 1))
    assert_size_stride(convert_element_type_74, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(convert_element_type_75, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_12, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_13, (32, 4, 1024), (4096, 1024, 1))
    assert_size_stride(getitem_14, (), ())
    assert_size_stride(getitem_15, (), ())
    assert_size_stride(add_22, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_7, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_43, (32768, 256), (256, 1))
    assert_size_stride(mm_17, (32768, 768), (768, 1))
    assert_size_stride(mm_18, (32768, 768), (768, 1))
    assert_size_stride(view_47, (32768, 768), (768, 1))
    assert_size_stride(add_24, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_8, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_49, (32768, 256), (256, 1))
    assert_size_stride(select_14, (32, 4, 1024, 64), (786432, 64, 768, 1))
    assert_size_stride(convert_element_type_96, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(convert_element_type_97, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_16, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_17, (32, 4, 1024), (4096, 1024, 1))
    assert_size_stride(getitem_18, (), ())
    assert_size_stride(getitem_19, (), ())
    assert_size_stride(add_28, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_9, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_55, (32768, 256), (256, 1))
    assert_size_stride(mm_22, (32768, 768), (768, 1))
    assert_size_stride(mm_23, (32768, 768), (768, 1))
    assert_size_stride(view_59, (32768, 768), (768, 1))
    assert_size_stride(add_30, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_10, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_61, (32768, 256), (256, 1))
    assert_size_stride(select_17, (32, 4, 1024, 64), (786432, 64, 768, 1))
    assert_size_stride(convert_element_type_118, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(convert_element_type_119, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_20, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_21, (32, 4, 1024), (4096, 1024, 1))
    assert_size_stride(getitem_22, (), ())
    assert_size_stride(getitem_23, (), ())
    assert_size_stride(add_34, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_11, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_67, (32768, 256), (256, 1))
    assert_size_stride(mm_27, (32768, 768), (768, 1))
    assert_size_stride(mm_28, (32768, 768), (768, 1))
    assert_size_stride(view_71, (32768, 768), (768, 1))
    assert_size_stride(add_36, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_12, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_73, (32768, 256), (256, 1))
    assert_size_stride(select_20, (32, 4, 1024, 64), (786432, 64, 768, 1))
    assert_size_stride(convert_element_type_140, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(convert_element_type_141, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_24, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_25, (32, 4, 1024), (4096, 1024, 1))
    assert_size_stride(getitem_26, (), ())
    assert_size_stride(getitem_27, (), ())
    assert_size_stride(add_40, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_13, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_79, (32768, 256), (256, 1))
    assert_size_stride(mm_32, (32768, 768), (768, 1))
    assert_size_stride(mm_33, (32768, 768), (768, 1))
    assert_size_stride(view_83, (32768, 768), (768, 1))
    assert_size_stride(add_42, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_14, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_85, (32768, 256), (256, 1))
    assert_size_stride(select_23, (32, 4, 1024, 64), (786432, 64, 768, 1))
    assert_size_stride(convert_element_type_162, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(convert_element_type_163, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_28, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_29, (32, 4, 1024), (4096, 1024, 1))
    assert_size_stride(getitem_30, (), ())
    assert_size_stride(getitem_31, (), ())
    assert_size_stride(add_46, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_15, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_91, (32768, 256), (256, 1))
    assert_size_stride(mm_37, (32768, 768), (768, 1))
    assert_size_stride(mm_38, (32768, 768), (768, 1))
    assert_size_stride(view_95, (32768, 768), (768, 1))
    assert_size_stride(add_48, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_16, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_97, (32768, 256), (256, 1))
    assert_size_stride(select_26, (32, 4, 1024, 64), (786432, 64, 768, 1))
    assert_size_stride(convert_element_type_184, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(convert_element_type_185, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_32, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_33, (32, 4, 1024), (4096, 1024, 1))
    assert_size_stride(getitem_34, (), ())
    assert_size_stride(getitem_35, (), ())
    assert_size_stride(add_52, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_17, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_103, (32768, 256), (256, 1))
    assert_size_stride(mm_42, (32768, 768), (768, 1))
    assert_size_stride(mm_43, (32768, 768), (768, 1))
    assert_size_stride(view_107, (32768, 768), (768, 1))
    assert_size_stride(add_54, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_18, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_109, (32768, 256), (256, 1))
    assert_size_stride(select_29, (32, 4, 1024, 64), (786432, 64, 768, 1))
    assert_size_stride(convert_element_type_206, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(convert_element_type_207, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_36, (32, 4, 1024, 64), (262144, 64, 256, 1))
    assert_size_stride(getitem_37, (32, 4, 1024), (4096, 1024, 1))
    assert_size_stride(getitem_38, (), ())
    assert_size_stride(getitem_39, (), ())
    assert_size_stride(add_58, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_19, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(view_115, (32768, 256), (256, 1))
    assert_size_stride(mm_47, (32768, 768), (768, 1))
    assert_size_stride(mm_48, (32768, 768), (768, 1))
    assert_size_stride(view_119, (32768, 768), (768, 1))
    assert_size_stride(add_60, (32, 1024, 256), (262144, 256, 1))
    assert_size_stride(rsqrt_20, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(permute_72, (256, 768), (768, 1))
    assert_size_stride(permute_76, (768, 256), (256, 1))
    assert_size_stride(permute_81, (768, 256), (256, 1))
    assert_size_stride(permute_85, (256, 256), (256, 1))
    assert_size_stride(permute_91, (768, 256), (256, 1))
    assert_size_stride(permute_95, (256, 768), (768, 1))
    assert_size_stride(permute_99, (768, 256), (256, 1))
    assert_size_stride(permute_104, (768, 256), (256, 1))
    assert_size_stride(permute_108, (256, 256), (256, 1))
    assert_size_stride(permute_114, (768, 256), (256, 1))
    assert_size_stride(permute_118, (256, 768), (768, 1))
    assert_size_stride(permute_122, (768, 256), (256, 1))
    assert_size_stride(permute_127, (768, 256), (256, 1))
    assert_size_stride(permute_131, (256, 256), (256, 1))
    assert_size_stride(permute_137, (768, 256), (256, 1))
    assert_size_stride(permute_141, (256, 768), (768, 1))
    assert_size_stride(permute_145, (768, 256), (256, 1))
    assert_size_stride(permute_150, (768, 256), (256, 1))
    assert_size_stride(permute_154, (256, 256), (256, 1))
    assert_size_stride(permute_160, (768, 256), (256, 1))
    assert_size_stride(permute_164, (256, 768), (768, 1))
    assert_size_stride(permute_168, (768, 256), (256, 1))
    assert_size_stride(permute_173, (768, 256), (256, 1))
    assert_size_stride(permute_177, (256, 256), (256, 1))
    assert_size_stride(permute_183, (768, 256), (256, 1))
    assert_size_stride(permute_187, (256, 768), (768, 1))
    assert_size_stride(permute_191, (768, 256), (256, 1))
    assert_size_stride(permute_196, (768, 256), (256, 1))
    assert_size_stride(permute_200, (256, 256), (256, 1))
    assert_size_stride(permute_206, (768, 256), (256, 1))
    assert_size_stride(permute_210, (256, 768), (768, 1))
    assert_size_stride(permute_214, (768, 256), (256, 1))
    assert_size_stride(permute_219, (768, 256), (256, 1))
    assert_size_stride(permute_223, (256, 256), (256, 1))
    assert_size_stride(permute_229, (768, 256), (256, 1))
    assert_size_stride(permute_233, (256, 768), (768, 1))
    assert_size_stride(permute_237, (768, 256), (256, 1))
    assert_size_stride(permute_242, (768, 256), (256, 1))
    assert_size_stride(permute_246, (256, 256), (256, 1))
    assert_size_stride(permute_252, (768, 256), (256, 1))
    assert_size_stride(permute_256, (256, 768), (768, 1))
    assert_size_stride(permute_260, (768, 256), (256, 1))
    assert_size_stride(permute_265, (768, 256), (256, 1))
    assert_size_stride(permute_269, (256, 256), (256, 1))
    assert_size_stride(permute_275, (768, 256), (256, 1))
    assert_size_stride(permute_279, (256, 768), (768, 1))
    assert_size_stride(permute_283, (768, 256), (256, 1))
    assert_size_stride(permute_288, (768, 256), (256, 1))
    assert_size_stride(permute_292, (256, 256), (256, 1))
    assert_size_stride(permute_298, (768, 256), (256, 1))
    assert_size_stride(tangents_1, (32, 1024, 256), (262144, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1, 256, 256), (65536, 65536, 1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [y_30], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_mul_sum_0.run(tangents_1, add_60, rsqrt_20, buf0, 65536, 128, grid=grid(65536), stream=stream0)
        buf1 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_30], Original ATen: [aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf0, buf1, 256, 256, grid=grid(256), stream=stream0)
        buf3 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf4 = empty_strided_cuda((32768, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.sum, aten.div, aten.pow, aten.add, aten._to_copy, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_2.run(tangents_1, primals_75, add_60, rsqrt_20, buf3, buf4, 32768, 256, grid=grid(32768), stream=stream0)
        del add_60
        del primals_75
        del rsqrt_20
        del tangents_1
        buf5 = empty_strided_cuda((256, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (256, 32768), (1, 256), 0), view_119, out=buf5)
        del view_119
        buf6 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, permute_72, out=buf6)
        del permute_72
        buf7 = empty_strided_cuda((256, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf5, buf7, 196608, grid=grid(196608), stream=stream0)
        buf8 = empty_strided_cuda((32, 1024, 768), (786432, 768, 1), torch.bfloat16)
        buf12 = empty_strided_cuda((32, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu_9], Original ATen: [aten.silu, aten.mul, aten.sigmoid, aten.fill, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_4.run(buf6, mm_47, mm_48, buf8, buf12, 25165824, grid=grid(25165824), stream=stream0)
        del mm_47
        del mm_48
        buf9 = reinterpret_tensor(buf5, (768, 256), (256, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (768, 32768), (1, 768), 0), view_115, out=buf9)
        buf10 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (32768, 768), (768, 1), 0), permute_76, out=buf10)
        del permute_76
        buf11 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf9, buf11, 196608, grid=grid(196608), stream=stream0)
        buf13 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (768, 32768), (1, 768), 0), view_115, out=buf13)
        del view_115
        buf14 = empty_strided_cuda((32768, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (32768, 768), (768, 1), 0), permute_81, out=buf14)
        del permute_81
        buf15 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf13, buf15, 196608, grid=grid(196608), stream=stream0)
        buf16 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [y_29], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused__to_copy_add_mul_sum_5.run(buf10, buf14, add_58, rsqrt_19, buf16, 65536, 128, grid=grid(65536), stream=stream0)
        buf17 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_29], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf16, buf17, 256, 256, grid=grid(256), stream=stream0)
        buf19 = buf3; del buf3  # reuse
        buf20 = empty_strided_cuda((32768, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_6.run(buf19, buf10, buf14, primals_71, add_58, rsqrt_19, buf20, 32768, 256, grid=grid(32768), stream=stream0)
        del add_58
        del buf10
        del primals_71
        del rsqrt_19
        buf21 = empty_strided_cuda((256, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (256, 32768), (1, 256), 0), reinterpret_tensor(getitem_36, (32768, 256), (256, 1), 0), out=buf21)
        buf22 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf20, permute_85, out=buf22)
        del permute_85
        buf23 = reinterpret_tensor(buf16, (256, 256), (256, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7.run(buf21, buf23, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf24 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf22, (32, 4, 1024, 64), (262144, 64, 256, 1), 0), convert_element_type_207, convert_element_type_206, select_29, reinterpret_tensor(where, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), getitem_36, getitem_37, getitem_38, getitem_39, 0.0, [True, True, True, False])
        del convert_element_type_206
        del convert_element_type_207
        del getitem_36
        del getitem_37
        del getitem_38
        del getitem_39
        del select_29
        buf25 = buf24[0]
        buf26 = buf24[1]
        buf27 = buf24[2]
        del buf24
        buf29 = reinterpret_tensor(buf22, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf22  # reuse
        buf31 = reinterpret_tensor(buf20, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.slice_backward, aten.add, aten.neg]
        triton_poi_fused__to_copy_add_mul_neg_slice_backward_8.run(buf26, primals_3, buf25, buf29, buf31, 8388608, grid=grid(8388608), stream=stream0)
        del buf25
        del buf26
        buf32 = reinterpret_tensor(buf12, (32, 1024, 3, 4, 64), (786432, 768, 256, 64, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf27, buf29, buf31, buf32, 25165824, grid=grid(25165824), stream=stream0)
        buf33 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (768, 32768), (1, 768), 0), view_109, out=buf33)
        del view_109
        buf34 = reinterpret_tensor(buf31, (32768, 256), (256, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (32768, 768), (768, 1), 0), permute_91, out=buf34)
        del permute_91
        buf35 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf33, buf35, 196608, grid=grid(196608), stream=stream0)
        buf36 = empty_strided_cuda((1, 1, 256, 256), (65536, 65536, 1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [y_27], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused__to_copy_mul_sum_10.run(buf34, add_54, rsqrt_18, buf36, 65536, 128, grid=grid(65536), stream=stream0)
        buf37 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_27], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf36, buf37, 256, 256, grid=grid(256), stream=stream0)
        buf39 = buf19; del buf19  # reuse
        buf40 = reinterpret_tensor(buf29, (32768, 256), (256, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.add, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_11.run(buf39, buf34, primals_68, add_54, rsqrt_18, buf40, 32768, 256, grid=grid(32768), stream=stream0)
        del add_54
        del primals_68
        del rsqrt_18
        buf41 = reinterpret_tensor(buf33, (256, 768), (768, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (256, 32768), (1, 256), 0), view_107, out=buf41)
        del view_107
        buf42 = reinterpret_tensor(buf32, (32768, 768), (768, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf40, permute_95, out=buf42)
        del permute_95
        buf43 = empty_strided_cuda((256, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf41, buf43, 196608, grid=grid(196608), stream=stream0)
        buf44 = buf8; del buf8  # reuse
        buf48 = reinterpret_tensor(buf6, (32, 1024, 768), (786432, 768, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [silu_8], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_4.run(buf42, mm_42, mm_43, buf44, buf48, 25165824, grid=grid(25165824), stream=stream0)
        del mm_42
        del mm_43
        buf45 = reinterpret_tensor(buf41, (768, 256), (256, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (768, 32768), (1, 768), 0), view_103, out=buf45)
        buf46 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (32768, 768), (768, 1), 0), permute_99, out=buf46)
        del permute_99
        buf47 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf45, buf47, 196608, grid=grid(196608), stream=stream0)
        buf49 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf48, (768, 32768), (1, 768), 0), view_103, out=buf49)
        del view_103
        buf50 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf48, (32768, 768), (768, 1), 0), permute_104, out=buf50)
        del permute_104
        buf51 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf49, buf51, 196608, grid=grid(196608), stream=stream0)
        buf52 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [y_26], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused__to_copy_add_mul_sum_5.run(buf46, buf50, add_52, rsqrt_17, buf52, 65536, 128, grid=grid(65536), stream=stream0)
        buf53 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_26], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf52, buf53, 256, 256, grid=grid(256), stream=stream0)
        buf55 = buf39; del buf39  # reuse
        buf56 = reinterpret_tensor(buf27, (32768, 256), (256, 1), 0); del buf27  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_6.run(buf55, buf46, buf50, primals_64, add_52, rsqrt_17, buf56, 32768, 256, grid=grid(32768), stream=stream0)
        del add_52
        del buf46
        del primals_64
        del rsqrt_17
        buf57 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf56, (256, 32768), (1, 256), 0), reinterpret_tensor(getitem_32, (32768, 256), (256, 1), 0), out=buf57)
        buf58 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf56, permute_108, out=buf58)
        del permute_108
        buf59 = reinterpret_tensor(buf52, (256, 256), (256, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7.run(buf57, buf59, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf60 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf58, (32, 4, 1024, 64), (262144, 64, 256, 1), 0), convert_element_type_185, convert_element_type_184, select_26, reinterpret_tensor(where, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), getitem_32, getitem_33, getitem_34, getitem_35, 0.0, [True, True, True, False])
        del convert_element_type_184
        del convert_element_type_185
        del getitem_32
        del getitem_33
        del getitem_34
        del getitem_35
        del select_26
        buf61 = buf60[0]
        buf62 = buf60[1]
        buf63 = buf60[2]
        del buf60
        buf65 = reinterpret_tensor(buf58, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf58  # reuse
        buf67 = reinterpret_tensor(buf56, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten._to_copy, aten.mul, aten.add, aten.neg]
        triton_poi_fused__to_copy_add_mul_neg_slice_backward_8.run(buf62, primals_3, buf61, buf65, buf67, 8388608, grid=grid(8388608), stream=stream0)
        del buf61
        del buf62
        buf68 = reinterpret_tensor(buf48, (32, 1024, 3, 4, 64), (786432, 768, 256, 64, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf63, buf65, buf67, buf68, 25165824, grid=grid(25165824), stream=stream0)
        buf69 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf68, (768, 32768), (1, 768), 0), view_97, out=buf69)
        del view_97
        buf70 = reinterpret_tensor(buf67, (32768, 256), (256, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf68, (32768, 768), (768, 1), 0), permute_114, out=buf70)
        del permute_114
        buf71 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf69, buf71, 196608, grid=grid(196608), stream=stream0)
        buf72 = empty_strided_cuda((1, 1, 256, 256), (65536, 65536, 1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [y_24], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused__to_copy_mul_sum_10.run(buf70, add_48, rsqrt_16, buf72, 65536, 128, grid=grid(65536), stream=stream0)
        buf73 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_24], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf72, buf73, 256, 256, grid=grid(256), stream=stream0)
        buf75 = buf55; del buf55  # reuse
        buf76 = reinterpret_tensor(buf65, (32768, 256), (256, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.add, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_11.run(buf75, buf70, primals_61, add_48, rsqrt_16, buf76, 32768, 256, grid=grid(32768), stream=stream0)
        del add_48
        del primals_61
        del rsqrt_16
        buf77 = reinterpret_tensor(buf69, (256, 768), (768, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf76, (256, 32768), (1, 256), 0), view_95, out=buf77)
        del view_95
        buf78 = reinterpret_tensor(buf68, (32768, 768), (768, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf76, permute_118, out=buf78)
        del permute_118
        buf79 = empty_strided_cuda((256, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf77, buf79, 196608, grid=grid(196608), stream=stream0)
        buf80 = buf44; del buf44  # reuse
        buf84 = reinterpret_tensor(buf42, (32, 1024, 768), (786432, 768, 1), 0); del buf42  # reuse
        # Topologically Sorted Source Nodes: [silu_7], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_4.run(buf78, mm_37, mm_38, buf80, buf84, 25165824, grid=grid(25165824), stream=stream0)
        del mm_37
        del mm_38
        buf81 = reinterpret_tensor(buf77, (768, 256), (256, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (768, 32768), (1, 768), 0), view_91, out=buf81)
        buf82 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (32768, 768), (768, 1), 0), permute_122, out=buf82)
        del permute_122
        buf83 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf81, buf83, 196608, grid=grid(196608), stream=stream0)
        buf85 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf84, (768, 32768), (1, 768), 0), view_91, out=buf85)
        del view_91
        buf86 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf84, (32768, 768), (768, 1), 0), permute_127, out=buf86)
        del permute_127
        buf87 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf85, buf87, 196608, grid=grid(196608), stream=stream0)
        buf88 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [y_23], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused__to_copy_add_mul_sum_5.run(buf82, buf86, add_46, rsqrt_15, buf88, 65536, 128, grid=grid(65536), stream=stream0)
        buf89 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_23], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf88, buf89, 256, 256, grid=grid(256), stream=stream0)
        buf91 = buf75; del buf75  # reuse
        buf92 = reinterpret_tensor(buf63, (32768, 256), (256, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_6.run(buf91, buf82, buf86, primals_57, add_46, rsqrt_15, buf92, 32768, 256, grid=grid(32768), stream=stream0)
        del add_46
        del buf82
        del primals_57
        del rsqrt_15
        buf93 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf92, (256, 32768), (1, 256), 0), reinterpret_tensor(getitem_28, (32768, 256), (256, 1), 0), out=buf93)
        buf94 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf92, permute_131, out=buf94)
        del permute_131
        buf95 = reinterpret_tensor(buf88, (256, 256), (256, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7.run(buf93, buf95, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf96 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf94, (32, 4, 1024, 64), (262144, 64, 256, 1), 0), convert_element_type_163, convert_element_type_162, select_23, reinterpret_tensor(where, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), getitem_28, getitem_29, getitem_30, getitem_31, 0.0, [True, True, True, False])
        del convert_element_type_162
        del convert_element_type_163
        del getitem_28
        del getitem_29
        del getitem_30
        del getitem_31
        del select_23
        buf97 = buf96[0]
        buf98 = buf96[1]
        buf99 = buf96[2]
        del buf96
        buf101 = reinterpret_tensor(buf94, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf94  # reuse
        buf103 = reinterpret_tensor(buf92, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten._to_copy, aten.mul, aten.add, aten.neg]
        triton_poi_fused__to_copy_add_mul_neg_slice_backward_8.run(buf98, primals_3, buf97, buf101, buf103, 8388608, grid=grid(8388608), stream=stream0)
        del buf97
        del buf98
        buf104 = reinterpret_tensor(buf84, (32, 1024, 3, 4, 64), (786432, 768, 256, 64, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf99, buf101, buf103, buf104, 25165824, grid=grid(25165824), stream=stream0)
        buf105 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (768, 32768), (1, 768), 0), view_85, out=buf105)
        del view_85
        buf106 = reinterpret_tensor(buf99, (32768, 256), (256, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (32768, 768), (768, 1), 0), permute_137, out=buf106)
        del permute_137
        buf107 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf105, buf107, 196608, grid=grid(196608), stream=stream0)
        buf108 = empty_strided_cuda((1, 1, 256, 256), (65536, 65536, 1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [y_21], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused__to_copy_mul_sum_10.run(buf106, add_42, rsqrt_14, buf108, 65536, 128, grid=grid(65536), stream=stream0)
        buf109 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_21], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf108, buf109, 256, 256, grid=grid(256), stream=stream0)
        buf111 = buf91; del buf91  # reuse
        buf112 = reinterpret_tensor(buf103, (32768, 256), (256, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.add, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_11.run(buf111, buf106, primals_54, add_42, rsqrt_14, buf112, 32768, 256, grid=grid(32768), stream=stream0)
        del add_42
        del primals_54
        del rsqrt_14
        buf113 = reinterpret_tensor(buf105, (256, 768), (768, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf112, (256, 32768), (1, 256), 0), view_83, out=buf113)
        del view_83
        buf114 = reinterpret_tensor(buf104, (32768, 768), (768, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf112, permute_141, out=buf114)
        del permute_141
        buf115 = empty_strided_cuda((256, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf113, buf115, 196608, grid=grid(196608), stream=stream0)
        buf116 = buf80; del buf80  # reuse
        buf120 = reinterpret_tensor(buf78, (32, 1024, 768), (786432, 768, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [silu_6], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_4.run(buf114, mm_32, mm_33, buf116, buf120, 25165824, grid=grid(25165824), stream=stream0)
        del mm_32
        del mm_33
        buf117 = reinterpret_tensor(buf113, (768, 256), (256, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (768, 32768), (1, 768), 0), view_79, out=buf117)
        buf118 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (32768, 768), (768, 1), 0), permute_145, out=buf118)
        del permute_145
        buf119 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf117, buf119, 196608, grid=grid(196608), stream=stream0)
        buf121 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf120, (768, 32768), (1, 768), 0), view_79, out=buf121)
        del view_79
        buf122 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf120, (32768, 768), (768, 1), 0), permute_150, out=buf122)
        del permute_150
        buf123 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf121, buf123, 196608, grid=grid(196608), stream=stream0)
        buf124 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [y_20], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused__to_copy_add_mul_sum_5.run(buf118, buf122, add_40, rsqrt_13, buf124, 65536, 128, grid=grid(65536), stream=stream0)
        buf125 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_20], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf124, buf125, 256, 256, grid=grid(256), stream=stream0)
        buf127 = buf111; del buf111  # reuse
        buf128 = reinterpret_tensor(buf101, (32768, 256), (256, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_6.run(buf127, buf118, buf122, primals_50, add_40, rsqrt_13, buf128, 32768, 256, grid=grid(32768), stream=stream0)
        del add_40
        del buf118
        del primals_50
        del rsqrt_13
        buf129 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf128, (256, 32768), (1, 256), 0), reinterpret_tensor(getitem_24, (32768, 256), (256, 1), 0), out=buf129)
        buf130 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf128, permute_154, out=buf130)
        del permute_154
        buf131 = reinterpret_tensor(buf124, (256, 256), (256, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7.run(buf129, buf131, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf132 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf130, (32, 4, 1024, 64), (262144, 64, 256, 1), 0), convert_element_type_141, convert_element_type_140, select_20, reinterpret_tensor(where, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), getitem_24, getitem_25, getitem_26, getitem_27, 0.0, [True, True, True, False])
        del convert_element_type_140
        del convert_element_type_141
        del getitem_24
        del getitem_25
        del getitem_26
        del getitem_27
        del select_20
        buf133 = buf132[0]
        buf134 = buf132[1]
        buf135 = buf132[2]
        del buf132
        buf137 = reinterpret_tensor(buf130, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf130  # reuse
        buf139 = reinterpret_tensor(buf128, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten._to_copy, aten.mul, aten.add, aten.neg]
        triton_poi_fused__to_copy_add_mul_neg_slice_backward_8.run(buf134, primals_3, buf133, buf137, buf139, 8388608, grid=grid(8388608), stream=stream0)
        del buf133
        del buf134
        buf140 = reinterpret_tensor(buf120, (32, 1024, 3, 4, 64), (786432, 768, 256, 64, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf135, buf137, buf139, buf140, 25165824, grid=grid(25165824), stream=stream0)
        buf141 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (768, 32768), (1, 768), 0), view_73, out=buf141)
        del view_73
        buf142 = reinterpret_tensor(buf139, (32768, 256), (256, 1), 0); del buf139  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (32768, 768), (768, 1), 0), permute_160, out=buf142)
        del permute_160
        buf143 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf141, buf143, 196608, grid=grid(196608), stream=stream0)
        buf144 = empty_strided_cuda((1, 1, 256, 256), (65536, 65536, 1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [y_18], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused__to_copy_mul_sum_10.run(buf142, add_36, rsqrt_12, buf144, 65536, 128, grid=grid(65536), stream=stream0)
        buf145 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_18], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf144, buf145, 256, 256, grid=grid(256), stream=stream0)
        buf147 = buf127; del buf127  # reuse
        buf148 = reinterpret_tensor(buf137, (32768, 256), (256, 1), 0); del buf137  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.add, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_11.run(buf147, buf142, primals_47, add_36, rsqrt_12, buf148, 32768, 256, grid=grid(32768), stream=stream0)
        del add_36
        del primals_47
        del rsqrt_12
        buf149 = reinterpret_tensor(buf141, (256, 768), (768, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf148, (256, 32768), (1, 256), 0), view_71, out=buf149)
        del view_71
        buf150 = reinterpret_tensor(buf140, (32768, 768), (768, 1), 0); del buf140  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf148, permute_164, out=buf150)
        del permute_164
        buf151 = empty_strided_cuda((256, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf149, buf151, 196608, grid=grid(196608), stream=stream0)
        buf152 = buf116; del buf116  # reuse
        buf156 = reinterpret_tensor(buf114, (32, 1024, 768), (786432, 768, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [silu_5], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_4.run(buf150, mm_27, mm_28, buf152, buf156, 25165824, grid=grid(25165824), stream=stream0)
        del mm_27
        del mm_28
        buf153 = reinterpret_tensor(buf149, (768, 256), (256, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (768, 32768), (1, 768), 0), view_67, out=buf153)
        buf154 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (32768, 768), (768, 1), 0), permute_168, out=buf154)
        del permute_168
        buf155 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf153, buf155, 196608, grid=grid(196608), stream=stream0)
        buf157 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf156, (768, 32768), (1, 768), 0), view_67, out=buf157)
        del view_67
        buf158 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf156, (32768, 768), (768, 1), 0), permute_173, out=buf158)
        del permute_173
        buf159 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf157, buf159, 196608, grid=grid(196608), stream=stream0)
        buf160 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [y_17], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused__to_copy_add_mul_sum_5.run(buf154, buf158, add_34, rsqrt_11, buf160, 65536, 128, grid=grid(65536), stream=stream0)
        buf161 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_17], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf160, buf161, 256, 256, grid=grid(256), stream=stream0)
        buf163 = buf147; del buf147  # reuse
        buf164 = reinterpret_tensor(buf135, (32768, 256), (256, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_6.run(buf163, buf154, buf158, primals_43, add_34, rsqrt_11, buf164, 32768, 256, grid=grid(32768), stream=stream0)
        del add_34
        del buf154
        del primals_43
        del rsqrt_11
        buf165 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf164, (256, 32768), (1, 256), 0), reinterpret_tensor(getitem_20, (32768, 256), (256, 1), 0), out=buf165)
        buf166 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf164, permute_177, out=buf166)
        del permute_177
        buf167 = reinterpret_tensor(buf160, (256, 256), (256, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7.run(buf165, buf167, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf168 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf166, (32, 4, 1024, 64), (262144, 64, 256, 1), 0), convert_element_type_119, convert_element_type_118, select_17, reinterpret_tensor(where, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), getitem_20, getitem_21, getitem_22, getitem_23, 0.0, [True, True, True, False])
        del convert_element_type_118
        del convert_element_type_119
        del getitem_20
        del getitem_21
        del getitem_22
        del getitem_23
        del select_17
        buf169 = buf168[0]
        buf170 = buf168[1]
        buf171 = buf168[2]
        del buf168
        buf173 = reinterpret_tensor(buf166, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf166  # reuse
        buf175 = reinterpret_tensor(buf164, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten._to_copy, aten.mul, aten.add, aten.neg]
        triton_poi_fused__to_copy_add_mul_neg_slice_backward_8.run(buf170, primals_3, buf169, buf173, buf175, 8388608, grid=grid(8388608), stream=stream0)
        del buf169
        del buf170
        buf176 = reinterpret_tensor(buf156, (32, 1024, 3, 4, 64), (786432, 768, 256, 64, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf171, buf173, buf175, buf176, 25165824, grid=grid(25165824), stream=stream0)
        buf177 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (768, 32768), (1, 768), 0), view_61, out=buf177)
        del view_61
        buf178 = reinterpret_tensor(buf175, (32768, 256), (256, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (32768, 768), (768, 1), 0), permute_183, out=buf178)
        del permute_183
        buf179 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf177, buf179, 196608, grid=grid(196608), stream=stream0)
        buf180 = empty_strided_cuda((1, 1, 256, 256), (65536, 65536, 1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [y_15], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused__to_copy_mul_sum_10.run(buf178, add_30, rsqrt_10, buf180, 65536, 128, grid=grid(65536), stream=stream0)
        buf181 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_15], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf180, buf181, 256, 256, grid=grid(256), stream=stream0)
        buf183 = buf163; del buf163  # reuse
        buf184 = reinterpret_tensor(buf173, (32768, 256), (256, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.add, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_11.run(buf183, buf178, primals_40, add_30, rsqrt_10, buf184, 32768, 256, grid=grid(32768), stream=stream0)
        del add_30
        del primals_40
        del rsqrt_10
        buf185 = reinterpret_tensor(buf177, (256, 768), (768, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf184, (256, 32768), (1, 256), 0), view_59, out=buf185)
        del view_59
        buf186 = reinterpret_tensor(buf176, (32768, 768), (768, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf184, permute_187, out=buf186)
        del permute_187
        buf187 = empty_strided_cuda((256, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf185, buf187, 196608, grid=grid(196608), stream=stream0)
        buf188 = buf152; del buf152  # reuse
        buf192 = reinterpret_tensor(buf150, (32, 1024, 768), (786432, 768, 1), 0); del buf150  # reuse
        # Topologically Sorted Source Nodes: [silu_4], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_4.run(buf186, mm_22, mm_23, buf188, buf192, 25165824, grid=grid(25165824), stream=stream0)
        del mm_22
        del mm_23
        buf189 = reinterpret_tensor(buf185, (768, 256), (256, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (768, 32768), (1, 768), 0), view_55, out=buf189)
        buf190 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (32768, 768), (768, 1), 0), permute_191, out=buf190)
        del permute_191
        buf191 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf189, buf191, 196608, grid=grid(196608), stream=stream0)
        buf193 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (768, 32768), (1, 768), 0), view_55, out=buf193)
        del view_55
        buf194 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf192, (32768, 768), (768, 1), 0), permute_196, out=buf194)
        del permute_196
        buf195 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf193, buf195, 196608, grid=grid(196608), stream=stream0)
        buf196 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [y_14], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused__to_copy_add_mul_sum_5.run(buf190, buf194, add_28, rsqrt_9, buf196, 65536, 128, grid=grid(65536), stream=stream0)
        buf197 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_14], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf196, buf197, 256, 256, grid=grid(256), stream=stream0)
        buf199 = buf183; del buf183  # reuse
        buf200 = reinterpret_tensor(buf171, (32768, 256), (256, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_6.run(buf199, buf190, buf194, primals_36, add_28, rsqrt_9, buf200, 32768, 256, grid=grid(32768), stream=stream0)
        del add_28
        del buf190
        del primals_36
        del rsqrt_9
        buf201 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (256, 32768), (1, 256), 0), reinterpret_tensor(getitem_16, (32768, 256), (256, 1), 0), out=buf201)
        buf202 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf200, permute_200, out=buf202)
        del permute_200
        buf203 = reinterpret_tensor(buf196, (256, 256), (256, 1), 0); del buf196  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7.run(buf201, buf203, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf204 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf202, (32, 4, 1024, 64), (262144, 64, 256, 1), 0), convert_element_type_97, convert_element_type_96, select_14, reinterpret_tensor(where, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), getitem_16, getitem_17, getitem_18, getitem_19, 0.0, [True, True, True, False])
        del convert_element_type_96
        del convert_element_type_97
        del getitem_16
        del getitem_17
        del getitem_18
        del getitem_19
        del select_14
        buf205 = buf204[0]
        buf206 = buf204[1]
        buf207 = buf204[2]
        del buf204
        buf209 = reinterpret_tensor(buf202, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf202  # reuse
        buf211 = reinterpret_tensor(buf200, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf200  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten._to_copy, aten.mul, aten.add, aten.neg]
        triton_poi_fused__to_copy_add_mul_neg_slice_backward_8.run(buf206, primals_3, buf205, buf209, buf211, 8388608, grid=grid(8388608), stream=stream0)
        del buf205
        del buf206
        buf212 = reinterpret_tensor(buf192, (32, 1024, 3, 4, 64), (786432, 768, 256, 64, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf207, buf209, buf211, buf212, 25165824, grid=grid(25165824), stream=stream0)
        buf213 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (768, 32768), (1, 768), 0), view_49, out=buf213)
        del view_49
        buf214 = reinterpret_tensor(buf211, (32768, 256), (256, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (32768, 768), (768, 1), 0), permute_206, out=buf214)
        del permute_206
        buf215 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf213, buf215, 196608, grid=grid(196608), stream=stream0)
        buf216 = empty_strided_cuda((1, 1, 256, 256), (65536, 65536, 1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [y_12], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused__to_copy_mul_sum_10.run(buf214, add_24, rsqrt_8, buf216, 65536, 128, grid=grid(65536), stream=stream0)
        buf217 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_12], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf216, buf217, 256, 256, grid=grid(256), stream=stream0)
        buf219 = buf199; del buf199  # reuse
        buf220 = reinterpret_tensor(buf209, (32768, 256), (256, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.add, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_11.run(buf219, buf214, primals_33, add_24, rsqrt_8, buf220, 32768, 256, grid=grid(32768), stream=stream0)
        del add_24
        del primals_33
        del rsqrt_8
        buf221 = reinterpret_tensor(buf213, (256, 768), (768, 1), 0); del buf213  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf220, (256, 32768), (1, 256), 0), view_47, out=buf221)
        del view_47
        buf222 = reinterpret_tensor(buf212, (32768, 768), (768, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf220, permute_210, out=buf222)
        del permute_210
        buf223 = empty_strided_cuda((256, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf221, buf223, 196608, grid=grid(196608), stream=stream0)
        buf224 = buf188; del buf188  # reuse
        buf228 = reinterpret_tensor(buf186, (32, 1024, 768), (786432, 768, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [silu_3], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_4.run(buf222, mm_17, mm_18, buf224, buf228, 25165824, grid=grid(25165824), stream=stream0)
        del mm_17
        del mm_18
        buf225 = reinterpret_tensor(buf221, (768, 256), (256, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (768, 32768), (1, 768), 0), view_43, out=buf225)
        buf226 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (32768, 768), (768, 1), 0), permute_214, out=buf226)
        del permute_214
        buf227 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf225, buf227, 196608, grid=grid(196608), stream=stream0)
        buf229 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (768, 32768), (1, 768), 0), view_43, out=buf229)
        del view_43
        buf230 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (32768, 768), (768, 1), 0), permute_219, out=buf230)
        del permute_219
        buf231 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf229, buf231, 196608, grid=grid(196608), stream=stream0)
        buf232 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [y_11], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused__to_copy_add_mul_sum_5.run(buf226, buf230, add_22, rsqrt_7, buf232, 65536, 128, grid=grid(65536), stream=stream0)
        buf233 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_11], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf232, buf233, 256, 256, grid=grid(256), stream=stream0)
        buf235 = buf219; del buf219  # reuse
        buf236 = reinterpret_tensor(buf207, (32768, 256), (256, 1), 0); del buf207  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_6.run(buf235, buf226, buf230, primals_29, add_22, rsqrt_7, buf236, 32768, 256, grid=grid(32768), stream=stream0)
        del add_22
        del buf226
        del primals_29
        del rsqrt_7
        buf237 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf236, (256, 32768), (1, 256), 0), reinterpret_tensor(getitem_12, (32768, 256), (256, 1), 0), out=buf237)
        buf238 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf236, permute_223, out=buf238)
        del permute_223
        buf239 = reinterpret_tensor(buf232, (256, 256), (256, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7.run(buf237, buf239, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf240 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf238, (32, 4, 1024, 64), (262144, 64, 256, 1), 0), convert_element_type_75, convert_element_type_74, select_11, reinterpret_tensor(where, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), getitem_12, getitem_13, getitem_14, getitem_15, 0.0, [True, True, True, False])
        del convert_element_type_74
        del convert_element_type_75
        del getitem_12
        del getitem_13
        del getitem_14
        del getitem_15
        del select_11
        buf241 = buf240[0]
        buf242 = buf240[1]
        buf243 = buf240[2]
        del buf240
        buf245 = reinterpret_tensor(buf238, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf238  # reuse
        buf247 = reinterpret_tensor(buf236, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten._to_copy, aten.mul, aten.add, aten.neg]
        triton_poi_fused__to_copy_add_mul_neg_slice_backward_8.run(buf242, primals_3, buf241, buf245, buf247, 8388608, grid=grid(8388608), stream=stream0)
        del buf241
        del buf242
        buf248 = reinterpret_tensor(buf228, (32, 1024, 3, 4, 64), (786432, 768, 256, 64, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf243, buf245, buf247, buf248, 25165824, grid=grid(25165824), stream=stream0)
        buf249 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (768, 32768), (1, 768), 0), view_37, out=buf249)
        del view_37
        buf250 = reinterpret_tensor(buf247, (32768, 256), (256, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (32768, 768), (768, 1), 0), permute_229, out=buf250)
        del permute_229
        buf251 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf249, buf251, 196608, grid=grid(196608), stream=stream0)
        buf252 = empty_strided_cuda((1, 1, 256, 256), (65536, 65536, 1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [y_9], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused__to_copy_mul_sum_10.run(buf250, add_18, rsqrt_6, buf252, 65536, 128, grid=grid(65536), stream=stream0)
        buf253 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_9], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf252, buf253, 256, 256, grid=grid(256), stream=stream0)
        buf255 = buf235; del buf235  # reuse
        buf256 = reinterpret_tensor(buf245, (32768, 256), (256, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.add, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_11.run(buf255, buf250, primals_26, add_18, rsqrt_6, buf256, 32768, 256, grid=grid(32768), stream=stream0)
        del add_18
        del primals_26
        del rsqrt_6
        buf257 = reinterpret_tensor(buf249, (256, 768), (768, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf256, (256, 32768), (1, 256), 0), view_35, out=buf257)
        del view_35
        buf258 = reinterpret_tensor(buf248, (32768, 768), (768, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf256, permute_233, out=buf258)
        del permute_233
        buf259 = empty_strided_cuda((256, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf257, buf259, 196608, grid=grid(196608), stream=stream0)
        buf260 = buf224; del buf224  # reuse
        buf264 = reinterpret_tensor(buf222, (32, 1024, 768), (786432, 768, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [silu_2], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_4.run(buf258, mm_12, mm_13, buf260, buf264, 25165824, grid=grid(25165824), stream=stream0)
        del mm_12
        del mm_13
        buf261 = reinterpret_tensor(buf257, (768, 256), (256, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf260, (768, 32768), (1, 768), 0), view_31, out=buf261)
        buf262 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf260, (32768, 768), (768, 1), 0), permute_237, out=buf262)
        del permute_237
        buf263 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf261, buf263, 196608, grid=grid(196608), stream=stream0)
        buf265 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (768, 32768), (1, 768), 0), view_31, out=buf265)
        del view_31
        buf266 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (32768, 768), (768, 1), 0), permute_242, out=buf266)
        del permute_242
        buf267 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf265, buf267, 196608, grid=grid(196608), stream=stream0)
        buf268 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [y_8], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused__to_copy_add_mul_sum_5.run(buf262, buf266, add_16, rsqrt_5, buf268, 65536, 128, grid=grid(65536), stream=stream0)
        buf269 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_8], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf268, buf269, 256, 256, grid=grid(256), stream=stream0)
        buf271 = buf255; del buf255  # reuse
        buf272 = reinterpret_tensor(buf243, (32768, 256), (256, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_6.run(buf271, buf262, buf266, primals_22, add_16, rsqrt_5, buf272, 32768, 256, grid=grid(32768), stream=stream0)
        del add_16
        del buf262
        del primals_22
        del rsqrt_5
        buf273 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf272, (256, 32768), (1, 256), 0), reinterpret_tensor(getitem_8, (32768, 256), (256, 1), 0), out=buf273)
        buf274 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf272, permute_246, out=buf274)
        del permute_246
        buf275 = reinterpret_tensor(buf268, (256, 256), (256, 1), 0); del buf268  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7.run(buf273, buf275, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf276 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf274, (32, 4, 1024, 64), (262144, 64, 256, 1), 0), convert_element_type_53, convert_element_type_52, select_8, reinterpret_tensor(where, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), getitem_8, getitem_9, getitem_10, getitem_11, 0.0, [True, True, True, False])
        del convert_element_type_52
        del convert_element_type_53
        del getitem_10
        del getitem_11
        del getitem_8
        del getitem_9
        del select_8
        buf277 = buf276[0]
        buf278 = buf276[1]
        buf279 = buf276[2]
        del buf276
        buf281 = reinterpret_tensor(buf274, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf274  # reuse
        buf283 = reinterpret_tensor(buf272, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten._to_copy, aten.mul, aten.add, aten.neg]
        triton_poi_fused__to_copy_add_mul_neg_slice_backward_8.run(buf278, primals_3, buf277, buf281, buf283, 8388608, grid=grid(8388608), stream=stream0)
        del buf277
        del buf278
        buf284 = reinterpret_tensor(buf264, (32, 1024, 3, 4, 64), (786432, 768, 256, 64, 1), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf279, buf281, buf283, buf284, 25165824, grid=grid(25165824), stream=stream0)
        buf285 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf284, (768, 32768), (1, 768), 0), view_25, out=buf285)
        del view_25
        buf286 = reinterpret_tensor(buf283, (32768, 256), (256, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf284, (32768, 768), (768, 1), 0), permute_252, out=buf286)
        del permute_252
        buf287 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf285, buf287, 196608, grid=grid(196608), stream=stream0)
        buf288 = empty_strided_cuda((1, 1, 256, 256), (65536, 65536, 1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [y_6], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused__to_copy_mul_sum_10.run(buf286, add_12, rsqrt_4, buf288, 65536, 128, grid=grid(65536), stream=stream0)
        buf289 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_6], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf288, buf289, 256, 256, grid=grid(256), stream=stream0)
        buf291 = buf271; del buf271  # reuse
        buf292 = reinterpret_tensor(buf281, (32768, 256), (256, 1), 0); del buf281  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.add, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_11.run(buf291, buf286, primals_19, add_12, rsqrt_4, buf292, 32768, 256, grid=grid(32768), stream=stream0)
        del add_12
        del primals_19
        del rsqrt_4
        buf293 = reinterpret_tensor(buf285, (256, 768), (768, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf292, (256, 32768), (1, 256), 0), view_23, out=buf293)
        del view_23
        buf294 = reinterpret_tensor(buf284, (32768, 768), (768, 1), 0); del buf284  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf292, permute_256, out=buf294)
        del permute_256
        buf295 = empty_strided_cuda((256, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf293, buf295, 196608, grid=grid(196608), stream=stream0)
        buf296 = buf260; del buf260  # reuse
        buf300 = reinterpret_tensor(buf258, (32, 1024, 768), (786432, 768, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [silu_1], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_4.run(buf294, mm_7, mm_8, buf296, buf300, 25165824, grid=grid(25165824), stream=stream0)
        del mm_7
        del mm_8
        buf297 = reinterpret_tensor(buf293, (768, 256), (256, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (768, 32768), (1, 768), 0), view_19, out=buf297)
        buf298 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf296, (32768, 768), (768, 1), 0), permute_260, out=buf298)
        del permute_260
        buf299 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf297, buf299, 196608, grid=grid(196608), stream=stream0)
        buf301 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf300, (768, 32768), (1, 768), 0), view_19, out=buf301)
        del view_19
        buf302 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf300, (32768, 768), (768, 1), 0), permute_265, out=buf302)
        del permute_265
        buf303 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf301, buf303, 196608, grid=grid(196608), stream=stream0)
        buf304 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [y_5], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused__to_copy_add_mul_sum_5.run(buf298, buf302, add_10, rsqrt_3, buf304, 65536, 128, grid=grid(65536), stream=stream0)
        buf305 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_5], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf304, buf305, 256, 256, grid=grid(256), stream=stream0)
        buf307 = buf291; del buf291  # reuse
        buf308 = reinterpret_tensor(buf279, (32768, 256), (256, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_6.run(buf307, buf298, buf302, primals_15, add_10, rsqrt_3, buf308, 32768, 256, grid=grid(32768), stream=stream0)
        del add_10
        del buf298
        del primals_15
        del rsqrt_3
        buf309 = buf273; del buf273  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf308, (256, 32768), (1, 256), 0), reinterpret_tensor(getitem_4, (32768, 256), (256, 1), 0), out=buf309)
        buf310 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf308, permute_269, out=buf310)
        del permute_269
        buf311 = reinterpret_tensor(buf304, (256, 256), (256, 1), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7.run(buf309, buf311, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf312 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf310, (32, 4, 1024, 64), (262144, 64, 256, 1), 0), convert_element_type_31, convert_element_type_30, select_5, reinterpret_tensor(where, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), getitem_4, getitem_5, getitem_6, getitem_7, 0.0, [True, True, True, False])
        del convert_element_type_30
        del convert_element_type_31
        del getitem_4
        del getitem_5
        del getitem_6
        del getitem_7
        del select_5
        buf313 = buf312[0]
        buf314 = buf312[1]
        buf315 = buf312[2]
        del buf312
        buf317 = reinterpret_tensor(buf310, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf310  # reuse
        buf319 = reinterpret_tensor(buf308, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf308  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten._to_copy, aten.mul, aten.add, aten.neg]
        triton_poi_fused__to_copy_add_mul_neg_slice_backward_8.run(buf314, primals_3, buf313, buf317, buf319, 8388608, grid=grid(8388608), stream=stream0)
        buf320 = reinterpret_tensor(buf300, (32, 1024, 3, 4, 64), (786432, 768, 256, 64, 1), 0); del buf300  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf315, buf317, buf319, buf320, 25165824, grid=grid(25165824), stream=stream0)
        buf321 = buf301; del buf301  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf320, (768, 32768), (1, 768), 0), view_13, out=buf321)
        del view_13
        buf322 = reinterpret_tensor(buf319, (32768, 256), (256, 1), 0); del buf319  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf320, (32768, 768), (768, 1), 0), permute_275, out=buf322)
        del permute_275
        buf323 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf321, buf323, 196608, grid=grid(196608), stream=stream0)
        buf324 = empty_strided_cuda((1, 1, 256, 256), (65536, 65536, 1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [y_3], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused__to_copy_mul_sum_10.run(buf322, add_6, rsqrt_2, buf324, 65536, 128, grid=grid(65536), stream=stream0)
        buf325 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_3], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf324, buf325, 256, 256, grid=grid(256), stream=stream0)
        buf327 = buf307; del buf307  # reuse
        buf328 = reinterpret_tensor(buf317, (32768, 256), (256, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.add, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_11.run(buf327, buf322, primals_12, add_6, rsqrt_2, buf328, 32768, 256, grid=grid(32768), stream=stream0)
        del add_6
        del primals_12
        del rsqrt_2
        buf329 = reinterpret_tensor(buf321, (256, 768), (768, 1), 0); del buf321  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf328, (256, 32768), (1, 256), 0), view_11, out=buf329)
        del view_11
        buf330 = reinterpret_tensor(buf320, (32768, 768), (768, 1), 0); del buf320  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf328, permute_279, out=buf330)
        del permute_279
        buf331 = empty_strided_cuda((256, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf329, buf331, 196608, grid=grid(196608), stream=stream0)
        buf332 = buf296; del buf296  # reuse
        buf336 = reinterpret_tensor(buf294, (32, 1024, 768), (786432, 768, 1), 0); del buf294  # reuse
        # Topologically Sorted Source Nodes: [silu], Original ATen: [aten.fill, aten.silu, aten.mul, aten.sigmoid, aten.sub, aten.add]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_4.run(buf330, mm_2, mm_3, buf332, buf336, 25165824, grid=grid(25165824), stream=stream0)
        del buf330
        del mm_2
        del mm_3
        buf333 = reinterpret_tensor(buf329, (768, 256), (256, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (768, 32768), (1, 768), 0), view_7, out=buf333)
        buf334 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (32768, 768), (768, 1), 0), permute_283, out=buf334)
        del buf332
        del permute_283
        buf335 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf333, buf335, 196608, grid=grid(196608), stream=stream0)
        buf337 = buf333; del buf333  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf336, (768, 32768), (1, 768), 0), view_7, out=buf337)
        del view_7
        buf338 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf336, (32768, 768), (768, 1), 0), permute_288, out=buf338)
        del permute_288
        buf339 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf337, buf339, 196608, grid=grid(196608), stream=stream0)
        buf343 = buf327; del buf327  # reuse
        buf344 = reinterpret_tensor(buf315, (32768, 256), (256, 1), 0); del buf315  # reuse
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum, aten.div, aten.pow, aten.view]
        triton_per_fused__to_copy_add_div_mul_pow_sum_view_12.run(buf343, buf334, buf338, primals_8, embedding, mm_1, rsqrt_1, buf344, 32768, 256, grid=grid(32768), stream=stream0)
        del primals_8
        buf346 = reinterpret_tensor(buf314, (32768, 256), (256, 1), 0); del buf314  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf344, permute_292, out=buf346)
        del permute_292
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._scaled_dot_product_efficient_attention_backward]
        buf348 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(reinterpret_tensor(buf346, (32, 4, 1024, 64), (262144, 64, 256, 1), 0), convert_element_type_9, convert_element_type_8, select_2, reinterpret_tensor(where, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), getitem, getitem_1, getitem_2, getitem_3, 0.0, [True, True, True, False])
        del convert_element_type_8
        del convert_element_type_9
        del getitem_1
        del getitem_2
        del getitem_3
        del select_2
        del where
        buf351 = buf348[2]
        buf349 = buf348[0]
        buf350 = buf348[1]
        del buf348
        buf353 = reinterpret_tensor(buf346, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf346  # reuse
        buf355 = reinterpret_tensor(buf313, (32, 4, 1024, 64), (262144, 65536, 64, 1), 0); del buf313  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.slice_backward, aten._to_copy, aten.mul, aten.add, aten.neg]
        triton_poi_fused__to_copy_add_mul_neg_slice_backward_8.run(buf350, primals_3, buf349, buf353, buf355, 8388608, grid=grid(8388608), stream=stream0)
        del buf349
        del buf350
        del primals_3
        buf356 = reinterpret_tensor(buf336, (32, 1024, 3, 4, 64), (786432, 768, 256, 64, 1), 0); del buf336  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_9.run(buf351, buf353, buf355, buf356, 25165824, grid=grid(25165824), stream=stream0)
        del buf351
        del buf353
        buf358 = reinterpret_tensor(buf355, (32768, 256), (256, 1), 0); del buf355  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf356, (32768, 768), (768, 1), 0), permute_298, out=buf358)
        del permute_298
        buf340 = buf324; del buf324  # reuse
        buf360 = empty_strided_cuda((1, 1, 256, 256), (65536, 65536, 1, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, y_2, y], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused__to_copy_add_mul_sum_13.run(buf334, buf338, embedding, mm_1, rsqrt_1, buf358, rsqrt, buf340, buf360, 65536, 128, grid=grid(65536), stream=stream0)
        del buf334
        del buf338
        del mm_1
        del rsqrt_1
        buf341 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, y_2], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf340, buf341, 256, 256, grid=grid(256), stream=stream0)
        buf345 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf344, (256, 32768), (1, 256), 0), reinterpret_tensor(getitem, (32768, 256), (256, 1), 0), out=buf345)
        del buf344
        del getitem
        buf347 = reinterpret_tensor(buf340, (256, 256), (256, 1), 0); del buf340  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_7.run(buf345, buf347, 65536, grid=grid(65536), stream=stream0)
        del buf345
        buf357 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf356, (768, 32768), (1, 768), 0), view_1, out=buf357)
        del buf356
        del view_1
        buf359 = empty_strided_cuda((768, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_3.run(buf357, buf359, 196608, grid=grid(196608), stream=stream0)
        del buf357
        buf361 = empty_strided_cuda((1, 1, 256), (256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten._to_copy, aten.mul, aten.sum]
        triton_red_fused_mul_sum_1.run(buf360, buf361, 256, 256, grid=grid(256), stream=stream0)
        del buf360
        buf363 = empty_strided_cuda((50304, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_14.run(buf363, 12877824, grid=grid(12877824), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.mul, aten.sum, aten.add, aten.div, aten.pow, aten.embedding_dense_backward]
        triton_per_fused__to_copy_add_div_embedding_dense_backward_mul_pow_sum_15.run(buf358, primals_5, embedding, primals_2, buf343, rsqrt, buf363, 32768, 256, grid=grid(32768), stream=stream0)
        del buf343
        del buf358
        del embedding
        del primals_2
        del primals_5
        del rsqrt
    return (buf363, None, None, None, reinterpret_tensor(buf361, (256, ), (1, ), 0), buf359, buf347, reinterpret_tensor(buf341, (256, ), (1, ), 0), buf339, buf335, buf331, reinterpret_tensor(buf325, (256, ), (1, ), 0), buf323, buf311, reinterpret_tensor(buf305, (256, ), (1, ), 0), buf303, buf299, buf295, reinterpret_tensor(buf289, (256, ), (1, ), 0), buf287, buf275, reinterpret_tensor(buf269, (256, ), (1, ), 0), buf267, buf263, buf259, reinterpret_tensor(buf253, (256, ), (1, ), 0), buf251, buf239, reinterpret_tensor(buf233, (256, ), (1, ), 0), buf231, buf227, buf223, reinterpret_tensor(buf217, (256, ), (1, ), 0), buf215, buf203, reinterpret_tensor(buf197, (256, ), (1, ), 0), buf195, buf191, buf187, reinterpret_tensor(buf181, (256, ), (1, ), 0), buf179, buf167, reinterpret_tensor(buf161, (256, ), (1, ), 0), buf159, buf155, buf151, reinterpret_tensor(buf145, (256, ), (1, ), 0), buf143, buf131, reinterpret_tensor(buf125, (256, ), (1, ), 0), buf123, buf119, buf115, reinterpret_tensor(buf109, (256, ), (1, ), 0), buf107, buf95, reinterpret_tensor(buf89, (256, ), (1, ), 0), buf87, buf83, buf79, reinterpret_tensor(buf73, (256, ), (1, ), 0), buf71, buf59, reinterpret_tensor(buf53, (256, ), (1, ), 0), buf51, buf47, buf43, reinterpret_tensor(buf37, (256, ), (1, ), 0), buf35, buf23, reinterpret_tensor(buf17, (256, ), (1, ), 0), buf15, buf11, buf7, reinterpret_tensor(buf1, (256, ), (1, ), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    primals_3 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    embedding = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_1 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    select_2 = rand_strided((32, 4, 1024, 64), (786432, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_8 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_9 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    where = rand_strided((32, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_1 = rand_strided((32, 4, 1024), (4096, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_3 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    mm_1 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_1 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_7 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_2 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_3 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_11 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    add_6 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_2 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_13 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    select_5 = rand_strided((32, 4, 1024, 64), (786432, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_30 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_31 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_4 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_5 = rand_strided((32, 4, 1024), (4096, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_7 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_10 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_3 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_19 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_7 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_8 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_23 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    add_12 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_4 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_25 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    select_8 = rand_strided((32, 4, 1024, 64), (786432, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_52 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_53 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_8 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_9 = rand_strided((32, 4, 1024), (4096, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_10 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_11 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_16 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_5 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_31 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_12 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_13 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_35 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    add_18 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_6 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    select_11 = rand_strided((32, 4, 1024, 64), (786432, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_74 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_75 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_12 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_13 = rand_strided((32, 4, 1024), (4096, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_14 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_15 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_22 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_7 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_43 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_17 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_18 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_47 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    add_24 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_8 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_49 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    select_14 = rand_strided((32, 4, 1024, 64), (786432, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_96 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_97 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_16 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_17 = rand_strided((32, 4, 1024), (4096, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_18 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_19 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_28 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_9 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_55 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_22 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_23 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_59 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    add_30 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_10 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_61 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    select_17 = rand_strided((32, 4, 1024, 64), (786432, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_118 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_119 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_20 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_21 = rand_strided((32, 4, 1024), (4096, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_22 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_23 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_34 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_11 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_67 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_27 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_28 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_71 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    add_36 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_12 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_73 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    select_20 = rand_strided((32, 4, 1024, 64), (786432, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_140 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_141 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_24 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_25 = rand_strided((32, 4, 1024), (4096, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_26 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_27 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_40 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_13 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_79 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_32 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_33 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_83 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    add_42 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_14 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_85 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    select_23 = rand_strided((32, 4, 1024, 64), (786432, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_162 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_163 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_28 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_29 = rand_strided((32, 4, 1024), (4096, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_30 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_31 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_46 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_15 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_91 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_37 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_38 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_95 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    add_48 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_16 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_97 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    select_26 = rand_strided((32, 4, 1024, 64), (786432, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_184 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_185 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_32 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_33 = rand_strided((32, 4, 1024), (4096, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_34 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_35 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_52 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_17 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_103 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_42 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_43 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_107 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    add_54 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_18 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_109 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    select_29 = rand_strided((32, 4, 1024, 64), (786432, 64, 768, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_206 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_207 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_36 = rand_strided((32, 4, 1024, 64), (262144, 64, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_37 = rand_strided((32, 4, 1024), (4096, 1024, 1), device='cuda:0', dtype=torch.float32)
    getitem_38 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_39 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    add_58 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_19 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    view_115 = rand_strided((32768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_47 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_48 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    view_119 = rand_strided((32768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    add_60 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_20 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_72 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_76 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_81 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_85 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_91 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_95 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_99 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_104 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_108 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_114 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_118 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_122 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_127 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_131 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_137 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_141 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_145 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_150 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_154 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_160 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_164 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_168 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_173 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_177 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_183 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_187 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_191 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_196 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_200 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_206 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_210 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_214 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_219 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_223 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_229 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_233 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_237 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_242 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_246 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_252 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_256 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_260 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_265 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_269 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_275 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_279 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_283 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_288 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_292 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_298 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((32, 1024, 256), (262144, 256, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_2, primals_3, primals_5, primals_8, primals_12, primals_15, primals_19, primals_22, primals_26, primals_29, primals_33, primals_36, primals_40, primals_43, primals_47, primals_50, primals_54, primals_57, primals_61, primals_64, primals_68, primals_71, primals_75, embedding, rsqrt, view_1, select_2, convert_element_type_8, convert_element_type_9, where, getitem, getitem_1, getitem_2, getitem_3, mm_1, rsqrt_1, view_7, mm_2, mm_3, view_11, add_6, rsqrt_2, view_13, select_5, convert_element_type_30, convert_element_type_31, getitem_4, getitem_5, getitem_6, getitem_7, add_10, rsqrt_3, view_19, mm_7, mm_8, view_23, add_12, rsqrt_4, view_25, select_8, convert_element_type_52, convert_element_type_53, getitem_8, getitem_9, getitem_10, getitem_11, add_16, rsqrt_5, view_31, mm_12, mm_13, view_35, add_18, rsqrt_6, view_37, select_11, convert_element_type_74, convert_element_type_75, getitem_12, getitem_13, getitem_14, getitem_15, add_22, rsqrt_7, view_43, mm_17, mm_18, view_47, add_24, rsqrt_8, view_49, select_14, convert_element_type_96, convert_element_type_97, getitem_16, getitem_17, getitem_18, getitem_19, add_28, rsqrt_9, view_55, mm_22, mm_23, view_59, add_30, rsqrt_10, view_61, select_17, convert_element_type_118, convert_element_type_119, getitem_20, getitem_21, getitem_22, getitem_23, add_34, rsqrt_11, view_67, mm_27, mm_28, view_71, add_36, rsqrt_12, view_73, select_20, convert_element_type_140, convert_element_type_141, getitem_24, getitem_25, getitem_26, getitem_27, add_40, rsqrt_13, view_79, mm_32, mm_33, view_83, add_42, rsqrt_14, view_85, select_23, convert_element_type_162, convert_element_type_163, getitem_28, getitem_29, getitem_30, getitem_31, add_46, rsqrt_15, view_91, mm_37, mm_38, view_95, add_48, rsqrt_16, view_97, select_26, convert_element_type_184, convert_element_type_185, getitem_32, getitem_33, getitem_34, getitem_35, add_52, rsqrt_17, view_103, mm_42, mm_43, view_107, add_54, rsqrt_18, view_109, select_29, convert_element_type_206, convert_element_type_207, getitem_36, getitem_37, getitem_38, getitem_39, add_58, rsqrt_19, view_115, mm_47, mm_48, view_119, add_60, rsqrt_20, permute_72, permute_76, permute_81, permute_85, permute_91, permute_95, permute_99, permute_104, permute_108, permute_114, permute_118, permute_122, permute_127, permute_131, permute_137, permute_141, permute_145, permute_150, permute_154, permute_160, permute_164, permute_168, permute_173, permute_177, permute_183, permute_187, permute_191, permute_196, permute_200, permute_206, permute_210, permute_214, permute_219, permute_223, permute_229, permute_233, permute_237, permute_242, permute_246, permute_252, permute_256, permute_260, permute_265, permute_269, permute_275, permute_279, permute_283, permute_288, permute_292, permute_298, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
