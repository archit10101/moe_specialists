# AOT ID: ['6_forward']
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/7q/c7qvqoneixb6k72ohuf6bmfjp36aldllyyeydb2aefobu72slrpw.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear => convert_element_type_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/dr/cdrmxirvmb46vwbocyax75tt62ys2iuh2x3gemxvarhra5xvhfeg.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear => convert_element_type, permute
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.bfloat16), kwargs = {})
#   %permute : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19316736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/wv/cwvedztsf7s2aujepfdu4tpqf7cpmvhkxnw4ns5etaaqjm5tozep.py
# Topologically Sorted Source Nodes: [lg, lse, gather_1, sub_1, mul_1, sum_2], Original ATen: [aten._to_copy, aten.logsumexp, aten.gather, aten.sub, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   gather_1 => gather_1
#   lg => convert_element_type_4
#   lse => abs_1, add, amax, eq, exp, full_default, log, sub, sum_1, where
#   mul_1 => mul_1
#   sub_1 => sub_2
#   sum_2 => sum_3
# Graph fragment:
#   %convert_element_type_4 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %amax : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_4, [-1], True), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax,), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_1, inf), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %amax), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_4, %where), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1]), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %squeeze), kwargs = {})
#   %gather_1 : [num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%convert_element_type_4, 1, %primals_4), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%gather_1, %unsqueeze), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_5, %sub_2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [-1]), kwargs = {dtype: torch.float32})
triton_red_fused__to_copy_gather_logsumexp_mul_sub_sum_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_gather_logsumexp_mul_sub_sum_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 2, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 50304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (50304*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = triton_helpers.maximum(_tmp3, tmp2)
        _tmp3 = tl.where(rmask, tmp4, _tmp3)
    tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, None)
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (50304*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl_math.abs(tmp3)
        tmp8 = float("inf")
        tmp9 = tmp7 == tmp8
        tmp10 = 0.0
        tmp11 = tl.where(tmp9, tmp10, tmp3)
        tmp12 = tmp6 - tmp11
        tmp13 = tl_math.exp(tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp15, None)
    tmp17 = tl.load(in_ptr1 + (4*x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr2 + (4*x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr1 + (1 + (4*x0)), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr2 + (1 + (4*x0)), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr1 + (2 + (4*x0)), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr2 + (2 + (4*x0)), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr1 + (3 + (4*x0)), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr2 + (3 + (4*x0)), None, eviction_policy='evict_last')
    tmp19 = tl.full([XBLOCK, 1], 50304, tl.int32)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp18)
    tl.device_assert((0 <= tmp22) & (tmp22 < 50304), "index out of bounds: 0 <= tmp22 < 50304")
    tmp24 = tl.load(in_ptr0 + (tmp22 + (50304*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tl_math.log(tmp15)
    tmp27 = tl_math.abs(tmp3)
    tmp28 = float("inf")
    tmp29 = tmp27 == tmp28
    tmp30 = 0.0
    tmp31 = tl.where(tmp29, tmp30, tmp3)
    tmp32 = tmp26 + tmp31
    tmp33 = tmp25 - tmp32
    tmp34 = tmp17 * tmp33
    tmp37 = tmp36 + tmp19
    tmp38 = tmp36 < 0
    tmp39 = tl.where(tmp38, tmp37, tmp36)
    tl.device_assert((0 <= tmp39) & (tmp39 < 50304), "index out of bounds: 0 <= tmp39 < 50304")
    tmp41 = tl.load(in_ptr0 + (tmp39 + (50304*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp42 - tmp32
    tmp44 = tmp35 * tmp43
    tmp45 = tmp34 + tmp44
    tmp48 = tmp47 + tmp19
    tmp49 = tmp47 < 0
    tmp50 = tl.where(tmp49, tmp48, tmp47)
    tl.device_assert((0 <= tmp50) & (tmp50 < 50304), "index out of bounds: 0 <= tmp50 < 50304")
    tmp52 = tl.load(in_ptr0 + (tmp50 + (50304*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp53 - tmp32
    tmp55 = tmp46 * tmp54
    tmp56 = tmp45 + tmp55
    tmp59 = tmp58 + tmp19
    tmp60 = tmp58 < 0
    tmp61 = tl.where(tmp60, tmp59, tmp58)
    tl.device_assert((0 <= tmp61) & (tmp61 < 50304), "index out of bounds: 0 <= tmp61 < 50304")
    tmp63 = tl.load(in_ptr0 + (tmp61 + (50304*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp64 - tmp32
    tmp66 = tmp57 * tmp65
    tmp67 = tmp56 + tmp66
    tl.store(out_ptr2 + (x0), tmp67, None)
    tl.store(out_ptr3 + (x0), tmp32, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/67/c673c3u3gowpvjlbj5het3tljrmmzgtzwoa2fvt75dpy3cedkl7p.py
# Topologically Sorted Source Nodes: [lse, valid, clamp, sub, mul, ce, kd, mul_2, sum_3], Original ATen: [aten.logsumexp, aten.ge, aten.clamp, aten.sub, aten.mul, aten.sum, aten.neg]
# Source node to ATen node mapping:
#   ce => sum_2
#   clamp => clamp_min
#   kd => neg
#   lse => add, log
#   mul => mul
#   mul_2 => mul_2
#   sub => sub_1
#   sum_3 => sum_4
#   valid => ge
# Graph fragment:
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %squeeze), kwargs = {})
#   %ge : [num_users=3] = call_function[target=torch.ops.aten.ge.Scalar](args = (%primals_3, 0), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%primals_3, 0), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %squeeze_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %ge), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul,), kwargs = {dtype: torch.float32})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sum_3,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %ge), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_2,), kwargs = {dtype: torch.float32})
triton_red_fused_clamp_ge_logsumexp_mul_neg_sub_sum_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[1, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: '*fp32', 5: '*i1', 6: '*i64', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {9: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_clamp_ge_logsumexp_mul_neg_sub_sum_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp24 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr4 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = triton_helpers.maximum(tmp0, tmp1)
        tmp5 = tl_math.log(tmp4)
        tmp7 = tl_math.abs(tmp6)
        tmp8 = float("inf")
        tmp9 = tmp7 == tmp8
        tmp10 = 0.0
        tmp11 = tl.where(tmp9, tmp10, tmp6)
        tmp12 = tmp5 + tmp11
        tmp13 = tl.full([XBLOCK, RBLOCK], 50304, tl.int32)
        tmp14 = tmp3 + tmp13
        tmp15 = tmp3 < 0
        tmp16 = tl.where(tmp15, tmp14, tmp3)
        tl.device_assert(((0 <= tmp16) & (tmp16 < 50304)) | ~(rmask), "index out of bounds: 0 <= tmp16 < 50304")
        tmp18 = tl.load(in_ptr3 + (tmp16 + (50304*r0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp12 - tmp19
        tmp21 = tmp2.to(tl.float32)
        tmp22 = tmp20 * tmp21
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(rmask, tmp25, _tmp24)
        tmp27 = -tmp26
        tmp28 = tmp27 * tmp21
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
        tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp2, rmask)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp3, rmask)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp24, None)
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    tl.store(out_ptr3 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp30, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (50304, 384), (384, 1))
    assert_size_stride(primals_2, (8192, 384), (384, 1))
    assert_size_stride(primals_3, (8192, ), (1, ))
    assert_size_stride(primals_4, (8192, 4), (4, 1))
    assert_size_stride(primals_5, (8192, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8192, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(primals_2, buf0, 3145728, grid=grid(3145728), stream=stream0)
        del primals_2
        buf1 = empty_strided_cuda((384, 50304), (1, 384), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_1, buf1, 19316736, grid=grid(19316736), stream=stream0)
        del primals_1
        buf2 = empty_strided_cuda((8192, 50304), (50304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, buf1, out=buf2)
        buf3 = empty_strided_cuda((8192, 1), (1, 8192), torch.float32)
        buf4 = empty_strided_cuda((8192, ), (1, ), torch.float32)
        buf8 = empty_strided_cuda((8192, ), (1, ), torch.float32)
        buf10 = empty_strided_cuda((8192, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [lg, lse, gather_1, sub_1, mul_1, sum_2], Original ATen: [aten._to_copy, aten.logsumexp, aten.gather, aten.sub, aten.mul, aten.sum]
        triton_red_fused__to_copy_gather_logsumexp_mul_sub_sum_2.run(buf2, primals_5, primals_4, buf3, buf4, buf8, buf10, 8192, 50304, grid=grid(8192), stream=stream0)
        buf5 = empty_strided_cuda((8192, ), (1, ), torch.bool)
        buf6 = empty_strided_cuda((8192, ), (1, ), torch.int64)
        buf7 = empty_strided_cuda((), (), torch.float32)
        buf9 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [lse, valid, clamp, sub, mul, ce, kd, mul_2, sum_3], Original ATen: [aten.logsumexp, aten.ge, aten.clamp, aten.sub, aten.mul, aten.sum, aten.neg]
        triton_red_fused_clamp_ge_logsumexp_mul_neg_sub_sum_3.run(primals_3, buf4, buf3, buf2, buf8, buf5, buf6, buf7, buf9, 1, 8192, grid=grid(1), stream=stream0)
        del buf3
        del buf4
        del buf8
        del primals_3
    return (buf7, buf9, primals_4, primals_5, buf0, buf2, buf5, reinterpret_tensor(buf6, (8192, 1), (1, 1), 0), reinterpret_tensor(buf10, (8192, 1), (1, 1), 0), reinterpret_tensor(buf1, (50304, 384), (384, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((50304, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((8192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_4 = rand_strided((8192, 4), (4, 1), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((8192, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
