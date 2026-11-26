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


# kernel path: /home/zapan4/moe-research/.cache/inductor/5e/c5ecdi4aj5peyuhwywll6ps5fmxjzmdnnppkdzum653jwkm6f5d2.py
# Topologically Sorted Source Nodes: [x, pow_2, mean, add, rsqrt, y, mul_2, linear], Original ATen: [aten.embedding, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   add => add_1
#   linear => convert_element_type_5
#   mean => mean
#   mul_2 => mul_4
#   pow_2 => pow_2
#   rsqrt => rsqrt
#   x => embedding
#   y => mul_3
# Graph fragment:
#   %embedding : [num_users=4] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_1, %primals_2), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%embedding, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%embedding, %rsqrt), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %primals_5), kwargs = {})
#   %convert_element_type_5 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
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
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp1 = tl.full([RBLOCK], 50304, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 50304), "index out of bounds: 0 <= tmp4 < 50304")
    tmp6 = tl.load(in_ptr1 + (r1 + (256*tmp4)), None)
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp11 = 256.0
    tmp12 = tmp10 / tmp11
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp6 * tmp15
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr0 + (r1 + (256*x0)), tmp6, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp15, None)
    tl.store(out_ptr1 + (r1 + (256*x0)), tmp19, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/qn/cqnfdhs6sff4tdtn3bncwn5zi54lphknaimotrucmu3qxhpmpbno.py
# Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear => convert_element_type_4, permute
# Graph fragment:
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_6, torch.bfloat16), kwargs = {})
#   %permute : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_4, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/jk/cjklarhv4usjl3orc3pm6dgn6ciyjdiyn4mahbrl7l3wcrs4bz6s.py
# Topologically Sorted Source Nodes: [rot, mul_3, mul_4, q_1, rot_1, mul_5, mul_6, k_1, y_1], Original ATen: [aten.cat, aten.mul, aten.add, aten._to_copy]
# Source node to ATen node mapping:
#   k_1 => add_3
#   mul_3 => mul_5
#   mul_4 => mul_6
#   mul_5 => mul_7
#   mul_6 => mul_8
#   q_1 => add_2
#   rot => cat
#   rot_1 => cat_1
#   y_1 => convert_element_type_8, convert_element_type_9
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_2], -1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %unsqueeze_2), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_3), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %mul_6), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_1, %slice_4], -1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %unsqueeze_2), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_1, %unsqueeze_3), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %mul_8), kwargs = {})
#   %convert_element_type_8 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.bfloat16), kwargs = {})
#   %convert_element_type_9 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_add_cat_mul_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*i64', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_cat_mul_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 4
    x2 = (xindex // 256)
    x4 = xindex
    x5 = xindex % 256
    tmp15 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (x5 + (768*x2)), None).to(tl.float32)
    tmp46 = tl.load(in_ptr0 + (256 + x5 + (768*x2)), None).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32 + (64*x1) + (768*x2) + x0), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = -tmp5
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = tmp0 >= tmp3
    tmp10 = tl.full([1], 64, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tl.load(in_ptr0 + ((64*x1) + (768*x2) + ((-32) + x0)), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tl.where(tmp4, tmp8, tmp12)
    tmp14 = tmp13.to(tl.float32)
    tmp16 = tmp15.to(tl.float64)
    tmp17 = 2*(x0 % 32)
    tmp18 = tmp17.to(tl.float64)
    tmp19 = tl.full([1], 0.015625, tl.float64)
    tmp20 = tmp18 * tmp19
    tmp21 = tl.full([1], 10000.0, tl.float64)
    tmp22 = libdevice.pow(tmp21, tmp20)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = tl.full([1], 1.0, tl.float64)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp16 * tmp26
    tmp28 = libdevice.sin(tmp27)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp14 * tmp29
    tmp32 = tmp31.to(tl.float32)
    tmp33 = libdevice.cos(tmp27)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 * tmp34
    tmp36 = tmp35 + tmp30
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tl.load(in_ptr0 + (288 + (64*x1) + (768*x2) + x0), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = -tmp38
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp4, tmp39, tmp40)
    tmp42 = tl.load(in_ptr0 + (256 + (64*x1) + (768*x2) + ((-32) + x0)), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp43 = tl.where(tmp4, tmp41, tmp42)
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp44 * tmp29
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp47 * tmp34
    tmp49 = tmp48 + tmp45
    tmp50 = tmp49.to(tl.float32)
    tl.store(out_ptr1 + (x4), tmp37, None)
    tl.store(out_ptr3 + (x4), tmp50, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ip/cipl53egul6jb4njdm5at52l4iyy7j3ptieauye24k3lc4pygqfa.py
# Topologically Sorted Source Nodes: [y_1], Original ATen: [aten.scalar_tensor, aten.logical_not, aten.where]
# Source node to ATen node mapping:
#   y_1 => full_default_1, full_default_2, logical_not, where
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %logical_not : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%unsqueeze_11,), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%logical_not, %full_default_2, %full_default_1), kwargs = {})
triton_poi_fused_logical_not_scalar_tensor_where_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_logical_not_scalar_tensor_where_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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
    tmp17 = tmp16 == 0
    tmp18 = float("-inf")
    tmp19 = 0.0
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tl.store(out_ptr0 + (x4), tmp20, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ia/cia3speqn65ndwffkg6nvsig4aivag3xy6mm7iwdv2sddzo5mci2.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear_1 => convert_element_type_10, permute_3
# Graph fragment:
#   %convert_element_type_10 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_7, torch.bfloat16), kwargs = {})
#   %permute_3 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_10, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/t5/ct55vasfsv5s6s6krm3sffujpwtofealsrgdxfaf5axbu2jscwey.py
# Topologically Sorted Source Nodes: [x_1, pow_3, mean_1, add_4, rsqrt_1, y_2, mul_8, linear_2], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   add_4 => add_5
#   linear_2 => convert_element_type_14
#   mean_1 => mean_1
#   mul_8 => mul_10
#   pow_3 => pow_3
#   rsqrt_1 => rsqrt_1
#   x_1 => add_4
#   y_2 => mul_9
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_6), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_4, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, %rsqrt_1), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %primals_8), kwargs = {})
#   %convert_element_type_14 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*bf16', 3: '*fp32', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), None).to(tl.float32)
    tmp14 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = 256.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp3 * tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, None)
    tl.store(out_ptr0 + (r1 + (256*x0)), tmp16, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/zv/czvzosm7gihgsujgplasszfvce5wftvgzis2zh2oacy7q5yxwfwo.py
# Topologically Sorted Source Nodes: [silu, mul_9], Original ATen: [aten.silu, aten.mul]
# Source node to ATen node mapping:
#   mul_9 => mul_12
#   silu => convert_element_type_17, convert_element_type_18, mul_11, sigmoid
# Graph fragment:
#   %convert_element_type_17 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_8, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_17,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_17, %sigmoid), kwargs = {})
#   %convert_element_type_18 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11, torch.bfloat16), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_18, %view_10), kwargs = {})
triton_poi_fused_mul_silu_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/6p/c6p2b55yanzevknnoao4xrc5x6t565sftrz7xa3f72jsml7uba2p.py
# Topologically Sorted Source Nodes: [x_1, x_2, pow_4, mean_2, add_6, rsqrt_2, y_3, mul_11, linear_5], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   add_6 => add_7
#   linear_5 => convert_element_type_27
#   mean_2 => mean_2
#   mul_11 => mul_14
#   pow_4 => pow_4
#   rsqrt_2 => rsqrt_2
#   x_1 => add_4
#   x_2 => add_6
#   y_3 => mul_13
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_6), kwargs = {})
#   %add_6 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_12), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_6, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-06), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, %rsqrt_2), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %primals_12), kwargs = {})
#   %convert_element_type_27 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_14, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_7', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1 + (256*x0)), None).to(tl.float32)
    tmp17 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp11 = 256.0
    tmp12 = tmp10 / tmp11
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp6 * tmp15
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.store(out_ptr0 + (r1 + (256*x0)), tmp6, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp15, None)
    tl.store(out_ptr1 + (r1 + (256*x0)), tmp19, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/vc/cvcxovyafwj5ojp6t6beuflxv52p2nz5pc45wcxef3uz57upjy7o.py
# Topologically Sorted Source Nodes: [x_3, pow_5, mean_3, add_10, rsqrt_3, y_5, mul_17, linear_7], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   add_10 => add_11
#   linear_7 => convert_element_type_36
#   mean_3 => mean_3
#   mul_17 => mul_20
#   pow_5 => pow_5
#   rsqrt_3 => rsqrt_3
#   x_3 => add_10
#   y_5 => mul_19
# Graph fragment:
#   %add_10 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %view_18), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_10, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [-1], True), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-06), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %rsqrt_3), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %primals_15), kwargs = {})
#   %convert_element_type_36 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_20, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), None).to(tl.float32)
    tmp14 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = 256.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp3 * tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (r1 + (256*x0)), tmp3, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, None)
    tl.store(out_ptr1 + (r1 + (256*x0)), tmp16, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/rn/crnfzclxvvmoidredefhlcbx62yhx7nre45ttxy2yaeh3lo5fbbw.py
# Topologically Sorted Source Nodes: [x_20, pow_22, mean_20, add_60, rsqrt_20, y_30, mul_92], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_60 => add_61
#   mean_20 => mean_20
#   mul_92 => mul_104
#   pow_22 => pow_22
#   rsqrt_20 => rsqrt_20
#   x_20 => add_60
#   y_30 => mul_103
# Graph fragment:
#   %add_60 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_58, %view_120), kwargs = {})
#   %pow_22 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_60, 2), kwargs = {})
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_22, [-1], True), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_20, 1e-06), kwargs = {})
#   %rsqrt_20 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_61,), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_60, %rsqrt_20), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_103, %primals_75), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_9', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (256*x0)), None).to(tl.float32)
    tmp14 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = 256.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-06
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp3 * tmp12
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + (r1 + (256*x0)), tmp3, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, None)
    tl.store(out_ptr1 + (r1 + (256*x0)), tmp15, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75 = args
    args.clear()
    assert_size_stride(primals_1, (50304, 256), (256, 1))
    assert_size_stride(primals_2, (32, 1024), (1024, 1))
    assert_size_stride(primals_3, (32, 1024), (1024, 1))
    assert_size_stride(primals_4, (32, 1024), (1024, 1))
    assert_size_stride(primals_5, (256, ), (1, ))
    assert_size_stride(primals_6, (768, 256), (256, 1))
    assert_size_stride(primals_7, (256, 256), (256, 1))
    assert_size_stride(primals_8, (256, ), (1, ))
    assert_size_stride(primals_9, (768, 256), (256, 1))
    assert_size_stride(primals_10, (768, 256), (256, 1))
    assert_size_stride(primals_11, (256, 768), (768, 1))
    assert_size_stride(primals_12, (256, ), (1, ))
    assert_size_stride(primals_13, (768, 256), (256, 1))
    assert_size_stride(primals_14, (256, 256), (256, 1))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (768, 256), (256, 1))
    assert_size_stride(primals_17, (768, 256), (256, 1))
    assert_size_stride(primals_18, (256, 768), (768, 1))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_20, (768, 256), (256, 1))
    assert_size_stride(primals_21, (256, 256), (256, 1))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_23, (768, 256), (256, 1))
    assert_size_stride(primals_24, (768, 256), (256, 1))
    assert_size_stride(primals_25, (256, 768), (768, 1))
    assert_size_stride(primals_26, (256, ), (1, ))
    assert_size_stride(primals_27, (768, 256), (256, 1))
    assert_size_stride(primals_28, (256, 256), (256, 1))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_30, (768, 256), (256, 1))
    assert_size_stride(primals_31, (768, 256), (256, 1))
    assert_size_stride(primals_32, (256, 768), (768, 1))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_34, (768, 256), (256, 1))
    assert_size_stride(primals_35, (256, 256), (256, 1))
    assert_size_stride(primals_36, (256, ), (1, ))
    assert_size_stride(primals_37, (768, 256), (256, 1))
    assert_size_stride(primals_38, (768, 256), (256, 1))
    assert_size_stride(primals_39, (256, 768), (768, 1))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_41, (768, 256), (256, 1))
    assert_size_stride(primals_42, (256, 256), (256, 1))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_44, (768, 256), (256, 1))
    assert_size_stride(primals_45, (768, 256), (256, 1))
    assert_size_stride(primals_46, (256, 768), (768, 1))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_48, (768, 256), (256, 1))
    assert_size_stride(primals_49, (256, 256), (256, 1))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_51, (768, 256), (256, 1))
    assert_size_stride(primals_52, (768, 256), (256, 1))
    assert_size_stride(primals_53, (256, 768), (768, 1))
    assert_size_stride(primals_54, (256, ), (1, ))
    assert_size_stride(primals_55, (768, 256), (256, 1))
    assert_size_stride(primals_56, (256, 256), (256, 1))
    assert_size_stride(primals_57, (256, ), (1, ))
    assert_size_stride(primals_58, (768, 256), (256, 1))
    assert_size_stride(primals_59, (768, 256), (256, 1))
    assert_size_stride(primals_60, (256, 768), (768, 1))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (768, 256), (256, 1))
    assert_size_stride(primals_63, (256, 256), (256, 1))
    assert_size_stride(primals_64, (256, ), (1, ))
    assert_size_stride(primals_65, (768, 256), (256, 1))
    assert_size_stride(primals_66, (768, 256), (256, 1))
    assert_size_stride(primals_67, (256, 768), (768, 1))
    assert_size_stride(primals_68, (256, ), (1, ))
    assert_size_stride(primals_69, (768, 256), (256, 1))
    assert_size_stride(primals_70, (256, 256), (256, 1))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_72, (768, 256), (256, 1))
    assert_size_stride(primals_73, (768, 256), (256, 1))
    assert_size_stride(primals_74, (256, 768), (768, 1))
    assert_size_stride(primals_75, (256, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf1 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf2 = reinterpret_tensor(buf1, (32, 1024, 1), (1024, 1, 1), 0); del buf1  # reuse
        buf4 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x, pow_2, mean, add, rsqrt, y, mul_2, linear], Original ATen: [aten.embedding, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0.run(buf2, primals_2, primals_1, primals_5, buf0, buf4, 32768, 256, grid=grid(32768), stream=stream0)
        del primals_1
        buf3 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_6, buf3, 196608, grid=grid(196608), stream=stream0)
        del primals_6
        buf5 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (32768, 256), (256, 1), 0), buf3, out=buf5)
        buf9 = empty_strided_cuda((32, 4, 1024, 64), (262144, 64, 256, 1), torch.bfloat16)
        buf8 = empty_strided_cuda((32, 4, 1024, 64), (262144, 64, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [rot, mul_3, mul_4, q_1, rot_1, mul_5, mul_6, k_1, y_1], Original ATen: [aten.cat, aten.mul, aten.add, aten._to_copy]
        triton_poi_fused__to_copy_add_cat_mul_2.run(buf5, primals_3, buf9, buf8, 8388608, grid=grid(8388608), stream=stream0)
        buf10 = empty_strided_cuda((32, 1, 1024, 1024), (1048576, 1048576, 1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [y_1], Original ATen: [aten.scalar_tensor, aten.logical_not, aten.where]
        triton_poi_fused_logical_not_scalar_tensor_where_3.run(primals_4, buf10, 33554432, grid=grid(33554432), stream=stream0)
        del primals_4
        # Topologically Sorted Source Nodes: [y_1], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf9, buf8, reinterpret_tensor(buf5, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), reinterpret_tensor(buf10, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), True)
        buf12 = buf11[0]
        buf13 = buf11[1]
        buf14 = buf11[2]
        buf15 = buf11[3]
        del buf11
        buf16 = empty_strided_cuda((256, 256), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_4.run(primals_7, buf16, 65536, grid=grid(65536), stream=stream0)
        del primals_7
        buf17 = empty_strided_cuda((32768, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (32768, 256), (256, 1), 0), buf16, out=buf17)
        buf18 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf19 = reinterpret_tensor(buf18, (32, 1024, 1), (1024, 1, 1), 0); del buf18  # reuse
        buf21 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_1, pow_3, mean_1, add_4, rsqrt_1, y_2, mul_8, linear_2], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf19, buf0, buf17, primals_8, buf21, 32768, 256, grid=grid(32768), stream=stream0)
        buf20 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_9, buf20, 196608, grid=grid(196608), stream=stream0)
        del primals_9
        buf22 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (32768, 256), (256, 1), 0), buf20, out=buf22)
        buf23 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_10, buf23, 196608, grid=grid(196608), stream=stream0)
        del primals_10
        buf24 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf21, (32768, 256), (256, 1), 0), buf23, out=buf24)
        buf25 = empty_strided_cuda((768, 256), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_11, buf25, 196608, grid=grid(196608), stream=stream0)
        del primals_11
        buf26 = empty_strided_cuda((32, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu, mul_9], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf22, buf24, buf26, 25165824, grid=grid(25165824), stream=stream0)
        buf27 = empty_strided_cuda((32768, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (32768, 768), (768, 1), 0), buf25, out=buf27)
        buf28 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf29 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf30 = reinterpret_tensor(buf29, (32, 1024, 1), (1024, 1, 1), 0); del buf29  # reuse
        buf32 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_1, x_2, pow_4, mean_2, add_6, rsqrt_2, y_3, mul_11, linear_5], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_7.run(buf30, buf0, buf17, buf27, primals_12, buf28, buf32, 32768, 256, grid=grid(32768), stream=stream0)
        buf31 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_13, buf31, 196608, grid=grid(196608), stream=stream0)
        del primals_13
        buf33 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (32768, 256), (256, 1), 0), buf31, out=buf33)
        buf37 = reinterpret_tensor(buf27, (32, 4, 1024, 64), (262144, 64, 256, 1), 0); del buf27  # reuse
        buf36 = empty_strided_cuda((32, 4, 1024, 64), (262144, 64, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [rot_2, mul_12, mul_13, q_3, rot_3, mul_14, mul_15, k_3, y_4], Original ATen: [aten.cat, aten.mul, aten.add, aten._to_copy]
        triton_poi_fused__to_copy_add_cat_mul_2.run(buf33, primals_3, buf37, buf36, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [y_4], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf38 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf37, buf36, reinterpret_tensor(buf33, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), reinterpret_tensor(buf10, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), True)
        buf39 = buf38[0]
        buf40 = buf38[1]
        buf41 = buf38[2]
        buf42 = buf38[3]
        del buf38
        buf43 = empty_strided_cuda((256, 256), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_4.run(primals_14, buf43, 65536, grid=grid(65536), stream=stream0)
        del primals_14
        buf44 = empty_strided_cuda((32768, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (32768, 256), (256, 1), 0), buf43, out=buf44)
        buf45 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf46 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf47 = reinterpret_tensor(buf46, (32, 1024, 1), (1024, 1, 1), 0); del buf46  # reuse
        buf49 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_3, pow_5, mean_3, add_10, rsqrt_3, y_5, mul_17, linear_7], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf47, buf28, buf44, primals_15, buf45, buf49, 32768, 256, grid=grid(32768), stream=stream0)
        buf48 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_16, buf48, 196608, grid=grid(196608), stream=stream0)
        del primals_16
        buf50 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (32768, 256), (256, 1), 0), buf48, out=buf50)
        buf51 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_17, buf51, 196608, grid=grid(196608), stream=stream0)
        del primals_17
        buf52 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (32768, 256), (256, 1), 0), buf51, out=buf52)
        buf53 = empty_strided_cuda((768, 256), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_18, buf53, 196608, grid=grid(196608), stream=stream0)
        del primals_18
        buf54 = empty_strided_cuda((32, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu_1, mul_18], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf50, buf52, buf54, 25165824, grid=grid(25165824), stream=stream0)
        buf55 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf54, (32768, 768), (768, 1), 0), buf53, out=buf55)
        buf56 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf57 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf58 = reinterpret_tensor(buf57, (32, 1024, 1), (1024, 1, 1), 0); del buf57  # reuse
        buf60 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_4, pow_6, mean_4, add_12, rsqrt_4, y_6, mul_20, linear_10], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf58, buf45, buf55, primals_19, buf56, buf60, 32768, 256, grid=grid(32768), stream=stream0)
        buf59 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_20, buf59, 196608, grid=grid(196608), stream=stream0)
        del primals_20
        buf61 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf60, (32768, 256), (256, 1), 0), buf59, out=buf61)
        buf65 = reinterpret_tensor(buf55, (32, 4, 1024, 64), (262144, 64, 256, 1), 0); del buf55  # reuse
        buf64 = empty_strided_cuda((32, 4, 1024, 64), (262144, 64, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [rot_4, mul_21, mul_22, q_5, rot_5, mul_23, mul_24, k_5, y_7], Original ATen: [aten.cat, aten.mul, aten.add, aten._to_copy]
        triton_poi_fused__to_copy_add_cat_mul_2.run(buf61, primals_3, buf65, buf64, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [y_7], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf66 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf65, buf64, reinterpret_tensor(buf61, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), reinterpret_tensor(buf10, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), True)
        buf67 = buf66[0]
        buf68 = buf66[1]
        buf69 = buf66[2]
        buf70 = buf66[3]
        del buf66
        buf71 = empty_strided_cuda((256, 256), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_4.run(primals_21, buf71, 65536, grid=grid(65536), stream=stream0)
        del primals_21
        buf72 = empty_strided_cuda((32768, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (32768, 256), (256, 1), 0), buf71, out=buf72)
        buf73 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf74 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf75 = reinterpret_tensor(buf74, (32, 1024, 1), (1024, 1, 1), 0); del buf74  # reuse
        buf77 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_5, pow_7, mean_5, add_16, rsqrt_5, y_8, mul_26, linear_12], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf75, buf56, buf72, primals_22, buf73, buf77, 32768, 256, grid=grid(32768), stream=stream0)
        buf76 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_23, buf76, 196608, grid=grid(196608), stream=stream0)
        del primals_23
        buf78 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (32768, 256), (256, 1), 0), buf76, out=buf78)
        buf79 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_24, buf79, 196608, grid=grid(196608), stream=stream0)
        del primals_24
        buf80 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (32768, 256), (256, 1), 0), buf79, out=buf80)
        buf81 = empty_strided_cuda((768, 256), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_25, buf81, 196608, grid=grid(196608), stream=stream0)
        del primals_25
        buf82 = empty_strided_cuda((32, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu_2, mul_27], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf78, buf80, buf82, 25165824, grid=grid(25165824), stream=stream0)
        buf83 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (32768, 768), (768, 1), 0), buf81, out=buf83)
        buf84 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf85 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf86 = reinterpret_tensor(buf85, (32, 1024, 1), (1024, 1, 1), 0); del buf85  # reuse
        buf88 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_6, pow_8, mean_6, add_18, rsqrt_6, y_9, mul_29, linear_15], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf86, buf73, buf83, primals_26, buf84, buf88, 32768, 256, grid=grid(32768), stream=stream0)
        buf87 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_27, buf87, 196608, grid=grid(196608), stream=stream0)
        del primals_27
        buf89 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (32768, 256), (256, 1), 0), buf87, out=buf89)
        buf93 = reinterpret_tensor(buf83, (32, 4, 1024, 64), (262144, 64, 256, 1), 0); del buf83  # reuse
        buf92 = empty_strided_cuda((32, 4, 1024, 64), (262144, 64, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [rot_6, mul_30, mul_31, q_7, rot_7, mul_32, mul_33, k_7, y_10], Original ATen: [aten.cat, aten.mul, aten.add, aten._to_copy]
        triton_poi_fused__to_copy_add_cat_mul_2.run(buf89, primals_3, buf93, buf92, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [y_10], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf94 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf93, buf92, reinterpret_tensor(buf89, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), reinterpret_tensor(buf10, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), True)
        buf95 = buf94[0]
        buf96 = buf94[1]
        buf97 = buf94[2]
        buf98 = buf94[3]
        del buf94
        buf99 = empty_strided_cuda((256, 256), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_4.run(primals_28, buf99, 65536, grid=grid(65536), stream=stream0)
        del primals_28
        buf100 = empty_strided_cuda((32768, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf95, (32768, 256), (256, 1), 0), buf99, out=buf100)
        buf101 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf102 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf103 = reinterpret_tensor(buf102, (32, 1024, 1), (1024, 1, 1), 0); del buf102  # reuse
        buf105 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_7, pow_9, mean_7, add_22, rsqrt_7, y_11, mul_35, linear_17], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf103, buf84, buf100, primals_29, buf101, buf105, 32768, 256, grid=grid(32768), stream=stream0)
        buf104 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_30, buf104, 196608, grid=grid(196608), stream=stream0)
        del primals_30
        buf106 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (32768, 256), (256, 1), 0), buf104, out=buf106)
        buf107 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_31, buf107, 196608, grid=grid(196608), stream=stream0)
        del primals_31
        buf108 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (32768, 256), (256, 1), 0), buf107, out=buf108)
        buf109 = empty_strided_cuda((768, 256), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_32, buf109, 196608, grid=grid(196608), stream=stream0)
        del primals_32
        buf110 = empty_strided_cuda((32, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu_3, mul_36], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf106, buf108, buf110, 25165824, grid=grid(25165824), stream=stream0)
        buf111 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf110, (32768, 768), (768, 1), 0), buf109, out=buf111)
        buf112 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf113 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf114 = reinterpret_tensor(buf113, (32, 1024, 1), (1024, 1, 1), 0); del buf113  # reuse
        buf116 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_8, pow_10, mean_8, add_24, rsqrt_8, y_12, mul_38, linear_20], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf114, buf101, buf111, primals_33, buf112, buf116, 32768, 256, grid=grid(32768), stream=stream0)
        buf115 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_34, buf115, 196608, grid=grid(196608), stream=stream0)
        del primals_34
        buf117 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (32768, 256), (256, 1), 0), buf115, out=buf117)
        buf121 = reinterpret_tensor(buf111, (32, 4, 1024, 64), (262144, 64, 256, 1), 0); del buf111  # reuse
        buf120 = empty_strided_cuda((32, 4, 1024, 64), (262144, 64, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [rot_8, mul_39, mul_40, q_9, rot_9, mul_41, mul_42, k_9, y_13], Original ATen: [aten.cat, aten.mul, aten.add, aten._to_copy]
        triton_poi_fused__to_copy_add_cat_mul_2.run(buf117, primals_3, buf121, buf120, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [y_13], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf122 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf121, buf120, reinterpret_tensor(buf117, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), reinterpret_tensor(buf10, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), True)
        buf123 = buf122[0]
        buf124 = buf122[1]
        buf125 = buf122[2]
        buf126 = buf122[3]
        del buf122
        buf127 = empty_strided_cuda((256, 256), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_4.run(primals_35, buf127, 65536, grid=grid(65536), stream=stream0)
        del primals_35
        buf128 = empty_strided_cuda((32768, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (32768, 256), (256, 1), 0), buf127, out=buf128)
        buf129 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf130 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf131 = reinterpret_tensor(buf130, (32, 1024, 1), (1024, 1, 1), 0); del buf130  # reuse
        buf133 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_9, pow_11, mean_9, add_28, rsqrt_9, y_14, mul_44, linear_22], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf131, buf112, buf128, primals_36, buf129, buf133, 32768, 256, grid=grid(32768), stream=stream0)
        buf132 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_37, buf132, 196608, grid=grid(196608), stream=stream0)
        del primals_37
        buf134 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (32768, 256), (256, 1), 0), buf132, out=buf134)
        buf135 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_38, buf135, 196608, grid=grid(196608), stream=stream0)
        del primals_38
        buf136 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (32768, 256), (256, 1), 0), buf135, out=buf136)
        buf137 = empty_strided_cuda((768, 256), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_39, buf137, 196608, grid=grid(196608), stream=stream0)
        del primals_39
        buf138 = empty_strided_cuda((32, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu_4, mul_45], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf134, buf136, buf138, 25165824, grid=grid(25165824), stream=stream0)
        buf139 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf138, (32768, 768), (768, 1), 0), buf137, out=buf139)
        buf140 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf141 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf142 = reinterpret_tensor(buf141, (32, 1024, 1), (1024, 1, 1), 0); del buf141  # reuse
        buf144 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_10, pow_12, mean_10, add_30, rsqrt_10, y_15, mul_47, linear_25], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf142, buf129, buf139, primals_40, buf140, buf144, 32768, 256, grid=grid(32768), stream=stream0)
        buf143 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_41, buf143, 196608, grid=grid(196608), stream=stream0)
        del primals_41
        buf145 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (32768, 256), (256, 1), 0), buf143, out=buf145)
        buf149 = reinterpret_tensor(buf139, (32, 4, 1024, 64), (262144, 64, 256, 1), 0); del buf139  # reuse
        buf148 = empty_strided_cuda((32, 4, 1024, 64), (262144, 64, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [rot_10, mul_48, mul_49, q_11, rot_11, mul_50, mul_51, k_11, y_16], Original ATen: [aten.cat, aten.mul, aten.add, aten._to_copy]
        triton_poi_fused__to_copy_add_cat_mul_2.run(buf145, primals_3, buf149, buf148, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [y_16], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf150 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf149, buf148, reinterpret_tensor(buf145, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), reinterpret_tensor(buf10, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), True)
        buf151 = buf150[0]
        buf152 = buf150[1]
        buf153 = buf150[2]
        buf154 = buf150[3]
        del buf150
        buf155 = empty_strided_cuda((256, 256), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_4.run(primals_42, buf155, 65536, grid=grid(65536), stream=stream0)
        del primals_42
        buf156 = empty_strided_cuda((32768, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf151, (32768, 256), (256, 1), 0), buf155, out=buf156)
        buf157 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf158 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf159 = reinterpret_tensor(buf158, (32, 1024, 1), (1024, 1, 1), 0); del buf158  # reuse
        buf161 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_11, pow_13, mean_11, add_34, rsqrt_11, y_17, mul_53, linear_27], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf159, buf140, buf156, primals_43, buf157, buf161, 32768, 256, grid=grid(32768), stream=stream0)
        buf160 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_44, buf160, 196608, grid=grid(196608), stream=stream0)
        del primals_44
        buf162 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (32768, 256), (256, 1), 0), buf160, out=buf162)
        buf163 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_45, buf163, 196608, grid=grid(196608), stream=stream0)
        del primals_45
        buf164 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf161, (32768, 256), (256, 1), 0), buf163, out=buf164)
        buf165 = empty_strided_cuda((768, 256), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_46, buf165, 196608, grid=grid(196608), stream=stream0)
        del primals_46
        buf166 = empty_strided_cuda((32, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu_5, mul_54], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf162, buf164, buf166, 25165824, grid=grid(25165824), stream=stream0)
        buf167 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (32768, 768), (768, 1), 0), buf165, out=buf167)
        buf168 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf169 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf170 = reinterpret_tensor(buf169, (32, 1024, 1), (1024, 1, 1), 0); del buf169  # reuse
        buf172 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_12, pow_14, mean_12, add_36, rsqrt_12, y_18, mul_56, linear_30], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf170, buf157, buf167, primals_47, buf168, buf172, 32768, 256, grid=grid(32768), stream=stream0)
        buf171 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_48, buf171, 196608, grid=grid(196608), stream=stream0)
        del primals_48
        buf173 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf172, (32768, 256), (256, 1), 0), buf171, out=buf173)
        buf177 = reinterpret_tensor(buf167, (32, 4, 1024, 64), (262144, 64, 256, 1), 0); del buf167  # reuse
        buf176 = empty_strided_cuda((32, 4, 1024, 64), (262144, 64, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [rot_12, mul_57, mul_58, q_13, rot_13, mul_59, mul_60, k_13, y_19], Original ATen: [aten.cat, aten.mul, aten.add, aten._to_copy]
        triton_poi_fused__to_copy_add_cat_mul_2.run(buf173, primals_3, buf177, buf176, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [y_19], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf178 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf177, buf176, reinterpret_tensor(buf173, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), reinterpret_tensor(buf10, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), True)
        buf179 = buf178[0]
        buf180 = buf178[1]
        buf181 = buf178[2]
        buf182 = buf178[3]
        del buf178
        buf183 = empty_strided_cuda((256, 256), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_4.run(primals_49, buf183, 65536, grid=grid(65536), stream=stream0)
        del primals_49
        buf184 = empty_strided_cuda((32768, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (32768, 256), (256, 1), 0), buf183, out=buf184)
        buf185 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf186 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf187 = reinterpret_tensor(buf186, (32, 1024, 1), (1024, 1, 1), 0); del buf186  # reuse
        buf189 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_13, pow_15, mean_13, add_40, rsqrt_13, y_20, mul_62, linear_32], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf187, buf168, buf184, primals_50, buf185, buf189, 32768, 256, grid=grid(32768), stream=stream0)
        buf188 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_51, buf188, 196608, grid=grid(196608), stream=stream0)
        del primals_51
        buf190 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (32768, 256), (256, 1), 0), buf188, out=buf190)
        buf191 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_52, buf191, 196608, grid=grid(196608), stream=stream0)
        del primals_52
        buf192 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf189, (32768, 256), (256, 1), 0), buf191, out=buf192)
        buf193 = empty_strided_cuda((768, 256), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_34], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_53, buf193, 196608, grid=grid(196608), stream=stream0)
        del primals_53
        buf194 = empty_strided_cuda((32, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu_6, mul_63], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf190, buf192, buf194, 25165824, grid=grid(25165824), stream=stream0)
        buf195 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [linear_34], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf194, (32768, 768), (768, 1), 0), buf193, out=buf195)
        buf196 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf197 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf198 = reinterpret_tensor(buf197, (32, 1024, 1), (1024, 1, 1), 0); del buf197  # reuse
        buf200 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_14, pow_16, mean_14, add_42, rsqrt_14, y_21, mul_65, linear_35], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf198, buf185, buf195, primals_54, buf196, buf200, 32768, 256, grid=grid(32768), stream=stream0)
        buf199 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_55, buf199, 196608, grid=grid(196608), stream=stream0)
        del primals_55
        buf201 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (32768, 256), (256, 1), 0), buf199, out=buf201)
        buf205 = reinterpret_tensor(buf195, (32, 4, 1024, 64), (262144, 64, 256, 1), 0); del buf195  # reuse
        buf204 = empty_strided_cuda((32, 4, 1024, 64), (262144, 64, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [rot_14, mul_66, mul_67, q_15, rot_15, mul_68, mul_69, k_15, y_22], Original ATen: [aten.cat, aten.mul, aten.add, aten._to_copy]
        triton_poi_fused__to_copy_add_cat_mul_2.run(buf201, primals_3, buf205, buf204, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [y_22], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf206 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf205, buf204, reinterpret_tensor(buf201, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), reinterpret_tensor(buf10, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), True)
        buf207 = buf206[0]
        buf208 = buf206[1]
        buf209 = buf206[2]
        buf210 = buf206[3]
        del buf206
        buf211 = empty_strided_cuda((256, 256), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_4.run(primals_56, buf211, 65536, grid=grid(65536), stream=stream0)
        del primals_56
        buf212 = empty_strided_cuda((32768, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (32768, 256), (256, 1), 0), buf211, out=buf212)
        buf213 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf214 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf215 = reinterpret_tensor(buf214, (32, 1024, 1), (1024, 1, 1), 0); del buf214  # reuse
        buf217 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_15, pow_17, mean_15, add_46, rsqrt_15, y_23, mul_71, linear_37], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf215, buf196, buf212, primals_57, buf213, buf217, 32768, 256, grid=grid(32768), stream=stream0)
        buf216 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_58, buf216, 196608, grid=grid(196608), stream=stream0)
        del primals_58
        buf218 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (32768, 256), (256, 1), 0), buf216, out=buf218)
        buf219 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_59, buf219, 196608, grid=grid(196608), stream=stream0)
        del primals_59
        buf220 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (32768, 256), (256, 1), 0), buf219, out=buf220)
        buf221 = empty_strided_cuda((768, 256), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_60, buf221, 196608, grid=grid(196608), stream=stream0)
        del primals_60
        buf222 = empty_strided_cuda((32, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu_7, mul_72], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf218, buf220, buf222, 25165824, grid=grid(25165824), stream=stream0)
        buf223 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf222, (32768, 768), (768, 1), 0), buf221, out=buf223)
        buf224 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf225 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf226 = reinterpret_tensor(buf225, (32, 1024, 1), (1024, 1, 1), 0); del buf225  # reuse
        buf228 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_16, pow_18, mean_16, add_48, rsqrt_16, y_24, mul_74, linear_40], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf226, buf213, buf223, primals_61, buf224, buf228, 32768, 256, grid=grid(32768), stream=stream0)
        buf227 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_62, buf227, 196608, grid=grid(196608), stream=stream0)
        del primals_62
        buf229 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (32768, 256), (256, 1), 0), buf227, out=buf229)
        buf233 = reinterpret_tensor(buf223, (32, 4, 1024, 64), (262144, 64, 256, 1), 0); del buf223  # reuse
        buf232 = empty_strided_cuda((32, 4, 1024, 64), (262144, 64, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [rot_16, mul_75, mul_76, q_17, rot_17, mul_77, mul_78, k_17, y_25], Original ATen: [aten.cat, aten.mul, aten.add, aten._to_copy]
        triton_poi_fused__to_copy_add_cat_mul_2.run(buf229, primals_3, buf233, buf232, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [y_25], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf234 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf233, buf232, reinterpret_tensor(buf229, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), reinterpret_tensor(buf10, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), True)
        buf235 = buf234[0]
        buf236 = buf234[1]
        buf237 = buf234[2]
        buf238 = buf234[3]
        del buf234
        buf239 = empty_strided_cuda((256, 256), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_4.run(primals_63, buf239, 65536, grid=grid(65536), stream=stream0)
        del primals_63
        buf240 = empty_strided_cuda((32768, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf235, (32768, 256), (256, 1), 0), buf239, out=buf240)
        buf241 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf242 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf243 = reinterpret_tensor(buf242, (32, 1024, 1), (1024, 1, 1), 0); del buf242  # reuse
        buf245 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_17, pow_19, mean_17, add_52, rsqrt_17, y_26, mul_80, linear_42], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf243, buf224, buf240, primals_64, buf241, buf245, 32768, 256, grid=grid(32768), stream=stream0)
        buf244 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_65, buf244, 196608, grid=grid(196608), stream=stream0)
        del primals_65
        buf246 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (32768, 256), (256, 1), 0), buf244, out=buf246)
        buf247 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_66, buf247, 196608, grid=grid(196608), stream=stream0)
        del primals_66
        buf248 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (32768, 256), (256, 1), 0), buf247, out=buf248)
        buf249 = empty_strided_cuda((768, 256), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_67, buf249, 196608, grid=grid(196608), stream=stream0)
        del primals_67
        buf250 = empty_strided_cuda((32, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu_8, mul_81], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf246, buf248, buf250, 25165824, grid=grid(25165824), stream=stream0)
        buf251 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf250, (32768, 768), (768, 1), 0), buf249, out=buf251)
        buf252 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf253 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf254 = reinterpret_tensor(buf253, (32, 1024, 1), (1024, 1, 1), 0); del buf253  # reuse
        buf256 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_18, pow_20, mean_18, add_54, rsqrt_18, y_27, mul_83, linear_45], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf254, buf241, buf251, primals_68, buf252, buf256, 32768, 256, grid=grid(32768), stream=stream0)
        buf255 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_69, buf255, 196608, grid=grid(196608), stream=stream0)
        del primals_69
        buf257 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf256, (32768, 256), (256, 1), 0), buf255, out=buf257)
        buf261 = reinterpret_tensor(buf251, (32, 4, 1024, 64), (262144, 64, 256, 1), 0); del buf251  # reuse
        buf260 = empty_strided_cuda((32, 4, 1024, 64), (262144, 64, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [rot_18, mul_84, mul_85, q_19, rot_19, mul_86, mul_87, k_19, y_28], Original ATen: [aten.cat, aten.mul, aten.add, aten._to_copy]
        triton_poi_fused__to_copy_add_cat_mul_2.run(buf257, primals_3, buf261, buf260, 8388608, grid=grid(8388608), stream=stream0)
        # Topologically Sorted Source Nodes: [y_28], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf262 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf261, buf260, reinterpret_tensor(buf257, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), reinterpret_tensor(buf10, (32, 4, 1024, 1024), (1048576, 0, 1024, 1), 0), True)
        buf263 = buf262[0]
        buf264 = buf262[1]
        buf265 = buf262[2]
        buf266 = buf262[3]
        del buf262
        buf267 = empty_strided_cuda((256, 256), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_4.run(primals_70, buf267, 65536, grid=grid(65536), stream=stream0)
        del primals_70
        buf268 = empty_strided_cuda((32768, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf263, (32768, 256), (256, 1), 0), buf267, out=buf268)
        buf269 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf270 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf271 = reinterpret_tensor(buf270, (32, 1024, 1), (1024, 1, 1), 0); del buf270  # reuse
        buf273 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_19, pow_21, mean_19, add_58, rsqrt_19, y_29, mul_89, linear_47], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_8.run(buf271, buf252, buf268, primals_71, buf269, buf273, 32768, 256, grid=grid(32768), stream=stream0)
        buf272 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_72, buf272, 196608, grid=grid(196608), stream=stream0)
        del primals_72
        buf274 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (32768, 256), (256, 1), 0), buf272, out=buf274)
        buf275 = empty_strided_cuda((256, 768), (1, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_48], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_73, buf275, 196608, grid=grid(196608), stream=stream0)
        del primals_73
        buf276 = empty_strided_cuda((32768, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_48], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (32768, 256), (256, 1), 0), buf275, out=buf276)
        buf277 = empty_strided_cuda((768, 256), (1, 768), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten._to_copy, aten.t]
        triton_poi_fused__to_copy_t_1.run(primals_74, buf277, 196608, grid=grid(196608), stream=stream0)
        del primals_74
        buf278 = empty_strided_cuda((32, 1024, 768), (786432, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu_9, mul_90], Original ATen: [aten.silu, aten.mul]
        triton_poi_fused_mul_silu_6.run(buf274, buf276, buf278, 25165824, grid=grid(25165824), stream=stream0)
        buf279 = buf268; del buf268  # reuse
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf278, (32768, 768), (768, 1), 0), buf277, out=buf279)
        buf280 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        buf281 = empty_strided_cuda((32, 1024, 1), (1024, 1, 32768), torch.float32)
        buf282 = reinterpret_tensor(buf281, (32, 1024, 1), (1024, 1, 1), 0); del buf281  # reuse
        buf283 = empty_strided_cuda((32, 1024, 256), (262144, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_20, pow_22, mean_20, add_60, rsqrt_20, y_30, mul_92], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        triton_per_fused_add_mean_mul_pow_rsqrt_9.run(buf282, buf269, buf279, primals_75, buf280, buf283, 32768, 256, grid=grid(32768), stream=stream0)
        del buf279
    return (buf283, primals_2, primals_3, primals_5, primals_8, primals_12, primals_15, primals_19, primals_22, primals_26, primals_29, primals_33, primals_36, primals_40, primals_43, primals_47, primals_50, primals_54, primals_57, primals_61, primals_64, primals_68, primals_71, primals_75, buf0, buf2, reinterpret_tensor(buf4, (32768, 256), (256, 1), 0), reinterpret_tensor(buf5, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), buf8, buf9, buf10, buf12, buf13, buf14, buf15, buf17, buf19, reinterpret_tensor(buf21, (32768, 256), (256, 1), 0), buf22, buf24, reinterpret_tensor(buf26, (32768, 768), (768, 1), 0), buf28, buf30, reinterpret_tensor(buf32, (32768, 256), (256, 1), 0), reinterpret_tensor(buf33, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), buf36, buf37, buf39, buf40, buf41, buf42, buf45, buf47, reinterpret_tensor(buf49, (32768, 256), (256, 1), 0), buf50, buf52, reinterpret_tensor(buf54, (32768, 768), (768, 1), 0), buf56, buf58, reinterpret_tensor(buf60, (32768, 256), (256, 1), 0), reinterpret_tensor(buf61, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), buf64, buf65, buf67, buf68, buf69, buf70, buf73, buf75, reinterpret_tensor(buf77, (32768, 256), (256, 1), 0), buf78, buf80, reinterpret_tensor(buf82, (32768, 768), (768, 1), 0), buf84, buf86, reinterpret_tensor(buf88, (32768, 256), (256, 1), 0), reinterpret_tensor(buf89, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), buf92, buf93, buf95, buf96, buf97, buf98, buf101, buf103, reinterpret_tensor(buf105, (32768, 256), (256, 1), 0), buf106, buf108, reinterpret_tensor(buf110, (32768, 768), (768, 1), 0), buf112, buf114, reinterpret_tensor(buf116, (32768, 256), (256, 1), 0), reinterpret_tensor(buf117, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), buf120, buf121, buf123, buf124, buf125, buf126, buf129, buf131, reinterpret_tensor(buf133, (32768, 256), (256, 1), 0), buf134, buf136, reinterpret_tensor(buf138, (32768, 768), (768, 1), 0), buf140, buf142, reinterpret_tensor(buf144, (32768, 256), (256, 1), 0), reinterpret_tensor(buf145, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), buf148, buf149, buf151, buf152, buf153, buf154, buf157, buf159, reinterpret_tensor(buf161, (32768, 256), (256, 1), 0), buf162, buf164, reinterpret_tensor(buf166, (32768, 768), (768, 1), 0), buf168, buf170, reinterpret_tensor(buf172, (32768, 256), (256, 1), 0), reinterpret_tensor(buf173, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), buf176, buf177, buf179, buf180, buf181, buf182, buf185, buf187, reinterpret_tensor(buf189, (32768, 256), (256, 1), 0), buf190, buf192, reinterpret_tensor(buf194, (32768, 768), (768, 1), 0), buf196, buf198, reinterpret_tensor(buf200, (32768, 256), (256, 1), 0), reinterpret_tensor(buf201, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), buf204, buf205, buf207, buf208, buf209, buf210, buf213, buf215, reinterpret_tensor(buf217, (32768, 256), (256, 1), 0), buf218, buf220, reinterpret_tensor(buf222, (32768, 768), (768, 1), 0), buf224, buf226, reinterpret_tensor(buf228, (32768, 256), (256, 1), 0), reinterpret_tensor(buf229, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), buf232, buf233, buf235, buf236, buf237, buf238, buf241, buf243, reinterpret_tensor(buf245, (32768, 256), (256, 1), 0), buf246, buf248, reinterpret_tensor(buf250, (32768, 768), (768, 1), 0), buf252, buf254, reinterpret_tensor(buf256, (32768, 256), (256, 1), 0), reinterpret_tensor(buf257, (32, 4, 1024, 64), (786432, 64, 768, 1), 512), buf260, buf261, buf263, buf264, buf265, buf266, buf269, buf271, reinterpret_tensor(buf273, (32768, 256), (256, 1), 0), buf274, buf276, reinterpret_tensor(buf278, (32768, 768), (768, 1), 0), buf280, buf282, reinterpret_tensor(buf277, (256, 768), (768, 1), 0), reinterpret_tensor(buf275, (768, 256), (256, 1), 0), reinterpret_tensor(buf272, (768, 256), (256, 1), 0), reinterpret_tensor(buf267, (256, 256), (256, 1), 0), reinterpret_tensor(buf255, (768, 256), (256, 1), 0), reinterpret_tensor(buf249, (256, 768), (768, 1), 0), reinterpret_tensor(buf247, (768, 256), (256, 1), 0), reinterpret_tensor(buf244, (768, 256), (256, 1), 0), reinterpret_tensor(buf239, (256, 256), (256, 1), 0), reinterpret_tensor(buf227, (768, 256), (256, 1), 0), reinterpret_tensor(buf221, (256, 768), (768, 1), 0), reinterpret_tensor(buf219, (768, 256), (256, 1), 0), reinterpret_tensor(buf216, (768, 256), (256, 1), 0), reinterpret_tensor(buf211, (256, 256), (256, 1), 0), reinterpret_tensor(buf199, (768, 256), (256, 1), 0), reinterpret_tensor(buf193, (256, 768), (768, 1), 0), reinterpret_tensor(buf191, (768, 256), (256, 1), 0), reinterpret_tensor(buf188, (768, 256), (256, 1), 0), reinterpret_tensor(buf183, (256, 256), (256, 1), 0), reinterpret_tensor(buf171, (768, 256), (256, 1), 0), reinterpret_tensor(buf165, (256, 768), (768, 1), 0), reinterpret_tensor(buf163, (768, 256), (256, 1), 0), reinterpret_tensor(buf160, (768, 256), (256, 1), 0), reinterpret_tensor(buf155, (256, 256), (256, 1), 0), reinterpret_tensor(buf143, (768, 256), (256, 1), 0), reinterpret_tensor(buf137, (256, 768), (768, 1), 0), reinterpret_tensor(buf135, (768, 256), (256, 1), 0), reinterpret_tensor(buf132, (768, 256), (256, 1), 0), reinterpret_tensor(buf127, (256, 256), (256, 1), 0), reinterpret_tensor(buf115, (768, 256), (256, 1), 0), reinterpret_tensor(buf109, (256, 768), (768, 1), 0), reinterpret_tensor(buf107, (768, 256), (256, 1), 0), reinterpret_tensor(buf104, (768, 256), (256, 1), 0), reinterpret_tensor(buf99, (256, 256), (256, 1), 0), reinterpret_tensor(buf87, (768, 256), (256, 1), 0), reinterpret_tensor(buf81, (256, 768), (768, 1), 0), reinterpret_tensor(buf79, (768, 256), (256, 1), 0), reinterpret_tensor(buf76, (768, 256), (256, 1), 0), reinterpret_tensor(buf71, (256, 256), (256, 1), 0), reinterpret_tensor(buf59, (768, 256), (256, 1), 0), reinterpret_tensor(buf53, (256, 768), (768, 1), 0), reinterpret_tensor(buf51, (768, 256), (256, 1), 0), reinterpret_tensor(buf48, (768, 256), (256, 1), 0), reinterpret_tensor(buf43, (256, 256), (256, 1), 0), reinterpret_tensor(buf31, (768, 256), (256, 1), 0), reinterpret_tensor(buf25, (256, 768), (768, 1), 0), reinterpret_tensor(buf23, (768, 256), (256, 1), 0), reinterpret_tensor(buf20, (768, 256), (256, 1), 0), reinterpret_tensor(buf16, (256, 256), (256, 1), 0), reinterpret_tensor(buf3, (768, 256), (256, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((50304, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    primals_3 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    primals_4 = rand_strided((32, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
