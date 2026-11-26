# AOT ID: ['6_backward']
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/2x/c2xwmdav2rbkoecerim2v5lfqe3hu5y3lscdb7ebauw32ef7v46t.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.new_zeros, aten.scatter_add]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_4 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_1, %primals_5), kwargs = {})
#   %full_default_1 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([8192, 50304], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_add : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_1, 1, %primals_4, %mul_4), kwargs = {})
triton_poi_fused_mul_new_zeros_scatter_add_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[536870912], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_new_zeros_scatter_add_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 412090368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/wi/cwi6evvinai4aiwbpc5jh4q7ixngll74shebwvr5mgkprgfibwkd.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.new_zeros, aten.scatter_add]
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_4 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_1, %primals_5), kwargs = {})
#   %full_default_1 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([8192, 50304], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_add : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_1, 1, %primals_4, %mul_4), kwargs = {})
triton_poi_fused_mul_new_zeros_scatter_add_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_new_zeros_scatter_add_1', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = (xindex // 4)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr3 + (x2), None)
    tl.device_assert((0 <= tmp0) & (tmp0 < 50304), "index out of bounds: 0 <= tmp0 < 50304")
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = -tmp6
    tmp9 = tmp7 * tmp8
    tl.atomic_add(out_ptr0 + (tmp0 + (50304*x1)), tmp9, None, sem='relaxed')
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/fn/cfntvsoh67ybhoqhmnn5f52vy772deoprqxmb2w2bqesp7bdwdqi.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.new_zeros, aten.mul, aten.add, aten.scatter_add]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default_1 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([8192, 50304], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_5 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %ge), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze_2, %mul_5), kwargs = {})
#   %scatter_add_1 : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_1, 1, %view, %unsqueeze_2), kwargs = {})
triton_poi_fused_add_mul_new_zeros_scatter_add_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_new_zeros_scatter_add_2', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (x0), None).to(tl.int1)
    tmp8 = tl.load(in_ptr3 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp12 = tl.load(in_ptr4 + (4*x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (1 + (4*x0)), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (2 + (4*x0)), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (3 + (4*x0)), None, eviction_policy='evict_last')
    tl.device_assert((0 <= tmp0) & (tmp0 < 50304), "index out of bounds: 0 <= tmp0 < 50304")
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = -tmp6
    tmp10 = tmp9 * tmp5
    tmp11 = -tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = -tmp13
    tmp16 = tmp11 * tmp15
    tmp17 = -tmp16
    tmp18 = tmp14 + tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = -tmp20
    tmp22 = tmp18 + tmp21
    tmp24 = tmp11 * tmp23
    tmp25 = -tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = tmp26 + tmp6
    tl.atomic_add(out_ptr0 + (tmp0 + (50304*x0)), tmp7, None, sem='relaxed')
    tl.store(out_ptr1 + (x0), tmp27, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ql/cqld6grbnssdysej7u52gsylxbg4vsuslbbjv3bajfvggbu22o3z.py
# Topologically Sorted Source Nodes: [lg], Original ATen: [aten.add, aten._to_copy, aten.sub, aten.exp, aten.mul]
# Source node to ATen node mapping:
#   lg => convert_element_type_4
# Graph fragment:
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%scatter_add, %scatter_add_1), kwargs = {})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_4, %unsqueeze_4), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_3,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_3, %exp_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_6), kwargs = {})
#   %convert_element_type_5 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_add_exp_mul_sub_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[536870912], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: '*fp32', 5: '*bf16', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_exp_mul_sub_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 412090368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = (xindex // 50304)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 - tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp9 = tmp3 * tmp8
    tmp10 = tmp2 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp11, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/7n/c7nfr347epln5wreaddte47jwbtqn5xnvhyukbx5cig7523i3x6y.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_10 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
triton_poi_fused__to_copy_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/jt/cjtsexsiwqptbqr6iwf3dgb5hdh5jijyjk67hnwwe4vdkkckwz4h.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_11 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_4, torch.float32), kwargs = {})
triton_poi_fused__to_copy_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_5', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19316736
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
    primals_4, primals_5, convert_element_type_1, mm, ge, view, unsqueeze_4, permute_3, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_4, (8192, 4), (4, 1))
    assert_size_stride(primals_5, (8192, 4), (4, 1))
    assert_size_stride(convert_element_type_1, (8192, 384), (384, 1))
    assert_size_stride(mm, (8192, 50304), (50304, 1))
    assert_size_stride(ge, (8192, ), (1, ))
    assert_size_stride(view, (8192, 1), (1, 1))
    assert_size_stride(unsqueeze_4, (8192, 1), (1, 1))
    assert_size_stride(permute_3, (50304, 384), (384, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8192, 50304), (50304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.new_zeros, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_new_zeros_scatter_add_0.run(buf0, 412090368, grid=grid(412090368), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mul, aten.new_zeros, aten.scatter_add]
        triton_poi_fused_mul_new_zeros_scatter_add_1.run(primals_4, tangents_2, ge, primals_5, buf0, 32768, grid=grid(32768), stream=stream0)
        del primals_4
        buf2 = empty_strided_cuda((8192, 50304), (50304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.new_zeros, aten.scatter_add]
        triton_poi_fused_mul_new_zeros_scatter_add_0.run(buf2, 412090368, grid=grid(412090368), stream=stream0)
        buf4 = empty_strided_cuda((8192, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.new_zeros, aten.mul, aten.add, aten.scatter_add]
        triton_poi_fused_add_mul_new_zeros_scatter_add_2.run(view, tangents_1, ge, tangents_2, primals_5, buf2, buf4, 8192, grid=grid(8192), stream=stream0)
        del ge
        del primals_5
        del tangents_1
        del tangents_2
        del view
        buf5 = empty_strided_cuda((8192, 50304), (50304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [lg], Original ATen: [aten.add, aten._to_copy, aten.sub, aten.exp, aten.mul]
        triton_poi_fused__to_copy_add_exp_mul_sub_3.run(buf0, buf2, buf4, mm, unsqueeze_4, buf5, 412090368, grid=grid(412090368), stream=stream0)
        del buf0
        del buf2
        del buf4
        del mm
        del unsqueeze_4
        buf6 = empty_strided_cuda((50304, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (50304, 8192), (1, 50304), 0), convert_element_type_1, out=buf6)
        del convert_element_type_1
        buf7 = empty_strided_cuda((8192, 384), (384, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf5, permute_3, out=buf7)
        del buf5
        del permute_3
        buf8 = empty_strided_cuda((8192, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf7, buf8, 3145728, grid=grid(3145728), stream=stream0)
        del buf7
        buf9 = empty_strided_cuda((50304, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_5.run(buf6, buf9, 19316736, grid=grid(19316736), stream=stream0)
        del buf6
    return (buf9, buf8, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_4 = rand_strided((8192, 4), (4, 1), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((8192, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_1 = rand_strided((8192, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    mm = rand_strided((8192, 50304), (50304, 1), device='cuda:0', dtype=torch.bfloat16)
    ge = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.bool)
    view = rand_strided((8192, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    unsqueeze_4 = rand_strided((8192, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    permute_3 = rand_strided((50304, 384), (384, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_4, primals_5, convert_element_type_1, mm, ge, view, unsqueeze_4, permute_3, tangents_1, tangents_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
