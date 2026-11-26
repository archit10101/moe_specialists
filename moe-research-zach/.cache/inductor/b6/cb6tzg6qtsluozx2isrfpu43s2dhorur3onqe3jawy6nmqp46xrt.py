# AOT ID: ['12_backward']
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/oe/coez43tamu4dmkmnuo7z6kv32aokgevpao4wuzst2cq35p2l4ax3.py
# Topologically Sorted Source Nodes: [pc], Original ATen: [aten.mul, aten.sub, aten.sum, aten.div, aten.pow, aten.add, aten.neg]
# Source node to ATen node mapping:
#   pc => sub
# Graph fragment:
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %sqrt), kwargs = {})
#   %sub : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mm, %mean), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %sub), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %rsqrt), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_4, [1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 32), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 1.0), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_4, 2.0), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %mul_8), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %mul_9), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_2,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%neg, [1], True), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_1, 32), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %div_1), kwargs = {})
triton_per_fused_add_div_mul_neg_pow_sub_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[32768, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_neg_pow_sub_sum_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (32*x0)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (r1 + (32*x0)), None)
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 - tmp4
    tmp6 = tmp2 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None]
    tmp11 = tmp2 * tmp10
    tmp12 = -0.5
    tmp13 = tmp9 * tmp12
    tmp14 = tmp10 * tmp10
    tmp15 = tmp14 * tmp10
    tmp16 = tmp13 * tmp15
    tmp17 = 0.03125
    tmp18 = tmp16 * tmp17
    tmp19 = 2.0
    tmp20 = tmp5 * tmp19
    tmp21 = tmp18 * tmp20
    tmp22 = tmp11 + tmp21
    tmp23 = -tmp22
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    tmp26 = tl.sum(tmp24, 1)[:, None]
    tmp27 = tmp26 * tmp17
    tmp28 = tmp22 + tmp27
    tl.store(out_ptr3 + (r1 + (32*x0)), tmp28, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/ae/cae2aogoxcpxh2y4yv6pxtxj45tjbdrdwenmznr4xszk3vsd3wa2.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([11, 64, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %mm_1, 0, %primals_2), kwargs = {})
triton_poi_fused_select_backward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_select_backward_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 22528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 2048)
    x0 = xindex % 2048
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = ks0
    tmp2 = tmp0 == tmp1
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_2, mm, mean, sqrt, rsqrt, permute, permute_1, tangents_1 = args
    args.clear()
    s0 = primals_2
    assert_size_stride(mm, (32768, 32), (32, 1))
    assert_size_stride(mean, (32768, 1), (1, 1))
    assert_size_stride(sqrt, (32768, 1), (1, 1))
    assert_size_stride(rsqrt, (32768, 1), (1, 1))
    assert_size_stride(permute, (64, 32768), (1, 64))
    assert_size_stride(permute_1, (32, 64), (1, 32))
    assert_size_stride(tangents_1, (32768, 32), (32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((32768, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pc], Original ATen: [aten.mul, aten.sub, aten.sum, aten.div, aten.pow, aten.add, aten.neg]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mul_neg_pow_sub_sum_0.run(tangents_1, sqrt, mm, mean, rsqrt, buf3, 32768, 32, grid=grid(32768), stream=stream0)
        del mean
        del mm
        del rsqrt
        del sqrt
        del tangents_1
        buf4 = empty_strided_cuda((64, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(permute, buf3, out=buf4)
        del permute
        buf5 = empty_strided_cuda((32768, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, permute_1, out=buf5)
        del buf3
        del permute_1
        buf6 = empty_strided_cuda((11, 64, 32), (2048, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.select_backward]
        triton_poi_fused_select_backward_1.run(buf4, buf6, s0, 22528, grid=grid(22528), stream=stream0)
        del buf4
    return (buf6, None, buf5, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = 1
    mm = rand_strided((32768, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    mean = rand_strided((32768, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    sqrt = rand_strided((32768, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((32768, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    permute = rand_strided((64, 32768), (1, 64), device='cuda:0', dtype=torch.float32)
    permute_1 = rand_strided((32, 64), (1, 32), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((32768, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_2, mm, mean, sqrt, rsqrt, permute, permute_1, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
