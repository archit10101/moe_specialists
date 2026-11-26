# AOT ID: ['12_forward']
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/n7/cn7jl6hjbzt6w6m6rsfviusn3sokiz63fxig7incuqlijzlpkhrn.py
# Topologically Sorted Source Nodes: [mean, pc, mean_1, sub_1, pow_1, mean_2, sc, pow_2, mean_3, add, rsqrt, mul, pr_1], Original ATen: [aten.mean, aten.sub, aten.pow, aten.sqrt, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add_1
#   mean => mean
#   mean_1 => mean_1
#   mean_2 => mean_2
#   mean_3 => mean_3
#   mul => mul_1
#   pc => sub
#   pow_1 => pow_1
#   pow_2 => pow_2
#   pr_1 => mul_2
#   rsqrt => rsqrt
#   sc => sqrt
#   sub_1 => sub_1
# Graph fragment:
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%mm, [-1], True), kwargs = {})
#   %sub : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mm, %mean), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%primals_4, [-1], True), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %mean_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_1, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %sqrt : [num_users=2] = call_function[target=torch.ops.aten.sqrt.default](args = (%mean_2,), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-08), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %sqrt), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_sqrt_sub_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_sqrt_sub_0', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp13 = tl.load(in_ptr1 + (r1 + (32*x0)), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None]
    tmp4 = 32.0
    tmp5 = tmp3 / tmp4
    tmp6 = tmp0 - tmp5
    tmp7 = tmp6 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.sum(tmp8, 1)[:, None]
    tmp11 = tmp10 / tmp4
    tmp12 = libdevice.sqrt(tmp11)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp16 = tl.sum(tmp14, 1)[:, None]
    tmp17 = tmp16 / tmp4
    tmp18 = tmp13 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.sum(tmp20, 1)[:, None]
    tmp23 = tmp22 / tmp4
    tmp24 = 1e-08
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp18 * tmp26
    tmp28 = tmp27 * tmp12
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp17, None)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp26, None)
    tl.store(out_ptr0 + (r1 + (32*x0)), tmp28, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    s0 = primals_2
    assert_size_stride(primals_1, (11, 64, 32), (2048, 32, 1))
    assert_size_stride(primals_3, (32768, 64), (64, 1))
    assert_size_stride(primals_4, (32768, 32), (32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32768, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pr], Original ATen: [aten.mm]
        extern_kernels.mm(primals_3, reinterpret_tensor(primals_1, (64, 32), (32, 1), 2048*s0), out=buf0)
        buf3 = empty_strided_cuda((32768, 1), (1, 32768), torch.float32)
        buf4 = buf3; del buf3  # reuse
        buf5 = reinterpret_tensor(buf4, (32768, 1), (1, 1), 0); del buf4  # reuse
        buf1 = empty_strided_cuda((32768, 1), (1, 32768), torch.float32)
        buf2 = reinterpret_tensor(buf1, (32768, 1), (1, 1), 0); del buf1  # reuse
        buf6 = empty_strided_cuda((32768, 1), (1, 32768), torch.float32)
        buf7 = reinterpret_tensor(buf6, (32768, 1), (1, 1), 0); del buf6  # reuse
        buf8 = empty_strided_cuda((32768, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mean, pc, mean_1, sub_1, pow_1, mean_2, sc, pow_2, mean_3, add, rsqrt, mul, pr_1], Original ATen: [aten.mean, aten.sub, aten.pow, aten.sqrt, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_sqrt_sub_0.run(buf5, buf2, buf7, primals_4, buf0, buf8, 32768, 32, grid=grid(32768), stream=stream0)
        del primals_4
    return (buf8, buf0, buf2, buf5, buf7, reinterpret_tensor(primals_3, (64, 32768), (1, 64), 0), reinterpret_tensor(primals_1, (32, 64), (1, 32), 2048*s0), s0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((11, 64, 32), (2048, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = 1
    primals_3 = rand_strided((32768, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32768, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
