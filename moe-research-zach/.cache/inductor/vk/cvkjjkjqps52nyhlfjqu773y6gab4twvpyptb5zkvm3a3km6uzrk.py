# AOT ID: ['22_inference']
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/gs/cgsgmobchsp6dlk542tbrbxkej54zty3mituvg3jwiovl42gbep2.py
# Topologically Sorted Source Nodes: [mask_block_sum, gt, lt, partial_blocks, partial_blocks_1, full_blocks, full_blocks_1], Original ATen: [aten.sum, aten.gt, aten.lt, aten.bitwise_and, aten._to_copy, aten.eq]
# Source node to ATen node mapping:
#   full_blocks => eq_1
#   full_blocks_1 => convert_element_type_1
#   gt => gt
#   lt => lt
#   mask_block_sum => sum_1
#   partial_blocks => bitwise_and_1
#   partial_blocks_1 => convert_element_type
# Graph fragment:
#   %sum_1 : [num_users=3] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute, [-2, -1]), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%sum_1, 0), kwargs = {})
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%sum_1, 16384), kwargs = {})
#   %bitwise_and_1 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%gt, %lt), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bitwise_and_1, torch.int8), kwargs = {})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%sum_1, 16384), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%eq_1, torch.int8), kwargs = {})
triton_red_fused__to_copy_bitwise_and_eq_gt_lt_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 16384],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*i64', 3: '*i8', 4: '*i8', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_bitwise_and_eq_gt_lt_sum_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.int64)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = (rindex // 128)
        r2 = rindex % 128
        tmp0 = tl.load(in_ptr0 + (r3 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (128*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 == tmp1
        tmp4 = r2 + (128*x0)
        tmp5 = tmp4 <= tmp3
        tmp6 = tmp2 & tmp5
        tmp7 = tmp6.to(tl.int64)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp11 = tl.full([1, 1], 0, tl.int64)
    tmp12 = tmp9 > tmp11
    tmp13 = tl.full([1, 1], 16384, tl.int64)
    tmp14 = tmp9 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp15.to(tl.int8)
    tmp17 = tmp9 == tmp13
    tmp18 = tmp17.to(tl.int8)
    tl.store(out_ptr1 + (x4), tmp16, None)
    tl.store(out_ptr2 + (x4), tmp18, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8192, ), (1, ))
    assert_size_stride(arg1_1, (16384, ), (1, ))
    assert_size_stride(arg2_1, (8192, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1, 1, 64, 128), (8192, 8192, 128, 1), torch.int8)
        buf2 = empty_strided_cuda((1, 1, 64, 128), (8192, 8192, 128, 1), torch.int8)
        # Topologically Sorted Source Nodes: [mask_block_sum, gt, lt, partial_blocks, partial_blocks_1, full_blocks, full_blocks_1], Original ATen: [aten.sum, aten.gt, aten.lt, aten.bitwise_and, aten._to_copy, aten.eq]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_bitwise_and_eq_gt_lt_sum_0.run(arg0_1, arg1_1, arg2_1, buf1, buf2, 8192, 16384, grid=grid(8192), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
    return (buf1, buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((16384, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((8192, ), (1, ), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
