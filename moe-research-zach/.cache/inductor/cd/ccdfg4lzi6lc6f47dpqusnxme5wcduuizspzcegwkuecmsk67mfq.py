# AOT ID: ['1_inference']
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/p6/cp6iao5qilykklui3jummi6zsjausgpypx4ixw5prrx36wgeiukb.py
# Topologically Sorted Source Nodes: [same, causal, ones, and_, m, eye, pad_self, or_], Original ATen: [aten.eq, aten.tril, aten.ones, aten.bitwise_and, aten.eye, aten.bitwise_or]
# Source node to ATen node mapping:
#   and_ => bitwise_and
#   causal => le, logical_and, sub
#   eye => eq_1, iota_3
#   m => bitwise_and_1
#   ones => full_default
#   or_ => bitwise_or
#   pad_self => bitwise_and_2
#   same => eq
# Graph fragment:
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_2, %unsqueeze_3), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub, 0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1024, 1024], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %logical_and : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le, %full_default), kwargs = {})
#   %bitwise_and : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%eq, %logical_and), kwargs = {})
#   %bitwise_and_1 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %unsqueeze_4), kwargs = {})
#   %iota_3 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1024,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze_6, %iota_3), kwargs = {})
#   %bitwise_and_2 : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%unsqueeze_5, %eq_1), kwargs = {})
#   %bitwise_or : [num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_and_1, %bitwise_and_2), kwargs = {})
triton_poi_fused_bitwise_and_bitwise_or_eq_eye_ones_tril_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_bitwise_or_eq_eye_ones_tril_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
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


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (16, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 1024, 1024), (1048576, 1024, 1), torch.bool)
        # Topologically Sorted Source Nodes: [same, causal, ones, and_, m, eye, pad_self, or_], Original ATen: [aten.eq, aten.tril, aten.ones, aten.bitwise_and, aten.eye, aten.bitwise_or]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bitwise_and_bitwise_or_eq_eye_ones_tril_0.run(arg0_1, buf0, 16777216, grid=grid(16777216), stream=stream0)
        del arg0_1
    return (reinterpret_tensor(buf0, (16, 1, 1024, 1024), (1048576, 1048576, 1024, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 1024), (1024, 1), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
