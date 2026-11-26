# AOT ID: ['8_backward']
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/4v/c4vcn2hvvqdzylywk7b3jz3av6y3e4hywkkshgulgvjsrgjvc5or.py
# Topologically Sorted Source Nodes: [to], Original ATen: [aten._to_copy, aten.mul, aten.new_zeros, aten.scatter_add]
# Source node to ATen node mapping:
#   to => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_1, torch.float32), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %convert_element_type), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 1024, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_add : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default, 1, %expand, %mul_1), kwargs = {})
triton_poi_fused__to_copy_mul_new_zeros_scatter_add_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_new_zeros_scatter_add_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /home/zapan4/moe-research/.cache/inductor/2w/c2w5ovih47pzm3yw4ftnuxniajpzgt3yksot7i263vi4lolkb3jx.py
# Topologically Sorted Source Nodes: [to], Original ATen: [aten._to_copy, aten.mul, aten.new_zeros, aten.scatter_add]
# Source node to ATen node mapping:
#   to => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_1, torch.float32), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %convert_element_type), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 1024, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_add : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default, 1, %expand, %mul_1), kwargs = {})
triton_poi_fused__to_copy_mul_new_zeros_scatter_add_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_mul_new_zeros_scatter_add_1', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = (xindex // 128)
    x4 = xindex
    x0 = xindex % 128
    x2 = (xindex // 131072)
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x4), None)
    tmp3 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last').to(tl.int1)
    tl.device_assert((0 <= tmp0) & (tmp0 < 1024), "index out of bounds: 0 <= tmp0 < 1024")
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tl.atomic_add(out_ptr0 + (x0 + (128*tmp0) + (131072*x2)), tmp5, None, sem='relaxed')
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    unsqueeze, unsqueeze_1, tangents_1 = args
    args.clear()
    assert_size_stride(unsqueeze, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(unsqueeze_1, (32, 1024, 1), (1024, 1, 1))
    assert_size_stride(tangents_1, (32, 1024, 128), (131072, 128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 1024, 128), (131072, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [to], Original ATen: [aten._to_copy, aten.mul, aten.new_zeros, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_mul_new_zeros_scatter_add_0.run(buf0, 4194304, grid=grid(4194304), stream=stream0)
        # Topologically Sorted Source Nodes: [to], Original ATen: [aten._to_copy, aten.mul, aten.new_zeros, aten.scatter_add]
        triton_poi_fused__to_copy_mul_new_zeros_scatter_add_1.run(unsqueeze, tangents_1, unsqueeze_1, buf0, 4194304, grid=grid(4194304), stream=stream0)
        del tangents_1
        del unsqueeze
        del unsqueeze_1
    return (None, buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    unsqueeze = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.int64)
    unsqueeze_1 = rand_strided((32, 1024, 1), (1024, 1, 1), device='cuda:0', dtype=torch.bool)
    tangents_1 = rand_strided((32, 1024, 128), (131072, 128, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([unsqueeze, unsqueeze_1, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
