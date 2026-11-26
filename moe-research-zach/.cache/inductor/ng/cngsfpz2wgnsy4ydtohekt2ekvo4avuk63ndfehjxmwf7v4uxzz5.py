# AOT ID: ['7_inference']
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


# kernel path: /home/zapan4/moe-research/.cache/inductor/wh/cwh76bicvq4gzk6jyqkzuuxczlh7oc4ql62a55igq2q2ckknqz5b.py
# Topologically Sorted Source Nodes: [lg, argmax, lse, softmax, mul, sum_1, ent], Original ATen: [aten._to_copy, aten.argmax, aten.logsumexp, aten._softmax, aten.mul, aten.sum, aten.sub]
# Source node to ATen node mapping:
#   argmax => argmax
#   ent => sub_2
#   lg => convert_element_type_2
#   lse => abs_1, add, amax, eq, exp, full_default, log, sub, sum_1, where
#   mul => mul
#   softmax => amax_1, div, exp_1, sub_1, sum_2
#   sum_1 => sum_3
# Graph fragment:
#   %convert_element_type_2 : [num_users=6] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %argmax : [num_users=1] = call_function[target=torch.ops.aten.argmax.default](args = (%convert_element_type_2, -1), kwargs = {})
#   %amax : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_2, [-1], True), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax,), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_1, inf), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %amax), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_2, %where), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1]), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %squeeze), kwargs = {})
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%convert_element_type_2, [-1], True), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_2, %amax_1), kwargs = {})
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_1, %sum_2), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %convert_element_type_2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [-1]), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %sum_3), kwargs = {})
triton_red_fused__softmax__to_copy_argmax_logsumexp_mul_sub_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[32768, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*i64', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_argmax_logsumexp_mul_sub_sum_0', 'mutated_arg_names': ['in_out_ptr1'], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 6, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 31744
    rnumel = 50304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    _tmp3_index = tl.full([XBLOCK, RBLOCK], 9223372036854775807, tl.int64)
    _tmp4 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (50304*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        _tmp3_next, _tmp3_index_next = triton_helpers.maximum_with_index(
            _tmp3, _tmp3_index, tmp2, rindex
        )
        _tmp3 = tl.where(rmask & xmask, _tmp3_next, _tmp3)
        _tmp3_index = tl.where(rmask & xmask, _tmp3_index_next, _tmp3_index)
        tmp5 = triton_helpers.maximum(_tmp4, tmp2)
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    _, tmp3_tmp = triton_helpers.max_with_index(_tmp3, _tmp3_index, 1)
    tmp3 = tmp3_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp19 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + (50304*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl_math.abs(tmp4)
        tmp9 = float("inf")
        tmp10 = tmp8 == tmp9
        tmp11 = 0.0
        tmp12 = tl.where(tmp10, tmp11, tmp4)
        tmp13 = tmp7 - tmp12
        tmp14 = tl_math.exp(tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
        tmp18 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp20 = triton_helpers.maximum(_tmp19, tmp18)
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp19 = triton_helpers.max2(_tmp19, 1)[:, None]
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp21 = tl.load(in_ptr0 + (r1 + (50304*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22 - tmp19
        tmp24 = tl_math.exp(tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    _tmp35 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp28 = tl.load(in_ptr0 + (r1 + (50304*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp29 - tmp19
        tmp31 = tl_math.exp(tmp30)
        tmp32 = tmp31 / tmp26
        tmp33 = tmp32 * tmp29
        tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
        tmp36 = _tmp35 + tmp34
        _tmp35 = tl.where(rmask & xmask, tmp36, _tmp35)
    tmp35 = tl.sum(_tmp35, 1)[:, None]
    tmp37 = tl_math.log(tmp16)
    tmp38 = tl_math.abs(tmp4)
    tmp39 = float("inf")
    tmp40 = tmp38 == tmp39
    tmp41 = 0.0
    tmp42 = tl.where(tmp40, tmp41, tmp4)
    tmp43 = tmp37 + tmp42
    tmp44 = tmp43 - tmp35
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp44, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (50304, 256), (256, 1))
    assert_size_stride(arg1_1, (31744, 256), (256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((31744, 50304), (50304, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.mm]
        extern_kernels.mm(arg1_1, reinterpret_tensor(arg0_1, (256, 50304), (1, 256), 0), out=buf0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((31744, ), (1, ), torch.int64)
        buf3 = empty_strided_cuda((31744, ), (1, ), torch.float32)
        buf7 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [lg, argmax, lse, softmax, mul, sum_1, ent], Original ATen: [aten._to_copy, aten.argmax, aten.logsumexp, aten._softmax, aten.mul, aten.sum, aten.sub]
        stream0 = get_raw_stream(0)
        triton_red_fused__softmax__to_copy_argmax_logsumexp_mul_sub_sum_0.run(buf7, buf0, buf1, 31744, 50304, grid=grid(31744), stream=stream0)
        del buf0
    return (buf1, buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((50304, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((31744, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
