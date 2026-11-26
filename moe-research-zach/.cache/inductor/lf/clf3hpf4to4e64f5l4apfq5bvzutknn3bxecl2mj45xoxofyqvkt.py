
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_bitwise_and_clamp_div_eq_mul_sum_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x3 = xindex
    tmp9 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    x1 = (xindex // 1024)
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp10 = tl.load(in_ptr0 + (r2 + (1024*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0 + ((-1)*r2)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = 0.0
        tmp3 = tmp1 >= tmp2
        tmp4 = 0.9945994019508362
        tmp5 = ((0.0) * ((0.0) >= (x0 + ((-1)*r2))) + (x0 + ((-1)*r2)) * ((x0 + ((-1)*r2)) > (0.0)))
        tmp6 = tmp5.to(tl.float32)
        tmp7 = libdevice.pow(tmp4, tmp6)
        tmp8 = tl.where(tmp3, tmp7, tmp2)
        tmp11 = tmp9 == tmp10
        tmp12 = tl.full([1, 1], 0, tl.int64)
        tmp13 = tmp9 >= tmp12
        tmp14 = tmp11 & tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp8 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp29 = tl.load(in_ptr0 + (r2 + (1024*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp20 = x0 + ((-1)*r2)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = 0.0
        tmp23 = tmp21 >= tmp22
        tmp24 = 0.9945994019508362
        tmp25 = ((0.0) * ((0.0) >= (x0 + ((-1)*r2))) + (x0 + ((-1)*r2)) * ((x0 + ((-1)*r2)) > (0.0)))
        tmp26 = tmp25.to(tl.float32)
        tmp27 = libdevice.pow(tmp24, tmp26)
        tmp28 = tl.where(tmp23, tmp27, tmp22)
        tmp30 = tmp9 == tmp29
        tmp31 = tl.full([1, 1], 0, tl.int64)
        tmp32 = tmp9 >= tmp31
        tmp33 = tmp30 & tmp32
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp28 * tmp34
        tmp36 = 1e-12
        tmp37 = triton_helpers.maximum(tmp18, tmp36)
        tmp38 = tmp35 / tmp37
        tl.store(out_ptr1 + (r2 + (1024*x3)), tmp38, rmask)
