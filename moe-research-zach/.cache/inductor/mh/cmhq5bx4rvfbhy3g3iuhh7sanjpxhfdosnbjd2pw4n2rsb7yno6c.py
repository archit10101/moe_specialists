
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[1024, 16384],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i8', 2: '*i8', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_bitwise_and_eq_gt_lt_sum_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x6 = (xindex // 8)
    x0 = xindex % 8
    x2 = (xindex // 64)
    x1 = (xindex // 8) % 8
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.int64)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = (rindex // 128)
        r3 = rindex % 128
        tmp0 = tl.load(in_ptr0 + (r4 + (128*x6)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + (128*x0) + (1024*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 == tmp1
        tmp3 = r3 + (128*x0)
        tmp4 = r4 + (128*x1)
        tmp5 = tmp3 <= tmp4
        tmp6 = tmp2 & tmp5
        tmp7 = r4 + ((-1)*r3) + ((-128)*x0) + (128*x1)
        tmp8 = tl.full([1, 1], 128, tl.int64)
        tmp9 = tmp7 < tmp8
        tmp10 = tmp6 & tmp9
        tmp11 = tmp10.to(tl.int64)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp15 = tl.full([1, 1], 0, tl.int64)
    tmp16 = tmp13 > tmp15
    tmp17 = tl.full([1, 1], 16384, tl.int64)
    tmp18 = tmp13 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tmp19.to(tl.int8)
    tmp21 = tmp13 == tmp17
    tmp22 = tmp21.to(tl.int8)
    tl.store(out_ptr1 + (x5), tmp20, xmask)
    tl.store(out_ptr2 + (x5), tmp22, xmask)
