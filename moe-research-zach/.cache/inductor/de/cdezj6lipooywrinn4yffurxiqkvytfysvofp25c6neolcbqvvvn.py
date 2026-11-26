
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton.jit
def _triton_helper_fn_gt_eq_gt_logical_and_logical_or_where_where0(arg0_0, arg0_1, arg1_0, arg1_1):
    tmp0 = arg0_0 > arg1_0
    tmp1 = arg0_0 == arg1_0
    tmp2 = arg0_1 > arg1_1
    tmp3 = tmp1 & tmp2
    tmp4 = tmp0 | tmp3
    tmp5 = tl.where(tmp4, arg0_0, arg1_0)
    tmp6 = tl.where(tmp4, arg0_1, arg1_1)
    return tmp5, tmp6

@triton_heuristics.reduction(
    size_hints=[32, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i1', 2: '*i64', 3: '*i64', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_bitwise_or_cummax_eq_full_like_index_lt_sub_where_zeros_like_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp9 = tl.full([XBLOCK, 1], -1, tl.int64)
    tmp12 = tl.full([XBLOCK, 1], -1, tl.int64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.int1)
        tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp40 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 == tmp2
        tmp4 = tmp0 | tmp3
        tmp5 = r1
        tmp6 = tl.where(tmp4, tmp5, tmp2)
        tmp7 = tmp6.to(tl.int64)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = rindex.to(tl.int64)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13, tmp14, = tl.associative_scan((tmp8, tmp11,), 1, _triton_helper_fn_gt_eq_gt_logical_and_logical_or_where_where0)
        tmp15 = triton_helpers.select_one((tmp13), rbase == (RBLOCK - 1), dim=-1, keep_dims=True)
        tmp16 = triton_helpers.select_one((tmp14), rbase == (RBLOCK - 1), dim=-1, keep_dims=True)
        tmp17 = tmp9 > tmp15
        tmp18 = tmp9 == tmp15
        tmp19 = tmp12 > tmp16
        tmp20 = tmp18 & tmp19
        tmp21 = tmp17 | tmp20
        tmp22 = tl.where(tmp21, tmp9, tmp15)
        tmp23 = tl.where(tmp21, tmp12, tmp16)
        tmp24 = tmp9 > tmp13
        tmp25 = tmp9 == tmp13
        tmp26 = tmp12 > tmp14
        tmp27 = tmp25 & tmp26
        tmp28 = tmp24 | tmp27
        tmp29 = tl.where(tmp28, tmp9, tmp13)
        tmp30 = tl.where(tmp28, tmp12, tmp14)
        tmp31 = tl.where(roffset > 0, tmp29, tmp13)
        tmp32 = tl.where(roffset > 0, tmp30, tmp14)
        tmp9 = tl.where(roffset > 0, tmp22, tmp15)
        tmp12 = tl.where(roffset > 0, tmp23, tmp16)
        tmp33 = tl.full([XBLOCK, RBLOCK], 1024, tl.int32)
        tmp34 = tmp31 + tmp33
        tmp35 = tmp31 < 0
        tmp36 = tl.where(tmp35, tmp34, tmp31)
        tl.device_assert(((0 <= tmp36) & (tmp36 < 1024)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp36 < 1024")
        tmp38 = tl.load(in_ptr1 + (tmp36 + (1024*x0)), rmask & xmask, eviction_policy='evict_last')
        tmp39 = tmp38 == tmp2
        tmp41 = tmp40 < tmp2
        tmp42 = tmp39 | tmp41
        tmp43 = tl.full([1, 1], 1, tl.int64)
        tmp44 = tmp31 - tmp43
        tmp45 = tl.full([1, 1], -1, tl.int64)
        tmp46 = tl.where(tmp42, tmp45, tmp44)
        tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp46, rmask & xmask)
