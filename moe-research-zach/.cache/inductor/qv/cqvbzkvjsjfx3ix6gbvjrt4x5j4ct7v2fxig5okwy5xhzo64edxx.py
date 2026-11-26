
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[65536, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*i64', 3: 'i64', 4: 'i64'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_argmax_logsumexp_mul_sub_sum_0', 'mutated_arg_names': ['in_out_ptr1'], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 6, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 50304
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :].to(tl.int64)
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    _tmp3_index = tl.full([XBLOCK, RBLOCK], 9223372036854775807, tl.int64)
    _tmp4 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (50304*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        _tmp3_next, _tmp3_index_next = triton_helpers.maximum_with_index(
            _tmp3, _tmp3_index, tmp2, rindex
        )
        _tmp3 = tl.where(rmask, _tmp3_next, _tmp3)
        _tmp3_index = tl.where(rmask, _tmp3_index_next, _tmp3_index)
        tmp5 = triton_helpers.maximum(_tmp4, tmp2)
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    _, tmp3_tmp = triton_helpers.max_with_index(_tmp3, _tmp3_index, 1)
    tmp3 = tmp3_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp3, None)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp19 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + (50304*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
        tmp18 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp20 = triton_helpers.maximum(_tmp19, tmp18)
        _tmp19 = tl.where(rmask, tmp20, _tmp19)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp19 = triton_helpers.max2(_tmp19, 1)[:, None]
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp21 = tl.load(in_ptr0 + (r1 + (50304*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22 - tmp19
        tmp24 = tl_math.exp(tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    _tmp35 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp28 = tl.load(in_ptr0 + (r1 + (50304*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp29 - tmp19
        tmp31 = tl_math.exp(tmp30)
        tmp32 = tmp31 / tmp26
        tmp33 = tmp32 * tmp29
        tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
        tmp36 = _tmp35 + tmp34
        _tmp35 = tl.where(rmask, tmp36, _tmp35)
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
    tl.store(in_out_ptr1 + (x0), tmp44, None)
