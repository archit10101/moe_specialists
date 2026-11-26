
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_gather_logsumexp_mul_sub_sum_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 2, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 50304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (50304*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = triton_helpers.maximum(_tmp3, tmp2)
        _tmp3 = tl.where(rmask, tmp4, _tmp3)
    tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, None)
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (50304*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl_math.abs(tmp3)
        tmp8 = float("inf")
        tmp9 = tmp7 == tmp8
        tmp10 = 0.0
        tmp11 = tl.where(tmp9, tmp10, tmp3)
        tmp12 = tmp6 - tmp11
        tmp13 = tl_math.exp(tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp15, None)
    tmp17 = tl.load(in_ptr1 + (4*x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr2 + (4*x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr1 + (1 + (4*x0)), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr2 + (1 + (4*x0)), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr1 + (2 + (4*x0)), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr2 + (2 + (4*x0)), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr1 + (3 + (4*x0)), None, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr2 + (3 + (4*x0)), None, eviction_policy='evict_last')
    tmp19 = tl.full([XBLOCK, 1], 50304, tl.int32)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp18 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp18)
    tl.device_assert((0 <= tmp22) & (tmp22 < 50304), "index out of bounds: 0 <= tmp22 < 50304")
    tmp24 = tl.load(in_ptr0 + (tmp22 + (50304*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tl_math.log(tmp15)
    tmp27 = tl_math.abs(tmp3)
    tmp28 = float("inf")
    tmp29 = tmp27 == tmp28
    tmp30 = 0.0
    tmp31 = tl.where(tmp29, tmp30, tmp3)
    tmp32 = tmp26 + tmp31
    tmp33 = tmp25 - tmp32
    tmp34 = tmp17 * tmp33
    tmp37 = tmp36 + tmp19
    tmp38 = tmp36 < 0
    tmp39 = tl.where(tmp38, tmp37, tmp36)
    tl.device_assert((0 <= tmp39) & (tmp39 < 50304), "index out of bounds: 0 <= tmp39 < 50304")
    tmp41 = tl.load(in_ptr0 + (tmp39 + (50304*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp42 - tmp32
    tmp44 = tmp35 * tmp43
    tmp45 = tmp34 + tmp44
    tmp48 = tmp47 + tmp19
    tmp49 = tmp47 < 0
    tmp50 = tl.where(tmp49, tmp48, tmp47)
    tl.device_assert((0 <= tmp50) & (tmp50 < 50304), "index out of bounds: 0 <= tmp50 < 50304")
    tmp52 = tl.load(in_ptr0 + (tmp50 + (50304*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp53 - tmp32
    tmp55 = tmp46 * tmp54
    tmp56 = tmp45 + tmp55
    tmp59 = tmp58 + tmp19
    tmp60 = tmp58 < 0
    tmp61 = tl.where(tmp60, tmp59, tmp58)
    tl.device_assert((0 <= tmp61) & (tmp61 < 50304), "index out of bounds: 0 <= tmp61 < 50304")
    tmp63 = tl.load(in_ptr0 + (tmp61 + (50304*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp64 - tmp32
    tmp66 = tmp57 * tmp65
    tmp67 = tmp56 + tmp66
    tl.store(out_ptr2 + (x0), tmp67, None)
    tl.store(out_ptr3 + (x0), tmp32, None)
