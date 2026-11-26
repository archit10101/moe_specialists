
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_clamp_min_div_linalg_vector_norm_mul_sum_11', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr3 + (r1 + (32*x0)), None)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 1.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.8408964276313782
    tmp6 = libdevice.pow(tmp5, tmp4)
    tmp7 = tmp3 - tmp6
    tmp8 = 6.285213507883244
    tmp9 = tmp7 * tmp8
    tmp10 = tmp0 / tmp9
    tmp12 = tl.full([1, 1], 0, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp10 * tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.sum(tmp17, 1)[:, None]
    tmp21 = 0.9892280101776123
    tmp22 = libdevice.pow(tmp21, tmp4)
    tmp23 = tmp3 - tmp22
    tmp24 = 92.83338515052041
    tmp25 = tmp23 * tmp24
    tmp26 = tmp20 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.sum(tmp29, 1)[:, None]
    tmp32 = libdevice.sqrt(tmp19)
    tmp33 = 1e-06
    tmp34 = triton_helpers.maximum(tmp32, tmp33)
    tmp35 = tmp15 / tmp34
    tmp36 = libdevice.sqrt(tmp31)
    tmp37 = triton_helpers.maximum(tmp36, tmp33)
    tmp38 = tmp27 / tmp37
    tmp39 = tmp35 * tmp38
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp42 = tl.sum(tmp40, 1)[:, None]
    tl.store(out_ptr3 + (x0), tmp42, None)
