
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*i1', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_eq_gt_remainder_rsub_zeros_like_12', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp15 = tl.load(in_ptr1 + (x2), None)
    tmp1 = 1.0
    tmp2 = tmp1 - tmp0
    tmp3 = 0.8335774540901184
    tmp4 = tmp2 > tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 1, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.load(in_ptr0 + ((-1) + x2), tmp7, other=0.0)
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 > tmp3
    tmp11 = tl.full(tmp10.shape, 0, tmp10.dtype)
    tmp12 = tl.where(tmp7, tmp10, tmp11)
    tmp13 = tl.full([1], False, tl.int1)
    tmp14 = tl.where(tmp7, tmp12, tmp13)
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 > tmp16
    tmp18 = tmp14 & tmp17
    tmp19 = tmp18 == 0
    tmp20 = tmp4 & tmp19
    tmp21 = tl.full([1], 256, tl.int64)
    tmp22 = tmp15 % tmp21
    tmp23 = tl.full([1], 0, tl.int32)
    tmp24 = tmp22 != tmp23
    tmp25 = libdevice.signbit(tmp22) if (tmp22).dtype is tl.float32 else tmp22 < 0
    tmp26 = libdevice.signbit(tmp21) if (tmp21).dtype is tl.float32 else tmp21 < 0
    tmp27 = tmp25 != tmp26
    tmp28 = tmp24 & tmp27
    tmp29 = tmp22 + tmp21
    tmp30 = tl.where(tmp28, tmp29, tmp22)
    tmp31 = tmp30 == tmp16
    tmp32 = tmp20 | tmp31
    tl.store(out_ptr0 + (x2), tmp32, None)
