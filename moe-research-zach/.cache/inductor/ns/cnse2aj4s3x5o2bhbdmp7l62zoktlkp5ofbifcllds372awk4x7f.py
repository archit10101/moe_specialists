
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 32768)
    x0 = xindex % 32768
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 63, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 42, tl.int64)
    tmp6 = tmp0 >= tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.load(in_ptr0 + ((-42) + (21*x0) + x1), tmp7, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.0
    tmp10 = tl.where(tmp6, tmp8, tmp9)
    tmp11 = tl.full([1], 21, tl.int64)
    tmp12 = tmp0 >= tmp11
    tmp13 = tmp0 < tmp5
    tmp14 = tmp12 & tmp13
    tmp15 = tmp14 & tmp4
    tmp16 = tl.load(in_ptr1 + ((-21) + (21*x0) + x1), tmp15, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.where(tmp14, tmp16, tmp9)
    tmp18 = tmp10 + tmp17
    tmp19 = tmp0 < tmp11
    tmp20 = tmp19 & tmp4
    tmp21 = tl.load(in_ptr2 + ((21*x0) + x1), tmp20, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.where(tmp19, tmp21, tmp9)
    tmp23 = tmp18 + tmp22
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp4, tmp23, tmp24)
    tmp26 = tmp0 >= tmp3
    tmp27 = tl.full([1], 64, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp30 = tl.where(tmp26, tmp9, tmp29)
    tmp31 = tl.where(tmp4, tmp25, tmp30)
    tl.store(out_ptr0 + (x2), tmp31, None)
