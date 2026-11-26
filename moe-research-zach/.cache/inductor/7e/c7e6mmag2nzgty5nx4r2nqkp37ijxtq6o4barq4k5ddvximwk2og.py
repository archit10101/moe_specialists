
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_neg_slice_backward_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 64
    x2 = (xindex // 384)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr3 + (x3), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = x0
    tmp6 = tl.full([1], 32, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = tl.load(in_ptr0 + (32 + x3), tmp7, other=0.0).to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (32 + x0 + (64*x2)), tmp7, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp7, tmp12, tmp13)
    tmp15 = 0.0
    tmp16 = tl.where(tmp7, tmp14, tmp15)
    tmp17 = tmp4 + tmp16
    tmp18 = tmp5 >= tmp6
    tmp19 = tl.load(in_ptr0 + ((-32) + x3), tmp18, other=0.0).to(tl.float32)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tl.load(in_ptr2 + ((-32) + x0 + (64*x2)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = -tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp18, tmp24, tmp25)
    tmp27 = tl.where(tmp18, tmp26, tmp15)
    tmp28 = tmp17 + tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp2
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.load(in_ptr3 + (32 + x3), tmp7, other=0.0).to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp34 * tmp10
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp7, tmp36, tmp37)
    tmp39 = tl.where(tmp7, tmp38, tmp15)
    tmp40 = tmp32 + tmp39
    tmp41 = tl.load(in_ptr3 + ((-32) + x3), tmp18, other=0.0).to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp42 * tmp21
    tmp44 = tmp43.to(tl.float32)
    tmp45 = -tmp44
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp18, tmp45, tmp46)
    tmp48 = tl.where(tmp18, tmp47, tmp15)
    tmp49 = tmp40 + tmp48
    tl.store(out_ptr0 + (x3), tmp28, None)
    tl.store(out_ptr1 + (x3), tmp49, None)
