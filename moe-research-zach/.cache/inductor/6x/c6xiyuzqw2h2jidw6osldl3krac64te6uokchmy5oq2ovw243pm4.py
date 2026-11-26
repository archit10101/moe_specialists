
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_28', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x4 = (xindex // 21)
    x0 = xindex % 21
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (2688 + x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2667 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (63 + (64*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1323 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (63 + (1024*x2)), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (1344 + x3 + (21504*x2)), None)
    tmp24 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (2688 + x3 + (21504*x2)), None)
    tmp28 = tl.load(in_ptr5 + (x4), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr4 + (2667 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (63 + (64*x2)), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr4 + (1323 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (1344 + x3 + (21504*x2)), None)
    tmp42 = tl.load(in_ptr6 + (x4), None, eviction_policy='evict_last')
    tmp6 = tl.full([1], -2, tl.int64)
    tmp7 = tmp5 == tmp6
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tmp5 >= tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 0.9945994019508362
    tmp13 = 64.0
    tmp14 = libdevice.pow(tmp12, tmp13)
    tmp15 = tmp11 * tmp14
    tmp16 = 0.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp4 + tmp17
    tmp19 = tmp3 * tmp18
    tmp20 = tmp2 + tmp19
    tmp21 = tmp1 * tmp20
    tmp22 = tmp0 + tmp21
    tmp25 = tmp24 * tmp18
    tmp26 = tmp23 + tmp25
    tmp32 = 0.999323308467865
    tmp33 = libdevice.pow(tmp32, tmp13)
    tmp34 = tmp11 * tmp33
    tmp35 = tmp34 * tmp16
    tmp36 = tmp31 + tmp35
    tmp37 = tmp30 * tmp36
    tmp38 = tmp29 + tmp37
    tmp39 = tmp28 * tmp38
    tmp40 = tmp27 + tmp39
    tmp43 = tmp42 * tmp36
    tmp44 = tmp41 + tmp43
    tl.store(out_ptr0 + (x5), tmp22, None)
    tl.store(out_ptr1 + (x3 + (21504*x2)), tmp22, None)
    tl.store(out_ptr2 + (x3 + (21504*x2)), tmp26, None)
    tl.store(out_ptr3 + (x5), tmp40, None)
    tl.store(out_ptr4 + (x3 + (21504*x2)), tmp40, None)
    tl.store(out_ptr5 + (x3 + (21504*x2)), tmp44, None)
