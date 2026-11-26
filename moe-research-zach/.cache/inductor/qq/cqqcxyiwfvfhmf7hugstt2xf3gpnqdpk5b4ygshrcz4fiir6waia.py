
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x1 = (xindex // 32) % 64
    x0 = xindex % 32
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (26624 + x3 + (32768*x2)), None)
    tmp1 = tl.load(in_ptr1 + (832 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (831 + (1024*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (26592 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (767 + (1024*x2)), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr3 + (26624 + x3 + (32768*x2)), None)
    tmp31 = tl.load(in_ptr3 + (26592 + x0 + (32768*x2)), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (2016 + x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (24576 + x3 + (32768*x2)), None)
    tmp40 = tl.load(in_ptr1 + (768 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr3 + (24576 + x3 + (32768*x2)), None)
    tmp3 = tmp1 == tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp1 >= tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 0.8408964276313782
    tmp9 = 1 + x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = libdevice.pow(tmp8, tmp10)
    tmp12 = tmp7 * tmp11
    tmp15 = tmp2 == tmp14
    tmp16 = tmp2 >= tmp4
    tmp17 = tmp15 & tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 64.0
    tmp20 = libdevice.pow(tmp8, tmp19)
    tmp21 = tmp18 * tmp20
    tmp23 = tmp21 * tmp22
    tmp24 = tmp13 + tmp23
    tmp25 = tmp12 * tmp24
    tmp26 = tmp0 + tmp25
    tmp28 = 0.9892280101776123
    tmp29 = libdevice.pow(tmp28, tmp10)
    tmp30 = tmp7 * tmp29
    tmp32 = libdevice.pow(tmp28, tmp19)
    tmp33 = tmp18 * tmp32
    tmp35 = tmp33 * tmp34
    tmp36 = tmp31 + tmp35
    tmp37 = tmp30 * tmp36
    tmp38 = tmp27 + tmp37
    tmp41 = tmp40 == tmp14
    tmp42 = tmp40 >= tmp4
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp44 * tmp11
    tmp46 = tmp45 * tmp22
    tmp47 = tmp39 + tmp46
    tmp49 = tmp44 * tmp29
    tmp50 = tmp49 * tmp34
    tmp51 = tmp48 + tmp50
    tl.store(out_ptr0 + (x4), tmp26, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp26, None)
    tl.store(out_ptr2 + (x4), tmp38, None)
    tl.store(out_ptr3 + (x3 + (32768*x2)), tmp38, None)
    tl.store(out_ptr4 + (x3 + (32768*x2)), tmp47, None)
    tl.store(out_ptr5 + (x3 + (32768*x2)), tmp51, None)
