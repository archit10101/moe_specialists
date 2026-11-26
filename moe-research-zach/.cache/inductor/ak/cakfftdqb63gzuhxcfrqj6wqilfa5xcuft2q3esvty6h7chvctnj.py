
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i1', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_unsqueeze_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 4096)
    x3 = xindex % 4096
    x1 = (xindex // 64) % 64
    x4 = (xindex // 64)
    x0 = xindex % 64
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (65536*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (12288 + x3 + (65536*x2)), None)
    tmp14 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp17 = tl.load(in_ptr0 + (12224 + x0 + (65536*x2)), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr4 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp20 = tl.load(in_ptr2 + (63))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp23 = tl.load(in_ptr5 + (4032 + x0 + (4096*x2)), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (8192 + x3 + (65536*x2)), None)
    tmp29 = tl.load(in_ptr4 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.full([1], -2, tl.int64)
    tmp3 = tmp1 == tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp1 >= tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 0.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp0 + tmp11
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 * tmp8
    tmp19 = tmp18.to(tl.float32)
    tmp22 = tmp19 * tmp21
    tmp24 = tmp22 * tmp23
    tmp25 = tmp17 + tmp24
    tmp26 = tmp16 * tmp25
    tmp27 = tmp13 + tmp26
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 * tmp8
    tmp32 = tmp31 * tmp23
    tmp33 = tmp28 + tmp32
    tl.store(out_ptr0 + (x3 + (65536*x2)), tmp12, None)
    tl.store(out_ptr1 + (x5), tmp27, None)
    tl.store(out_ptr2 + (x3 + (65536*x2)), tmp27, None)
    tl.store(out_ptr3 + (x3 + (65536*x2)), tmp33, None)
