
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_stack_15', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 1344)
    x3 = xindex % 1344
    x4 = (xindex // 21)
    x1 = (xindex // 21) % 64
    x0 = xindex % 21
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (12096 + x3 + (21504*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (12075 + x0 + (21504*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (63 + (64*x2)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (63))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr4 + (1323 + x0 + (1344*x2)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (10752 + x3 + (21504*x2)), None)
    tmp17 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tmp5 + tmp12
    tmp14 = tmp4 * tmp13
    tmp15 = tmp0 + tmp14
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp3
    tmp20 = tmp19 * tmp11
    tmp21 = tmp16 + tmp20
    tl.store(out_ptr0 + (x5), tmp15, None)
    tl.store(out_ptr1 + (x3 + (21504*x2)), tmp15, None)
    tl.store(out_ptr2 + (x3 + (21504*x2)), tmp21, None)
