
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_new_zeros_scatter_add_2', 'mutated_arg_names': ['out_ptr0'], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (x0), None).to(tl.int1)
    tmp8 = tl.load(in_ptr3 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp12 = tl.load(in_ptr4 + (4*x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (1 + (4*x0)), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (2 + (4*x0)), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (3 + (4*x0)), None, eviction_policy='evict_last')
    tl.device_assert((0 <= tmp0) & (tmp0 < 50304), "index out of bounds: 0 <= tmp0 < 50304")
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = -tmp6
    tmp10 = tmp9 * tmp5
    tmp11 = -tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = -tmp13
    tmp16 = tmp11 * tmp15
    tmp17 = -tmp16
    tmp18 = tmp14 + tmp17
    tmp20 = tmp11 * tmp19
    tmp21 = -tmp20
    tmp22 = tmp18 + tmp21
    tmp24 = tmp11 * tmp23
    tmp25 = -tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = tmp26 + tmp6
    tl.atomic_add(out_ptr0 + (tmp0 + (50304*x0)), tmp7, None, sem='relaxed')
    tl.store(out_ptr1 + (x0), tmp27, None)
