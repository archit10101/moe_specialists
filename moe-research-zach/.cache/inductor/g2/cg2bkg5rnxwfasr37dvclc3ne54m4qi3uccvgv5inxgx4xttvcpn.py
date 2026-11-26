
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*i64', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_unsqueeze_38', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048)
    x3 = xindex % 2048
    x1 = (xindex // 32) % 64
    tmp0 = tl.load(in_ptr0 + (x3 + (32768*x2)), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr2 + (x3 + (32768*x2)), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.full([1], -2, tl.int64)
    tmp4 = tmp2 == tmp3
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tmp2 >= tmp5
    tmp7 = tmp4 & tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = 0.99609375
    tmp11 = 1 + x1
    tmp12 = tmp11.to(tl.float32)
    tmp13 = libdevice.pow(tmp10, tmp12)
    tmp14 = tmp9 * tmp13
    tmp15 = 0.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp1 + tmp16
    tmp19 = tmp18.to(tl.float32)
    tmp20 = 1.0
    tmp21 = libdevice.pow(tmp20, tmp12)
    tmp22 = tmp9 * tmp21
    tmp23 = tmp22 * tmp15
    tmp24 = tmp19 + tmp23
    tl.store(out_ptr0 + (x3 + (32768*x2)), tmp17, None)
    tl.store(out_ptr1 + (x3 + (32768*x2)), tmp24, None)
