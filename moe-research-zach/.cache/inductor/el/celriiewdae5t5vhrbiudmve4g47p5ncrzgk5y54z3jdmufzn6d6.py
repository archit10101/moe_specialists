
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_bitwise_and_clamp_eq_ge_mul_pow_sub_where_zeros_29', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 64
    x3 = (xindex // 64)
    x2 = (xindex // 4096)
    x4 = xindex
    tmp9 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp0 = x1 + ((-1)*x0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.0
    tmp3 = tmp1 >= tmp2
    tmp4 = 0.99609375
    tmp5 = ((0.0) * ((0.0) >= (x1 + ((-1)*x0))) + (x1 + ((-1)*x0)) * ((x1 + ((-1)*x0)) > (0.0)))
    tmp6 = tmp5.to(tl.float32)
    tmp7 = libdevice.pow(tmp4, tmp6)
    tmp8 = tl.where(tmp3, tmp7, tmp2)
    tmp11 = tmp9 == tmp10
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = tmp9 >= tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp8 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 1.0
    tmp20 = libdevice.pow(tmp19, tmp6)
    tmp21 = tl.where(tmp3, tmp20, tmp2)
    tmp22 = tmp21 * tmp16
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp18, None)
    tl.store(out_ptr1 + (x4), tmp23, None)
