
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*i64', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_cat_mul_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 64
    x1 = (xindex // 64) % 4
    x2 = (xindex // 256)
    x4 = xindex
    x5 = xindex % 256
    tmp15 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (x5 + (768*x2)), None).to(tl.float32)
    tmp46 = tl.load(in_ptr0 + (256 + x5 + (768*x2)), None).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32 + (64*x1) + (768*x2) + x0), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = -tmp5
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp4, tmp6, tmp7)
    tmp9 = tmp0 >= tmp3
    tmp10 = tl.full([1], 64, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tl.load(in_ptr0 + ((64*x1) + (768*x2) + ((-32) + x0)), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tl.where(tmp4, tmp8, tmp12)
    tmp14 = tmp13.to(tl.float32)
    tmp16 = tmp15.to(tl.float64)
    tmp17 = 2*(x0 % 32)
    tmp18 = tmp17.to(tl.float64)
    tmp19 = tl.full([1], 0.015625, tl.float64)
    tmp20 = tmp18 * tmp19
    tmp21 = tl.full([1], 10000.0, tl.float64)
    tmp22 = libdevice.pow(tmp21, tmp20)
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = tmp23 / tmp22
    tmp25 = tl.full([1], 1.0, tl.float64)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp16 * tmp26
    tmp28 = libdevice.sin(tmp27)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp14 * tmp29
    tmp32 = tmp31.to(tl.float32)
    tmp33 = libdevice.cos(tmp27)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp32 * tmp34
    tmp36 = tmp35 + tmp30
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tl.load(in_ptr0 + (288 + (64*x1) + (768*x2) + x0), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = -tmp38
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp4, tmp39, tmp40)
    tmp42 = tl.load(in_ptr0 + (256 + (64*x1) + (768*x2) + ((-32) + x0)), tmp9, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp43 = tl.where(tmp4, tmp41, tmp42)
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp44 * tmp29
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp47 * tmp34
    tmp49 = tmp48 + tmp45
    tmp50 = tmp49.to(tl.float32)
    tl.store(out_ptr1 + (x4), tmp37, None)
    tl.store(out_ptr3 + (x4), tmp50, None)
