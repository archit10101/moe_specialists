
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*i64', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_neg_slice_backward_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x5 = xindex
    x2 = (xindex // 256)
    x0 = xindex % 64
    x1 = (xindex // 64) % 4
    x3 = (xindex // 256) % 1024
    x4 = (xindex // 262144)
    tmp0 = tl.load(in_ptr0 + (x5), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr2 + (x5), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float64)
    tmp4 = 2*(x0 % 32)
    tmp5 = tmp4.to(tl.float64)
    tmp6 = tl.full([1], 0.015625, tl.float64)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full([1], 10000.0, tl.float64)
    tmp9 = libdevice.pow(tmp8, tmp7)
    tmp10 = tl.full([1], 1, tl.int32)
    tmp11 = tmp10 / tmp9
    tmp12 = tl.full([1], 1.0, tl.float64)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp3 * tmp13
    tmp15 = libdevice.cos(tmp14)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp1 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = x0
    tmp20 = tl.full([1], 32, tl.int64)
    tmp21 = tmp19 < tmp20
    tmp22 = tl.load(in_ptr0 + (32 + x5), tmp21, other=0.0).to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tl.load(in_ptr1 + (x2), tmp21, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24.to(tl.float64)
    tmp26 = tmp25 * tmp13
    tmp27 = libdevice.sin(tmp26)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp23 * tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp21, tmp30, tmp31)
    tmp33 = 0.0
    tmp34 = tl.where(tmp21, tmp32, tmp33)
    tmp35 = tmp18 + tmp34
    tmp36 = tmp19 >= tmp20
    tmp37 = tl.load(in_ptr0 + ((-32) + x5), tmp36, other=0.0).to(tl.float32)
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.load(in_ptr1 + (x2), tmp36, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp39.to(tl.float64)
    tmp41 = tmp40 * tmp13
    tmp42 = libdevice.sin(tmp41)
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp38 * tmp43
    tmp45 = tmp44.to(tl.float32)
    tmp46 = -tmp45
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp36, tmp46, tmp47)
    tmp49 = tl.where(tmp36, tmp48, tmp33)
    tmp50 = tmp35 + tmp49
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tmp52 * tmp16
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tl.load(in_ptr2 + (32 + x5), tmp21, other=0.0).to(tl.float32)
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tmp56 * tmp28
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tl.full(tmp58.shape, 0.0, tmp58.dtype)
    tmp60 = tl.where(tmp21, tmp58, tmp59)
    tmp61 = tl.where(tmp21, tmp60, tmp33)
    tmp62 = tmp54 + tmp61
    tmp63 = tl.load(in_ptr2 + ((-32) + x5), tmp36, other=0.0).to(tl.float32)
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp64 * tmp43
    tmp66 = tmp65.to(tl.float32)
    tmp67 = -tmp66
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp36, tmp67, tmp68)
    tmp70 = tl.where(tmp36, tmp69, tmp33)
    tmp71 = tmp62 + tmp70
    tl.store(out_ptr1 + (x0 + (64*x3) + (65536*x1) + (262144*x4)), tmp50, None)
    tl.store(out_ptr3 + (x0 + (64*x3) + (65536*x1) + (262144*x4)), tmp71, None)
