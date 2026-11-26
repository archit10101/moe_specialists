
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: '*fp32', 37: '*fp32', 38: '*fp32', 39: '*fp32', 40: '*fp32', 41: '*fp32', 42: '*fp32', 43: '*fp32', 44: '*fp32', 45: '*fp32', 46: '*fp32', 47: '*fp32', 48: '*fp32', 49: '*fp32', 50: '*fp32', 51: '*fp32', 52: '*fp32', 53: '*fp32', 54: '*fp32', 55: '*fp32', 56: '*fp32', 57: '*fp32', 58: '*fp32', 59: '*fp32', 60: '*fp32', 61: '*fp32', 62: '*fp32', 63: '*fp32', 64: 'i32', 65: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_select_backward_sum_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 105, 'num_reduction': 30, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 32
    x1 = (xindex // 32)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (92224 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (960 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp5 = tl.load(in_ptr2 + (960 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr0 + (86080 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp14 = tl.load(in_ptr1 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp18 = tl.load(in_ptr2 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr4 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr0 + (79936 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp33 = tl.load(in_ptr1 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp37 = tl.load(in_ptr2 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.load(in_ptr5 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.load(in_ptr0 + (73792 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp48 = tl.load(in_ptr1 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp52 = tl.load(in_ptr2 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr6 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.load(in_ptr0 + (67648 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp63 = tl.load(in_ptr1 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp67 = tl.load(in_ptr2 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp71 = tl.load(in_ptr7 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp77 = tl.load(in_ptr0 + (61504 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp78 = tl.load(in_ptr1 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp82 = tl.load(in_ptr2 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp86 = tl.load(in_ptr8 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp92 = tl.load(in_ptr0 + (55360 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp93 = tl.load(in_ptr1 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp97 = tl.load(in_ptr2 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp101 = tl.load(in_ptr9 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp107 = tl.load(in_ptr0 + (49216 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp108 = tl.load(in_ptr1 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp112 = tl.load(in_ptr2 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp116 = tl.load(in_ptr10 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp122 = tl.load(in_ptr0 + (43072 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp123 = tl.load(in_ptr1 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp127 = tl.load(in_ptr2 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp131 = tl.load(in_ptr11 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp137 = tl.load(in_ptr0 + (36928 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp138 = tl.load(in_ptr1 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp142 = tl.load(in_ptr2 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp146 = tl.load(in_ptr12 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp152 = tl.load(in_ptr0 + (30784 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp153 = tl.load(in_ptr1 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp157 = tl.load(in_ptr2 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp161 = tl.load(in_ptr13 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp167 = tl.load(in_ptr0 + (24640 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp168 = tl.load(in_ptr1 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp172 = tl.load(in_ptr2 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp176 = tl.load(in_ptr14 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp182 = tl.load(in_ptr0 + (18496 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp183 = tl.load(in_ptr1 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp187 = tl.load(in_ptr2 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp191 = tl.load(in_ptr15 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp197 = tl.load(in_ptr0 + (12352 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp198 = tl.load(in_ptr1 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp202 = tl.load(in_ptr2 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp206 = tl.load(in_ptr16 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp212 = tl.load(in_ptr0 + (6208 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp213 = tl.load(in_ptr1 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp217 = tl.load(in_ptr2 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp221 = tl.load(in_ptr17 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp227 = tl.load(in_ptr0 + (92192 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp229 = tl.load(in_ptr18 + (960 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp231 = tl.load(in_ptr19 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp237 = tl.load(in_ptr0 + (86048 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp239 = tl.load(in_ptr18 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp243 = tl.load(in_ptr20 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp249 = tl.load(in_ptr0 + (79904 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp251 = tl.load(in_ptr18 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp255 = tl.load(in_ptr21 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp261 = tl.load(in_ptr0 + (73760 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp263 = tl.load(in_ptr18 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp267 = tl.load(in_ptr22 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp273 = tl.load(in_ptr0 + (67616 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp275 = tl.load(in_ptr18 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp279 = tl.load(in_ptr23 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp285 = tl.load(in_ptr0 + (61472 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp287 = tl.load(in_ptr18 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp291 = tl.load(in_ptr24 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp297 = tl.load(in_ptr0 + (55328 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp299 = tl.load(in_ptr18 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp303 = tl.load(in_ptr25 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp309 = tl.load(in_ptr0 + (49184 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp311 = tl.load(in_ptr18 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp315 = tl.load(in_ptr26 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp321 = tl.load(in_ptr0 + (43040 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp323 = tl.load(in_ptr18 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp327 = tl.load(in_ptr27 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp333 = tl.load(in_ptr0 + (36896 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp335 = tl.load(in_ptr18 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp339 = tl.load(in_ptr28 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp345 = tl.load(in_ptr0 + (30752 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp347 = tl.load(in_ptr18 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp351 = tl.load(in_ptr29 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp357 = tl.load(in_ptr0 + (24608 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp359 = tl.load(in_ptr18 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp363 = tl.load(in_ptr30 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp369 = tl.load(in_ptr0 + (18464 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp371 = tl.load(in_ptr18 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp375 = tl.load(in_ptr31 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp381 = tl.load(in_ptr0 + (12320 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp383 = tl.load(in_ptr18 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp387 = tl.load(in_ptr32 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp393 = tl.load(in_ptr0 + (6176 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp395 = tl.load(in_ptr18 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp399 = tl.load(in_ptr33 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp0 * tmp3
    tmp6 = tmp4 / tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None]
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp13 * tmp16
    tmp19 = tmp17 / tmp18
    tmp20 = r2
    tmp21 = tl.full([1, 1], 63, tl.int32)
    tmp22 = tmp20 == tmp21
    tmp23 = 0.0
    tmp24 = tl.where(tmp22, tmp12, tmp23)
    tmp25 = tmp19 + tmp24
    tmp27 = tmp25 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.where(xmask, tmp28, 0)
    tmp31 = tl.sum(tmp30, 1)[:, None]
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp32 * tmp35
    tmp38 = tmp36 / tmp37
    tmp39 = tl.where(tmp22, tmp31, tmp23)
    tmp40 = tmp38 + tmp39
    tmp42 = tmp40 * tmp41
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, RBLOCK])
    tmp45 = tl.where(xmask, tmp43, 0)
    tmp46 = tl.sum(tmp45, 1)[:, None]
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp47 * tmp50
    tmp53 = tmp51 / tmp52
    tmp54 = tl.where(tmp22, tmp46, tmp23)
    tmp55 = tmp53 + tmp54
    tmp57 = tmp55 * tmp56
    tmp58 = tl.broadcast_to(tmp57, [XBLOCK, RBLOCK])
    tmp60 = tl.where(xmask, tmp58, 0)
    tmp61 = tl.sum(tmp60, 1)[:, None]
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tmp62 * tmp65
    tmp68 = tmp66 / tmp67
    tmp69 = tl.where(tmp22, tmp61, tmp23)
    tmp70 = tmp68 + tmp69
    tmp72 = tmp70 * tmp71
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp75 = tl.where(xmask, tmp73, 0)
    tmp76 = tl.sum(tmp75, 1)[:, None]
    tmp79 = tmp78.to(tl.float32)
    tmp80 = tmp79.to(tl.float32)
    tmp81 = tmp77 * tmp80
    tmp83 = tmp81 / tmp82
    tmp84 = tl.where(tmp22, tmp76, tmp23)
    tmp85 = tmp83 + tmp84
    tmp87 = tmp85 * tmp86
    tmp88 = tl.broadcast_to(tmp87, [XBLOCK, RBLOCK])
    tmp90 = tl.where(xmask, tmp88, 0)
    tmp91 = tl.sum(tmp90, 1)[:, None]
    tmp94 = tmp93.to(tl.float32)
    tmp95 = tmp94.to(tl.float32)
    tmp96 = tmp92 * tmp95
    tmp98 = tmp96 / tmp97
    tmp99 = tl.where(tmp22, tmp91, tmp23)
    tmp100 = tmp98 + tmp99
    tmp102 = tmp100 * tmp101
    tmp103 = tl.broadcast_to(tmp102, [XBLOCK, RBLOCK])
    tmp105 = tl.where(xmask, tmp103, 0)
    tmp106 = tl.sum(tmp105, 1)[:, None]
    tmp109 = tmp108.to(tl.float32)
    tmp110 = tmp109.to(tl.float32)
    tmp111 = tmp107 * tmp110
    tmp113 = tmp111 / tmp112
    tmp114 = tl.where(tmp22, tmp106, tmp23)
    tmp115 = tmp113 + tmp114
    tmp117 = tmp115 * tmp116
    tmp118 = tl.broadcast_to(tmp117, [XBLOCK, RBLOCK])
    tmp120 = tl.where(xmask, tmp118, 0)
    tmp121 = tl.sum(tmp120, 1)[:, None]
    tmp124 = tmp123.to(tl.float32)
    tmp125 = tmp124.to(tl.float32)
    tmp126 = tmp122 * tmp125
    tmp128 = tmp126 / tmp127
    tmp129 = tl.where(tmp22, tmp121, tmp23)
    tmp130 = tmp128 + tmp129
    tmp132 = tmp130 * tmp131
    tmp133 = tl.broadcast_to(tmp132, [XBLOCK, RBLOCK])
    tmp135 = tl.where(xmask, tmp133, 0)
    tmp136 = tl.sum(tmp135, 1)[:, None]
    tmp139 = tmp138.to(tl.float32)
    tmp140 = tmp139.to(tl.float32)
    tmp141 = tmp137 * tmp140
    tmp143 = tmp141 / tmp142
    tmp144 = tl.where(tmp22, tmp136, tmp23)
    tmp145 = tmp143 + tmp144
    tmp147 = tmp145 * tmp146
    tmp148 = tl.broadcast_to(tmp147, [XBLOCK, RBLOCK])
    tmp150 = tl.where(xmask, tmp148, 0)
    tmp151 = tl.sum(tmp150, 1)[:, None]
    tmp154 = tmp153.to(tl.float32)
    tmp155 = tmp154.to(tl.float32)
    tmp156 = tmp152 * tmp155
    tmp158 = tmp156 / tmp157
    tmp159 = tl.where(tmp22, tmp151, tmp23)
    tmp160 = tmp158 + tmp159
    tmp162 = tmp160 * tmp161
    tmp163 = tl.broadcast_to(tmp162, [XBLOCK, RBLOCK])
    tmp165 = tl.where(xmask, tmp163, 0)
    tmp166 = tl.sum(tmp165, 1)[:, None]
    tmp169 = tmp168.to(tl.float32)
    tmp170 = tmp169.to(tl.float32)
    tmp171 = tmp167 * tmp170
    tmp173 = tmp171 / tmp172
    tmp174 = tl.where(tmp22, tmp166, tmp23)
    tmp175 = tmp173 + tmp174
    tmp177 = tmp175 * tmp176
    tmp178 = tl.broadcast_to(tmp177, [XBLOCK, RBLOCK])
    tmp180 = tl.where(xmask, tmp178, 0)
    tmp181 = tl.sum(tmp180, 1)[:, None]
    tmp184 = tmp183.to(tl.float32)
    tmp185 = tmp184.to(tl.float32)
    tmp186 = tmp182 * tmp185
    tmp188 = tmp186 / tmp187
    tmp189 = tl.where(tmp22, tmp181, tmp23)
    tmp190 = tmp188 + tmp189
    tmp192 = tmp190 * tmp191
    tmp193 = tl.broadcast_to(tmp192, [XBLOCK, RBLOCK])
    tmp195 = tl.where(xmask, tmp193, 0)
    tmp196 = tl.sum(tmp195, 1)[:, None]
    tmp199 = tmp198.to(tl.float32)
    tmp200 = tmp199.to(tl.float32)
    tmp201 = tmp197 * tmp200
    tmp203 = tmp201 / tmp202
    tmp204 = tl.where(tmp22, tmp196, tmp23)
    tmp205 = tmp203 + tmp204
    tmp207 = tmp205 * tmp206
    tmp208 = tl.broadcast_to(tmp207, [XBLOCK, RBLOCK])
    tmp210 = tl.where(xmask, tmp208, 0)
    tmp211 = tl.sum(tmp210, 1)[:, None]
    tmp214 = tmp213.to(tl.float32)
    tmp215 = tmp214.to(tl.float32)
    tmp216 = tmp212 * tmp215
    tmp218 = tmp216 / tmp217
    tmp219 = tl.where(tmp22, tmp211, tmp23)
    tmp220 = tmp218 + tmp219
    tmp222 = tmp220 * tmp221
    tmp223 = tl.broadcast_to(tmp222, [XBLOCK, RBLOCK])
    tmp225 = tl.where(xmask, tmp223, 0)
    tmp226 = tl.sum(tmp225, 1)[:, None]
    tmp228 = tmp227 * tmp3
    tmp230 = tmp228 / tmp229
    tmp232 = tmp230 * tmp231
    tmp233 = tl.broadcast_to(tmp232, [XBLOCK, RBLOCK])
    tmp235 = tl.where(xmask, tmp233, 0)
    tmp236 = tl.sum(tmp235, 1)[:, None]
    tmp238 = tmp237 * tmp16
    tmp240 = tmp238 / tmp239
    tmp241 = tl.where(tmp22, tmp236, tmp23)
    tmp242 = tmp240 + tmp241
    tmp244 = tmp242 * tmp243
    tmp245 = tl.broadcast_to(tmp244, [XBLOCK, RBLOCK])
    tmp247 = tl.where(xmask, tmp245, 0)
    tmp248 = tl.sum(tmp247, 1)[:, None]
    tmp250 = tmp249 * tmp35
    tmp252 = tmp250 / tmp251
    tmp253 = tl.where(tmp22, tmp248, tmp23)
    tmp254 = tmp252 + tmp253
    tmp256 = tmp254 * tmp255
    tmp257 = tl.broadcast_to(tmp256, [XBLOCK, RBLOCK])
    tmp259 = tl.where(xmask, tmp257, 0)
    tmp260 = tl.sum(tmp259, 1)[:, None]
    tmp262 = tmp261 * tmp50
    tmp264 = tmp262 / tmp263
    tmp265 = tl.where(tmp22, tmp260, tmp23)
    tmp266 = tmp264 + tmp265
    tmp268 = tmp266 * tmp267
    tmp269 = tl.broadcast_to(tmp268, [XBLOCK, RBLOCK])
    tmp271 = tl.where(xmask, tmp269, 0)
    tmp272 = tl.sum(tmp271, 1)[:, None]
    tmp274 = tmp273 * tmp65
    tmp276 = tmp274 / tmp275
    tmp277 = tl.where(tmp22, tmp272, tmp23)
    tmp278 = tmp276 + tmp277
    tmp280 = tmp278 * tmp279
    tmp281 = tl.broadcast_to(tmp280, [XBLOCK, RBLOCK])
    tmp283 = tl.where(xmask, tmp281, 0)
    tmp284 = tl.sum(tmp283, 1)[:, None]
    tmp286 = tmp285 * tmp80
    tmp288 = tmp286 / tmp287
    tmp289 = tl.where(tmp22, tmp284, tmp23)
    tmp290 = tmp288 + tmp289
    tmp292 = tmp290 * tmp291
    tmp293 = tl.broadcast_to(tmp292, [XBLOCK, RBLOCK])
    tmp295 = tl.where(xmask, tmp293, 0)
    tmp296 = tl.sum(tmp295, 1)[:, None]
    tmp298 = tmp297 * tmp95
    tmp300 = tmp298 / tmp299
    tmp301 = tl.where(tmp22, tmp296, tmp23)
    tmp302 = tmp300 + tmp301
    tmp304 = tmp302 * tmp303
    tmp305 = tl.broadcast_to(tmp304, [XBLOCK, RBLOCK])
    tmp307 = tl.where(xmask, tmp305, 0)
    tmp308 = tl.sum(tmp307, 1)[:, None]
    tmp310 = tmp309 * tmp110
    tmp312 = tmp310 / tmp311
    tmp313 = tl.where(tmp22, tmp308, tmp23)
    tmp314 = tmp312 + tmp313
    tmp316 = tmp314 * tmp315
    tmp317 = tl.broadcast_to(tmp316, [XBLOCK, RBLOCK])
    tmp319 = tl.where(xmask, tmp317, 0)
    tmp320 = tl.sum(tmp319, 1)[:, None]
    tmp322 = tmp321 * tmp125
    tmp324 = tmp322 / tmp323
    tmp325 = tl.where(tmp22, tmp320, tmp23)
    tmp326 = tmp324 + tmp325
    tmp328 = tmp326 * tmp327
    tmp329 = tl.broadcast_to(tmp328, [XBLOCK, RBLOCK])
    tmp331 = tl.where(xmask, tmp329, 0)
    tmp332 = tl.sum(tmp331, 1)[:, None]
    tmp334 = tmp333 * tmp140
    tmp336 = tmp334 / tmp335
    tmp337 = tl.where(tmp22, tmp332, tmp23)
    tmp338 = tmp336 + tmp337
    tmp340 = tmp338 * tmp339
    tmp341 = tl.broadcast_to(tmp340, [XBLOCK, RBLOCK])
    tmp343 = tl.where(xmask, tmp341, 0)
    tmp344 = tl.sum(tmp343, 1)[:, None]
    tmp346 = tmp345 * tmp155
    tmp348 = tmp346 / tmp347
    tmp349 = tl.where(tmp22, tmp344, tmp23)
    tmp350 = tmp348 + tmp349
    tmp352 = tmp350 * tmp351
    tmp353 = tl.broadcast_to(tmp352, [XBLOCK, RBLOCK])
    tmp355 = tl.where(xmask, tmp353, 0)
    tmp356 = tl.sum(tmp355, 1)[:, None]
    tmp358 = tmp357 * tmp170
    tmp360 = tmp358 / tmp359
    tmp361 = tl.where(tmp22, tmp356, tmp23)
    tmp362 = tmp360 + tmp361
    tmp364 = tmp362 * tmp363
    tmp365 = tl.broadcast_to(tmp364, [XBLOCK, RBLOCK])
    tmp367 = tl.where(xmask, tmp365, 0)
    tmp368 = tl.sum(tmp367, 1)[:, None]
    tmp370 = tmp369 * tmp185
    tmp372 = tmp370 / tmp371
    tmp373 = tl.where(tmp22, tmp368, tmp23)
    tmp374 = tmp372 + tmp373
    tmp376 = tmp374 * tmp375
    tmp377 = tl.broadcast_to(tmp376, [XBLOCK, RBLOCK])
    tmp379 = tl.where(xmask, tmp377, 0)
    tmp380 = tl.sum(tmp379, 1)[:, None]
    tmp382 = tmp381 * tmp200
    tmp384 = tmp382 / tmp383
    tmp385 = tl.where(tmp22, tmp380, tmp23)
    tmp386 = tmp384 + tmp385
    tmp388 = tmp386 * tmp387
    tmp389 = tl.broadcast_to(tmp388, [XBLOCK, RBLOCK])
    tmp391 = tl.where(xmask, tmp389, 0)
    tmp392 = tl.sum(tmp391, 1)[:, None]
    tmp394 = tmp393 * tmp215
    tmp396 = tmp394 / tmp395
    tmp397 = tl.where(tmp22, tmp392, tmp23)
    tmp398 = tmp396 + tmp397
    tmp400 = tmp398 * tmp399
    tmp401 = tl.broadcast_to(tmp400, [XBLOCK, RBLOCK])
    tmp403 = tl.where(xmask, tmp401, 0)
    tmp404 = tl.sum(tmp403, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tl.store(out_ptr1 + (x3), tmp31, xmask)
    tl.store(out_ptr2 + (x3), tmp46, xmask)
    tl.store(out_ptr3 + (x3), tmp61, xmask)
    tl.store(out_ptr4 + (x3), tmp76, xmask)
    tl.store(out_ptr5 + (x3), tmp91, xmask)
    tl.store(out_ptr6 + (x3), tmp106, xmask)
    tl.store(out_ptr7 + (x3), tmp121, xmask)
    tl.store(out_ptr8 + (x3), tmp136, xmask)
    tl.store(out_ptr9 + (x3), tmp151, xmask)
    tl.store(out_ptr10 + (x3), tmp166, xmask)
    tl.store(out_ptr11 + (x3), tmp181, xmask)
    tl.store(out_ptr12 + (x3), tmp196, xmask)
    tl.store(out_ptr13 + (x3), tmp211, xmask)
    tl.store(out_ptr14 + (x3), tmp226, xmask)
    tl.store(out_ptr15 + (x3), tmp236, xmask)
    tl.store(out_ptr16 + (x3), tmp248, xmask)
    tl.store(out_ptr17 + (x3), tmp260, xmask)
    tl.store(out_ptr18 + (x3), tmp272, xmask)
    tl.store(out_ptr19 + (x3), tmp284, xmask)
    tl.store(out_ptr20 + (x3), tmp296, xmask)
    tl.store(out_ptr21 + (x3), tmp308, xmask)
    tl.store(out_ptr22 + (x3), tmp320, xmask)
    tl.store(out_ptr23 + (x3), tmp332, xmask)
    tl.store(out_ptr24 + (x3), tmp344, xmask)
    tl.store(out_ptr25 + (x3), tmp356, xmask)
    tl.store(out_ptr26 + (x3), tmp368, xmask)
    tl.store(out_ptr27 + (x3), tmp380, xmask)
    tl.store(out_ptr28 + (x3), tmp392, xmask)
    tl.store(out_ptr29 + (x3), tmp404, xmask)
