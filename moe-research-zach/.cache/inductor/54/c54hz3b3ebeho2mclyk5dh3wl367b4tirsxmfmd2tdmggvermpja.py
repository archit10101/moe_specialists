
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
    xnumel = 672
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 21
    x1 = (xindex // 21)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (60522 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (960 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (960 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + (56490 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp13 = tl.load(in_ptr1 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp16 = tl.load(in_ptr2 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr4 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr0 + (52458 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp31 = tl.load(in_ptr1 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp34 = tl.load(in_ptr2 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr5 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr0 + (48426 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp45 = tl.load(in_ptr1 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp48 = tl.load(in_ptr2 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tl.load(in_ptr6 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr0 + (44394 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp59 = tl.load(in_ptr1 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp62 = tl.load(in_ptr2 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tl.load(in_ptr7 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp72 = tl.load(in_ptr0 + (40362 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp73 = tl.load(in_ptr1 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp76 = tl.load(in_ptr2 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp80 = tl.load(in_ptr8 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp86 = tl.load(in_ptr0 + (36330 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp87 = tl.load(in_ptr1 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp90 = tl.load(in_ptr2 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp94 = tl.load(in_ptr9 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp100 = tl.load(in_ptr0 + (32298 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp101 = tl.load(in_ptr1 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp104 = tl.load(in_ptr2 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp108 = tl.load(in_ptr10 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp114 = tl.load(in_ptr0 + (28266 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp115 = tl.load(in_ptr1 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp118 = tl.load(in_ptr2 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp122 = tl.load(in_ptr11 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp128 = tl.load(in_ptr0 + (24234 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp129 = tl.load(in_ptr1 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp132 = tl.load(in_ptr2 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp136 = tl.load(in_ptr12 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp142 = tl.load(in_ptr0 + (20202 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp143 = tl.load(in_ptr1 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp146 = tl.load(in_ptr2 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp150 = tl.load(in_ptr13 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp156 = tl.load(in_ptr0 + (16170 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp157 = tl.load(in_ptr1 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp160 = tl.load(in_ptr2 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp164 = tl.load(in_ptr14 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp170 = tl.load(in_ptr0 + (12138 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp171 = tl.load(in_ptr1 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp174 = tl.load(in_ptr2 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp178 = tl.load(in_ptr15 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp184 = tl.load(in_ptr0 + (8106 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp185 = tl.load(in_ptr1 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp188 = tl.load(in_ptr2 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp192 = tl.load(in_ptr16 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp198 = tl.load(in_ptr0 + (4074 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp199 = tl.load(in_ptr1 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp202 = tl.load(in_ptr2 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp206 = tl.load(in_ptr17 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp212 = tl.load(in_ptr0 + (60501 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp214 = tl.load(in_ptr18 + (960 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp216 = tl.load(in_ptr19 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp222 = tl.load(in_ptr0 + (56469 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp224 = tl.load(in_ptr18 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp228 = tl.load(in_ptr20 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp234 = tl.load(in_ptr0 + (52437 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp236 = tl.load(in_ptr18 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp240 = tl.load(in_ptr21 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp246 = tl.load(in_ptr0 + (48405 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp248 = tl.load(in_ptr18 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp252 = tl.load(in_ptr22 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp258 = tl.load(in_ptr0 + (44373 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp260 = tl.load(in_ptr18 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp264 = tl.load(in_ptr23 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp270 = tl.load(in_ptr0 + (40341 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp272 = tl.load(in_ptr18 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp276 = tl.load(in_ptr24 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp282 = tl.load(in_ptr0 + (36309 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp284 = tl.load(in_ptr18 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp288 = tl.load(in_ptr25 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp294 = tl.load(in_ptr0 + (32277 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp296 = tl.load(in_ptr18 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp300 = tl.load(in_ptr26 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp306 = tl.load(in_ptr0 + (28245 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp308 = tl.load(in_ptr18 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp312 = tl.load(in_ptr27 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp318 = tl.load(in_ptr0 + (24213 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp320 = tl.load(in_ptr18 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp324 = tl.load(in_ptr28 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp330 = tl.load(in_ptr0 + (20181 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp332 = tl.load(in_ptr18 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp336 = tl.load(in_ptr29 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp342 = tl.load(in_ptr0 + (16149 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp344 = tl.load(in_ptr18 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp348 = tl.load(in_ptr30 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp354 = tl.load(in_ptr0 + (12117 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp356 = tl.load(in_ptr18 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp360 = tl.load(in_ptr31 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp366 = tl.load(in_ptr0 + (8085 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp368 = tl.load(in_ptr18 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp372 = tl.load(in_ptr32 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp378 = tl.load(in_ptr0 + (4053 + x0 + (63*r2) + (64512*x1)), xmask, other=0.0)
    tmp380 = tl.load(in_ptr18 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp384 = tl.load(in_ptr33 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tmp5 = tmp3 / tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp15 / tmp16
    tmp18 = r2
    tmp19 = tl.full([1, 1], 63, tl.int32)
    tmp20 = tmp18 == tmp19
    tmp21 = 0.0
    tmp22 = tl.where(tmp20, tmp11, tmp21)
    tmp23 = tmp17 + tmp22
    tmp25 = tmp23 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
    tmp28 = tl.where(xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None]
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp30 * tmp32
    tmp35 = tmp33 / tmp34
    tmp36 = tl.where(tmp20, tmp29, tmp21)
    tmp37 = tmp35 + tmp36
    tmp39 = tmp37 * tmp38
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK, RBLOCK])
    tmp42 = tl.where(xmask, tmp40, 0)
    tmp43 = tl.sum(tmp42, 1)[:, None]
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp44 * tmp46
    tmp49 = tmp47 / tmp48
    tmp50 = tl.where(tmp20, tmp43, tmp21)
    tmp51 = tmp49 + tmp50
    tmp53 = tmp51 * tmp52
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK, RBLOCK])
    tmp56 = tl.where(xmask, tmp54, 0)
    tmp57 = tl.sum(tmp56, 1)[:, None]
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp58 * tmp60
    tmp63 = tmp61 / tmp62
    tmp64 = tl.where(tmp20, tmp57, tmp21)
    tmp65 = tmp63 + tmp64
    tmp67 = tmp65 * tmp66
    tmp68 = tl.broadcast_to(tmp67, [XBLOCK, RBLOCK])
    tmp70 = tl.where(xmask, tmp68, 0)
    tmp71 = tl.sum(tmp70, 1)[:, None]
    tmp74 = tmp73.to(tl.float32)
    tmp75 = tmp72 * tmp74
    tmp77 = tmp75 / tmp76
    tmp78 = tl.where(tmp20, tmp71, tmp21)
    tmp79 = tmp77 + tmp78
    tmp81 = tmp79 * tmp80
    tmp82 = tl.broadcast_to(tmp81, [XBLOCK, RBLOCK])
    tmp84 = tl.where(xmask, tmp82, 0)
    tmp85 = tl.sum(tmp84, 1)[:, None]
    tmp88 = tmp87.to(tl.float32)
    tmp89 = tmp86 * tmp88
    tmp91 = tmp89 / tmp90
    tmp92 = tl.where(tmp20, tmp85, tmp21)
    tmp93 = tmp91 + tmp92
    tmp95 = tmp93 * tmp94
    tmp96 = tl.broadcast_to(tmp95, [XBLOCK, RBLOCK])
    tmp98 = tl.where(xmask, tmp96, 0)
    tmp99 = tl.sum(tmp98, 1)[:, None]
    tmp102 = tmp101.to(tl.float32)
    tmp103 = tmp100 * tmp102
    tmp105 = tmp103 / tmp104
    tmp106 = tl.where(tmp20, tmp99, tmp21)
    tmp107 = tmp105 + tmp106
    tmp109 = tmp107 * tmp108
    tmp110 = tl.broadcast_to(tmp109, [XBLOCK, RBLOCK])
    tmp112 = tl.where(xmask, tmp110, 0)
    tmp113 = tl.sum(tmp112, 1)[:, None]
    tmp116 = tmp115.to(tl.float32)
    tmp117 = tmp114 * tmp116
    tmp119 = tmp117 / tmp118
    tmp120 = tl.where(tmp20, tmp113, tmp21)
    tmp121 = tmp119 + tmp120
    tmp123 = tmp121 * tmp122
    tmp124 = tl.broadcast_to(tmp123, [XBLOCK, RBLOCK])
    tmp126 = tl.where(xmask, tmp124, 0)
    tmp127 = tl.sum(tmp126, 1)[:, None]
    tmp130 = tmp129.to(tl.float32)
    tmp131 = tmp128 * tmp130
    tmp133 = tmp131 / tmp132
    tmp134 = tl.where(tmp20, tmp127, tmp21)
    tmp135 = tmp133 + tmp134
    tmp137 = tmp135 * tmp136
    tmp138 = tl.broadcast_to(tmp137, [XBLOCK, RBLOCK])
    tmp140 = tl.where(xmask, tmp138, 0)
    tmp141 = tl.sum(tmp140, 1)[:, None]
    tmp144 = tmp143.to(tl.float32)
    tmp145 = tmp142 * tmp144
    tmp147 = tmp145 / tmp146
    tmp148 = tl.where(tmp20, tmp141, tmp21)
    tmp149 = tmp147 + tmp148
    tmp151 = tmp149 * tmp150
    tmp152 = tl.broadcast_to(tmp151, [XBLOCK, RBLOCK])
    tmp154 = tl.where(xmask, tmp152, 0)
    tmp155 = tl.sum(tmp154, 1)[:, None]
    tmp158 = tmp157.to(tl.float32)
    tmp159 = tmp156 * tmp158
    tmp161 = tmp159 / tmp160
    tmp162 = tl.where(tmp20, tmp155, tmp21)
    tmp163 = tmp161 + tmp162
    tmp165 = tmp163 * tmp164
    tmp166 = tl.broadcast_to(tmp165, [XBLOCK, RBLOCK])
    tmp168 = tl.where(xmask, tmp166, 0)
    tmp169 = tl.sum(tmp168, 1)[:, None]
    tmp172 = tmp171.to(tl.float32)
    tmp173 = tmp170 * tmp172
    tmp175 = tmp173 / tmp174
    tmp176 = tl.where(tmp20, tmp169, tmp21)
    tmp177 = tmp175 + tmp176
    tmp179 = tmp177 * tmp178
    tmp180 = tl.broadcast_to(tmp179, [XBLOCK, RBLOCK])
    tmp182 = tl.where(xmask, tmp180, 0)
    tmp183 = tl.sum(tmp182, 1)[:, None]
    tmp186 = tmp185.to(tl.float32)
    tmp187 = tmp184 * tmp186
    tmp189 = tmp187 / tmp188
    tmp190 = tl.where(tmp20, tmp183, tmp21)
    tmp191 = tmp189 + tmp190
    tmp193 = tmp191 * tmp192
    tmp194 = tl.broadcast_to(tmp193, [XBLOCK, RBLOCK])
    tmp196 = tl.where(xmask, tmp194, 0)
    tmp197 = tl.sum(tmp196, 1)[:, None]
    tmp200 = tmp199.to(tl.float32)
    tmp201 = tmp198 * tmp200
    tmp203 = tmp201 / tmp202
    tmp204 = tl.where(tmp20, tmp197, tmp21)
    tmp205 = tmp203 + tmp204
    tmp207 = tmp205 * tmp206
    tmp208 = tl.broadcast_to(tmp207, [XBLOCK, RBLOCK])
    tmp210 = tl.where(xmask, tmp208, 0)
    tmp211 = tl.sum(tmp210, 1)[:, None]
    tmp213 = tmp212 * tmp2
    tmp215 = tmp213 / tmp214
    tmp217 = tmp215 * tmp216
    tmp218 = tl.broadcast_to(tmp217, [XBLOCK, RBLOCK])
    tmp220 = tl.where(xmask, tmp218, 0)
    tmp221 = tl.sum(tmp220, 1)[:, None]
    tmp223 = tmp222 * tmp14
    tmp225 = tmp223 / tmp224
    tmp226 = tl.where(tmp20, tmp221, tmp21)
    tmp227 = tmp225 + tmp226
    tmp229 = tmp227 * tmp228
    tmp230 = tl.broadcast_to(tmp229, [XBLOCK, RBLOCK])
    tmp232 = tl.where(xmask, tmp230, 0)
    tmp233 = tl.sum(tmp232, 1)[:, None]
    tmp235 = tmp234 * tmp32
    tmp237 = tmp235 / tmp236
    tmp238 = tl.where(tmp20, tmp233, tmp21)
    tmp239 = tmp237 + tmp238
    tmp241 = tmp239 * tmp240
    tmp242 = tl.broadcast_to(tmp241, [XBLOCK, RBLOCK])
    tmp244 = tl.where(xmask, tmp242, 0)
    tmp245 = tl.sum(tmp244, 1)[:, None]
    tmp247 = tmp246 * tmp46
    tmp249 = tmp247 / tmp248
    tmp250 = tl.where(tmp20, tmp245, tmp21)
    tmp251 = tmp249 + tmp250
    tmp253 = tmp251 * tmp252
    tmp254 = tl.broadcast_to(tmp253, [XBLOCK, RBLOCK])
    tmp256 = tl.where(xmask, tmp254, 0)
    tmp257 = tl.sum(tmp256, 1)[:, None]
    tmp259 = tmp258 * tmp60
    tmp261 = tmp259 / tmp260
    tmp262 = tl.where(tmp20, tmp257, tmp21)
    tmp263 = tmp261 + tmp262
    tmp265 = tmp263 * tmp264
    tmp266 = tl.broadcast_to(tmp265, [XBLOCK, RBLOCK])
    tmp268 = tl.where(xmask, tmp266, 0)
    tmp269 = tl.sum(tmp268, 1)[:, None]
    tmp271 = tmp270 * tmp74
    tmp273 = tmp271 / tmp272
    tmp274 = tl.where(tmp20, tmp269, tmp21)
    tmp275 = tmp273 + tmp274
    tmp277 = tmp275 * tmp276
    tmp278 = tl.broadcast_to(tmp277, [XBLOCK, RBLOCK])
    tmp280 = tl.where(xmask, tmp278, 0)
    tmp281 = tl.sum(tmp280, 1)[:, None]
    tmp283 = tmp282 * tmp88
    tmp285 = tmp283 / tmp284
    tmp286 = tl.where(tmp20, tmp281, tmp21)
    tmp287 = tmp285 + tmp286
    tmp289 = tmp287 * tmp288
    tmp290 = tl.broadcast_to(tmp289, [XBLOCK, RBLOCK])
    tmp292 = tl.where(xmask, tmp290, 0)
    tmp293 = tl.sum(tmp292, 1)[:, None]
    tmp295 = tmp294 * tmp102
    tmp297 = tmp295 / tmp296
    tmp298 = tl.where(tmp20, tmp293, tmp21)
    tmp299 = tmp297 + tmp298
    tmp301 = tmp299 * tmp300
    tmp302 = tl.broadcast_to(tmp301, [XBLOCK, RBLOCK])
    tmp304 = tl.where(xmask, tmp302, 0)
    tmp305 = tl.sum(tmp304, 1)[:, None]
    tmp307 = tmp306 * tmp116
    tmp309 = tmp307 / tmp308
    tmp310 = tl.where(tmp20, tmp305, tmp21)
    tmp311 = tmp309 + tmp310
    tmp313 = tmp311 * tmp312
    tmp314 = tl.broadcast_to(tmp313, [XBLOCK, RBLOCK])
    tmp316 = tl.where(xmask, tmp314, 0)
    tmp317 = tl.sum(tmp316, 1)[:, None]
    tmp319 = tmp318 * tmp130
    tmp321 = tmp319 / tmp320
    tmp322 = tl.where(tmp20, tmp317, tmp21)
    tmp323 = tmp321 + tmp322
    tmp325 = tmp323 * tmp324
    tmp326 = tl.broadcast_to(tmp325, [XBLOCK, RBLOCK])
    tmp328 = tl.where(xmask, tmp326, 0)
    tmp329 = tl.sum(tmp328, 1)[:, None]
    tmp331 = tmp330 * tmp144
    tmp333 = tmp331 / tmp332
    tmp334 = tl.where(tmp20, tmp329, tmp21)
    tmp335 = tmp333 + tmp334
    tmp337 = tmp335 * tmp336
    tmp338 = tl.broadcast_to(tmp337, [XBLOCK, RBLOCK])
    tmp340 = tl.where(xmask, tmp338, 0)
    tmp341 = tl.sum(tmp340, 1)[:, None]
    tmp343 = tmp342 * tmp158
    tmp345 = tmp343 / tmp344
    tmp346 = tl.where(tmp20, tmp341, tmp21)
    tmp347 = tmp345 + tmp346
    tmp349 = tmp347 * tmp348
    tmp350 = tl.broadcast_to(tmp349, [XBLOCK, RBLOCK])
    tmp352 = tl.where(xmask, tmp350, 0)
    tmp353 = tl.sum(tmp352, 1)[:, None]
    tmp355 = tmp354 * tmp172
    tmp357 = tmp355 / tmp356
    tmp358 = tl.where(tmp20, tmp353, tmp21)
    tmp359 = tmp357 + tmp358
    tmp361 = tmp359 * tmp360
    tmp362 = tl.broadcast_to(tmp361, [XBLOCK, RBLOCK])
    tmp364 = tl.where(xmask, tmp362, 0)
    tmp365 = tl.sum(tmp364, 1)[:, None]
    tmp367 = tmp366 * tmp186
    tmp369 = tmp367 / tmp368
    tmp370 = tl.where(tmp20, tmp365, tmp21)
    tmp371 = tmp369 + tmp370
    tmp373 = tmp371 * tmp372
    tmp374 = tl.broadcast_to(tmp373, [XBLOCK, RBLOCK])
    tmp376 = tl.where(xmask, tmp374, 0)
    tmp377 = tl.sum(tmp376, 1)[:, None]
    tmp379 = tmp378 * tmp200
    tmp381 = tmp379 / tmp380
    tmp382 = tl.where(tmp20, tmp377, tmp21)
    tmp383 = tmp381 + tmp382
    tmp385 = tmp383 * tmp384
    tmp386 = tl.broadcast_to(tmp385, [XBLOCK, RBLOCK])
    tmp388 = tl.where(xmask, tmp386, 0)
    tmp389 = tl.sum(tmp388, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
    tl.store(out_ptr1 + (x3), tmp29, xmask)
    tl.store(out_ptr2 + (x3), tmp43, xmask)
    tl.store(out_ptr3 + (x3), tmp57, xmask)
    tl.store(out_ptr4 + (x3), tmp71, xmask)
    tl.store(out_ptr5 + (x3), tmp85, xmask)
    tl.store(out_ptr6 + (x3), tmp99, xmask)
    tl.store(out_ptr7 + (x3), tmp113, xmask)
    tl.store(out_ptr8 + (x3), tmp127, xmask)
    tl.store(out_ptr9 + (x3), tmp141, xmask)
    tl.store(out_ptr10 + (x3), tmp155, xmask)
    tl.store(out_ptr11 + (x3), tmp169, xmask)
    tl.store(out_ptr12 + (x3), tmp183, xmask)
    tl.store(out_ptr13 + (x3), tmp197, xmask)
    tl.store(out_ptr14 + (x3), tmp211, xmask)
    tl.store(out_ptr15 + (x3), tmp221, xmask)
    tl.store(out_ptr16 + (x3), tmp233, xmask)
    tl.store(out_ptr17 + (x3), tmp245, xmask)
    tl.store(out_ptr18 + (x3), tmp257, xmask)
    tl.store(out_ptr19 + (x3), tmp269, xmask)
    tl.store(out_ptr20 + (x3), tmp281, xmask)
    tl.store(out_ptr21 + (x3), tmp293, xmask)
    tl.store(out_ptr22 + (x3), tmp305, xmask)
    tl.store(out_ptr23 + (x3), tmp317, xmask)
    tl.store(out_ptr24 + (x3), tmp329, xmask)
    tl.store(out_ptr25 + (x3), tmp341, xmask)
    tl.store(out_ptr26 + (x3), tmp353, xmask)
    tl.store(out_ptr27 + (x3), tmp365, xmask)
    tl.store(out_ptr28 + (x3), tmp377, xmask)
    tl.store(out_ptr29 + (x3), tmp389, xmask)
