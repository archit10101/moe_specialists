
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
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*i1', 7: '*i1', 8: '*i1', 9: '*i1', 10: '*i1', 11: '*i1', 12: '*i1', 13: '*i1', 14: '*i1', 15: '*i1', 16: '*i1', 17: '*i1', 18: '*i1', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: 'i32', 35: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_select_backward_sum_2', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 61, 'num_reduction': 15, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (92160 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (960 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (960 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr3 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr0 + (86016 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp13 = tl.load(in_ptr1 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp16 = tl.load(in_ptr2 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr4 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp26 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (79872 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp34 = tl.load(in_ptr1 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp37 = tl.load(in_ptr2 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.load(in_ptr6 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp49 = tl.load(in_ptr0 + (73728 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp50 = tl.load(in_ptr1 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp53 = tl.load(in_ptr2 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.load(in_ptr7 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp65 = tl.load(in_ptr0 + (67584 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp66 = tl.load(in_ptr1 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp69 = tl.load(in_ptr2 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp73 = tl.load(in_ptr8 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp81 = tl.load(in_ptr0 + (61440 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp82 = tl.load(in_ptr1 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp85 = tl.load(in_ptr2 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp89 = tl.load(in_ptr9 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp97 = tl.load(in_ptr0 + (55296 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp98 = tl.load(in_ptr1 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp101 = tl.load(in_ptr2 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp105 = tl.load(in_ptr10 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp113 = tl.load(in_ptr0 + (49152 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp114 = tl.load(in_ptr1 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp117 = tl.load(in_ptr2 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp121 = tl.load(in_ptr11 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp129 = tl.load(in_ptr0 + (43008 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp130 = tl.load(in_ptr1 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp133 = tl.load(in_ptr2 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp137 = tl.load(in_ptr12 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp145 = tl.load(in_ptr0 + (36864 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp146 = tl.load(in_ptr1 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp149 = tl.load(in_ptr2 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp153 = tl.load(in_ptr13 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp161 = tl.load(in_ptr0 + (30720 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp162 = tl.load(in_ptr1 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp165 = tl.load(in_ptr2 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp169 = tl.load(in_ptr14 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp177 = tl.load(in_ptr0 + (24576 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp178 = tl.load(in_ptr1 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp181 = tl.load(in_ptr2 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp185 = tl.load(in_ptr15 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp193 = tl.load(in_ptr0 + (18432 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp194 = tl.load(in_ptr1 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp197 = tl.load(in_ptr2 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp201 = tl.load(in_ptr16 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp209 = tl.load(in_ptr0 + (12288 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp210 = tl.load(in_ptr1 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp213 = tl.load(in_ptr2 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp217 = tl.load(in_ptr17 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp225 = tl.load(in_ptr0 + (6144 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp226 = tl.load(in_ptr1 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp229 = tl.load(in_ptr2 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp233 = tl.load(in_ptr18 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
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
    tmp25 = tmp24.to(tl.float32)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp23 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.where(xmask, tmp29, 0)
    tmp32 = tl.sum(tmp31, 1)[:, None]
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp33 * tmp35
    tmp38 = tmp36 / tmp37
    tmp39 = tl.where(tmp20, tmp32, tmp21)
    tmp40 = tmp38 + tmp39
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp42 * tmp26
    tmp44 = tmp40 * tmp43
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK, RBLOCK])
    tmp47 = tl.where(xmask, tmp45, 0)
    tmp48 = tl.sum(tmp47, 1)[:, None]
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp49 * tmp51
    tmp54 = tmp52 / tmp53
    tmp55 = tl.where(tmp20, tmp48, tmp21)
    tmp56 = tmp54 + tmp55
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tmp58 * tmp26
    tmp60 = tmp56 * tmp59
    tmp61 = tl.broadcast_to(tmp60, [XBLOCK, RBLOCK])
    tmp63 = tl.where(xmask, tmp61, 0)
    tmp64 = tl.sum(tmp63, 1)[:, None]
    tmp67 = tmp66.to(tl.float32)
    tmp68 = tmp65 * tmp67
    tmp70 = tmp68 / tmp69
    tmp71 = tl.where(tmp20, tmp64, tmp21)
    tmp72 = tmp70 + tmp71
    tmp74 = tmp73.to(tl.float32)
    tmp75 = tmp74 * tmp26
    tmp76 = tmp72 * tmp75
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK, RBLOCK])
    tmp79 = tl.where(xmask, tmp77, 0)
    tmp80 = tl.sum(tmp79, 1)[:, None]
    tmp83 = tmp82.to(tl.float32)
    tmp84 = tmp81 * tmp83
    tmp86 = tmp84 / tmp85
    tmp87 = tl.where(tmp20, tmp80, tmp21)
    tmp88 = tmp86 + tmp87
    tmp90 = tmp89.to(tl.float32)
    tmp91 = tmp90 * tmp26
    tmp92 = tmp88 * tmp91
    tmp93 = tl.broadcast_to(tmp92, [XBLOCK, RBLOCK])
    tmp95 = tl.where(xmask, tmp93, 0)
    tmp96 = tl.sum(tmp95, 1)[:, None]
    tmp99 = tmp98.to(tl.float32)
    tmp100 = tmp97 * tmp99
    tmp102 = tmp100 / tmp101
    tmp103 = tl.where(tmp20, tmp96, tmp21)
    tmp104 = tmp102 + tmp103
    tmp106 = tmp105.to(tl.float32)
    tmp107 = tmp106 * tmp26
    tmp108 = tmp104 * tmp107
    tmp109 = tl.broadcast_to(tmp108, [XBLOCK, RBLOCK])
    tmp111 = tl.where(xmask, tmp109, 0)
    tmp112 = tl.sum(tmp111, 1)[:, None]
    tmp115 = tmp114.to(tl.float32)
    tmp116 = tmp113 * tmp115
    tmp118 = tmp116 / tmp117
    tmp119 = tl.where(tmp20, tmp112, tmp21)
    tmp120 = tmp118 + tmp119
    tmp122 = tmp121.to(tl.float32)
    tmp123 = tmp122 * tmp26
    tmp124 = tmp120 * tmp123
    tmp125 = tl.broadcast_to(tmp124, [XBLOCK, RBLOCK])
    tmp127 = tl.where(xmask, tmp125, 0)
    tmp128 = tl.sum(tmp127, 1)[:, None]
    tmp131 = tmp130.to(tl.float32)
    tmp132 = tmp129 * tmp131
    tmp134 = tmp132 / tmp133
    tmp135 = tl.where(tmp20, tmp128, tmp21)
    tmp136 = tmp134 + tmp135
    tmp138 = tmp137.to(tl.float32)
    tmp139 = tmp138 * tmp26
    tmp140 = tmp136 * tmp139
    tmp141 = tl.broadcast_to(tmp140, [XBLOCK, RBLOCK])
    tmp143 = tl.where(xmask, tmp141, 0)
    tmp144 = tl.sum(tmp143, 1)[:, None]
    tmp147 = tmp146.to(tl.float32)
    tmp148 = tmp145 * tmp147
    tmp150 = tmp148 / tmp149
    tmp151 = tl.where(tmp20, tmp144, tmp21)
    tmp152 = tmp150 + tmp151
    tmp154 = tmp153.to(tl.float32)
    tmp155 = tmp154 * tmp26
    tmp156 = tmp152 * tmp155
    tmp157 = tl.broadcast_to(tmp156, [XBLOCK, RBLOCK])
    tmp159 = tl.where(xmask, tmp157, 0)
    tmp160 = tl.sum(tmp159, 1)[:, None]
    tmp163 = tmp162.to(tl.float32)
    tmp164 = tmp161 * tmp163
    tmp166 = tmp164 / tmp165
    tmp167 = tl.where(tmp20, tmp160, tmp21)
    tmp168 = tmp166 + tmp167
    tmp170 = tmp169.to(tl.float32)
    tmp171 = tmp170 * tmp26
    tmp172 = tmp168 * tmp171
    tmp173 = tl.broadcast_to(tmp172, [XBLOCK, RBLOCK])
    tmp175 = tl.where(xmask, tmp173, 0)
    tmp176 = tl.sum(tmp175, 1)[:, None]
    tmp179 = tmp178.to(tl.float32)
    tmp180 = tmp177 * tmp179
    tmp182 = tmp180 / tmp181
    tmp183 = tl.where(tmp20, tmp176, tmp21)
    tmp184 = tmp182 + tmp183
    tmp186 = tmp185.to(tl.float32)
    tmp187 = tmp186 * tmp26
    tmp188 = tmp184 * tmp187
    tmp189 = tl.broadcast_to(tmp188, [XBLOCK, RBLOCK])
    tmp191 = tl.where(xmask, tmp189, 0)
    tmp192 = tl.sum(tmp191, 1)[:, None]
    tmp195 = tmp194.to(tl.float32)
    tmp196 = tmp193 * tmp195
    tmp198 = tmp196 / tmp197
    tmp199 = tl.where(tmp20, tmp192, tmp21)
    tmp200 = tmp198 + tmp199
    tmp202 = tmp201.to(tl.float32)
    tmp203 = tmp202 * tmp26
    tmp204 = tmp200 * tmp203
    tmp205 = tl.broadcast_to(tmp204, [XBLOCK, RBLOCK])
    tmp207 = tl.where(xmask, tmp205, 0)
    tmp208 = tl.sum(tmp207, 1)[:, None]
    tmp211 = tmp210.to(tl.float32)
    tmp212 = tmp209 * tmp211
    tmp214 = tmp212 / tmp213
    tmp215 = tl.where(tmp20, tmp208, tmp21)
    tmp216 = tmp214 + tmp215
    tmp218 = tmp217.to(tl.float32)
    tmp219 = tmp218 * tmp26
    tmp220 = tmp216 * tmp219
    tmp221 = tl.broadcast_to(tmp220, [XBLOCK, RBLOCK])
    tmp223 = tl.where(xmask, tmp221, 0)
    tmp224 = tl.sum(tmp223, 1)[:, None]
    tmp227 = tmp226.to(tl.float32)
    tmp228 = tmp225 * tmp227
    tmp230 = tmp228 / tmp229
    tmp231 = tl.where(tmp20, tmp224, tmp21)
    tmp232 = tmp230 + tmp231
    tmp234 = tmp233.to(tl.float32)
    tmp235 = tmp234 * tmp26
    tmp236 = tmp232 * tmp235
    tmp237 = tl.broadcast_to(tmp236, [XBLOCK, RBLOCK])
    tmp239 = tl.where(xmask, tmp237, 0)
    tmp240 = tl.sum(tmp239, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp11, xmask)
    tl.store(out_ptr1 + (x3), tmp32, xmask)
    tl.store(out_ptr2 + (x3), tmp48, xmask)
    tl.store(out_ptr3 + (x3), tmp64, xmask)
    tl.store(out_ptr4 + (x3), tmp80, xmask)
    tl.store(out_ptr5 + (x3), tmp96, xmask)
    tl.store(out_ptr6 + (x3), tmp112, xmask)
    tl.store(out_ptr7 + (x3), tmp128, xmask)
    tl.store(out_ptr8 + (x3), tmp144, xmask)
    tl.store(out_ptr9 + (x3), tmp160, xmask)
    tl.store(out_ptr10 + (x3), tmp176, xmask)
    tl.store(out_ptr11 + (x3), tmp192, xmask)
    tl.store(out_ptr12 + (x3), tmp208, xmask)
    tl.store(out_ptr13 + (x3), tmp224, xmask)
    tl.store(out_ptr14 + (x3), tmp240, xmask)
