
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
    tmp5 = tl.load(in_ptr2 + (960 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr0 + (86016 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp14 = tl.load(in_ptr1 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp18 = tl.load(in_ptr2 + (896 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr4 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp29 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr0 + (79872 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp37 = tl.load(in_ptr1 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp41 = tl.load(in_ptr2 + (832 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr6 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp54 = tl.load(in_ptr0 + (73728 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp55 = tl.load(in_ptr1 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp59 = tl.load(in_ptr2 + (768 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp63 = tl.load(in_ptr7 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp72 = tl.load(in_ptr0 + (67584 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp73 = tl.load(in_ptr1 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp77 = tl.load(in_ptr2 + (704 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp81 = tl.load(in_ptr8 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp90 = tl.load(in_ptr0 + (61440 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp91 = tl.load(in_ptr1 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp95 = tl.load(in_ptr2 + (640 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp99 = tl.load(in_ptr9 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp108 = tl.load(in_ptr0 + (55296 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp109 = tl.load(in_ptr1 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp113 = tl.load(in_ptr2 + (576 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp117 = tl.load(in_ptr10 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp126 = tl.load(in_ptr0 + (49152 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp127 = tl.load(in_ptr1 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp131 = tl.load(in_ptr2 + (512 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp135 = tl.load(in_ptr11 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp144 = tl.load(in_ptr0 + (43008 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp145 = tl.load(in_ptr1 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp149 = tl.load(in_ptr2 + (448 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp153 = tl.load(in_ptr12 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp162 = tl.load(in_ptr0 + (36864 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp163 = tl.load(in_ptr1 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp167 = tl.load(in_ptr2 + (384 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp171 = tl.load(in_ptr13 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp180 = tl.load(in_ptr0 + (30720 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp181 = tl.load(in_ptr1 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp185 = tl.load(in_ptr2 + (320 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp189 = tl.load(in_ptr14 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp198 = tl.load(in_ptr0 + (24576 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp199 = tl.load(in_ptr1 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp203 = tl.load(in_ptr2 + (256 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp207 = tl.load(in_ptr15 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp216 = tl.load(in_ptr0 + (18432 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp217 = tl.load(in_ptr1 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp221 = tl.load(in_ptr2 + (192 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp225 = tl.load(in_ptr16 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp234 = tl.load(in_ptr0 + (12288 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp235 = tl.load(in_ptr1 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp239 = tl.load(in_ptr2 + (128 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp243 = tl.load(in_ptr17 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp252 = tl.load(in_ptr0 + (6144 + x0 + (96*r2) + (98304*x1)), xmask, other=0.0)
    tmp253 = tl.load(in_ptr1 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp257 = tl.load(in_ptr2 + (64 + r2 + (1024*x1)), xmask, eviction_policy='evict_last', other=0.0)
    tmp261 = tl.load(in_ptr18 + (r2 + (64*x1)), xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
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
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27.to(tl.float32)
    tmp30 = tmp28 * tmp29
    tmp31 = tmp25 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
    tmp34 = tl.where(xmask, tmp32, 0)
    tmp35 = tl.sum(tmp34, 1)[:, None]
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp36 * tmp39
    tmp42 = tmp40 / tmp41
    tmp43 = tl.where(tmp22, tmp35, tmp23)
    tmp44 = tmp42 + tmp43
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp46.to(tl.float32)
    tmp48 = tmp47 * tmp29
    tmp49 = tmp44 * tmp48
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK, RBLOCK])
    tmp52 = tl.where(xmask, tmp50, 0)
    tmp53 = tl.sum(tmp52, 1)[:, None]
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tmp54 * tmp57
    tmp60 = tmp58 / tmp59
    tmp61 = tl.where(tmp22, tmp53, tmp23)
    tmp62 = tmp60 + tmp61
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tmp65 * tmp29
    tmp67 = tmp62 * tmp66
    tmp68 = tl.broadcast_to(tmp67, [XBLOCK, RBLOCK])
    tmp70 = tl.where(xmask, tmp68, 0)
    tmp71 = tl.sum(tmp70, 1)[:, None]
    tmp74 = tmp73.to(tl.float32)
    tmp75 = tmp74.to(tl.float32)
    tmp76 = tmp72 * tmp75
    tmp78 = tmp76 / tmp77
    tmp79 = tl.where(tmp22, tmp71, tmp23)
    tmp80 = tmp78 + tmp79
    tmp82 = tmp81.to(tl.float32)
    tmp83 = tmp82.to(tl.float32)
    tmp84 = tmp83 * tmp29
    tmp85 = tmp80 * tmp84
    tmp86 = tl.broadcast_to(tmp85, [XBLOCK, RBLOCK])
    tmp88 = tl.where(xmask, tmp86, 0)
    tmp89 = tl.sum(tmp88, 1)[:, None]
    tmp92 = tmp91.to(tl.float32)
    tmp93 = tmp92.to(tl.float32)
    tmp94 = tmp90 * tmp93
    tmp96 = tmp94 / tmp95
    tmp97 = tl.where(tmp22, tmp89, tmp23)
    tmp98 = tmp96 + tmp97
    tmp100 = tmp99.to(tl.float32)
    tmp101 = tmp100.to(tl.float32)
    tmp102 = tmp101 * tmp29
    tmp103 = tmp98 * tmp102
    tmp104 = tl.broadcast_to(tmp103, [XBLOCK, RBLOCK])
    tmp106 = tl.where(xmask, tmp104, 0)
    tmp107 = tl.sum(tmp106, 1)[:, None]
    tmp110 = tmp109.to(tl.float32)
    tmp111 = tmp110.to(tl.float32)
    tmp112 = tmp108 * tmp111
    tmp114 = tmp112 / tmp113
    tmp115 = tl.where(tmp22, tmp107, tmp23)
    tmp116 = tmp114 + tmp115
    tmp118 = tmp117.to(tl.float32)
    tmp119 = tmp118.to(tl.float32)
    tmp120 = tmp119 * tmp29
    tmp121 = tmp116 * tmp120
    tmp122 = tl.broadcast_to(tmp121, [XBLOCK, RBLOCK])
    tmp124 = tl.where(xmask, tmp122, 0)
    tmp125 = tl.sum(tmp124, 1)[:, None]
    tmp128 = tmp127.to(tl.float32)
    tmp129 = tmp128.to(tl.float32)
    tmp130 = tmp126 * tmp129
    tmp132 = tmp130 / tmp131
    tmp133 = tl.where(tmp22, tmp125, tmp23)
    tmp134 = tmp132 + tmp133
    tmp136 = tmp135.to(tl.float32)
    tmp137 = tmp136.to(tl.float32)
    tmp138 = tmp137 * tmp29
    tmp139 = tmp134 * tmp138
    tmp140 = tl.broadcast_to(tmp139, [XBLOCK, RBLOCK])
    tmp142 = tl.where(xmask, tmp140, 0)
    tmp143 = tl.sum(tmp142, 1)[:, None]
    tmp146 = tmp145.to(tl.float32)
    tmp147 = tmp146.to(tl.float32)
    tmp148 = tmp144 * tmp147
    tmp150 = tmp148 / tmp149
    tmp151 = tl.where(tmp22, tmp143, tmp23)
    tmp152 = tmp150 + tmp151
    tmp154 = tmp153.to(tl.float32)
    tmp155 = tmp154.to(tl.float32)
    tmp156 = tmp155 * tmp29
    tmp157 = tmp152 * tmp156
    tmp158 = tl.broadcast_to(tmp157, [XBLOCK, RBLOCK])
    tmp160 = tl.where(xmask, tmp158, 0)
    tmp161 = tl.sum(tmp160, 1)[:, None]
    tmp164 = tmp163.to(tl.float32)
    tmp165 = tmp164.to(tl.float32)
    tmp166 = tmp162 * tmp165
    tmp168 = tmp166 / tmp167
    tmp169 = tl.where(tmp22, tmp161, tmp23)
    tmp170 = tmp168 + tmp169
    tmp172 = tmp171.to(tl.float32)
    tmp173 = tmp172.to(tl.float32)
    tmp174 = tmp173 * tmp29
    tmp175 = tmp170 * tmp174
    tmp176 = tl.broadcast_to(tmp175, [XBLOCK, RBLOCK])
    tmp178 = tl.where(xmask, tmp176, 0)
    tmp179 = tl.sum(tmp178, 1)[:, None]
    tmp182 = tmp181.to(tl.float32)
    tmp183 = tmp182.to(tl.float32)
    tmp184 = tmp180 * tmp183
    tmp186 = tmp184 / tmp185
    tmp187 = tl.where(tmp22, tmp179, tmp23)
    tmp188 = tmp186 + tmp187
    tmp190 = tmp189.to(tl.float32)
    tmp191 = tmp190.to(tl.float32)
    tmp192 = tmp191 * tmp29
    tmp193 = tmp188 * tmp192
    tmp194 = tl.broadcast_to(tmp193, [XBLOCK, RBLOCK])
    tmp196 = tl.where(xmask, tmp194, 0)
    tmp197 = tl.sum(tmp196, 1)[:, None]
    tmp200 = tmp199.to(tl.float32)
    tmp201 = tmp200.to(tl.float32)
    tmp202 = tmp198 * tmp201
    tmp204 = tmp202 / tmp203
    tmp205 = tl.where(tmp22, tmp197, tmp23)
    tmp206 = tmp204 + tmp205
    tmp208 = tmp207.to(tl.float32)
    tmp209 = tmp208.to(tl.float32)
    tmp210 = tmp209 * tmp29
    tmp211 = tmp206 * tmp210
    tmp212 = tl.broadcast_to(tmp211, [XBLOCK, RBLOCK])
    tmp214 = tl.where(xmask, tmp212, 0)
    tmp215 = tl.sum(tmp214, 1)[:, None]
    tmp218 = tmp217.to(tl.float32)
    tmp219 = tmp218.to(tl.float32)
    tmp220 = tmp216 * tmp219
    tmp222 = tmp220 / tmp221
    tmp223 = tl.where(tmp22, tmp215, tmp23)
    tmp224 = tmp222 + tmp223
    tmp226 = tmp225.to(tl.float32)
    tmp227 = tmp226.to(tl.float32)
    tmp228 = tmp227 * tmp29
    tmp229 = tmp224 * tmp228
    tmp230 = tl.broadcast_to(tmp229, [XBLOCK, RBLOCK])
    tmp232 = tl.where(xmask, tmp230, 0)
    tmp233 = tl.sum(tmp232, 1)[:, None]
    tmp236 = tmp235.to(tl.float32)
    tmp237 = tmp236.to(tl.float32)
    tmp238 = tmp234 * tmp237
    tmp240 = tmp238 / tmp239
    tmp241 = tl.where(tmp22, tmp233, tmp23)
    tmp242 = tmp240 + tmp241
    tmp244 = tmp243.to(tl.float32)
    tmp245 = tmp244.to(tl.float32)
    tmp246 = tmp245 * tmp29
    tmp247 = tmp242 * tmp246
    tmp248 = tl.broadcast_to(tmp247, [XBLOCK, RBLOCK])
    tmp250 = tl.where(xmask, tmp248, 0)
    tmp251 = tl.sum(tmp250, 1)[:, None]
    tmp254 = tmp253.to(tl.float32)
    tmp255 = tmp254.to(tl.float32)
    tmp256 = tmp252 * tmp255
    tmp258 = tmp256 / tmp257
    tmp259 = tl.where(tmp22, tmp251, tmp23)
    tmp260 = tmp258 + tmp259
    tmp262 = tmp261.to(tl.float32)
    tmp263 = tmp262.to(tl.float32)
    tmp264 = tmp263 * tmp29
    tmp265 = tmp260 * tmp264
    tmp266 = tl.broadcast_to(tmp265, [XBLOCK, RBLOCK])
    tmp268 = tl.where(xmask, tmp266, 0)
    tmp269 = tl.sum(tmp268, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, xmask)
    tl.store(out_ptr1 + (x3), tmp35, xmask)
    tl.store(out_ptr2 + (x3), tmp53, xmask)
    tl.store(out_ptr3 + (x3), tmp71, xmask)
    tl.store(out_ptr4 + (x3), tmp89, xmask)
    tl.store(out_ptr5 + (x3), tmp107, xmask)
    tl.store(out_ptr6 + (x3), tmp125, xmask)
    tl.store(out_ptr7 + (x3), tmp143, xmask)
    tl.store(out_ptr8 + (x3), tmp161, xmask)
    tl.store(out_ptr9 + (x3), tmp179, xmask)
    tl.store(out_ptr10 + (x3), tmp197, xmask)
    tl.store(out_ptr11 + (x3), tmp215, xmask)
    tl.store(out_ptr12 + (x3), tmp233, xmask)
    tl.store(out_ptr13 + (x3), tmp251, xmask)
    tl.store(out_ptr14 + (x3), tmp269, xmask)
