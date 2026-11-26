
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*i1', 7: '*i1', 8: '*i1', 9: '*i1', 10: '*i1', 11: '*i1', 12: '*i1', 13: '*i1', 14: '*i1', 15: '*i1', 16: '*i1', 17: '*i1', 18: '*i1', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: 'i32', 35: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_select_backward_sum_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 61, 'num_reduction': 15, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (61440 + x0 + (64*r2) + (65536*x1)), None)
    tmp1 = tl.load(in_ptr1 + (960 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp4 = tl.load(in_ptr2 + (960 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (r2 + (64*x1)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (57344 + x0 + (64*r2) + (65536*x1)), None)
    tmp12 = tl.load(in_ptr1 + (896 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp15 = tl.load(in_ptr2 + (896 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr4 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp25 = tl.load(in_ptr5 + (r2), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (53248 + x0 + (64*r2) + (65536*x1)), None)
    tmp32 = tl.load(in_ptr1 + (832 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp35 = tl.load(in_ptr2 + (832 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr6 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp46 = tl.load(in_ptr0 + (49152 + x0 + (64*r2) + (65536*x1)), None)
    tmp47 = tl.load(in_ptr1 + (768 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp50 = tl.load(in_ptr2 + (768 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr7 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp61 = tl.load(in_ptr0 + (45056 + x0 + (64*r2) + (65536*x1)), None)
    tmp62 = tl.load(in_ptr1 + (704 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp65 = tl.load(in_ptr2 + (704 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr8 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp76 = tl.load(in_ptr0 + (40960 + x0 + (64*r2) + (65536*x1)), None)
    tmp77 = tl.load(in_ptr1 + (640 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp80 = tl.load(in_ptr2 + (640 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp84 = tl.load(in_ptr9 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp91 = tl.load(in_ptr0 + (36864 + x0 + (64*r2) + (65536*x1)), None)
    tmp92 = tl.load(in_ptr1 + (576 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp95 = tl.load(in_ptr2 + (576 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp99 = tl.load(in_ptr10 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp106 = tl.load(in_ptr0 + (32768 + x0 + (64*r2) + (65536*x1)), None)
    tmp107 = tl.load(in_ptr1 + (512 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp110 = tl.load(in_ptr2 + (512 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp114 = tl.load(in_ptr11 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp121 = tl.load(in_ptr0 + (28672 + x0 + (64*r2) + (65536*x1)), None)
    tmp122 = tl.load(in_ptr1 + (448 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp125 = tl.load(in_ptr2 + (448 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp129 = tl.load(in_ptr12 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp136 = tl.load(in_ptr0 + (24576 + x0 + (64*r2) + (65536*x1)), None)
    tmp137 = tl.load(in_ptr1 + (384 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp140 = tl.load(in_ptr2 + (384 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp144 = tl.load(in_ptr13 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp151 = tl.load(in_ptr0 + (20480 + x0 + (64*r2) + (65536*x1)), None)
    tmp152 = tl.load(in_ptr1 + (320 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp155 = tl.load(in_ptr2 + (320 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp159 = tl.load(in_ptr14 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp166 = tl.load(in_ptr0 + (16384 + x0 + (64*r2) + (65536*x1)), None)
    tmp167 = tl.load(in_ptr1 + (256 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp170 = tl.load(in_ptr2 + (256 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp174 = tl.load(in_ptr15 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp181 = tl.load(in_ptr0 + (12288 + x0 + (64*r2) + (65536*x1)), None)
    tmp182 = tl.load(in_ptr1 + (192 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp185 = tl.load(in_ptr2 + (192 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp189 = tl.load(in_ptr16 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp196 = tl.load(in_ptr0 + (8192 + x0 + (64*r2) + (65536*x1)), None)
    tmp197 = tl.load(in_ptr1 + (128 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp200 = tl.load(in_ptr2 + (128 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp204 = tl.load(in_ptr17 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp211 = tl.load(in_ptr0 + (4096 + x0 + (64*r2) + (65536*x1)), None)
    tmp212 = tl.load(in_ptr1 + (64 + r2 + (1024*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp215 = tl.load(in_ptr2 + (64 + r2 + (1024*x1)), None, eviction_policy='evict_last')
    tmp219 = tl.load(in_ptr18 + (r2 + (64*x1)), None, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tmp5 = tmp3 / tmp4
    tmp7 = tmp5 * tmp6
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp10 = tl.sum(tmp8, 1)[:, None]
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 * tmp13
    tmp16 = tmp14 / tmp15
    tmp17 = r2
    tmp18 = tl.full([1, 1], 63, tl.int32)
    tmp19 = tmp17 == tmp18
    tmp20 = 0.0
    tmp21 = tl.where(tmp19, tmp10, tmp20)
    tmp22 = tmp16 + tmp21
    tmp24 = tmp23.to(tl.float32)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp22 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
    tmp30 = tl.sum(tmp28, 1)[:, None]
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 * tmp33
    tmp36 = tmp34 / tmp35
    tmp37 = tl.where(tmp19, tmp30, tmp20)
    tmp38 = tmp36 + tmp37
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp40 * tmp25
    tmp42 = tmp38 * tmp41
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, RBLOCK])
    tmp45 = tl.sum(tmp43, 1)[:, None]
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp46 * tmp48
    tmp51 = tmp49 / tmp50
    tmp52 = tl.where(tmp19, tmp45, tmp20)
    tmp53 = tmp51 + tmp52
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tmp55 * tmp25
    tmp57 = tmp53 * tmp56
    tmp58 = tl.broadcast_to(tmp57, [XBLOCK, RBLOCK])
    tmp60 = tl.sum(tmp58, 1)[:, None]
    tmp63 = tmp62.to(tl.float32)
    tmp64 = tmp61 * tmp63
    tmp66 = tmp64 / tmp65
    tmp67 = tl.where(tmp19, tmp60, tmp20)
    tmp68 = tmp66 + tmp67
    tmp70 = tmp69.to(tl.float32)
    tmp71 = tmp70 * tmp25
    tmp72 = tmp68 * tmp71
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp75 = tl.sum(tmp73, 1)[:, None]
    tmp78 = tmp77.to(tl.float32)
    tmp79 = tmp76 * tmp78
    tmp81 = tmp79 / tmp80
    tmp82 = tl.where(tmp19, tmp75, tmp20)
    tmp83 = tmp81 + tmp82
    tmp85 = tmp84.to(tl.float32)
    tmp86 = tmp85 * tmp25
    tmp87 = tmp83 * tmp86
    tmp88 = tl.broadcast_to(tmp87, [XBLOCK, RBLOCK])
    tmp90 = tl.sum(tmp88, 1)[:, None]
    tmp93 = tmp92.to(tl.float32)
    tmp94 = tmp91 * tmp93
    tmp96 = tmp94 / tmp95
    tmp97 = tl.where(tmp19, tmp90, tmp20)
    tmp98 = tmp96 + tmp97
    tmp100 = tmp99.to(tl.float32)
    tmp101 = tmp100 * tmp25
    tmp102 = tmp98 * tmp101
    tmp103 = tl.broadcast_to(tmp102, [XBLOCK, RBLOCK])
    tmp105 = tl.sum(tmp103, 1)[:, None]
    tmp108 = tmp107.to(tl.float32)
    tmp109 = tmp106 * tmp108
    tmp111 = tmp109 / tmp110
    tmp112 = tl.where(tmp19, tmp105, tmp20)
    tmp113 = tmp111 + tmp112
    tmp115 = tmp114.to(tl.float32)
    tmp116 = tmp115 * tmp25
    tmp117 = tmp113 * tmp116
    tmp118 = tl.broadcast_to(tmp117, [XBLOCK, RBLOCK])
    tmp120 = tl.sum(tmp118, 1)[:, None]
    tmp123 = tmp122.to(tl.float32)
    tmp124 = tmp121 * tmp123
    tmp126 = tmp124 / tmp125
    tmp127 = tl.where(tmp19, tmp120, tmp20)
    tmp128 = tmp126 + tmp127
    tmp130 = tmp129.to(tl.float32)
    tmp131 = tmp130 * tmp25
    tmp132 = tmp128 * tmp131
    tmp133 = tl.broadcast_to(tmp132, [XBLOCK, RBLOCK])
    tmp135 = tl.sum(tmp133, 1)[:, None]
    tmp138 = tmp137.to(tl.float32)
    tmp139 = tmp136 * tmp138
    tmp141 = tmp139 / tmp140
    tmp142 = tl.where(tmp19, tmp135, tmp20)
    tmp143 = tmp141 + tmp142
    tmp145 = tmp144.to(tl.float32)
    tmp146 = tmp145 * tmp25
    tmp147 = tmp143 * tmp146
    tmp148 = tl.broadcast_to(tmp147, [XBLOCK, RBLOCK])
    tmp150 = tl.sum(tmp148, 1)[:, None]
    tmp153 = tmp152.to(tl.float32)
    tmp154 = tmp151 * tmp153
    tmp156 = tmp154 / tmp155
    tmp157 = tl.where(tmp19, tmp150, tmp20)
    tmp158 = tmp156 + tmp157
    tmp160 = tmp159.to(tl.float32)
    tmp161 = tmp160 * tmp25
    tmp162 = tmp158 * tmp161
    tmp163 = tl.broadcast_to(tmp162, [XBLOCK, RBLOCK])
    tmp165 = tl.sum(tmp163, 1)[:, None]
    tmp168 = tmp167.to(tl.float32)
    tmp169 = tmp166 * tmp168
    tmp171 = tmp169 / tmp170
    tmp172 = tl.where(tmp19, tmp165, tmp20)
    tmp173 = tmp171 + tmp172
    tmp175 = tmp174.to(tl.float32)
    tmp176 = tmp175 * tmp25
    tmp177 = tmp173 * tmp176
    tmp178 = tl.broadcast_to(tmp177, [XBLOCK, RBLOCK])
    tmp180 = tl.sum(tmp178, 1)[:, None]
    tmp183 = tmp182.to(tl.float32)
    tmp184 = tmp181 * tmp183
    tmp186 = tmp184 / tmp185
    tmp187 = tl.where(tmp19, tmp180, tmp20)
    tmp188 = tmp186 + tmp187
    tmp190 = tmp189.to(tl.float32)
    tmp191 = tmp190 * tmp25
    tmp192 = tmp188 * tmp191
    tmp193 = tl.broadcast_to(tmp192, [XBLOCK, RBLOCK])
    tmp195 = tl.sum(tmp193, 1)[:, None]
    tmp198 = tmp197.to(tl.float32)
    tmp199 = tmp196 * tmp198
    tmp201 = tmp199 / tmp200
    tmp202 = tl.where(tmp19, tmp195, tmp20)
    tmp203 = tmp201 + tmp202
    tmp205 = tmp204.to(tl.float32)
    tmp206 = tmp205 * tmp25
    tmp207 = tmp203 * tmp206
    tmp208 = tl.broadcast_to(tmp207, [XBLOCK, RBLOCK])
    tmp210 = tl.sum(tmp208, 1)[:, None]
    tmp213 = tmp212.to(tl.float32)
    tmp214 = tmp211 * tmp213
    tmp216 = tmp214 / tmp215
    tmp217 = tl.where(tmp19, tmp210, tmp20)
    tmp218 = tmp216 + tmp217
    tmp220 = tmp219.to(tl.float32)
    tmp221 = tmp220 * tmp25
    tmp222 = tmp218 * tmp221
    tmp223 = tl.broadcast_to(tmp222, [XBLOCK, RBLOCK])
    tmp225 = tl.sum(tmp223, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp30, None)
    tl.store(out_ptr2 + (x3), tmp45, None)
    tl.store(out_ptr3 + (x3), tmp60, None)
    tl.store(out_ptr4 + (x3), tmp75, None)
    tl.store(out_ptr5 + (x3), tmp90, None)
    tl.store(out_ptr6 + (x3), tmp105, None)
    tl.store(out_ptr7 + (x3), tmp120, None)
    tl.store(out_ptr8 + (x3), tmp135, None)
    tl.store(out_ptr9 + (x3), tmp150, None)
    tl.store(out_ptr10 + (x3), tmp165, None)
    tl.store(out_ptr11 + (x3), tmp180, None)
    tl.store(out_ptr12 + (x3), tmp195, None)
    tl.store(out_ptr13 + (x3), tmp210, None)
    tl.store(out_ptr14 + (x3), tmp225, None)
