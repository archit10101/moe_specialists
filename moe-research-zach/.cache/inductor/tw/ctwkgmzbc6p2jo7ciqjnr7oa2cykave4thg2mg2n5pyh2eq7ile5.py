
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*i1', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_select_backward_3', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 63, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048) % 16
    x0 = xindex % 32
    x1 = (xindex // 32) % 64
    x3 = (xindex // 32768)
    x5 = xindex
    tmp3 = tl.load(in_ptr0 + (92160 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (86016 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp20 = tl.load(in_ptr2 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (79872 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr1 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp38 = tl.load(in_ptr2 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr4 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr0 + (73728 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr1 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp53 = tl.load(in_ptr2 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr5 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr0 + (67584 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr1 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp68 = tl.load(in_ptr2 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr6 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr0 + (61440 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr1 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp83 = tl.load(in_ptr2 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr7 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr0 + (55296 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr1 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp98 = tl.load(in_ptr2 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp100 = tl.load(in_ptr8 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr0 + (49152 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp109 = tl.load(in_ptr1 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp113 = tl.load(in_ptr2 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp115 = tl.load(in_ptr9 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp123 = tl.load(in_ptr0 + (43008 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp124 = tl.load(in_ptr1 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp128 = tl.load(in_ptr2 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp130 = tl.load(in_ptr10 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp138 = tl.load(in_ptr0 + (36864 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp139 = tl.load(in_ptr1 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp143 = tl.load(in_ptr2 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp145 = tl.load(in_ptr11 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp153 = tl.load(in_ptr0 + (30720 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp154 = tl.load(in_ptr1 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp158 = tl.load(in_ptr2 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp160 = tl.load(in_ptr12 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp168 = tl.load(in_ptr0 + (24576 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp169 = tl.load(in_ptr1 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp173 = tl.load(in_ptr2 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp175 = tl.load(in_ptr13 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp183 = tl.load(in_ptr0 + (18432 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp184 = tl.load(in_ptr1 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp188 = tl.load(in_ptr2 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp190 = tl.load(in_ptr14 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp198 = tl.load(in_ptr0 + (12288 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp199 = tl.load(in_ptr1 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp203 = tl.load(in_ptr2 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp205 = tl.load(in_ptr15 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp213 = tl.load(in_ptr0 + (6144 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp214 = tl.load(in_ptr1 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp218 = tl.load(in_ptr2 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp220 = tl.load(in_ptr16 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp228 = tl.load(in_ptr0 + (x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp229 = tl.load(in_ptr1 + (x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp233 = tl.load(in_ptr2 + (x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp235 = tl.load(in_ptr17 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 15, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp3 * tmp6
    tmp9 = tmp7 / tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = 0.0
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = tl.full([1], 14, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp15 * tmp18
    tmp21 = tmp19 / tmp20
    tmp22 = x1
    tmp23 = tl.full([1], 63, tl.int32)
    tmp24 = tmp22 == tmp23
    tmp26 = tl.where(tmp24, tmp25, tmp11)
    tmp27 = tmp21 + tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.where(tmp14, tmp28, tmp11)
    tmp30 = tmp12 + tmp29
    tmp31 = tl.full([1], 13, tl.int32)
    tmp32 = tmp0 == tmp31
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp33 * tmp36
    tmp39 = tmp37 / tmp38
    tmp41 = tl.where(tmp24, tmp40, tmp11)
    tmp42 = tmp39 + tmp41
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tl.where(tmp32, tmp43, tmp11)
    tmp45 = tmp30 + tmp44
    tmp46 = tl.full([1], 12, tl.int32)
    tmp47 = tmp0 == tmp46
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp50.to(tl.float32)
    tmp52 = tmp48 * tmp51
    tmp54 = tmp52 / tmp53
    tmp56 = tl.where(tmp24, tmp55, tmp11)
    tmp57 = tmp54 + tmp56
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tl.where(tmp47, tmp58, tmp11)
    tmp60 = tmp45 + tmp59
    tmp61 = tl.full([1], 11, tl.int32)
    tmp62 = tmp0 == tmp61
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tmp65.to(tl.float32)
    tmp67 = tmp63 * tmp66
    tmp69 = tmp67 / tmp68
    tmp71 = tl.where(tmp24, tmp70, tmp11)
    tmp72 = tmp69 + tmp71
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tl.where(tmp62, tmp73, tmp11)
    tmp75 = tmp60 + tmp74
    tmp76 = tl.full([1], 10, tl.int32)
    tmp77 = tmp0 == tmp76
    tmp80 = tmp79.to(tl.float32)
    tmp81 = tmp80.to(tl.float32)
    tmp82 = tmp78 * tmp81
    tmp84 = tmp82 / tmp83
    tmp86 = tl.where(tmp24, tmp85, tmp11)
    tmp87 = tmp84 + tmp86
    tmp88 = tmp87.to(tl.float32)
    tmp89 = tl.where(tmp77, tmp88, tmp11)
    tmp90 = tmp75 + tmp89
    tmp91 = tl.full([1], 9, tl.int32)
    tmp92 = tmp0 == tmp91
    tmp95 = tmp94.to(tl.float32)
    tmp96 = tmp95.to(tl.float32)
    tmp97 = tmp93 * tmp96
    tmp99 = tmp97 / tmp98
    tmp101 = tl.where(tmp24, tmp100, tmp11)
    tmp102 = tmp99 + tmp101
    tmp103 = tmp102.to(tl.float32)
    tmp104 = tl.where(tmp92, tmp103, tmp11)
    tmp105 = tmp90 + tmp104
    tmp106 = tl.full([1], 8, tl.int32)
    tmp107 = tmp0 == tmp106
    tmp110 = tmp109.to(tl.float32)
    tmp111 = tmp110.to(tl.float32)
    tmp112 = tmp108 * tmp111
    tmp114 = tmp112 / tmp113
    tmp116 = tl.where(tmp24, tmp115, tmp11)
    tmp117 = tmp114 + tmp116
    tmp118 = tmp117.to(tl.float32)
    tmp119 = tl.where(tmp107, tmp118, tmp11)
    tmp120 = tmp105 + tmp119
    tmp121 = tl.full([1], 7, tl.int32)
    tmp122 = tmp0 == tmp121
    tmp125 = tmp124.to(tl.float32)
    tmp126 = tmp125.to(tl.float32)
    tmp127 = tmp123 * tmp126
    tmp129 = tmp127 / tmp128
    tmp131 = tl.where(tmp24, tmp130, tmp11)
    tmp132 = tmp129 + tmp131
    tmp133 = tmp132.to(tl.float32)
    tmp134 = tl.where(tmp122, tmp133, tmp11)
    tmp135 = tmp120 + tmp134
    tmp136 = tl.full([1], 6, tl.int32)
    tmp137 = tmp0 == tmp136
    tmp140 = tmp139.to(tl.float32)
    tmp141 = tmp140.to(tl.float32)
    tmp142 = tmp138 * tmp141
    tmp144 = tmp142 / tmp143
    tmp146 = tl.where(tmp24, tmp145, tmp11)
    tmp147 = tmp144 + tmp146
    tmp148 = tmp147.to(tl.float32)
    tmp149 = tl.where(tmp137, tmp148, tmp11)
    tmp150 = tmp135 + tmp149
    tmp151 = tl.full([1], 5, tl.int32)
    tmp152 = tmp0 == tmp151
    tmp155 = tmp154.to(tl.float32)
    tmp156 = tmp155.to(tl.float32)
    tmp157 = tmp153 * tmp156
    tmp159 = tmp157 / tmp158
    tmp161 = tl.where(tmp24, tmp160, tmp11)
    tmp162 = tmp159 + tmp161
    tmp163 = tmp162.to(tl.float32)
    tmp164 = tl.where(tmp152, tmp163, tmp11)
    tmp165 = tmp150 + tmp164
    tmp166 = tl.full([1], 4, tl.int32)
    tmp167 = tmp0 == tmp166
    tmp170 = tmp169.to(tl.float32)
    tmp171 = tmp170.to(tl.float32)
    tmp172 = tmp168 * tmp171
    tmp174 = tmp172 / tmp173
    tmp176 = tl.where(tmp24, tmp175, tmp11)
    tmp177 = tmp174 + tmp176
    tmp178 = tmp177.to(tl.float32)
    tmp179 = tl.where(tmp167, tmp178, tmp11)
    tmp180 = tmp165 + tmp179
    tmp181 = tl.full([1], 3, tl.int32)
    tmp182 = tmp0 == tmp181
    tmp185 = tmp184.to(tl.float32)
    tmp186 = tmp185.to(tl.float32)
    tmp187 = tmp183 * tmp186
    tmp189 = tmp187 / tmp188
    tmp191 = tl.where(tmp24, tmp190, tmp11)
    tmp192 = tmp189 + tmp191
    tmp193 = tmp192.to(tl.float32)
    tmp194 = tl.where(tmp182, tmp193, tmp11)
    tmp195 = tmp180 + tmp194
    tmp196 = tl.full([1], 2, tl.int32)
    tmp197 = tmp0 == tmp196
    tmp200 = tmp199.to(tl.float32)
    tmp201 = tmp200.to(tl.float32)
    tmp202 = tmp198 * tmp201
    tmp204 = tmp202 / tmp203
    tmp206 = tl.where(tmp24, tmp205, tmp11)
    tmp207 = tmp204 + tmp206
    tmp208 = tmp207.to(tl.float32)
    tmp209 = tl.where(tmp197, tmp208, tmp11)
    tmp210 = tmp195 + tmp209
    tmp211 = tl.full([1], 1, tl.int32)
    tmp212 = tmp0 == tmp211
    tmp215 = tmp214.to(tl.float32)
    tmp216 = tmp215.to(tl.float32)
    tmp217 = tmp213 * tmp216
    tmp219 = tmp217 / tmp218
    tmp221 = tl.where(tmp24, tmp220, tmp11)
    tmp222 = tmp219 + tmp221
    tmp223 = tmp222.to(tl.float32)
    tmp224 = tl.where(tmp212, tmp223, tmp11)
    tmp225 = tmp210 + tmp224
    tmp226 = tl.full([1], 0, tl.int32)
    tmp227 = tmp0 == tmp226
    tmp230 = tmp229.to(tl.float32)
    tmp231 = tmp230.to(tl.float32)
    tmp232 = tmp228 * tmp231
    tmp234 = tmp232 / tmp233
    tmp236 = tl.where(tmp24, tmp235, tmp11)
    tmp237 = tmp234 + tmp236
    tmp238 = tmp237.to(tl.float32)
    tmp239 = tl.where(tmp227, tmp238, tmp11)
    tmp240 = tmp225 + tmp239
    tl.store(in_out_ptr0 + (x5), tmp240, None)
