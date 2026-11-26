
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*bf16', 35: '*bf16', 36: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_select_backward_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 110, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048) % 16
    x0 = xindex % 32
    x1 = (xindex // 32) % 64
    x3 = (xindex // 32768)
    x6 = xindex
    x4 = (xindex // 2048)
    tmp3 = tl.load(in_ptr0 + (92224 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp7 = tl.load(in_ptr2 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (86080 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp17 = tl.load(in_ptr2 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr0 + (79936 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr1 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp33 = tl.load(in_ptr2 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr4 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr0 + (73792 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr1 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp46 = tl.load(in_ptr2 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr5 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr0 + (67648 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr1 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp59 = tl.load(in_ptr2 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr6 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr0 + (61504 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr1 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp72 = tl.load(in_ptr2 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr7 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr0 + (55360 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr1 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp85 = tl.load(in_ptr2 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr8 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr0 + (49216 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr1 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp98 = tl.load(in_ptr2 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp100 = tl.load(in_ptr9 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr0 + (43072 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr1 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp111 = tl.load(in_ptr2 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp113 = tl.load(in_ptr10 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp120 = tl.load(in_ptr0 + (36928 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp121 = tl.load(in_ptr1 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp124 = tl.load(in_ptr2 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp126 = tl.load(in_ptr11 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp133 = tl.load(in_ptr0 + (30784 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp134 = tl.load(in_ptr1 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp137 = tl.load(in_ptr2 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp139 = tl.load(in_ptr12 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp146 = tl.load(in_ptr0 + (24640 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp147 = tl.load(in_ptr1 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp150 = tl.load(in_ptr2 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp152 = tl.load(in_ptr13 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp159 = tl.load(in_ptr0 + (18496 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp160 = tl.load(in_ptr1 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp163 = tl.load(in_ptr2 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp165 = tl.load(in_ptr14 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp172 = tl.load(in_ptr0 + (12352 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp173 = tl.load(in_ptr1 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp176 = tl.load(in_ptr2 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp178 = tl.load(in_ptr15 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp185 = tl.load(in_ptr0 + (6208 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp186 = tl.load(in_ptr1 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp189 = tl.load(in_ptr2 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp191 = tl.load(in_ptr16 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp199 = tl.load(in_ptr0 + (64 + x0 + (96*x1) + (98304*(x4 // 16))), None)
    tmp200 = tl.load(in_ptr1 + (x1 + (1024*(x4 // 16))), None, eviction_policy='evict_last').to(tl.int1)
    tmp203 = tl.load(in_ptr2 + (x1 + (1024*(x4 // 16))), None, eviction_policy='evict_last')
    tmp205 = tl.load(in_ptr17 + (x0 + (32*(x4 // 16))), None, eviction_policy='evict_last')
    tmp211 = tl.load(in_ptr0 + (92192 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp213 = tl.load(in_ptr18 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp216 = tl.load(in_ptr0 + (86048 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp218 = tl.load(in_ptr18 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp220 = tl.load(in_ptr19 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp225 = tl.load(in_ptr0 + (79904 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp227 = tl.load(in_ptr18 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp229 = tl.load(in_ptr20 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp234 = tl.load(in_ptr0 + (73760 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp236 = tl.load(in_ptr18 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp238 = tl.load(in_ptr21 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp243 = tl.load(in_ptr0 + (67616 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp245 = tl.load(in_ptr18 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp247 = tl.load(in_ptr22 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp252 = tl.load(in_ptr0 + (61472 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp254 = tl.load(in_ptr18 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp256 = tl.load(in_ptr23 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp261 = tl.load(in_ptr0 + (55328 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp263 = tl.load(in_ptr18 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp265 = tl.load(in_ptr24 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp270 = tl.load(in_ptr0 + (49184 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp272 = tl.load(in_ptr18 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp274 = tl.load(in_ptr25 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp279 = tl.load(in_ptr0 + (43040 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp281 = tl.load(in_ptr18 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp283 = tl.load(in_ptr26 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp288 = tl.load(in_ptr0 + (36896 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp290 = tl.load(in_ptr18 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp292 = tl.load(in_ptr27 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp297 = tl.load(in_ptr0 + (30752 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp299 = tl.load(in_ptr18 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp301 = tl.load(in_ptr28 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp306 = tl.load(in_ptr0 + (24608 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp308 = tl.load(in_ptr18 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp310 = tl.load(in_ptr29 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp315 = tl.load(in_ptr0 + (18464 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp317 = tl.load(in_ptr18 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp319 = tl.load(in_ptr30 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp324 = tl.load(in_ptr0 + (12320 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp326 = tl.load(in_ptr18 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp328 = tl.load(in_ptr31 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp333 = tl.load(in_ptr0 + (6176 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp335 = tl.load(in_ptr18 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp337 = tl.load(in_ptr32 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp342 = tl.load(in_ptr0 + (32 + x0 + (96*x1) + (98304*(x4 // 16))), None)
    tmp344 = tl.load(in_ptr18 + (x1 + (1024*(x4 // 16))), None, eviction_policy='evict_last')
    tmp346 = tl.load(in_ptr33 + (x0 + (32*(x4 // 16))), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 15, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp8 = tmp6 / tmp7
    tmp9 = 0.0
    tmp10 = tl.where(tmp2, tmp8, tmp9)
    tmp11 = tl.full([1], 14, tl.int32)
    tmp12 = tmp0 == tmp11
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 * tmp15
    tmp18 = tmp16 / tmp17
    tmp19 = x1
    tmp20 = tl.full([1], 63, tl.int32)
    tmp21 = tmp19 == tmp20
    tmp23 = tl.where(tmp21, tmp22, tmp9)
    tmp24 = tmp18 + tmp23
    tmp25 = tl.where(tmp12, tmp24, tmp9)
    tmp26 = tmp10 + tmp25
    tmp27 = tl.full([1], 13, tl.int32)
    tmp28 = tmp0 == tmp27
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 * tmp31
    tmp34 = tmp32 / tmp33
    tmp36 = tl.where(tmp21, tmp35, tmp9)
    tmp37 = tmp34 + tmp36
    tmp38 = tl.where(tmp28, tmp37, tmp9)
    tmp39 = tmp26 + tmp38
    tmp40 = tl.full([1], 12, tl.int32)
    tmp41 = tmp0 == tmp40
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp42 * tmp44
    tmp47 = tmp45 / tmp46
    tmp49 = tl.where(tmp21, tmp48, tmp9)
    tmp50 = tmp47 + tmp49
    tmp51 = tl.where(tmp41, tmp50, tmp9)
    tmp52 = tmp39 + tmp51
    tmp53 = tl.full([1], 11, tl.int32)
    tmp54 = tmp0 == tmp53
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tmp55 * tmp57
    tmp60 = tmp58 / tmp59
    tmp62 = tl.where(tmp21, tmp61, tmp9)
    tmp63 = tmp60 + tmp62
    tmp64 = tl.where(tmp54, tmp63, tmp9)
    tmp65 = tmp52 + tmp64
    tmp66 = tl.full([1], 10, tl.int32)
    tmp67 = tmp0 == tmp66
    tmp70 = tmp69.to(tl.float32)
    tmp71 = tmp68 * tmp70
    tmp73 = tmp71 / tmp72
    tmp75 = tl.where(tmp21, tmp74, tmp9)
    tmp76 = tmp73 + tmp75
    tmp77 = tl.where(tmp67, tmp76, tmp9)
    tmp78 = tmp65 + tmp77
    tmp79 = tl.full([1], 9, tl.int32)
    tmp80 = tmp0 == tmp79
    tmp83 = tmp82.to(tl.float32)
    tmp84 = tmp81 * tmp83
    tmp86 = tmp84 / tmp85
    tmp88 = tl.where(tmp21, tmp87, tmp9)
    tmp89 = tmp86 + tmp88
    tmp90 = tl.where(tmp80, tmp89, tmp9)
    tmp91 = tmp78 + tmp90
    tmp92 = tl.full([1], 8, tl.int32)
    tmp93 = tmp0 == tmp92
    tmp96 = tmp95.to(tl.float32)
    tmp97 = tmp94 * tmp96
    tmp99 = tmp97 / tmp98
    tmp101 = tl.where(tmp21, tmp100, tmp9)
    tmp102 = tmp99 + tmp101
    tmp103 = tl.where(tmp93, tmp102, tmp9)
    tmp104 = tmp91 + tmp103
    tmp105 = tl.full([1], 7, tl.int32)
    tmp106 = tmp0 == tmp105
    tmp109 = tmp108.to(tl.float32)
    tmp110 = tmp107 * tmp109
    tmp112 = tmp110 / tmp111
    tmp114 = tl.where(tmp21, tmp113, tmp9)
    tmp115 = tmp112 + tmp114
    tmp116 = tl.where(tmp106, tmp115, tmp9)
    tmp117 = tmp104 + tmp116
    tmp118 = tl.full([1], 6, tl.int32)
    tmp119 = tmp0 == tmp118
    tmp122 = tmp121.to(tl.float32)
    tmp123 = tmp120 * tmp122
    tmp125 = tmp123 / tmp124
    tmp127 = tl.where(tmp21, tmp126, tmp9)
    tmp128 = tmp125 + tmp127
    tmp129 = tl.where(tmp119, tmp128, tmp9)
    tmp130 = tmp117 + tmp129
    tmp131 = tl.full([1], 5, tl.int32)
    tmp132 = tmp0 == tmp131
    tmp135 = tmp134.to(tl.float32)
    tmp136 = tmp133 * tmp135
    tmp138 = tmp136 / tmp137
    tmp140 = tl.where(tmp21, tmp139, tmp9)
    tmp141 = tmp138 + tmp140
    tmp142 = tl.where(tmp132, tmp141, tmp9)
    tmp143 = tmp130 + tmp142
    tmp144 = tl.full([1], 4, tl.int32)
    tmp145 = tmp0 == tmp144
    tmp148 = tmp147.to(tl.float32)
    tmp149 = tmp146 * tmp148
    tmp151 = tmp149 / tmp150
    tmp153 = tl.where(tmp21, tmp152, tmp9)
    tmp154 = tmp151 + tmp153
    tmp155 = tl.where(tmp145, tmp154, tmp9)
    tmp156 = tmp143 + tmp155
    tmp157 = tl.full([1], 3, tl.int32)
    tmp158 = tmp0 == tmp157
    tmp161 = tmp160.to(tl.float32)
    tmp162 = tmp159 * tmp161
    tmp164 = tmp162 / tmp163
    tmp166 = tl.where(tmp21, tmp165, tmp9)
    tmp167 = tmp164 + tmp166
    tmp168 = tl.where(tmp158, tmp167, tmp9)
    tmp169 = tmp156 + tmp168
    tmp170 = tl.full([1], 2, tl.int32)
    tmp171 = tmp0 == tmp170
    tmp174 = tmp173.to(tl.float32)
    tmp175 = tmp172 * tmp174
    tmp177 = tmp175 / tmp176
    tmp179 = tl.where(tmp21, tmp178, tmp9)
    tmp180 = tmp177 + tmp179
    tmp181 = tl.where(tmp171, tmp180, tmp9)
    tmp182 = tmp169 + tmp181
    tmp183 = tl.full([1], 1, tl.int32)
    tmp184 = tmp0 == tmp183
    tmp187 = tmp186.to(tl.float32)
    tmp188 = tmp185 * tmp187
    tmp190 = tmp188 / tmp189
    tmp192 = tl.where(tmp21, tmp191, tmp9)
    tmp193 = tmp190 + tmp192
    tmp194 = tl.where(tmp184, tmp193, tmp9)
    tmp195 = tmp182 + tmp194
    tmp196 = (x6 // 2048) % 16
    tmp197 = tl.full([1], 0, tl.int32)
    tmp198 = tmp196 == tmp197
    tmp201 = tmp200.to(tl.float32)
    tmp202 = tmp199 * tmp201
    tmp204 = tmp202 / tmp203
    tmp206 = tl.where(tmp21, tmp205, tmp9)
    tmp207 = tmp204 + tmp206
    tmp208 = tl.where(tmp198, tmp207, tmp9)
    tmp209 = tmp195 + tmp208
    tmp210 = tmp209.to(tl.float32)
    tmp212 = tmp211 * tmp5
    tmp214 = tmp212 / tmp213
    tmp215 = tl.where(tmp2, tmp214, tmp9)
    tmp217 = tmp216 * tmp15
    tmp219 = tmp217 / tmp218
    tmp221 = tl.where(tmp21, tmp220, tmp9)
    tmp222 = tmp219 + tmp221
    tmp223 = tl.where(tmp12, tmp222, tmp9)
    tmp224 = tmp215 + tmp223
    tmp226 = tmp225 * tmp31
    tmp228 = tmp226 / tmp227
    tmp230 = tl.where(tmp21, tmp229, tmp9)
    tmp231 = tmp228 + tmp230
    tmp232 = tl.where(tmp28, tmp231, tmp9)
    tmp233 = tmp224 + tmp232
    tmp235 = tmp234 * tmp44
    tmp237 = tmp235 / tmp236
    tmp239 = tl.where(tmp21, tmp238, tmp9)
    tmp240 = tmp237 + tmp239
    tmp241 = tl.where(tmp41, tmp240, tmp9)
    tmp242 = tmp233 + tmp241
    tmp244 = tmp243 * tmp57
    tmp246 = tmp244 / tmp245
    tmp248 = tl.where(tmp21, tmp247, tmp9)
    tmp249 = tmp246 + tmp248
    tmp250 = tl.where(tmp54, tmp249, tmp9)
    tmp251 = tmp242 + tmp250
    tmp253 = tmp252 * tmp70
    tmp255 = tmp253 / tmp254
    tmp257 = tl.where(tmp21, tmp256, tmp9)
    tmp258 = tmp255 + tmp257
    tmp259 = tl.where(tmp67, tmp258, tmp9)
    tmp260 = tmp251 + tmp259
    tmp262 = tmp261 * tmp83
    tmp264 = tmp262 / tmp263
    tmp266 = tl.where(tmp21, tmp265, tmp9)
    tmp267 = tmp264 + tmp266
    tmp268 = tl.where(tmp80, tmp267, tmp9)
    tmp269 = tmp260 + tmp268
    tmp271 = tmp270 * tmp96
    tmp273 = tmp271 / tmp272
    tmp275 = tl.where(tmp21, tmp274, tmp9)
    tmp276 = tmp273 + tmp275
    tmp277 = tl.where(tmp93, tmp276, tmp9)
    tmp278 = tmp269 + tmp277
    tmp280 = tmp279 * tmp109
    tmp282 = tmp280 / tmp281
    tmp284 = tl.where(tmp21, tmp283, tmp9)
    tmp285 = tmp282 + tmp284
    tmp286 = tl.where(tmp106, tmp285, tmp9)
    tmp287 = tmp278 + tmp286
    tmp289 = tmp288 * tmp122
    tmp291 = tmp289 / tmp290
    tmp293 = tl.where(tmp21, tmp292, tmp9)
    tmp294 = tmp291 + tmp293
    tmp295 = tl.where(tmp119, tmp294, tmp9)
    tmp296 = tmp287 + tmp295
    tmp298 = tmp297 * tmp135
    tmp300 = tmp298 / tmp299
    tmp302 = tl.where(tmp21, tmp301, tmp9)
    tmp303 = tmp300 + tmp302
    tmp304 = tl.where(tmp132, tmp303, tmp9)
    tmp305 = tmp296 + tmp304
    tmp307 = tmp306 * tmp148
    tmp309 = tmp307 / tmp308
    tmp311 = tl.where(tmp21, tmp310, tmp9)
    tmp312 = tmp309 + tmp311
    tmp313 = tl.where(tmp145, tmp312, tmp9)
    tmp314 = tmp305 + tmp313
    tmp316 = tmp315 * tmp161
    tmp318 = tmp316 / tmp317
    tmp320 = tl.where(tmp21, tmp319, tmp9)
    tmp321 = tmp318 + tmp320
    tmp322 = tl.where(tmp158, tmp321, tmp9)
    tmp323 = tmp314 + tmp322
    tmp325 = tmp324 * tmp174
    tmp327 = tmp325 / tmp326
    tmp329 = tl.where(tmp21, tmp328, tmp9)
    tmp330 = tmp327 + tmp329
    tmp331 = tl.where(tmp171, tmp330, tmp9)
    tmp332 = tmp323 + tmp331
    tmp334 = tmp333 * tmp187
    tmp336 = tmp334 / tmp335
    tmp338 = tl.where(tmp21, tmp337, tmp9)
    tmp339 = tmp336 + tmp338
    tmp340 = tl.where(tmp184, tmp339, tmp9)
    tmp341 = tmp332 + tmp340
    tmp343 = tmp342 * tmp201
    tmp345 = tmp343 / tmp344
    tmp347 = tl.where(tmp21, tmp346, tmp9)
    tmp348 = tmp345 + tmp347
    tmp349 = tl.where(tmp198, tmp348, tmp9)
    tmp350 = tmp341 + tmp349
    tmp351 = tmp350.to(tl.float32)
    tl.store(out_ptr0 + (x6), tmp210, None)
    tl.store(out_ptr1 + (x6), tmp351, None)
