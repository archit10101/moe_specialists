
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: '*fp32', 29: '*fp32', 30: '*fp32', 31: '*fp32', 32: '*fp32', 33: '*fp32', 34: '*fp32', 35: '*fp32', 36: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_select_backward_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'no_x_dim': False, 'num_load': 110, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 2048) % 16
    x0 = xindex % 32
    x1 = (xindex // 32) % 64
    x3 = (xindex // 32768)
    x5 = xindex
    tmp3 = tl.load(in_ptr0 + (92224 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (86080 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp20 = tl.load(in_ptr2 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (79936 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr1 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp38 = tl.load(in_ptr2 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr4 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr0 + (73792 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr1 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp53 = tl.load(in_ptr2 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr5 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr0 + (67648 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr1 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp68 = tl.load(in_ptr2 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr6 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr0 + (61504 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr1 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp83 = tl.load(in_ptr2 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr7 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr0 + (55360 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr1 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp98 = tl.load(in_ptr2 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp100 = tl.load(in_ptr8 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr0 + (49216 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp109 = tl.load(in_ptr1 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp113 = tl.load(in_ptr2 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp115 = tl.load(in_ptr9 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp123 = tl.load(in_ptr0 + (43072 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp124 = tl.load(in_ptr1 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp128 = tl.load(in_ptr2 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp130 = tl.load(in_ptr10 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp138 = tl.load(in_ptr0 + (36928 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp139 = tl.load(in_ptr1 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp143 = tl.load(in_ptr2 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp145 = tl.load(in_ptr11 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp153 = tl.load(in_ptr0 + (30784 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp154 = tl.load(in_ptr1 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp158 = tl.load(in_ptr2 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp160 = tl.load(in_ptr12 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp168 = tl.load(in_ptr0 + (24640 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp169 = tl.load(in_ptr1 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp173 = tl.load(in_ptr2 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp175 = tl.load(in_ptr13 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp183 = tl.load(in_ptr0 + (18496 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp184 = tl.load(in_ptr1 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp188 = tl.load(in_ptr2 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp190 = tl.load(in_ptr14 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp198 = tl.load(in_ptr0 + (12352 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp199 = tl.load(in_ptr1 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp203 = tl.load(in_ptr2 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp205 = tl.load(in_ptr15 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp213 = tl.load(in_ptr0 + (6208 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp214 = tl.load(in_ptr1 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp218 = tl.load(in_ptr2 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp220 = tl.load(in_ptr16 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp228 = tl.load(in_ptr0 + (64 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp229 = tl.load(in_ptr1 + (x1 + (1024*x3)), None, eviction_policy='evict_last').to(tl.int1)
    tmp233 = tl.load(in_ptr2 + (x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp235 = tl.load(in_ptr17 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp241 = tl.load(in_ptr0 + (92192 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp243 = tl.load(in_ptr18 + (960 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp247 = tl.load(in_ptr0 + (86048 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp249 = tl.load(in_ptr18 + (896 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp251 = tl.load(in_ptr19 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp257 = tl.load(in_ptr0 + (79904 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp259 = tl.load(in_ptr18 + (832 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp261 = tl.load(in_ptr20 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp267 = tl.load(in_ptr0 + (73760 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp269 = tl.load(in_ptr18 + (768 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp271 = tl.load(in_ptr21 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp277 = tl.load(in_ptr0 + (67616 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp279 = tl.load(in_ptr18 + (704 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp281 = tl.load(in_ptr22 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp287 = tl.load(in_ptr0 + (61472 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp289 = tl.load(in_ptr18 + (640 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp291 = tl.load(in_ptr23 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp297 = tl.load(in_ptr0 + (55328 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp299 = tl.load(in_ptr18 + (576 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp301 = tl.load(in_ptr24 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp307 = tl.load(in_ptr0 + (49184 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp309 = tl.load(in_ptr18 + (512 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp311 = tl.load(in_ptr25 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp317 = tl.load(in_ptr0 + (43040 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp319 = tl.load(in_ptr18 + (448 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp321 = tl.load(in_ptr26 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp327 = tl.load(in_ptr0 + (36896 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp329 = tl.load(in_ptr18 + (384 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp331 = tl.load(in_ptr27 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp337 = tl.load(in_ptr0 + (30752 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp339 = tl.load(in_ptr18 + (320 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp341 = tl.load(in_ptr28 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp347 = tl.load(in_ptr0 + (24608 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp349 = tl.load(in_ptr18 + (256 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp351 = tl.load(in_ptr29 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp357 = tl.load(in_ptr0 + (18464 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp359 = tl.load(in_ptr18 + (192 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp361 = tl.load(in_ptr30 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp367 = tl.load(in_ptr0 + (12320 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp369 = tl.load(in_ptr18 + (128 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp371 = tl.load(in_ptr31 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp377 = tl.load(in_ptr0 + (6176 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp379 = tl.load(in_ptr18 + (64 + x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp381 = tl.load(in_ptr32 + (x0 + (32*x3)), None, eviction_policy='evict_last')
    tmp387 = tl.load(in_ptr0 + (32 + x0 + (96*x1) + (98304*x3)), None, eviction_policy='evict_last')
    tmp389 = tl.load(in_ptr18 + (x1 + (1024*x3)), None, eviction_policy='evict_last')
    tmp391 = tl.load(in_ptr33 + (x0 + (32*x3)), None, eviction_policy='evict_last')
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
    tmp242 = tmp241 * tmp6
    tmp244 = tmp242 / tmp243
    tmp245 = tmp244.to(tl.float32)
    tmp246 = tl.where(tmp2, tmp245, tmp11)
    tmp248 = tmp247 * tmp18
    tmp250 = tmp248 / tmp249
    tmp252 = tl.where(tmp24, tmp251, tmp11)
    tmp253 = tmp250 + tmp252
    tmp254 = tmp253.to(tl.float32)
    tmp255 = tl.where(tmp14, tmp254, tmp11)
    tmp256 = tmp246 + tmp255
    tmp258 = tmp257 * tmp36
    tmp260 = tmp258 / tmp259
    tmp262 = tl.where(tmp24, tmp261, tmp11)
    tmp263 = tmp260 + tmp262
    tmp264 = tmp263.to(tl.float32)
    tmp265 = tl.where(tmp32, tmp264, tmp11)
    tmp266 = tmp256 + tmp265
    tmp268 = tmp267 * tmp51
    tmp270 = tmp268 / tmp269
    tmp272 = tl.where(tmp24, tmp271, tmp11)
    tmp273 = tmp270 + tmp272
    tmp274 = tmp273.to(tl.float32)
    tmp275 = tl.where(tmp47, tmp274, tmp11)
    tmp276 = tmp266 + tmp275
    tmp278 = tmp277 * tmp66
    tmp280 = tmp278 / tmp279
    tmp282 = tl.where(tmp24, tmp281, tmp11)
    tmp283 = tmp280 + tmp282
    tmp284 = tmp283.to(tl.float32)
    tmp285 = tl.where(tmp62, tmp284, tmp11)
    tmp286 = tmp276 + tmp285
    tmp288 = tmp287 * tmp81
    tmp290 = tmp288 / tmp289
    tmp292 = tl.where(tmp24, tmp291, tmp11)
    tmp293 = tmp290 + tmp292
    tmp294 = tmp293.to(tl.float32)
    tmp295 = tl.where(tmp77, tmp294, tmp11)
    tmp296 = tmp286 + tmp295
    tmp298 = tmp297 * tmp96
    tmp300 = tmp298 / tmp299
    tmp302 = tl.where(tmp24, tmp301, tmp11)
    tmp303 = tmp300 + tmp302
    tmp304 = tmp303.to(tl.float32)
    tmp305 = tl.where(tmp92, tmp304, tmp11)
    tmp306 = tmp296 + tmp305
    tmp308 = tmp307 * tmp111
    tmp310 = tmp308 / tmp309
    tmp312 = tl.where(tmp24, tmp311, tmp11)
    tmp313 = tmp310 + tmp312
    tmp314 = tmp313.to(tl.float32)
    tmp315 = tl.where(tmp107, tmp314, tmp11)
    tmp316 = tmp306 + tmp315
    tmp318 = tmp317 * tmp126
    tmp320 = tmp318 / tmp319
    tmp322 = tl.where(tmp24, tmp321, tmp11)
    tmp323 = tmp320 + tmp322
    tmp324 = tmp323.to(tl.float32)
    tmp325 = tl.where(tmp122, tmp324, tmp11)
    tmp326 = tmp316 + tmp325
    tmp328 = tmp327 * tmp141
    tmp330 = tmp328 / tmp329
    tmp332 = tl.where(tmp24, tmp331, tmp11)
    tmp333 = tmp330 + tmp332
    tmp334 = tmp333.to(tl.float32)
    tmp335 = tl.where(tmp137, tmp334, tmp11)
    tmp336 = tmp326 + tmp335
    tmp338 = tmp337 * tmp156
    tmp340 = tmp338 / tmp339
    tmp342 = tl.where(tmp24, tmp341, tmp11)
    tmp343 = tmp340 + tmp342
    tmp344 = tmp343.to(tl.float32)
    tmp345 = tl.where(tmp152, tmp344, tmp11)
    tmp346 = tmp336 + tmp345
    tmp348 = tmp347 * tmp171
    tmp350 = tmp348 / tmp349
    tmp352 = tl.where(tmp24, tmp351, tmp11)
    tmp353 = tmp350 + tmp352
    tmp354 = tmp353.to(tl.float32)
    tmp355 = tl.where(tmp167, tmp354, tmp11)
    tmp356 = tmp346 + tmp355
    tmp358 = tmp357 * tmp186
    tmp360 = tmp358 / tmp359
    tmp362 = tl.where(tmp24, tmp361, tmp11)
    tmp363 = tmp360 + tmp362
    tmp364 = tmp363.to(tl.float32)
    tmp365 = tl.where(tmp182, tmp364, tmp11)
    tmp366 = tmp356 + tmp365
    tmp368 = tmp367 * tmp201
    tmp370 = tmp368 / tmp369
    tmp372 = tl.where(tmp24, tmp371, tmp11)
    tmp373 = tmp370 + tmp372
    tmp374 = tmp373.to(tl.float32)
    tmp375 = tl.where(tmp197, tmp374, tmp11)
    tmp376 = tmp366 + tmp375
    tmp378 = tmp377 * tmp216
    tmp380 = tmp378 / tmp379
    tmp382 = tl.where(tmp24, tmp381, tmp11)
    tmp383 = tmp380 + tmp382
    tmp384 = tmp383.to(tl.float32)
    tmp385 = tl.where(tmp212, tmp384, tmp11)
    tmp386 = tmp376 + tmp385
    tmp388 = tmp387 * tmp231
    tmp390 = tmp388 / tmp389
    tmp392 = tl.where(tmp24, tmp391, tmp11)
    tmp393 = tmp390 + tmp392
    tmp394 = tmp393.to(tl.float32)
    tmp395 = tl.where(tmp227, tmp394, tmp11)
    tmp396 = tmp386 + tmp395
    tl.store(in_out_ptr0 + (x5), tmp240, None)
    tl.store(in_out_ptr1 + (x5), tmp396, None)
