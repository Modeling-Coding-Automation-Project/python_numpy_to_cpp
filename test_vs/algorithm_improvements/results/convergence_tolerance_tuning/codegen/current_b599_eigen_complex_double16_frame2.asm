
/tmp/matrix_peak_stack_audit/current_b599/current_b599_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000001be00 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeINS0_7ComplexIdEELm16EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS7_XT0_EEERSC_m.part.0>:
   1be00:	sub    rsp,0xd8
   1be07:	movsd  xmm0,QWORD PTR [rdi]
   1be0b:	movsd  xmm2,QWORD PTR [rdi+0x8]
   1be10:	movsd  xmm3,QWORD PTR [rsi+0x8]
   1be15:	movsd  xmm1,QWORD PTR [rsi]
   1be19:	movapd xmm10,xmm0
   1be1e:	movapd xmm4,xmm2
   1be22:	mulsd  xmm10,xmm1
   1be27:	mulsd  xmm0,xmm3
   1be2b:	mulsd  xmm1,xmm2
   1be2f:	mulsd  xmm4,xmm3
   1be33:	addsd  xmm0,xmm1
   1be37:	subsd  xmm10,xmm4
   1be3c:	movsd  xmm4,QWORD PTR [rdx+0x8]
   1be41:	addsd  xmm10,QWORD PTR [rdx]
   1be46:	addsd  xmm4,xmm0
   1be4a:	movsd  QWORD PTR [rdx],xmm10
   1be4f:	movsd  xmm0,QWORD PTR [rdi+0x10]
   1be54:	movsd  xmm1,QWORD PTR [rsi]
   1be58:	movsd  QWORD PTR [rdx+0x8],xmm4
   1be5d:	movsd  xmm2,QWORD PTR [rdi+0x18]
   1be62:	movsd  xmm3,QWORD PTR [rsi+0x8]
   1be67:	movapd xmm8,xmm0
   1be6c:	mulsd  xmm8,xmm1
   1be71:	mulsd  xmm0,xmm3
   1be75:	movapd xmm5,xmm2
   1be79:	mulsd  xmm1,xmm2
   1be7d:	mulsd  xmm5,xmm3
   1be81:	movsd  xmm3,QWORD PTR [rdx+0x18]
   1be86:	addsd  xmm0,xmm1
   1be8a:	subsd  xmm8,xmm5
   1be8f:	addsd  xmm8,QWORD PTR [rdx+0x10]
   1be95:	addsd  xmm3,xmm0
   1be99:	movsd  QWORD PTR [rdx+0x10],xmm8
   1be9f:	movsd  xmm0,QWORD PTR [rdi+0x20]
   1bea4:	movsd  xmm1,QWORD PTR [rsi]
   1bea8:	movsd  QWORD PTR [rdx+0x18],xmm3
   1bead:	movsd  xmm2,QWORD PTR [rdi+0x28]
   1beb2:	movsd  xmm5,QWORD PTR [rsi+0x8]
   1beb7:	movapd xmm7,xmm0
   1bebb:	mulsd  xmm7,xmm1
   1bebf:	mulsd  xmm0,xmm5
   1bec3:	movapd xmm6,xmm2
   1bec7:	mulsd  xmm1,xmm2
   1becb:	movsd  xmm2,QWORD PTR [rdx+0x28]
   1bed0:	mulsd  xmm6,xmm5
   1bed4:	addsd  xmm0,xmm1
   1bed8:	subsd  xmm7,xmm6
   1bedc:	addsd  xmm7,QWORD PTR [rdx+0x20]
   1bee1:	addsd  xmm2,xmm0
   1bee5:	movsd  QWORD PTR [rdx+0x20],xmm7
   1beea:	movsd  xmm0,QWORD PTR [rdi+0x30]
   1beef:	movsd  xmm1,QWORD PTR [rsi]
   1bef3:	movsd  QWORD PTR [rdx+0x28],xmm2
   1bef8:	movsd  xmm5,QWORD PTR [rdi+0x38]
   1befd:	movsd  xmm9,QWORD PTR [rsi+0x8]
   1bf03:	movapd xmm6,xmm0
   1bf07:	mulsd  xmm6,xmm1
   1bf0b:	mulsd  xmm0,xmm9
   1bf10:	movapd xmm11,xmm5
   1bf15:	mulsd  xmm11,xmm9
   1bf1a:	mulsd  xmm1,xmm5
   1bf1e:	subsd  xmm6,xmm11
   1bf23:	addsd  xmm6,QWORD PTR [rdx+0x30]
   1bf28:	addsd  xmm1,xmm0
   1bf2c:	movsd  QWORD PTR [rdx+0x30],xmm6
   1bf31:	addsd  xmm1,QWORD PTR [rdx+0x38]
   1bf36:	movsd  xmm9,QWORD PTR [rdi+0x40]
   1bf3c:	movsd  xmm0,QWORD PTR [rsi]
   1bf40:	movapd xmm5,xmm9
   1bf45:	movsd  QWORD PTR [rdx+0x38],xmm1
   1bf4a:	movsd  xmm11,QWORD PTR [rdi+0x48]
   1bf50:	movsd  xmm12,QWORD PTR [rsi+0x8]
   1bf56:	mulsd  xmm5,xmm0
   1bf5a:	mulsd  xmm9,xmm12
   1bf5f:	movapd xmm13,xmm11
   1bf64:	mulsd  xmm13,xmm12
   1bf69:	mulsd  xmm0,xmm11
   1bf6e:	subsd  xmm5,xmm13
   1bf73:	addsd  xmm5,QWORD PTR [rdx+0x40]
   1bf78:	addsd  xmm0,xmm9
   1bf7d:	addsd  xmm0,QWORD PTR [rdx+0x48]
   1bf82:	movsd  QWORD PTR [rdx+0x40],xmm5
   1bf87:	movsd  xmm9,QWORD PTR [rdi+0x50]
   1bf8d:	movsd  xmm11,QWORD PTR [rsi]
   1bf92:	movsd  QWORD PTR [rdx+0x48],xmm0
   1bf97:	movsd  xmm13,QWORD PTR [rdi+0x58]
   1bf9d:	movsd  xmm14,QWORD PTR [rsi+0x8]
   1bfa3:	movapd xmm12,xmm9
   1bfa8:	mulsd  xmm12,xmm11
   1bfad:	movapd xmm15,xmm13
   1bfb2:	mulsd  xmm15,xmm14
   1bfb7:	mulsd  xmm11,xmm13
   1bfbc:	mulsd  xmm9,xmm14
   1bfc1:	subsd  xmm12,xmm15
   1bfc6:	addsd  xmm12,QWORD PTR [rdx+0x50]
   1bfcc:	addsd  xmm9,xmm11
   1bfd1:	addsd  xmm9,QWORD PTR [rdx+0x58]
   1bfd7:	movsd  QWORD PTR [rdx+0x50],xmm12
   1bfdd:	movsd  xmm13,QWORD PTR [rdi+0x60]
   1bfe3:	movsd  xmm11,QWORD PTR [rsi]
   1bfe8:	movsd  QWORD PTR [rsp],xmm12
   1bfee:	movsd  QWORD PTR [rdx+0x58],xmm9
   1bff4:	movsd  xmm12,QWORD PTR [rdi+0x68]
   1bffa:	movapd xmm15,xmm13
   1bfff:	movsd  QWORD PTR [rsp+0x8],xmm9
   1c006:	movsd  xmm9,QWORD PTR [rsi+0x8]
   1c00c:	mulsd  xmm15,xmm11
   1c011:	mulsd  xmm11,xmm12
   1c016:	movapd xmm14,xmm12
   1c01b:	mulsd  xmm14,xmm9
   1c020:	mulsd  xmm9,xmm13
   1c025:	subsd  xmm15,xmm14
   1c02a:	movsd  xmm14,QWORD PTR [rdx+0x68]
   1c030:	addsd  xmm15,QWORD PTR [rdx+0x60]
   1c036:	addsd  xmm9,xmm11
   1c03b:	movsd  QWORD PTR [rdx+0x60],xmm15
   1c041:	movsd  xmm11,QWORD PTR [rsi]
   1c046:	addsd  xmm14,xmm9
   1c04b:	movsd  QWORD PTR [rsp+0x10],xmm15
   1c052:	movsd  xmm9,QWORD PTR [rdi+0x70]
   1c058:	movapd xmm15,xmm9
   1c05d:	movsd  QWORD PTR [rsp+0x18],xmm14
   1c064:	mulsd  xmm15,xmm11
   1c069:	movsd  QWORD PTR [rdx+0x68],xmm14
   1c06f:	movsd  xmm12,QWORD PTR [rdi+0x78]
   1c075:	movsd  xmm13,QWORD PTR [rsi+0x8]
   1c07b:	mulsd  xmm11,xmm12
   1c080:	movapd xmm14,xmm12
   1c085:	mulsd  xmm14,xmm13
   1c08a:	mulsd  xmm9,xmm13
   1c08f:	movsd  xmm13,QWORD PTR [rdx+0x78]
   1c095:	subsd  xmm15,xmm14
   1c09a:	addsd  xmm15,QWORD PTR [rdx+0x70]
   1c0a0:	addsd  xmm9,xmm11
   1c0a5:	movsd  QWORD PTR [rdx+0x70],xmm15
   1c0ab:	movsd  xmm11,QWORD PTR [rsi]
   1c0b0:	addsd  xmm13,xmm9
   1c0b5:	movsd  QWORD PTR [rsp+0x20],xmm15
   1c0bc:	movsd  xmm9,QWORD PTR [rdi+0x80]
   1c0c5:	movapd xmm14,xmm9
   1c0ca:	mulsd  xmm14,xmm11
   1c0cf:	movsd  QWORD PTR [rdx+0x78],xmm13
   1c0d5:	movsd  xmm12,QWORD PTR [rdi+0x88]
   1c0de:	movsd  QWORD PTR [rsp+0x28],xmm13
   1c0e5:	movsd  xmm13,QWORD PTR [rsi+0x8]
   1c0eb:	mulsd  xmm11,xmm12
   1c0f0:	mulsd  xmm9,xmm13
   1c0f5:	movapd xmm15,xmm14
   1c0fa:	movapd xmm14,xmm12
   1c0ff:	mulsd  xmm14,xmm13
   1c104:	addsd  xmm9,xmm11
   1c109:	movsd  xmm11,QWORD PTR [rdx+0x88]
   1c112:	subsd  xmm15,xmm14
   1c117:	addsd  xmm11,xmm9
   1c11c:	movsd  xmm14,QWORD PTR [rdx+0x80]
   1c125:	addsd  xmm14,xmm15
   1c12a:	movsd  QWORD PTR [rsp+0x38],xmm11
   1c131:	movsd  QWORD PTR [rdx+0x88],xmm11
   1c13a:	movsd  xmm13,QWORD PTR [rsi+0x8]
   1c140:	movsd  xmm12,QWORD PTR [rdi+0x98]
   1c149:	movsd  QWORD PTR [rdx+0x80],xmm14
   1c152:	movsd  xmm11,QWORD PTR [rsi]
   1c157:	movsd  xmm9,QWORD PTR [rdi+0x90]
   1c160:	movsd  QWORD PTR [rsp+0x30],xmm14
   1c167:	movapd xmm14,xmm12
   1c16c:	mulsd  xmm14,xmm13
   1c171:	movapd xmm15,xmm9
   1c176:	mulsd  xmm15,xmm11
   1c17b:	mulsd  xmm9,xmm13
   1c180:	mulsd  xmm11,xmm12
   1c185:	subsd  xmm15,xmm14
   1c18a:	addsd  xmm15,QWORD PTR [rdx+0x90]
   1c193:	addsd  xmm9,xmm11
   1c198:	addsd  xmm9,QWORD PTR [rdx+0x98]
   1c1a1:	movsd  QWORD PTR [rdx+0x90],xmm15
   1c1aa:	movsd  xmm11,QWORD PTR [rsi]
   1c1af:	movsd  QWORD PTR [rsp+0x40],xmm15
   1c1b6:	movsd  QWORD PTR [rdx+0x98],xmm9
   1c1bf:	movsd  xmm13,QWORD PTR [rsi+0x8]
   1c1c5:	movsd  xmm12,QWORD PTR [rdi+0xa8]
   1c1ce:	movsd  QWORD PTR [rsp+0x48],xmm9
   1c1d5:	movsd  xmm9,QWORD PTR [rdi+0xa0]
   1c1de:	movapd xmm14,xmm9
   1c1e3:	mulsd  xmm14,xmm11
   1c1e8:	mulsd  xmm9,xmm13
   1c1ed:	mulsd  xmm11,xmm12
   1c1f2:	movapd xmm15,xmm14
   1c1f7:	movapd xmm14,xmm12
   1c1fc:	mulsd  xmm14,xmm13
   1c201:	addsd  xmm9,xmm11
   1c206:	subsd  xmm15,xmm14
   1c20b:	movsd  xmm14,QWORD PTR [rdx+0xa0]
   1c214:	movsd  xmm13,QWORD PTR [rdx+0xa8]
   1c21d:	addsd  xmm13,xmm9
   1c222:	addsd  xmm14,xmm15
   1c227:	movsd  QWORD PTR [rdx+0xa8],xmm13
   1c230:	movsd  xmm12,QWORD PTR [rdi+0xb8]
   1c239:	movsd  QWORD PTR [rdx+0xa0],xmm14
   1c242:	movsd  xmm11,QWORD PTR [rsi]
   1c247:	movsd  xmm9,QWORD PTR [rdi+0xb0]
   1c250:	movsd  QWORD PTR [rsp+0x58],xmm13
   1c257:	movsd  xmm13,QWORD PTR [rsi+0x8]
   1c25d:	movsd  QWORD PTR [rsp+0x50],xmm14
   1c264:	movapd xmm15,xmm9
   1c269:	movapd xmm14,xmm12
   1c26e:	mulsd  xmm14,xmm13
   1c273:	mulsd  xmm15,xmm11
   1c278:	mulsd  xmm9,xmm13
   1c27d:	mulsd  xmm11,xmm12
   1c282:	movsd  xmm12,QWORD PTR [rdx+0xb8]
   1c28b:	subsd  xmm15,xmm14
   1c290:	addsd  xmm15,QWORD PTR [rdx+0xb0]
   1c299:	addsd  xmm9,xmm11
   1c29e:	movsd  QWORD PTR [rdx+0xb0],xmm15
   1c2a7:	movsd  xmm11,QWORD PTR [rsi]
   1c2ac:	movsd  QWORD PTR [rsp+0x60],xmm15
   1c2b3:	addsd  xmm12,xmm9
   1c2b8:	movsd  xmm9,QWORD PTR [rdi+0xc0]
   1c2c1:	movapd xmm14,xmm9
   1c2c6:	mulsd  xmm14,xmm11
   1c2cb:	movsd  QWORD PTR [rsp+0x68],xmm12
   1c2d2:	movsd  QWORD PTR [rdx+0xb8],xmm12
   1c2db:	movsd  xmm13,QWORD PTR [rsi+0x8]
   1c2e1:	movsd  xmm12,QWORD PTR [rdi+0xc8]
   1c2ea:	mulsd  xmm9,xmm13
   1c2ef:	mulsd  xmm11,xmm12
   1c2f4:	movapd xmm15,xmm14
   1c2f9:	movapd xmm14,xmm12
   1c2fe:	mulsd  xmm14,xmm13
   1c303:	addsd  xmm9,xmm11
   1c308:	movsd  xmm11,QWORD PTR [rdx+0xc8]
   1c311:	subsd  xmm15,xmm14
   1c316:	movsd  xmm14,QWORD PTR [rdx+0xc0]
   1c31f:	addsd  xmm11,xmm9
   1c324:	addsd  xmm14,xmm15
   1c329:	movsd  QWORD PTR [rdx+0xc8],xmm11
   1c332:	movsd  xmm13,QWORD PTR [rsi+0x8]
   1c338:	movsd  xmm12,QWORD PTR [rdi+0xd8]
   1c341:	movsd  QWORD PTR [rsp+0x78],xmm11
   1c348:	movsd  QWORD PTR [rdx+0xc0],xmm14
   1c351:	movsd  xmm11,QWORD PTR [rsi]
   1c356:	movsd  xmm9,QWORD PTR [rdi+0xd0]
   1c35f:	movsd  QWORD PTR [rsp+0x70],xmm14
   1c366:	movapd xmm14,xmm12
   1c36b:	mulsd  xmm14,xmm13
   1c370:	movapd xmm15,xmm9
   1c375:	mulsd  xmm15,xmm11
   1c37a:	mulsd  xmm9,xmm13
   1c37f:	mulsd  xmm11,xmm12
   1c384:	subsd  xmm15,xmm14
   1c389:	addsd  xmm15,QWORD PTR [rdx+0xd0]
   1c392:	addsd  xmm9,xmm11
   1c397:	movsd  QWORD PTR [rsp+0x80],xmm15
   1c3a1:	movsd  QWORD PTR [rdx+0xd0],xmm15
   1c3aa:	movsd  xmm11,QWORD PTR [rsi]
   1c3af:	addsd  xmm9,QWORD PTR [rdx+0xd8]
   1c3b8:	movsd  QWORD PTR [rdx+0xd8],xmm9
   1c3c1:	movsd  xmm13,QWORD PTR [rsi+0x8]
   1c3c7:	movsd  xmm12,QWORD PTR [rdi+0xe8]
   1c3d0:	movsd  QWORD PTR [rsp+0x88],xmm9
   1c3da:	movsd  xmm9,QWORD PTR [rdi+0xe0]
   1c3e3:	movapd xmm14,xmm9
   1c3e8:	mulsd  xmm14,xmm11
   1c3ed:	mulsd  xmm9,xmm13
   1c3f2:	mulsd  xmm11,xmm12
   1c3f7:	movapd xmm15,xmm14
   1c3fc:	movapd xmm14,xmm12
   1c401:	mulsd  xmm14,xmm13
   1c406:	movsd  xmm13,QWORD PTR [rdx+0xe8]
   1c40f:	addsd  xmm9,xmm11
   1c414:	addsd  xmm13,xmm9
   1c419:	subsd  xmm15,xmm14
   1c41e:	movsd  xmm14,QWORD PTR [rdx+0xe0]
   1c427:	movsd  QWORD PTR [rdx+0xe8],xmm13
   1c430:	movsd  xmm12,QWORD PTR [rdi+0xf8]
   1c439:	movsd  QWORD PTR [rsp+0x98],xmm13
   1c443:	addsd  xmm14,xmm15
   1c448:	movsd  xmm13,QWORD PTR [rsi+0x8]
   1c44e:	movsd  QWORD PTR [rdx+0xe0],xmm14
   1c457:	movsd  xmm11,QWORD PTR [rsi]
   1c45c:	movsd  xmm9,QWORD PTR [rdi+0xf0]
   1c465:	movsd  QWORD PTR [rsp+0x90],xmm14
   1c46f:	movapd xmm14,xmm12
   1c474:	mulsd  xmm14,xmm13
   1c479:	movapd xmm15,xmm9
   1c47e:	mulsd  xmm15,xmm11
   1c483:	mulsd  xmm9,xmm13
   1c488:	mulsd  xmm11,xmm12
   1c48d:	movsd  xmm12,QWORD PTR [rdx+0xf8]
   1c496:	subsd  xmm15,xmm14
   1c49b:	addsd  xmm15,QWORD PTR [rdx+0xf0]
   1c4a4:	addsd  xmm9,xmm11
   1c4a9:	movsd  QWORD PTR [rdx+0xf0],xmm15
   1c4b2:	movsd  xmm11,QWORD PTR [rsi+0x10]
   1c4b8:	movsd  QWORD PTR [rsp+0xa0],xmm15
   1c4c2:	addsd  xmm12,xmm9
   1c4c7:	movsd  xmm9,QWORD PTR [rdi+0x100]
   1c4d0:	movsd  QWORD PTR [rdx+0xf8],xmm12
   1c4d9:	movsd  xmm14,QWORD PTR [rsi+0x18]
   1c4df:	movsd  xmm13,QWORD PTR [rdi+0x108]
   1c4e8:	movsd  QWORD PTR [rsp+0xa8],xmm12
   1c4f2:	movapd xmm12,xmm9
   1c4f7:	mulsd  xmm12,xmm11
   1c4fc:	movapd xmm15,xmm13
   1c501:	mulsd  xmm15,xmm14
   1c506:	mulsd  xmm11,xmm13
   1c50b:	mulsd  xmm9,xmm14
   1c510:	subsd  xmm12,xmm15
   1c515:	addsd  xmm9,xmm11
   1c51a:	addsd  xmm12,xmm10
   1c51f:	movapd xmm10,xmm9
   1c524:	addsd  xmm10,xmm4
   1c529:	movsd  QWORD PTR [rdx],xmm12
   1c52e:	movapd xmm15,xmm12
   1c533:	movsd  xmm4,QWORD PTR [rdi+0x110]
   1c53b:	movsd  xmm9,QWORD PTR [rsi+0x10]
   1c541:	movsd  QWORD PTR [rdx+0x8],xmm10
   1c547:	movapd xmm14,xmm10
   1c54c:	movapd xmm10,xmm4
   1c551:	movsd  xmm11,QWORD PTR [rdi+0x118]
   1c55a:	movsd  xmm12,QWORD PTR [rsi+0x18]
   1c560:	mulsd  xmm10,xmm9
   1c565:	mulsd  xmm9,xmm11
   1c56a:	movapd xmm13,xmm11
   1c56f:	mulsd  xmm13,xmm12
   1c574:	mulsd  xmm4,xmm12
   1c579:	subsd  xmm10,xmm13
   1c57e:	addsd  xmm4,xmm9
   1c583:	addsd  xmm10,xmm8
   1c588:	addsd  xmm4,xmm3
   1c58c:	movsd  QWORD PTR [rdx+0x10],xmm10
   1c592:	movsd  xmm3,QWORD PTR [rdi+0x120]
   1c59a:	movapd xmm13,xmm10
   1c59f:	movsd  QWORD PTR [rdx+0x18],xmm4
   1c5a4:	movapd xmm12,xmm4
   1c5a9:	movsd  xmm10,QWORD PTR [rsi+0x18]
   1c5af:	movsd  xmm9,QWORD PTR [rdi+0x128]
   1c5b8:	movsd  xmm4,QWORD PTR [rsi+0x10]
   1c5bd:	movapd xmm8,xmm3
   1c5c2:	mulsd  xmm3,xmm10
   1c5c7:	movapd xmm11,xmm9
   1c5cc:	mulsd  xmm11,xmm10
   1c5d1:	mulsd  xmm8,xmm4
   1c5d6:	mulsd  xmm4,xmm9
   1c5db:	subsd  xmm8,xmm11
   1c5e0:	addsd  xmm3,xmm4
   1c5e4:	addsd  xmm8,xmm7
   1c5e9:	addsd  xmm3,xmm2
   1c5ed:	movsd  QWORD PTR [rdx+0x20],xmm8
   1c5f3:	movsd  xmm2,QWORD PTR [rdi+0x130]
   1c5fb:	movapd xmm11,xmm8
   1c600:	movsd  QWORD PTR [rdx+0x28],xmm3
   1c605:	movapd xmm10,xmm3
   1c60a:	movsd  xmm3,QWORD PTR [rsi+0x10]
   1c60f:	movsd  xmm7,QWORD PTR [rdi+0x138]
   1c617:	movsd  xmm8,QWORD PTR [rsi+0x18]
   1c61d:	movapd xmm4,xmm2
   1c621:	mulsd  xmm4,xmm3
   1c625:	movapd xmm9,xmm7
   1c62a:	mulsd  xmm9,xmm8
   1c62f:	mulsd  xmm3,xmm7
   1c633:	mulsd  xmm2,xmm8
   1c638:	subsd  xmm4,xmm9
   1c63d:	addsd  xmm2,xmm3
   1c641:	addsd  xmm4,xmm6
   1c645:	addsd  xmm2,xmm1
   1c649:	movsd  QWORD PTR [rdx+0x30],xmm4
   1c64e:	movapd xmm9,xmm4
   1c653:	movsd  xmm1,QWORD PTR [rdi+0x140]
   1c65b:	movsd  QWORD PTR [rdx+0x38],xmm2
   1c660:	movapd xmm8,xmm2
   1c665:	movsd  xmm6,QWORD PTR [rsi+0x18]
   1c66a:	movsd  xmm4,QWORD PTR [rdi+0x148]
   1c672:	movsd  xmm2,QWORD PTR [rsi+0x10]
   1c677:	movapd xmm3,xmm1
   1c67b:	mulsd  xmm1,xmm6
   1c67f:	movapd xmm7,xmm4
   1c683:	mulsd  xmm7,xmm6
   1c687:	mulsd  xmm3,xmm2
   1c68b:	mulsd  xmm2,xmm4
   1c68f:	subsd  xmm3,xmm7
   1c693:	addsd  xmm1,xmm2
   1c697:	movapd xmm7,xmm3
   1c69b:	movapd xmm6,xmm1
   1c69f:	addsd  xmm7,xmm5
   1c6a3:	addsd  xmm6,xmm0
   1c6a7:	movsd  QWORD PTR [rdx+0x40],xmm7
   1c6ac:	movsd  xmm0,QWORD PTR [rdi+0x150]
   1c6b4:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1c6b9:	movsd  QWORD PTR [rdx+0x48],xmm6
   1c6be:	movsd  xmm3,QWORD PTR [rdi+0x158]
   1c6c6:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1c6cb:	movapd xmm2,xmm0
   1c6cf:	mulsd  xmm2,xmm1
   1c6d3:	movapd xmm5,xmm3
   1c6d7:	mulsd  xmm5,xmm4
   1c6db:	mulsd  xmm1,xmm3
   1c6df:	mulsd  xmm0,xmm4
   1c6e3:	subsd  xmm2,xmm5
   1c6e7:	addsd  xmm2,QWORD PTR [rsp]
   1c6ec:	addsd  xmm0,xmm1
   1c6f0:	addsd  xmm0,QWORD PTR [rsp+0x8]
   1c6f6:	movsd  QWORD PTR [rdx+0x50],xmm2
   1c6fb:	movsd  QWORD PTR [rsp],xmm2
   1c700:	movsd  QWORD PTR [rsp+0x8],xmm0
   1c706:	movsd  QWORD PTR [rdx+0x58],xmm0
   1c70b:	movsd  xmm0,QWORD PTR [rdi+0x160]
   1c713:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1c718:	movsd  xmm3,QWORD PTR [rdi+0x168]
   1c720:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1c725:	movapd xmm2,xmm0
   1c729:	mulsd  xmm2,xmm1
   1c72d:	movapd xmm5,xmm3
   1c731:	mulsd  xmm1,xmm3
   1c735:	mulsd  xmm0,xmm4
   1c739:	mulsd  xmm5,xmm4
   1c73d:	movsd  xmm4,QWORD PTR [rsp+0x18]
   1c743:	addsd  xmm0,xmm1
   1c747:	subsd  xmm2,xmm5
   1c74b:	addsd  xmm2,QWORD PTR [rsp+0x10]
   1c751:	addsd  xmm4,xmm0
   1c755:	movsd  QWORD PTR [rdx+0x60],xmm2
   1c75a:	movsd  xmm0,QWORD PTR [rdi+0x170]
   1c762:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1c767:	movsd  QWORD PTR [rsp+0x10],xmm2
   1c76d:	movsd  QWORD PTR [rdx+0x68],xmm4
   1c772:	movsd  xmm3,QWORD PTR [rdi+0x178]
   1c77a:	movapd xmm2,xmm0
   1c77e:	movsd  QWORD PTR [rsp+0x18],xmm4
   1c784:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1c789:	mulsd  xmm2,xmm1
   1c78d:	mulsd  xmm1,xmm3
   1c791:	movapd xmm5,xmm3
   1c795:	movsd  xmm3,QWORD PTR [rsp+0x28]
   1c79b:	mulsd  xmm0,xmm4
   1c79f:	mulsd  xmm5,xmm4
   1c7a3:	addsd  xmm0,xmm1
   1c7a7:	subsd  xmm2,xmm5
   1c7ab:	addsd  xmm2,QWORD PTR [rsp+0x20]
   1c7b1:	addsd  xmm3,xmm0
   1c7b5:	movsd  QWORD PTR [rdx+0x70],xmm2
   1c7ba:	movsd  xmm0,QWORD PTR [rdi+0x180]
   1c7c2:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1c7c7:	movsd  QWORD PTR [rsp+0x20],xmm2
   1c7cd:	movsd  QWORD PTR [rdx+0x78],xmm3
   1c7d2:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1c7d7:	movapd xmm2,xmm0
   1c7db:	movsd  QWORD PTR [rsp+0x28],xmm3
   1c7e1:	movsd  xmm3,QWORD PTR [rdi+0x188]
   1c7e9:	mulsd  xmm2,xmm1
   1c7ed:	mulsd  xmm0,xmm4
   1c7f1:	mulsd  xmm1,xmm3
   1c7f5:	movapd xmm5,xmm3
   1c7f9:	mulsd  xmm5,xmm4
   1c7fd:	addsd  xmm0,xmm1
   1c801:	movsd  xmm1,QWORD PTR [rsp+0x38]
   1c807:	subsd  xmm2,xmm5
   1c80b:	addsd  xmm2,QWORD PTR [rsp+0x30]
   1c811:	addsd  xmm1,xmm0
   1c815:	movsd  QWORD PTR [rdx+0x80],xmm2
   1c81d:	movsd  xmm0,QWORD PTR [rdi+0x190]
   1c825:	movsd  QWORD PTR [rsp+0x30],xmm2
   1c82b:	movsd  QWORD PTR [rdx+0x88],xmm1
   1c833:	movsd  xmm3,QWORD PTR [rdi+0x198]
   1c83b:	movapd xmm2,xmm0
   1c83f:	movsd  QWORD PTR [rsp+0x38],xmm1
   1c845:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1c84a:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1c84f:	movapd xmm5,xmm3
   1c853:	mulsd  xmm5,xmm4
   1c857:	mulsd  xmm2,xmm1
   1c85b:	mulsd  xmm0,xmm4
   1c85f:	mulsd  xmm1,xmm3
   1c863:	subsd  xmm2,xmm5
   1c867:	addsd  xmm2,QWORD PTR [rsp+0x40]
   1c86d:	addsd  xmm0,xmm1
   1c871:	addsd  xmm0,QWORD PTR [rsp+0x48]
   1c877:	movsd  QWORD PTR [rdx+0x90],xmm2
   1c87f:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1c884:	movsd  QWORD PTR [rsp+0x40],xmm2
   1c88a:	movsd  QWORD PTR [rdx+0x98],xmm0
   1c892:	movsd  xmm3,QWORD PTR [rdi+0x1a8]
   1c89a:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1c89f:	movsd  QWORD PTR [rsp+0x48],xmm0
   1c8a5:	movsd  xmm0,QWORD PTR [rdi+0x1a0]
   1c8ad:	movapd xmm5,xmm3
   1c8b1:	mulsd  xmm5,xmm4
   1c8b5:	movapd xmm2,xmm0
   1c8b9:	mulsd  xmm2,xmm1
   1c8bd:	mulsd  xmm0,xmm4
   1c8c1:	movsd  xmm4,QWORD PTR [rsp+0x58]
   1c8c7:	mulsd  xmm1,xmm3
   1c8cb:	subsd  xmm2,xmm5
   1c8cf:	addsd  xmm2,QWORD PTR [rsp+0x50]
   1c8d5:	addsd  xmm0,xmm1
   1c8d9:	movsd  QWORD PTR [rdx+0xa0],xmm2
   1c8e1:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1c8e6:	movsd  QWORD PTR [rsp+0x50],xmm2
   1c8ec:	addsd  xmm4,xmm0
   1c8f0:	movsd  xmm0,QWORD PTR [rdi+0x1b0]
   1c8f8:	movapd xmm2,xmm0
   1c8fc:	movsd  QWORD PTR [rdx+0xa8],xmm4
   1c904:	movsd  xmm3,QWORD PTR [rdi+0x1b8]
   1c90c:	mulsd  xmm2,xmm1
   1c910:	movsd  QWORD PTR [rsp+0x58],xmm4
   1c916:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1c91b:	mulsd  xmm1,xmm3
   1c91f:	movapd xmm5,xmm3
   1c923:	movsd  xmm3,QWORD PTR [rsp+0x68]
   1c929:	mulsd  xmm0,xmm4
   1c92d:	mulsd  xmm5,xmm4
   1c931:	addsd  xmm0,xmm1
   1c935:	subsd  xmm2,xmm5
   1c939:	addsd  xmm2,QWORD PTR [rsp+0x60]
   1c93f:	addsd  xmm3,xmm0
   1c943:	movsd  QWORD PTR [rdx+0xb0],xmm2
   1c94b:	movsd  xmm0,QWORD PTR [rdi+0x1c0]
   1c953:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1c958:	movsd  QWORD PTR [rsp+0x60],xmm2
   1c95e:	movsd  QWORD PTR [rsp+0x68],xmm3
   1c964:	movapd xmm2,xmm0
   1c968:	movsd  QWORD PTR [rdx+0xb8],xmm3
   1c970:	movsd  xmm3,QWORD PTR [rdi+0x1c8]
   1c978:	mulsd  xmm2,xmm1
   1c97c:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1c981:	mulsd  xmm1,xmm3
   1c985:	movapd xmm5,xmm3
   1c989:	mulsd  xmm5,xmm4
   1c98d:	mulsd  xmm0,xmm4
   1c991:	subsd  xmm2,xmm5
   1c995:	addsd  xmm2,QWORD PTR [rsp+0x70]
   1c99b:	addsd  xmm0,xmm1
   1c99f:	movsd  QWORD PTR [rsp+0x70],xmm2
   1c9a5:	movsd  xmm1,QWORD PTR [rsp+0x78]
   1c9ab:	movsd  QWORD PTR [rdx+0xc0],xmm2
   1c9b3:	addsd  xmm1,xmm0
   1c9b7:	movsd  xmm0,QWORD PTR [rdi+0x1d0]
   1c9bf:	movapd xmm2,xmm0
   1c9c3:	movsd  QWORD PTR [rdx+0xc8],xmm1
   1c9cb:	movsd  xmm3,QWORD PTR [rdi+0x1d8]
   1c9d3:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1c9d8:	movsd  QWORD PTR [rsp+0x78],xmm1
   1c9de:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1c9e3:	mulsd  xmm0,xmm4
   1c9e7:	movapd xmm5,xmm3
   1c9eb:	mulsd  xmm5,xmm4
   1c9ef:	mulsd  xmm2,xmm1
   1c9f3:	mulsd  xmm1,xmm3
   1c9f7:	subsd  xmm2,xmm5
   1c9fb:	addsd  xmm2,QWORD PTR [rsp+0x80]
   1ca04:	addsd  xmm0,xmm1
   1ca08:	addsd  xmm0,QWORD PTR [rsp+0x88]
   1ca11:	movsd  QWORD PTR [rdx+0xd0],xmm2
   1ca19:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1ca1e:	movsd  QWORD PTR [rdx+0xd8],xmm0
   1ca26:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1ca2b:	movsd  xmm3,QWORD PTR [rdi+0x1e8]
   1ca33:	movsd  QWORD PTR [rsp+0x88],xmm0
   1ca3c:	movsd  xmm0,QWORD PTR [rdi+0x1e0]
   1ca44:	movsd  QWORD PTR [rsp+0x80],xmm2
   1ca4d:	movapd xmm5,xmm3
   1ca51:	mulsd  xmm5,xmm4
   1ca55:	movapd xmm2,xmm0
   1ca59:	mulsd  xmm2,xmm1
   1ca5d:	mulsd  xmm0,xmm4
   1ca61:	mulsd  xmm1,xmm3
   1ca65:	subsd  xmm2,xmm5
   1ca69:	addsd  xmm2,QWORD PTR [rsp+0x90]
   1ca72:	addsd  xmm0,xmm1
   1ca76:	addsd  xmm0,QWORD PTR [rsp+0x98]
   1ca7f:	movsd  QWORD PTR [rdx+0xe0],xmm2
   1ca87:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1ca8c:	movsd  QWORD PTR [rsp+0x90],xmm2
   1ca95:	movsd  QWORD PTR [rdx+0xe8],xmm0
   1ca9d:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1caa2:	movsd  xmm3,QWORD PTR [rdi+0x1f8]
   1caaa:	movsd  QWORD PTR [rsp+0x98],xmm0
   1cab3:	movsd  xmm0,QWORD PTR [rdi+0x1f0]
   1cabb:	movapd xmm5,xmm0
   1cabf:	mulsd  xmm5,xmm1
   1cac3:	mulsd  xmm0,xmm4
   1cac7:	mulsd  xmm1,xmm3
   1cacb:	movapd xmm2,xmm5
   1cacf:	movapd xmm5,xmm3
   1cad3:	movsd  xmm3,QWORD PTR [rsp+0xa8]
   1cadc:	mulsd  xmm5,xmm4
   1cae0:	addsd  xmm0,xmm1
   1cae4:	addsd  xmm3,xmm0
   1cae8:	subsd  xmm2,xmm5
   1caec:	addsd  xmm2,QWORD PTR [rsp+0xa0]
   1caf5:	movsd  QWORD PTR [rsp+0xa0],xmm2
   1cafe:	movsd  QWORD PTR [rdx+0xf0],xmm2
   1cb06:	movsd  QWORD PTR [rsp+0xa8],xmm3
   1cb0f:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1cb14:	movsd  xmm0,QWORD PTR [rdi+0x200]
   1cb1c:	movsd  QWORD PTR [rdx+0xf8],xmm3
   1cb24:	movsd  xmm3,QWORD PTR [rdi+0x208]
   1cb2c:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1cb31:	movapd xmm5,xmm0
   1cb35:	mulsd  xmm5,xmm1
   1cb39:	mulsd  xmm0,xmm4
   1cb3d:	mulsd  xmm1,xmm3
   1cb41:	movapd xmm2,xmm5
   1cb45:	movapd xmm5,xmm3
   1cb49:	mulsd  xmm5,xmm4
   1cb4d:	addsd  xmm0,xmm1
   1cb51:	addsd  xmm14,xmm0
   1cb56:	subsd  xmm2,xmm5
   1cb5a:	movsd  QWORD PTR [rdx+0x8],xmm14
   1cb60:	movsd  xmm3,QWORD PTR [rdi+0x218]
   1cb68:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1cb6d:	addsd  xmm15,xmm2
   1cb72:	movsd  QWORD PTR [rdx],xmm15
   1cb77:	movsd  xmm0,QWORD PTR [rdi+0x210]
   1cb7f:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1cb84:	movapd xmm5,xmm0
   1cb88:	mulsd  xmm5,xmm1
   1cb8c:	mulsd  xmm0,xmm4
   1cb90:	mulsd  xmm1,xmm3
   1cb94:	movapd xmm2,xmm5
   1cb98:	movapd xmm5,xmm3
   1cb9c:	mulsd  xmm5,xmm4
   1cba0:	addsd  xmm0,xmm1
   1cba4:	addsd  xmm12,xmm0
   1cba9:	subsd  xmm2,xmm5
   1cbad:	movsd  QWORD PTR [rdx+0x18],xmm12
   1cbb3:	movsd  xmm3,QWORD PTR [rdi+0x228]
   1cbbb:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1cbc0:	addsd  xmm13,xmm2
   1cbc5:	movsd  QWORD PTR [rdx+0x10],xmm13
   1cbcb:	movsd  xmm0,QWORD PTR [rdi+0x220]
   1cbd3:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1cbd8:	movapd xmm5,xmm0
   1cbdc:	mulsd  xmm5,xmm1
   1cbe0:	mulsd  xmm0,xmm4
   1cbe4:	mulsd  xmm1,xmm3
   1cbe8:	movapd xmm2,xmm5
   1cbec:	movapd xmm5,xmm3
   1cbf0:	mulsd  xmm5,xmm4
   1cbf4:	addsd  xmm0,xmm1
   1cbf8:	addsd  xmm10,xmm0
   1cbfd:	subsd  xmm2,xmm5
   1cc01:	movsd  QWORD PTR [rdx+0x28],xmm10
   1cc07:	movsd  xmm3,QWORD PTR [rdi+0x238]
   1cc0f:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1cc14:	addsd  xmm11,xmm2
   1cc19:	movsd  QWORD PTR [rdx+0x20],xmm11
   1cc1f:	movsd  xmm0,QWORD PTR [rdi+0x230]
   1cc27:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1cc2c:	movapd xmm5,xmm0
   1cc30:	mulsd  xmm5,xmm1
   1cc34:	mulsd  xmm0,xmm4
   1cc38:	mulsd  xmm1,xmm3
   1cc3c:	movapd xmm2,xmm5
   1cc40:	movapd xmm5,xmm3
   1cc44:	mulsd  xmm5,xmm4
   1cc48:	addsd  xmm0,xmm1
   1cc4c:	addsd  xmm8,xmm0
   1cc51:	subsd  xmm2,xmm5
   1cc55:	movsd  QWORD PTR [rdx+0x38],xmm8
   1cc5b:	movsd  xmm3,QWORD PTR [rdi+0x248]
   1cc63:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1cc68:	addsd  xmm9,xmm2
   1cc6d:	movsd  QWORD PTR [rdx+0x30],xmm9
   1cc73:	movsd  xmm0,QWORD PTR [rdi+0x240]
   1cc7b:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1cc80:	movapd xmm5,xmm0
   1cc84:	mulsd  xmm5,xmm1
   1cc88:	mulsd  xmm0,xmm4
   1cc8c:	mulsd  xmm1,xmm3
   1cc90:	movapd xmm2,xmm5
   1cc94:	movapd xmm5,xmm3
   1cc98:	mulsd  xmm5,xmm4
   1cc9c:	addsd  xmm0,xmm1
   1cca0:	addsd  xmm6,xmm0
   1cca4:	subsd  xmm2,xmm5
   1cca8:	movsd  QWORD PTR [rdx+0x48],xmm6
   1ccad:	addsd  xmm7,xmm2
   1ccb1:	movsd  QWORD PTR [rdx+0x40],xmm7
   1ccb6:	movsd  xmm0,QWORD PTR [rdi+0x250]
   1ccbe:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1ccc3:	movsd  xmm3,QWORD PTR [rdi+0x258]
   1cccb:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1ccd0:	movapd xmm5,xmm0
   1ccd4:	mulsd  xmm5,xmm1
   1ccd8:	mulsd  xmm0,xmm4
   1ccdc:	mulsd  xmm1,xmm3
   1cce0:	movapd xmm2,xmm5
   1cce4:	movapd xmm5,xmm3
   1cce8:	mulsd  xmm5,xmm4
   1ccec:	addsd  xmm0,xmm1
   1ccf0:	movsd  xmm1,QWORD PTR [rsp+0x8]
   1ccf6:	addsd  xmm1,xmm0
   1ccfa:	subsd  xmm2,xmm5
   1ccfe:	addsd  xmm2,QWORD PTR [rsp]
   1cd03:	movsd  QWORD PTR [rdx+0x58],xmm1
   1cd08:	movsd  xmm3,QWORD PTR [rdi+0x268]
   1cd10:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1cd15:	movsd  QWORD PTR [rdx+0x50],xmm2
   1cd1a:	movsd  xmm0,QWORD PTR [rdi+0x260]
   1cd22:	movsd  QWORD PTR [rsp+0x8],xmm1
   1cd28:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1cd2d:	movapd xmm5,xmm0
   1cd31:	movsd  QWORD PTR [rsp],xmm2
   1cd36:	mulsd  xmm5,xmm1
   1cd3a:	mulsd  xmm0,xmm4
   1cd3e:	mulsd  xmm1,xmm3
   1cd42:	movapd xmm2,xmm5
   1cd46:	movapd xmm5,xmm3
   1cd4a:	mulsd  xmm5,xmm4
   1cd4e:	addsd  xmm0,xmm1
   1cd52:	addsd  xmm0,QWORD PTR [rsp+0x18]
   1cd58:	movsd  QWORD PTR [rdx+0x68],xmm0
   1cd5d:	movsd  xmm3,QWORD PTR [rdi+0x278]
   1cd65:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1cd6a:	subsd  xmm2,xmm5
   1cd6e:	addsd  xmm2,QWORD PTR [rsp+0x10]
   1cd74:	movsd  QWORD PTR [rsp+0x18],xmm0
   1cd7a:	movapd xmm5,xmm3
   1cd7e:	mulsd  xmm5,xmm4
   1cd82:	movsd  QWORD PTR [rdx+0x60],xmm2
   1cd87:	movsd  xmm0,QWORD PTR [rdi+0x270]
   1cd8f:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1cd94:	movsd  QWORD PTR [rsp+0x10],xmm2
   1cd9a:	movapd xmm2,xmm0
   1cd9e:	mulsd  xmm2,xmm1
   1cda2:	mulsd  xmm0,xmm4
   1cda6:	mulsd  xmm1,xmm3
   1cdaa:	subsd  xmm2,xmm5
   1cdae:	addsd  xmm2,QWORD PTR [rsp+0x20]
   1cdb4:	addsd  xmm0,xmm1
   1cdb8:	addsd  xmm0,QWORD PTR [rsp+0x28]
   1cdbe:	movsd  QWORD PTR [rdx+0x70],xmm2
   1cdc3:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1cdc8:	movsd  QWORD PTR [rsp+0x20],xmm2
   1cdce:	movsd  QWORD PTR [rsp+0x28],xmm0
   1cdd4:	movsd  QWORD PTR [rdx+0x78],xmm0
   1cdd9:	movsd  xmm0,QWORD PTR [rdi+0x280]
   1cde1:	movsd  xmm3,QWORD PTR [rdi+0x288]
   1cde9:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1cdee:	movapd xmm2,xmm0
   1cdf2:	movapd xmm5,xmm3
   1cdf6:	mulsd  xmm5,xmm4
   1cdfa:	mulsd  xmm2,xmm1
   1cdfe:	mulsd  xmm0,xmm4
   1ce02:	mulsd  xmm1,xmm3
   1ce06:	subsd  xmm2,xmm5
   1ce0a:	addsd  xmm2,QWORD PTR [rsp+0x30]
   1ce10:	addsd  xmm0,xmm1
   1ce14:	addsd  xmm0,QWORD PTR [rsp+0x38]
   1ce1a:	movsd  QWORD PTR [rdx+0x80],xmm2
   1ce22:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1ce27:	movsd  QWORD PTR [rsp+0x30],xmm2
   1ce2d:	movsd  QWORD PTR [rdx+0x88],xmm0
   1ce35:	movsd  xmm3,QWORD PTR [rdi+0x298]
   1ce3d:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1ce42:	movsd  QWORD PTR [rsp+0x38],xmm0
   1ce48:	movsd  xmm0,QWORD PTR [rdi+0x290]
   1ce50:	movapd xmm5,xmm3
   1ce54:	mulsd  xmm5,xmm4
   1ce58:	movapd xmm2,xmm0
   1ce5c:	mulsd  xmm2,xmm1
   1ce60:	mulsd  xmm0,xmm4
   1ce64:	mulsd  xmm1,xmm3
   1ce68:	subsd  xmm2,xmm5
   1ce6c:	addsd  xmm2,QWORD PTR [rsp+0x40]
   1ce72:	addsd  xmm0,xmm1
   1ce76:	movsd  xmm1,QWORD PTR [rsp+0x48]
   1ce7c:	movsd  QWORD PTR [rdx+0x90],xmm2
   1ce84:	movsd  QWORD PTR [rsp+0x40],xmm2
   1ce8a:	addsd  xmm1,xmm0
   1ce8e:	movsd  xmm0,QWORD PTR [rdi+0x2a0]
   1ce96:	movapd xmm2,xmm0
   1ce9a:	movsd  QWORD PTR [rdx+0x98],xmm1
   1cea2:	movsd  xmm3,QWORD PTR [rdi+0x2a8]
   1ceaa:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1ceaf:	movsd  QWORD PTR [rsp+0x48],xmm1
   1ceb5:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1ceba:	mulsd  xmm0,xmm4
   1cebe:	movapd xmm5,xmm3
   1cec2:	mulsd  xmm5,xmm4
   1cec6:	mulsd  xmm2,xmm1
   1ceca:	mulsd  xmm1,xmm3
   1cece:	subsd  xmm2,xmm5
   1ced2:	addsd  xmm2,QWORD PTR [rsp+0x50]
   1ced8:	addsd  xmm0,xmm1
   1cedc:	addsd  xmm0,QWORD PTR [rsp+0x58]
   1cee2:	movsd  QWORD PTR [rdx+0xa0],xmm2
   1ceea:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1ceef:	movsd  QWORD PTR [rdx+0xa8],xmm0
   1cef7:	movsd  xmm3,QWORD PTR [rdi+0x2b8]
   1ceff:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1cf04:	movsd  QWORD PTR [rsp+0x58],xmm0
   1cf0a:	movsd  xmm0,QWORD PTR [rdi+0x2b0]
   1cf12:	movsd  QWORD PTR [rsp+0x50],xmm2
   1cf18:	movapd xmm5,xmm3
   1cf1c:	mulsd  xmm5,xmm4
   1cf20:	movapd xmm2,xmm0
   1cf24:	mulsd  xmm2,xmm1
   1cf28:	mulsd  xmm0,xmm4
   1cf2c:	mulsd  xmm1,xmm3
   1cf30:	subsd  xmm2,xmm5
   1cf34:	addsd  xmm2,QWORD PTR [rsp+0x60]
   1cf3a:	addsd  xmm0,xmm1
   1cf3e:	movsd  QWORD PTR [rsp+0x60],xmm2
   1cf44:	addsd  xmm0,QWORD PTR [rsp+0x68]
   1cf4a:	movsd  QWORD PTR [rdx+0xb0],xmm2
   1cf52:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1cf57:	movsd  QWORD PTR [rdx+0xb8],xmm0
   1cf5f:	movsd  xmm3,QWORD PTR [rdi+0x2c8]
   1cf67:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1cf6c:	movsd  QWORD PTR [rsp+0x68],xmm0
   1cf72:	movsd  xmm0,QWORD PTR [rdi+0x2c0]
   1cf7a:	movapd xmm5,xmm3
   1cf7e:	mulsd  xmm5,xmm4
   1cf82:	movapd xmm2,xmm0
   1cf86:	mulsd  xmm2,xmm1
   1cf8a:	mulsd  xmm0,xmm4
   1cf8e:	mulsd  xmm1,xmm3
   1cf92:	subsd  xmm2,xmm5
   1cf96:	addsd  xmm2,QWORD PTR [rsp+0x70]
   1cf9c:	addsd  xmm0,xmm1
   1cfa0:	addsd  xmm0,QWORD PTR [rsp+0x78]
   1cfa6:	movsd  QWORD PTR [rdx+0xc0],xmm2
   1cfae:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1cfb3:	movsd  QWORD PTR [rsp+0x70],xmm2
   1cfb9:	movsd  QWORD PTR [rdx+0xc8],xmm0
   1cfc1:	movsd  xmm3,QWORD PTR [rdi+0x2d8]
   1cfc9:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1cfce:	movsd  QWORD PTR [rsp+0x78],xmm0
   1cfd4:	movsd  xmm0,QWORD PTR [rdi+0x2d0]
   1cfdc:	movapd xmm5,xmm3
   1cfe0:	mulsd  xmm5,xmm4
   1cfe4:	movapd xmm2,xmm0
   1cfe8:	mulsd  xmm2,xmm1
   1cfec:	mulsd  xmm0,xmm4
   1cff0:	movsd  xmm4,QWORD PTR [rsp+0x88]
   1cff9:	mulsd  xmm1,xmm3
   1cffd:	subsd  xmm2,xmm5
   1d001:	addsd  xmm2,QWORD PTR [rsp+0x80]
   1d00a:	addsd  xmm0,xmm1
   1d00e:	movsd  QWORD PTR [rdx+0xd0],xmm2
   1d016:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1d01b:	movsd  QWORD PTR [rsp+0x80],xmm2
   1d024:	addsd  xmm4,xmm0
   1d028:	movsd  xmm0,QWORD PTR [rdi+0x2e0]
   1d030:	movapd xmm2,xmm0
   1d034:	movsd  QWORD PTR [rdx+0xd8],xmm4
   1d03c:	movsd  xmm3,QWORD PTR [rdi+0x2e8]
   1d044:	mulsd  xmm2,xmm1
   1d048:	movsd  QWORD PTR [rsp+0x88],xmm4
   1d051:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1d056:	mulsd  xmm1,xmm3
   1d05a:	movapd xmm5,xmm3
   1d05e:	mulsd  xmm5,xmm4
   1d062:	mulsd  xmm0,xmm4
   1d066:	subsd  xmm2,xmm5
   1d06a:	addsd  xmm2,QWORD PTR [rsp+0x90]
   1d073:	addsd  xmm0,xmm1
   1d077:	addsd  xmm0,QWORD PTR [rsp+0x98]
   1d080:	movsd  QWORD PTR [rsp+0x90],xmm2
   1d089:	movsd  QWORD PTR [rdx+0xe0],xmm2
   1d091:	movsd  QWORD PTR [rsp+0x98],xmm0
   1d09a:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1d09f:	movsd  QWORD PTR [rdx+0xe8],xmm0
   1d0a7:	movsd  xmm0,QWORD PTR [rdi+0x2f0]
   1d0af:	movsd  xmm3,QWORD PTR [rdi+0x2f8]
   1d0b7:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1d0bc:	movapd xmm2,xmm0
   1d0c0:	movapd xmm5,xmm3
   1d0c4:	mulsd  xmm5,xmm4
   1d0c8:	mulsd  xmm2,xmm1
   1d0cc:	mulsd  xmm0,xmm4
   1d0d0:	mulsd  xmm1,xmm3
   1d0d4:	subsd  xmm2,xmm5
   1d0d8:	addsd  xmm2,QWORD PTR [rsp+0xa0]
   1d0e1:	addsd  xmm0,xmm1
   1d0e5:	addsd  xmm0,QWORD PTR [rsp+0xa8]
   1d0ee:	movsd  QWORD PTR [rdx+0xf0],xmm2
   1d0f6:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d0fb:	movsd  QWORD PTR [rsp+0xa0],xmm2
   1d104:	movsd  QWORD PTR [rdx+0xf8],xmm0
   1d10c:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d111:	movsd  xmm3,QWORD PTR [rdi+0x308]
   1d119:	movsd  QWORD PTR [rsp+0xa8],xmm0
   1d122:	movsd  xmm0,QWORD PTR [rdi+0x300]
   1d12a:	movapd xmm5,xmm3
   1d12e:	mulsd  xmm5,xmm4
   1d132:	movapd xmm2,xmm0
   1d136:	mulsd  xmm2,xmm1
   1d13a:	mulsd  xmm0,xmm4
   1d13e:	mulsd  xmm1,xmm3
   1d142:	subsd  xmm2,xmm5
   1d146:	addsd  xmm0,xmm1
   1d14a:	addsd  xmm15,xmm2
   1d14f:	addsd  xmm14,xmm0
   1d154:	movsd  QWORD PTR [rdx],xmm15
   1d159:	movsd  xmm0,QWORD PTR [rdi+0x310]
   1d161:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d166:	movsd  QWORD PTR [rdx+0x8],xmm14
   1d16c:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d171:	movapd xmm2,xmm0
   1d175:	movsd  xmm3,QWORD PTR [rdi+0x318]
   1d17d:	mulsd  xmm2,xmm1
   1d181:	mulsd  xmm0,xmm4
   1d185:	movapd xmm5,xmm3
   1d189:	mulsd  xmm5,xmm4
   1d18d:	mulsd  xmm1,xmm3
   1d191:	subsd  xmm2,xmm5
   1d195:	addsd  xmm0,xmm1
   1d199:	addsd  xmm13,xmm2
   1d19e:	addsd  xmm12,xmm0
   1d1a3:	movsd  QWORD PTR [rdx+0x10],xmm13
   1d1a9:	movsd  xmm0,QWORD PTR [rdi+0x320]
   1d1b1:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d1b6:	movsd  QWORD PTR [rdx+0x18],xmm12
   1d1bc:	movsd  xmm3,QWORD PTR [rdi+0x328]
   1d1c4:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d1c9:	movapd xmm2,xmm0
   1d1cd:	mulsd  xmm2,xmm1
   1d1d1:	movapd xmm5,xmm3
   1d1d5:	mulsd  xmm5,xmm4
   1d1d9:	mulsd  xmm1,xmm3
   1d1dd:	mulsd  xmm0,xmm4
   1d1e1:	subsd  xmm2,xmm5
   1d1e5:	addsd  xmm0,xmm1
   1d1e9:	addsd  xmm11,xmm2
   1d1ee:	addsd  xmm10,xmm0
   1d1f3:	movsd  QWORD PTR [rdx+0x20],xmm11
   1d1f9:	movsd  xmm0,QWORD PTR [rdi+0x330]
   1d201:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d206:	movsd  QWORD PTR [rdx+0x28],xmm10
   1d20c:	movsd  xmm3,QWORD PTR [rdi+0x338]
   1d214:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d219:	movapd xmm2,xmm0
   1d21d:	mulsd  xmm2,xmm1
   1d221:	mulsd  xmm0,xmm4
   1d225:	movapd xmm5,xmm3
   1d229:	mulsd  xmm5,xmm4
   1d22d:	mulsd  xmm1,xmm3
   1d231:	subsd  xmm2,xmm5
   1d235:	addsd  xmm0,xmm1
   1d239:	addsd  xmm9,xmm2
   1d23e:	addsd  xmm8,xmm0
   1d243:	movsd  QWORD PTR [rdx+0x30],xmm9
   1d249:	movsd  xmm0,QWORD PTR [rdi+0x340]
   1d251:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d256:	movsd  QWORD PTR [rdx+0x38],xmm8
   1d25c:	movsd  xmm3,QWORD PTR [rdi+0x348]
   1d264:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d269:	movapd xmm2,xmm0
   1d26d:	mulsd  xmm2,xmm1
   1d271:	movapd xmm5,xmm3
   1d275:	mulsd  xmm5,xmm4
   1d279:	mulsd  xmm1,xmm3
   1d27d:	mulsd  xmm0,xmm4
   1d281:	subsd  xmm2,xmm5
   1d285:	addsd  xmm0,xmm1
   1d289:	addsd  xmm7,xmm2
   1d28d:	addsd  xmm6,xmm0
   1d291:	movsd  QWORD PTR [rdx+0x40],xmm7
   1d296:	movsd  xmm0,QWORD PTR [rdi+0x350]
   1d29e:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d2a3:	movsd  QWORD PTR [rdx+0x48],xmm6
   1d2a8:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d2ad:	movapd xmm2,xmm0
   1d2b1:	movsd  xmm3,QWORD PTR [rdi+0x358]
   1d2b9:	mulsd  xmm2,xmm1
   1d2bd:	mulsd  xmm0,xmm4
   1d2c1:	movapd xmm5,xmm3
   1d2c5:	mulsd  xmm5,xmm4
   1d2c9:	mulsd  xmm1,xmm3
   1d2cd:	movsd  xmm3,QWORD PTR [rsp+0x8]
   1d2d3:	subsd  xmm2,xmm5
   1d2d7:	addsd  xmm2,QWORD PTR [rsp]
   1d2dc:	addsd  xmm0,xmm1
   1d2e0:	movsd  QWORD PTR [rdx+0x50],xmm2
   1d2e5:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d2ea:	addsd  xmm3,xmm0
   1d2ee:	movsd  xmm0,QWORD PTR [rdi+0x360]
   1d2f6:	movsd  QWORD PTR [rsp],xmm2
   1d2fb:	movapd xmm5,xmm0
   1d2ff:	mulsd  xmm5,xmm1
   1d303:	movsd  QWORD PTR [rdx+0x58],xmm3
   1d308:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d30d:	movsd  QWORD PTR [rsp+0x8],xmm3
   1d313:	movsd  xmm3,QWORD PTR [rdi+0x368]
   1d31b:	mulsd  xmm0,xmm4
   1d31f:	mulsd  xmm1,xmm3
   1d323:	movapd xmm2,xmm5
   1d327:	movapd xmm5,xmm3
   1d32b:	mulsd  xmm5,xmm4
   1d32f:	addsd  xmm0,xmm1
   1d333:	addsd  xmm0,QWORD PTR [rsp+0x18]
   1d339:	subsd  xmm2,xmm5
   1d33d:	addsd  xmm2,QWORD PTR [rsp+0x10]
   1d343:	movsd  QWORD PTR [rsp+0x18],xmm0
   1d349:	movsd  QWORD PTR [rdx+0x68],xmm0
   1d34e:	movsd  xmm3,QWORD PTR [rdi+0x378]
   1d356:	movsd  QWORD PTR [rdx+0x60],xmm2
   1d35b:	movsd  xmm0,QWORD PTR [rdi+0x370]
   1d363:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d368:	movsd  QWORD PTR [rsp+0x10],xmm2
   1d36e:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d373:	movapd xmm5,xmm0
   1d377:	mulsd  xmm5,xmm1
   1d37b:	mulsd  xmm0,xmm4
   1d37f:	mulsd  xmm1,xmm3
   1d383:	movapd xmm2,xmm5
   1d387:	movapd xmm5,xmm3
   1d38b:	mulsd  xmm5,xmm4
   1d38f:	movsd  xmm4,QWORD PTR [rsp+0x28]
   1d395:	addsd  xmm0,xmm1
   1d399:	addsd  xmm4,xmm0
   1d39d:	subsd  xmm2,xmm5
   1d3a1:	addsd  xmm2,QWORD PTR [rsp+0x20]
   1d3a7:	movsd  QWORD PTR [rdx+0x78],xmm4
   1d3ac:	movsd  xmm3,QWORD PTR [rdi+0x388]
   1d3b4:	movsd  QWORD PTR [rdx+0x70],xmm2
   1d3b9:	movsd  xmm0,QWORD PTR [rdi+0x380]
   1d3c1:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d3c6:	movsd  QWORD PTR [rsp+0x28],xmm4
   1d3cc:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d3d1:	movapd xmm5,xmm3
   1d3d5:	movsd  QWORD PTR [rsp+0x20],xmm2
   1d3db:	movapd xmm2,xmm0
   1d3df:	mulsd  xmm2,xmm1
   1d3e3:	mulsd  xmm0,xmm4
   1d3e7:	mulsd  xmm1,xmm3
   1d3eb:	mulsd  xmm5,xmm4
   1d3ef:	movsd  xmm4,QWORD PTR [rsp+0x38]
   1d3f5:	addsd  xmm0,xmm1
   1d3f9:	subsd  xmm2,xmm5
   1d3fd:	addsd  xmm2,QWORD PTR [rsp+0x30]
   1d403:	addsd  xmm4,xmm0
   1d407:	movsd  QWORD PTR [rdx+0x80],xmm2
   1d40f:	movsd  xmm0,QWORD PTR [rdi+0x390]
   1d417:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d41c:	movsd  QWORD PTR [rsp+0x30],xmm2
   1d422:	movsd  QWORD PTR [rdx+0x88],xmm4
   1d42a:	movsd  xmm3,QWORD PTR [rdi+0x398]
   1d432:	movapd xmm2,xmm0
   1d436:	movsd  QWORD PTR [rsp+0x38],xmm4
   1d43c:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d441:	mulsd  xmm2,xmm1
   1d445:	mulsd  xmm1,xmm3
   1d449:	movapd xmm5,xmm3
   1d44d:	mulsd  xmm5,xmm4
   1d451:	mulsd  xmm0,xmm4
   1d455:	subsd  xmm2,xmm5
   1d459:	addsd  xmm2,QWORD PTR [rsp+0x40]
   1d45f:	addsd  xmm0,xmm1
   1d463:	addsd  xmm0,QWORD PTR [rsp+0x48]
   1d469:	movsd  QWORD PTR [rdx+0x90],xmm2
   1d471:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d476:	movsd  QWORD PTR [rdx+0x98],xmm0
   1d47e:	movsd  xmm3,QWORD PTR [rdi+0x3a8]
   1d486:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d48b:	movsd  QWORD PTR [rsp+0x48],xmm0
   1d491:	movsd  xmm0,QWORD PTR [rdi+0x3a0]
   1d499:	movsd  QWORD PTR [rsp+0x40],xmm2
   1d49f:	movapd xmm5,xmm3
   1d4a3:	mulsd  xmm5,xmm4
   1d4a7:	movapd xmm2,xmm0
   1d4ab:	mulsd  xmm2,xmm1
   1d4af:	mulsd  xmm0,xmm4
   1d4b3:	mulsd  xmm1,xmm3
   1d4b7:	subsd  xmm2,xmm5
   1d4bb:	addsd  xmm2,QWORD PTR [rsp+0x50]
   1d4c1:	addsd  xmm0,xmm1
   1d4c5:	movsd  QWORD PTR [rsp+0x50],xmm2
   1d4cb:	addsd  xmm0,QWORD PTR [rsp+0x58]
   1d4d1:	movsd  QWORD PTR [rdx+0xa0],xmm2
   1d4d9:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d4de:	movsd  QWORD PTR [rdx+0xa8],xmm0
   1d4e6:	movsd  xmm3,QWORD PTR [rdi+0x3b8]
   1d4ee:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d4f3:	movsd  QWORD PTR [rsp+0x58],xmm0
   1d4f9:	movsd  xmm0,QWORD PTR [rdi+0x3b0]
   1d501:	movapd xmm5,xmm0
   1d505:	mulsd  xmm5,xmm1
   1d509:	mulsd  xmm0,xmm4
   1d50d:	mulsd  xmm1,xmm3
   1d511:	movapd xmm2,xmm5
   1d515:	movapd xmm5,xmm3
   1d519:	mulsd  xmm5,xmm4
   1d51d:	movsd  xmm4,QWORD PTR [rsp+0x68]
   1d523:	addsd  xmm0,xmm1
   1d527:	addsd  xmm4,xmm0
   1d52b:	subsd  xmm2,xmm5
   1d52f:	addsd  xmm2,QWORD PTR [rsp+0x60]
   1d535:	movsd  QWORD PTR [rdx+0xb8],xmm4
   1d53d:	movsd  xmm3,QWORD PTR [rdi+0x3c8]
   1d545:	movsd  QWORD PTR [rdx+0xb0],xmm2
   1d54d:	movsd  xmm0,QWORD PTR [rdi+0x3c0]
   1d555:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d55a:	movsd  QWORD PTR [rsp+0x68],xmm4
   1d560:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d565:	movapd xmm5,xmm0
   1d569:	movsd  QWORD PTR [rsp+0x60],xmm2
   1d56f:	mulsd  xmm5,xmm1
   1d573:	mulsd  xmm0,xmm4
   1d577:	mulsd  xmm1,xmm3
   1d57b:	movapd xmm2,xmm5
   1d57f:	movapd xmm5,xmm3
   1d583:	mulsd  xmm5,xmm4
   1d587:	addsd  xmm0,xmm1
   1d58b:	addsd  xmm0,QWORD PTR [rsp+0x78]
   1d591:	movsd  QWORD PTR [rdx+0xc8],xmm0
   1d599:	movsd  xmm3,QWORD PTR [rdi+0x3d8]
   1d5a1:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d5a6:	subsd  xmm2,xmm5
   1d5aa:	addsd  xmm2,QWORD PTR [rsp+0x70]
   1d5b0:	movsd  QWORD PTR [rsp+0x78],xmm0
   1d5b6:	movsd  QWORD PTR [rdx+0xc0],xmm2
   1d5be:	movsd  xmm0,QWORD PTR [rdi+0x3d0]
   1d5c6:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d5cb:	movsd  QWORD PTR [rsp+0x70],xmm2
   1d5d1:	movapd xmm5,xmm0
   1d5d5:	mulsd  xmm5,xmm1
   1d5d9:	mulsd  xmm0,xmm4
   1d5dd:	mulsd  xmm1,xmm3
   1d5e1:	movapd xmm2,xmm5
   1d5e5:	movapd xmm5,xmm3
   1d5e9:	mulsd  xmm5,xmm4
   1d5ed:	addsd  xmm0,xmm1
   1d5f1:	addsd  xmm0,QWORD PTR [rsp+0x88]
   1d5fa:	subsd  xmm2,xmm5
   1d5fe:	addsd  xmm2,QWORD PTR [rsp+0x80]
   1d607:	movsd  QWORD PTR [rsp+0x80],xmm2
   1d610:	movsd  QWORD PTR [rdx+0xd0],xmm2
   1d618:	movsd  QWORD PTR [rsp+0x88],xmm0
   1d621:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d626:	movsd  QWORD PTR [rdx+0xd8],xmm0
   1d62e:	movsd  xmm0,QWORD PTR [rdi+0x3e0]
   1d636:	movsd  xmm3,QWORD PTR [rdi+0x3e8]
   1d63e:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d643:	movapd xmm5,xmm0
   1d647:	mulsd  xmm5,xmm1
   1d64b:	mulsd  xmm0,xmm4
   1d64f:	mulsd  xmm1,xmm3
   1d653:	movapd xmm2,xmm5
   1d657:	movapd xmm5,xmm3
   1d65b:	mulsd  xmm5,xmm4
   1d65f:	addsd  xmm0,xmm1
   1d663:	addsd  xmm0,QWORD PTR [rsp+0x98]
   1d66c:	movsd  QWORD PTR [rdx+0xe8],xmm0
   1d674:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d679:	movsd  xmm3,QWORD PTR [rdi+0x3f8]
   1d681:	movsd  QWORD PTR [rsp+0x98],xmm0
   1d68a:	subsd  xmm2,xmm5
   1d68e:	addsd  xmm2,QWORD PTR [rsp+0x90]
   1d697:	movapd xmm5,xmm3
   1d69b:	mulsd  xmm5,xmm4
   1d69f:	movsd  QWORD PTR [rdx+0xe0],xmm2
   1d6a7:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d6ac:	movsd  xmm0,QWORD PTR [rdi+0x3f0]
   1d6b4:	movsd  QWORD PTR [rsp+0x90],xmm2
   1d6bd:	movapd xmm2,xmm0
   1d6c1:	mulsd  xmm2,xmm1
   1d6c5:	mulsd  xmm0,xmm4
   1d6c9:	movsd  xmm4,QWORD PTR [rsp+0xa8]
   1d6d2:	mulsd  xmm1,xmm3
   1d6d6:	subsd  xmm2,xmm5
   1d6da:	addsd  xmm2,QWORD PTR [rsp+0xa0]
   1d6e3:	addsd  xmm0,xmm1
   1d6e7:	movsd  QWORD PTR [rdx+0xf0],xmm2
   1d6ef:	movsd  xmm1,QWORD PTR [rsi+0x40]
   1d6f4:	movsd  QWORD PTR [rsp+0xa0],xmm2
   1d6fd:	addsd  xmm4,xmm0
   1d701:	movsd  xmm0,QWORD PTR [rdi+0x400]
   1d709:	movapd xmm2,xmm0
   1d70d:	movsd  QWORD PTR [rdx+0xf8],xmm4
   1d715:	movsd  xmm3,QWORD PTR [rdi+0x408]
   1d71d:	mulsd  xmm2,xmm1
   1d721:	movsd  QWORD PTR [rsp+0xa8],xmm4
   1d72a:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1d72f:	mulsd  xmm1,xmm3
   1d733:	movapd xmm5,xmm3
   1d737:	mulsd  xmm5,xmm4
   1d73b:	mulsd  xmm0,xmm4
   1d73f:	subsd  xmm2,xmm5
   1d743:	addsd  xmm0,xmm1
   1d747:	addsd  xmm15,xmm2
   1d74c:	addsd  xmm14,xmm0
   1d751:	movsd  QWORD PTR [rdx],xmm15
   1d756:	movsd  xmm0,QWORD PTR [rdi+0x410]
   1d75e:	movsd  xmm1,QWORD PTR [rsi+0x40]
   1d763:	movsd  QWORD PTR [rdx+0x8],xmm14
   1d769:	movsd  xmm3,QWORD PTR [rdi+0x418]
   1d771:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1d776:	movapd xmm2,xmm0
   1d77a:	mulsd  xmm2,xmm1
   1d77e:	movapd xmm5,xmm3
   1d782:	mulsd  xmm5,xmm4
   1d786:	mulsd  xmm1,xmm3
   1d78a:	mulsd  xmm0,xmm4
   1d78e:	subsd  xmm2,xmm5
   1d792:	addsd  xmm0,xmm1
   1d796:	addsd  xmm2,xmm13
   1d79b:	movapd xmm13,xmm0
   1d7a0:	movsd  QWORD PTR [rsp+0xb0],xmm2
   1d7a9:	addsd  xmm13,xmm12
   1d7ae:	movsd  QWORD PTR [rdx+0x10],xmm2
   1d7b3:	movsd  xmm0,QWORD PTR [rdi+0x420]
   1d7bb:	movsd  xmm1,QWORD PTR [rsi+0x40]
   1d7c0:	movsd  QWORD PTR [rdx+0x18],xmm13
   1d7c6:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1d7cb:	movsd  xmm3,QWORD PTR [rdi+0x428]
   1d7d3:	movsd  QWORD PTR [rsp+0xb8],xmm13
   1d7dd:	movapd xmm13,xmm0
   1d7e2:	mulsd  xmm13,xmm1
   1d7e7:	mulsd  xmm0,xmm4
   1d7eb:	mulsd  xmm1,xmm3
   1d7ef:	movapd xmm2,xmm13
   1d7f4:	movapd xmm13,xmm3
   1d7f9:	mulsd  xmm13,xmm4
   1d7fe:	addsd  xmm0,xmm1
   1d802:	subsd  xmm2,xmm13
   1d807:	movapd xmm12,xmm2
   1d80c:	addsd  xmm12,xmm11
   1d811:	movapd xmm11,xmm0
   1d816:	addsd  xmm11,xmm10
   1d81b:	movsd  QWORD PTR [rdx+0x20],xmm12
   1d821:	movsd  xmm0,QWORD PTR [rdi+0x430]
   1d829:	movsd  xmm1,QWORD PTR [rsi+0x40]
   1d82e:	movsd  QWORD PTR [rdx+0x28],xmm11
   1d834:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1d839:	movapd xmm13,xmm11
   1d83e:	movsd  xmm3,QWORD PTR [rdi+0x438]
   1d846:	movapd xmm10,xmm0
   1d84b:	mulsd  xmm10,xmm1
   1d850:	mulsd  xmm0,xmm4
   1d854:	mulsd  xmm1,xmm3
   1d858:	movapd xmm2,xmm10
   1d85d:	movapd xmm10,xmm3
   1d862:	mulsd  xmm10,xmm4
   1d867:	addsd  xmm0,xmm1
   1d86b:	movapd xmm11,xmm0
   1d870:	subsd  xmm2,xmm10
   1d875:	addsd  xmm11,xmm8
   1d87a:	movapd xmm10,xmm2
   1d87f:	movsd  QWORD PTR [rdx+0x38],xmm11
   1d885:	movsd  xmm3,QWORD PTR [rdi+0x448]
   1d88d:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1d892:	movsd  QWORD PTR [rsp+0xc8],xmm11
   1d89c:	addsd  xmm10,xmm9
   1d8a1:	movsd  QWORD PTR [rdx+0x30],xmm10
   1d8a7:	movsd  xmm1,QWORD PTR [rsi+0x40]
   1d8ac:	movsd  xmm0,QWORD PTR [rdi+0x440]
   1d8b4:	movsd  QWORD PTR [rsp+0xc0],xmm10
   1d8be:	movapd xmm8,xmm0
   1d8c3:	mulsd  xmm8,xmm1
   1d8c8:	mulsd  xmm0,xmm4
   1d8cc:	mulsd  xmm1,xmm3
   1d8d0:	movapd xmm2,xmm8
   1d8d5:	movapd xmm8,xmm3
   1d8da:	mulsd  xmm8,xmm4
   1d8df:	addsd  xmm0,xmm1
   1d8e3:	addsd  xmm6,xmm0
   1d8e7:	subsd  xmm2,xmm8
   1d8ec:	movsd  QWORD PTR [rdx+0x48],xmm6
   1d8f1:	movsd  xmm3,QWORD PTR [rdi+0x458]
   1d8f9:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1d8fe:	movapd xmm8,xmm2
   1d903:	addsd  xmm8,xmm7
   1d908:	movapd xmm9,xmm3
   1d90d:	mulsd  xmm9,xmm4
   1d912:	movsd  QWORD PTR [rdx+0x40],xmm8
   1d918:	movsd  xmm1,QWORD PTR [rsi+0x40]
   1d91d:	movapd xmm5,xmm8
   1d922:	movsd  xmm0,QWORD PTR [rdi+0x450]
   1d92a:	movapd xmm8,xmm0
   1d92f:	mulsd  xmm8,xmm1
   1d934:	mulsd  xmm0,xmm4
   1d938:	mulsd  xmm1,xmm3
   1d93c:	movapd xmm2,xmm8
   1d941:	movsd  xmm8,QWORD PTR [rsp]
   1d947:	subsd  xmm2,xmm9
   1d94c:	movsd  xmm9,QWORD PTR [rsp+0x8]
   1d953:	addsd  xmm0,xmm1
   1d957:	addsd  xmm8,xmm2
   1d95c:	addsd  xmm9,xmm0
   1d961:	movsd  QWORD PTR [rsp],xmm8
   1d967:	movsd  QWORD PTR [rsp+0x8],xmm9
   1d96e:	movsd  QWORD PTR [rdx+0x50],xmm8
   1d974:	movsd  QWORD PTR [rdx+0x58],xmm9
   1d97a:	movsd  xmm0,QWORD PTR [rdi+0x460]
   1d982:	movsd  xmm3,QWORD PTR [rdi+0x468]
   1d98a:	movsd  xmm1,QWORD PTR [rsi+0x40]
   1d98f:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1d994:	movapd xmm8,xmm0
   1d999:	movapd xmm9,xmm3
   1d99e:	movsd  xmm10,QWORD PTR [rsp+0x18]
   1d9a5:	mulsd  xmm8,xmm1
   1d9aa:	mulsd  xmm9,xmm4
   1d9af:	mulsd  xmm1,xmm3
   1d9b3:	mulsd  xmm0,xmm4
   1d9b7:	movapd xmm2,xmm8
   1d9bc:	movsd  xmm8,QWORD PTR [rsp+0x10]
   1d9c3:	subsd  xmm2,xmm9
   1d9c8:	addsd  xmm0,xmm1
   1d9cc:	addsd  xmm8,xmm2
   1d9d1:	addsd  xmm10,xmm0
   1d9d6:	movsd  QWORD PTR [rdx+0x60],xmm8
   1d9dc:	movsd  xmm0,QWORD PTR [rdi+0x470]
   1d9e4:	movsd  xmm1,QWORD PTR [rsi+0x40]
   1d9e9:	movsd  QWORD PTR [rsp+0x10],xmm8
   1d9f0:	movsd  QWORD PTR [rdx+0x68],xmm10
   1d9f6:	movsd  xmm3,QWORD PTR [rdi+0x478]
   1d9fe:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1da03:	movapd xmm11,xmm0
   1da08:	mulsd  xmm11,xmm1
   1da0d:	movsd  QWORD PTR [rsp+0x18],xmm10
   1da14:	mulsd  xmm1,xmm3
   1da18:	movapd xmm9,xmm3
   1da1d:	mulsd  xmm9,xmm4
   1da22:	mulsd  xmm0,xmm4
   1da26:	movapd xmm2,xmm11
   1da2b:	movsd  xmm11,QWORD PTR [rsp+0x20]
   1da32:	subsd  xmm2,xmm9
   1da37:	movsd  xmm9,QWORD PTR [rsp+0x28]
   1da3e:	addsd  xmm0,xmm1
   1da42:	addsd  xmm11,xmm2
   1da47:	addsd  xmm9,xmm0
   1da4c:	movsd  QWORD PTR [rdx+0x70],xmm11
   1da52:	movsd  xmm0,QWORD PTR [rdi+0x480]
   1da5a:	movsd  xmm2,QWORD PTR [rsi+0x40]
   1da5f:	movsd  QWORD PTR [rdx+0x78],xmm9
   1da65:	movsd  xmm3,QWORD PTR [rdi+0x488]
   1da6d:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1da72:	movsd  QWORD PTR [rsp+0x28],xmm9
   1da79:	movapd xmm1,xmm0
   1da7d:	mulsd  xmm1,xmm2
   1da81:	movapd xmm9,xmm3
   1da86:	movsd  QWORD PTR [rsp+0x20],xmm11
   1da8d:	mulsd  xmm9,xmm4
   1da92:	mulsd  xmm2,xmm3
   1da96:	mulsd  xmm0,xmm4
   1da9a:	subsd  xmm1,xmm9
   1da9f:	addsd  xmm1,QWORD PTR [rsp+0x30]
   1daa5:	addsd  xmm0,xmm2
   1daa9:	addsd  xmm0,QWORD PTR [rsp+0x38]
   1daaf:	movsd  QWORD PTR [rdx+0x80],xmm1
   1dab7:	movsd  xmm2,QWORD PTR [rsi+0x40]
   1dabc:	movsd  QWORD PTR [rdx+0x88],xmm0
   1dac4:	movsd  xmm0,QWORD PTR [rdi+0x490]
   1dacc:	movsd  xmm3,QWORD PTR [rdi+0x498]
   1dad4:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1dad9:	movapd xmm1,xmm0
   1dadd:	movapd xmm9,xmm3
   1dae2:	mulsd  xmm9,xmm4
   1dae7:	mulsd  xmm1,xmm2
   1daeb:	mulsd  xmm0,xmm4
   1daef:	mulsd  xmm2,xmm3
   1daf3:	subsd  xmm1,xmm9
   1daf8:	addsd  xmm1,QWORD PTR [rsp+0x40]
   1dafe:	addsd  xmm0,xmm2
   1db02:	addsd  xmm0,QWORD PTR [rsp+0x48]
   1db08:	movsd  QWORD PTR [rdx+0x90],xmm1
   1db10:	movsd  xmm2,QWORD PTR [rsi+0x40]
   1db15:	movsd  QWORD PTR [rdx+0x98],xmm0
   1db1d:	movsd  xmm0,QWORD PTR [rdi+0x4a0]
   1db25:	movsd  xmm3,QWORD PTR [rdi+0x4a8]
   1db2d:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1db32:	movapd xmm1,xmm0
   1db36:	movapd xmm9,xmm3
   1db3b:	mulsd  xmm9,xmm4
   1db40:	mulsd  xmm1,xmm2
   1db44:	mulsd  xmm0,xmm4
   1db48:	mulsd  xmm2,xmm3
   1db4c:	subsd  xmm1,xmm9
   1db51:	addsd  xmm1,QWORD PTR [rsp+0x50]
   1db57:	addsd  xmm0,xmm2
   1db5b:	addsd  xmm0,QWORD PTR [rsp+0x58]
   1db61:	movsd  QWORD PTR [rdx+0xa0],xmm1
   1db69:	movsd  xmm2,QWORD PTR [rsi+0x40]
   1db6e:	movsd  QWORD PTR [rdx+0xa8],xmm0
   1db76:	movsd  xmm0,QWORD PTR [rdi+0x4b0]
   1db7e:	movsd  xmm3,QWORD PTR [rdi+0x4b8]
   1db86:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1db8b:	movapd xmm1,xmm0
   1db8f:	movapd xmm9,xmm3
   1db94:	mulsd  xmm9,xmm4
   1db99:	mulsd  xmm1,xmm2
   1db9d:	mulsd  xmm0,xmm4
   1dba1:	mulsd  xmm2,xmm3
   1dba5:	subsd  xmm1,xmm9
   1dbaa:	addsd  xmm1,QWORD PTR [rsp+0x60]
   1dbb0:	addsd  xmm0,xmm2
   1dbb4:	addsd  xmm0,QWORD PTR [rsp+0x68]
   1dbba:	movsd  QWORD PTR [rdx+0xb0],xmm1
   1dbc2:	movsd  xmm2,QWORD PTR [rsi+0x40]
   1dbc7:	movsd  QWORD PTR [rdx+0xb8],xmm0
   1dbcf:	movsd  xmm0,QWORD PTR [rdi+0x4c0]
   1dbd7:	movsd  xmm3,QWORD PTR [rdi+0x4c8]
   1dbdf:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1dbe4:	movapd xmm1,xmm0
   1dbe8:	movapd xmm9,xmm3
   1dbed:	mulsd  xmm9,xmm4
   1dbf2:	mulsd  xmm1,xmm2
   1dbf6:	mulsd  xmm0,xmm4
   1dbfa:	mulsd  xmm2,xmm3
   1dbfe:	subsd  xmm1,xmm9
   1dc03:	addsd  xmm1,QWORD PTR [rsp+0x70]
   1dc09:	addsd  xmm0,xmm2
   1dc0d:	addsd  xmm0,QWORD PTR [rsp+0x78]
   1dc13:	movsd  QWORD PTR [rdx+0xc0],xmm1
   1dc1b:	movsd  xmm2,QWORD PTR [rsi+0x40]
   1dc20:	movsd  QWORD PTR [rdx+0xc8],xmm0
   1dc28:	movsd  xmm0,QWORD PTR [rdi+0x4d0]
   1dc30:	movsd  xmm3,QWORD PTR [rdi+0x4d8]
   1dc38:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1dc3d:	movapd xmm1,xmm0
   1dc41:	movapd xmm9,xmm3
   1dc46:	mulsd  xmm9,xmm4
   1dc4b:	mulsd  xmm1,xmm2
   1dc4f:	mulsd  xmm0,xmm4
   1dc53:	mulsd  xmm2,xmm3
   1dc57:	subsd  xmm1,xmm9
   1dc5c:	addsd  xmm1,QWORD PTR [rsp+0x80]
   1dc65:	addsd  xmm0,xmm2
   1dc69:	addsd  xmm0,QWORD PTR [rsp+0x88]
   1dc72:	movsd  QWORD PTR [rdx+0xd0],xmm1
   1dc7a:	movsd  xmm2,QWORD PTR [rsi+0x40]
   1dc7f:	movsd  QWORD PTR [rdx+0xd8],xmm0
   1dc87:	movsd  xmm0,QWORD PTR [rdi+0x4e0]
   1dc8f:	movsd  xmm3,QWORD PTR [rdi+0x4e8]
   1dc97:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1dc9c:	movapd xmm1,xmm0
   1dca0:	movapd xmm9,xmm3
   1dca5:	mulsd  xmm9,xmm4
   1dcaa:	mulsd  xmm1,xmm2
   1dcae:	mulsd  xmm0,xmm4
   1dcb2:	mulsd  xmm2,xmm3
   1dcb6:	subsd  xmm1,xmm9
   1dcbb:	addsd  xmm1,QWORD PTR [rsp+0x90]
   1dcc4:	addsd  xmm0,xmm2
   1dcc8:	addsd  xmm0,QWORD PTR [rsp+0x98]
   1dcd1:	movsd  QWORD PTR [rdx+0xe0],xmm1
   1dcd9:	movsd  xmm2,QWORD PTR [rsi+0x40]
   1dcde:	movsd  QWORD PTR [rdx+0xe8],xmm0
   1dce6:	movsd  xmm0,QWORD PTR [rdi+0x4f0]
   1dcee:	movsd  xmm3,QWORD PTR [rdi+0x4f8]
   1dcf6:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1dcfb:	movapd xmm1,xmm0
   1dcff:	movapd xmm9,xmm3
   1dd04:	mulsd  xmm9,xmm4
   1dd09:	mulsd  xmm1,xmm2
   1dd0d:	mulsd  xmm0,xmm4
   1dd11:	mulsd  xmm2,xmm3
   1dd15:	subsd  xmm1,xmm9
   1dd1a:	addsd  xmm1,QWORD PTR [rsp+0xa0]
   1dd23:	addsd  xmm0,xmm2
   1dd27:	addsd  xmm0,QWORD PTR [rsp+0xa8]
   1dd30:	movsd  QWORD PTR [rdx+0xf0],xmm1
   1dd38:	movsd  xmm11,QWORD PTR [rdi+0x500]
   1dd41:	movsd  QWORD PTR [rdx+0xf8],xmm0
   1dd49:	movsd  xmm2,QWORD PTR [rdi+0x508]
   1dd51:	movsd  xmm3,QWORD PTR [rsi+0x58]
   1dd56:	movapd xmm1,xmm11
   1dd5b:	movsd  xmm0,QWORD PTR [rsi+0x50]
   1dd60:	mulsd  xmm11,xmm3
   1dd65:	movapd xmm4,xmm2
   1dd69:	mulsd  xmm4,xmm3
   1dd6d:	mulsd  xmm1,xmm0
   1dd71:	mulsd  xmm0,xmm2
   1dd75:	subsd  xmm1,xmm4
   1dd79:	addsd  xmm11,xmm0
   1dd7e:	addsd  xmm1,xmm15
   1dd83:	addsd  xmm11,xmm14
   1dd88:	movsd  QWORD PTR [rdx],xmm1
   1dd8c:	movsd  xmm0,QWORD PTR [rsi+0x50]
   1dd91:	movsd  xmm10,QWORD PTR [rdi+0x510]
   1dd9a:	movsd  QWORD PTR [rdx+0x8],xmm11
   1dda0:	movsd  xmm2,QWORD PTR [rdi+0x518]
   1dda8:	movsd  xmm3,QWORD PTR [rsi+0x58]
   1ddad:	movapd xmm15,xmm10
   1ddb2:	movsd  QWORD PTR [rsp+0x30],xmm1
   1ddb8:	mulsd  xmm15,xmm0
   1ddbd:	movapd xmm4,xmm2
   1ddc1:	mulsd  xmm4,xmm3
   1ddc5:	mulsd  xmm0,xmm2
   1ddc9:	mulsd  xmm10,xmm3
   1ddce:	subsd  xmm15,xmm4
   1ddd3:	addsd  xmm15,QWORD PTR [rsp+0xb0]
   1dddd:	addsd  xmm10,xmm0
   1dde2:	movsd  QWORD PTR [rdx+0x10],xmm15
   1dde8:	movsd  xmm0,QWORD PTR [rsi+0x50]
   1dded:	addsd  xmm10,QWORD PTR [rsp+0xb8]
   1ddf7:	movsd  xmm9,QWORD PTR [rdi+0x520]
   1de00:	movsd  QWORD PTR [rdx+0x18],xmm10
   1de06:	movsd  xmm3,QWORD PTR [rsi+0x58]
   1de0b:	movapd xmm14,xmm9
   1de10:	movsd  xmm2,QWORD PTR [rdi+0x528]
   1de18:	mulsd  xmm14,xmm0
   1de1d:	mulsd  xmm9,xmm3
   1de22:	movapd xmm4,xmm2
   1de26:	mulsd  xmm4,xmm3
   1de2a:	mulsd  xmm0,xmm2
   1de2e:	subsd  xmm14,xmm4
   1de33:	addsd  xmm9,xmm0
   1de38:	addsd  xmm14,xmm12
   1de3d:	addsd  xmm9,xmm13
   1de42:	movsd  QWORD PTR [rdx+0x20],xmm14
   1de48:	movsd  xmm0,QWORD PTR [rsi+0x50]
   1de4d:	movsd  xmm8,QWORD PTR [rdi+0x530]
   1de56:	movsd  QWORD PTR [rdx+0x28],xmm9
   1de5c:	movsd  xmm2,QWORD PTR [rdi+0x538]
   1de64:	movsd  xmm3,QWORD PTR [rsi+0x58]
   1de69:	movapd xmm13,xmm8
   1de6e:	mulsd  xmm13,xmm0
   1de73:	movapd xmm4,xmm2
   1de77:	mulsd  xmm4,xmm3
   1de7b:	mulsd  xmm0,xmm2
   1de7f:	mulsd  xmm8,xmm3
   1de84:	subsd  xmm13,xmm4
   1de89:	addsd  xmm13,QWORD PTR [rsp+0xc0]
   1de93:	addsd  xmm8,xmm0
   1de98:	addsd  xmm8,QWORD PTR [rsp+0xc8]
   1dea2:	movsd  QWORD PTR [rdx+0x30],xmm13
   1dea8:	movsd  xmm7,QWORD PTR [rdi+0x540]
   1deb0:	movsd  xmm0,QWORD PTR [rsi+0x50]
   1deb5:	movsd  QWORD PTR [rdx+0x38],xmm8
   1debb:	movsd  xmm3,QWORD PTR [rsi+0x58]
   1dec0:	movapd xmm12,xmm7
   1dec5:	movsd  xmm2,QWORD PTR [rdi+0x548]
   1decd:	mulsd  xmm12,xmm0
   1ded2:	mulsd  xmm7,xmm3
   1ded6:	movapd xmm4,xmm2
   1deda:	mulsd  xmm4,xmm3
   1dede:	mulsd  xmm0,xmm2
   1dee2:	subsd  xmm12,xmm4
   1dee7:	addsd  xmm7,xmm0
   1deeb:	addsd  xmm12,xmm5
   1def0:	addsd  xmm7,xmm6
   1def4:	movsd  QWORD PTR [rdx+0x40],xmm12
   1defa:	movsd  xmm0,QWORD PTR [rdi+0x550]
   1df02:	movsd  xmm2,QWORD PTR [rsi+0x50]
   1df07:	movsd  QWORD PTR [rdx+0x48],xmm7
   1df0c:	movsd  xmm4,QWORD PTR [rdi+0x558]
   1df14:	movsd  xmm5,QWORD PTR [rsi+0x58]
   1df19:	movapd xmm3,xmm0
   1df1d:	mulsd  xmm3,xmm2
   1df21:	movapd xmm1,xmm4
   1df25:	mulsd  xmm1,xmm5
   1df29:	mulsd  xmm2,xmm4
   1df2d:	mulsd  xmm0,xmm5
   1df31:	subsd  xmm3,xmm1
   1df35:	addsd  xmm3,QWORD PTR [rsp]
   1df3a:	addsd  xmm0,xmm2
   1df3e:	addsd  xmm0,QWORD PTR [rsp+0x8]
   1df44:	movsd  QWORD PTR [rsp],xmm3
   1df49:	movsd  QWORD PTR [rdx+0x50],xmm3
   1df4e:	movsd  QWORD PTR [rsp+0x8],xmm0
   1df54:	movsd  QWORD PTR [rdx+0x58],xmm0
   1df59:	movsd  xmm0,QWORD PTR [rdi+0x560]
   1df61:	movsd  xmm2,QWORD PTR [rsi+0x50]
   1df66:	movsd  xmm4,QWORD PTR [rdi+0x568]
   1df6e:	movsd  xmm5,QWORD PTR [rsi+0x58]
   1df73:	movapd xmm1,xmm0
   1df77:	mulsd  xmm1,xmm2
   1df7b:	mulsd  xmm0,xmm5
   1df7f:	mulsd  xmm2,xmm4
   1df83:	movapd xmm3,xmm1
   1df87:	movapd xmm1,xmm4
   1df8b:	mulsd  xmm1,xmm5
   1df8f:	movsd  xmm5,QWORD PTR [rsp+0x18]
   1df95:	addsd  xmm0,xmm2
   1df99:	addsd  xmm5,xmm0
   1df9d:	subsd  xmm3,xmm1
   1dfa1:	movsd  xmm1,QWORD PTR [rsp+0x10]
   1dfa7:	movsd  QWORD PTR [rsp+0x18],xmm5
   1dfad:	addsd  xmm1,xmm3
   1dfb1:	movsd  QWORD PTR [rdx+0x68],xmm5
   1dfb6:	movsd  xmm5,QWORD PTR [rsi+0x58]
   1dfbb:	movsd  xmm4,QWORD PTR [rdi+0x578]
   1dfc3:	movsd  QWORD PTR [rdx+0x60],xmm1
   1dfc8:	movsd  xmm0,QWORD PTR [rdi+0x570]
   1dfd0:	movsd  xmm2,QWORD PTR [rsi+0x50]
   1dfd5:	movsd  QWORD PTR [rsp+0x10],xmm1
   1dfdb:	movapd xmm1,xmm4
   1dfdf:	mulsd  xmm1,xmm5
   1dfe3:	movapd xmm3,xmm0
   1dfe7:	mulsd  xmm3,xmm2
   1dfeb:	mulsd  xmm0,xmm5
   1dfef:	mulsd  xmm2,xmm4
   1dff3:	movsd  xmm4,QWORD PTR [rsp+0x28]
   1dff9:	subsd  xmm3,xmm1
   1dffd:	addsd  xmm3,QWORD PTR [rsp+0x20]
   1e003:	addsd  xmm0,xmm2
   1e007:	movsd  QWORD PTR [rdx+0x70],xmm3
   1e00c:	movsd  QWORD PTR [rsp+0x20],xmm3
   1e012:	addsd  xmm4,xmm0
   1e016:	movsd  QWORD PTR [rdx+0x78],xmm4
   1e01b:	movsd  QWORD PTR [rsp+0x28],xmm4
   1e021:	call   178e0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm5ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1e026:	movsd  xmm2,QWORD PTR [rsi+0x60]
   1e02b:	movsd  xmm5,QWORD PTR [rsi+0x68]
   1e030:	movsd  xmm0,QWORD PTR [rdi+0x600]
   1e038:	movsd  xmm4,QWORD PTR [rdi+0x608]
   1e040:	movsd  xmm6,QWORD PTR [rsp+0x30]
   1e046:	movapd xmm1,xmm0
   1e04a:	mulsd  xmm1,xmm2
   1e04e:	mulsd  xmm0,xmm5
   1e052:	mulsd  xmm2,xmm4
   1e056:	movapd xmm3,xmm1
   1e05a:	movapd xmm1,xmm4
   1e05e:	mulsd  xmm1,xmm5
   1e062:	addsd  xmm0,xmm2
   1e066:	addsd  xmm11,xmm0
   1e06b:	subsd  xmm3,xmm1
   1e06f:	movsd  QWORD PTR [rdx+0x8],xmm11
   1e075:	movsd  xmm4,QWORD PTR [rsi+0x68]
   1e07a:	addsd  xmm6,xmm3
   1e07e:	movsd  xmm3,QWORD PTR [rdi+0x618]
   1e086:	movapd xmm5,xmm3
   1e08a:	movsd  QWORD PTR [rdx],xmm6
   1e08e:	movsd  xmm1,QWORD PTR [rsi+0x60]
   1e093:	mulsd  xmm5,xmm4
   1e097:	movsd  xmm0,QWORD PTR [rdi+0x610]
   1e09f:	movsd  QWORD PTR [rsp+0x30],xmm6
   1e0a5:	movapd xmm2,xmm0
   1e0a9:	mulsd  xmm2,xmm1
   1e0ad:	mulsd  xmm0,xmm4
   1e0b1:	mulsd  xmm1,xmm3
   1e0b5:	subsd  xmm2,xmm5
   1e0b9:	addsd  xmm0,xmm1
   1e0bd:	addsd  xmm15,xmm2
   1e0c2:	addsd  xmm10,xmm0
   1e0c7:	movsd  QWORD PTR [rdx+0x10],xmm15
   1e0cd:	movsd  xmm0,QWORD PTR [rdi+0x620]
   1e0d5:	movsd  xmm1,QWORD PTR [rsi+0x60]
   1e0da:	movsd  QWORD PTR [rdx+0x18],xmm10
   1e0e0:	movsd  xmm4,QWORD PTR [rsi+0x68]
   1e0e5:	movapd xmm2,xmm0
   1e0e9:	movsd  xmm3,QWORD PTR [rdi+0x628]
   1e0f1:	mulsd  xmm2,xmm1
   1e0f5:	mulsd  xmm0,xmm4
   1e0f9:	movapd xmm5,xmm3
   1e0fd:	mulsd  xmm5,xmm4
   1e101:	mulsd  xmm1,xmm3
   1e105:	subsd  xmm2,xmm5
   1e109:	addsd  xmm0,xmm1
   1e10d:	addsd  xmm14,xmm2
   1e112:	addsd  xmm9,xmm0
   1e117:	movsd  QWORD PTR [rdx+0x20],xmm14
   1e11d:	movsd  xmm0,QWORD PTR [rdi+0x630]
   1e125:	movsd  xmm1,QWORD PTR [rsi+0x60]
   1e12a:	movsd  QWORD PTR [rdx+0x28],xmm9
   1e130:	movsd  xmm3,QWORD PTR [rdi+0x638]
   1e138:	movsd  xmm4,QWORD PTR [rsi+0x68]
   1e13d:	movapd xmm2,xmm0
   1e141:	mulsd  xmm2,xmm1
   1e145:	movapd xmm5,xmm3
   1e149:	mulsd  xmm5,xmm4
   1e14d:	mulsd  xmm1,xmm3
   1e151:	mulsd  xmm0,xmm4
   1e155:	subsd  xmm2,xmm5
   1e159:	addsd  xmm0,xmm1
   1e15d:	addsd  xmm13,xmm2
   1e162:	addsd  xmm8,xmm0
   1e167:	movsd  QWORD PTR [rdx+0x30],xmm13
   1e16d:	movsd  xmm0,QWORD PTR [rdi+0x640]
   1e175:	movsd  xmm1,QWORD PTR [rsi+0x60]
   1e17a:	movsd  QWORD PTR [rdx+0x38],xmm8
   1e180:	movsd  xmm4,QWORD PTR [rsi+0x68]
   1e185:	movapd xmm2,xmm0
   1e189:	movsd  xmm3,QWORD PTR [rdi+0x648]
   1e191:	mulsd  xmm2,xmm1
   1e195:	mulsd  xmm0,xmm4
   1e199:	movapd xmm5,xmm3
   1e19d:	mulsd  xmm5,xmm4
   1e1a1:	mulsd  xmm1,xmm3
   1e1a5:	subsd  xmm2,xmm5
   1e1a9:	addsd  xmm0,xmm1
   1e1ad:	addsd  xmm12,xmm2
   1e1b2:	addsd  xmm7,xmm0
   1e1b6:	movsd  QWORD PTR [rdx+0x40],xmm12
   1e1bc:	movsd  QWORD PTR [rdx+0x48],xmm7
   1e1c1:	movsd  xmm0,QWORD PTR [rdi+0x650]
   1e1c9:	movsd  xmm3,QWORD PTR [rdi+0x658]
   1e1d1:	movsd  xmm4,QWORD PTR [rsi+0x68]
   1e1d6:	movsd  xmm1,QWORD PTR [rsi+0x60]
   1e1db:	movapd xmm2,xmm0
   1e1df:	movapd xmm5,xmm3
   1e1e3:	mulsd  xmm5,xmm4
   1e1e7:	mulsd  xmm2,xmm1
   1e1eb:	mulsd  xmm0,xmm4
   1e1ef:	mulsd  xmm1,xmm3
   1e1f3:	subsd  xmm2,xmm5
   1e1f7:	addsd  xmm2,QWORD PTR [rsp]
   1e1fc:	addsd  xmm0,xmm1
   1e200:	addsd  xmm0,QWORD PTR [rsp+0x8]
   1e206:	movsd  QWORD PTR [rdx+0x50],xmm2
   1e20b:	movsd  xmm1,QWORD PTR [rsi+0x60]
   1e210:	movsd  QWORD PTR [rsp],xmm2
   1e215:	movsd  QWORD PTR [rdx+0x58],xmm0
   1e21a:	movsd  xmm3,QWORD PTR [rdi+0x668]
   1e222:	movsd  xmm4,QWORD PTR [rsi+0x68]
   1e227:	movsd  QWORD PTR [rsp+0x8],xmm0
   1e22d:	movsd  xmm0,QWORD PTR [rdi+0x660]
   1e235:	movapd xmm5,xmm3
   1e239:	mulsd  xmm5,xmm4
   1e23d:	movapd xmm2,xmm0
   1e241:	mulsd  xmm2,xmm1
   1e245:	mulsd  xmm0,xmm4
   1e249:	mulsd  xmm1,xmm3
   1e24d:	subsd  xmm2,xmm5
   1e251:	addsd  xmm2,QWORD PTR [rsp+0x10]
   1e257:	addsd  xmm0,xmm1
   1e25b:	movsd  xmm1,QWORD PTR [rsp+0x18]
   1e261:	movsd  QWORD PTR [rdx+0x60],xmm2
   1e266:	movsd  QWORD PTR [rsp+0x10],xmm2
   1e26c:	addsd  xmm1,xmm0
   1e270:	movsd  xmm0,QWORD PTR [rdi+0x670]
   1e278:	movapd xmm5,xmm0
   1e27c:	movsd  QWORD PTR [rdx+0x68],xmm1
   1e281:	movsd  xmm3,QWORD PTR [rdi+0x678]
   1e289:	movsd  xmm4,QWORD PTR [rsi+0x68]
   1e28e:	movsd  QWORD PTR [rsp+0x18],xmm1
   1e294:	movsd  xmm1,QWORD PTR [rsi+0x60]
   1e299:	mulsd  xmm0,xmm4
   1e29d:	mulsd  xmm5,xmm1
   1e2a1:	mulsd  xmm1,xmm3
   1e2a5:	movapd xmm2,xmm5
   1e2a9:	movapd xmm5,xmm3
   1e2ad:	movsd  xmm3,QWORD PTR [rsp+0x28]
   1e2b3:	mulsd  xmm5,xmm4
   1e2b7:	addsd  xmm0,xmm1
   1e2bb:	addsd  xmm3,xmm0
   1e2bf:	subsd  xmm2,xmm5
   1e2c3:	movsd  xmm5,QWORD PTR [rsp+0x20]
   1e2c9:	movsd  QWORD PTR [rdx+0x78],xmm3
   1e2ce:	movsd  QWORD PTR [rsp+0x28],xmm3
   1e2d4:	addsd  xmm5,xmm2
   1e2d8:	movsd  QWORD PTR [rdx+0x70],xmm5
   1e2dd:	movsd  QWORD PTR [rsp+0x20],xmm5
   1e2e3:	call   173b0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm6ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1e2e8:	movsd  xmm1,QWORD PTR [rsi+0x70]
   1e2ed:	movsd  xmm4,QWORD PTR [rsi+0x78]
   1e2f2:	movsd  xmm0,QWORD PTR [rdi+0x700]
   1e2fa:	movsd  xmm3,QWORD PTR [rdi+0x708]
   1e302:	movapd xmm6,xmm0
   1e306:	movapd xmm5,xmm3
   1e30a:	mulsd  xmm6,xmm1
   1e30e:	mulsd  xmm5,xmm4
   1e312:	mulsd  xmm1,xmm3
   1e316:	mulsd  xmm0,xmm4
   1e31a:	movapd xmm2,xmm6
   1e31e:	movsd  xmm6,QWORD PTR [rsp+0x30]
   1e324:	subsd  xmm2,xmm5
   1e328:	addsd  xmm0,xmm1
   1e32c:	addsd  xmm6,xmm2
   1e330:	addsd  xmm11,xmm0
   1e335:	movsd  QWORD PTR [rdx],xmm6
   1e339:	movsd  xmm0,QWORD PTR [rdi+0x710]
   1e341:	movsd  xmm1,QWORD PTR [rsi+0x70]
   1e346:	movsd  QWORD PTR [rsp+0x30],xmm6
   1e34c:	movsd  QWORD PTR [rdx+0x8],xmm11
   1e352:	movapd xmm6,xmm0
   1e356:	movsd  xmm4,QWORD PTR [rsi+0x78]
   1e35b:	movsd  xmm3,QWORD PTR [rdi+0x718]
   1e363:	mulsd  xmm6,xmm1
   1e367:	mulsd  xmm0,xmm4
   1e36b:	movapd xmm5,xmm3
   1e36f:	mulsd  xmm5,xmm4
   1e373:	mulsd  xmm1,xmm3
   1e377:	movapd xmm2,xmm6
   1e37b:	subsd  xmm2,xmm5
   1e37f:	addsd  xmm0,xmm1
   1e383:	addsd  xmm15,xmm2
   1e388:	addsd  xmm10,xmm0
   1e38d:	movsd  QWORD PTR [rdx+0x10],xmm15
   1e393:	movsd  xmm0,QWORD PTR [rdi+0x720]
   1e39b:	movsd  xmm1,QWORD PTR [rsi+0x70]
   1e3a0:	movsd  QWORD PTR [rdx+0x18],xmm10
   1e3a6:	movsd  xmm3,QWORD PTR [rdi+0x728]
   1e3ae:	movsd  xmm4,QWORD PTR [rsi+0x78]
   1e3b3:	movapd xmm6,xmm0
   1e3b7:	mulsd  xmm6,xmm1
   1e3bb:	movapd xmm5,xmm3
   1e3bf:	mulsd  xmm5,xmm4
   1e3c3:	mulsd  xmm1,xmm3
   1e3c7:	mulsd  xmm0,xmm4
   1e3cb:	movapd xmm2,xmm6
   1e3cf:	subsd  xmm2,xmm5
   1e3d3:	addsd  xmm0,xmm1
   1e3d7:	addsd  xmm14,xmm2
   1e3dc:	addsd  xmm9,xmm0
   1e3e1:	movsd  QWORD PTR [rdx+0x20],xmm14
   1e3e7:	movsd  xmm0,QWORD PTR [rdi+0x730]
   1e3ef:	movsd  xmm1,QWORD PTR [rsi+0x70]
   1e3f4:	movsd  QWORD PTR [rdx+0x28],xmm9
   1e3fa:	movapd xmm6,xmm0
   1e3fe:	movsd  xmm4,QWORD PTR [rsi+0x78]
   1e403:	movsd  xmm3,QWORD PTR [rdi+0x738]
   1e40b:	mulsd  xmm6,xmm1
   1e40f:	mulsd  xmm0,xmm4
   1e413:	movapd xmm5,xmm3
   1e417:	mulsd  xmm5,xmm4
   1e41b:	mulsd  xmm1,xmm3
   1e41f:	movapd xmm2,xmm6
   1e423:	subsd  xmm2,xmm5
   1e427:	addsd  xmm0,xmm1
   1e42b:	addsd  xmm13,xmm2
   1e430:	addsd  xmm8,xmm0
   1e435:	movsd  QWORD PTR [rdx+0x30],xmm13
   1e43b:	movsd  xmm0,QWORD PTR [rdi+0x740]
   1e443:	movsd  xmm1,QWORD PTR [rsi+0x70]
   1e448:	movsd  QWORD PTR [rdx+0x38],xmm8
   1e44e:	movsd  xmm3,QWORD PTR [rdi+0x748]
   1e456:	movsd  xmm4,QWORD PTR [rsi+0x78]
   1e45b:	movapd xmm6,xmm0
   1e45f:	mulsd  xmm6,xmm1
   1e463:	movapd xmm5,xmm3
   1e467:	mulsd  xmm5,xmm4
   1e46b:	mulsd  xmm1,xmm3
   1e46f:	mulsd  xmm0,xmm4
   1e473:	movapd xmm2,xmm6
   1e477:	subsd  xmm2,xmm5
   1e47b:	addsd  xmm0,xmm1
   1e47f:	addsd  xmm12,xmm2
   1e484:	addsd  xmm7,xmm0
   1e488:	movsd  QWORD PTR [rdx+0x40],xmm12
   1e48e:	movsd  QWORD PTR [rdx+0x48],xmm7
   1e493:	movsd  xmm0,QWORD PTR [rdi+0x750]
   1e49b:	movsd  xmm3,QWORD PTR [rdi+0x758]
   1e4a3:	movsd  xmm1,QWORD PTR [rsi+0x70]
   1e4a8:	movsd  xmm4,QWORD PTR [rsi+0x78]
   1e4ad:	movapd xmm6,xmm0
   1e4b1:	movapd xmm5,xmm3
   1e4b5:	mulsd  xmm6,xmm1
   1e4b9:	mulsd  xmm5,xmm4
   1e4bd:	mulsd  xmm1,xmm3
   1e4c1:	mulsd  xmm0,xmm4
   1e4c5:	movsd  xmm4,QWORD PTR [rsp+0x8]
   1e4cb:	movapd xmm2,xmm6
   1e4cf:	subsd  xmm2,xmm5
   1e4d3:	movsd  xmm5,QWORD PTR [rsp]
   1e4d8:	addsd  xmm0,xmm1
   1e4dc:	addsd  xmm5,xmm2
   1e4e0:	addsd  xmm4,xmm0
   1e4e4:	movsd  QWORD PTR [rdx+0x50],xmm5
   1e4e9:	movsd  xmm0,QWORD PTR [rdi+0x760]
   1e4f1:	movsd  xmm1,QWORD PTR [rsi+0x70]
   1e4f6:	movsd  QWORD PTR [rsp],xmm5
   1e4fb:	movsd  QWORD PTR [rdx+0x58],xmm4
   1e500:	movsd  xmm3,QWORD PTR [rdi+0x768]
   1e508:	movapd xmm2,xmm0
   1e50c:	movsd  QWORD PTR [rsp+0x8],xmm4
   1e512:	movsd  xmm4,QWORD PTR [rsi+0x78]
   1e517:	mulsd  xmm2,xmm1
   1e51b:	mulsd  xmm1,xmm3
   1e51f:	movapd xmm5,xmm3
   1e523:	mulsd  xmm5,xmm4
   1e527:	mulsd  xmm0,xmm4
   1e52b:	subsd  xmm2,xmm5
   1e52f:	addsd  xmm2,QWORD PTR [rsp+0x10]
   1e535:	addsd  xmm0,xmm1
   1e539:	addsd  xmm0,QWORD PTR [rsp+0x18]
   1e53f:	movsd  QWORD PTR [rdx+0x60],xmm2
   1e544:	movsd  xmm1,QWORD PTR [rsi+0x70]
   1e549:	movsd  QWORD PTR [rdx+0x68],xmm0
   1e54e:	movsd  xmm3,QWORD PTR [rdi+0x778]
   1e556:	movsd  xmm4,QWORD PTR [rsi+0x78]
   1e55b:	movsd  QWORD PTR [rsp+0x18],xmm0
   1e561:	movsd  xmm0,QWORD PTR [rdi+0x770]
   1e569:	movsd  QWORD PTR [rsp+0x10],xmm2
   1e56f:	movapd xmm5,xmm3
   1e573:	mulsd  xmm5,xmm4
   1e577:	movapd xmm2,xmm0
   1e57b:	mulsd  xmm2,xmm1
   1e57f:	mulsd  xmm0,xmm4
   1e583:	mulsd  xmm1,xmm3
   1e587:	subsd  xmm2,xmm5
   1e58b:	addsd  xmm2,QWORD PTR [rsp+0x20]
   1e591:	addsd  xmm0,xmm1
   1e595:	movsd  xmm1,QWORD PTR [rsp+0x28]
   1e59b:	movsd  QWORD PTR [rdx+0x70],xmm2
   1e5a0:	movsd  QWORD PTR [rsp+0x20],xmm2
   1e5a6:	addsd  xmm1,xmm0
   1e5aa:	movsd  QWORD PTR [rdx+0x78],xmm1
   1e5af:	movsd  QWORD PTR [rsp+0x28],xmm1
   1e5b5:	call   16e80 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm7ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1e5ba:	movsd  xmm3,QWORD PTR [rdi+0x808]
   1e5c2:	movsd  xmm0,QWORD PTR [rdi+0x800]
   1e5ca:	movsd  xmm4,QWORD PTR [rsi+0x88]
   1e5d2:	movsd  xmm1,QWORD PTR [rsi+0x80]
   1e5da:	movapd xmm2,xmm0
   1e5de:	movapd xmm5,xmm3
   1e5e2:	movsd  xmm6,QWORD PTR [rsp+0x30]
   1e5e8:	mulsd  xmm5,xmm4
   1e5ec:	mulsd  xmm2,xmm1
   1e5f0:	mulsd  xmm0,xmm4
   1e5f4:	mulsd  xmm1,xmm3
   1e5f8:	subsd  xmm2,xmm5
   1e5fc:	addsd  xmm0,xmm1
   1e600:	addsd  xmm6,xmm2
   1e604:	addsd  xmm11,xmm0
   1e609:	movsd  QWORD PTR [rdx],xmm6
   1e60d:	movsd  xmm0,QWORD PTR [rdi+0x810]
   1e615:	movsd  xmm1,QWORD PTR [rsi+0x80]
   1e61d:	movsd  QWORD PTR [rsp+0x30],xmm6
   1e623:	movsd  QWORD PTR [rdx+0x8],xmm11
   1e629:	movsd  xmm3,QWORD PTR [rdi+0x818]
   1e631:	movapd xmm2,xmm0
   1e635:	movsd  xmm4,QWORD PTR [rsi+0x88]
   1e63d:	mulsd  xmm2,xmm1
   1e641:	mulsd  xmm0,xmm4
   1e645:	movapd xmm5,xmm3
   1e649:	mulsd  xmm5,xmm4
   1e64d:	mulsd  xmm1,xmm3
   1e651:	subsd  xmm2,xmm5
   1e655:	addsd  xmm0,xmm1
   1e659:	addsd  xmm15,xmm2
   1e65e:	addsd  xmm10,xmm0
   1e663:	movsd  QWORD PTR [rdx+0x10],xmm15
   1e669:	movsd  xmm0,QWORD PTR [rdi+0x820]
   1e671:	movsd  xmm1,QWORD PTR [rsi+0x80]
   1e679:	movsd  QWORD PTR [rdx+0x18],xmm10
   1e67f:	movsd  xmm3,QWORD PTR [rdi+0x828]
   1e687:	movsd  xmm4,QWORD PTR [rsi+0x88]
   1e68f:	movapd xmm2,xmm0
   1e693:	mulsd  xmm2,xmm1
   1e697:	movapd xmm5,xmm3
   1e69b:	mulsd  xmm5,xmm4
   1e69f:	mulsd  xmm1,xmm3
   1e6a3:	mulsd  xmm0,xmm4
   1e6a7:	subsd  xmm2,xmm5
   1e6ab:	addsd  xmm0,xmm1
   1e6af:	addsd  xmm14,xmm2
   1e6b4:	addsd  xmm9,xmm0
   1e6b9:	movsd  QWORD PTR [rdx+0x20],xmm14
   1e6bf:	movsd  xmm0,QWORD PTR [rdi+0x830]
   1e6c7:	movsd  xmm1,QWORD PTR [rsi+0x80]
   1e6cf:	movsd  QWORD PTR [rdx+0x28],xmm9
   1e6d5:	movsd  xmm3,QWORD PTR [rdi+0x838]
   1e6dd:	movapd xmm2,xmm0
   1e6e1:	movsd  xmm4,QWORD PTR [rsi+0x88]
   1e6e9:	mulsd  xmm2,xmm1
   1e6ed:	mulsd  xmm0,xmm4
   1e6f1:	movapd xmm5,xmm3
   1e6f5:	mulsd  xmm5,xmm4
   1e6f9:	mulsd  xmm1,xmm3
   1e6fd:	subsd  xmm2,xmm5
   1e701:	addsd  xmm0,xmm1
   1e705:	addsd  xmm13,xmm2
   1e70a:	addsd  xmm8,xmm0
   1e70f:	movsd  QWORD PTR [rdx+0x30],xmm13
   1e715:	movsd  xmm0,QWORD PTR [rdi+0x840]
   1e71d:	movsd  xmm1,QWORD PTR [rsi+0x80]
   1e725:	movsd  QWORD PTR [rdx+0x38],xmm8
   1e72b:	movsd  xmm3,QWORD PTR [rdi+0x848]
   1e733:	movsd  xmm4,QWORD PTR [rsi+0x88]
   1e73b:	movapd xmm2,xmm0
   1e73f:	mulsd  xmm2,xmm1
   1e743:	movapd xmm5,xmm3
   1e747:	mulsd  xmm5,xmm4
   1e74b:	mulsd  xmm1,xmm3
   1e74f:	mulsd  xmm0,xmm4
   1e753:	subsd  xmm2,xmm5
   1e757:	addsd  xmm0,xmm1
   1e75b:	addsd  xmm12,xmm2
   1e760:	addsd  xmm7,xmm0
   1e764:	movsd  QWORD PTR [rdx+0x40],xmm12
   1e76a:	movsd  QWORD PTR [rdx+0x48],xmm7
   1e76f:	movsd  xmm0,QWORD PTR [rdi+0x850]
   1e777:	movsd  xmm3,QWORD PTR [rdi+0x858]
   1e77f:	movsd  xmm4,QWORD PTR [rsi+0x88]
   1e787:	movsd  xmm1,QWORD PTR [rsi+0x80]
   1e78f:	movapd xmm2,xmm0
   1e793:	movapd xmm5,xmm3
   1e797:	mulsd  xmm2,xmm1
   1e79b:	mulsd  xmm0,xmm4
   1e79f:	mulsd  xmm1,xmm3
   1e7a3:	movsd  xmm3,QWORD PTR [rsp+0x8]
   1e7a9:	mulsd  xmm5,xmm4
   1e7ad:	addsd  xmm0,xmm1
   1e7b1:	subsd  xmm2,xmm5
   1e7b5:	addsd  xmm2,QWORD PTR [rsp]
   1e7ba:	addsd  xmm3,xmm0
   1e7be:	movsd  QWORD PTR [rdx+0x50],xmm2
   1e7c3:	movsd  xmm0,QWORD PTR [rdi+0x860]
   1e7cb:	movsd  xmm1,QWORD PTR [rsi+0x80]
   1e7d3:	movsd  QWORD PTR [rsp],xmm2
   1e7d8:	movsd  QWORD PTR [rdx+0x58],xmm3
   1e7dd:	movsd  xmm4,QWORD PTR [rsi+0x88]
   1e7e5:	movapd xmm2,xmm0
   1e7e9:	movsd  QWORD PTR [rsp+0x8],xmm3
   1e7ef:	movsd  xmm3,QWORD PTR [rdi+0x868]
   1e7f7:	mulsd  xmm2,xmm1
   1e7fb:	mulsd  xmm0,xmm4
   1e7ff:	mulsd  xmm1,xmm3
   1e803:	movapd xmm5,xmm3
   1e807:	mulsd  xmm5,xmm4
   1e80b:	movsd  xmm4,QWORD PTR [rsp+0x18]
   1e811:	addsd  xmm0,xmm1
   1e815:	subsd  xmm2,xmm5
   1e819:	addsd  xmm2,QWORD PTR [rsp+0x10]
   1e81f:	addsd  xmm4,xmm0
   1e823:	movsd  QWORD PTR [rdx+0x60],xmm2
   1e828:	movsd  xmm0,QWORD PTR [rdi+0x870]
   1e830:	movsd  xmm1,QWORD PTR [rsi+0x80]
   1e838:	movsd  QWORD PTR [rsp+0x10],xmm2
   1e83e:	movsd  QWORD PTR [rdx+0x68],xmm4
   1e843:	movsd  xmm3,QWORD PTR [rdi+0x878]
   1e84b:	movapd xmm2,xmm0
   1e84f:	movsd  QWORD PTR [rsp+0x18],xmm4
   1e855:	movsd  xmm4,QWORD PTR [rsi+0x88]
   1e85d:	mulsd  xmm2,xmm1
   1e861:	mulsd  xmm1,xmm3
   1e865:	movapd xmm5,xmm3
   1e869:	mulsd  xmm5,xmm4
   1e86d:	mulsd  xmm0,xmm4
   1e871:	subsd  xmm2,xmm5
   1e875:	addsd  xmm2,QWORD PTR [rsp+0x20]
   1e87b:	addsd  xmm0,xmm1
   1e87f:	addsd  xmm0,QWORD PTR [rsp+0x28]
   1e885:	movsd  QWORD PTR [rdx+0x70],xmm2
   1e88a:	movsd  QWORD PTR [rdx+0x78],xmm0
   1e88f:	movsd  QWORD PTR [rsp+0x20],xmm2
   1e895:	movsd  QWORD PTR [rsp+0x28],xmm0
   1e89b:	call   16920 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm8ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1e8a0:	movsd  xmm3,QWORD PTR [rdi+0x908]
   1e8a8:	movsd  xmm0,QWORD PTR [rdi+0x900]
   1e8b0:	movsd  xmm4,QWORD PTR [rsi+0x98]
   1e8b8:	movsd  xmm1,QWORD PTR [rsi+0x90]
   1e8c0:	movapd xmm2,xmm0
   1e8c4:	movapd xmm5,xmm3
   1e8c8:	movsd  xmm6,QWORD PTR [rsp+0x30]
   1e8ce:	mulsd  xmm5,xmm4
   1e8d2:	mulsd  xmm2,xmm1
   1e8d6:	mulsd  xmm0,xmm4
   1e8da:	mulsd  xmm1,xmm3
   1e8de:	subsd  xmm2,xmm5
   1e8e2:	addsd  xmm0,xmm1
   1e8e6:	addsd  xmm6,xmm2
   1e8ea:	addsd  xmm11,xmm0
   1e8ef:	movsd  QWORD PTR [rdx],xmm6
   1e8f3:	movsd  xmm0,QWORD PTR [rdi+0x910]
   1e8fb:	movsd  xmm1,QWORD PTR [rsi+0x90]
   1e903:	movsd  QWORD PTR [rsp+0x30],xmm6
   1e909:	movsd  QWORD PTR [rdx+0x8],xmm11
   1e90f:	movsd  xmm3,QWORD PTR [rdi+0x918]
   1e917:	movapd xmm2,xmm0
   1e91b:	movsd  xmm4,QWORD PTR [rsi+0x98]
   1e923:	mulsd  xmm2,xmm1
   1e927:	mulsd  xmm0,xmm4
   1e92b:	movapd xmm5,xmm3
   1e92f:	mulsd  xmm5,xmm4
   1e933:	mulsd  xmm1,xmm3
   1e937:	subsd  xmm2,xmm5
   1e93b:	addsd  xmm0,xmm1
   1e93f:	addsd  xmm15,xmm2
   1e944:	addsd  xmm10,xmm0
   1e949:	movsd  QWORD PTR [rdx+0x10],xmm15
   1e94f:	movsd  xmm0,QWORD PTR [rdi+0x920]
   1e957:	movsd  xmm1,QWORD PTR [rsi+0x90]
   1e95f:	movsd  QWORD PTR [rdx+0x18],xmm10
   1e965:	movsd  xmm3,QWORD PTR [rdi+0x928]
   1e96d:	movsd  xmm4,QWORD PTR [rsi+0x98]
   1e975:	movapd xmm2,xmm0
   1e979:	mulsd  xmm2,xmm1
   1e97d:	movapd xmm5,xmm3
   1e981:	mulsd  xmm5,xmm4
   1e985:	mulsd  xmm1,xmm3
   1e989:	mulsd  xmm0,xmm4
   1e98d:	subsd  xmm2,xmm5
   1e991:	addsd  xmm0,xmm1
   1e995:	addsd  xmm14,xmm2
   1e99a:	addsd  xmm9,xmm0
   1e99f:	movsd  QWORD PTR [rdx+0x20],xmm14
   1e9a5:	movsd  xmm0,QWORD PTR [rdi+0x930]
   1e9ad:	movsd  xmm1,QWORD PTR [rsi+0x90]
   1e9b5:	movsd  QWORD PTR [rdx+0x28],xmm9
   1e9bb:	movsd  xmm3,QWORD PTR [rdi+0x938]
   1e9c3:	movapd xmm2,xmm0
   1e9c7:	movsd  xmm4,QWORD PTR [rsi+0x98]
   1e9cf:	mulsd  xmm2,xmm1
   1e9d3:	mulsd  xmm0,xmm4
   1e9d7:	movapd xmm5,xmm3
   1e9db:	mulsd  xmm5,xmm4
   1e9df:	mulsd  xmm1,xmm3
   1e9e3:	subsd  xmm2,xmm5
   1e9e7:	addsd  xmm0,xmm1
   1e9eb:	addsd  xmm13,xmm2
   1e9f0:	addsd  xmm8,xmm0
   1e9f5:	movsd  QWORD PTR [rdx+0x30],xmm13
   1e9fb:	movsd  xmm0,QWORD PTR [rdi+0x940]
   1ea03:	movsd  xmm1,QWORD PTR [rsi+0x90]
   1ea0b:	movsd  QWORD PTR [rdx+0x38],xmm8
   1ea11:	movsd  xmm3,QWORD PTR [rdi+0x948]
   1ea19:	movsd  xmm4,QWORD PTR [rsi+0x98]
   1ea21:	movapd xmm2,xmm0
   1ea25:	mulsd  xmm2,xmm1
   1ea29:	movapd xmm5,xmm3
   1ea2d:	mulsd  xmm5,xmm4
   1ea31:	mulsd  xmm1,xmm3
   1ea35:	mulsd  xmm0,xmm4
   1ea39:	subsd  xmm2,xmm5
   1ea3d:	addsd  xmm0,xmm1
   1ea41:	addsd  xmm12,xmm2
   1ea46:	addsd  xmm7,xmm0
   1ea4a:	movsd  QWORD PTR [rdx+0x40],xmm12
   1ea50:	movsd  QWORD PTR [rdx+0x48],xmm7
   1ea55:	movsd  xmm0,QWORD PTR [rdi+0x950]
   1ea5d:	movsd  xmm3,QWORD PTR [rdi+0x958]
   1ea65:	movsd  xmm4,QWORD PTR [rsi+0x98]
   1ea6d:	movsd  xmm1,QWORD PTR [rsi+0x90]
   1ea75:	movapd xmm2,xmm0
   1ea79:	movapd xmm5,xmm3
   1ea7d:	mulsd  xmm2,xmm1
   1ea81:	mulsd  xmm0,xmm4
   1ea85:	mulsd  xmm1,xmm3
   1ea89:	mulsd  xmm5,xmm4
   1ea8d:	addsd  xmm0,xmm1
   1ea91:	movsd  xmm1,QWORD PTR [rsp+0x8]
   1ea97:	subsd  xmm2,xmm5
   1ea9b:	addsd  xmm2,QWORD PTR [rsp]
   1eaa0:	addsd  xmm1,xmm0
   1eaa4:	movsd  QWORD PTR [rdx+0x50],xmm2
   1eaa9:	movsd  xmm0,QWORD PTR [rdi+0x960]
   1eab1:	movsd  QWORD PTR [rsp],xmm2
   1eab6:	movsd  QWORD PTR [rdx+0x58],xmm1
   1eabb:	movsd  xmm3,QWORD PTR [rdi+0x968]
   1eac3:	movsd  xmm4,QWORD PTR [rsi+0x98]
   1eacb:	movapd xmm2,xmm0
   1eacf:	movsd  QWORD PTR [rsp+0x8],xmm1
   1ead5:	movsd  xmm1,QWORD PTR [rsi+0x90]
   1eadd:	mulsd  xmm0,xmm4
   1eae1:	movapd xmm5,xmm3
   1eae5:	mulsd  xmm2,xmm1
   1eae9:	mulsd  xmm1,xmm3
   1eaed:	movsd  xmm3,QWORD PTR [rsp+0x18]
   1eaf3:	mulsd  xmm5,xmm4
   1eaf7:	addsd  xmm0,xmm1
   1eafb:	subsd  xmm2,xmm5
   1eaff:	addsd  xmm2,QWORD PTR [rsp+0x10]
   1eb05:	addsd  xmm3,xmm0
   1eb09:	movsd  QWORD PTR [rdx+0x60],xmm2
   1eb0e:	movsd  xmm0,QWORD PTR [rdi+0x970]
   1eb16:	movsd  xmm1,QWORD PTR [rsi+0x90]
   1eb1e:	movsd  QWORD PTR [rsp+0x10],xmm2
   1eb24:	movsd  QWORD PTR [rsp+0x18],xmm3
   1eb2a:	movapd xmm2,xmm0
   1eb2e:	movsd  QWORD PTR [rdx+0x68],xmm3
   1eb33:	movsd  xmm3,QWORD PTR [rdi+0x978]
   1eb3b:	mulsd  xmm2,xmm1
   1eb3f:	movsd  xmm4,QWORD PTR [rsi+0x98]
   1eb47:	mulsd  xmm1,xmm3
   1eb4b:	movapd xmm5,xmm3
   1eb4f:	mulsd  xmm0,xmm4
   1eb53:	mulsd  xmm5,xmm4
   1eb57:	movsd  xmm4,QWORD PTR [rsp+0x28]
   1eb5d:	addsd  xmm0,xmm1
   1eb61:	subsd  xmm2,xmm5
   1eb65:	addsd  xmm2,QWORD PTR [rsp+0x20]
   1eb6b:	addsd  xmm4,xmm0
   1eb6f:	movsd  QWORD PTR [rdx+0x70],xmm2
   1eb74:	movsd  QWORD PTR [rsp+0x20],xmm2
   1eb7a:	movsd  QWORD PTR [rdx+0x78],xmm4
   1eb7f:	movsd  QWORD PTR [rsp+0x28],xmm4
   1eb85:	call   163c0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm9ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1eb8a:	movsd  xmm3,QWORD PTR [rdi+0xa08]
   1eb92:	movsd  xmm0,QWORD PTR [rdi+0xa00]
   1eb9a:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   1eba2:	movsd  xmm1,QWORD PTR [rsi+0xa0]
   1ebaa:	movapd xmm2,xmm0
   1ebae:	movapd xmm5,xmm3
   1ebb2:	movsd  xmm6,QWORD PTR [rsp+0x30]
   1ebb8:	mulsd  xmm5,xmm4
   1ebbc:	mulsd  xmm2,xmm1
   1ebc0:	mulsd  xmm0,xmm4
   1ebc4:	mulsd  xmm1,xmm3
   1ebc8:	subsd  xmm2,xmm5
   1ebcc:	addsd  xmm0,xmm1
   1ebd0:	addsd  xmm6,xmm2
   1ebd4:	addsd  xmm11,xmm0
   1ebd9:	movsd  QWORD PTR [rdx],xmm6
   1ebdd:	movsd  xmm0,QWORD PTR [rdi+0xa10]
   1ebe5:	movsd  xmm1,QWORD PTR [rsi+0xa0]
   1ebed:	movsd  QWORD PTR [rsp+0x30],xmm6
   1ebf3:	movsd  QWORD PTR [rdx+0x8],xmm11
   1ebf9:	movsd  xmm3,QWORD PTR [rdi+0xa18]
   1ec01:	movapd xmm2,xmm0
   1ec05:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   1ec0d:	mulsd  xmm2,xmm1
   1ec11:	mulsd  xmm0,xmm4
   1ec15:	movapd xmm5,xmm3
   1ec19:	mulsd  xmm5,xmm4
   1ec1d:	mulsd  xmm1,xmm3
   1ec21:	subsd  xmm2,xmm5
   1ec25:	addsd  xmm0,xmm1
   1ec29:	addsd  xmm15,xmm2
   1ec2e:	addsd  xmm10,xmm0
   1ec33:	movsd  QWORD PTR [rdx+0x10],xmm15
   1ec39:	movsd  xmm0,QWORD PTR [rdi+0xa20]
   1ec41:	movsd  xmm1,QWORD PTR [rsi+0xa0]
   1ec49:	movsd  QWORD PTR [rdx+0x18],xmm10
   1ec4f:	movsd  xmm3,QWORD PTR [rdi+0xa28]
   1ec57:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   1ec5f:	movapd xmm2,xmm0
   1ec63:	mulsd  xmm2,xmm1
   1ec67:	movapd xmm5,xmm3
   1ec6b:	mulsd  xmm5,xmm4
   1ec6f:	mulsd  xmm1,xmm3
   1ec73:	mulsd  xmm0,xmm4
   1ec77:	subsd  xmm2,xmm5
   1ec7b:	addsd  xmm0,xmm1
   1ec7f:	addsd  xmm14,xmm2
   1ec84:	addsd  xmm9,xmm0
   1ec89:	movsd  QWORD PTR [rdx+0x20],xmm14
   1ec8f:	movsd  xmm0,QWORD PTR [rdi+0xa30]
   1ec97:	movsd  xmm1,QWORD PTR [rsi+0xa0]
   1ec9f:	movsd  QWORD PTR [rdx+0x28],xmm9
   1eca5:	movsd  xmm3,QWORD PTR [rdi+0xa38]
   1ecad:	movapd xmm2,xmm0
   1ecb1:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   1ecb9:	mulsd  xmm2,xmm1
   1ecbd:	mulsd  xmm0,xmm4
   1ecc1:	movapd xmm5,xmm3
   1ecc5:	mulsd  xmm5,xmm4
   1ecc9:	mulsd  xmm1,xmm3
   1eccd:	subsd  xmm2,xmm5
   1ecd1:	addsd  xmm0,xmm1
   1ecd5:	addsd  xmm13,xmm2
   1ecda:	addsd  xmm8,xmm0
   1ecdf:	movsd  QWORD PTR [rdx+0x30],xmm13
   1ece5:	movsd  xmm0,QWORD PTR [rdi+0xa40]
   1eced:	movsd  xmm1,QWORD PTR [rsi+0xa0]
   1ecf5:	movsd  QWORD PTR [rdx+0x38],xmm8
   1ecfb:	movsd  xmm3,QWORD PTR [rdi+0xa48]
   1ed03:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   1ed0b:	movapd xmm2,xmm0
   1ed0f:	mulsd  xmm2,xmm1
   1ed13:	movapd xmm5,xmm3
   1ed17:	mulsd  xmm5,xmm4
   1ed1b:	mulsd  xmm1,xmm3
   1ed1f:	mulsd  xmm0,xmm4
   1ed23:	subsd  xmm2,xmm5
   1ed27:	addsd  xmm0,xmm1
   1ed2b:	addsd  xmm12,xmm2
   1ed30:	addsd  xmm7,xmm0
   1ed34:	movsd  QWORD PTR [rdx+0x40],xmm12
   1ed3a:	movsd  QWORD PTR [rdx+0x48],xmm7
   1ed3f:	movsd  xmm0,QWORD PTR [rdi+0xa50]
   1ed47:	movsd  xmm3,QWORD PTR [rdi+0xa58]
   1ed4f:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   1ed57:	movsd  xmm1,QWORD PTR [rsi+0xa0]
   1ed5f:	movapd xmm2,xmm0
   1ed63:	movapd xmm5,xmm3
   1ed67:	mulsd  xmm5,xmm4
   1ed6b:	mulsd  xmm2,xmm1
   1ed6f:	mulsd  xmm0,xmm4
   1ed73:	mulsd  xmm1,xmm3
   1ed77:	subsd  xmm2,xmm5
   1ed7b:	addsd  xmm2,QWORD PTR [rsp]
   1ed80:	addsd  xmm0,xmm1
   1ed84:	addsd  xmm0,QWORD PTR [rsp+0x8]
   1ed8a:	movsd  QWORD PTR [rdx+0x50],xmm2
   1ed8f:	movsd  xmm1,QWORD PTR [rsi+0xa0]
   1ed97:	movsd  QWORD PTR [rsp],xmm2
   1ed9c:	movsd  QWORD PTR [rdx+0x58],xmm0
   1eda1:	movsd  xmm3,QWORD PTR [rdi+0xa68]
   1eda9:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   1edb1:	movsd  QWORD PTR [rsp+0x8],xmm0
   1edb7:	movsd  xmm0,QWORD PTR [rdi+0xa60]
   1edbf:	movapd xmm5,xmm3
   1edc3:	mulsd  xmm5,xmm4
   1edc7:	movapd xmm2,xmm0
   1edcb:	mulsd  xmm2,xmm1
   1edcf:	mulsd  xmm0,xmm4
   1edd3:	mulsd  xmm1,xmm3
   1edd7:	subsd  xmm2,xmm5
   1eddb:	addsd  xmm2,QWORD PTR [rsp+0x10]
   1ede1:	addsd  xmm0,xmm1
   1ede5:	movsd  xmm1,QWORD PTR [rsp+0x18]
   1edeb:	movsd  QWORD PTR [rdx+0x60],xmm2
   1edf0:	movsd  QWORD PTR [rsp+0x10],xmm2
   1edf6:	addsd  xmm1,xmm0
   1edfa:	movsd  xmm0,QWORD PTR [rdi+0xa70]
   1ee02:	movapd xmm2,xmm0
   1ee06:	movsd  QWORD PTR [rdx+0x68],xmm1
   1ee0b:	movsd  xmm3,QWORD PTR [rdi+0xa78]
   1ee13:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   1ee1b:	movsd  QWORD PTR [rsp+0x18],xmm1
   1ee21:	movsd  xmm1,QWORD PTR [rsi+0xa0]
   1ee29:	mulsd  xmm0,xmm4
   1ee2d:	movapd xmm5,xmm3
   1ee31:	mulsd  xmm2,xmm1
   1ee35:	mulsd  xmm1,xmm3
   1ee39:	movsd  xmm3,QWORD PTR [rsp+0x28]
   1ee3f:	mulsd  xmm5,xmm4
   1ee43:	addsd  xmm0,xmm1
   1ee47:	subsd  xmm2,xmm5
   1ee4b:	addsd  xmm2,QWORD PTR [rsp+0x20]
   1ee51:	addsd  xmm3,xmm0
   1ee55:	movsd  QWORD PTR [rdx+0x70],xmm2
   1ee5a:	movsd  QWORD PTR [rsp+0x20],xmm2
   1ee60:	movsd  QWORD PTR [rdx+0x78],xmm3
   1ee65:	movsd  QWORD PTR [rsp+0x28],xmm3
   1ee6b:	call   15e60 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm10ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1ee70:	movsd  xmm3,QWORD PTR [rdi+0xb08]
   1ee78:	movsd  xmm0,QWORD PTR [rdi+0xb00]
   1ee80:	movsd  xmm4,QWORD PTR [rsi+0xb8]
   1ee88:	movsd  xmm1,QWORD PTR [rsi+0xb0]
   1ee90:	movapd xmm2,xmm0
   1ee94:	movapd xmm5,xmm3
   1ee98:	mulsd  xmm5,xmm4
   1ee9c:	mulsd  xmm2,xmm1
   1eea0:	mulsd  xmm0,xmm4
   1eea4:	mulsd  xmm1,xmm3
   1eea8:	subsd  xmm2,xmm5
   1eeac:	addsd  xmm2,QWORD PTR [rsp+0x30]
   1eeb2:	addsd  xmm0,xmm1
   1eeb6:	movsd  QWORD PTR [rdx],xmm2
   1eeba:	movsd  xmm1,QWORD PTR [rsi+0xb0]
   1eec2:	movsd  QWORD PTR [rsp+0x30],xmm2
   1eec8:	movapd xmm4,xmm0
   1eecc:	movsd  xmm0,QWORD PTR [rdi+0xb10]
   1eed4:	addsd  xmm4,xmm11
   1eed9:	movapd xmm11,xmm0
   1eede:	mulsd  xmm11,xmm1
   1eee3:	movsd  QWORD PTR [rdx+0x8],xmm4
   1eee8:	movsd  xmm3,QWORD PTR [rdi+0xb18]
   1eef0:	movsd  QWORD PTR [rsp+0x38],xmm4
   1eef6:	movsd  xmm4,QWORD PTR [rsi+0xb8]
   1eefe:	mulsd  xmm1,xmm3
   1ef02:	mulsd  xmm0,xmm4
   1ef06:	movapd xmm2,xmm11
   1ef0b:	movapd xmm11,xmm3
   1ef10:	mulsd  xmm11,xmm4
   1ef15:	addsd  xmm0,xmm1
   1ef19:	subsd  xmm2,xmm11
   1ef1e:	movapd xmm11,xmm2
   1ef23:	addsd  xmm11,xmm15
   1ef28:	movapd xmm15,xmm0
   1ef2d:	addsd  xmm15,xmm10
   1ef32:	movsd  QWORD PTR [rdx+0x10],xmm11
   1ef38:	movsd  xmm0,QWORD PTR [rdi+0xb20]
   1ef40:	movsd  xmm1,QWORD PTR [rsi+0xb0]
   1ef48:	movsd  QWORD PTR [rdx+0x18],xmm15
   1ef4e:	movsd  xmm3,QWORD PTR [rdi+0xb28]
   1ef56:	movsd  xmm4,QWORD PTR [rsi+0xb8]
   1ef5e:	movapd xmm10,xmm0
   1ef63:	movsd  QWORD PTR [rsp+0x40],xmm11
   1ef6a:	mulsd  xmm10,xmm1
   1ef6f:	movsd  QWORD PTR [rsp+0x48],xmm15
   1ef76:	mulsd  xmm0,xmm4
   1ef7a:	mulsd  xmm1,xmm3
   1ef7e:	movapd xmm2,xmm10
   1ef83:	movapd xmm10,xmm3
   1ef88:	mulsd  xmm10,xmm4
   1ef8d:	addsd  xmm0,xmm1
   1ef91:	subsd  xmm2,xmm10
   1ef96:	movapd xmm10,xmm2
   1ef9b:	addsd  xmm10,xmm14
   1efa0:	movapd xmm14,xmm0
   1efa5:	addsd  xmm14,xmm9
   1efaa:	movsd  QWORD PTR [rdx+0x20],xmm10
   1efb0:	movsd  xmm0,QWORD PTR [rsi+0xb0]
   1efb8:	movsd  xmm11,QWORD PTR [rdi+0xb30]
   1efc1:	movsd  QWORD PTR [rdx+0x28],xmm14
   1efc7:	movsd  xmm2,QWORD PTR [rdi+0xb38]
   1efcf:	movsd  xmm3,QWORD PTR [rsi+0xb8]
   1efd7:	movapd xmm9,xmm11
   1efdc:	movsd  QWORD PTR [rsp+0x58],xmm14
   1efe3:	mulsd  xmm9,xmm0
   1efe8:	movsd  QWORD PTR [rsp+0x50],xmm10
   1efef:	mulsd  xmm0,xmm2
   1eff3:	mulsd  xmm11,xmm3
   1eff8:	movapd xmm1,xmm9
   1effd:	movapd xmm9,xmm2
   1f002:	mulsd  xmm9,xmm3
   1f007:	addsd  xmm11,xmm0
   1f00c:	addsd  xmm11,xmm8
   1f011:	subsd  xmm1,xmm9
   1f016:	movsd  QWORD PTR [rdx+0x38],xmm11
   1f01c:	addsd  xmm1,xmm13
   1f021:	movsd  QWORD PTR [rsp+0x60],xmm1
   1f027:	movsd  QWORD PTR [rdx+0x30],xmm1
   1f02c:	movsd  xmm10,QWORD PTR [rdi+0xb40]
   1f035:	movsd  xmm0,QWORD PTR [rsi+0xb0]
   1f03d:	movsd  xmm2,QWORD PTR [rdi+0xb48]
   1f045:	movsd  xmm3,QWORD PTR [rsi+0xb8]
   1f04d:	movapd xmm8,xmm10
   1f052:	mulsd  xmm8,xmm0
   1f057:	mulsd  xmm10,xmm3
   1f05c:	mulsd  xmm0,xmm2
   1f060:	movapd xmm15,xmm8
   1f065:	movapd xmm8,xmm2
   1f06a:	mulsd  xmm8,xmm3
   1f06f:	addsd  xmm10,xmm0
   1f074:	addsd  xmm10,xmm7
   1f079:	subsd  xmm15,xmm8
   1f07e:	movsd  QWORD PTR [rdx+0x48],xmm10
   1f084:	movsd  xmm2,QWORD PTR [rdi+0xb58]
   1f08c:	movsd  xmm3,QWORD PTR [rsi+0xb8]
   1f094:	addsd  xmm15,xmm12
   1f099:	movsd  QWORD PTR [rdx+0x40],xmm15
   1f09f:	movsd  xmm0,QWORD PTR [rsi+0xb0]
   1f0a7:	movsd  xmm9,QWORD PTR [rdi+0xb50]
   1f0b0:	movapd xmm8,xmm9
   1f0b5:	mulsd  xmm8,xmm0
   1f0ba:	mulsd  xmm9,xmm3
   1f0bf:	mulsd  xmm0,xmm2
   1f0c3:	movapd xmm14,xmm8
   1f0c8:	movapd xmm8,xmm2
   1f0cd:	mulsd  xmm8,xmm3
   1f0d2:	addsd  xmm9,xmm0
   1f0d7:	addsd  xmm9,QWORD PTR [rsp+0x8]
   1f0de:	movsd  QWORD PTR [rdx+0x58],xmm9
   1f0e4:	movsd  xmm2,QWORD PTR [rdi+0xb68]
   1f0ec:	movsd  xmm3,QWORD PTR [rsi+0xb8]
   1f0f4:	subsd  xmm14,xmm8
   1f0f9:	addsd  xmm14,QWORD PTR [rsp]
   1f0ff:	movapd xmm12,xmm2
   1f104:	mulsd  xmm12,xmm3
   1f109:	movsd  QWORD PTR [rdx+0x50],xmm14
   1f10f:	movsd  xmm0,QWORD PTR [rsi+0xb0]
   1f117:	movsd  xmm8,QWORD PTR [rdi+0xb60]
   1f120:	movapd xmm13,xmm8
   1f125:	mulsd  xmm13,xmm0
   1f12a:	mulsd  xmm8,xmm3
   1f12f:	mulsd  xmm0,xmm2
   1f133:	subsd  xmm13,xmm12
   1f138:	addsd  xmm13,QWORD PTR [rsp+0x10]
   1f13f:	addsd  xmm8,xmm0
   1f144:	addsd  xmm8,QWORD PTR [rsp+0x18]
   1f14b:	movsd  QWORD PTR [rdx+0x60],xmm13
   1f151:	movsd  xmm7,QWORD PTR [rdi+0xb70]
   1f159:	movsd  xmm0,QWORD PTR [rsi+0xb0]
   1f161:	movsd  QWORD PTR [rdx+0x68],xmm8
   1f167:	movsd  xmm2,QWORD PTR [rdi+0xb78]
   1f16f:	movapd xmm12,xmm7
   1f174:	movsd  xmm3,QWORD PTR [rsi+0xb8]
   1f17c:	mulsd  xmm12,xmm0
   1f181:	mulsd  xmm7,xmm3
   1f185:	movapd xmm1,xmm2
   1f189:	mulsd  xmm1,xmm3
   1f18d:	mulsd  xmm0,xmm2
   1f191:	subsd  xmm12,xmm1
   1f196:	addsd  xmm12,QWORD PTR [rsp+0x20]
   1f19d:	addsd  xmm7,xmm0
   1f1a1:	addsd  xmm7,QWORD PTR [rsp+0x28]
   1f1a7:	movsd  QWORD PTR [rdx+0x70],xmm12
   1f1ad:	movsd  QWORD PTR [rdx+0x78],xmm7
   1f1b2:	call   15900 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm11ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1f1b7:	movsd  xmm4,QWORD PTR [rdi+0xc08]
   1f1bf:	movsd  xmm0,QWORD PTR [rdi+0xc00]
   1f1c7:	movsd  xmm5,QWORD PTR [rsi+0xc8]
   1f1cf:	movsd  xmm3,QWORD PTR [rsi+0xc0]
   1f1d7:	movapd xmm2,xmm0
   1f1db:	movapd xmm1,xmm4
   1f1df:	mulsd  xmm1,xmm5
   1f1e3:	mulsd  xmm2,xmm3
   1f1e7:	mulsd  xmm0,xmm5
   1f1eb:	mulsd  xmm3,xmm4
   1f1ef:	subsd  xmm2,xmm1
   1f1f3:	addsd  xmm2,QWORD PTR [rsp+0x30]
   1f1f9:	addsd  xmm0,xmm3
   1f1fd:	addsd  xmm0,QWORD PTR [rsp+0x38]
   1f203:	movsd  QWORD PTR [rdx],xmm2
   1f207:	movsd  xmm5,QWORD PTR [rdi+0xc10]
   1f20f:	movsd  xmm3,QWORD PTR [rsi+0xc0]
   1f217:	movsd  QWORD PTR [rdx+0x8],xmm0
   1f21c:	movsd  xmm4,QWORD PTR [rdi+0xc18]
   1f224:	movapd xmm2,xmm5
   1f228:	movsd  xmm0,QWORD PTR [rsi+0xc8]
   1f230:	mulsd  xmm2,xmm3
   1f234:	mulsd  xmm3,xmm4
   1f238:	movapd xmm1,xmm4
   1f23c:	mulsd  xmm1,xmm0
   1f240:	mulsd  xmm0,xmm5
   1f244:	subsd  xmm2,xmm1
   1f248:	addsd  xmm2,QWORD PTR [rsp+0x40]
   1f24e:	addsd  xmm0,xmm3
   1f252:	addsd  xmm0,QWORD PTR [rsp+0x48]
   1f258:	movsd  QWORD PTR [rdx+0x10],xmm2
   1f25d:	movsd  xmm3,QWORD PTR [rsi+0xc0]
   1f265:	movsd  QWORD PTR [rdx+0x18],xmm0
   1f26a:	movsd  xmm0,QWORD PTR [rdi+0xc20]
   1f272:	movsd  xmm4,QWORD PTR [rdi+0xc28]
   1f27a:	movsd  xmm5,QWORD PTR [rsi+0xc8]
   1f282:	movapd xmm2,xmm0
   1f286:	movapd xmm1,xmm4
   1f28a:	mulsd  xmm1,xmm5
   1f28e:	mulsd  xmm2,xmm3
   1f292:	mulsd  xmm0,xmm5
   1f296:	mulsd  xmm3,xmm4
   1f29a:	subsd  xmm2,xmm1
   1f29e:	addsd  xmm2,QWORD PTR [rsp+0x50]
   1f2a4:	addsd  xmm0,xmm3
   1f2a8:	addsd  xmm0,QWORD PTR [rsp+0x58]
   1f2ae:	movsd  QWORD PTR [rdx+0x20],xmm2
   1f2b3:	movsd  xmm3,QWORD PTR [rsi+0xc0]
   1f2bb:	movsd  QWORD PTR [rdx+0x28],xmm0
   1f2c0:	movsd  xmm0,QWORD PTR [rdi+0xc30]
   1f2c8:	movsd  xmm4,QWORD PTR [rdi+0xc38]
   1f2d0:	movsd  xmm5,QWORD PTR [rsi+0xc8]
   1f2d8:	movapd xmm2,xmm0
   1f2dc:	movapd xmm6,xmm4
   1f2e0:	mulsd  xmm2,xmm3
   1f2e4:	mulsd  xmm0,xmm5
   1f2e8:	mulsd  xmm3,xmm4
   1f2ec:	mulsd  xmm6,xmm5
   1f2f0:	addsd  xmm0,xmm3
   1f2f4:	subsd  xmm2,xmm6
   1f2f8:	addsd  xmm2,QWORD PTR [rsp+0x60]
   1f2fe:	addsd  xmm0,xmm11
   1f303:	movsd  QWORD PTR [rdx+0x30],xmm2
   1f308:	movsd  QWORD PTR [rdx+0x38],xmm0
   1f30d:	movsd  xmm0,QWORD PTR [rdi+0xc40]
   1f315:	movsd  xmm2,QWORD PTR [rsi+0xc0]
   1f31d:	movsd  xmm3,QWORD PTR [rdi+0xc48]
   1f325:	movsd  xmm4,QWORD PTR [rsi+0xc8]
   1f32d:	movapd xmm1,xmm0
   1f331:	mulsd  xmm1,xmm2
   1f335:	movapd xmm5,xmm3
   1f339:	mulsd  xmm5,xmm4
   1f33d:	mulsd  xmm2,xmm3
   1f341:	mulsd  xmm0,xmm4
   1f345:	subsd  xmm1,xmm5
   1f349:	addsd  xmm0,xmm2
   1f34d:	addsd  xmm1,xmm15
   1f352:	addsd  xmm0,xmm10
   1f357:	movsd  QWORD PTR [rdx+0x40],xmm1
   1f35c:	movsd  xmm2,QWORD PTR [rsi+0xc0]
   1f364:	movsd  QWORD PTR [rdx+0x48],xmm0
   1f369:	movsd  xmm0,QWORD PTR [rdi+0xc50]
   1f371:	movsd  xmm3,QWORD PTR [rdi+0xc58]
   1f379:	movsd  xmm4,QWORD PTR [rsi+0xc8]
   1f381:	movapd xmm1,xmm0
   1f385:	movapd xmm5,xmm3
   1f389:	mulsd  xmm5,xmm4
   1f38d:	mulsd  xmm1,xmm2
   1f391:	mulsd  xmm0,xmm4
   1f395:	mulsd  xmm2,xmm3
   1f399:	subsd  xmm1,xmm5
   1f39d:	addsd  xmm0,xmm2
   1f3a1:	addsd  xmm1,xmm14
   1f3a6:	addsd  xmm0,xmm9
   1f3ab:	movsd  QWORD PTR [rdx+0x50],xmm1
   1f3b0:	movsd  xmm2,QWORD PTR [rsi+0xc0]
   1f3b8:	movsd  QWORD PTR [rdx+0x58],xmm0
   1f3bd:	movsd  xmm0,QWORD PTR [rdi+0xc60]
   1f3c5:	movsd  xmm3,QWORD PTR [rdi+0xc68]
   1f3cd:	movsd  xmm4,QWORD PTR [rsi+0xc8]
   1f3d5:	movapd xmm1,xmm0
   1f3d9:	movapd xmm5,xmm3
   1f3dd:	mulsd  xmm5,xmm4
   1f3e1:	mulsd  xmm1,xmm2
   1f3e5:	mulsd  xmm0,xmm4
   1f3e9:	mulsd  xmm2,xmm3
   1f3ed:	subsd  xmm1,xmm5
   1f3f1:	addsd  xmm0,xmm2
   1f3f5:	addsd  xmm1,xmm13
   1f3fa:	addsd  xmm0,xmm8
   1f3ff:	movsd  QWORD PTR [rdx+0x60],xmm1
   1f404:	movsd  xmm2,QWORD PTR [rsi+0xc0]
   1f40c:	movsd  QWORD PTR [rdx+0x68],xmm0
   1f411:	movsd  xmm0,QWORD PTR [rdi+0xc70]
   1f419:	movsd  xmm3,QWORD PTR [rdi+0xc78]
   1f421:	movsd  xmm4,QWORD PTR [rsi+0xc8]
   1f429:	movapd xmm5,xmm3
   1f42d:	movapd xmm1,xmm0
   1f431:	mulsd  xmm5,xmm4
   1f435:	mulsd  xmm0,xmm4
   1f439:	mulsd  xmm1,xmm2
   1f43d:	mulsd  xmm2,xmm3
   1f441:	subsd  xmm1,xmm5
   1f445:	addsd  xmm0,xmm2
   1f449:	addsd  xmm1,xmm12
   1f44e:	addsd  xmm0,xmm7
   1f452:	movsd  QWORD PTR [rdx+0x70],xmm1
   1f457:	movsd  QWORD PTR [rdx+0x78],xmm0
   1f45c:	call   153a0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm12ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1f461:	call   18870 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm13ELm0ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1f466:	call   17e10 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm14ELm0ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1f46b:	movsd  xmm3,QWORD PTR [rdi+0xf08]
   1f473:	movsd  xmm0,QWORD PTR [rdi+0xf00]
   1f47b:	movsd  xmm4,QWORD PTR [rsi+0xf8]
   1f483:	movsd  xmm2,QWORD PTR [rsi+0xf0]
   1f48b:	movapd xmm1,xmm0
   1f48f:	movapd xmm5,xmm3
   1f493:	mulsd  xmm5,xmm4
   1f497:	mulsd  xmm1,xmm2
   1f49b:	mulsd  xmm0,xmm4
   1f49f:	mulsd  xmm2,xmm3
   1f4a3:	subsd  xmm1,xmm5
   1f4a7:	addsd  xmm1,QWORD PTR [rdx]
   1f4ab:	addsd  xmm0,xmm2
   1f4af:	addsd  xmm0,QWORD PTR [rdx+0x8]
   1f4b4:	movsd  QWORD PTR [rdx],xmm1
   1f4b8:	movsd  xmm2,QWORD PTR [rsi+0xf0]
   1f4c0:	movsd  QWORD PTR [rdx+0x8],xmm0
   1f4c5:	movsd  xmm0,QWORD PTR [rdi+0xf10]
   1f4cd:	movsd  xmm3,QWORD PTR [rdi+0xf18]
   1f4d5:	movsd  xmm4,QWORD PTR [rsi+0xf8]
   1f4dd:	movapd xmm1,xmm0
   1f4e1:	movapd xmm5,xmm3
   1f4e5:	mulsd  xmm5,xmm4
   1f4e9:	mulsd  xmm1,xmm2
   1f4ed:	mulsd  xmm0,xmm4
   1f4f1:	mulsd  xmm2,xmm3
   1f4f5:	subsd  xmm1,xmm5
   1f4f9:	addsd  xmm1,QWORD PTR [rdx+0x10]
   1f4fe:	addsd  xmm0,xmm2
   1f502:	addsd  xmm0,QWORD PTR [rdx+0x18]
   1f507:	movsd  QWORD PTR [rdx+0x10],xmm1
   1f50c:	movsd  xmm2,QWORD PTR [rsi+0xf0]
   1f514:	movsd  QWORD PTR [rdx+0x18],xmm0
   1f519:	movsd  xmm0,QWORD PTR [rdi+0xf20]
   1f521:	movsd  xmm3,QWORD PTR [rdi+0xf28]
   1f529:	movsd  xmm4,QWORD PTR [rsi+0xf8]
   1f531:	movapd xmm1,xmm0
   1f535:	movapd xmm5,xmm3
   1f539:	mulsd  xmm5,xmm4
   1f53d:	mulsd  xmm1,xmm2
   1f541:	mulsd  xmm0,xmm4
   1f545:	mulsd  xmm2,xmm3
   1f549:	subsd  xmm1,xmm5
   1f54d:	addsd  xmm1,QWORD PTR [rdx+0x20]
   1f552:	addsd  xmm0,xmm2
   1f556:	addsd  xmm0,QWORD PTR [rdx+0x28]
   1f55b:	movsd  QWORD PTR [rdx+0x20],xmm1
   1f560:	movsd  xmm2,QWORD PTR [rsi+0xf0]
   1f568:	movsd  QWORD PTR [rdx+0x28],xmm0
   1f56d:	movsd  xmm0,QWORD PTR [rdi+0xf30]
   1f575:	movsd  xmm3,QWORD PTR [rdi+0xf38]
   1f57d:	movsd  xmm4,QWORD PTR [rsi+0xf8]
   1f585:	movapd xmm1,xmm0
   1f589:	movapd xmm5,xmm3
   1f58d:	mulsd  xmm5,xmm4
   1f591:	mulsd  xmm1,xmm2
   1f595:	mulsd  xmm0,xmm4
   1f599:	mulsd  xmm2,xmm3
   1f59d:	subsd  xmm1,xmm5
   1f5a1:	addsd  xmm1,QWORD PTR [rdx+0x30]
   1f5a6:	addsd  xmm0,xmm2
   1f5aa:	addsd  xmm0,QWORD PTR [rdx+0x38]
   1f5af:	movsd  QWORD PTR [rdx+0x30],xmm1
   1f5b4:	movsd  QWORD PTR [rdx+0x38],xmm0
   1f5b9:	movsd  xmm0,QWORD PTR [rdi+0xf40]
   1f5c1:	movsd  xmm3,QWORD PTR [rdi+0xf48]
   1f5c9:	movsd  xmm4,QWORD PTR [rsi+0xf8]
   1f5d1:	movsd  xmm2,QWORD PTR [rsi+0xf0]
   1f5d9:	movapd xmm1,xmm0
   1f5dd:	movapd xmm5,xmm3
   1f5e1:	mulsd  xmm5,xmm4
   1f5e5:	mulsd  xmm1,xmm2
   1f5e9:	mulsd  xmm0,xmm4
   1f5ed:	mulsd  xmm2,xmm3
   1f5f1:	subsd  xmm1,xmm5
   1f5f5:	addsd  xmm1,QWORD PTR [rdx+0x40]
   1f5fa:	addsd  xmm0,xmm2
   1f5fe:	addsd  xmm0,QWORD PTR [rdx+0x48]
   1f603:	movsd  QWORD PTR [rdx+0x40],xmm1
   1f608:	movsd  xmm2,QWORD PTR [rsi+0xf0]
   1f610:	movsd  QWORD PTR [rdx+0x48],xmm0
   1f615:	movsd  xmm0,QWORD PTR [rdi+0xf50]
   1f61d:	movsd  xmm3,QWORD PTR [rdi+0xf58]
   1f625:	movsd  xmm4,QWORD PTR [rsi+0xf8]
   1f62d:	movapd xmm1,xmm0
   1f631:	movapd xmm5,xmm3
   1f635:	mulsd  xmm5,xmm4
   1f639:	mulsd  xmm1,xmm2
   1f63d:	mulsd  xmm0,xmm4
   1f641:	mulsd  xmm2,xmm3
   1f645:	subsd  xmm1,xmm5
   1f649:	addsd  xmm1,QWORD PTR [rdx+0x50]
   1f64e:	addsd  xmm0,xmm2
   1f652:	addsd  xmm0,QWORD PTR [rdx+0x58]
   1f657:	movsd  QWORD PTR [rdx+0x50],xmm1
   1f65c:	movsd  xmm2,QWORD PTR [rsi+0xf0]
   1f664:	movsd  QWORD PTR [rdx+0x58],xmm0
   1f669:	movsd  xmm0,QWORD PTR [rdi+0xf60]
   1f671:	movsd  xmm3,QWORD PTR [rdi+0xf68]
   1f679:	movsd  xmm4,QWORD PTR [rsi+0xf8]
   1f681:	movapd xmm1,xmm0
   1f685:	movapd xmm5,xmm3
   1f689:	mulsd  xmm5,xmm4
   1f68d:	mulsd  xmm1,xmm2
   1f691:	mulsd  xmm0,xmm4
   1f695:	mulsd  xmm2,xmm3
   1f699:	subsd  xmm1,xmm5
   1f69d:	addsd  xmm1,QWORD PTR [rdx+0x60]
   1f6a2:	addsd  xmm0,xmm2
   1f6a6:	addsd  xmm0,QWORD PTR [rdx+0x68]
   1f6ab:	movsd  QWORD PTR [rdx+0x60],xmm1
   1f6b0:	movsd  xmm2,QWORD PTR [rsi+0xf0]
   1f6b8:	movsd  QWORD PTR [rdx+0x68],xmm0
   1f6bd:	movsd  xmm0,QWORD PTR [rdi+0xf70]
   1f6c5:	movsd  xmm3,QWORD PTR [rdi+0xf78]
   1f6cd:	movsd  xmm4,QWORD PTR [rsi+0xf8]
   1f6d5:	movapd xmm1,xmm0
   1f6d9:	movapd xmm5,xmm3
   1f6dd:	mulsd  xmm1,xmm2
   1f6e1:	mulsd  xmm5,xmm4
   1f6e5:	mulsd  xmm0,xmm4
   1f6e9:	mulsd  xmm2,xmm3
   1f6ed:	subsd  xmm1,xmm5
   1f6f1:	addsd  xmm1,QWORD PTR [rdx+0x70]
   1f6f6:	addsd  xmm0,xmm2
   1f6fa:	addsd  xmm0,QWORD PTR [rdx+0x78]
   1f6ff:	movsd  QWORD PTR [rdx+0x70],xmm1
   1f704:	movsd  QWORD PTR [rdx+0x78],xmm0
   1f709:	add    rsp,0xd8
   1f710:	jmp    14e40 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm15ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>

Disassembly of section .fini:
