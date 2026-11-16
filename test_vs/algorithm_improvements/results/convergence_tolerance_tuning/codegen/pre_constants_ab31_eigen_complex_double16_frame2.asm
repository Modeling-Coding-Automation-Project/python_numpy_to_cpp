
/tmp/matrix_peak_stack_audit/pre_constants_ab31/pre_constants_ab31_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000001c0c0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeINS0_7ComplexIdEELm16EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS7_XT0_EEERSC_m.part.0>:
   1c0c0:	sub    rsp,0xd8
   1c0c7:	movsd  xmm0,QWORD PTR [rdi]
   1c0cb:	movsd  xmm2,QWORD PTR [rdi+0x8]
   1c0d0:	movsd  xmm3,QWORD PTR [rsi+0x8]
   1c0d5:	movsd  xmm1,QWORD PTR [rsi]
   1c0d9:	movapd xmm10,xmm0
   1c0de:	movapd xmm4,xmm2
   1c0e2:	mulsd  xmm10,xmm1
   1c0e7:	mulsd  xmm0,xmm3
   1c0eb:	mulsd  xmm1,xmm2
   1c0ef:	mulsd  xmm4,xmm3
   1c0f3:	addsd  xmm0,xmm1
   1c0f7:	subsd  xmm10,xmm4
   1c0fc:	movsd  xmm4,QWORD PTR [rdx+0x8]
   1c101:	addsd  xmm10,QWORD PTR [rdx]
   1c106:	addsd  xmm4,xmm0
   1c10a:	movsd  QWORD PTR [rdx],xmm10
   1c10f:	movsd  xmm0,QWORD PTR [rdi+0x10]
   1c114:	movsd  xmm1,QWORD PTR [rsi]
   1c118:	movsd  QWORD PTR [rdx+0x8],xmm4
   1c11d:	movsd  xmm2,QWORD PTR [rdi+0x18]
   1c122:	movsd  xmm3,QWORD PTR [rsi+0x8]
   1c127:	movapd xmm8,xmm0
   1c12c:	mulsd  xmm8,xmm1
   1c131:	mulsd  xmm0,xmm3
   1c135:	movapd xmm5,xmm2
   1c139:	mulsd  xmm1,xmm2
   1c13d:	mulsd  xmm5,xmm3
   1c141:	movsd  xmm3,QWORD PTR [rdx+0x18]
   1c146:	addsd  xmm0,xmm1
   1c14a:	subsd  xmm8,xmm5
   1c14f:	addsd  xmm8,QWORD PTR [rdx+0x10]
   1c155:	addsd  xmm3,xmm0
   1c159:	movsd  QWORD PTR [rdx+0x10],xmm8
   1c15f:	movsd  xmm0,QWORD PTR [rdi+0x20]
   1c164:	movsd  xmm1,QWORD PTR [rsi]
   1c168:	movsd  QWORD PTR [rdx+0x18],xmm3
   1c16d:	movsd  xmm2,QWORD PTR [rdi+0x28]
   1c172:	movsd  xmm5,QWORD PTR [rsi+0x8]
   1c177:	movapd xmm7,xmm0
   1c17b:	mulsd  xmm7,xmm1
   1c17f:	mulsd  xmm0,xmm5
   1c183:	movapd xmm6,xmm2
   1c187:	mulsd  xmm1,xmm2
   1c18b:	movsd  xmm2,QWORD PTR [rdx+0x28]
   1c190:	mulsd  xmm6,xmm5
   1c194:	addsd  xmm0,xmm1
   1c198:	subsd  xmm7,xmm6
   1c19c:	addsd  xmm7,QWORD PTR [rdx+0x20]
   1c1a1:	addsd  xmm2,xmm0
   1c1a5:	movsd  QWORD PTR [rdx+0x20],xmm7
   1c1aa:	movsd  xmm0,QWORD PTR [rdi+0x30]
   1c1af:	movsd  xmm1,QWORD PTR [rsi]
   1c1b3:	movsd  QWORD PTR [rdx+0x28],xmm2
   1c1b8:	movsd  xmm5,QWORD PTR [rdi+0x38]
   1c1bd:	movsd  xmm9,QWORD PTR [rsi+0x8]
   1c1c3:	movapd xmm6,xmm0
   1c1c7:	mulsd  xmm6,xmm1
   1c1cb:	mulsd  xmm0,xmm9
   1c1d0:	movapd xmm11,xmm5
   1c1d5:	mulsd  xmm11,xmm9
   1c1da:	mulsd  xmm1,xmm5
   1c1de:	subsd  xmm6,xmm11
   1c1e3:	addsd  xmm6,QWORD PTR [rdx+0x30]
   1c1e8:	addsd  xmm1,xmm0
   1c1ec:	movsd  QWORD PTR [rdx+0x30],xmm6
   1c1f1:	addsd  xmm1,QWORD PTR [rdx+0x38]
   1c1f6:	movsd  xmm9,QWORD PTR [rdi+0x40]
   1c1fc:	movsd  xmm0,QWORD PTR [rsi]
   1c200:	movapd xmm5,xmm9
   1c205:	movsd  QWORD PTR [rdx+0x38],xmm1
   1c20a:	movsd  xmm11,QWORD PTR [rdi+0x48]
   1c210:	movsd  xmm12,QWORD PTR [rsi+0x8]
   1c216:	mulsd  xmm5,xmm0
   1c21a:	mulsd  xmm9,xmm12
   1c21f:	movapd xmm13,xmm11
   1c224:	mulsd  xmm13,xmm12
   1c229:	mulsd  xmm0,xmm11
   1c22e:	subsd  xmm5,xmm13
   1c233:	addsd  xmm5,QWORD PTR [rdx+0x40]
   1c238:	addsd  xmm0,xmm9
   1c23d:	addsd  xmm0,QWORD PTR [rdx+0x48]
   1c242:	movsd  QWORD PTR [rdx+0x40],xmm5
   1c247:	movsd  xmm9,QWORD PTR [rdi+0x50]
   1c24d:	movsd  xmm11,QWORD PTR [rsi]
   1c252:	movsd  QWORD PTR [rdx+0x48],xmm0
   1c257:	movsd  xmm13,QWORD PTR [rdi+0x58]
   1c25d:	movsd  xmm14,QWORD PTR [rsi+0x8]
   1c263:	movapd xmm12,xmm9
   1c268:	mulsd  xmm12,xmm11
   1c26d:	movapd xmm15,xmm13
   1c272:	mulsd  xmm15,xmm14
   1c277:	mulsd  xmm11,xmm13
   1c27c:	mulsd  xmm9,xmm14
   1c281:	subsd  xmm12,xmm15
   1c286:	addsd  xmm12,QWORD PTR [rdx+0x50]
   1c28c:	addsd  xmm9,xmm11
   1c291:	addsd  xmm9,QWORD PTR [rdx+0x58]
   1c297:	movsd  QWORD PTR [rdx+0x50],xmm12
   1c29d:	movsd  xmm13,QWORD PTR [rdi+0x60]
   1c2a3:	movsd  xmm11,QWORD PTR [rsi]
   1c2a8:	movsd  QWORD PTR [rsp],xmm12
   1c2ae:	movsd  QWORD PTR [rdx+0x58],xmm9
   1c2b4:	movsd  xmm12,QWORD PTR [rdi+0x68]
   1c2ba:	movapd xmm15,xmm13
   1c2bf:	movsd  QWORD PTR [rsp+0x8],xmm9
   1c2c6:	movsd  xmm9,QWORD PTR [rsi+0x8]
   1c2cc:	mulsd  xmm15,xmm11
   1c2d1:	mulsd  xmm11,xmm12
   1c2d6:	movapd xmm14,xmm12
   1c2db:	mulsd  xmm14,xmm9
   1c2e0:	mulsd  xmm9,xmm13
   1c2e5:	subsd  xmm15,xmm14
   1c2ea:	movsd  xmm14,QWORD PTR [rdx+0x68]
   1c2f0:	addsd  xmm15,QWORD PTR [rdx+0x60]
   1c2f6:	addsd  xmm9,xmm11
   1c2fb:	movsd  QWORD PTR [rdx+0x60],xmm15
   1c301:	movsd  xmm11,QWORD PTR [rsi]
   1c306:	addsd  xmm14,xmm9
   1c30b:	movsd  QWORD PTR [rsp+0x10],xmm15
   1c312:	movsd  xmm9,QWORD PTR [rdi+0x70]
   1c318:	movapd xmm15,xmm9
   1c31d:	movsd  QWORD PTR [rsp+0x18],xmm14
   1c324:	mulsd  xmm15,xmm11
   1c329:	movsd  QWORD PTR [rdx+0x68],xmm14
   1c32f:	movsd  xmm12,QWORD PTR [rdi+0x78]
   1c335:	movsd  xmm13,QWORD PTR [rsi+0x8]
   1c33b:	mulsd  xmm11,xmm12
   1c340:	movapd xmm14,xmm12
   1c345:	mulsd  xmm14,xmm13
   1c34a:	mulsd  xmm9,xmm13
   1c34f:	movsd  xmm13,QWORD PTR [rdx+0x78]
   1c355:	subsd  xmm15,xmm14
   1c35a:	addsd  xmm15,QWORD PTR [rdx+0x70]
   1c360:	addsd  xmm9,xmm11
   1c365:	movsd  QWORD PTR [rdx+0x70],xmm15
   1c36b:	movsd  xmm11,QWORD PTR [rsi]
   1c370:	addsd  xmm13,xmm9
   1c375:	movsd  QWORD PTR [rsp+0x20],xmm15
   1c37c:	movsd  xmm9,QWORD PTR [rdi+0x80]
   1c385:	movapd xmm14,xmm9
   1c38a:	mulsd  xmm14,xmm11
   1c38f:	movsd  QWORD PTR [rdx+0x78],xmm13
   1c395:	movsd  xmm12,QWORD PTR [rdi+0x88]
   1c39e:	movsd  QWORD PTR [rsp+0x28],xmm13
   1c3a5:	movsd  xmm13,QWORD PTR [rsi+0x8]
   1c3ab:	mulsd  xmm11,xmm12
   1c3b0:	mulsd  xmm9,xmm13
   1c3b5:	movapd xmm15,xmm14
   1c3ba:	movapd xmm14,xmm12
   1c3bf:	mulsd  xmm14,xmm13
   1c3c4:	addsd  xmm9,xmm11
   1c3c9:	movsd  xmm11,QWORD PTR [rdx+0x88]
   1c3d2:	subsd  xmm15,xmm14
   1c3d7:	addsd  xmm11,xmm9
   1c3dc:	movsd  xmm14,QWORD PTR [rdx+0x80]
   1c3e5:	addsd  xmm14,xmm15
   1c3ea:	movsd  QWORD PTR [rsp+0x38],xmm11
   1c3f1:	movsd  QWORD PTR [rdx+0x88],xmm11
   1c3fa:	movsd  xmm13,QWORD PTR [rsi+0x8]
   1c400:	movsd  xmm12,QWORD PTR [rdi+0x98]
   1c409:	movsd  QWORD PTR [rdx+0x80],xmm14
   1c412:	movsd  xmm11,QWORD PTR [rsi]
   1c417:	movsd  xmm9,QWORD PTR [rdi+0x90]
   1c420:	movsd  QWORD PTR [rsp+0x30],xmm14
   1c427:	movapd xmm14,xmm12
   1c42c:	mulsd  xmm14,xmm13
   1c431:	movapd xmm15,xmm9
   1c436:	mulsd  xmm15,xmm11
   1c43b:	mulsd  xmm9,xmm13
   1c440:	mulsd  xmm11,xmm12
   1c445:	subsd  xmm15,xmm14
   1c44a:	addsd  xmm15,QWORD PTR [rdx+0x90]
   1c453:	addsd  xmm9,xmm11
   1c458:	addsd  xmm9,QWORD PTR [rdx+0x98]
   1c461:	movsd  QWORD PTR [rdx+0x90],xmm15
   1c46a:	movsd  xmm11,QWORD PTR [rsi]
   1c46f:	movsd  QWORD PTR [rsp+0x40],xmm15
   1c476:	movsd  QWORD PTR [rdx+0x98],xmm9
   1c47f:	movsd  xmm13,QWORD PTR [rsi+0x8]
   1c485:	movsd  xmm12,QWORD PTR [rdi+0xa8]
   1c48e:	movsd  QWORD PTR [rsp+0x48],xmm9
   1c495:	movsd  xmm9,QWORD PTR [rdi+0xa0]
   1c49e:	movapd xmm14,xmm9
   1c4a3:	mulsd  xmm14,xmm11
   1c4a8:	mulsd  xmm9,xmm13
   1c4ad:	mulsd  xmm11,xmm12
   1c4b2:	movapd xmm15,xmm14
   1c4b7:	movapd xmm14,xmm12
   1c4bc:	mulsd  xmm14,xmm13
   1c4c1:	addsd  xmm9,xmm11
   1c4c6:	subsd  xmm15,xmm14
   1c4cb:	movsd  xmm14,QWORD PTR [rdx+0xa0]
   1c4d4:	movsd  xmm13,QWORD PTR [rdx+0xa8]
   1c4dd:	addsd  xmm13,xmm9
   1c4e2:	addsd  xmm14,xmm15
   1c4e7:	movsd  QWORD PTR [rdx+0xa8],xmm13
   1c4f0:	movsd  xmm12,QWORD PTR [rdi+0xb8]
   1c4f9:	movsd  QWORD PTR [rdx+0xa0],xmm14
   1c502:	movsd  xmm11,QWORD PTR [rsi]
   1c507:	movsd  xmm9,QWORD PTR [rdi+0xb0]
   1c510:	movsd  QWORD PTR [rsp+0x58],xmm13
   1c517:	movsd  xmm13,QWORD PTR [rsi+0x8]
   1c51d:	movsd  QWORD PTR [rsp+0x50],xmm14
   1c524:	movapd xmm15,xmm9
   1c529:	movapd xmm14,xmm12
   1c52e:	mulsd  xmm14,xmm13
   1c533:	mulsd  xmm15,xmm11
   1c538:	mulsd  xmm9,xmm13
   1c53d:	mulsd  xmm11,xmm12
   1c542:	movsd  xmm12,QWORD PTR [rdx+0xb8]
   1c54b:	subsd  xmm15,xmm14
   1c550:	addsd  xmm15,QWORD PTR [rdx+0xb0]
   1c559:	addsd  xmm9,xmm11
   1c55e:	movsd  QWORD PTR [rdx+0xb0],xmm15
   1c567:	movsd  xmm11,QWORD PTR [rsi]
   1c56c:	movsd  QWORD PTR [rsp+0x60],xmm15
   1c573:	addsd  xmm12,xmm9
   1c578:	movsd  xmm9,QWORD PTR [rdi+0xc0]
   1c581:	movapd xmm14,xmm9
   1c586:	mulsd  xmm14,xmm11
   1c58b:	movsd  QWORD PTR [rsp+0x68],xmm12
   1c592:	movsd  QWORD PTR [rdx+0xb8],xmm12
   1c59b:	movsd  xmm13,QWORD PTR [rsi+0x8]
   1c5a1:	movsd  xmm12,QWORD PTR [rdi+0xc8]
   1c5aa:	mulsd  xmm9,xmm13
   1c5af:	mulsd  xmm11,xmm12
   1c5b4:	movapd xmm15,xmm14
   1c5b9:	movapd xmm14,xmm12
   1c5be:	mulsd  xmm14,xmm13
   1c5c3:	addsd  xmm9,xmm11
   1c5c8:	movsd  xmm11,QWORD PTR [rdx+0xc8]
   1c5d1:	subsd  xmm15,xmm14
   1c5d6:	movsd  xmm14,QWORD PTR [rdx+0xc0]
   1c5df:	addsd  xmm11,xmm9
   1c5e4:	addsd  xmm14,xmm15
   1c5e9:	movsd  QWORD PTR [rdx+0xc8],xmm11
   1c5f2:	movsd  xmm13,QWORD PTR [rsi+0x8]
   1c5f8:	movsd  xmm12,QWORD PTR [rdi+0xd8]
   1c601:	movsd  QWORD PTR [rsp+0x78],xmm11
   1c608:	movsd  QWORD PTR [rdx+0xc0],xmm14
   1c611:	movsd  xmm11,QWORD PTR [rsi]
   1c616:	movsd  xmm9,QWORD PTR [rdi+0xd0]
   1c61f:	movsd  QWORD PTR [rsp+0x70],xmm14
   1c626:	movapd xmm14,xmm12
   1c62b:	mulsd  xmm14,xmm13
   1c630:	movapd xmm15,xmm9
   1c635:	mulsd  xmm15,xmm11
   1c63a:	mulsd  xmm9,xmm13
   1c63f:	mulsd  xmm11,xmm12
   1c644:	subsd  xmm15,xmm14
   1c649:	addsd  xmm15,QWORD PTR [rdx+0xd0]
   1c652:	addsd  xmm9,xmm11
   1c657:	movsd  QWORD PTR [rsp+0x80],xmm15
   1c661:	movsd  QWORD PTR [rdx+0xd0],xmm15
   1c66a:	movsd  xmm11,QWORD PTR [rsi]
   1c66f:	addsd  xmm9,QWORD PTR [rdx+0xd8]
   1c678:	movsd  QWORD PTR [rdx+0xd8],xmm9
   1c681:	movsd  xmm13,QWORD PTR [rsi+0x8]
   1c687:	movsd  xmm12,QWORD PTR [rdi+0xe8]
   1c690:	movsd  QWORD PTR [rsp+0x88],xmm9
   1c69a:	movsd  xmm9,QWORD PTR [rdi+0xe0]
   1c6a3:	movapd xmm14,xmm9
   1c6a8:	mulsd  xmm14,xmm11
   1c6ad:	mulsd  xmm9,xmm13
   1c6b2:	mulsd  xmm11,xmm12
   1c6b7:	movapd xmm15,xmm14
   1c6bc:	movapd xmm14,xmm12
   1c6c1:	mulsd  xmm14,xmm13
   1c6c6:	movsd  xmm13,QWORD PTR [rdx+0xe8]
   1c6cf:	addsd  xmm9,xmm11
   1c6d4:	addsd  xmm13,xmm9
   1c6d9:	subsd  xmm15,xmm14
   1c6de:	movsd  xmm14,QWORD PTR [rdx+0xe0]
   1c6e7:	movsd  QWORD PTR [rdx+0xe8],xmm13
   1c6f0:	movsd  xmm12,QWORD PTR [rdi+0xf8]
   1c6f9:	movsd  QWORD PTR [rsp+0x98],xmm13
   1c703:	addsd  xmm14,xmm15
   1c708:	movsd  xmm13,QWORD PTR [rsi+0x8]
   1c70e:	movsd  QWORD PTR [rdx+0xe0],xmm14
   1c717:	movsd  xmm11,QWORD PTR [rsi]
   1c71c:	movsd  xmm9,QWORD PTR [rdi+0xf0]
   1c725:	movsd  QWORD PTR [rsp+0x90],xmm14
   1c72f:	movapd xmm14,xmm12
   1c734:	mulsd  xmm14,xmm13
   1c739:	movapd xmm15,xmm9
   1c73e:	mulsd  xmm15,xmm11
   1c743:	mulsd  xmm9,xmm13
   1c748:	mulsd  xmm11,xmm12
   1c74d:	movsd  xmm12,QWORD PTR [rdx+0xf8]
   1c756:	subsd  xmm15,xmm14
   1c75b:	addsd  xmm15,QWORD PTR [rdx+0xf0]
   1c764:	addsd  xmm9,xmm11
   1c769:	movsd  QWORD PTR [rdx+0xf0],xmm15
   1c772:	movsd  xmm11,QWORD PTR [rsi+0x10]
   1c778:	movsd  QWORD PTR [rsp+0xa0],xmm15
   1c782:	addsd  xmm12,xmm9
   1c787:	movsd  xmm9,QWORD PTR [rdi+0x100]
   1c790:	movsd  QWORD PTR [rdx+0xf8],xmm12
   1c799:	movsd  xmm14,QWORD PTR [rsi+0x18]
   1c79f:	movsd  xmm13,QWORD PTR [rdi+0x108]
   1c7a8:	movsd  QWORD PTR [rsp+0xa8],xmm12
   1c7b2:	movapd xmm12,xmm9
   1c7b7:	mulsd  xmm12,xmm11
   1c7bc:	movapd xmm15,xmm13
   1c7c1:	mulsd  xmm15,xmm14
   1c7c6:	mulsd  xmm11,xmm13
   1c7cb:	mulsd  xmm9,xmm14
   1c7d0:	subsd  xmm12,xmm15
   1c7d5:	addsd  xmm9,xmm11
   1c7da:	addsd  xmm12,xmm10
   1c7df:	movapd xmm10,xmm9
   1c7e4:	addsd  xmm10,xmm4
   1c7e9:	movsd  QWORD PTR [rdx],xmm12
   1c7ee:	movapd xmm15,xmm12
   1c7f3:	movsd  xmm4,QWORD PTR [rdi+0x110]
   1c7fb:	movsd  xmm9,QWORD PTR [rsi+0x10]
   1c801:	movsd  QWORD PTR [rdx+0x8],xmm10
   1c807:	movapd xmm14,xmm10
   1c80c:	movapd xmm10,xmm4
   1c811:	movsd  xmm11,QWORD PTR [rdi+0x118]
   1c81a:	movsd  xmm12,QWORD PTR [rsi+0x18]
   1c820:	mulsd  xmm10,xmm9
   1c825:	mulsd  xmm9,xmm11
   1c82a:	movapd xmm13,xmm11
   1c82f:	mulsd  xmm13,xmm12
   1c834:	mulsd  xmm4,xmm12
   1c839:	subsd  xmm10,xmm13
   1c83e:	addsd  xmm4,xmm9
   1c843:	addsd  xmm10,xmm8
   1c848:	addsd  xmm4,xmm3
   1c84c:	movsd  QWORD PTR [rdx+0x10],xmm10
   1c852:	movsd  xmm3,QWORD PTR [rdi+0x120]
   1c85a:	movapd xmm13,xmm10
   1c85f:	movsd  QWORD PTR [rdx+0x18],xmm4
   1c864:	movapd xmm12,xmm4
   1c869:	movsd  xmm10,QWORD PTR [rsi+0x18]
   1c86f:	movsd  xmm9,QWORD PTR [rdi+0x128]
   1c878:	movsd  xmm4,QWORD PTR [rsi+0x10]
   1c87d:	movapd xmm8,xmm3
   1c882:	mulsd  xmm3,xmm10
   1c887:	movapd xmm11,xmm9
   1c88c:	mulsd  xmm11,xmm10
   1c891:	mulsd  xmm8,xmm4
   1c896:	mulsd  xmm4,xmm9
   1c89b:	subsd  xmm8,xmm11
   1c8a0:	addsd  xmm3,xmm4
   1c8a4:	addsd  xmm8,xmm7
   1c8a9:	addsd  xmm3,xmm2
   1c8ad:	movsd  QWORD PTR [rdx+0x20],xmm8
   1c8b3:	movsd  xmm2,QWORD PTR [rdi+0x130]
   1c8bb:	movapd xmm11,xmm8
   1c8c0:	movsd  QWORD PTR [rdx+0x28],xmm3
   1c8c5:	movapd xmm10,xmm3
   1c8ca:	movsd  xmm3,QWORD PTR [rsi+0x10]
   1c8cf:	movsd  xmm7,QWORD PTR [rdi+0x138]
   1c8d7:	movsd  xmm8,QWORD PTR [rsi+0x18]
   1c8dd:	movapd xmm4,xmm2
   1c8e1:	mulsd  xmm4,xmm3
   1c8e5:	movapd xmm9,xmm7
   1c8ea:	mulsd  xmm9,xmm8
   1c8ef:	mulsd  xmm3,xmm7
   1c8f3:	mulsd  xmm2,xmm8
   1c8f8:	subsd  xmm4,xmm9
   1c8fd:	addsd  xmm2,xmm3
   1c901:	addsd  xmm4,xmm6
   1c905:	addsd  xmm2,xmm1
   1c909:	movsd  QWORD PTR [rdx+0x30],xmm4
   1c90e:	movapd xmm9,xmm4
   1c913:	movsd  xmm1,QWORD PTR [rdi+0x140]
   1c91b:	movsd  QWORD PTR [rdx+0x38],xmm2
   1c920:	movapd xmm8,xmm2
   1c925:	movsd  xmm6,QWORD PTR [rsi+0x18]
   1c92a:	movsd  xmm4,QWORD PTR [rdi+0x148]
   1c932:	movsd  xmm2,QWORD PTR [rsi+0x10]
   1c937:	movapd xmm3,xmm1
   1c93b:	mulsd  xmm1,xmm6
   1c93f:	movapd xmm7,xmm4
   1c943:	mulsd  xmm7,xmm6
   1c947:	mulsd  xmm3,xmm2
   1c94b:	mulsd  xmm2,xmm4
   1c94f:	subsd  xmm3,xmm7
   1c953:	addsd  xmm1,xmm2
   1c957:	movapd xmm7,xmm3
   1c95b:	movapd xmm6,xmm1
   1c95f:	addsd  xmm7,xmm5
   1c963:	addsd  xmm6,xmm0
   1c967:	movsd  QWORD PTR [rdx+0x40],xmm7
   1c96c:	movsd  xmm0,QWORD PTR [rdi+0x150]
   1c974:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1c979:	movsd  QWORD PTR [rdx+0x48],xmm6
   1c97e:	movsd  xmm3,QWORD PTR [rdi+0x158]
   1c986:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1c98b:	movapd xmm2,xmm0
   1c98f:	mulsd  xmm2,xmm1
   1c993:	movapd xmm5,xmm3
   1c997:	mulsd  xmm5,xmm4
   1c99b:	mulsd  xmm1,xmm3
   1c99f:	mulsd  xmm0,xmm4
   1c9a3:	subsd  xmm2,xmm5
   1c9a7:	addsd  xmm2,QWORD PTR [rsp]
   1c9ac:	addsd  xmm0,xmm1
   1c9b0:	addsd  xmm0,QWORD PTR [rsp+0x8]
   1c9b6:	movsd  QWORD PTR [rdx+0x50],xmm2
   1c9bb:	movsd  QWORD PTR [rsp],xmm2
   1c9c0:	movsd  QWORD PTR [rsp+0x8],xmm0
   1c9c6:	movsd  QWORD PTR [rdx+0x58],xmm0
   1c9cb:	movsd  xmm0,QWORD PTR [rdi+0x160]
   1c9d3:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1c9d8:	movsd  xmm3,QWORD PTR [rdi+0x168]
   1c9e0:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1c9e5:	movapd xmm2,xmm0
   1c9e9:	mulsd  xmm2,xmm1
   1c9ed:	movapd xmm5,xmm3
   1c9f1:	mulsd  xmm1,xmm3
   1c9f5:	mulsd  xmm0,xmm4
   1c9f9:	mulsd  xmm5,xmm4
   1c9fd:	movsd  xmm4,QWORD PTR [rsp+0x18]
   1ca03:	addsd  xmm0,xmm1
   1ca07:	subsd  xmm2,xmm5
   1ca0b:	addsd  xmm2,QWORD PTR [rsp+0x10]
   1ca11:	addsd  xmm4,xmm0
   1ca15:	movsd  QWORD PTR [rdx+0x60],xmm2
   1ca1a:	movsd  xmm0,QWORD PTR [rdi+0x170]
   1ca22:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1ca27:	movsd  QWORD PTR [rsp+0x10],xmm2
   1ca2d:	movsd  QWORD PTR [rdx+0x68],xmm4
   1ca32:	movsd  xmm3,QWORD PTR [rdi+0x178]
   1ca3a:	movapd xmm2,xmm0
   1ca3e:	movsd  QWORD PTR [rsp+0x18],xmm4
   1ca44:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1ca49:	mulsd  xmm2,xmm1
   1ca4d:	mulsd  xmm1,xmm3
   1ca51:	movapd xmm5,xmm3
   1ca55:	movsd  xmm3,QWORD PTR [rsp+0x28]
   1ca5b:	mulsd  xmm0,xmm4
   1ca5f:	mulsd  xmm5,xmm4
   1ca63:	addsd  xmm0,xmm1
   1ca67:	subsd  xmm2,xmm5
   1ca6b:	addsd  xmm2,QWORD PTR [rsp+0x20]
   1ca71:	addsd  xmm3,xmm0
   1ca75:	movsd  QWORD PTR [rdx+0x70],xmm2
   1ca7a:	movsd  xmm0,QWORD PTR [rdi+0x180]
   1ca82:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1ca87:	movsd  QWORD PTR [rsp+0x20],xmm2
   1ca8d:	movsd  QWORD PTR [rdx+0x78],xmm3
   1ca92:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1ca97:	movapd xmm2,xmm0
   1ca9b:	movsd  QWORD PTR [rsp+0x28],xmm3
   1caa1:	movsd  xmm3,QWORD PTR [rdi+0x188]
   1caa9:	mulsd  xmm2,xmm1
   1caad:	mulsd  xmm0,xmm4
   1cab1:	mulsd  xmm1,xmm3
   1cab5:	movapd xmm5,xmm3
   1cab9:	mulsd  xmm5,xmm4
   1cabd:	addsd  xmm0,xmm1
   1cac1:	movsd  xmm1,QWORD PTR [rsp+0x38]
   1cac7:	subsd  xmm2,xmm5
   1cacb:	addsd  xmm2,QWORD PTR [rsp+0x30]
   1cad1:	addsd  xmm1,xmm0
   1cad5:	movsd  QWORD PTR [rdx+0x80],xmm2
   1cadd:	movsd  xmm0,QWORD PTR [rdi+0x190]
   1cae5:	movsd  QWORD PTR [rsp+0x30],xmm2
   1caeb:	movsd  QWORD PTR [rdx+0x88],xmm1
   1caf3:	movsd  xmm3,QWORD PTR [rdi+0x198]
   1cafb:	movapd xmm2,xmm0
   1caff:	movsd  QWORD PTR [rsp+0x38],xmm1
   1cb05:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1cb0a:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1cb0f:	movapd xmm5,xmm3
   1cb13:	mulsd  xmm5,xmm4
   1cb17:	mulsd  xmm2,xmm1
   1cb1b:	mulsd  xmm0,xmm4
   1cb1f:	mulsd  xmm1,xmm3
   1cb23:	subsd  xmm2,xmm5
   1cb27:	addsd  xmm2,QWORD PTR [rsp+0x40]
   1cb2d:	addsd  xmm0,xmm1
   1cb31:	addsd  xmm0,QWORD PTR [rsp+0x48]
   1cb37:	movsd  QWORD PTR [rdx+0x90],xmm2
   1cb3f:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1cb44:	movsd  QWORD PTR [rsp+0x40],xmm2
   1cb4a:	movsd  QWORD PTR [rdx+0x98],xmm0
   1cb52:	movsd  xmm3,QWORD PTR [rdi+0x1a8]
   1cb5a:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1cb5f:	movsd  QWORD PTR [rsp+0x48],xmm0
   1cb65:	movsd  xmm0,QWORD PTR [rdi+0x1a0]
   1cb6d:	movapd xmm5,xmm3
   1cb71:	mulsd  xmm5,xmm4
   1cb75:	movapd xmm2,xmm0
   1cb79:	mulsd  xmm2,xmm1
   1cb7d:	mulsd  xmm0,xmm4
   1cb81:	movsd  xmm4,QWORD PTR [rsp+0x58]
   1cb87:	mulsd  xmm1,xmm3
   1cb8b:	subsd  xmm2,xmm5
   1cb8f:	addsd  xmm2,QWORD PTR [rsp+0x50]
   1cb95:	addsd  xmm0,xmm1
   1cb99:	movsd  QWORD PTR [rdx+0xa0],xmm2
   1cba1:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1cba6:	movsd  QWORD PTR [rsp+0x50],xmm2
   1cbac:	addsd  xmm4,xmm0
   1cbb0:	movsd  xmm0,QWORD PTR [rdi+0x1b0]
   1cbb8:	movapd xmm2,xmm0
   1cbbc:	movsd  QWORD PTR [rdx+0xa8],xmm4
   1cbc4:	movsd  xmm3,QWORD PTR [rdi+0x1b8]
   1cbcc:	mulsd  xmm2,xmm1
   1cbd0:	movsd  QWORD PTR [rsp+0x58],xmm4
   1cbd6:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1cbdb:	mulsd  xmm1,xmm3
   1cbdf:	movapd xmm5,xmm3
   1cbe3:	movsd  xmm3,QWORD PTR [rsp+0x68]
   1cbe9:	mulsd  xmm0,xmm4
   1cbed:	mulsd  xmm5,xmm4
   1cbf1:	addsd  xmm0,xmm1
   1cbf5:	subsd  xmm2,xmm5
   1cbf9:	addsd  xmm2,QWORD PTR [rsp+0x60]
   1cbff:	addsd  xmm3,xmm0
   1cc03:	movsd  QWORD PTR [rdx+0xb0],xmm2
   1cc0b:	movsd  xmm0,QWORD PTR [rdi+0x1c0]
   1cc13:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1cc18:	movsd  QWORD PTR [rsp+0x60],xmm2
   1cc1e:	movsd  QWORD PTR [rsp+0x68],xmm3
   1cc24:	movapd xmm2,xmm0
   1cc28:	movsd  QWORD PTR [rdx+0xb8],xmm3
   1cc30:	movsd  xmm3,QWORD PTR [rdi+0x1c8]
   1cc38:	mulsd  xmm2,xmm1
   1cc3c:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1cc41:	mulsd  xmm1,xmm3
   1cc45:	movapd xmm5,xmm3
   1cc49:	mulsd  xmm5,xmm4
   1cc4d:	mulsd  xmm0,xmm4
   1cc51:	subsd  xmm2,xmm5
   1cc55:	addsd  xmm2,QWORD PTR [rsp+0x70]
   1cc5b:	addsd  xmm0,xmm1
   1cc5f:	movsd  QWORD PTR [rsp+0x70],xmm2
   1cc65:	movsd  xmm1,QWORD PTR [rsp+0x78]
   1cc6b:	movsd  QWORD PTR [rdx+0xc0],xmm2
   1cc73:	addsd  xmm1,xmm0
   1cc77:	movsd  xmm0,QWORD PTR [rdi+0x1d0]
   1cc7f:	movapd xmm2,xmm0
   1cc83:	movsd  QWORD PTR [rdx+0xc8],xmm1
   1cc8b:	movsd  xmm3,QWORD PTR [rdi+0x1d8]
   1cc93:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1cc98:	movsd  QWORD PTR [rsp+0x78],xmm1
   1cc9e:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1cca3:	mulsd  xmm0,xmm4
   1cca7:	movapd xmm5,xmm3
   1ccab:	mulsd  xmm5,xmm4
   1ccaf:	mulsd  xmm2,xmm1
   1ccb3:	mulsd  xmm1,xmm3
   1ccb7:	subsd  xmm2,xmm5
   1ccbb:	addsd  xmm2,QWORD PTR [rsp+0x80]
   1ccc4:	addsd  xmm0,xmm1
   1ccc8:	addsd  xmm0,QWORD PTR [rsp+0x88]
   1ccd1:	movsd  QWORD PTR [rdx+0xd0],xmm2
   1ccd9:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1ccde:	movsd  QWORD PTR [rdx+0xd8],xmm0
   1cce6:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1cceb:	movsd  xmm3,QWORD PTR [rdi+0x1e8]
   1ccf3:	movsd  QWORD PTR [rsp+0x88],xmm0
   1ccfc:	movsd  xmm0,QWORD PTR [rdi+0x1e0]
   1cd04:	movsd  QWORD PTR [rsp+0x80],xmm2
   1cd0d:	movapd xmm5,xmm3
   1cd11:	mulsd  xmm5,xmm4
   1cd15:	movapd xmm2,xmm0
   1cd19:	mulsd  xmm2,xmm1
   1cd1d:	mulsd  xmm0,xmm4
   1cd21:	mulsd  xmm1,xmm3
   1cd25:	subsd  xmm2,xmm5
   1cd29:	addsd  xmm2,QWORD PTR [rsp+0x90]
   1cd32:	addsd  xmm0,xmm1
   1cd36:	addsd  xmm0,QWORD PTR [rsp+0x98]
   1cd3f:	movsd  QWORD PTR [rdx+0xe0],xmm2
   1cd47:	movsd  xmm1,QWORD PTR [rsi+0x10]
   1cd4c:	movsd  QWORD PTR [rsp+0x90],xmm2
   1cd55:	movsd  QWORD PTR [rdx+0xe8],xmm0
   1cd5d:	movsd  xmm4,QWORD PTR [rsi+0x18]
   1cd62:	movsd  xmm3,QWORD PTR [rdi+0x1f8]
   1cd6a:	movsd  QWORD PTR [rsp+0x98],xmm0
   1cd73:	movsd  xmm0,QWORD PTR [rdi+0x1f0]
   1cd7b:	movapd xmm5,xmm0
   1cd7f:	mulsd  xmm5,xmm1
   1cd83:	mulsd  xmm0,xmm4
   1cd87:	mulsd  xmm1,xmm3
   1cd8b:	movapd xmm2,xmm5
   1cd8f:	movapd xmm5,xmm3
   1cd93:	movsd  xmm3,QWORD PTR [rsp+0xa8]
   1cd9c:	mulsd  xmm5,xmm4
   1cda0:	addsd  xmm0,xmm1
   1cda4:	addsd  xmm3,xmm0
   1cda8:	subsd  xmm2,xmm5
   1cdac:	addsd  xmm2,QWORD PTR [rsp+0xa0]
   1cdb5:	movsd  QWORD PTR [rsp+0xa0],xmm2
   1cdbe:	movsd  QWORD PTR [rdx+0xf0],xmm2
   1cdc6:	movsd  QWORD PTR [rsp+0xa8],xmm3
   1cdcf:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1cdd4:	movsd  xmm0,QWORD PTR [rdi+0x200]
   1cddc:	movsd  QWORD PTR [rdx+0xf8],xmm3
   1cde4:	movsd  xmm3,QWORD PTR [rdi+0x208]
   1cdec:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1cdf1:	movapd xmm5,xmm0
   1cdf5:	mulsd  xmm5,xmm1
   1cdf9:	mulsd  xmm0,xmm4
   1cdfd:	mulsd  xmm1,xmm3
   1ce01:	movapd xmm2,xmm5
   1ce05:	movapd xmm5,xmm3
   1ce09:	mulsd  xmm5,xmm4
   1ce0d:	addsd  xmm0,xmm1
   1ce11:	addsd  xmm14,xmm0
   1ce16:	subsd  xmm2,xmm5
   1ce1a:	movsd  QWORD PTR [rdx+0x8],xmm14
   1ce20:	movsd  xmm3,QWORD PTR [rdi+0x218]
   1ce28:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1ce2d:	addsd  xmm15,xmm2
   1ce32:	movsd  QWORD PTR [rdx],xmm15
   1ce37:	movsd  xmm0,QWORD PTR [rdi+0x210]
   1ce3f:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1ce44:	movapd xmm5,xmm0
   1ce48:	mulsd  xmm5,xmm1
   1ce4c:	mulsd  xmm0,xmm4
   1ce50:	mulsd  xmm1,xmm3
   1ce54:	movapd xmm2,xmm5
   1ce58:	movapd xmm5,xmm3
   1ce5c:	mulsd  xmm5,xmm4
   1ce60:	addsd  xmm0,xmm1
   1ce64:	addsd  xmm12,xmm0
   1ce69:	subsd  xmm2,xmm5
   1ce6d:	movsd  QWORD PTR [rdx+0x18],xmm12
   1ce73:	movsd  xmm3,QWORD PTR [rdi+0x228]
   1ce7b:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1ce80:	addsd  xmm13,xmm2
   1ce85:	movsd  QWORD PTR [rdx+0x10],xmm13
   1ce8b:	movsd  xmm0,QWORD PTR [rdi+0x220]
   1ce93:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1ce98:	movapd xmm5,xmm0
   1ce9c:	mulsd  xmm5,xmm1
   1cea0:	mulsd  xmm0,xmm4
   1cea4:	mulsd  xmm1,xmm3
   1cea8:	movapd xmm2,xmm5
   1ceac:	movapd xmm5,xmm3
   1ceb0:	mulsd  xmm5,xmm4
   1ceb4:	addsd  xmm0,xmm1
   1ceb8:	addsd  xmm10,xmm0
   1cebd:	subsd  xmm2,xmm5
   1cec1:	movsd  QWORD PTR [rdx+0x28],xmm10
   1cec7:	movsd  xmm3,QWORD PTR [rdi+0x238]
   1cecf:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1ced4:	addsd  xmm11,xmm2
   1ced9:	movsd  QWORD PTR [rdx+0x20],xmm11
   1cedf:	movsd  xmm0,QWORD PTR [rdi+0x230]
   1cee7:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1ceec:	movapd xmm5,xmm0
   1cef0:	mulsd  xmm5,xmm1
   1cef4:	mulsd  xmm0,xmm4
   1cef8:	mulsd  xmm1,xmm3
   1cefc:	movapd xmm2,xmm5
   1cf00:	movapd xmm5,xmm3
   1cf04:	mulsd  xmm5,xmm4
   1cf08:	addsd  xmm0,xmm1
   1cf0c:	addsd  xmm8,xmm0
   1cf11:	subsd  xmm2,xmm5
   1cf15:	movsd  QWORD PTR [rdx+0x38],xmm8
   1cf1b:	movsd  xmm3,QWORD PTR [rdi+0x248]
   1cf23:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1cf28:	addsd  xmm9,xmm2
   1cf2d:	movsd  QWORD PTR [rdx+0x30],xmm9
   1cf33:	movsd  xmm0,QWORD PTR [rdi+0x240]
   1cf3b:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1cf40:	movapd xmm5,xmm0
   1cf44:	mulsd  xmm5,xmm1
   1cf48:	mulsd  xmm0,xmm4
   1cf4c:	mulsd  xmm1,xmm3
   1cf50:	movapd xmm2,xmm5
   1cf54:	movapd xmm5,xmm3
   1cf58:	mulsd  xmm5,xmm4
   1cf5c:	addsd  xmm0,xmm1
   1cf60:	addsd  xmm6,xmm0
   1cf64:	subsd  xmm2,xmm5
   1cf68:	movsd  QWORD PTR [rdx+0x48],xmm6
   1cf6d:	addsd  xmm7,xmm2
   1cf71:	movsd  QWORD PTR [rdx+0x40],xmm7
   1cf76:	movsd  xmm0,QWORD PTR [rdi+0x250]
   1cf7e:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1cf83:	movsd  xmm3,QWORD PTR [rdi+0x258]
   1cf8b:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1cf90:	movapd xmm5,xmm0
   1cf94:	mulsd  xmm5,xmm1
   1cf98:	mulsd  xmm0,xmm4
   1cf9c:	mulsd  xmm1,xmm3
   1cfa0:	movapd xmm2,xmm5
   1cfa4:	movapd xmm5,xmm3
   1cfa8:	mulsd  xmm5,xmm4
   1cfac:	addsd  xmm0,xmm1
   1cfb0:	movsd  xmm1,QWORD PTR [rsp+0x8]
   1cfb6:	addsd  xmm1,xmm0
   1cfba:	subsd  xmm2,xmm5
   1cfbe:	addsd  xmm2,QWORD PTR [rsp]
   1cfc3:	movsd  QWORD PTR [rdx+0x58],xmm1
   1cfc8:	movsd  xmm3,QWORD PTR [rdi+0x268]
   1cfd0:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1cfd5:	movsd  QWORD PTR [rdx+0x50],xmm2
   1cfda:	movsd  xmm0,QWORD PTR [rdi+0x260]
   1cfe2:	movsd  QWORD PTR [rsp+0x8],xmm1
   1cfe8:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1cfed:	movapd xmm5,xmm0
   1cff1:	movsd  QWORD PTR [rsp],xmm2
   1cff6:	mulsd  xmm5,xmm1
   1cffa:	mulsd  xmm0,xmm4
   1cffe:	mulsd  xmm1,xmm3
   1d002:	movapd xmm2,xmm5
   1d006:	movapd xmm5,xmm3
   1d00a:	mulsd  xmm5,xmm4
   1d00e:	addsd  xmm0,xmm1
   1d012:	addsd  xmm0,QWORD PTR [rsp+0x18]
   1d018:	movsd  QWORD PTR [rdx+0x68],xmm0
   1d01d:	movsd  xmm3,QWORD PTR [rdi+0x278]
   1d025:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1d02a:	subsd  xmm2,xmm5
   1d02e:	addsd  xmm2,QWORD PTR [rsp+0x10]
   1d034:	movsd  QWORD PTR [rsp+0x18],xmm0
   1d03a:	movapd xmm5,xmm3
   1d03e:	mulsd  xmm5,xmm4
   1d042:	movsd  QWORD PTR [rdx+0x60],xmm2
   1d047:	movsd  xmm0,QWORD PTR [rdi+0x270]
   1d04f:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1d054:	movsd  QWORD PTR [rsp+0x10],xmm2
   1d05a:	movapd xmm2,xmm0
   1d05e:	mulsd  xmm2,xmm1
   1d062:	mulsd  xmm0,xmm4
   1d066:	mulsd  xmm1,xmm3
   1d06a:	subsd  xmm2,xmm5
   1d06e:	addsd  xmm2,QWORD PTR [rsp+0x20]
   1d074:	addsd  xmm0,xmm1
   1d078:	addsd  xmm0,QWORD PTR [rsp+0x28]
   1d07e:	movsd  QWORD PTR [rdx+0x70],xmm2
   1d083:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1d088:	movsd  QWORD PTR [rsp+0x20],xmm2
   1d08e:	movsd  QWORD PTR [rsp+0x28],xmm0
   1d094:	movsd  QWORD PTR [rdx+0x78],xmm0
   1d099:	movsd  xmm0,QWORD PTR [rdi+0x280]
   1d0a1:	movsd  xmm3,QWORD PTR [rdi+0x288]
   1d0a9:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1d0ae:	movapd xmm2,xmm0
   1d0b2:	movapd xmm5,xmm3
   1d0b6:	mulsd  xmm5,xmm4
   1d0ba:	mulsd  xmm2,xmm1
   1d0be:	mulsd  xmm0,xmm4
   1d0c2:	mulsd  xmm1,xmm3
   1d0c6:	subsd  xmm2,xmm5
   1d0ca:	addsd  xmm2,QWORD PTR [rsp+0x30]
   1d0d0:	addsd  xmm0,xmm1
   1d0d4:	addsd  xmm0,QWORD PTR [rsp+0x38]
   1d0da:	movsd  QWORD PTR [rdx+0x80],xmm2
   1d0e2:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1d0e7:	movsd  QWORD PTR [rsp+0x30],xmm2
   1d0ed:	movsd  QWORD PTR [rdx+0x88],xmm0
   1d0f5:	movsd  xmm3,QWORD PTR [rdi+0x298]
   1d0fd:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1d102:	movsd  QWORD PTR [rsp+0x38],xmm0
   1d108:	movsd  xmm0,QWORD PTR [rdi+0x290]
   1d110:	movapd xmm5,xmm3
   1d114:	mulsd  xmm5,xmm4
   1d118:	movapd xmm2,xmm0
   1d11c:	mulsd  xmm2,xmm1
   1d120:	mulsd  xmm0,xmm4
   1d124:	mulsd  xmm1,xmm3
   1d128:	subsd  xmm2,xmm5
   1d12c:	addsd  xmm2,QWORD PTR [rsp+0x40]
   1d132:	addsd  xmm0,xmm1
   1d136:	movsd  xmm1,QWORD PTR [rsp+0x48]
   1d13c:	movsd  QWORD PTR [rdx+0x90],xmm2
   1d144:	movsd  QWORD PTR [rsp+0x40],xmm2
   1d14a:	addsd  xmm1,xmm0
   1d14e:	movsd  xmm0,QWORD PTR [rdi+0x2a0]
   1d156:	movapd xmm2,xmm0
   1d15a:	movsd  QWORD PTR [rdx+0x98],xmm1
   1d162:	movsd  xmm3,QWORD PTR [rdi+0x2a8]
   1d16a:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1d16f:	movsd  QWORD PTR [rsp+0x48],xmm1
   1d175:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1d17a:	mulsd  xmm0,xmm4
   1d17e:	movapd xmm5,xmm3
   1d182:	mulsd  xmm5,xmm4
   1d186:	mulsd  xmm2,xmm1
   1d18a:	mulsd  xmm1,xmm3
   1d18e:	subsd  xmm2,xmm5
   1d192:	addsd  xmm2,QWORD PTR [rsp+0x50]
   1d198:	addsd  xmm0,xmm1
   1d19c:	addsd  xmm0,QWORD PTR [rsp+0x58]
   1d1a2:	movsd  QWORD PTR [rdx+0xa0],xmm2
   1d1aa:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1d1af:	movsd  QWORD PTR [rdx+0xa8],xmm0
   1d1b7:	movsd  xmm3,QWORD PTR [rdi+0x2b8]
   1d1bf:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1d1c4:	movsd  QWORD PTR [rsp+0x58],xmm0
   1d1ca:	movsd  xmm0,QWORD PTR [rdi+0x2b0]
   1d1d2:	movsd  QWORD PTR [rsp+0x50],xmm2
   1d1d8:	movapd xmm5,xmm3
   1d1dc:	mulsd  xmm5,xmm4
   1d1e0:	movapd xmm2,xmm0
   1d1e4:	mulsd  xmm2,xmm1
   1d1e8:	mulsd  xmm0,xmm4
   1d1ec:	mulsd  xmm1,xmm3
   1d1f0:	subsd  xmm2,xmm5
   1d1f4:	addsd  xmm2,QWORD PTR [rsp+0x60]
   1d1fa:	addsd  xmm0,xmm1
   1d1fe:	movsd  QWORD PTR [rsp+0x60],xmm2
   1d204:	addsd  xmm0,QWORD PTR [rsp+0x68]
   1d20a:	movsd  QWORD PTR [rdx+0xb0],xmm2
   1d212:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1d217:	movsd  QWORD PTR [rdx+0xb8],xmm0
   1d21f:	movsd  xmm3,QWORD PTR [rdi+0x2c8]
   1d227:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1d22c:	movsd  QWORD PTR [rsp+0x68],xmm0
   1d232:	movsd  xmm0,QWORD PTR [rdi+0x2c0]
   1d23a:	movapd xmm5,xmm3
   1d23e:	mulsd  xmm5,xmm4
   1d242:	movapd xmm2,xmm0
   1d246:	mulsd  xmm2,xmm1
   1d24a:	mulsd  xmm0,xmm4
   1d24e:	mulsd  xmm1,xmm3
   1d252:	subsd  xmm2,xmm5
   1d256:	addsd  xmm2,QWORD PTR [rsp+0x70]
   1d25c:	addsd  xmm0,xmm1
   1d260:	addsd  xmm0,QWORD PTR [rsp+0x78]
   1d266:	movsd  QWORD PTR [rdx+0xc0],xmm2
   1d26e:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1d273:	movsd  QWORD PTR [rsp+0x70],xmm2
   1d279:	movsd  QWORD PTR [rdx+0xc8],xmm0
   1d281:	movsd  xmm3,QWORD PTR [rdi+0x2d8]
   1d289:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1d28e:	movsd  QWORD PTR [rsp+0x78],xmm0
   1d294:	movsd  xmm0,QWORD PTR [rdi+0x2d0]
   1d29c:	movapd xmm5,xmm3
   1d2a0:	mulsd  xmm5,xmm4
   1d2a4:	movapd xmm2,xmm0
   1d2a8:	mulsd  xmm2,xmm1
   1d2ac:	mulsd  xmm0,xmm4
   1d2b0:	movsd  xmm4,QWORD PTR [rsp+0x88]
   1d2b9:	mulsd  xmm1,xmm3
   1d2bd:	subsd  xmm2,xmm5
   1d2c1:	addsd  xmm2,QWORD PTR [rsp+0x80]
   1d2ca:	addsd  xmm0,xmm1
   1d2ce:	movsd  QWORD PTR [rdx+0xd0],xmm2
   1d2d6:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1d2db:	movsd  QWORD PTR [rsp+0x80],xmm2
   1d2e4:	addsd  xmm4,xmm0
   1d2e8:	movsd  xmm0,QWORD PTR [rdi+0x2e0]
   1d2f0:	movapd xmm2,xmm0
   1d2f4:	movsd  QWORD PTR [rdx+0xd8],xmm4
   1d2fc:	movsd  xmm3,QWORD PTR [rdi+0x2e8]
   1d304:	mulsd  xmm2,xmm1
   1d308:	movsd  QWORD PTR [rsp+0x88],xmm4
   1d311:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1d316:	mulsd  xmm1,xmm3
   1d31a:	movapd xmm5,xmm3
   1d31e:	mulsd  xmm5,xmm4
   1d322:	mulsd  xmm0,xmm4
   1d326:	subsd  xmm2,xmm5
   1d32a:	addsd  xmm2,QWORD PTR [rsp+0x90]
   1d333:	addsd  xmm0,xmm1
   1d337:	addsd  xmm0,QWORD PTR [rsp+0x98]
   1d340:	movsd  QWORD PTR [rsp+0x90],xmm2
   1d349:	movsd  QWORD PTR [rdx+0xe0],xmm2
   1d351:	movsd  QWORD PTR [rsp+0x98],xmm0
   1d35a:	movsd  xmm1,QWORD PTR [rsi+0x20]
   1d35f:	movsd  QWORD PTR [rdx+0xe8],xmm0
   1d367:	movsd  xmm0,QWORD PTR [rdi+0x2f0]
   1d36f:	movsd  xmm3,QWORD PTR [rdi+0x2f8]
   1d377:	movsd  xmm4,QWORD PTR [rsi+0x28]
   1d37c:	movapd xmm2,xmm0
   1d380:	movapd xmm5,xmm3
   1d384:	mulsd  xmm5,xmm4
   1d388:	mulsd  xmm2,xmm1
   1d38c:	mulsd  xmm0,xmm4
   1d390:	mulsd  xmm1,xmm3
   1d394:	subsd  xmm2,xmm5
   1d398:	addsd  xmm2,QWORD PTR [rsp+0xa0]
   1d3a1:	addsd  xmm0,xmm1
   1d3a5:	addsd  xmm0,QWORD PTR [rsp+0xa8]
   1d3ae:	movsd  QWORD PTR [rdx+0xf0],xmm2
   1d3b6:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d3bb:	movsd  QWORD PTR [rsp+0xa0],xmm2
   1d3c4:	movsd  QWORD PTR [rdx+0xf8],xmm0
   1d3cc:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d3d1:	movsd  xmm3,QWORD PTR [rdi+0x308]
   1d3d9:	movsd  QWORD PTR [rsp+0xa8],xmm0
   1d3e2:	movsd  xmm0,QWORD PTR [rdi+0x300]
   1d3ea:	movapd xmm5,xmm3
   1d3ee:	mulsd  xmm5,xmm4
   1d3f2:	movapd xmm2,xmm0
   1d3f6:	mulsd  xmm2,xmm1
   1d3fa:	mulsd  xmm0,xmm4
   1d3fe:	mulsd  xmm1,xmm3
   1d402:	subsd  xmm2,xmm5
   1d406:	addsd  xmm0,xmm1
   1d40a:	addsd  xmm15,xmm2
   1d40f:	addsd  xmm14,xmm0
   1d414:	movsd  QWORD PTR [rdx],xmm15
   1d419:	movsd  xmm0,QWORD PTR [rdi+0x310]
   1d421:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d426:	movsd  QWORD PTR [rdx+0x8],xmm14
   1d42c:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d431:	movapd xmm2,xmm0
   1d435:	movsd  xmm3,QWORD PTR [rdi+0x318]
   1d43d:	mulsd  xmm2,xmm1
   1d441:	mulsd  xmm0,xmm4
   1d445:	movapd xmm5,xmm3
   1d449:	mulsd  xmm5,xmm4
   1d44d:	mulsd  xmm1,xmm3
   1d451:	subsd  xmm2,xmm5
   1d455:	addsd  xmm0,xmm1
   1d459:	addsd  xmm13,xmm2
   1d45e:	addsd  xmm12,xmm0
   1d463:	movsd  QWORD PTR [rdx+0x10],xmm13
   1d469:	movsd  xmm0,QWORD PTR [rdi+0x320]
   1d471:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d476:	movsd  QWORD PTR [rdx+0x18],xmm12
   1d47c:	movsd  xmm3,QWORD PTR [rdi+0x328]
   1d484:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d489:	movapd xmm2,xmm0
   1d48d:	mulsd  xmm2,xmm1
   1d491:	movapd xmm5,xmm3
   1d495:	mulsd  xmm5,xmm4
   1d499:	mulsd  xmm1,xmm3
   1d49d:	mulsd  xmm0,xmm4
   1d4a1:	subsd  xmm2,xmm5
   1d4a5:	addsd  xmm0,xmm1
   1d4a9:	addsd  xmm11,xmm2
   1d4ae:	addsd  xmm10,xmm0
   1d4b3:	movsd  QWORD PTR [rdx+0x20],xmm11
   1d4b9:	movsd  xmm0,QWORD PTR [rdi+0x330]
   1d4c1:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d4c6:	movsd  QWORD PTR [rdx+0x28],xmm10
   1d4cc:	movsd  xmm3,QWORD PTR [rdi+0x338]
   1d4d4:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d4d9:	movapd xmm2,xmm0
   1d4dd:	mulsd  xmm2,xmm1
   1d4e1:	mulsd  xmm0,xmm4
   1d4e5:	movapd xmm5,xmm3
   1d4e9:	mulsd  xmm5,xmm4
   1d4ed:	mulsd  xmm1,xmm3
   1d4f1:	subsd  xmm2,xmm5
   1d4f5:	addsd  xmm0,xmm1
   1d4f9:	addsd  xmm9,xmm2
   1d4fe:	addsd  xmm8,xmm0
   1d503:	movsd  QWORD PTR [rdx+0x30],xmm9
   1d509:	movsd  xmm0,QWORD PTR [rdi+0x340]
   1d511:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d516:	movsd  QWORD PTR [rdx+0x38],xmm8
   1d51c:	movsd  xmm3,QWORD PTR [rdi+0x348]
   1d524:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d529:	movapd xmm2,xmm0
   1d52d:	mulsd  xmm2,xmm1
   1d531:	movapd xmm5,xmm3
   1d535:	mulsd  xmm5,xmm4
   1d539:	mulsd  xmm1,xmm3
   1d53d:	mulsd  xmm0,xmm4
   1d541:	subsd  xmm2,xmm5
   1d545:	addsd  xmm0,xmm1
   1d549:	addsd  xmm7,xmm2
   1d54d:	addsd  xmm6,xmm0
   1d551:	movsd  QWORD PTR [rdx+0x40],xmm7
   1d556:	movsd  xmm0,QWORD PTR [rdi+0x350]
   1d55e:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d563:	movsd  QWORD PTR [rdx+0x48],xmm6
   1d568:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d56d:	movapd xmm2,xmm0
   1d571:	movsd  xmm3,QWORD PTR [rdi+0x358]
   1d579:	mulsd  xmm2,xmm1
   1d57d:	mulsd  xmm0,xmm4
   1d581:	movapd xmm5,xmm3
   1d585:	mulsd  xmm5,xmm4
   1d589:	mulsd  xmm1,xmm3
   1d58d:	movsd  xmm3,QWORD PTR [rsp+0x8]
   1d593:	subsd  xmm2,xmm5
   1d597:	addsd  xmm2,QWORD PTR [rsp]
   1d59c:	addsd  xmm0,xmm1
   1d5a0:	movsd  QWORD PTR [rdx+0x50],xmm2
   1d5a5:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d5aa:	addsd  xmm3,xmm0
   1d5ae:	movsd  xmm0,QWORD PTR [rdi+0x360]
   1d5b6:	movsd  QWORD PTR [rsp],xmm2
   1d5bb:	movapd xmm5,xmm0
   1d5bf:	mulsd  xmm5,xmm1
   1d5c3:	movsd  QWORD PTR [rdx+0x58],xmm3
   1d5c8:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d5cd:	movsd  QWORD PTR [rsp+0x8],xmm3
   1d5d3:	movsd  xmm3,QWORD PTR [rdi+0x368]
   1d5db:	mulsd  xmm0,xmm4
   1d5df:	mulsd  xmm1,xmm3
   1d5e3:	movapd xmm2,xmm5
   1d5e7:	movapd xmm5,xmm3
   1d5eb:	mulsd  xmm5,xmm4
   1d5ef:	addsd  xmm0,xmm1
   1d5f3:	addsd  xmm0,QWORD PTR [rsp+0x18]
   1d5f9:	subsd  xmm2,xmm5
   1d5fd:	addsd  xmm2,QWORD PTR [rsp+0x10]
   1d603:	movsd  QWORD PTR [rsp+0x18],xmm0
   1d609:	movsd  QWORD PTR [rdx+0x68],xmm0
   1d60e:	movsd  xmm3,QWORD PTR [rdi+0x378]
   1d616:	movsd  QWORD PTR [rdx+0x60],xmm2
   1d61b:	movsd  xmm0,QWORD PTR [rdi+0x370]
   1d623:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d628:	movsd  QWORD PTR [rsp+0x10],xmm2
   1d62e:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d633:	movapd xmm5,xmm0
   1d637:	mulsd  xmm5,xmm1
   1d63b:	mulsd  xmm0,xmm4
   1d63f:	mulsd  xmm1,xmm3
   1d643:	movapd xmm2,xmm5
   1d647:	movapd xmm5,xmm3
   1d64b:	mulsd  xmm5,xmm4
   1d64f:	movsd  xmm4,QWORD PTR [rsp+0x28]
   1d655:	addsd  xmm0,xmm1
   1d659:	addsd  xmm4,xmm0
   1d65d:	subsd  xmm2,xmm5
   1d661:	addsd  xmm2,QWORD PTR [rsp+0x20]
   1d667:	movsd  QWORD PTR [rdx+0x78],xmm4
   1d66c:	movsd  xmm3,QWORD PTR [rdi+0x388]
   1d674:	movsd  QWORD PTR [rdx+0x70],xmm2
   1d679:	movsd  xmm0,QWORD PTR [rdi+0x380]
   1d681:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d686:	movsd  QWORD PTR [rsp+0x28],xmm4
   1d68c:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d691:	movapd xmm5,xmm3
   1d695:	movsd  QWORD PTR [rsp+0x20],xmm2
   1d69b:	movapd xmm2,xmm0
   1d69f:	mulsd  xmm2,xmm1
   1d6a3:	mulsd  xmm0,xmm4
   1d6a7:	mulsd  xmm1,xmm3
   1d6ab:	mulsd  xmm5,xmm4
   1d6af:	movsd  xmm4,QWORD PTR [rsp+0x38]
   1d6b5:	addsd  xmm0,xmm1
   1d6b9:	subsd  xmm2,xmm5
   1d6bd:	addsd  xmm2,QWORD PTR [rsp+0x30]
   1d6c3:	addsd  xmm4,xmm0
   1d6c7:	movsd  QWORD PTR [rdx+0x80],xmm2
   1d6cf:	movsd  xmm0,QWORD PTR [rdi+0x390]
   1d6d7:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d6dc:	movsd  QWORD PTR [rsp+0x30],xmm2
   1d6e2:	movsd  QWORD PTR [rdx+0x88],xmm4
   1d6ea:	movsd  xmm3,QWORD PTR [rdi+0x398]
   1d6f2:	movapd xmm2,xmm0
   1d6f6:	movsd  QWORD PTR [rsp+0x38],xmm4
   1d6fc:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d701:	mulsd  xmm2,xmm1
   1d705:	mulsd  xmm1,xmm3
   1d709:	movapd xmm5,xmm3
   1d70d:	mulsd  xmm5,xmm4
   1d711:	mulsd  xmm0,xmm4
   1d715:	subsd  xmm2,xmm5
   1d719:	addsd  xmm2,QWORD PTR [rsp+0x40]
   1d71f:	addsd  xmm0,xmm1
   1d723:	addsd  xmm0,QWORD PTR [rsp+0x48]
   1d729:	movsd  QWORD PTR [rdx+0x90],xmm2
   1d731:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d736:	movsd  QWORD PTR [rdx+0x98],xmm0
   1d73e:	movsd  xmm3,QWORD PTR [rdi+0x3a8]
   1d746:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d74b:	movsd  QWORD PTR [rsp+0x48],xmm0
   1d751:	movsd  xmm0,QWORD PTR [rdi+0x3a0]
   1d759:	movsd  QWORD PTR [rsp+0x40],xmm2
   1d75f:	movapd xmm5,xmm3
   1d763:	mulsd  xmm5,xmm4
   1d767:	movapd xmm2,xmm0
   1d76b:	mulsd  xmm2,xmm1
   1d76f:	mulsd  xmm0,xmm4
   1d773:	mulsd  xmm1,xmm3
   1d777:	subsd  xmm2,xmm5
   1d77b:	addsd  xmm2,QWORD PTR [rsp+0x50]
   1d781:	addsd  xmm0,xmm1
   1d785:	movsd  QWORD PTR [rsp+0x50],xmm2
   1d78b:	addsd  xmm0,QWORD PTR [rsp+0x58]
   1d791:	movsd  QWORD PTR [rdx+0xa0],xmm2
   1d799:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d79e:	movsd  QWORD PTR [rdx+0xa8],xmm0
   1d7a6:	movsd  xmm3,QWORD PTR [rdi+0x3b8]
   1d7ae:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d7b3:	movsd  QWORD PTR [rsp+0x58],xmm0
   1d7b9:	movsd  xmm0,QWORD PTR [rdi+0x3b0]
   1d7c1:	movapd xmm5,xmm0
   1d7c5:	mulsd  xmm5,xmm1
   1d7c9:	mulsd  xmm0,xmm4
   1d7cd:	mulsd  xmm1,xmm3
   1d7d1:	movapd xmm2,xmm5
   1d7d5:	movapd xmm5,xmm3
   1d7d9:	mulsd  xmm5,xmm4
   1d7dd:	movsd  xmm4,QWORD PTR [rsp+0x68]
   1d7e3:	addsd  xmm0,xmm1
   1d7e7:	addsd  xmm4,xmm0
   1d7eb:	subsd  xmm2,xmm5
   1d7ef:	addsd  xmm2,QWORD PTR [rsp+0x60]
   1d7f5:	movsd  QWORD PTR [rdx+0xb8],xmm4
   1d7fd:	movsd  xmm3,QWORD PTR [rdi+0x3c8]
   1d805:	movsd  QWORD PTR [rdx+0xb0],xmm2
   1d80d:	movsd  xmm0,QWORD PTR [rdi+0x3c0]
   1d815:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d81a:	movsd  QWORD PTR [rsp+0x68],xmm4
   1d820:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d825:	movapd xmm5,xmm0
   1d829:	movsd  QWORD PTR [rsp+0x60],xmm2
   1d82f:	mulsd  xmm5,xmm1
   1d833:	mulsd  xmm0,xmm4
   1d837:	mulsd  xmm1,xmm3
   1d83b:	movapd xmm2,xmm5
   1d83f:	movapd xmm5,xmm3
   1d843:	mulsd  xmm5,xmm4
   1d847:	addsd  xmm0,xmm1
   1d84b:	addsd  xmm0,QWORD PTR [rsp+0x78]
   1d851:	movsd  QWORD PTR [rdx+0xc8],xmm0
   1d859:	movsd  xmm3,QWORD PTR [rdi+0x3d8]
   1d861:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d866:	subsd  xmm2,xmm5
   1d86a:	addsd  xmm2,QWORD PTR [rsp+0x70]
   1d870:	movsd  QWORD PTR [rsp+0x78],xmm0
   1d876:	movsd  QWORD PTR [rdx+0xc0],xmm2
   1d87e:	movsd  xmm0,QWORD PTR [rdi+0x3d0]
   1d886:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d88b:	movsd  QWORD PTR [rsp+0x70],xmm2
   1d891:	movapd xmm5,xmm0
   1d895:	mulsd  xmm5,xmm1
   1d899:	mulsd  xmm0,xmm4
   1d89d:	mulsd  xmm1,xmm3
   1d8a1:	movapd xmm2,xmm5
   1d8a5:	movapd xmm5,xmm3
   1d8a9:	mulsd  xmm5,xmm4
   1d8ad:	addsd  xmm0,xmm1
   1d8b1:	addsd  xmm0,QWORD PTR [rsp+0x88]
   1d8ba:	subsd  xmm2,xmm5
   1d8be:	addsd  xmm2,QWORD PTR [rsp+0x80]
   1d8c7:	movsd  QWORD PTR [rsp+0x80],xmm2
   1d8d0:	movsd  QWORD PTR [rdx+0xd0],xmm2
   1d8d8:	movsd  QWORD PTR [rsp+0x88],xmm0
   1d8e1:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d8e6:	movsd  QWORD PTR [rdx+0xd8],xmm0
   1d8ee:	movsd  xmm0,QWORD PTR [rdi+0x3e0]
   1d8f6:	movsd  xmm3,QWORD PTR [rdi+0x3e8]
   1d8fe:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d903:	movapd xmm5,xmm0
   1d907:	mulsd  xmm5,xmm1
   1d90b:	mulsd  xmm0,xmm4
   1d90f:	mulsd  xmm1,xmm3
   1d913:	movapd xmm2,xmm5
   1d917:	movapd xmm5,xmm3
   1d91b:	mulsd  xmm5,xmm4
   1d91f:	addsd  xmm0,xmm1
   1d923:	addsd  xmm0,QWORD PTR [rsp+0x98]
   1d92c:	movsd  QWORD PTR [rdx+0xe8],xmm0
   1d934:	movsd  xmm4,QWORD PTR [rsi+0x38]
   1d939:	movsd  xmm3,QWORD PTR [rdi+0x3f8]
   1d941:	movsd  QWORD PTR [rsp+0x98],xmm0
   1d94a:	subsd  xmm2,xmm5
   1d94e:	addsd  xmm2,QWORD PTR [rsp+0x90]
   1d957:	movapd xmm5,xmm3
   1d95b:	mulsd  xmm5,xmm4
   1d95f:	movsd  QWORD PTR [rdx+0xe0],xmm2
   1d967:	movsd  xmm1,QWORD PTR [rsi+0x30]
   1d96c:	movsd  xmm0,QWORD PTR [rdi+0x3f0]
   1d974:	movsd  QWORD PTR [rsp+0x90],xmm2
   1d97d:	movapd xmm2,xmm0
   1d981:	mulsd  xmm2,xmm1
   1d985:	mulsd  xmm0,xmm4
   1d989:	movsd  xmm4,QWORD PTR [rsp+0xa8]
   1d992:	mulsd  xmm1,xmm3
   1d996:	subsd  xmm2,xmm5
   1d99a:	addsd  xmm2,QWORD PTR [rsp+0xa0]
   1d9a3:	addsd  xmm0,xmm1
   1d9a7:	movsd  QWORD PTR [rdx+0xf0],xmm2
   1d9af:	movsd  xmm1,QWORD PTR [rsi+0x40]
   1d9b4:	movsd  QWORD PTR [rsp+0xa0],xmm2
   1d9bd:	addsd  xmm4,xmm0
   1d9c1:	movsd  xmm0,QWORD PTR [rdi+0x400]
   1d9c9:	movapd xmm2,xmm0
   1d9cd:	movsd  QWORD PTR [rdx+0xf8],xmm4
   1d9d5:	movsd  xmm3,QWORD PTR [rdi+0x408]
   1d9dd:	mulsd  xmm2,xmm1
   1d9e1:	movsd  QWORD PTR [rsp+0xa8],xmm4
   1d9ea:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1d9ef:	mulsd  xmm1,xmm3
   1d9f3:	movapd xmm5,xmm3
   1d9f7:	mulsd  xmm5,xmm4
   1d9fb:	mulsd  xmm0,xmm4
   1d9ff:	subsd  xmm2,xmm5
   1da03:	addsd  xmm0,xmm1
   1da07:	addsd  xmm15,xmm2
   1da0c:	addsd  xmm14,xmm0
   1da11:	movsd  QWORD PTR [rdx],xmm15
   1da16:	movsd  xmm0,QWORD PTR [rdi+0x410]
   1da1e:	movsd  xmm1,QWORD PTR [rsi+0x40]
   1da23:	movsd  QWORD PTR [rdx+0x8],xmm14
   1da29:	movsd  xmm3,QWORD PTR [rdi+0x418]
   1da31:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1da36:	movapd xmm2,xmm0
   1da3a:	mulsd  xmm2,xmm1
   1da3e:	movapd xmm5,xmm3
   1da42:	mulsd  xmm5,xmm4
   1da46:	mulsd  xmm1,xmm3
   1da4a:	mulsd  xmm0,xmm4
   1da4e:	subsd  xmm2,xmm5
   1da52:	addsd  xmm0,xmm1
   1da56:	addsd  xmm2,xmm13
   1da5b:	movapd xmm13,xmm0
   1da60:	movsd  QWORD PTR [rsp+0xb0],xmm2
   1da69:	addsd  xmm13,xmm12
   1da6e:	movsd  QWORD PTR [rdx+0x10],xmm2
   1da73:	movsd  xmm0,QWORD PTR [rdi+0x420]
   1da7b:	movsd  xmm1,QWORD PTR [rsi+0x40]
   1da80:	movsd  QWORD PTR [rdx+0x18],xmm13
   1da86:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1da8b:	movsd  xmm3,QWORD PTR [rdi+0x428]
   1da93:	movsd  QWORD PTR [rsp+0xb8],xmm13
   1da9d:	movapd xmm13,xmm0
   1daa2:	mulsd  xmm13,xmm1
   1daa7:	mulsd  xmm0,xmm4
   1daab:	mulsd  xmm1,xmm3
   1daaf:	movapd xmm2,xmm13
   1dab4:	movapd xmm13,xmm3
   1dab9:	mulsd  xmm13,xmm4
   1dabe:	addsd  xmm0,xmm1
   1dac2:	subsd  xmm2,xmm13
   1dac7:	movapd xmm12,xmm2
   1dacc:	addsd  xmm12,xmm11
   1dad1:	movapd xmm11,xmm0
   1dad6:	addsd  xmm11,xmm10
   1dadb:	movsd  QWORD PTR [rdx+0x20],xmm12
   1dae1:	movsd  xmm0,QWORD PTR [rdi+0x430]
   1dae9:	movsd  xmm1,QWORD PTR [rsi+0x40]
   1daee:	movsd  QWORD PTR [rdx+0x28],xmm11
   1daf4:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1daf9:	movapd xmm13,xmm11
   1dafe:	movsd  xmm3,QWORD PTR [rdi+0x438]
   1db06:	movapd xmm10,xmm0
   1db0b:	mulsd  xmm10,xmm1
   1db10:	mulsd  xmm0,xmm4
   1db14:	mulsd  xmm1,xmm3
   1db18:	movapd xmm2,xmm10
   1db1d:	movapd xmm10,xmm3
   1db22:	mulsd  xmm10,xmm4
   1db27:	addsd  xmm0,xmm1
   1db2b:	movapd xmm11,xmm0
   1db30:	subsd  xmm2,xmm10
   1db35:	addsd  xmm11,xmm8
   1db3a:	movapd xmm10,xmm2
   1db3f:	movsd  QWORD PTR [rdx+0x38],xmm11
   1db45:	movsd  xmm3,QWORD PTR [rdi+0x448]
   1db4d:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1db52:	movsd  QWORD PTR [rsp+0xc8],xmm11
   1db5c:	addsd  xmm10,xmm9
   1db61:	movsd  QWORD PTR [rdx+0x30],xmm10
   1db67:	movsd  xmm1,QWORD PTR [rsi+0x40]
   1db6c:	movsd  xmm0,QWORD PTR [rdi+0x440]
   1db74:	movsd  QWORD PTR [rsp+0xc0],xmm10
   1db7e:	movapd xmm8,xmm0
   1db83:	mulsd  xmm8,xmm1
   1db88:	mulsd  xmm0,xmm4
   1db8c:	mulsd  xmm1,xmm3
   1db90:	movapd xmm2,xmm8
   1db95:	movapd xmm8,xmm3
   1db9a:	mulsd  xmm8,xmm4
   1db9f:	addsd  xmm0,xmm1
   1dba3:	addsd  xmm6,xmm0
   1dba7:	subsd  xmm2,xmm8
   1dbac:	movsd  QWORD PTR [rdx+0x48],xmm6
   1dbb1:	movsd  xmm3,QWORD PTR [rdi+0x458]
   1dbb9:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1dbbe:	movapd xmm8,xmm2
   1dbc3:	addsd  xmm8,xmm7
   1dbc8:	movapd xmm9,xmm3
   1dbcd:	mulsd  xmm9,xmm4
   1dbd2:	movsd  QWORD PTR [rdx+0x40],xmm8
   1dbd8:	movsd  xmm1,QWORD PTR [rsi+0x40]
   1dbdd:	movapd xmm5,xmm8
   1dbe2:	movsd  xmm0,QWORD PTR [rdi+0x450]
   1dbea:	movapd xmm8,xmm0
   1dbef:	mulsd  xmm8,xmm1
   1dbf4:	mulsd  xmm0,xmm4
   1dbf8:	mulsd  xmm1,xmm3
   1dbfc:	movapd xmm2,xmm8
   1dc01:	movsd  xmm8,QWORD PTR [rsp]
   1dc07:	subsd  xmm2,xmm9
   1dc0c:	movsd  xmm9,QWORD PTR [rsp+0x8]
   1dc13:	addsd  xmm0,xmm1
   1dc17:	addsd  xmm8,xmm2
   1dc1c:	addsd  xmm9,xmm0
   1dc21:	movsd  QWORD PTR [rsp],xmm8
   1dc27:	movsd  QWORD PTR [rsp+0x8],xmm9
   1dc2e:	movsd  QWORD PTR [rdx+0x50],xmm8
   1dc34:	movsd  QWORD PTR [rdx+0x58],xmm9
   1dc3a:	movsd  xmm0,QWORD PTR [rdi+0x460]
   1dc42:	movsd  xmm3,QWORD PTR [rdi+0x468]
   1dc4a:	movsd  xmm1,QWORD PTR [rsi+0x40]
   1dc4f:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1dc54:	movapd xmm8,xmm0
   1dc59:	movapd xmm9,xmm3
   1dc5e:	movsd  xmm10,QWORD PTR [rsp+0x18]
   1dc65:	mulsd  xmm8,xmm1
   1dc6a:	mulsd  xmm9,xmm4
   1dc6f:	mulsd  xmm1,xmm3
   1dc73:	mulsd  xmm0,xmm4
   1dc77:	movapd xmm2,xmm8
   1dc7c:	movsd  xmm8,QWORD PTR [rsp+0x10]
   1dc83:	subsd  xmm2,xmm9
   1dc88:	addsd  xmm0,xmm1
   1dc8c:	addsd  xmm8,xmm2
   1dc91:	addsd  xmm10,xmm0
   1dc96:	movsd  QWORD PTR [rdx+0x60],xmm8
   1dc9c:	movsd  xmm0,QWORD PTR [rdi+0x470]
   1dca4:	movsd  xmm1,QWORD PTR [rsi+0x40]
   1dca9:	movsd  QWORD PTR [rsp+0x10],xmm8
   1dcb0:	movsd  QWORD PTR [rdx+0x68],xmm10
   1dcb6:	movsd  xmm3,QWORD PTR [rdi+0x478]
   1dcbe:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1dcc3:	movapd xmm11,xmm0
   1dcc8:	mulsd  xmm11,xmm1
   1dccd:	movsd  QWORD PTR [rsp+0x18],xmm10
   1dcd4:	mulsd  xmm1,xmm3
   1dcd8:	movapd xmm9,xmm3
   1dcdd:	mulsd  xmm9,xmm4
   1dce2:	mulsd  xmm0,xmm4
   1dce6:	movapd xmm2,xmm11
   1dceb:	movsd  xmm11,QWORD PTR [rsp+0x20]
   1dcf2:	subsd  xmm2,xmm9
   1dcf7:	movsd  xmm9,QWORD PTR [rsp+0x28]
   1dcfe:	addsd  xmm0,xmm1
   1dd02:	addsd  xmm11,xmm2
   1dd07:	addsd  xmm9,xmm0
   1dd0c:	movsd  QWORD PTR [rdx+0x70],xmm11
   1dd12:	movsd  xmm0,QWORD PTR [rdi+0x480]
   1dd1a:	movsd  xmm2,QWORD PTR [rsi+0x40]
   1dd1f:	movsd  QWORD PTR [rdx+0x78],xmm9
   1dd25:	movsd  xmm3,QWORD PTR [rdi+0x488]
   1dd2d:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1dd32:	movsd  QWORD PTR [rsp+0x28],xmm9
   1dd39:	movapd xmm1,xmm0
   1dd3d:	mulsd  xmm1,xmm2
   1dd41:	movapd xmm9,xmm3
   1dd46:	movsd  QWORD PTR [rsp+0x20],xmm11
   1dd4d:	mulsd  xmm9,xmm4
   1dd52:	mulsd  xmm2,xmm3
   1dd56:	mulsd  xmm0,xmm4
   1dd5a:	subsd  xmm1,xmm9
   1dd5f:	addsd  xmm1,QWORD PTR [rsp+0x30]
   1dd65:	addsd  xmm0,xmm2
   1dd69:	addsd  xmm0,QWORD PTR [rsp+0x38]
   1dd6f:	movsd  QWORD PTR [rdx+0x80],xmm1
   1dd77:	movsd  xmm2,QWORD PTR [rsi+0x40]
   1dd7c:	movsd  QWORD PTR [rdx+0x88],xmm0
   1dd84:	movsd  xmm0,QWORD PTR [rdi+0x490]
   1dd8c:	movsd  xmm3,QWORD PTR [rdi+0x498]
   1dd94:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1dd99:	movapd xmm1,xmm0
   1dd9d:	movapd xmm9,xmm3
   1dda2:	mulsd  xmm9,xmm4
   1dda7:	mulsd  xmm1,xmm2
   1ddab:	mulsd  xmm0,xmm4
   1ddaf:	mulsd  xmm2,xmm3
   1ddb3:	subsd  xmm1,xmm9
   1ddb8:	addsd  xmm1,QWORD PTR [rsp+0x40]
   1ddbe:	addsd  xmm0,xmm2
   1ddc2:	addsd  xmm0,QWORD PTR [rsp+0x48]
   1ddc8:	movsd  QWORD PTR [rdx+0x90],xmm1
   1ddd0:	movsd  xmm2,QWORD PTR [rsi+0x40]
   1ddd5:	movsd  QWORD PTR [rdx+0x98],xmm0
   1dddd:	movsd  xmm0,QWORD PTR [rdi+0x4a0]
   1dde5:	movsd  xmm3,QWORD PTR [rdi+0x4a8]
   1dded:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1ddf2:	movapd xmm1,xmm0
   1ddf6:	movapd xmm9,xmm3
   1ddfb:	mulsd  xmm9,xmm4
   1de00:	mulsd  xmm1,xmm2
   1de04:	mulsd  xmm0,xmm4
   1de08:	mulsd  xmm2,xmm3
   1de0c:	subsd  xmm1,xmm9
   1de11:	addsd  xmm1,QWORD PTR [rsp+0x50]
   1de17:	addsd  xmm0,xmm2
   1de1b:	addsd  xmm0,QWORD PTR [rsp+0x58]
   1de21:	movsd  QWORD PTR [rdx+0xa0],xmm1
   1de29:	movsd  xmm2,QWORD PTR [rsi+0x40]
   1de2e:	movsd  QWORD PTR [rdx+0xa8],xmm0
   1de36:	movsd  xmm0,QWORD PTR [rdi+0x4b0]
   1de3e:	movsd  xmm3,QWORD PTR [rdi+0x4b8]
   1de46:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1de4b:	movapd xmm1,xmm0
   1de4f:	movapd xmm9,xmm3
   1de54:	mulsd  xmm9,xmm4
   1de59:	mulsd  xmm1,xmm2
   1de5d:	mulsd  xmm0,xmm4
   1de61:	mulsd  xmm2,xmm3
   1de65:	subsd  xmm1,xmm9
   1de6a:	addsd  xmm1,QWORD PTR [rsp+0x60]
   1de70:	addsd  xmm0,xmm2
   1de74:	addsd  xmm0,QWORD PTR [rsp+0x68]
   1de7a:	movsd  QWORD PTR [rdx+0xb0],xmm1
   1de82:	movsd  xmm2,QWORD PTR [rsi+0x40]
   1de87:	movsd  QWORD PTR [rdx+0xb8],xmm0
   1de8f:	movsd  xmm0,QWORD PTR [rdi+0x4c0]
   1de97:	movsd  xmm3,QWORD PTR [rdi+0x4c8]
   1de9f:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1dea4:	movapd xmm1,xmm0
   1dea8:	movapd xmm9,xmm3
   1dead:	mulsd  xmm9,xmm4
   1deb2:	mulsd  xmm1,xmm2
   1deb6:	mulsd  xmm0,xmm4
   1deba:	mulsd  xmm2,xmm3
   1debe:	subsd  xmm1,xmm9
   1dec3:	addsd  xmm1,QWORD PTR [rsp+0x70]
   1dec9:	addsd  xmm0,xmm2
   1decd:	addsd  xmm0,QWORD PTR [rsp+0x78]
   1ded3:	movsd  QWORD PTR [rdx+0xc0],xmm1
   1dedb:	movsd  xmm2,QWORD PTR [rsi+0x40]
   1dee0:	movsd  QWORD PTR [rdx+0xc8],xmm0
   1dee8:	movsd  xmm0,QWORD PTR [rdi+0x4d0]
   1def0:	movsd  xmm3,QWORD PTR [rdi+0x4d8]
   1def8:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1defd:	movapd xmm1,xmm0
   1df01:	movapd xmm9,xmm3
   1df06:	mulsd  xmm9,xmm4
   1df0b:	mulsd  xmm1,xmm2
   1df0f:	mulsd  xmm0,xmm4
   1df13:	mulsd  xmm2,xmm3
   1df17:	subsd  xmm1,xmm9
   1df1c:	addsd  xmm1,QWORD PTR [rsp+0x80]
   1df25:	addsd  xmm0,xmm2
   1df29:	addsd  xmm0,QWORD PTR [rsp+0x88]
   1df32:	movsd  QWORD PTR [rdx+0xd0],xmm1
   1df3a:	movsd  xmm2,QWORD PTR [rsi+0x40]
   1df3f:	movsd  QWORD PTR [rdx+0xd8],xmm0
   1df47:	movsd  xmm0,QWORD PTR [rdi+0x4e0]
   1df4f:	movsd  xmm3,QWORD PTR [rdi+0x4e8]
   1df57:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1df5c:	movapd xmm1,xmm0
   1df60:	movapd xmm9,xmm3
   1df65:	mulsd  xmm9,xmm4
   1df6a:	mulsd  xmm1,xmm2
   1df6e:	mulsd  xmm0,xmm4
   1df72:	mulsd  xmm2,xmm3
   1df76:	subsd  xmm1,xmm9
   1df7b:	addsd  xmm1,QWORD PTR [rsp+0x90]
   1df84:	addsd  xmm0,xmm2
   1df88:	addsd  xmm0,QWORD PTR [rsp+0x98]
   1df91:	movsd  QWORD PTR [rdx+0xe0],xmm1
   1df99:	movsd  xmm2,QWORD PTR [rsi+0x40]
   1df9e:	movsd  QWORD PTR [rdx+0xe8],xmm0
   1dfa6:	movsd  xmm0,QWORD PTR [rdi+0x4f0]
   1dfae:	movsd  xmm3,QWORD PTR [rdi+0x4f8]
   1dfb6:	movsd  xmm4,QWORD PTR [rsi+0x48]
   1dfbb:	movapd xmm1,xmm0
   1dfbf:	movapd xmm9,xmm3
   1dfc4:	mulsd  xmm9,xmm4
   1dfc9:	mulsd  xmm1,xmm2
   1dfcd:	mulsd  xmm0,xmm4
   1dfd1:	mulsd  xmm2,xmm3
   1dfd5:	subsd  xmm1,xmm9
   1dfda:	addsd  xmm1,QWORD PTR [rsp+0xa0]
   1dfe3:	addsd  xmm0,xmm2
   1dfe7:	addsd  xmm0,QWORD PTR [rsp+0xa8]
   1dff0:	movsd  QWORD PTR [rdx+0xf0],xmm1
   1dff8:	movsd  xmm11,QWORD PTR [rdi+0x500]
   1e001:	movsd  QWORD PTR [rdx+0xf8],xmm0
   1e009:	movsd  xmm2,QWORD PTR [rdi+0x508]
   1e011:	movsd  xmm3,QWORD PTR [rsi+0x58]
   1e016:	movapd xmm1,xmm11
   1e01b:	movsd  xmm0,QWORD PTR [rsi+0x50]
   1e020:	mulsd  xmm11,xmm3
   1e025:	movapd xmm4,xmm2
   1e029:	mulsd  xmm4,xmm3
   1e02d:	mulsd  xmm1,xmm0
   1e031:	mulsd  xmm0,xmm2
   1e035:	subsd  xmm1,xmm4
   1e039:	addsd  xmm11,xmm0
   1e03e:	addsd  xmm1,xmm15
   1e043:	addsd  xmm11,xmm14
   1e048:	movsd  QWORD PTR [rdx],xmm1
   1e04c:	movsd  xmm0,QWORD PTR [rsi+0x50]
   1e051:	movsd  xmm10,QWORD PTR [rdi+0x510]
   1e05a:	movsd  QWORD PTR [rdx+0x8],xmm11
   1e060:	movsd  xmm2,QWORD PTR [rdi+0x518]
   1e068:	movsd  xmm3,QWORD PTR [rsi+0x58]
   1e06d:	movapd xmm15,xmm10
   1e072:	movsd  QWORD PTR [rsp+0x30],xmm1
   1e078:	mulsd  xmm15,xmm0
   1e07d:	movapd xmm4,xmm2
   1e081:	mulsd  xmm4,xmm3
   1e085:	mulsd  xmm0,xmm2
   1e089:	mulsd  xmm10,xmm3
   1e08e:	subsd  xmm15,xmm4
   1e093:	addsd  xmm15,QWORD PTR [rsp+0xb0]
   1e09d:	addsd  xmm10,xmm0
   1e0a2:	movsd  QWORD PTR [rdx+0x10],xmm15
   1e0a8:	movsd  xmm0,QWORD PTR [rsi+0x50]
   1e0ad:	addsd  xmm10,QWORD PTR [rsp+0xb8]
   1e0b7:	movsd  xmm9,QWORD PTR [rdi+0x520]
   1e0c0:	movsd  QWORD PTR [rdx+0x18],xmm10
   1e0c6:	movsd  xmm3,QWORD PTR [rsi+0x58]
   1e0cb:	movapd xmm14,xmm9
   1e0d0:	movsd  xmm2,QWORD PTR [rdi+0x528]
   1e0d8:	mulsd  xmm14,xmm0
   1e0dd:	mulsd  xmm9,xmm3
   1e0e2:	movapd xmm4,xmm2
   1e0e6:	mulsd  xmm4,xmm3
   1e0ea:	mulsd  xmm0,xmm2
   1e0ee:	subsd  xmm14,xmm4
   1e0f3:	addsd  xmm9,xmm0
   1e0f8:	addsd  xmm14,xmm12
   1e0fd:	addsd  xmm9,xmm13
   1e102:	movsd  QWORD PTR [rdx+0x20],xmm14
   1e108:	movsd  xmm0,QWORD PTR [rsi+0x50]
   1e10d:	movsd  xmm8,QWORD PTR [rdi+0x530]
   1e116:	movsd  QWORD PTR [rdx+0x28],xmm9
   1e11c:	movsd  xmm2,QWORD PTR [rdi+0x538]
   1e124:	movsd  xmm3,QWORD PTR [rsi+0x58]
   1e129:	movapd xmm13,xmm8
   1e12e:	mulsd  xmm13,xmm0
   1e133:	movapd xmm4,xmm2
   1e137:	mulsd  xmm4,xmm3
   1e13b:	mulsd  xmm0,xmm2
   1e13f:	mulsd  xmm8,xmm3
   1e144:	subsd  xmm13,xmm4
   1e149:	addsd  xmm13,QWORD PTR [rsp+0xc0]
   1e153:	addsd  xmm8,xmm0
   1e158:	addsd  xmm8,QWORD PTR [rsp+0xc8]
   1e162:	movsd  QWORD PTR [rdx+0x30],xmm13
   1e168:	movsd  xmm7,QWORD PTR [rdi+0x540]
   1e170:	movsd  xmm0,QWORD PTR [rsi+0x50]
   1e175:	movsd  QWORD PTR [rdx+0x38],xmm8
   1e17b:	movsd  xmm3,QWORD PTR [rsi+0x58]
   1e180:	movapd xmm12,xmm7
   1e185:	movsd  xmm2,QWORD PTR [rdi+0x548]
   1e18d:	mulsd  xmm12,xmm0
   1e192:	mulsd  xmm7,xmm3
   1e196:	movapd xmm4,xmm2
   1e19a:	mulsd  xmm4,xmm3
   1e19e:	mulsd  xmm0,xmm2
   1e1a2:	subsd  xmm12,xmm4
   1e1a7:	addsd  xmm7,xmm0
   1e1ab:	addsd  xmm12,xmm5
   1e1b0:	addsd  xmm7,xmm6
   1e1b4:	movsd  QWORD PTR [rdx+0x40],xmm12
   1e1ba:	movsd  xmm0,QWORD PTR [rdi+0x550]
   1e1c2:	movsd  xmm2,QWORD PTR [rsi+0x50]
   1e1c7:	movsd  QWORD PTR [rdx+0x48],xmm7
   1e1cc:	movsd  xmm4,QWORD PTR [rdi+0x558]
   1e1d4:	movsd  xmm5,QWORD PTR [rsi+0x58]
   1e1d9:	movapd xmm3,xmm0
   1e1dd:	mulsd  xmm3,xmm2
   1e1e1:	movapd xmm1,xmm4
   1e1e5:	mulsd  xmm1,xmm5
   1e1e9:	mulsd  xmm2,xmm4
   1e1ed:	mulsd  xmm0,xmm5
   1e1f1:	subsd  xmm3,xmm1
   1e1f5:	addsd  xmm3,QWORD PTR [rsp]
   1e1fa:	addsd  xmm0,xmm2
   1e1fe:	addsd  xmm0,QWORD PTR [rsp+0x8]
   1e204:	movsd  QWORD PTR [rsp],xmm3
   1e209:	movsd  QWORD PTR [rdx+0x50],xmm3
   1e20e:	movsd  QWORD PTR [rsp+0x8],xmm0
   1e214:	movsd  QWORD PTR [rdx+0x58],xmm0
   1e219:	movsd  xmm0,QWORD PTR [rdi+0x560]
   1e221:	movsd  xmm2,QWORD PTR [rsi+0x50]
   1e226:	movsd  xmm4,QWORD PTR [rdi+0x568]
   1e22e:	movsd  xmm5,QWORD PTR [rsi+0x58]
   1e233:	movapd xmm1,xmm0
   1e237:	mulsd  xmm1,xmm2
   1e23b:	mulsd  xmm0,xmm5
   1e23f:	mulsd  xmm2,xmm4
   1e243:	movapd xmm3,xmm1
   1e247:	movapd xmm1,xmm4
   1e24b:	mulsd  xmm1,xmm5
   1e24f:	movsd  xmm5,QWORD PTR [rsp+0x18]
   1e255:	addsd  xmm0,xmm2
   1e259:	addsd  xmm5,xmm0
   1e25d:	subsd  xmm3,xmm1
   1e261:	movsd  xmm1,QWORD PTR [rsp+0x10]
   1e267:	movsd  QWORD PTR [rsp+0x18],xmm5
   1e26d:	addsd  xmm1,xmm3
   1e271:	movsd  QWORD PTR [rdx+0x68],xmm5
   1e276:	movsd  xmm5,QWORD PTR [rsi+0x58]
   1e27b:	movsd  xmm4,QWORD PTR [rdi+0x578]
   1e283:	movsd  QWORD PTR [rdx+0x60],xmm1
   1e288:	movsd  xmm0,QWORD PTR [rdi+0x570]
   1e290:	movsd  xmm2,QWORD PTR [rsi+0x50]
   1e295:	movsd  QWORD PTR [rsp+0x10],xmm1
   1e29b:	movapd xmm1,xmm4
   1e29f:	mulsd  xmm1,xmm5
   1e2a3:	movapd xmm3,xmm0
   1e2a7:	mulsd  xmm3,xmm2
   1e2ab:	mulsd  xmm0,xmm5
   1e2af:	mulsd  xmm2,xmm4
   1e2b3:	movsd  xmm4,QWORD PTR [rsp+0x28]
   1e2b9:	subsd  xmm3,xmm1
   1e2bd:	addsd  xmm3,QWORD PTR [rsp+0x20]
   1e2c3:	addsd  xmm0,xmm2
   1e2c7:	movsd  QWORD PTR [rdx+0x70],xmm3
   1e2cc:	movsd  QWORD PTR [rsp+0x20],xmm3
   1e2d2:	addsd  xmm4,xmm0
   1e2d6:	movsd  QWORD PTR [rdx+0x78],xmm4
   1e2db:	movsd  QWORD PTR [rsp+0x28],xmm4
   1e2e1:	call   17ba0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm5ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1e2e6:	movsd  xmm2,QWORD PTR [rsi+0x60]
   1e2eb:	movsd  xmm5,QWORD PTR [rsi+0x68]
   1e2f0:	movsd  xmm0,QWORD PTR [rdi+0x600]
   1e2f8:	movsd  xmm4,QWORD PTR [rdi+0x608]
   1e300:	movsd  xmm6,QWORD PTR [rsp+0x30]
   1e306:	movapd xmm1,xmm0
   1e30a:	mulsd  xmm1,xmm2
   1e30e:	mulsd  xmm0,xmm5
   1e312:	mulsd  xmm2,xmm4
   1e316:	movapd xmm3,xmm1
   1e31a:	movapd xmm1,xmm4
   1e31e:	mulsd  xmm1,xmm5
   1e322:	addsd  xmm0,xmm2
   1e326:	addsd  xmm11,xmm0
   1e32b:	subsd  xmm3,xmm1
   1e32f:	movsd  QWORD PTR [rdx+0x8],xmm11
   1e335:	movsd  xmm4,QWORD PTR [rsi+0x68]
   1e33a:	addsd  xmm6,xmm3
   1e33e:	movsd  xmm3,QWORD PTR [rdi+0x618]
   1e346:	movapd xmm5,xmm3
   1e34a:	movsd  QWORD PTR [rdx],xmm6
   1e34e:	movsd  xmm1,QWORD PTR [rsi+0x60]
   1e353:	mulsd  xmm5,xmm4
   1e357:	movsd  xmm0,QWORD PTR [rdi+0x610]
   1e35f:	movsd  QWORD PTR [rsp+0x30],xmm6
   1e365:	movapd xmm2,xmm0
   1e369:	mulsd  xmm2,xmm1
   1e36d:	mulsd  xmm0,xmm4
   1e371:	mulsd  xmm1,xmm3
   1e375:	subsd  xmm2,xmm5
   1e379:	addsd  xmm0,xmm1
   1e37d:	addsd  xmm15,xmm2
   1e382:	addsd  xmm10,xmm0
   1e387:	movsd  QWORD PTR [rdx+0x10],xmm15
   1e38d:	movsd  xmm0,QWORD PTR [rdi+0x620]
   1e395:	movsd  xmm1,QWORD PTR [rsi+0x60]
   1e39a:	movsd  QWORD PTR [rdx+0x18],xmm10
   1e3a0:	movsd  xmm4,QWORD PTR [rsi+0x68]
   1e3a5:	movapd xmm2,xmm0
   1e3a9:	movsd  xmm3,QWORD PTR [rdi+0x628]
   1e3b1:	mulsd  xmm2,xmm1
   1e3b5:	mulsd  xmm0,xmm4
   1e3b9:	movapd xmm5,xmm3
   1e3bd:	mulsd  xmm5,xmm4
   1e3c1:	mulsd  xmm1,xmm3
   1e3c5:	subsd  xmm2,xmm5
   1e3c9:	addsd  xmm0,xmm1
   1e3cd:	addsd  xmm14,xmm2
   1e3d2:	addsd  xmm9,xmm0
   1e3d7:	movsd  QWORD PTR [rdx+0x20],xmm14
   1e3dd:	movsd  xmm0,QWORD PTR [rdi+0x630]
   1e3e5:	movsd  xmm1,QWORD PTR [rsi+0x60]
   1e3ea:	movsd  QWORD PTR [rdx+0x28],xmm9
   1e3f0:	movsd  xmm3,QWORD PTR [rdi+0x638]
   1e3f8:	movsd  xmm4,QWORD PTR [rsi+0x68]
   1e3fd:	movapd xmm2,xmm0
   1e401:	mulsd  xmm2,xmm1
   1e405:	movapd xmm5,xmm3
   1e409:	mulsd  xmm5,xmm4
   1e40d:	mulsd  xmm1,xmm3
   1e411:	mulsd  xmm0,xmm4
   1e415:	subsd  xmm2,xmm5
   1e419:	addsd  xmm0,xmm1
   1e41d:	addsd  xmm13,xmm2
   1e422:	addsd  xmm8,xmm0
   1e427:	movsd  QWORD PTR [rdx+0x30],xmm13
   1e42d:	movsd  xmm0,QWORD PTR [rdi+0x640]
   1e435:	movsd  xmm1,QWORD PTR [rsi+0x60]
   1e43a:	movsd  QWORD PTR [rdx+0x38],xmm8
   1e440:	movsd  xmm4,QWORD PTR [rsi+0x68]
   1e445:	movapd xmm2,xmm0
   1e449:	movsd  xmm3,QWORD PTR [rdi+0x648]
   1e451:	mulsd  xmm2,xmm1
   1e455:	mulsd  xmm0,xmm4
   1e459:	movapd xmm5,xmm3
   1e45d:	mulsd  xmm5,xmm4
   1e461:	mulsd  xmm1,xmm3
   1e465:	subsd  xmm2,xmm5
   1e469:	addsd  xmm0,xmm1
   1e46d:	addsd  xmm12,xmm2
   1e472:	addsd  xmm7,xmm0
   1e476:	movsd  QWORD PTR [rdx+0x40],xmm12
   1e47c:	movsd  QWORD PTR [rdx+0x48],xmm7
   1e481:	movsd  xmm0,QWORD PTR [rdi+0x650]
   1e489:	movsd  xmm3,QWORD PTR [rdi+0x658]
   1e491:	movsd  xmm4,QWORD PTR [rsi+0x68]
   1e496:	movsd  xmm1,QWORD PTR [rsi+0x60]
   1e49b:	movapd xmm2,xmm0
   1e49f:	movapd xmm5,xmm3
   1e4a3:	mulsd  xmm5,xmm4
   1e4a7:	mulsd  xmm2,xmm1
   1e4ab:	mulsd  xmm0,xmm4
   1e4af:	mulsd  xmm1,xmm3
   1e4b3:	subsd  xmm2,xmm5
   1e4b7:	addsd  xmm2,QWORD PTR [rsp]
   1e4bc:	addsd  xmm0,xmm1
   1e4c0:	addsd  xmm0,QWORD PTR [rsp+0x8]
   1e4c6:	movsd  QWORD PTR [rdx+0x50],xmm2
   1e4cb:	movsd  xmm1,QWORD PTR [rsi+0x60]
   1e4d0:	movsd  QWORD PTR [rsp],xmm2
   1e4d5:	movsd  QWORD PTR [rdx+0x58],xmm0
   1e4da:	movsd  xmm3,QWORD PTR [rdi+0x668]
   1e4e2:	movsd  xmm4,QWORD PTR [rsi+0x68]
   1e4e7:	movsd  QWORD PTR [rsp+0x8],xmm0
   1e4ed:	movsd  xmm0,QWORD PTR [rdi+0x660]
   1e4f5:	movapd xmm5,xmm3
   1e4f9:	mulsd  xmm5,xmm4
   1e4fd:	movapd xmm2,xmm0
   1e501:	mulsd  xmm2,xmm1
   1e505:	mulsd  xmm0,xmm4
   1e509:	mulsd  xmm1,xmm3
   1e50d:	subsd  xmm2,xmm5
   1e511:	addsd  xmm2,QWORD PTR [rsp+0x10]
   1e517:	addsd  xmm0,xmm1
   1e51b:	movsd  xmm1,QWORD PTR [rsp+0x18]
   1e521:	movsd  QWORD PTR [rdx+0x60],xmm2
   1e526:	movsd  QWORD PTR [rsp+0x10],xmm2
   1e52c:	addsd  xmm1,xmm0
   1e530:	movsd  xmm0,QWORD PTR [rdi+0x670]
   1e538:	movapd xmm5,xmm0
   1e53c:	movsd  QWORD PTR [rdx+0x68],xmm1
   1e541:	movsd  xmm3,QWORD PTR [rdi+0x678]
   1e549:	movsd  xmm4,QWORD PTR [rsi+0x68]
   1e54e:	movsd  QWORD PTR [rsp+0x18],xmm1
   1e554:	movsd  xmm1,QWORD PTR [rsi+0x60]
   1e559:	mulsd  xmm0,xmm4
   1e55d:	mulsd  xmm5,xmm1
   1e561:	mulsd  xmm1,xmm3
   1e565:	movapd xmm2,xmm5
   1e569:	movapd xmm5,xmm3
   1e56d:	movsd  xmm3,QWORD PTR [rsp+0x28]
   1e573:	mulsd  xmm5,xmm4
   1e577:	addsd  xmm0,xmm1
   1e57b:	addsd  xmm3,xmm0
   1e57f:	subsd  xmm2,xmm5
   1e583:	movsd  xmm5,QWORD PTR [rsp+0x20]
   1e589:	movsd  QWORD PTR [rdx+0x78],xmm3
   1e58e:	movsd  QWORD PTR [rsp+0x28],xmm3
   1e594:	addsd  xmm5,xmm2
   1e598:	movsd  QWORD PTR [rdx+0x70],xmm5
   1e59d:	movsd  QWORD PTR [rsp+0x20],xmm5
   1e5a3:	call   17670 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm6ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1e5a8:	movsd  xmm1,QWORD PTR [rsi+0x70]
   1e5ad:	movsd  xmm4,QWORD PTR [rsi+0x78]
   1e5b2:	movsd  xmm0,QWORD PTR [rdi+0x700]
   1e5ba:	movsd  xmm3,QWORD PTR [rdi+0x708]
   1e5c2:	movapd xmm6,xmm0
   1e5c6:	movapd xmm5,xmm3
   1e5ca:	mulsd  xmm6,xmm1
   1e5ce:	mulsd  xmm5,xmm4
   1e5d2:	mulsd  xmm1,xmm3
   1e5d6:	mulsd  xmm0,xmm4
   1e5da:	movapd xmm2,xmm6
   1e5de:	movsd  xmm6,QWORD PTR [rsp+0x30]
   1e5e4:	subsd  xmm2,xmm5
   1e5e8:	addsd  xmm0,xmm1
   1e5ec:	addsd  xmm6,xmm2
   1e5f0:	addsd  xmm11,xmm0
   1e5f5:	movsd  QWORD PTR [rdx],xmm6
   1e5f9:	movsd  xmm0,QWORD PTR [rdi+0x710]
   1e601:	movsd  xmm1,QWORD PTR [rsi+0x70]
   1e606:	movsd  QWORD PTR [rsp+0x30],xmm6
   1e60c:	movsd  QWORD PTR [rdx+0x8],xmm11
   1e612:	movapd xmm6,xmm0
   1e616:	movsd  xmm4,QWORD PTR [rsi+0x78]
   1e61b:	movsd  xmm3,QWORD PTR [rdi+0x718]
   1e623:	mulsd  xmm6,xmm1
   1e627:	mulsd  xmm0,xmm4
   1e62b:	movapd xmm5,xmm3
   1e62f:	mulsd  xmm5,xmm4
   1e633:	mulsd  xmm1,xmm3
   1e637:	movapd xmm2,xmm6
   1e63b:	subsd  xmm2,xmm5
   1e63f:	addsd  xmm0,xmm1
   1e643:	addsd  xmm15,xmm2
   1e648:	addsd  xmm10,xmm0
   1e64d:	movsd  QWORD PTR [rdx+0x10],xmm15
   1e653:	movsd  xmm0,QWORD PTR [rdi+0x720]
   1e65b:	movsd  xmm1,QWORD PTR [rsi+0x70]
   1e660:	movsd  QWORD PTR [rdx+0x18],xmm10
   1e666:	movsd  xmm3,QWORD PTR [rdi+0x728]
   1e66e:	movsd  xmm4,QWORD PTR [rsi+0x78]
   1e673:	movapd xmm6,xmm0
   1e677:	mulsd  xmm6,xmm1
   1e67b:	movapd xmm5,xmm3
   1e67f:	mulsd  xmm5,xmm4
   1e683:	mulsd  xmm1,xmm3
   1e687:	mulsd  xmm0,xmm4
   1e68b:	movapd xmm2,xmm6
   1e68f:	subsd  xmm2,xmm5
   1e693:	addsd  xmm0,xmm1
   1e697:	addsd  xmm14,xmm2
   1e69c:	addsd  xmm9,xmm0
   1e6a1:	movsd  QWORD PTR [rdx+0x20],xmm14
   1e6a7:	movsd  xmm0,QWORD PTR [rdi+0x730]
   1e6af:	movsd  xmm1,QWORD PTR [rsi+0x70]
   1e6b4:	movsd  QWORD PTR [rdx+0x28],xmm9
   1e6ba:	movapd xmm6,xmm0
   1e6be:	movsd  xmm4,QWORD PTR [rsi+0x78]
   1e6c3:	movsd  xmm3,QWORD PTR [rdi+0x738]
   1e6cb:	mulsd  xmm6,xmm1
   1e6cf:	mulsd  xmm0,xmm4
   1e6d3:	movapd xmm5,xmm3
   1e6d7:	mulsd  xmm5,xmm4
   1e6db:	mulsd  xmm1,xmm3
   1e6df:	movapd xmm2,xmm6
   1e6e3:	subsd  xmm2,xmm5
   1e6e7:	addsd  xmm0,xmm1
   1e6eb:	addsd  xmm13,xmm2
   1e6f0:	addsd  xmm8,xmm0
   1e6f5:	movsd  QWORD PTR [rdx+0x30],xmm13
   1e6fb:	movsd  xmm0,QWORD PTR [rdi+0x740]
   1e703:	movsd  xmm1,QWORD PTR [rsi+0x70]
   1e708:	movsd  QWORD PTR [rdx+0x38],xmm8
   1e70e:	movsd  xmm3,QWORD PTR [rdi+0x748]
   1e716:	movsd  xmm4,QWORD PTR [rsi+0x78]
   1e71b:	movapd xmm6,xmm0
   1e71f:	mulsd  xmm6,xmm1
   1e723:	movapd xmm5,xmm3
   1e727:	mulsd  xmm5,xmm4
   1e72b:	mulsd  xmm1,xmm3
   1e72f:	mulsd  xmm0,xmm4
   1e733:	movapd xmm2,xmm6
   1e737:	subsd  xmm2,xmm5
   1e73b:	addsd  xmm0,xmm1
   1e73f:	addsd  xmm12,xmm2
   1e744:	addsd  xmm7,xmm0
   1e748:	movsd  QWORD PTR [rdx+0x40],xmm12
   1e74e:	movsd  QWORD PTR [rdx+0x48],xmm7
   1e753:	movsd  xmm0,QWORD PTR [rdi+0x750]
   1e75b:	movsd  xmm3,QWORD PTR [rdi+0x758]
   1e763:	movsd  xmm1,QWORD PTR [rsi+0x70]
   1e768:	movsd  xmm4,QWORD PTR [rsi+0x78]
   1e76d:	movapd xmm6,xmm0
   1e771:	movapd xmm5,xmm3
   1e775:	mulsd  xmm6,xmm1
   1e779:	mulsd  xmm5,xmm4
   1e77d:	mulsd  xmm1,xmm3
   1e781:	mulsd  xmm0,xmm4
   1e785:	movsd  xmm4,QWORD PTR [rsp+0x8]
   1e78b:	movapd xmm2,xmm6
   1e78f:	subsd  xmm2,xmm5
   1e793:	movsd  xmm5,QWORD PTR [rsp]
   1e798:	addsd  xmm0,xmm1
   1e79c:	addsd  xmm5,xmm2
   1e7a0:	addsd  xmm4,xmm0
   1e7a4:	movsd  QWORD PTR [rdx+0x50],xmm5
   1e7a9:	movsd  xmm0,QWORD PTR [rdi+0x760]
   1e7b1:	movsd  xmm1,QWORD PTR [rsi+0x70]
   1e7b6:	movsd  QWORD PTR [rsp],xmm5
   1e7bb:	movsd  QWORD PTR [rdx+0x58],xmm4
   1e7c0:	movsd  xmm3,QWORD PTR [rdi+0x768]
   1e7c8:	movapd xmm2,xmm0
   1e7cc:	movsd  QWORD PTR [rsp+0x8],xmm4
   1e7d2:	movsd  xmm4,QWORD PTR [rsi+0x78]
   1e7d7:	mulsd  xmm2,xmm1
   1e7db:	mulsd  xmm1,xmm3
   1e7df:	movapd xmm5,xmm3
   1e7e3:	mulsd  xmm5,xmm4
   1e7e7:	mulsd  xmm0,xmm4
   1e7eb:	subsd  xmm2,xmm5
   1e7ef:	addsd  xmm2,QWORD PTR [rsp+0x10]
   1e7f5:	addsd  xmm0,xmm1
   1e7f9:	addsd  xmm0,QWORD PTR [rsp+0x18]
   1e7ff:	movsd  QWORD PTR [rdx+0x60],xmm2
   1e804:	movsd  xmm1,QWORD PTR [rsi+0x70]
   1e809:	movsd  QWORD PTR [rdx+0x68],xmm0
   1e80e:	movsd  xmm3,QWORD PTR [rdi+0x778]
   1e816:	movsd  xmm4,QWORD PTR [rsi+0x78]
   1e81b:	movsd  QWORD PTR [rsp+0x18],xmm0
   1e821:	movsd  xmm0,QWORD PTR [rdi+0x770]
   1e829:	movsd  QWORD PTR [rsp+0x10],xmm2
   1e82f:	movapd xmm5,xmm3
   1e833:	mulsd  xmm5,xmm4
   1e837:	movapd xmm2,xmm0
   1e83b:	mulsd  xmm2,xmm1
   1e83f:	mulsd  xmm0,xmm4
   1e843:	mulsd  xmm1,xmm3
   1e847:	subsd  xmm2,xmm5
   1e84b:	addsd  xmm2,QWORD PTR [rsp+0x20]
   1e851:	addsd  xmm0,xmm1
   1e855:	movsd  xmm1,QWORD PTR [rsp+0x28]
   1e85b:	movsd  QWORD PTR [rdx+0x70],xmm2
   1e860:	movsd  QWORD PTR [rsp+0x20],xmm2
   1e866:	addsd  xmm1,xmm0
   1e86a:	movsd  QWORD PTR [rdx+0x78],xmm1
   1e86f:	movsd  QWORD PTR [rsp+0x28],xmm1
   1e875:	call   17140 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm7ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1e87a:	movsd  xmm3,QWORD PTR [rdi+0x808]
   1e882:	movsd  xmm0,QWORD PTR [rdi+0x800]
   1e88a:	movsd  xmm4,QWORD PTR [rsi+0x88]
   1e892:	movsd  xmm1,QWORD PTR [rsi+0x80]
   1e89a:	movapd xmm2,xmm0
   1e89e:	movapd xmm5,xmm3
   1e8a2:	movsd  xmm6,QWORD PTR [rsp+0x30]
   1e8a8:	mulsd  xmm5,xmm4
   1e8ac:	mulsd  xmm2,xmm1
   1e8b0:	mulsd  xmm0,xmm4
   1e8b4:	mulsd  xmm1,xmm3
   1e8b8:	subsd  xmm2,xmm5
   1e8bc:	addsd  xmm0,xmm1
   1e8c0:	addsd  xmm6,xmm2
   1e8c4:	addsd  xmm11,xmm0
   1e8c9:	movsd  QWORD PTR [rdx],xmm6
   1e8cd:	movsd  xmm0,QWORD PTR [rdi+0x810]
   1e8d5:	movsd  xmm1,QWORD PTR [rsi+0x80]
   1e8dd:	movsd  QWORD PTR [rsp+0x30],xmm6
   1e8e3:	movsd  QWORD PTR [rdx+0x8],xmm11
   1e8e9:	movsd  xmm3,QWORD PTR [rdi+0x818]
   1e8f1:	movapd xmm2,xmm0
   1e8f5:	movsd  xmm4,QWORD PTR [rsi+0x88]
   1e8fd:	mulsd  xmm2,xmm1
   1e901:	mulsd  xmm0,xmm4
   1e905:	movapd xmm5,xmm3
   1e909:	mulsd  xmm5,xmm4
   1e90d:	mulsd  xmm1,xmm3
   1e911:	subsd  xmm2,xmm5
   1e915:	addsd  xmm0,xmm1
   1e919:	addsd  xmm15,xmm2
   1e91e:	addsd  xmm10,xmm0
   1e923:	movsd  QWORD PTR [rdx+0x10],xmm15
   1e929:	movsd  xmm0,QWORD PTR [rdi+0x820]
   1e931:	movsd  xmm1,QWORD PTR [rsi+0x80]
   1e939:	movsd  QWORD PTR [rdx+0x18],xmm10
   1e93f:	movsd  xmm3,QWORD PTR [rdi+0x828]
   1e947:	movsd  xmm4,QWORD PTR [rsi+0x88]
   1e94f:	movapd xmm2,xmm0
   1e953:	mulsd  xmm2,xmm1
   1e957:	movapd xmm5,xmm3
   1e95b:	mulsd  xmm5,xmm4
   1e95f:	mulsd  xmm1,xmm3
   1e963:	mulsd  xmm0,xmm4
   1e967:	subsd  xmm2,xmm5
   1e96b:	addsd  xmm0,xmm1
   1e96f:	addsd  xmm14,xmm2
   1e974:	addsd  xmm9,xmm0
   1e979:	movsd  QWORD PTR [rdx+0x20],xmm14
   1e97f:	movsd  xmm0,QWORD PTR [rdi+0x830]
   1e987:	movsd  xmm1,QWORD PTR [rsi+0x80]
   1e98f:	movsd  QWORD PTR [rdx+0x28],xmm9
   1e995:	movsd  xmm3,QWORD PTR [rdi+0x838]
   1e99d:	movapd xmm2,xmm0
   1e9a1:	movsd  xmm4,QWORD PTR [rsi+0x88]
   1e9a9:	mulsd  xmm2,xmm1
   1e9ad:	mulsd  xmm0,xmm4
   1e9b1:	movapd xmm5,xmm3
   1e9b5:	mulsd  xmm5,xmm4
   1e9b9:	mulsd  xmm1,xmm3
   1e9bd:	subsd  xmm2,xmm5
   1e9c1:	addsd  xmm0,xmm1
   1e9c5:	addsd  xmm13,xmm2
   1e9ca:	addsd  xmm8,xmm0
   1e9cf:	movsd  QWORD PTR [rdx+0x30],xmm13
   1e9d5:	movsd  xmm0,QWORD PTR [rdi+0x840]
   1e9dd:	movsd  xmm1,QWORD PTR [rsi+0x80]
   1e9e5:	movsd  QWORD PTR [rdx+0x38],xmm8
   1e9eb:	movsd  xmm3,QWORD PTR [rdi+0x848]
   1e9f3:	movsd  xmm4,QWORD PTR [rsi+0x88]
   1e9fb:	movapd xmm2,xmm0
   1e9ff:	mulsd  xmm2,xmm1
   1ea03:	movapd xmm5,xmm3
   1ea07:	mulsd  xmm5,xmm4
   1ea0b:	mulsd  xmm1,xmm3
   1ea0f:	mulsd  xmm0,xmm4
   1ea13:	subsd  xmm2,xmm5
   1ea17:	addsd  xmm0,xmm1
   1ea1b:	addsd  xmm12,xmm2
   1ea20:	addsd  xmm7,xmm0
   1ea24:	movsd  QWORD PTR [rdx+0x40],xmm12
   1ea2a:	movsd  QWORD PTR [rdx+0x48],xmm7
   1ea2f:	movsd  xmm0,QWORD PTR [rdi+0x850]
   1ea37:	movsd  xmm3,QWORD PTR [rdi+0x858]
   1ea3f:	movsd  xmm4,QWORD PTR [rsi+0x88]
   1ea47:	movsd  xmm1,QWORD PTR [rsi+0x80]
   1ea4f:	movapd xmm2,xmm0
   1ea53:	movapd xmm5,xmm3
   1ea57:	mulsd  xmm2,xmm1
   1ea5b:	mulsd  xmm0,xmm4
   1ea5f:	mulsd  xmm1,xmm3
   1ea63:	movsd  xmm3,QWORD PTR [rsp+0x8]
   1ea69:	mulsd  xmm5,xmm4
   1ea6d:	addsd  xmm0,xmm1
   1ea71:	subsd  xmm2,xmm5
   1ea75:	addsd  xmm2,QWORD PTR [rsp]
   1ea7a:	addsd  xmm3,xmm0
   1ea7e:	movsd  QWORD PTR [rdx+0x50],xmm2
   1ea83:	movsd  xmm0,QWORD PTR [rdi+0x860]
   1ea8b:	movsd  xmm1,QWORD PTR [rsi+0x80]
   1ea93:	movsd  QWORD PTR [rsp],xmm2
   1ea98:	movsd  QWORD PTR [rdx+0x58],xmm3
   1ea9d:	movsd  xmm4,QWORD PTR [rsi+0x88]
   1eaa5:	movapd xmm2,xmm0
   1eaa9:	movsd  QWORD PTR [rsp+0x8],xmm3
   1eaaf:	movsd  xmm3,QWORD PTR [rdi+0x868]
   1eab7:	mulsd  xmm2,xmm1
   1eabb:	mulsd  xmm0,xmm4
   1eabf:	mulsd  xmm1,xmm3
   1eac3:	movapd xmm5,xmm3
   1eac7:	mulsd  xmm5,xmm4
   1eacb:	movsd  xmm4,QWORD PTR [rsp+0x18]
   1ead1:	addsd  xmm0,xmm1
   1ead5:	subsd  xmm2,xmm5
   1ead9:	addsd  xmm2,QWORD PTR [rsp+0x10]
   1eadf:	addsd  xmm4,xmm0
   1eae3:	movsd  QWORD PTR [rdx+0x60],xmm2
   1eae8:	movsd  xmm0,QWORD PTR [rdi+0x870]
   1eaf0:	movsd  xmm1,QWORD PTR [rsi+0x80]
   1eaf8:	movsd  QWORD PTR [rsp+0x10],xmm2
   1eafe:	movsd  QWORD PTR [rdx+0x68],xmm4
   1eb03:	movsd  xmm3,QWORD PTR [rdi+0x878]
   1eb0b:	movapd xmm2,xmm0
   1eb0f:	movsd  QWORD PTR [rsp+0x18],xmm4
   1eb15:	movsd  xmm4,QWORD PTR [rsi+0x88]
   1eb1d:	mulsd  xmm2,xmm1
   1eb21:	mulsd  xmm1,xmm3
   1eb25:	movapd xmm5,xmm3
   1eb29:	mulsd  xmm5,xmm4
   1eb2d:	mulsd  xmm0,xmm4
   1eb31:	subsd  xmm2,xmm5
   1eb35:	addsd  xmm2,QWORD PTR [rsp+0x20]
   1eb3b:	addsd  xmm0,xmm1
   1eb3f:	addsd  xmm0,QWORD PTR [rsp+0x28]
   1eb45:	movsd  QWORD PTR [rdx+0x70],xmm2
   1eb4a:	movsd  QWORD PTR [rdx+0x78],xmm0
   1eb4f:	movsd  QWORD PTR [rsp+0x20],xmm2
   1eb55:	movsd  QWORD PTR [rsp+0x28],xmm0
   1eb5b:	call   16be0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm8ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1eb60:	movsd  xmm3,QWORD PTR [rdi+0x908]
   1eb68:	movsd  xmm0,QWORD PTR [rdi+0x900]
   1eb70:	movsd  xmm4,QWORD PTR [rsi+0x98]
   1eb78:	movsd  xmm1,QWORD PTR [rsi+0x90]
   1eb80:	movapd xmm2,xmm0
   1eb84:	movapd xmm5,xmm3
   1eb88:	movsd  xmm6,QWORD PTR [rsp+0x30]
   1eb8e:	mulsd  xmm5,xmm4
   1eb92:	mulsd  xmm2,xmm1
   1eb96:	mulsd  xmm0,xmm4
   1eb9a:	mulsd  xmm1,xmm3
   1eb9e:	subsd  xmm2,xmm5
   1eba2:	addsd  xmm0,xmm1
   1eba6:	addsd  xmm6,xmm2
   1ebaa:	addsd  xmm11,xmm0
   1ebaf:	movsd  QWORD PTR [rdx],xmm6
   1ebb3:	movsd  xmm0,QWORD PTR [rdi+0x910]
   1ebbb:	movsd  xmm1,QWORD PTR [rsi+0x90]
   1ebc3:	movsd  QWORD PTR [rsp+0x30],xmm6
   1ebc9:	movsd  QWORD PTR [rdx+0x8],xmm11
   1ebcf:	movsd  xmm3,QWORD PTR [rdi+0x918]
   1ebd7:	movapd xmm2,xmm0
   1ebdb:	movsd  xmm4,QWORD PTR [rsi+0x98]
   1ebe3:	mulsd  xmm2,xmm1
   1ebe7:	mulsd  xmm0,xmm4
   1ebeb:	movapd xmm5,xmm3
   1ebef:	mulsd  xmm5,xmm4
   1ebf3:	mulsd  xmm1,xmm3
   1ebf7:	subsd  xmm2,xmm5
   1ebfb:	addsd  xmm0,xmm1
   1ebff:	addsd  xmm15,xmm2
   1ec04:	addsd  xmm10,xmm0
   1ec09:	movsd  QWORD PTR [rdx+0x10],xmm15
   1ec0f:	movsd  xmm0,QWORD PTR [rdi+0x920]
   1ec17:	movsd  xmm1,QWORD PTR [rsi+0x90]
   1ec1f:	movsd  QWORD PTR [rdx+0x18],xmm10
   1ec25:	movsd  xmm3,QWORD PTR [rdi+0x928]
   1ec2d:	movsd  xmm4,QWORD PTR [rsi+0x98]
   1ec35:	movapd xmm2,xmm0
   1ec39:	mulsd  xmm2,xmm1
   1ec3d:	movapd xmm5,xmm3
   1ec41:	mulsd  xmm5,xmm4
   1ec45:	mulsd  xmm1,xmm3
   1ec49:	mulsd  xmm0,xmm4
   1ec4d:	subsd  xmm2,xmm5
   1ec51:	addsd  xmm0,xmm1
   1ec55:	addsd  xmm14,xmm2
   1ec5a:	addsd  xmm9,xmm0
   1ec5f:	movsd  QWORD PTR [rdx+0x20],xmm14
   1ec65:	movsd  xmm0,QWORD PTR [rdi+0x930]
   1ec6d:	movsd  xmm1,QWORD PTR [rsi+0x90]
   1ec75:	movsd  QWORD PTR [rdx+0x28],xmm9
   1ec7b:	movsd  xmm3,QWORD PTR [rdi+0x938]
   1ec83:	movapd xmm2,xmm0
   1ec87:	movsd  xmm4,QWORD PTR [rsi+0x98]
   1ec8f:	mulsd  xmm2,xmm1
   1ec93:	mulsd  xmm0,xmm4
   1ec97:	movapd xmm5,xmm3
   1ec9b:	mulsd  xmm5,xmm4
   1ec9f:	mulsd  xmm1,xmm3
   1eca3:	subsd  xmm2,xmm5
   1eca7:	addsd  xmm0,xmm1
   1ecab:	addsd  xmm13,xmm2
   1ecb0:	addsd  xmm8,xmm0
   1ecb5:	movsd  QWORD PTR [rdx+0x30],xmm13
   1ecbb:	movsd  xmm0,QWORD PTR [rdi+0x940]
   1ecc3:	movsd  xmm1,QWORD PTR [rsi+0x90]
   1eccb:	movsd  QWORD PTR [rdx+0x38],xmm8
   1ecd1:	movsd  xmm3,QWORD PTR [rdi+0x948]
   1ecd9:	movsd  xmm4,QWORD PTR [rsi+0x98]
   1ece1:	movapd xmm2,xmm0
   1ece5:	mulsd  xmm2,xmm1
   1ece9:	movapd xmm5,xmm3
   1eced:	mulsd  xmm5,xmm4
   1ecf1:	mulsd  xmm1,xmm3
   1ecf5:	mulsd  xmm0,xmm4
   1ecf9:	subsd  xmm2,xmm5
   1ecfd:	addsd  xmm0,xmm1
   1ed01:	addsd  xmm12,xmm2
   1ed06:	addsd  xmm7,xmm0
   1ed0a:	movsd  QWORD PTR [rdx+0x40],xmm12
   1ed10:	movsd  QWORD PTR [rdx+0x48],xmm7
   1ed15:	movsd  xmm0,QWORD PTR [rdi+0x950]
   1ed1d:	movsd  xmm3,QWORD PTR [rdi+0x958]
   1ed25:	movsd  xmm4,QWORD PTR [rsi+0x98]
   1ed2d:	movsd  xmm1,QWORD PTR [rsi+0x90]
   1ed35:	movapd xmm2,xmm0
   1ed39:	movapd xmm5,xmm3
   1ed3d:	mulsd  xmm2,xmm1
   1ed41:	mulsd  xmm0,xmm4
   1ed45:	mulsd  xmm1,xmm3
   1ed49:	mulsd  xmm5,xmm4
   1ed4d:	addsd  xmm0,xmm1
   1ed51:	movsd  xmm1,QWORD PTR [rsp+0x8]
   1ed57:	subsd  xmm2,xmm5
   1ed5b:	addsd  xmm2,QWORD PTR [rsp]
   1ed60:	addsd  xmm1,xmm0
   1ed64:	movsd  QWORD PTR [rdx+0x50],xmm2
   1ed69:	movsd  xmm0,QWORD PTR [rdi+0x960]
   1ed71:	movsd  QWORD PTR [rsp],xmm2
   1ed76:	movsd  QWORD PTR [rdx+0x58],xmm1
   1ed7b:	movsd  xmm3,QWORD PTR [rdi+0x968]
   1ed83:	movsd  xmm4,QWORD PTR [rsi+0x98]
   1ed8b:	movapd xmm2,xmm0
   1ed8f:	movsd  QWORD PTR [rsp+0x8],xmm1
   1ed95:	movsd  xmm1,QWORD PTR [rsi+0x90]
   1ed9d:	mulsd  xmm0,xmm4
   1eda1:	movapd xmm5,xmm3
   1eda5:	mulsd  xmm2,xmm1
   1eda9:	mulsd  xmm1,xmm3
   1edad:	movsd  xmm3,QWORD PTR [rsp+0x18]
   1edb3:	mulsd  xmm5,xmm4
   1edb7:	addsd  xmm0,xmm1
   1edbb:	subsd  xmm2,xmm5
   1edbf:	addsd  xmm2,QWORD PTR [rsp+0x10]
   1edc5:	addsd  xmm3,xmm0
   1edc9:	movsd  QWORD PTR [rdx+0x60],xmm2
   1edce:	movsd  xmm0,QWORD PTR [rdi+0x970]
   1edd6:	movsd  xmm1,QWORD PTR [rsi+0x90]
   1edde:	movsd  QWORD PTR [rsp+0x10],xmm2
   1ede4:	movsd  QWORD PTR [rsp+0x18],xmm3
   1edea:	movapd xmm2,xmm0
   1edee:	movsd  QWORD PTR [rdx+0x68],xmm3
   1edf3:	movsd  xmm3,QWORD PTR [rdi+0x978]
   1edfb:	mulsd  xmm2,xmm1
   1edff:	movsd  xmm4,QWORD PTR [rsi+0x98]
   1ee07:	mulsd  xmm1,xmm3
   1ee0b:	movapd xmm5,xmm3
   1ee0f:	mulsd  xmm0,xmm4
   1ee13:	mulsd  xmm5,xmm4
   1ee17:	movsd  xmm4,QWORD PTR [rsp+0x28]
   1ee1d:	addsd  xmm0,xmm1
   1ee21:	subsd  xmm2,xmm5
   1ee25:	addsd  xmm2,QWORD PTR [rsp+0x20]
   1ee2b:	addsd  xmm4,xmm0
   1ee2f:	movsd  QWORD PTR [rdx+0x70],xmm2
   1ee34:	movsd  QWORD PTR [rsp+0x20],xmm2
   1ee3a:	movsd  QWORD PTR [rdx+0x78],xmm4
   1ee3f:	movsd  QWORD PTR [rsp+0x28],xmm4
   1ee45:	call   16680 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm9ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1ee4a:	movsd  xmm3,QWORD PTR [rdi+0xa08]
   1ee52:	movsd  xmm0,QWORD PTR [rdi+0xa00]
   1ee5a:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   1ee62:	movsd  xmm1,QWORD PTR [rsi+0xa0]
   1ee6a:	movapd xmm2,xmm0
   1ee6e:	movapd xmm5,xmm3
   1ee72:	movsd  xmm6,QWORD PTR [rsp+0x30]
   1ee78:	mulsd  xmm5,xmm4
   1ee7c:	mulsd  xmm2,xmm1
   1ee80:	mulsd  xmm0,xmm4
   1ee84:	mulsd  xmm1,xmm3
   1ee88:	subsd  xmm2,xmm5
   1ee8c:	addsd  xmm0,xmm1
   1ee90:	addsd  xmm6,xmm2
   1ee94:	addsd  xmm11,xmm0
   1ee99:	movsd  QWORD PTR [rdx],xmm6
   1ee9d:	movsd  xmm0,QWORD PTR [rdi+0xa10]
   1eea5:	movsd  xmm1,QWORD PTR [rsi+0xa0]
   1eead:	movsd  QWORD PTR [rsp+0x30],xmm6
   1eeb3:	movsd  QWORD PTR [rdx+0x8],xmm11
   1eeb9:	movsd  xmm3,QWORD PTR [rdi+0xa18]
   1eec1:	movapd xmm2,xmm0
   1eec5:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   1eecd:	mulsd  xmm2,xmm1
   1eed1:	mulsd  xmm0,xmm4
   1eed5:	movapd xmm5,xmm3
   1eed9:	mulsd  xmm5,xmm4
   1eedd:	mulsd  xmm1,xmm3
   1eee1:	subsd  xmm2,xmm5
   1eee5:	addsd  xmm0,xmm1
   1eee9:	addsd  xmm15,xmm2
   1eeee:	addsd  xmm10,xmm0
   1eef3:	movsd  QWORD PTR [rdx+0x10],xmm15
   1eef9:	movsd  xmm0,QWORD PTR [rdi+0xa20]
   1ef01:	movsd  xmm1,QWORD PTR [rsi+0xa0]
   1ef09:	movsd  QWORD PTR [rdx+0x18],xmm10
   1ef0f:	movsd  xmm3,QWORD PTR [rdi+0xa28]
   1ef17:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   1ef1f:	movapd xmm2,xmm0
   1ef23:	mulsd  xmm2,xmm1
   1ef27:	movapd xmm5,xmm3
   1ef2b:	mulsd  xmm5,xmm4
   1ef2f:	mulsd  xmm1,xmm3
   1ef33:	mulsd  xmm0,xmm4
   1ef37:	subsd  xmm2,xmm5
   1ef3b:	addsd  xmm0,xmm1
   1ef3f:	addsd  xmm14,xmm2
   1ef44:	addsd  xmm9,xmm0
   1ef49:	movsd  QWORD PTR [rdx+0x20],xmm14
   1ef4f:	movsd  xmm0,QWORD PTR [rdi+0xa30]
   1ef57:	movsd  xmm1,QWORD PTR [rsi+0xa0]
   1ef5f:	movsd  QWORD PTR [rdx+0x28],xmm9
   1ef65:	movsd  xmm3,QWORD PTR [rdi+0xa38]
   1ef6d:	movapd xmm2,xmm0
   1ef71:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   1ef79:	mulsd  xmm2,xmm1
   1ef7d:	mulsd  xmm0,xmm4
   1ef81:	movapd xmm5,xmm3
   1ef85:	mulsd  xmm5,xmm4
   1ef89:	mulsd  xmm1,xmm3
   1ef8d:	subsd  xmm2,xmm5
   1ef91:	addsd  xmm0,xmm1
   1ef95:	addsd  xmm13,xmm2
   1ef9a:	addsd  xmm8,xmm0
   1ef9f:	movsd  QWORD PTR [rdx+0x30],xmm13
   1efa5:	movsd  xmm0,QWORD PTR [rdi+0xa40]
   1efad:	movsd  xmm1,QWORD PTR [rsi+0xa0]
   1efb5:	movsd  QWORD PTR [rdx+0x38],xmm8
   1efbb:	movsd  xmm3,QWORD PTR [rdi+0xa48]
   1efc3:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   1efcb:	movapd xmm2,xmm0
   1efcf:	mulsd  xmm2,xmm1
   1efd3:	movapd xmm5,xmm3
   1efd7:	mulsd  xmm5,xmm4
   1efdb:	mulsd  xmm1,xmm3
   1efdf:	mulsd  xmm0,xmm4
   1efe3:	subsd  xmm2,xmm5
   1efe7:	addsd  xmm0,xmm1
   1efeb:	addsd  xmm12,xmm2
   1eff0:	addsd  xmm7,xmm0
   1eff4:	movsd  QWORD PTR [rdx+0x40],xmm12
   1effa:	movsd  QWORD PTR [rdx+0x48],xmm7
   1efff:	movsd  xmm0,QWORD PTR [rdi+0xa50]
   1f007:	movsd  xmm3,QWORD PTR [rdi+0xa58]
   1f00f:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   1f017:	movsd  xmm1,QWORD PTR [rsi+0xa0]
   1f01f:	movapd xmm2,xmm0
   1f023:	movapd xmm5,xmm3
   1f027:	mulsd  xmm5,xmm4
   1f02b:	mulsd  xmm2,xmm1
   1f02f:	mulsd  xmm0,xmm4
   1f033:	mulsd  xmm1,xmm3
   1f037:	subsd  xmm2,xmm5
   1f03b:	addsd  xmm2,QWORD PTR [rsp]
   1f040:	addsd  xmm0,xmm1
   1f044:	addsd  xmm0,QWORD PTR [rsp+0x8]
   1f04a:	movsd  QWORD PTR [rdx+0x50],xmm2
   1f04f:	movsd  xmm1,QWORD PTR [rsi+0xa0]
   1f057:	movsd  QWORD PTR [rsp],xmm2
   1f05c:	movsd  QWORD PTR [rdx+0x58],xmm0
   1f061:	movsd  xmm3,QWORD PTR [rdi+0xa68]
   1f069:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   1f071:	movsd  QWORD PTR [rsp+0x8],xmm0
   1f077:	movsd  xmm0,QWORD PTR [rdi+0xa60]
   1f07f:	movapd xmm5,xmm3
   1f083:	mulsd  xmm5,xmm4
   1f087:	movapd xmm2,xmm0
   1f08b:	mulsd  xmm2,xmm1
   1f08f:	mulsd  xmm0,xmm4
   1f093:	mulsd  xmm1,xmm3
   1f097:	subsd  xmm2,xmm5
   1f09b:	addsd  xmm2,QWORD PTR [rsp+0x10]
   1f0a1:	addsd  xmm0,xmm1
   1f0a5:	movsd  xmm1,QWORD PTR [rsp+0x18]
   1f0ab:	movsd  QWORD PTR [rdx+0x60],xmm2
   1f0b0:	movsd  QWORD PTR [rsp+0x10],xmm2
   1f0b6:	addsd  xmm1,xmm0
   1f0ba:	movsd  xmm0,QWORD PTR [rdi+0xa70]
   1f0c2:	movapd xmm2,xmm0
   1f0c6:	movsd  QWORD PTR [rdx+0x68],xmm1
   1f0cb:	movsd  xmm3,QWORD PTR [rdi+0xa78]
   1f0d3:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   1f0db:	movsd  QWORD PTR [rsp+0x18],xmm1
   1f0e1:	movsd  xmm1,QWORD PTR [rsi+0xa0]
   1f0e9:	mulsd  xmm0,xmm4
   1f0ed:	movapd xmm5,xmm3
   1f0f1:	mulsd  xmm2,xmm1
   1f0f5:	mulsd  xmm1,xmm3
   1f0f9:	movsd  xmm3,QWORD PTR [rsp+0x28]
   1f0ff:	mulsd  xmm5,xmm4
   1f103:	addsd  xmm0,xmm1
   1f107:	subsd  xmm2,xmm5
   1f10b:	addsd  xmm2,QWORD PTR [rsp+0x20]
   1f111:	addsd  xmm3,xmm0
   1f115:	movsd  QWORD PTR [rdx+0x70],xmm2
   1f11a:	movsd  QWORD PTR [rsp+0x20],xmm2
   1f120:	movsd  QWORD PTR [rdx+0x78],xmm3
   1f125:	movsd  QWORD PTR [rsp+0x28],xmm3
   1f12b:	call   16120 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm10ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1f130:	movsd  xmm3,QWORD PTR [rdi+0xb08]
   1f138:	movsd  xmm0,QWORD PTR [rdi+0xb00]
   1f140:	movsd  xmm4,QWORD PTR [rsi+0xb8]
   1f148:	movsd  xmm1,QWORD PTR [rsi+0xb0]
   1f150:	movapd xmm2,xmm0
   1f154:	movapd xmm5,xmm3
   1f158:	mulsd  xmm5,xmm4
   1f15c:	mulsd  xmm2,xmm1
   1f160:	mulsd  xmm0,xmm4
   1f164:	mulsd  xmm1,xmm3
   1f168:	subsd  xmm2,xmm5
   1f16c:	addsd  xmm2,QWORD PTR [rsp+0x30]
   1f172:	addsd  xmm0,xmm1
   1f176:	movsd  QWORD PTR [rdx],xmm2
   1f17a:	movsd  xmm1,QWORD PTR [rsi+0xb0]
   1f182:	movsd  QWORD PTR [rsp+0x30],xmm2
   1f188:	movapd xmm4,xmm0
   1f18c:	movsd  xmm0,QWORD PTR [rdi+0xb10]
   1f194:	addsd  xmm4,xmm11
   1f199:	movapd xmm11,xmm0
   1f19e:	mulsd  xmm11,xmm1
   1f1a3:	movsd  QWORD PTR [rdx+0x8],xmm4
   1f1a8:	movsd  xmm3,QWORD PTR [rdi+0xb18]
   1f1b0:	movsd  QWORD PTR [rsp+0x38],xmm4
   1f1b6:	movsd  xmm4,QWORD PTR [rsi+0xb8]
   1f1be:	mulsd  xmm1,xmm3
   1f1c2:	mulsd  xmm0,xmm4
   1f1c6:	movapd xmm2,xmm11
   1f1cb:	movapd xmm11,xmm3
   1f1d0:	mulsd  xmm11,xmm4
   1f1d5:	addsd  xmm0,xmm1
   1f1d9:	subsd  xmm2,xmm11
   1f1de:	movapd xmm11,xmm2
   1f1e3:	addsd  xmm11,xmm15
   1f1e8:	movapd xmm15,xmm0
   1f1ed:	addsd  xmm15,xmm10
   1f1f2:	movsd  QWORD PTR [rdx+0x10],xmm11
   1f1f8:	movsd  xmm0,QWORD PTR [rdi+0xb20]
   1f200:	movsd  xmm1,QWORD PTR [rsi+0xb0]
   1f208:	movsd  QWORD PTR [rdx+0x18],xmm15
   1f20e:	movsd  xmm3,QWORD PTR [rdi+0xb28]
   1f216:	movsd  xmm4,QWORD PTR [rsi+0xb8]
   1f21e:	movapd xmm10,xmm0
   1f223:	movsd  QWORD PTR [rsp+0x40],xmm11
   1f22a:	mulsd  xmm10,xmm1
   1f22f:	movsd  QWORD PTR [rsp+0x48],xmm15
   1f236:	mulsd  xmm0,xmm4
   1f23a:	mulsd  xmm1,xmm3
   1f23e:	movapd xmm2,xmm10
   1f243:	movapd xmm10,xmm3
   1f248:	mulsd  xmm10,xmm4
   1f24d:	addsd  xmm0,xmm1
   1f251:	subsd  xmm2,xmm10
   1f256:	movapd xmm10,xmm2
   1f25b:	addsd  xmm10,xmm14
   1f260:	movapd xmm14,xmm0
   1f265:	addsd  xmm14,xmm9
   1f26a:	movsd  QWORD PTR [rdx+0x20],xmm10
   1f270:	movsd  xmm0,QWORD PTR [rsi+0xb0]
   1f278:	movsd  xmm11,QWORD PTR [rdi+0xb30]
   1f281:	movsd  QWORD PTR [rdx+0x28],xmm14
   1f287:	movsd  xmm2,QWORD PTR [rdi+0xb38]
   1f28f:	movsd  xmm3,QWORD PTR [rsi+0xb8]
   1f297:	movapd xmm9,xmm11
   1f29c:	movsd  QWORD PTR [rsp+0x58],xmm14
   1f2a3:	mulsd  xmm9,xmm0
   1f2a8:	movsd  QWORD PTR [rsp+0x50],xmm10
   1f2af:	mulsd  xmm0,xmm2
   1f2b3:	mulsd  xmm11,xmm3
   1f2b8:	movapd xmm1,xmm9
   1f2bd:	movapd xmm9,xmm2
   1f2c2:	mulsd  xmm9,xmm3
   1f2c7:	addsd  xmm11,xmm0
   1f2cc:	addsd  xmm11,xmm8
   1f2d1:	subsd  xmm1,xmm9
   1f2d6:	movsd  QWORD PTR [rdx+0x38],xmm11
   1f2dc:	addsd  xmm1,xmm13
   1f2e1:	movsd  QWORD PTR [rsp+0x60],xmm1
   1f2e7:	movsd  QWORD PTR [rdx+0x30],xmm1
   1f2ec:	movsd  xmm10,QWORD PTR [rdi+0xb40]
   1f2f5:	movsd  xmm0,QWORD PTR [rsi+0xb0]
   1f2fd:	movsd  xmm2,QWORD PTR [rdi+0xb48]
   1f305:	movsd  xmm3,QWORD PTR [rsi+0xb8]
   1f30d:	movapd xmm8,xmm10
   1f312:	mulsd  xmm8,xmm0
   1f317:	mulsd  xmm10,xmm3
   1f31c:	mulsd  xmm0,xmm2
   1f320:	movapd xmm15,xmm8
   1f325:	movapd xmm8,xmm2
   1f32a:	mulsd  xmm8,xmm3
   1f32f:	addsd  xmm10,xmm0
   1f334:	addsd  xmm10,xmm7
   1f339:	subsd  xmm15,xmm8
   1f33e:	movsd  QWORD PTR [rdx+0x48],xmm10
   1f344:	movsd  xmm2,QWORD PTR [rdi+0xb58]
   1f34c:	movsd  xmm3,QWORD PTR [rsi+0xb8]
   1f354:	addsd  xmm15,xmm12
   1f359:	movsd  QWORD PTR [rdx+0x40],xmm15
   1f35f:	movsd  xmm0,QWORD PTR [rsi+0xb0]
   1f367:	movsd  xmm9,QWORD PTR [rdi+0xb50]
   1f370:	movapd xmm8,xmm9
   1f375:	mulsd  xmm8,xmm0
   1f37a:	mulsd  xmm9,xmm3
   1f37f:	mulsd  xmm0,xmm2
   1f383:	movapd xmm14,xmm8
   1f388:	movapd xmm8,xmm2
   1f38d:	mulsd  xmm8,xmm3
   1f392:	addsd  xmm9,xmm0
   1f397:	addsd  xmm9,QWORD PTR [rsp+0x8]
   1f39e:	movsd  QWORD PTR [rdx+0x58],xmm9
   1f3a4:	movsd  xmm2,QWORD PTR [rdi+0xb68]
   1f3ac:	movsd  xmm3,QWORD PTR [rsi+0xb8]
   1f3b4:	subsd  xmm14,xmm8
   1f3b9:	addsd  xmm14,QWORD PTR [rsp]
   1f3bf:	movapd xmm12,xmm2
   1f3c4:	mulsd  xmm12,xmm3
   1f3c9:	movsd  QWORD PTR [rdx+0x50],xmm14
   1f3cf:	movsd  xmm0,QWORD PTR [rsi+0xb0]
   1f3d7:	movsd  xmm8,QWORD PTR [rdi+0xb60]
   1f3e0:	movapd xmm13,xmm8
   1f3e5:	mulsd  xmm13,xmm0
   1f3ea:	mulsd  xmm8,xmm3
   1f3ef:	mulsd  xmm0,xmm2
   1f3f3:	subsd  xmm13,xmm12
   1f3f8:	addsd  xmm13,QWORD PTR [rsp+0x10]
   1f3ff:	addsd  xmm8,xmm0
   1f404:	addsd  xmm8,QWORD PTR [rsp+0x18]
   1f40b:	movsd  QWORD PTR [rdx+0x60],xmm13
   1f411:	movsd  xmm7,QWORD PTR [rdi+0xb70]
   1f419:	movsd  xmm0,QWORD PTR [rsi+0xb0]
   1f421:	movsd  QWORD PTR [rdx+0x68],xmm8
   1f427:	movsd  xmm2,QWORD PTR [rdi+0xb78]
   1f42f:	movapd xmm12,xmm7
   1f434:	movsd  xmm3,QWORD PTR [rsi+0xb8]
   1f43c:	mulsd  xmm12,xmm0
   1f441:	mulsd  xmm7,xmm3
   1f445:	movapd xmm1,xmm2
   1f449:	mulsd  xmm1,xmm3
   1f44d:	mulsd  xmm0,xmm2
   1f451:	subsd  xmm12,xmm1
   1f456:	addsd  xmm12,QWORD PTR [rsp+0x20]
   1f45d:	addsd  xmm7,xmm0
   1f461:	addsd  xmm7,QWORD PTR [rsp+0x28]
   1f467:	movsd  QWORD PTR [rdx+0x70],xmm12
   1f46d:	movsd  QWORD PTR [rdx+0x78],xmm7
   1f472:	call   15bc0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm11ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1f477:	movsd  xmm4,QWORD PTR [rdi+0xc08]
   1f47f:	movsd  xmm0,QWORD PTR [rdi+0xc00]
   1f487:	movsd  xmm5,QWORD PTR [rsi+0xc8]
   1f48f:	movsd  xmm3,QWORD PTR [rsi+0xc0]
   1f497:	movapd xmm2,xmm0
   1f49b:	movapd xmm1,xmm4
   1f49f:	mulsd  xmm1,xmm5
   1f4a3:	mulsd  xmm2,xmm3
   1f4a7:	mulsd  xmm0,xmm5
   1f4ab:	mulsd  xmm3,xmm4
   1f4af:	subsd  xmm2,xmm1
   1f4b3:	addsd  xmm2,QWORD PTR [rsp+0x30]
   1f4b9:	addsd  xmm0,xmm3
   1f4bd:	addsd  xmm0,QWORD PTR [rsp+0x38]
   1f4c3:	movsd  QWORD PTR [rdx],xmm2
   1f4c7:	movsd  xmm5,QWORD PTR [rdi+0xc10]
   1f4cf:	movsd  xmm3,QWORD PTR [rsi+0xc0]
   1f4d7:	movsd  QWORD PTR [rdx+0x8],xmm0
   1f4dc:	movsd  xmm4,QWORD PTR [rdi+0xc18]
   1f4e4:	movapd xmm2,xmm5
   1f4e8:	movsd  xmm0,QWORD PTR [rsi+0xc8]
   1f4f0:	mulsd  xmm2,xmm3
   1f4f4:	mulsd  xmm3,xmm4
   1f4f8:	movapd xmm1,xmm4
   1f4fc:	mulsd  xmm1,xmm0
   1f500:	mulsd  xmm0,xmm5
   1f504:	subsd  xmm2,xmm1
   1f508:	addsd  xmm2,QWORD PTR [rsp+0x40]
   1f50e:	addsd  xmm0,xmm3
   1f512:	addsd  xmm0,QWORD PTR [rsp+0x48]
   1f518:	movsd  QWORD PTR [rdx+0x10],xmm2
   1f51d:	movsd  xmm3,QWORD PTR [rsi+0xc0]
   1f525:	movsd  QWORD PTR [rdx+0x18],xmm0
   1f52a:	movsd  xmm0,QWORD PTR [rdi+0xc20]
   1f532:	movsd  xmm4,QWORD PTR [rdi+0xc28]
   1f53a:	movsd  xmm5,QWORD PTR [rsi+0xc8]
   1f542:	movapd xmm2,xmm0
   1f546:	movapd xmm1,xmm4
   1f54a:	mulsd  xmm1,xmm5
   1f54e:	mulsd  xmm2,xmm3
   1f552:	mulsd  xmm0,xmm5
   1f556:	mulsd  xmm3,xmm4
   1f55a:	subsd  xmm2,xmm1
   1f55e:	addsd  xmm2,QWORD PTR [rsp+0x50]
   1f564:	addsd  xmm0,xmm3
   1f568:	addsd  xmm0,QWORD PTR [rsp+0x58]
   1f56e:	movsd  QWORD PTR [rdx+0x20],xmm2
   1f573:	movsd  xmm3,QWORD PTR [rsi+0xc0]
   1f57b:	movsd  QWORD PTR [rdx+0x28],xmm0
   1f580:	movsd  xmm0,QWORD PTR [rdi+0xc30]
   1f588:	movsd  xmm4,QWORD PTR [rdi+0xc38]
   1f590:	movsd  xmm5,QWORD PTR [rsi+0xc8]
   1f598:	movapd xmm2,xmm0
   1f59c:	movapd xmm6,xmm4
   1f5a0:	mulsd  xmm2,xmm3
   1f5a4:	mulsd  xmm0,xmm5
   1f5a8:	mulsd  xmm3,xmm4
   1f5ac:	mulsd  xmm6,xmm5
   1f5b0:	addsd  xmm0,xmm3
   1f5b4:	subsd  xmm2,xmm6
   1f5b8:	addsd  xmm2,QWORD PTR [rsp+0x60]
   1f5be:	addsd  xmm0,xmm11
   1f5c3:	movsd  QWORD PTR [rdx+0x30],xmm2
   1f5c8:	movsd  QWORD PTR [rdx+0x38],xmm0
   1f5cd:	movsd  xmm0,QWORD PTR [rdi+0xc40]
   1f5d5:	movsd  xmm2,QWORD PTR [rsi+0xc0]
   1f5dd:	movsd  xmm3,QWORD PTR [rdi+0xc48]
   1f5e5:	movsd  xmm4,QWORD PTR [rsi+0xc8]
   1f5ed:	movapd xmm1,xmm0
   1f5f1:	mulsd  xmm1,xmm2
   1f5f5:	movapd xmm5,xmm3
   1f5f9:	mulsd  xmm5,xmm4
   1f5fd:	mulsd  xmm2,xmm3
   1f601:	mulsd  xmm0,xmm4
   1f605:	subsd  xmm1,xmm5
   1f609:	addsd  xmm0,xmm2
   1f60d:	addsd  xmm1,xmm15
   1f612:	addsd  xmm0,xmm10
   1f617:	movsd  QWORD PTR [rdx+0x40],xmm1
   1f61c:	movsd  xmm2,QWORD PTR [rsi+0xc0]
   1f624:	movsd  QWORD PTR [rdx+0x48],xmm0
   1f629:	movsd  xmm0,QWORD PTR [rdi+0xc50]
   1f631:	movsd  xmm3,QWORD PTR [rdi+0xc58]
   1f639:	movsd  xmm4,QWORD PTR [rsi+0xc8]
   1f641:	movapd xmm1,xmm0
   1f645:	movapd xmm5,xmm3
   1f649:	mulsd  xmm5,xmm4
   1f64d:	mulsd  xmm1,xmm2
   1f651:	mulsd  xmm0,xmm4
   1f655:	mulsd  xmm2,xmm3
   1f659:	subsd  xmm1,xmm5
   1f65d:	addsd  xmm0,xmm2
   1f661:	addsd  xmm1,xmm14
   1f666:	addsd  xmm0,xmm9
   1f66b:	movsd  QWORD PTR [rdx+0x50],xmm1
   1f670:	movsd  xmm2,QWORD PTR [rsi+0xc0]
   1f678:	movsd  QWORD PTR [rdx+0x58],xmm0
   1f67d:	movsd  xmm0,QWORD PTR [rdi+0xc60]
   1f685:	movsd  xmm3,QWORD PTR [rdi+0xc68]
   1f68d:	movsd  xmm4,QWORD PTR [rsi+0xc8]
   1f695:	movapd xmm1,xmm0
   1f699:	movapd xmm5,xmm3
   1f69d:	mulsd  xmm5,xmm4
   1f6a1:	mulsd  xmm1,xmm2
   1f6a5:	mulsd  xmm0,xmm4
   1f6a9:	mulsd  xmm2,xmm3
   1f6ad:	subsd  xmm1,xmm5
   1f6b1:	addsd  xmm0,xmm2
   1f6b5:	addsd  xmm1,xmm13
   1f6ba:	addsd  xmm0,xmm8
   1f6bf:	movsd  QWORD PTR [rdx+0x60],xmm1
   1f6c4:	movsd  xmm2,QWORD PTR [rsi+0xc0]
   1f6cc:	movsd  QWORD PTR [rdx+0x68],xmm0
   1f6d1:	movsd  xmm0,QWORD PTR [rdi+0xc70]
   1f6d9:	movsd  xmm3,QWORD PTR [rdi+0xc78]
   1f6e1:	movsd  xmm4,QWORD PTR [rsi+0xc8]
   1f6e9:	movapd xmm5,xmm3
   1f6ed:	movapd xmm1,xmm0
   1f6f1:	mulsd  xmm5,xmm4
   1f6f5:	mulsd  xmm0,xmm4
   1f6f9:	mulsd  xmm1,xmm2
   1f6fd:	mulsd  xmm2,xmm3
   1f701:	subsd  xmm1,xmm5
   1f705:	addsd  xmm0,xmm2
   1f709:	addsd  xmm1,xmm12
   1f70e:	addsd  xmm0,xmm7
   1f712:	movsd  QWORD PTR [rdx+0x70],xmm1
   1f717:	movsd  QWORD PTR [rdx+0x78],xmm0
   1f71c:	call   15660 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm12ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1f721:	call   18b30 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm13ELm0ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1f726:	call   180d0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm14ELm0ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>
   1f72b:	movsd  xmm3,QWORD PTR [rdi+0xf08]
   1f733:	movsd  xmm0,QWORD PTR [rdi+0xf00]
   1f73b:	movsd  xmm4,QWORD PTR [rsi+0xf8]
   1f743:	movsd  xmm2,QWORD PTR [rsi+0xf0]
   1f74b:	movapd xmm1,xmm0
   1f74f:	movapd xmm5,xmm3
   1f753:	mulsd  xmm5,xmm4
   1f757:	mulsd  xmm1,xmm2
   1f75b:	mulsd  xmm0,xmm4
   1f75f:	mulsd  xmm2,xmm3
   1f763:	subsd  xmm1,xmm5
   1f767:	addsd  xmm1,QWORD PTR [rdx]
   1f76b:	addsd  xmm0,xmm2
   1f76f:	addsd  xmm0,QWORD PTR [rdx+0x8]
   1f774:	movsd  QWORD PTR [rdx],xmm1
   1f778:	movsd  xmm2,QWORD PTR [rsi+0xf0]
   1f780:	movsd  QWORD PTR [rdx+0x8],xmm0
   1f785:	movsd  xmm0,QWORD PTR [rdi+0xf10]
   1f78d:	movsd  xmm3,QWORD PTR [rdi+0xf18]
   1f795:	movsd  xmm4,QWORD PTR [rsi+0xf8]
   1f79d:	movapd xmm1,xmm0
   1f7a1:	movapd xmm5,xmm3
   1f7a5:	mulsd  xmm5,xmm4
   1f7a9:	mulsd  xmm1,xmm2
   1f7ad:	mulsd  xmm0,xmm4
   1f7b1:	mulsd  xmm2,xmm3
   1f7b5:	subsd  xmm1,xmm5
   1f7b9:	addsd  xmm1,QWORD PTR [rdx+0x10]
   1f7be:	addsd  xmm0,xmm2
   1f7c2:	addsd  xmm0,QWORD PTR [rdx+0x18]
   1f7c7:	movsd  QWORD PTR [rdx+0x10],xmm1
   1f7cc:	movsd  xmm2,QWORD PTR [rsi+0xf0]
   1f7d4:	movsd  QWORD PTR [rdx+0x18],xmm0
   1f7d9:	movsd  xmm0,QWORD PTR [rdi+0xf20]
   1f7e1:	movsd  xmm3,QWORD PTR [rdi+0xf28]
   1f7e9:	movsd  xmm4,QWORD PTR [rsi+0xf8]
   1f7f1:	movapd xmm1,xmm0
   1f7f5:	movapd xmm5,xmm3
   1f7f9:	mulsd  xmm5,xmm4
   1f7fd:	mulsd  xmm1,xmm2
   1f801:	mulsd  xmm0,xmm4
   1f805:	mulsd  xmm2,xmm3
   1f809:	subsd  xmm1,xmm5
   1f80d:	addsd  xmm1,QWORD PTR [rdx+0x20]
   1f812:	addsd  xmm0,xmm2
   1f816:	addsd  xmm0,QWORD PTR [rdx+0x28]
   1f81b:	movsd  QWORD PTR [rdx+0x20],xmm1
   1f820:	movsd  xmm2,QWORD PTR [rsi+0xf0]
   1f828:	movsd  QWORD PTR [rdx+0x28],xmm0
   1f82d:	movsd  xmm0,QWORD PTR [rdi+0xf30]
   1f835:	movsd  xmm3,QWORD PTR [rdi+0xf38]
   1f83d:	movsd  xmm4,QWORD PTR [rsi+0xf8]
   1f845:	movapd xmm1,xmm0
   1f849:	movapd xmm5,xmm3
   1f84d:	mulsd  xmm5,xmm4
   1f851:	mulsd  xmm1,xmm2
   1f855:	mulsd  xmm0,xmm4
   1f859:	mulsd  xmm2,xmm3
   1f85d:	subsd  xmm1,xmm5
   1f861:	addsd  xmm1,QWORD PTR [rdx+0x30]
   1f866:	addsd  xmm0,xmm2
   1f86a:	addsd  xmm0,QWORD PTR [rdx+0x38]
   1f86f:	movsd  QWORD PTR [rdx+0x30],xmm1
   1f874:	movsd  QWORD PTR [rdx+0x38],xmm0
   1f879:	movsd  xmm0,QWORD PTR [rdi+0xf40]
   1f881:	movsd  xmm3,QWORD PTR [rdi+0xf48]
   1f889:	movsd  xmm4,QWORD PTR [rsi+0xf8]
   1f891:	movsd  xmm2,QWORD PTR [rsi+0xf0]
   1f899:	movapd xmm1,xmm0
   1f89d:	movapd xmm5,xmm3
   1f8a1:	mulsd  xmm5,xmm4
   1f8a5:	mulsd  xmm1,xmm2
   1f8a9:	mulsd  xmm0,xmm4
   1f8ad:	mulsd  xmm2,xmm3
   1f8b1:	subsd  xmm1,xmm5
   1f8b5:	addsd  xmm1,QWORD PTR [rdx+0x40]
   1f8ba:	addsd  xmm0,xmm2
   1f8be:	addsd  xmm0,QWORD PTR [rdx+0x48]
   1f8c3:	movsd  QWORD PTR [rdx+0x40],xmm1
   1f8c8:	movsd  xmm2,QWORD PTR [rsi+0xf0]
   1f8d0:	movsd  QWORD PTR [rdx+0x48],xmm0
   1f8d5:	movsd  xmm0,QWORD PTR [rdi+0xf50]
   1f8dd:	movsd  xmm3,QWORD PTR [rdi+0xf58]
   1f8e5:	movsd  xmm4,QWORD PTR [rsi+0xf8]
   1f8ed:	movapd xmm1,xmm0
   1f8f1:	movapd xmm5,xmm3
   1f8f5:	mulsd  xmm5,xmm4
   1f8f9:	mulsd  xmm1,xmm2
   1f8fd:	mulsd  xmm0,xmm4
   1f901:	mulsd  xmm2,xmm3
   1f905:	subsd  xmm1,xmm5
   1f909:	addsd  xmm1,QWORD PTR [rdx+0x50]
   1f90e:	addsd  xmm0,xmm2
   1f912:	addsd  xmm0,QWORD PTR [rdx+0x58]
   1f917:	movsd  QWORD PTR [rdx+0x50],xmm1
   1f91c:	movsd  xmm2,QWORD PTR [rsi+0xf0]
   1f924:	movsd  QWORD PTR [rdx+0x58],xmm0
   1f929:	movsd  xmm0,QWORD PTR [rdi+0xf60]
   1f931:	movsd  xmm3,QWORD PTR [rdi+0xf68]
   1f939:	movsd  xmm4,QWORD PTR [rsi+0xf8]
   1f941:	movapd xmm1,xmm0
   1f945:	movapd xmm5,xmm3
   1f949:	mulsd  xmm5,xmm4
   1f94d:	mulsd  xmm1,xmm2
   1f951:	mulsd  xmm0,xmm4
   1f955:	mulsd  xmm2,xmm3
   1f959:	subsd  xmm1,xmm5
   1f95d:	addsd  xmm1,QWORD PTR [rdx+0x60]
   1f962:	addsd  xmm0,xmm2
   1f966:	addsd  xmm0,QWORD PTR [rdx+0x68]
   1f96b:	movsd  QWORD PTR [rdx+0x60],xmm1
   1f970:	movsd  xmm2,QWORD PTR [rsi+0xf0]
   1f978:	movsd  QWORD PTR [rdx+0x68],xmm0
   1f97d:	movsd  xmm0,QWORD PTR [rdi+0xf70]
   1f985:	movsd  xmm3,QWORD PTR [rdi+0xf78]
   1f98d:	movsd  xmm4,QWORD PTR [rsi+0xf8]
   1f995:	movapd xmm1,xmm0
   1f999:	movapd xmm5,xmm3
   1f99d:	mulsd  xmm1,xmm2
   1f9a1:	mulsd  xmm5,xmm4
   1f9a5:	mulsd  xmm0,xmm4
   1f9a9:	mulsd  xmm2,xmm3
   1f9ad:	subsd  xmm1,xmm5
   1f9b1:	addsd  xmm1,QWORD PTR [rdx+0x70]
   1f9b6:	addsd  xmm0,xmm2
   1f9ba:	addsd  xmm0,QWORD PTR [rdx+0x78]
   1f9bf:	movsd  QWORD PTR [rdx+0x70],xmm1
   1f9c4:	movsd  QWORD PTR [rdx+0x78],xmm0
   1f9c9:	add    rsp,0xd8
   1f9d0:	jmp    15100 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm15ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>

Disassembly of section .fini:
