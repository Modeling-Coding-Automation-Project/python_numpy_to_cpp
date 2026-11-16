
/tmp/matrix_peak_stack_audit/current_b599/current_b599_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000002e2e0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>:
   2e2e0:	endbr64
   2e2e4:	pxor   xmm0,xmm0
   2e2e8:	test   rsi,rsi
   2e2eb:	je     2e33c <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x5c>
   2e2ed:	movsd  xmm7,QWORD PTR [rdi]
   2e2f1:	comisd xmm0,xmm7
   2e2f5:	ja     2e340 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x60>
   2e2f7:	comisd xmm7,xmm0
   2e2fb:	ja     2e6a8 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x3c8>
   2e301:	cmp    rsi,0x1
   2e305:	je     2e33c <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x5c>
   2e307:	movsd  xmm3,QWORD PTR [rdi+0x8]
   2e30c:	pxor   xmm6,xmm6
   2e310:	comisd xmm0,xmm3
   2e314:	ja     2e3e0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x100>
   2e31a:	movapd xmm4,xmm3
   2e31e:	maxsd  xmm4,xmm6
   2e322:	movapd xmm6,xmm4
   2e326:	cmp    rsi,0x2
   2e32a:	jne    2e3f2 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x112>
   2e330:	pxor   xmm0,xmm0
   2e334:	ucomisd xmm4,xmm0
   2e338:	jp     2e37b <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x9b>
   2e33a:	jne    2e37b <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x9b>
   2e33c:	ret
   2e33d:	nop    DWORD PTR [rax]
   2e340:	movapd xmm6,xmm7
   2e344:	xorpd  xmm6,XMMWORD PTR [rip+0x35444]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   2e34c:	cmp    rsi,0x1
   2e350:	je     2e3a0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xc0>
   2e352:	movsd  xmm3,QWORD PTR [rdi+0x8]
   2e357:	pxor   xmm0,xmm0
   2e35b:	comisd xmm0,xmm3
   2e35f:	jbe    2e31a <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x3a>
   2e361:	movapd xmm0,xmm3
   2e365:	xorpd  xmm0,XMMWORD PTR [rip+0x35423]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   2e36d:	maxsd  xmm0,xmm6
   2e371:	movapd xmm6,xmm0
   2e375:	cmp    rsi,0x2
   2e379:	jne    2e3f2 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x112>
   2e37b:	divsd  xmm7,xmm6
   2e37f:	pxor   xmm0,xmm0
   2e383:	divsd  xmm3,xmm6
   2e387:	movapd xmm2,xmm7
   2e38b:	mulsd  xmm2,xmm7
   2e38f:	addsd  xmm2,xmm0
   2e393:	mulsd  xmm3,xmm3
   2e397:	addsd  xmm2,xmm3
   2e39b:	jmp    2e3b0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd0>
   2e39d:	nop    DWORD PTR [rax]
   2e3a0:	movapd xmm2,xmm6
   2e3a4:	divsd  xmm2,xmm7
   2e3a8:	mulsd  xmm2,xmm2
   2e3ac:	addsd  xmm2,xmm0
   2e3b0:	pxor   xmm0,xmm0
   2e3b4:	ucomisd xmm2,xmm0
   2e3b8:	jp     2e3bc <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xdc>
   2e3ba:	je     2e3d9 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xf9>
   2e3bc:	movsd  xmm1,QWORD PTR [rip+0x3543c]        # 63800 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
   2e3c4:	movapd xmm5,xmm1
   2e3c8:	maxsd  xmm5,xmm2
   2e3cc:	movapd xmm0,xmm5
   2e3d0:	call   6a10 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   2e3d5:	mulsd  xmm0,xmm5
   2e3d9:	mulsd  xmm0,xmm6
   2e3dd:	ret
   2e3de:	xchg   ax,ax
   2e3e0:	movapd xmm6,xmm3
   2e3e4:	xorpd  xmm6,XMMWORD PTR [rip+0x353a4]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   2e3ec:	cmp    rsi,0x2
   2e3f0:	je     2e37b <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x9b>
   2e3f2:	movsd  xmm4,QWORD PTR [rdi+0x10]
   2e3f7:	pxor   xmm5,xmm5
   2e3fb:	movapd xmm1,xmm4
   2e3ff:	movapd xmm2,xmm4
   2e403:	xorpd  xmm2,XMMWORD PTR [rip+0x35385]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   2e40b:	cmpltsd xmm1,xmm5
   2e410:	andpd  xmm2,xmm1
   2e414:	andnpd xmm1,xmm4
   2e418:	orpd   xmm1,xmm2
   2e41c:	maxsd  xmm1,xmm6
   2e420:	movapd xmm6,xmm1
   2e424:	cmp    rsi,0x3
   2e428:	jne    2e470 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x190>
   2e42a:	ucomisd xmm1,xmm5
   2e42e:	movapd xmm0,xmm5
   2e432:	jp     2e43a <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x15a>
   2e434:	je     2e33c <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x5c>
   2e43a:	divsd  xmm7,xmm6
   2e43e:	pxor   xmm0,xmm0
   2e442:	divsd  xmm3,xmm6
   2e446:	movapd xmm2,xmm7
   2e44a:	mulsd  xmm2,xmm7
   2e44e:	addsd  xmm2,xmm0
   2e452:	divsd  xmm4,xmm6
   2e456:	mulsd  xmm3,xmm3
   2e45a:	addsd  xmm2,xmm3
   2e45e:	mulsd  xmm4,xmm4
   2e462:	addsd  xmm2,xmm4
   2e466:	jmp    2e3b0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd0>
   2e46b:	nop    DWORD PTR [rax+rax*1+0x0]
   2e470:	movsd  xmm5,QWORD PTR [rdi+0x18]
   2e475:	pxor   xmm2,xmm2
   2e479:	movapd xmm1,xmm5
   2e47d:	movapd xmm0,xmm5
   2e481:	xorpd  xmm0,XMMWORD PTR [rip+0x35307]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   2e489:	cmpltsd xmm1,xmm2
   2e48e:	andpd  xmm0,xmm1
   2e492:	andnpd xmm1,xmm5
   2e496:	orpd   xmm1,xmm0
   2e49a:	maxsd  xmm1,xmm6
   2e49e:	movapd xmm6,xmm1
   2e4a2:	cmp    rsi,0x4
   2e4a6:	je     2e5e8 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x308>
   2e4ac:	movsd  xmm1,QWORD PTR [rdi+0x20]
   2e4b1:	pxor   xmm8,xmm8
   2e4b6:	movapd xmm0,xmm1
   2e4ba:	movapd xmm2,xmm1
   2e4be:	xorpd  xmm2,XMMWORD PTR [rip+0x352ca]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   2e4c6:	cmpltsd xmm0,xmm8
   2e4cc:	andpd  xmm2,xmm0
   2e4d0:	andnpd xmm0,xmm1
   2e4d4:	orpd   xmm0,xmm2
   2e4d8:	maxsd  xmm0,xmm6
   2e4dc:	movapd xmm6,xmm0
   2e4e0:	cmp    rsi,0x5
   2e4e4:	je     2e6d0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x3f0>
   2e4ea:	movsd  xmm0,QWORD PTR [rdi+0x28]
   2e4ef:	pxor   xmm8,xmm8
   2e4f4:	movapd xmm9,xmm0
   2e4f9:	movapd xmm2,xmm0
   2e4fd:	xorpd  xmm2,XMMWORD PTR [rip+0x3528b]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   2e505:	cmpltsd xmm0,xmm8
   2e50b:	andpd  xmm2,xmm0
   2e50f:	andnpd xmm0,xmm9
   2e514:	orpd   xmm0,xmm2
   2e518:	maxsd  xmm0,xmm6
   2e51c:	movapd xmm6,xmm0
   2e520:	cmp    rsi,0x6
   2e524:	je     2e6d0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x3f0>
   2e52a:	movsd  xmm0,QWORD PTR [rdi+0x30]
   2e52f:	pxor   xmm8,xmm8
   2e534:	movapd xmm9,xmm0
   2e539:	movapd xmm2,xmm0
   2e53d:	xorpd  xmm2,XMMWORD PTR [rip+0x3524b]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   2e545:	cmpltsd xmm0,xmm8
   2e54b:	andpd  xmm2,xmm0
   2e54f:	andnpd xmm0,xmm9
   2e554:	orpd   xmm0,xmm2
   2e558:	maxsd  xmm0,xmm6
   2e55c:	movapd xmm6,xmm0
   2e560:	cmp    rsi,0x7
   2e564:	je     2e6d0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x3f0>
   2e56a:	movsd  xmm2,QWORD PTR [rdi+0x38]
   2e56f:	pxor   xmm0,xmm0
   2e573:	comisd xmm0,xmm2
   2e577:	ja     2e6e5 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x405>
   2e57d:	comisd xmm2,xmm6
   2e581:	ja     2e6fa <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x41a>
   2e587:	ucomisd xmm6,xmm0
   2e58b:	jp     2e597 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2b7>
   2e58d:	pxor   xmm0,xmm0
   2e591:	je     2e33c <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x5c>
   2e597:	divsd  xmm7,xmm6
   2e59b:	pxor   xmm2,xmm2
   2e59f:	movapd xmm0,xmm7
   2e5a3:	mulsd  xmm0,xmm7
   2e5a7:	addsd  xmm2,xmm0
   2e5ab:	cmp    rsi,0x1
   2e5af:	je     2e3b0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd0>
   2e5b5:	divsd  xmm3,xmm6
   2e5b9:	divsd  xmm4,xmm6
   2e5bd:	mulsd  xmm3,xmm3
   2e5c1:	addsd  xmm3,xmm2
   2e5c5:	movapd xmm0,xmm3
   2e5c9:	divsd  xmm5,xmm6
   2e5cd:	mulsd  xmm4,xmm4
   2e5d1:	addsd  xmm0,xmm4
   2e5d5:	movapd xmm2,xmm0
   2e5d9:	mulsd  xmm5,xmm5
   2e5dd:	addsd  xmm2,xmm5
   2e5e1:	jmp    2e643 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x363>
   2e5e3:	nop    DWORD PTR [rax+rax*1+0x0]
   2e5e8:	ucomisd xmm1,xmm2
   2e5ec:	movapd xmm0,xmm2
   2e5f0:	jp     2e5f8 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x318>
   2e5f2:	je     2e33c <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x5c>
   2e5f8:	divsd  xmm7,xmm6
   2e5fc:	pxor   xmm1,xmm1
   2e600:	divsd  xmm3,xmm6
   2e604:	movapd xmm0,xmm7
   2e608:	mulsd  xmm0,xmm7
   2e60c:	addsd  xmm0,xmm1
   2e610:	divsd  xmm4,xmm6
   2e614:	mulsd  xmm3,xmm3
   2e618:	addsd  xmm0,xmm3
   2e61c:	divsd  xmm5,xmm6
   2e620:	mulsd  xmm4,xmm4
   2e624:	addsd  xmm0,xmm4
   2e628:	movapd xmm2,xmm5
   2e62c:	mulsd  xmm2,xmm5
   2e630:	addsd  xmm2,xmm0
   2e634:	cmp    rsi,0x4
   2e638:	je     2e3b0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd0>
   2e63e:	movsd  xmm1,QWORD PTR [rdi+0x20]
   2e643:	divsd  xmm1,xmm6
   2e647:	mulsd  xmm1,xmm1
   2e64b:	addsd  xmm2,xmm1
   2e64f:	cmp    rsi,0x5
   2e653:	je     2e3b0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd0>
   2e659:	movsd  xmm0,QWORD PTR [rdi+0x28]
   2e65e:	divsd  xmm0,xmm6
   2e662:	mulsd  xmm0,xmm0
   2e666:	addsd  xmm2,xmm0
   2e66a:	cmp    rsi,0x6
   2e66e:	je     2e3b0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd0>
   2e674:	movsd  xmm0,QWORD PTR [rdi+0x30]
   2e679:	divsd  xmm0,xmm6
   2e67d:	mulsd  xmm0,xmm0
   2e681:	addsd  xmm2,xmm0
   2e685:	cmp    rsi,0x7
   2e689:	je     2e3b0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd0>
   2e68f:	movsd  xmm0,QWORD PTR [rdi+0x38]
   2e694:	divsd  xmm0,xmm6
   2e698:	mulsd  xmm0,xmm0
   2e69c:	addsd  xmm2,xmm0
   2e6a0:	jmp    2e3b0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd0>
   2e6a5:	nop    DWORD PTR [rax]
   2e6a8:	movapd xmm6,xmm7
   2e6ac:	cmp    rsi,0x1
   2e6b0:	jne    2e352 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x72>
   2e6b6:	movapd xmm2,xmm7
   2e6ba:	divsd  xmm2,xmm7
   2e6be:	mulsd  xmm2,xmm2
   2e6c2:	addsd  xmm2,xmm0
   2e6c6:	jmp    2e3b0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd0>
   2e6cb:	nop    DWORD PTR [rax+rax*1+0x0]
   2e6d0:	ucomisd xmm6,xmm8
   2e6d5:	movapd xmm0,xmm8
   2e6da:	jnp    2e5f2 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x312>
   2e6e0:	jmp    2e5f8 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x318>
   2e6e5:	xorpd  xmm2,XMMWORD PTR [rip+0x350a3]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   2e6ed:	maxsd  xmm2,xmm6
   2e6f1:	movapd xmm6,xmm2
   2e6f5:	jmp    2e597 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2b7>
   2e6fa:	ucomisd xmm2,xmm0
   2e6fe:	jp     2e70a <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x42a>
   2e700:	pxor   xmm0,xmm0
   2e704:	je     2e33c <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x5c>
   2e70a:	movapd xmm6,xmm2
   2e70e:	jmp    2e597 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2b7>

Disassembly of section .fini:
