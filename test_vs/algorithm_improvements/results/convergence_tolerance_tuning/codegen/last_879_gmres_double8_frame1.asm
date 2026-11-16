
/tmp/matrix_peak_stack_audit/last_879/last_879_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000008400 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0>:
    8400:	push   r15
    8402:	pxor   xmm0,xmm0
    8406:	mov    r11,rsi
    8409:	pxor   xmm5,xmm5
    840d:	push   r14
    840f:	movapd xmm1,xmm0
    8413:	mov    rax,rcx
    8416:	mov    r15,r9
    8419:	push   r13
    841b:	push   r12
    841d:	push   rbp
    841e:	push   rbx
    841f:	sub    rsp,0x1a8
    8426:	movsd  xmm2,QWORD PTR [rsi+0x18]
    842b:	movsd  xmm3,QWORD PTR [rip+0x2f3bd]        # 377f0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x70>
    8433:	mov    QWORD PTR [rsp+0x30],rdi
    8438:	movsd  xmm7,QWORD PTR [r11+0x20]
    843e:	mov    QWORD PTR [rsp+0x38],rdx
    8443:	cmpnltsd xmm1,xmm2
    8448:	movsd  xmm6,QWORD PTR [r11+0x28]
    844e:	mov    QWORD PTR [rsp+0x40],rcx
    8453:	mov    rbx,QWORD PTR [rsi+0x8]
    8457:	andpd  xmm3,xmm1
    845b:	andnpd xmm1,xmm2
    845f:	mov    rdx,QWORD PTR fs:0x28
    8468:	mov    QWORD PTR [rsp+0x198],rdx
    8470:	xor    edx,edx
    8472:	comisd xmm0,xmm2
    8476:	orpd   xmm1,xmm3
    847a:	mov    QWORD PTR [rsi+0x4a8],0x0
    8485:	mov    BYTE PTR [rsi+0x4b1],0x0
    848c:	movapd xmm2,xmm5
    8490:	movsd  QWORD PTR [rsp+0x60],xmm1
    8496:	movapd xmm1,xmm0
    849a:	cmpltsd xmm0,xmm6
    849f:	setae  BYTE PTR [rsi+0x4b2]
    84a6:	cmpltsd xmm1,xmm7
    84ab:	xor    esi,esi
    84ad:	cmp    QWORD PTR [r11+0x10],0x1
    84b2:	seta   sil
    84b6:	add    rsi,0x1
    84ba:	andpd  xmm6,xmm0
    84be:	mov    QWORD PTR [rsp+0x28],rsi
    84c3:	andpd  xmm7,xmm1
    84c7:	xchg   ax,ax
    84c9:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    84d4:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    84df:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    84ea:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    84f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8500:	cmp    rdx,r15
    8503:	jae    8533 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x133>
    8505:	movsd  xmm0,QWORD PTR [rax+rdx*8]
    850a:	movapd xmm3,xmm0
    850e:	movapd xmm1,xmm0
    8512:	xorpd  xmm1,XMMWORD PTR [rip+0x2f286]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    851a:	cmpltsd xmm0,xmm2
    851f:	andpd  xmm1,xmm0
    8523:	andnpd xmm0,xmm3
    8527:	orpd   xmm0,xmm1
    852b:	maxsd  xmm0,xmm5
    852f:	movapd xmm5,xmm0
    8533:	add    rdx,0x1
    8537:	cmp    rdx,0x8
    853b:	jne    8500 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x100>
    853d:	pxor   xmm0,xmm0
    8541:	ucomisd xmm5,xmm0
    8545:	jp     8549 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x149>
    8547:	je     8598 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x198>
    8549:	mov    rax,QWORD PTR [rsp+0x40]
    854e:	pxor   xmm2,xmm2
    8552:	xor    edx,edx
    8554:	nop
    8555:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8560:	cmp    rdx,r15
    8563:	jae    8576 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x176>
    8565:	movsd  xmm0,QWORD PTR [rax+rdx*8]
    856a:	divsd  xmm0,xmm5
    856e:	mulsd  xmm0,xmm0
    8572:	addsd  xmm2,xmm0
    8576:	add    rdx,0x1
    857a:	cmp    rdx,0x8
    857e:	jne    8560 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x160>
    8580:	pxor   xmm0,xmm0
    8584:	ucomisd xmm2,xmm0
    8588:	jp     90d3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xcd3>
    858e:	jne    90d3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xcd3>
    8594:	mulsd  xmm0,xmm5
    8598:	mov    rdx,QWORD PTR [rsp+0x30]
    859d:	mulsd  xmm7,xmm0
    85a1:	movdqu xmm0,XMMWORD PTR [r8]
    85a6:	lea    r14,[rsp+0x90]
    85ae:	mov    rcx,QWORD PTR [rsp+0x38]
    85b3:	mov    QWORD PTR [rsp+0x8],r11
    85b8:	movups XMMWORD PTR [rdx],xmm0
    85bb:	movdqu xmm0,XMMWORD PTR [r8+0x10]
    85c1:	mov    rdi,QWORD PTR [rcx]
    85c4:	mov    rsi,QWORD PTR [rcx+0x8]
    85c8:	mov    rcx,r14
    85cb:	mov    QWORD PTR [rsp+0x18],r14
    85d0:	movups XMMWORD PTR [rdx+0x10],xmm0
    85d4:	movdqu xmm0,XMMWORD PTR [r8+0x20]
    85da:	addsd  xmm7,xmm6
    85de:	mov    QWORD PTR [rsp+0x50],rdi
    85e3:	movups XMMWORD PTR [rdx+0x20],xmm0
    85e7:	movdqu xmm0,XMMWORD PTR [r8+0x30]
    85ed:	mov    r8,r15
    85f0:	movups XMMWORD PTR [rdx+0x30],xmm0
    85f4:	pxor   xmm0,xmm0
    85f8:	movsd  QWORD PTR [rsp+0x48],xmm7
    85fe:	movaps XMMWORD PTR [rsp+0x90],xmm0
    8606:	movaps XMMWORD PTR [rsp+0xa0],xmm0
    860e:	movaps XMMWORD PTR [rsp+0xb0],xmm0
    8616:	movaps XMMWORD PTR [rsp+0xc0],xmm0
    861e:	call   4260 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    8623:	mov    r11,QWORD PTR [rsp+0x8]
    8628:	mov    rcx,r14
    862b:	xor    edx,edx
    862d:	mov    rax,QWORD PTR [rsp+0x40]
    8632:	nop    DWORD PTR [rax]
    8635:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8640:	pxor   xmm0,xmm0
    8644:	cmp    rdx,r15
    8647:	jae    8652 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x252>
    8649:	movsd  xmm0,QWORD PTR [rax+rdx*8]
    864e:	subsd  xmm0,QWORD PTR [rcx]
    8652:	add    rdx,0x1
    8656:	movsd  QWORD PTR [rcx],xmm0
    865a:	add    rcx,0x8
    865e:	cmp    rdx,0x8
    8662:	jne    8640 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x240>
    8664:	pxor   xmm5,xmm5
    8668:	mov    rcx,QWORD PTR [rsp+0x18]
    866d:	xor    edx,edx
    866f:	movapd xmm2,xmm5
    8673:	xchg   ax,ax
    8675:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8680:	cmp    rdx,r15
    8683:	jae    86b6 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x2b6>
    8685:	movsd  xmm0,QWORD PTR [rcx]
    8689:	movapd xmm3,xmm0
    868d:	movapd xmm1,xmm0
    8691:	xorpd  xmm1,XMMWORD PTR [rip+0x2f107]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    8699:	cmpltsd xmm0,xmm2
    869e:	andpd  xmm1,xmm0
    86a2:	andnpd xmm0,xmm3
    86a6:	orpd   xmm0,xmm1
    86aa:	movapd xmm1,xmm0
    86ae:	maxsd  xmm1,xmm5
    86b2:	movapd xmm5,xmm1
    86b6:	add    rdx,0x1
    86ba:	add    rcx,0x8
    86be:	cmp    rdx,0x8
    86c2:	jne    8680 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x280>
    86c4:	pxor   xmm0,xmm0
    86c8:	ucomisd xmm5,xmm0
    86cc:	jp     86d0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x2d0>
    86ce:	je     873b <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x33b>
    86d0:	mov    rcx,QWORD PTR [rsp+0x18]
    86d5:	pxor   xmm2,xmm2
    86d9:	xor    edx,edx
    86db:	nop    DWORD PTR [rax+0x0]
    86df:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    86ea:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    86f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8700:	cmp    rdx,r15
    8703:	jae    8715 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x315>
    8705:	movsd  xmm0,QWORD PTR [rcx]
    8709:	divsd  xmm0,xmm5
    870d:	mulsd  xmm0,xmm0
    8711:	addsd  xmm2,xmm0
    8715:	add    rdx,0x1
    8719:	add    rcx,0x8
    871d:	cmp    rdx,0x8
    8721:	jne    8700 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x300>
    8723:	pxor   xmm0,xmm0
    8727:	ucomisd xmm2,xmm0
    872b:	jp     90b1 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xcb1>
    8731:	jne    90b1 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xcb1>
    8737:	mulsd  xmm0,xmm5
    873b:	movsd  xmm4,QWORD PTR [rsp+0x48]
    8741:	mov    rsi,QWORD PTR [r11]
    8744:	movsd  QWORD PTR [r11+0x4a0],xmm0
    874d:	comisd xmm4,xmm0
    8751:	setae  dl
    8754:	mov    BYTE PTR [r11+0x4b0],dl
    875b:	test   rsi,rsi
    875e:	je     87cd <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x3cd>
    8760:	cmp    rbx,0x8
    8764:	mov    eax,0x8
    8769:	mov    ecx,0x1
    876e:	mov    rbp,r11
    8771:	cmovbe rax,rbx
    8775:	mov    rbx,rsi
    8778:	test   rax,rax
    877b:	cmove  rax,rcx
    877f:	cmp    rax,r15
    8782:	cmova  rax,r15
    8786:	test   rax,rax
    8789:	mov    QWORD PTR [rsp+0x8],rax
    878e:	lea    rax,[rsp+0xd0]
    8796:	setne  r9b
    879a:	mov    QWORD PTR [rsp+0x58],rax
    879f:	xor    r14d,r14d
    87a2:	jmp    87af <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x3af>
    87a4:	nop    DWORD PTR [rax+0x0]
    87a8:	movzx  edx,BYTE PTR [rbp+0x4b0]
    87af:	test   dl,dl
    87b1:	jne    87c4 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x3c4>
    87b3:	movzx  eax,BYTE PTR [rbp+0x4b1]
    87ba:	xor    eax,0x1
    87bd:	and    al,r9b
    87c0:	mov    edx,eax
    87c2:	jne    87fb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x3fb>
    87c4:	add    r14,0x1
    87c8:	cmp    r14,rbx
    87cb:	jne    87a8 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x3a8>
    87cd:	mov    rax,QWORD PTR [rsp+0x198]
    87d5:	sub    rax,QWORD PTR fs:0x28
    87de:	jne    9616 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1216>
    87e4:	mov    rax,QWORD PTR [rsp+0x30]
    87e9:	add    rsp,0x1a8
    87f0:	pop    rbx
    87f1:	pop    rbp
    87f2:	pop    r12
    87f4:	pop    r13
    87f6:	pop    r14
    87f8:	pop    r15
    87fa:	ret
    87fb:	movsd  xmm2,QWORD PTR [rbp+0x4a0]
    8803:	comisd xmm2,QWORD PTR [rip+0x2f17d]        # 37988 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x208>
    880b:	pxor   xmm0,xmm0
    880f:	pxor   xmm1,xmm1
    8813:	movups XMMWORD PTR [rbp+0x398],xmm0
    881a:	mov    QWORD PTR [rbp+0x3d8],0x0
    8825:	movups XMMWORD PTR [rbp+0x3a8],xmm0
    882c:	movups XMMWORD PTR [rbp+0x3b8],xmm0
    8833:	movups XMMWORD PTR [rbp+0x3c8],xmm0
    883a:	movsd  QWORD PTR [rbp+0x398],xmm2
    8842:	jb     8850 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x450>
    8844:	movsd  xmm1,QWORD PTR [rip+0x2efdc]        # 37828 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
    884c:	divsd  xmm1,xmm2
    8850:	comisd xmm2,QWORD PTR [rip+0x2f130]        # 37988 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x208>
    8858:	mov    rax,QWORD PTR [rsp+0x18]
    885d:	lea    rdi,[rbp+0x30]
    8861:	movsd  xmm0,QWORD PTR [rax]
    8865:	lea    rsi,[rax+0x40]
    8869:	jae    88c0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x4c0>
    886b:	mov    rax,QWORD PTR [rsp+0x18]
    8870:	add    rax,0x10
    8874:	divsd  xmm0,xmm2
    8878:	movsd  QWORD PTR [rdi],xmm0
    887c:	movsd  xmm0,QWORD PTR [rax-0x8]
    8881:	divsd  xmm0,xmm2
    8885:	movsd  QWORD PTR [rdi+0x8],xmm0
    888a:	cmp    rax,rsi
    888d:	je     88f0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x4f0>
    888f:	movsd  xmm0,QWORD PTR [rax]
    8893:	add    rdi,0x10
    8897:	jmp    8870 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x470>
    8899:	nop    WORD PTR [rax+rax*1+0x0]
    889f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    88aa:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    88b5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    88c0:	mulsd  xmm0,xmm1
    88c4:	add    rax,0x10
    88c8:	movsd  QWORD PTR [rdi],xmm0
    88cc:	movsd  xmm0,QWORD PTR [rax-0x8]
    88d1:	mulsd  xmm0,xmm1
    88d5:	movsd  QWORD PTR [rdi+0x8],xmm0
    88da:	cmp    rax,rsi
    88dd:	je     88f0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x4f0>
    88df:	movsd  xmm0,QWORD PTR [rax]
    88e3:	add    rdi,0x10
    88e7:	jmp    88c0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x4c0>
    88e9:	nop    DWORD PTR [rax+0x0]
    88f0:	pxor   xmm0,xmm0
    88f4:	mov    QWORD PTR [rsp+0x70],r14
    88f9:	xor    r10d,r10d
    88fc:	xor    esi,esi
    88fe:	mov    BYTE PTR [rsp+0x8e],dl
    8905:	xor    r14d,r14d
    8908:	mov    r8,r15
    890b:	lea    r13,[rbp+0x358]
    8912:	mov    QWORD PTR [rsp+0x78],rbp
    8917:	lea    r12,[rbp+0xb0]
    891e:	mov    QWORD PTR [rsp+0x80],rbx
    8926:	mov    BYTE PTR [rsp+0x8f],r9b
    892e:	movaps XMMWORD PTR [rsp+0xd0],xmm0
    8936:	movaps XMMWORD PTR [rsp+0xe0],xmm0
    893e:	movaps XMMWORD PTR [rsp+0xf0],xmm0
    8946:	movaps XMMWORD PTR [rsp+0x100],xmm0
    894e:	movaps XMMWORD PTR [rsp+0x110],xmm0
    8956:	movaps XMMWORD PTR [rsp+0x120],xmm0
    895e:	movaps XMMWORD PTR [rsp+0x130],xmm0
    8966:	movaps XMMWORD PTR [rsp+0x140],xmm0
    896e:	movsd  QWORD PTR [rsp+0x68],xmm2
    8974:	jmp    8995 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x595>
    8976:	cs nop WORD PTR [rax+rax*1+0x0]
    8980:	mov    r14,r15
    8983:	add    r13,0x8
    8987:	add    r12,0x40
    898b:	cmp    r14,0x8
    898f:	je     8ef7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xaf7>
    8995:	cmp    r14,QWORD PTR [rsp+0x8]
    899a:	lea    r15,[r14+0x1]
    899e:	setb   al
    89a1:	and    al,dl
    89a3:	mov    BYTE PTR [rsp+0x10],al
    89a7:	je     8980 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x580>
    89a9:	mov    rdx,QWORD PTR [rsp+0x58]
    89ae:	movdqu xmm0,XMMWORD PTR [r12-0x80]
    89b5:	lea    rbx,[rsp+0x110]
    89bd:	mov    QWORD PTR [rsp+0x20],r8
    89c2:	mov    rax,QWORD PTR [rsp+0x38]
    89c7:	mov    rdi,QWORD PTR [rsp+0x50]
    89cc:	mov    rcx,rbx
    89cf:	movaps XMMWORD PTR [rdx],xmm0
    89d2:	movdqu xmm0,XMMWORD PTR [r12-0x70]
    89d9:	mov    rsi,QWORD PTR [rax+0x8]
    89dd:	movaps XMMWORD PTR [rdx+0x10],xmm0
    89e1:	movdqu xmm0,XMMWORD PTR [r12-0x60]
    89e8:	movaps XMMWORD PTR [rdx+0x20],xmm0
    89ec:	movdqu xmm0,XMMWORD PTR [r12-0x50]
    89f3:	movaps XMMWORD PTR [rdx+0x30],xmm0
    89f7:	call   4260 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    89fc:	pxor   xmm6,xmm6
    8a00:	mov    r8,QWORD PTR [rsp+0x20]
    8a05:	xor    eax,eax
    8a07:	mov    r10,rbx
    8a0a:	mov    rdx,rbx
    8a0d:	movapd xmm4,xmm6
    8a11:	nop    DWORD PTR [rax]
    8a14:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8a1f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8a2a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8a35:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8a40:	cmp    rax,r8
    8a43:	jae    8a72 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x672>
    8a45:	movsd  xmm0,QWORD PTR [rdx]
    8a49:	movapd xmm1,xmm0
    8a4d:	movapd xmm2,xmm0
    8a51:	xorpd  xmm2,XMMWORD PTR [rip+0x2ed47]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    8a59:	cmpltsd xmm1,xmm4
    8a5e:	andpd  xmm2,xmm1
    8a62:	andnpd xmm1,xmm0
    8a66:	orpd   xmm1,xmm2
    8a6a:	maxsd  xmm1,xmm6
    8a6e:	movapd xmm6,xmm1
    8a72:	add    rax,0x1
    8a76:	add    rdx,0x8
    8a7a:	cmp    rax,0x8
    8a7e:	jne    8a40 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x640>
    8a80:	ucomisd xmm6,xmm4
    8a84:	jp     8a8c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x68c>
    8a86:	pxor   xmm5,xmm5
    8a8a:	je     8afb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x6fb>
    8a8c:	mov    rdx,rbx
    8a8f:	pxor   xmm3,xmm3
    8a93:	xor    eax,eax
    8a95:	cs nop WORD PTR [rax+rax*1+0x0]
    8a9f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8aaa:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8ab5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8ac0:	cmp    rax,r8
    8ac3:	jae    8ad5 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x6d5>
    8ac5:	movsd  xmm0,QWORD PTR [rdx]
    8ac9:	divsd  xmm0,xmm6
    8acd:	mulsd  xmm0,xmm0
    8ad1:	addsd  xmm3,xmm0
    8ad5:	add    rax,0x1
    8ad9:	add    rdx,0x8
    8add:	cmp    rax,0x8
    8ae1:	jne    8ac0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x6c0>
    8ae3:	ucomisd xmm3,xmm4
    8ae7:	jp     908b <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc8b>
    8aed:	pxor   xmm5,xmm5
    8af1:	jne    908b <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc8b>
    8af7:	mulsd  xmm5,xmm6
    8afb:	mov    QWORD PTR [rbp+0x390],0x0
    8b06:	pxor   xmm0,xmm0
    8b0a:	xor    ecx,ecx
    8b0c:	mov    rdx,r15
    8b0f:	movups XMMWORD PTR [rbp+0x350],xmm0
    8b16:	movups XMMWORD PTR [rbp+0x360],xmm0
    8b1d:	movups XMMWORD PTR [rbp+0x370],xmm0
    8b24:	movups XMMWORD PTR [rbp+0x380],xmm0
    8b2b:	lea    rax,[rbp+0x350]
    8b32:	lea    rdi,[rbp+0x70]
    8b36:	mov    QWORD PTR [rsp+0x20],rax
    8b3b:	mov    r15,rax
    8b3e:	lea    rsi,[rbp+0x30]
    8b42:	nop    DWORD PTR [rax]
    8b45:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8b50:	pxor   xmm2,xmm2
    8b54:	xor    eax,eax
    8b56:	movapd xmm1,xmm2
    8b5a:	movupd xmm0,XMMWORD PTR [rsi+rax*1]
    8b5f:	movapd xmm6,xmm0
    8b63:	mulpd  xmm6,xmm0
    8b67:	mulpd  xmm0,XMMWORD PTR [rbx+rax*1]
    8b6c:	add    rax,0x10
    8b70:	movapd xmm3,xmm6
    8b74:	unpckhpd xmm6,xmm6
    8b78:	addsd  xmm3,xmm2
    8b7c:	addsd  xmm1,xmm0
    8b80:	movapd xmm2,xmm6
    8b84:	unpckhpd xmm0,xmm0
    8b88:	addsd  xmm1,xmm0
    8b8c:	addsd  xmm2,xmm3
    8b90:	cmp    rax,0x40
    8b94:	jne    8b5a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x75a>
    8b96:	divsd  xmm1,xmm2
    8b9a:	movsd  xmm0,QWORD PTR [r15]
    8b9f:	mov    r9,rsi
    8ba2:	mov    rax,rbx
    8ba5:	addsd  xmm0,xmm1
    8ba9:	unpcklpd xmm1,xmm1
    8bad:	movsd  QWORD PTR [r15],xmm0
    8bb2:	movupd xmm3,XMMWORD PTR [r9]
    8bb7:	movapd xmm0,XMMWORD PTR [rax]
    8bbb:	add    r9,0x10
    8bbf:	add    rax,0x10
    8bc3:	mulpd  xmm3,xmm1
    8bc7:	subpd  xmm0,xmm3
    8bcb:	movaps XMMWORD PTR [rax-0x10],xmm0
    8bcf:	cmp    rdi,r9
    8bd2:	jne    8bb2 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x7b2>
    8bd4:	add    rdi,0x40
    8bd8:	add    rsi,0x40
    8bdc:	add    r15,0x8
    8be0:	cmp    rdi,r12
    8be3:	jne    8b50 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x750>
    8be9:	add    rcx,0x1
    8bed:	cmp    rcx,QWORD PTR [rsp+0x28]
    8bf2:	jne    8b2b <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x72b>
    8bf8:	mov    r9,QWORD PTR [rsp+0x20]
    8bfd:	mov    r15,rdx
    8c00:	pxor   xmm6,xmm6
    8c04:	mov    rdx,rbx
    8c07:	xor    eax,eax
    8c09:	nop    DWORD PTR [rax+0x0]
    8c10:	cmp    rax,r8
    8c13:	jae    8c46 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x846>
    8c15:	movsd  xmm0,QWORD PTR [rdx]
    8c19:	movapd xmm1,xmm0
    8c1d:	movapd xmm2,xmm0
    8c21:	xorpd  xmm2,XMMWORD PTR [rip+0x2eb77]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    8c29:	cmpltsd xmm1,xmm4
    8c2e:	andpd  xmm2,xmm1
    8c32:	andnpd xmm1,xmm0
    8c36:	orpd   xmm1,xmm2
    8c3a:	movapd xmm0,xmm1
    8c3e:	maxsd  xmm0,xmm6
    8c42:	movapd xmm6,xmm0
    8c46:	add    rax,0x1
    8c4a:	add    rdx,0x8
    8c4e:	cmp    rax,0x8
    8c52:	jne    8c10 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x810>
    8c54:	ucomisd xmm6,xmm4
    8c58:	jp     8c60 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x860>
    8c5a:	je     90f9 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xcf9>
    8c60:	mov    rdx,rbx
    8c63:	pxor   xmm3,xmm3
    8c67:	xor    eax,eax
    8c69:	nop
    8c6a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8c75:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8c80:	cmp    rax,r8
    8c83:	jae    8c95 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x895>
    8c85:	movsd  xmm0,QWORD PTR [rdx]
    8c89:	divsd  xmm0,xmm6
    8c8d:	mulsd  xmm0,xmm0
    8c91:	addsd  xmm3,xmm0
    8c95:	add    rax,0x1
    8c99:	add    rdx,0x8
    8c9d:	cmp    rax,0x8
    8ca1:	jne    8c80 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x880>
    8ca3:	ucomisd xmm3,xmm4
    8ca7:	jp     9024 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc24>
    8cad:	jne    9024 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc24>
    8cb3:	mulsd  xmm6,xmm4
    8cb7:	mulsd  xmm5,QWORD PTR [rip+0x2eb51]        # 37810 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x90>
    8cbf:	movsd  QWORD PTR [r13+0x0],xmm6
    8cc5:	comisd xmm5,xmm6
    8cc9:	jb     9117 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xd17>
    8ccf:	test   r14,r14
    8cd2:	je     8d49 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x949>
    8cd4:	movsd  xmm0,QWORD PTR [rbp+0x350]
    8cdc:	mov    rax,rbp
    8cdf:	lea    rdx,[r13-0x358]
    8ce6:	cs nop WORD PTR [rax+rax*1+0x0]
    8cf0:	movsd  xmm1,QWORD PTR [rax+0x358]
    8cf8:	movsd  xmm7,QWORD PTR [rax+0x420]
    8d00:	movapd xmm2,xmm0
    8d04:	add    rax,0x8
    8d08:	movsd  xmm3,QWORD PTR [rax+0x3d8]
    8d10:	mulsd  xmm2,xmm7
    8d14:	movapd xmm8,xmm1
    8d19:	mulsd  xmm1,xmm7
    8d1d:	mulsd  xmm8,xmm3
    8d22:	mulsd  xmm3,xmm0
    8d26:	movapd xmm0,xmm1
    8d2a:	addsd  xmm8,xmm2
    8d2f:	subsd  xmm0,xmm3
    8d33:	movsd  QWORD PTR [rax+0x348],xmm8
    8d3c:	movsd  QWORD PTR [rax+0x350],xmm0
    8d44:	cmp    rax,rdx
    8d47:	jne    8cf0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x8f0>
    8d49:	movsd  xmm8,QWORD PTR [r13-0x8]
    8d4f:	movsd  xmm9,QWORD PTR [r13+0x0]
    8d55:	movapd xmm1,xmm8
    8d5a:	comisd xmm4,xmm9
    8d5f:	movapd xmm0,xmm8
    8d64:	cmpltsd xmm1,xmm4
    8d69:	xorpd  xmm0,XMMWORD PTR [rip+0x2ea2f]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    8d71:	andpd  xmm0,xmm1
    8d75:	andnpd xmm1,xmm8
    8d7a:	orpd   xmm0,xmm1
    8d7e:	ja     91b0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xdb0>
    8d84:	ucomisd xmm9,xmm4
    8d89:	jp     8d91 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x991>
    8d8b:	je     9239 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xe39>
    8d91:	movapd xmm2,xmm9
    8d96:	maxsd  xmm2,xmm0
    8d9a:	movapd xmm7,xmm8
    8d9f:	movapd xmm10,xmm9
    8da4:	divsd  xmm7,xmm2
    8da8:	divsd  xmm10,xmm2
    8dad:	movapd xmm1,xmm7
    8db1:	comisd xmm4,xmm7
    8db5:	mulsd  xmm1,xmm7
    8db9:	movapd xmm0,xmm10
    8dbe:	mulsd  xmm0,xmm10
    8dc3:	addsd  xmm0,xmm1
    8dc7:	ja     9202 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xe02>
    8dcd:	ucomisd xmm0,xmm4
    8dd1:	jp     8dd9 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x9d9>
    8dd3:	pxor   xmm1,xmm1
    8dd7:	je     8e02 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xa02>
    8dd9:	movsd  xmm1,QWORD PTR [rip+0x2ea7f]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
    8de1:	movapd xmm11,xmm1
    8de6:	maxsd  xmm11,xmm0
    8deb:	movapd xmm0,xmm11
    8df0:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
    8df5:	movapd xmm1,xmm11
    8dfa:	pxor   xmm4,xmm4
    8dfe:	mulsd  xmm1,xmm0
    8e02:	ucomisd xmm7,xmm4
    8e06:	jp     94f8 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x10f8>
    8e0c:	jne    94f8 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x10f8>
    8e12:	movapd xmm7,xmm10
    8e17:	pxor   xmm0,xmm0
    8e1b:	divsd  xmm7,xmm1
    8e1f:	mulsd  xmm8,xmm0
    8e24:	mov    rax,r15
    8e27:	movsd  QWORD PTR [r13+0x88],xmm7
    8e30:	mulsd  xmm7,xmm9
    8e35:	imul   rax,r14
    8e39:	movsd  QWORD PTR [r13+0xc8],xmm0
    8e42:	shr    rax,1
    8e45:	addsd  xmm7,xmm8
    8e4a:	movsd  QWORD PTR [r13-0x8],xmm7
    8e50:	nop    DWORD PTR [rax+rax*1+0x0]
    8e55:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8e60:	movsd  xmm0,QWORD PTR [r9]
    8e65:	movsd  QWORD PTR [r9+rax*8-0x120],xmm0
    8e6f:	add    r9,0x8
    8e73:	cmp    r9,r13
    8e76:	jne    8e60 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xa60>
    8e78:	movsd  xmm0,QWORD PTR [r13+0x88]
    8e81:	movsd  xmm1,QWORD PTR [r13+0x40]
    8e87:	xorpd  xmm0,XMMWORD PTR [rip+0x2e911]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    8e8f:	mulsd  xmm0,xmm1
    8e93:	mulsd  xmm1,QWORD PTR [r13+0xc8]
    8e9c:	movsd  QWORD PTR [r13+0x48],xmm0
    8ea2:	movsd  QWORD PTR [r13+0x40],xmm1
    8ea8:	add    QWORD PTR [rbp+0x4a8],0x1
    8eb0:	comisd xmm5,xmm6
    8eb4:	jb     9178 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xd78>
    8eba:	cmp    r15,0x8
    8ebe:	je     9502 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1102>
    8ec4:	cmp    r14,0x6
    8ec8:	je     953e <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x113e>
    8ece:	add    r13,0x10
    8ed2:	sub    r12,0xffffffffffffff80
    8ed6:	add    r14,0x3
    8eda:	xor    edx,edx
    8edc:	movzx  r10d,BYTE PTR [rsp+0x10]
    8ee2:	mov    rsi,r15
    8ee5:	add    r13,0x8
    8ee9:	add    r12,0x40
    8eed:	cmp    r14,0x8
    8ef1:	jne    8995 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x595>
    8ef7:	movsd  xmm2,QWORD PTR [rsp+0x68]
    8efd:	movzx  edx,BYTE PTR [rsp+0x8e]
    8f05:	mov    r15,r8
    8f08:	mov    r14,QWORD PTR [rsp+0x70]
    8f0d:	mov    rcx,QWORD PTR [rsp+0x78]
    8f12:	mov    rbx,QWORD PTR [rsp+0x80]
    8f1a:	movzx  r9d,BYTE PTR [rsp+0x8f]
    8f23:	movsd  xmm5,QWORD PTR [rsp+0x60]
    8f29:	mov    eax,edx
    8f2b:	mov    r8d,0x7
    8f31:	pxor   xmm3,xmm3
    8f35:	jmp    8f4a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb4a>
    8f37:	nop    WORD PTR [rax+rax*1+0x0]
    8f40:	sub    r8,0x1
    8f44:	jb     9011 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc11>
    8f4a:	mov    QWORD PTR [rbp+r8*8+0x460],0x0
    8f56:	cmp    r8,rsi
    8f59:	setb   r11b
    8f5d:	and    r11b,al
    8f60:	je     8f40 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb40>
    8f62:	movsd  xmm1,QWORD PTR [rbp+r8*8+0x398]
    8f6c:	lea    r12,[r8+0x1]
    8f70:	cmp    r8,0x7
    8f74:	je     8fb2 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xbb2>
    8f76:	mov    rdi,r12
    8f79:	nop    DWORD PTR [rax+0x0]
    8f80:	mov    rax,rdi
    8f83:	add    rdi,0x1
    8f87:	cmp    rax,rsi
    8f8a:	jae    8fac <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xbac>
    8f8c:	imul   rax,rdi
    8f90:	shr    rax,1
    8f93:	add    rax,r8
    8f96:	movsd  xmm0,QWORD PTR [rbp+rax*8+0x230]
    8f9f:	mulsd  xmm0,QWORD PTR [rbp+rdi*8+0x458]
    8fa8:	subsd  xmm1,xmm0
    8fac:	cmp    rdi,0x8
    8fb0:	jne    8f80 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb80>
    8fb2:	imul   r12,r8
    8fb6:	shr    r12,1
    8fb9:	lea    rax,[r12+r8*1]
    8fbd:	movsd  xmm0,QWORD PTR [rbp+rax*8+0x230]
    8fc6:	comisd xmm3,xmm0
    8fca:	ja     91c2 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xdc2>
    8fd0:	comisd xmm5,xmm0
    8fd4:	jbe    91ec <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xdec>
    8fda:	lea    rax,[r8-0x1]
    8fde:	test   r8,r8
    8fe1:	je     9019 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc19>
    8fe3:	mov    QWORD PTR [rbp+rax*8+0x460],0x0
    8fef:	sub    r8,0x2
    8ff3:	cmp    r8,0xffffffffffffffff
    8ff7:	je     9019 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc19>
    8ff9:	mov    QWORD PTR [rbp+r8*8+0x460],0x0
    9005:	xor    eax,eax
    9007:	sub    r8,0x1
    900b:	jae    8f4a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb4a>
    9011:	test   al,al
    9013:	jne    9275 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xe75>
    9019:	mov    BYTE PTR [rbp+0x4b1],dl
    901f:	jmp    87c4 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x3c4>
    9024:	movsd  xmm1,QWORD PTR [rip+0x2e834]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
    902c:	movapd xmm7,xmm1
    9030:	maxsd  xmm7,xmm3
    9034:	movapd xmm0,xmm7
    9038:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
    903d:	mulsd  xmm5,QWORD PTR [rip+0x2e7cb]        # 37810 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x90>
    9045:	pxor   xmm4,xmm4
    9049:	mulsd  xmm7,xmm0
    904d:	mulsd  xmm6,xmm7
    9051:	comisd xmm5,xmm6
    9055:	movsd  QWORD PTR [r13+0x0],xmm6
    905b:	jae    8ccf <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x8cf>
    9061:	cmp    r15,QWORD PTR [rsp+0x8]
    9066:	jae    8ccf <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x8cf>
    906c:	comisd xmm6,QWORD PTR [rip+0x2e914]        # 37988 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x208>
    9074:	jb     9122 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xd22>
    907a:	movsd  xmm1,QWORD PTR [rip+0x2e7a6]        # 37828 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
    9082:	divsd  xmm1,xmm6
    9086:	jmp    9126 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xd26>
    908b:	movsd  xmm1,QWORD PTR [rip+0x2e7cd]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
    9093:	movapd xmm5,xmm1
    9097:	maxsd  xmm5,xmm3
    909b:	movapd xmm0,xmm5
    909f:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
    90a4:	pxor   xmm4,xmm4
    90a8:	mulsd  xmm5,xmm0
    90ac:	jmp    8af7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x6f7>
    90b1:	movsd  xmm1,QWORD PTR [rip+0x2e7a7]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
    90b9:	movapd xmm6,xmm1
    90bd:	maxsd  xmm6,xmm2
    90c1:	movapd xmm0,xmm6
    90c5:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
    90ca:	mulsd  xmm0,xmm6
    90ce:	jmp    8737 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x337>
    90d3:	movsd  xmm1,QWORD PTR [rip+0x2e785]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
    90db:	movapd xmm8,xmm1
    90e0:	maxsd  xmm8,xmm2
    90e5:	movapd xmm0,xmm8
    90ea:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
    90ef:	mulsd  xmm0,xmm8
    90f4:	jmp    8594 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x194>
    90f9:	mulsd  xmm5,QWORD PTR [rip+0x2e70f]        # 37810 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x90>
    9101:	mov    QWORD PTR [r13+0x0],0x0
    9109:	pxor   xmm6,xmm6
    910d:	comisd xmm5,xmm4
    9111:	jae    8ccf <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x8cf>
    9117:	cmp    r15,QWORD PTR [rsp+0x8]
    911c:	jae    8ccf <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x8cf>
    9122:	pxor   xmm1,xmm1
    9126:	add    rbx,0x40
    912a:	comisd xmm6,QWORD PTR [rip+0x2e856]        # 37988 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x208>
    9132:	lea    rax,[r12-0x40]
    9137:	movsd  xmm0,QWORD PTR [rsp+0x110]
    9140:	jb     94c6 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x10c6>
    9146:	mulsd  xmm0,xmm1
    914a:	add    r10,0x10
    914e:	movsd  QWORD PTR [rax],xmm0
    9152:	movsd  xmm0,QWORD PTR [r10-0x8]
    9158:	mulsd  xmm0,xmm1
    915c:	movsd  QWORD PTR [rax+0x8],xmm0
    9161:	cmp    r10,rbx
    9164:	je     8ccf <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x8cf>
    916a:	movsd  xmm0,QWORD PTR [r10]
    916f:	add    rax,0x10
    9173:	jmp    9146 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xd46>
    9175:	nop    DWORD PTR [rax]
    9178:	movapd xmm1,xmm0
    917c:	movapd xmm2,xmm0
    9180:	mov    r14,r15
    9183:	mov    rsi,r15
    9186:	cmpltsd xmm1,xmm4
    918b:	xorpd  xmm2,XMMWORD PTR [rip+0x2e60d]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    9193:	andpd  xmm2,xmm1
    9197:	andnpd xmm1,xmm0
    919b:	orpd   xmm1,xmm2
    919f:	comisd xmm1,QWORD PTR [rsp+0x48]
    91a5:	seta   dl
    91a8:	xor    r10d,r10d
    91ab:	jmp    8983 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x583>
    91b0:	movapd xmm2,xmm9
    91b5:	xorpd  xmm2,XMMWORD PTR [rip+0x2e5e3]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    91bd:	jmp    8d96 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x996>
    91c2:	movapd xmm4,xmm0
    91c6:	xorpd  xmm4,XMMWORD PTR [rip+0x2e5d2]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    91ce:	comisd xmm5,xmm4
    91d2:	ja     8fda <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xbda>
    91d8:	movapd xmm4,xmm5
    91dc:	xorpd  xmm4,XMMWORD PTR [rip+0x2e5bc]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    91e4:	minsd  xmm4,xmm0
    91e8:	movapd xmm0,xmm4
    91ec:	divsd  xmm1,xmm0
    91f0:	mov    eax,r11d
    91f3:	movsd  QWORD PTR [rbp+r8*8+0x460],xmm1
    91fd:	jmp    8f40 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb40>
    9202:	ucomisd xmm0,xmm4
    9206:	movapd xmm12,xmm7
    920b:	xorpd  xmm12,XMMWORD PTR [rip+0x2e58c]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    9214:	jp     924a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xe4a>
    9216:	pxor   xmm1,xmm1
    921a:	jne    924a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xe4a>
    921c:	divsd  xmm7,xmm12
    9221:	movapd xmm0,xmm12
    9226:	divsd  xmm10,xmm1
    922b:	divsd  xmm0,xmm1
    922f:	mulsd  xmm7,xmm10
    9234:	jmp    8e1f <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xa1f>
    9239:	movsd  xmm0,QWORD PTR [rip+0x2e5e7]        # 37828 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
    9241:	pxor   xmm7,xmm7
    9245:	jmp    8e1f <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xa1f>
    924a:	movsd  xmm1,QWORD PTR [rip+0x2e60e]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
    9252:	movapd xmm11,xmm1
    9257:	maxsd  xmm11,xmm0
    925c:	movapd xmm0,xmm11
    9261:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
    9266:	movapd xmm1,xmm11
    926b:	pxor   xmm4,xmm4
    926f:	mulsd  xmm1,xmm0
    9273:	jmp    921c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xe1c>
    9275:	mov    rax,QWORD PTR [rsp+0x30]
    927a:	lea    rdx,[rsp+0x150]
    9282:	xor    r11d,r11d
    9285:	mov    r8,rdx
    9288:	movdqu xmm0,XMMWORD PTR [rax]
    928c:	movaps XMMWORD PTR [rsp+0x150],xmm0
    9294:	movdqu xmm0,XMMWORD PTR [rax+0x10]
    9299:	movaps XMMWORD PTR [rsp+0x160],xmm0
    92a1:	movdqu xmm0,XMMWORD PTR [rax+0x20]
    92a6:	movaps XMMWORD PTR [rsp+0x170],xmm0
    92ae:	movdqu xmm0,XMMWORD PTR [rax+0x30]
    92b3:	movaps XMMWORD PTR [rsp+0x180],xmm0
    92bb:	mov    rdi,rcx
    92be:	pxor   xmm1,xmm1
    92c2:	xor    eax,eax
    92c4:	nop    DWORD PTR [rax+rax*1+0x0]
    92c9:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    92d4:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    92df:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    92ea:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    92f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    9300:	cmp    rax,rsi
    9303:	jae    9317 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf17>
    9305:	movsd  xmm0,QWORD PTR [rdi+0x30]
    930a:	mulsd  xmm0,QWORD PTR [rbp+rax*8+0x460]
    9313:	addsd  xmm1,xmm0
    9317:	add    rax,0x1
    931b:	add    rdi,0x40
    931f:	cmp    rax,0x8
    9323:	jne    9300 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf00>
    9325:	addsd  xmm1,QWORD PTR [r8]
    932a:	add    r11,0x1
    932e:	add    r8,0x8
    9332:	add    rcx,0x8
    9336:	movsd  QWORD PTR [r8-0x8],xmm1
    933c:	cmp    r11,0x8
    9340:	jne    92bb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xebb>
    9346:	mov    rax,QWORD PTR [rsp+0x38]
    934b:	mov    r13,QWORD PTR [rsp+0x18]
    9350:	mov    r8,r15
    9353:	mov    BYTE PTR [rsp+0x20],r10b
    9358:	mov    rdi,QWORD PTR [rsp+0x50]
    935d:	mov    BYTE PTR [rsp+0x10],r9b
    9362:	mov    r12,QWORD PTR [rax+0x8]
    9366:	mov    rcx,r13
    9369:	movsd  QWORD PTR [rsp+0x68],xmm2
    936f:	mov    rsi,r12
    9372:	call   4260 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    9377:	movzx  r10d,BYTE PTR [rsp+0x20]
    937d:	mov    rdx,r13
    9380:	xor    eax,eax
    9382:	movzx  r9d,BYTE PTR [rsp+0x10]
    9388:	mov    rcx,QWORD PTR [rsp+0x40]
    938d:	movsd  xmm2,QWORD PTR [rsp+0x68]
    9393:	pxor   xmm0,xmm0
    9397:	cmp    rax,r15
    939a:	jae    93a5 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xfa5>
    939c:	movsd  xmm0,QWORD PTR [rcx+rax*8]
    93a1:	subsd  xmm0,QWORD PTR [rdx]
    93a5:	add    rax,0x1
    93a9:	movsd  QWORD PTR [rdx],xmm0
    93ad:	add    rdx,0x8
    93b1:	cmp    rax,0x8
    93b5:	jne    9393 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf93>
    93b7:	pxor   xmm5,xmm5
    93bb:	mov    rdx,QWORD PTR [rsp+0x18]
    93c0:	xor    eax,eax
    93c2:	movapd xmm3,xmm5
    93c6:	cmp    rax,r15
    93c9:	jae    93f8 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xff8>
    93cb:	movsd  xmm0,QWORD PTR [rdx]
    93cf:	movapd xmm4,xmm0
    93d3:	movapd xmm1,xmm0
    93d7:	xorpd  xmm1,XMMWORD PTR [rip+0x2e3c1]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    93df:	cmpltsd xmm0,xmm3
    93e4:	andpd  xmm1,xmm0
    93e8:	andnpd xmm0,xmm4
    93ec:	orpd   xmm0,xmm1
    93f0:	maxsd  xmm0,xmm5
    93f4:	movapd xmm5,xmm0
    93f8:	add    rax,0x1
    93fc:	add    rdx,0x8
    9400:	cmp    rax,0x8
    9404:	jne    93c6 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xfc6>
    9406:	pxor   xmm0,xmm0
    940a:	ucomisd xmm5,xmm0
    940e:	jp     9412 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1012>
    9410:	je     9458 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1058>
    9412:	mov    rdx,QWORD PTR [rsp+0x18]
    9417:	pxor   xmm3,xmm3
    941b:	xor    eax,eax
    941d:	cmp    rax,r15
    9420:	jae    9432 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1032>
    9422:	movsd  xmm0,QWORD PTR [rdx]
    9426:	divsd  xmm0,xmm5
    942a:	mulsd  xmm0,xmm0
    942e:	addsd  xmm3,xmm0
    9432:	add    rax,0x1
    9436:	add    rdx,0x8
    943a:	cmp    rax,0x8
    943e:	jne    941d <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x101d>
    9440:	pxor   xmm0,xmm0
    9444:	ucomisd xmm3,xmm0
    9448:	jp     957a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x117a>
    944e:	jne    957a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x117a>
    9454:	mulsd  xmm0,xmm5
    9458:	comisd xmm2,xmm0
    945c:	jb     95a8 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x11a8>
    9462:	movdqa xmm1,XMMWORD PTR [rsp+0x150]
    946b:	mov    rax,QWORD PTR [rsp+0x30]
    9470:	movups XMMWORD PTR [rax],xmm1
    9473:	movdqa xmm1,XMMWORD PTR [rsp+0x160]
    947c:	movups XMMWORD PTR [rax+0x10],xmm1
    9480:	movdqa xmm1,XMMWORD PTR [rsp+0x170]
    9489:	movups XMMWORD PTR [rax+0x20],xmm1
    948d:	movdqa xmm1,XMMWORD PTR [rsp+0x180]
    9496:	movups XMMWORD PTR [rax+0x30],xmm1
    949a:	movsd  QWORD PTR [rbp+0x4a0],xmm0
    94a2:	movsd  xmm4,QWORD PTR [rsp+0x48]
    94a8:	comisd xmm4,xmm0
    94ac:	setb   al
    94af:	setae  BYTE PTR [rbp+0x4b0]
    94b6:	and    eax,r10d
    94b9:	mov    edx,eax
    94bb:	mov    BYTE PTR [rbp+0x4b1],dl
    94c1:	jmp    87c4 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x3c4>
    94c6:	divsd  xmm0,xmm6
    94ca:	add    r10,0x10
    94ce:	movsd  QWORD PTR [rax],xmm0
    94d2:	movsd  xmm0,QWORD PTR [r10-0x8]
    94d8:	divsd  xmm0,xmm6
    94dc:	movsd  QWORD PTR [rax+0x8],xmm0
    94e1:	cmp    rbx,r10
    94e4:	je     8ccf <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x8cf>
    94ea:	movsd  xmm0,QWORD PTR [r10]
    94ef:	add    rax,0x10
    94f3:	jmp    94c6 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x10c6>
    94f5:	nop    DWORD PTR [rax]
    94f8:	movapd xmm12,xmm7
    94fd:	jmp    921c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xe1c>
    9502:	movsd  xmm2,QWORD PTR [rsp+0x68]
    9508:	movzx  edx,BYTE PTR [rsp+0x8e]
    9510:	mov    r15,r8
    9513:	mov    esi,0x8
    9518:	mov    r14,QWORD PTR [rsp+0x70]
    951d:	mov    rcx,QWORD PTR [rsp+0x78]
    9522:	mov    rbx,QWORD PTR [rsp+0x80]
    952a:	movzx  r10d,BYTE PTR [rsp+0x10]
    9530:	movzx  r9d,BYTE PTR [rsp+0x8f]
    9539:	jmp    8f23 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb23>
    953e:	movsd  xmm2,QWORD PTR [rsp+0x68]
    9544:	movzx  edx,BYTE PTR [rsp+0x8e]
    954c:	mov    r15,r8
    954f:	mov    esi,0x7
    9554:	mov    r14,QWORD PTR [rsp+0x70]
    9559:	mov    rcx,QWORD PTR [rsp+0x78]
    955e:	mov    rbx,QWORD PTR [rsp+0x80]
    9566:	movzx  r10d,BYTE PTR [rsp+0x10]
    956c:	movzx  r9d,BYTE PTR [rsp+0x8f]
    9575:	jmp    8f23 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb23>
    957a:	movsd  xmm1,QWORD PTR [rip+0x2e2de]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
    9582:	movsd  QWORD PTR [rsp+0x10],xmm2
    9588:	movapd xmm6,xmm1
    958c:	maxsd  xmm6,xmm3
    9590:	movapd xmm0,xmm6
    9594:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
    9599:	movsd  xmm2,QWORD PTR [rsp+0x10]
    959f:	mulsd  xmm0,xmm6
    95a3:	jmp    9454 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1054>
    95a8:	mov    r13,QWORD PTR [rsp+0x18]
    95ad:	mov    rdx,QWORD PTR [rsp+0x30]
    95b2:	mov    r8,r15
    95b5:	mov    rsi,r12
    95b8:	mov    rdi,QWORD PTR [rsp+0x50]
    95bd:	mov    BYTE PTR [rsp+0x20],r9b
    95c2:	mov    rcx,r13
    95c5:	mov    BYTE PTR [rsp+0x10],r10b
    95ca:	call   4260 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    95cf:	movzx  r10d,BYTE PTR [rsp+0x10]
    95d5:	mov    rdx,r13
    95d8:	xor    eax,eax
    95da:	movzx  r9d,BYTE PTR [rsp+0x20]
    95e0:	mov    rcx,QWORD PTR [rsp+0x40]
    95e5:	pxor   xmm0,xmm0
    95e9:	cmp    rax,r15
    95ec:	jae    95f7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x11f7>
    95ee:	movsd  xmm0,QWORD PTR [rcx+rax*8]
    95f3:	subsd  xmm0,QWORD PTR [rdx]
    95f7:	add    rax,0x1
    95fb:	movsd  QWORD PTR [rdx],xmm0
    95ff:	add    rdx,0x8
    9603:	cmp    rax,0x8
    9607:	jne    95e5 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x11e5>
    9609:	movsd  xmm0,QWORD PTR [rbp+0x4a0]
    9611:	jmp    94a2 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x10a2>
    9616:	call   1170 <__stack_chk_fail@plt>

Disassembly of section .fini:
