
/tmp/matrix_peak_stack_audit/current_b599/current_b599_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000008320 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0>:
    8320:	push   r15
    8322:	pxor   xmm0,xmm0
    8326:	mov    r15,rsi
    8329:	push   r14
    832b:	movapd xmm1,xmm0
    832f:	mov    r14,rdi
    8332:	push   r13
    8334:	push   r12
    8336:	push   rbp
    8337:	mov    rbp,r8
    833a:	push   rbx
    833b:	sub    rsp,0x1f8
    8342:	movsd  xmm2,QWORD PTR [rsi+0x18]
    8347:	mov    rbx,QWORD PTR [rsi+0x8]
    834b:	movsd  xmm3,QWORD PTR [rip+0x5b49d]        # 637f0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x70>
    8353:	mov    QWORD PTR [rsp+0x18],rdi
    8358:	mov    rdi,rcx
    835b:	cmpnltsd xmm1,xmm2
    8360:	mov    QWORD PTR [rsp+0x30],rdx
    8365:	comisd xmm0,xmm2
    8369:	mov    QWORD PTR [rsp+0x40],rcx
    836e:	andpd  xmm3,xmm1
    8372:	andnpd xmm1,xmm2
    8376:	movapd xmm2,xmm0
    837a:	mov    r12,QWORD PTR fs:0x28
    8383:	mov    QWORD PTR [rsp+0x1e8],r12
    838b:	mov    r12,r9
    838e:	orpd   xmm1,xmm3
    8392:	setae  BYTE PTR [rsi+0x4b2]
    8399:	xor    eax,eax
    839b:	movsd  QWORD PTR [rsp+0x70],xmm1
    83a1:	movsd  xmm1,QWORD PTR [rsi+0x20]
    83a6:	mov    WORD PTR [rsi+0x4b0],ax
    83ad:	xor    eax,eax
    83af:	cmp    QWORD PTR [rsi+0x10],0x1
    83b4:	cmpltsd xmm2,xmm1
    83b9:	seta   al
    83bc:	mov    QWORD PTR [rsi+0x4a8],0x0
    83c7:	add    rax,0x1
    83cb:	mov    QWORD PTR [rsp+0x28],rax
    83d0:	andpd  xmm1,xmm2
    83d4:	movsd  QWORD PTR [rsp+0x8],xmm1
    83da:	movsd  xmm1,QWORD PTR [rsi+0x28]
    83df:	mov    rsi,r9
    83e2:	cmpltsd xmm0,xmm1
    83e7:	movapd xmm6,xmm1
    83eb:	andpd  xmm6,xmm0
    83ef:	movsd  QWORD PTR [rsp+0x10],xmm6
    83f5:	call   2e2e0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    83fa:	mulsd  xmm0,QWORD PTR [rsp+0x8]
    8400:	addsd  xmm0,QWORD PTR [rsp+0x10]
    8406:	movsd  QWORD PTR [rsp+0x38],xmm0
    840c:	movdqu xmm0,XMMWORD PTR [rbp+0x0]
    8411:	movups XMMWORD PTR [r14],xmm0
    8415:	movdqu xmm0,XMMWORD PTR [rbp+0x10]
    841a:	movups XMMWORD PTR [r14+0x10],xmm0
    841f:	movdqu xmm0,XMMWORD PTR [rbp+0x20]
    8424:	movups XMMWORD PTR [r14+0x20],xmm0
    8429:	movdqu xmm0,XMMWORD PTR [rbp+0x30]
    842e:	movups XMMWORD PTR [r14+0x30],xmm0
    8433:	pxor   xmm0,xmm0
    8437:	movaps XMMWORD PTR [rsp+0xa0],xmm0
    843f:	movaps XMMWORD PTR [rsp+0xb0],xmm0
    8447:	movaps XMMWORD PTR [rsp+0xc0],xmm0
    844f:	movaps XMMWORD PTR [rsp+0xd0],xmm0
    8457:	cmp    r12,0x8
    845b:	je     9052 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xd32>
    8461:	movaps XMMWORD PTR [rsp+0x1a0],xmm0
    8469:	movaps XMMWORD PTR [rsp+0x1b0],xmm0
    8471:	movaps XMMWORD PTR [rsp+0x1c0],xmm0
    8479:	movaps XMMWORD PTR [rsp+0x1d0],xmm0
    8481:	test   r12,r12
    8484:	jne    8629 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x309>
    848a:	lea    rax,[rsp+0xa0]
    8492:	mov    QWORD PTR [rsp+0xa0],0x0
    849e:	mov    QWORD PTR [rsp+0x8],rax
    84a3:	mov    QWORD PTR [rsp+0xa8],0x0
    84af:	mov    QWORD PTR [rsp+0xb0],0x0
    84bb:	mov    QWORD PTR [rsp+0xb8],0x0
    84c7:	mov    QWORD PTR [rsp+0xc0],0x0
    84d3:	mov    QWORD PTR [rsp+0xc8],0x0
    84df:	mov    QWORD PTR [rsp+0xd0],0x0
    84eb:	mov    QWORD PTR [rsp+0xd8],0x0
    84f7:	mov    rdx,QWORD PTR [rsp+0x8]
    84fc:	mov    rcx,QWORD PTR [rsp+0x40]
    8501:	xor    eax,eax
    8503:	nop    WORD PTR [rax+rax*1+0x0]
    8509:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8514:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    851f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    852a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8535:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8540:	pxor   xmm0,xmm0
    8544:	cmp    rax,r12
    8547:	jae    8552 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x232>
    8549:	movsd  xmm0,QWORD PTR [rcx+rax*8]
    854e:	subsd  xmm0,QWORD PTR [rdx]
    8552:	add    rax,0x1
    8556:	movsd  QWORD PTR [rdx],xmm0
    855a:	add    rdx,0x8
    855e:	cmp    rax,0x8
    8562:	jne    8540 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x220>
    8564:	mov    rdi,QWORD PTR [rsp+0x8]
    8569:	mov    rsi,r12
    856c:	call   2e2e0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    8571:	movsd  xmm7,QWORD PTR [rsp+0x38]
    8577:	mov    r10,QWORD PTR [r15]
    857a:	movsd  QWORD PTR [r15+0x4a0],xmm0
    8583:	comisd xmm7,xmm0
    8587:	setae  dl
    858a:	mov    BYTE PTR [r15+0x4b0],dl
    8591:	test   r10,r10
    8594:	je     85fb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x2db>
    8596:	mov    ecx,0x8
    859b:	cmp    rbx,rcx
    859e:	cmova  rbx,rcx
    85a2:	mov    ecx,0x1
    85a7:	test   rbx,rbx
    85aa:	cmove  rbx,rcx
    85ae:	cmp    rbx,r12
    85b1:	cmova  rbx,r12
    85b5:	test   rbx,rbx
    85b8:	mov    QWORD PTR [rsp+0x10],rbx
    85bd:	mov    rbx,r10
    85c0:	mov    r10,r12
    85c3:	setne  r14b
    85c7:	xor    ebp,ebp
    85c9:	jmp    85d8 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x2b8>
    85cb:	nop    DWORD PTR [rax+rax*1+0x0]
    85d0:	movzx  edx,BYTE PTR [r15+0x4b0]
    85d8:	test   dl,dl
    85da:	jne    85f2 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x2d2>
    85dc:	movzx  eax,BYTE PTR [r15+0x4b1]
    85e4:	xor    eax,0x1
    85e7:	and    al,r14b
    85ea:	mov    edx,eax
    85ec:	jne    86b0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x390>
    85f2:	add    rbp,0x1
    85f6:	cmp    rbp,rbx
    85f9:	jne    85d0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x2b0>
    85fb:	mov    rax,QWORD PTR [rsp+0x1e8]
    8603:	sub    rax,QWORD PTR fs:0x28
    860c:	jne    996d <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x164d>
    8612:	mov    rax,QWORD PTR [rsp+0x18]
    8617:	add    rsp,0x1f8
    861e:	pop    rbx
    861f:	pop    rbp
    8620:	pop    r12
    8622:	pop    r13
    8624:	pop    r14
    8626:	pop    r15
    8628:	ret
    8629:	mov    rax,QWORD PTR [rsp+0x30]
    862e:	mov    rdi,QWORD PTR [rax]
    8631:	mov    rcx,QWORD PTR [rax+0x8]
    8635:	mov    rax,QWORD PTR [rsp+0x18]
    863a:	movsd  xmm0,QWORD PTR [rax]
    863e:	movsd  QWORD PTR [rsp+0x1a0],xmm0
    8647:	cmp    r12,0x1
    864b:	je     94db <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x11bb>
    8651:	mov    rax,QWORD PTR [rsp+0x18]
    8656:	movsd  xmm0,QWORD PTR [rax+0x8]
    865b:	movsd  QWORD PTR [rsp+0x1a8],xmm0
    8664:	cmp    r12,0x2
    8668:	je     9490 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1170>
    866e:	mov    rax,QWORD PTR [rsp+0x18]
    8673:	movsd  xmm0,QWORD PTR [rax+0x10]
    8678:	movsd  QWORD PTR [rsp+0x1b0],xmm0
    8681:	cmp    r12,0x3
    8685:	jne    96a2 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1382>
    868b:	cmp    rcx,0x8
    868f:	je     9775 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1455>
    8695:	test   rcx,rcx
    8698:	jne    9853 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1533>
    869e:	lea    rax,[rsp+0xa0]
    86a6:	mov    QWORD PTR [rsp+0x8],rax
    86ab:	jmp    84bb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x19b>
    86b0:	movsd  xmm2,QWORD PTR [r15+0x4a0]
    86b9:	comisd xmm2,QWORD PTR [rip+0x5b367]        # 63a28 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x2a8>
    86c1:	pxor   xmm0,xmm0
    86c5:	pxor   xmm1,xmm1
    86c9:	movups XMMWORD PTR [r15+0x398],xmm0
    86d1:	mov    QWORD PTR [r15+0x3d8],0x0
    86dc:	movups XMMWORD PTR [r15+0x3a8],xmm0
    86e4:	movups XMMWORD PTR [r15+0x3b8],xmm0
    86ec:	movups XMMWORD PTR [r15+0x3c8],xmm0
    86f4:	movsd  QWORD PTR [r15+0x398],xmm2
    86fd:	jb     870b <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x3eb>
    86ff:	movsd  xmm1,QWORD PTR [rip+0x5b249]        # 63950 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x1d0>
    8707:	divsd  xmm1,xmm2
    870b:	comisd xmm2,QWORD PTR [rip+0x5b315]        # 63a28 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x2a8>
    8713:	mov    rax,QWORD PTR [rsp+0x8]
    8718:	movsd  xmm0,QWORD PTR [rax]
    871c:	lea    rsi,[rax+0x40]
    8720:	jb     8faf <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc8f>
    8726:	lea    rdi,[r15+0x30]
    872a:	mulsd  xmm0,xmm1
    872e:	add    rax,0x10
    8732:	movsd  QWORD PTR [rdi],xmm0
    8736:	movsd  xmm0,QWORD PTR [rax-0x8]
    873b:	mulsd  xmm0,xmm1
    873f:	movsd  QWORD PTR [rdi+0x8],xmm0
    8744:	cmp    rsi,rax
    8747:	je     8758 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x438>
    8749:	movsd  xmm0,QWORD PTR [rax]
    874d:	add    rdi,0x10
    8751:	jmp    872a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x40a>
    8753:	nop    DWORD PTR [rax+rax*1+0x0]
    8758:	pxor   xmm0,xmm0
    875c:	mov    BYTE PTR [rsp+0x97],r14b
    8764:	mov    esi,edx
    8766:	xor    edi,edi
    8768:	mov    BYTE PTR [rsp+0x96],dl
    876f:	pxor   xmm5,xmm5
    8773:	xor    r14d,r14d
    8776:	xor    ecx,ecx
    8778:	mov    QWORD PTR [rsp+0x80],rbp
    8780:	lea    r13,[r15+0x358]
    8787:	lea    r12,[r15+0xb0]
    878e:	mov    QWORD PTR [rsp+0x88],rbx
    8796:	mov    QWORD PTR [rsp+0x98],r15
    879e:	movaps XMMWORD PTR [rsp+0xe0],xmm0
    87a6:	movaps XMMWORD PTR [rsp+0xf0],xmm0
    87ae:	movaps XMMWORD PTR [rsp+0x100],xmm0
    87b6:	movaps XMMWORD PTR [rsp+0x110],xmm0
    87be:	movaps XMMWORD PTR [rsp+0x120],xmm0
    87c6:	movaps XMMWORD PTR [rsp+0x130],xmm0
    87ce:	movaps XMMWORD PTR [rsp+0x140],xmm0
    87d6:	movaps XMMWORD PTR [rsp+0x150],xmm0
    87de:	movsd  QWORD PTR [rsp+0x78],xmm2
    87e4:	jmp    8815 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x4f5>
    87e6:	nop    DWORD PTR [rax+0x0]
    87ea:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    87f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8800:	mov    r14,rax
    8803:	add    r13,0x8
    8807:	add    r12,0x40
    880b:	cmp    r14,0x8
    880f:	je     8d51 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xa31>
    8815:	cmp    r14,QWORD PTR [rsp+0x10]
    881a:	lea    rax,[r14+0x1]
    881e:	setb   dl
    8821:	and    dl,sil
    8824:	mov    BYTE PTR [rsp+0x20],dl
    8828:	je     8800 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x4e0>
    882a:	movdqu xmm0,XMMWORD PTR [r12-0x80]
    8831:	lea    rsi,[rsp+0xe0]
    8839:	movaps XMMWORD PTR [rsp+0xe0],xmm0
    8841:	movdqu xmm0,XMMWORD PTR [r12-0x70]
    8848:	movaps XMMWORD PTR [rsp+0xf0],xmm0
    8850:	movdqu xmm0,XMMWORD PTR [r12-0x60]
    8857:	movaps XMMWORD PTR [rsp+0x100],xmm0
    885f:	movdqu xmm0,XMMWORD PTR [r12-0x50]
    8866:	movaps XMMWORD PTR [rsp+0x110],xmm0
    886e:	cmp    r10,0x8
    8872:	je     8fe8 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xcc8>
    8878:	pxor   xmm0,xmm0
    887c:	movaps XMMWORD PTR [rsp+0x1a0],xmm0
    8884:	movaps XMMWORD PTR [rsp+0x1b0],xmm0
    888c:	movaps XMMWORD PTR [rsp+0x1c0],xmm0
    8894:	movaps XMMWORD PTR [rsp+0x1d0],xmm0
    889c:	test   r10,r10
    889f:	je     88d5 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x5b5>
    88a1:	movsd  xmm0,QWORD PTR [rsp+0xe0]
    88aa:	movsd  QWORD PTR [rsp+0x1a0],xmm0
    88b3:	cmp    r10,0x1
    88b7:	je     88d5 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x5b5>
    88b9:	movsd  xmm0,QWORD PTR [rsp+0xe8]
    88c2:	movsd  QWORD PTR [rsp+0x1a8],xmm0
    88cb:	cmp    r10,0x2
    88cf:	jne    93ec <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x10cc>
    88d5:	mov    rbx,QWORD PTR [rsp+0x30]
    88da:	pxor   xmm0,xmm0
    88de:	lea    rbp,[rsp+0x120]
    88e6:	movaps XMMWORD PTR [rsp+0x120],xmm0
    88ee:	mov    rcx,QWORD PTR [rbx+0x8]
    88f2:	mov    rdi,QWORD PTR [rbx]
    88f5:	movaps XMMWORD PTR [rsp+0x130],xmm0
    88fd:	movaps XMMWORD PTR [rsp+0x140],xmm0
    8905:	movaps XMMWORD PTR [rsp+0x150],xmm0
    890d:	cmp    rcx,0x8
    8911:	je     94c2 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x11a2>
    8917:	test   rcx,rcx
    891a:	je     8930 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x610>
    891c:	lea    rsi,[rsp+0x1a0]
    8924:	mov    rdx,rbp
    8927:	call   4180 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    892c:	pxor   xmm5,xmm5
    8930:	test   r10,r10
    8933:	jne    8ee9 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xbc9>
    8939:	mov    QWORD PTR [rsp+0x120],0x0
    8945:	movsd  QWORD PTR [rsp+0x128],xmm5
    894e:	mov    QWORD PTR [rsp+0x130],0x0
    895a:	movsd  QWORD PTR [rsp+0x138],xmm5
    8963:	mov    QWORD PTR [rsp+0x140],0x0
    896f:	movsd  QWORD PTR [rsp+0x148],xmm5
    8978:	movsd  QWORD PTR [rsp+0x150],xmm5
    8981:	movsd  QWORD PTR [rsp+0x158],xmm5
    898a:	mov    rsi,r10
    898d:	mov    rdi,rbp
    8990:	mov    QWORD PTR [rsp+0x60],rax
    8995:	mov    QWORD PTR [rsp+0x58],r10
    899a:	call   2e2e0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    899f:	mov    QWORD PTR [rsp+0x50],r14
    89a4:	mov    rax,QWORD PTR [rsp+0x60]
    89a9:	xor    r9d,r9d
    89ac:	mov    QWORD PTR [r15+0x390],0x0
    89b7:	mov    r10,QWORD PTR [rsp+0x58]
    89bc:	movsd  QWORD PTR [rsp+0x48],xmm0
    89c2:	pxor   xmm0,xmm0
    89c6:	movups XMMWORD PTR [r15+0x350],xmm0
    89ce:	movups XMMWORD PTR [r15+0x360],xmm0
    89d6:	movups XMMWORD PTR [r15+0x370],xmm0
    89de:	movups XMMWORD PTR [r15+0x380],xmm0
    89e6:	lea    rcx,[r15+0x350]
    89ed:	lea    rbx,[r15+0x70]
    89f1:	mov    r14,rcx
    89f4:	lea    rdi,[r15+0x30]
    89f8:	nop    DWORD PTR [rax+rax*1+0x0]
    8a00:	pxor   xmm2,xmm2
    8a04:	xor    edx,edx
    8a06:	movapd xmm1,xmm2
    8a0a:	movupd xmm0,XMMWORD PTR [rdi+rdx*1]
    8a0f:	movapd xmm4,xmm0
    8a13:	mulpd  xmm4,xmm0
    8a17:	mulpd  xmm0,XMMWORD PTR [rbp+rdx*1+0x0]
    8a1d:	add    rdx,0x10
    8a21:	movapd xmm3,xmm4
    8a25:	unpckhpd xmm4,xmm4
    8a29:	addsd  xmm3,xmm2
    8a2d:	addsd  xmm1,xmm0
    8a31:	movapd xmm2,xmm4
    8a35:	unpckhpd xmm0,xmm0
    8a39:	addsd  xmm1,xmm0
    8a3d:	addsd  xmm2,xmm3
    8a41:	cmp    rdx,0x40
    8a45:	jne    8a0a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x6ea>
    8a47:	divsd  xmm1,xmm2
    8a4b:	movsd  xmm0,QWORD PTR [r14]
    8a50:	mov    r8,rbp
    8a53:	mov    rsi,rdi
    8a56:	mov    rdx,rbp
    8a59:	addsd  xmm0,xmm1
    8a5d:	unpcklpd xmm1,xmm1
    8a61:	movsd  QWORD PTR [r14],xmm0
    8a66:	movupd xmm3,XMMWORD PTR [rsi]
    8a6a:	movapd xmm0,XMMWORD PTR [rdx]
    8a6e:	add    rsi,0x10
    8a72:	add    rdx,0x10
    8a76:	mulpd  xmm3,xmm1
    8a7a:	subpd  xmm0,xmm3
    8a7e:	movaps XMMWORD PTR [rdx-0x10],xmm0
    8a82:	cmp    rsi,rbx
    8a85:	jne    8a66 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x746>
    8a87:	add    rbx,0x40
    8a8b:	add    rdi,0x40
    8a8f:	add    r14,0x8
    8a93:	cmp    r12,rbx
    8a96:	jne    8a00 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x6e0>
    8a9c:	add    r9,0x1
    8aa0:	cmp    r9,QWORD PTR [rsp+0x28]
    8aa5:	jne    89e6 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x6c6>
    8aab:	mov    rsi,r10
    8aae:	mov    rdi,rbp
    8ab1:	mov    QWORD PTR [rsp+0x60],rcx
    8ab6:	mov    r14,QWORD PTR [rsp+0x50]
    8abb:	mov    QWORD PTR [rsp+0x58],rax
    8ac0:	mov    QWORD PTR [rsp+0x50],r10
    8ac5:	mov    QWORD PTR [rsp+0x68],r8
    8aca:	call   2e2e0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    8acf:	mov    r10,QWORD PTR [rsp+0x50]
    8ad4:	mov    rax,QWORD PTR [rsp+0x58]
    8ad9:	pxor   xmm5,xmm5
    8add:	mov    rcx,QWORD PTR [rsp+0x60]
    8ae2:	movapd xmm7,xmm0
    8ae6:	movsd  QWORD PTR [r13+0x0],xmm0
    8aec:	movsd  xmm8,QWORD PTR [rip+0x5af3b]        # 63a30 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x2b0>
    8af5:	mulsd  xmm8,QWORD PTR [rsp+0x48]
    8afc:	comisd xmm8,xmm0
    8b01:	jae    8b13 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x7f3>
    8b03:	cmp    rax,QWORD PTR [rsp+0x10]
    8b08:	mov    r8,QWORD PTR [rsp+0x68]
    8b0d:	jb     9500 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x11e0>
    8b13:	movq   xmm6,QWORD PTR [rip+0x5ac75]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    8b1b:	test   r14,r14
    8b1e:	je     8b98 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x878>
    8b20:	movsd  xmm0,QWORD PTR [r15+0x350]
    8b29:	mov    rdx,r15
    8b2c:	lea    rsi,[r13-0x358]
    8b33:	movq   xmm6,QWORD PTR [rip+0x5ac55]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    8b3b:	nop    DWORD PTR [rax+rax*1+0x0]
    8b40:	movsd  xmm1,QWORD PTR [rdx+0x358]
    8b48:	movsd  xmm2,QWORD PTR [rdx+0x420]
    8b50:	movapd xmm4,xmm0
    8b54:	add    rdx,0x8
    8b58:	movsd  xmm3,QWORD PTR [rdx+0x3d8]
    8b60:	mulsd  xmm4,xmm2
    8b64:	movapd xmm9,xmm1
    8b69:	mulsd  xmm1,xmm2
    8b6d:	mulsd  xmm9,xmm3
    8b72:	mulsd  xmm3,xmm0
    8b76:	movapd xmm0,xmm1
    8b7a:	addsd  xmm4,xmm9
    8b7f:	subsd  xmm0,xmm3
    8b83:	movsd  QWORD PTR [rdx+0x348],xmm4
    8b8b:	movsd  QWORD PTR [rdx+0x350],xmm0
    8b93:	cmp    rdx,rsi
    8b96:	jne    8b40 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x820>
    8b98:	movsd  xmm9,QWORD PTR [r13-0x8]
    8b9e:	movsd  xmm11,QWORD PTR [r13+0x0]
    8ba4:	movapd xmm1,xmm9
    8ba9:	comisd xmm5,xmm11
    8bae:	movapd xmm0,xmm9
    8bb3:	cmpltsd xmm1,xmm5
    8bb8:	xorpd  xmm0,xmm6
    8bbc:	andpd  xmm0,xmm1
    8bc0:	andnpd xmm1,xmm9
    8bc5:	orpd   xmm0,xmm1
    8bc9:	ja     8edb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xbbb>
    8bcf:	ucomisd xmm11,xmm5
    8bd4:	jp     8bdc <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x8bc>
    8bd6:	je     9040 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xd20>
    8bdc:	movapd xmm2,xmm11
    8be1:	maxsd  xmm2,xmm0
    8be5:	movapd xmm10,xmm9
    8bea:	movapd xmm12,xmm11
    8bef:	divsd  xmm10,xmm2
    8bf4:	divsd  xmm12,xmm2
    8bf9:	movapd xmm0,xmm10
    8bfe:	comisd xmm5,xmm10
    8c03:	mulsd  xmm0,xmm10
    8c08:	movapd xmm1,xmm12
    8c0d:	mulsd  xmm1,xmm12
    8c12:	addsd  xmm0,xmm1
    8c16:	ja     8f74 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc54>
    8c1c:	ucomisd xmm0,xmm5
    8c20:	jp     8c28 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x908>
    8c22:	pxor   xmm1,xmm1
    8c26:	je     8c61 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x941>
    8c28:	movsd  xmm1,QWORD PTR [rip+0x5abd0]        # 63800 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
    8c30:	mov    QWORD PTR [rsp+0x50],rax
    8c35:	mov    QWORD PTR [rsp+0x48],rcx
    8c3a:	movapd xmm13,xmm1
    8c3f:	maxsd  xmm13,xmm0
    8c44:	movapd xmm0,xmm13
    8c49:	call   6a10 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
    8c4e:	mov    rax,QWORD PTR [rsp+0x50]
    8c53:	mov    rcx,QWORD PTR [rsp+0x48]
    8c58:	movapd xmm1,xmm0
    8c5c:	mulsd  xmm1,xmm13
    8c61:	ucomisd xmm10,xmm5
    8c66:	jp     95b0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1290>
    8c6c:	jne    95b0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1290>
    8c72:	divsd  xmm12,xmm1
    8c77:	pxor   xmm0,xmm0
    8c7b:	movapd xmm10,xmm12
    8c80:	mulsd  xmm9,xmm0
    8c85:	mov    rdx,r14
    8c88:	movsd  QWORD PTR [r13+0xc8],xmm0
    8c91:	mulsd  xmm11,xmm10
    8c96:	imul   rdx,rax
    8c9a:	movsd  QWORD PTR [r13+0x88],xmm10
    8ca3:	shr    rdx,1
    8ca6:	addsd  xmm9,xmm11
    8cab:	movsd  QWORD PTR [r13-0x8],xmm9
    8cb1:	nop    DWORD PTR [rax+0x0]
    8cb5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8cc0:	movsd  xmm0,QWORD PTR [rcx]
    8cc4:	movsd  QWORD PTR [rcx+rdx*8-0x120],xmm0
    8ccd:	add    rcx,0x8
    8cd1:	cmp    r13,rcx
    8cd4:	jne    8cc0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x9a0>
    8cd6:	movsd  xmm0,QWORD PTR [r13+0x88]
    8cdf:	movsd  xmm1,QWORD PTR [r13+0x40]
    8ce5:	xorpd  xmm0,xmm6
    8ce9:	mulsd  xmm0,xmm1
    8ced:	mulsd  xmm1,QWORD PTR [r13+0xc8]
    8cf6:	movsd  QWORD PTR [r13+0x48],xmm0
    8cfc:	movsd  QWORD PTR [r13+0x40],xmm1
    8d02:	add    QWORD PTR [r15+0x4a8],0x1
    8d0a:	comisd xmm8,xmm7
    8d0f:	jb     8ea8 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb88>
    8d15:	cmp    rax,0x8
    8d19:	je     9618 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x12f8>
    8d1f:	cmp    r14,0x6
    8d23:	je     95d9 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x12b9>
    8d29:	add    r13,0x10
    8d2d:	sub    r12,0xffffffffffffff80
    8d31:	add    r14,0x3
    8d35:	xor    esi,esi
    8d37:	movzx  ecx,BYTE PTR [rsp+0x20]
    8d3c:	mov    rdi,rax
    8d3f:	add    r13,0x8
    8d43:	add    r12,0x40
    8d47:	cmp    r14,0x8
    8d4b:	jne    8815 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x4f5>
    8d51:	mov    r12d,ecx
    8d54:	movsd  xmm2,QWORD PTR [rsp+0x78]
    8d5a:	movzx  edx,BYTE PTR [rsp+0x96]
    8d62:	mov    rbp,QWORD PTR [rsp+0x80]
    8d6a:	mov    rbx,QWORD PTR [rsp+0x88]
    8d72:	movzx  r14d,BYTE PTR [rsp+0x97]
    8d7b:	mov    rcx,QWORD PTR [rsp+0x98]
    8d83:	movsd  xmm5,QWORD PTR [rsp+0x70]
    8d89:	mov    eax,edx
    8d8b:	mov    r8d,0x7
    8d91:	pxor   xmm3,xmm3
    8d95:	jmp    8daa <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xa8a>
    8d97:	nop    WORD PTR [rax+rax*1+0x0]
    8da0:	sub    r8,0x1
    8da4:	jb     8e94 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb74>
    8daa:	mov    QWORD PTR [r15+r8*8+0x460],0x0
    8db6:	cmp    r8,rdi
    8db9:	setb   r9b
    8dbd:	and    r9b,al
    8dc0:	je     8da0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xa80>
    8dc2:	movsd  xmm1,QWORD PTR [r15+r8*8+0x398]
    8dcc:	lea    r11,[r8+0x1]
    8dd0:	cmp    r8,0x7
    8dd4:	je     8e34 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb14>
    8dd6:	mov    rsi,r11
    8dd9:	nop    WORD PTR [rax+rax*1+0x0]
    8ddf:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8dea:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8df5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8e00:	mov    rax,rsi
    8e03:	add    rsi,0x1
    8e07:	cmp    rax,rdi
    8e0a:	jae    8e2e <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb0e>
    8e0c:	imul   rax,rsi
    8e10:	shr    rax,1
    8e13:	add    rax,r8
    8e16:	movsd  xmm0,QWORD PTR [r15+rax*8+0x230]
    8e20:	mulsd  xmm0,QWORD PTR [r15+rsi*8+0x458]
    8e2a:	subsd  xmm1,xmm0
    8e2e:	cmp    rsi,0x8
    8e32:	jne    8e00 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xae0>
    8e34:	imul   r11,r8
    8e38:	shr    r11,1
    8e3b:	lea    rax,[r11+r8*1]
    8e3f:	movsd  xmm0,QWORD PTR [r15+rax*8+0x230]
    8e49:	comisd xmm3,xmm0
    8e4d:	ja     8f34 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc14>
    8e53:	comisd xmm5,xmm0
    8e57:	jbe    8f5e <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc3e>
    8e5d:	lea    rax,[r8-0x1]
    8e61:	test   r8,r8
    8e64:	je     8e9c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb7c>
    8e66:	mov    QWORD PTR [r15+rax*8+0x460],0x0
    8e72:	sub    r8,0x2
    8e76:	cmp    r8,0xffffffffffffffff
    8e7a:	je     8e9c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb7c>
    8e7c:	mov    QWORD PTR [r15+r8*8+0x460],0x0
    8e88:	xor    eax,eax
    8e8a:	sub    r8,0x1
    8e8e:	jae    8daa <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xa8a>
    8e94:	test   al,al
    8e96:	jne    90cb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xdab>
    8e9c:	mov    BYTE PTR [r15+0x4b1],dl
    8ea3:	jmp    85f2 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x2d2>
    8ea8:	movapd xmm1,xmm0
    8eac:	xorpd  xmm6,xmm0
    8eb0:	mov    r14,rax
    8eb3:	mov    r12,rbx
    8eb6:	cmpltsd xmm1,xmm5
    8ebb:	mov    rdi,rax
    8ebe:	andpd  xmm6,xmm1
    8ec2:	andnpd xmm1,xmm0
    8ec6:	orpd   xmm1,xmm6
    8eca:	comisd xmm1,QWORD PTR [rsp+0x38]
    8ed0:	seta   sil
    8ed4:	xor    ecx,ecx
    8ed6:	jmp    8803 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x4e3>
    8edb:	movapd xmm2,xmm11
    8ee0:	xorpd  xmm2,xmm6
    8ee4:	jmp    8be1 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x8c1>
    8ee9:	cmp    r10,0x1
    8eed:	je     8945 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x625>
    8ef3:	cmp    r10,0x2
    8ef7:	je     894e <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x62e>
    8efd:	cmp    r10,0x3
    8f01:	je     895a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x63a>
    8f07:	cmp    r10,0x4
    8f0b:	je     8963 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x643>
    8f11:	cmp    r10,0x5
    8f15:	je     896f <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x64f>
    8f1b:	cmp    r10,0x6
    8f1f:	je     8978 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x658>
    8f25:	cmp    r10,0x7
    8f29:	jne    898a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x66a>
    8f2f:	jmp    8981 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x661>
    8f34:	movapd xmm4,xmm0
    8f38:	xorpd  xmm4,XMMWORD PTR [rip+0x5a850]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    8f40:	comisd xmm5,xmm4
    8f44:	ja     8e5d <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb3d>
    8f4a:	movapd xmm4,xmm5
    8f4e:	xorpd  xmm4,XMMWORD PTR [rip+0x5a83a]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    8f56:	minsd  xmm4,xmm0
    8f5a:	movapd xmm0,xmm4
    8f5e:	divsd  xmm1,xmm0
    8f62:	mov    eax,r9d
    8f65:	movsd  QWORD PTR [r15+r8*8+0x460],xmm1
    8f6f:	jmp    8da0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xa80>
    8f74:	ucomisd xmm0,xmm5
    8f78:	movapd xmm14,xmm10
    8f7d:	xorpd  xmm14,xmm6
    8f82:	jp     908d <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xd6d>
    8f88:	pxor   xmm1,xmm1
    8f8c:	jne    908d <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xd6d>
    8f92:	divsd  xmm10,xmm14
    8f97:	movapd xmm0,xmm14
    8f9c:	divsd  xmm12,xmm1
    8fa1:	divsd  xmm0,xmm1
    8fa5:	mulsd  xmm10,xmm12
    8faa:	jmp    8c80 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x960>
    8faf:	mov    rax,QWORD PTR [rsp+0x8]
    8fb4:	lea    rdi,[r15+0x30]
    8fb8:	divsd  xmm0,xmm2
    8fbc:	add    rax,0x10
    8fc0:	movsd  QWORD PTR [rdi],xmm0
    8fc4:	movsd  xmm0,QWORD PTR [rax-0x8]
    8fc9:	divsd  xmm0,xmm2
    8fcd:	movsd  QWORD PTR [rdi+0x8],xmm0
    8fd2:	cmp    rsi,rax
    8fd5:	je     8758 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x438>
    8fdb:	movsd  xmm0,QWORD PTR [rax]
    8fdf:	add    rdi,0x10
    8fe3:	jmp    8fb8 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc98>
    8fe5:	nop    DWORD PTR [rax]
    8fe8:	mov    rcx,QWORD PTR [rsp+0x30]
    8fed:	pxor   xmm0,xmm0
    8ff1:	lea    rbp,[rsp+0x120]
    8ff9:	movaps XMMWORD PTR [rsp+0x120],xmm0
    9001:	mov    rdi,QWORD PTR [rcx]
    9004:	mov    rcx,QWORD PTR [rcx+0x8]
    9008:	movaps XMMWORD PTR [rsp+0x130],xmm0
    9010:	movaps XMMWORD PTR [rsp+0x140],xmm0
    9018:	movaps XMMWORD PTR [rsp+0x150],xmm0
    9020:	cmp    rcx,0x8
    9024:	je     9724 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1404>
    902a:	test   rcx,rcx
    902d:	je     898a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x66a>
    9033:	mov    rdx,rbp
    9036:	call   4180 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    903b:	jmp    898a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x66a>
    9040:	movsd  xmm0,QWORD PTR [rip+0x5a908]        # 63950 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x1d0>
    9048:	pxor   xmm10,xmm10
    904d:	jmp    8c80 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x960>
    9052:	mov    rax,QWORD PTR [rsp+0x30]
    9057:	lea    rdx,[rsp+0xa0]
    905f:	mov    QWORD PTR [rsp+0x8],rdx
    9064:	mov    rcx,QWORD PTR [rax+0x8]
    9068:	mov    rdi,QWORD PTR [rax]
    906b:	cmp    rcx,0x8
    906f:	je     9743 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1423>
    9075:	test   rcx,rcx
    9078:	je     84f7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1d7>
    907e:	mov    rsi,QWORD PTR [rsp+0x18]
    9083:	call   4180 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    9088:	jmp    84f7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1d7>
    908d:	movsd  xmm1,QWORD PTR [rip+0x5a76b]        # 63800 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
    9095:	mov    QWORD PTR [rsp+0x50],rax
    909a:	mov    QWORD PTR [rsp+0x48],rcx
    909f:	movapd xmm13,xmm1
    90a4:	maxsd  xmm13,xmm0
    90a9:	movapd xmm0,xmm13
    90ae:	call   6a10 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
    90b3:	movapd xmm1,xmm13
    90b8:	mov    rcx,QWORD PTR [rsp+0x48]
    90bd:	mov    rax,QWORD PTR [rsp+0x50]
    90c2:	mulsd  xmm1,xmm0
    90c6:	jmp    8f92 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc72>
    90cb:	mov    rax,QWORD PTR [rsp+0x18]
    90d0:	lea    rsi,[rsp+0x160]
    90d8:	xor    r9d,r9d
    90db:	mov    r8,rsi
    90de:	movdqu xmm0,XMMWORD PTR [rax]
    90e2:	movaps XMMWORD PTR [rsp+0x160],xmm0
    90ea:	movdqu xmm0,XMMWORD PTR [rax+0x10]
    90ef:	movaps XMMWORD PTR [rsp+0x170],xmm0
    90f7:	movdqu xmm0,XMMWORD PTR [rax+0x20]
    90fc:	movaps XMMWORD PTR [rsp+0x180],xmm0
    9104:	movdqu xmm0,XMMWORD PTR [rax+0x30]
    9109:	movaps XMMWORD PTR [rsp+0x190],xmm0
    9111:	mov    rdx,rcx
    9114:	pxor   xmm1,xmm1
    9118:	xor    eax,eax
    911a:	nop    DWORD PTR [rax+rax*1+0x0]
    911f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    912a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    9135:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    9140:	cmp    rax,rdi
    9143:	jae    9158 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xe38>
    9145:	movsd  xmm0,QWORD PTR [rdx+0x30]
    914a:	mulsd  xmm0,QWORD PTR [r15+rax*8+0x460]
    9154:	addsd  xmm1,xmm0
    9158:	add    rax,0x1
    915c:	add    rdx,0x40
    9160:	cmp    rax,0x8
    9164:	jne    9140 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xe20>
    9166:	addsd  xmm1,QWORD PTR [r8]
    916b:	add    r9,0x1
    916f:	add    r8,0x8
    9173:	add    rcx,0x8
    9177:	movsd  QWORD PTR [r8-0x8],xmm1
    917d:	cmp    r9,0x8
    9181:	jne    9111 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xdf1>
    9183:	cmp    r10,0x8
    9187:	je     9794 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1474>
    918d:	pxor   xmm0,xmm0
    9191:	movaps XMMWORD PTR [rsp+0x1a0],xmm0
    9199:	movaps XMMWORD PTR [rsp+0x1b0],xmm0
    91a1:	movaps XMMWORD PTR [rsp+0x1c0],xmm0
    91a9:	movaps XMMWORD PTR [rsp+0x1d0],xmm0
    91b1:	test   r10,r10
    91b4:	je     927c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf5c>
    91ba:	movsd  xmm0,QWORD PTR [rsp+0x160]
    91c3:	movsd  QWORD PTR [rsp+0x1a0],xmm0
    91cc:	cmp    r10,0x1
    91d0:	je     927c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf5c>
    91d6:	movsd  xmm0,QWORD PTR [rsp+0x168]
    91df:	movsd  QWORD PTR [rsp+0x1a8],xmm0
    91e8:	cmp    r10,0x2
    91ec:	je     927c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf5c>
    91f2:	movsd  xmm0,QWORD PTR [rsp+0x170]
    91fb:	movsd  QWORD PTR [rsp+0x1b0],xmm0
    9204:	cmp    r10,0x3
    9208:	je     927c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf5c>
    920a:	movsd  xmm0,QWORD PTR [rsp+0x178]
    9213:	movsd  QWORD PTR [rsp+0x1b8],xmm0
    921c:	cmp    r10,0x4
    9220:	je     927c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf5c>
    9222:	movsd  xmm0,QWORD PTR [rsp+0x180]
    922b:	movsd  QWORD PTR [rsp+0x1c0],xmm0
    9234:	cmp    r10,0x5
    9238:	je     927c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf5c>
    923a:	movsd  xmm0,QWORD PTR [rsp+0x188]
    9243:	movsd  QWORD PTR [rsp+0x1c8],xmm0
    924c:	cmp    r10,0x6
    9250:	je     927c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf5c>
    9252:	movsd  xmm0,QWORD PTR [rsp+0x190]
    925b:	movsd  QWORD PTR [rsp+0x1d0],xmm0
    9264:	cmp    r10,0x7
    9268:	je     927c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf5c>
    926a:	movsd  xmm0,QWORD PTR [rsp+0x198]
    9273:	movsd  QWORD PTR [rsp+0x1d8],xmm0
    927c:	mov    rax,QWORD PTR [rsp+0x30]
    9281:	pxor   xmm0,xmm0
    9285:	movaps XMMWORD PTR [rsp+0xa0],xmm0
    928d:	mov    rcx,QWORD PTR [rax+0x8]
    9291:	mov    rdi,QWORD PTR [rax]
    9294:	movaps XMMWORD PTR [rsp+0xb0],xmm0
    929c:	movaps XMMWORD PTR [rsp+0xc0],xmm0
    92a4:	movaps XMMWORD PTR [rsp+0xd0],xmm0
    92ac:	cmp    rcx,0x8
    92b0:	je     9872 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1552>
    92b6:	test   rcx,rcx
    92b9:	jne    9701 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x13e1>
    92bf:	test   r10,r10
    92c2:	jne    9657 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1337>
    92c8:	mov    QWORD PTR [rsp+0xa0],0x0
    92d4:	mov    QWORD PTR [rsp+0xa8],0x0
    92e0:	mov    QWORD PTR [rsp+0xb0],0x0
    92ec:	mov    QWORD PTR [rsp+0xb8],0x0
    92f8:	mov    QWORD PTR [rsp+0xc0],0x0
    9304:	mov    QWORD PTR [rsp+0xc8],0x0
    9310:	mov    QWORD PTR [rsp+0xd0],0x0
    931c:	mov    QWORD PTR [rsp+0xd8],0x0
    9328:	mov    rdx,QWORD PTR [rsp+0x8]
    932d:	mov    rcx,QWORD PTR [rsp+0x40]
    9332:	xor    eax,eax
    9334:	pxor   xmm0,xmm0
    9338:	cmp    rax,r10
    933b:	jae    9346 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1026>
    933d:	movsd  xmm0,QWORD PTR [rcx+rax*8]
    9342:	subsd  xmm0,QWORD PTR [rdx]
    9346:	add    rax,0x1
    934a:	movsd  QWORD PTR [rdx],xmm0
    934e:	add    rdx,0x8
    9352:	cmp    rax,0x8
    9356:	jne    9334 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1014>
    9358:	mov    rdi,QWORD PTR [rsp+0x8]
    935d:	mov    rsi,r10
    9360:	mov    QWORD PTR [rsp+0x20],r10
    9365:	movsd  QWORD PTR [rsp+0x48],xmm2
    936b:	call   2e2e0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    9370:	movsd  xmm2,QWORD PTR [rsp+0x48]
    9376:	mov    r10,QWORD PTR [rsp+0x20]
    937b:	comisd xmm2,xmm0
    937f:	jb     98d3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x15b3>
    9385:	movdqa xmm1,XMMWORD PTR [rsp+0x160]
    938e:	mov    rax,QWORD PTR [rsp+0x18]
    9393:	movups XMMWORD PTR [rax],xmm1
    9396:	movdqa xmm1,XMMWORD PTR [rsp+0x170]
    939f:	movups XMMWORD PTR [rax+0x10],xmm1
    93a3:	movdqa xmm1,XMMWORD PTR [rsp+0x180]
    93ac:	movups XMMWORD PTR [rax+0x20],xmm1
    93b0:	movdqa xmm1,XMMWORD PTR [rsp+0x190]
    93b9:	movups XMMWORD PTR [rax+0x30],xmm1
    93bd:	movsd  QWORD PTR [r15+0x4a0],xmm0
    93c6:	movsd  xmm7,QWORD PTR [rsp+0x38]
    93cc:	mov    edx,r12d
    93cf:	comisd xmm7,xmm0
    93d3:	setb   al
    93d6:	setae  BYTE PTR [r15+0x4b0]
    93de:	and    edx,eax
    93e0:	mov    BYTE PTR [r15+0x4b1],dl
    93e7:	jmp    85f2 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x2d2>
    93ec:	movsd  xmm0,QWORD PTR [rsp+0xf0]
    93f5:	movsd  QWORD PTR [rsp+0x1b0],xmm0
    93fe:	cmp    r10,0x3
    9402:	je     88d5 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x5b5>
    9408:	movsd  xmm0,QWORD PTR [rsp+0xf8]
    9411:	movsd  QWORD PTR [rsp+0x1b8],xmm0
    941a:	cmp    r10,0x4
    941e:	je     88d5 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x5b5>
    9424:	movsd  xmm0,QWORD PTR [rsp+0x100]
    942d:	movsd  QWORD PTR [rsp+0x1c0],xmm0
    9436:	cmp    r10,0x5
    943a:	je     88d5 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x5b5>
    9440:	movsd  xmm0,QWORD PTR [rsp+0x108]
    9449:	movsd  QWORD PTR [rsp+0x1c8],xmm0
    9452:	cmp    r10,0x6
    9456:	je     88d5 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x5b5>
    945c:	movsd  xmm0,QWORD PTR [rsp+0x110]
    9465:	movsd  QWORD PTR [rsp+0x1d0],xmm0
    946e:	cmp    r10,0x7
    9472:	je     88d5 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x5b5>
    9478:	movsd  xmm0,QWORD PTR [rsp+0x118]
    9481:	movsd  QWORD PTR [rsp+0x1d8],xmm0
    948a:	jmp    88d5 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x5b5>
    948f:	nop
    9490:	cmp    rcx,0x8
    9494:	je     9895 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1575>
    949a:	test   rcx,rcx
    949d:	je     9731 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1411>
    94a3:	lea    rdx,[rsp+0xa0]
    94ab:	lea    rsi,[rsp+0x1a0]
    94b3:	mov    QWORD PTR [rsp+0x8],rdx
    94b8:	call   4180 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    94bd:	jmp    84af <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x18f>
    94c2:	lea    rsi,[rsp+0x1a0]
    94ca:	mov    rdx,rbp
    94cd:	call   5a50 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    94d2:	pxor   xmm5,xmm5
    94d6:	jmp    8930 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x610>
    94db:	cmp    rcx,0x8
    94df:	je     97f2 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x14d2>
    94e5:	test   rcx,rcx
    94e8:	jne    95ba <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x129a>
    94ee:	lea    rax,[rsp+0xa0]
    94f6:	mov    QWORD PTR [rsp+0x8],rax
    94fb:	jmp    84a3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x183>
    9500:	comisd xmm0,QWORD PTR [rip+0x5a520]        # 63a28 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x2a8>
    9508:	pxor   xmm1,xmm1
    950c:	jb     951a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x11fa>
    950e:	movsd  xmm1,QWORD PTR [rip+0x5a43a]        # 63950 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x1d0>
    9516:	divsd  xmm1,xmm0
    951a:	movsd  xmm0,QWORD PTR [rbp+0x0]
    951f:	comisd xmm7,QWORD PTR [rip+0x5a501]        # 63a28 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x2a8>
    9527:	lea    rdx,[r12-0x40]
    952c:	lea    rsi,[rbp+0x40]
    9530:	jb     9580 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1260>
    9532:	mulsd  xmm0,xmm1
    9536:	add    r8,0x10
    953a:	movsd  QWORD PTR [rdx],xmm0
    953e:	movsd  xmm0,QWORD PTR [r8-0x8]
    9544:	mulsd  xmm0,xmm1
    9548:	movsd  QWORD PTR [rdx+0x8],xmm0
    954d:	cmp    rsi,r8
    9550:	je     8b13 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x7f3>
    9556:	movsd  xmm0,QWORD PTR [r8]
    955b:	add    rdx,0x10
    955f:	jmp    9532 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1212>
    9561:	nop    WORD PTR [rax+rax*1+0x0]
    956a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    9575:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    9580:	divsd  xmm0,xmm7
    9584:	add    r8,0x10
    9588:	movsd  QWORD PTR [rdx],xmm0
    958c:	movsd  xmm0,QWORD PTR [r8-0x8]
    9592:	divsd  xmm0,xmm7
    9596:	movsd  QWORD PTR [rdx+0x8],xmm0
    959b:	cmp    rsi,r8
    959e:	je     8b13 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x7f3>
    95a4:	movsd  xmm0,QWORD PTR [r8]
    95a9:	add    rdx,0x10
    95ad:	jmp    9580 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1260>
    95af:	nop
    95b0:	movapd xmm14,xmm10
    95b5:	jmp    8f92 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc72>
    95ba:	lea    rdx,[rsp+0xa0]
    95c2:	lea    rsi,[rsp+0x1a0]
    95ca:	mov    QWORD PTR [rsp+0x8],rdx
    95cf:	call   4180 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    95d4:	jmp    84a3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x183>
    95d9:	movsd  xmm2,QWORD PTR [rsp+0x78]
    95df:	movzx  edx,BYTE PTR [rsp+0x96]
    95e7:	mov    edi,0x7
    95ec:	mov    rbp,QWORD PTR [rsp+0x80]
    95f4:	mov    rbx,QWORD PTR [rsp+0x88]
    95fc:	movzx  r14d,BYTE PTR [rsp+0x97]
    9605:	mov    rcx,QWORD PTR [rsp+0x98]
    960d:	movzx  r12d,BYTE PTR [rsp+0x20]
    9613:	jmp    8d83 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xa63>
    9618:	movsd  xmm2,QWORD PTR [rsp+0x78]
    961e:	movzx  edx,BYTE PTR [rsp+0x96]
    9626:	mov    edi,0x8
    962b:	mov    rbp,QWORD PTR [rsp+0x80]
    9633:	mov    rbx,QWORD PTR [rsp+0x88]
    963b:	movzx  r14d,BYTE PTR [rsp+0x97]
    9644:	mov    rcx,QWORD PTR [rsp+0x98]
    964c:	movzx  r12d,BYTE PTR [rsp+0x20]
    9652:	jmp    8d83 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xa63>
    9657:	cmp    r10,0x1
    965b:	je     92d4 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xfb4>
    9661:	cmp    r10,0x2
    9665:	je     92e0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xfc0>
    966b:	cmp    r10,0x3
    966f:	je     92ec <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xfcc>
    9675:	cmp    r10,0x4
    9679:	je     92f8 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xfd8>
    967f:	cmp    r10,0x5
    9683:	je     9304 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xfe4>
    9689:	cmp    r10,0x6
    968d:	je     9310 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xff0>
    9693:	cmp    r10,0x7
    9697:	jne    9328 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1008>
    969d:	jmp    931c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xffc>
    96a2:	mov    rax,QWORD PTR [rsp+0x18]
    96a7:	movsd  xmm0,QWORD PTR [rax+0x18]
    96ac:	movsd  QWORD PTR [rsp+0x1b8],xmm0
    96b5:	cmp    r12,0x4
    96b9:	je     9750 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1430>
    96bf:	mov    rax,QWORD PTR [rsp+0x18]
    96c4:	movsd  xmm0,QWORD PTR [rax+0x20]
    96c9:	movsd  QWORD PTR [rsp+0x1c0],xmm0
    96d2:	cmp    r12,0x5
    96d6:	jne    9811 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x14f1>
    96dc:	cmp    rcx,0x8
    96e0:	je     9972 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1652>
    96e6:	test   rcx,rcx
    96e9:	jne    9991 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1671>
    96ef:	lea    rax,[rsp+0xa0]
    96f7:	mov    QWORD PTR [rsp+0x8],rax
    96fc:	jmp    84d3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1b3>
    9701:	mov    rdx,QWORD PTR [rsp+0x8]
    9706:	lea    rsi,[rsp+0x1a0]
    970e:	movsd  QWORD PTR [rsp+0x20],xmm2
    9714:	call   4180 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    9719:	movsd  xmm2,QWORD PTR [rsp+0x20]
    971f:	jmp    92bf <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf9f>
    9724:	mov    rdx,rbp
    9727:	call   5a50 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    972c:	jmp    898a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x66a>
    9731:	lea    rax,[rsp+0xa0]
    9739:	mov    QWORD PTR [rsp+0x8],rax
    973e:	jmp    84af <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x18f>
    9743:	mov    rsi,r14
    9746:	call   5a50 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    974b:	jmp    84f7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1d7>
    9750:	cmp    rcx,0x8
    9754:	je     9933 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1613>
    975a:	test   rcx,rcx
    975d:	jne    98b4 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1594>
    9763:	lea    rax,[rsp+0xa0]
    976b:	mov    QWORD PTR [rsp+0x8],rax
    9770:	jmp    84c7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1a7>
    9775:	lea    rdx,[rsp+0xa0]
    977d:	lea    rsi,[rsp+0x1a0]
    9785:	mov    QWORD PTR [rsp+0x8],rdx
    978a:	call   5a50 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    978f:	jmp    84bb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x19b>
    9794:	mov    rax,QWORD PTR [rsp+0x30]
    9799:	pxor   xmm0,xmm0
    979d:	movaps XMMWORD PTR [rsp+0xa0],xmm0
    97a5:	mov    rcx,QWORD PTR [rax+0x8]
    97a9:	mov    rdi,QWORD PTR [rax]
    97ac:	movaps XMMWORD PTR [rsp+0xb0],xmm0
    97b4:	movaps XMMWORD PTR [rsp+0xc0],xmm0
    97bc:	movaps XMMWORD PTR [rsp+0xd0],xmm0
    97c4:	cmp    rcx,0x8
    97c8:	je     9952 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1632>
    97ce:	test   rcx,rcx
    97d1:	je     9328 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1008>
    97d7:	mov    rdx,QWORD PTR [rsp+0x8]
    97dc:	movsd  QWORD PTR [rsp+0x20],xmm2
    97e2:	call   4180 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    97e7:	movsd  xmm2,QWORD PTR [rsp+0x20]
    97ed:	jmp    9328 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1008>
    97f2:	lea    rdx,[rsp+0xa0]
    97fa:	lea    rsi,[rsp+0x1a0]
    9802:	mov    QWORD PTR [rsp+0x8],rdx
    9807:	call   5a50 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    980c:	jmp    84a3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x183>
    9811:	mov    rax,QWORD PTR [rsp+0x18]
    9816:	movsd  xmm0,QWORD PTR [rax+0x28]
    981b:	movsd  QWORD PTR [rsp+0x1c8],xmm0
    9824:	cmp    r12,0x6
    9828:	jne    99ee <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x16ce>
    982e:	cmp    rcx,0x8
    9832:	je     99cf <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x16af>
    9838:	test   rcx,rcx
    983b:	jne    99b0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1690>
    9841:	lea    rax,[rsp+0xa0]
    9849:	mov    QWORD PTR [rsp+0x8],rax
    984e:	jmp    84df <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1bf>
    9853:	lea    rdx,[rsp+0xa0]
    985b:	lea    rsi,[rsp+0x1a0]
    9863:	mov    QWORD PTR [rsp+0x8],rdx
    9868:	call   4180 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    986d:	jmp    84bb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x19b>
    9872:	mov    rdx,QWORD PTR [rsp+0x8]
    9877:	lea    rsi,[rsp+0x1a0]
    987f:	movsd  QWORD PTR [rsp+0x20],xmm2
    9885:	call   5a50 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    988a:	movsd  xmm2,QWORD PTR [rsp+0x20]
    9890:	jmp    92bf <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf9f>
    9895:	lea    rdx,[rsp+0xa0]
    989d:	lea    rsi,[rsp+0x1a0]
    98a5:	mov    QWORD PTR [rsp+0x8],rdx
    98aa:	call   5a50 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    98af:	jmp    84af <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x18f>
    98b4:	lea    rdx,[rsp+0xa0]
    98bc:	lea    rsi,[rsp+0x1a0]
    98c4:	mov    QWORD PTR [rsp+0x8],rdx
    98c9:	call   4180 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    98ce:	jmp    84c7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1a7>
    98d3:	mov    r13,QWORD PTR [rsp+0x8]
    98d8:	mov    rsi,QWORD PTR [rsp+0x18]
    98dd:	mov    rcx,r10
    98e0:	mov    QWORD PTR [rsp+0x20],r10
    98e5:	mov    rdi,QWORD PTR [rsp+0x30]
    98ea:	mov    rdx,r13
    98ed:	call   4d130 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m>
    98f2:	mov    rcx,QWORD PTR [rsp+0x40]
    98f7:	mov    rdx,r13
    98fa:	xor    eax,eax
    98fc:	mov    r10,QWORD PTR [rsp+0x20]
    9901:	pxor   xmm0,xmm0
    9905:	cmp    rax,r10
    9908:	jae    9913 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x15f3>
    990a:	movsd  xmm0,QWORD PTR [rcx+rax*8]
    990f:	subsd  xmm0,QWORD PTR [rdx]
    9913:	add    rax,0x1
    9917:	movsd  QWORD PTR [rdx],xmm0
    991b:	add    rdx,0x8
    991f:	cmp    rax,0x8
    9923:	jne    9901 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x15e1>
    9925:	movsd  xmm0,QWORD PTR [r15+0x4a0]
    992e:	jmp    93c6 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x10a6>
    9933:	lea    rdx,[rsp+0xa0]
    993b:	lea    rsi,[rsp+0x1a0]
    9943:	mov    QWORD PTR [rsp+0x8],rdx
    9948:	call   5a50 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    994d:	jmp    84c7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1a7>
    9952:	mov    rdx,QWORD PTR [rsp+0x8]
    9957:	movsd  QWORD PTR [rsp+0x20],xmm2
    995d:	call   5a50 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    9962:	movsd  xmm2,QWORD PTR [rsp+0x20]
    9968:	jmp    9328 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1008>
    996d:	call   1170 <__stack_chk_fail@plt>
    9972:	lea    rdx,[rsp+0xa0]
    997a:	lea    rsi,[rsp+0x1a0]
    9982:	mov    QWORD PTR [rsp+0x8],rdx
    9987:	call   5a50 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    998c:	jmp    84d3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1b3>
    9991:	lea    rdx,[rsp+0xa0]
    9999:	lea    rsi,[rsp+0x1a0]
    99a1:	mov    QWORD PTR [rsp+0x8],rdx
    99a6:	call   4180 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    99ab:	jmp    84d3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1b3>
    99b0:	lea    rdx,[rsp+0xa0]
    99b8:	lea    rsi,[rsp+0x1a0]
    99c0:	mov    QWORD PTR [rsp+0x8],rdx
    99c5:	call   4180 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    99ca:	jmp    84df <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1bf>
    99cf:	lea    rdx,[rsp+0xa0]
    99d7:	lea    rsi,[rsp+0x1a0]
    99df:	mov    QWORD PTR [rsp+0x8],rdx
    99e4:	call   5a50 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    99e9:	jmp    84df <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1bf>
    99ee:	mov    rax,QWORD PTR [rsp+0x18]
    99f3:	movsd  xmm0,QWORD PTR [rax+0x30]
    99f8:	movsd  QWORD PTR [rsp+0x1d0],xmm0
    9a01:	cmp    r12,0x7
    9a05:	jne    9a24 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1704>
    9a07:	cmp    rcx,0x8
    9a0b:	je     9a77 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1757>
    9a0d:	test   rcx,rcx
    9a10:	jne    9a58 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1738>
    9a12:	lea    rax,[rsp+0xa0]
    9a1a:	mov    QWORD PTR [rsp+0x8],rax
    9a1f:	jmp    84eb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1cb>
    9a24:	mov    rax,QWORD PTR [rsp+0x18]
    9a29:	movsd  xmm0,QWORD PTR [rax+0x38]
    9a2e:	movsd  QWORD PTR [rsp+0x1d8],xmm0
    9a37:	cmp    rcx,0x8
    9a3b:	je     9af1 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x17d1>
    9a41:	test   rcx,rcx
    9a44:	jne    9a96 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1776>
    9a46:	lea    rax,[rsp+0xa0]
    9a4e:	mov    QWORD PTR [rsp+0x8],rax
    9a53:	jmp    84f7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1d7>
    9a58:	lea    rdx,[rsp+0xa0]
    9a60:	lea    rsi,[rsp+0x1a0]
    9a68:	mov    QWORD PTR [rsp+0x8],rdx
    9a6d:	call   4180 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    9a72:	jmp    84eb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1cb>
    9a77:	lea    rdx,[rsp+0xa0]
    9a7f:	lea    rsi,[rsp+0x1a0]
    9a87:	mov    QWORD PTR [rsp+0x8],rdx
    9a8c:	call   5a50 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    9a91:	jmp    84eb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1cb>
    9a96:	lea    rdx,[rsp+0xa0]
    9a9e:	lea    rsi,[rsp+0x1a0]
    9aa6:	mov    QWORD PTR [rsp+0x8],rdx
    9aab:	call   4180 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    9ab0:	cmp    r12,0x2
    9ab4:	je     84af <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x18f>
    9aba:	cmp    r12,0x3
    9abe:	je     84bb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x19b>
    9ac4:	cmp    r12,0x4
    9ac8:	je     84c7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1a7>
    9ace:	cmp    r12,0x5
    9ad2:	je     84d3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1b3>
    9ad8:	cmp    r12,0x6
    9adc:	je     84df <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1bf>
    9ae2:	cmp    r12,0x7
    9ae6:	jne    84f7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1d7>
    9aec:	jmp    84eb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1cb>
    9af1:	lea    rdx,[rsp+0xa0]
    9af9:	lea    rsi,[rsp+0x1a0]
    9b01:	mov    QWORD PTR [rsp+0x8],rdx
    9b06:	call   5a50 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    9b0b:	jmp    9ace <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x17ae>

Disassembly of section .fini:
