
/tmp/matrix_peak_stack_audit/pre_constants_ab31/pre_constants_ab31_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000008600 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0>:
    8600:	push   r15
    8602:	pxor   xmm0,xmm0
    8606:	mov    r15,rsi
    8609:	push   r14
    860b:	movapd xmm1,xmm0
    860f:	mov    r14,rdi
    8612:	push   r13
    8614:	push   r12
    8616:	push   rbp
    8617:	mov    rbp,r8
    861a:	push   rbx
    861b:	sub    rsp,0x1f8
    8622:	movsd  xmm2,QWORD PTR [rsi+0x18]
    8627:	mov    rbx,QWORD PTR [rsi+0x8]
    862b:	movsd  xmm3,QWORD PTR [rip+0x5b1bd]        # 637f0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x70>
    8633:	mov    QWORD PTR [rsp+0x18],rdi
    8638:	mov    rdi,rcx
    863b:	cmpnltsd xmm1,xmm2
    8640:	mov    QWORD PTR [rsp+0x30],rdx
    8645:	comisd xmm0,xmm2
    8649:	mov    QWORD PTR [rsp+0x40],rcx
    864e:	andpd  xmm3,xmm1
    8652:	andnpd xmm1,xmm2
    8656:	movapd xmm2,xmm0
    865a:	mov    r12,QWORD PTR fs:0x28
    8663:	mov    QWORD PTR [rsp+0x1e8],r12
    866b:	mov    r12,r9
    866e:	orpd   xmm1,xmm3
    8672:	setae  BYTE PTR [rsi+0x4b2]
    8679:	xor    eax,eax
    867b:	movsd  QWORD PTR [rsp+0x70],xmm1
    8681:	movsd  xmm1,QWORD PTR [rsi+0x20]
    8686:	mov    WORD PTR [rsi+0x4b0],ax
    868d:	xor    eax,eax
    868f:	cmp    QWORD PTR [rsi+0x10],0x1
    8694:	cmpltsd xmm2,xmm1
    8699:	seta   al
    869c:	mov    QWORD PTR [rsi+0x4a8],0x0
    86a7:	add    rax,0x1
    86ab:	mov    QWORD PTR [rsp+0x28],rax
    86b0:	andpd  xmm1,xmm2
    86b4:	movsd  QWORD PTR [rsp+0x8],xmm1
    86ba:	movsd  xmm1,QWORD PTR [rsi+0x28]
    86bf:	mov    rsi,r9
    86c2:	cmpltsd xmm0,xmm1
    86c7:	movapd xmm6,xmm1
    86cb:	andpd  xmm6,xmm0
    86cf:	movsd  QWORD PTR [rsp+0x10],xmm6
    86d5:	call   33820 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    86da:	mulsd  xmm0,QWORD PTR [rsp+0x8]
    86e0:	addsd  xmm0,QWORD PTR [rsp+0x10]
    86e6:	movsd  QWORD PTR [rsp+0x38],xmm0
    86ec:	movdqu xmm0,XMMWORD PTR [rbp+0x0]
    86f1:	movups XMMWORD PTR [r14],xmm0
    86f5:	movdqu xmm0,XMMWORD PTR [rbp+0x10]
    86fa:	movups XMMWORD PTR [r14+0x10],xmm0
    86ff:	movdqu xmm0,XMMWORD PTR [rbp+0x20]
    8704:	movups XMMWORD PTR [r14+0x20],xmm0
    8709:	movdqu xmm0,XMMWORD PTR [rbp+0x30]
    870e:	movups XMMWORD PTR [r14+0x30],xmm0
    8713:	pxor   xmm0,xmm0
    8717:	movaps XMMWORD PTR [rsp+0xa0],xmm0
    871f:	movaps XMMWORD PTR [rsp+0xb0],xmm0
    8727:	movaps XMMWORD PTR [rsp+0xc0],xmm0
    872f:	movaps XMMWORD PTR [rsp+0xd0],xmm0
    8737:	cmp    r12,0x8
    873b:	je     9312 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xd12>
    8741:	movaps XMMWORD PTR [rsp+0x1a0],xmm0
    8749:	movaps XMMWORD PTR [rsp+0x1b0],xmm0
    8751:	movaps XMMWORD PTR [rsp+0x1c0],xmm0
    8759:	movaps XMMWORD PTR [rsp+0x1d0],xmm0
    8761:	test   r12,r12
    8764:	jne    88e9 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x2e9>
    876a:	lea    rax,[rsp+0xa0]
    8772:	mov    QWORD PTR [rsp+0xa0],0x0
    877e:	mov    QWORD PTR [rsp+0x8],rax
    8783:	mov    QWORD PTR [rsp+0xa8],0x0
    878f:	mov    QWORD PTR [rsp+0xb0],0x0
    879b:	mov    QWORD PTR [rsp+0xb8],0x0
    87a7:	mov    QWORD PTR [rsp+0xc0],0x0
    87b3:	mov    QWORD PTR [rsp+0xc8],0x0
    87bf:	mov    QWORD PTR [rsp+0xd0],0x0
    87cb:	mov    QWORD PTR [rsp+0xd8],0x0
    87d7:	mov    rdx,QWORD PTR [rsp+0x8]
    87dc:	mov    rcx,QWORD PTR [rsp+0x40]
    87e1:	xor    eax,eax
    87e3:	nop    DWORD PTR [rax+0x0]
    87ea:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    87f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8800:	pxor   xmm0,xmm0
    8804:	cmp    rax,r12
    8807:	jae    8812 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x212>
    8809:	movsd  xmm0,QWORD PTR [rcx+rax*8]
    880e:	subsd  xmm0,QWORD PTR [rdx]
    8812:	add    rax,0x1
    8816:	movsd  QWORD PTR [rdx],xmm0
    881a:	add    rdx,0x8
    881e:	cmp    rax,0x8
    8822:	jne    8800 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x200>
    8824:	mov    rdi,QWORD PTR [rsp+0x8]
    8829:	mov    rsi,r12
    882c:	call   33820 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    8831:	movsd  xmm7,QWORD PTR [rsp+0x38]
    8837:	mov    r10,QWORD PTR [r15]
    883a:	movsd  QWORD PTR [r15+0x4a0],xmm0
    8843:	comisd xmm7,xmm0
    8847:	setae  dl
    884a:	mov    BYTE PTR [r15+0x4b0],dl
    8851:	test   r10,r10
    8854:	je     88bb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x2bb>
    8856:	mov    ecx,0x8
    885b:	cmp    rbx,rcx
    885e:	cmova  rbx,rcx
    8862:	mov    ecx,0x1
    8867:	test   rbx,rbx
    886a:	cmove  rbx,rcx
    886e:	cmp    rbx,r12
    8871:	cmova  rbx,r12
    8875:	test   rbx,rbx
    8878:	mov    QWORD PTR [rsp+0x10],rbx
    887d:	mov    rbx,r10
    8880:	mov    r10,r12
    8883:	setne  r14b
    8887:	xor    ebp,ebp
    8889:	jmp    8898 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x298>
    888b:	nop    DWORD PTR [rax+rax*1+0x0]
    8890:	movzx  edx,BYTE PTR [r15+0x4b0]
    8898:	test   dl,dl
    889a:	jne    88b2 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x2b2>
    889c:	movzx  eax,BYTE PTR [r15+0x4b1]
    88a4:	xor    eax,0x1
    88a7:	and    al,r14b
    88aa:	mov    edx,eax
    88ac:	jne    8970 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x370>
    88b2:	add    rbp,0x1
    88b6:	cmp    rbp,rbx
    88b9:	jne    8890 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x290>
    88bb:	mov    rax,QWORD PTR [rsp+0x1e8]
    88c3:	sub    rax,QWORD PTR fs:0x28
    88cc:	jne    9c2d <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x162d>
    88d2:	mov    rax,QWORD PTR [rsp+0x18]
    88d7:	add    rsp,0x1f8
    88de:	pop    rbx
    88df:	pop    rbp
    88e0:	pop    r12
    88e2:	pop    r13
    88e4:	pop    r14
    88e6:	pop    r15
    88e8:	ret
    88e9:	mov    rax,QWORD PTR [rsp+0x30]
    88ee:	mov    rdi,QWORD PTR [rax]
    88f1:	mov    rcx,QWORD PTR [rax+0x8]
    88f5:	mov    rax,QWORD PTR [rsp+0x18]
    88fa:	movsd  xmm0,QWORD PTR [rax]
    88fe:	movsd  QWORD PTR [rsp+0x1a0],xmm0
    8907:	cmp    r12,0x1
    890b:	je     979b <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x119b>
    8911:	mov    rax,QWORD PTR [rsp+0x18]
    8916:	movsd  xmm0,QWORD PTR [rax+0x8]
    891b:	movsd  QWORD PTR [rsp+0x1a8],xmm0
    8924:	cmp    r12,0x2
    8928:	je     9750 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1150>
    892e:	mov    rax,QWORD PTR [rsp+0x18]
    8933:	movsd  xmm0,QWORD PTR [rax+0x10]
    8938:	movsd  QWORD PTR [rsp+0x1b0],xmm0
    8941:	cmp    r12,0x3
    8945:	jne    9962 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1362>
    894b:	cmp    rcx,0x8
    894f:	je     9a35 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1435>
    8955:	test   rcx,rcx
    8958:	jne    9b13 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1513>
    895e:	lea    rax,[rsp+0xa0]
    8966:	mov    QWORD PTR [rsp+0x8],rax
    896b:	jmp    879b <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x19b>
    8970:	movsd  xmm2,QWORD PTR [r15+0x4a0]
    8979:	comisd xmm2,QWORD PTR [rip+0x5b077]        # 639f8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x278>
    8981:	pxor   xmm0,xmm0
    8985:	pxor   xmm1,xmm1
    8989:	movups XMMWORD PTR [r15+0x398],xmm0
    8991:	mov    QWORD PTR [r15+0x3d8],0x0
    899c:	movups XMMWORD PTR [r15+0x3a8],xmm0
    89a4:	movups XMMWORD PTR [r15+0x3b8],xmm0
    89ac:	movups XMMWORD PTR [r15+0x3c8],xmm0
    89b4:	movsd  QWORD PTR [r15+0x398],xmm2
    89bd:	jb     89cb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x3cb>
    89bf:	movsd  xmm1,QWORD PTR [rip+0x5ae61]        # 63828 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
    89c7:	divsd  xmm1,xmm2
    89cb:	comisd xmm2,QWORD PTR [rip+0x5b025]        # 639f8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x278>
    89d3:	mov    rax,QWORD PTR [rsp+0x8]
    89d8:	movsd  xmm0,QWORD PTR [rax]
    89dc:	lea    rsi,[rax+0x40]
    89e0:	jb     926f <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc6f>
    89e6:	lea    rdi,[r15+0x30]
    89ea:	mulsd  xmm0,xmm1
    89ee:	add    rax,0x10
    89f2:	movsd  QWORD PTR [rdi],xmm0
    89f6:	movsd  xmm0,QWORD PTR [rax-0x8]
    89fb:	mulsd  xmm0,xmm1
    89ff:	movsd  QWORD PTR [rdi+0x8],xmm0
    8a04:	cmp    rsi,rax
    8a07:	je     8a18 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x418>
    8a09:	movsd  xmm0,QWORD PTR [rax]
    8a0d:	add    rdi,0x10
    8a11:	jmp    89ea <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x3ea>
    8a13:	nop    DWORD PTR [rax+rax*1+0x0]
    8a18:	pxor   xmm0,xmm0
    8a1c:	mov    BYTE PTR [rsp+0x97],r14b
    8a24:	mov    esi,edx
    8a26:	xor    edi,edi
    8a28:	mov    BYTE PTR [rsp+0x96],dl
    8a2f:	pxor   xmm5,xmm5
    8a33:	xor    r14d,r14d
    8a36:	xor    ecx,ecx
    8a38:	mov    QWORD PTR [rsp+0x80],rbp
    8a40:	lea    r13,[r15+0x358]
    8a47:	lea    r12,[r15+0xb0]
    8a4e:	mov    QWORD PTR [rsp+0x88],rbx
    8a56:	mov    QWORD PTR [rsp+0x98],r15
    8a5e:	movaps XMMWORD PTR [rsp+0xe0],xmm0
    8a66:	movaps XMMWORD PTR [rsp+0xf0],xmm0
    8a6e:	movaps XMMWORD PTR [rsp+0x100],xmm0
    8a76:	movaps XMMWORD PTR [rsp+0x110],xmm0
    8a7e:	movaps XMMWORD PTR [rsp+0x120],xmm0
    8a86:	movaps XMMWORD PTR [rsp+0x130],xmm0
    8a8e:	movaps XMMWORD PTR [rsp+0x140],xmm0
    8a96:	movaps XMMWORD PTR [rsp+0x150],xmm0
    8a9e:	movsd  QWORD PTR [rsp+0x78],xmm2
    8aa4:	jmp    8ad5 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x4d5>
    8aa6:	nop    DWORD PTR [rax+0x0]
    8aaa:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8ab5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8ac0:	mov    r14,rax
    8ac3:	add    r13,0x8
    8ac7:	add    r12,0x40
    8acb:	cmp    r14,0x8
    8acf:	je     9011 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xa11>
    8ad5:	cmp    r14,QWORD PTR [rsp+0x10]
    8ada:	lea    rax,[r14+0x1]
    8ade:	setb   dl
    8ae1:	and    dl,sil
    8ae4:	mov    BYTE PTR [rsp+0x20],dl
    8ae8:	je     8ac0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x4c0>
    8aea:	movdqu xmm0,XMMWORD PTR [r12-0x80]
    8af1:	lea    rsi,[rsp+0xe0]
    8af9:	movaps XMMWORD PTR [rsp+0xe0],xmm0
    8b01:	movdqu xmm0,XMMWORD PTR [r12-0x70]
    8b08:	movaps XMMWORD PTR [rsp+0xf0],xmm0
    8b10:	movdqu xmm0,XMMWORD PTR [r12-0x60]
    8b17:	movaps XMMWORD PTR [rsp+0x100],xmm0
    8b1f:	movdqu xmm0,XMMWORD PTR [r12-0x50]
    8b26:	movaps XMMWORD PTR [rsp+0x110],xmm0
    8b2e:	cmp    r10,0x8
    8b32:	je     92a8 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xca8>
    8b38:	pxor   xmm0,xmm0
    8b3c:	movaps XMMWORD PTR [rsp+0x1a0],xmm0
    8b44:	movaps XMMWORD PTR [rsp+0x1b0],xmm0
    8b4c:	movaps XMMWORD PTR [rsp+0x1c0],xmm0
    8b54:	movaps XMMWORD PTR [rsp+0x1d0],xmm0
    8b5c:	test   r10,r10
    8b5f:	je     8b95 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x595>
    8b61:	movsd  xmm0,QWORD PTR [rsp+0xe0]
    8b6a:	movsd  QWORD PTR [rsp+0x1a0],xmm0
    8b73:	cmp    r10,0x1
    8b77:	je     8b95 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x595>
    8b79:	movsd  xmm0,QWORD PTR [rsp+0xe8]
    8b82:	movsd  QWORD PTR [rsp+0x1a8],xmm0
    8b8b:	cmp    r10,0x2
    8b8f:	jne    96ac <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x10ac>
    8b95:	mov    rbx,QWORD PTR [rsp+0x30]
    8b9a:	pxor   xmm0,xmm0
    8b9e:	lea    rbp,[rsp+0x120]
    8ba6:	movaps XMMWORD PTR [rsp+0x120],xmm0
    8bae:	mov    rcx,QWORD PTR [rbx+0x8]
    8bb2:	mov    rdi,QWORD PTR [rbx]
    8bb5:	movaps XMMWORD PTR [rsp+0x130],xmm0
    8bbd:	movaps XMMWORD PTR [rsp+0x140],xmm0
    8bc5:	movaps XMMWORD PTR [rsp+0x150],xmm0
    8bcd:	cmp    rcx,0x8
    8bd1:	je     9782 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1182>
    8bd7:	test   rcx,rcx
    8bda:	je     8bf0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x5f0>
    8bdc:	lea    rsi,[rsp+0x1a0]
    8be4:	mov    rdx,rbp
    8be7:	call   4240 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    8bec:	pxor   xmm5,xmm5
    8bf0:	test   r10,r10
    8bf3:	jne    91a9 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xba9>
    8bf9:	mov    QWORD PTR [rsp+0x120],0x0
    8c05:	movsd  QWORD PTR [rsp+0x128],xmm5
    8c0e:	mov    QWORD PTR [rsp+0x130],0x0
    8c1a:	movsd  QWORD PTR [rsp+0x138],xmm5
    8c23:	mov    QWORD PTR [rsp+0x140],0x0
    8c2f:	movsd  QWORD PTR [rsp+0x148],xmm5
    8c38:	movsd  QWORD PTR [rsp+0x150],xmm5
    8c41:	movsd  QWORD PTR [rsp+0x158],xmm5
    8c4a:	mov    rsi,r10
    8c4d:	mov    rdi,rbp
    8c50:	mov    QWORD PTR [rsp+0x60],rax
    8c55:	mov    QWORD PTR [rsp+0x58],r10
    8c5a:	call   33820 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    8c5f:	mov    QWORD PTR [rsp+0x50],r14
    8c64:	mov    rax,QWORD PTR [rsp+0x60]
    8c69:	xor    r9d,r9d
    8c6c:	mov    QWORD PTR [r15+0x390],0x0
    8c77:	mov    r10,QWORD PTR [rsp+0x58]
    8c7c:	movsd  QWORD PTR [rsp+0x48],xmm0
    8c82:	pxor   xmm0,xmm0
    8c86:	movups XMMWORD PTR [r15+0x350],xmm0
    8c8e:	movups XMMWORD PTR [r15+0x360],xmm0
    8c96:	movups XMMWORD PTR [r15+0x370],xmm0
    8c9e:	movups XMMWORD PTR [r15+0x380],xmm0
    8ca6:	lea    rcx,[r15+0x350]
    8cad:	lea    rbx,[r15+0x70]
    8cb1:	mov    r14,rcx
    8cb4:	lea    rdi,[r15+0x30]
    8cb8:	nop    DWORD PTR [rax+rax*1+0x0]
    8cc0:	pxor   xmm2,xmm2
    8cc4:	xor    edx,edx
    8cc6:	movapd xmm1,xmm2
    8cca:	movupd xmm0,XMMWORD PTR [rdi+rdx*1]
    8ccf:	movapd xmm4,xmm0
    8cd3:	mulpd  xmm4,xmm0
    8cd7:	mulpd  xmm0,XMMWORD PTR [rbp+rdx*1+0x0]
    8cdd:	add    rdx,0x10
    8ce1:	movapd xmm3,xmm4
    8ce5:	unpckhpd xmm4,xmm4
    8ce9:	addsd  xmm3,xmm2
    8ced:	addsd  xmm1,xmm0
    8cf1:	movapd xmm2,xmm4
    8cf5:	unpckhpd xmm0,xmm0
    8cf9:	addsd  xmm1,xmm0
    8cfd:	addsd  xmm2,xmm3
    8d01:	cmp    rdx,0x40
    8d05:	jne    8cca <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x6ca>
    8d07:	divsd  xmm1,xmm2
    8d0b:	movsd  xmm0,QWORD PTR [r14]
    8d10:	mov    r8,rbp
    8d13:	mov    rsi,rdi
    8d16:	mov    rdx,rbp
    8d19:	addsd  xmm0,xmm1
    8d1d:	unpcklpd xmm1,xmm1
    8d21:	movsd  QWORD PTR [r14],xmm0
    8d26:	movupd xmm3,XMMWORD PTR [rsi]
    8d2a:	movapd xmm0,XMMWORD PTR [rdx]
    8d2e:	add    rsi,0x10
    8d32:	add    rdx,0x10
    8d36:	mulpd  xmm3,xmm1
    8d3a:	subpd  xmm0,xmm3
    8d3e:	movaps XMMWORD PTR [rdx-0x10],xmm0
    8d42:	cmp    rsi,rbx
    8d45:	jne    8d26 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x726>
    8d47:	add    rbx,0x40
    8d4b:	add    rdi,0x40
    8d4f:	add    r14,0x8
    8d53:	cmp    r12,rbx
    8d56:	jne    8cc0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x6c0>
    8d5c:	add    r9,0x1
    8d60:	cmp    r9,QWORD PTR [rsp+0x28]
    8d65:	jne    8ca6 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x6a6>
    8d6b:	mov    rsi,r10
    8d6e:	mov    rdi,rbp
    8d71:	mov    QWORD PTR [rsp+0x60],rcx
    8d76:	mov    r14,QWORD PTR [rsp+0x50]
    8d7b:	mov    QWORD PTR [rsp+0x58],rax
    8d80:	mov    QWORD PTR [rsp+0x50],r10
    8d85:	mov    QWORD PTR [rsp+0x68],r8
    8d8a:	call   33820 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    8d8f:	mov    r10,QWORD PTR [rsp+0x50]
    8d94:	mov    rax,QWORD PTR [rsp+0x58]
    8d99:	pxor   xmm5,xmm5
    8d9d:	mov    rcx,QWORD PTR [rsp+0x60]
    8da2:	movapd xmm7,xmm0
    8da6:	movsd  QWORD PTR [r13+0x0],xmm0
    8dac:	movsd  xmm8,QWORD PTR [rip+0x5aa5b]        # 63810 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x90>
    8db5:	mulsd  xmm8,QWORD PTR [rsp+0x48]
    8dbc:	comisd xmm8,xmm0
    8dc1:	jae    8dd3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x7d3>
    8dc3:	cmp    rax,QWORD PTR [rsp+0x10]
    8dc8:	mov    r8,QWORD PTR [rsp+0x68]
    8dcd:	jb     97c0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x11c0>
    8dd3:	movq   xmm6,QWORD PTR [rip+0x5a9b5]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    8ddb:	test   r14,r14
    8dde:	je     8e58 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x858>
    8de0:	movsd  xmm0,QWORD PTR [r15+0x350]
    8de9:	mov    rdx,r15
    8dec:	lea    rsi,[r13-0x358]
    8df3:	movq   xmm6,QWORD PTR [rip+0x5a995]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    8dfb:	nop    DWORD PTR [rax+rax*1+0x0]
    8e00:	movsd  xmm1,QWORD PTR [rdx+0x358]
    8e08:	movsd  xmm2,QWORD PTR [rdx+0x420]
    8e10:	movapd xmm4,xmm0
    8e14:	add    rdx,0x8
    8e18:	movsd  xmm3,QWORD PTR [rdx+0x3d8]
    8e20:	mulsd  xmm4,xmm2
    8e24:	movapd xmm9,xmm1
    8e29:	mulsd  xmm1,xmm2
    8e2d:	mulsd  xmm9,xmm3
    8e32:	mulsd  xmm3,xmm0
    8e36:	movapd xmm0,xmm1
    8e3a:	addsd  xmm4,xmm9
    8e3f:	subsd  xmm0,xmm3
    8e43:	movsd  QWORD PTR [rdx+0x348],xmm4
    8e4b:	movsd  QWORD PTR [rdx+0x350],xmm0
    8e53:	cmp    rdx,rsi
    8e56:	jne    8e00 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x800>
    8e58:	movsd  xmm9,QWORD PTR [r13-0x8]
    8e5e:	movsd  xmm11,QWORD PTR [r13+0x0]
    8e64:	movapd xmm1,xmm9
    8e69:	comisd xmm5,xmm11
    8e6e:	movapd xmm0,xmm9
    8e73:	cmpltsd xmm1,xmm5
    8e78:	xorpd  xmm0,xmm6
    8e7c:	andpd  xmm0,xmm1
    8e80:	andnpd xmm1,xmm9
    8e85:	orpd   xmm0,xmm1
    8e89:	ja     919b <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb9b>
    8e8f:	ucomisd xmm11,xmm5
    8e94:	jp     8e9c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x89c>
    8e96:	je     9300 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xd00>
    8e9c:	movapd xmm2,xmm11
    8ea1:	maxsd  xmm2,xmm0
    8ea5:	movapd xmm10,xmm9
    8eaa:	movapd xmm12,xmm11
    8eaf:	divsd  xmm10,xmm2
    8eb4:	divsd  xmm12,xmm2
    8eb9:	movapd xmm0,xmm10
    8ebe:	comisd xmm5,xmm10
    8ec3:	mulsd  xmm0,xmm10
    8ec8:	movapd xmm1,xmm12
    8ecd:	mulsd  xmm1,xmm12
    8ed2:	addsd  xmm0,xmm1
    8ed6:	ja     9234 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc34>
    8edc:	ucomisd xmm0,xmm5
    8ee0:	jp     8ee8 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x8e8>
    8ee2:	pxor   xmm1,xmm1
    8ee6:	je     8f21 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x921>
    8ee8:	movsd  xmm1,QWORD PTR [rip+0x5a970]        # 63860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
    8ef0:	mov    QWORD PTR [rsp+0x50],rax
    8ef5:	mov    QWORD PTR [rsp+0x48],rcx
    8efa:	movapd xmm13,xmm1
    8eff:	maxsd  xmm13,xmm0
    8f04:	movapd xmm0,xmm13
    8f09:	call   6ad0 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
    8f0e:	mov    rax,QWORD PTR [rsp+0x50]
    8f13:	mov    rcx,QWORD PTR [rsp+0x48]
    8f18:	movapd xmm1,xmm0
    8f1c:	mulsd  xmm1,xmm13
    8f21:	ucomisd xmm10,xmm5
    8f26:	jp     9870 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1270>
    8f2c:	jne    9870 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1270>
    8f32:	divsd  xmm12,xmm1
    8f37:	pxor   xmm0,xmm0
    8f3b:	movapd xmm10,xmm12
    8f40:	mulsd  xmm9,xmm0
    8f45:	mov    rdx,r14
    8f48:	movsd  QWORD PTR [r13+0xc8],xmm0
    8f51:	mulsd  xmm11,xmm10
    8f56:	imul   rdx,rax
    8f5a:	movsd  QWORD PTR [r13+0x88],xmm10
    8f63:	shr    rdx,1
    8f66:	addsd  xmm9,xmm11
    8f6b:	movsd  QWORD PTR [r13-0x8],xmm9
    8f71:	nop    DWORD PTR [rax+0x0]
    8f75:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8f80:	movsd  xmm0,QWORD PTR [rcx]
    8f84:	movsd  QWORD PTR [rcx+rdx*8-0x120],xmm0
    8f8d:	add    rcx,0x8
    8f91:	cmp    r13,rcx
    8f94:	jne    8f80 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x980>
    8f96:	movsd  xmm0,QWORD PTR [r13+0x88]
    8f9f:	movsd  xmm1,QWORD PTR [r13+0x40]
    8fa5:	xorpd  xmm0,xmm6
    8fa9:	mulsd  xmm0,xmm1
    8fad:	mulsd  xmm1,QWORD PTR [r13+0xc8]
    8fb6:	movsd  QWORD PTR [r13+0x48],xmm0
    8fbc:	movsd  QWORD PTR [r13+0x40],xmm1
    8fc2:	add    QWORD PTR [r15+0x4a8],0x1
    8fca:	comisd xmm8,xmm7
    8fcf:	jb     9168 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb68>
    8fd5:	cmp    rax,0x8
    8fd9:	je     98d8 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x12d8>
    8fdf:	cmp    r14,0x6
    8fe3:	je     9899 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1299>
    8fe9:	add    r13,0x10
    8fed:	sub    r12,0xffffffffffffff80
    8ff1:	add    r14,0x3
    8ff5:	xor    esi,esi
    8ff7:	movzx  ecx,BYTE PTR [rsp+0x20]
    8ffc:	mov    rdi,rax
    8fff:	add    r13,0x8
    9003:	add    r12,0x40
    9007:	cmp    r14,0x8
    900b:	jne    8ad5 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x4d5>
    9011:	mov    r12d,ecx
    9014:	movsd  xmm2,QWORD PTR [rsp+0x78]
    901a:	movzx  edx,BYTE PTR [rsp+0x96]
    9022:	mov    rbp,QWORD PTR [rsp+0x80]
    902a:	mov    rbx,QWORD PTR [rsp+0x88]
    9032:	movzx  r14d,BYTE PTR [rsp+0x97]
    903b:	mov    rcx,QWORD PTR [rsp+0x98]
    9043:	movsd  xmm5,QWORD PTR [rsp+0x70]
    9049:	mov    eax,edx
    904b:	mov    r8d,0x7
    9051:	pxor   xmm3,xmm3
    9055:	jmp    906a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xa6a>
    9057:	nop    WORD PTR [rax+rax*1+0x0]
    9060:	sub    r8,0x1
    9064:	jb     9154 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb54>
    906a:	mov    QWORD PTR [r15+r8*8+0x460],0x0
    9076:	cmp    r8,rdi
    9079:	setb   r9b
    907d:	and    r9b,al
    9080:	je     9060 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xa60>
    9082:	movsd  xmm1,QWORD PTR [r15+r8*8+0x398]
    908c:	lea    r11,[r8+0x1]
    9090:	cmp    r8,0x7
    9094:	je     90f4 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xaf4>
    9096:	mov    rsi,r11
    9099:	nop    WORD PTR [rax+rax*1+0x0]
    909f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    90aa:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    90b5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    90c0:	mov    rax,rsi
    90c3:	add    rsi,0x1
    90c7:	cmp    rax,rdi
    90ca:	jae    90ee <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xaee>
    90cc:	imul   rax,rsi
    90d0:	shr    rax,1
    90d3:	add    rax,r8
    90d6:	movsd  xmm0,QWORD PTR [r15+rax*8+0x230]
    90e0:	mulsd  xmm0,QWORD PTR [r15+rsi*8+0x458]
    90ea:	subsd  xmm1,xmm0
    90ee:	cmp    rsi,0x8
    90f2:	jne    90c0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xac0>
    90f4:	imul   r11,r8
    90f8:	shr    r11,1
    90fb:	lea    rax,[r11+r8*1]
    90ff:	movsd  xmm0,QWORD PTR [r15+rax*8+0x230]
    9109:	comisd xmm3,xmm0
    910d:	ja     91f4 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xbf4>
    9113:	comisd xmm5,xmm0
    9117:	jbe    921e <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc1e>
    911d:	lea    rax,[r8-0x1]
    9121:	test   r8,r8
    9124:	je     915c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb5c>
    9126:	mov    QWORD PTR [r15+rax*8+0x460],0x0
    9132:	sub    r8,0x2
    9136:	cmp    r8,0xffffffffffffffff
    913a:	je     915c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb5c>
    913c:	mov    QWORD PTR [r15+r8*8+0x460],0x0
    9148:	xor    eax,eax
    914a:	sub    r8,0x1
    914e:	jae    906a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xa6a>
    9154:	test   al,al
    9156:	jne    938b <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xd8b>
    915c:	mov    BYTE PTR [r15+0x4b1],dl
    9163:	jmp    88b2 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x2b2>
    9168:	movapd xmm1,xmm0
    916c:	xorpd  xmm6,xmm0
    9170:	mov    r14,rax
    9173:	mov    r12,rbx
    9176:	cmpltsd xmm1,xmm5
    917b:	mov    rdi,rax
    917e:	andpd  xmm6,xmm1
    9182:	andnpd xmm1,xmm0
    9186:	orpd   xmm1,xmm6
    918a:	comisd xmm1,QWORD PTR [rsp+0x38]
    9190:	seta   sil
    9194:	xor    ecx,ecx
    9196:	jmp    8ac3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x4c3>
    919b:	movapd xmm2,xmm11
    91a0:	xorpd  xmm2,xmm6
    91a4:	jmp    8ea1 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x8a1>
    91a9:	cmp    r10,0x1
    91ad:	je     8c05 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x605>
    91b3:	cmp    r10,0x2
    91b7:	je     8c0e <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x60e>
    91bd:	cmp    r10,0x3
    91c1:	je     8c1a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x61a>
    91c7:	cmp    r10,0x4
    91cb:	je     8c23 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x623>
    91d1:	cmp    r10,0x5
    91d5:	je     8c2f <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x62f>
    91db:	cmp    r10,0x6
    91df:	je     8c38 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x638>
    91e5:	cmp    r10,0x7
    91e9:	jne    8c4a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x64a>
    91ef:	jmp    8c41 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x641>
    91f4:	movapd xmm4,xmm0
    91f8:	xorpd  xmm4,XMMWORD PTR [rip+0x5a590]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    9200:	comisd xmm5,xmm4
    9204:	ja     911d <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xb1d>
    920a:	movapd xmm4,xmm5
    920e:	xorpd  xmm4,XMMWORD PTR [rip+0x5a57a]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    9216:	minsd  xmm4,xmm0
    921a:	movapd xmm0,xmm4
    921e:	divsd  xmm1,xmm0
    9222:	mov    eax,r9d
    9225:	movsd  QWORD PTR [r15+r8*8+0x460],xmm1
    922f:	jmp    9060 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xa60>
    9234:	ucomisd xmm0,xmm5
    9238:	movapd xmm14,xmm10
    923d:	xorpd  xmm14,xmm6
    9242:	jp     934d <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xd4d>
    9248:	pxor   xmm1,xmm1
    924c:	jne    934d <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xd4d>
    9252:	divsd  xmm10,xmm14
    9257:	movapd xmm0,xmm14
    925c:	divsd  xmm12,xmm1
    9261:	divsd  xmm0,xmm1
    9265:	mulsd  xmm10,xmm12
    926a:	jmp    8f40 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x940>
    926f:	mov    rax,QWORD PTR [rsp+0x8]
    9274:	lea    rdi,[r15+0x30]
    9278:	divsd  xmm0,xmm2
    927c:	add    rax,0x10
    9280:	movsd  QWORD PTR [rdi],xmm0
    9284:	movsd  xmm0,QWORD PTR [rax-0x8]
    9289:	divsd  xmm0,xmm2
    928d:	movsd  QWORD PTR [rdi+0x8],xmm0
    9292:	cmp    rsi,rax
    9295:	je     8a18 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x418>
    929b:	movsd  xmm0,QWORD PTR [rax]
    929f:	add    rdi,0x10
    92a3:	jmp    9278 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc78>
    92a5:	nop    DWORD PTR [rax]
    92a8:	mov    rcx,QWORD PTR [rsp+0x30]
    92ad:	pxor   xmm0,xmm0
    92b1:	lea    rbp,[rsp+0x120]
    92b9:	movaps XMMWORD PTR [rsp+0x120],xmm0
    92c1:	mov    rdi,QWORD PTR [rcx]
    92c4:	mov    rcx,QWORD PTR [rcx+0x8]
    92c8:	movaps XMMWORD PTR [rsp+0x130],xmm0
    92d0:	movaps XMMWORD PTR [rsp+0x140],xmm0
    92d8:	movaps XMMWORD PTR [rsp+0x150],xmm0
    92e0:	cmp    rcx,0x8
    92e4:	je     99e4 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x13e4>
    92ea:	test   rcx,rcx
    92ed:	je     8c4a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x64a>
    92f3:	mov    rdx,rbp
    92f6:	call   4240 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    92fb:	jmp    8c4a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x64a>
    9300:	movsd  xmm0,QWORD PTR [rip+0x5a520]        # 63828 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
    9308:	pxor   xmm10,xmm10
    930d:	jmp    8f40 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x940>
    9312:	mov    rax,QWORD PTR [rsp+0x30]
    9317:	lea    rdx,[rsp+0xa0]
    931f:	mov    QWORD PTR [rsp+0x8],rdx
    9324:	mov    rcx,QWORD PTR [rax+0x8]
    9328:	mov    rdi,QWORD PTR [rax]
    932b:	cmp    rcx,0x8
    932f:	je     9a03 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1403>
    9335:	test   rcx,rcx
    9338:	je     87d7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1d7>
    933e:	mov    rsi,QWORD PTR [rsp+0x18]
    9343:	call   4240 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    9348:	jmp    87d7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1d7>
    934d:	movsd  xmm1,QWORD PTR [rip+0x5a50b]        # 63860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
    9355:	mov    QWORD PTR [rsp+0x50],rax
    935a:	mov    QWORD PTR [rsp+0x48],rcx
    935f:	movapd xmm13,xmm1
    9364:	maxsd  xmm13,xmm0
    9369:	movapd xmm0,xmm13
    936e:	call   6ad0 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
    9373:	movapd xmm1,xmm13
    9378:	mov    rcx,QWORD PTR [rsp+0x48]
    937d:	mov    rax,QWORD PTR [rsp+0x50]
    9382:	mulsd  xmm1,xmm0
    9386:	jmp    9252 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc52>
    938b:	mov    rax,QWORD PTR [rsp+0x18]
    9390:	lea    rsi,[rsp+0x160]
    9398:	xor    r9d,r9d
    939b:	mov    r8,rsi
    939e:	movdqu xmm0,XMMWORD PTR [rax]
    93a2:	movaps XMMWORD PTR [rsp+0x160],xmm0
    93aa:	movdqu xmm0,XMMWORD PTR [rax+0x10]
    93af:	movaps XMMWORD PTR [rsp+0x170],xmm0
    93b7:	movdqu xmm0,XMMWORD PTR [rax+0x20]
    93bc:	movaps XMMWORD PTR [rsp+0x180],xmm0
    93c4:	movdqu xmm0,XMMWORD PTR [rax+0x30]
    93c9:	movaps XMMWORD PTR [rsp+0x190],xmm0
    93d1:	mov    rdx,rcx
    93d4:	pxor   xmm1,xmm1
    93d8:	xor    eax,eax
    93da:	nop    DWORD PTR [rax+rax*1+0x0]
    93df:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    93ea:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    93f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    9400:	cmp    rax,rdi
    9403:	jae    9418 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xe18>
    9405:	movsd  xmm0,QWORD PTR [rdx+0x30]
    940a:	mulsd  xmm0,QWORD PTR [r15+rax*8+0x460]
    9414:	addsd  xmm1,xmm0
    9418:	add    rax,0x1
    941c:	add    rdx,0x40
    9420:	cmp    rax,0x8
    9424:	jne    9400 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xe00>
    9426:	addsd  xmm1,QWORD PTR [r8]
    942b:	add    r9,0x1
    942f:	add    r8,0x8
    9433:	add    rcx,0x8
    9437:	movsd  QWORD PTR [r8-0x8],xmm1
    943d:	cmp    r9,0x8
    9441:	jne    93d1 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xdd1>
    9443:	cmp    r10,0x8
    9447:	je     9a54 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1454>
    944d:	pxor   xmm0,xmm0
    9451:	movaps XMMWORD PTR [rsp+0x1a0],xmm0
    9459:	movaps XMMWORD PTR [rsp+0x1b0],xmm0
    9461:	movaps XMMWORD PTR [rsp+0x1c0],xmm0
    9469:	movaps XMMWORD PTR [rsp+0x1d0],xmm0
    9471:	test   r10,r10
    9474:	je     953c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf3c>
    947a:	movsd  xmm0,QWORD PTR [rsp+0x160]
    9483:	movsd  QWORD PTR [rsp+0x1a0],xmm0
    948c:	cmp    r10,0x1
    9490:	je     953c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf3c>
    9496:	movsd  xmm0,QWORD PTR [rsp+0x168]
    949f:	movsd  QWORD PTR [rsp+0x1a8],xmm0
    94a8:	cmp    r10,0x2
    94ac:	je     953c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf3c>
    94b2:	movsd  xmm0,QWORD PTR [rsp+0x170]
    94bb:	movsd  QWORD PTR [rsp+0x1b0],xmm0
    94c4:	cmp    r10,0x3
    94c8:	je     953c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf3c>
    94ca:	movsd  xmm0,QWORD PTR [rsp+0x178]
    94d3:	movsd  QWORD PTR [rsp+0x1b8],xmm0
    94dc:	cmp    r10,0x4
    94e0:	je     953c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf3c>
    94e2:	movsd  xmm0,QWORD PTR [rsp+0x180]
    94eb:	movsd  QWORD PTR [rsp+0x1c0],xmm0
    94f4:	cmp    r10,0x5
    94f8:	je     953c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf3c>
    94fa:	movsd  xmm0,QWORD PTR [rsp+0x188]
    9503:	movsd  QWORD PTR [rsp+0x1c8],xmm0
    950c:	cmp    r10,0x6
    9510:	je     953c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf3c>
    9512:	movsd  xmm0,QWORD PTR [rsp+0x190]
    951b:	movsd  QWORD PTR [rsp+0x1d0],xmm0
    9524:	cmp    r10,0x7
    9528:	je     953c <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf3c>
    952a:	movsd  xmm0,QWORD PTR [rsp+0x198]
    9533:	movsd  QWORD PTR [rsp+0x1d8],xmm0
    953c:	mov    rax,QWORD PTR [rsp+0x30]
    9541:	pxor   xmm0,xmm0
    9545:	movaps XMMWORD PTR [rsp+0xa0],xmm0
    954d:	mov    rcx,QWORD PTR [rax+0x8]
    9551:	mov    rdi,QWORD PTR [rax]
    9554:	movaps XMMWORD PTR [rsp+0xb0],xmm0
    955c:	movaps XMMWORD PTR [rsp+0xc0],xmm0
    9564:	movaps XMMWORD PTR [rsp+0xd0],xmm0
    956c:	cmp    rcx,0x8
    9570:	je     9b32 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1532>
    9576:	test   rcx,rcx
    9579:	jne    99c1 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x13c1>
    957f:	test   r10,r10
    9582:	jne    9917 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1317>
    9588:	mov    QWORD PTR [rsp+0xa0],0x0
    9594:	mov    QWORD PTR [rsp+0xa8],0x0
    95a0:	mov    QWORD PTR [rsp+0xb0],0x0
    95ac:	mov    QWORD PTR [rsp+0xb8],0x0
    95b8:	mov    QWORD PTR [rsp+0xc0],0x0
    95c4:	mov    QWORD PTR [rsp+0xc8],0x0
    95d0:	mov    QWORD PTR [rsp+0xd0],0x0
    95dc:	mov    QWORD PTR [rsp+0xd8],0x0
    95e8:	mov    rdx,QWORD PTR [rsp+0x8]
    95ed:	mov    rcx,QWORD PTR [rsp+0x40]
    95f2:	xor    eax,eax
    95f4:	pxor   xmm0,xmm0
    95f8:	cmp    rax,r10
    95fb:	jae    9606 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1006>
    95fd:	movsd  xmm0,QWORD PTR [rcx+rax*8]
    9602:	subsd  xmm0,QWORD PTR [rdx]
    9606:	add    rax,0x1
    960a:	movsd  QWORD PTR [rdx],xmm0
    960e:	add    rdx,0x8
    9612:	cmp    rax,0x8
    9616:	jne    95f4 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xff4>
    9618:	mov    rdi,QWORD PTR [rsp+0x8]
    961d:	mov    rsi,r10
    9620:	mov    QWORD PTR [rsp+0x20],r10
    9625:	movsd  QWORD PTR [rsp+0x48],xmm2
    962b:	call   33820 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    9630:	movsd  xmm2,QWORD PTR [rsp+0x48]
    9636:	mov    r10,QWORD PTR [rsp+0x20]
    963b:	comisd xmm2,xmm0
    963f:	jb     9b93 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1593>
    9645:	movdqa xmm1,XMMWORD PTR [rsp+0x160]
    964e:	mov    rax,QWORD PTR [rsp+0x18]
    9653:	movups XMMWORD PTR [rax],xmm1
    9656:	movdqa xmm1,XMMWORD PTR [rsp+0x170]
    965f:	movups XMMWORD PTR [rax+0x10],xmm1
    9663:	movdqa xmm1,XMMWORD PTR [rsp+0x180]
    966c:	movups XMMWORD PTR [rax+0x20],xmm1
    9670:	movdqa xmm1,XMMWORD PTR [rsp+0x190]
    9679:	movups XMMWORD PTR [rax+0x30],xmm1
    967d:	movsd  QWORD PTR [r15+0x4a0],xmm0
    9686:	movsd  xmm7,QWORD PTR [rsp+0x38]
    968c:	mov    edx,r12d
    968f:	comisd xmm7,xmm0
    9693:	setb   al
    9696:	setae  BYTE PTR [r15+0x4b0]
    969e:	and    edx,eax
    96a0:	mov    BYTE PTR [r15+0x4b1],dl
    96a7:	jmp    88b2 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x2b2>
    96ac:	movsd  xmm0,QWORD PTR [rsp+0xf0]
    96b5:	movsd  QWORD PTR [rsp+0x1b0],xmm0
    96be:	cmp    r10,0x3
    96c2:	je     8b95 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x595>
    96c8:	movsd  xmm0,QWORD PTR [rsp+0xf8]
    96d1:	movsd  QWORD PTR [rsp+0x1b8],xmm0
    96da:	cmp    r10,0x4
    96de:	je     8b95 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x595>
    96e4:	movsd  xmm0,QWORD PTR [rsp+0x100]
    96ed:	movsd  QWORD PTR [rsp+0x1c0],xmm0
    96f6:	cmp    r10,0x5
    96fa:	je     8b95 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x595>
    9700:	movsd  xmm0,QWORD PTR [rsp+0x108]
    9709:	movsd  QWORD PTR [rsp+0x1c8],xmm0
    9712:	cmp    r10,0x6
    9716:	je     8b95 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x595>
    971c:	movsd  xmm0,QWORD PTR [rsp+0x110]
    9725:	movsd  QWORD PTR [rsp+0x1d0],xmm0
    972e:	cmp    r10,0x7
    9732:	je     8b95 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x595>
    9738:	movsd  xmm0,QWORD PTR [rsp+0x118]
    9741:	movsd  QWORD PTR [rsp+0x1d8],xmm0
    974a:	jmp    8b95 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x595>
    974f:	nop
    9750:	cmp    rcx,0x8
    9754:	je     9b55 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1555>
    975a:	test   rcx,rcx
    975d:	je     99f1 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x13f1>
    9763:	lea    rdx,[rsp+0xa0]
    976b:	lea    rsi,[rsp+0x1a0]
    9773:	mov    QWORD PTR [rsp+0x8],rdx
    9778:	call   4240 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    977d:	jmp    878f <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x18f>
    9782:	lea    rsi,[rsp+0x1a0]
    978a:	mov    rdx,rbp
    978d:	call   5b10 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    9792:	pxor   xmm5,xmm5
    9796:	jmp    8bf0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x5f0>
    979b:	cmp    rcx,0x8
    979f:	je     9ab2 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x14b2>
    97a5:	test   rcx,rcx
    97a8:	jne    987a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x127a>
    97ae:	lea    rax,[rsp+0xa0]
    97b6:	mov    QWORD PTR [rsp+0x8],rax
    97bb:	jmp    8783 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x183>
    97c0:	comisd xmm0,QWORD PTR [rip+0x5a230]        # 639f8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x278>
    97c8:	pxor   xmm1,xmm1
    97cc:	jb     97da <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x11da>
    97ce:	movsd  xmm1,QWORD PTR [rip+0x5a052]        # 63828 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
    97d6:	divsd  xmm1,xmm0
    97da:	movsd  xmm0,QWORD PTR [rbp+0x0]
    97df:	comisd xmm7,QWORD PTR [rip+0x5a211]        # 639f8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x278>
    97e7:	lea    rdx,[r12-0x40]
    97ec:	lea    rsi,[rbp+0x40]
    97f0:	jb     9840 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1240>
    97f2:	mulsd  xmm0,xmm1
    97f6:	add    r8,0x10
    97fa:	movsd  QWORD PTR [rdx],xmm0
    97fe:	movsd  xmm0,QWORD PTR [r8-0x8]
    9804:	mulsd  xmm0,xmm1
    9808:	movsd  QWORD PTR [rdx+0x8],xmm0
    980d:	cmp    rsi,r8
    9810:	je     8dd3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x7d3>
    9816:	movsd  xmm0,QWORD PTR [r8]
    981b:	add    rdx,0x10
    981f:	jmp    97f2 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x11f2>
    9821:	nop    WORD PTR [rax+rax*1+0x0]
    982a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    9835:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    9840:	divsd  xmm0,xmm7
    9844:	add    r8,0x10
    9848:	movsd  QWORD PTR [rdx],xmm0
    984c:	movsd  xmm0,QWORD PTR [r8-0x8]
    9852:	divsd  xmm0,xmm7
    9856:	movsd  QWORD PTR [rdx+0x8],xmm0
    985b:	cmp    rsi,r8
    985e:	je     8dd3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x7d3>
    9864:	movsd  xmm0,QWORD PTR [r8]
    9869:	add    rdx,0x10
    986d:	jmp    9840 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1240>
    986f:	nop
    9870:	movapd xmm14,xmm10
    9875:	jmp    9252 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xc52>
    987a:	lea    rdx,[rsp+0xa0]
    9882:	lea    rsi,[rsp+0x1a0]
    988a:	mov    QWORD PTR [rsp+0x8],rdx
    988f:	call   4240 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    9894:	jmp    8783 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x183>
    9899:	movsd  xmm2,QWORD PTR [rsp+0x78]
    989f:	movzx  edx,BYTE PTR [rsp+0x96]
    98a7:	mov    edi,0x7
    98ac:	mov    rbp,QWORD PTR [rsp+0x80]
    98b4:	mov    rbx,QWORD PTR [rsp+0x88]
    98bc:	movzx  r14d,BYTE PTR [rsp+0x97]
    98c5:	mov    rcx,QWORD PTR [rsp+0x98]
    98cd:	movzx  r12d,BYTE PTR [rsp+0x20]
    98d3:	jmp    9043 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xa43>
    98d8:	movsd  xmm2,QWORD PTR [rsp+0x78]
    98de:	movzx  edx,BYTE PTR [rsp+0x96]
    98e6:	mov    edi,0x8
    98eb:	mov    rbp,QWORD PTR [rsp+0x80]
    98f3:	mov    rbx,QWORD PTR [rsp+0x88]
    98fb:	movzx  r14d,BYTE PTR [rsp+0x97]
    9904:	mov    rcx,QWORD PTR [rsp+0x98]
    990c:	movzx  r12d,BYTE PTR [rsp+0x20]
    9912:	jmp    9043 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xa43>
    9917:	cmp    r10,0x1
    991b:	je     9594 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf94>
    9921:	cmp    r10,0x2
    9925:	je     95a0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xfa0>
    992b:	cmp    r10,0x3
    992f:	je     95ac <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xfac>
    9935:	cmp    r10,0x4
    9939:	je     95b8 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xfb8>
    993f:	cmp    r10,0x5
    9943:	je     95c4 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xfc4>
    9949:	cmp    r10,0x6
    994d:	je     95d0 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xfd0>
    9953:	cmp    r10,0x7
    9957:	jne    95e8 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xfe8>
    995d:	jmp    95dc <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xfdc>
    9962:	mov    rax,QWORD PTR [rsp+0x18]
    9967:	movsd  xmm0,QWORD PTR [rax+0x18]
    996c:	movsd  QWORD PTR [rsp+0x1b8],xmm0
    9975:	cmp    r12,0x4
    9979:	je     9a10 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1410>
    997f:	mov    rax,QWORD PTR [rsp+0x18]
    9984:	movsd  xmm0,QWORD PTR [rax+0x20]
    9989:	movsd  QWORD PTR [rsp+0x1c0],xmm0
    9992:	cmp    r12,0x5
    9996:	jne    9ad1 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x14d1>
    999c:	cmp    rcx,0x8
    99a0:	je     9c32 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1632>
    99a6:	test   rcx,rcx
    99a9:	jne    9c51 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1651>
    99af:	lea    rax,[rsp+0xa0]
    99b7:	mov    QWORD PTR [rsp+0x8],rax
    99bc:	jmp    87b3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1b3>
    99c1:	mov    rdx,QWORD PTR [rsp+0x8]
    99c6:	lea    rsi,[rsp+0x1a0]
    99ce:	movsd  QWORD PTR [rsp+0x20],xmm2
    99d4:	call   4240 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    99d9:	movsd  xmm2,QWORD PTR [rsp+0x20]
    99df:	jmp    957f <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf7f>
    99e4:	mov    rdx,rbp
    99e7:	call   5b10 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    99ec:	jmp    8c4a <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x64a>
    99f1:	lea    rax,[rsp+0xa0]
    99f9:	mov    QWORD PTR [rsp+0x8],rax
    99fe:	jmp    878f <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x18f>
    9a03:	mov    rsi,r14
    9a06:	call   5b10 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    9a0b:	jmp    87d7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1d7>
    9a10:	cmp    rcx,0x8
    9a14:	je     9bf3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x15f3>
    9a1a:	test   rcx,rcx
    9a1d:	jne    9b74 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1574>
    9a23:	lea    rax,[rsp+0xa0]
    9a2b:	mov    QWORD PTR [rsp+0x8],rax
    9a30:	jmp    87a7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1a7>
    9a35:	lea    rdx,[rsp+0xa0]
    9a3d:	lea    rsi,[rsp+0x1a0]
    9a45:	mov    QWORD PTR [rsp+0x8],rdx
    9a4a:	call   5b10 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    9a4f:	jmp    879b <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x19b>
    9a54:	mov    rax,QWORD PTR [rsp+0x30]
    9a59:	pxor   xmm0,xmm0
    9a5d:	movaps XMMWORD PTR [rsp+0xa0],xmm0
    9a65:	mov    rcx,QWORD PTR [rax+0x8]
    9a69:	mov    rdi,QWORD PTR [rax]
    9a6c:	movaps XMMWORD PTR [rsp+0xb0],xmm0
    9a74:	movaps XMMWORD PTR [rsp+0xc0],xmm0
    9a7c:	movaps XMMWORD PTR [rsp+0xd0],xmm0
    9a84:	cmp    rcx,0x8
    9a88:	je     9c12 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1612>
    9a8e:	test   rcx,rcx
    9a91:	je     95e8 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xfe8>
    9a97:	mov    rdx,QWORD PTR [rsp+0x8]
    9a9c:	movsd  QWORD PTR [rsp+0x20],xmm2
    9aa2:	call   4240 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    9aa7:	movsd  xmm2,QWORD PTR [rsp+0x20]
    9aad:	jmp    95e8 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xfe8>
    9ab2:	lea    rdx,[rsp+0xa0]
    9aba:	lea    rsi,[rsp+0x1a0]
    9ac2:	mov    QWORD PTR [rsp+0x8],rdx
    9ac7:	call   5b10 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    9acc:	jmp    8783 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x183>
    9ad1:	mov    rax,QWORD PTR [rsp+0x18]
    9ad6:	movsd  xmm0,QWORD PTR [rax+0x28]
    9adb:	movsd  QWORD PTR [rsp+0x1c8],xmm0
    9ae4:	cmp    r12,0x6
    9ae8:	jne    9cae <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x16ae>
    9aee:	cmp    rcx,0x8
    9af2:	je     9c8f <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x168f>
    9af8:	test   rcx,rcx
    9afb:	jne    9c70 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1670>
    9b01:	lea    rax,[rsp+0xa0]
    9b09:	mov    QWORD PTR [rsp+0x8],rax
    9b0e:	jmp    87bf <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1bf>
    9b13:	lea    rdx,[rsp+0xa0]
    9b1b:	lea    rsi,[rsp+0x1a0]
    9b23:	mov    QWORD PTR [rsp+0x8],rdx
    9b28:	call   4240 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    9b2d:	jmp    879b <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x19b>
    9b32:	mov    rdx,QWORD PTR [rsp+0x8]
    9b37:	lea    rsi,[rsp+0x1a0]
    9b3f:	movsd  QWORD PTR [rsp+0x20],xmm2
    9b45:	call   5b10 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    9b4a:	movsd  xmm2,QWORD PTR [rsp+0x20]
    9b50:	jmp    957f <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xf7f>
    9b55:	lea    rdx,[rsp+0xa0]
    9b5d:	lea    rsi,[rsp+0x1a0]
    9b65:	mov    QWORD PTR [rsp+0x8],rdx
    9b6a:	call   5b10 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    9b6f:	jmp    878f <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x18f>
    9b74:	lea    rdx,[rsp+0xa0]
    9b7c:	lea    rsi,[rsp+0x1a0]
    9b84:	mov    QWORD PTR [rsp+0x8],rdx
    9b89:	call   4240 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    9b8e:	jmp    87a7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1a7>
    9b93:	mov    r13,QWORD PTR [rsp+0x8]
    9b98:	mov    rsi,QWORD PTR [rsp+0x18]
    9b9d:	mov    rcx,r10
    9ba0:	mov    QWORD PTR [rsp+0x20],r10
    9ba5:	mov    rdi,QWORD PTR [rsp+0x30]
    9baa:	mov    rdx,r13
    9bad:	call   4d230 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m>
    9bb2:	mov    rcx,QWORD PTR [rsp+0x40]
    9bb7:	mov    rdx,r13
    9bba:	xor    eax,eax
    9bbc:	mov    r10,QWORD PTR [rsp+0x20]
    9bc1:	pxor   xmm0,xmm0
    9bc5:	cmp    rax,r10
    9bc8:	jae    9bd3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x15d3>
    9bca:	movsd  xmm0,QWORD PTR [rcx+rax*8]
    9bcf:	subsd  xmm0,QWORD PTR [rdx]
    9bd3:	add    rax,0x1
    9bd7:	movsd  QWORD PTR [rdx],xmm0
    9bdb:	add    rdx,0x8
    9bdf:	cmp    rax,0x8
    9be3:	jne    9bc1 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x15c1>
    9be5:	movsd  xmm0,QWORD PTR [r15+0x4a0]
    9bee:	jmp    9686 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1086>
    9bf3:	lea    rdx,[rsp+0xa0]
    9bfb:	lea    rsi,[rsp+0x1a0]
    9c03:	mov    QWORD PTR [rsp+0x8],rdx
    9c08:	call   5b10 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    9c0d:	jmp    87a7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1a7>
    9c12:	mov    rdx,QWORD PTR [rsp+0x8]
    9c17:	movsd  QWORD PTR [rsp+0x20],xmm2
    9c1d:	call   5b10 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    9c22:	movsd  xmm2,QWORD PTR [rsp+0x20]
    9c28:	jmp    95e8 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0xfe8>
    9c2d:	call   1170 <__stack_chk_fail@plt>
    9c32:	lea    rdx,[rsp+0xa0]
    9c3a:	lea    rsi,[rsp+0x1a0]
    9c42:	mov    QWORD PTR [rsp+0x8],rdx
    9c47:	call   5b10 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    9c4c:	jmp    87b3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1b3>
    9c51:	lea    rdx,[rsp+0xa0]
    9c59:	lea    rsi,[rsp+0x1a0]
    9c61:	mov    QWORD PTR [rsp+0x8],rdx
    9c66:	call   4240 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    9c6b:	jmp    87b3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1b3>
    9c70:	lea    rdx,[rsp+0xa0]
    9c78:	lea    rsi,[rsp+0x1a0]
    9c80:	mov    QWORD PTR [rsp+0x8],rdx
    9c85:	call   4240 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    9c8a:	jmp    87bf <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1bf>
    9c8f:	lea    rdx,[rsp+0xa0]
    9c97:	lea    rsi,[rsp+0x1a0]
    9c9f:	mov    QWORD PTR [rsp+0x8],rdx
    9ca4:	call   5b10 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    9ca9:	jmp    87bf <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1bf>
    9cae:	mov    rax,QWORD PTR [rsp+0x18]
    9cb3:	movsd  xmm0,QWORD PTR [rax+0x30]
    9cb8:	movsd  QWORD PTR [rsp+0x1d0],xmm0
    9cc1:	cmp    r12,0x7
    9cc5:	jne    9ce4 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x16e4>
    9cc7:	cmp    rcx,0x8
    9ccb:	je     9d37 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1737>
    9ccd:	test   rcx,rcx
    9cd0:	jne    9d18 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1718>
    9cd2:	lea    rax,[rsp+0xa0]
    9cda:	mov    QWORD PTR [rsp+0x8],rax
    9cdf:	jmp    87cb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1cb>
    9ce4:	mov    rax,QWORD PTR [rsp+0x18]
    9ce9:	movsd  xmm0,QWORD PTR [rax+0x38]
    9cee:	movsd  QWORD PTR [rsp+0x1d8],xmm0
    9cf7:	cmp    rcx,0x8
    9cfb:	je     9db1 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x17b1>
    9d01:	test   rcx,rcx
    9d04:	jne    9d56 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1756>
    9d06:	lea    rax,[rsp+0xa0]
    9d0e:	mov    QWORD PTR [rsp+0x8],rax
    9d13:	jmp    87d7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1d7>
    9d18:	lea    rdx,[rsp+0xa0]
    9d20:	lea    rsi,[rsp+0x1a0]
    9d28:	mov    QWORD PTR [rsp+0x8],rdx
    9d2d:	call   4240 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    9d32:	jmp    87cb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1cb>
    9d37:	lea    rdx,[rsp+0xa0]
    9d3f:	lea    rsi,[rsp+0x1a0]
    9d47:	mov    QWORD PTR [rsp+0x8],rdx
    9d4c:	call   5b10 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    9d51:	jmp    87cb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1cb>
    9d56:	lea    rdx,[rsp+0xa0]
    9d5e:	lea    rsi,[rsp+0x1a0]
    9d66:	mov    QWORD PTR [rsp+0x8],rdx
    9d6b:	call   4240 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
    9d70:	cmp    r12,0x2
    9d74:	je     878f <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x18f>
    9d7a:	cmp    r12,0x3
    9d7e:	je     879b <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x19b>
    9d84:	cmp    r12,0x4
    9d88:	je     87a7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1a7>
    9d8e:	cmp    r12,0x5
    9d92:	je     87b3 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1b3>
    9d98:	cmp    r12,0x6
    9d9c:	je     87bf <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1bf>
    9da2:	cmp    r12,0x7
    9da6:	jne    87d7 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1d7>
    9dac:	jmp    87cb <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x1cb>
    9db1:	lea    rdx,[rsp+0xa0]
    9db9:	lea    rsi,[rsp+0x1a0]
    9dc1:	mov    QWORD PTR [rsp+0x8],rdx
    9dc6:	call   5b10 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
    9dcb:	jmp    9d8e <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0+0x178e>

Disassembly of section .fini:
