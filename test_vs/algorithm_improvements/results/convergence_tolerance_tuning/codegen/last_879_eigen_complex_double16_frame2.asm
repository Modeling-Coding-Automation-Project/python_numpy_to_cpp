
/tmp/matrix_peak_stack_audit/last_879/last_879_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

00000000000099c0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0>:
    99c0:	push   rbx
    99c1:	mov    r9,rsi
    99c4:	sub    rsp,0x110
    99cb:	mov    r10,QWORD PTR fs:0x28
    99d4:	mov    QWORD PTR [rsp+0x108],r10
    99dc:	mov    r10,rcx
    99df:	cmp    r8,0x10
    99e3:	je     9baa <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x1ea>
    99e9:	mov    r11,rsp
    99ec:	lea    rsi,[rsp+0x100]
    99f4:	pxor   xmm0,xmm0
    99f8:	mov    rax,rsp
    99fb:	nop    DWORD PTR [rax+rax*1+0x0]
    9a00:	movaps XMMWORD PTR [rax],xmm0
    9a03:	add    rax,0x20
    9a07:	movaps XMMWORD PTR [rax-0x10],xmm0
    9a0b:	cmp    rax,rsi
    9a0e:	jne    9a00 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x40>
    9a10:	mov    rcx,r11
    9a13:	xor    eax,eax
    9a15:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    9a20:	cmp    rax,r8
    9a23:	jae    9a2c <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x6c>
    9a25:	movupd xmm0,XMMWORD PTR [rdx]
    9a29:	movaps XMMWORD PTR [rcx],xmm0
    9a2c:	add    rax,0x1
    9a30:	add    rdx,0x10
    9a34:	add    rcx,0x10
    9a38:	cmp    rax,0x10
    9a3c:	jne    9a20 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x60>
    9a3e:	lea    rdx,[r10+0x100]
    9a45:	mov    rax,r10
    9a48:	pxor   xmm0,xmm0
    9a4c:	nop    DWORD PTR [rax+0x0]
    9a50:	movups XMMWORD PTR [rax],xmm0
    9a53:	add    rax,0x20
    9a57:	movups XMMWORD PTR [rax-0x10],xmm0
    9a5b:	cmp    rax,rdx
    9a5e:	jne    9a50 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x90>
    9a60:	cmp    r9,0x10
    9a64:	je     9b3b <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x17b>
    9a6a:	mov    rbx,rdi
    9a6d:	xor    esi,esi
    9a6f:	nop
    9a70:	mov    rdx,r10
    9a73:	mov    rcx,rbx
    9a76:	xor    eax,eax
    9a78:	nop    DWORD PTR [rax+rax*1+0x0]
    9a80:	cmp    rax,rsi
    9a83:	mov    rdi,rsi
    9a86:	cmovae rdi,rax
    9a8a:	cmp    rdi,r9
    9a8d:	jae    9acc <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x10c>
    9a8f:	movupd xmm2,XMMWORD PTR [rcx]
    9a93:	movapd xmm0,XMMWORD PTR [r11]
    9a98:	movupd xmm1,XMMWORD PTR [rcx]
    9a9c:	movupd xmm6,XMMWORD PTR [rdx]
    9aa0:	unpcklpd xmm2,xmm2
    9aa4:	mulpd  xmm2,xmm0
    9aa8:	unpckhpd xmm1,xmm1
    9aac:	shufpd xmm0,xmm0,0x1
    9ab1:	mulpd  xmm0,xmm1
    9ab5:	movapd xmm1,xmm2
    9ab9:	subpd  xmm1,xmm0
    9abd:	addpd  xmm0,xmm2
    9ac1:	movsd  xmm0,xmm1
    9ac5:	addpd  xmm0,xmm6
    9ac9:	movups XMMWORD PTR [rdx],xmm0
    9acc:	add    rax,0x1
    9ad0:	add    rcx,0x10
    9ad4:	add    rdx,0x10
    9ad8:	cmp    rax,0x10
    9adc:	jne    9a80 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0xc0>
    9ade:	add    rsi,0x1
    9ae2:	add    rbx,0x100
    9ae9:	add    r11,0x10
    9aed:	cmp    rsi,0x10
    9af1:	jne    9a70 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0xb0>
    9af7:	xor    eax,eax
    9af9:	pxor   xmm0,xmm0
    9afd:	nop    DWORD PTR [rax]
    9b00:	cmp    rax,r8
    9b03:	jb     9b11 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x151>
    9b05:	mov    rdx,rax
    9b08:	shl    rdx,0x4
    9b0c:	movups XMMWORD PTR [r10+rdx*1],xmm0
    9b11:	add    rax,0x1
    9b15:	cmp    rax,0x10
    9b19:	jne    9b00 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x140>
    9b1b:	mov    rax,QWORD PTR [rsp+0x108]
    9b23:	sub    rax,QWORD PTR fs:0x28
    9b2c:	jne    9cf6 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x336>
    9b32:	add    rsp,0x110
    9b39:	pop    rbx
    9b3a:	ret
    9b3b:	mov    rdx,rdi
    9b3e:	xchg   ax,ax
    9b40:	movapd xmm3,XMMWORD PTR [r11]
    9b45:	xor    eax,eax
    9b47:	movapd xmm4,xmm3
    9b4b:	shufpd xmm4,xmm3,0x1
    9b50:	movupd xmm1,XMMWORD PTR [rdx+rax*1]
    9b55:	movupd xmm0,XMMWORD PTR [rdx+rax*1]
    9b5a:	movupd xmm5,XMMWORD PTR [r10+rax*1]
    9b60:	unpcklpd xmm1,xmm1
    9b64:	unpckhpd xmm0,xmm0
    9b68:	mulpd  xmm1,xmm3
    9b6c:	mulpd  xmm0,xmm4
    9b70:	movapd xmm2,xmm1
    9b74:	subpd  xmm2,xmm0
    9b78:	addpd  xmm0,xmm1
    9b7c:	movsd  xmm0,xmm2
    9b80:	addpd  xmm0,xmm5
    9b84:	movups XMMWORD PTR [r10+rax*1],xmm0
    9b89:	add    rax,0x10
    9b8d:	cmp    rax,0x100
    9b93:	jne    9b50 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x190>
    9b95:	add    r11,0x10
    9b99:	add    rdx,0x100
    9ba0:	cmp    r11,rsi
    9ba3:	jne    9b40 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x180>
    9ba5:	jmp    9af7 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x137>
    9baa:	lea    rcx,[r10+0x100]
    9bb1:	mov    rax,r10
    9bb4:	pxor   xmm0,xmm0
    9bb8:	nop    DWORD PTR [rax+rax*1+0x0]
    9bc0:	movups XMMWORD PTR [rax],xmm0
    9bc3:	add    rax,0x20
    9bc7:	movups XMMWORD PTR [rax-0x10],xmm0
    9bcb:	cmp    rax,rcx
    9bce:	jne    9bc0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x200>
    9bd0:	cmp    r9,0x10
    9bd4:	je     9c6b <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x2ab>
    9bda:	mov    rsi,rdx
    9bdd:	xor    r8d,r8d
    9be0:	mov    rdx,r10
    9be3:	mov    rcx,rdi
    9be6:	xor    eax,eax
    9be8:	nop    DWORD PTR [rax+rax*1+0x0]
    9bf0:	cmp    rax,r8
    9bf3:	mov    r11,r8
    9bf6:	cmovae r11,rax
    9bfa:	cmp    r11,r9
    9bfd:	jae    9c3b <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x27b>
    9bff:	movupd xmm0,XMMWORD PTR [rcx]
    9c03:	movupd xmm2,XMMWORD PTR [rsi]
    9c07:	movupd xmm1,XMMWORD PTR [rcx]
    9c0b:	movupd xmm7,XMMWORD PTR [rdx]
    9c0f:	unpcklpd xmm0,xmm0
    9c13:	mulpd  xmm0,xmm2
    9c17:	unpckhpd xmm1,xmm1
    9c1b:	shufpd xmm2,xmm2,0x1
    9c20:	mulpd  xmm1,xmm2
    9c24:	movapd xmm2,xmm0
    9c28:	subpd  xmm2,xmm1
    9c2c:	addpd  xmm0,xmm1
    9c30:	movsd  xmm0,xmm2
    9c34:	addpd  xmm0,xmm7
    9c38:	movups XMMWORD PTR [rdx],xmm0
    9c3b:	add    rax,0x1
    9c3f:	add    rcx,0x10
    9c43:	add    rdx,0x10
    9c47:	cmp    rax,0x10
    9c4b:	jne    9bf0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x230>
    9c4d:	add    r8,0x1
    9c51:	add    rdi,0x100
    9c58:	add    rsi,0x10
    9c5c:	cmp    r8,0x10
    9c60:	jne    9be0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x220>
    9c66:	jmp    9b1b <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x15b>
    9c6b:	mov    rax,rdx
    9c6e:	lea    rdx,[rdi+0x100]
    9c75:	xor    edi,edi
    9c77:	lea    rsi,[rdx-0x100]
    9c7e:	mov    rcx,r10
    9c81:	nop    DWORD PTR [rax+0x0]
    9c85:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    9c90:	movupd xmm0,XMMWORD PTR [rsi]
    9c94:	movupd xmm2,XMMWORD PTR [rax]
    9c98:	add    rsi,0x10
    9c9c:	add    rcx,0x10
    9ca0:	movupd xmm1,XMMWORD PTR [rsi-0x10]
    9ca5:	movupd xmm7,XMMWORD PTR [rcx-0x10]
    9caa:	unpcklpd xmm0,xmm0
    9cae:	mulpd  xmm0,xmm2
    9cb2:	unpckhpd xmm1,xmm1
    9cb6:	shufpd xmm2,xmm2,0x1
    9cbb:	mulpd  xmm1,xmm2
    9cbf:	movapd xmm2,xmm0
    9cc3:	subpd  xmm2,xmm1
    9cc7:	addpd  xmm0,xmm1
    9ccb:	movsd  xmm0,xmm2
    9ccf:	addpd  xmm0,xmm7
    9cd3:	movups XMMWORD PTR [rcx-0x10],xmm0
    9cd7:	cmp    rsi,rdx
    9cda:	jne    9c90 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x2d0>
    9cdc:	add    rdi,0x1
    9ce0:	add    rax,0x10
    9ce4:	add    rdx,0x100
    9ceb:	cmp    rdi,0x10
    9cef:	jne    9c77 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x2b7>
    9cf1:	jmp    9b1b <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0+0x15b>
    9cf6:	call   1170 <__stack_chk_fail@plt>

Disassembly of section .fini:
