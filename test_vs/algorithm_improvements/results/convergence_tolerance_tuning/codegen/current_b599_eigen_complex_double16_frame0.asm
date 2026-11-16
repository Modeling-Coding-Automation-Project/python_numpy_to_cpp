
/tmp/matrix_peak_stack_audit/current_b599/current_b599_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000005ee70 <peak_eigen_complex_double_16>:
   5ee70:	endbr64
   5ee74:	push   rbp
   5ee75:	pxor   xmm0,xmm0
   5ee79:	lea    rbp,[rdi+0x800]
   5ee80:	push   rbx
   5ee81:	mov    rbx,rdi
   5ee84:	sub    rsp,0x8
   5ee88:	comisd xmm0,QWORD PTR [rdi+0x828]
   5ee90:	mov    BYTE PTR [rdi+0x295e],0x0
   5ee97:	jb     5eeae <peak_eigen_complex_double_16+0x3e>
   5ee99:	mov    rax,QWORD PTR [rip+0x4960]        # 63800 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
   5eea0:	mov    BYTE PTR [rdi+0x295e],0x1
   5eea7:	mov    QWORD PTR [rdi+0x828],rax
   5eeae:	mov    QWORD PTR [rbx+0x2968],0x0
   5eeb9:	pxor   xmm0,xmm0
   5eebd:	lea    rax,[rbx+0x80]
   5eec4:	lea    rcx,[rbx+0x880]
   5eecb:	movapd xmm2,xmm0
   5eecf:	nop
   5eed0:	lea    rdx,[rax-0x80]
   5eed4:	jmp    5eee8 <peak_eigen_complex_double_16+0x78>
   5eed6:	cs nop WORD PTR [rax+rax*1+0x0]
   5eee0:	movsd  xmm0,QWORD PTR [rbx+0x2968]
   5eee8:	movsd  xmm1,QWORD PTR [rdx]
   5eeec:	movapd xmm4,xmm1
   5eef0:	movapd xmm3,xmm1
   5eef4:	xorpd  xmm3,XMMWORD PTR [rip+0x4894]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   5eefc:	cmpltsd xmm1,xmm2
   5ef01:	andpd  xmm3,xmm1
   5ef05:	andnpd xmm1,xmm4
   5ef09:	orpd   xmm1,xmm3
   5ef0d:	comisd xmm1,xmm0
   5ef11:	jbe    5ef1b <peak_eigen_complex_double_16+0xab>
   5ef13:	movsd  QWORD PTR [rbx+0x2968],xmm1
   5ef1b:	add    rdx,0x8
   5ef1f:	cmp    rdx,rax
   5ef22:	jne    5eee0 <peak_eigen_complex_double_16+0x70>
   5ef24:	sub    rax,0xffffffffffffff80
   5ef28:	cmp    rax,rcx
   5ef2b:	je     5ef37 <peak_eigen_complex_double_16+0xc7>
   5ef2d:	movsd  xmm0,QWORD PTR [rbx+0x2968]
   5ef35:	jmp    5eed0 <peak_eigen_complex_double_16+0x60>
   5ef37:	movsd  xmm0,QWORD PTR [rbx+0x828]
   5ef3f:	lea    rsi,[rbx+0x858]
   5ef46:	mov    rdi,rbx
   5ef49:	call   2fac0 <_ZN4Base6Matrix14EigenOperation10hessenbergINS0_7ComplexIdEEdLm16EEEvRKNS0_6MatrixIT0_XT1_EXT1_EEERNS5_IT_XT1_EXT1_EEES6_>
   5ef4e:	xor    eax,eax
   5ef50:	mov    edx,0x1
   5ef55:	mov    rdi,rbp
   5ef58:	mov    WORD PTR [rbx+0x295c],ax
   5ef5f:	mov    WORD PTR [rbx+0x2960],dx
   5ef66:	mov    DWORD PTR [rbx+0x2958],0x1
   5ef70:	mov    QWORD PTR [rbx+0x2978],0x0
   5ef7b:	call   492c0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE29continue_solving_eigen_valuesEv>
   5ef80:	mov    rsi,rbx
   5ef83:	mov    rdi,rbp
   5ef86:	call   5ae80 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE>
   5ef8b:	add    rsp,0x8
   5ef8f:	pop    rbx
   5ef90:	pop    rbp
   5ef91:	ret

Disassembly of section .fini:
