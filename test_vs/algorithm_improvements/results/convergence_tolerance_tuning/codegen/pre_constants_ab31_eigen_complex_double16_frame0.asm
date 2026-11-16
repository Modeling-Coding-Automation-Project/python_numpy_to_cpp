
/tmp/matrix_peak_stack_audit/pre_constants_ab31/pre_constants_ab31_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000005eff0 <peak_eigen_complex_double_16>:
   5eff0:	endbr64
   5eff4:	push   rbp
   5eff5:	pxor   xmm0,xmm0
   5eff9:	lea    rbp,[rdi+0x800]
   5f000:	push   rbx
   5f001:	mov    rbx,rdi
   5f004:	sub    rsp,0x8
   5f008:	comisd xmm0,QWORD PTR [rdi+0x828]
   5f010:	mov    BYTE PTR [rdi+0x295e],0x0
   5f017:	jb     5f02e <peak_eigen_complex_double_16+0x3e>
   5f019:	mov    rax,QWORD PTR [rip+0x4840]        # 63860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   5f020:	mov    BYTE PTR [rdi+0x295e],0x1
   5f027:	mov    QWORD PTR [rdi+0x828],rax
   5f02e:	mov    QWORD PTR [rbx+0x2968],0x0
   5f039:	pxor   xmm0,xmm0
   5f03d:	lea    rax,[rbx+0x80]
   5f044:	lea    rcx,[rbx+0x880]
   5f04b:	movapd xmm2,xmm0
   5f04f:	nop
   5f050:	lea    rdx,[rax-0x80]
   5f054:	jmp    5f068 <peak_eigen_complex_double_16+0x78>
   5f056:	cs nop WORD PTR [rax+rax*1+0x0]
   5f060:	movsd  xmm0,QWORD PTR [rbx+0x2968]
   5f068:	movsd  xmm1,QWORD PTR [rdx]
   5f06c:	movapd xmm4,xmm1
   5f070:	movapd xmm3,xmm1
   5f074:	xorpd  xmm3,XMMWORD PTR [rip+0x4714]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   5f07c:	cmpltsd xmm1,xmm2
   5f081:	andpd  xmm3,xmm1
   5f085:	andnpd xmm1,xmm4
   5f089:	orpd   xmm1,xmm3
   5f08d:	comisd xmm1,xmm0
   5f091:	jbe    5f09b <peak_eigen_complex_double_16+0xab>
   5f093:	movsd  QWORD PTR [rbx+0x2968],xmm1
   5f09b:	add    rdx,0x8
   5f09f:	cmp    rdx,rax
   5f0a2:	jne    5f060 <peak_eigen_complex_double_16+0x70>
   5f0a4:	sub    rax,0xffffffffffffff80
   5f0a8:	cmp    rax,rcx
   5f0ab:	je     5f0b7 <peak_eigen_complex_double_16+0xc7>
   5f0ad:	movsd  xmm0,QWORD PTR [rbx+0x2968]
   5f0b5:	jmp    5f050 <peak_eigen_complex_double_16+0x60>
   5f0b7:	movsd  xmm0,QWORD PTR [rbx+0x828]
   5f0bf:	lea    rsi,[rbx+0x858]
   5f0c6:	mov    rdi,rbx
   5f0c9:	call   35000 <_ZN4Base6Matrix14EigenOperation10hessenbergINS0_7ComplexIdEEdLm16EEEvRKNS0_6MatrixIT0_XT1_EXT1_EEERNS5_IT_XT1_EXT1_EEES6_>
   5f0ce:	xor    eax,eax
   5f0d0:	mov    edx,0x1
   5f0d5:	mov    rdi,rbp
   5f0d8:	mov    WORD PTR [rbx+0x295c],ax
   5f0df:	mov    WORD PTR [rbx+0x2960],dx
   5f0e6:	mov    DWORD PTR [rbx+0x2958],0x1
   5f0f0:	mov    QWORD PTR [rbx+0x2978],0x0
   5f0fb:	call   4b240 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE29continue_solving_eigen_valuesEv>
   5f100:	mov    rsi,rbx
   5f103:	mov    rdi,rbp
   5f106:	call   5b000 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE>
   5f10b:	add    rsp,0x8
   5f10f:	pop    rbx
   5f110:	pop    rbp
   5f111:	ret

Disassembly of section .fini:
