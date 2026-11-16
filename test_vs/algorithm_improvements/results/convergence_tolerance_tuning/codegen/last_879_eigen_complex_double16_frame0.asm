
/tmp/matrix_peak_stack_audit/last_879/last_879_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000033230 <peak_eigen_complex_double_16>:
   33230:	endbr64
   33234:	push   rbp
   33235:	pxor   xmm0,xmm0
   33239:	lea    rbp,[rdi+0x800]
   33240:	push   rbx
   33241:	mov    rbx,rdi
   33244:	sub    rsp,0x8
   33248:	comisd xmm0,QWORD PTR [rdi+0x828]
   33250:	mov    BYTE PTR [rdi+0x295e],0x0
   33257:	jb     3326e <peak_eigen_complex_double_16+0x3e>
   33259:	mov    rax,QWORD PTR [rip+0x4600]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   33260:	mov    BYTE PTR [rdi+0x295e],0x1
   33267:	mov    QWORD PTR [rdi+0x828],rax
   3326e:	mov    QWORD PTR [rbx+0x2968],0x0
   33279:	pxor   xmm0,xmm0
   3327d:	lea    rax,[rbx+0x80]
   33284:	lea    rcx,[rbx+0x880]
   3328b:	movapd xmm2,xmm0
   3328f:	nop
   33290:	lea    rdx,[rax-0x80]
   33294:	jmp    332a8 <peak_eigen_complex_double_16+0x78>
   33296:	cs nop WORD PTR [rax+rax*1+0x0]
   332a0:	movsd  xmm0,QWORD PTR [rbx+0x2968]
   332a8:	movsd  xmm1,QWORD PTR [rdx]
   332ac:	movapd xmm4,xmm1
   332b0:	movapd xmm3,xmm1
   332b4:	xorpd  xmm3,XMMWORD PTR [rip+0x44e4]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   332bc:	cmpltsd xmm1,xmm2
   332c1:	andpd  xmm3,xmm1
   332c5:	andnpd xmm1,xmm4
   332c9:	orpd   xmm1,xmm3
   332cd:	comisd xmm1,xmm0
   332d1:	jbe    332db <peak_eigen_complex_double_16+0xab>
   332d3:	movsd  QWORD PTR [rbx+0x2968],xmm1
   332db:	add    rdx,0x8
   332df:	cmp    rdx,rax
   332e2:	jne    332a0 <peak_eigen_complex_double_16+0x70>
   332e4:	sub    rax,0xffffffffffffff80
   332e8:	cmp    rax,rcx
   332eb:	je     332f7 <peak_eigen_complex_double_16+0xc7>
   332ed:	movsd  xmm0,QWORD PTR [rbx+0x2968]
   332f5:	jmp    33290 <peak_eigen_complex_double_16+0x60>
   332f7:	movsd  xmm0,QWORD PTR [rbx+0x828]
   332ff:	lea    rsi,[rbx+0x858]
   33306:	mov    rdi,rbx
   33309:	call   1c8c0 <_ZN4Base6Matrix14EigenOperation10hessenbergINS0_7ComplexIdEEdLm16EEEvRKNS0_6MatrixIT0_XT1_EXT1_EEERNS5_IT_XT1_EXT1_EEES6_>
   3330e:	xor    eax,eax
   33310:	mov    edx,0x1
   33315:	mov    rdi,rbp
   33318:	mov    WORD PTR [rbx+0x295c],ax
   3331f:	mov    WORD PTR [rbx+0x2960],dx
   33326:	mov    DWORD PTR [rbx+0x2958],0x1
   33330:	mov    QWORD PTR [rbx+0x2978],0x0
   3333b:	call   29a00 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE29continue_solving_eigen_valuesEv>
   33340:	mov    rsi,rbx
   33343:	mov    rdi,rbp
   33346:	call   30e80 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE>
   3334b:	add    rsp,0x8
   3334f:	pop    rbx
   33350:	pop    rbp
   33351:	ret

Disassembly of section .fini:
