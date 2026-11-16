
/tmp/matrix_peak_stack_audit/pre_constants_ab31/pre_constants_ab31_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000004f340 <peak_gmres_full_double_8>:
   4f340:	endbr64
   4f344:	push   rbx
   4f345:	mov    r9d,0x8
   4f34b:	sub    rsp,0x60
   4f34f:	mov    rbx,QWORD PTR fs:0x28
   4f358:	mov    QWORD PTR [rsp+0x58],rbx
   4f35d:	mov    rbx,rdi
   4f360:	mov    rdx,rsp
   4f363:	mov    QWORD PTR [rsp],rdi
   4f367:	lea    rdi,[rsp+0x10]
   4f36c:	mov    QWORD PTR [rsp+0x8],0x8
   4f375:	lea    rcx,[rbx+0x200]
   4f37c:	lea    rsi,[rbx+0x2c0]
   4f383:	lea    r8,[rbx+0x240]
   4f38a:	call   8600 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0>
   4f38f:	movdqa xmm0,XMMWORD PTR [rsp+0x10]
   4f395:	movups XMMWORD PTR [rbx+0x280],xmm0
   4f39c:	movdqa xmm0,XMMWORD PTR [rsp+0x20]
   4f3a2:	movups XMMWORD PTR [rbx+0x290],xmm0
   4f3a9:	movdqa xmm0,XMMWORD PTR [rsp+0x30]
   4f3af:	movups XMMWORD PTR [rbx+0x2a0],xmm0
   4f3b6:	movdqa xmm0,XMMWORD PTR [rsp+0x40]
   4f3bc:	movups XMMWORD PTR [rbx+0x2b0],xmm0
   4f3c3:	mov    rax,QWORD PTR [rsp+0x58]
   4f3c8:	sub    rax,QWORD PTR fs:0x28
   4f3d1:	jne    4f3d9 <peak_gmres_full_double_8+0x99>
   4f3d3:	add    rsp,0x60
   4f3d7:	pop    rbx
   4f3d8:	ret
   4f3d9:	call   1170 <__stack_chk_fail@plt>

Disassembly of section .fini:
