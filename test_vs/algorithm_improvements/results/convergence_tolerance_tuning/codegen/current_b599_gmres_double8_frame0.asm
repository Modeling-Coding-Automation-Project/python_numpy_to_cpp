
/tmp/matrix_peak_stack_audit/current_b599/current_b599_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000004f240 <peak_gmres_full_double_8>:
   4f240:	endbr64
   4f244:	push   rbx
   4f245:	mov    r9d,0x8
   4f24b:	sub    rsp,0x60
   4f24f:	mov    rbx,QWORD PTR fs:0x28
   4f258:	mov    QWORD PTR [rsp+0x58],rbx
   4f25d:	mov    rbx,rdi
   4f260:	mov    rdx,rsp
   4f263:	mov    QWORD PTR [rsp],rdi
   4f267:	lea    rdi,[rsp+0x10]
   4f26c:	mov    QWORD PTR [rsp+0x8],0x8
   4f275:	lea    rcx,[rbx+0x200]
   4f27c:	lea    rsi,[rbx+0x2c0]
   4f283:	lea    r8,[rbx+0x240]
   4f28a:	call   8320 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0>
   4f28f:	movdqa xmm0,XMMWORD PTR [rsp+0x10]
   4f295:	movups XMMWORD PTR [rbx+0x280],xmm0
   4f29c:	movdqa xmm0,XMMWORD PTR [rsp+0x20]
   4f2a2:	movups XMMWORD PTR [rbx+0x290],xmm0
   4f2a9:	movdqa xmm0,XMMWORD PTR [rsp+0x30]
   4f2af:	movups XMMWORD PTR [rbx+0x2a0],xmm0
   4f2b6:	movdqa xmm0,XMMWORD PTR [rsp+0x40]
   4f2bc:	movups XMMWORD PTR [rbx+0x2b0],xmm0
   4f2c3:	mov    rax,QWORD PTR [rsp+0x58]
   4f2c8:	sub    rax,QWORD PTR fs:0x28
   4f2d1:	jne    4f2d9 <peak_gmres_full_double_8+0x99>
   4f2d3:	add    rsp,0x60
   4f2d7:	pop    rbx
   4f2d8:	ret
   4f2d9:	call   1170 <__stack_chk_fail@plt>

Disassembly of section .fini:
