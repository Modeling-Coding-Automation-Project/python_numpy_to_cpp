
/tmp/matrix_peak_stack_audit/last_879/last_879_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000010500 <peak_gmres_full_double_8>:
   10500:	endbr64
   10504:	push   rbx
   10505:	mov    r9d,0x8
   1050b:	sub    rsp,0x60
   1050f:	mov    rbx,QWORD PTR fs:0x28
   10518:	mov    QWORD PTR [rsp+0x58],rbx
   1051d:	mov    rbx,rdi
   10520:	mov    rdx,rsp
   10523:	mov    QWORD PTR [rsp],rdi
   10527:	lea    rdi,[rsp+0x10]
   1052c:	mov    QWORD PTR [rsp+0x8],0x8
   10535:	lea    rcx,[rbx+0x200]
   1053c:	lea    rsi,[rbx+0x2c0]
   10543:	lea    r8,[rbx+0x240]
   1054a:	call   8400 <_ZN4Base6Matrix11GmresSolverIdLm8ELm8EE6_solveINS0_14GmresOperation14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEENS4_22IdentityPreconditionerIdEEEENS0_6VectorIdLm8EEERKT_RKSC_SH_RKT0_m.isra.0>
   1054f:	movdqa xmm0,XMMWORD PTR [rsp+0x10]
   10555:	movups XMMWORD PTR [rbx+0x280],xmm0
   1055c:	movdqa xmm0,XMMWORD PTR [rsp+0x20]
   10562:	movups XMMWORD PTR [rbx+0x290],xmm0
   10569:	movdqa xmm0,XMMWORD PTR [rsp+0x30]
   1056f:	movups XMMWORD PTR [rbx+0x2a0],xmm0
   10576:	movdqa xmm0,XMMWORD PTR [rsp+0x40]
   1057c:	movups XMMWORD PTR [rbx+0x2b0],xmm0
   10583:	mov    rax,QWORD PTR [rsp+0x58]
   10588:	sub    rax,QWORD PTR fs:0x28
   10591:	jne    10599 <peak_gmres_full_double_8+0x99>
   10593:	add    rsp,0x60
   10597:	pop    rbx
   10598:	ret
   10599:	call   1170 <__stack_chk_fail@plt>

Disassembly of section .fini:
