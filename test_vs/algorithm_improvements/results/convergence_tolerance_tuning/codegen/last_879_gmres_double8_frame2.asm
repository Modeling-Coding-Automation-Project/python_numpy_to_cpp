
/tmp/matrix_peak_stack_audit/last_879/last_879_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000004260 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>:
    4260:	sub    rsp,0x58
    4264:	mov    r9,rsi
    4267:	pxor   xmm0,xmm0
    426b:	mov    rsi,QWORD PTR fs:0x28
    4274:	mov    QWORD PTR [rsp+0x48],rsi
    4279:	mov    rsi,rdx
    427c:	cmp    r8,0x8
    4280:	je     441d <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x1bd>
    4286:	movaps XMMWORD PTR [rsp],xmm0
    428a:	mov    r10,rsp
    428d:	mov    rdx,rsp
    4290:	xor    eax,eax
    4292:	movaps XMMWORD PTR [rsp+0x10],xmm0
    4297:	movaps XMMWORD PTR [rsp+0x20],xmm0
    429c:	movaps XMMWORD PTR [rsp+0x30],xmm0
    42a1:	nop    WORD PTR [rax+rax*1+0x0]
    42aa:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    42b5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    42c0:	cmp    rax,r8
    42c3:	jae    42ce <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x6e>
    42c5:	movsd  xmm0,QWORD PTR [rsi+rax*8]
    42ca:	movsd  QWORD PTR [rdx],xmm0
    42ce:	add    rax,0x1
    42d2:	add    rdx,0x8
    42d6:	cmp    rax,0x8
    42da:	jne    42c0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x60>
    42dc:	pxor   xmm0,xmm0
    42e0:	xor    edx,edx
    42e2:	movups XMMWORD PTR [rcx],xmm0
    42e5:	movups XMMWORD PTR [rcx+0x10],xmm0
    42e9:	movups XMMWORD PTR [rcx+0x20],xmm0
    42ed:	movups XMMWORD PTR [rcx+0x30],xmm0
    42f1:	cmp    r9,0x8
    42f5:	je     43d0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x170>
    42fb:	nop    DWORD PTR [rax+rax*1+0x0]
    4300:	xor    eax,eax
    4302:	nop    DWORD PTR [rax+0x0]
    4309:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    4314:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    431f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    432a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    4335:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    4340:	cmp    rdx,rax
    4343:	mov    rsi,rax
    4346:	cmovae rsi,rdx
    434a:	cmp    rsi,r9
    434d:	jae    4363 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x103>
    434f:	movsd  xmm0,QWORD PTR [rdi+rax*8]
    4354:	mulsd  xmm0,QWORD PTR [r10]
    4359:	addsd  xmm0,QWORD PTR [rcx+rax*8]
    435e:	movsd  QWORD PTR [rcx+rax*8],xmm0
    4363:	add    rax,0x1
    4367:	cmp    rax,0x8
    436b:	jne    4340 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0xe0>
    436d:	add    rdx,0x1
    4371:	add    rdi,0x40
    4375:	add    r10,0x8
    4379:	cmp    rdx,0x8
    437d:	jne    4300 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0xa0>
    437f:	xor    eax,eax
    4381:	nop    WORD PTR [rax+rax*1+0x0]
    438a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    4395:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    43a0:	cmp    rax,r8
    43a3:	jb     43ad <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x14d>
    43a5:	mov    QWORD PTR [rcx+rax*8],0x0
    43ad:	add    rax,0x1
    43b1:	cmp    rax,0x8
    43b5:	jne    43a0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x140>
    43b7:	mov    rax,QWORD PTR [rsp+0x48]
    43bc:	sub    rax,QWORD PTR fs:0x28
    43c5:	jne    450f <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x2af>
    43cb:	add    rsp,0x58
    43cf:	ret
    43d0:	lea    rdx,[rdi+0x200]
    43d7:	nop    WORD PTR [rax+rax*1+0x0]
    43e0:	movsd  xmm1,QWORD PTR [r10]
    43e5:	xor    eax,eax
    43e7:	unpcklpd xmm1,xmm1
    43eb:	movupd xmm0,XMMWORD PTR [rdi+rax*1]
    43f0:	movupd xmm2,XMMWORD PTR [rcx+rax*1]
    43f5:	mulpd  xmm0,xmm1
    43f9:	addpd  xmm0,xmm2
    43fd:	movups XMMWORD PTR [rcx+rax*1],xmm0
    4401:	add    rax,0x10
    4405:	cmp    rax,0x40
    4409:	jne    43eb <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x18b>
    440b:	add    rdi,0x40
    440f:	add    r10,0x8
    4413:	cmp    rdi,rdx
    4416:	jne    43e0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x180>
    4418:	jmp    437f <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x11f>
    441d:	movups XMMWORD PTR [rcx],xmm0
    4420:	xor    edx,edx
    4422:	movups XMMWORD PTR [rcx+0x10],xmm0
    4426:	movups XMMWORD PTR [rcx+0x20],xmm0
    442a:	movups XMMWORD PTR [rcx+0x30],xmm0
    442e:	cmp    r9,0x8
    4432:	je     44c0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x260>
    4438:	nop    DWORD PTR [rax+rax*1+0x0]
    4440:	xor    eax,eax
    4442:	nop    DWORD PTR [rax+0x0]
    4449:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    4454:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    445f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    446a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    4475:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    4480:	cmp    rdx,rax
    4483:	mov    r8,rax
    4486:	cmovae r8,rdx
    448a:	cmp    r8,r9
    448d:	jae    44a3 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x243>
    448f:	movsd  xmm0,QWORD PTR [rdi+rax*8]
    4494:	mulsd  xmm0,QWORD PTR [rsi+rdx*8]
    4499:	addsd  xmm0,QWORD PTR [rcx+rax*8]
    449e:	movsd  QWORD PTR [rcx+rax*8],xmm0
    44a3:	add    rax,0x1
    44a7:	cmp    rax,0x8
    44ab:	jne    4480 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x220>
    44ad:	add    rdx,0x1
    44b1:	add    rdi,0x40
    44b5:	cmp    rdx,0x8
    44b9:	jne    4440 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x1e0>
    44bb:	jmp    43b7 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x157>
    44c0:	mov    rax,rdi
    44c3:	lea    rdi,[rdi+0x200]
    44ca:	nop    WORD PTR [rax+rax*1+0x0]
    44d0:	xor    edx,edx
    44d2:	nop    DWORD PTR [rax]
    44d5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    44e0:	movsd  xmm0,QWORD PTR [rax+rdx*8]
    44e5:	mulsd  xmm0,QWORD PTR [rsi]
    44e9:	addsd  xmm0,QWORD PTR [rcx+rdx*8]
    44ee:	movsd  QWORD PTR [rcx+rdx*8],xmm0
    44f3:	add    rdx,0x1
    44f7:	cmp    rdx,0x8
    44fb:	jne    44e0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x280>
    44fd:	add    rax,0x40
    4501:	add    rsi,0x8
    4505:	cmp    rax,rdi
    4508:	jne    44d0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x270>
    450a:	jmp    43b7 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x157>
    450f:	call   1170 <__stack_chk_fail@plt>

Disassembly of section .fini:
