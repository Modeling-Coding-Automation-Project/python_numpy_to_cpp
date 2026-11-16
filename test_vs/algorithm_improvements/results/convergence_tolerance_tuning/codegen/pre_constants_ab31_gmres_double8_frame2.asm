
/tmp/matrix_peak_stack_audit/pre_constants_ab31/pre_constants_ab31_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000004d230 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m>:
   4d230:	endbr64
   4d234:	sub    rsp,0x58
   4d238:	mov    rax,QWORD PTR fs:0x28
   4d241:	mov    QWORD PTR [rsp+0x48],rax
   4d246:	mov    rax,rcx
   4d249:	cmp    rax,0x8
   4d24d:	je     4d3c0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x190>
   4d253:	pxor   xmm0,xmm0
   4d257:	movaps XMMWORD PTR [rsp],xmm0
   4d25b:	movaps XMMWORD PTR [rsp+0x10],xmm0
   4d260:	movaps XMMWORD PTR [rsp+0x20],xmm0
   4d265:	movaps XMMWORD PTR [rsp+0x30],xmm0
   4d26a:	test   rax,rax
   4d26d:	je     4d300 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0xd0>
   4d273:	movsd  xmm0,QWORD PTR [rsi]
   4d277:	movsd  QWORD PTR [rsp],xmm0
   4d27c:	cmp    rax,0x1
   4d280:	je     4d300 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0xd0>
   4d282:	movsd  xmm0,QWORD PTR [rsi+0x8]
   4d287:	movsd  QWORD PTR [rsp+0x8],xmm0
   4d28d:	cmp    rax,0x2
   4d291:	je     4d300 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0xd0>
   4d293:	movsd  xmm0,QWORD PTR [rsi+0x10]
   4d298:	movsd  QWORD PTR [rsp+0x10],xmm0
   4d29e:	cmp    rax,0x3
   4d2a2:	je     4d300 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0xd0>
   4d2a4:	movsd  xmm0,QWORD PTR [rsi+0x18]
   4d2a9:	movsd  QWORD PTR [rsp+0x18],xmm0
   4d2af:	cmp    rax,0x4
   4d2b3:	je     4d300 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0xd0>
   4d2b5:	movsd  xmm0,QWORD PTR [rsi+0x20]
   4d2ba:	movsd  QWORD PTR [rsp+0x20],xmm0
   4d2c0:	cmp    rax,0x5
   4d2c4:	je     4d300 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0xd0>
   4d2c6:	movsd  xmm0,QWORD PTR [rsi+0x28]
   4d2cb:	movsd  QWORD PTR [rsp+0x28],xmm0
   4d2d1:	cmp    rax,0x6
   4d2d5:	je     4d300 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0xd0>
   4d2d7:	movsd  xmm0,QWORD PTR [rsi+0x30]
   4d2dc:	movsd  QWORD PTR [rsp+0x30],xmm0
   4d2e2:	cmp    rax,0x7
   4d2e6:	je     4d300 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0xd0>
   4d2e8:	movsd  xmm0,QWORD PTR [rsi+0x38]
   4d2ed:	movsd  QWORD PTR [rsp+0x38],xmm0
   4d2f3:	xchg   ax,ax
   4d2f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   4d300:	mov    rcx,QWORD PTR [rdi+0x8]
   4d304:	pxor   xmm0,xmm0
   4d308:	mov    rdi,QWORD PTR [rdi]
   4d30b:	movups XMMWORD PTR [rdx],xmm0
   4d30e:	movups XMMWORD PTR [rdx+0x10],xmm0
   4d312:	movups XMMWORD PTR [rdx+0x20],xmm0
   4d316:	movups XMMWORD PTR [rdx+0x30],xmm0
   4d31a:	cmp    rcx,0x8
   4d31e:	je     4d408 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x1d8>
   4d324:	test   rcx,rcx
   4d327:	je     4d331 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x101>
   4d329:	mov    rsi,rsp
   4d32c:	call   4240 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
   4d331:	test   rax,rax
   4d334:	jne    4d390 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x160>
   4d336:	mov    QWORD PTR [rdx],0x0
   4d33d:	mov    QWORD PTR [rdx+0x8],0x0
   4d345:	mov    QWORD PTR [rdx+0x10],0x0
   4d34d:	mov    QWORD PTR [rdx+0x18],0x0
   4d355:	mov    QWORD PTR [rdx+0x20],0x0
   4d35d:	mov    QWORD PTR [rdx+0x28],0x0
   4d365:	mov    QWORD PTR [rdx+0x30],0x0
   4d36d:	mov    QWORD PTR [rdx+0x38],0x0
   4d375:	mov    rax,QWORD PTR [rsp+0x48]
   4d37a:	sub    rax,QWORD PTR fs:0x28
   4d383:	jne    4d43c <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x20c>
   4d389:	add    rsp,0x58
   4d38d:	ret
   4d38e:	xchg   ax,ax
   4d390:	cmp    rax,0x1
   4d394:	je     4d33d <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x10d>
   4d396:	cmp    rax,0x2
   4d39a:	je     4d345 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x115>
   4d39c:	cmp    rax,0x3
   4d3a0:	je     4d34d <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x11d>
   4d3a2:	cmp    rax,0x4
   4d3a6:	je     4d355 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x125>
   4d3a8:	cmp    rax,0x5
   4d3ac:	je     4d35d <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x12d>
   4d3ae:	cmp    rax,0x6
   4d3b2:	je     4d365 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x135>
   4d3b4:	cmp    rax,0x7
   4d3b8:	jne    4d375 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x145>
   4d3ba:	jmp    4d36d <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x13d>
   4d3bc:	nop    DWORD PTR [rax+0x0]
   4d3c0:	mov    rcx,QWORD PTR [rdi+0x8]
   4d3c4:	pxor   xmm0,xmm0
   4d3c8:	mov    rax,QWORD PTR [rdi]
   4d3cb:	movups XMMWORD PTR [rdx],xmm0
   4d3ce:	movups XMMWORD PTR [rdx+0x10],xmm0
   4d3d2:	movups XMMWORD PTR [rdx+0x20],xmm0
   4d3d6:	movups XMMWORD PTR [rdx+0x30],xmm0
   4d3da:	cmp    rcx,0x8
   4d3de:	je     4d420 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x1f0>
   4d3e0:	test   rcx,rcx
   4d3e3:	je     4d375 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x145>
   4d3e5:	mov    rdi,QWORD PTR [rsp+0x48]
   4d3ea:	sub    rdi,QWORD PTR fs:0x28
   4d3f3:	jne    4d43c <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x20c>
   4d3f5:	mov    rdi,rax
   4d3f8:	add    rsp,0x58
   4d3fc:	jmp    4240 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
   4d401:	nop    DWORD PTR [rax+0x0]
   4d408:	mov    rsi,rsp
   4d40b:	call   5b10 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
   4d410:	test   rax,rax
   4d413:	jne    4d390 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x160>
   4d419:	jmp    4d336 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x106>
   4d41e:	xchg   ax,ax
   4d420:	mov    rcx,QWORD PTR [rsp+0x48]
   4d425:	sub    rcx,QWORD PTR fs:0x28
   4d42e:	jne    4d43c <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x20c>
   4d430:	mov    rdi,rax
   4d433:	add    rsp,0x58
   4d437:	jmp    5b10 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
   4d43c:	call   1170 <__stack_chk_fail@plt>

Disassembly of section .fini:
