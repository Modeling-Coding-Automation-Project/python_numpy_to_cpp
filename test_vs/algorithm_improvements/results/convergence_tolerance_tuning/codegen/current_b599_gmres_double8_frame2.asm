
/tmp/matrix_peak_stack_audit/current_b599/current_b599_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000004d130 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m>:
   4d130:	endbr64
   4d134:	sub    rsp,0x58
   4d138:	mov    rax,QWORD PTR fs:0x28
   4d141:	mov    QWORD PTR [rsp+0x48],rax
   4d146:	mov    rax,rcx
   4d149:	cmp    rax,0x8
   4d14d:	je     4d2c0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x190>
   4d153:	pxor   xmm0,xmm0
   4d157:	movaps XMMWORD PTR [rsp],xmm0
   4d15b:	movaps XMMWORD PTR [rsp+0x10],xmm0
   4d160:	movaps XMMWORD PTR [rsp+0x20],xmm0
   4d165:	movaps XMMWORD PTR [rsp+0x30],xmm0
   4d16a:	test   rax,rax
   4d16d:	je     4d200 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0xd0>
   4d173:	movsd  xmm0,QWORD PTR [rsi]
   4d177:	movsd  QWORD PTR [rsp],xmm0
   4d17c:	cmp    rax,0x1
   4d180:	je     4d200 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0xd0>
   4d182:	movsd  xmm0,QWORD PTR [rsi+0x8]
   4d187:	movsd  QWORD PTR [rsp+0x8],xmm0
   4d18d:	cmp    rax,0x2
   4d191:	je     4d200 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0xd0>
   4d193:	movsd  xmm0,QWORD PTR [rsi+0x10]
   4d198:	movsd  QWORD PTR [rsp+0x10],xmm0
   4d19e:	cmp    rax,0x3
   4d1a2:	je     4d200 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0xd0>
   4d1a4:	movsd  xmm0,QWORD PTR [rsi+0x18]
   4d1a9:	movsd  QWORD PTR [rsp+0x18],xmm0
   4d1af:	cmp    rax,0x4
   4d1b3:	je     4d200 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0xd0>
   4d1b5:	movsd  xmm0,QWORD PTR [rsi+0x20]
   4d1ba:	movsd  QWORD PTR [rsp+0x20],xmm0
   4d1c0:	cmp    rax,0x5
   4d1c4:	je     4d200 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0xd0>
   4d1c6:	movsd  xmm0,QWORD PTR [rsi+0x28]
   4d1cb:	movsd  QWORD PTR [rsp+0x28],xmm0
   4d1d1:	cmp    rax,0x6
   4d1d5:	je     4d200 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0xd0>
   4d1d7:	movsd  xmm0,QWORD PTR [rsi+0x30]
   4d1dc:	movsd  QWORD PTR [rsp+0x30],xmm0
   4d1e2:	cmp    rax,0x7
   4d1e6:	je     4d200 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0xd0>
   4d1e8:	movsd  xmm0,QWORD PTR [rsi+0x38]
   4d1ed:	movsd  QWORD PTR [rsp+0x38],xmm0
   4d1f3:	xchg   ax,ax
   4d1f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   4d200:	mov    rcx,QWORD PTR [rdi+0x8]
   4d204:	pxor   xmm0,xmm0
   4d208:	mov    rdi,QWORD PTR [rdi]
   4d20b:	movups XMMWORD PTR [rdx],xmm0
   4d20e:	movups XMMWORD PTR [rdx+0x10],xmm0
   4d212:	movups XMMWORD PTR [rdx+0x20],xmm0
   4d216:	movups XMMWORD PTR [rdx+0x30],xmm0
   4d21a:	cmp    rcx,0x8
   4d21e:	je     4d308 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x1d8>
   4d224:	test   rcx,rcx
   4d227:	je     4d231 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x101>
   4d229:	mov    rsi,rsp
   4d22c:	call   4180 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
   4d231:	test   rax,rax
   4d234:	jne    4d290 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x160>
   4d236:	mov    QWORD PTR [rdx],0x0
   4d23d:	mov    QWORD PTR [rdx+0x8],0x0
   4d245:	mov    QWORD PTR [rdx+0x10],0x0
   4d24d:	mov    QWORD PTR [rdx+0x18],0x0
   4d255:	mov    QWORD PTR [rdx+0x20],0x0
   4d25d:	mov    QWORD PTR [rdx+0x28],0x0
   4d265:	mov    QWORD PTR [rdx+0x30],0x0
   4d26d:	mov    QWORD PTR [rdx+0x38],0x0
   4d275:	mov    rax,QWORD PTR [rsp+0x48]
   4d27a:	sub    rax,QWORD PTR fs:0x28
   4d283:	jne    4d33c <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x20c>
   4d289:	add    rsp,0x58
   4d28d:	ret
   4d28e:	xchg   ax,ax
   4d290:	cmp    rax,0x1
   4d294:	je     4d23d <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x10d>
   4d296:	cmp    rax,0x2
   4d29a:	je     4d245 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x115>
   4d29c:	cmp    rax,0x3
   4d2a0:	je     4d24d <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x11d>
   4d2a2:	cmp    rax,0x4
   4d2a6:	je     4d255 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x125>
   4d2a8:	cmp    rax,0x5
   4d2ac:	je     4d25d <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x12d>
   4d2ae:	cmp    rax,0x6
   4d2b2:	je     4d265 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x135>
   4d2b4:	cmp    rax,0x7
   4d2b8:	jne    4d275 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x145>
   4d2ba:	jmp    4d26d <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x13d>
   4d2bc:	nop    DWORD PTR [rax+0x0]
   4d2c0:	mov    rcx,QWORD PTR [rdi+0x8]
   4d2c4:	pxor   xmm0,xmm0
   4d2c8:	mov    rax,QWORD PTR [rdi]
   4d2cb:	movups XMMWORD PTR [rdx],xmm0
   4d2ce:	movups XMMWORD PTR [rdx+0x10],xmm0
   4d2d2:	movups XMMWORD PTR [rdx+0x20],xmm0
   4d2d6:	movups XMMWORD PTR [rdx+0x30],xmm0
   4d2da:	cmp    rcx,0x8
   4d2de:	je     4d320 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x1f0>
   4d2e0:	test   rcx,rcx
   4d2e3:	je     4d275 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x145>
   4d2e5:	mov    rdi,QWORD PTR [rsp+0x48]
   4d2ea:	sub    rdi,QWORD PTR fs:0x28
   4d2f3:	jne    4d33c <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x20c>
   4d2f5:	mov    rdi,rax
   4d2f8:	add    rsp,0x58
   4d2fc:	jmp    4180 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>
   4d301:	nop    DWORD PTR [rax+0x0]
   4d308:	mov    rsi,rsp
   4d30b:	call   5a50 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
   4d310:	test   rax,rax
   4d313:	jne    4d290 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x160>
   4d319:	jmp    4d236 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x106>
   4d31e:	xchg   ax,ax
   4d320:	mov    rcx,QWORD PTR [rsp+0x48]
   4d325:	sub    rcx,QWORD PTR fs:0x28
   4d32e:	jne    4d33c <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m+0x20c>
   4d330:	mov    rdi,rax
   4d333:	add    rsp,0x58
   4d337:	jmp    5a50 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeIdLm8EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS5_XT0_EEERSA_m.part.0>
   4d33c:	call   1170 <__stack_chk_fail@plt>

Disassembly of section .fini:
