
/tmp/matrix_peak_stack_audit/pre_constants_ab31/pre_constants_ab31_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000016120 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm10ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>:
   16120:	movsd  xmm3,QWORD PTR [rdi+0xa88]
   16128:	movsd  xmm0,QWORD PTR [rdi+0xa80]
   16130:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   16138:	movsd  xmm2,QWORD PTR [rsi+0xa0]
   16140:	movapd xmm1,xmm0
   16144:	movapd xmm5,xmm3
   16148:	mulsd  xmm5,xmm4
   1614c:	mulsd  xmm1,xmm2
   16150:	mulsd  xmm0,xmm4
   16154:	mulsd  xmm2,xmm3
   16158:	subsd  xmm1,xmm5
   1615c:	addsd  xmm1,QWORD PTR [rdx+0x80]
   16164:	addsd  xmm0,xmm2
   16168:	addsd  xmm0,QWORD PTR [rdx+0x88]
   16170:	movsd  QWORD PTR [rdx+0x80],xmm1
   16178:	movsd  xmm2,QWORD PTR [rsi+0xa0]
   16180:	movsd  QWORD PTR [rdx+0x88],xmm0
   16188:	movsd  xmm0,QWORD PTR [rdi+0xa90]
   16190:	movsd  xmm3,QWORD PTR [rdi+0xa98]
   16198:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   161a0:	movapd xmm1,xmm0
   161a4:	movapd xmm5,xmm3
   161a8:	mulsd  xmm5,xmm4
   161ac:	mulsd  xmm1,xmm2
   161b0:	mulsd  xmm0,xmm4
   161b4:	mulsd  xmm2,xmm3
   161b8:	subsd  xmm1,xmm5
   161bc:	addsd  xmm1,QWORD PTR [rdx+0x90]
   161c4:	addsd  xmm0,xmm2
   161c8:	addsd  xmm0,QWORD PTR [rdx+0x98]
   161d0:	movsd  QWORD PTR [rdx+0x90],xmm1
   161d8:	movsd  xmm2,QWORD PTR [rsi+0xa0]
   161e0:	movsd  QWORD PTR [rdx+0x98],xmm0
   161e8:	movsd  xmm0,QWORD PTR [rdi+0xaa0]
   161f0:	movsd  xmm3,QWORD PTR [rdi+0xaa8]
   161f8:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   16200:	movapd xmm1,xmm0
   16204:	movapd xmm5,xmm3
   16208:	mulsd  xmm5,xmm4
   1620c:	mulsd  xmm1,xmm2
   16210:	mulsd  xmm0,xmm4
   16214:	mulsd  xmm2,xmm3
   16218:	subsd  xmm1,xmm5
   1621c:	addsd  xmm1,QWORD PTR [rdx+0xa0]
   16224:	addsd  xmm0,xmm2
   16228:	addsd  xmm0,QWORD PTR [rdx+0xa8]
   16230:	movsd  QWORD PTR [rdx+0xa0],xmm1
   16238:	movsd  xmm2,QWORD PTR [rsi+0xa0]
   16240:	movsd  QWORD PTR [rdx+0xa8],xmm0
   16248:	movsd  xmm0,QWORD PTR [rdi+0xab0]
   16250:	movsd  xmm3,QWORD PTR [rdi+0xab8]
   16258:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   16260:	movapd xmm1,xmm0
   16264:	movapd xmm5,xmm3
   16268:	mulsd  xmm5,xmm4
   1626c:	mulsd  xmm1,xmm2
   16270:	mulsd  xmm0,xmm4
   16274:	mulsd  xmm2,xmm3
   16278:	subsd  xmm1,xmm5
   1627c:	addsd  xmm1,QWORD PTR [rdx+0xb0]
   16284:	addsd  xmm0,xmm2
   16288:	addsd  xmm0,QWORD PTR [rdx+0xb8]
   16290:	movsd  QWORD PTR [rdx+0xb0],xmm1
   16298:	movsd  QWORD PTR [rdx+0xb8],xmm0
   162a0:	movsd  xmm0,QWORD PTR [rdi+0xac0]
   162a8:	movsd  xmm3,QWORD PTR [rdi+0xac8]
   162b0:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   162b8:	movsd  xmm2,QWORD PTR [rsi+0xa0]
   162c0:	movapd xmm1,xmm0
   162c4:	movapd xmm5,xmm3
   162c8:	movupd xmm6,XMMWORD PTR [rdx+0xf0]
   162d0:	mulsd  xmm5,xmm4
   162d4:	mulsd  xmm1,xmm2
   162d8:	mulsd  xmm0,xmm4
   162dc:	mulsd  xmm2,xmm3
   162e0:	subsd  xmm1,xmm5
   162e4:	addsd  xmm1,QWORD PTR [rdx+0xc0]
   162ec:	addsd  xmm0,xmm2
   162f0:	addsd  xmm0,QWORD PTR [rdx+0xc8]
   162f8:	movsd  QWORD PTR [rdx+0xc0],xmm1
   16300:	movsd  xmm2,QWORD PTR [rsi+0xa0]
   16308:	movsd  QWORD PTR [rdx+0xc8],xmm0
   16310:	movsd  xmm0,QWORD PTR [rdi+0xad0]
   16318:	movsd  xmm3,QWORD PTR [rdi+0xad8]
   16320:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   16328:	movapd xmm1,xmm0
   1632c:	movapd xmm5,xmm3
   16330:	mulsd  xmm5,xmm4
   16334:	mulsd  xmm1,xmm2
   16338:	mulsd  xmm0,xmm4
   1633c:	mulsd  xmm2,xmm3
   16340:	subsd  xmm1,xmm5
   16344:	addsd  xmm1,QWORD PTR [rdx+0xd0]
   1634c:	addsd  xmm0,xmm2
   16350:	addsd  xmm0,QWORD PTR [rdx+0xd8]
   16358:	movsd  QWORD PTR [rdx+0xd0],xmm1
   16360:	movsd  xmm2,QWORD PTR [rsi+0xa0]
   16368:	movsd  QWORD PTR [rdx+0xd8],xmm0
   16370:	movsd  xmm0,QWORD PTR [rdi+0xae0]
   16378:	movsd  xmm3,QWORD PTR [rdi+0xae8]
   16380:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   16388:	movapd xmm1,xmm0
   1638c:	movapd xmm5,xmm3
   16390:	mulsd  xmm5,xmm4
   16394:	mulsd  xmm1,xmm2
   16398:	mulsd  xmm0,xmm4
   1639c:	mulsd  xmm2,xmm3
   163a0:	subsd  xmm1,xmm5
   163a4:	addsd  xmm1,QWORD PTR [rdx+0xe0]
   163ac:	addsd  xmm0,xmm2
   163b0:	addsd  xmm0,QWORD PTR [rdx+0xe8]
   163b8:	movsd  QWORD PTR [rdx+0xe0],xmm1
   163c0:	movsd  xmm1,QWORD PTR [rdi+0xaf0]
   163c8:	movsd  xmm2,QWORD PTR [rsi+0xa0]
   163d0:	movsd  QWORD PTR [rdx+0xe8],xmm0
   163d8:	movsd  xmm3,QWORD PTR [rdi+0xaf8]
   163e0:	movapd xmm0,xmm1
   163e4:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   163ec:	mulsd  xmm0,xmm2
   163f0:	mulsd  xmm1,xmm4
   163f4:	movapd xmm5,xmm3
   163f8:	mulsd  xmm5,xmm4
   163fc:	mulsd  xmm2,xmm3
   16400:	subsd  xmm0,xmm5
   16404:	addsd  xmm1,xmm2
   16408:	unpcklpd xmm0,xmm1
   1640c:	addpd  xmm0,xmm6
   16410:	movups XMMWORD PTR [rdx+0xf0],xmm0
   16417:	ret

Disassembly of section .fini:
