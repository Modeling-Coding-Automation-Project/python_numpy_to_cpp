
/tmp/matrix_peak_stack_audit/current_b599/current_b599_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000004180 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>:
    4180:	movsd  xmm0,QWORD PTR [rdi]
    4184:	mulsd  xmm0,QWORD PTR [rsi]
    4188:	addsd  xmm0,QWORD PTR [rdx]
    418c:	movsd  QWORD PTR [rdx],xmm0
    4190:	cmp    rcx,0x1
    4194:	je     451d <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x39d>
    419a:	movsd  xmm1,QWORD PTR [rdi+0x8]
    419f:	mulsd  xmm1,QWORD PTR [rsi]
    41a3:	addsd  xmm1,QWORD PTR [rdx+0x8]
    41a8:	movsd  QWORD PTR [rdx+0x8],xmm1
    41ad:	cmp    rcx,0x2
    41b1:	je     4520 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x3a0>
    41b7:	movsd  xmm2,QWORD PTR [rdi+0x10]
    41bc:	mulsd  xmm2,QWORD PTR [rsi]
    41c0:	addsd  xmm2,QWORD PTR [rdx+0x10]
    41c5:	movsd  QWORD PTR [rdx+0x10],xmm2
    41ca:	cmp    rcx,0x3
    41ce:	je     4550 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x3d0>
    41d4:	movsd  xmm3,QWORD PTR [rdi+0x18]
    41d9:	mulsd  xmm3,QWORD PTR [rsi]
    41dd:	addsd  xmm3,QWORD PTR [rdx+0x18]
    41e2:	movsd  QWORD PTR [rdx+0x18],xmm3
    41e7:	cmp    rcx,0x4
    41eb:	je     4400 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x280>
    41f1:	movsd  xmm4,QWORD PTR [rdi+0x20]
    41f6:	mulsd  xmm4,QWORD PTR [rsi]
    41fa:	addsd  xmm4,QWORD PTR [rdx+0x20]
    41ff:	movsd  QWORD PTR [rdx+0x20],xmm4
    4204:	cmp    rcx,0x5
    4208:	je     4400 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x280>
    420e:	movsd  xmm4,QWORD PTR [rdi+0x28]
    4213:	mulsd  xmm4,QWORD PTR [rsi]
    4217:	addsd  xmm4,QWORD PTR [rdx+0x28]
    421c:	movsd  QWORD PTR [rdx+0x28],xmm4
    4221:	cmp    rcx,0x6
    4225:	je     4400 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x280>
    422b:	movsd  xmm4,QWORD PTR [rdi+0x30]
    4230:	mulsd  xmm4,QWORD PTR [rsi]
    4234:	addsd  xmm4,QWORD PTR [rdx+0x30]
    4239:	movsd  QWORD PTR [rdx+0x30],xmm4
    423e:	cmp    rcx,0x7
    4242:	je     4400 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x280>
    4248:	movsd  xmm4,QWORD PTR [rdi+0x38]
    424d:	mulsd  xmm4,QWORD PTR [rsi]
    4251:	addsd  xmm4,QWORD PTR [rdx+0x38]
    4256:	movsd  QWORD PTR [rdx+0x38],xmm4
    425b:	movsd  xmm4,QWORD PTR [rsi+0x8]
    4260:	mulsd  xmm4,QWORD PTR [rdi+0x40]
    4265:	addsd  xmm4,xmm0
    4269:	movsd  QWORD PTR [rdx],xmm4
    426d:	movsd  xmm0,QWORD PTR [rsi+0x8]
    4272:	mulsd  xmm0,QWORD PTR [rdi+0x48]
    4277:	addsd  xmm0,xmm1
    427b:	movsd  QWORD PTR [rdx+0x8],xmm0
    4280:	movsd  xmm1,QWORD PTR [rsi+0x8]
    4285:	mulsd  xmm1,QWORD PTR [rdi+0x50]
    428a:	addsd  xmm1,xmm2
    428e:	movsd  QWORD PTR [rdx+0x10],xmm1
    4293:	movsd  xmm2,QWORD PTR [rsi+0x8]
    4298:	mulsd  xmm2,QWORD PTR [rdi+0x58]
    429d:	addsd  xmm3,xmm2
    42a1:	movsd  QWORD PTR [rdx+0x18],xmm3
    42a6:	cs nop WORD PTR [rax+rax*1+0x0]
    42b0:	movsd  xmm2,QWORD PTR [rdi+0x60]
    42b5:	mulsd  xmm2,QWORD PTR [rsi+0x8]
    42ba:	addsd  xmm2,QWORD PTR [rdx+0x20]
    42bf:	movsd  QWORD PTR [rdx+0x20],xmm2
    42c4:	cmp    rcx,0x5
    42c8:	je     4456 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x2d6>
    42ce:	movsd  xmm2,QWORD PTR [rdi+0x68]
    42d3:	mulsd  xmm2,QWORD PTR [rsi+0x8]
    42d8:	addsd  xmm2,QWORD PTR [rdx+0x28]
    42dd:	movsd  QWORD PTR [rdx+0x28],xmm2
    42e2:	cmp    rcx,0x6
    42e6:	je     4456 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x2d6>
    42ec:	movsd  xmm2,QWORD PTR [rdi+0x70]
    42f1:	mulsd  xmm2,QWORD PTR [rsi+0x8]
    42f6:	addsd  xmm2,QWORD PTR [rdx+0x30]
    42fb:	movsd  QWORD PTR [rdx+0x30],xmm2
    4300:	cmp    rcx,0x7
    4304:	je     4456 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x2d6>
    430a:	movsd  xmm2,QWORD PTR [rdi+0x78]
    430f:	mulsd  xmm2,QWORD PTR [rsi+0x8]
    4314:	addsd  xmm2,QWORD PTR [rdx+0x38]
    4319:	movsd  QWORD PTR [rdx+0x38],xmm2
    431e:	movsd  xmm2,QWORD PTR [rsi+0x10]
    4323:	mulsd  xmm2,QWORD PTR [rdi+0x80]
    432b:	addsd  xmm4,xmm2
    432f:	movsd  QWORD PTR [rdx],xmm4
    4333:	movsd  xmm2,QWORD PTR [rsi+0x10]
    4338:	mulsd  xmm2,QWORD PTR [rdi+0x88]
    4340:	movapd xmm3,xmm2
    4344:	addsd  xmm3,xmm0
    4348:	movsd  QWORD PTR [rdx+0x8],xmm3
    434d:	movsd  xmm0,QWORD PTR [rsi+0x10]
    4352:	mulsd  xmm0,QWORD PTR [rdi+0x90]
    435a:	addsd  xmm1,xmm0
    435e:	movsd  QWORD PTR [rdx+0x10],xmm1
    4363:	movsd  xmm0,QWORD PTR [rsi+0x10]
    4368:	mulsd  xmm0,QWORD PTR [rdi+0x98]
    4370:	addsd  xmm0,QWORD PTR [rdx+0x18]
    4375:	movsd  QWORD PTR [rdx+0x18],xmm0
    437a:	nop    WORD PTR [rax+rax*1+0x0]
    4380:	movsd  xmm2,QWORD PTR [rdi+0xa0]
    4388:	mulsd  xmm2,QWORD PTR [rsi+0x10]
    438d:	addsd  xmm2,QWORD PTR [rdx+0x20]
    4392:	movsd  QWORD PTR [rdx+0x20],xmm2
    4397:	cmp    rcx,0x5
    439b:	je     44bc <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x33c>
    43a1:	movsd  xmm2,QWORD PTR [rdi+0xa8]
    43a9:	mulsd  xmm2,QWORD PTR [rsi+0x10]
    43ae:	addsd  xmm2,QWORD PTR [rdx+0x28]
    43b3:	movsd  QWORD PTR [rdx+0x28],xmm2
    43b8:	cmp    rcx,0x6
    43bc:	je     44bc <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x33c>
    43c2:	movsd  xmm2,QWORD PTR [rdi+0xb0]
    43ca:	mulsd  xmm2,QWORD PTR [rsi+0x10]
    43cf:	addsd  xmm2,QWORD PTR [rdx+0x30]
    43d4:	movsd  QWORD PTR [rdx+0x30],xmm2
    43d9:	cmp    rcx,0x7
    43dd:	je     44bc <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x33c>
    43e3:	movsd  xmm2,QWORD PTR [rdi+0xb8]
    43eb:	mulsd  xmm2,QWORD PTR [rsi+0x10]
    43f0:	addsd  xmm2,QWORD PTR [rdx+0x38]
    43f5:	movsd  QWORD PTR [rdx+0x38],xmm2
    43fa:	jmp    44bc <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x33c>
    43ff:	nop
    4400:	movsd  xmm4,QWORD PTR [rsi+0x8]
    4405:	mulsd  xmm4,QWORD PTR [rdi+0x40]
    440a:	addsd  xmm4,xmm0
    440e:	movsd  QWORD PTR [rdx],xmm4
    4412:	movsd  xmm0,QWORD PTR [rsi+0x8]
    4417:	mulsd  xmm0,QWORD PTR [rdi+0x48]
    441c:	addsd  xmm0,xmm1
    4420:	movsd  QWORD PTR [rdx+0x8],xmm0
    4425:	movsd  xmm1,QWORD PTR [rdi+0x50]
    442a:	mulsd  xmm1,QWORD PTR [rsi+0x8]
    442f:	addsd  xmm1,xmm2
    4433:	movsd  QWORD PTR [rdx+0x10],xmm1
    4438:	movsd  xmm2,QWORD PTR [rdi+0x58]
    443d:	mulsd  xmm2,QWORD PTR [rsi+0x8]
    4442:	addsd  xmm2,QWORD PTR [rdx+0x18]
    4447:	movsd  QWORD PTR [rdx+0x18],xmm2
    444c:	cmp    rcx,0x4
    4450:	jne    42b0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x130>
    4456:	movsd  xmm2,QWORD PTR [rsi+0x10]
    445b:	mulsd  xmm2,QWORD PTR [rdi+0x80]
    4463:	addsd  xmm4,xmm2
    4467:	movsd  QWORD PTR [rdx],xmm4
    446b:	movsd  xmm2,QWORD PTR [rsi+0x10]
    4470:	mulsd  xmm2,QWORD PTR [rdi+0x88]
    4478:	movapd xmm3,xmm2
    447c:	addsd  xmm3,xmm0
    4480:	movsd  QWORD PTR [rdx+0x8],xmm3
    4485:	movsd  xmm0,QWORD PTR [rsi+0x10]
    448a:	mulsd  xmm0,QWORD PTR [rdi+0x90]
    4492:	addsd  xmm1,xmm0
    4496:	movsd  QWORD PTR [rdx+0x10],xmm1
    449b:	movsd  xmm0,QWORD PTR [rdi+0x98]
    44a3:	mulsd  xmm0,QWORD PTR [rsi+0x10]
    44a8:	addsd  xmm0,QWORD PTR [rdx+0x18]
    44ad:	movsd  QWORD PTR [rdx+0x18],xmm0
    44b2:	cmp    rcx,0x4
    44b6:	jne    4380 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x200>
    44bc:	movsd  xmm2,QWORD PTR [rdi+0xc0]
    44c4:	mulsd  xmm2,QWORD PTR [rsi+0x18]
    44c9:	addsd  xmm4,xmm2
    44cd:	movsd  QWORD PTR [rdx],xmm4
    44d1:	movsd  xmm2,QWORD PTR [rdi+0xc8]
    44d9:	mulsd  xmm2,QWORD PTR [rsi+0x18]
    44de:	addsd  xmm3,xmm2
    44e2:	movsd  QWORD PTR [rdx+0x8],xmm3
    44e7:	movsd  xmm2,QWORD PTR [rdi+0xd0]
    44ef:	mulsd  xmm2,QWORD PTR [rsi+0x18]
    44f4:	addsd  xmm2,xmm1
    44f8:	movsd  QWORD PTR [rdx+0x10],xmm2
    44fd:	movsd  xmm1,QWORD PTR [rdi+0xd8]
    4505:	mulsd  xmm1,QWORD PTR [rsi+0x18]
    450a:	addsd  xmm0,xmm1
    450e:	movsd  QWORD PTR [rdx+0x18],xmm0
    4513:	cmp    rcx,0x4
    4517:	jne    45d0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x450>
    451d:	ret
    451e:	xchg   ax,ax
    4520:	movsd  xmm2,QWORD PTR [rsi+0x8]
    4525:	mulsd  xmm2,QWORD PTR [rdi+0x40]
    452a:	addsd  xmm2,xmm0
    452e:	movsd  QWORD PTR [rdx],xmm2
    4532:	movsd  xmm0,QWORD PTR [rsi+0x8]
    4537:	mulsd  xmm0,QWORD PTR [rdi+0x48]
    453c:	addsd  xmm0,xmm1
    4540:	movsd  QWORD PTR [rdx+0x8],xmm0
    4545:	ret
    4546:	cs nop WORD PTR [rax+rax*1+0x0]
    4550:	movsd  xmm3,QWORD PTR [rdi+0x40]
    4555:	mulsd  xmm3,QWORD PTR [rsi+0x8]
    455a:	addsd  xmm3,xmm0
    455e:	movsd  QWORD PTR [rdx],xmm3
    4562:	movsd  xmm0,QWORD PTR [rsi+0x8]
    4567:	mulsd  xmm0,QWORD PTR [rdi+0x48]
    456c:	addsd  xmm1,xmm0
    4570:	movsd  QWORD PTR [rdx+0x8],xmm1
    4575:	movsd  xmm0,QWORD PTR [rdi+0x50]
    457a:	mulsd  xmm0,QWORD PTR [rsi+0x8]
    457f:	addsd  xmm0,xmm2
    4583:	movsd  QWORD PTR [rdx+0x10],xmm0
    4588:	movsd  xmm2,QWORD PTR [rsi+0x10]
    458d:	mulsd  xmm2,QWORD PTR [rdi+0x80]
    4595:	addsd  xmm2,xmm3
    4599:	movsd  QWORD PTR [rdx],xmm2
    459d:	movsd  xmm2,QWORD PTR [rsi+0x10]
    45a2:	mulsd  xmm2,QWORD PTR [rdi+0x88]
    45aa:	addsd  xmm2,xmm1
    45ae:	movsd  QWORD PTR [rdx+0x8],xmm2
    45b3:	movsd  xmm1,QWORD PTR [rsi+0x10]
    45b8:	mulsd  xmm1,QWORD PTR [rdi+0x90]
    45c0:	addsd  xmm1,xmm0
    45c4:	movsd  QWORD PTR [rdx+0x10],xmm1
    45c9:	ret
    45ca:	nop    WORD PTR [rax+rax*1+0x0]
    45d0:	movsd  xmm1,QWORD PTR [rdi+0xe0]
    45d8:	mulsd  xmm1,QWORD PTR [rsi+0x18]
    45dd:	addsd  xmm1,QWORD PTR [rdx+0x20]
    45e2:	movsd  QWORD PTR [rdx+0x20],xmm1
    45e7:	cmp    rcx,0x5
    45eb:	je     4640 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x4c0>
    45ed:	movsd  xmm5,QWORD PTR [rdi+0xe8]
    45f5:	mulsd  xmm5,QWORD PTR [rsi+0x18]
    45fa:	addsd  xmm5,QWORD PTR [rdx+0x28]
    45ff:	movsd  QWORD PTR [rdx+0x28],xmm5
    4604:	cmp    rcx,0x6
    4608:	je     4640 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x4c0>
    460a:	movsd  xmm5,QWORD PTR [rdi+0xf0]
    4612:	mulsd  xmm5,QWORD PTR [rsi+0x18]
    4617:	addsd  xmm5,QWORD PTR [rdx+0x30]
    461c:	movsd  QWORD PTR [rdx+0x30],xmm5
    4621:	cmp    rcx,0x7
    4625:	je     4640 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x4c0>
    4627:	movsd  xmm5,QWORD PTR [rdi+0xf8]
    462f:	mulsd  xmm5,QWORD PTR [rsi+0x18]
    4634:	addsd  xmm5,QWORD PTR [rdx+0x38]
    4639:	movsd  QWORD PTR [rdx+0x38],xmm5
    463e:	xchg   ax,ax
    4640:	movsd  xmm5,QWORD PTR [rdi+0x100]
    4648:	mulsd  xmm5,QWORD PTR [rsi+0x20]
    464d:	addsd  xmm5,xmm4
    4651:	movsd  QWORD PTR [rdx],xmm5
    4655:	movsd  xmm4,QWORD PTR [rdi+0x108]
    465d:	mulsd  xmm4,QWORD PTR [rsi+0x20]
    4662:	addsd  xmm4,xmm3
    4666:	movsd  QWORD PTR [rdx+0x8],xmm4
    466b:	movsd  xmm3,QWORD PTR [rdi+0x110]
    4673:	mulsd  xmm3,QWORD PTR [rsi+0x20]
    4678:	addsd  xmm3,xmm2
    467c:	movsd  QWORD PTR [rdx+0x10],xmm3
    4681:	movsd  xmm2,QWORD PTR [rdi+0x118]
    4689:	mulsd  xmm2,QWORD PTR [rsi+0x20]
    468e:	addsd  xmm2,xmm0
    4692:	movsd  QWORD PTR [rdx+0x18],xmm2
    4697:	movsd  xmm0,QWORD PTR [rdi+0x120]
    469f:	mulsd  xmm0,QWORD PTR [rsi+0x20]
    46a4:	addsd  xmm0,xmm1
    46a8:	movsd  QWORD PTR [rdx+0x20],xmm0
    46ad:	cmp    rcx,0x5
    46b1:	je     451d <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x39d>
    46b7:	movsd  xmm1,QWORD PTR [rdi+0x128]
    46bf:	mulsd  xmm1,QWORD PTR [rsi+0x20]
    46c4:	addsd  xmm1,QWORD PTR [rdx+0x28]
    46c9:	movsd  QWORD PTR [rdx+0x28],xmm1
    46ce:	cmp    rcx,0x6
    46d2:	je     4708 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x588>
    46d4:	movsd  xmm6,QWORD PTR [rdi+0x130]
    46dc:	mulsd  xmm6,QWORD PTR [rsi+0x20]
    46e1:	addsd  xmm6,QWORD PTR [rdx+0x30]
    46e6:	movsd  QWORD PTR [rdx+0x30],xmm6
    46eb:	cmp    rcx,0x7
    46ef:	je     4708 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x588>
    46f1:	movsd  xmm6,QWORD PTR [rdi+0x138]
    46f9:	mulsd  xmm6,QWORD PTR [rsi+0x20]
    46fe:	addsd  xmm6,QWORD PTR [rdx+0x38]
    4703:	movsd  QWORD PTR [rdx+0x38],xmm6
    4708:	movsd  xmm6,QWORD PTR [rdi+0x140]
    4710:	mulsd  xmm6,QWORD PTR [rsi+0x28]
    4715:	addsd  xmm5,xmm6
    4719:	movsd  QWORD PTR [rdx],xmm5
    471d:	movsd  xmm6,QWORD PTR [rdi+0x148]
    4725:	mulsd  xmm6,QWORD PTR [rsi+0x28]
    472a:	addsd  xmm6,xmm4
    472e:	movsd  QWORD PTR [rdx+0x8],xmm6
    4733:	movsd  xmm4,QWORD PTR [rdi+0x150]
    473b:	mulsd  xmm4,QWORD PTR [rsi+0x28]
    4740:	addsd  xmm4,xmm3
    4744:	movsd  QWORD PTR [rdx+0x10],xmm4
    4749:	movsd  xmm3,QWORD PTR [rdi+0x158]
    4751:	mulsd  xmm3,QWORD PTR [rsi+0x28]
    4756:	addsd  xmm3,xmm2
    475a:	movsd  QWORD PTR [rdx+0x18],xmm3
    475f:	movsd  xmm2,QWORD PTR [rdi+0x160]
    4767:	mulsd  xmm2,QWORD PTR [rsi+0x28]
    476c:	addsd  xmm2,xmm0
    4770:	movsd  QWORD PTR [rdx+0x20],xmm2
    4775:	movsd  xmm0,QWORD PTR [rdi+0x168]
    477d:	mulsd  xmm0,QWORD PTR [rsi+0x28]
    4782:	addsd  xmm0,xmm1
    4786:	movsd  QWORD PTR [rdx+0x28],xmm0
    478b:	cmp    rcx,0x6
    478f:	je     451d <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x39d>
    4795:	movsd  xmm1,QWORD PTR [rdi+0x170]
    479d:	mulsd  xmm1,QWORD PTR [rsi+0x28]
    47a2:	addsd  xmm1,QWORD PTR [rdx+0x30]
    47a7:	movsd  QWORD PTR [rdx+0x30],xmm1
    47ac:	cmp    rcx,0x7
    47b0:	je     47c9 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x649>
    47b2:	movsd  xmm7,QWORD PTR [rdi+0x178]
    47ba:	mulsd  xmm7,QWORD PTR [rsi+0x28]
    47bf:	addsd  xmm7,QWORD PTR [rdx+0x38]
    47c4:	movsd  QWORD PTR [rdx+0x38],xmm7
    47c9:	movsd  xmm7,QWORD PTR [rdi+0x180]
    47d1:	mulsd  xmm7,QWORD PTR [rsi+0x30]
    47d6:	addsd  xmm7,xmm5
    47da:	movsd  QWORD PTR [rdx],xmm7
    47de:	movsd  xmm5,QWORD PTR [rdi+0x188]
    47e6:	mulsd  xmm5,QWORD PTR [rsi+0x30]
    47eb:	addsd  xmm6,xmm5
    47ef:	movsd  QWORD PTR [rdx+0x8],xmm6
    47f4:	movsd  xmm5,QWORD PTR [rdi+0x190]
    47fc:	mulsd  xmm5,QWORD PTR [rsi+0x30]
    4801:	addsd  xmm5,xmm4
    4805:	movsd  QWORD PTR [rdx+0x10],xmm5
    480a:	movsd  xmm4,QWORD PTR [rdi+0x198]
    4812:	mulsd  xmm4,QWORD PTR [rsi+0x30]
    4817:	addsd  xmm4,xmm3
    481b:	movsd  QWORD PTR [rdx+0x18],xmm4
    4820:	movsd  xmm3,QWORD PTR [rdi+0x1a0]
    4828:	mulsd  xmm3,QWORD PTR [rsi+0x30]
    482d:	addsd  xmm3,xmm2
    4831:	movsd  QWORD PTR [rdx+0x20],xmm3
    4836:	movsd  xmm2,QWORD PTR [rdi+0x1a8]
    483e:	mulsd  xmm2,QWORD PTR [rsi+0x30]
    4843:	addsd  xmm2,xmm0
    4847:	movsd  QWORD PTR [rdx+0x28],xmm2
    484c:	movsd  xmm0,QWORD PTR [rdi+0x1b0]
    4854:	mulsd  xmm0,QWORD PTR [rsi+0x30]
    4859:	addsd  xmm1,xmm0
    485d:	movsd  QWORD PTR [rdx+0x30],xmm1
    4862:	cmp    rcx,0x7
    4866:	je     451d <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x39d>
    486c:	movsd  xmm0,QWORD PTR [rdi+0x1b8]
    4874:	mulsd  xmm0,QWORD PTR [rsi+0x30]
    4879:	addsd  xmm0,QWORD PTR [rdx+0x38]
    487e:	movsd  QWORD PTR [rdx+0x38],xmm0
    4883:	movsd  xmm8,QWORD PTR [rdi+0x1c0]
    488c:	mulsd  xmm8,QWORD PTR [rsi+0x38]
    4892:	addsd  xmm8,xmm7
    4897:	movsd  QWORD PTR [rdx],xmm8
    489c:	movsd  xmm7,QWORD PTR [rdi+0x1c8]
    48a4:	mulsd  xmm7,QWORD PTR [rsi+0x38]
    48a9:	addsd  xmm7,xmm6
    48ad:	movsd  QWORD PTR [rdx+0x8],xmm7
    48b2:	movsd  xmm6,QWORD PTR [rdi+0x1d0]
    48ba:	mulsd  xmm6,QWORD PTR [rsi+0x38]
    48bf:	addsd  xmm6,xmm5
    48c3:	movsd  QWORD PTR [rdx+0x10],xmm6
    48c8:	movsd  xmm5,QWORD PTR [rdi+0x1d8]
    48d0:	mulsd  xmm5,QWORD PTR [rsi+0x38]
    48d5:	addsd  xmm5,xmm4
    48d9:	movsd  QWORD PTR [rdx+0x18],xmm5
    48de:	movsd  xmm4,QWORD PTR [rdi+0x1e0]
    48e6:	mulsd  xmm4,QWORD PTR [rsi+0x38]
    48eb:	addsd  xmm4,xmm3
    48ef:	movsd  QWORD PTR [rdx+0x20],xmm4
    48f4:	movsd  xmm3,QWORD PTR [rdi+0x1e8]
    48fc:	mulsd  xmm3,QWORD PTR [rsi+0x38]
    4901:	addsd  xmm3,xmm2
    4905:	movsd  QWORD PTR [rdx+0x28],xmm3
    490a:	movsd  xmm2,QWORD PTR [rdi+0x1f0]
    4912:	mulsd  xmm2,QWORD PTR [rsi+0x38]
    4917:	addsd  xmm2,xmm1
    491b:	movsd  QWORD PTR [rdx+0x30],xmm2
    4920:	movsd  xmm1,QWORD PTR [rdi+0x1f8]
    4928:	mulsd  xmm1,QWORD PTR [rsi+0x38]
    492d:	addsd  xmm1,xmm0
    4931:	movsd  QWORD PTR [rdx+0x38],xmm1
    4936:	ret

Disassembly of section .fini:
