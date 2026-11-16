
/tmp/matrix_peak_stack_audit/pre_constants_ab31/pre_constants_ab31_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000004240 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0>:
    4240:	movsd  xmm0,QWORD PTR [rdi]
    4244:	mulsd  xmm0,QWORD PTR [rsi]
    4248:	addsd  xmm0,QWORD PTR [rdx]
    424c:	movsd  QWORD PTR [rdx],xmm0
    4250:	cmp    rcx,0x1
    4254:	je     45dd <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x39d>
    425a:	movsd  xmm1,QWORD PTR [rdi+0x8]
    425f:	mulsd  xmm1,QWORD PTR [rsi]
    4263:	addsd  xmm1,QWORD PTR [rdx+0x8]
    4268:	movsd  QWORD PTR [rdx+0x8],xmm1
    426d:	cmp    rcx,0x2
    4271:	je     45e0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x3a0>
    4277:	movsd  xmm2,QWORD PTR [rdi+0x10]
    427c:	mulsd  xmm2,QWORD PTR [rsi]
    4280:	addsd  xmm2,QWORD PTR [rdx+0x10]
    4285:	movsd  QWORD PTR [rdx+0x10],xmm2
    428a:	cmp    rcx,0x3
    428e:	je     4610 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x3d0>
    4294:	movsd  xmm3,QWORD PTR [rdi+0x18]
    4299:	mulsd  xmm3,QWORD PTR [rsi]
    429d:	addsd  xmm3,QWORD PTR [rdx+0x18]
    42a2:	movsd  QWORD PTR [rdx+0x18],xmm3
    42a7:	cmp    rcx,0x4
    42ab:	je     44c0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x280>
    42b1:	movsd  xmm4,QWORD PTR [rdi+0x20]
    42b6:	mulsd  xmm4,QWORD PTR [rsi]
    42ba:	addsd  xmm4,QWORD PTR [rdx+0x20]
    42bf:	movsd  QWORD PTR [rdx+0x20],xmm4
    42c4:	cmp    rcx,0x5
    42c8:	je     44c0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x280>
    42ce:	movsd  xmm4,QWORD PTR [rdi+0x28]
    42d3:	mulsd  xmm4,QWORD PTR [rsi]
    42d7:	addsd  xmm4,QWORD PTR [rdx+0x28]
    42dc:	movsd  QWORD PTR [rdx+0x28],xmm4
    42e1:	cmp    rcx,0x6
    42e5:	je     44c0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x280>
    42eb:	movsd  xmm4,QWORD PTR [rdi+0x30]
    42f0:	mulsd  xmm4,QWORD PTR [rsi]
    42f4:	addsd  xmm4,QWORD PTR [rdx+0x30]
    42f9:	movsd  QWORD PTR [rdx+0x30],xmm4
    42fe:	cmp    rcx,0x7
    4302:	je     44c0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x280>
    4308:	movsd  xmm4,QWORD PTR [rdi+0x38]
    430d:	mulsd  xmm4,QWORD PTR [rsi]
    4311:	addsd  xmm4,QWORD PTR [rdx+0x38]
    4316:	movsd  QWORD PTR [rdx+0x38],xmm4
    431b:	movsd  xmm4,QWORD PTR [rsi+0x8]
    4320:	mulsd  xmm4,QWORD PTR [rdi+0x40]
    4325:	addsd  xmm4,xmm0
    4329:	movsd  QWORD PTR [rdx],xmm4
    432d:	movsd  xmm0,QWORD PTR [rsi+0x8]
    4332:	mulsd  xmm0,QWORD PTR [rdi+0x48]
    4337:	addsd  xmm0,xmm1
    433b:	movsd  QWORD PTR [rdx+0x8],xmm0
    4340:	movsd  xmm1,QWORD PTR [rsi+0x8]
    4345:	mulsd  xmm1,QWORD PTR [rdi+0x50]
    434a:	addsd  xmm1,xmm2
    434e:	movsd  QWORD PTR [rdx+0x10],xmm1
    4353:	movsd  xmm2,QWORD PTR [rsi+0x8]
    4358:	mulsd  xmm2,QWORD PTR [rdi+0x58]
    435d:	addsd  xmm3,xmm2
    4361:	movsd  QWORD PTR [rdx+0x18],xmm3
    4366:	cs nop WORD PTR [rax+rax*1+0x0]
    4370:	movsd  xmm2,QWORD PTR [rdi+0x60]
    4375:	mulsd  xmm2,QWORD PTR [rsi+0x8]
    437a:	addsd  xmm2,QWORD PTR [rdx+0x20]
    437f:	movsd  QWORD PTR [rdx+0x20],xmm2
    4384:	cmp    rcx,0x5
    4388:	je     4516 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x2d6>
    438e:	movsd  xmm2,QWORD PTR [rdi+0x68]
    4393:	mulsd  xmm2,QWORD PTR [rsi+0x8]
    4398:	addsd  xmm2,QWORD PTR [rdx+0x28]
    439d:	movsd  QWORD PTR [rdx+0x28],xmm2
    43a2:	cmp    rcx,0x6
    43a6:	je     4516 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x2d6>
    43ac:	movsd  xmm2,QWORD PTR [rdi+0x70]
    43b1:	mulsd  xmm2,QWORD PTR [rsi+0x8]
    43b6:	addsd  xmm2,QWORD PTR [rdx+0x30]
    43bb:	movsd  QWORD PTR [rdx+0x30],xmm2
    43c0:	cmp    rcx,0x7
    43c4:	je     4516 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x2d6>
    43ca:	movsd  xmm2,QWORD PTR [rdi+0x78]
    43cf:	mulsd  xmm2,QWORD PTR [rsi+0x8]
    43d4:	addsd  xmm2,QWORD PTR [rdx+0x38]
    43d9:	movsd  QWORD PTR [rdx+0x38],xmm2
    43de:	movsd  xmm2,QWORD PTR [rsi+0x10]
    43e3:	mulsd  xmm2,QWORD PTR [rdi+0x80]
    43eb:	addsd  xmm4,xmm2
    43ef:	movsd  QWORD PTR [rdx],xmm4
    43f3:	movsd  xmm2,QWORD PTR [rsi+0x10]
    43f8:	mulsd  xmm2,QWORD PTR [rdi+0x88]
    4400:	movapd xmm3,xmm2
    4404:	addsd  xmm3,xmm0
    4408:	movsd  QWORD PTR [rdx+0x8],xmm3
    440d:	movsd  xmm0,QWORD PTR [rsi+0x10]
    4412:	mulsd  xmm0,QWORD PTR [rdi+0x90]
    441a:	addsd  xmm1,xmm0
    441e:	movsd  QWORD PTR [rdx+0x10],xmm1
    4423:	movsd  xmm0,QWORD PTR [rsi+0x10]
    4428:	mulsd  xmm0,QWORD PTR [rdi+0x98]
    4430:	addsd  xmm0,QWORD PTR [rdx+0x18]
    4435:	movsd  QWORD PTR [rdx+0x18],xmm0
    443a:	nop    WORD PTR [rax+rax*1+0x0]
    4440:	movsd  xmm2,QWORD PTR [rdi+0xa0]
    4448:	mulsd  xmm2,QWORD PTR [rsi+0x10]
    444d:	addsd  xmm2,QWORD PTR [rdx+0x20]
    4452:	movsd  QWORD PTR [rdx+0x20],xmm2
    4457:	cmp    rcx,0x5
    445b:	je     457c <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x33c>
    4461:	movsd  xmm2,QWORD PTR [rdi+0xa8]
    4469:	mulsd  xmm2,QWORD PTR [rsi+0x10]
    446e:	addsd  xmm2,QWORD PTR [rdx+0x28]
    4473:	movsd  QWORD PTR [rdx+0x28],xmm2
    4478:	cmp    rcx,0x6
    447c:	je     457c <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x33c>
    4482:	movsd  xmm2,QWORD PTR [rdi+0xb0]
    448a:	mulsd  xmm2,QWORD PTR [rsi+0x10]
    448f:	addsd  xmm2,QWORD PTR [rdx+0x30]
    4494:	movsd  QWORD PTR [rdx+0x30],xmm2
    4499:	cmp    rcx,0x7
    449d:	je     457c <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x33c>
    44a3:	movsd  xmm2,QWORD PTR [rdi+0xb8]
    44ab:	mulsd  xmm2,QWORD PTR [rsi+0x10]
    44b0:	addsd  xmm2,QWORD PTR [rdx+0x38]
    44b5:	movsd  QWORD PTR [rdx+0x38],xmm2
    44ba:	jmp    457c <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x33c>
    44bf:	nop
    44c0:	movsd  xmm4,QWORD PTR [rsi+0x8]
    44c5:	mulsd  xmm4,QWORD PTR [rdi+0x40]
    44ca:	addsd  xmm4,xmm0
    44ce:	movsd  QWORD PTR [rdx],xmm4
    44d2:	movsd  xmm0,QWORD PTR [rsi+0x8]
    44d7:	mulsd  xmm0,QWORD PTR [rdi+0x48]
    44dc:	addsd  xmm0,xmm1
    44e0:	movsd  QWORD PTR [rdx+0x8],xmm0
    44e5:	movsd  xmm1,QWORD PTR [rdi+0x50]
    44ea:	mulsd  xmm1,QWORD PTR [rsi+0x8]
    44ef:	addsd  xmm1,xmm2
    44f3:	movsd  QWORD PTR [rdx+0x10],xmm1
    44f8:	movsd  xmm2,QWORD PTR [rdi+0x58]
    44fd:	mulsd  xmm2,QWORD PTR [rsi+0x8]
    4502:	addsd  xmm2,QWORD PTR [rdx+0x18]
    4507:	movsd  QWORD PTR [rdx+0x18],xmm2
    450c:	cmp    rcx,0x4
    4510:	jne    4370 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x130>
    4516:	movsd  xmm2,QWORD PTR [rsi+0x10]
    451b:	mulsd  xmm2,QWORD PTR [rdi+0x80]
    4523:	addsd  xmm4,xmm2
    4527:	movsd  QWORD PTR [rdx],xmm4
    452b:	movsd  xmm2,QWORD PTR [rsi+0x10]
    4530:	mulsd  xmm2,QWORD PTR [rdi+0x88]
    4538:	movapd xmm3,xmm2
    453c:	addsd  xmm3,xmm0
    4540:	movsd  QWORD PTR [rdx+0x8],xmm3
    4545:	movsd  xmm0,QWORD PTR [rsi+0x10]
    454a:	mulsd  xmm0,QWORD PTR [rdi+0x90]
    4552:	addsd  xmm1,xmm0
    4556:	movsd  QWORD PTR [rdx+0x10],xmm1
    455b:	movsd  xmm0,QWORD PTR [rdi+0x98]
    4563:	mulsd  xmm0,QWORD PTR [rsi+0x10]
    4568:	addsd  xmm0,QWORD PTR [rdx+0x18]
    456d:	movsd  QWORD PTR [rdx+0x18],xmm0
    4572:	cmp    rcx,0x4
    4576:	jne    4440 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x200>
    457c:	movsd  xmm2,QWORD PTR [rdi+0xc0]
    4584:	mulsd  xmm2,QWORD PTR [rsi+0x18]
    4589:	addsd  xmm4,xmm2
    458d:	movsd  QWORD PTR [rdx],xmm4
    4591:	movsd  xmm2,QWORD PTR [rdi+0xc8]
    4599:	mulsd  xmm2,QWORD PTR [rsi+0x18]
    459e:	addsd  xmm3,xmm2
    45a2:	movsd  QWORD PTR [rdx+0x8],xmm3
    45a7:	movsd  xmm2,QWORD PTR [rdi+0xd0]
    45af:	mulsd  xmm2,QWORD PTR [rsi+0x18]
    45b4:	addsd  xmm2,xmm1
    45b8:	movsd  QWORD PTR [rdx+0x10],xmm2
    45bd:	movsd  xmm1,QWORD PTR [rdi+0xd8]
    45c5:	mulsd  xmm1,QWORD PTR [rsi+0x18]
    45ca:	addsd  xmm0,xmm1
    45ce:	movsd  QWORD PTR [rdx+0x18],xmm0
    45d3:	cmp    rcx,0x4
    45d7:	jne    4690 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x450>
    45dd:	ret
    45de:	xchg   ax,ax
    45e0:	movsd  xmm2,QWORD PTR [rsi+0x8]
    45e5:	mulsd  xmm2,QWORD PTR [rdi+0x40]
    45ea:	addsd  xmm2,xmm0
    45ee:	movsd  QWORD PTR [rdx],xmm2
    45f2:	movsd  xmm0,QWORD PTR [rsi+0x8]
    45f7:	mulsd  xmm0,QWORD PTR [rdi+0x48]
    45fc:	addsd  xmm0,xmm1
    4600:	movsd  QWORD PTR [rdx+0x8],xmm0
    4605:	ret
    4606:	cs nop WORD PTR [rax+rax*1+0x0]
    4610:	movsd  xmm3,QWORD PTR [rdi+0x40]
    4615:	mulsd  xmm3,QWORD PTR [rsi+0x8]
    461a:	addsd  xmm3,xmm0
    461e:	movsd  QWORD PTR [rdx],xmm3
    4622:	movsd  xmm0,QWORD PTR [rsi+0x8]
    4627:	mulsd  xmm0,QWORD PTR [rdi+0x48]
    462c:	addsd  xmm1,xmm0
    4630:	movsd  QWORD PTR [rdx+0x8],xmm1
    4635:	movsd  xmm0,QWORD PTR [rdi+0x50]
    463a:	mulsd  xmm0,QWORD PTR [rsi+0x8]
    463f:	addsd  xmm0,xmm2
    4643:	movsd  QWORD PTR [rdx+0x10],xmm0
    4648:	movsd  xmm2,QWORD PTR [rsi+0x10]
    464d:	mulsd  xmm2,QWORD PTR [rdi+0x80]
    4655:	addsd  xmm2,xmm3
    4659:	movsd  QWORD PTR [rdx],xmm2
    465d:	movsd  xmm2,QWORD PTR [rsi+0x10]
    4662:	mulsd  xmm2,QWORD PTR [rdi+0x88]
    466a:	addsd  xmm2,xmm1
    466e:	movsd  QWORD PTR [rdx+0x8],xmm2
    4673:	movsd  xmm1,QWORD PTR [rsi+0x10]
    4678:	mulsd  xmm1,QWORD PTR [rdi+0x90]
    4680:	addsd  xmm1,xmm0
    4684:	movsd  QWORD PTR [rdx+0x10],xmm1
    4689:	ret
    468a:	nop    WORD PTR [rax+rax*1+0x0]
    4690:	movsd  xmm1,QWORD PTR [rdi+0xe0]
    4698:	mulsd  xmm1,QWORD PTR [rsi+0x18]
    469d:	addsd  xmm1,QWORD PTR [rdx+0x20]
    46a2:	movsd  QWORD PTR [rdx+0x20],xmm1
    46a7:	cmp    rcx,0x5
    46ab:	je     4700 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x4c0>
    46ad:	movsd  xmm5,QWORD PTR [rdi+0xe8]
    46b5:	mulsd  xmm5,QWORD PTR [rsi+0x18]
    46ba:	addsd  xmm5,QWORD PTR [rdx+0x28]
    46bf:	movsd  QWORD PTR [rdx+0x28],xmm5
    46c4:	cmp    rcx,0x6
    46c8:	je     4700 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x4c0>
    46ca:	movsd  xmm5,QWORD PTR [rdi+0xf0]
    46d2:	mulsd  xmm5,QWORD PTR [rsi+0x18]
    46d7:	addsd  xmm5,QWORD PTR [rdx+0x30]
    46dc:	movsd  QWORD PTR [rdx+0x30],xmm5
    46e1:	cmp    rcx,0x7
    46e5:	je     4700 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x4c0>
    46e7:	movsd  xmm5,QWORD PTR [rdi+0xf8]
    46ef:	mulsd  xmm5,QWORD PTR [rsi+0x18]
    46f4:	addsd  xmm5,QWORD PTR [rdx+0x38]
    46f9:	movsd  QWORD PTR [rdx+0x38],xmm5
    46fe:	xchg   ax,ax
    4700:	movsd  xmm5,QWORD PTR [rdi+0x100]
    4708:	mulsd  xmm5,QWORD PTR [rsi+0x20]
    470d:	addsd  xmm5,xmm4
    4711:	movsd  QWORD PTR [rdx],xmm5
    4715:	movsd  xmm4,QWORD PTR [rdi+0x108]
    471d:	mulsd  xmm4,QWORD PTR [rsi+0x20]
    4722:	addsd  xmm4,xmm3
    4726:	movsd  QWORD PTR [rdx+0x8],xmm4
    472b:	movsd  xmm3,QWORD PTR [rdi+0x110]
    4733:	mulsd  xmm3,QWORD PTR [rsi+0x20]
    4738:	addsd  xmm3,xmm2
    473c:	movsd  QWORD PTR [rdx+0x10],xmm3
    4741:	movsd  xmm2,QWORD PTR [rdi+0x118]
    4749:	mulsd  xmm2,QWORD PTR [rsi+0x20]
    474e:	addsd  xmm2,xmm0
    4752:	movsd  QWORD PTR [rdx+0x18],xmm2
    4757:	movsd  xmm0,QWORD PTR [rdi+0x120]
    475f:	mulsd  xmm0,QWORD PTR [rsi+0x20]
    4764:	addsd  xmm0,xmm1
    4768:	movsd  QWORD PTR [rdx+0x20],xmm0
    476d:	cmp    rcx,0x5
    4771:	je     45dd <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x39d>
    4777:	movsd  xmm1,QWORD PTR [rdi+0x128]
    477f:	mulsd  xmm1,QWORD PTR [rsi+0x20]
    4784:	addsd  xmm1,QWORD PTR [rdx+0x28]
    4789:	movsd  QWORD PTR [rdx+0x28],xmm1
    478e:	cmp    rcx,0x6
    4792:	je     47c8 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x588>
    4794:	movsd  xmm6,QWORD PTR [rdi+0x130]
    479c:	mulsd  xmm6,QWORD PTR [rsi+0x20]
    47a1:	addsd  xmm6,QWORD PTR [rdx+0x30]
    47a6:	movsd  QWORD PTR [rdx+0x30],xmm6
    47ab:	cmp    rcx,0x7
    47af:	je     47c8 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x588>
    47b1:	movsd  xmm6,QWORD PTR [rdi+0x138]
    47b9:	mulsd  xmm6,QWORD PTR [rsi+0x20]
    47be:	addsd  xmm6,QWORD PTR [rdx+0x38]
    47c3:	movsd  QWORD PTR [rdx+0x38],xmm6
    47c8:	movsd  xmm6,QWORD PTR [rdi+0x140]
    47d0:	mulsd  xmm6,QWORD PTR [rsi+0x28]
    47d5:	addsd  xmm5,xmm6
    47d9:	movsd  QWORD PTR [rdx],xmm5
    47dd:	movsd  xmm6,QWORD PTR [rdi+0x148]
    47e5:	mulsd  xmm6,QWORD PTR [rsi+0x28]
    47ea:	addsd  xmm6,xmm4
    47ee:	movsd  QWORD PTR [rdx+0x8],xmm6
    47f3:	movsd  xmm4,QWORD PTR [rdi+0x150]
    47fb:	mulsd  xmm4,QWORD PTR [rsi+0x28]
    4800:	addsd  xmm4,xmm3
    4804:	movsd  QWORD PTR [rdx+0x10],xmm4
    4809:	movsd  xmm3,QWORD PTR [rdi+0x158]
    4811:	mulsd  xmm3,QWORD PTR [rsi+0x28]
    4816:	addsd  xmm3,xmm2
    481a:	movsd  QWORD PTR [rdx+0x18],xmm3
    481f:	movsd  xmm2,QWORD PTR [rdi+0x160]
    4827:	mulsd  xmm2,QWORD PTR [rsi+0x28]
    482c:	addsd  xmm2,xmm0
    4830:	movsd  QWORD PTR [rdx+0x20],xmm2
    4835:	movsd  xmm0,QWORD PTR [rdi+0x168]
    483d:	mulsd  xmm0,QWORD PTR [rsi+0x28]
    4842:	addsd  xmm0,xmm1
    4846:	movsd  QWORD PTR [rdx+0x28],xmm0
    484b:	cmp    rcx,0x6
    484f:	je     45dd <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x39d>
    4855:	movsd  xmm1,QWORD PTR [rdi+0x170]
    485d:	mulsd  xmm1,QWORD PTR [rsi+0x28]
    4862:	addsd  xmm1,QWORD PTR [rdx+0x30]
    4867:	movsd  QWORD PTR [rdx+0x30],xmm1
    486c:	cmp    rcx,0x7
    4870:	je     4889 <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x649>
    4872:	movsd  xmm7,QWORD PTR [rdi+0x178]
    487a:	mulsd  xmm7,QWORD PTR [rsi+0x28]
    487f:	addsd  xmm7,QWORD PTR [rdx+0x38]
    4884:	movsd  QWORD PTR [rdx+0x38],xmm7
    4889:	movsd  xmm7,QWORD PTR [rdi+0x180]
    4891:	mulsd  xmm7,QWORD PTR [rsi+0x30]
    4896:	addsd  xmm7,xmm5
    489a:	movsd  QWORD PTR [rdx],xmm7
    489e:	movsd  xmm5,QWORD PTR [rdi+0x188]
    48a6:	mulsd  xmm5,QWORD PTR [rsi+0x30]
    48ab:	addsd  xmm6,xmm5
    48af:	movsd  QWORD PTR [rdx+0x8],xmm6
    48b4:	movsd  xmm5,QWORD PTR [rdi+0x190]
    48bc:	mulsd  xmm5,QWORD PTR [rsi+0x30]
    48c1:	addsd  xmm5,xmm4
    48c5:	movsd  QWORD PTR [rdx+0x10],xmm5
    48ca:	movsd  xmm4,QWORD PTR [rdi+0x198]
    48d2:	mulsd  xmm4,QWORD PTR [rsi+0x30]
    48d7:	addsd  xmm4,xmm3
    48db:	movsd  QWORD PTR [rdx+0x18],xmm4
    48e0:	movsd  xmm3,QWORD PTR [rdi+0x1a0]
    48e8:	mulsd  xmm3,QWORD PTR [rsi+0x30]
    48ed:	addsd  xmm3,xmm2
    48f1:	movsd  QWORD PTR [rdx+0x20],xmm3
    48f6:	movsd  xmm2,QWORD PTR [rdi+0x1a8]
    48fe:	mulsd  xmm2,QWORD PTR [rsi+0x30]
    4903:	addsd  xmm2,xmm0
    4907:	movsd  QWORD PTR [rdx+0x28],xmm2
    490c:	movsd  xmm0,QWORD PTR [rdi+0x1b0]
    4914:	mulsd  xmm0,QWORD PTR [rsi+0x30]
    4919:	addsd  xmm1,xmm0
    491d:	movsd  QWORD PTR [rdx+0x30],xmm1
    4922:	cmp    rcx,0x7
    4926:	je     45dd <_ZN4Base6Matrix14GmresOperation13DenseMultiply6ColumnIdLm8ELm0ELb0EE7computeERKNS0_6MatrixIdLm8ELm8EEERKNS0_6VectorIdLm8EEERSA_m.part.0+0x39d>
    492c:	movsd  xmm0,QWORD PTR [rdi+0x1b8]
    4934:	mulsd  xmm0,QWORD PTR [rsi+0x30]
    4939:	addsd  xmm0,QWORD PTR [rdx+0x38]
    493e:	movsd  QWORD PTR [rdx+0x38],xmm0
    4943:	movsd  xmm8,QWORD PTR [rdi+0x1c0]
    494c:	mulsd  xmm8,QWORD PTR [rsi+0x38]
    4952:	addsd  xmm8,xmm7
    4957:	movsd  QWORD PTR [rdx],xmm8
    495c:	movsd  xmm7,QWORD PTR [rdi+0x1c8]
    4964:	mulsd  xmm7,QWORD PTR [rsi+0x38]
    4969:	addsd  xmm7,xmm6
    496d:	movsd  QWORD PTR [rdx+0x8],xmm7
    4972:	movsd  xmm6,QWORD PTR [rdi+0x1d0]
    497a:	mulsd  xmm6,QWORD PTR [rsi+0x38]
    497f:	addsd  xmm6,xmm5
    4983:	movsd  QWORD PTR [rdx+0x10],xmm6
    4988:	movsd  xmm5,QWORD PTR [rdi+0x1d8]
    4990:	mulsd  xmm5,QWORD PTR [rsi+0x38]
    4995:	addsd  xmm5,xmm4
    4999:	movsd  QWORD PTR [rdx+0x18],xmm5
    499e:	movsd  xmm4,QWORD PTR [rdi+0x1e0]
    49a6:	mulsd  xmm4,QWORD PTR [rsi+0x38]
    49ab:	addsd  xmm4,xmm3
    49af:	movsd  QWORD PTR [rdx+0x20],xmm4
    49b4:	movsd  xmm3,QWORD PTR [rdi+0x1e8]
    49bc:	mulsd  xmm3,QWORD PTR [rsi+0x38]
    49c1:	addsd  xmm3,xmm2
    49c5:	movsd  QWORD PTR [rdx+0x28],xmm3
    49ca:	movsd  xmm2,QWORD PTR [rdi+0x1f0]
    49d2:	mulsd  xmm2,QWORD PTR [rsi+0x38]
    49d7:	addsd  xmm2,xmm1
    49db:	movsd  QWORD PTR [rdx+0x30],xmm2
    49e0:	movsd  xmm1,QWORD PTR [rdi+0x1f8]
    49e8:	mulsd  xmm1,QWORD PTR [rsi+0x38]
    49ed:	addsd  xmm1,xmm0
    49f1:	movsd  QWORD PTR [rdx+0x38],xmm1
    49f6:	ret

Disassembly of section .fini:
