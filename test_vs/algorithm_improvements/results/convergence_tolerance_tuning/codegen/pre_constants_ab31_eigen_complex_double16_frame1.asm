
/tmp/matrix_peak_stack_audit/pre_constants_ab31/pre_constants_ab31_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000005b000 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE>:
   5b000:	endbr64
   5b004:	push   r15
   5b006:	push   r14
   5b008:	push   r13
   5b00a:	push   r12
   5b00c:	push   rbp
   5b00d:	push   rbx
   5b00e:	sub    rsp,0x1000
   5b015:	or     QWORD PTR [rsp],0x0
   5b01a:	sub    rsp,0x1000
   5b021:	or     QWORD PTR [rsp],0x0
   5b026:	sub    rsp,0x1000
   5b02d:	or     QWORD PTR [rsp],0x0
   5b032:	sub    rsp,0x6b8
   5b039:	pxor   xmm0,xmm0
   5b03d:	movsd  xmm6,QWORD PTR [rdi+0x28]
   5b042:	mov    QWORD PTR [rsp+0xd0],rdi
   5b04a:	mov    QWORD PTR [rsp+0x1a8],rsi
   5b052:	mov    rax,QWORD PTR fs:0x28
   5b05b:	mov    QWORD PTR [rsp+0x36a8],rax
   5b063:	xor    eax,eax
   5b065:	comisd xmm0,xmm6
   5b069:	movsd  QWORD PTR [rsp+0xb8],xmm6
   5b072:	jb     5b091 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x91>
   5b074:	movsd  xmm0,QWORD PTR [rip+0x87e4]        # 63860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   5b07c:	mov    BYTE PTR [rdi+0x215e],0x1
   5b083:	movsd  QWORD PTR [rdi+0x28],xmm0
   5b088:	movsd  QWORD PTR [rsp+0xb8],xmm0
   5b091:	mov    rax,QWORD PTR [rsp+0xd0]
   5b099:	movzx  r10d,BYTE PTR [rax+0x215c]
   5b0a1:	mov    BYTE PTR [rax+0x215d],r10b
   5b0a8:	test   r10b,r10b
   5b0ab:	je     5b0b7 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xb7>
   5b0ad:	mov    DWORD PTR [rax+0x2158],0x1
   5b0b7:	mov    rax,QWORD PTR [rsp+0xd0]
   5b0bf:	pxor   xmm2,xmm2
   5b0c3:	movsd  xmm1,QWORD PTR [rip+0x875d]        # 63828 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
   5b0cb:	lea    rsi,[rsp+0x240]
   5b0d3:	movapd xmm0,xmm2
   5b0d7:	mov    r9d,r10d
   5b0da:	mov    QWORD PTR [rsp+0x10],rsi
   5b0df:	lea    r11,[rsp+0x2970]
   5b0e7:	movsd  xmm6,QWORD PTR [rax+0x2168]
   5b0ef:	lea    rbx,[rax+0x1158]
   5b0f6:	mov    BYTE PTR [rax+0x215f],0x0
   5b0fd:	lea    r15,[rsp+0x1940]
   5b105:	mov    QWORD PTR [rsp+0x150],rbx
   5b10d:	lea    rbx,[rsp+0x940]
   5b115:	mov    r13,rax
   5b118:	lea    rbp,[rsp+0x840]
   5b120:	cmpnltsd xmm0,xmm6
   5b125:	mov    QWORD PTR [rsp+0x30],rbx
   5b12a:	lea    rbx,[rsp+0x230]
   5b132:	lea    r12,[rsp+0x340]
   5b13a:	mov    QWORD PTR [rsp+0x198],rsi
   5b142:	mov    QWORD PTR [rsp+0xb0],0x0
   5b14e:	andpd  xmm1,xmm0
   5b152:	andnpd xmm0,xmm6
   5b156:	mov    QWORD PTR [rsp+0x20],rbx
   5b15b:	lea    rbx,[rsp+0x3510]
   5b163:	movapd xmm6,xmm0
   5b167:	movsd  xmm0,QWORD PTR [rax+0x30]
   5b16c:	mov    QWORD PTR [rsp+0x1c8],rbx
   5b174:	lea    rbx,[rsp+0x1a40]
   5b17c:	orpd   xmm6,xmm1
   5b180:	movsd  QWORD PTR [rax+0x2170],xmm2
   5b188:	movapd xmm3,xmm0
   5b18c:	movapd xmm1,xmm0
   5b190:	xorpd  xmm1,XMMWORD PTR [rip+0x85f8]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   5b198:	movsd  QWORD PTR [rsp+0x190],xmm6
   5b1a1:	cmpltsd xmm0,xmm2
   5b1a6:	andpd  xmm1,xmm0
   5b1aa:	andnpd xmm0,xmm3
   5b1ae:	orpd   xmm0,xmm1
   5b1b2:	mulsd  xmm0,xmm6
   5b1b6:	movsd  xmm6,QWORD PTR [rsp+0xb8]
   5b1bf:	maxsd  xmm6,xmm0
   5b1c3:	movsd  QWORD PTR [rsp+0x1b8],xmm6
   5b1cc:	mov    rcx,QWORD PTR [rsp+0x30]
   5b1d1:	lea    rdx,[rsp+0xa40]
   5b1d9:	pxor   xmm0,xmm0
   5b1dd:	lea    rax,[rdx-0x100]
   5b1e4:	nop
   5b1e5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5b1f0:	movaps XMMWORD PTR [rax],xmm0
   5b1f3:	add    rax,0x20
   5b1f7:	movaps XMMWORD PTR [rax-0x10],xmm0
   5b1fb:	cmp    rdx,rax
   5b1fe:	jne    5b1f0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1f0>
   5b200:	add    rdx,0x100
   5b207:	cmp    rbx,rdx
   5b20a:	jne    5b1dd <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1dd>
   5b20c:	mov    rax,QWORD PTR [rsp+0xb0]
   5b214:	mov    rsi,QWORD PTR [rsp+0x1a8]
   5b21c:	movsd  xmm1,QWORD PTR [rsp+0x1b8]
   5b225:	mov    rdx,QWORD PTR [rsp+0x30]
   5b22a:	shl    rax,0x4
   5b22e:	mov    r10,rsi
   5b231:	movsd  xmm10,QWORD PTR [r13+rax*1+0x1058]
   5b23b:	movsd  xmm9,QWORD PTR [r13+rax*1+0x1060]
   5b245:	addsd  xmm1,xmm10
   5b24a:	unpcklpd xmm1,xmm9
   5b24f:	xor    eax,eax
   5b251:	nop    DWORD PTR [rax]
   5b254:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5b25f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5b26a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5b275:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5b280:	movupd xmm0,XMMWORD PTR [rsi+rax*1]
   5b285:	movq   xmm2,xmm0
   5b289:	psrldq xmm0,0x8
   5b28e:	movaps XMMWORD PTR [rdx+rax*2],xmm2
   5b292:	movaps XMMWORD PTR [rdx+rax*2+0x10],xmm0
   5b297:	add    rax,0x10
   5b29b:	cmp    rax,0x80
   5b2a1:	jne    5b280 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x280>
   5b2a3:	movapd xmm0,XMMWORD PTR [rcx]
   5b2a7:	add    rcx,0x110
   5b2ae:	sub    rsi,0xffffffffffffff80
   5b2b2:	add    rdx,0x100
   5b2b9:	subpd  xmm0,xmm1
   5b2bd:	movaps XMMWORD PTR [rcx-0x110],xmm0
   5b2c4:	cmp    rbx,rcx
   5b2c7:	jne    5b24f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x24f>
   5b2c9:	mov    rax,QWORD PTR [rsp+0x10]
   5b2ce:	pxor   xmm0,xmm0
   5b2d2:	movaps XMMWORD PTR [rax],xmm0
   5b2d5:	add    rax,0x20
   5b2d9:	movaps XMMWORD PTR [rax-0x10],xmm0
   5b2dd:	cmp    r12,rax
   5b2e0:	jne    5b2d2 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2d2>
   5b2e2:	movzx  ecx,BYTE PTR [r13+0x2161]
   5b2ea:	mov    rax,QWORD PTR [rsp+0x10]
   5b2ef:	test   cl,cl
   5b2f1:	jne    5ef24 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3f24>
   5b2f7:	movq   xmm0,QWORD PTR [rip+0x8651]        # 63950 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x1d0>
   5b2ff:	jmp    5b30e <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x30e>
   5b301:	movaps XMMWORD PTR [rax+0x10],xmm0
   5b305:	add    rax,0x20
   5b309:	cmp    rax,r12
   5b30c:	je     5b31a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x31a>
   5b30e:	lea    rdx,[rax+0x10]
   5b312:	movaps XMMWORD PTR [rax],xmm0
   5b315:	cmp    rdx,r12
   5b318:	jne    5b301 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x301>
   5b31a:	mov    rax,QWORD PTR [rsp+0x198]
   5b322:	movsd  xmm0,QWORD PTR [rip+0x84fe]        # 63828 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
   5b32a:	addsd  xmm0,QWORD PTR [rax]
   5b32e:	movsd  QWORD PTR [rax],xmm0
   5b332:	mov    rdx,QWORD PTR [rsp+0x20]
   5b337:	mov    rdi,QWORD PTR [rsp+0x10]
   5b33c:	mov    esi,0x10
   5b341:	mov    QWORD PTR [rsp+0x230],0x0
   5b34d:	call   fc90 <_ZN4Base6Matrix9Numerical10VectorNorm9ScaleLoopINS0_7ComplexIdEELm16ELm0EE7computeERKNS0_6VectorIS5_Lm16EEEmRd.part.0.constprop.0>
   5b352:	pxor   xmm0,xmm0
   5b356:	movsd  xmm5,QWORD PTR [rsp+0x230]
   5b35f:	ucomisd xmm5,xmm0
   5b363:	jp     5b36b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x36b>
   5b365:	je     5b79f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x79f>
   5b36b:	movsd  xmm15,QWORD PTR [rsp+0x2a0]
   5b375:	movsd  xmm6,QWORD PTR [rsp+0x240]
   5b37e:	movsd  xmm14,QWORD PTR [rsp+0x248]
   5b388:	movsd  xmm4,QWORD PTR [rsp+0x250]
   5b391:	movsd  xmm13,QWORD PTR [rsp+0x258]
   5b39b:	divsd  xmm15,xmm5
   5b3a0:	movsd  xmm3,QWORD PTR [rsp+0x260]
   5b3a9:	movsd  xmm12,QWORD PTR [rsp+0x268]
   5b3b3:	movsd  xmm2,QWORD PTR [rsp+0x270]
   5b3bc:	movsd  xmm11,QWORD PTR [rsp+0x278]
   5b3c6:	movsd  xmm1,QWORD PTR [rsp+0x280]
   5b3cf:	movsd  xmm8,QWORD PTR [rsp+0x288]
   5b3d9:	movsd  xmm0,QWORD PTR [rsp+0x290]
   5b3e2:	movsd  xmm7,QWORD PTR [rsp+0x298]
   5b3eb:	divsd  xmm6,xmm5
   5b3ef:	movsd  QWORD PTR [rsp],xmm15
   5b3f5:	movsd  xmm15,QWORD PTR [rsp+0x2a8]
   5b3ff:	divsd  xmm15,xmm5
   5b404:	mulsd  xmm6,xmm6
   5b408:	divsd  xmm14,xmm5
   5b40d:	movsd  QWORD PTR [rsp+0x8],xmm15
   5b414:	movsd  xmm15,QWORD PTR [rsp+0x2b0]
   5b41e:	divsd  xmm15,xmm5
   5b423:	mulsd  xmm14,xmm14
   5b428:	addsd  xmm6,xmm14
   5b42d:	pxor   xmm14,xmm14
   5b432:	addsd  xmm6,xmm14
   5b437:	divsd  xmm4,xmm5
   5b43b:	movsd  QWORD PTR [rsp+0x18],xmm15
   5b442:	movsd  xmm15,QWORD PTR [rsp+0x2b8]
   5b44c:	divsd  xmm15,xmm5
   5b451:	mulsd  xmm4,xmm4
   5b455:	divsd  xmm13,xmm5
   5b45a:	movsd  QWORD PTR [rsp+0x28],xmm15
   5b461:	movsd  xmm15,QWORD PTR [rsp+0x2c0]
   5b46b:	divsd  xmm15,xmm5
   5b470:	mulsd  xmm13,xmm13
   5b475:	addsd  xmm4,xmm13
   5b47a:	addsd  xmm4,xmm6
   5b47e:	divsd  xmm3,xmm5
   5b482:	movsd  QWORD PTR [rsp+0x38],xmm15
   5b489:	movsd  xmm15,QWORD PTR [rsp+0x2c8]
   5b493:	divsd  xmm15,xmm5
   5b498:	mulsd  xmm3,xmm3
   5b49c:	divsd  xmm12,xmm5
   5b4a1:	movsd  QWORD PTR [rsp+0x40],xmm15
   5b4a8:	movsd  xmm15,QWORD PTR [rsp+0x2d0]
   5b4b2:	divsd  xmm15,xmm5
   5b4b7:	mulsd  xmm12,xmm12
   5b4bc:	addsd  xmm3,xmm12
   5b4c1:	addsd  xmm3,xmm4
   5b4c5:	divsd  xmm2,xmm5
   5b4c9:	movsd  QWORD PTR [rsp+0x48],xmm15
   5b4d0:	movsd  xmm15,QWORD PTR [rsp+0x2d8]
   5b4da:	divsd  xmm15,xmm5
   5b4df:	mulsd  xmm2,xmm2
   5b4e3:	divsd  xmm11,xmm5
   5b4e8:	movsd  QWORD PTR [rsp+0x50],xmm15
   5b4ef:	movsd  xmm15,QWORD PTR [rsp+0x2e0]
   5b4f9:	divsd  xmm15,xmm5
   5b4fe:	mulsd  xmm11,xmm11
   5b503:	addsd  xmm2,xmm11
   5b508:	addsd  xmm2,xmm3
   5b50c:	divsd  xmm1,xmm5
   5b510:	movsd  QWORD PTR [rsp+0x58],xmm15
   5b517:	movsd  xmm15,QWORD PTR [rsp+0x2e8]
   5b521:	divsd  xmm15,xmm5
   5b526:	mulsd  xmm1,xmm1
   5b52a:	divsd  xmm8,xmm5
   5b52f:	movsd  QWORD PTR [rsp+0x60],xmm15
   5b536:	movsd  xmm15,QWORD PTR [rsp+0x2f0]
   5b540:	divsd  xmm15,xmm5
   5b545:	mulsd  xmm8,xmm8
   5b54a:	addsd  xmm1,xmm8
   5b54f:	addsd  xmm1,xmm2
   5b553:	movsd  xmm2,QWORD PTR [rsp+0x8]
   5b559:	mulsd  xmm2,xmm2
   5b55d:	divsd  xmm0,xmm5
   5b561:	movsd  QWORD PTR [rsp+0x68],xmm15
   5b568:	movsd  xmm15,QWORD PTR [rsp+0x2f8]
   5b572:	divsd  xmm15,xmm5
   5b577:	mulsd  xmm0,xmm0
   5b57b:	divsd  xmm7,xmm5
   5b57f:	movsd  QWORD PTR [rsp+0x70],xmm15
   5b586:	movsd  xmm15,QWORD PTR [rsp+0x300]
   5b590:	divsd  xmm15,xmm5
   5b595:	mulsd  xmm7,xmm7
   5b599:	addsd  xmm0,xmm7
   5b59d:	addsd  xmm0,xmm1
   5b5a1:	movsd  xmm1,QWORD PTR [rsp]
   5b5a6:	mulsd  xmm1,xmm1
   5b5aa:	addsd  xmm1,xmm2
   5b5ae:	movsd  xmm2,QWORD PTR [rsp+0x28]
   5b5b4:	mulsd  xmm2,xmm2
   5b5b8:	addsd  xmm1,xmm0
   5b5bc:	movsd  xmm0,QWORD PTR [rsp+0x18]
   5b5c2:	movsd  QWORD PTR [rsp+0x78],xmm15
   5b5c9:	mulsd  xmm0,xmm0
   5b5cd:	movsd  xmm15,QWORD PTR [rsp+0x308]
   5b5d7:	divsd  xmm15,xmm5
   5b5dc:	addsd  xmm0,xmm2
   5b5e0:	movsd  xmm2,QWORD PTR [rsp+0x40]
   5b5e6:	mulsd  xmm2,xmm2
   5b5ea:	addsd  xmm0,xmm1
   5b5ee:	movsd  xmm1,QWORD PTR [rsp+0x38]
   5b5f4:	mulsd  xmm1,xmm1
   5b5f8:	addsd  xmm1,xmm2
   5b5fc:	movsd  xmm2,QWORD PTR [rsp+0x50]
   5b602:	mulsd  xmm2,xmm2
   5b606:	addsd  xmm1,xmm0
   5b60a:	movsd  xmm0,QWORD PTR [rsp+0x48]
   5b610:	mulsd  xmm0,xmm0
   5b614:	addsd  xmm0,xmm2
   5b618:	movsd  xmm2,QWORD PTR [rsp+0x60]
   5b61e:	mulsd  xmm2,xmm2
   5b622:	addsd  xmm0,xmm1
   5b626:	movsd  xmm1,QWORD PTR [rsp+0x58]
   5b62c:	movsd  QWORD PTR [rsp+0x80],xmm15
   5b636:	mulsd  xmm1,xmm1
   5b63a:	movsd  xmm15,QWORD PTR [rsp+0x310]
   5b644:	divsd  xmm15,xmm5
   5b649:	addsd  xmm1,xmm2
   5b64d:	movsd  xmm2,QWORD PTR [rsp+0x70]
   5b653:	mulsd  xmm2,xmm2
   5b657:	addsd  xmm0,xmm1
   5b65b:	movsd  xmm1,QWORD PTR [rsp+0x68]
   5b661:	mulsd  xmm1,xmm1
   5b665:	addsd  xmm1,xmm2
   5b669:	addsd  xmm0,xmm1
   5b66d:	movsd  xmm1,QWORD PTR [rsp+0x78]
   5b673:	mulsd  xmm1,xmm1
   5b677:	movsd  QWORD PTR [rsp+0x88],xmm15
   5b681:	movsd  xmm15,QWORD PTR [rsp+0x318]
   5b68b:	divsd  xmm15,xmm5
   5b690:	movsd  QWORD PTR [rsp+0xa0],xmm15
   5b69a:	movsd  xmm15,QWORD PTR [rsp+0x320]
   5b6a4:	divsd  xmm15,xmm5
   5b6a9:	movsd  QWORD PTR [rsp+0x90],xmm15
   5b6b3:	movsd  xmm15,QWORD PTR [rsp+0x328]
   5b6bd:	divsd  xmm15,xmm5
   5b6c2:	movsd  QWORD PTR [rsp+0xa8],xmm15
   5b6cc:	movsd  xmm15,QWORD PTR [rsp+0x330]
   5b6d6:	divsd  xmm15,xmm5
   5b6db:	movsd  QWORD PTR [rsp+0xc0],xmm15
   5b6e5:	movsd  xmm15,QWORD PTR [rsp+0x338]
   5b6ef:	movsd  xmm2,QWORD PTR [rsp+0x80]
   5b6f8:	mulsd  xmm2,xmm2
   5b6fc:	divsd  xmm15,xmm5
   5b701:	addsd  xmm1,xmm2
   5b705:	movsd  xmm2,QWORD PTR [rsp+0xa0]
   5b70e:	mulsd  xmm2,xmm2
   5b712:	addsd  xmm0,xmm1
   5b716:	movsd  xmm1,QWORD PTR [rsp+0x88]
   5b71f:	mulsd  xmm1,xmm1
   5b723:	addsd  xmm1,xmm2
   5b727:	movsd  xmm2,QWORD PTR [rsp+0xa8]
   5b730:	mulsd  xmm2,xmm2
   5b734:	addsd  xmm0,xmm1
   5b738:	movsd  xmm1,QWORD PTR [rsp+0x90]
   5b741:	mulsd  xmm1,xmm1
   5b745:	addsd  xmm1,xmm2
   5b749:	addsd  xmm0,xmm1
   5b74d:	movsd  xmm1,QWORD PTR [rsp+0xc0]
   5b756:	mulsd  xmm15,xmm15
   5b75b:	mulsd  xmm1,xmm1
   5b75f:	addsd  xmm1,xmm15
   5b764:	addsd  xmm0,xmm1
   5b768:	ucomisd xmm0,xmm14
   5b76d:	jp     5b776 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x776>
   5b76f:	movapd xmm1,xmm14
   5b774:	je     5b797 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x797>
   5b776:	movsd  xmm1,QWORD PTR [rip+0x80e2]        # 63860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   5b77e:	movapd xmm6,xmm1
   5b782:	maxsd  xmm6,xmm0
   5b786:	movapd xmm0,xmm6
   5b78a:	call   6ad0 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5b78f:	movapd xmm1,xmm0
   5b793:	mulsd  xmm1,xmm6
   5b797:	mulsd  xmm5,xmm1
   5b79b:	movapd xmm0,xmm5
   5b79f:	mov    rax,QWORD PTR [rsp+0x10]
   5b7a4:	unpcklpd xmm0,xmm0
   5b7a8:	movapd xmm1,XMMWORD PTR [rax]
   5b7ac:	add    rax,0x10
   5b7b0:	divpd  xmm1,xmm0
   5b7b4:	movaps XMMWORD PTR [rax-0x10],xmm1
   5b7b8:	cmp    r12,rax
   5b7bb:	jne    5b7a8 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x7a8>
   5b7bd:	mov    QWORD PTR [rsp+0x1968],0x0
   5b7c9:	lea    rax,[rsp+0x1970]
   5b7d1:	pxor   xmm0,xmm0
   5b7d5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5b7e0:	movaps XMMWORD PTR [rax],xmm0
   5b7e3:	add    rax,0x20
   5b7e7:	movaps XMMWORD PTR [rax-0x10],xmm0
   5b7eb:	cmp    r11,rax
   5b7ee:	jne    5b7e0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x7e0>
   5b7f0:	lea    rdx,[r15+0x18b0]
   5b7f7:	mov    rax,r11
   5b7fa:	pxor   xmm0,xmm0
   5b7fe:	mov    QWORD PTR [rsp+0x38],rdx
   5b803:	xchg   ax,ax
   5b805:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5b810:	movaps XMMWORD PTR [rax],xmm0
   5b813:	add    rax,0x20
   5b817:	movaps XMMWORD PTR [rax-0x10],xmm0
   5b81b:	cmp    rdx,rax
   5b81e:	jne    5b810 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x810>
   5b820:	mov    rsi,QWORD PTR [rsp+0x38]
   5b825:	lea    rdx,[r15+0x19c0]
   5b82c:	pxor   xmm0,xmm0
   5b830:	mov    QWORD PTR [rsp+0x40],rdx
   5b835:	sub    rdx,rsi
   5b838:	mov    rax,rsi
   5b83b:	and    edx,0x10
   5b83e:	je     5b858 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x858>
   5b840:	mov    rax,QWORD PTR [rsp+0x38]
   5b845:	movaps XMMWORD PTR [r15+0x18b0],xmm0
   5b84d:	add    rax,0x10
   5b851:	cmp    QWORD PTR [rsp+0x40],rax
   5b856:	je     5b86a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x86a>
   5b858:	movaps XMMWORD PTR [rax],xmm0
   5b85b:	add    rax,0x20
   5b85f:	movaps XMMWORD PTR [rax-0x10],xmm0
   5b863:	cmp    QWORD PTR [rsp+0x40],rax
   5b868:	jne    5b858 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x858>
   5b86a:	mov    rsi,QWORD PTR [rsp+0x40]
   5b86f:	lea    rdx,[r15+0x1ad0]
   5b876:	pxor   xmm0,xmm0
   5b87a:	mov    QWORD PTR [rsp+0x48],rdx
   5b87f:	sub    rdx,rsi
   5b882:	mov    rax,rsi
   5b885:	and    edx,0x10
   5b888:	je     5b8a2 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x8a2>
   5b88a:	mov    rax,QWORD PTR [rsp+0x40]
   5b88f:	movaps XMMWORD PTR [r15+0x19c0],xmm0
   5b897:	add    rax,0x10
   5b89b:	cmp    QWORD PTR [rsp+0x48],rax
   5b8a0:	je     5b8b4 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x8b4>
   5b8a2:	movaps XMMWORD PTR [rax],xmm0
   5b8a5:	add    rax,0x20
   5b8a9:	movaps XMMWORD PTR [rax-0x10],xmm0
   5b8ad:	cmp    QWORD PTR [rsp+0x48],rax
   5b8b2:	jne    5b8a2 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x8a2>
   5b8b4:	mov    rsi,QWORD PTR [rsp+0x48]
   5b8b9:	lea    rdx,[r15+0x1bd0]
   5b8c0:	pxor   xmm0,xmm0
   5b8c4:	mov    QWORD PTR [rsp+0x158],rdx
   5b8cc:	sub    rdx,rsi
   5b8cf:	mov    rax,rsi
   5b8d2:	and    edx,0x10
   5b8d5:	je     5b8f2 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x8f2>
   5b8d7:	mov    rax,QWORD PTR [rsp+0x48]
   5b8dc:	movaps XMMWORD PTR [r15+0x1ad0],xmm0
   5b8e4:	add    rax,0x10
   5b8e8:	cmp    QWORD PTR [rsp+0x158],rax
   5b8f0:	je     5b907 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x907>
   5b8f2:	movaps XMMWORD PTR [rax],xmm0
   5b8f5:	add    rax,0x20
   5b8f9:	movaps XMMWORD PTR [rax-0x10],xmm0
   5b8fd:	cmp    QWORD PTR [rsp+0x158],rax
   5b905:	jne    5b8f2 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x8f2>
   5b907:	mov    rdi,QWORD PTR [rsp+0x1c8]
   5b90f:	xor    eax,eax
   5b911:	mov    ecx,0x10
   5b916:	pxor   xmm0,xmm0
   5b91a:	rep stos QWORD PTR [rdi],rax
   5b91d:	lea    rax,[r15+0x1d50]
   5b924:	mov    QWORD PTR [rsp+0x1a0],rax
   5b92c:	lea    rax,[r15+0x1c50]
   5b933:	movaps XMMWORD PTR [rax],xmm0
   5b936:	add    rax,0x20
   5b93a:	movaps XMMWORD PTR [rax-0x10],xmm0
   5b93e:	cmp    QWORD PTR [rsp+0x1a0],rax
   5b946:	jne    5b933 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x933>
   5b948:	movdqu xmm5,XMMWORD PTR [r13+0x10]
   5b94e:	pxor   xmm1,xmm1
   5b952:	mov    r8,QWORD PTR [r13+0x20]
   5b956:	movsd  xmm6,QWORD PTR [rsp+0xb8]
   5b95f:	movdqa xmm0,xmm5
   5b963:	mov    QWORD PTR [rsp+0x1950],r8
   5b96b:	movsd  QWORD PTR [rsp+0x1958],xmm6
   5b974:	shufpd xmm0,xmm5,0x1
   5b979:	movaps XMMWORD PTR [rsp+0x1940],xmm0
   5b981:	movsd  xmm0,QWORD PTR [r13+0x38]
   5b987:	comisd xmm0,xmm1
   5b98b:	ja     5b9b2 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x9b2>
   5b98d:	movsd  xmm6,QWORD PTR [rip+0x801b]        # 639b0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x230>
   5b995:	movsd  xmm1,QWORD PTR [rip+0x800b]        # 639a8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x228>
   5b99d:	movapd xmm0,xmm6
   5b9a1:	call   6ad0 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5b9a6:	mulsd  xmm0,xmm6
   5b9aa:	mulsd  xmm0,QWORD PTR [rip+0x804e]        # 63a00 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x280>
   5b9b2:	movsd  QWORD PTR [rsp+0x1960],xmm0
   5b9bb:	mov    rsi,QWORD PTR [r13+0x8]
   5b9bf:	test   rsi,rsi
   5b9c2:	je     5ef50 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3f50>
   5b9c8:	cmp    r8,0x2
   5b9cc:	movapd xmm6,xmm10
   5b9d1:	mov    BYTE PTR [rsp+0xd8],0x0
   5b9d9:	movsd  xmm14,QWORD PTR [rsp+0x240]
   5b9e3:	sbb    rax,rax
   5b9e6:	unpcklpd xmm6,xmm6
   5b9ea:	mov    QWORD PTR [rsp+0x68],r11
   5b9ef:	movsd  xmm7,QWORD PTR [rsp+0x248]
   5b9f8:	add    rax,0x2
   5b9fc:	movaps XMMWORD PTR [rsp+0x160],xmm6
   5ba04:	movapd xmm6,xmm9
   5ba09:	xor    r14d,r14d
   5ba0c:	mov    QWORD PTR [rsp+0x188],r10
   5ba14:	unpcklpd xmm6,xmm6
   5ba18:	mov    QWORD PTR [rsp+0x70],r12
   5ba1d:	mov    QWORD PTR [rsp+0x180],rax
   5ba25:	mov    rax,r13
   5ba28:	mov    r13,rbp
   5ba2b:	mov    QWORD PTR [rsp+0x8],r15
   5ba30:	mov    r15,rsi
   5ba33:	mov    rsi,rbx
   5ba36:	mov    rbp,rax
   5ba39:	mov    BYTE PTR [rsp+0x78],r9b
   5ba3e:	mov    ebx,r9d
   5ba41:	mov    r9,rsi
   5ba44:	movq   QWORD PTR [rsp+0x1b0],xmm5
   5ba4d:	movhps QWORD PTR [rsp+0xc8],xmm5
   5ba55:	movaps XMMWORD PTR [rsp+0x170],xmm6
   5ba5d:	cmp    BYTE PTR [rsp+0x78],0x0
   5ba62:	je     5efac <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3fac>
   5ba68:	lea    rdx,[rsp+0x440]
   5ba70:	lea    r8,[rsp+0x340]
   5ba78:	pxor   xmm0,xmm0
   5ba7c:	mov    QWORD PTR [rsp+0x60],rdx
   5ba81:	mov    rax,r8
   5ba84:	movaps XMMWORD PTR [rax],xmm0
   5ba87:	add    rax,0x20
   5ba8b:	movaps XMMWORD PTR [rax-0x10],xmm0
   5ba8f:	cmp    rdx,rax
   5ba92:	jne    5ba84 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xa84>
   5ba94:	pxor   xmm6,xmm6
   5ba98:	movsd  xmm2,QWORD PTR [rip+0x7d50]        # 637f0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x70>
   5baa0:	movsd  xmm1,QWORD PTR [rsp+0x1958]
   5baa9:	lea    rax,[rsp+0x218]
   5bab1:	movapd xmm0,xmm6
   5bab5:	mov    rdi,QWORD PTR [rsp+0x10]
   5baba:	mov    rdx,rax
   5babd:	mov    esi,0x10
   5bac2:	cmpnltsd xmm0,xmm1
   5bac7:	comisd xmm6,xmm1
   5bacb:	mov    QWORD PTR [rsp+0x3698],0x0
   5bad7:	mov    BYTE PTR [rsp+0x36a1],0x0
   5badf:	mov    QWORD PTR [rsp+0x1c0],rax
   5bae7:	andpd  xmm2,xmm0
   5baeb:	andnpd xmm0,xmm1
   5baef:	movapd xmm1,xmm6
   5baf3:	setae  BYTE PTR [rsp+0x36a2]
   5bafb:	movsd  QWORD PTR [rsp+0x218],xmm6
   5bb04:	orpd   xmm0,xmm2
   5bb08:	movsd  QWORD PTR [rsp+0xc0],xmm0
   5bb11:	movsd  xmm0,QWORD PTR [rsp+0x1960]
   5bb1a:	cmpltsd xmm1,xmm0
   5bb1f:	andpd  xmm0,xmm1
   5bb23:	movapd xmm1,xmm6
   5bb27:	movsd  QWORD PTR [rsp+0x18],xmm0
   5bb2d:	movsd  xmm0,QWORD PTR [rsp+0x1968]
   5bb36:	cmpltsd xmm1,xmm0
   5bb3b:	andpd  xmm0,xmm1
   5bb3f:	movsd  QWORD PTR [rsp+0x28],xmm0
   5bb45:	call   fc90 <_ZN4Base6Matrix9Numerical10VectorNorm9ScaleLoopINS0_7ComplexIdEELm16ELm0EE7computeERKNS0_6VectorIS5_Lm16EEEmRd.part.0.constprop.0>
   5bb4a:	movsd  xmm5,QWORD PTR [rsp+0x218]
   5bb53:	ucomisd xmm5,xmm6
   5bb57:	jp     5bb63 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xb63>
   5bb59:	movapd xmm0,xmm6
   5bb5d:	je     5bd33 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xd33>
   5bb63:	divsd  xmm14,xmm5
   5bb68:	mov    rdx,QWORD PTR [rsp+0x20]
   5bb6d:	movsd  xmm13,QWORD PTR [rsp+0x2a8]
   5bb77:	mov    esi,0x10
   5bb7c:	movsd  xmm1,QWORD PTR [rsp+0x250]
   5bb85:	mov    rdi,QWORD PTR [rsp+0x10]
   5bb8a:	movsd  xmm12,QWORD PTR [rsp+0x258]
   5bb94:	movsd  xmm0,QWORD PTR [rsp+0x260]
   5bb9d:	movsd  xmm11,QWORD PTR [rsp+0x268]
   5bba7:	movsd  xmm6,QWORD PTR [rsp+0x270]
   5bbb0:	movsd  xmm10,QWORD PTR [rsp+0x278]
   5bbba:	movsd  xmm4,QWORD PTR [rsp+0x280]
   5bbc3:	movsd  xmm9,QWORD PTR [rsp+0x288]
   5bbcd:	movsd  xmm3,QWORD PTR [rsp+0x290]
   5bbd6:	movsd  xmm8,QWORD PTR [rsp+0x298]
   5bbe0:	movsd  xmm2,QWORD PTR [rsp+0x2a0]
   5bbe9:	divsd  xmm13,xmm5
   5bbee:	mulsd  xmm14,xmm14
   5bbf3:	divsd  xmm7,xmm5
   5bbf7:	movsd  QWORD PTR [rsp],xmm13
   5bbfd:	movsd  xmm13,QWORD PTR [rsp+0x2b0]
   5bc07:	divsd  xmm13,xmm5
   5bc0c:	mulsd  xmm7,xmm7
   5bc10:	addsd  xmm14,xmm7
   5bc15:	pxor   xmm7,xmm7
   5bc19:	addsd  xmm14,xmm7
   5bc1e:	divsd  xmm1,xmm5
   5bc22:	movapd xmm15,xmm13
   5bc27:	movsd  xmm13,QWORD PTR [rsp+0x2b8]
   5bc31:	divsd  xmm12,xmm5
   5bc36:	mulsd  xmm1,xmm1
   5bc3a:	divsd  xmm0,xmm5
   5bc3e:	mulsd  xmm12,xmm12
   5bc43:	addsd  xmm1,xmm12
   5bc48:	addsd  xmm1,xmm14
   5bc4d:	divsd  xmm11,xmm5
   5bc52:	mulsd  xmm0,xmm0
   5bc56:	divsd  xmm6,xmm5
   5bc5a:	mulsd  xmm11,xmm11
   5bc5f:	addsd  xmm0,xmm11
   5bc64:	addsd  xmm0,xmm1
   5bc68:	movsd  xmm1,QWORD PTR [rsp]
   5bc6d:	mulsd  xmm1,xmm1
   5bc71:	divsd  xmm10,xmm5
   5bc76:	mulsd  xmm6,xmm6
   5bc7a:	divsd  xmm4,xmm5
   5bc7e:	mulsd  xmm10,xmm10
   5bc83:	addsd  xmm6,xmm10
   5bc88:	addsd  xmm0,xmm6
   5bc8c:	divsd  xmm9,xmm5
   5bc91:	mulsd  xmm4,xmm4
   5bc95:	divsd  xmm3,xmm5
   5bc99:	mulsd  xmm9,xmm9
   5bc9e:	addsd  xmm4,xmm9
   5bca3:	addsd  xmm0,xmm4
   5bca7:	divsd  xmm8,xmm5
   5bcac:	mulsd  xmm3,xmm3
   5bcb0:	divsd  xmm2,xmm5
   5bcb4:	mulsd  xmm8,xmm8
   5bcb9:	addsd  xmm3,xmm8
   5bcbe:	addsd  xmm0,xmm3
   5bcc2:	divsd  xmm13,xmm5
   5bcc7:	mulsd  xmm2,xmm2
   5bccb:	addsd  xmm1,xmm2
   5bccf:	addsd  xmm0,xmm1
   5bcd3:	movapd xmm1,xmm15
   5bcd8:	mulsd  xmm1,xmm15
   5bcdd:	mulsd  xmm13,xmm13
   5bce2:	addsd  xmm1,xmm13
   5bce7:	addsd  xmm0,xmm1
   5bceb:	movsd  QWORD PTR [rsp+0x230],xmm0
   5bcf4:	movapd xmm0,xmm5
   5bcf8:	call   c140 <_ZN4Base6Matrix9Numerical10VectorNorm14SquaredSumLoopINS0_7ComplexIdEELm16ELm8EE7computeERKNS0_6VectorIS5_Lm16EEEmdRd.part.0>
   5bcfd:	movsd  xmm2,QWORD PTR [rsp+0x230]
   5bd06:	ucomisd xmm2,xmm7
   5bd0a:	jp     5bd12 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xd12>
   5bd0c:	movapd xmm0,xmm7
   5bd10:	je     5bd2f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xd2f>
   5bd12:	movsd  xmm1,QWORD PTR [rip+0x7b46]        # 63860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   5bd1a:	movapd xmm6,xmm1
   5bd1e:	maxsd  xmm6,xmm2
   5bd22:	movapd xmm0,xmm6
   5bd26:	call   6ad0 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5bd2b:	mulsd  xmm0,xmm6
   5bd2f:	mulsd  xmm0,xmm5
   5bd33:	mulsd  xmm0,QWORD PTR [rsp+0x18]
   5bd39:	mov    rdx,QWORD PTR [rsp+0x60]
   5bd3e:	addsd  xmm0,QWORD PTR [rsp+0x28]
   5bd44:	mov    r12,rdx
   5bd47:	mov    rax,rdx
   5bd4a:	movsd  QWORD PTR [rsp+0xe0],xmm0
   5bd53:	movapd xmm0,XMMWORD PTR [r8]
   5bd58:	add    r8,0x10
   5bd5c:	add    rax,0x10
   5bd60:	movaps XMMWORD PTR [rax-0x10],xmm0
   5bd64:	cmp    rdx,r8
   5bd67:	jne    5bd53 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xd53>
   5bd69:	lea    rax,[rsp+0x540]
   5bd71:	lea    r10,[rsp+0x640]
   5bd79:	pxor   xmm0,xmm0
   5bd7d:	mov    QWORD PTR [rsp+0x50],rax
   5bd82:	movaps XMMWORD PTR [rax],xmm0
   5bd85:	add    rax,0x20
   5bd89:	movaps XMMWORD PTR [rax-0x10],xmm0
   5bd8d:	cmp    r10,rax
   5bd90:	jne    5bd82 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xd82>
   5bd92:	mov    rdx,QWORD PTR [rsp+0x50]
   5bd97:	mov    rsi,QWORD PTR [rsp+0x60]
   5bd9c:	movaps XMMWORD PTR [rsp+0x540],xmm0
   5bda4:	mov    rdi,QWORD PTR [rsp+0x30]
   5bda9:	movaps XMMWORD PTR [rsp+0x550],xmm0
   5bdb1:	movaps XMMWORD PTR [rsp+0x560],xmm0
   5bdb9:	movaps XMMWORD PTR [rsp+0x570],xmm0
   5bdc1:	movaps XMMWORD PTR [rsp+0x580],xmm0
   5bdc9:	movaps XMMWORD PTR [rsp+0x590],xmm0
   5bdd1:	movaps XMMWORD PTR [rsp+0x5a0],xmm0
   5bdd9:	movaps XMMWORD PTR [rsp+0x5b0],xmm0
   5bde1:	movaps XMMWORD PTR [rsp+0x5c0],xmm0
   5bde9:	movaps XMMWORD PTR [rsp+0x5d0],xmm0
   5bdf1:	movaps XMMWORD PTR [rsp+0x5e0],xmm0
   5bdf9:	movaps XMMWORD PTR [rsp+0x5f0],xmm0
   5be01:	movaps XMMWORD PTR [rsp+0x600],xmm0
   5be09:	movaps XMMWORD PTR [rsp+0x610],xmm0
   5be11:	movaps XMMWORD PTR [rsp+0x620],xmm0
   5be19:	movaps XMMWORD PTR [rsp+0x630],xmm0
   5be21:	call   1c0c0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeINS0_7ComplexIdEELm16EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS7_XT0_EEERSC_m.part.0>
   5be26:	mov    rax,QWORD PTR [rsp+0x10]
   5be2b:	mov    rcx,QWORD PTR [rsp+0x70]
   5be30:	movapd xmm0,XMMWORD PTR [rax]
   5be34:	subpd  xmm0,XMMWORD PTR [rdx]
   5be38:	add    rax,0x10
   5be3c:	add    rdx,0x10
   5be40:	movaps XMMWORD PTR [rdx-0x10],xmm0
   5be44:	cmp    rcx,rax
   5be47:	jne    5be30 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xe30>
   5be49:	mov    rdx,QWORD PTR [rsp+0x20]
   5be4e:	mov    rdi,QWORD PTR [rsp+0x50]
   5be53:	mov    esi,0x10
   5be58:	mov    QWORD PTR [rsp+0x230],0x0
   5be64:	call   fc90 <_ZN4Base6Matrix9Numerical10VectorNorm9ScaleLoopINS0_7ComplexIdEELm16ELm0EE7computeERKNS0_6VectorIS5_Lm16EEEmRd.part.0.constprop.0>
   5be69:	pxor   xmm0,xmm0
   5be6d:	movsd  xmm5,QWORD PTR [rsp+0x230]
   5be76:	ucomisd xmm5,xmm0
   5be7a:	jp     5be82 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xe82>
   5be7c:	je     5c2d3 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x12d3>
   5be82:	movsd  xmm14,QWORD PTR [rsp+0x5a8]
   5be8c:	movsd  xmm7,QWORD PTR [rsp+0x540]
   5be95:	movsd  xmm13,QWORD PTR [rsp+0x548]
   5be9f:	movsd  xmm6,QWORD PTR [rsp+0x550]
   5bea8:	movsd  xmm12,QWORD PTR [rsp+0x558]
   5beb2:	divsd  xmm14,xmm5
   5beb7:	movsd  xmm4,QWORD PTR [rsp+0x560]
   5bec0:	movsd  xmm11,QWORD PTR [rsp+0x568]
   5beca:	movsd  xmm3,QWORD PTR [rsp+0x570]
   5bed3:	movsd  xmm10,QWORD PTR [rsp+0x578]
   5bedd:	movsd  xmm2,QWORD PTR [rsp+0x580]
   5bee6:	movsd  xmm9,QWORD PTR [rsp+0x588]
   5bef0:	movsd  xmm1,QWORD PTR [rsp+0x590]
   5bef9:	movsd  xmm8,QWORD PTR [rsp+0x598]
   5bf03:	movsd  xmm0,QWORD PTR [rsp+0x5a0]
   5bf0c:	divsd  xmm7,xmm5
   5bf10:	movsd  QWORD PTR [rsp],xmm14
   5bf16:	movsd  xmm14,QWORD PTR [rsp+0x5b0]
   5bf20:	divsd  xmm14,xmm5
   5bf25:	mulsd  xmm7,xmm7
   5bf29:	divsd  xmm13,xmm5
   5bf2e:	movsd  QWORD PTR [rsp+0x18],xmm14
   5bf35:	movsd  xmm14,QWORD PTR [rsp+0x5b8]
   5bf3f:	divsd  xmm14,xmm5
   5bf44:	mulsd  xmm13,xmm13
   5bf49:	addsd  xmm7,xmm13
   5bf4e:	pxor   xmm13,xmm13
   5bf53:	addsd  xmm7,xmm13
   5bf58:	divsd  xmm6,xmm5
   5bf5c:	movsd  QWORD PTR [rsp+0x28],xmm14
   5bf63:	movsd  xmm14,QWORD PTR [rsp+0x5c0]
   5bf6d:	divsd  xmm14,xmm5
   5bf72:	mulsd  xmm6,xmm6
   5bf76:	divsd  xmm12,xmm5
   5bf7b:	movsd  QWORD PTR [rsp+0x58],xmm14
   5bf82:	movsd  xmm14,QWORD PTR [rsp+0x5c8]
   5bf8c:	divsd  xmm14,xmm5
   5bf91:	mulsd  xmm12,xmm12
   5bf96:	addsd  xmm6,xmm12
   5bf9b:	addsd  xmm6,xmm7
   5bf9f:	divsd  xmm4,xmm5
   5bfa3:	movsd  QWORD PTR [rsp+0x80],xmm14
   5bfad:	movsd  xmm14,QWORD PTR [rsp+0x5d0]
   5bfb7:	divsd  xmm14,xmm5
   5bfbc:	mulsd  xmm4,xmm4
   5bfc0:	divsd  xmm11,xmm5
   5bfc5:	movsd  QWORD PTR [rsp+0x88],xmm14
   5bfcf:	movsd  xmm14,QWORD PTR [rsp+0x5d8]
   5bfd9:	divsd  xmm14,xmm5
   5bfde:	mulsd  xmm11,xmm11
   5bfe3:	addsd  xmm4,xmm11
   5bfe8:	addsd  xmm4,xmm6
   5bfec:	divsd  xmm3,xmm5
   5bff0:	movsd  QWORD PTR [rsp+0xa0],xmm14
   5bffa:	movsd  xmm14,QWORD PTR [rsp+0x5e0]
   5c004:	divsd  xmm14,xmm5
   5c009:	mulsd  xmm3,xmm3
   5c00d:	divsd  xmm10,xmm5
   5c012:	movsd  QWORD PTR [rsp+0x90],xmm14
   5c01c:	movsd  xmm14,QWORD PTR [rsp+0x5e8]
   5c026:	divsd  xmm14,xmm5
   5c02b:	mulsd  xmm10,xmm10
   5c030:	addsd  xmm3,xmm10
   5c035:	addsd  xmm3,xmm4
   5c039:	divsd  xmm2,xmm5
   5c03d:	movsd  QWORD PTR [rsp+0xa8],xmm14
   5c047:	movsd  xmm14,QWORD PTR [rsp+0x5f0]
   5c051:	divsd  xmm14,xmm5
   5c056:	mulsd  xmm2,xmm2
   5c05a:	divsd  xmm9,xmm5
   5c05f:	movsd  QWORD PTR [rsp+0xe8],xmm14
   5c069:	movsd  xmm14,QWORD PTR [rsp+0x5f8]
   5c073:	divsd  xmm14,xmm5
   5c078:	mulsd  xmm9,xmm9
   5c07d:	addsd  xmm2,xmm9
   5c082:	addsd  xmm2,xmm3
   5c086:	divsd  xmm1,xmm5
   5c08a:	movsd  QWORD PTR [rsp+0xf0],xmm14
   5c094:	movsd  xmm14,QWORD PTR [rsp+0x600]
   5c09e:	divsd  xmm14,xmm5
   5c0a3:	mulsd  xmm1,xmm1
   5c0a7:	divsd  xmm8,xmm5
   5c0ac:	movsd  QWORD PTR [rsp+0xf8],xmm14
   5c0b6:	movsd  xmm14,QWORD PTR [rsp+0x608]
   5c0c0:	divsd  xmm14,xmm5
   5c0c5:	mulsd  xmm8,xmm8
   5c0ca:	addsd  xmm1,xmm8
   5c0cf:	addsd  xmm1,xmm2
   5c0d3:	movsd  xmm2,QWORD PTR [rsp]
   5c0d8:	mulsd  xmm2,xmm2
   5c0dc:	divsd  xmm0,xmm5
   5c0e0:	movsd  QWORD PTR [rsp+0x100],xmm14
   5c0ea:	movsd  xmm14,QWORD PTR [rsp+0x610]
   5c0f4:	divsd  xmm14,xmm5
   5c0f9:	mulsd  xmm0,xmm0
   5c0fd:	addsd  xmm0,xmm2
   5c101:	movsd  xmm2,QWORD PTR [rsp+0x28]
   5c107:	mulsd  xmm2,xmm2
   5c10b:	addsd  xmm0,xmm1
   5c10f:	movsd  xmm1,QWORD PTR [rsp+0x18]
   5c115:	mulsd  xmm1,xmm1
   5c119:	addsd  xmm1,xmm2
   5c11d:	movsd  xmm2,QWORD PTR [rsp+0x80]
   5c126:	mulsd  xmm2,xmm2
   5c12a:	addsd  xmm1,xmm0
   5c12e:	movsd  xmm0,QWORD PTR [rsp+0x58]
   5c134:	movsd  QWORD PTR [rsp+0x108],xmm14
   5c13e:	mulsd  xmm0,xmm0
   5c142:	movsd  xmm14,QWORD PTR [rsp+0x618]
   5c14c:	divsd  xmm14,xmm5
   5c151:	addsd  xmm0,xmm2
   5c155:	movsd  xmm2,QWORD PTR [rsp+0xa0]
   5c15e:	mulsd  xmm2,xmm2
   5c162:	addsd  xmm0,xmm1
   5c166:	movsd  xmm1,QWORD PTR [rsp+0x88]
   5c16f:	mulsd  xmm1,xmm1
   5c173:	addsd  xmm1,xmm2
   5c177:	movsd  xmm2,QWORD PTR [rsp+0xa8]
   5c180:	mulsd  xmm2,xmm2
   5c184:	addsd  xmm0,xmm1
   5c188:	movsd  xmm1,QWORD PTR [rsp+0x90]
   5c191:	mulsd  xmm1,xmm1
   5c195:	addsd  xmm1,xmm2
   5c199:	movsd  xmm2,QWORD PTR [rsp+0xf0]
   5c1a2:	mulsd  xmm2,xmm2
   5c1a6:	addsd  xmm0,xmm1
   5c1aa:	movsd  xmm1,QWORD PTR [rsp+0xe8]
   5c1b3:	movsd  QWORD PTR [rsp+0x110],xmm14
   5c1bd:	mulsd  xmm1,xmm1
   5c1c1:	movsd  xmm14,QWORD PTR [rsp+0x620]
   5c1cb:	divsd  xmm14,xmm5
   5c1d0:	addsd  xmm1,xmm2
   5c1d4:	movsd  xmm2,QWORD PTR [rsp+0x100]
   5c1dd:	mulsd  xmm2,xmm2
   5c1e1:	addsd  xmm0,xmm1
   5c1e5:	movsd  xmm1,QWORD PTR [rsp+0xf8]
   5c1ee:	mulsd  xmm1,xmm1
   5c1f2:	addsd  xmm1,xmm2
   5c1f6:	movsd  xmm2,QWORD PTR [rsp+0x110]
   5c1ff:	mulsd  xmm2,xmm2
   5c203:	addsd  xmm0,xmm1
   5c207:	movsd  xmm1,QWORD PTR [rsp+0x108]
   5c210:	mulsd  xmm1,xmm1
   5c214:	addsd  xmm1,xmm2
   5c218:	addsd  xmm0,xmm1
   5c21c:	movsd  QWORD PTR [rsp+0x118],xmm14
   5c226:	movsd  xmm14,QWORD PTR [rsp+0x628]
   5c230:	movsd  xmm1,QWORD PTR [rsp+0x118]
   5c239:	divsd  xmm14,xmm5
   5c23e:	mulsd  xmm1,xmm1
   5c242:	movsd  QWORD PTR [rsp+0x120],xmm14
   5c24c:	movsd  xmm14,QWORD PTR [rsp+0x630]
   5c256:	divsd  xmm14,xmm5
   5c25b:	movapd xmm15,xmm14
   5c260:	movsd  xmm14,QWORD PTR [rsp+0x638]
   5c26a:	movsd  xmm2,QWORD PTR [rsp+0x120]
   5c273:	divsd  xmm14,xmm5
   5c278:	mulsd  xmm2,xmm2
   5c27c:	addsd  xmm1,xmm2
   5c280:	addsd  xmm0,xmm1
   5c284:	movapd xmm1,xmm15
   5c289:	mulsd  xmm1,xmm15
   5c28e:	mulsd  xmm14,xmm14
   5c293:	addsd  xmm1,xmm14
   5c298:	addsd  xmm0,xmm1
   5c29c:	ucomisd xmm0,xmm13
   5c2a1:	jp     5c2aa <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x12aa>
   5c2a3:	movapd xmm1,xmm13
   5c2a8:	je     5c2cb <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x12cb>
   5c2aa:	movsd  xmm1,QWORD PTR [rip+0x75ae]        # 63860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   5c2b2:	movapd xmm6,xmm1
   5c2b6:	maxsd  xmm6,xmm0
   5c2ba:	movapd xmm0,xmm6
   5c2be:	call   6ad0 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5c2c3:	movapd xmm1,xmm0
   5c2c7:	mulsd  xmm1,xmm6
   5c2cb:	mulsd  xmm5,xmm1
   5c2cf:	movapd xmm0,xmm5
   5c2d3:	movsd  xmm6,QWORD PTR [rsp+0xe0]
   5c2dc:	movsd  QWORD PTR [rsp+0x3690],xmm0
   5c2e5:	comisd xmm6,xmm0
   5c2e9:	setae  al
   5c2ec:	mov    BYTE PTR [rsp+0x36a0],al
   5c2f3:	cmp    QWORD PTR [rsp+0xc8],0x0
   5c2fc:	je     5d8d4 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x28d4>
   5c302:	mov    rcx,QWORD PTR [rsp+0x1b0]
   5c30a:	mov    esi,0x10
   5c30f:	mov    edx,0x1
   5c314:	mov    BYTE PTR [rsp+0x1d7],bl
   5c31b:	mov    QWORD PTR [rsp+0x1d8],r14
   5c323:	mov    r11,r10
   5c326:	cmp    rcx,0x10
   5c32a:	mov    QWORD PTR [rsp+0x1e0],r12
   5c332:	cmovbe rsi,rcx
   5c336:	mov    QWORD PTR [rsp+0x1e8],r15
   5c33e:	mov    QWORD PTR [rsp+0x1f0],r9
   5c346:	test   rsi,rsi
   5c349:	mov    QWORD PTR [rsp+0x1f8],rbp
   5c351:	cmovne rdx,rsi
   5c355:	xor    ebp,ebp
   5c357:	mov    QWORD PTR [rsp+0x18],rdx
   5c35c:	test   al,al
   5c35e:	jne    5d57b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x257b>
   5c364:	cmp    BYTE PTR [rsp+0x36a1],0x0
   5c36c:	jne    5d57b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x257b>
   5c372:	mov    rbx,QWORD PTR [rsp+0x40]
   5c377:	mov    rdx,QWORD PTR [rsp+0x48]
   5c37c:	pxor   xmm0,xmm0
   5c380:	movsd  xmm7,QWORD PTR [rsp+0x3690]
   5c389:	sub    rdx,rbx
   5c38c:	mov    rax,rbx
   5c38f:	movapd xmm2,xmm7
   5c393:	and    edx,0x10
   5c396:	unpcklpd xmm2,xmm2
   5c39a:	je     5c3ac <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x13ac>
   5c39c:	mov    rax,QWORD PTR [rsp+0x40]
   5c3a1:	add    rax,0x10
   5c3a5:	cmp    QWORD PTR [rsp+0x48],rax
   5c3aa:	je     5c3c1 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x13c1>
   5c3ac:	mov    rdx,QWORD PTR [rsp+0x48]
   5c3b1:	movaps XMMWORD PTR [rax],xmm0
   5c3b4:	add    rax,0x20
   5c3b8:	movaps XMMWORD PTR [rax-0x10],xmm0
   5c3bc:	cmp    rdx,rax
   5c3bf:	jne    5c3b1 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x13b1>
   5c3c1:	comisd xmm7,QWORD PTR [rip+0x762f]        # 639f8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x278>
   5c3c9:	pxor   xmm1,xmm1
   5c3cd:	mov    QWORD PTR [rsp+0x3308],0x0
   5c3d9:	movsd  QWORD PTR [rsp+0x3300],xmm7
   5c3e2:	jb     5c3f0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x13f0>
   5c3e4:	movsd  xmm1,QWORD PTR [rip+0x743c]        # 63828 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
   5c3ec:	divsd  xmm1,xmm7
   5c3f0:	mov    rax,QWORD PTR [rsp+0x50]
   5c3f5:	unpcklpd xmm1,xmm1
   5c3f9:	lea    rdx,[rsp+0x1970]
   5c401:	jmp    5c418 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1418>
   5c403:	mulpd  xmm0,xmm1
   5c407:	add    rax,0x10
   5c40b:	add    rdx,0x10
   5c40f:	movaps XMMWORD PTR [rdx-0x10],xmm0
   5c413:	cmp    r11,rax
   5c416:	je     5c43b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x143b>
   5c418:	comisd xmm7,QWORD PTR [rip+0x75d8]        # 639f8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x278>
   5c420:	movapd xmm0,XMMWORD PTR [rax]
   5c424:	jae    5c403 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1403>
   5c426:	divpd  xmm0,xmm2
   5c42a:	add    rax,0x10
   5c42e:	add    rdx,0x10
   5c432:	movaps XMMWORD PTR [rdx-0x10],xmm0
   5c436:	cmp    r11,rax
   5c439:	jne    5c418 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1418>
   5c43b:	lea    r12,[rsp+0x740]
   5c443:	mov    rax,r11
   5c446:	pxor   xmm0,xmm0
   5c44a:	movaps XMMWORD PTR [rax],xmm0
   5c44d:	add    rax,0x20
   5c451:	movaps XMMWORD PTR [rax-0x10],xmm0
   5c455:	cmp    r12,rax
   5c458:	jne    5c44a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x144a>
   5c45a:	mov    rax,r12
   5c45d:	pxor   xmm0,xmm0
   5c461:	movaps XMMWORD PTR [rax],xmm0
   5c464:	add    rax,0x20
   5c468:	movaps XMMWORD PTR [rax-0x10],xmm0
   5c46c:	cmp    r13,rax
   5c46f:	jne    5c461 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1461>
   5c471:	mov    QWORD PTR [rsp+0x58],r11
   5c476:	mov    r14,QWORD PTR [rsp+0x38]
   5c47b:	xor    edi,edi
   5c47d:	xor    r8d,r8d
   5c480:	mov    QWORD PTR [rsp+0x208],rbp
   5c488:	movzx  edx,BYTE PTR [rsp+0x78]
   5c48d:	xor    r15d,r15d
   5c490:	lea    rax,[rsp+0x1970]
   5c498:	mov    rbx,QWORD PTR [rsp+0x158]
   5c4a0:	mov    rbp,QWORD PTR [rsp+0x40]
   5c4a5:	movsd  QWORD PTR [rsp+0x200],xmm7
   5c4ae:	jmp    5c4d7 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x14d7>
   5c4b0:	nop    DWORD PTR [rax+rax*1+0x0]
   5c4b5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5c4c0:	mov    r14,rsi
   5c4c3:	add    rbx,0x8
   5c4c7:	add    rax,0x100
   5c4cd:	cmp    r15,0x10
   5c4d1:	je     5d5a9 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x25a9>
   5c4d7:	cmp    r15,QWORD PTR [rsp+0x18]
   5c4dc:	mov    QWORD PTR [rsp],r15
   5c4e0:	lea    rsi,[r14+0x10]
   5c4e4:	setb   cl
   5c4e7:	add    r15,0x1
   5c4eb:	and    cl,dl
   5c4ed:	mov    BYTE PTR [rsp+0x28],cl
   5c4f1:	je     5c4c0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x14c0>
   5c4f3:	mov    rdx,QWORD PTR [rsp+0x58]
   5c4f8:	mov    rcx,rax
   5c4fb:	pxor   xmm3,xmm3
   5c4ff:	nop
   5c500:	movapd xmm1,XMMWORD PTR [rcx]
   5c504:	add    rdx,0x10
   5c508:	add    rcx,0x10
   5c50c:	movapd xmm0,xmm1
   5c510:	movapd xmm2,xmm1
   5c514:	shufpd xmm0,xmm1,0x1
   5c519:	mulpd  xmm0,xmm3
   5c51d:	subpd  xmm2,xmm0
   5c521:	addpd  xmm0,xmm1
   5c525:	movsd  xmm0,xmm2
   5c529:	movaps XMMWORD PTR [rdx-0x10],xmm0
   5c52d:	cmp    rdx,r12
   5c530:	jne    5c500 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1500>
   5c532:	mov    rsi,QWORD PTR [rsp+0x58]
   5c537:	mov    rdi,QWORD PTR [rsp+0x30]
   5c53c:	mov    rdx,r12
   5c53f:	mov    QWORD PTR [rsp+0x80],rax
   5c547:	movaps XMMWORD PTR [rsp+0x740],xmm3
   5c54f:	movaps XMMWORD PTR [rsp+0x750],xmm3
   5c557:	movaps XMMWORD PTR [rsp+0x760],xmm3
   5c55f:	movaps XMMWORD PTR [rsp+0x770],xmm3
   5c567:	movaps XMMWORD PTR [rsp+0x780],xmm3
   5c56f:	movaps XMMWORD PTR [rsp+0x790],xmm3
   5c577:	movaps XMMWORD PTR [rsp+0x7a0],xmm3
   5c57f:	movaps XMMWORD PTR [rsp+0x7b0],xmm3
   5c587:	movaps XMMWORD PTR [rsp+0x7c0],xmm3
   5c58f:	movaps XMMWORD PTR [rsp+0x7d0],xmm3
   5c597:	movaps XMMWORD PTR [rsp+0x7e0],xmm3
   5c59f:	movaps XMMWORD PTR [rsp+0x7f0],xmm3
   5c5a7:	movaps XMMWORD PTR [rsp+0x800],xmm3
   5c5af:	movaps XMMWORD PTR [rsp+0x810],xmm3
   5c5b7:	movaps XMMWORD PTR [rsp+0x820],xmm3
   5c5bf:	movaps XMMWORD PTR [rsp+0x830],xmm3
   5c5c7:	call   1c0c0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeINS0_7ComplexIdEELm16EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS7_XT0_EEERSC_m.part.0>
   5c5cc:	mov    rdx,QWORD PTR [rsp+0x20]
   5c5d1:	mov    esi,0x10
   5c5d6:	mov    rdi,r12
   5c5d9:	mov    QWORD PTR [rsp+0x230],0x0
   5c5e5:	call   fc90 <_ZN4Base6Matrix9Numerical10VectorNorm9ScaleLoopINS0_7ComplexIdEELm16ELm0EE7computeERKNS0_6VectorIS5_Lm16EEEmRd.part.0.constprop.0>
   5c5ea:	pxor   xmm0,xmm0
   5c5ee:	movsd  xmm5,QWORD PTR [rsp+0x230]
   5c5f7:	mov    rax,QWORD PTR [rsp+0x80]
   5c5ff:	ucomisd xmm5,xmm0
   5c603:	jp     5c60f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x160f>
   5c605:	movapd xmm6,xmm0
   5c609:	je     5ca8f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1a8f>
   5c60f:	movsd  xmm14,QWORD PTR [rsp+0x7a8]
   5c619:	movsd  xmm7,QWORD PTR [rsp+0x740]
   5c622:	movsd  xmm13,QWORD PTR [rsp+0x748]
   5c62c:	movsd  xmm6,QWORD PTR [rsp+0x750]
   5c635:	movsd  xmm12,QWORD PTR [rsp+0x758]
   5c63f:	divsd  xmm14,xmm5
   5c644:	movsd  xmm4,QWORD PTR [rsp+0x760]
   5c64d:	movsd  xmm11,QWORD PTR [rsp+0x768]
   5c657:	movsd  xmm3,QWORD PTR [rsp+0x770]
   5c660:	movsd  xmm10,QWORD PTR [rsp+0x778]
   5c66a:	movsd  xmm2,QWORD PTR [rsp+0x780]
   5c673:	movsd  xmm9,QWORD PTR [rsp+0x788]
   5c67d:	movsd  xmm1,QWORD PTR [rsp+0x790]
   5c686:	movsd  xmm8,QWORD PTR [rsp+0x798]
   5c690:	movsd  xmm0,QWORD PTR [rsp+0x7a0]
   5c699:	divsd  xmm7,xmm5
   5c69d:	movsd  QWORD PTR [rsp+0x80],xmm14
   5c6a7:	movsd  xmm14,QWORD PTR [rsp+0x7b0]
   5c6b1:	divsd  xmm14,xmm5
   5c6b6:	mulsd  xmm7,xmm7
   5c6ba:	divsd  xmm13,xmm5
   5c6bf:	movsd  QWORD PTR [rsp+0x88],xmm14
   5c6c9:	movsd  xmm14,QWORD PTR [rsp+0x7b8]
   5c6d3:	divsd  xmm14,xmm5
   5c6d8:	mulsd  xmm13,xmm13
   5c6dd:	addsd  xmm7,xmm13
   5c6e2:	pxor   xmm13,xmm13
   5c6e7:	addsd  xmm7,xmm13
   5c6ec:	divsd  xmm6,xmm5
   5c6f0:	movsd  QWORD PTR [rsp+0xa0],xmm14
   5c6fa:	movsd  xmm14,QWORD PTR [rsp+0x7c0]
   5c704:	divsd  xmm14,xmm5
   5c709:	mulsd  xmm6,xmm6
   5c70d:	divsd  xmm12,xmm5
   5c712:	movsd  QWORD PTR [rsp+0x90],xmm14
   5c71c:	movsd  xmm14,QWORD PTR [rsp+0x7c8]
   5c726:	divsd  xmm14,xmm5
   5c72b:	mulsd  xmm12,xmm12
   5c730:	addsd  xmm6,xmm12
   5c735:	addsd  xmm6,xmm7
   5c739:	divsd  xmm4,xmm5
   5c73d:	movsd  QWORD PTR [rsp+0xa8],xmm14
   5c747:	movsd  xmm14,QWORD PTR [rsp+0x7d0]
   5c751:	divsd  xmm14,xmm5
   5c756:	mulsd  xmm4,xmm4
   5c75a:	divsd  xmm11,xmm5
   5c75f:	movsd  QWORD PTR [rsp+0xe8],xmm14
   5c769:	movsd  xmm14,QWORD PTR [rsp+0x7d8]
   5c773:	divsd  xmm14,xmm5
   5c778:	mulsd  xmm11,xmm11
   5c77d:	addsd  xmm4,xmm11
   5c782:	addsd  xmm4,xmm6
   5c786:	divsd  xmm3,xmm5
   5c78a:	movsd  QWORD PTR [rsp+0xf0],xmm14
   5c794:	movsd  xmm14,QWORD PTR [rsp+0x7e0]
   5c79e:	divsd  xmm14,xmm5
   5c7a3:	mulsd  xmm3,xmm3
   5c7a7:	divsd  xmm10,xmm5
   5c7ac:	movsd  QWORD PTR [rsp+0xf8],xmm14
   5c7b6:	movsd  xmm14,QWORD PTR [rsp+0x7e8]
   5c7c0:	divsd  xmm14,xmm5
   5c7c5:	mulsd  xmm10,xmm10
   5c7ca:	addsd  xmm3,xmm10
   5c7cf:	addsd  xmm3,xmm4
   5c7d3:	divsd  xmm2,xmm5
   5c7d7:	movsd  QWORD PTR [rsp+0x100],xmm14
   5c7e1:	movsd  xmm14,QWORD PTR [rsp+0x7f0]
   5c7eb:	divsd  xmm14,xmm5
   5c7f0:	mulsd  xmm2,xmm2
   5c7f4:	divsd  xmm9,xmm5
   5c7f9:	movsd  QWORD PTR [rsp+0x108],xmm14
   5c803:	movsd  xmm14,QWORD PTR [rsp+0x7f8]
   5c80d:	divsd  xmm14,xmm5
   5c812:	mulsd  xmm9,xmm9
   5c817:	addsd  xmm2,xmm9
   5c81c:	addsd  xmm2,xmm3
   5c820:	divsd  xmm1,xmm5
   5c824:	movsd  QWORD PTR [rsp+0x110],xmm14
   5c82e:	movsd  xmm14,QWORD PTR [rsp+0x800]
   5c838:	divsd  xmm14,xmm5
   5c83d:	mulsd  xmm1,xmm1
   5c841:	divsd  xmm8,xmm5
   5c846:	movsd  QWORD PTR [rsp+0x118],xmm14
   5c850:	movsd  xmm14,QWORD PTR [rsp+0x808]
   5c85a:	divsd  xmm14,xmm5
   5c85f:	mulsd  xmm8,xmm8
   5c864:	addsd  xmm1,xmm8
   5c869:	addsd  xmm1,xmm2
   5c86d:	movsd  xmm2,QWORD PTR [rsp+0x80]
   5c876:	mulsd  xmm2,xmm2
   5c87a:	divsd  xmm0,xmm5
   5c87e:	movsd  QWORD PTR [rsp+0x120],xmm14
   5c888:	movsd  xmm14,QWORD PTR [rsp+0x810]
   5c892:	divsd  xmm14,xmm5
   5c897:	mulsd  xmm0,xmm0
   5c89b:	addsd  xmm0,xmm2
   5c89f:	movsd  xmm2,QWORD PTR [rsp+0xa0]
   5c8a8:	mulsd  xmm2,xmm2
   5c8ac:	addsd  xmm0,xmm1
   5c8b0:	movsd  xmm1,QWORD PTR [rsp+0x88]
   5c8b9:	mulsd  xmm1,xmm1
   5c8bd:	addsd  xmm1,xmm2
   5c8c1:	movsd  xmm2,QWORD PTR [rsp+0xa8]
   5c8ca:	mulsd  xmm2,xmm2
   5c8ce:	addsd  xmm1,xmm0
   5c8d2:	movsd  xmm0,QWORD PTR [rsp+0x90]
   5c8db:	movsd  QWORD PTR [rsp+0x128],xmm14
   5c8e5:	mulsd  xmm0,xmm0
   5c8e9:	movsd  xmm14,QWORD PTR [rsp+0x818]
   5c8f3:	divsd  xmm14,xmm5
   5c8f8:	addsd  xmm0,xmm2
   5c8fc:	movsd  xmm2,QWORD PTR [rsp+0xf0]
   5c905:	mulsd  xmm2,xmm2
   5c909:	addsd  xmm0,xmm1
   5c90d:	movsd  xmm1,QWORD PTR [rsp+0xe8]
   5c916:	mulsd  xmm1,xmm1
   5c91a:	addsd  xmm1,xmm2
   5c91e:	movsd  xmm2,QWORD PTR [rsp+0x100]
   5c927:	mulsd  xmm2,xmm2
   5c92b:	addsd  xmm0,xmm1
   5c92f:	movsd  xmm1,QWORD PTR [rsp+0xf8]
   5c938:	mulsd  xmm1,xmm1
   5c93c:	addsd  xmm1,xmm2
   5c940:	movsd  xmm2,QWORD PTR [rsp+0x110]
   5c949:	mulsd  xmm2,xmm2
   5c94d:	addsd  xmm0,xmm1
   5c951:	movsd  xmm1,QWORD PTR [rsp+0x108]
   5c95a:	movsd  QWORD PTR [rsp+0x130],xmm14
   5c964:	mulsd  xmm1,xmm1
   5c968:	movsd  xmm14,QWORD PTR [rsp+0x820]
   5c972:	divsd  xmm14,xmm5
   5c977:	addsd  xmm1,xmm2
   5c97b:	movsd  xmm2,QWORD PTR [rsp+0x120]
   5c984:	mulsd  xmm2,xmm2
   5c988:	addsd  xmm0,xmm1
   5c98c:	movsd  xmm1,QWORD PTR [rsp+0x118]
   5c995:	mulsd  xmm1,xmm1
   5c999:	addsd  xmm1,xmm2
   5c99d:	movsd  xmm2,QWORD PTR [rsp+0x130]
   5c9a6:	mulsd  xmm2,xmm2
   5c9aa:	addsd  xmm0,xmm1
   5c9ae:	movsd  xmm1,QWORD PTR [rsp+0x128]
   5c9b7:	mulsd  xmm1,xmm1
   5c9bb:	addsd  xmm1,xmm2
   5c9bf:	addsd  xmm0,xmm1
   5c9c3:	movsd  QWORD PTR [rsp+0x138],xmm14
   5c9cd:	movsd  xmm14,QWORD PTR [rsp+0x828]
   5c9d7:	divsd  xmm14,xmm5
   5c9dc:	movsd  QWORD PTR [rsp+0x140],xmm14
   5c9e6:	movsd  xmm14,QWORD PTR [rsp+0x830]
   5c9f0:	divsd  xmm14,xmm5
   5c9f5:	movsd  QWORD PTR [rsp+0x148],xmm14
   5c9ff:	movsd  xmm14,QWORD PTR [rsp+0x838]
   5ca09:	movsd  xmm1,QWORD PTR [rsp+0x138]
   5ca12:	movsd  xmm2,QWORD PTR [rsp+0x140]
   5ca1b:	divsd  xmm14,xmm5
   5ca20:	mulsd  xmm1,xmm1
   5ca24:	mulsd  xmm2,xmm2
   5ca28:	addsd  xmm1,xmm2
   5ca2c:	addsd  xmm0,xmm1
   5ca30:	movsd  xmm1,QWORD PTR [rsp+0x148]
   5ca39:	mulsd  xmm1,xmm1
   5ca3d:	mulsd  xmm14,xmm14
   5ca42:	addsd  xmm14,xmm1
   5ca47:	addsd  xmm0,xmm14
   5ca4c:	ucomisd xmm0,xmm13
   5ca51:	jp     5ca5a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1a5a>
   5ca53:	movapd xmm6,xmm13
   5ca58:	je     5ca8b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1a8b>
   5ca5a:	movsd  xmm1,QWORD PTR [rip+0x6dfe]        # 63860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   5ca62:	mov    QWORD PTR [rsp+0x80],rax
   5ca6a:	movapd xmm7,xmm1
   5ca6e:	maxsd  xmm7,xmm0
   5ca72:	movapd xmm0,xmm7
   5ca76:	call   6ad0 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5ca7b:	mov    rax,QWORD PTR [rsp+0x80]
   5ca83:	movapd xmm6,xmm0
   5ca87:	mulsd  xmm6,xmm7
   5ca8b:	mulsd  xmm6,xmm5
   5ca8f:	mov    rsi,QWORD PTR [rsp+0x38]
   5ca94:	mov    rcx,rbp
   5ca97:	pxor   xmm0,xmm0
   5ca9b:	sub    rcx,rsi
   5ca9e:	mov    rdx,rsi
   5caa1:	and    ecx,0x10
   5caa4:	je     5cac0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1ac0>
   5caa6:	mov    rsi,QWORD PTR [rsp+0x38]
   5caab:	lea    rdx,[rsi+0x10]
   5caaf:	movaps XMMWORD PTR [rsi],xmm0
   5cab2:	cmp    rdx,rbp
   5cab5:	je     5cad0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1ad0>
   5cab7:	nop    WORD PTR [rax+rax*1+0x0]
   5cac0:	movaps XMMWORD PTR [rdx],xmm0
   5cac3:	add    rdx,0x20
   5cac7:	movaps XMMWORD PTR [rdx-0x10],xmm0
   5cacb:	cmp    rdx,rbp
   5cace:	jne    5cac0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1ac0>
   5cad0:	mov    rsi,QWORD PTR [rsp+0x8]
   5cad5:	xor    ecx,ecx
   5cad7:	mov    QWORD PTR [rsp+0x80],rbp
   5cadf:	lea    r10,[rax+0x200]
   5cae6:	lea    r11,[rsi+0x130]
   5caed:	mov    rsi,QWORD PTR [rsp+0x180]
   5caf5:	mov    r8,QWORD PTR [rsp+0x8]
   5cafa:	mov    rdi,r11
   5cafd:	lea    r9,[rsp+0x1970]
   5cb05:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5cb10:	pxor   xmm2,xmm2
   5cb14:	mov    rdx,r9
   5cb17:	mov    rbp,r12
   5cb1a:	movapd xmm8,xmm2
   5cb1f:	movapd xmm7,xmm2
   5cb23:	xchg   ax,ax
   5cb25:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5cb30:	movapd xmm5,XMMWORD PTR [rdx]
   5cb34:	movapd xmm0,XMMWORD PTR [rdx+0x10]
   5cb39:	add    rdx,0x20
   5cb3d:	add    rbp,0x20
   5cb41:	movlpd xmm0,QWORD PTR [rdx-0x18]
   5cb46:	movapd xmm10,XMMWORD PTR [rbp-0x20]
   5cb4c:	movhpd xmm5,QWORD PTR [rdx-0x10]
   5cb51:	movapd xmm3,xmm0
   5cb55:	movapd xmm1,xmm5
   5cb59:	movhpd xmm10,QWORD PTR [rbp-0x10]
   5cb5f:	mulpd  xmm1,xmm5
   5cb63:	movapd xmm4,xmm5
   5cb67:	mulpd  xmm3,xmm0
   5cb6b:	mulpd  xmm4,xmm10
   5cb70:	addpd  xmm3,xmm1
   5cb74:	movapd xmm1,XMMWORD PTR [rip+0x6d44]        # 638c0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x140>
   5cb7c:	xorpd  xmm1,xmm0
   5cb80:	movapd xmm0,XMMWORD PTR [rbp-0x10]
   5cb85:	movlpd xmm0,QWORD PTR [rbp-0x18]
   5cb8a:	movapd xmm9,xmm1
   5cb8f:	addsd  xmm7,xmm3
   5cb93:	unpckhpd xmm3,xmm3
   5cb97:	mulpd  xmm9,xmm0
   5cb9c:	mulpd  xmm1,xmm10
   5cba1:	mulpd  xmm0,xmm5
   5cba5:	movapd xmm5,xmm8
   5cbaa:	addsd  xmm7,xmm3
   5cbae:	subpd  xmm4,xmm9
   5cbb3:	addpd  xmm0,xmm1
   5cbb7:	addsd  xmm5,xmm4
   5cbbb:	unpckhpd xmm4,xmm4
   5cbbf:	movapd xmm1,xmm0
   5cbc3:	unpckhpd xmm0,xmm0
   5cbc7:	addsd  xmm1,xmm2
   5cbcb:	addsd  xmm5,xmm4
   5cbcf:	movapd xmm2,xmm1
   5cbd3:	movapd xmm8,xmm5
   5cbd8:	addsd  xmm2,xmm0
   5cbdc:	cmp    rdx,rdi
   5cbdf:	jne    5cb30 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1b30>
   5cbe5:	divsd  xmm8,xmm7
   5cbea:	mov    rbp,r9
   5cbed:	mov    rdx,r12
   5cbf0:	movsd  xmm0,QWORD PTR [r8+0x18b0]
   5cbf9:	divsd  xmm2,xmm7
   5cbfd:	addsd  xmm0,xmm8
   5cc02:	movapd xmm4,xmm8
   5cc07:	movsd  QWORD PTR [r8+0x18b0],xmm0
   5cc10:	movsd  xmm0,QWORD PTR [r8+0x18b8]
   5cc19:	addsd  xmm0,xmm2
   5cc1d:	unpcklpd xmm4,xmm2
   5cc21:	unpcklpd xmm2,xmm8
   5cc26:	movsd  QWORD PTR [r8+0x18b8],xmm0
   5cc2f:	nop
   5cc30:	movapd xmm1,XMMWORD PTR [rbp+0x0]
   5cc35:	movapd xmm0,XMMWORD PTR [rbp+0x0]
   5cc3a:	add    rdx,0x10
   5cc3e:	add    rbp,0x10
   5cc42:	unpcklpd xmm1,xmm1
   5cc46:	unpckhpd xmm0,xmm0
   5cc4a:	mulpd  xmm1,xmm4
   5cc4e:	mulpd  xmm0,xmm2
   5cc52:	movapd xmm3,xmm1
   5cc56:	subpd  xmm3,xmm0
   5cc5a:	addpd  xmm0,xmm1
   5cc5e:	movapd xmm1,XMMWORD PTR [rdx-0x10]
   5cc63:	movsd  xmm0,xmm3
   5cc67:	subpd  xmm1,xmm0
   5cc6b:	movaps XMMWORD PTR [rdx-0x10],xmm1
   5cc6f:	cmp    r13,rdx
   5cc72:	jne    5cc30 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1c30>
   5cc74:	add    rdi,0x100
   5cc7b:	add    r9,0x100
   5cc82:	add    r8,0x10
   5cc86:	cmp    r10,rdi
   5cc89:	jne    5cb10 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1b10>
   5cc8f:	lea    rdx,[rcx+0x1]
   5cc93:	mov    ecx,0x1
   5cc98:	cmp    rdx,rsi
   5cc9b:	jb     5caf5 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1af5>
   5cca1:	mov    rdx,QWORD PTR [rsp+0x20]
   5cca6:	mov    esi,0x10
   5ccab:	mov    rdi,r12
   5ccae:	mov    rbp,QWORD PTR [rsp+0x80]
   5ccb6:	mov    QWORD PTR [rsp+0x230],0x0
   5ccc2:	mov    QWORD PTR [rsp+0x80],rax
   5ccca:	call   fc90 <_ZN4Base6Matrix9Numerical10VectorNorm9ScaleLoopINS0_7ComplexIdEELm16ELm0EE7computeERKNS0_6VectorIS5_Lm16EEEmRd.part.0.constprop.0>
   5cccf:	pxor   xmm0,xmm0
   5ccd3:	movsd  xmm5,QWORD PTR [rsp+0x230]
   5ccdc:	mov    rax,QWORD PTR [rsp+0x80]
   5cce4:	ucomisd xmm5,xmm0
   5cce8:	jp     5ccf4 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1cf4>
   5ccea:	movapd xmm1,xmm0
   5ccee:	je     5d185 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2185>
   5ccf4:	movsd  xmm14,QWORD PTR [rsp+0x7a0]
   5ccfe:	movsd  xmm7,QWORD PTR [rsp+0x740]
   5cd07:	movsd  xmm13,QWORD PTR [rsp+0x748]
   5cd11:	movsd  xmm4,QWORD PTR [rsp+0x750]
   5cd1a:	movsd  xmm12,QWORD PTR [rsp+0x758]
   5cd24:	divsd  xmm14,xmm5
   5cd29:	movsd  xmm3,QWORD PTR [rsp+0x760]
   5cd32:	movsd  xmm11,QWORD PTR [rsp+0x768]
   5cd3c:	movsd  xmm2,QWORD PTR [rsp+0x770]
   5cd45:	movsd  xmm10,QWORD PTR [rsp+0x778]
   5cd4f:	movsd  xmm1,QWORD PTR [rsp+0x780]
   5cd58:	movsd  xmm9,QWORD PTR [rsp+0x788]
   5cd62:	movsd  xmm0,QWORD PTR [rsp+0x790]
   5cd6b:	movsd  xmm8,QWORD PTR [rsp+0x798]
   5cd75:	divsd  xmm7,xmm5
   5cd79:	movsd  QWORD PTR [rsp+0x80],xmm14
   5cd83:	movsd  xmm14,QWORD PTR [rsp+0x7a8]
   5cd8d:	divsd  xmm14,xmm5
   5cd92:	mulsd  xmm7,xmm7
   5cd96:	divsd  xmm13,xmm5
   5cd9b:	movsd  QWORD PTR [rsp+0x88],xmm14
   5cda5:	movsd  xmm14,QWORD PTR [rsp+0x7b0]
   5cdaf:	divsd  xmm14,xmm5
   5cdb4:	mulsd  xmm13,xmm13
   5cdb9:	addsd  xmm7,xmm13
   5cdbe:	pxor   xmm13,xmm13
   5cdc3:	addsd  xmm7,xmm13
   5cdc8:	divsd  xmm4,xmm5
   5cdcc:	movsd  QWORD PTR [rsp+0xa0],xmm14
   5cdd6:	movsd  xmm14,QWORD PTR [rsp+0x7b8]
   5cde0:	divsd  xmm14,xmm5
   5cde5:	mulsd  xmm4,xmm4
   5cde9:	divsd  xmm12,xmm5
   5cdee:	movsd  QWORD PTR [rsp+0x90],xmm14
   5cdf8:	movsd  xmm14,QWORD PTR [rsp+0x7c0]
   5ce02:	divsd  xmm14,xmm5
   5ce07:	mulsd  xmm12,xmm12
   5ce0c:	addsd  xmm4,xmm12
   5ce11:	addsd  xmm4,xmm7
   5ce15:	divsd  xmm3,xmm5
   5ce19:	movsd  QWORD PTR [rsp+0xa8],xmm14
   5ce23:	movsd  xmm14,QWORD PTR [rsp+0x7c8]
   5ce2d:	divsd  xmm14,xmm5
   5ce32:	mulsd  xmm3,xmm3
   5ce36:	divsd  xmm11,xmm5
   5ce3b:	movsd  QWORD PTR [rsp+0xe8],xmm14
   5ce45:	movsd  xmm14,QWORD PTR [rsp+0x7d0]
   5ce4f:	divsd  xmm14,xmm5
   5ce54:	mulsd  xmm11,xmm11
   5ce59:	addsd  xmm3,xmm11
   5ce5e:	addsd  xmm3,xmm4
   5ce62:	divsd  xmm2,xmm5
   5ce66:	movsd  QWORD PTR [rsp+0xf0],xmm14
   5ce70:	movsd  xmm14,QWORD PTR [rsp+0x7d8]
   5ce7a:	divsd  xmm14,xmm5
   5ce7f:	mulsd  xmm2,xmm2
   5ce83:	divsd  xmm10,xmm5
   5ce88:	movsd  QWORD PTR [rsp+0xf8],xmm14
   5ce92:	movsd  xmm14,QWORD PTR [rsp+0x7e0]
   5ce9c:	divsd  xmm14,xmm5
   5cea1:	mulsd  xmm10,xmm10
   5cea6:	addsd  xmm2,xmm10
   5ceab:	addsd  xmm2,xmm3
   5ceaf:	divsd  xmm1,xmm5
   5ceb3:	movsd  QWORD PTR [rsp+0x100],xmm14
   5cebd:	movsd  xmm14,QWORD PTR [rsp+0x7e8]
   5cec7:	divsd  xmm14,xmm5
   5cecc:	mulsd  xmm1,xmm1
   5ced0:	divsd  xmm9,xmm5
   5ced5:	movsd  QWORD PTR [rsp+0x108],xmm14
   5cedf:	movsd  xmm14,QWORD PTR [rsp+0x7f0]
   5cee9:	divsd  xmm14,xmm5
   5ceee:	mulsd  xmm9,xmm9
   5cef3:	addsd  xmm1,xmm9
   5cef8:	addsd  xmm1,xmm2
   5cefc:	movsd  xmm2,QWORD PTR [rsp+0x88]
   5cf05:	mulsd  xmm2,xmm2
   5cf09:	divsd  xmm0,xmm5
   5cf0d:	movsd  QWORD PTR [rsp+0x110],xmm14
   5cf17:	movsd  xmm14,QWORD PTR [rsp+0x7f8]
   5cf21:	divsd  xmm14,xmm5
   5cf26:	mulsd  xmm0,xmm0
   5cf2a:	divsd  xmm8,xmm5
   5cf2f:	movsd  QWORD PTR [rsp+0x118],xmm14
   5cf39:	movsd  xmm14,QWORD PTR [rsp+0x800]
   5cf43:	divsd  xmm14,xmm5
   5cf48:	mulsd  xmm8,xmm8
   5cf4d:	addsd  xmm0,xmm8
   5cf52:	addsd  xmm0,xmm1
   5cf56:	movsd  xmm1,QWORD PTR [rsp+0x80]
   5cf5f:	mulsd  xmm1,xmm1
   5cf63:	addsd  xmm1,xmm2
   5cf67:	movsd  xmm2,QWORD PTR [rsp+0x90]
   5cf70:	mulsd  xmm2,xmm2
   5cf74:	addsd  xmm1,xmm0
   5cf78:	movsd  xmm0,QWORD PTR [rsp+0xa0]
   5cf81:	movsd  QWORD PTR [rsp+0x120],xmm14
   5cf8b:	mulsd  xmm0,xmm0
   5cf8f:	movsd  xmm14,QWORD PTR [rsp+0x808]
   5cf99:	divsd  xmm14,xmm5
   5cf9e:	addsd  xmm0,xmm2
   5cfa2:	movsd  xmm2,QWORD PTR [rsp+0xe8]
   5cfab:	mulsd  xmm2,xmm2
   5cfaf:	addsd  xmm0,xmm1
   5cfb3:	movsd  xmm1,QWORD PTR [rsp+0xa8]
   5cfbc:	mulsd  xmm1,xmm1
   5cfc0:	addsd  xmm1,xmm2
   5cfc4:	movsd  xmm2,QWORD PTR [rsp+0xf8]
   5cfcd:	mulsd  xmm2,xmm2
   5cfd1:	addsd  xmm1,xmm0
   5cfd5:	movsd  xmm0,QWORD PTR [rsp+0xf0]
   5cfde:	mulsd  xmm0,xmm0
   5cfe2:	addsd  xmm0,xmm2
   5cfe6:	movsd  xmm2,QWORD PTR [rsp+0x108]
   5cfef:	mulsd  xmm2,xmm2
   5cff3:	addsd  xmm0,xmm1
   5cff7:	movsd  xmm1,QWORD PTR [rsp+0x100]
   5d000:	movsd  QWORD PTR [rsp+0x128],xmm14
   5d00a:	mulsd  xmm1,xmm1
   5d00e:	movsd  xmm14,QWORD PTR [rsp+0x810]
   5d018:	divsd  xmm14,xmm5
   5d01d:	addsd  xmm1,xmm2
   5d021:	movsd  xmm2,QWORD PTR [rsp+0x118]
   5d02a:	mulsd  xmm2,xmm2
   5d02e:	addsd  xmm0,xmm1
   5d032:	movsd  xmm1,QWORD PTR [rsp+0x110]
   5d03b:	mulsd  xmm1,xmm1
   5d03f:	addsd  xmm1,xmm2
   5d043:	movsd  xmm2,QWORD PTR [rsp+0x128]
   5d04c:	mulsd  xmm2,xmm2
   5d050:	addsd  xmm0,xmm1
   5d054:	movsd  xmm1,QWORD PTR [rsp+0x120]
   5d05d:	mulsd  xmm1,xmm1
   5d061:	addsd  xmm1,xmm2
   5d065:	addsd  xmm0,xmm1
   5d069:	movsd  QWORD PTR [rsp+0x130],xmm14
   5d073:	movsd  xmm14,QWORD PTR [rsp+0x818]
   5d07d:	divsd  xmm14,xmm5
   5d082:	movsd  QWORD PTR [rsp+0x138],xmm14
   5d08c:	movsd  xmm14,QWORD PTR [rsp+0x820]
   5d096:	divsd  xmm14,xmm5
   5d09b:	movsd  QWORD PTR [rsp+0x140],xmm14
   5d0a5:	movsd  xmm14,QWORD PTR [rsp+0x828]
   5d0af:	divsd  xmm14,xmm5
   5d0b4:	movsd  QWORD PTR [rsp+0x148],xmm14
   5d0be:	movsd  xmm14,QWORD PTR [rsp+0x830]
   5d0c8:	divsd  xmm14,xmm5
   5d0cd:	movapd xmm15,xmm14
   5d0d2:	movsd  xmm14,QWORD PTR [rsp+0x838]
   5d0dc:	movsd  xmm1,QWORD PTR [rsp+0x130]
   5d0e5:	movsd  xmm2,QWORD PTR [rsp+0x138]
   5d0ee:	divsd  xmm14,xmm5
   5d0f3:	mulsd  xmm2,xmm2
   5d0f7:	mulsd  xmm1,xmm1
   5d0fb:	addsd  xmm1,xmm2
   5d0ff:	movsd  xmm2,QWORD PTR [rsp+0x148]
   5d108:	mulsd  xmm2,xmm2
   5d10c:	addsd  xmm0,xmm1
   5d110:	movsd  xmm1,QWORD PTR [rsp+0x140]
   5d119:	mulsd  xmm1,xmm1
   5d11d:	addsd  xmm1,xmm2
   5d121:	addsd  xmm0,xmm1
   5d125:	movapd xmm1,xmm15
   5d12a:	mulsd  xmm1,xmm15
   5d12f:	mulsd  xmm14,xmm14
   5d134:	addsd  xmm14,xmm1
   5d139:	addsd  xmm0,xmm14
   5d13e:	ucomisd xmm0,xmm13
   5d143:	jp     5d14c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x214c>
   5d145:	movapd xmm1,xmm13
   5d14a:	je     5d181 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2181>
   5d14c:	movsd  xmm2,QWORD PTR [rip+0x670c]        # 63860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   5d154:	mov    QWORD PTR [rsp+0x80],rax
   5d15c:	movapd xmm7,xmm2
   5d160:	movapd xmm1,xmm2
   5d164:	maxsd  xmm7,xmm0
   5d168:	movapd xmm0,xmm7
   5d16c:	call   6ad0 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5d171:	mov    rax,QWORD PTR [rsp+0x80]
   5d179:	movapd xmm1,xmm0
   5d17d:	mulsd  xmm1,xmm7
   5d181:	mulsd  xmm1,xmm5
   5d185:	mulsd  xmm6,QWORD PTR [rip+0x6683]        # 63810 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x90>
   5d18d:	lea    rsi,[r14+0x10]
   5d191:	mov    QWORD PTR [r14+0x18],0x0
   5d199:	movsd  QWORD PTR [r14+0x10],xmm1
   5d19f:	comisd xmm6,xmm1
   5d1a3:	jae    5d1b0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x21b0>
   5d1a5:	cmp    r15,QWORD PTR [rsp+0x18]
   5d1aa:	jb     5d83c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x283c>
   5d1b0:	movq   xmm3,QWORD PTR [rip+0x65d8]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   5d1b8:	cmp    QWORD PTR [rsp],0x0
   5d1bd:	je     5d2e6 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x22e6>
   5d1c3:	mov    rdx,QWORD PTR [rsp+0x8]
   5d1c8:	lea    rdi,[r14-0x18b0]
   5d1cf:	movsd  xmm2,QWORD PTR [rsp+0x31f8]
   5d1d8:	movq   xmm3,QWORD PTR [rip+0x65b0]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   5d1e0:	movsd  xmm0,QWORD PTR [rsp+0x31f0]
   5d1e9:	mov    rcx,rdx
   5d1ec:	nop    DWORD PTR [rax+0x0]
   5d1f0:	movsd  xmm7,QWORD PTR [rdx+0x18c8]
   5d1f8:	add    rdx,0x10
   5d1fc:	add    rcx,0x8
   5d200:	movsd  xmm8,QWORD PTR [rdx+0x18b0]
   5d209:	movsd  xmm5,QWORD PTR [rdx+0x1ac8]
   5d211:	movsd  xmm4,QWORD PTR [rdx+0x1ac0]
   5d219:	movapd xmm9,xmm8
   5d21e:	movapd xmm11,xmm7
   5d223:	movsd  xmm10,QWORD PTR [rcx+0x1bc8]
   5d22c:	mulsd  xmm11,xmm5
   5d231:	mulsd  xmm9,xmm4
   5d236:	subsd  xmm9,xmm11
   5d23b:	movapd xmm11,xmm0
   5d240:	mulsd  xmm11,xmm10
   5d245:	addsd  xmm9,xmm11
   5d24a:	movapd xmm11,xmm8
   5d24f:	mulsd  xmm11,xmm5
   5d254:	mulsd  xmm8,xmm10
   5d259:	movsd  QWORD PTR [rdx+0x18a0],xmm9
   5d262:	movapd xmm9,xmm7
   5d267:	mulsd  xmm9,xmm4
   5d26c:	xorpd  xmm4,XMMWORD PTR [rip+0x651c]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   5d274:	mulsd  xmm7,xmm10
   5d279:	addsd  xmm9,xmm11
   5d27e:	movapd xmm11,xmm10
   5d283:	mulsd  xmm11,xmm2
   5d288:	addsd  xmm9,xmm11
   5d28d:	movapd xmm11,xmm0
   5d292:	mulsd  xmm11,xmm5
   5d297:	mulsd  xmm5,xmm2
   5d29b:	movsd  QWORD PTR [rdx+0x18a8],xmm9
   5d2a4:	movapd xmm9,xmm4
   5d2a9:	mulsd  xmm9,xmm2
   5d2ae:	mulsd  xmm4,xmm0
   5d2b2:	addsd  xmm9,xmm11
   5d2b7:	subsd  xmm4,xmm5
   5d2bb:	addsd  xmm7,xmm9
   5d2c0:	addsd  xmm4,xmm8
   5d2c5:	movsd  QWORD PTR [rdx+0x18b8],xmm7
   5d2cd:	movapd xmm2,xmm7
   5d2d1:	movsd  QWORD PTR [rdx+0x18b0],xmm4
   5d2d9:	movapd xmm0,xmm4
   5d2dd:	cmp    rdx,rdi
   5d2e0:	jne    5d1f0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x21f0>
   5d2e6:	lea    rcx,[r14+0x220]
   5d2ed:	mov    rdx,rbx
   5d2f0:	mov    rdi,r14
   5d2f3:	mov    QWORD PTR [rsp+0xa8],rax
   5d2fb:	mov    QWORD PTR [rsp+0x80],rsi
   5d303:	movaps XMMWORD PTR [rsp+0x90],xmm3
   5d30b:	movsd  QWORD PTR [rsp+0xa0],xmm1
   5d314:	movsd  QWORD PTR [rsp+0x88],xmm6
   5d31d:	call   4f3e0 <_ZN4Base6Matrix9Numerical6givensINS0_7ComplexIdEEEEvRKT_S7_RNS1_6ScalarIS5_E4RealERS5_>
   5d322:	movsd  xmm8,QWORD PTR [r14+0x10]
   5d328:	movsd  xmm7,QWORD PTR [r14+0x18]
   5d32e:	movsd  xmm0,QWORD PTR [r14+0x220]
   5d337:	movsd  xmm4,QWORD PTR [rbx]
   5d33b:	movapd xmm9,xmm8
   5d340:	mov    rcx,QWORD PTR [rsp]
   5d344:	movsd  xmm5,QWORD PTR [r14+0x228]
   5d34d:	movapd xmm2,xmm0
   5d351:	mov    rdx,QWORD PTR [rsp+0x38]
   5d356:	movsd  xmm6,QWORD PTR [rsp+0x88]
   5d35f:	mulsd  xmm2,xmm7
   5d363:	imul   rcx,r15
   5d367:	movsd  xmm1,QWORD PTR [rsp+0xa0]
   5d370:	mov    rax,QWORD PTR [rsp+0xa8]
   5d378:	mulsd  xmm9,xmm5
   5d37d:	mov    rsi,QWORD PTR [rsp+0x80]
   5d385:	movapd xmm3,XMMWORD PTR [rsp+0x90]
   5d38e:	mulsd  xmm0,xmm8
   5d393:	mulsd  xmm5,xmm7
   5d397:	shr    rcx,1
   5d39a:	shl    rcx,0x4
   5d39e:	add    rcx,QWORD PTR [rsp+0x68]
   5d3a3:	addsd  xmm2,xmm9
   5d3a8:	movsd  xmm9,QWORD PTR [r14+0x8]
   5d3ae:	mulsd  xmm9,xmm4
   5d3b3:	subsd  xmm0,xmm5
   5d3b7:	mulsd  xmm4,QWORD PTR [r14]
   5d3bc:	addsd  xmm2,xmm9
   5d3c1:	addsd  xmm0,xmm4
   5d3c5:	movsd  QWORD PTR [r14+0x8],xmm2
   5d3cb:	movsd  QWORD PTR [r14],xmm0
   5d3d0:	jmp    5d404 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2404>
   5d3d2:	xchg   ax,ax
   5d3d4:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5d3df:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5d3ea:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5d3f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5d400:	add    rdx,0x10
   5d404:	cmp    rcx,rdx
   5d407:	je     5d41b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x241b>
   5d409:	movsd  xmm0,QWORD PTR [rdx]
   5d40d:	movsd  QWORD PTR [rcx],xmm0
   5d411:	movsd  xmm0,QWORD PTR [rdx+0x8]
   5d416:	movsd  QWORD PTR [rcx+0x8],xmm0
   5d41b:	add    rcx,0x10
   5d41f:	cmp    r14,rdx
   5d422:	jne    5d400 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2400>
   5d424:	movsd  xmm0,QWORD PTR [r14+0x220]
   5d42d:	movsd  xmm4,QWORD PTR [r14+0x228]
   5d436:	movsd  xmm5,QWORD PTR [r14+0x110]
   5d43f:	movsd  xmm7,QWORD PTR [r14+0x118]
   5d448:	xorpd  xmm0,xmm3
   5d44c:	movapd xmm8,xmm4
   5d451:	mulsd  xmm8,xmm7
   5d456:	movapd xmm2,xmm0
   5d45a:	mulsd  xmm4,xmm5
   5d45e:	mulsd  xmm2,xmm5
   5d462:	mulsd  xmm0,xmm7
   5d466:	subsd  xmm2,xmm8
   5d46b:	addsd  xmm0,xmm4
   5d46f:	movsd  QWORD PTR [r14+0x120],xmm2
   5d478:	movsd  QWORD PTR [r14+0x128],xmm0
   5d481:	movsd  xmm4,QWORD PTR [rbx]
   5d485:	unpcklpd xmm4,xmm4
   5d489:	mulpd  xmm4,XMMWORD PTR [r14+0x110]
   5d492:	movaps XMMWORD PTR [r14+0x110],xmm4
   5d49a:	add    QWORD PTR [rsp+0x3698],0x1
   5d4a3:	comisd xmm6,xmm1
   5d4a7:	jae    5d59a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x259a>
   5d4ad:	pxor   xmm4,xmm4
   5d4b1:	comisd xmm4,xmm2
   5d4b5:	ja     5d80a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x280a>
   5d4bb:	comisd xmm4,xmm0
   5d4bf:	movapd xmm1,xmm2
   5d4c3:	ja     5d89b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x289b>
   5d4c9:	movapd xmm3,xmm0
   5d4cd:	movapd xmm5,xmm1
   5d4d1:	pxor   xmm1,xmm1
   5d4d5:	maxsd  xmm5,xmm3
   5d4d9:	ucomisd xmm5,xmm1
   5d4dd:	jp     5d4e1 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x24e1>
   5d4df:	je     5d53c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x253c>
   5d4e1:	divsd  xmm2,xmm5
   5d4e5:	divsd  xmm0,xmm5
   5d4e9:	movapd xmm1,xmm2
   5d4ed:	mulsd  xmm1,xmm2
   5d4f1:	mulsd  xmm0,xmm0
   5d4f5:	addsd  xmm0,xmm1
   5d4f9:	pxor   xmm1,xmm1
   5d4fd:	ucomisd xmm0,xmm1
   5d501:	jp     5d505 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2505>
   5d503:	je     5d538 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2538>
   5d505:	movsd  xmm1,QWORD PTR [rip+0x6353]        # 63860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   5d50d:	mov    QWORD PTR [rsp+0x28],rax
   5d512:	mov    QWORD PTR [rsp],rsi
   5d516:	movapd xmm6,xmm1
   5d51a:	maxsd  xmm6,xmm0
   5d51e:	movapd xmm0,xmm6
   5d522:	call   6ad0 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5d527:	mov    rax,QWORD PTR [rsp+0x28]
   5d52c:	mov    rsi,QWORD PTR [rsp]
   5d530:	movapd xmm1,xmm0
   5d534:	mulsd  xmm1,xmm6
   5d538:	mulsd  xmm1,xmm5
   5d53c:	comisd xmm1,QWORD PTR [rsp+0xe0]
   5d545:	mov    r8,r15
   5d548:	seta   dl
   5d54b:	xor    edi,edi
   5d54d:	jmp    5c4c0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x14c0>
   5d552:	mov    r8,rbp
   5d555:	movsd  xmm7,QWORD PTR [rsp+0x58]
   5d55b:	mov    rbp,QWORD PTR [rsp]
   5d55f:	mov    r11,r14
   5d562:	mov    r13,QWORD PTR [rsp+0x28]
   5d567:	test   al,al
   5d569:	jne    5e3ce <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x33ce>
   5d56f:	movzx  eax,BYTE PTR [rsp+0x78]
   5d574:	mov    BYTE PTR [rsp+0x36a1],al
   5d57b:	add    rbp,0x1
   5d57f:	cmp    rbp,QWORD PTR [rsp+0xc8]
   5d587:	je     5d8a4 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x28a4>
   5d58d:	movzx  eax,BYTE PTR [rsp+0x36a0]
   5d595:	jmp    5c35c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x135c>
   5d59a:	movzx  edi,BYTE PTR [rsp+0x28]
   5d59f:	xor    edx,edx
   5d5a1:	mov    r8,r15
   5d5a4:	jmp    5c4c0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x14c0>
   5d5a9:	mov    QWORD PTR [rsp+0x28],r13
   5d5ae:	mov    r14,QWORD PTR [rsp+0x58]
   5d5b3:	mov    ebx,0xf
   5d5b8:	pxor   xmm3,xmm3
   5d5bc:	mov    rbp,QWORD PTR [rsp+0x208]
   5d5c4:	mov    BYTE PTR [rsp+0x80],dil
   5d5cc:	movsd  xmm7,QWORD PTR [rsp+0x200]
   5d5d5:	mov    r12,QWORD PTR [rsp+0x1a0]
   5d5dd:	movzx  eax,BYTE PTR [rsp+0x78]
   5d5e2:	mov    r13,QWORD PTR [rsp+0x8]
   5d5e7:	mov    QWORD PTR [rsp],rbp
   5d5eb:	mov    rbp,r8
   5d5ee:	movsd  QWORD PTR [rsp+0x58],xmm7
   5d5f4:	jmp    5d612 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2612>
   5d5f6:	cs nop WORD PTR [rax+rax*1+0x0]
   5d600:	sub    rbx,0x1
   5d604:	sub    r12,0x10
   5d608:	cmp    rbx,0xffffffffffffffff
   5d60c:	je     5d552 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2552>
   5d612:	cmp    rbx,rbp
   5d615:	movaps XMMWORD PTR [r12-0x10],xmm3
   5d61b:	setb   r15b
   5d61f:	and    r15b,al
   5d622:	je     5d600 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2600>
   5d624:	movapd xmm5,XMMWORD PTR [r12-0x2a0]
   5d62e:	lea    rdi,[rbx+0x1]
   5d632:	cmp    rbx,0xf
   5d636:	je     5d69f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x269f>
   5d638:	mov    rsi,r12
   5d63b:	mov    rdx,rdi
   5d63e:	xchg   ax,ax
   5d640:	mov    rax,rdx
   5d643:	add    rdx,0x1
   5d647:	cmp    rax,rbp
   5d64a:	jae    5d695 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2695>
   5d64c:	imul   rax,rdx
   5d650:	movapd xmm1,XMMWORD PTR [rsi]
   5d654:	shr    rax,1
   5d657:	add    rax,rbx
   5d65a:	shl    rax,0x4
   5d65e:	movapd xmm2,XMMWORD PTR [r13+rax*1+0x1030]
   5d668:	movapd xmm0,xmm2
   5d66c:	unpcklpd xmm0,xmm2
   5d670:	unpckhpd xmm2,xmm2
   5d674:	mulpd  xmm0,xmm1
   5d678:	shufpd xmm1,xmm1,0x1
   5d67d:	mulpd  xmm1,xmm2
   5d681:	movapd xmm2,xmm0
   5d685:	subpd  xmm2,xmm1
   5d689:	addpd  xmm0,xmm1
   5d68d:	movsd  xmm0,xmm2
   5d691:	subpd  xmm5,xmm0
   5d695:	add    rsi,0x10
   5d699:	cmp    rdx,0x10
   5d69d:	jne    5d640 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2640>
   5d69f:	imul   rdi,rbx
   5d6a3:	pxor   xmm2,xmm2
   5d6a7:	shr    rdi,1
   5d6aa:	lea    r9,[rbx+rdi*1+0x103]
   5d6b2:	shl    r9,0x4
   5d6b6:	add    r9,r13
   5d6b9:	movsd  xmm0,QWORD PTR [r9]
   5d6be:	movsd  xmm1,QWORD PTR [r9+0x8]
   5d6c4:	comisd xmm2,xmm0
   5d6c8:	ja     5d791 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2791>
   5d6ce:	comisd xmm2,xmm1
   5d6d2:	movapd xmm6,xmm0
   5d6d6:	ja     5d7f9 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x27f9>
   5d6dc:	movapd xmm2,xmm1
   5d6e0:	maxsd  xmm6,xmm2
   5d6e4:	pxor   xmm2,xmm2
   5d6e8:	ucomisd xmm6,xmm2
   5d6ec:	jp     5d6f4 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x26f4>
   5d6ee:	je     5d7cb <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x27cb>
   5d6f4:	divsd  xmm0,xmm6
   5d6f8:	divsd  xmm1,xmm6
   5d6fc:	mulsd  xmm0,xmm0
   5d700:	mulsd  xmm1,xmm1
   5d704:	addsd  xmm0,xmm1
   5d708:	pxor   xmm1,xmm1
   5d70c:	ucomisd xmm0,xmm1
   5d710:	jp     5d714 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2714>
   5d712:	je     5d73d <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x273d>
   5d714:	movsd  xmm2,QWORD PTR [rip+0x6144]        # 63860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   5d71c:	movapd xmm7,xmm2
   5d720:	movapd xmm1,xmm2
   5d724:	maxsd  xmm7,xmm0
   5d728:	movapd xmm0,xmm7
   5d72c:	call   6ad0 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5d731:	pxor   xmm3,xmm3
   5d735:	movapd xmm1,xmm0
   5d739:	mulsd  xmm1,xmm7
   5d73d:	movsd  xmm7,QWORD PTR [rsp+0xc0]
   5d746:	mulsd  xmm6,xmm1
   5d74a:	comisd xmm7,xmm6
   5d74e:	ja     5d7cb <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x27cb>
   5d750:	mov    rdi,QWORD PTR [rsp+0x20]
   5d755:	lea    rsi,[rsp+0x220]
   5d75d:	mov    rdx,r9
   5d760:	movsd  xmm0,QWORD PTR [rsp+0xc0]
   5d769:	movaps XMMWORD PTR [rsp+0x220],xmm5
   5d771:	call   5add0 <_ZN4Base6Matrix9Numerical6ScalarINS0_7ComplexIdEEE6divideERKS4_S7_d>
   5d776:	mov    eax,r15d
   5d779:	pxor   xmm3,xmm3
   5d77d:	movapd xmm0,XMMWORD PTR [rsp+0x230]
   5d786:	movaps XMMWORD PTR [r12-0x10],xmm0
   5d78c:	jmp    5d600 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2600>
   5d791:	comisd xmm2,xmm1
   5d795:	movapd xmm6,xmm0
   5d799:	xorpd  xmm6,XMMWORD PTR [rip+0x5fef]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   5d7a1:	jbe    5d6dc <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x26dc>
   5d7a7:	comisd xmm1,xmm0
   5d7ab:	movapd xmm2,xmm1
   5d7af:	xorpd  xmm2,XMMWORD PTR [rip+0x5fd9]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   5d7b7:	seta   al
   5d7ba:	test   al,al
   5d7bc:	je     5d7c2 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x27c2>
   5d7be:	movapd xmm2,xmm6
   5d7c2:	movapd xmm6,xmm2
   5d7c6:	jmp    5d6f4 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x26f4>
   5d7cb:	test   rbx,rbx
   5d7ce:	je     5e803 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3803>
   5d7d4:	sub    rbx,0x2
   5d7d8:	movaps XMMWORD PTR [r12-0x20],xmm3
   5d7de:	cmp    rbx,0xffffffffffffffff
   5d7e2:	je     5e803 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3803>
   5d7e8:	sub    r12,0x20
   5d7ec:	xor    eax,eax
   5d7ee:	movaps XMMWORD PTR [r12-0x10],xmm3
   5d7f4:	jmp    5d600 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2600>
   5d7f9:	movapd xmm2,xmm1
   5d7fd:	xorpd  xmm2,XMMWORD PTR [rip+0x5f8b]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   5d805:	jmp    5d6e0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x26e0>
   5d80a:	comisd xmm4,xmm0
   5d80e:	movapd xmm1,xmm2
   5d812:	xorpd  xmm1,xmm3
   5d816:	jbe    5d4c9 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x24c9>
   5d81c:	comisd xmm0,xmm2
   5d820:	xorpd  xmm3,xmm0
   5d824:	movapd xmm5,xmm3
   5d828:	seta   dl
   5d82b:	test   dl,dl
   5d82d:	je     5d4e1 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x24e1>
   5d833:	movapd xmm5,xmm1
   5d837:	jmp    5d4e1 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x24e1>
   5d83c:	comisd xmm1,QWORD PTR [rip+0x61b4]        # 639f8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x278>
   5d844:	movapd xmm3,xmm1
   5d848:	pxor   xmm2,xmm2
   5d84c:	unpcklpd xmm3,xmm3
   5d850:	jb     5d85e <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x285e>
   5d852:	movsd  xmm2,QWORD PTR [rip+0x5fce]        # 63828 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
   5d85a:	divsd  xmm2,xmm1
   5d85e:	unpcklpd xmm2,xmm2
   5d862:	lea    rcx,[rax-0x30]
   5d866:	mov    rdx,r12
   5d869:	jmp    5d887 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2887>
   5d86b:	mulpd  xmm0,xmm2
   5d86f:	add    rdx,0x10
   5d873:	movaps XMMWORD PTR [rcx+0x130],xmm0
   5d87a:	add    rcx,0x10
   5d87e:	cmp    rdx,r13
   5d881:	je     5d1b0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x21b0>
   5d887:	comisd xmm1,QWORD PTR [rip+0x6169]        # 639f8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x278>
   5d88f:	movapd xmm0,XMMWORD PTR [rdx]
   5d893:	jae    5d86b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x286b>
   5d895:	divpd  xmm0,xmm3
   5d899:	jmp    5d86f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x286f>
   5d89b:	xorpd  xmm3,xmm0
   5d89f:	jmp    5d4cd <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x24cd>
   5d8a4:	movzx  ebx,BYTE PTR [rsp+0x1d7]
   5d8ac:	mov    r14,QWORD PTR [rsp+0x1d8]
   5d8b4:	mov    r12,QWORD PTR [rsp+0x1e0]
   5d8bc:	mov    r15,QWORD PTR [rsp+0x1e8]
   5d8c4:	mov    r9,QWORD PTR [rsp+0x1f0]
   5d8cc:	mov    rbp,QWORD PTR [rsp+0x1f8]
   5d8d4:	mov    rdx,QWORD PTR [rsp+0x20]
   5d8d9:	mov    rdi,QWORD PTR [rsp+0x60]
   5d8de:	mov    esi,0x10
   5d8e3:	mov    QWORD PTR [rsp+0x230],0x0
   5d8ef:	call   fc90 <_ZN4Base6Matrix9Numerical10VectorNorm9ScaleLoopINS0_7ComplexIdEELm16ELm0EE7computeERKNS0_6VectorIS5_Lm16EEEmRd.part.0.constprop.0>
   5d8f4:	pxor   xmm0,xmm0
   5d8f8:	movsd  xmm5,QWORD PTR [rsp+0x230]
   5d901:	ucomisd xmm5,xmm0
   5d905:	jp     5d90d <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x290d>
   5d907:	je     5dd4c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2d4c>
   5d90d:	movsd  xmm14,QWORD PTR [rsp+0x4a8]
   5d917:	movsd  xmm7,QWORD PTR [rsp+0x440]
   5d920:	movsd  xmm13,QWORD PTR [rsp+0x448]
   5d92a:	movsd  xmm6,QWORD PTR [rsp+0x450]
   5d933:	movsd  xmm12,QWORD PTR [rsp+0x458]
   5d93d:	divsd  xmm14,xmm5
   5d942:	movsd  xmm4,QWORD PTR [rsp+0x460]
   5d94b:	movsd  xmm11,QWORD PTR [rsp+0x468]
   5d955:	movsd  xmm3,QWORD PTR [rsp+0x470]
   5d95e:	movsd  xmm10,QWORD PTR [rsp+0x478]
   5d968:	movsd  xmm2,QWORD PTR [rsp+0x480]
   5d971:	movsd  xmm9,QWORD PTR [rsp+0x488]
   5d97b:	movsd  xmm1,QWORD PTR [rsp+0x490]
   5d984:	movsd  xmm8,QWORD PTR [rsp+0x498]
   5d98e:	movsd  xmm0,QWORD PTR [rsp+0x4a0]
   5d997:	divsd  xmm7,xmm5
   5d99b:	movsd  QWORD PTR [rsp],xmm14
   5d9a1:	movsd  xmm14,QWORD PTR [rsp+0x4b0]
   5d9ab:	divsd  xmm14,xmm5
   5d9b0:	mulsd  xmm7,xmm7
   5d9b4:	divsd  xmm13,xmm5
   5d9b9:	movsd  QWORD PTR [rsp+0x18],xmm14
   5d9c0:	movsd  xmm14,QWORD PTR [rsp+0x4b8]
   5d9ca:	divsd  xmm14,xmm5
   5d9cf:	mulsd  xmm13,xmm13
   5d9d4:	addsd  xmm7,xmm13
   5d9d9:	pxor   xmm13,xmm13
   5d9de:	addsd  xmm7,xmm13
   5d9e3:	divsd  xmm6,xmm5
   5d9e7:	movsd  QWORD PTR [rsp+0x28],xmm14
   5d9ee:	movsd  xmm14,QWORD PTR [rsp+0x4c0]
   5d9f8:	divsd  xmm14,xmm5
   5d9fd:	mulsd  xmm6,xmm6
   5da01:	divsd  xmm12,xmm5
   5da06:	movsd  QWORD PTR [rsp+0x50],xmm14
   5da0d:	movsd  xmm14,QWORD PTR [rsp+0x4c8]
   5da17:	divsd  xmm14,xmm5
   5da1c:	mulsd  xmm12,xmm12
   5da21:	addsd  xmm6,xmm12
   5da26:	addsd  xmm6,xmm7
   5da2a:	divsd  xmm4,xmm5
   5da2e:	movsd  QWORD PTR [rsp+0x58],xmm14
   5da35:	movsd  xmm14,QWORD PTR [rsp+0x4d0]
   5da3f:	divsd  xmm14,xmm5
   5da44:	mulsd  xmm4,xmm4
   5da48:	divsd  xmm11,xmm5
   5da4d:	movsd  QWORD PTR [rsp+0x60],xmm14
   5da54:	movsd  xmm14,QWORD PTR [rsp+0x4d8]
   5da5e:	divsd  xmm14,xmm5
   5da63:	mulsd  xmm11,xmm11
   5da68:	addsd  xmm4,xmm11
   5da6d:	addsd  xmm4,xmm6
   5da71:	divsd  xmm3,xmm5
   5da75:	movsd  QWORD PTR [rsp+0x78],xmm14
   5da7c:	movsd  xmm14,QWORD PTR [rsp+0x4e0]
   5da86:	divsd  xmm14,xmm5
   5da8b:	mulsd  xmm3,xmm3
   5da8f:	divsd  xmm10,xmm5
   5da94:	movsd  QWORD PTR [rsp+0x80],xmm14
   5da9e:	movsd  xmm14,QWORD PTR [rsp+0x4e8]
   5daa8:	divsd  xmm14,xmm5
   5daad:	mulsd  xmm10,xmm10
   5dab2:	addsd  xmm3,xmm10
   5dab7:	addsd  xmm3,xmm4
   5dabb:	divsd  xmm2,xmm5
   5dabf:	movsd  QWORD PTR [rsp+0x88],xmm14
   5dac9:	movsd  xmm14,QWORD PTR [rsp+0x4f0]
   5dad3:	divsd  xmm14,xmm5
   5dad8:	mulsd  xmm2,xmm2
   5dadc:	divsd  xmm9,xmm5
   5dae1:	movsd  QWORD PTR [rsp+0xa0],xmm14
   5daeb:	movsd  xmm14,QWORD PTR [rsp+0x4f8]
   5daf5:	divsd  xmm14,xmm5
   5dafa:	mulsd  xmm9,xmm9
   5daff:	addsd  xmm2,xmm9
   5db04:	addsd  xmm2,xmm3
   5db08:	divsd  xmm1,xmm5
   5db0c:	movsd  QWORD PTR [rsp+0x90],xmm14
   5db16:	movsd  xmm14,QWORD PTR [rsp+0x500]
   5db20:	divsd  xmm14,xmm5
   5db25:	mulsd  xmm1,xmm1
   5db29:	divsd  xmm8,xmm5
   5db2e:	movsd  QWORD PTR [rsp+0xa8],xmm14
   5db38:	movsd  xmm14,QWORD PTR [rsp+0x508]
   5db42:	divsd  xmm14,xmm5
   5db47:	mulsd  xmm8,xmm8
   5db4c:	addsd  xmm1,xmm8
   5db51:	addsd  xmm1,xmm2
   5db55:	movsd  xmm2,QWORD PTR [rsp]
   5db5a:	mulsd  xmm2,xmm2
   5db5e:	divsd  xmm0,xmm5
   5db62:	movsd  QWORD PTR [rsp+0xc0],xmm14
   5db6c:	movsd  xmm14,QWORD PTR [rsp+0x510]
   5db76:	divsd  xmm14,xmm5
   5db7b:	mulsd  xmm0,xmm0
   5db7f:	addsd  xmm0,xmm2
   5db83:	movsd  xmm2,QWORD PTR [rsp+0x28]
   5db89:	mulsd  xmm2,xmm2
   5db8d:	addsd  xmm0,xmm1
   5db91:	movsd  xmm1,QWORD PTR [rsp+0x18]
   5db97:	mulsd  xmm1,xmm1
   5db9b:	addsd  xmm1,xmm2
   5db9f:	movsd  xmm2,QWORD PTR [rsp+0x58]
   5dba5:	mulsd  xmm2,xmm2
   5dba9:	addsd  xmm1,xmm0
   5dbad:	movsd  xmm0,QWORD PTR [rsp+0x50]
   5dbb3:	movsd  QWORD PTR [rsp+0xe0],xmm14
   5dbbd:	mulsd  xmm0,xmm0
   5dbc1:	movsd  xmm14,QWORD PTR [rsp+0x518]
   5dbcb:	divsd  xmm14,xmm5
   5dbd0:	addsd  xmm0,xmm2
   5dbd4:	movsd  xmm2,QWORD PTR [rsp+0x78]
   5dbda:	mulsd  xmm2,xmm2
   5dbde:	addsd  xmm0,xmm1
   5dbe2:	movsd  xmm1,QWORD PTR [rsp+0x60]
   5dbe8:	mulsd  xmm1,xmm1
   5dbec:	addsd  xmm1,xmm2
   5dbf0:	movsd  xmm2,QWORD PTR [rsp+0x88]
   5dbf9:	mulsd  xmm2,xmm2
   5dbfd:	addsd  xmm0,xmm1
   5dc01:	movsd  xmm1,QWORD PTR [rsp+0x80]
   5dc0a:	mulsd  xmm1,xmm1
   5dc0e:	addsd  xmm1,xmm2
   5dc12:	movsd  xmm2,QWORD PTR [rsp+0x90]
   5dc1b:	mulsd  xmm2,xmm2
   5dc1f:	addsd  xmm0,xmm1
   5dc23:	movsd  xmm1,QWORD PTR [rsp+0xa0]
   5dc2c:	movsd  QWORD PTR [rsp+0xe8],xmm14
   5dc36:	mulsd  xmm1,xmm1
   5dc3a:	movsd  xmm14,QWORD PTR [rsp+0x520]
   5dc44:	divsd  xmm14,xmm5
   5dc49:	addsd  xmm1,xmm2
   5dc4d:	movsd  xmm2,QWORD PTR [rsp+0xc0]
   5dc56:	mulsd  xmm2,xmm2
   5dc5a:	addsd  xmm0,xmm1
   5dc5e:	movsd  xmm1,QWORD PTR [rsp+0xa8]
   5dc67:	mulsd  xmm1,xmm1
   5dc6b:	addsd  xmm1,xmm2
   5dc6f:	movsd  xmm2,QWORD PTR [rsp+0xe8]
   5dc78:	mulsd  xmm2,xmm2
   5dc7c:	addsd  xmm0,xmm1
   5dc80:	movsd  xmm1,QWORD PTR [rsp+0xe0]
   5dc89:	mulsd  xmm1,xmm1
   5dc8d:	addsd  xmm1,xmm2
   5dc91:	addsd  xmm0,xmm1
   5dc95:	movsd  QWORD PTR [rsp+0xf0],xmm14
   5dc9f:	movsd  xmm14,QWORD PTR [rsp+0x528]
   5dca9:	movsd  xmm1,QWORD PTR [rsp+0xf0]
   5dcb2:	divsd  xmm14,xmm5
   5dcb7:	mulsd  xmm1,xmm1
   5dcbb:	movsd  QWORD PTR [rsp+0xf8],xmm14
   5dcc5:	movsd  xmm14,QWORD PTR [rsp+0x530]
   5dccf:	divsd  xmm14,xmm5
   5dcd4:	movapd xmm15,xmm14
   5dcd9:	movsd  xmm14,QWORD PTR [rsp+0x538]
   5dce3:	movsd  xmm2,QWORD PTR [rsp+0xf8]
   5dcec:	divsd  xmm14,xmm5
   5dcf1:	mulsd  xmm2,xmm2
   5dcf5:	addsd  xmm1,xmm2
   5dcf9:	addsd  xmm0,xmm1
   5dcfd:	movapd xmm1,xmm15
   5dd02:	mulsd  xmm1,xmm15
   5dd07:	mulsd  xmm14,xmm14
   5dd0c:	addsd  xmm1,xmm14
   5dd11:	addsd  xmm0,xmm1
   5dd15:	ucomisd xmm0,xmm13
   5dd1a:	jp     5dd23 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2d23>
   5dd1c:	movapd xmm1,xmm13
   5dd21:	je     5dd44 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2d44>
   5dd23:	movsd  xmm1,QWORD PTR [rip+0x5b35]        # 63860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   5dd2b:	movapd xmm6,xmm1
   5dd2f:	maxsd  xmm6,xmm0
   5dd33:	movapd xmm0,xmm6
   5dd37:	call   6ad0 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5dd3c:	movapd xmm1,xmm0
   5dd40:	mulsd  xmm1,xmm6
   5dd44:	mulsd  xmm5,xmm1
   5dd48:	movapd xmm0,xmm5
   5dd4c:	movsd  xmm6,QWORD PTR [rsp+0xb8]
   5dd55:	comisd xmm6,xmm0
   5dd59:	jb     5e8db <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x38db>
   5dd5f:	mov    rax,rbp
   5dd62:	mov    rcx,r9
   5dd65:	mov    rsi,r15
   5dd68:	mov    r11,QWORD PTR [rsp+0x68]
   5dd6d:	mov    r15,QWORD PTR [rsp+0x8]
   5dd72:	mov    r12,QWORD PTR [rsp+0x70]
   5dd77:	mov    r9d,ebx
   5dd7a:	mov    rbp,r13
   5dd7d:	mov    DWORD PTR [rax+0x2158],0x5
   5dd87:	mov    r10,QWORD PTR [rsp+0x188]
   5dd8f:	mov    r13,rax
   5dd92:	mov    rbx,rcx
   5dd95:	lea    rax,[r14+0x1]
   5dd99:	cmp    rax,rsi
   5dd9c:	je     5dda7 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2da7>
   5dd9e:	add    r14,0x2
   5dda2:	cmp    r14,rsi
   5dda5:	jne    5dd95 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2d95>
   5dda7:	mov    rcx,rbp
   5ddaa:	mov    rax,rbp
   5ddad:	pxor   xmm0,xmm0
   5ddb1:	movaps XMMWORD PTR [rax],xmm0
   5ddb4:	add    rax,0x20
   5ddb8:	movaps XMMWORD PTR [rax-0x10],xmm0
   5ddbc:	cmp    QWORD PTR [rsp+0x30],rax
   5ddc1:	jne    5ddb1 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2db1>
   5ddc3:	mov    rsi,QWORD PTR [rsp+0x10]
   5ddc8:	movapd xmm3,XMMWORD PTR [rcx]
   5ddcc:	movapd xmm2,XMMWORD PTR [rcx+0x10]
   5ddd1:	mov    rdx,r10
   5ddd4:	movlpd xmm2,QWORD PTR [rcx+0x8]
   5ddd9:	mov    rax,QWORD PTR [rsp+0x10]
   5ddde:	movhpd xmm3,QWORD PTR [rcx+0x10]
   5dde3:	nop    DWORD PTR [rax+0x0]
   5ddea:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5ddf5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5de00:	movsd  xmm0,QWORD PTR [rax]
   5de04:	movupd xmm1,XMMWORD PTR [rdx]
   5de08:	add    rax,0x10
   5de0c:	sub    rdx,0xffffffffffffff80
   5de10:	unpcklpd xmm0,xmm0
   5de14:	mulpd  xmm0,xmm1
   5de18:	addpd  xmm3,xmm0
   5de1c:	movsd  xmm0,QWORD PTR [rax-0x8]
   5de21:	unpcklpd xmm0,xmm0
   5de25:	mulpd  xmm0,xmm1
   5de29:	addpd  xmm2,xmm0
   5de2d:	cmp    r12,rax
   5de30:	jne    5de00 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2e00>
   5de32:	movapd xmm1,XMMWORD PTR [rsi]
   5de36:	add    rcx,0x20
   5de3a:	add    rsi,0x20
   5de3e:	add    r10,0x10
   5de42:	movapd xmm6,XMMWORD PTR [rsp+0x160]
   5de4b:	movapd xmm0,XMMWORD PTR [rsi-0x10]
   5de50:	movapd xmm7,XMMWORD PTR [rsp+0x170]
   5de59:	movlpd xmm0,QWORD PTR [rsi-0x18]
   5de5e:	movhpd xmm1,QWORD PTR [rsi-0x10]
   5de63:	movapd xmm4,xmm6
   5de67:	mulpd  xmm4,xmm0
   5de6b:	movapd xmm5,xmm7
   5de6f:	mulpd  xmm5,xmm1
   5de73:	mulpd  xmm0,xmm7
   5de77:	mulpd  xmm1,xmm6
   5de7b:	addpd  xmm4,xmm5
   5de7f:	subpd  xmm1,xmm0
   5de83:	subpd  xmm2,xmm4
   5de87:	subpd  xmm3,xmm1
   5de8b:	movapd xmm0,xmm3
   5de8f:	unpckhpd xmm3,xmm2
   5de93:	unpcklpd xmm0,xmm2
   5de97:	movaps XMMWORD PTR [rcx-0x10],xmm3
   5de9b:	movaps XMMWORD PTR [rcx-0x20],xmm0
   5de9f:	cmp    QWORD PTR [rsp+0x30],rcx
   5dea4:	jne    5ddc8 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2dc8>
   5deaa:	mov    rdx,QWORD PTR [rsp+0x20]
   5deaf:	mov    rdi,rbp
   5deb2:	mov    esi,0x10
   5deb7:	mov    QWORD PTR [rsp+0x230],0x0
   5dec3:	call   fc90 <_ZN4Base6Matrix9Numerical10VectorNorm9ScaleLoopINS0_7ComplexIdEELm16ELm0EE7computeERKNS0_6VectorIS5_Lm16EEEmRd.part.0.constprop.0>
   5dec8:	pxor   xmm0,xmm0
   5decc:	movsd  xmm5,QWORD PTR [rsp+0x230]
   5ded5:	ucomisd xmm5,xmm0
   5ded9:	jp     5dee5 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2ee5>
   5dedb:	movapd xmm6,xmm0
   5dedf:	je     5e2ef <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x32ef>
   5dee5:	movsd  xmm15,QWORD PTR [rsp+0x8b0]
   5deef:	movsd  xmm7,QWORD PTR [rsp+0x840]
   5def8:	movsd  xmm14,QWORD PTR [rsp+0x848]
   5df02:	movsd  xmm6,QWORD PTR [rsp+0x850]
   5df0b:	movsd  xmm13,QWORD PTR [rsp+0x858]
   5df15:	divsd  xmm15,xmm5
   5df1a:	movsd  xmm4,QWORD PTR [rsp+0x860]
   5df23:	movsd  xmm12,QWORD PTR [rsp+0x868]
   5df2d:	movsd  xmm3,QWORD PTR [rsp+0x870]
   5df36:	movsd  xmm11,QWORD PTR [rsp+0x878]
   5df40:	movsd  xmm2,QWORD PTR [rsp+0x880]
   5df49:	movsd  xmm10,QWORD PTR [rsp+0x888]
   5df53:	movsd  xmm1,QWORD PTR [rsp+0x890]
   5df5c:	movsd  xmm9,QWORD PTR [rsp+0x898]
   5df66:	movsd  xmm0,QWORD PTR [rsp+0x8a0]
   5df6f:	movsd  xmm8,QWORD PTR [rsp+0x8a8]
   5df79:	divsd  xmm7,xmm5
   5df7d:	movsd  QWORD PTR [rsp],xmm15
   5df83:	movsd  xmm15,QWORD PTR [rsp+0x8b8]
   5df8d:	divsd  xmm15,xmm5
   5df92:	mulsd  xmm7,xmm7
   5df96:	divsd  xmm14,xmm5
   5df9b:	movsd  QWORD PTR [rsp+0x8],xmm15
   5dfa2:	movsd  xmm15,QWORD PTR [rsp+0x8c0]
   5dfac:	divsd  xmm15,xmm5
   5dfb1:	mulsd  xmm14,xmm14
   5dfb6:	addsd  xmm7,xmm14
   5dfbb:	pxor   xmm14,xmm14
   5dfc0:	addsd  xmm7,xmm14
   5dfc5:	divsd  xmm6,xmm5
   5dfc9:	movsd  QWORD PTR [rsp+0x18],xmm15
   5dfd0:	movsd  xmm15,QWORD PTR [rsp+0x8c8]
   5dfda:	divsd  xmm15,xmm5
   5dfdf:	mulsd  xmm6,xmm6
   5dfe3:	divsd  xmm13,xmm5
   5dfe8:	movsd  QWORD PTR [rsp+0x28],xmm15
   5dfef:	movsd  xmm15,QWORD PTR [rsp+0x8d0]
   5dff9:	divsd  xmm15,xmm5
   5dffe:	mulsd  xmm13,xmm13
   5e003:	addsd  xmm6,xmm13
   5e008:	addsd  xmm6,xmm7
   5e00c:	divsd  xmm4,xmm5
   5e010:	movsd  QWORD PTR [rsp+0x38],xmm15
   5e017:	movsd  xmm15,QWORD PTR [rsp+0x8d8]
   5e021:	divsd  xmm15,xmm5
   5e026:	mulsd  xmm4,xmm4
   5e02a:	divsd  xmm12,xmm5
   5e02f:	movsd  QWORD PTR [rsp+0x40],xmm15
   5e036:	movsd  xmm15,QWORD PTR [rsp+0x8e0]
   5e040:	divsd  xmm15,xmm5
   5e045:	mulsd  xmm12,xmm12
   5e04a:	addsd  xmm4,xmm12
   5e04f:	addsd  xmm4,xmm6
   5e053:	divsd  xmm3,xmm5
   5e057:	movsd  QWORD PTR [rsp+0x48],xmm15
   5e05e:	movsd  xmm15,QWORD PTR [rsp+0x8e8]
   5e068:	divsd  xmm15,xmm5
   5e06d:	mulsd  xmm3,xmm3
   5e071:	divsd  xmm11,xmm5
   5e076:	movsd  QWORD PTR [rsp+0x50],xmm15
   5e07d:	movsd  xmm15,QWORD PTR [rsp+0x8f0]
   5e087:	divsd  xmm15,xmm5
   5e08c:	mulsd  xmm11,xmm11
   5e091:	addsd  xmm3,xmm11
   5e096:	addsd  xmm3,xmm4
   5e09a:	divsd  xmm2,xmm5
   5e09e:	movsd  QWORD PTR [rsp+0x58],xmm15
   5e0a5:	movsd  xmm15,QWORD PTR [rsp+0x8f8]
   5e0af:	divsd  xmm15,xmm5
   5e0b4:	mulsd  xmm2,xmm2
   5e0b8:	divsd  xmm10,xmm5
   5e0bd:	movsd  QWORD PTR [rsp+0x60],xmm15
   5e0c4:	movsd  xmm15,QWORD PTR [rsp+0x900]
   5e0ce:	divsd  xmm15,xmm5
   5e0d3:	mulsd  xmm10,xmm10
   5e0d8:	addsd  xmm2,xmm10
   5e0dd:	addsd  xmm2,xmm3
   5e0e1:	divsd  xmm1,xmm5
   5e0e5:	movsd  QWORD PTR [rsp+0x68],xmm15
   5e0ec:	movsd  xmm15,QWORD PTR [rsp+0x908]
   5e0f6:	divsd  xmm15,xmm5
   5e0fb:	mulsd  xmm1,xmm1
   5e0ff:	divsd  xmm9,xmm5
   5e104:	movsd  QWORD PTR [rsp+0x70],xmm15
   5e10b:	movsd  xmm15,QWORD PTR [rsp+0x910]
   5e115:	divsd  xmm15,xmm5
   5e11a:	mulsd  xmm9,xmm9
   5e11f:	addsd  xmm1,xmm9
   5e124:	addsd  xmm1,xmm2
   5e128:	movsd  xmm2,QWORD PTR [rsp+0x8]
   5e12e:	mulsd  xmm2,xmm2
   5e132:	divsd  xmm0,xmm5
   5e136:	movsd  QWORD PTR [rsp+0x78],xmm15
   5e13d:	movsd  xmm15,QWORD PTR [rsp+0x918]
   5e147:	divsd  xmm15,xmm5
   5e14c:	mulsd  xmm0,xmm0
   5e150:	divsd  xmm8,xmm5
   5e155:	movsd  QWORD PTR [rsp+0x80],xmm15
   5e15f:	movsd  xmm15,QWORD PTR [rsp+0x920]
   5e169:	divsd  xmm15,xmm5
   5e16e:	mulsd  xmm8,xmm8
   5e173:	addsd  xmm0,xmm8
   5e178:	addsd  xmm0,xmm1
   5e17c:	movsd  xmm1,QWORD PTR [rsp]
   5e181:	mulsd  xmm1,xmm1
   5e185:	addsd  xmm1,xmm2
   5e189:	movsd  xmm2,QWORD PTR [rsp+0x28]
   5e18f:	mulsd  xmm2,xmm2
   5e193:	addsd  xmm1,xmm0
   5e197:	movsd  xmm0,QWORD PTR [rsp+0x18]
   5e19d:	movsd  QWORD PTR [rsp+0x88],xmm15
   5e1a7:	mulsd  xmm0,xmm0
   5e1ab:	movsd  xmm15,QWORD PTR [rsp+0x928]
   5e1b5:	divsd  xmm15,xmm5
   5e1ba:	addsd  xmm0,xmm2
   5e1be:	movsd  xmm2,QWORD PTR [rsp+0x40]
   5e1c4:	mulsd  xmm2,xmm2
   5e1c8:	addsd  xmm0,xmm1
   5e1cc:	movsd  xmm1,QWORD PTR [rsp+0x38]
   5e1d2:	mulsd  xmm1,xmm1
   5e1d6:	addsd  xmm1,xmm2
   5e1da:	movsd  xmm2,QWORD PTR [rsp+0x50]
   5e1e0:	mulsd  xmm2,xmm2
   5e1e4:	addsd  xmm0,xmm1
   5e1e8:	movsd  xmm1,QWORD PTR [rsp+0x48]
   5e1ee:	mulsd  xmm1,xmm1
   5e1f2:	addsd  xmm1,xmm2
   5e1f6:	movsd  xmm2,QWORD PTR [rsp+0x60]
   5e1fc:	mulsd  xmm2,xmm2
   5e200:	addsd  xmm0,xmm1
   5e204:	movsd  xmm1,QWORD PTR [rsp+0x58]
   5e20a:	movsd  QWORD PTR [rsp+0xa0],xmm15
   5e214:	mulsd  xmm1,xmm1
   5e218:	movsd  xmm15,QWORD PTR [rsp+0x930]
   5e222:	divsd  xmm15,xmm5
   5e227:	addsd  xmm1,xmm2
   5e22b:	movsd  xmm2,QWORD PTR [rsp+0x70]
   5e231:	mulsd  xmm2,xmm2
   5e235:	addsd  xmm0,xmm1
   5e239:	movsd  xmm1,QWORD PTR [rsp+0x68]
   5e23f:	mulsd  xmm1,xmm1
   5e243:	addsd  xmm1,xmm2
   5e247:	movsd  xmm2,QWORD PTR [rsp+0x80]
   5e250:	mulsd  xmm2,xmm2
   5e254:	addsd  xmm0,xmm1
   5e258:	movsd  xmm1,QWORD PTR [rsp+0x78]
   5e25e:	mulsd  xmm1,xmm1
   5e262:	addsd  xmm1,xmm2
   5e266:	addsd  xmm0,xmm1
   5e26a:	movsd  xmm1,QWORD PTR [rsp+0x88]
   5e273:	movsd  QWORD PTR [rsp+0x90],xmm15
   5e27d:	mulsd  xmm1,xmm1
   5e281:	movsd  xmm15,QWORD PTR [rsp+0x938]
   5e28b:	movsd  xmm2,QWORD PTR [rsp+0xa0]
   5e294:	divsd  xmm15,xmm5
   5e299:	mulsd  xmm2,xmm2
   5e29d:	addsd  xmm1,xmm2
   5e2a1:	addsd  xmm0,xmm1
   5e2a5:	movsd  xmm1,QWORD PTR [rsp+0x90]
   5e2ae:	mulsd  xmm1,xmm1
   5e2b2:	mulsd  xmm15,xmm15
   5e2b7:	addsd  xmm1,xmm15
   5e2bc:	addsd  xmm0,xmm1
   5e2c0:	ucomisd xmm0,xmm14
   5e2c5:	jp     5e2ce <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x32ce>
   5e2c7:	movapd xmm6,xmm14
   5e2cc:	je     5e2eb <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x32eb>
   5e2ce:	movsd  xmm1,QWORD PTR [rip+0x558a]        # 63860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   5e2d6:	movapd xmm6,xmm1
   5e2da:	maxsd  xmm6,xmm0
   5e2de:	movapd xmm0,xmm6
   5e2e2:	call   6ad0 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5e2e7:	mulsd  xmm6,xmm0
   5e2eb:	mulsd  xmm6,xmm5
   5e2ef:	divsd  xmm6,QWORD PTR [rsp+0x190]
   5e2f8:	comisd xmm6,QWORD PTR [r13+0x2170]
   5e301:	jbe    5e30c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x330c>
   5e303:	movsd  QWORD PTR [r13+0x2170],xmm6
   5e30c:	movzx  ecx,BYTE PTR [rsp+0xd8]
   5e314:	and    cl,BYTE PTR [r13+0x215d]
   5e31b:	mov    BYTE PTR [r13+0x215d],cl
   5e322:	mov    rdx,QWORD PTR [rsp+0x150]
   5e32a:	mov    rax,QWORD PTR [rsp+0x10]
   5e32f:	movapd xmm0,XMMWORD PTR [rax]
   5e333:	add    rax,0x10
   5e337:	add    rdx,0x10
   5e33b:	movups XMMWORD PTR [rdx-0x10],xmm0
   5e33f:	cmp    rax,r12
   5e342:	jne    5e32f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x332f>
   5e344:	add    QWORD PTR [rsp+0xb0],0x1
   5e34d:	mov    rax,QWORD PTR [rsp+0xb0]
   5e355:	add    QWORD PTR [rsp+0x150],0x100
   5e361:	add    QWORD PTR [rsp+0x198],0x10
   5e36a:	cmp    rax,0x10
   5e36e:	jne    5b1cc <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1cc>
   5e374:	mov    rax,QWORD PTR [rsp+0xd0]
   5e37c:	movzx  r15d,cl
   5e380:	mov    BYTE PTR [rax+0x2161],0x1
   5e387:	cmp    BYTE PTR [rax+0x215f],0x0
   5e38e:	je     5ef7f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3f7f>
   5e394:	mov    BYTE PTR [rax+0x215d],0x0
   5e39b:	mov    DWORD PTR [rax+0x2158],0x4
   5e3a5:	mov    rax,QWORD PTR [rsp+0x36a8]
   5e3ad:	sub    rax,QWORD PTR fs:0x28
   5e3b6:	jne    5efdd <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3fdd>
   5e3bc:	add    rsp,0x36b8
   5e3c3:	pop    rbx
   5e3c4:	pop    rbp
   5e3c5:	pop    r12
   5e3c7:	pop    r13
   5e3c9:	pop    r14
   5e3cb:	pop    r15
   5e3cd:	ret
   5e3ce:	mov    rdx,QWORD PTR [rsp+0x60]
   5e3d3:	mov    rcx,QWORD PTR [rsp+0x30]
   5e3d8:	mov    r9,r13
   5e3db:	mov    rax,r13
   5e3de:	movapd xmm0,XMMWORD PTR [rdx]
   5e3e2:	add    rax,0x10
   5e3e6:	add    rdx,0x10
   5e3ea:	movaps XMMWORD PTR [rax-0x10],xmm0
   5e3ee:	cmp    rcx,rax
   5e3f1:	jne    5e3de <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x33de>
   5e3f3:	mov    rsi,QWORD PTR [rsp+0x8]
   5e3f8:	mov    rdi,r13
   5e3fb:	xor    r10d,r10d
   5e3fe:	pxor   xmm6,xmm6
   5e402:	mov    rbx,rsi
   5e405:	pxor   xmm8,xmm8
   5e40a:	lea    rcx,[rsi+0x1c50]
   5e411:	mov    rdx,rbx
   5e414:	xor    eax,eax
   5e416:	movapd xmm5,xmm8
   5e41b:	nop    DWORD PTR [rax+rax*1+0x0]
   5e420:	cmp    rax,r8
   5e423:	jae    5e464 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3464>
   5e425:	movsd  xmm0,QWORD PTR [rdx+0x30]
   5e42a:	movsd  xmm3,QWORD PTR [rdx+0x38]
   5e42f:	movsd  xmm1,QWORD PTR [rcx]
   5e433:	movsd  xmm4,QWORD PTR [rcx+0x8]
   5e438:	movapd xmm2,xmm0
   5e43c:	movapd xmm9,xmm3
   5e441:	mulsd  xmm2,xmm1
   5e445:	mulsd  xmm9,xmm4
   5e44a:	mulsd  xmm0,xmm4
   5e44e:	mulsd  xmm1,xmm3
   5e452:	subsd  xmm2,xmm9
   5e457:	addsd  xmm0,xmm1
   5e45b:	addsd  xmm5,xmm2
   5e45f:	addsd  xmm8,xmm0
   5e464:	add    rax,0x1
   5e468:	add    rdx,0x100
   5e46f:	add    rcx,0x10
   5e473:	cmp    rax,0x10
   5e477:	jne    5e420 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3420>
   5e479:	movapd xmm0,xmm5
   5e47d:	add    r10,0x1
   5e481:	add    rdi,0x10
   5e485:	add    rbx,0x10
   5e489:	mulsd  xmm5,xmm6
   5e48d:	movapd xmm1,xmm8
   5e492:	mulsd  xmm1,xmm6
   5e496:	addsd  xmm5,xmm8
   5e49b:	addsd  xmm5,QWORD PTR [rdi-0x8]
   5e4a0:	subsd  xmm0,xmm1
   5e4a4:	addsd  xmm0,QWORD PTR [rdi-0x10]
   5e4a9:	movsd  QWORD PTR [rdi-0x8],xmm5
   5e4ae:	movsd  QWORD PTR [rdi-0x10],xmm0
   5e4b3:	cmp    r10,0x10
   5e4b7:	jne    5e405 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3405>
   5e4bd:	mov    rax,QWORD PTR [rsp+0x50]
   5e4c2:	mov    rdi,QWORD PTR [rsp+0x30]
   5e4c7:	pxor   xmm0,xmm0
   5e4cb:	mov    rsi,r13
   5e4ce:	movsd  QWORD PTR [rsp],xmm7
   5e4d3:	mov    rdx,rax
   5e4d6:	movaps XMMWORD PTR [rsp+0x540],xmm0
   5e4de:	movaps XMMWORD PTR [rsp+0x550],xmm0
   5e4e6:	movaps XMMWORD PTR [rsp+0x560],xmm0
   5e4ee:	movaps XMMWORD PTR [rsp+0x570],xmm0
   5e4f6:	movaps XMMWORD PTR [rsp+0x580],xmm0
   5e4fe:	movaps XMMWORD PTR [rsp+0x590],xmm0
   5e506:	movaps XMMWORD PTR [rsp+0x5a0],xmm0
   5e50e:	movaps XMMWORD PTR [rsp+0x5b0],xmm0
   5e516:	movaps XMMWORD PTR [rsp+0x5c0],xmm0
   5e51e:	movaps XMMWORD PTR [rsp+0x5d0],xmm0
   5e526:	movaps XMMWORD PTR [rsp+0x5e0],xmm0
   5e52e:	movaps XMMWORD PTR [rsp+0x5f0],xmm0
   5e536:	movaps XMMWORD PTR [rsp+0x600],xmm0
   5e53e:	movaps XMMWORD PTR [rsp+0x610],xmm0
   5e546:	movaps XMMWORD PTR [rsp+0x620],xmm0
   5e54e:	movaps XMMWORD PTR [rsp+0x630],xmm0
   5e556:	call   1c0c0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeINS0_7ComplexIdEELm16EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS7_XT0_EEERSC_m.part.0>
   5e55b:	mov    rdx,QWORD PTR [rsp+0x10]
   5e560:	movsd  xmm7,QWORD PTR [rsp]
   5e565:	movapd xmm0,XMMWORD PTR [rdx]
   5e569:	subpd  xmm0,XMMWORD PTR [rax]
   5e56d:	add    rax,0x10
   5e571:	add    rdx,0x10
   5e575:	movaps XMMWORD PTR [rax-0x10],xmm0
   5e579:	cmp    rax,r11
   5e57c:	jne    5e565 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3565>
   5e57e:	mov    rdx,QWORD PTR [rsp+0x1c0]
   5e586:	mov    rdi,QWORD PTR [rsp+0x50]
   5e58b:	mov    esi,0x10
   5e590:	mov    QWORD PTR [rsp+0x218],0x0
   5e59c:	call   fc90 <_ZN4Base6Matrix9Numerical10VectorNorm9ScaleLoopINS0_7ComplexIdEELm16ELm0EE7computeERKNS0_6VectorIS5_Lm16EEEmRd.part.0.constprop.0>
   5e5a1:	pxor   xmm0,xmm0
   5e5a5:	movsd  xmm5,QWORD PTR [rsp+0x218]
   5e5ae:	ucomisd xmm5,xmm0
   5e5b2:	jp     5e5ba <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x35ba>
   5e5b4:	je     5e7ae <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x37ae>
   5e5ba:	movsd  xmm14,QWORD PTR [rsp+0x5a0]
   5e5c4:	mov    rdx,QWORD PTR [rsp+0x20]
   5e5c9:	mov    esi,0x10
   5e5ce:	movsd  xmm2,QWORD PTR [rsp+0x540]
   5e5d7:	mov    rdi,QWORD PTR [rsp+0x50]
   5e5dc:	movsd  xmm13,QWORD PTR [rsp+0x548]
   5e5e6:	divsd  xmm14,xmm5
   5e5eb:	movsd  xmm1,QWORD PTR [rsp+0x550]
   5e5f4:	movsd  xmm12,QWORD PTR [rsp+0x558]
   5e5fe:	movsd  xmm0,QWORD PTR [rsp+0x560]
   5e607:	movsd  xmm11,QWORD PTR [rsp+0x568]
   5e611:	movsd  xmm6,QWORD PTR [rsp+0x570]
   5e61a:	movsd  xmm10,QWORD PTR [rsp+0x578]
   5e624:	movsd  xmm4,QWORD PTR [rsp+0x580]
   5e62d:	movsd  xmm9,QWORD PTR [rsp+0x588]
   5e637:	movsd  xmm3,QWORD PTR [rsp+0x590]
   5e640:	movsd  xmm8,QWORD PTR [rsp+0x598]
   5e64a:	divsd  xmm2,xmm5
   5e64e:	movsd  QWORD PTR [rsp],xmm14
   5e654:	movsd  xmm14,QWORD PTR [rsp+0x5a8]
   5e65e:	divsd  xmm14,xmm5
   5e663:	mulsd  xmm2,xmm2
   5e667:	divsd  xmm13,xmm5
   5e66c:	movsd  QWORD PTR [rsp+0x28],xmm14
   5e673:	movsd  xmm14,QWORD PTR [rsp+0x5b0]
   5e67d:	divsd  xmm14,xmm5
   5e682:	mulsd  xmm13,xmm13
   5e687:	addsd  xmm2,xmm13
   5e68c:	pxor   xmm13,xmm13
   5e691:	addsd  xmm2,xmm13
   5e696:	divsd  xmm1,xmm5
   5e69a:	movapd xmm15,xmm14
   5e69f:	movsd  xmm14,QWORD PTR [rsp+0x5b8]
   5e6a9:	divsd  xmm12,xmm5
   5e6ae:	mulsd  xmm1,xmm1
   5e6b2:	divsd  xmm0,xmm5
   5e6b6:	mulsd  xmm12,xmm12
   5e6bb:	addsd  xmm1,xmm12
   5e6c0:	addsd  xmm1,xmm2
   5e6c4:	movsd  xmm2,QWORD PTR [rsp+0x28]
   5e6ca:	mulsd  xmm2,xmm2
   5e6ce:	divsd  xmm11,xmm5
   5e6d3:	mulsd  xmm0,xmm0
   5e6d7:	divsd  xmm6,xmm5
   5e6db:	mulsd  xmm11,xmm11
   5e6e0:	addsd  xmm0,xmm11
   5e6e5:	addsd  xmm0,xmm1
   5e6e9:	movsd  xmm1,QWORD PTR [rsp]
   5e6ee:	mulsd  xmm1,xmm1
   5e6f2:	divsd  xmm10,xmm5
   5e6f7:	addsd  xmm1,xmm2
   5e6fb:	mulsd  xmm6,xmm6
   5e6ff:	divsd  xmm4,xmm5
   5e703:	mulsd  xmm10,xmm10
   5e708:	addsd  xmm6,xmm10
   5e70d:	addsd  xmm0,xmm6
   5e711:	divsd  xmm9,xmm5
   5e716:	mulsd  xmm4,xmm4
   5e71a:	divsd  xmm3,xmm5
   5e71e:	mulsd  xmm9,xmm9
   5e723:	addsd  xmm4,xmm9
   5e728:	addsd  xmm0,xmm4
   5e72c:	divsd  xmm8,xmm5
   5e731:	mulsd  xmm3,xmm3
   5e735:	divsd  xmm14,xmm5
   5e73a:	mulsd  xmm8,xmm8
   5e73f:	addsd  xmm3,xmm8
   5e744:	addsd  xmm0,xmm3
   5e748:	addsd  xmm0,xmm1
   5e74c:	movapd xmm1,xmm15
   5e751:	mulsd  xmm1,xmm15
   5e756:	mulsd  xmm14,xmm14
   5e75b:	addsd  xmm1,xmm14
   5e760:	addsd  xmm0,xmm1
   5e764:	movsd  QWORD PTR [rsp+0x230],xmm0
   5e76d:	movapd xmm0,xmm5
   5e771:	call   c140 <_ZN4Base6Matrix9Numerical10VectorNorm14SquaredSumLoopINS0_7ComplexIdEELm16ELm8EE7computeERKNS0_6VectorIS5_Lm16EEEmdRd.part.0>
   5e776:	movsd  xmm2,QWORD PTR [rsp+0x230]
   5e77f:	ucomisd xmm2,xmm13
   5e784:	jp     5e78d <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x378d>
   5e786:	movapd xmm0,xmm13
   5e78b:	je     5e7aa <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x37aa>
   5e78d:	movsd  xmm1,QWORD PTR [rip+0x50cb]        # 63860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   5e795:	movapd xmm6,xmm1
   5e799:	maxsd  xmm6,xmm2
   5e79d:	movapd xmm0,xmm6
   5e7a1:	call   6ad0 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5e7a6:	mulsd  xmm0,xmm6
   5e7aa:	mulsd  xmm0,xmm5
   5e7ae:	comisd xmm7,xmm0
   5e7b2:	mov    rax,QWORD PTR [rsp+0x60]
   5e7b7:	lea    rdx,[rax+0x100]
   5e7be:	jb     5e814 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3814>
   5e7c0:	movapd xmm1,XMMWORD PTR [r9]
   5e7c5:	add    rax,0x10
   5e7c9:	add    r9,0x10
   5e7cd:	movaps XMMWORD PTR [rax-0x10],xmm1
   5e7d1:	cmp    rdx,rax
   5e7d4:	jne    5e7c0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x37c0>
   5e7d6:	movsd  QWORD PTR [rsp+0x3690],xmm0
   5e7df:	movsd  xmm6,QWORD PTR [rsp+0xe0]
   5e7e8:	comisd xmm6,xmm0
   5e7ec:	setae  BYTE PTR [rsp+0x36a0]
   5e7f4:	setb   al
   5e7f7:	and    al,BYTE PTR [rsp+0x80]
   5e7fe:	jmp    5d574 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2574>
   5e803:	mov    rbp,QWORD PTR [rsp]
   5e807:	mov    r13,QWORD PTR [rsp+0x28]
   5e80c:	mov    r11,r14
   5e80f:	jmp    5d56f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x256f>
   5e814:	mov    rax,QWORD PTR [rsp+0x50]
   5e819:	mov    rsi,QWORD PTR [rsp+0x60]
   5e81e:	pxor   xmm0,xmm0
   5e822:	mov    rdi,QWORD PTR [rsp+0x30]
   5e827:	movaps XMMWORD PTR [rsp+0x540],xmm0
   5e82f:	mov    rdx,rax
   5e832:	movaps XMMWORD PTR [rsp+0x550],xmm0
   5e83a:	movaps XMMWORD PTR [rsp+0x560],xmm0
   5e842:	movaps XMMWORD PTR [rsp+0x570],xmm0
   5e84a:	movaps XMMWORD PTR [rsp+0x580],xmm0
   5e852:	movaps XMMWORD PTR [rsp+0x590],xmm0
   5e85a:	movaps XMMWORD PTR [rsp+0x5a0],xmm0
   5e862:	movaps XMMWORD PTR [rsp+0x5b0],xmm0
   5e86a:	movaps XMMWORD PTR [rsp+0x5c0],xmm0
   5e872:	movaps XMMWORD PTR [rsp+0x5d0],xmm0
   5e87a:	movaps XMMWORD PTR [rsp+0x5e0],xmm0
   5e882:	movaps XMMWORD PTR [rsp+0x5f0],xmm0
   5e88a:	movaps XMMWORD PTR [rsp+0x600],xmm0
   5e892:	movaps XMMWORD PTR [rsp+0x610],xmm0
   5e89a:	movaps XMMWORD PTR [rsp+0x620],xmm0
   5e8a2:	movaps XMMWORD PTR [rsp+0x630],xmm0
   5e8aa:	call   1c0c0 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeINS0_7ComplexIdEELm16EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS7_XT0_EEERSC_m.part.0>
   5e8af:	mov    rdx,QWORD PTR [rsp+0x10]
   5e8b4:	movapd xmm0,XMMWORD PTR [rdx]
   5e8b8:	subpd  xmm0,XMMWORD PTR [rax]
   5e8bc:	add    rax,0x10
   5e8c0:	add    rdx,0x10
   5e8c4:	movaps XMMWORD PTR [rax-0x10],xmm0
   5e8c8:	cmp    rax,r11
   5e8cb:	jne    5e8b4 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x38b4>
   5e8cd:	movsd  xmm0,QWORD PTR [rsp+0x3690]
   5e8d6:	jmp    5e7df <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x37df>
   5e8db:	mov    rax,QWORD PTR [rsp+0x10]
   5e8e0:	unpcklpd xmm0,xmm0
   5e8e4:	movapd xmm1,XMMWORD PTR [r12]
   5e8ea:	add    rax,0x10
   5e8ee:	add    r12,0x10
   5e8f2:	divpd  xmm1,xmm0
   5e8f6:	movaps XMMWORD PTR [rax-0x10],xmm1
   5e8fa:	cmp    QWORD PTR [rsp+0x70],rax
   5e8ff:	jne    5e8e4 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x38e4>
   5e901:	mov    rcx,r13
   5e904:	mov    rax,r13
   5e907:	pxor   xmm0,xmm0
   5e90b:	movaps XMMWORD PTR [rax],xmm0
   5e90e:	add    rax,0x20
   5e912:	movaps XMMWORD PTR [rax-0x10],xmm0
   5e916:	cmp    QWORD PTR [rsp+0x30],rax
   5e91b:	jne    5e90b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x390b>
   5e91d:	mov    rdi,QWORD PTR [rsp+0x1a8]
   5e925:	mov    rsi,QWORD PTR [rsp+0x10]
   5e92a:	mov    r8,QWORD PTR [rsp+0x70]
   5e92f:	movapd xmm3,XMMWORD PTR [rcx]
   5e933:	movapd xmm2,XMMWORD PTR [rcx+0x10]
   5e938:	mov    rdx,rdi
   5e93b:	movlpd xmm2,QWORD PTR [rcx+0x8]
   5e940:	mov    rax,QWORD PTR [rsp+0x10]
   5e945:	movhpd xmm3,QWORD PTR [rcx+0x10]
   5e94a:	cs nop WORD PTR [rax+rax*1+0x0]
   5e954:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5e95f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5e96a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5e975:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5e980:	movsd  xmm0,QWORD PTR [rax]
   5e984:	movupd xmm1,XMMWORD PTR [rdx]
   5e988:	add    rax,0x10
   5e98c:	sub    rdx,0xffffffffffffff80
   5e990:	unpcklpd xmm0,xmm0
   5e994:	mulpd  xmm0,xmm1
   5e998:	addpd  xmm3,xmm0
   5e99c:	movsd  xmm0,QWORD PTR [rax-0x8]
   5e9a1:	unpcklpd xmm0,xmm0
   5e9a5:	mulpd  xmm0,xmm1
   5e9a9:	addpd  xmm2,xmm0
   5e9ad:	cmp    r8,rax
   5e9b0:	jne    5e980 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3980>
   5e9b2:	movapd xmm1,XMMWORD PTR [rsi]
   5e9b6:	add    rcx,0x20
   5e9ba:	add    rsi,0x20
   5e9be:	add    rdi,0x10
   5e9c2:	movapd xmm6,XMMWORD PTR [rsp+0x160]
   5e9cb:	movapd xmm0,XMMWORD PTR [rsi-0x10]
   5e9d0:	movapd xmm7,XMMWORD PTR [rsp+0x170]
   5e9d9:	movlpd xmm0,QWORD PTR [rsi-0x18]
   5e9de:	movhpd xmm1,QWORD PTR [rsi-0x10]
   5e9e3:	movapd xmm4,xmm6
   5e9e7:	mulpd  xmm4,xmm0
   5e9eb:	movapd xmm5,xmm7
   5e9ef:	mulpd  xmm5,xmm1
   5e9f3:	mulpd  xmm0,xmm7
   5e9f7:	mulpd  xmm1,xmm6
   5e9fb:	addpd  xmm4,xmm5
   5e9ff:	subpd  xmm1,xmm0
   5ea03:	subpd  xmm2,xmm4
   5ea07:	subpd  xmm3,xmm1
   5ea0b:	movapd xmm0,xmm3
   5ea0f:	unpckhpd xmm3,xmm2
   5ea13:	unpcklpd xmm0,xmm2
   5ea17:	movaps XMMWORD PTR [rcx-0x10],xmm3
   5ea1b:	movaps XMMWORD PTR [rcx-0x20],xmm0
   5ea1f:	cmp    QWORD PTR [rsp+0x30],rcx
   5ea24:	jne    5e92f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x392f>
   5ea2a:	mov    rdx,QWORD PTR [rsp+0x20]
   5ea2f:	mov    rdi,r13
   5ea32:	mov    esi,0x10
   5ea37:	mov    QWORD PTR [rsp+0x230],0x0
   5ea43:	call   fc90 <_ZN4Base6Matrix9Numerical10VectorNorm9ScaleLoopINS0_7ComplexIdEELm16ELm0EE7computeERKNS0_6VectorIS5_Lm16EEEmRd.part.0.constprop.0>
   5ea48:	pxor   xmm0,xmm0
   5ea4c:	movsd  xmm5,QWORD PTR [rsp+0x230]
   5ea55:	ucomisd xmm5,xmm0
   5ea59:	jp     5ea65 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3a65>
   5ea5b:	movapd xmm6,xmm0
   5ea5f:	je     5ee9c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3e9c>
   5ea65:	movsd  xmm14,QWORD PTR [rsp+0x8a8]
   5ea6f:	movsd  xmm7,QWORD PTR [rsp+0x840]
   5ea78:	movsd  xmm13,QWORD PTR [rsp+0x848]
   5ea82:	movsd  xmm6,QWORD PTR [rsp+0x850]
   5ea8b:	movsd  xmm12,QWORD PTR [rsp+0x858]
   5ea95:	divsd  xmm14,xmm5
   5ea9a:	movsd  xmm4,QWORD PTR [rsp+0x860]
   5eaa3:	movsd  xmm11,QWORD PTR [rsp+0x868]
   5eaad:	movsd  xmm3,QWORD PTR [rsp+0x870]
   5eab6:	movsd  xmm10,QWORD PTR [rsp+0x878]
   5eac0:	movsd  xmm2,QWORD PTR [rsp+0x880]
   5eac9:	movsd  xmm9,QWORD PTR [rsp+0x888]
   5ead3:	movsd  xmm1,QWORD PTR [rsp+0x890]
   5eadc:	movsd  xmm8,QWORD PTR [rsp+0x898]
   5eae6:	movsd  xmm0,QWORD PTR [rsp+0x8a0]
   5eaef:	divsd  xmm7,xmm5
   5eaf3:	movsd  QWORD PTR [rsp],xmm14
   5eaf9:	movsd  xmm14,QWORD PTR [rsp+0x8b0]
   5eb03:	divsd  xmm14,xmm5
   5eb08:	mulsd  xmm7,xmm7
   5eb0c:	divsd  xmm13,xmm5
   5eb11:	movsd  QWORD PTR [rsp+0x18],xmm14
   5eb18:	movsd  xmm14,QWORD PTR [rsp+0x8b8]
   5eb22:	divsd  xmm14,xmm5
   5eb27:	mulsd  xmm13,xmm13
   5eb2c:	addsd  xmm7,xmm13
   5eb31:	pxor   xmm13,xmm13
   5eb36:	addsd  xmm7,xmm13
   5eb3b:	divsd  xmm6,xmm5
   5eb3f:	movsd  QWORD PTR [rsp+0x28],xmm14
   5eb46:	movsd  xmm14,QWORD PTR [rsp+0x8c0]
   5eb50:	divsd  xmm14,xmm5
   5eb55:	mulsd  xmm6,xmm6
   5eb59:	divsd  xmm12,xmm5
   5eb5e:	movsd  QWORD PTR [rsp+0x50],xmm14
   5eb65:	movsd  xmm14,QWORD PTR [rsp+0x8c8]
   5eb6f:	divsd  xmm14,xmm5
   5eb74:	mulsd  xmm12,xmm12
   5eb79:	addsd  xmm6,xmm12
   5eb7e:	addsd  xmm6,xmm7
   5eb82:	divsd  xmm4,xmm5
   5eb86:	movsd  QWORD PTR [rsp+0x58],xmm14
   5eb8d:	movsd  xmm14,QWORD PTR [rsp+0x8d0]
   5eb97:	divsd  xmm14,xmm5
   5eb9c:	mulsd  xmm4,xmm4
   5eba0:	divsd  xmm11,xmm5
   5eba5:	movsd  QWORD PTR [rsp+0x60],xmm14
   5ebac:	movsd  xmm14,QWORD PTR [rsp+0x8d8]
   5ebb6:	divsd  xmm14,xmm5
   5ebbb:	mulsd  xmm11,xmm11
   5ebc0:	addsd  xmm4,xmm11
   5ebc5:	addsd  xmm4,xmm6
   5ebc9:	divsd  xmm3,xmm5
   5ebcd:	movsd  QWORD PTR [rsp+0x78],xmm14
   5ebd4:	movsd  xmm14,QWORD PTR [rsp+0x8e0]
   5ebde:	divsd  xmm14,xmm5
   5ebe3:	mulsd  xmm3,xmm3
   5ebe7:	divsd  xmm10,xmm5
   5ebec:	movsd  QWORD PTR [rsp+0x80],xmm14
   5ebf6:	movsd  xmm14,QWORD PTR [rsp+0x8e8]
   5ec00:	divsd  xmm14,xmm5
   5ec05:	mulsd  xmm10,xmm10
   5ec0a:	addsd  xmm3,xmm10
   5ec0f:	addsd  xmm3,xmm4
   5ec13:	divsd  xmm2,xmm5
   5ec17:	movsd  QWORD PTR [rsp+0x88],xmm14
   5ec21:	movsd  xmm14,QWORD PTR [rsp+0x8f0]
   5ec2b:	divsd  xmm14,xmm5
   5ec30:	mulsd  xmm2,xmm2
   5ec34:	divsd  xmm9,xmm5
   5ec39:	movsd  QWORD PTR [rsp+0xa0],xmm14
   5ec43:	movsd  xmm14,QWORD PTR [rsp+0x8f8]
   5ec4d:	divsd  xmm14,xmm5
   5ec52:	mulsd  xmm9,xmm9
   5ec57:	addsd  xmm2,xmm9
   5ec5c:	addsd  xmm2,xmm3
   5ec60:	divsd  xmm1,xmm5
   5ec64:	movsd  QWORD PTR [rsp+0x90],xmm14
   5ec6e:	movsd  xmm14,QWORD PTR [rsp+0x900]
   5ec78:	divsd  xmm14,xmm5
   5ec7d:	mulsd  xmm1,xmm1
   5ec81:	divsd  xmm8,xmm5
   5ec86:	movsd  QWORD PTR [rsp+0xa8],xmm14
   5ec90:	movsd  xmm14,QWORD PTR [rsp+0x908]
   5ec9a:	divsd  xmm14,xmm5
   5ec9f:	mulsd  xmm8,xmm8
   5eca4:	addsd  xmm1,xmm8
   5eca9:	addsd  xmm1,xmm2
   5ecad:	movsd  xmm2,QWORD PTR [rsp]
   5ecb2:	mulsd  xmm2,xmm2
   5ecb6:	divsd  xmm0,xmm5
   5ecba:	movsd  QWORD PTR [rsp+0xc0],xmm14
   5ecc4:	movsd  xmm14,QWORD PTR [rsp+0x910]
   5ecce:	divsd  xmm14,xmm5
   5ecd3:	mulsd  xmm0,xmm0
   5ecd7:	addsd  xmm0,xmm2
   5ecdb:	movsd  xmm2,QWORD PTR [rsp+0x28]
   5ece1:	mulsd  xmm2,xmm2
   5ece5:	addsd  xmm0,xmm1
   5ece9:	movsd  xmm1,QWORD PTR [rsp+0x18]
   5ecef:	mulsd  xmm1,xmm1
   5ecf3:	addsd  xmm1,xmm2
   5ecf7:	movsd  xmm2,QWORD PTR [rsp+0x58]
   5ecfd:	mulsd  xmm2,xmm2
   5ed01:	addsd  xmm1,xmm0
   5ed05:	movsd  xmm0,QWORD PTR [rsp+0x50]
   5ed0b:	movsd  QWORD PTR [rsp+0xd8],xmm14
   5ed15:	mulsd  xmm0,xmm0
   5ed19:	movsd  xmm14,QWORD PTR [rsp+0x918]
   5ed23:	divsd  xmm14,xmm5
   5ed28:	addsd  xmm0,xmm2
   5ed2c:	movsd  xmm2,QWORD PTR [rsp+0x78]
   5ed32:	mulsd  xmm2,xmm2
   5ed36:	addsd  xmm0,xmm1
   5ed3a:	movsd  xmm1,QWORD PTR [rsp+0x60]
   5ed40:	mulsd  xmm1,xmm1
   5ed44:	addsd  xmm1,xmm2
   5ed48:	movsd  xmm2,QWORD PTR [rsp+0x88]
   5ed51:	mulsd  xmm2,xmm2
   5ed55:	addsd  xmm0,xmm1
   5ed59:	movsd  xmm1,QWORD PTR [rsp+0x80]
   5ed62:	mulsd  xmm1,xmm1
   5ed66:	addsd  xmm1,xmm2
   5ed6a:	movsd  xmm2,QWORD PTR [rsp+0x90]
   5ed73:	mulsd  xmm2,xmm2
   5ed77:	addsd  xmm0,xmm1
   5ed7b:	movsd  xmm1,QWORD PTR [rsp+0xa0]
   5ed84:	movsd  QWORD PTR [rsp+0xe0],xmm14
   5ed8e:	mulsd  xmm1,xmm1
   5ed92:	movsd  xmm14,QWORD PTR [rsp+0x920]
   5ed9c:	divsd  xmm14,xmm5
   5eda1:	addsd  xmm1,xmm2
   5eda5:	movsd  xmm2,QWORD PTR [rsp+0xc0]
   5edae:	mulsd  xmm2,xmm2
   5edb2:	addsd  xmm0,xmm1
   5edb6:	movsd  xmm1,QWORD PTR [rsp+0xa8]
   5edbf:	mulsd  xmm1,xmm1
   5edc3:	addsd  xmm1,xmm2
   5edc7:	movsd  xmm2,QWORD PTR [rsp+0xe0]
   5edd0:	mulsd  xmm2,xmm2
   5edd4:	addsd  xmm0,xmm1
   5edd8:	movsd  xmm1,QWORD PTR [rsp+0xd8]
   5ede1:	mulsd  xmm1,xmm1
   5ede5:	addsd  xmm1,xmm2
   5ede9:	addsd  xmm0,xmm1
   5eded:	movsd  QWORD PTR [rsp+0xe8],xmm14
   5edf7:	movsd  xmm14,QWORD PTR [rsp+0x928]
   5ee01:	movsd  xmm1,QWORD PTR [rsp+0xe8]
   5ee0a:	divsd  xmm14,xmm5
   5ee0f:	mulsd  xmm1,xmm1
   5ee13:	movsd  QWORD PTR [rsp+0xf0],xmm14
   5ee1d:	movsd  xmm14,QWORD PTR [rsp+0x930]
   5ee27:	divsd  xmm14,xmm5
   5ee2c:	movapd xmm15,xmm14
   5ee31:	movsd  xmm14,QWORD PTR [rsp+0x938]
   5ee3b:	movsd  xmm2,QWORD PTR [rsp+0xf0]
   5ee44:	divsd  xmm14,xmm5
   5ee49:	mulsd  xmm2,xmm2
   5ee4d:	addsd  xmm1,xmm2
   5ee51:	addsd  xmm0,xmm1
   5ee55:	movapd xmm1,xmm15
   5ee5a:	mulsd  xmm1,xmm15
   5ee5f:	mulsd  xmm14,xmm14
   5ee64:	addsd  xmm1,xmm14
   5ee69:	addsd  xmm0,xmm1
   5ee6d:	ucomisd xmm0,xmm13
   5ee72:	jp     5ee7b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3e7b>
   5ee74:	movapd xmm6,xmm13
   5ee79:	je     5ee98 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3e98>
   5ee7b:	movsd  xmm1,QWORD PTR [rip+0x49dd]        # 63860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   5ee83:	movapd xmm6,xmm1
   5ee87:	maxsd  xmm6,xmm0
   5ee8b:	movapd xmm0,xmm6
   5ee8f:	call   6ad0 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5ee94:	mulsd  xmm6,xmm0
   5ee98:	mulsd  xmm6,xmm5
   5ee9c:	divsd  xmm6,QWORD PTR [rsp+0x190]
   5eea5:	movsd  xmm0,QWORD PTR [rbp+0x50]
   5eeaa:	movsd  xmm14,QWORD PTR [rsp+0x240]
   5eeb4:	movsd  xmm7,QWORD PTR [rsp+0x248]
   5eebd:	comisd xmm0,xmm6
   5eec1:	setae  BYTE PTR [rsp+0xd8]
   5eec9:	setb   BYTE PTR [rsp+0x78]
   5eece:	cmp    BYTE PTR [rsp+0x36a1],0x0
   5eed6:	je     5eee9 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3ee9>
   5eed8:	cmp    BYTE PTR [rsp+0x36a0],0x0
   5eee0:	jne    5eee9 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3ee9>
   5eee2:	mov    BYTE PTR [rbp+0x215f],0x1
   5eee9:	add    r14,0x1
   5eeed:	cmp    r14,r15
   5eef0:	jne    5ba5d <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xa5d>
   5eef6:	mov    rsi,r9
   5eef9:	mov    rax,rbp
   5eefc:	mov    r9d,ebx
   5eeff:	mov    rbp,r13
   5ef02:	mov    r11,QWORD PTR [rsp+0x68]
   5ef07:	mov    r15,QWORD PTR [rsp+0x8]
   5ef0c:	mov    r13,rax
   5ef0f:	mov    rbx,rsi
   5ef12:	mov    r10,QWORD PTR [rsp+0x188]
   5ef1a:	mov    r12,QWORD PTR [rsp+0x70]
   5ef1f:	jmp    5dda7 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2da7>
   5ef24:	mov    rdx,QWORD PTR [rsp+0x150]
   5ef2c:	movupd xmm0,XMMWORD PTR [rdx]
   5ef30:	add    rax,0x10
   5ef34:	add    rdx,0x10
   5ef38:	movaps XMMWORD PTR [rax-0x10],xmm0
   5ef3c:	cmp    r12,rax
   5ef3f:	jne    5ef2c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3f2c>
   5ef41:	test   cl,cl
   5ef43:	jne    5b332 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x332>
   5ef49:	jmp    5b31a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x31a>
   5ef4e:	xchg   ax,ax
   5ef50:	movapd xmm6,xmm10
   5ef55:	mov    BYTE PTR [rsp+0xd8],0x0
   5ef5d:	unpcklpd xmm6,xmm6
   5ef61:	movaps XMMWORD PTR [rsp+0x160],xmm6
   5ef69:	movapd xmm6,xmm9
   5ef6e:	unpcklpd xmm6,xmm6
   5ef72:	movaps XMMWORD PTR [rsp+0x170],xmm6
   5ef7a:	jmp    5dda7 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2da7>
   5ef7f:	test   r9b,r9b
   5ef82:	je     5e3a5 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x33a5>
   5ef88:	mov    rbx,QWORD PTR [rsp+0xd0]
   5ef90:	cmp    DWORD PTR [rbx+0x2158],0x5
   5ef97:	je     5e3a5 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x33a5>
   5ef9d:	lea    eax,[r15+0x1]
   5efa1:	mov    DWORD PTR [rbx+0x2158],eax
   5efa7:	jmp    5e3a5 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x33a5>
   5efac:	mov    rcx,r9
   5efaf:	mov    rax,rbp
   5efb2:	mov    rsi,r15
   5efb5:	mov    r9d,ebx
   5efb8:	mov    rbp,r13
   5efbb:	mov    r11,QWORD PTR [rsp+0x68]
   5efc0:	mov    r15,QWORD PTR [rsp+0x8]
   5efc5:	mov    r13,rax
   5efc8:	mov    r10,QWORD PTR [rsp+0x188]
   5efd0:	mov    r12,QWORD PTR [rsp+0x70]
   5efd5:	mov    rbx,rcx
   5efd8:	jmp    5dd95 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2d95>
   5efdd:	call   1170 <__stack_chk_fail@plt>

Disassembly of section .fini:
