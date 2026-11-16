
/tmp/matrix_peak_stack_audit/current_b599/current_b599_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

000000000005ae80 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE>:
   5ae80:	endbr64
   5ae84:	push   r15
   5ae86:	push   r14
   5ae88:	push   r13
   5ae8a:	push   r12
   5ae8c:	push   rbp
   5ae8d:	push   rbx
   5ae8e:	sub    rsp,0x1000
   5ae95:	or     QWORD PTR [rsp],0x0
   5ae9a:	sub    rsp,0x1000
   5aea1:	or     QWORD PTR [rsp],0x0
   5aea6:	sub    rsp,0x1000
   5aead:	or     QWORD PTR [rsp],0x0
   5aeb2:	sub    rsp,0x6b8
   5aeb9:	pxor   xmm0,xmm0
   5aebd:	movsd  xmm6,QWORD PTR [rdi+0x28]
   5aec2:	mov    QWORD PTR [rsp+0xd0],rdi
   5aeca:	mov    QWORD PTR [rsp+0x1a0],rsi
   5aed2:	mov    rax,QWORD PTR fs:0x28
   5aedb:	mov    QWORD PTR [rsp+0x36a8],rax
   5aee3:	xor    eax,eax
   5aee5:	comisd xmm0,xmm6
   5aee9:	movsd  QWORD PTR [rsp+0xb8],xmm6
   5aef2:	jb     5af11 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x91>
   5aef4:	movsd  xmm0,QWORD PTR [rip+0x8904]        # 63800 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
   5aefc:	mov    BYTE PTR [rdi+0x215e],0x1
   5af03:	movsd  QWORD PTR [rdi+0x28],xmm0
   5af08:	movsd  QWORD PTR [rsp+0xb8],xmm0
   5af11:	mov    rax,QWORD PTR [rsp+0xd0]
   5af19:	movzx  r8d,BYTE PTR [rax+0x215c]
   5af21:	mov    BYTE PTR [rax+0x215d],r8b
   5af28:	test   r8b,r8b
   5af2b:	je     5af37 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xb7>
   5af2d:	mov    DWORD PTR [rax+0x2158],0x1
   5af37:	mov    rax,QWORD PTR [rsp+0xd0]
   5af3f:	pxor   xmm2,xmm2
   5af43:	movsd  xmm1,QWORD PTR [rip+0x8a05]        # 63950 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x1d0>
   5af4b:	lea    rsi,[rsp+0x240]
   5af53:	movapd xmm0,xmm2
   5af57:	mov    QWORD PTR [rsp+0x10],rsi
   5af5c:	lea    r10,[rsp+0x2970]
   5af64:	mov    r9d,r8d
   5af67:	movsd  xmm6,QWORD PTR [rax+0x2168]
   5af6f:	mov    r13,rax
   5af72:	lea    rbx,[rax+0x1158]
   5af79:	mov    BYTE PTR [rax+0x215f],0x0
   5af80:	mov    QWORD PTR [rsp+0x150],rbx
   5af88:	lea    rbx,[rsp+0x940]
   5af90:	lea    r15,[rsp+0x1940]
   5af98:	cmpnltsd xmm0,xmm6
   5af9d:	mov    QWORD PTR [rsp+0x30],rbx
   5afa2:	lea    rbx,[rsp+0x230]
   5afaa:	lea    rbp,[rsp+0x840]
   5afb2:	mov    QWORD PTR [rsp+0x20],rbx
   5afb7:	lea    rbx,[rsp+0x3510]
   5afbf:	lea    r12,[rsp+0x340]
   5afc7:	mov    QWORD PTR [rsp+0xb0],0x0
   5afd3:	andpd  xmm1,xmm0
   5afd7:	andnpd xmm0,xmm6
   5afdb:	mov    QWORD PTR [rsp+0x1c0],rbx
   5afe3:	lea    rbx,[rsp+0x1a40]
   5afeb:	movapd xmm6,xmm0
   5afef:	movsd  xmm0,QWORD PTR [rax+0x30]
   5aff4:	mov    QWORD PTR [rsp+0x78],r10
   5aff9:	mov    r10,rsi
   5affc:	orpd   xmm6,xmm1
   5b000:	movsd  QWORD PTR [rax+0x2170],xmm2
   5b008:	movapd xmm3,xmm0
   5b00c:	movapd xmm1,xmm0
   5b010:	xorpd  xmm1,XMMWORD PTR [rip+0x8778]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   5b018:	movsd  QWORD PTR [rsp+0x190],xmm6
   5b021:	cmpltsd xmm0,xmm2
   5b026:	andpd  xmm1,xmm0
   5b02a:	andnpd xmm0,xmm3
   5b02e:	orpd   xmm0,xmm1
   5b032:	mulsd  xmm0,xmm6
   5b036:	movsd  xmm6,QWORD PTR [rsp+0xb8]
   5b03f:	maxsd  xmm6,xmm0
   5b043:	movsd  QWORD PTR [rsp+0x1b0],xmm6
   5b04c:	mov    rcx,QWORD PTR [rsp+0x30]
   5b051:	lea    rdx,[rsp+0xa40]
   5b059:	pxor   xmm0,xmm0
   5b05d:	lea    rax,[rdx-0x100]
   5b064:	nop
   5b065:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5b070:	movaps XMMWORD PTR [rax],xmm0
   5b073:	add    rax,0x20
   5b077:	movaps XMMWORD PTR [rax-0x10],xmm0
   5b07b:	cmp    rdx,rax
   5b07e:	jne    5b070 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1f0>
   5b080:	add    rdx,0x100
   5b087:	cmp    rbx,rdx
   5b08a:	jne    5b05d <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1dd>
   5b08c:	mov    rax,QWORD PTR [rsp+0xb0]
   5b094:	mov    rsi,QWORD PTR [rsp+0x1a0]
   5b09c:	movsd  xmm1,QWORD PTR [rsp+0x1b0]
   5b0a5:	mov    rdx,QWORD PTR [rsp+0x30]
   5b0aa:	shl    rax,0x4
   5b0ae:	mov    r8,rsi
   5b0b1:	movsd  xmm10,QWORD PTR [r13+rax*1+0x1058]
   5b0bb:	movsd  xmm12,QWORD PTR [r13+rax*1+0x1060]
   5b0c5:	addsd  xmm1,xmm10
   5b0ca:	unpcklpd xmm1,xmm12
   5b0cf:	xor    eax,eax
   5b0d1:	nop    DWORD PTR [rax]
   5b0d4:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5b0df:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5b0ea:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5b0f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5b100:	movupd xmm0,XMMWORD PTR [rsi+rax*1]
   5b105:	movq   xmm2,xmm0
   5b109:	psrldq xmm0,0x8
   5b10e:	movaps XMMWORD PTR [rdx+rax*2],xmm2
   5b112:	movaps XMMWORD PTR [rdx+rax*2+0x10],xmm0
   5b117:	add    rax,0x10
   5b11b:	cmp    rax,0x80
   5b121:	jne    5b100 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x280>
   5b123:	movapd xmm0,XMMWORD PTR [rcx]
   5b127:	add    rcx,0x110
   5b12e:	sub    rsi,0xffffffffffffff80
   5b132:	add    rdx,0x100
   5b139:	subpd  xmm0,xmm1
   5b13d:	movaps XMMWORD PTR [rcx-0x110],xmm0
   5b144:	cmp    rbx,rcx
   5b147:	jne    5b0cf <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x24f>
   5b149:	mov    rax,QWORD PTR [rsp+0x10]
   5b14e:	pxor   xmm0,xmm0
   5b152:	movaps XMMWORD PTR [rax],xmm0
   5b155:	add    rax,0x20
   5b159:	movaps XMMWORD PTR [rax-0x10],xmm0
   5b15d:	cmp    r12,rax
   5b160:	jne    5b152 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2d2>
   5b162:	movzx  ecx,BYTE PTR [r13+0x2161]
   5b16a:	mov    rax,QWORD PTR [rsp+0x10]
   5b16f:	test   cl,cl
   5b171:	jne    5eda7 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3f27>
   5b177:	movq   xmm0,QWORD PTR [rip+0x8811]        # 63990 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x210>
   5b17f:	jmp    5b18e <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x30e>
   5b181:	movaps XMMWORD PTR [rax+0x10],xmm0
   5b185:	add    rax,0x20
   5b189:	cmp    rax,r12
   5b18c:	je     5b19a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x31a>
   5b18e:	lea    rdx,[rax+0x10]
   5b192:	movaps XMMWORD PTR [rax],xmm0
   5b195:	cmp    rdx,r12
   5b198:	jne    5b181 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x301>
   5b19a:	movsd  xmm0,QWORD PTR [rip+0x87ae]        # 63950 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x1d0>
   5b1a2:	addsd  xmm0,QWORD PTR [r10]
   5b1a7:	movsd  QWORD PTR [r10],xmm0
   5b1ac:	mov    rdx,QWORD PTR [rsp+0x20]
   5b1b1:	mov    rdi,QWORD PTR [rsp+0x10]
   5b1b6:	mov    esi,0x10
   5b1bb:	mov    QWORD PTR [rsp+0x230],0x0
   5b1c7:	call   f9d0 <_ZN4Base6Matrix9Numerical10VectorNorm9ScaleLoopINS0_7ComplexIdEELm16ELm0EE7computeERKNS0_6VectorIS5_Lm16EEEmRd.part.0.constprop.0>
   5b1cc:	pxor   xmm0,xmm0
   5b1d0:	movsd  xmm5,QWORD PTR [rsp+0x230]
   5b1d9:	ucomisd xmm5,xmm0
   5b1dd:	jp     5b1e5 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x365>
   5b1df:	je     5b61f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x79f>
   5b1e5:	movsd  xmm15,QWORD PTR [rsp+0x2a0]
   5b1ef:	movsd  xmm6,QWORD PTR [rsp+0x240]
   5b1f8:	movsd  xmm14,QWORD PTR [rsp+0x248]
   5b202:	movsd  xmm4,QWORD PTR [rsp+0x250]
   5b20b:	movsd  xmm13,QWORD PTR [rsp+0x258]
   5b215:	divsd  xmm15,xmm5
   5b21a:	movsd  xmm3,QWORD PTR [rsp+0x260]
   5b223:	movsd  xmm11,QWORD PTR [rsp+0x268]
   5b22d:	movsd  xmm2,QWORD PTR [rsp+0x270]
   5b236:	movsd  xmm9,QWORD PTR [rsp+0x278]
   5b240:	movsd  xmm1,QWORD PTR [rsp+0x280]
   5b249:	movsd  xmm8,QWORD PTR [rsp+0x288]
   5b253:	movsd  xmm0,QWORD PTR [rsp+0x290]
   5b25c:	movsd  xmm7,QWORD PTR [rsp+0x298]
   5b265:	divsd  xmm6,xmm5
   5b269:	movsd  QWORD PTR [rsp],xmm15
   5b26f:	movsd  xmm15,QWORD PTR [rsp+0x2a8]
   5b279:	divsd  xmm15,xmm5
   5b27e:	mulsd  xmm6,xmm6
   5b282:	divsd  xmm14,xmm5
   5b287:	movsd  QWORD PTR [rsp+0x8],xmm15
   5b28e:	movsd  xmm15,QWORD PTR [rsp+0x2b0]
   5b298:	divsd  xmm15,xmm5
   5b29d:	mulsd  xmm14,xmm14
   5b2a2:	addsd  xmm6,xmm14
   5b2a7:	pxor   xmm14,xmm14
   5b2ac:	addsd  xmm6,xmm14
   5b2b1:	divsd  xmm4,xmm5
   5b2b5:	movsd  QWORD PTR [rsp+0x18],xmm15
   5b2bc:	movsd  xmm15,QWORD PTR [rsp+0x2b8]
   5b2c6:	divsd  xmm15,xmm5
   5b2cb:	mulsd  xmm4,xmm4
   5b2cf:	divsd  xmm13,xmm5
   5b2d4:	movsd  QWORD PTR [rsp+0x28],xmm15
   5b2db:	movsd  xmm15,QWORD PTR [rsp+0x2c0]
   5b2e5:	divsd  xmm15,xmm5
   5b2ea:	mulsd  xmm13,xmm13
   5b2ef:	addsd  xmm4,xmm13
   5b2f4:	addsd  xmm4,xmm6
   5b2f8:	divsd  xmm3,xmm5
   5b2fc:	movsd  QWORD PTR [rsp+0x38],xmm15
   5b303:	movsd  xmm15,QWORD PTR [rsp+0x2c8]
   5b30d:	divsd  xmm15,xmm5
   5b312:	mulsd  xmm3,xmm3
   5b316:	divsd  xmm11,xmm5
   5b31b:	movsd  QWORD PTR [rsp+0x40],xmm15
   5b322:	movsd  xmm15,QWORD PTR [rsp+0x2d0]
   5b32c:	divsd  xmm15,xmm5
   5b331:	mulsd  xmm11,xmm11
   5b336:	addsd  xmm3,xmm11
   5b33b:	addsd  xmm3,xmm4
   5b33f:	divsd  xmm2,xmm5
   5b343:	movsd  QWORD PTR [rsp+0x48],xmm15
   5b34a:	movsd  xmm15,QWORD PTR [rsp+0x2d8]
   5b354:	divsd  xmm15,xmm5
   5b359:	mulsd  xmm2,xmm2
   5b35d:	divsd  xmm9,xmm5
   5b362:	movsd  QWORD PTR [rsp+0x50],xmm15
   5b369:	movsd  xmm15,QWORD PTR [rsp+0x2e0]
   5b373:	divsd  xmm15,xmm5
   5b378:	mulsd  xmm9,xmm9
   5b37d:	addsd  xmm2,xmm9
   5b382:	addsd  xmm2,xmm3
   5b386:	divsd  xmm1,xmm5
   5b38a:	movsd  QWORD PTR [rsp+0x58],xmm15
   5b391:	movsd  xmm15,QWORD PTR [rsp+0x2e8]
   5b39b:	divsd  xmm15,xmm5
   5b3a0:	mulsd  xmm1,xmm1
   5b3a4:	divsd  xmm8,xmm5
   5b3a9:	movsd  QWORD PTR [rsp+0x60],xmm15
   5b3b0:	movsd  xmm15,QWORD PTR [rsp+0x2f0]
   5b3ba:	divsd  xmm15,xmm5
   5b3bf:	mulsd  xmm8,xmm8
   5b3c4:	addsd  xmm1,xmm8
   5b3c9:	addsd  xmm1,xmm2
   5b3cd:	movsd  xmm2,QWORD PTR [rsp+0x8]
   5b3d3:	mulsd  xmm2,xmm2
   5b3d7:	divsd  xmm0,xmm5
   5b3db:	movsd  QWORD PTR [rsp+0x68],xmm15
   5b3e2:	movsd  xmm15,QWORD PTR [rsp+0x2f8]
   5b3ec:	divsd  xmm15,xmm5
   5b3f1:	mulsd  xmm0,xmm0
   5b3f5:	divsd  xmm7,xmm5
   5b3f9:	movsd  QWORD PTR [rsp+0x70],xmm15
   5b400:	movsd  xmm15,QWORD PTR [rsp+0x300]
   5b40a:	divsd  xmm15,xmm5
   5b40f:	mulsd  xmm7,xmm7
   5b413:	addsd  xmm0,xmm7
   5b417:	addsd  xmm0,xmm1
   5b41b:	movsd  xmm1,QWORD PTR [rsp]
   5b420:	mulsd  xmm1,xmm1
   5b424:	addsd  xmm1,xmm2
   5b428:	movsd  xmm2,QWORD PTR [rsp+0x28]
   5b42e:	mulsd  xmm2,xmm2
   5b432:	addsd  xmm1,xmm0
   5b436:	movsd  xmm0,QWORD PTR [rsp+0x18]
   5b43c:	movsd  QWORD PTR [rsp+0x80],xmm15
   5b446:	mulsd  xmm0,xmm0
   5b44a:	movsd  xmm15,QWORD PTR [rsp+0x308]
   5b454:	divsd  xmm15,xmm5
   5b459:	addsd  xmm0,xmm2
   5b45d:	movsd  xmm2,QWORD PTR [rsp+0x40]
   5b463:	mulsd  xmm2,xmm2
   5b467:	addsd  xmm0,xmm1
   5b46b:	movsd  xmm1,QWORD PTR [rsp+0x38]
   5b471:	mulsd  xmm1,xmm1
   5b475:	addsd  xmm1,xmm2
   5b479:	movsd  xmm2,QWORD PTR [rsp+0x50]
   5b47f:	mulsd  xmm2,xmm2
   5b483:	addsd  xmm1,xmm0
   5b487:	movsd  xmm0,QWORD PTR [rsp+0x48]
   5b48d:	mulsd  xmm0,xmm0
   5b491:	addsd  xmm0,xmm2
   5b495:	movsd  xmm2,QWORD PTR [rsp+0x60]
   5b49b:	mulsd  xmm2,xmm2
   5b49f:	addsd  xmm0,xmm1
   5b4a3:	movsd  xmm1,QWORD PTR [rsp+0x58]
   5b4a9:	movsd  QWORD PTR [rsp+0x88],xmm15
   5b4b3:	mulsd  xmm1,xmm1
   5b4b7:	movsd  xmm15,QWORD PTR [rsp+0x310]
   5b4c1:	divsd  xmm15,xmm5
   5b4c6:	addsd  xmm1,xmm2
   5b4ca:	movsd  xmm2,QWORD PTR [rsp+0x70]
   5b4d0:	mulsd  xmm2,xmm2
   5b4d4:	addsd  xmm0,xmm1
   5b4d8:	movsd  xmm1,QWORD PTR [rsp+0x68]
   5b4de:	mulsd  xmm1,xmm1
   5b4e2:	addsd  xmm1,xmm2
   5b4e6:	addsd  xmm0,xmm1
   5b4ea:	movsd  xmm1,QWORD PTR [rsp+0x80]
   5b4f3:	mulsd  xmm1,xmm1
   5b4f7:	movsd  QWORD PTR [rsp+0xa0],xmm15
   5b501:	movsd  xmm15,QWORD PTR [rsp+0x318]
   5b50b:	divsd  xmm15,xmm5
   5b510:	movsd  QWORD PTR [rsp+0x90],xmm15
   5b51a:	movsd  xmm15,QWORD PTR [rsp+0x320]
   5b524:	divsd  xmm15,xmm5
   5b529:	movsd  QWORD PTR [rsp+0xa8],xmm15
   5b533:	movsd  xmm15,QWORD PTR [rsp+0x328]
   5b53d:	divsd  xmm15,xmm5
   5b542:	movsd  QWORD PTR [rsp+0xc0],xmm15
   5b54c:	movsd  xmm15,QWORD PTR [rsp+0x330]
   5b556:	divsd  xmm15,xmm5
   5b55b:	movsd  QWORD PTR [rsp+0xc8],xmm15
   5b565:	movsd  xmm15,QWORD PTR [rsp+0x338]
   5b56f:	movsd  xmm2,QWORD PTR [rsp+0x88]
   5b578:	mulsd  xmm2,xmm2
   5b57c:	divsd  xmm15,xmm5
   5b581:	addsd  xmm1,xmm2
   5b585:	movsd  xmm2,QWORD PTR [rsp+0x90]
   5b58e:	mulsd  xmm2,xmm2
   5b592:	addsd  xmm0,xmm1
   5b596:	movsd  xmm1,QWORD PTR [rsp+0xa0]
   5b59f:	mulsd  xmm1,xmm1
   5b5a3:	addsd  xmm1,xmm2
   5b5a7:	movsd  xmm2,QWORD PTR [rsp+0xc0]
   5b5b0:	mulsd  xmm2,xmm2
   5b5b4:	addsd  xmm0,xmm1
   5b5b8:	movsd  xmm1,QWORD PTR [rsp+0xa8]
   5b5c1:	mulsd  xmm1,xmm1
   5b5c5:	addsd  xmm1,xmm2
   5b5c9:	addsd  xmm0,xmm1
   5b5cd:	movsd  xmm1,QWORD PTR [rsp+0xc8]
   5b5d6:	mulsd  xmm15,xmm15
   5b5db:	mulsd  xmm1,xmm1
   5b5df:	addsd  xmm1,xmm15
   5b5e4:	addsd  xmm0,xmm1
   5b5e8:	ucomisd xmm0,xmm14
   5b5ed:	jp     5b5f6 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x776>
   5b5ef:	movapd xmm1,xmm14
   5b5f4:	je     5b617 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x797>
   5b5f6:	movsd  xmm1,QWORD PTR [rip+0x8202]        # 63800 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
   5b5fe:	movapd xmm6,xmm1
   5b602:	maxsd  xmm6,xmm0
   5b606:	movapd xmm0,xmm6
   5b60a:	call   6a10 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5b60f:	movapd xmm1,xmm0
   5b613:	mulsd  xmm1,xmm6
   5b617:	mulsd  xmm5,xmm1
   5b61b:	movapd xmm0,xmm5
   5b61f:	mov    rax,QWORD PTR [rsp+0x10]
   5b624:	unpcklpd xmm0,xmm0
   5b628:	movapd xmm1,XMMWORD PTR [rax]
   5b62c:	add    rax,0x10
   5b630:	divpd  xmm1,xmm0
   5b634:	movaps XMMWORD PTR [rax-0x10],xmm1
   5b638:	cmp    r12,rax
   5b63b:	jne    5b628 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x7a8>
   5b63d:	mov    QWORD PTR [rsp+0x1968],0x0
   5b649:	mov    rdx,QWORD PTR [rsp+0x78]
   5b64e:	lea    rax,[rsp+0x1970]
   5b656:	pxor   xmm0,xmm0
   5b65a:	nop    WORD PTR [rax+rax*1+0x0]
   5b660:	movaps XMMWORD PTR [rax],xmm0
   5b663:	add    rax,0x20
   5b667:	movaps XMMWORD PTR [rax-0x10],xmm0
   5b66b:	cmp    rdx,rax
   5b66e:	jne    5b660 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x7e0>
   5b670:	lea    rdx,[r15+0x18b0]
   5b677:	mov    rax,QWORD PTR [rsp+0x78]
   5b67c:	pxor   xmm0,xmm0
   5b680:	mov    QWORD PTR [rsp+0x38],rdx
   5b685:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5b690:	movaps XMMWORD PTR [rax],xmm0
   5b693:	add    rax,0x20
   5b697:	movaps XMMWORD PTR [rax-0x10],xmm0
   5b69b:	cmp    rdx,rax
   5b69e:	jne    5b690 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x810>
   5b6a0:	mov    rsi,QWORD PTR [rsp+0x38]
   5b6a5:	lea    rdx,[r15+0x19c0]
   5b6ac:	pxor   xmm0,xmm0
   5b6b0:	mov    QWORD PTR [rsp+0x40],rdx
   5b6b5:	sub    rdx,rsi
   5b6b8:	mov    rax,rsi
   5b6bb:	and    edx,0x10
   5b6be:	je     5b6d8 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x858>
   5b6c0:	mov    rax,QWORD PTR [rsp+0x38]
   5b6c5:	movaps XMMWORD PTR [r15+0x18b0],xmm0
   5b6cd:	add    rax,0x10
   5b6d1:	cmp    QWORD PTR [rsp+0x40],rax
   5b6d6:	je     5b6ea <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x86a>
   5b6d8:	movaps XMMWORD PTR [rax],xmm0
   5b6db:	add    rax,0x20
   5b6df:	movaps XMMWORD PTR [rax-0x10],xmm0
   5b6e3:	cmp    QWORD PTR [rsp+0x40],rax
   5b6e8:	jne    5b6d8 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x858>
   5b6ea:	mov    rsi,QWORD PTR [rsp+0x40]
   5b6ef:	lea    rdx,[r15+0x1ad0]
   5b6f6:	pxor   xmm0,xmm0
   5b6fa:	mov    QWORD PTR [rsp+0x48],rdx
   5b6ff:	sub    rdx,rsi
   5b702:	mov    rax,rsi
   5b705:	and    edx,0x10
   5b708:	je     5b722 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x8a2>
   5b70a:	mov    rax,QWORD PTR [rsp+0x40]
   5b70f:	movaps XMMWORD PTR [r15+0x19c0],xmm0
   5b717:	add    rax,0x10
   5b71b:	cmp    QWORD PTR [rsp+0x48],rax
   5b720:	je     5b734 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x8b4>
   5b722:	movaps XMMWORD PTR [rax],xmm0
   5b725:	add    rax,0x20
   5b729:	movaps XMMWORD PTR [rax-0x10],xmm0
   5b72d:	cmp    QWORD PTR [rsp+0x48],rax
   5b732:	jne    5b722 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x8a2>
   5b734:	mov    rcx,QWORD PTR [rsp+0x48]
   5b739:	lea    rsi,[r15+0x1bd0]
   5b740:	pxor   xmm0,xmm0
   5b744:	mov    rdx,rsi
   5b747:	sub    rdx,rcx
   5b74a:	mov    rax,rcx
   5b74d:	and    edx,0x10
   5b750:	je     5b768 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x8e8>
   5b752:	mov    rax,QWORD PTR [rsp+0x48]
   5b757:	movaps XMMWORD PTR [r15+0x1ad0],xmm0
   5b75f:	add    rax,0x10
   5b763:	cmp    rsi,rax
   5b766:	je     5b778 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x8f8>
   5b768:	movaps XMMWORD PTR [rax],xmm0
   5b76b:	add    rax,0x20
   5b76f:	movaps XMMWORD PTR [rax-0x10],xmm0
   5b773:	cmp    rsi,rax
   5b776:	jne    5b768 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x8e8>
   5b778:	mov    rdi,QWORD PTR [rsp+0x1c0]
   5b780:	xor    eax,eax
   5b782:	mov    ecx,0x10
   5b787:	pxor   xmm0,xmm0
   5b78b:	rep stos QWORD PTR [rdi],rax
   5b78e:	lea    rax,[r15+0x1d50]
   5b795:	mov    QWORD PTR [rsp+0x198],rax
   5b79d:	lea    rax,[r15+0x1c50]
   5b7a4:	movaps XMMWORD PTR [rax],xmm0
   5b7a7:	add    rax,0x20
   5b7ab:	movaps XMMWORD PTR [rax-0x10],xmm0
   5b7af:	cmp    QWORD PTR [rsp+0x198],rax
   5b7b7:	jne    5b7a4 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x924>
   5b7b9:	movdqu xmm1,XMMWORD PTR [r13+0x10]
   5b7bf:	movsd  xmm2,QWORD PTR [r13+0x38]
   5b7c5:	movsd  xmm3,QWORD PTR [rip+0x826b]        # 63a38 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x2b8>
   5b7cd:	mov    rax,QWORD PTR [r13+0x20]
   5b7d1:	movdqa xmm0,xmm1
   5b7d5:	mov    r11,QWORD PTR [r13+0x8]
   5b7d9:	movsd  xmm6,QWORD PTR [rsp+0xb8]
   5b7e2:	shufpd xmm0,xmm1,0x1
   5b7e7:	movaps XMMWORD PTR [rsp+0x1940],xmm0
   5b7ef:	pxor   xmm0,xmm0
   5b7f3:	cmpnltsd xmm0,xmm2
   5b7f8:	mov    QWORD PTR [rsp+0x1950],rax
   5b800:	movsd  QWORD PTR [rsp+0x1958],xmm6
   5b809:	andpd  xmm3,xmm0
   5b80d:	andnpd xmm0,xmm2
   5b811:	movapd xmm2,xmm0
   5b815:	orpd   xmm2,xmm3
   5b819:	movsd  QWORD PTR [rsp+0x1960],xmm2
   5b822:	test   r11,r11
   5b825:	je     5edd8 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3f58>
   5b82b:	cmp    rax,0x2
   5b82f:	movapd xmm6,xmm10
   5b834:	mov    BYTE PTR [rsp+0xd8],0x0
   5b83c:	movsd  xmm14,QWORD PTR [rsp+0x240]
   5b846:	sbb    rax,rax
   5b849:	unpcklpd xmm6,xmm6
   5b84d:	xor    r14d,r14d
   5b850:	mov    QWORD PTR [rsp+0x180],r10
   5b858:	add    rax,0x2
   5b85c:	movaps XMMWORD PTR [rsp+0x160],xmm6
   5b864:	movapd xmm6,xmm12
   5b869:	movsd  xmm7,QWORD PTR [rsp+0x248]
   5b872:	mov    QWORD PTR [rsp+0x188],r8
   5b87a:	unpcklpd xmm6,xmm6
   5b87e:	mov    QWORD PTR [rsp+0x8],r15
   5b883:	mov    QWORD PTR [rsp+0x158],rax
   5b88b:	mov    rax,r13
   5b88e:	mov    r13,rbp
   5b891:	mov    QWORD PTR [rsp+0x1c8],rsi
   5b899:	mov    rsi,rbx
   5b89c:	mov    rbp,rax
   5b89f:	mov    ebx,r9d
   5b8a2:	mov    BYTE PTR [rsp+0x70],r9b
   5b8a7:	mov    r9,rsi
   5b8aa:	mov    QWORD PTR [rsp+0x68],r12
   5b8af:	mov    r12,r11
   5b8b2:	movq   QWORD PTR [rsp+0x1a8],xmm1
   5b8bb:	movhps QWORD PTR [rsp+0xc8],xmm1
   5b8c3:	movaps XMMWORD PTR [rsp+0x170],xmm6
   5b8cb:	cmp    BYTE PTR [rsp+0x70],0x0
   5b8d0:	je     5ee34 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3fb4>
   5b8d6:	lea    rdx,[rsp+0x440]
   5b8de:	lea    r8,[rsp+0x340]
   5b8e6:	pxor   xmm0,xmm0
   5b8ea:	mov    QWORD PTR [rsp+0x60],rdx
   5b8ef:	mov    rax,r8
   5b8f2:	movaps XMMWORD PTR [rax],xmm0
   5b8f5:	add    rax,0x20
   5b8f9:	movaps XMMWORD PTR [rax-0x10],xmm0
   5b8fd:	cmp    rdx,rax
   5b900:	jne    5b8f2 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xa72>
   5b902:	pxor   xmm6,xmm6
   5b906:	movsd  xmm2,QWORD PTR [rip+0x7ee2]        # 637f0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x70>
   5b90e:	movsd  xmm1,QWORD PTR [rsp+0x1958]
   5b917:	lea    rax,[rsp+0x218]
   5b91f:	movapd xmm0,xmm6
   5b923:	mov    rdi,QWORD PTR [rsp+0x10]
   5b928:	mov    rdx,rax
   5b92b:	mov    esi,0x10
   5b930:	cmpnltsd xmm0,xmm1
   5b935:	comisd xmm6,xmm1
   5b939:	mov    QWORD PTR [rsp+0x3698],0x0
   5b945:	mov    BYTE PTR [rsp+0x36a1],0x0
   5b94d:	mov    QWORD PTR [rsp+0x1b8],rax
   5b955:	andpd  xmm2,xmm0
   5b959:	andnpd xmm0,xmm1
   5b95d:	movapd xmm1,xmm6
   5b961:	setae  BYTE PTR [rsp+0x36a2]
   5b969:	movsd  QWORD PTR [rsp+0x218],xmm6
   5b972:	orpd   xmm0,xmm2
   5b976:	movsd  QWORD PTR [rsp+0xc0],xmm0
   5b97f:	movsd  xmm0,QWORD PTR [rsp+0x1960]
   5b988:	cmpltsd xmm1,xmm0
   5b98d:	andpd  xmm0,xmm1
   5b991:	movapd xmm1,xmm6
   5b995:	movsd  QWORD PTR [rsp+0x18],xmm0
   5b99b:	movsd  xmm0,QWORD PTR [rsp+0x1968]
   5b9a4:	cmpltsd xmm1,xmm0
   5b9a9:	andpd  xmm0,xmm1
   5b9ad:	movsd  QWORD PTR [rsp+0x28],xmm0
   5b9b3:	call   f9d0 <_ZN4Base6Matrix9Numerical10VectorNorm9ScaleLoopINS0_7ComplexIdEELm16ELm0EE7computeERKNS0_6VectorIS5_Lm16EEEmRd.part.0.constprop.0>
   5b9b8:	movsd  xmm5,QWORD PTR [rsp+0x218]
   5b9c1:	ucomisd xmm5,xmm6
   5b9c5:	jp     5b9d1 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xb51>
   5b9c7:	movapd xmm0,xmm6
   5b9cb:	je     5bba1 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xd21>
   5b9d1:	divsd  xmm14,xmm5
   5b9d6:	mov    rdx,QWORD PTR [rsp+0x20]
   5b9db:	movsd  xmm13,QWORD PTR [rsp+0x2a8]
   5b9e5:	mov    esi,0x10
   5b9ea:	movsd  xmm1,QWORD PTR [rsp+0x250]
   5b9f3:	mov    rdi,QWORD PTR [rsp+0x10]
   5b9f8:	movsd  xmm12,QWORD PTR [rsp+0x258]
   5ba02:	movsd  xmm0,QWORD PTR [rsp+0x260]
   5ba0b:	movsd  xmm11,QWORD PTR [rsp+0x268]
   5ba15:	movsd  xmm6,QWORD PTR [rsp+0x270]
   5ba1e:	movsd  xmm10,QWORD PTR [rsp+0x278]
   5ba28:	movsd  xmm4,QWORD PTR [rsp+0x280]
   5ba31:	movsd  xmm9,QWORD PTR [rsp+0x288]
   5ba3b:	movsd  xmm3,QWORD PTR [rsp+0x290]
   5ba44:	movsd  xmm8,QWORD PTR [rsp+0x298]
   5ba4e:	movsd  xmm2,QWORD PTR [rsp+0x2a0]
   5ba57:	divsd  xmm13,xmm5
   5ba5c:	mulsd  xmm14,xmm14
   5ba61:	divsd  xmm7,xmm5
   5ba65:	movsd  QWORD PTR [rsp],xmm13
   5ba6b:	movsd  xmm13,QWORD PTR [rsp+0x2b0]
   5ba75:	divsd  xmm13,xmm5
   5ba7a:	mulsd  xmm7,xmm7
   5ba7e:	addsd  xmm14,xmm7
   5ba83:	pxor   xmm7,xmm7
   5ba87:	addsd  xmm14,xmm7
   5ba8c:	divsd  xmm1,xmm5
   5ba90:	movapd xmm15,xmm13
   5ba95:	movsd  xmm13,QWORD PTR [rsp+0x2b8]
   5ba9f:	divsd  xmm12,xmm5
   5baa4:	mulsd  xmm1,xmm1
   5baa8:	divsd  xmm0,xmm5
   5baac:	mulsd  xmm12,xmm12
   5bab1:	addsd  xmm1,xmm12
   5bab6:	addsd  xmm1,xmm14
   5babb:	divsd  xmm11,xmm5
   5bac0:	mulsd  xmm0,xmm0
   5bac4:	divsd  xmm6,xmm5
   5bac8:	mulsd  xmm11,xmm11
   5bacd:	addsd  xmm0,xmm11
   5bad2:	addsd  xmm0,xmm1
   5bad6:	movsd  xmm1,QWORD PTR [rsp]
   5badb:	mulsd  xmm1,xmm1
   5badf:	divsd  xmm10,xmm5
   5bae4:	mulsd  xmm6,xmm6
   5bae8:	divsd  xmm4,xmm5
   5baec:	mulsd  xmm10,xmm10
   5baf1:	addsd  xmm6,xmm10
   5baf6:	addsd  xmm0,xmm6
   5bafa:	divsd  xmm9,xmm5
   5baff:	mulsd  xmm4,xmm4
   5bb03:	divsd  xmm3,xmm5
   5bb07:	mulsd  xmm9,xmm9
   5bb0c:	addsd  xmm4,xmm9
   5bb11:	addsd  xmm0,xmm4
   5bb15:	divsd  xmm8,xmm5
   5bb1a:	mulsd  xmm3,xmm3
   5bb1e:	divsd  xmm2,xmm5
   5bb22:	mulsd  xmm8,xmm8
   5bb27:	addsd  xmm3,xmm8
   5bb2c:	addsd  xmm0,xmm3
   5bb30:	divsd  xmm13,xmm5
   5bb35:	mulsd  xmm2,xmm2
   5bb39:	addsd  xmm1,xmm2
   5bb3d:	addsd  xmm0,xmm1
   5bb41:	movapd xmm1,xmm15
   5bb46:	mulsd  xmm1,xmm15
   5bb4b:	mulsd  xmm13,xmm13
   5bb50:	addsd  xmm1,xmm13
   5bb55:	addsd  xmm0,xmm1
   5bb59:	movsd  QWORD PTR [rsp+0x230],xmm0
   5bb62:	movapd xmm0,xmm5
   5bb66:	call   be80 <_ZN4Base6Matrix9Numerical10VectorNorm14SquaredSumLoopINS0_7ComplexIdEELm16ELm8EE7computeERKNS0_6VectorIS5_Lm16EEEmdRd.part.0>
   5bb6b:	movsd  xmm2,QWORD PTR [rsp+0x230]
   5bb74:	ucomisd xmm2,xmm7
   5bb78:	jp     5bb80 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xd00>
   5bb7a:	movapd xmm0,xmm7
   5bb7e:	je     5bb9d <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xd1d>
   5bb80:	movsd  xmm1,QWORD PTR [rip+0x7c78]        # 63800 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
   5bb88:	movapd xmm6,xmm1
   5bb8c:	maxsd  xmm6,xmm2
   5bb90:	movapd xmm0,xmm6
   5bb94:	call   6a10 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5bb99:	mulsd  xmm0,xmm6
   5bb9d:	mulsd  xmm0,xmm5
   5bba1:	mulsd  xmm0,QWORD PTR [rsp+0x18]
   5bba7:	mov    rdx,QWORD PTR [rsp+0x60]
   5bbac:	addsd  xmm0,QWORD PTR [rsp+0x28]
   5bbb2:	mov    r15,rdx
   5bbb5:	mov    rax,rdx
   5bbb8:	movsd  QWORD PTR [rsp+0xe0],xmm0
   5bbc1:	movapd xmm0,XMMWORD PTR [r8]
   5bbc6:	add    r8,0x10
   5bbca:	add    rax,0x10
   5bbce:	movaps XMMWORD PTR [rax-0x10],xmm0
   5bbd2:	cmp    rdx,r8
   5bbd5:	jne    5bbc1 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xd41>
   5bbd7:	lea    rax,[rsp+0x540]
   5bbdf:	lea    r10,[rsp+0x640]
   5bbe7:	pxor   xmm0,xmm0
   5bbeb:	mov    QWORD PTR [rsp+0x50],rax
   5bbf0:	movaps XMMWORD PTR [rax],xmm0
   5bbf3:	add    rax,0x20
   5bbf7:	movaps XMMWORD PTR [rax-0x10],xmm0
   5bbfb:	cmp    r10,rax
   5bbfe:	jne    5bbf0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xd70>
   5bc00:	mov    rdx,QWORD PTR [rsp+0x50]
   5bc05:	mov    rsi,QWORD PTR [rsp+0x60]
   5bc0a:	movaps XMMWORD PTR [rsp+0x540],xmm0
   5bc12:	mov    rdi,QWORD PTR [rsp+0x30]
   5bc17:	movaps XMMWORD PTR [rsp+0x550],xmm0
   5bc1f:	movaps XMMWORD PTR [rsp+0x560],xmm0
   5bc27:	movaps XMMWORD PTR [rsp+0x570],xmm0
   5bc2f:	movaps XMMWORD PTR [rsp+0x580],xmm0
   5bc37:	movaps XMMWORD PTR [rsp+0x590],xmm0
   5bc3f:	movaps XMMWORD PTR [rsp+0x5a0],xmm0
   5bc47:	movaps XMMWORD PTR [rsp+0x5b0],xmm0
   5bc4f:	movaps XMMWORD PTR [rsp+0x5c0],xmm0
   5bc57:	movaps XMMWORD PTR [rsp+0x5d0],xmm0
   5bc5f:	movaps XMMWORD PTR [rsp+0x5e0],xmm0
   5bc67:	movaps XMMWORD PTR [rsp+0x5f0],xmm0
   5bc6f:	movaps XMMWORD PTR [rsp+0x600],xmm0
   5bc77:	movaps XMMWORD PTR [rsp+0x610],xmm0
   5bc7f:	movaps XMMWORD PTR [rsp+0x620],xmm0
   5bc87:	movaps XMMWORD PTR [rsp+0x630],xmm0
   5bc8f:	call   1be00 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeINS0_7ComplexIdEELm16EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS7_XT0_EEERSC_m.part.0>
   5bc94:	mov    rax,QWORD PTR [rsp+0x10]
   5bc99:	mov    rcx,QWORD PTR [rsp+0x68]
   5bc9e:	movapd xmm0,XMMWORD PTR [rax]
   5bca2:	subpd  xmm0,XMMWORD PTR [rdx]
   5bca6:	add    rax,0x10
   5bcaa:	add    rdx,0x10
   5bcae:	movaps XMMWORD PTR [rdx-0x10],xmm0
   5bcb2:	cmp    rcx,rax
   5bcb5:	jne    5bc9e <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xe1e>
   5bcb7:	mov    rdx,QWORD PTR [rsp+0x20]
   5bcbc:	mov    rdi,QWORD PTR [rsp+0x50]
   5bcc1:	mov    esi,0x10
   5bcc6:	mov    QWORD PTR [rsp+0x230],0x0
   5bcd2:	call   f9d0 <_ZN4Base6Matrix9Numerical10VectorNorm9ScaleLoopINS0_7ComplexIdEELm16ELm0EE7computeERKNS0_6VectorIS5_Lm16EEEmRd.part.0.constprop.0>
   5bcd7:	pxor   xmm0,xmm0
   5bcdb:	movsd  xmm5,QWORD PTR [rsp+0x230]
   5bce4:	ucomisd xmm5,xmm0
   5bce8:	jp     5bcf0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xe70>
   5bcea:	je     5c141 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x12c1>
   5bcf0:	movsd  xmm14,QWORD PTR [rsp+0x5a8]
   5bcfa:	movsd  xmm7,QWORD PTR [rsp+0x540]
   5bd03:	movsd  xmm13,QWORD PTR [rsp+0x548]
   5bd0d:	movsd  xmm6,QWORD PTR [rsp+0x550]
   5bd16:	movsd  xmm12,QWORD PTR [rsp+0x558]
   5bd20:	divsd  xmm14,xmm5
   5bd25:	movsd  xmm4,QWORD PTR [rsp+0x560]
   5bd2e:	movsd  xmm11,QWORD PTR [rsp+0x568]
   5bd38:	movsd  xmm3,QWORD PTR [rsp+0x570]
   5bd41:	movsd  xmm10,QWORD PTR [rsp+0x578]
   5bd4b:	movsd  xmm2,QWORD PTR [rsp+0x580]
   5bd54:	movsd  xmm9,QWORD PTR [rsp+0x588]
   5bd5e:	movsd  xmm1,QWORD PTR [rsp+0x590]
   5bd67:	movsd  xmm8,QWORD PTR [rsp+0x598]
   5bd71:	movsd  xmm0,QWORD PTR [rsp+0x5a0]
   5bd7a:	divsd  xmm7,xmm5
   5bd7e:	movsd  QWORD PTR [rsp],xmm14
   5bd84:	movsd  xmm14,QWORD PTR [rsp+0x5b0]
   5bd8e:	divsd  xmm14,xmm5
   5bd93:	mulsd  xmm7,xmm7
   5bd97:	divsd  xmm13,xmm5
   5bd9c:	movsd  QWORD PTR [rsp+0x18],xmm14
   5bda3:	movsd  xmm14,QWORD PTR [rsp+0x5b8]
   5bdad:	divsd  xmm14,xmm5
   5bdb2:	mulsd  xmm13,xmm13
   5bdb7:	addsd  xmm7,xmm13
   5bdbc:	pxor   xmm13,xmm13
   5bdc1:	addsd  xmm7,xmm13
   5bdc6:	divsd  xmm6,xmm5
   5bdca:	movsd  QWORD PTR [rsp+0x28],xmm14
   5bdd1:	movsd  xmm14,QWORD PTR [rsp+0x5c0]
   5bddb:	divsd  xmm14,xmm5
   5bde0:	mulsd  xmm6,xmm6
   5bde4:	divsd  xmm12,xmm5
   5bde9:	movsd  QWORD PTR [rsp+0x58],xmm14
   5bdf0:	movsd  xmm14,QWORD PTR [rsp+0x5c8]
   5bdfa:	divsd  xmm14,xmm5
   5bdff:	mulsd  xmm12,xmm12
   5be04:	addsd  xmm6,xmm12
   5be09:	addsd  xmm6,xmm7
   5be0d:	divsd  xmm4,xmm5
   5be11:	movsd  QWORD PTR [rsp+0x80],xmm14
   5be1b:	movsd  xmm14,QWORD PTR [rsp+0x5d0]
   5be25:	divsd  xmm14,xmm5
   5be2a:	mulsd  xmm4,xmm4
   5be2e:	divsd  xmm11,xmm5
   5be33:	movsd  QWORD PTR [rsp+0x88],xmm14
   5be3d:	movsd  xmm14,QWORD PTR [rsp+0x5d8]
   5be47:	divsd  xmm14,xmm5
   5be4c:	mulsd  xmm11,xmm11
   5be51:	addsd  xmm4,xmm11
   5be56:	addsd  xmm4,xmm6
   5be5a:	divsd  xmm3,xmm5
   5be5e:	movsd  QWORD PTR [rsp+0xa0],xmm14
   5be68:	movsd  xmm14,QWORD PTR [rsp+0x5e0]
   5be72:	divsd  xmm14,xmm5
   5be77:	mulsd  xmm3,xmm3
   5be7b:	divsd  xmm10,xmm5
   5be80:	movsd  QWORD PTR [rsp+0x90],xmm14
   5be8a:	movsd  xmm14,QWORD PTR [rsp+0x5e8]
   5be94:	divsd  xmm14,xmm5
   5be99:	mulsd  xmm10,xmm10
   5be9e:	addsd  xmm3,xmm10
   5bea3:	addsd  xmm3,xmm4
   5bea7:	divsd  xmm2,xmm5
   5beab:	movsd  QWORD PTR [rsp+0xa8],xmm14
   5beb5:	movsd  xmm14,QWORD PTR [rsp+0x5f0]
   5bebf:	divsd  xmm14,xmm5
   5bec4:	mulsd  xmm2,xmm2
   5bec8:	divsd  xmm9,xmm5
   5becd:	movsd  QWORD PTR [rsp+0xe8],xmm14
   5bed7:	movsd  xmm14,QWORD PTR [rsp+0x5f8]
   5bee1:	divsd  xmm14,xmm5
   5bee6:	mulsd  xmm9,xmm9
   5beeb:	addsd  xmm2,xmm9
   5bef0:	addsd  xmm2,xmm3
   5bef4:	divsd  xmm1,xmm5
   5bef8:	movsd  QWORD PTR [rsp+0xf0],xmm14
   5bf02:	movsd  xmm14,QWORD PTR [rsp+0x600]
   5bf0c:	divsd  xmm14,xmm5
   5bf11:	mulsd  xmm1,xmm1
   5bf15:	divsd  xmm8,xmm5
   5bf1a:	movsd  QWORD PTR [rsp+0xf8],xmm14
   5bf24:	movsd  xmm14,QWORD PTR [rsp+0x608]
   5bf2e:	divsd  xmm14,xmm5
   5bf33:	mulsd  xmm8,xmm8
   5bf38:	addsd  xmm1,xmm8
   5bf3d:	addsd  xmm1,xmm2
   5bf41:	movsd  xmm2,QWORD PTR [rsp]
   5bf46:	mulsd  xmm2,xmm2
   5bf4a:	divsd  xmm0,xmm5
   5bf4e:	movsd  QWORD PTR [rsp+0x100],xmm14
   5bf58:	movsd  xmm14,QWORD PTR [rsp+0x610]
   5bf62:	divsd  xmm14,xmm5
   5bf67:	mulsd  xmm0,xmm0
   5bf6b:	addsd  xmm0,xmm2
   5bf6f:	movsd  xmm2,QWORD PTR [rsp+0x28]
   5bf75:	mulsd  xmm2,xmm2
   5bf79:	addsd  xmm0,xmm1
   5bf7d:	movsd  xmm1,QWORD PTR [rsp+0x18]
   5bf83:	mulsd  xmm1,xmm1
   5bf87:	addsd  xmm1,xmm2
   5bf8b:	movsd  xmm2,QWORD PTR [rsp+0x80]
   5bf94:	mulsd  xmm2,xmm2
   5bf98:	addsd  xmm1,xmm0
   5bf9c:	movsd  xmm0,QWORD PTR [rsp+0x58]
   5bfa2:	movsd  QWORD PTR [rsp+0x108],xmm14
   5bfac:	mulsd  xmm0,xmm0
   5bfb0:	movsd  xmm14,QWORD PTR [rsp+0x618]
   5bfba:	divsd  xmm14,xmm5
   5bfbf:	addsd  xmm0,xmm2
   5bfc3:	movsd  xmm2,QWORD PTR [rsp+0xa0]
   5bfcc:	mulsd  xmm2,xmm2
   5bfd0:	addsd  xmm0,xmm1
   5bfd4:	movsd  xmm1,QWORD PTR [rsp+0x88]
   5bfdd:	mulsd  xmm1,xmm1
   5bfe1:	addsd  xmm1,xmm2
   5bfe5:	movsd  xmm2,QWORD PTR [rsp+0xa8]
   5bfee:	mulsd  xmm2,xmm2
   5bff2:	addsd  xmm0,xmm1
   5bff6:	movsd  xmm1,QWORD PTR [rsp+0x90]
   5bfff:	mulsd  xmm1,xmm1
   5c003:	addsd  xmm1,xmm2
   5c007:	movsd  xmm2,QWORD PTR [rsp+0xf0]
   5c010:	mulsd  xmm2,xmm2
   5c014:	addsd  xmm0,xmm1
   5c018:	movsd  xmm1,QWORD PTR [rsp+0xe8]
   5c021:	movsd  QWORD PTR [rsp+0x110],xmm14
   5c02b:	mulsd  xmm1,xmm1
   5c02f:	movsd  xmm14,QWORD PTR [rsp+0x620]
   5c039:	divsd  xmm14,xmm5
   5c03e:	addsd  xmm1,xmm2
   5c042:	movsd  xmm2,QWORD PTR [rsp+0x100]
   5c04b:	mulsd  xmm2,xmm2
   5c04f:	addsd  xmm0,xmm1
   5c053:	movsd  xmm1,QWORD PTR [rsp+0xf8]
   5c05c:	mulsd  xmm1,xmm1
   5c060:	addsd  xmm1,xmm2
   5c064:	movsd  xmm2,QWORD PTR [rsp+0x110]
   5c06d:	mulsd  xmm2,xmm2
   5c071:	addsd  xmm0,xmm1
   5c075:	movsd  xmm1,QWORD PTR [rsp+0x108]
   5c07e:	mulsd  xmm1,xmm1
   5c082:	addsd  xmm1,xmm2
   5c086:	addsd  xmm0,xmm1
   5c08a:	movsd  QWORD PTR [rsp+0x118],xmm14
   5c094:	movsd  xmm14,QWORD PTR [rsp+0x628]
   5c09e:	movsd  xmm1,QWORD PTR [rsp+0x118]
   5c0a7:	divsd  xmm14,xmm5
   5c0ac:	mulsd  xmm1,xmm1
   5c0b0:	movsd  QWORD PTR [rsp+0x120],xmm14
   5c0ba:	movsd  xmm14,QWORD PTR [rsp+0x630]
   5c0c4:	divsd  xmm14,xmm5
   5c0c9:	movapd xmm15,xmm14
   5c0ce:	movsd  xmm14,QWORD PTR [rsp+0x638]
   5c0d8:	movsd  xmm2,QWORD PTR [rsp+0x120]
   5c0e1:	divsd  xmm14,xmm5
   5c0e6:	mulsd  xmm2,xmm2
   5c0ea:	addsd  xmm1,xmm2
   5c0ee:	addsd  xmm0,xmm1
   5c0f2:	movapd xmm1,xmm15
   5c0f7:	mulsd  xmm1,xmm15
   5c0fc:	mulsd  xmm14,xmm14
   5c101:	addsd  xmm1,xmm14
   5c106:	addsd  xmm0,xmm1
   5c10a:	ucomisd xmm0,xmm13
   5c10f:	jp     5c118 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1298>
   5c111:	movapd xmm1,xmm13
   5c116:	je     5c139 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x12b9>
   5c118:	movsd  xmm1,QWORD PTR [rip+0x76e0]        # 63800 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
   5c120:	movapd xmm6,xmm1
   5c124:	maxsd  xmm6,xmm0
   5c128:	movapd xmm0,xmm6
   5c12c:	call   6a10 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5c131:	movapd xmm1,xmm0
   5c135:	mulsd  xmm1,xmm6
   5c139:	mulsd  xmm5,xmm1
   5c13d:	movapd xmm0,xmm5
   5c141:	movsd  xmm6,QWORD PTR [rsp+0xe0]
   5c14a:	movsd  QWORD PTR [rsp+0x3690],xmm0
   5c153:	comisd xmm6,xmm0
   5c157:	setae  al
   5c15a:	mov    BYTE PTR [rsp+0x36a0],al
   5c161:	cmp    QWORD PTR [rsp+0xc8],0x0
   5c16a:	je     5d754 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x28d4>
   5c170:	mov    rcx,QWORD PTR [rsp+0x1a8]
   5c178:	mov    esi,0x10
   5c17d:	mov    edx,0x1
   5c182:	mov    BYTE PTR [rsp+0x1d7],bl
   5c189:	mov    QWORD PTR [rsp+0x1d8],r14
   5c191:	mov    r11,r10
   5c194:	cmp    rcx,0x10
   5c198:	mov    QWORD PTR [rsp+0x1e0],r15
   5c1a0:	cmovbe rsi,rcx
   5c1a4:	mov    QWORD PTR [rsp+0x1e8],r12
   5c1ac:	mov    QWORD PTR [rsp+0x1f0],r9
   5c1b4:	test   rsi,rsi
   5c1b7:	mov    QWORD PTR [rsp+0x1f8],rbp
   5c1bf:	cmovne rdx,rsi
   5c1c3:	xor    ebp,ebp
   5c1c5:	mov    QWORD PTR [rsp+0x18],rdx
   5c1ca:	test   al,al
   5c1cc:	jne    5d3fb <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x257b>
   5c1d2:	cmp    BYTE PTR [rsp+0x36a1],0x0
   5c1da:	jne    5d3fb <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x257b>
   5c1e0:	mov    rbx,QWORD PTR [rsp+0x40]
   5c1e5:	mov    rdx,QWORD PTR [rsp+0x48]
   5c1ea:	pxor   xmm0,xmm0
   5c1ee:	movsd  xmm7,QWORD PTR [rsp+0x3690]
   5c1f7:	sub    rdx,rbx
   5c1fa:	mov    rax,rbx
   5c1fd:	movapd xmm2,xmm7
   5c201:	and    edx,0x10
   5c204:	unpcklpd xmm2,xmm2
   5c208:	je     5c21a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x139a>
   5c20a:	mov    rax,QWORD PTR [rsp+0x40]
   5c20f:	add    rax,0x10
   5c213:	cmp    QWORD PTR [rsp+0x48],rax
   5c218:	je     5c22f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x13af>
   5c21a:	mov    rdx,QWORD PTR [rsp+0x48]
   5c21f:	movaps XMMWORD PTR [rax],xmm0
   5c222:	add    rax,0x20
   5c226:	movaps XMMWORD PTR [rax-0x10],xmm0
   5c22a:	cmp    rdx,rax
   5c22d:	jne    5c21f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x139f>
   5c22f:	comisd xmm7,QWORD PTR [rip+0x77f1]        # 63a28 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x2a8>
   5c237:	pxor   xmm1,xmm1
   5c23b:	mov    QWORD PTR [rsp+0x3308],0x0
   5c247:	movsd  QWORD PTR [rsp+0x3300],xmm7
   5c250:	jb     5c25e <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x13de>
   5c252:	movsd  xmm1,QWORD PTR [rip+0x76f6]        # 63950 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x1d0>
   5c25a:	divsd  xmm1,xmm7
   5c25e:	mov    rax,QWORD PTR [rsp+0x50]
   5c263:	unpcklpd xmm1,xmm1
   5c267:	lea    rdx,[rsp+0x1970]
   5c26f:	jmp    5c286 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1406>
   5c271:	mulpd  xmm0,xmm1
   5c275:	add    rax,0x10
   5c279:	add    rdx,0x10
   5c27d:	movaps XMMWORD PTR [rdx-0x10],xmm0
   5c281:	cmp    rax,r11
   5c284:	je     5c2a9 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1429>
   5c286:	comisd xmm7,QWORD PTR [rip+0x779a]        # 63a28 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x2a8>
   5c28e:	movapd xmm0,XMMWORD PTR [rax]
   5c292:	jae    5c271 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x13f1>
   5c294:	divpd  xmm0,xmm2
   5c298:	add    rax,0x10
   5c29c:	add    rdx,0x10
   5c2a0:	movaps XMMWORD PTR [rdx-0x10],xmm0
   5c2a4:	cmp    rax,r11
   5c2a7:	jne    5c286 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1406>
   5c2a9:	lea    r12,[rsp+0x740]
   5c2b1:	mov    rax,r11
   5c2b4:	pxor   xmm0,xmm0
   5c2b8:	movaps XMMWORD PTR [rax],xmm0
   5c2bb:	add    rax,0x20
   5c2bf:	movaps XMMWORD PTR [rax-0x10],xmm0
   5c2c3:	cmp    r12,rax
   5c2c6:	jne    5c2b8 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1438>
   5c2c8:	mov    rax,r12
   5c2cb:	pxor   xmm0,xmm0
   5c2cf:	movaps XMMWORD PTR [rax],xmm0
   5c2d2:	add    rax,0x20
   5c2d6:	movaps XMMWORD PTR [rax-0x10],xmm0
   5c2da:	cmp    r13,rax
   5c2dd:	jne    5c2cf <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x144f>
   5c2df:	mov    QWORD PTR [rsp+0x58],r11
   5c2e4:	mov    r14,QWORD PTR [rsp+0x38]
   5c2e9:	xor    edi,edi
   5c2eb:	xor    r8d,r8d
   5c2ee:	mov    QWORD PTR [rsp+0x208],rbp
   5c2f6:	movzx  edx,BYTE PTR [rsp+0x70]
   5c2fb:	xor    r15d,r15d
   5c2fe:	lea    rax,[rsp+0x1970]
   5c306:	mov    rbx,QWORD PTR [rsp+0x1c8]
   5c30e:	mov    rbp,QWORD PTR [rsp+0x40]
   5c313:	movsd  QWORD PTR [rsp+0x200],xmm7
   5c31c:	jmp    5c357 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x14d7>
   5c31e:	nop
   5c31f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5c32a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5c335:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5c340:	mov    r14,rsi
   5c343:	add    rbx,0x8
   5c347:	add    rax,0x100
   5c34d:	cmp    r15,0x10
   5c351:	je     5d429 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x25a9>
   5c357:	cmp    r15,QWORD PTR [rsp+0x18]
   5c35c:	mov    QWORD PTR [rsp],r15
   5c360:	lea    rsi,[r14+0x10]
   5c364:	setb   cl
   5c367:	add    r15,0x1
   5c36b:	and    cl,dl
   5c36d:	mov    BYTE PTR [rsp+0x28],cl
   5c371:	je     5c340 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x14c0>
   5c373:	mov    rdx,QWORD PTR [rsp+0x58]
   5c378:	mov    rcx,rax
   5c37b:	pxor   xmm3,xmm3
   5c37f:	nop
   5c380:	movapd xmm1,XMMWORD PTR [rcx]
   5c384:	add    rdx,0x10
   5c388:	add    rcx,0x10
   5c38c:	movapd xmm0,xmm1
   5c390:	movapd xmm2,xmm1
   5c394:	shufpd xmm0,xmm1,0x1
   5c399:	mulpd  xmm0,xmm3
   5c39d:	subpd  xmm2,xmm0
   5c3a1:	addpd  xmm0,xmm1
   5c3a5:	movsd  xmm0,xmm2
   5c3a9:	movaps XMMWORD PTR [rdx-0x10],xmm0
   5c3ad:	cmp    rdx,r12
   5c3b0:	jne    5c380 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1500>
   5c3b2:	mov    rsi,QWORD PTR [rsp+0x58]
   5c3b7:	mov    rdi,QWORD PTR [rsp+0x30]
   5c3bc:	mov    rdx,r12
   5c3bf:	mov    QWORD PTR [rsp+0x80],rax
   5c3c7:	movaps XMMWORD PTR [rsp+0x740],xmm3
   5c3cf:	movaps XMMWORD PTR [rsp+0x750],xmm3
   5c3d7:	movaps XMMWORD PTR [rsp+0x760],xmm3
   5c3df:	movaps XMMWORD PTR [rsp+0x770],xmm3
   5c3e7:	movaps XMMWORD PTR [rsp+0x780],xmm3
   5c3ef:	movaps XMMWORD PTR [rsp+0x790],xmm3
   5c3f7:	movaps XMMWORD PTR [rsp+0x7a0],xmm3
   5c3ff:	movaps XMMWORD PTR [rsp+0x7b0],xmm3
   5c407:	movaps XMMWORD PTR [rsp+0x7c0],xmm3
   5c40f:	movaps XMMWORD PTR [rsp+0x7d0],xmm3
   5c417:	movaps XMMWORD PTR [rsp+0x7e0],xmm3
   5c41f:	movaps XMMWORD PTR [rsp+0x7f0],xmm3
   5c427:	movaps XMMWORD PTR [rsp+0x800],xmm3
   5c42f:	movaps XMMWORD PTR [rsp+0x810],xmm3
   5c437:	movaps XMMWORD PTR [rsp+0x820],xmm3
   5c43f:	movaps XMMWORD PTR [rsp+0x830],xmm3
   5c447:	call   1be00 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeINS0_7ComplexIdEELm16EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS7_XT0_EEERSC_m.part.0>
   5c44c:	mov    rdx,QWORD PTR [rsp+0x20]
   5c451:	mov    esi,0x10
   5c456:	mov    rdi,r12
   5c459:	mov    QWORD PTR [rsp+0x230],0x0
   5c465:	call   f9d0 <_ZN4Base6Matrix9Numerical10VectorNorm9ScaleLoopINS0_7ComplexIdEELm16ELm0EE7computeERKNS0_6VectorIS5_Lm16EEEmRd.part.0.constprop.0>
   5c46a:	pxor   xmm0,xmm0
   5c46e:	movsd  xmm5,QWORD PTR [rsp+0x230]
   5c477:	mov    rax,QWORD PTR [rsp+0x80]
   5c47f:	ucomisd xmm5,xmm0
   5c483:	jp     5c48f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x160f>
   5c485:	movapd xmm6,xmm0
   5c489:	je     5c90f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1a8f>
   5c48f:	movsd  xmm14,QWORD PTR [rsp+0x7a8]
   5c499:	movsd  xmm7,QWORD PTR [rsp+0x740]
   5c4a2:	movsd  xmm13,QWORD PTR [rsp+0x748]
   5c4ac:	movsd  xmm6,QWORD PTR [rsp+0x750]
   5c4b5:	movsd  xmm12,QWORD PTR [rsp+0x758]
   5c4bf:	divsd  xmm14,xmm5
   5c4c4:	movsd  xmm4,QWORD PTR [rsp+0x760]
   5c4cd:	movsd  xmm11,QWORD PTR [rsp+0x768]
   5c4d7:	movsd  xmm3,QWORD PTR [rsp+0x770]
   5c4e0:	movsd  xmm10,QWORD PTR [rsp+0x778]
   5c4ea:	movsd  xmm2,QWORD PTR [rsp+0x780]
   5c4f3:	movsd  xmm9,QWORD PTR [rsp+0x788]
   5c4fd:	movsd  xmm1,QWORD PTR [rsp+0x790]
   5c506:	movsd  xmm8,QWORD PTR [rsp+0x798]
   5c510:	movsd  xmm0,QWORD PTR [rsp+0x7a0]
   5c519:	divsd  xmm7,xmm5
   5c51d:	movsd  QWORD PTR [rsp+0x80],xmm14
   5c527:	movsd  xmm14,QWORD PTR [rsp+0x7b0]
   5c531:	divsd  xmm14,xmm5
   5c536:	mulsd  xmm7,xmm7
   5c53a:	divsd  xmm13,xmm5
   5c53f:	movsd  QWORD PTR [rsp+0x88],xmm14
   5c549:	movsd  xmm14,QWORD PTR [rsp+0x7b8]
   5c553:	divsd  xmm14,xmm5
   5c558:	mulsd  xmm13,xmm13
   5c55d:	addsd  xmm7,xmm13
   5c562:	pxor   xmm13,xmm13
   5c567:	addsd  xmm7,xmm13
   5c56c:	divsd  xmm6,xmm5
   5c570:	movsd  QWORD PTR [rsp+0xa0],xmm14
   5c57a:	movsd  xmm14,QWORD PTR [rsp+0x7c0]
   5c584:	divsd  xmm14,xmm5
   5c589:	mulsd  xmm6,xmm6
   5c58d:	divsd  xmm12,xmm5
   5c592:	movsd  QWORD PTR [rsp+0x90],xmm14
   5c59c:	movsd  xmm14,QWORD PTR [rsp+0x7c8]
   5c5a6:	divsd  xmm14,xmm5
   5c5ab:	mulsd  xmm12,xmm12
   5c5b0:	addsd  xmm6,xmm12
   5c5b5:	addsd  xmm6,xmm7
   5c5b9:	divsd  xmm4,xmm5
   5c5bd:	movsd  QWORD PTR [rsp+0xa8],xmm14
   5c5c7:	movsd  xmm14,QWORD PTR [rsp+0x7d0]
   5c5d1:	divsd  xmm14,xmm5
   5c5d6:	mulsd  xmm4,xmm4
   5c5da:	divsd  xmm11,xmm5
   5c5df:	movsd  QWORD PTR [rsp+0xe8],xmm14
   5c5e9:	movsd  xmm14,QWORD PTR [rsp+0x7d8]
   5c5f3:	divsd  xmm14,xmm5
   5c5f8:	mulsd  xmm11,xmm11
   5c5fd:	addsd  xmm4,xmm11
   5c602:	addsd  xmm4,xmm6
   5c606:	divsd  xmm3,xmm5
   5c60a:	movsd  QWORD PTR [rsp+0xf0],xmm14
   5c614:	movsd  xmm14,QWORD PTR [rsp+0x7e0]
   5c61e:	divsd  xmm14,xmm5
   5c623:	mulsd  xmm3,xmm3
   5c627:	divsd  xmm10,xmm5
   5c62c:	movsd  QWORD PTR [rsp+0xf8],xmm14
   5c636:	movsd  xmm14,QWORD PTR [rsp+0x7e8]
   5c640:	divsd  xmm14,xmm5
   5c645:	mulsd  xmm10,xmm10
   5c64a:	addsd  xmm3,xmm10
   5c64f:	addsd  xmm3,xmm4
   5c653:	divsd  xmm2,xmm5
   5c657:	movsd  QWORD PTR [rsp+0x100],xmm14
   5c661:	movsd  xmm14,QWORD PTR [rsp+0x7f0]
   5c66b:	divsd  xmm14,xmm5
   5c670:	mulsd  xmm2,xmm2
   5c674:	divsd  xmm9,xmm5
   5c679:	movsd  QWORD PTR [rsp+0x108],xmm14
   5c683:	movsd  xmm14,QWORD PTR [rsp+0x7f8]
   5c68d:	divsd  xmm14,xmm5
   5c692:	mulsd  xmm9,xmm9
   5c697:	addsd  xmm2,xmm9
   5c69c:	addsd  xmm2,xmm3
   5c6a0:	divsd  xmm1,xmm5
   5c6a4:	movsd  QWORD PTR [rsp+0x110],xmm14
   5c6ae:	movsd  xmm14,QWORD PTR [rsp+0x800]
   5c6b8:	divsd  xmm14,xmm5
   5c6bd:	mulsd  xmm1,xmm1
   5c6c1:	divsd  xmm8,xmm5
   5c6c6:	movsd  QWORD PTR [rsp+0x118],xmm14
   5c6d0:	movsd  xmm14,QWORD PTR [rsp+0x808]
   5c6da:	divsd  xmm14,xmm5
   5c6df:	mulsd  xmm8,xmm8
   5c6e4:	addsd  xmm1,xmm8
   5c6e9:	addsd  xmm1,xmm2
   5c6ed:	movsd  xmm2,QWORD PTR [rsp+0x80]
   5c6f6:	mulsd  xmm2,xmm2
   5c6fa:	divsd  xmm0,xmm5
   5c6fe:	movsd  QWORD PTR [rsp+0x120],xmm14
   5c708:	movsd  xmm14,QWORD PTR [rsp+0x810]
   5c712:	divsd  xmm14,xmm5
   5c717:	mulsd  xmm0,xmm0
   5c71b:	addsd  xmm0,xmm2
   5c71f:	movsd  xmm2,QWORD PTR [rsp+0xa0]
   5c728:	mulsd  xmm2,xmm2
   5c72c:	addsd  xmm0,xmm1
   5c730:	movsd  xmm1,QWORD PTR [rsp+0x88]
   5c739:	mulsd  xmm1,xmm1
   5c73d:	addsd  xmm1,xmm2
   5c741:	movsd  xmm2,QWORD PTR [rsp+0xa8]
   5c74a:	mulsd  xmm2,xmm2
   5c74e:	addsd  xmm1,xmm0
   5c752:	movsd  xmm0,QWORD PTR [rsp+0x90]
   5c75b:	movsd  QWORD PTR [rsp+0x128],xmm14
   5c765:	mulsd  xmm0,xmm0
   5c769:	movsd  xmm14,QWORD PTR [rsp+0x818]
   5c773:	divsd  xmm14,xmm5
   5c778:	addsd  xmm0,xmm2
   5c77c:	movsd  xmm2,QWORD PTR [rsp+0xf0]
   5c785:	mulsd  xmm2,xmm2
   5c789:	addsd  xmm0,xmm1
   5c78d:	movsd  xmm1,QWORD PTR [rsp+0xe8]
   5c796:	mulsd  xmm1,xmm1
   5c79a:	addsd  xmm1,xmm2
   5c79e:	movsd  xmm2,QWORD PTR [rsp+0x100]
   5c7a7:	mulsd  xmm2,xmm2
   5c7ab:	addsd  xmm0,xmm1
   5c7af:	movsd  xmm1,QWORD PTR [rsp+0xf8]
   5c7b8:	mulsd  xmm1,xmm1
   5c7bc:	addsd  xmm1,xmm2
   5c7c0:	movsd  xmm2,QWORD PTR [rsp+0x110]
   5c7c9:	mulsd  xmm2,xmm2
   5c7cd:	addsd  xmm0,xmm1
   5c7d1:	movsd  xmm1,QWORD PTR [rsp+0x108]
   5c7da:	movsd  QWORD PTR [rsp+0x130],xmm14
   5c7e4:	mulsd  xmm1,xmm1
   5c7e8:	movsd  xmm14,QWORD PTR [rsp+0x820]
   5c7f2:	divsd  xmm14,xmm5
   5c7f7:	addsd  xmm1,xmm2
   5c7fb:	movsd  xmm2,QWORD PTR [rsp+0x120]
   5c804:	mulsd  xmm2,xmm2
   5c808:	addsd  xmm0,xmm1
   5c80c:	movsd  xmm1,QWORD PTR [rsp+0x118]
   5c815:	mulsd  xmm1,xmm1
   5c819:	addsd  xmm1,xmm2
   5c81d:	movsd  xmm2,QWORD PTR [rsp+0x130]
   5c826:	mulsd  xmm2,xmm2
   5c82a:	addsd  xmm0,xmm1
   5c82e:	movsd  xmm1,QWORD PTR [rsp+0x128]
   5c837:	mulsd  xmm1,xmm1
   5c83b:	addsd  xmm1,xmm2
   5c83f:	addsd  xmm0,xmm1
   5c843:	movsd  QWORD PTR [rsp+0x138],xmm14
   5c84d:	movsd  xmm14,QWORD PTR [rsp+0x828]
   5c857:	divsd  xmm14,xmm5
   5c85c:	movsd  QWORD PTR [rsp+0x140],xmm14
   5c866:	movsd  xmm14,QWORD PTR [rsp+0x830]
   5c870:	divsd  xmm14,xmm5
   5c875:	movsd  QWORD PTR [rsp+0x148],xmm14
   5c87f:	movsd  xmm14,QWORD PTR [rsp+0x838]
   5c889:	movsd  xmm1,QWORD PTR [rsp+0x138]
   5c892:	movsd  xmm2,QWORD PTR [rsp+0x140]
   5c89b:	divsd  xmm14,xmm5
   5c8a0:	mulsd  xmm1,xmm1
   5c8a4:	mulsd  xmm2,xmm2
   5c8a8:	addsd  xmm1,xmm2
   5c8ac:	addsd  xmm0,xmm1
   5c8b0:	movsd  xmm1,QWORD PTR [rsp+0x148]
   5c8b9:	mulsd  xmm1,xmm1
   5c8bd:	mulsd  xmm14,xmm14
   5c8c2:	addsd  xmm14,xmm1
   5c8c7:	addsd  xmm0,xmm14
   5c8cc:	ucomisd xmm0,xmm13
   5c8d1:	jp     5c8da <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1a5a>
   5c8d3:	movapd xmm6,xmm13
   5c8d8:	je     5c90b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1a8b>
   5c8da:	movsd  xmm1,QWORD PTR [rip+0x6f1e]        # 63800 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
   5c8e2:	mov    QWORD PTR [rsp+0x80],rax
   5c8ea:	movapd xmm7,xmm1
   5c8ee:	maxsd  xmm7,xmm0
   5c8f2:	movapd xmm0,xmm7
   5c8f6:	call   6a10 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5c8fb:	mov    rax,QWORD PTR [rsp+0x80]
   5c903:	movapd xmm6,xmm0
   5c907:	mulsd  xmm6,xmm7
   5c90b:	mulsd  xmm6,xmm5
   5c90f:	mov    rsi,QWORD PTR [rsp+0x38]
   5c914:	mov    rcx,rbp
   5c917:	pxor   xmm0,xmm0
   5c91b:	sub    rcx,rsi
   5c91e:	mov    rdx,rsi
   5c921:	and    ecx,0x10
   5c924:	je     5c940 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1ac0>
   5c926:	mov    rsi,QWORD PTR [rsp+0x38]
   5c92b:	lea    rdx,[rsi+0x10]
   5c92f:	movaps XMMWORD PTR [rsi],xmm0
   5c932:	cmp    rdx,rbp
   5c935:	je     5c950 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1ad0>
   5c937:	nop    WORD PTR [rax+rax*1+0x0]
   5c940:	movaps XMMWORD PTR [rdx],xmm0
   5c943:	add    rdx,0x20
   5c947:	movaps XMMWORD PTR [rdx-0x10],xmm0
   5c94b:	cmp    rdx,rbp
   5c94e:	jne    5c940 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1ac0>
   5c950:	mov    rsi,QWORD PTR [rsp+0x8]
   5c955:	xor    ecx,ecx
   5c957:	mov    QWORD PTR [rsp+0x80],r14
   5c95f:	lea    r10,[rax+0x200]
   5c966:	lea    r11,[rsi+0x130]
   5c96d:	mov    rsi,QWORD PTR [rsp+0x158]
   5c975:	mov    r8,QWORD PTR [rsp+0x8]
   5c97a:	mov    rdi,r11
   5c97d:	lea    r9,[rsp+0x1970]
   5c985:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5c990:	pxor   xmm2,xmm2
   5c994:	mov    rdx,r9
   5c997:	mov    r14,r12
   5c99a:	movapd xmm8,xmm2
   5c99f:	movapd xmm7,xmm2
   5c9a3:	xchg   ax,ax
   5c9a5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5c9b0:	movapd xmm5,XMMWORD PTR [rdx]
   5c9b4:	movapd xmm0,XMMWORD PTR [rdx+0x10]
   5c9b9:	add    rdx,0x20
   5c9bd:	add    r14,0x20
   5c9c1:	movlpd xmm0,QWORD PTR [rdx-0x18]
   5c9c6:	movapd xmm10,XMMWORD PTR [r14-0x20]
   5c9cc:	movhpd xmm5,QWORD PTR [rdx-0x10]
   5c9d1:	movapd xmm3,xmm0
   5c9d5:	movapd xmm1,xmm5
   5c9d9:	movhpd xmm10,QWORD PTR [r14-0x10]
   5c9df:	mulpd  xmm1,xmm5
   5c9e3:	movapd xmm4,xmm5
   5c9e7:	mulpd  xmm3,xmm0
   5c9eb:	mulpd  xmm4,xmm10
   5c9f0:	addpd  xmm3,xmm1
   5c9f4:	movapd xmm1,XMMWORD PTR [rip+0x6f04]        # 63900 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x180>
   5c9fc:	xorpd  xmm1,xmm0
   5ca00:	movapd xmm0,XMMWORD PTR [r14-0x10]
   5ca06:	movlpd xmm0,QWORD PTR [r14-0x18]
   5ca0c:	movapd xmm9,xmm1
   5ca11:	addsd  xmm7,xmm3
   5ca15:	unpckhpd xmm3,xmm3
   5ca19:	mulpd  xmm9,xmm0
   5ca1e:	mulpd  xmm1,xmm10
   5ca23:	mulpd  xmm0,xmm5
   5ca27:	movapd xmm5,xmm8
   5ca2c:	addsd  xmm7,xmm3
   5ca30:	subpd  xmm4,xmm9
   5ca35:	addpd  xmm0,xmm1
   5ca39:	addsd  xmm5,xmm4
   5ca3d:	unpckhpd xmm4,xmm4
   5ca41:	movapd xmm1,xmm0
   5ca45:	unpckhpd xmm0,xmm0
   5ca49:	addsd  xmm1,xmm2
   5ca4d:	addsd  xmm5,xmm4
   5ca51:	movapd xmm2,xmm1
   5ca55:	movapd xmm8,xmm5
   5ca5a:	addsd  xmm2,xmm0
   5ca5e:	cmp    rdx,rdi
   5ca61:	jne    5c9b0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1b30>
   5ca67:	divsd  xmm8,xmm7
   5ca6c:	mov    r14,r9
   5ca6f:	mov    rdx,r12
   5ca72:	movsd  xmm0,QWORD PTR [r8+0x18b0]
   5ca7b:	divsd  xmm2,xmm7
   5ca7f:	addsd  xmm0,xmm8
   5ca84:	movapd xmm4,xmm8
   5ca89:	movsd  QWORD PTR [r8+0x18b0],xmm0
   5ca92:	movsd  xmm0,QWORD PTR [r8+0x18b8]
   5ca9b:	addsd  xmm0,xmm2
   5ca9f:	unpcklpd xmm4,xmm2
   5caa3:	unpcklpd xmm2,xmm8
   5caa8:	movsd  QWORD PTR [r8+0x18b8],xmm0
   5cab1:	nop    DWORD PTR [rax+0x0]
   5cab5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5cac0:	movapd xmm1,XMMWORD PTR [r14]
   5cac5:	movapd xmm0,XMMWORD PTR [r14]
   5caca:	add    rdx,0x10
   5cace:	add    r14,0x10
   5cad2:	unpcklpd xmm1,xmm1
   5cad6:	unpckhpd xmm0,xmm0
   5cada:	mulpd  xmm1,xmm4
   5cade:	mulpd  xmm0,xmm2
   5cae2:	movapd xmm3,xmm1
   5cae6:	subpd  xmm3,xmm0
   5caea:	addpd  xmm0,xmm1
   5caee:	movapd xmm1,XMMWORD PTR [rdx-0x10]
   5caf3:	movsd  xmm0,xmm3
   5caf7:	subpd  xmm1,xmm0
   5cafb:	movaps XMMWORD PTR [rdx-0x10],xmm1
   5caff:	cmp    r13,rdx
   5cb02:	jne    5cac0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1c40>
   5cb04:	add    rdi,0x100
   5cb0b:	add    r9,0x100
   5cb12:	add    r8,0x10
   5cb16:	cmp    rdi,r10
   5cb19:	jne    5c990 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1b10>
   5cb1f:	lea    rdx,[rcx+0x1]
   5cb23:	mov    ecx,0x1
   5cb28:	cmp    rdx,rsi
   5cb2b:	jb     5c975 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1af5>
   5cb31:	mov    rdx,QWORD PTR [rsp+0x20]
   5cb36:	mov    esi,0x10
   5cb3b:	mov    rdi,r12
   5cb3e:	mov    r14,QWORD PTR [rsp+0x80]
   5cb46:	mov    QWORD PTR [rsp+0x230],0x0
   5cb52:	mov    QWORD PTR [rsp+0x80],rax
   5cb5a:	call   f9d0 <_ZN4Base6Matrix9Numerical10VectorNorm9ScaleLoopINS0_7ComplexIdEELm16ELm0EE7computeERKNS0_6VectorIS5_Lm16EEEmRd.part.0.constprop.0>
   5cb5f:	pxor   xmm0,xmm0
   5cb63:	movsd  xmm5,QWORD PTR [rsp+0x230]
   5cb6c:	mov    rax,QWORD PTR [rsp+0x80]
   5cb74:	ucomisd xmm5,xmm0
   5cb78:	jp     5cb84 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1d04>
   5cb7a:	movapd xmm1,xmm0
   5cb7e:	je     5d015 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2195>
   5cb84:	movsd  xmm14,QWORD PTR [rsp+0x7a0]
   5cb8e:	movsd  xmm7,QWORD PTR [rsp+0x740]
   5cb97:	movsd  xmm13,QWORD PTR [rsp+0x748]
   5cba1:	movsd  xmm4,QWORD PTR [rsp+0x750]
   5cbaa:	movsd  xmm12,QWORD PTR [rsp+0x758]
   5cbb4:	divsd  xmm14,xmm5
   5cbb9:	movsd  xmm3,QWORD PTR [rsp+0x760]
   5cbc2:	movsd  xmm11,QWORD PTR [rsp+0x768]
   5cbcc:	movsd  xmm2,QWORD PTR [rsp+0x770]
   5cbd5:	movsd  xmm10,QWORD PTR [rsp+0x778]
   5cbdf:	movsd  xmm1,QWORD PTR [rsp+0x780]
   5cbe8:	movsd  xmm9,QWORD PTR [rsp+0x788]
   5cbf2:	movsd  xmm0,QWORD PTR [rsp+0x790]
   5cbfb:	movsd  xmm8,QWORD PTR [rsp+0x798]
   5cc05:	divsd  xmm7,xmm5
   5cc09:	movsd  QWORD PTR [rsp+0x80],xmm14
   5cc13:	movsd  xmm14,QWORD PTR [rsp+0x7a8]
   5cc1d:	divsd  xmm14,xmm5
   5cc22:	mulsd  xmm7,xmm7
   5cc26:	divsd  xmm13,xmm5
   5cc2b:	movsd  QWORD PTR [rsp+0x88],xmm14
   5cc35:	movsd  xmm14,QWORD PTR [rsp+0x7b0]
   5cc3f:	divsd  xmm14,xmm5
   5cc44:	mulsd  xmm13,xmm13
   5cc49:	addsd  xmm7,xmm13
   5cc4e:	pxor   xmm13,xmm13
   5cc53:	addsd  xmm7,xmm13
   5cc58:	divsd  xmm4,xmm5
   5cc5c:	movsd  QWORD PTR [rsp+0xa0],xmm14
   5cc66:	movsd  xmm14,QWORD PTR [rsp+0x7b8]
   5cc70:	divsd  xmm14,xmm5
   5cc75:	mulsd  xmm4,xmm4
   5cc79:	divsd  xmm12,xmm5
   5cc7e:	movsd  QWORD PTR [rsp+0x90],xmm14
   5cc88:	movsd  xmm14,QWORD PTR [rsp+0x7c0]
   5cc92:	divsd  xmm14,xmm5
   5cc97:	mulsd  xmm12,xmm12
   5cc9c:	addsd  xmm4,xmm12
   5cca1:	addsd  xmm4,xmm7
   5cca5:	divsd  xmm3,xmm5
   5cca9:	movsd  QWORD PTR [rsp+0xa8],xmm14
   5ccb3:	movsd  xmm14,QWORD PTR [rsp+0x7c8]
   5ccbd:	divsd  xmm14,xmm5
   5ccc2:	mulsd  xmm3,xmm3
   5ccc6:	divsd  xmm11,xmm5
   5cccb:	movsd  QWORD PTR [rsp+0xe8],xmm14
   5ccd5:	movsd  xmm14,QWORD PTR [rsp+0x7d0]
   5ccdf:	divsd  xmm14,xmm5
   5cce4:	mulsd  xmm11,xmm11
   5cce9:	addsd  xmm3,xmm11
   5ccee:	addsd  xmm3,xmm4
   5ccf2:	divsd  xmm2,xmm5
   5ccf6:	movsd  QWORD PTR [rsp+0xf0],xmm14
   5cd00:	movsd  xmm14,QWORD PTR [rsp+0x7d8]
   5cd0a:	divsd  xmm14,xmm5
   5cd0f:	mulsd  xmm2,xmm2
   5cd13:	divsd  xmm10,xmm5
   5cd18:	movsd  QWORD PTR [rsp+0xf8],xmm14
   5cd22:	movsd  xmm14,QWORD PTR [rsp+0x7e0]
   5cd2c:	divsd  xmm14,xmm5
   5cd31:	mulsd  xmm10,xmm10
   5cd36:	addsd  xmm2,xmm10
   5cd3b:	addsd  xmm2,xmm3
   5cd3f:	divsd  xmm1,xmm5
   5cd43:	movsd  QWORD PTR [rsp+0x100],xmm14
   5cd4d:	movsd  xmm14,QWORD PTR [rsp+0x7e8]
   5cd57:	divsd  xmm14,xmm5
   5cd5c:	mulsd  xmm1,xmm1
   5cd60:	divsd  xmm9,xmm5
   5cd65:	movsd  QWORD PTR [rsp+0x108],xmm14
   5cd6f:	movsd  xmm14,QWORD PTR [rsp+0x7f0]
   5cd79:	divsd  xmm14,xmm5
   5cd7e:	mulsd  xmm9,xmm9
   5cd83:	addsd  xmm1,xmm9
   5cd88:	addsd  xmm1,xmm2
   5cd8c:	movsd  xmm2,QWORD PTR [rsp+0x88]
   5cd95:	mulsd  xmm2,xmm2
   5cd99:	divsd  xmm0,xmm5
   5cd9d:	movsd  QWORD PTR [rsp+0x110],xmm14
   5cda7:	movsd  xmm14,QWORD PTR [rsp+0x7f8]
   5cdb1:	divsd  xmm14,xmm5
   5cdb6:	mulsd  xmm0,xmm0
   5cdba:	divsd  xmm8,xmm5
   5cdbf:	movsd  QWORD PTR [rsp+0x118],xmm14
   5cdc9:	movsd  xmm14,QWORD PTR [rsp+0x800]
   5cdd3:	divsd  xmm14,xmm5
   5cdd8:	mulsd  xmm8,xmm8
   5cddd:	addsd  xmm0,xmm8
   5cde2:	addsd  xmm0,xmm1
   5cde6:	movsd  xmm1,QWORD PTR [rsp+0x80]
   5cdef:	mulsd  xmm1,xmm1
   5cdf3:	addsd  xmm1,xmm2
   5cdf7:	movsd  xmm2,QWORD PTR [rsp+0x90]
   5ce00:	mulsd  xmm2,xmm2
   5ce04:	addsd  xmm1,xmm0
   5ce08:	movsd  xmm0,QWORD PTR [rsp+0xa0]
   5ce11:	movsd  QWORD PTR [rsp+0x120],xmm14
   5ce1b:	mulsd  xmm0,xmm0
   5ce1f:	movsd  xmm14,QWORD PTR [rsp+0x808]
   5ce29:	divsd  xmm14,xmm5
   5ce2e:	addsd  xmm0,xmm2
   5ce32:	movsd  xmm2,QWORD PTR [rsp+0xe8]
   5ce3b:	mulsd  xmm2,xmm2
   5ce3f:	addsd  xmm0,xmm1
   5ce43:	movsd  xmm1,QWORD PTR [rsp+0xa8]
   5ce4c:	mulsd  xmm1,xmm1
   5ce50:	addsd  xmm1,xmm2
   5ce54:	movsd  xmm2,QWORD PTR [rsp+0xf8]
   5ce5d:	mulsd  xmm2,xmm2
   5ce61:	addsd  xmm1,xmm0
   5ce65:	movsd  xmm0,QWORD PTR [rsp+0xf0]
   5ce6e:	mulsd  xmm0,xmm0
   5ce72:	addsd  xmm0,xmm2
   5ce76:	movsd  xmm2,QWORD PTR [rsp+0x108]
   5ce7f:	mulsd  xmm2,xmm2
   5ce83:	addsd  xmm0,xmm1
   5ce87:	movsd  xmm1,QWORD PTR [rsp+0x100]
   5ce90:	movsd  QWORD PTR [rsp+0x128],xmm14
   5ce9a:	mulsd  xmm1,xmm1
   5ce9e:	movsd  xmm14,QWORD PTR [rsp+0x810]
   5cea8:	divsd  xmm14,xmm5
   5cead:	addsd  xmm1,xmm2
   5ceb1:	movsd  xmm2,QWORD PTR [rsp+0x118]
   5ceba:	mulsd  xmm2,xmm2
   5cebe:	addsd  xmm0,xmm1
   5cec2:	movsd  xmm1,QWORD PTR [rsp+0x110]
   5cecb:	mulsd  xmm1,xmm1
   5cecf:	addsd  xmm1,xmm2
   5ced3:	movsd  xmm2,QWORD PTR [rsp+0x128]
   5cedc:	mulsd  xmm2,xmm2
   5cee0:	addsd  xmm0,xmm1
   5cee4:	movsd  xmm1,QWORD PTR [rsp+0x120]
   5ceed:	mulsd  xmm1,xmm1
   5cef1:	addsd  xmm1,xmm2
   5cef5:	addsd  xmm0,xmm1
   5cef9:	movsd  QWORD PTR [rsp+0x130],xmm14
   5cf03:	movsd  xmm14,QWORD PTR [rsp+0x818]
   5cf0d:	divsd  xmm14,xmm5
   5cf12:	movsd  QWORD PTR [rsp+0x138],xmm14
   5cf1c:	movsd  xmm14,QWORD PTR [rsp+0x820]
   5cf26:	divsd  xmm14,xmm5
   5cf2b:	movsd  QWORD PTR [rsp+0x140],xmm14
   5cf35:	movsd  xmm14,QWORD PTR [rsp+0x828]
   5cf3f:	divsd  xmm14,xmm5
   5cf44:	movsd  QWORD PTR [rsp+0x148],xmm14
   5cf4e:	movsd  xmm14,QWORD PTR [rsp+0x830]
   5cf58:	divsd  xmm14,xmm5
   5cf5d:	movapd xmm15,xmm14
   5cf62:	movsd  xmm14,QWORD PTR [rsp+0x838]
   5cf6c:	movsd  xmm1,QWORD PTR [rsp+0x130]
   5cf75:	movsd  xmm2,QWORD PTR [rsp+0x138]
   5cf7e:	divsd  xmm14,xmm5
   5cf83:	mulsd  xmm2,xmm2
   5cf87:	mulsd  xmm1,xmm1
   5cf8b:	addsd  xmm1,xmm2
   5cf8f:	movsd  xmm2,QWORD PTR [rsp+0x148]
   5cf98:	mulsd  xmm2,xmm2
   5cf9c:	addsd  xmm0,xmm1
   5cfa0:	movsd  xmm1,QWORD PTR [rsp+0x140]
   5cfa9:	mulsd  xmm1,xmm1
   5cfad:	addsd  xmm1,xmm2
   5cfb1:	addsd  xmm0,xmm1
   5cfb5:	movapd xmm1,xmm15
   5cfba:	mulsd  xmm1,xmm15
   5cfbf:	mulsd  xmm14,xmm14
   5cfc4:	addsd  xmm14,xmm1
   5cfc9:	addsd  xmm0,xmm14
   5cfce:	ucomisd xmm0,xmm13
   5cfd3:	jp     5cfdc <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x215c>
   5cfd5:	movapd xmm1,xmm13
   5cfda:	je     5d011 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2191>
   5cfdc:	movsd  xmm2,QWORD PTR [rip+0x681c]        # 63800 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
   5cfe4:	mov    QWORD PTR [rsp+0x80],rax
   5cfec:	movapd xmm7,xmm2
   5cff0:	movapd xmm1,xmm2
   5cff4:	maxsd  xmm7,xmm0
   5cff8:	movapd xmm0,xmm7
   5cffc:	call   6a10 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5d001:	mov    rax,QWORD PTR [rsp+0x80]
   5d009:	movapd xmm1,xmm0
   5d00d:	mulsd  xmm1,xmm7
   5d011:	mulsd  xmm1,xmm5
   5d015:	mulsd  xmm6,QWORD PTR [rip+0x6a13]        # 63a30 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x2b0>
   5d01d:	lea    rsi,[r14+0x10]
   5d021:	mov    QWORD PTR [r14+0x18],0x0
   5d029:	movsd  QWORD PTR [r14+0x10],xmm1
   5d02f:	comisd xmm6,xmm1
   5d033:	jae    5d040 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x21c0>
   5d035:	cmp    r15,QWORD PTR [rsp+0x18]
   5d03a:	jb     5d6bc <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x283c>
   5d040:	movq   xmm3,QWORD PTR [rip+0x6748]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   5d048:	cmp    QWORD PTR [rsp],0x0
   5d04d:	je     5d176 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x22f6>
   5d053:	mov    rdx,QWORD PTR [rsp+0x8]
   5d058:	lea    rdi,[r14-0x18b0]
   5d05f:	movsd  xmm2,QWORD PTR [rsp+0x31f8]
   5d068:	movq   xmm3,QWORD PTR [rip+0x6720]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   5d070:	movsd  xmm0,QWORD PTR [rsp+0x31f0]
   5d079:	mov    rcx,rdx
   5d07c:	nop    DWORD PTR [rax+0x0]
   5d080:	movsd  xmm7,QWORD PTR [rdx+0x18c8]
   5d088:	add    rdx,0x10
   5d08c:	add    rcx,0x8
   5d090:	movsd  xmm8,QWORD PTR [rdx+0x18b0]
   5d099:	movsd  xmm5,QWORD PTR [rdx+0x1ac8]
   5d0a1:	movsd  xmm4,QWORD PTR [rdx+0x1ac0]
   5d0a9:	movapd xmm9,xmm8
   5d0ae:	movapd xmm11,xmm7
   5d0b3:	movsd  xmm10,QWORD PTR [rcx+0x1bc8]
   5d0bc:	mulsd  xmm11,xmm5
   5d0c1:	mulsd  xmm9,xmm4
   5d0c6:	subsd  xmm9,xmm11
   5d0cb:	movapd xmm11,xmm0
   5d0d0:	mulsd  xmm11,xmm10
   5d0d5:	addsd  xmm9,xmm11
   5d0da:	movapd xmm11,xmm8
   5d0df:	mulsd  xmm11,xmm5
   5d0e4:	mulsd  xmm8,xmm10
   5d0e9:	movsd  QWORD PTR [rdx+0x18a0],xmm9
   5d0f2:	movapd xmm9,xmm7
   5d0f7:	mulsd  xmm9,xmm4
   5d0fc:	xorpd  xmm4,XMMWORD PTR [rip+0x668c]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   5d104:	mulsd  xmm7,xmm10
   5d109:	addsd  xmm9,xmm11
   5d10e:	movapd xmm11,xmm10
   5d113:	mulsd  xmm11,xmm2
   5d118:	addsd  xmm9,xmm11
   5d11d:	movapd xmm11,xmm0
   5d122:	mulsd  xmm11,xmm5
   5d127:	mulsd  xmm5,xmm2
   5d12b:	movsd  QWORD PTR [rdx+0x18a8],xmm9
   5d134:	movapd xmm9,xmm4
   5d139:	mulsd  xmm9,xmm2
   5d13e:	mulsd  xmm4,xmm0
   5d142:	addsd  xmm9,xmm11
   5d147:	subsd  xmm4,xmm5
   5d14b:	addsd  xmm7,xmm9
   5d150:	addsd  xmm4,xmm8
   5d155:	movsd  QWORD PTR [rdx+0x18b8],xmm7
   5d15d:	movapd xmm2,xmm7
   5d161:	movsd  QWORD PTR [rdx+0x18b0],xmm4
   5d169:	movapd xmm0,xmm4
   5d16d:	cmp    rdx,rdi
   5d170:	jne    5d080 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2200>
   5d176:	lea    rcx,[r14+0x220]
   5d17d:	mov    rdx,rbx
   5d180:	mov    rdi,r14
   5d183:	mov    QWORD PTR [rsp+0xa8],rax
   5d18b:	mov    QWORD PTR [rsp+0x80],rsi
   5d193:	movaps XMMWORD PTR [rsp+0x90],xmm3
   5d19b:	movsd  QWORD PTR [rsp+0xa0],xmm1
   5d1a4:	movsd  QWORD PTR [rsp+0x88],xmm6
   5d1ad:	call   4f2e0 <_ZN4Base6Matrix9Numerical6givensINS0_7ComplexIdEEEEvRKT_S7_RNS1_6ScalarIS5_E4RealERS5_>
   5d1b2:	movsd  xmm8,QWORD PTR [r14+0x10]
   5d1b8:	movsd  xmm7,QWORD PTR [r14+0x18]
   5d1be:	movsd  xmm0,QWORD PTR [r14+0x220]
   5d1c7:	movsd  xmm4,QWORD PTR [rbx]
   5d1cb:	movapd xmm9,xmm8
   5d1d0:	mov    rcx,QWORD PTR [rsp]
   5d1d4:	movsd  xmm5,QWORD PTR [r14+0x228]
   5d1dd:	movapd xmm2,xmm0
   5d1e1:	mov    rdx,QWORD PTR [rsp+0x38]
   5d1e6:	movsd  xmm6,QWORD PTR [rsp+0x88]
   5d1ef:	mulsd  xmm2,xmm7
   5d1f3:	imul   rcx,r15
   5d1f7:	movsd  xmm1,QWORD PTR [rsp+0xa0]
   5d200:	mov    rax,QWORD PTR [rsp+0xa8]
   5d208:	mulsd  xmm9,xmm5
   5d20d:	mov    rsi,QWORD PTR [rsp+0x80]
   5d215:	movapd xmm3,XMMWORD PTR [rsp+0x90]
   5d21e:	mulsd  xmm0,xmm8
   5d223:	mulsd  xmm5,xmm7
   5d227:	shr    rcx,1
   5d22a:	shl    rcx,0x4
   5d22e:	add    rcx,QWORD PTR [rsp+0x78]
   5d233:	addsd  xmm2,xmm9
   5d238:	movsd  xmm9,QWORD PTR [r14+0x8]
   5d23e:	mulsd  xmm9,xmm4
   5d243:	subsd  xmm0,xmm5
   5d247:	mulsd  xmm4,QWORD PTR [r14]
   5d24c:	addsd  xmm2,xmm9
   5d251:	addsd  xmm0,xmm4
   5d255:	movsd  QWORD PTR [r14+0x8],xmm2
   5d25b:	movsd  QWORD PTR [r14],xmm0
   5d260:	jmp    5d284 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2404>
   5d262:	nop    DWORD PTR [rax+rax*1+0x0]
   5d26a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5d275:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5d280:	add    rdx,0x10
   5d284:	cmp    rcx,rdx
   5d287:	je     5d29b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x241b>
   5d289:	movsd  xmm0,QWORD PTR [rdx]
   5d28d:	movsd  QWORD PTR [rcx],xmm0
   5d291:	movsd  xmm0,QWORD PTR [rdx+0x8]
   5d296:	movsd  QWORD PTR [rcx+0x8],xmm0
   5d29b:	add    rcx,0x10
   5d29f:	cmp    r14,rdx
   5d2a2:	jne    5d280 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2400>
   5d2a4:	movsd  xmm0,QWORD PTR [r14+0x220]
   5d2ad:	movsd  xmm4,QWORD PTR [r14+0x228]
   5d2b6:	movsd  xmm5,QWORD PTR [r14+0x110]
   5d2bf:	movsd  xmm7,QWORD PTR [r14+0x118]
   5d2c8:	xorpd  xmm0,xmm3
   5d2cc:	movapd xmm8,xmm4
   5d2d1:	mulsd  xmm8,xmm7
   5d2d6:	movapd xmm2,xmm0
   5d2da:	mulsd  xmm4,xmm5
   5d2de:	mulsd  xmm2,xmm5
   5d2e2:	mulsd  xmm0,xmm7
   5d2e6:	subsd  xmm2,xmm8
   5d2eb:	addsd  xmm0,xmm4
   5d2ef:	movsd  QWORD PTR [r14+0x120],xmm2
   5d2f8:	movsd  QWORD PTR [r14+0x128],xmm0
   5d301:	movsd  xmm4,QWORD PTR [rbx]
   5d305:	unpcklpd xmm4,xmm4
   5d309:	mulpd  xmm4,XMMWORD PTR [r14+0x110]
   5d312:	movaps XMMWORD PTR [r14+0x110],xmm4
   5d31a:	add    QWORD PTR [rsp+0x3698],0x1
   5d323:	comisd xmm6,xmm1
   5d327:	jae    5d41a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x259a>
   5d32d:	pxor   xmm4,xmm4
   5d331:	comisd xmm4,xmm2
   5d335:	ja     5d68a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x280a>
   5d33b:	comisd xmm4,xmm0
   5d33f:	movapd xmm1,xmm2
   5d343:	ja     5d71b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x289b>
   5d349:	movapd xmm3,xmm0
   5d34d:	movapd xmm5,xmm1
   5d351:	pxor   xmm1,xmm1
   5d355:	maxsd  xmm5,xmm3
   5d359:	ucomisd xmm5,xmm1
   5d35d:	jp     5d361 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x24e1>
   5d35f:	je     5d3bc <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x253c>
   5d361:	divsd  xmm2,xmm5
   5d365:	divsd  xmm0,xmm5
   5d369:	movapd xmm1,xmm2
   5d36d:	mulsd  xmm1,xmm2
   5d371:	mulsd  xmm0,xmm0
   5d375:	addsd  xmm0,xmm1
   5d379:	pxor   xmm1,xmm1
   5d37d:	ucomisd xmm0,xmm1
   5d381:	jp     5d385 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2505>
   5d383:	je     5d3b8 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2538>
   5d385:	movsd  xmm1,QWORD PTR [rip+0x6473]        # 63800 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
   5d38d:	mov    QWORD PTR [rsp+0x28],rax
   5d392:	mov    QWORD PTR [rsp],rsi
   5d396:	movapd xmm6,xmm1
   5d39a:	maxsd  xmm6,xmm0
   5d39e:	movapd xmm0,xmm6
   5d3a2:	call   6a10 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5d3a7:	mov    rax,QWORD PTR [rsp+0x28]
   5d3ac:	mov    rsi,QWORD PTR [rsp]
   5d3b0:	movapd xmm1,xmm0
   5d3b4:	mulsd  xmm1,xmm6
   5d3b8:	mulsd  xmm1,xmm5
   5d3bc:	comisd xmm1,QWORD PTR [rsp+0xe0]
   5d3c5:	mov    r8,r15
   5d3c8:	seta   dl
   5d3cb:	xor    edi,edi
   5d3cd:	jmp    5c340 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x14c0>
   5d3d2:	mov    r8,rbp
   5d3d5:	movsd  xmm7,QWORD PTR [rsp+0x58]
   5d3db:	mov    rbp,QWORD PTR [rsp]
   5d3df:	mov    r11,r14
   5d3e2:	mov    r13,QWORD PTR [rsp+0x28]
   5d3e7:	test   al,al
   5d3e9:	jne    5e24f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x33cf>
   5d3ef:	movzx  eax,BYTE PTR [rsp+0x70]
   5d3f4:	mov    BYTE PTR [rsp+0x36a1],al
   5d3fb:	add    rbp,0x1
   5d3ff:	cmp    rbp,QWORD PTR [rsp+0xc8]
   5d407:	je     5d724 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x28a4>
   5d40d:	movzx  eax,BYTE PTR [rsp+0x36a0]
   5d415:	jmp    5c1ca <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x134a>
   5d41a:	movzx  edi,BYTE PTR [rsp+0x28]
   5d41f:	xor    edx,edx
   5d421:	mov    r8,r15
   5d424:	jmp    5c340 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x14c0>
   5d429:	mov    QWORD PTR [rsp+0x28],r13
   5d42e:	mov    r14,QWORD PTR [rsp+0x58]
   5d433:	mov    ebx,0xf
   5d438:	pxor   xmm3,xmm3
   5d43c:	mov    rbp,QWORD PTR [rsp+0x208]
   5d444:	mov    BYTE PTR [rsp+0x80],dil
   5d44c:	movsd  xmm7,QWORD PTR [rsp+0x200]
   5d455:	mov    r12,QWORD PTR [rsp+0x198]
   5d45d:	movzx  eax,BYTE PTR [rsp+0x70]
   5d462:	mov    r13,QWORD PTR [rsp+0x8]
   5d467:	mov    QWORD PTR [rsp],rbp
   5d46b:	mov    rbp,r8
   5d46e:	movsd  QWORD PTR [rsp+0x58],xmm7
   5d474:	jmp    5d492 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2612>
   5d476:	cs nop WORD PTR [rax+rax*1+0x0]
   5d480:	sub    rbx,0x1
   5d484:	sub    r12,0x10
   5d488:	cmp    rbx,0xffffffffffffffff
   5d48c:	je     5d3d2 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2552>
   5d492:	cmp    rbx,rbp
   5d495:	movaps XMMWORD PTR [r12-0x10],xmm3
   5d49b:	setb   r15b
   5d49f:	and    r15b,al
   5d4a2:	je     5d480 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2600>
   5d4a4:	movapd xmm5,XMMWORD PTR [r12-0x2a0]
   5d4ae:	lea    rdi,[rbx+0x1]
   5d4b2:	cmp    rbx,0xf
   5d4b6:	je     5d51f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x269f>
   5d4b8:	mov    rsi,r12
   5d4bb:	mov    rdx,rdi
   5d4be:	xchg   ax,ax
   5d4c0:	mov    rax,rdx
   5d4c3:	add    rdx,0x1
   5d4c7:	cmp    rax,rbp
   5d4ca:	jae    5d515 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2695>
   5d4cc:	imul   rax,rdx
   5d4d0:	movapd xmm1,XMMWORD PTR [rsi]
   5d4d4:	shr    rax,1
   5d4d7:	add    rax,rbx
   5d4da:	shl    rax,0x4
   5d4de:	movapd xmm2,XMMWORD PTR [r13+rax*1+0x1030]
   5d4e8:	movapd xmm0,xmm2
   5d4ec:	unpcklpd xmm0,xmm2
   5d4f0:	unpckhpd xmm2,xmm2
   5d4f4:	mulpd  xmm0,xmm1
   5d4f8:	shufpd xmm1,xmm1,0x1
   5d4fd:	mulpd  xmm1,xmm2
   5d501:	movapd xmm2,xmm0
   5d505:	subpd  xmm2,xmm1
   5d509:	addpd  xmm0,xmm1
   5d50d:	movsd  xmm0,xmm2
   5d511:	subpd  xmm5,xmm0
   5d515:	add    rsi,0x10
   5d519:	cmp    rdx,0x10
   5d51d:	jne    5d4c0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2640>
   5d51f:	imul   rdi,rbx
   5d523:	pxor   xmm2,xmm2
   5d527:	shr    rdi,1
   5d52a:	lea    r9,[rbx+rdi*1+0x103]
   5d532:	shl    r9,0x4
   5d536:	add    r9,r13
   5d539:	movsd  xmm0,QWORD PTR [r9]
   5d53e:	movsd  xmm1,QWORD PTR [r9+0x8]
   5d544:	comisd xmm2,xmm0
   5d548:	ja     5d611 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2791>
   5d54e:	comisd xmm2,xmm1
   5d552:	movapd xmm6,xmm0
   5d556:	ja     5d679 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x27f9>
   5d55c:	movapd xmm2,xmm1
   5d560:	maxsd  xmm6,xmm2
   5d564:	pxor   xmm2,xmm2
   5d568:	ucomisd xmm6,xmm2
   5d56c:	jp     5d574 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x26f4>
   5d56e:	je     5d64b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x27cb>
   5d574:	divsd  xmm0,xmm6
   5d578:	divsd  xmm1,xmm6
   5d57c:	mulsd  xmm0,xmm0
   5d580:	mulsd  xmm1,xmm1
   5d584:	addsd  xmm0,xmm1
   5d588:	pxor   xmm1,xmm1
   5d58c:	ucomisd xmm0,xmm1
   5d590:	jp     5d594 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2714>
   5d592:	je     5d5bd <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x273d>
   5d594:	movsd  xmm2,QWORD PTR [rip+0x6264]        # 63800 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
   5d59c:	movapd xmm7,xmm2
   5d5a0:	movapd xmm1,xmm2
   5d5a4:	maxsd  xmm7,xmm0
   5d5a8:	movapd xmm0,xmm7
   5d5ac:	call   6a10 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5d5b1:	pxor   xmm3,xmm3
   5d5b5:	movapd xmm1,xmm0
   5d5b9:	mulsd  xmm1,xmm7
   5d5bd:	movsd  xmm7,QWORD PTR [rsp+0xc0]
   5d5c6:	mulsd  xmm6,xmm1
   5d5ca:	comisd xmm7,xmm6
   5d5ce:	ja     5d64b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x27cb>
   5d5d0:	mov    rdi,QWORD PTR [rsp+0x20]
   5d5d5:	lea    rsi,[rsp+0x220]
   5d5dd:	mov    rdx,r9
   5d5e0:	movsd  xmm0,QWORD PTR [rsp+0xc0]
   5d5e9:	movaps XMMWORD PTR [rsp+0x220],xmm5
   5d5f1:	call   5ac50 <_ZN4Base6Matrix9Numerical6ScalarINS0_7ComplexIdEEE6divideERKS4_S7_d>
   5d5f6:	mov    eax,r15d
   5d5f9:	pxor   xmm3,xmm3
   5d5fd:	movapd xmm0,XMMWORD PTR [rsp+0x230]
   5d606:	movaps XMMWORD PTR [r12-0x10],xmm0
   5d60c:	jmp    5d480 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2600>
   5d611:	comisd xmm2,xmm1
   5d615:	movapd xmm6,xmm0
   5d619:	xorpd  xmm6,XMMWORD PTR [rip+0x616f]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   5d621:	jbe    5d55c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x26dc>
   5d627:	comisd xmm1,xmm0
   5d62b:	movapd xmm2,xmm1
   5d62f:	xorpd  xmm2,XMMWORD PTR [rip+0x6159]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   5d637:	seta   al
   5d63a:	test   al,al
   5d63c:	je     5d642 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x27c2>
   5d63e:	movapd xmm2,xmm6
   5d642:	movapd xmm6,xmm2
   5d646:	jmp    5d574 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x26f4>
   5d64b:	test   rbx,rbx
   5d64e:	je     5e683 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3803>
   5d654:	sub    rbx,0x2
   5d658:	movaps XMMWORD PTR [r12-0x20],xmm3
   5d65e:	cmp    rbx,0xffffffffffffffff
   5d662:	je     5e683 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3803>
   5d668:	sub    r12,0x20
   5d66c:	xor    eax,eax
   5d66e:	movaps XMMWORD PTR [r12-0x10],xmm3
   5d674:	jmp    5d480 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2600>
   5d679:	movapd xmm2,xmm1
   5d67d:	xorpd  xmm2,XMMWORD PTR [rip+0x610b]        # 63790 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
   5d685:	jmp    5d560 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x26e0>
   5d68a:	comisd xmm4,xmm0
   5d68e:	movapd xmm1,xmm2
   5d692:	xorpd  xmm1,xmm3
   5d696:	jbe    5d349 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x24c9>
   5d69c:	comisd xmm0,xmm2
   5d6a0:	xorpd  xmm3,xmm0
   5d6a4:	movapd xmm5,xmm3
   5d6a8:	seta   dl
   5d6ab:	test   dl,dl
   5d6ad:	je     5d361 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x24e1>
   5d6b3:	movapd xmm5,xmm1
   5d6b7:	jmp    5d361 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x24e1>
   5d6bc:	comisd xmm1,QWORD PTR [rip+0x6364]        # 63a28 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x2a8>
   5d6c4:	movapd xmm3,xmm1
   5d6c8:	pxor   xmm2,xmm2
   5d6cc:	unpcklpd xmm3,xmm3
   5d6d0:	jb     5d6de <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x285e>
   5d6d2:	movsd  xmm2,QWORD PTR [rip+0x6276]        # 63950 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x1d0>
   5d6da:	divsd  xmm2,xmm1
   5d6de:	unpcklpd xmm2,xmm2
   5d6e2:	lea    rcx,[rax-0x30]
   5d6e6:	mov    rdx,r12
   5d6e9:	jmp    5d707 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2887>
   5d6eb:	mulpd  xmm0,xmm2
   5d6ef:	add    rdx,0x10
   5d6f3:	movaps XMMWORD PTR [rcx+0x130],xmm0
   5d6fa:	add    rcx,0x10
   5d6fe:	cmp    rdx,r13
   5d701:	je     5d040 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x21c0>
   5d707:	comisd xmm1,QWORD PTR [rip+0x6319]        # 63a28 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x2a8>
   5d70f:	movapd xmm0,XMMWORD PTR [rdx]
   5d713:	jae    5d6eb <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x286b>
   5d715:	divpd  xmm0,xmm3
   5d719:	jmp    5d6ef <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x286f>
   5d71b:	xorpd  xmm3,xmm0
   5d71f:	jmp    5d34d <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x24cd>
   5d724:	movzx  ebx,BYTE PTR [rsp+0x1d7]
   5d72c:	mov    r14,QWORD PTR [rsp+0x1d8]
   5d734:	mov    r15,QWORD PTR [rsp+0x1e0]
   5d73c:	mov    r12,QWORD PTR [rsp+0x1e8]
   5d744:	mov    r9,QWORD PTR [rsp+0x1f0]
   5d74c:	mov    rbp,QWORD PTR [rsp+0x1f8]
   5d754:	mov    rdx,QWORD PTR [rsp+0x20]
   5d759:	mov    rdi,QWORD PTR [rsp+0x60]
   5d75e:	mov    esi,0x10
   5d763:	mov    QWORD PTR [rsp+0x230],0x0
   5d76f:	call   f9d0 <_ZN4Base6Matrix9Numerical10VectorNorm9ScaleLoopINS0_7ComplexIdEELm16ELm0EE7computeERKNS0_6VectorIS5_Lm16EEEmRd.part.0.constprop.0>
   5d774:	pxor   xmm0,xmm0
   5d778:	movsd  xmm5,QWORD PTR [rsp+0x230]
   5d781:	ucomisd xmm5,xmm0
   5d785:	jp     5d78d <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x290d>
   5d787:	je     5dbcc <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2d4c>
   5d78d:	movsd  xmm14,QWORD PTR [rsp+0x4a8]
   5d797:	movsd  xmm7,QWORD PTR [rsp+0x440]
   5d7a0:	movsd  xmm13,QWORD PTR [rsp+0x448]
   5d7aa:	movsd  xmm6,QWORD PTR [rsp+0x450]
   5d7b3:	movsd  xmm12,QWORD PTR [rsp+0x458]
   5d7bd:	divsd  xmm14,xmm5
   5d7c2:	movsd  xmm4,QWORD PTR [rsp+0x460]
   5d7cb:	movsd  xmm11,QWORD PTR [rsp+0x468]
   5d7d5:	movsd  xmm3,QWORD PTR [rsp+0x470]
   5d7de:	movsd  xmm10,QWORD PTR [rsp+0x478]
   5d7e8:	movsd  xmm2,QWORD PTR [rsp+0x480]
   5d7f1:	movsd  xmm9,QWORD PTR [rsp+0x488]
   5d7fb:	movsd  xmm1,QWORD PTR [rsp+0x490]
   5d804:	movsd  xmm8,QWORD PTR [rsp+0x498]
   5d80e:	movsd  xmm0,QWORD PTR [rsp+0x4a0]
   5d817:	divsd  xmm7,xmm5
   5d81b:	movsd  QWORD PTR [rsp],xmm14
   5d821:	movsd  xmm14,QWORD PTR [rsp+0x4b0]
   5d82b:	divsd  xmm14,xmm5
   5d830:	mulsd  xmm7,xmm7
   5d834:	divsd  xmm13,xmm5
   5d839:	movsd  QWORD PTR [rsp+0x18],xmm14
   5d840:	movsd  xmm14,QWORD PTR [rsp+0x4b8]
   5d84a:	divsd  xmm14,xmm5
   5d84f:	mulsd  xmm13,xmm13
   5d854:	addsd  xmm7,xmm13
   5d859:	pxor   xmm13,xmm13
   5d85e:	addsd  xmm7,xmm13
   5d863:	divsd  xmm6,xmm5
   5d867:	movsd  QWORD PTR [rsp+0x28],xmm14
   5d86e:	movsd  xmm14,QWORD PTR [rsp+0x4c0]
   5d878:	divsd  xmm14,xmm5
   5d87d:	mulsd  xmm6,xmm6
   5d881:	divsd  xmm12,xmm5
   5d886:	movsd  QWORD PTR [rsp+0x50],xmm14
   5d88d:	movsd  xmm14,QWORD PTR [rsp+0x4c8]
   5d897:	divsd  xmm14,xmm5
   5d89c:	mulsd  xmm12,xmm12
   5d8a1:	addsd  xmm6,xmm12
   5d8a6:	addsd  xmm6,xmm7
   5d8aa:	divsd  xmm4,xmm5
   5d8ae:	movsd  QWORD PTR [rsp+0x58],xmm14
   5d8b5:	movsd  xmm14,QWORD PTR [rsp+0x4d0]
   5d8bf:	divsd  xmm14,xmm5
   5d8c4:	mulsd  xmm4,xmm4
   5d8c8:	divsd  xmm11,xmm5
   5d8cd:	movsd  QWORD PTR [rsp+0x60],xmm14
   5d8d4:	movsd  xmm14,QWORD PTR [rsp+0x4d8]
   5d8de:	divsd  xmm14,xmm5
   5d8e3:	mulsd  xmm11,xmm11
   5d8e8:	addsd  xmm4,xmm11
   5d8ed:	addsd  xmm4,xmm6
   5d8f1:	divsd  xmm3,xmm5
   5d8f5:	movsd  QWORD PTR [rsp+0x70],xmm14
   5d8fc:	movsd  xmm14,QWORD PTR [rsp+0x4e0]
   5d906:	divsd  xmm14,xmm5
   5d90b:	mulsd  xmm3,xmm3
   5d90f:	divsd  xmm10,xmm5
   5d914:	movsd  QWORD PTR [rsp+0x80],xmm14
   5d91e:	movsd  xmm14,QWORD PTR [rsp+0x4e8]
   5d928:	divsd  xmm14,xmm5
   5d92d:	mulsd  xmm10,xmm10
   5d932:	addsd  xmm3,xmm10
   5d937:	addsd  xmm3,xmm4
   5d93b:	divsd  xmm2,xmm5
   5d93f:	movsd  QWORD PTR [rsp+0x88],xmm14
   5d949:	movsd  xmm14,QWORD PTR [rsp+0x4f0]
   5d953:	divsd  xmm14,xmm5
   5d958:	mulsd  xmm2,xmm2
   5d95c:	divsd  xmm9,xmm5
   5d961:	movsd  QWORD PTR [rsp+0xa0],xmm14
   5d96b:	movsd  xmm14,QWORD PTR [rsp+0x4f8]
   5d975:	divsd  xmm14,xmm5
   5d97a:	mulsd  xmm9,xmm9
   5d97f:	addsd  xmm2,xmm9
   5d984:	addsd  xmm2,xmm3
   5d988:	divsd  xmm1,xmm5
   5d98c:	movsd  QWORD PTR [rsp+0x90],xmm14
   5d996:	movsd  xmm14,QWORD PTR [rsp+0x500]
   5d9a0:	divsd  xmm14,xmm5
   5d9a5:	mulsd  xmm1,xmm1
   5d9a9:	divsd  xmm8,xmm5
   5d9ae:	movsd  QWORD PTR [rsp+0xa8],xmm14
   5d9b8:	movsd  xmm14,QWORD PTR [rsp+0x508]
   5d9c2:	divsd  xmm14,xmm5
   5d9c7:	mulsd  xmm8,xmm8
   5d9cc:	addsd  xmm1,xmm8
   5d9d1:	addsd  xmm1,xmm2
   5d9d5:	movsd  xmm2,QWORD PTR [rsp]
   5d9da:	mulsd  xmm2,xmm2
   5d9de:	divsd  xmm0,xmm5
   5d9e2:	movsd  QWORD PTR [rsp+0xc0],xmm14
   5d9ec:	movsd  xmm14,QWORD PTR [rsp+0x510]
   5d9f6:	divsd  xmm14,xmm5
   5d9fb:	mulsd  xmm0,xmm0
   5d9ff:	addsd  xmm0,xmm2
   5da03:	movsd  xmm2,QWORD PTR [rsp+0x28]
   5da09:	mulsd  xmm2,xmm2
   5da0d:	addsd  xmm0,xmm1
   5da11:	movsd  xmm1,QWORD PTR [rsp+0x18]
   5da17:	mulsd  xmm1,xmm1
   5da1b:	addsd  xmm1,xmm2
   5da1f:	movsd  xmm2,QWORD PTR [rsp+0x58]
   5da25:	mulsd  xmm2,xmm2
   5da29:	addsd  xmm1,xmm0
   5da2d:	movsd  xmm0,QWORD PTR [rsp+0x50]
   5da33:	movsd  QWORD PTR [rsp+0xe0],xmm14
   5da3d:	mulsd  xmm0,xmm0
   5da41:	movsd  xmm14,QWORD PTR [rsp+0x518]
   5da4b:	divsd  xmm14,xmm5
   5da50:	addsd  xmm0,xmm2
   5da54:	movsd  xmm2,QWORD PTR [rsp+0x70]
   5da5a:	mulsd  xmm2,xmm2
   5da5e:	addsd  xmm0,xmm1
   5da62:	movsd  xmm1,QWORD PTR [rsp+0x60]
   5da68:	mulsd  xmm1,xmm1
   5da6c:	addsd  xmm1,xmm2
   5da70:	movsd  xmm2,QWORD PTR [rsp+0x88]
   5da79:	mulsd  xmm2,xmm2
   5da7d:	addsd  xmm0,xmm1
   5da81:	movsd  xmm1,QWORD PTR [rsp+0x80]
   5da8a:	mulsd  xmm1,xmm1
   5da8e:	addsd  xmm1,xmm2
   5da92:	movsd  xmm2,QWORD PTR [rsp+0x90]
   5da9b:	mulsd  xmm2,xmm2
   5da9f:	addsd  xmm0,xmm1
   5daa3:	movsd  xmm1,QWORD PTR [rsp+0xa0]
   5daac:	movsd  QWORD PTR [rsp+0xe8],xmm14
   5dab6:	mulsd  xmm1,xmm1
   5daba:	movsd  xmm14,QWORD PTR [rsp+0x520]
   5dac4:	divsd  xmm14,xmm5
   5dac9:	addsd  xmm1,xmm2
   5dacd:	movsd  xmm2,QWORD PTR [rsp+0xc0]
   5dad6:	mulsd  xmm2,xmm2
   5dada:	addsd  xmm0,xmm1
   5dade:	movsd  xmm1,QWORD PTR [rsp+0xa8]
   5dae7:	mulsd  xmm1,xmm1
   5daeb:	addsd  xmm1,xmm2
   5daef:	movsd  xmm2,QWORD PTR [rsp+0xe8]
   5daf8:	mulsd  xmm2,xmm2
   5dafc:	addsd  xmm0,xmm1
   5db00:	movsd  xmm1,QWORD PTR [rsp+0xe0]
   5db09:	mulsd  xmm1,xmm1
   5db0d:	addsd  xmm1,xmm2
   5db11:	addsd  xmm0,xmm1
   5db15:	movsd  QWORD PTR [rsp+0xf0],xmm14
   5db1f:	movsd  xmm14,QWORD PTR [rsp+0x528]
   5db29:	movsd  xmm1,QWORD PTR [rsp+0xf0]
   5db32:	divsd  xmm14,xmm5
   5db37:	mulsd  xmm1,xmm1
   5db3b:	movsd  QWORD PTR [rsp+0xf8],xmm14
   5db45:	movsd  xmm14,QWORD PTR [rsp+0x530]
   5db4f:	divsd  xmm14,xmm5
   5db54:	movapd xmm15,xmm14
   5db59:	movsd  xmm14,QWORD PTR [rsp+0x538]
   5db63:	movsd  xmm2,QWORD PTR [rsp+0xf8]
   5db6c:	divsd  xmm14,xmm5
   5db71:	mulsd  xmm2,xmm2
   5db75:	addsd  xmm1,xmm2
   5db79:	addsd  xmm0,xmm1
   5db7d:	movapd xmm1,xmm15
   5db82:	mulsd  xmm1,xmm15
   5db87:	mulsd  xmm14,xmm14
   5db8c:	addsd  xmm1,xmm14
   5db91:	addsd  xmm0,xmm1
   5db95:	ucomisd xmm0,xmm13
   5db9a:	jp     5dba3 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2d23>
   5db9c:	movapd xmm1,xmm13
   5dba1:	je     5dbc4 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2d44>
   5dba3:	movsd  xmm1,QWORD PTR [rip+0x5c55]        # 63800 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
   5dbab:	movapd xmm6,xmm1
   5dbaf:	maxsd  xmm6,xmm0
   5dbb3:	movapd xmm0,xmm6
   5dbb7:	call   6a10 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5dbbc:	movapd xmm1,xmm0
   5dbc0:	mulsd  xmm1,xmm6
   5dbc4:	mulsd  xmm5,xmm1
   5dbc8:	movapd xmm0,xmm5
   5dbcc:	movsd  xmm6,QWORD PTR [rsp+0xb8]
   5dbd5:	comisd xmm6,xmm0
   5dbd9:	jb     5e75b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x38db>
   5dbdf:	mov    rax,rbp
   5dbe2:	mov    rsi,r9
   5dbe5:	mov    r11,r12
   5dbe8:	mov    r15,QWORD PTR [rsp+0x8]
   5dbed:	mov    DWORD PTR [rax+0x2158],0x5
   5dbf7:	mov    r12,QWORD PTR [rsp+0x68]
   5dbfc:	mov    r9d,ebx
   5dbff:	mov    rbp,r13
   5dc02:	mov    r10,QWORD PTR [rsp+0x180]
   5dc0a:	mov    r8,QWORD PTR [rsp+0x188]
   5dc12:	mov    r13,rax
   5dc15:	mov    rbx,rsi
   5dc18:	lea    rax,[r14+0x1]
   5dc1c:	cmp    rax,r11
   5dc1f:	je     5dc2a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2daa>
   5dc21:	add    r14,0x2
   5dc25:	cmp    r14,r11
   5dc28:	jne    5dc18 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2d98>
   5dc2a:	mov    rcx,rbp
   5dc2d:	mov    rax,rbp
   5dc30:	pxor   xmm0,xmm0
   5dc34:	movaps XMMWORD PTR [rax],xmm0
   5dc37:	add    rax,0x20
   5dc3b:	movaps XMMWORD PTR [rax-0x10],xmm0
   5dc3f:	cmp    QWORD PTR [rsp+0x30],rax
   5dc44:	jne    5dc34 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2db4>
   5dc46:	mov    rsi,QWORD PTR [rsp+0x10]
   5dc4b:	movapd xmm3,XMMWORD PTR [rcx]
   5dc4f:	movapd xmm2,XMMWORD PTR [rcx+0x10]
   5dc54:	mov    rdx,r8
   5dc57:	movlpd xmm2,QWORD PTR [rcx+0x8]
   5dc5c:	mov    rax,QWORD PTR [rsp+0x10]
   5dc61:	movhpd xmm3,QWORD PTR [rcx+0x10]
   5dc66:	nop    DWORD PTR [rax+0x0]
   5dc6a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5dc75:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5dc80:	movsd  xmm0,QWORD PTR [rax]
   5dc84:	movupd xmm1,XMMWORD PTR [rdx]
   5dc88:	add    rax,0x10
   5dc8c:	sub    rdx,0xffffffffffffff80
   5dc90:	unpcklpd xmm0,xmm0
   5dc94:	mulpd  xmm0,xmm1
   5dc98:	addpd  xmm3,xmm0
   5dc9c:	movsd  xmm0,QWORD PTR [rax-0x8]
   5dca1:	unpcklpd xmm0,xmm0
   5dca5:	mulpd  xmm0,xmm1
   5dca9:	addpd  xmm2,xmm0
   5dcad:	cmp    r12,rax
   5dcb0:	jne    5dc80 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2e00>
   5dcb2:	movapd xmm1,XMMWORD PTR [rsi]
   5dcb6:	add    rcx,0x20
   5dcba:	add    rsi,0x20
   5dcbe:	add    r8,0x10
   5dcc2:	movapd xmm6,XMMWORD PTR [rsp+0x160]
   5dccb:	movapd xmm0,XMMWORD PTR [rsi-0x10]
   5dcd0:	movapd xmm7,XMMWORD PTR [rsp+0x170]
   5dcd9:	movlpd xmm0,QWORD PTR [rsi-0x18]
   5dcde:	movhpd xmm1,QWORD PTR [rsi-0x10]
   5dce3:	movapd xmm4,xmm6
   5dce7:	mulpd  xmm4,xmm0
   5dceb:	movapd xmm5,xmm7
   5dcef:	mulpd  xmm5,xmm1
   5dcf3:	mulpd  xmm0,xmm7
   5dcf7:	mulpd  xmm1,xmm6
   5dcfb:	addpd  xmm4,xmm5
   5dcff:	subpd  xmm1,xmm0
   5dd03:	subpd  xmm2,xmm4
   5dd07:	subpd  xmm3,xmm1
   5dd0b:	movapd xmm0,xmm3
   5dd0f:	unpckhpd xmm3,xmm2
   5dd13:	unpcklpd xmm0,xmm2
   5dd17:	movaps XMMWORD PTR [rcx-0x10],xmm3
   5dd1b:	movaps XMMWORD PTR [rcx-0x20],xmm0
   5dd1f:	cmp    QWORD PTR [rsp+0x30],rcx
   5dd24:	jne    5dc4b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2dcb>
   5dd2a:	mov    rdx,QWORD PTR [rsp+0x20]
   5dd2f:	mov    rdi,rbp
   5dd32:	mov    esi,0x10
   5dd37:	mov    QWORD PTR [rsp+0x230],0x0
   5dd43:	call   f9d0 <_ZN4Base6Matrix9Numerical10VectorNorm9ScaleLoopINS0_7ComplexIdEELm16ELm0EE7computeERKNS0_6VectorIS5_Lm16EEEmRd.part.0.constprop.0>
   5dd48:	pxor   xmm0,xmm0
   5dd4c:	movsd  xmm5,QWORD PTR [rsp+0x230]
   5dd55:	ucomisd xmm5,xmm0
   5dd59:	jp     5dd65 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2ee5>
   5dd5b:	movapd xmm6,xmm0
   5dd5f:	je     5e175 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x32f5>
   5dd65:	movsd  xmm15,QWORD PTR [rsp+0x8b0]
   5dd6f:	movsd  xmm7,QWORD PTR [rsp+0x840]
   5dd78:	movsd  xmm14,QWORD PTR [rsp+0x848]
   5dd82:	movsd  xmm6,QWORD PTR [rsp+0x850]
   5dd8b:	movsd  xmm13,QWORD PTR [rsp+0x858]
   5dd95:	divsd  xmm15,xmm5
   5dd9a:	movsd  xmm4,QWORD PTR [rsp+0x860]
   5dda3:	movsd  xmm12,QWORD PTR [rsp+0x868]
   5ddad:	movsd  xmm3,QWORD PTR [rsp+0x870]
   5ddb6:	movsd  xmm11,QWORD PTR [rsp+0x878]
   5ddc0:	movsd  xmm2,QWORD PTR [rsp+0x880]
   5ddc9:	movsd  xmm10,QWORD PTR [rsp+0x888]
   5ddd3:	movsd  xmm1,QWORD PTR [rsp+0x890]
   5dddc:	movsd  xmm9,QWORD PTR [rsp+0x898]
   5dde6:	movsd  xmm0,QWORD PTR [rsp+0x8a0]
   5ddef:	movsd  xmm8,QWORD PTR [rsp+0x8a8]
   5ddf9:	divsd  xmm7,xmm5
   5ddfd:	movsd  QWORD PTR [rsp],xmm15
   5de03:	movsd  xmm15,QWORD PTR [rsp+0x8b8]
   5de0d:	divsd  xmm15,xmm5
   5de12:	mulsd  xmm7,xmm7
   5de16:	divsd  xmm14,xmm5
   5de1b:	movsd  QWORD PTR [rsp+0x8],xmm15
   5de22:	movsd  xmm15,QWORD PTR [rsp+0x8c0]
   5de2c:	divsd  xmm15,xmm5
   5de31:	mulsd  xmm14,xmm14
   5de36:	addsd  xmm7,xmm14
   5de3b:	pxor   xmm14,xmm14
   5de40:	addsd  xmm7,xmm14
   5de45:	divsd  xmm6,xmm5
   5de49:	movsd  QWORD PTR [rsp+0x18],xmm15
   5de50:	movsd  xmm15,QWORD PTR [rsp+0x8c8]
   5de5a:	divsd  xmm15,xmm5
   5de5f:	mulsd  xmm6,xmm6
   5de63:	divsd  xmm13,xmm5
   5de68:	movsd  QWORD PTR [rsp+0x28],xmm15
   5de6f:	movsd  xmm15,QWORD PTR [rsp+0x8d0]
   5de79:	divsd  xmm15,xmm5
   5de7e:	mulsd  xmm13,xmm13
   5de83:	addsd  xmm6,xmm13
   5de88:	addsd  xmm6,xmm7
   5de8c:	divsd  xmm4,xmm5
   5de90:	movsd  QWORD PTR [rsp+0x38],xmm15
   5de97:	movsd  xmm15,QWORD PTR [rsp+0x8d8]
   5dea1:	divsd  xmm15,xmm5
   5dea6:	mulsd  xmm4,xmm4
   5deaa:	divsd  xmm12,xmm5
   5deaf:	movsd  QWORD PTR [rsp+0x40],xmm15
   5deb6:	movsd  xmm15,QWORD PTR [rsp+0x8e0]
   5dec0:	divsd  xmm15,xmm5
   5dec5:	mulsd  xmm12,xmm12
   5deca:	addsd  xmm4,xmm12
   5decf:	addsd  xmm4,xmm6
   5ded3:	divsd  xmm3,xmm5
   5ded7:	movsd  QWORD PTR [rsp+0x48],xmm15
   5dede:	movsd  xmm15,QWORD PTR [rsp+0x8e8]
   5dee8:	divsd  xmm15,xmm5
   5deed:	mulsd  xmm3,xmm3
   5def1:	divsd  xmm11,xmm5
   5def6:	movsd  QWORD PTR [rsp+0x50],xmm15
   5defd:	movsd  xmm15,QWORD PTR [rsp+0x8f0]
   5df07:	divsd  xmm15,xmm5
   5df0c:	mulsd  xmm11,xmm11
   5df11:	addsd  xmm3,xmm11
   5df16:	addsd  xmm3,xmm4
   5df1a:	divsd  xmm2,xmm5
   5df1e:	movsd  QWORD PTR [rsp+0x58],xmm15
   5df25:	movsd  xmm15,QWORD PTR [rsp+0x8f8]
   5df2f:	divsd  xmm15,xmm5
   5df34:	mulsd  xmm2,xmm2
   5df38:	divsd  xmm10,xmm5
   5df3d:	movsd  QWORD PTR [rsp+0x60],xmm15
   5df44:	movsd  xmm15,QWORD PTR [rsp+0x900]
   5df4e:	divsd  xmm15,xmm5
   5df53:	mulsd  xmm10,xmm10
   5df58:	addsd  xmm2,xmm10
   5df5d:	addsd  xmm2,xmm3
   5df61:	divsd  xmm1,xmm5
   5df65:	movsd  QWORD PTR [rsp+0x68],xmm15
   5df6c:	movsd  xmm15,QWORD PTR [rsp+0x908]
   5df76:	divsd  xmm15,xmm5
   5df7b:	mulsd  xmm1,xmm1
   5df7f:	divsd  xmm9,xmm5
   5df84:	movsd  QWORD PTR [rsp+0x70],xmm15
   5df8b:	movsd  xmm15,QWORD PTR [rsp+0x910]
   5df95:	divsd  xmm15,xmm5
   5df9a:	mulsd  xmm9,xmm9
   5df9f:	addsd  xmm1,xmm9
   5dfa4:	addsd  xmm1,xmm2
   5dfa8:	movsd  xmm2,QWORD PTR [rsp+0x8]
   5dfae:	mulsd  xmm2,xmm2
   5dfb2:	divsd  xmm0,xmm5
   5dfb6:	movsd  QWORD PTR [rsp+0x80],xmm15
   5dfc0:	movsd  xmm15,QWORD PTR [rsp+0x918]
   5dfca:	divsd  xmm15,xmm5
   5dfcf:	mulsd  xmm0,xmm0
   5dfd3:	divsd  xmm8,xmm5
   5dfd8:	movsd  QWORD PTR [rsp+0x88],xmm15
   5dfe2:	movsd  xmm15,QWORD PTR [rsp+0x920]
   5dfec:	divsd  xmm15,xmm5
   5dff1:	mulsd  xmm8,xmm8
   5dff6:	addsd  xmm0,xmm8
   5dffb:	addsd  xmm0,xmm1
   5dfff:	movsd  xmm1,QWORD PTR [rsp]
   5e004:	mulsd  xmm1,xmm1
   5e008:	addsd  xmm1,xmm2
   5e00c:	movsd  xmm2,QWORD PTR [rsp+0x28]
   5e012:	mulsd  xmm2,xmm2
   5e016:	addsd  xmm1,xmm0
   5e01a:	movsd  xmm0,QWORD PTR [rsp+0x18]
   5e020:	movsd  QWORD PTR [rsp+0xa0],xmm15
   5e02a:	mulsd  xmm0,xmm0
   5e02e:	movsd  xmm15,QWORD PTR [rsp+0x928]
   5e038:	divsd  xmm15,xmm5
   5e03d:	addsd  xmm0,xmm2
   5e041:	movsd  xmm2,QWORD PTR [rsp+0x40]
   5e047:	mulsd  xmm2,xmm2
   5e04b:	addsd  xmm0,xmm1
   5e04f:	movsd  xmm1,QWORD PTR [rsp+0x38]
   5e055:	mulsd  xmm1,xmm1
   5e059:	addsd  xmm1,xmm2
   5e05d:	movsd  xmm2,QWORD PTR [rsp+0x50]
   5e063:	mulsd  xmm2,xmm2
   5e067:	addsd  xmm0,xmm1
   5e06b:	movsd  xmm1,QWORD PTR [rsp+0x48]
   5e071:	mulsd  xmm1,xmm1
   5e075:	addsd  xmm1,xmm2
   5e079:	movsd  xmm2,QWORD PTR [rsp+0x60]
   5e07f:	mulsd  xmm2,xmm2
   5e083:	addsd  xmm0,xmm1
   5e087:	movsd  xmm1,QWORD PTR [rsp+0x58]
   5e08d:	movsd  QWORD PTR [rsp+0x90],xmm15
   5e097:	mulsd  xmm1,xmm1
   5e09b:	movsd  xmm15,QWORD PTR [rsp+0x930]
   5e0a5:	divsd  xmm15,xmm5
   5e0aa:	addsd  xmm1,xmm2
   5e0ae:	movsd  xmm2,QWORD PTR [rsp+0x70]
   5e0b4:	mulsd  xmm2,xmm2
   5e0b8:	addsd  xmm0,xmm1
   5e0bc:	movsd  xmm1,QWORD PTR [rsp+0x68]
   5e0c2:	mulsd  xmm1,xmm1
   5e0c6:	addsd  xmm1,xmm2
   5e0ca:	movsd  xmm2,QWORD PTR [rsp+0x88]
   5e0d3:	mulsd  xmm2,xmm2
   5e0d7:	addsd  xmm0,xmm1
   5e0db:	movsd  xmm1,QWORD PTR [rsp+0x80]
   5e0e4:	mulsd  xmm1,xmm1
   5e0e8:	addsd  xmm1,xmm2
   5e0ec:	addsd  xmm0,xmm1
   5e0f0:	movsd  xmm1,QWORD PTR [rsp+0xa0]
   5e0f9:	movsd  QWORD PTR [rsp+0xa8],xmm15
   5e103:	mulsd  xmm1,xmm1
   5e107:	movsd  xmm15,QWORD PTR [rsp+0x938]
   5e111:	movsd  xmm2,QWORD PTR [rsp+0x90]
   5e11a:	divsd  xmm15,xmm5
   5e11f:	mulsd  xmm2,xmm2
   5e123:	addsd  xmm1,xmm2
   5e127:	addsd  xmm0,xmm1
   5e12b:	movsd  xmm1,QWORD PTR [rsp+0xa8]
   5e134:	mulsd  xmm1,xmm1
   5e138:	mulsd  xmm15,xmm15
   5e13d:	addsd  xmm1,xmm15
   5e142:	addsd  xmm0,xmm1
   5e146:	ucomisd xmm0,xmm14
   5e14b:	jp     5e154 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x32d4>
   5e14d:	movapd xmm6,xmm14
   5e152:	je     5e171 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x32f1>
   5e154:	movsd  xmm1,QWORD PTR [rip+0x56a4]        # 63800 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
   5e15c:	movapd xmm6,xmm1
   5e160:	maxsd  xmm6,xmm0
   5e164:	movapd xmm0,xmm6
   5e168:	call   6a10 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5e16d:	mulsd  xmm6,xmm0
   5e171:	mulsd  xmm6,xmm5
   5e175:	divsd  xmm6,QWORD PTR [rsp+0x190]
   5e17e:	comisd xmm6,QWORD PTR [r13+0x2170]
   5e187:	jbe    5e192 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3312>
   5e189:	movsd  QWORD PTR [r13+0x2170],xmm6
   5e192:	movzx  ecx,BYTE PTR [rsp+0xd8]
   5e19a:	and    cl,BYTE PTR [r13+0x215d]
   5e1a1:	mov    BYTE PTR [r13+0x215d],cl
   5e1a8:	mov    rdx,QWORD PTR [rsp+0x150]
   5e1b0:	mov    rax,QWORD PTR [rsp+0x10]
   5e1b5:	movapd xmm0,XMMWORD PTR [rax]
   5e1b9:	add    rax,0x10
   5e1bd:	add    rdx,0x10
   5e1c1:	movups XMMWORD PTR [rdx-0x10],xmm0
   5e1c5:	cmp    rax,r12
   5e1c8:	jne    5e1b5 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3335>
   5e1ca:	add    QWORD PTR [rsp+0xb0],0x1
   5e1d3:	mov    rax,QWORD PTR [rsp+0xb0]
   5e1db:	add    r10,0x10
   5e1df:	add    QWORD PTR [rsp+0x150],0x100
   5e1eb:	cmp    rax,0x10
   5e1ef:	jne    5b04c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1cc>
   5e1f5:	mov    rax,QWORD PTR [rsp+0xd0]
   5e1fd:	movzx  r15d,cl
   5e201:	mov    BYTE PTR [rax+0x2161],0x1
   5e208:	cmp    BYTE PTR [rax+0x215f],0x0
   5e20f:	je     5ee07 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3f87>
   5e215:	mov    BYTE PTR [rax+0x215d],0x0
   5e21c:	mov    DWORD PTR [rax+0x2158],0x4
   5e226:	mov    rax,QWORD PTR [rsp+0x36a8]
   5e22e:	sub    rax,QWORD PTR fs:0x28
   5e237:	jne    5ee68 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3fe8>
   5e23d:	add    rsp,0x36b8
   5e244:	pop    rbx
   5e245:	pop    rbp
   5e246:	pop    r12
   5e248:	pop    r13
   5e24a:	pop    r14
   5e24c:	pop    r15
   5e24e:	ret
   5e24f:	mov    rdx,QWORD PTR [rsp+0x60]
   5e254:	mov    rcx,QWORD PTR [rsp+0x30]
   5e259:	mov    r9,r13
   5e25c:	mov    rax,r13
   5e25f:	movapd xmm0,XMMWORD PTR [rdx]
   5e263:	add    rax,0x10
   5e267:	add    rdx,0x10
   5e26b:	movaps XMMWORD PTR [rax-0x10],xmm0
   5e26f:	cmp    rcx,rax
   5e272:	jne    5e25f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x33df>
   5e274:	mov    rsi,QWORD PTR [rsp+0x8]
   5e279:	mov    rdi,r13
   5e27c:	xor    r10d,r10d
   5e27f:	pxor   xmm6,xmm6
   5e283:	mov    rbx,rsi
   5e286:	pxor   xmm8,xmm8
   5e28b:	lea    rcx,[rsi+0x1c50]
   5e292:	mov    rdx,rbx
   5e295:	xor    eax,eax
   5e297:	movapd xmm5,xmm8
   5e29c:	nop    DWORD PTR [rax+0x0]
   5e2a0:	cmp    rax,r8
   5e2a3:	jae    5e2e4 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3464>
   5e2a5:	movsd  xmm0,QWORD PTR [rdx+0x30]
   5e2aa:	movsd  xmm3,QWORD PTR [rdx+0x38]
   5e2af:	movsd  xmm1,QWORD PTR [rcx]
   5e2b3:	movsd  xmm4,QWORD PTR [rcx+0x8]
   5e2b8:	movapd xmm2,xmm0
   5e2bc:	movapd xmm9,xmm3
   5e2c1:	mulsd  xmm2,xmm1
   5e2c5:	mulsd  xmm9,xmm4
   5e2ca:	mulsd  xmm0,xmm4
   5e2ce:	mulsd  xmm1,xmm3
   5e2d2:	subsd  xmm2,xmm9
   5e2d7:	addsd  xmm0,xmm1
   5e2db:	addsd  xmm5,xmm2
   5e2df:	addsd  xmm8,xmm0
   5e2e4:	add    rax,0x1
   5e2e8:	add    rdx,0x100
   5e2ef:	add    rcx,0x10
   5e2f3:	cmp    rax,0x10
   5e2f7:	jne    5e2a0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3420>
   5e2f9:	movapd xmm0,xmm5
   5e2fd:	add    r10,0x1
   5e301:	add    rdi,0x10
   5e305:	add    rbx,0x10
   5e309:	mulsd  xmm5,xmm6
   5e30d:	movapd xmm1,xmm8
   5e312:	mulsd  xmm1,xmm6
   5e316:	addsd  xmm5,xmm8
   5e31b:	addsd  xmm5,QWORD PTR [rdi-0x8]
   5e320:	subsd  xmm0,xmm1
   5e324:	addsd  xmm0,QWORD PTR [rdi-0x10]
   5e329:	movsd  QWORD PTR [rdi-0x8],xmm5
   5e32e:	movsd  QWORD PTR [rdi-0x10],xmm0
   5e333:	cmp    r10,0x10
   5e337:	jne    5e286 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3406>
   5e33d:	mov    rax,QWORD PTR [rsp+0x50]
   5e342:	mov    rdi,QWORD PTR [rsp+0x30]
   5e347:	pxor   xmm0,xmm0
   5e34b:	mov    rsi,r13
   5e34e:	movsd  QWORD PTR [rsp],xmm7
   5e353:	mov    rdx,rax
   5e356:	movaps XMMWORD PTR [rsp+0x540],xmm0
   5e35e:	movaps XMMWORD PTR [rsp+0x550],xmm0
   5e366:	movaps XMMWORD PTR [rsp+0x560],xmm0
   5e36e:	movaps XMMWORD PTR [rsp+0x570],xmm0
   5e376:	movaps XMMWORD PTR [rsp+0x580],xmm0
   5e37e:	movaps XMMWORD PTR [rsp+0x590],xmm0
   5e386:	movaps XMMWORD PTR [rsp+0x5a0],xmm0
   5e38e:	movaps XMMWORD PTR [rsp+0x5b0],xmm0
   5e396:	movaps XMMWORD PTR [rsp+0x5c0],xmm0
   5e39e:	movaps XMMWORD PTR [rsp+0x5d0],xmm0
   5e3a6:	movaps XMMWORD PTR [rsp+0x5e0],xmm0
   5e3ae:	movaps XMMWORD PTR [rsp+0x5f0],xmm0
   5e3b6:	movaps XMMWORD PTR [rsp+0x600],xmm0
   5e3be:	movaps XMMWORD PTR [rsp+0x610],xmm0
   5e3c6:	movaps XMMWORD PTR [rsp+0x620],xmm0
   5e3ce:	movaps XMMWORD PTR [rsp+0x630],xmm0
   5e3d6:	call   1be00 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeINS0_7ComplexIdEELm16EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS7_XT0_EEERSC_m.part.0>
   5e3db:	mov    rdx,QWORD PTR [rsp+0x10]
   5e3e0:	movsd  xmm7,QWORD PTR [rsp]
   5e3e5:	movapd xmm0,XMMWORD PTR [rdx]
   5e3e9:	subpd  xmm0,XMMWORD PTR [rax]
   5e3ed:	add    rax,0x10
   5e3f1:	add    rdx,0x10
   5e3f5:	movaps XMMWORD PTR [rax-0x10],xmm0
   5e3f9:	cmp    r11,rax
   5e3fc:	jne    5e3e5 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3565>
   5e3fe:	mov    rdx,QWORD PTR [rsp+0x1b8]
   5e406:	mov    rdi,QWORD PTR [rsp+0x50]
   5e40b:	mov    esi,0x10
   5e410:	mov    QWORD PTR [rsp+0x218],0x0
   5e41c:	call   f9d0 <_ZN4Base6Matrix9Numerical10VectorNorm9ScaleLoopINS0_7ComplexIdEELm16ELm0EE7computeERKNS0_6VectorIS5_Lm16EEEmRd.part.0.constprop.0>
   5e421:	pxor   xmm0,xmm0
   5e425:	movsd  xmm5,QWORD PTR [rsp+0x218]
   5e42e:	ucomisd xmm5,xmm0
   5e432:	jp     5e43a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x35ba>
   5e434:	je     5e62e <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x37ae>
   5e43a:	movsd  xmm14,QWORD PTR [rsp+0x5a0]
   5e444:	mov    rdx,QWORD PTR [rsp+0x20]
   5e449:	mov    esi,0x10
   5e44e:	movsd  xmm2,QWORD PTR [rsp+0x540]
   5e457:	mov    rdi,QWORD PTR [rsp+0x50]
   5e45c:	movsd  xmm13,QWORD PTR [rsp+0x548]
   5e466:	divsd  xmm14,xmm5
   5e46b:	movsd  xmm1,QWORD PTR [rsp+0x550]
   5e474:	movsd  xmm12,QWORD PTR [rsp+0x558]
   5e47e:	movsd  xmm0,QWORD PTR [rsp+0x560]
   5e487:	movsd  xmm11,QWORD PTR [rsp+0x568]
   5e491:	movsd  xmm6,QWORD PTR [rsp+0x570]
   5e49a:	movsd  xmm10,QWORD PTR [rsp+0x578]
   5e4a4:	movsd  xmm4,QWORD PTR [rsp+0x580]
   5e4ad:	movsd  xmm9,QWORD PTR [rsp+0x588]
   5e4b7:	movsd  xmm3,QWORD PTR [rsp+0x590]
   5e4c0:	movsd  xmm8,QWORD PTR [rsp+0x598]
   5e4ca:	divsd  xmm2,xmm5
   5e4ce:	movsd  QWORD PTR [rsp],xmm14
   5e4d4:	movsd  xmm14,QWORD PTR [rsp+0x5a8]
   5e4de:	divsd  xmm14,xmm5
   5e4e3:	mulsd  xmm2,xmm2
   5e4e7:	divsd  xmm13,xmm5
   5e4ec:	movsd  QWORD PTR [rsp+0x28],xmm14
   5e4f3:	movsd  xmm14,QWORD PTR [rsp+0x5b0]
   5e4fd:	divsd  xmm14,xmm5
   5e502:	mulsd  xmm13,xmm13
   5e507:	addsd  xmm2,xmm13
   5e50c:	pxor   xmm13,xmm13
   5e511:	addsd  xmm2,xmm13
   5e516:	divsd  xmm1,xmm5
   5e51a:	movapd xmm15,xmm14
   5e51f:	movsd  xmm14,QWORD PTR [rsp+0x5b8]
   5e529:	divsd  xmm12,xmm5
   5e52e:	mulsd  xmm1,xmm1
   5e532:	divsd  xmm0,xmm5
   5e536:	mulsd  xmm12,xmm12
   5e53b:	addsd  xmm1,xmm12
   5e540:	addsd  xmm1,xmm2
   5e544:	movsd  xmm2,QWORD PTR [rsp+0x28]
   5e54a:	mulsd  xmm2,xmm2
   5e54e:	divsd  xmm11,xmm5
   5e553:	mulsd  xmm0,xmm0
   5e557:	divsd  xmm6,xmm5
   5e55b:	mulsd  xmm11,xmm11
   5e560:	addsd  xmm0,xmm11
   5e565:	addsd  xmm0,xmm1
   5e569:	movsd  xmm1,QWORD PTR [rsp]
   5e56e:	mulsd  xmm1,xmm1
   5e572:	divsd  xmm10,xmm5
   5e577:	addsd  xmm1,xmm2
   5e57b:	mulsd  xmm6,xmm6
   5e57f:	divsd  xmm4,xmm5
   5e583:	mulsd  xmm10,xmm10
   5e588:	addsd  xmm6,xmm10
   5e58d:	addsd  xmm0,xmm6
   5e591:	divsd  xmm9,xmm5
   5e596:	mulsd  xmm4,xmm4
   5e59a:	divsd  xmm3,xmm5
   5e59e:	mulsd  xmm9,xmm9
   5e5a3:	addsd  xmm4,xmm9
   5e5a8:	addsd  xmm0,xmm4
   5e5ac:	divsd  xmm8,xmm5
   5e5b1:	mulsd  xmm3,xmm3
   5e5b5:	divsd  xmm14,xmm5
   5e5ba:	mulsd  xmm8,xmm8
   5e5bf:	addsd  xmm3,xmm8
   5e5c4:	addsd  xmm0,xmm3
   5e5c8:	addsd  xmm0,xmm1
   5e5cc:	movapd xmm1,xmm15
   5e5d1:	mulsd  xmm1,xmm15
   5e5d6:	mulsd  xmm14,xmm14
   5e5db:	addsd  xmm1,xmm14
   5e5e0:	addsd  xmm0,xmm1
   5e5e4:	movsd  QWORD PTR [rsp+0x230],xmm0
   5e5ed:	movapd xmm0,xmm5
   5e5f1:	call   be80 <_ZN4Base6Matrix9Numerical10VectorNorm14SquaredSumLoopINS0_7ComplexIdEELm16ELm8EE7computeERKNS0_6VectorIS5_Lm16EEEmdRd.part.0>
   5e5f6:	movsd  xmm2,QWORD PTR [rsp+0x230]
   5e5ff:	ucomisd xmm2,xmm13
   5e604:	jp     5e60d <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x378d>
   5e606:	movapd xmm0,xmm13
   5e60b:	je     5e62a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x37aa>
   5e60d:	movsd  xmm1,QWORD PTR [rip+0x51eb]        # 63800 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
   5e615:	movapd xmm6,xmm1
   5e619:	maxsd  xmm6,xmm2
   5e61d:	movapd xmm0,xmm6
   5e621:	call   6a10 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5e626:	mulsd  xmm0,xmm6
   5e62a:	mulsd  xmm0,xmm5
   5e62e:	comisd xmm7,xmm0
   5e632:	mov    rax,QWORD PTR [rsp+0x60]
   5e637:	lea    rdx,[rax+0x100]
   5e63e:	jb     5e694 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3814>
   5e640:	movapd xmm1,XMMWORD PTR [r9]
   5e645:	add    rax,0x10
   5e649:	add    r9,0x10
   5e64d:	movaps XMMWORD PTR [rax-0x10],xmm1
   5e651:	cmp    rdx,rax
   5e654:	jne    5e640 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x37c0>
   5e656:	movsd  QWORD PTR [rsp+0x3690],xmm0
   5e65f:	movsd  xmm6,QWORD PTR [rsp+0xe0]
   5e668:	comisd xmm6,xmm0
   5e66c:	setae  BYTE PTR [rsp+0x36a0]
   5e674:	setb   al
   5e677:	and    al,BYTE PTR [rsp+0x80]
   5e67e:	jmp    5d3f4 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2574>
   5e683:	mov    rbp,QWORD PTR [rsp]
   5e687:	mov    r13,QWORD PTR [rsp+0x28]
   5e68c:	mov    r11,r14
   5e68f:	jmp    5d3ef <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x256f>
   5e694:	mov    rax,QWORD PTR [rsp+0x50]
   5e699:	mov    rsi,QWORD PTR [rsp+0x60]
   5e69e:	pxor   xmm0,xmm0
   5e6a2:	mov    rdi,QWORD PTR [rsp+0x30]
   5e6a7:	movaps XMMWORD PTR [rsp+0x540],xmm0
   5e6af:	mov    rdx,rax
   5e6b2:	movaps XMMWORD PTR [rsp+0x550],xmm0
   5e6ba:	movaps XMMWORD PTR [rsp+0x560],xmm0
   5e6c2:	movaps XMMWORD PTR [rsp+0x570],xmm0
   5e6ca:	movaps XMMWORD PTR [rsp+0x580],xmm0
   5e6d2:	movaps XMMWORD PTR [rsp+0x590],xmm0
   5e6da:	movaps XMMWORD PTR [rsp+0x5a0],xmm0
   5e6e2:	movaps XMMWORD PTR [rsp+0x5b0],xmm0
   5e6ea:	movaps XMMWORD PTR [rsp+0x5c0],xmm0
   5e6f2:	movaps XMMWORD PTR [rsp+0x5d0],xmm0
   5e6fa:	movaps XMMWORD PTR [rsp+0x5e0],xmm0
   5e702:	movaps XMMWORD PTR [rsp+0x5f0],xmm0
   5e70a:	movaps XMMWORD PTR [rsp+0x600],xmm0
   5e712:	movaps XMMWORD PTR [rsp+0x610],xmm0
   5e71a:	movaps XMMWORD PTR [rsp+0x620],xmm0
   5e722:	movaps XMMWORD PTR [rsp+0x630],xmm0
   5e72a:	call   1be00 <_ZN4Base6Matrix14GmresOperation13DenseMultiply7computeINS0_7ComplexIdEELm16EEEvRKNS0_6MatrixIT_XT0_EXT0_EEERKNS0_6VectorIS7_XT0_EEERSC_m.part.0>
   5e72f:	mov    rdx,QWORD PTR [rsp+0x10]
   5e734:	movapd xmm0,XMMWORD PTR [rdx]
   5e738:	subpd  xmm0,XMMWORD PTR [rax]
   5e73c:	add    rax,0x10
   5e740:	add    rdx,0x10
   5e744:	movaps XMMWORD PTR [rax-0x10],xmm0
   5e748:	cmp    r11,rax
   5e74b:	jne    5e734 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x38b4>
   5e74d:	movsd  xmm0,QWORD PTR [rsp+0x3690]
   5e756:	jmp    5e65f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x37df>
   5e75b:	mov    rax,QWORD PTR [rsp+0x10]
   5e760:	unpcklpd xmm0,xmm0
   5e764:	movapd xmm1,XMMWORD PTR [r15]
   5e769:	add    rax,0x10
   5e76d:	add    r15,0x10
   5e771:	divpd  xmm1,xmm0
   5e775:	movaps XMMWORD PTR [rax-0x10],xmm1
   5e779:	cmp    QWORD PTR [rsp+0x68],rax
   5e77e:	jne    5e764 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x38e4>
   5e780:	mov    rcx,r13
   5e783:	mov    rax,r13
   5e786:	pxor   xmm0,xmm0
   5e78a:	movaps XMMWORD PTR [rax],xmm0
   5e78d:	add    rax,0x20
   5e791:	movaps XMMWORD PTR [rax-0x10],xmm0
   5e795:	cmp    QWORD PTR [rsp+0x30],rax
   5e79a:	jne    5e78a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x390a>
   5e79c:	mov    rdi,QWORD PTR [rsp+0x1a0]
   5e7a4:	mov    rsi,QWORD PTR [rsp+0x10]
   5e7a9:	mov    r8,QWORD PTR [rsp+0x68]
   5e7ae:	movapd xmm3,XMMWORD PTR [rcx]
   5e7b2:	movapd xmm2,XMMWORD PTR [rcx+0x10]
   5e7b7:	mov    rdx,rdi
   5e7ba:	movlpd xmm2,QWORD PTR [rcx+0x8]
   5e7bf:	mov    rax,QWORD PTR [rsp+0x10]
   5e7c4:	movhpd xmm3,QWORD PTR [rcx+0x10]
   5e7c9:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5e7d4:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5e7df:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5e7ea:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5e7f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   5e800:	movsd  xmm0,QWORD PTR [rax]
   5e804:	movupd xmm1,XMMWORD PTR [rdx]
   5e808:	add    rax,0x10
   5e80c:	sub    rdx,0xffffffffffffff80
   5e810:	unpcklpd xmm0,xmm0
   5e814:	mulpd  xmm0,xmm1
   5e818:	addpd  xmm3,xmm0
   5e81c:	movsd  xmm0,QWORD PTR [rax-0x8]
   5e821:	unpcklpd xmm0,xmm0
   5e825:	mulpd  xmm0,xmm1
   5e829:	addpd  xmm2,xmm0
   5e82d:	cmp    r8,rax
   5e830:	jne    5e800 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3980>
   5e832:	movapd xmm1,XMMWORD PTR [rsi]
   5e836:	add    rcx,0x20
   5e83a:	add    rsi,0x20
   5e83e:	add    rdi,0x10
   5e842:	movapd xmm6,XMMWORD PTR [rsp+0x160]
   5e84b:	movapd xmm0,XMMWORD PTR [rsi-0x10]
   5e850:	movapd xmm7,XMMWORD PTR [rsp+0x170]
   5e859:	movlpd xmm0,QWORD PTR [rsi-0x18]
   5e85e:	movhpd xmm1,QWORD PTR [rsi-0x10]
   5e863:	movapd xmm4,xmm6
   5e867:	mulpd  xmm4,xmm0
   5e86b:	movapd xmm5,xmm7
   5e86f:	mulpd  xmm5,xmm1
   5e873:	mulpd  xmm0,xmm7
   5e877:	mulpd  xmm1,xmm6
   5e87b:	addpd  xmm4,xmm5
   5e87f:	subpd  xmm1,xmm0
   5e883:	subpd  xmm2,xmm4
   5e887:	subpd  xmm3,xmm1
   5e88b:	movapd xmm0,xmm3
   5e88f:	unpckhpd xmm3,xmm2
   5e893:	unpcklpd xmm0,xmm2
   5e897:	movaps XMMWORD PTR [rcx-0x10],xmm3
   5e89b:	movaps XMMWORD PTR [rcx-0x20],xmm0
   5e89f:	cmp    QWORD PTR [rsp+0x30],rcx
   5e8a4:	jne    5e7ae <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x392e>
   5e8aa:	mov    rdx,QWORD PTR [rsp+0x20]
   5e8af:	mov    rdi,r13
   5e8b2:	mov    esi,0x10
   5e8b7:	mov    QWORD PTR [rsp+0x230],0x0
   5e8c3:	call   f9d0 <_ZN4Base6Matrix9Numerical10VectorNorm9ScaleLoopINS0_7ComplexIdEELm16ELm0EE7computeERKNS0_6VectorIS5_Lm16EEEmRd.part.0.constprop.0>
   5e8c8:	pxor   xmm0,xmm0
   5e8cc:	movsd  xmm5,QWORD PTR [rsp+0x230]
   5e8d5:	ucomisd xmm5,xmm0
   5e8d9:	jp     5e8e5 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3a65>
   5e8db:	movapd xmm6,xmm0
   5e8df:	je     5ed1c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3e9c>
   5e8e5:	movsd  xmm14,QWORD PTR [rsp+0x8a8]
   5e8ef:	movsd  xmm7,QWORD PTR [rsp+0x840]
   5e8f8:	movsd  xmm13,QWORD PTR [rsp+0x848]
   5e902:	movsd  xmm6,QWORD PTR [rsp+0x850]
   5e90b:	movsd  xmm12,QWORD PTR [rsp+0x858]
   5e915:	divsd  xmm14,xmm5
   5e91a:	movsd  xmm4,QWORD PTR [rsp+0x860]
   5e923:	movsd  xmm11,QWORD PTR [rsp+0x868]
   5e92d:	movsd  xmm3,QWORD PTR [rsp+0x870]
   5e936:	movsd  xmm10,QWORD PTR [rsp+0x878]
   5e940:	movsd  xmm2,QWORD PTR [rsp+0x880]
   5e949:	movsd  xmm9,QWORD PTR [rsp+0x888]
   5e953:	movsd  xmm1,QWORD PTR [rsp+0x890]
   5e95c:	movsd  xmm8,QWORD PTR [rsp+0x898]
   5e966:	movsd  xmm0,QWORD PTR [rsp+0x8a0]
   5e96f:	divsd  xmm7,xmm5
   5e973:	movsd  QWORD PTR [rsp],xmm14
   5e979:	movsd  xmm14,QWORD PTR [rsp+0x8b0]
   5e983:	divsd  xmm14,xmm5
   5e988:	mulsd  xmm7,xmm7
   5e98c:	divsd  xmm13,xmm5
   5e991:	movsd  QWORD PTR [rsp+0x18],xmm14
   5e998:	movsd  xmm14,QWORD PTR [rsp+0x8b8]
   5e9a2:	divsd  xmm14,xmm5
   5e9a7:	mulsd  xmm13,xmm13
   5e9ac:	addsd  xmm7,xmm13
   5e9b1:	pxor   xmm13,xmm13
   5e9b6:	addsd  xmm7,xmm13
   5e9bb:	divsd  xmm6,xmm5
   5e9bf:	movsd  QWORD PTR [rsp+0x28],xmm14
   5e9c6:	movsd  xmm14,QWORD PTR [rsp+0x8c0]
   5e9d0:	divsd  xmm14,xmm5
   5e9d5:	mulsd  xmm6,xmm6
   5e9d9:	divsd  xmm12,xmm5
   5e9de:	movsd  QWORD PTR [rsp+0x50],xmm14
   5e9e5:	movsd  xmm14,QWORD PTR [rsp+0x8c8]
   5e9ef:	divsd  xmm14,xmm5
   5e9f4:	mulsd  xmm12,xmm12
   5e9f9:	addsd  xmm6,xmm12
   5e9fe:	addsd  xmm6,xmm7
   5ea02:	divsd  xmm4,xmm5
   5ea06:	movsd  QWORD PTR [rsp+0x58],xmm14
   5ea0d:	movsd  xmm14,QWORD PTR [rsp+0x8d0]
   5ea17:	divsd  xmm14,xmm5
   5ea1c:	mulsd  xmm4,xmm4
   5ea20:	divsd  xmm11,xmm5
   5ea25:	movsd  QWORD PTR [rsp+0x60],xmm14
   5ea2c:	movsd  xmm14,QWORD PTR [rsp+0x8d8]
   5ea36:	divsd  xmm14,xmm5
   5ea3b:	mulsd  xmm11,xmm11
   5ea40:	addsd  xmm4,xmm11
   5ea45:	addsd  xmm4,xmm6
   5ea49:	divsd  xmm3,xmm5
   5ea4d:	movsd  QWORD PTR [rsp+0x70],xmm14
   5ea54:	movsd  xmm14,QWORD PTR [rsp+0x8e0]
   5ea5e:	divsd  xmm14,xmm5
   5ea63:	mulsd  xmm3,xmm3
   5ea67:	divsd  xmm10,xmm5
   5ea6c:	movsd  QWORD PTR [rsp+0x80],xmm14
   5ea76:	movsd  xmm14,QWORD PTR [rsp+0x8e8]
   5ea80:	divsd  xmm14,xmm5
   5ea85:	mulsd  xmm10,xmm10
   5ea8a:	addsd  xmm3,xmm10
   5ea8f:	addsd  xmm3,xmm4
   5ea93:	divsd  xmm2,xmm5
   5ea97:	movsd  QWORD PTR [rsp+0x88],xmm14
   5eaa1:	movsd  xmm14,QWORD PTR [rsp+0x8f0]
   5eaab:	divsd  xmm14,xmm5
   5eab0:	mulsd  xmm2,xmm2
   5eab4:	divsd  xmm9,xmm5
   5eab9:	movsd  QWORD PTR [rsp+0xa0],xmm14
   5eac3:	movsd  xmm14,QWORD PTR [rsp+0x8f8]
   5eacd:	divsd  xmm14,xmm5
   5ead2:	mulsd  xmm9,xmm9
   5ead7:	addsd  xmm2,xmm9
   5eadc:	addsd  xmm2,xmm3
   5eae0:	divsd  xmm1,xmm5
   5eae4:	movsd  QWORD PTR [rsp+0x90],xmm14
   5eaee:	movsd  xmm14,QWORD PTR [rsp+0x900]
   5eaf8:	divsd  xmm14,xmm5
   5eafd:	mulsd  xmm1,xmm1
   5eb01:	divsd  xmm8,xmm5
   5eb06:	movsd  QWORD PTR [rsp+0xa8],xmm14
   5eb10:	movsd  xmm14,QWORD PTR [rsp+0x908]
   5eb1a:	divsd  xmm14,xmm5
   5eb1f:	mulsd  xmm8,xmm8
   5eb24:	addsd  xmm1,xmm8
   5eb29:	addsd  xmm1,xmm2
   5eb2d:	movsd  xmm2,QWORD PTR [rsp]
   5eb32:	mulsd  xmm2,xmm2
   5eb36:	divsd  xmm0,xmm5
   5eb3a:	movsd  QWORD PTR [rsp+0xc0],xmm14
   5eb44:	movsd  xmm14,QWORD PTR [rsp+0x910]
   5eb4e:	divsd  xmm14,xmm5
   5eb53:	mulsd  xmm0,xmm0
   5eb57:	addsd  xmm0,xmm2
   5eb5b:	movsd  xmm2,QWORD PTR [rsp+0x28]
   5eb61:	mulsd  xmm2,xmm2
   5eb65:	addsd  xmm0,xmm1
   5eb69:	movsd  xmm1,QWORD PTR [rsp+0x18]
   5eb6f:	mulsd  xmm1,xmm1
   5eb73:	addsd  xmm1,xmm2
   5eb77:	movsd  xmm2,QWORD PTR [rsp+0x58]
   5eb7d:	mulsd  xmm2,xmm2
   5eb81:	addsd  xmm1,xmm0
   5eb85:	movsd  xmm0,QWORD PTR [rsp+0x50]
   5eb8b:	movsd  QWORD PTR [rsp+0xd8],xmm14
   5eb95:	mulsd  xmm0,xmm0
   5eb99:	movsd  xmm14,QWORD PTR [rsp+0x918]
   5eba3:	divsd  xmm14,xmm5
   5eba8:	addsd  xmm0,xmm2
   5ebac:	movsd  xmm2,QWORD PTR [rsp+0x70]
   5ebb2:	mulsd  xmm2,xmm2
   5ebb6:	addsd  xmm0,xmm1
   5ebba:	movsd  xmm1,QWORD PTR [rsp+0x60]
   5ebc0:	mulsd  xmm1,xmm1
   5ebc4:	addsd  xmm1,xmm2
   5ebc8:	movsd  xmm2,QWORD PTR [rsp+0x88]
   5ebd1:	mulsd  xmm2,xmm2
   5ebd5:	addsd  xmm0,xmm1
   5ebd9:	movsd  xmm1,QWORD PTR [rsp+0x80]
   5ebe2:	mulsd  xmm1,xmm1
   5ebe6:	addsd  xmm1,xmm2
   5ebea:	movsd  xmm2,QWORD PTR [rsp+0x90]
   5ebf3:	mulsd  xmm2,xmm2
   5ebf7:	addsd  xmm0,xmm1
   5ebfb:	movsd  xmm1,QWORD PTR [rsp+0xa0]
   5ec04:	movsd  QWORD PTR [rsp+0xe0],xmm14
   5ec0e:	mulsd  xmm1,xmm1
   5ec12:	movsd  xmm14,QWORD PTR [rsp+0x920]
   5ec1c:	divsd  xmm14,xmm5
   5ec21:	addsd  xmm1,xmm2
   5ec25:	movsd  xmm2,QWORD PTR [rsp+0xc0]
   5ec2e:	mulsd  xmm2,xmm2
   5ec32:	addsd  xmm0,xmm1
   5ec36:	movsd  xmm1,QWORD PTR [rsp+0xa8]
   5ec3f:	mulsd  xmm1,xmm1
   5ec43:	addsd  xmm1,xmm2
   5ec47:	movsd  xmm2,QWORD PTR [rsp+0xe0]
   5ec50:	mulsd  xmm2,xmm2
   5ec54:	addsd  xmm0,xmm1
   5ec58:	movsd  xmm1,QWORD PTR [rsp+0xd8]
   5ec61:	mulsd  xmm1,xmm1
   5ec65:	addsd  xmm1,xmm2
   5ec69:	addsd  xmm0,xmm1
   5ec6d:	movsd  QWORD PTR [rsp+0xe8],xmm14
   5ec77:	movsd  xmm14,QWORD PTR [rsp+0x928]
   5ec81:	movsd  xmm1,QWORD PTR [rsp+0xe8]
   5ec8a:	divsd  xmm14,xmm5
   5ec8f:	mulsd  xmm1,xmm1
   5ec93:	movsd  QWORD PTR [rsp+0xf0],xmm14
   5ec9d:	movsd  xmm14,QWORD PTR [rsp+0x930]
   5eca7:	divsd  xmm14,xmm5
   5ecac:	movapd xmm15,xmm14
   5ecb1:	movsd  xmm14,QWORD PTR [rsp+0x938]
   5ecbb:	movsd  xmm2,QWORD PTR [rsp+0xf0]
   5ecc4:	divsd  xmm14,xmm5
   5ecc9:	mulsd  xmm2,xmm2
   5eccd:	addsd  xmm1,xmm2
   5ecd1:	addsd  xmm0,xmm1
   5ecd5:	movapd xmm1,xmm15
   5ecda:	mulsd  xmm1,xmm15
   5ecdf:	mulsd  xmm14,xmm14
   5ece4:	addsd  xmm1,xmm14
   5ece9:	addsd  xmm0,xmm1
   5eced:	ucomisd xmm0,xmm13
   5ecf2:	jp     5ecfb <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3e7b>
   5ecf4:	movapd xmm6,xmm13
   5ecf9:	je     5ed18 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3e98>
   5ecfb:	movsd  xmm1,QWORD PTR [rip+0x4afd]        # 63800 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
   5ed03:	movapd xmm6,xmm1
   5ed07:	maxsd  xmm6,xmm0
   5ed0b:	movapd xmm0,xmm6
   5ed0f:	call   6a10 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   5ed14:	mulsd  xmm6,xmm0
   5ed18:	mulsd  xmm6,xmm5
   5ed1c:	divsd  xmm6,QWORD PTR [rsp+0x190]
   5ed25:	movsd  xmm0,QWORD PTR [rbp+0x50]
   5ed2a:	movsd  xmm14,QWORD PTR [rsp+0x240]
   5ed34:	movsd  xmm7,QWORD PTR [rsp+0x248]
   5ed3d:	comisd xmm0,xmm6
   5ed41:	setae  BYTE PTR [rsp+0xd8]
   5ed49:	setb   BYTE PTR [rsp+0x70]
   5ed4e:	cmp    BYTE PTR [rsp+0x36a1],0x0
   5ed56:	je     5ed69 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3ee9>
   5ed58:	cmp    BYTE PTR [rsp+0x36a0],0x0
   5ed60:	jne    5ed69 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3ee9>
   5ed62:	mov    BYTE PTR [rbp+0x215f],0x1
   5ed69:	add    r14,0x1
   5ed6d:	cmp    r14,r12
   5ed70:	jne    5b8cb <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xa4b>
   5ed76:	mov    rsi,r9
   5ed79:	mov    rax,rbp
   5ed7c:	mov    r9d,ebx
   5ed7f:	mov    rbp,r13
   5ed82:	mov    r15,QWORD PTR [rsp+0x8]
   5ed87:	mov    r12,QWORD PTR [rsp+0x68]
   5ed8c:	mov    r13,rax
   5ed8f:	mov    rbx,rsi
   5ed92:	mov    r10,QWORD PTR [rsp+0x180]
   5ed9a:	mov    r8,QWORD PTR [rsp+0x188]
   5eda2:	jmp    5dc2a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2daa>
   5eda7:	mov    rdx,QWORD PTR [rsp+0x150]
   5edaf:	movupd xmm0,XMMWORD PTR [rdx]
   5edb3:	add    rax,0x10
   5edb7:	add    rdx,0x10
   5edbb:	movaps XMMWORD PTR [rax-0x10],xmm0
   5edbf:	cmp    r12,rax
   5edc2:	jne    5edaf <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3f2f>
   5edc4:	test   cl,cl
   5edc6:	jne    5b1ac <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x32c>
   5edcc:	jmp    5b19a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x31a>
   5edd1:	nop    DWORD PTR [rax+0x0]
   5edd8:	movapd xmm6,xmm10
   5eddd:	mov    BYTE PTR [rsp+0xd8],0x0
   5ede5:	unpcklpd xmm6,xmm6
   5ede9:	movaps XMMWORD PTR [rsp+0x160],xmm6
   5edf1:	movapd xmm6,xmm12
   5edf6:	unpcklpd xmm6,xmm6
   5edfa:	movaps XMMWORD PTR [rsp+0x170],xmm6
   5ee02:	jmp    5dc2a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2daa>
   5ee07:	test   r9b,r9b
   5ee0a:	je     5e226 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x33a6>
   5ee10:	mov    rbx,QWORD PTR [rsp+0xd0]
   5ee18:	cmp    DWORD PTR [rbx+0x2158],0x5
   5ee1f:	je     5e226 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x33a6>
   5ee25:	lea    eax,[r15+0x1]
   5ee29:	mov    DWORD PTR [rbx+0x2158],eax
   5ee2f:	jmp    5e226 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x33a6>
   5ee34:	mov    rsi,r9
   5ee37:	mov    rax,rbp
   5ee3a:	mov    r11,r12
   5ee3d:	mov    r9d,ebx
   5ee40:	mov    rbp,r13
   5ee43:	mov    r15,QWORD PTR [rsp+0x8]
   5ee48:	mov    r12,QWORD PTR [rsp+0x68]
   5ee4d:	mov    r13,rax
   5ee50:	mov    r10,QWORD PTR [rsp+0x180]
   5ee58:	mov    r8,QWORD PTR [rsp+0x188]
   5ee60:	mov    rbx,rsi
   5ee63:	jmp    5dc18 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2d98>
   5ee68:	call   1170 <__stack_chk_fail@plt>

Disassembly of section .fini:
