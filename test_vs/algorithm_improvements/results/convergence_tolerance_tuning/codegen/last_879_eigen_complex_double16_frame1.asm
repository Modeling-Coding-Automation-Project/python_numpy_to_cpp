
/tmp/matrix_peak_stack_audit/last_879/last_879_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000030e80 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE>:
   30e80:	endbr64
   30e84:	push   r15
   30e86:	push   r14
   30e88:	push   r13
   30e8a:	push   r12
   30e8c:	push   rbp
   30e8d:	push   rbx
   30e8e:	sub    rsp,0x1000
   30e95:	or     QWORD PTR [rsp],0x0
   30e9a:	sub    rsp,0x1000
   30ea1:	or     QWORD PTR [rsp],0x0
   30ea6:	sub    rsp,0x1000
   30ead:	or     QWORD PTR [rsp],0x0
   30eb2:	sub    rsp,0x5f8
   30eb9:	pxor   xmm0,xmm0
   30ebd:	movsd  xmm1,QWORD PTR [rdi+0x28]
   30ec2:	mov    QWORD PTR [rsp+0xa0],rdi
   30eca:	mov    QWORD PTR [rsp+0x118],rsi
   30ed2:	mov    rax,QWORD PTR fs:0x28
   30edb:	mov    QWORD PTR [rsp+0x35e8],rax
   30ee3:	xor    eax,eax
   30ee5:	comisd xmm0,xmm1
   30ee9:	jb     30eff <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x7f>
   30eeb:	movsd  xmm1,QWORD PTR [rip+0x696d]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   30ef3:	mov    BYTE PTR [rdi+0x215e],0x1
   30efa:	movsd  QWORD PTR [rdi+0x28],xmm1
   30eff:	mov    rax,QWORD PTR [rsp+0xa0]
   30f07:	movzx  edi,BYTE PTR [rax+0x215c]
   30f0e:	mov    BYTE PTR [rsp+0x107],dil
   30f16:	mov    BYTE PTR [rax+0x215d],dil
   30f1d:	test   dil,dil
   30f20:	je     30f2c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xac>
   30f22:	mov    DWORD PTR [rax+0x2158],0x1
   30f2c:	mov    r12,QWORD PTR [rsp+0xa0]
   30f34:	pxor   xmm3,xmm3
   30f38:	movsd  xmm2,QWORD PTR [rip+0x68e8]        # 37828 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
   30f40:	lea    rdi,[rsp+0x180]
   30f48:	movapd xmm0,xmm3
   30f4c:	lea    rbx,[rsp+0x280]
   30f54:	movapd xmm7,xmm1
   30f58:	mov    QWORD PTR [rsp+0x110],rdi
   30f60:	movsd  xmm6,QWORD PTR [r12+0x2168]
   30f6a:	mov    QWORD PTR [rsp+0x28],rbx
   30f6f:	lea    rax,[r12+0x1158]
   30f77:	lea    rdi,[rsp+0x880]
   30f7f:	lea    r10,[rsp+0x28b0]
   30f87:	lea    rbx,[rsp+0x3450]
   30f8f:	mov    BYTE PTR [r12+0x215f],0x0
   30f98:	cmpnltsd xmm0,xmm6
   30f9d:	mov    QWORD PTR [rsp+0xd8],rax
   30fa5:	lea    r15,[rsp+0x1880]
   30fad:	lea    rbp,[rsp+0x780]
   30fb5:	mov    QWORD PTR [rsp+0xc0],0x0
   30fc1:	mov    r13,rbp
   30fc4:	lea    rbp,[rsp+0x1980]
   30fcc:	mov    QWORD PTR [rsp+0x10],rdi
   30fd1:	andpd  xmm2,xmm0
   30fd5:	andnpd xmm0,xmm6
   30fd9:	mov    QWORD PTR [rsp+0x140],rbx
   30fe1:	movapd xmm6,xmm0
   30fe5:	mov    QWORD PTR [rsp+0x60],r10
   30fea:	movsd  xmm0,QWORD PTR [r12+0x30]
   30ff1:	orpd   xmm6,xmm2
   30ff5:	mov    QWORD PTR [rsp],r15
   30ff9:	lea    r15,[r12+0x1060]
   31001:	movapd xmm4,xmm0
   31005:	movapd xmm2,xmm0
   31009:	xorpd  xmm2,XMMWORD PTR [rip+0x678f]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   31011:	movsd  QWORD PTR [r12+0x2170],xmm3
   3101b:	cmpltsd xmm0,xmm3
   31020:	andpd  xmm2,xmm0
   31024:	andnpd xmm0,xmm4
   31028:	orpd   xmm0,xmm2
   3102c:	mulsd  xmm0,xmm6
   31030:	maxsd  xmm7,xmm0
   31034:	mov    rdi,QWORD PTR [rsp+0x10]
   31039:	lea    rdx,[rsp+0x980]
   31041:	pxor   xmm0,xmm0
   31045:	lea    rax,[rdx-0x100]
   3104c:	nop    DWORD PTR [rax+0x0]
   31050:	movaps XMMWORD PTR [rax],xmm0
   31053:	add    rax,0x20
   31057:	movaps XMMWORD PTR [rax-0x10],xmm0
   3105b:	cmp    rax,rdx
   3105e:	jne    31050 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1d0>
   31060:	lea    rdx,[rax+0x100]
   31067:	cmp    rdx,rbp
   3106a:	jne    31045 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1c5>
   3106c:	mov    rcx,QWORD PTR [rsp+0x118]
   31074:	movsd  xmm1,QWORD PTR [r15-0x8]
   3107a:	mov    rdx,QWORD PTR [rsp+0x10]
   3107f:	mov    rbx,rcx
   31082:	addsd  xmm1,xmm7
   31086:	movhpd xmm1,QWORD PTR [r15]
   3108b:	xor    eax,eax
   3108d:	nop    DWORD PTR [rax+0x0]
   31094:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   3109f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   310aa:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   310b5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   310c0:	movupd xmm0,XMMWORD PTR [rcx+rax*1]
   310c5:	movq   xmm2,xmm0
   310c9:	psrldq xmm0,0x8
   310ce:	movaps XMMWORD PTR [rdx+rax*2],xmm2
   310d2:	movaps XMMWORD PTR [rdx+rax*2+0x10],xmm0
   310d7:	add    rax,0x10
   310db:	cmp    rax,0x80
   310e1:	jne    310c0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x240>
   310e3:	movapd xmm0,XMMWORD PTR [rdi]
   310e7:	add    rdi,0x110
   310ee:	sub    rcx,0xffffffffffffff80
   310f2:	add    rdx,0x100
   310f9:	subpd  xmm0,xmm1
   310fd:	movaps XMMWORD PTR [rdi-0x110],xmm0
   31104:	cmp    rdi,rbp
   31107:	jne    3108b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x20b>
   31109:	lea    rax,[rsp+0x180]
   31111:	pxor   xmm0,xmm0
   31115:	movaps XMMWORD PTR [rax],xmm0
   31118:	add    rax,0x20
   3111c:	movaps XMMWORD PTR [rax-0x10],xmm0
   31120:	cmp    rax,QWORD PTR [rsp+0x28]
   31125:	jne    31115 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x295>
   31127:	movzx  ecx,BYTE PTR [r12+0x2161]
   31130:	test   cl,cl
   31132:	jne    33164 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x22e4>
   31138:	lea    rax,[rsp+0x180]
   31140:	movq   xmm0,QWORD PTR [rip+0x6808]        # 37950 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x1d0>
   31148:	jmp    31157 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2d7>
   3114a:	movaps XMMWORD PTR [rax+0x10],xmm0
   3114e:	add    rax,0x20
   31152:	cmp    rax,rdi
   31155:	je     31168 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2e8>
   31157:	mov    rdi,QWORD PTR [rsp+0x28]
   3115c:	lea    rdx,[rax+0x10]
   31160:	movaps XMMWORD PTR [rax],xmm0
   31163:	cmp    rdi,rdx
   31166:	jne    3114a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2ca>
   31168:	mov    rax,QWORD PTR [rsp+0x110]
   31170:	movsd  xmm0,QWORD PTR [rip+0x66b0]        # 37828 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
   31178:	addsd  xmm0,QWORD PTR [rax]
   3117c:	movsd  QWORD PTR [rax],xmm0
   31180:	pxor   xmm5,xmm5
   31184:	lea    rax,[rsp+0x180]
   3118c:	movapd xmm2,xmm5
   31190:	movsd  xmm1,QWORD PTR [rax]
   31194:	add    rax,0x10
   31198:	movapd xmm3,xmm1
   3119c:	movapd xmm0,xmm1
   311a0:	xorpd  xmm0,XMMWORD PTR [rip+0x65f8]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   311a8:	cmpltsd xmm1,xmm2
   311ad:	andpd  xmm0,xmm1
   311b1:	andnpd xmm1,xmm3
   311b5:	orpd   xmm1,xmm0
   311b9:	movsd  xmm0,QWORD PTR [rax-0x8]
   311be:	movapd xmm4,xmm0
   311c2:	movapd xmm3,xmm0
   311c6:	xorpd  xmm3,XMMWORD PTR [rip+0x65d2]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   311ce:	cmpltsd xmm0,xmm2
   311d3:	andpd  xmm3,xmm0
   311d7:	andnpd xmm0,xmm4
   311db:	orpd   xmm0,xmm3
   311df:	movapd xmm3,xmm1
   311e3:	maxsd  xmm3,xmm0
   311e7:	maxsd  xmm3,xmm5
   311eb:	movapd xmm5,xmm3
   311ef:	cmp    rax,QWORD PTR [rsp+0x28]
   311f4:	jne    31190 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x310>
   311f6:	pxor   xmm0,xmm0
   311fa:	ucomisd xmm3,xmm0
   311fe:	jp     31206 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x386>
   31200:	je     31289 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x409>
   31206:	movapd xmm3,xmm5
   3120a:	lea    rax,[rsp+0x180]
   31212:	pxor   xmm2,xmm2
   31216:	unpcklpd xmm3,xmm3
   3121a:	movapd xmm0,XMMWORD PTR [rax]
   3121e:	movapd xmm1,XMMWORD PTR [rax+0x10]
   31223:	add    rax,0x20
   31227:	movlpd xmm1,QWORD PTR [rax-0x18]
   3122c:	movhpd xmm0,QWORD PTR [rax-0x10]
   31231:	divpd  xmm0,xmm3
   31235:	divpd  xmm1,xmm3
   31239:	mulpd  xmm0,xmm0
   3123d:	mulpd  xmm1,xmm1
   31241:	addpd  xmm0,xmm1
   31245:	addsd  xmm2,xmm0
   31249:	unpckhpd xmm0,xmm0
   3124d:	addsd  xmm2,xmm0
   31251:	cmp    rax,QWORD PTR [rsp+0x28]
   31256:	jne    3121a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x39a>
   31258:	pxor   xmm0,xmm0
   3125c:	ucomisd xmm2,xmm0
   31260:	jp     31264 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x3e4>
   31262:	je     31285 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x405>
   31264:	movsd  xmm1,QWORD PTR [rip+0x65f4]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   3126c:	movapd xmm8,xmm1
   31271:	maxsd  xmm8,xmm2
   31276:	movapd xmm0,xmm8
   3127b:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   31280:	mulsd  xmm0,xmm8
   31285:	mulsd  xmm0,xmm5
   31289:	mov    rdx,QWORD PTR [rsp+0x28]
   3128e:	unpcklpd xmm0,xmm0
   31292:	lea    rax,[rsp+0x180]
   3129a:	movapd xmm1,XMMWORD PTR [rax]
   3129e:	add    rax,0x10
   312a2:	divpd  xmm1,xmm0
   312a6:	movaps XMMWORD PTR [rax-0x10],xmm1
   312aa:	cmp    rax,rdx
   312ad:	jne    3129a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x41a>
   312af:	mov    QWORD PTR [rsp+0x18a8],0x0
   312bb:	mov    rdx,QWORD PTR [rsp+0x60]
   312c0:	lea    rax,[rsp+0x18b0]
   312c8:	pxor   xmm0,xmm0
   312cc:	nop    DWORD PTR [rax+0x0]
   312d0:	movaps XMMWORD PTR [rax],xmm0
   312d3:	add    rax,0x20
   312d7:	movaps XMMWORD PTR [rax-0x10],xmm0
   312db:	cmp    rax,rdx
   312de:	jne    312d0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x450>
   312e0:	mov    rax,QWORD PTR [rsp]
   312e4:	pxor   xmm0,xmm0
   312e8:	lea    r10,[rax+0x18b0]
   312ef:	mov    rax,QWORD PTR [rsp+0x60]
   312f4:	nop
   312f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   31300:	movaps XMMWORD PTR [rax],xmm0
   31303:	add    rax,0x20
   31307:	movaps XMMWORD PTR [rax-0x10],xmm0
   3130b:	cmp    rax,r10
   3130e:	jne    31300 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x480>
   31310:	mov    rax,QWORD PTR [rsp]
   31314:	pxor   xmm0,xmm0
   31318:	lea    rdx,[rax+0x19c0]
   3131f:	mov    rax,r10
   31322:	mov    QWORD PTR [rsp+0x58],rdx
   31327:	sub    rdx,r10
   3132a:	and    edx,0x10
   3132d:	je     31345 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x4c5>
   3132f:	mov    rax,QWORD PTR [rsp]
   31333:	movaps XMMWORD PTR [rax+0x18b0],xmm0
   3133a:	lea    rax,[r10+0x10]
   3133e:	cmp    rax,QWORD PTR [rsp+0x58]
   31343:	je     31357 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x4d7>
   31345:	movaps XMMWORD PTR [rax],xmm0
   31348:	add    rax,0x20
   3134c:	movaps XMMWORD PTR [rax-0x10],xmm0
   31350:	cmp    rax,QWORD PTR [rsp+0x58]
   31355:	jne    31345 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x4c5>
   31357:	mov    rax,QWORD PTR [rsp]
   3135b:	mov    rdi,QWORD PTR [rsp+0x58]
   31360:	pxor   xmm0,xmm0
   31364:	lea    rdx,[rax+0x1ad0]
   3136b:	mov    rax,rdi
   3136e:	mov    QWORD PTR [rsp+0x40],rdx
   31373:	sub    rdx,rdi
   31376:	and    edx,0x10
   31379:	je     31396 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x516>
   3137b:	mov    rax,QWORD PTR [rsp]
   3137f:	movaps XMMWORD PTR [rax+0x19c0],xmm0
   31386:	mov    rax,QWORD PTR [rsp+0x58]
   3138b:	add    rax,0x10
   3138f:	cmp    rax,QWORD PTR [rsp+0x40]
   31394:	je     313a8 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x528>
   31396:	movaps XMMWORD PTR [rax],xmm0
   31399:	add    rax,0x20
   3139d:	movaps XMMWORD PTR [rax-0x10],xmm0
   313a1:	cmp    rax,QWORD PTR [rsp+0x40]
   313a6:	jne    31396 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x516>
   313a8:	mov    rax,QWORD PTR [rsp]
   313ac:	mov    rdi,QWORD PTR [rsp+0x40]
   313b1:	pxor   xmm0,xmm0
   313b5:	lea    rdx,[rax+0x1bd0]
   313bc:	mov    rax,rdi
   313bf:	mov    QWORD PTR [rsp+0xd0],rdx
   313c7:	sub    rdx,rdi
   313ca:	and    edx,0x10
   313cd:	je     313ed <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x56d>
   313cf:	mov    rax,QWORD PTR [rsp]
   313d3:	movaps XMMWORD PTR [rax+0x1ad0],xmm0
   313da:	mov    rax,QWORD PTR [rsp+0x40]
   313df:	add    rax,0x10
   313e3:	cmp    rax,QWORD PTR [rsp+0xd0]
   313eb:	je     31402 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x582>
   313ed:	movaps XMMWORD PTR [rax],xmm0
   313f0:	add    rax,0x20
   313f4:	movaps XMMWORD PTR [rax-0x10],xmm0
   313f8:	cmp    rax,QWORD PTR [rsp+0xd0]
   31400:	jne    313ed <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x56d>
   31402:	mov    rdi,QWORD PTR [rsp+0x140]
   3140a:	xor    eax,eax
   3140c:	mov    ecx,0x10
   31411:	pxor   xmm0,xmm0
   31415:	rep stos QWORD PTR [rdi],rax
   31418:	mov    rax,QWORD PTR [rsp]
   3141c:	lea    rdi,[rax+0x1d50]
   31423:	add    rax,0x1c50
   31429:	mov    QWORD PTR [rsp+0x108],rdi
   31431:	movaps XMMWORD PTR [rax],xmm0
   31434:	add    rax,0x20
   31438:	movaps XMMWORD PTR [rax-0x10],xmm0
   3143c:	cmp    rax,QWORD PTR [rsp+0x108]
   31444:	jne    31431 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x5b1>
   31446:	movsd  xmm0,QWORD PTR [r12+0x28]
   3144d:	movdqu xmm5,XMMWORD PTR [r12+0x10]
   31454:	pxor   xmm1,xmm1
   31458:	mov    r8,QWORD PTR [r12+0x20]
   3145d:	movsd  QWORD PTR [rsp+0x1898],xmm0
   31466:	movdqa xmm0,xmm5
   3146a:	shufpd xmm0,xmm5,0x1
   3146f:	movaps XMMWORD PTR [rsp+0x1880],xmm0
   31477:	movsd  xmm0,QWORD PTR [r12+0x38]
   3147e:	mov    QWORD PTR [rsp+0x1890],r8
   31486:	comisd xmm0,xmm1
   3148a:	ja     314b4 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x634>
   3148c:	movsd  xmm8,QWORD PTR [rip+0x6523]        # 379b8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x238>
   31495:	movsd  xmm1,QWORD PTR [rip+0x6513]        # 379b0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x230>
   3149d:	movapd xmm0,xmm8
   314a2:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   314a7:	mulsd  xmm0,xmm8
   314ac:	mulsd  xmm0,QWORD PTR [rip+0x652c]        # 379e0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x260>
   314b4:	movsd  QWORD PTR [rsp+0x18a0],xmm0
   314bd:	mov    r9,QWORD PTR [r12+0x8]
   314c2:	test   r9,r9
   314c5:	je     331a0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2320>
   314cb:	cmp    r8,0x2
   314cf:	mov    BYTE PTR [rsp+0x106],0x0
   314d7:	sbb    rax,rax
   314da:	mov    QWORD PTR [rsp+0xe8],r9
   314e2:	add    rax,0x2
   314e6:	mov    QWORD PTR [rsp+0x38],r10
   314eb:	mov    QWORD PTR [rsp+0x48],rax
   314f0:	movzx  eax,BYTE PTR [rsp+0x107]
   314f8:	mov    QWORD PTR [rsp+0xf8],rbp
   31500:	mov    rbp,QWORD PTR [rsp+0x28]
   31505:	mov    BYTE PTR [rsp+0x105],al
   3150c:	mov    QWORD PTR [rsp+0xe0],rbx
   31514:	xor    ebx,ebx
   31516:	mov    QWORD PTR [rsp+0xb0],r15
   3151e:	mov    r15,r13
   31521:	movq   QWORD PTR [rsp+0x138],xmm5
   3152a:	movhps QWORD PTR [rsp+0x90],xmm5
   31532:	movsd  QWORD PTR [rsp+0xa8],xmm6
   3153b:	movsd  QWORD PTR [rsp+0xf0],xmm7
   31544:	cmp    BYTE PTR [rsp+0x105],0x0
   3154c:	je     331de <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x235e>
   31552:	lea    r8,[rsp+0x280]
   3155a:	lea    r11,[rsp+0x380]
   31562:	pxor   xmm0,xmm0
   31566:	mov    rax,r8
   31569:	movaps XMMWORD PTR [rax],xmm0
   3156c:	add    rax,0x20
   31570:	movaps XMMWORD PTR [rax-0x10],xmm0
   31574:	cmp    rax,r11
   31577:	jne    31569 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x6e9>
   31579:	pxor   xmm0,xmm0
   3157d:	pxor   xmm7,xmm7
   31581:	movsd  xmm2,QWORD PTR [rsp+0x1898]
   3158a:	movsd  xmm3,QWORD PTR [rip+0x625e]        # 377f0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x70>
   31592:	movapd xmm1,xmm0
   31596:	movsd  xmm6,QWORD PTR [rsp+0x18a0]
   3159f:	movsd  xmm5,QWORD PTR [rsp+0x18a8]
   315a8:	mov    QWORD PTR [rsp+0x35d8],0x0
   315b4:	cmpnltsd xmm1,xmm2
   315b9:	comisd xmm0,xmm2
   315bd:	mov    BYTE PTR [rsp+0x35e1],0x0
   315c5:	lea    rax,[rsp+0x180]
   315cd:	andpd  xmm3,xmm1
   315d1:	andnpd xmm1,xmm2
   315d5:	setae  BYTE PTR [rsp+0x35e2]
   315dd:	orpd   xmm1,xmm3
   315e1:	movapd xmm3,xmm7
   315e5:	movsd  QWORD PTR [rsp+0x98],xmm1
   315ee:	movapd xmm1,xmm0
   315f2:	cmpltsd xmm0,xmm5
   315f7:	cmpltsd xmm1,xmm6
   315fc:	andpd  xmm5,xmm0
   31600:	andpd  xmm6,xmm1
   31604:	movsd  xmm1,QWORD PTR [rax]
   31608:	add    rax,0x10
   3160c:	movapd xmm2,xmm1
   31610:	movapd xmm0,xmm1
   31614:	xorpd  xmm0,XMMWORD PTR [rip+0x6184]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   3161c:	cmpltsd xmm1,xmm3
   31621:	andpd  xmm0,xmm1
   31625:	andnpd xmm1,xmm2
   31629:	orpd   xmm1,xmm0
   3162d:	movsd  xmm0,QWORD PTR [rax-0x8]
   31632:	movapd xmm4,xmm0
   31636:	movapd xmm2,xmm0
   3163a:	xorpd  xmm2,XMMWORD PTR [rip+0x615e]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   31642:	cmpltsd xmm0,xmm3
   31647:	andpd  xmm2,xmm0
   3164b:	andnpd xmm0,xmm4
   3164f:	orpd   xmm0,xmm2
   31653:	movapd xmm2,xmm1
   31657:	maxsd  xmm2,xmm0
   3165b:	maxsd  xmm2,xmm7
   3165f:	movapd xmm7,xmm2
   31663:	cmp    rax,rbp
   31666:	jne    31604 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x784>
   31668:	pxor   xmm0,xmm0
   3166c:	ucomisd xmm2,xmm0
   31670:	jp     31678 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x7f8>
   31672:	je     316f9 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x879>
   31678:	movapd xmm3,xmm7
   3167c:	lea    rax,[rsp+0x180]
   31684:	pxor   xmm2,xmm2
   31688:	unpcklpd xmm3,xmm3
   3168c:	movapd xmm0,XMMWORD PTR [rax]
   31690:	movapd xmm1,XMMWORD PTR [rax+0x10]
   31695:	add    rax,0x20
   31699:	movlpd xmm1,QWORD PTR [rax-0x18]
   3169e:	movhpd xmm0,QWORD PTR [rax-0x10]
   316a3:	divpd  xmm0,xmm3
   316a7:	divpd  xmm1,xmm3
   316ab:	mulpd  xmm0,xmm0
   316af:	mulpd  xmm1,xmm1
   316b3:	addpd  xmm0,xmm1
   316b7:	addsd  xmm2,xmm0
   316bb:	unpckhpd xmm0,xmm0
   316bf:	addsd  xmm2,xmm0
   316c3:	cmp    rax,rbp
   316c6:	jne    3168c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x80c>
   316c8:	pxor   xmm0,xmm0
   316cc:	ucomisd xmm2,xmm0
   316d0:	jp     316d4 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x854>
   316d2:	je     316f5 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x875>
   316d4:	movsd  xmm1,QWORD PTR [rip+0x6184]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   316dc:	movapd xmm8,xmm1
   316e1:	maxsd  xmm8,xmm2
   316e6:	movapd xmm0,xmm8
   316eb:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   316f0:	mulsd  xmm0,xmm8
   316f5:	mulsd  xmm0,xmm7
   316f9:	mulsd  xmm6,xmm0
   316fd:	mov    QWORD PTR [rsp+0x130],r11
   31705:	mov    rax,r11
   31708:	addsd  xmm6,xmm5
   3170c:	movsd  QWORD PTR [rsp+0xc8],xmm6
   31715:	movapd xmm0,XMMWORD PTR [r8]
   3171a:	add    r8,0x10
   3171e:	add    rax,0x10
   31722:	movaps XMMWORD PTR [rax-0x10],xmm0
   31726:	cmp    r11,r8
   31729:	jne    31715 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x895>
   3172b:	lea    rax,[rsp+0x480]
   31733:	lea    r14,[rsp+0x580]
   3173b:	pxor   xmm0,xmm0
   3173f:	mov    QWORD PTR [rsp+0x30],rax
   31744:	mov    rdx,rax
   31747:	movaps XMMWORD PTR [rdx],xmm0
   3174a:	add    rdx,0x20
   3174e:	movaps XMMWORD PTR [rdx-0x10],xmm0
   31752:	cmp    r14,rdx
   31755:	jne    31747 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x8c7>
   31757:	mov    r13,QWORD PTR [rsp+0x30]
   3175c:	mov    rdi,QWORD PTR [rsp+0x10]
   31761:	mov    rdx,r11
   31764:	mov    r8d,0x10
   3176a:	mov    esi,0x10
   3176f:	mov    QWORD PTR [rsp+0x8],r11
   31774:	mov    rcx,r13
   31777:	call   99c0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0>
   3177c:	mov    r11,QWORD PTR [rsp+0x8]
   31781:	mov    rcx,r13
   31784:	lea    rdx,[rsp+0x180]
   3178c:	movapd xmm0,XMMWORD PTR [rdx]
   31790:	subpd  xmm0,XMMWORD PTR [rcx]
   31794:	add    rdx,0x10
   31798:	add    rcx,0x10
   3179c:	movaps XMMWORD PTR [rcx-0x10],xmm0
   317a0:	cmp    rdx,rbp
   317a3:	jne    3178c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x90c>
   317a5:	pxor   xmm5,xmm5
   317a9:	mov    rdx,QWORD PTR [rsp+0x30]
   317ae:	movapd xmm2,xmm5
   317b2:	movsd  xmm1,QWORD PTR [rdx]
   317b6:	add    rdx,0x10
   317ba:	movapd xmm3,xmm1
   317be:	movapd xmm0,xmm1
   317c2:	xorpd  xmm0,XMMWORD PTR [rip+0x5fd6]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   317ca:	cmpltsd xmm1,xmm2
   317cf:	andpd  xmm0,xmm1
   317d3:	andnpd xmm1,xmm3
   317d7:	orpd   xmm1,xmm0
   317db:	movsd  xmm0,QWORD PTR [rdx-0x8]
   317e0:	movapd xmm4,xmm0
   317e4:	movapd xmm3,xmm0
   317e8:	xorpd  xmm3,XMMWORD PTR [rip+0x5fb0]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   317f0:	cmpltsd xmm0,xmm2
   317f5:	andpd  xmm3,xmm0
   317f9:	andnpd xmm0,xmm4
   317fd:	orpd   xmm0,xmm3
   31801:	movapd xmm3,xmm1
   31805:	maxsd  xmm3,xmm0
   31809:	maxsd  xmm3,xmm5
   3180d:	movapd xmm5,xmm3
   31811:	cmp    rdx,r14
   31814:	jne    317b2 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x932>
   31816:	pxor   xmm0,xmm0
   3181a:	ucomisd xmm3,xmm0
   3181e:	jp     31822 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x9a2>
   31820:	je     3189c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xa1c>
   31822:	movapd xmm3,xmm5
   31826:	mov    rdx,QWORD PTR [rsp+0x30]
   3182b:	pxor   xmm2,xmm2
   3182f:	unpcklpd xmm3,xmm3
   31833:	movapd xmm0,XMMWORD PTR [rdx]
   31837:	movapd xmm1,XMMWORD PTR [rdx+0x10]
   3183c:	add    rdx,0x20
   31840:	movlpd xmm1,QWORD PTR [rdx-0x18]
   31845:	movhpd xmm0,QWORD PTR [rdx-0x10]
   3184a:	divpd  xmm0,xmm3
   3184e:	divpd  xmm1,xmm3
   31852:	mulpd  xmm0,xmm0
   31856:	mulpd  xmm1,xmm1
   3185a:	addpd  xmm0,xmm1
   3185e:	addsd  xmm2,xmm0
   31862:	unpckhpd xmm0,xmm0
   31866:	addsd  xmm2,xmm0
   3186a:	cmp    rdx,r14
   3186d:	jne    31833 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x9b3>
   3186f:	pxor   xmm0,xmm0
   31873:	ucomisd xmm2,xmm0
   31877:	jp     3187b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x9fb>
   31879:	je     31898 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xa18>
   3187b:	movsd  xmm1,QWORD PTR [rip+0x5fdd]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   31883:	movapd xmm6,xmm1
   31887:	maxsd  xmm6,xmm2
   3188b:	movapd xmm0,xmm6
   3188f:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   31894:	mulsd  xmm0,xmm6
   31898:	mulsd  xmm0,xmm5
   3189c:	movsd  xmm3,QWORD PTR [rsp+0xc8]
   318a5:	lea    r13,[r11+0x100]
   318ac:	movsd  QWORD PTR [rsp+0x35d0],xmm0
   318b5:	comisd xmm3,xmm0
   318b9:	setae  dl
   318bc:	mov    BYTE PTR [rsp+0x35e0],dl
   318c3:	cmp    QWORD PTR [rsp+0x90],0x0
   318cc:	je     326f8 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1878>
   318d2:	mov    rdi,QWORD PTR [rsp+0x138]
   318da:	mov    eax,0x10
   318df:	mov    ecx,0x1
   318e4:	lea    r13,[r11+0x100]
   318eb:	mov    QWORD PTR [rsp+0x20],0x0
   318f4:	mov    r10,r14
   318f7:	cmp    rdi,0x10
   318fb:	mov    QWORD PTR [rsp+0x148],rbx
   31903:	cmovbe rax,rdi
   31907:	mov    QWORD PTR [rsp+0x120],r13
   3190f:	mov    QWORD PTR [rsp+0xb8],r11
   31917:	test   rax,rax
   3191a:	mov    QWORD PTR [rsp+0x128],rbp
   31922:	cmovne rcx,rax
   31926:	mov    QWORD PTR [rsp+0x150],r12
   3192e:	mov    r12,QWORD PTR [rsp+0x58]
   31933:	mov    QWORD PTR [rsp+0x8],rcx
   31938:	test   dl,dl
   3193a:	jne    323b0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1530>
   31940:	cmp    BYTE PTR [rsp+0x35e1],0x0
   31948:	jne    323b0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1530>
   3194e:	mov    rdx,QWORD PTR [rsp+0x40]
   31953:	mov    rax,r12
   31956:	pxor   xmm0,xmm0
   3195a:	movsd  xmm4,QWORD PTR [rsp+0x35d0]
   31963:	sub    rdx,r12
   31966:	movapd xmm5,xmm4
   3196a:	and    edx,0x10
   3196d:	unpcklpd xmm5,xmm5
   31971:	je     3197f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xaff>
   31973:	lea    rax,[r12+0x10]
   31978:	cmp    rax,QWORD PTR [rsp+0x40]
   3197d:	je     31994 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xb14>
   3197f:	mov    rdx,QWORD PTR [rsp+0x40]
   31984:	movaps XMMWORD PTR [rax],xmm0
   31987:	add    rax,0x20
   3198b:	movaps XMMWORD PTR [rax-0x10],xmm0
   3198f:	cmp    rax,rdx
   31992:	jne    31984 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xb04>
   31994:	movsd  xmm3,QWORD PTR [rip+0x5fec]        # 37988 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x208>
   3199c:	pxor   xmm1,xmm1
   319a0:	mov    QWORD PTR [rsp+0x3248],0x0
   319ac:	movsd  QWORD PTR [rsp+0x3240],xmm4
   319b5:	comisd xmm4,xmm3
   319b9:	jb     319c7 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xb47>
   319bb:	movsd  xmm1,QWORD PTR [rip+0x5e65]        # 37828 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
   319c3:	divsd  xmm1,xmm4
   319c7:	mov    rax,QWORD PTR [rsp+0x30]
   319cc:	unpcklpd xmm1,xmm1
   319d0:	lea    rdx,[rsp+0x18b0]
   319d8:	jmp    319ef <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xb6f>
   319da:	mulpd  xmm0,xmm1
   319de:	add    rax,0x10
   319e2:	add    rdx,0x10
   319e6:	movaps XMMWORD PTR [rdx-0x10],xmm0
   319ea:	cmp    rax,r10
   319ed:	je     31a0e <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xb8e>
   319ef:	comisd xmm4,xmm3
   319f3:	movapd xmm0,XMMWORD PTR [rax]
   319f7:	jae    319da <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xb5a>
   319f9:	divpd  xmm0,xmm5
   319fd:	add    rax,0x10
   31a01:	add    rdx,0x10
   31a05:	movaps XMMWORD PTR [rdx-0x10],xmm0
   31a09:	cmp    rax,r10
   31a0c:	jne    319ef <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xb6f>
   31a0e:	mov    rax,r10
   31a11:	lea    rbx,[rsp+0x680]
   31a19:	pxor   xmm0,xmm0
   31a1d:	movaps XMMWORD PTR [rax],xmm0
   31a20:	add    rax,0x20
   31a24:	movaps XMMWORD PTR [rax-0x10],xmm0
   31a28:	cmp    rbx,rax
   31a2b:	jne    31a1d <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xb9d>
   31a2d:	mov    rax,rbx
   31a30:	pxor   xmm0,xmm0
   31a34:	movaps XMMWORD PTR [rax],xmm0
   31a37:	add    rax,0x20
   31a3b:	movaps XMMWORD PTR [rax-0x10],xmm0
   31a3f:	cmp    r15,rax
   31a42:	jne    31a34 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xbb4>
   31a44:	mov    rax,QWORD PTR [rsp+0xd0]
   31a4c:	mov    r9,r12
   31a4f:	xor    edi,edi
   31a51:	xor    ecx,ecx
   31a53:	mov    QWORD PTR [rsp+0x50],r10
   31a58:	mov    r13,QWORD PTR [rsp+0x38]
   31a5d:	xor    r14d,r14d
   31a60:	lea    r11,[rsp+0x18b0]
   31a68:	movzx  edx,BYTE PTR [rsp+0x105]
   31a70:	mov    r12,rax
   31a73:	movsd  QWORD PTR [rsp+0x158],xmm4
   31a7c:	jmp    31a98 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xc18>
   31a7e:	xchg   ax,ax
   31a80:	mov    r13,rax
   31a83:	add    r12,0x8
   31a87:	add    r11,0x100
   31a8e:	cmp    r14,0x10
   31a92:	je     323e5 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1565>
   31a98:	cmp    r14,QWORD PTR [rsp+0x8]
   31a9d:	mov    rbp,r14
   31aa0:	lea    rax,[r13+0x10]
   31aa4:	setb   sil
   31aa8:	add    r14,0x1
   31aac:	and    sil,dl
   31aaf:	mov    BYTE PTR [rsp+0x18],sil
   31ab4:	je     31a80 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xc00>
   31ab6:	mov    rax,QWORD PTR [rsp+0x50]
   31abb:	mov    rdx,r11
   31abe:	pxor   xmm5,xmm5
   31ac2:	nop    DWORD PTR [rax+0x0]
   31ac9:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   31ad4:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   31adf:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   31aea:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   31af5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   31b00:	movapd xmm0,XMMWORD PTR [rdx]
   31b04:	add    rax,0x10
   31b08:	add    rdx,0x10
   31b0c:	movapd xmm1,xmm0
   31b10:	movapd xmm4,xmm0
   31b14:	shufpd xmm1,xmm0,0x1
   31b19:	mulpd  xmm1,xmm5
   31b1d:	subpd  xmm4,xmm1
   31b21:	addpd  xmm0,xmm1
   31b25:	movsd  xmm0,xmm4
   31b29:	movaps XMMWORD PTR [rax-0x10],xmm0
   31b2d:	cmp    rax,rbx
   31b30:	jne    31b00 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xc80>
   31b32:	mov    rdx,QWORD PTR [rsp+0x50]
   31b37:	mov    rdi,QWORD PTR [rsp+0x10]
   31b3c:	mov    rcx,rbx
   31b3f:	mov    esi,0x10
   31b44:	mov    r8d,0x10
   31b4a:	mov    QWORD PTR [rsp+0x70],r11
   31b4f:	mov    QWORD PTR [rsp+0x68],r9
   31b54:	call   99c0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0>
   31b59:	pxor   xmm6,xmm6
   31b5d:	mov    rax,rbx
   31b60:	mov    r11,QWORD PTR [rsp+0x70]
   31b65:	mov    r9,QWORD PTR [rsp+0x68]
   31b6a:	movapd xmm2,XMMWORD PTR [rip+0x5d4e]        # 378c0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x140>
   31b72:	movapd xmm7,xmm6
   31b76:	movsd  xmm3,QWORD PTR [rip+0x5e0a]        # 37988 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x208>
   31b7e:	xchg   ax,ax
   31b80:	movsd  xmm0,QWORD PTR [rax]
   31b84:	add    rax,0x10
   31b88:	movapd xmm1,xmm0
   31b8c:	movapd xmm4,xmm0
   31b90:	xorpd  xmm4,XMMWORD PTR [rip+0x5c08]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   31b98:	cmpltsd xmm1,xmm7
   31b9d:	andpd  xmm4,xmm1
   31ba1:	andnpd xmm1,xmm0
   31ba5:	orpd   xmm1,xmm4
   31ba9:	movapd xmm0,xmm1
   31bad:	movsd  xmm1,QWORD PTR [rax-0x8]
   31bb2:	movapd xmm4,xmm1
   31bb6:	movapd xmm5,xmm1
   31bba:	xorpd  xmm5,XMMWORD PTR [rip+0x5bde]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   31bc2:	cmpltsd xmm4,xmm7
   31bc7:	andpd  xmm5,xmm4
   31bcb:	andnpd xmm4,xmm1
   31bcf:	orpd   xmm4,xmm5
   31bd3:	movapd xmm1,xmm4
   31bd7:	movapd xmm4,xmm0
   31bdb:	maxsd  xmm4,xmm1
   31bdf:	maxsd  xmm4,xmm6
   31be3:	movapd xmm6,xmm4
   31be7:	cmp    r15,rax
   31bea:	jne    31b80 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xd00>
   31bec:	pxor   xmm0,xmm0
   31bf0:	ucomisd xmm4,xmm0
   31bf4:	jp     31c00 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xd80>
   31bf6:	movapd xmm5,xmm0
   31bfa:	je     31cbd <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xe3d>
   31c00:	movapd xmm5,xmm6
   31c04:	mov    rax,rbx
   31c07:	pxor   xmm4,xmm4
   31c0b:	unpcklpd xmm5,xmm5
   31c0f:	nop    DWORD PTR [rax+rax*1+0x0]
   31c14:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   31c1f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   31c2a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   31c35:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   31c40:	movapd xmm0,XMMWORD PTR [rax]
   31c44:	movapd xmm1,XMMWORD PTR [rax+0x10]
   31c49:	add    rax,0x20
   31c4d:	movlpd xmm1,QWORD PTR [rax-0x18]
   31c52:	movhpd xmm0,QWORD PTR [rax-0x10]
   31c57:	divpd  xmm0,xmm5
   31c5b:	divpd  xmm1,xmm5
   31c5f:	mulpd  xmm0,xmm0
   31c63:	mulpd  xmm1,xmm1
   31c67:	addpd  xmm0,xmm1
   31c6b:	addsd  xmm4,xmm0
   31c6f:	unpckhpd xmm0,xmm0
   31c73:	addsd  xmm4,xmm0
   31c77:	cmp    r15,rax
   31c7a:	jne    31c40 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xdc0>
   31c7c:	pxor   xmm0,xmm0
   31c80:	ucomisd xmm4,xmm0
   31c84:	jp     31c8c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xe0c>
   31c86:	movapd xmm5,xmm0
   31c8a:	je     31cb9 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xe39>
   31c8c:	movsd  xmm1,QWORD PTR [rip+0x5bcc]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   31c94:	movapd xmm5,xmm1
   31c98:	maxsd  xmm5,xmm4
   31c9c:	movapd xmm0,xmm5
   31ca0:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   31ca5:	movapd xmm2,XMMWORD PTR [rip+0x5c13]        # 378c0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x140>
   31cad:	movsd  xmm3,QWORD PTR [rip+0x5cd3]        # 37988 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x208>
   31cb5:	mulsd  xmm5,xmm0
   31cb9:	mulsd  xmm5,xmm6
   31cbd:	mov    rdi,QWORD PTR [rsp+0x38]
   31cc2:	mov    rdx,r9
   31cc5:	pxor   xmm0,xmm0
   31cc9:	sub    rdx,rdi
   31ccc:	mov    rax,rdi
   31ccf:	and    edx,0x10
   31cd2:	je     31cf0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xe70>
   31cd4:	mov    rax,QWORD PTR [rsp+0x38]
   31cd9:	movaps XMMWORD PTR [rax],xmm0
   31cdc:	add    rax,0x10
   31ce0:	cmp    rax,r9
   31ce3:	je     31d00 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xe80>
   31ce5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   31cf0:	movaps XMMWORD PTR [rax],xmm0
   31cf3:	add    rax,0x20
   31cf7:	movaps XMMWORD PTR [rax-0x10],xmm0
   31cfb:	cmp    rax,r9
   31cfe:	jne    31cf0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xe70>
   31d00:	mov    rax,QWORD PTR [rsp]
   31d04:	mov    QWORD PTR [rsp+0x68],rbp
   31d09:	xor    edx,edx
   31d0b:	lea    r8,[r11+0x200]
   31d12:	lea    r10,[rax+0x130]
   31d19:	mov    rdi,QWORD PTR [rsp]
   31d1d:	mov    rsi,r10
   31d20:	lea    rbp,[rsp+0x18b0]
   31d28:	nop    DWORD PTR [rax+rax*1+0x0]
   31d30:	pxor   xmm4,xmm4
   31d34:	mov    rcx,rbx
   31d37:	mov    rax,rbp
   31d3a:	movapd xmm8,xmm4
   31d3f:	movapd xmm9,xmm4
   31d44:	nop
   31d45:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   31d50:	movapd xmm0,XMMWORD PTR [rax]
   31d54:	movapd xmm1,XMMWORD PTR [rax+0x10]
   31d59:	add    rax,0x20
   31d5d:	add    rcx,0x20
   31d61:	movlpd xmm1,QWORD PTR [rax-0x18]
   31d66:	movapd xmm12,XMMWORD PTR [rcx-0x20]
   31d6c:	movhpd xmm0,QWORD PTR [rax-0x10]
   31d71:	movapd xmm10,XMMWORD PTR [rcx-0x10]
   31d77:	movlpd xmm10,QWORD PTR [rcx-0x18]
   31d7d:	movapd xmm6,xmm0
   31d81:	movapd xmm7,xmm1
   31d85:	movhpd xmm12,QWORD PTR [rcx-0x10]
   31d8b:	mulpd  xmm7,xmm1
   31d8f:	movapd xmm11,xmm10
   31d94:	xorpd  xmm1,xmm2
   31d98:	mulpd  xmm6,xmm0
   31d9c:	mulpd  xmm11,xmm1
   31da1:	mulpd  xmm1,xmm12
   31da6:	addpd  xmm6,xmm7
   31daa:	movapd xmm7,xmm0
   31dae:	mulpd  xmm0,xmm10
   31db3:	mulpd  xmm7,xmm12
   31db8:	addsd  xmm9,xmm6
   31dbd:	unpckhpd xmm6,xmm6
   31dc1:	addpd  xmm0,xmm1
   31dc5:	addsd  xmm9,xmm6
   31dca:	subpd  xmm7,xmm11
   31dcf:	movapd xmm1,xmm0
   31dd3:	unpckhpd xmm0,xmm0
   31dd7:	addsd  xmm1,xmm4
   31ddb:	addsd  xmm8,xmm7
   31de0:	unpckhpd xmm7,xmm7
   31de4:	movapd xmm4,xmm1
   31de8:	addsd  xmm8,xmm7
   31ded:	addsd  xmm4,xmm0
   31df1:	cmp    rsi,rax
   31df4:	jne    31d50 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xed0>
   31dfa:	divsd  xmm8,xmm9
   31dff:	movsd  xmm0,QWORD PTR [rdi+0x18b0]
   31e07:	mov    rcx,rbp
   31e0a:	mov    rax,rbx
   31e0d:	divsd  xmm4,xmm9
   31e12:	addsd  xmm0,xmm8
   31e17:	movapd xmm7,xmm8
   31e1c:	movsd  QWORD PTR [rdi+0x18b0],xmm0
   31e24:	movsd  xmm0,QWORD PTR [rdi+0x18b8]
   31e2c:	addsd  xmm0,xmm4
   31e30:	unpcklpd xmm7,xmm4
   31e34:	unpcklpd xmm4,xmm8
   31e39:	movsd  QWORD PTR [rdi+0x18b8],xmm0
   31e41:	nop    DWORD PTR [rax+0x0]
   31e45:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   31e50:	movapd xmm1,XMMWORD PTR [rcx]
   31e54:	movapd xmm0,XMMWORD PTR [rcx]
   31e58:	add    rax,0x10
   31e5c:	add    rcx,0x10
   31e60:	unpcklpd xmm1,xmm1
   31e64:	unpckhpd xmm0,xmm0
   31e68:	mulpd  xmm1,xmm7
   31e6c:	mulpd  xmm0,xmm4
   31e70:	movapd xmm6,xmm1
   31e74:	subpd  xmm6,xmm0
   31e78:	addpd  xmm1,xmm0
   31e7c:	movapd xmm0,XMMWORD PTR [rax-0x10]
   31e81:	movsd  xmm1,xmm6
   31e85:	subpd  xmm0,xmm1
   31e89:	movaps XMMWORD PTR [rax-0x10],xmm0
   31e8d:	cmp    r15,rax
   31e90:	jne    31e50 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xfd0>
   31e92:	add    rsi,0x100
   31e99:	add    rbp,0x100
   31ea0:	add    rdi,0x10
   31ea4:	cmp    rsi,r8
   31ea7:	jne    31d30 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xeb0>
   31ead:	lea    rax,[rdx+0x1]
   31eb1:	mov    edx,0x1
   31eb6:	cmp    rax,QWORD PTR [rsp+0x48]
   31ebb:	jb     31d19 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xe99>
   31ec1:	pxor   xmm6,xmm6
   31ec5:	mov    rbp,QWORD PTR [rsp+0x68]
   31eca:	mov    rax,rbx
   31ecd:	movapd xmm8,xmm6
   31ed2:	nop    DWORD PTR [rax]
   31ed5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   31ee0:	movsd  xmm0,QWORD PTR [rax]
   31ee4:	add    rax,0x10
   31ee8:	movapd xmm1,xmm0
   31eec:	movapd xmm4,xmm0
   31ef0:	xorpd  xmm4,XMMWORD PTR [rip+0x58a8]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   31ef8:	cmpltsd xmm1,xmm8
   31efe:	andpd  xmm4,xmm1
   31f02:	andnpd xmm1,xmm0
   31f06:	orpd   xmm1,xmm4
   31f0a:	movapd xmm0,xmm1
   31f0e:	movsd  xmm1,QWORD PTR [rax-0x8]
   31f13:	movapd xmm4,xmm1
   31f17:	movapd xmm7,xmm1
   31f1b:	xorpd  xmm7,XMMWORD PTR [rip+0x587d]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   31f23:	cmpltsd xmm4,xmm8
   31f29:	andpd  xmm7,xmm4
   31f2d:	andnpd xmm4,xmm1
   31f31:	orpd   xmm4,xmm7
   31f35:	movapd xmm1,xmm4
   31f39:	movapd xmm4,xmm0
   31f3d:	maxsd  xmm4,xmm1
   31f41:	maxsd  xmm4,xmm6
   31f45:	movapd xmm6,xmm4
   31f49:	cmp    r15,rax
   31f4c:	jne    31ee0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1060>
   31f4e:	pxor   xmm0,xmm0
   31f52:	ucomisd xmm4,xmm0
   31f56:	jp     31f62 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x10e2>
   31f58:	movapd xmm1,xmm0
   31f5c:	je     31ffd <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x117d>
   31f62:	movapd xmm7,xmm6
   31f66:	mov    rax,rbx
   31f69:	pxor   xmm4,xmm4
   31f6d:	unpcklpd xmm7,xmm7
   31f71:	nop    DWORD PTR [rax+0x0]
   31f75:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   31f80:	movapd xmm0,XMMWORD PTR [rax]
   31f84:	movapd xmm1,XMMWORD PTR [rax+0x10]
   31f89:	add    rax,0x20
   31f8d:	movlpd xmm1,QWORD PTR [rax-0x18]
   31f92:	movhpd xmm0,QWORD PTR [rax-0x10]
   31f97:	divpd  xmm0,xmm7
   31f9b:	divpd  xmm1,xmm7
   31f9f:	mulpd  xmm0,xmm0
   31fa3:	mulpd  xmm1,xmm1
   31fa7:	addpd  xmm0,xmm1
   31fab:	addsd  xmm4,xmm0
   31faf:	unpckhpd xmm0,xmm0
   31fb3:	addsd  xmm4,xmm0
   31fb7:	cmp    r15,rax
   31fba:	jne    31f80 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1100>
   31fbc:	pxor   xmm0,xmm0
   31fc0:	ucomisd xmm4,xmm0
   31fc4:	jp     31fcc <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x114c>
   31fc6:	movapd xmm1,xmm0
   31fca:	je     31ff9 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1179>
   31fcc:	movsd  xmm3,QWORD PTR [rip+0x588c]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   31fd4:	movapd xmm7,xmm3
   31fd8:	movapd xmm1,xmm3
   31fdc:	maxsd  xmm7,xmm4
   31fe0:	movapd xmm0,xmm7
   31fe4:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   31fe9:	movsd  xmm3,QWORD PTR [rip+0x5997]        # 37988 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x208>
   31ff1:	mulsd  xmm7,xmm0
   31ff5:	movapd xmm1,xmm7
   31ff9:	mulsd  xmm1,xmm6
   31ffd:	mulsd  xmm5,QWORD PTR [rip+0x580b]        # 37810 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x90>
   32005:	lea    rax,[r13+0x10]
   32009:	mov    QWORD PTR [r13+0x18],0x0
   32011:	movsd  QWORD PTR [r13+0x10],xmm1
   32017:	comisd xmm5,xmm1
   3201b:	jae    32028 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x11a8>
   3201d:	cmp    r14,QWORD PTR [rsp+0x8]
   32022:	jb     32668 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x17e8>
   32028:	test   rbp,rbp
   3202b:	je     32158 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x12d8>
   32031:	mov    rdx,QWORD PTR [rsp]
   32035:	lea    rsi,[r13-0x18b0]
   3203c:	movsd  xmm4,QWORD PTR [rsp+0x3138]
   32045:	movsd  xmm0,QWORD PTR [rsp+0x3130]
   3204e:	mov    rcx,rdx
   32051:	nop    DWORD PTR [rax+0x0]
   32055:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   32060:	movsd  xmm7,QWORD PTR [rdx+0x1ad8]
   32068:	add    rdx,0x10
   3206c:	add    rcx,0x8
   32070:	movsd  xmm9,QWORD PTR [rdx+0x18b0]
   32079:	movsd  xmm8,QWORD PTR [rdx+0x18b8]
   32082:	movsd  xmm6,QWORD PTR [rdx+0x1ac0]
   3208a:	movsd  xmm11,QWORD PTR [rcx+0x1bc8]
   32093:	movapd xmm10,xmm9
   32098:	mulsd  xmm10,xmm6
   3209d:	movapd xmm12,xmm8
   320a2:	mulsd  xmm12,xmm7
   320a7:	subsd  xmm10,xmm12
   320ac:	movapd xmm12,xmm0
   320b1:	mulsd  xmm12,xmm11
   320b6:	addsd  xmm10,xmm12
   320bb:	movapd xmm12,xmm9
   320c0:	mulsd  xmm12,xmm7
   320c5:	mulsd  xmm9,xmm11
   320ca:	movsd  QWORD PTR [rdx+0x18a0],xmm10
   320d3:	movapd xmm10,xmm8
   320d8:	mulsd  xmm10,xmm6
   320dd:	xorpd  xmm6,XMMWORD PTR [rip+0x56bb]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   320e5:	mulsd  xmm8,xmm11
   320ea:	addsd  xmm10,xmm12
   320ef:	movapd xmm12,xmm11
   320f4:	mulsd  xmm12,xmm4
   320f9:	addsd  xmm10,xmm12
   320fe:	movapd xmm12,xmm0
   32103:	mulsd  xmm12,xmm7
   32108:	mulsd  xmm7,xmm4
   3210c:	movsd  QWORD PTR [rdx+0x18a8],xmm10
   32115:	movapd xmm10,xmm6
   3211a:	mulsd  xmm10,xmm4
   3211f:	movapd xmm4,xmm8
   32124:	mulsd  xmm6,xmm0
   32128:	addsd  xmm10,xmm12
   3212d:	subsd  xmm6,xmm7
   32131:	addsd  xmm4,xmm10
   32136:	addsd  xmm6,xmm9
   3213b:	movsd  QWORD PTR [rdx+0x18b8],xmm4
   32143:	movsd  QWORD PTR [rdx+0x18b0],xmm6
   3214b:	movapd xmm0,xmm6
   3214f:	cmp    rsi,rdx
   32152:	jne    32060 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x11e0>
   32158:	lea    rcx,[r13+0x220]
   3215f:	mov    rdx,r12
   32162:	mov    rsi,rax
   32165:	mov    rdi,r13
   32168:	mov    QWORD PTR [rsp+0x88],r9
   32170:	mov    QWORD PTR [rsp+0x80],r11
   32178:	mov    QWORD PTR [rsp+0x68],rax
   3217d:	movsd  QWORD PTR [rsp+0x78],xmm1
   32183:	movsd  QWORD PTR [rsp+0x70],xmm5
   32189:	call   2b9f0 <_ZN4Base6Matrix9Numerical6givensINS0_7ComplexIdEEEEvRKT_S7_RNS1_6ScalarIS5_E4RealERS5_>
   3218e:	movsd  xmm9,QWORD PTR [r13+0x10]
   32194:	mov    rcx,rbp
   32197:	mov    rdx,QWORD PTR [rsp+0x38]
   3219c:	movsd  xmm8,QWORD PTR [r13+0x18]
   321a2:	movsd  xmm6,QWORD PTR [r12]
   321a8:	imul   rcx,r14
   321ac:	movapd xmm10,xmm9
   321b1:	movsd  xmm5,QWORD PTR [rsp+0x70]
   321b7:	movsd  xmm0,QWORD PTR [r13+0x220]
   321c0:	movsd  xmm7,QWORD PTR [r13+0x228]
   321c9:	mov    r11,QWORD PTR [rsp+0x80]
   321d1:	movapd xmm4,xmm0
   321d5:	shr    rcx,1
   321d8:	movsd  xmm1,QWORD PTR [rsp+0x78]
   321de:	mov    r9,QWORD PTR [rsp+0x88]
   321e6:	mulsd  xmm10,xmm7
   321eb:	shl    rcx,0x4
   321ef:	mov    rax,QWORD PTR [rsp+0x68]
   321f4:	add    rcx,QWORD PTR [rsp+0x60]
   321f9:	mulsd  xmm4,xmm8
   321fe:	mulsd  xmm0,xmm9
   32203:	mulsd  xmm7,xmm8
   32208:	addsd  xmm4,xmm10
   3220d:	movsd  xmm10,QWORD PTR [r13+0x8]
   32213:	mulsd  xmm10,xmm6
   32218:	subsd  xmm0,xmm7
   3221c:	mulsd  xmm6,QWORD PTR [r13+0x0]
   32222:	addsd  xmm4,xmm10
   32227:	addsd  xmm0,xmm6
   3222b:	movsd  QWORD PTR [r13+0x8],xmm4
   32231:	movsd  QWORD PTR [r13+0x0],xmm0
   32237:	jmp    32244 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x13c4>
   32239:	nop    DWORD PTR [rax+0x0]
   32240:	add    rdx,0x10
   32244:	cmp    rdx,rcx
   32247:	je     3225b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x13db>
   32249:	movsd  xmm0,QWORD PTR [rdx]
   3224d:	movsd  QWORD PTR [rcx],xmm0
   32251:	movsd  xmm0,QWORD PTR [rdx+0x8]
   32256:	movsd  QWORD PTR [rcx+0x8],xmm0
   3225b:	add    rcx,0x10
   3225f:	cmp    rdx,r13
   32262:	jne    32240 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x13c0>
   32264:	movsd  xmm6,QWORD PTR [rdx+0x228]
   3226c:	movsd  xmm0,QWORD PTR [rdx+0x220]
   32274:	xorpd  xmm0,XMMWORD PTR [rip+0x5524]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   3227c:	movsd  xmm7,QWORD PTR [rdx+0x110]
   32284:	movsd  xmm8,QWORD PTR [rdx+0x118]
   3228d:	movapd xmm9,xmm6
   32292:	mulsd  xmm6,xmm7
   32296:	movapd xmm4,xmm0
   3229a:	mulsd  xmm4,xmm7
   3229e:	mulsd  xmm9,xmm8
   322a3:	mulsd  xmm0,xmm8
   322a8:	subsd  xmm4,xmm9
   322ad:	addsd  xmm0,xmm6
   322b1:	movsd  QWORD PTR [rdx+0x120],xmm4
   322b9:	movsd  QWORD PTR [rdx+0x128],xmm0
   322c1:	movsd  xmm6,QWORD PTR [r12]
   322c7:	unpcklpd xmm6,xmm6
   322cb:	mulpd  xmm6,XMMWORD PTR [rdx+0x110]
   322d3:	movaps XMMWORD PTR [rdx+0x110],xmm6
   322da:	add    QWORD PTR [rsp+0x35d8],0x1
   322e3:	comisd xmm5,xmm1
   322e7:	jae    323d6 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1556>
   322ed:	pxor   xmm1,xmm1
   322f1:	comisd xmm1,xmm4
   322f5:	ja     3262e <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x17ae>
   322fb:	comisd xmm1,xmm0
   322ff:	movapd xmm6,xmm4
   32303:	ja     326bf <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x183f>
   32309:	movapd xmm1,xmm0
   3230d:	maxsd  xmm6,xmm1
   32311:	pxor   xmm1,xmm1
   32315:	ucomisd xmm6,xmm1
   32319:	jp     32321 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x14a1>
   3231b:	movapd xmm5,xmm1
   3231f:	je     32370 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x14f0>
   32321:	divsd  xmm4,xmm6
   32325:	pxor   xmm1,xmm1
   32329:	divsd  xmm0,xmm6
   3232d:	mulsd  xmm4,xmm4
   32331:	mulsd  xmm0,xmm0
   32335:	addsd  xmm0,xmm4
   32339:	ucomisd xmm0,xmm1
   3233d:	jp     32345 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x14c5>
   3233f:	movapd xmm5,xmm1
   32343:	je     3236c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x14ec>
   32345:	movsd  xmm1,QWORD PTR [rip+0x5513]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   3234d:	mov    QWORD PTR [rsp+0x18],rax
   32352:	movapd xmm5,xmm1
   32356:	maxsd  xmm5,xmm0
   3235a:	movapd xmm0,xmm5
   3235e:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   32363:	mov    rax,QWORD PTR [rsp+0x18]
   32368:	mulsd  xmm5,xmm0
   3236c:	mulsd  xmm5,xmm6
   32370:	comisd xmm5,QWORD PTR [rsp+0xc8]
   32379:	mov    rcx,r14
   3237c:	seta   dl
   3237f:	xor    edi,edi
   32381:	jmp    31a80 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xc00>
   32386:	mov    rcx,r12
   32389:	movsd  xmm4,QWORD PTR [rsp+0x70]
   3238f:	mov    r12,QWORD PTR [rsp+0x18]
   32394:	mov    r10,QWORD PTR [rsp+0x50]
   32399:	test   al,al
   3239b:	jne    32b7d <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1cfd>
   323a1:	movzx  eax,BYTE PTR [rsp+0x105]
   323a9:	mov    BYTE PTR [rsp+0x35e1],al
   323b0:	add    QWORD PTR [rsp+0x20],0x1
   323b6:	mov    rax,QWORD PTR [rsp+0x20]
   323bb:	cmp    QWORD PTR [rsp+0x90],rax
   323c3:	je     326d0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1850>
   323c9:	movzx  edx,BYTE PTR [rsp+0x35e0]
   323d1:	jmp    31938 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xab8>
   323d6:	movzx  edi,BYTE PTR [rsp+0x18]
   323db:	xor    edx,edx
   323dd:	mov    rcx,r14
   323e0:	jmp    31a80 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0xc00>
   323e5:	mov    BYTE PTR [rsp+0x68],dil
   323ea:	mov    r13,QWORD PTR [rsp]
   323ee:	pxor   xmm6,xmm6
   323f2:	mov    r12,rcx
   323f5:	movsd  xmm4,QWORD PTR [rsp+0x158]
   323fe:	mov    QWORD PTR [rsp+0x18],r9
   32403:	mov    ebx,0xf
   32408:	mov    rbp,QWORD PTR [rsp+0x108]
   32410:	movzx  eax,BYTE PTR [rsp+0x105]
   32418:	movsd  QWORD PTR [rsp+0x70],xmm4
   3241e:	jmp    32432 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x15b2>
   32420:	sub    rbx,0x1
   32424:	sub    rbp,0x10
   32428:	cmp    rbx,0xffffffffffffffff
   3242c:	je     32386 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1506>
   32432:	cmp    rbx,r12
   32435:	movaps XMMWORD PTR [rbp-0x10],xmm6
   32439:	setb   r14b
   3243d:	and    r14b,al
   32440:	je     32420 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x15a0>
   32442:	movapd xmm5,XMMWORD PTR [rbp-0x2a0]
   3244a:	lea    rdi,[rbx+0x1]
   3244e:	cmp    rbx,0xf
   32452:	je     324bf <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x163f>
   32454:	mov    rsi,rbp
   32457:	mov    rdx,rdi
   3245a:	nop    WORD PTR [rax+rax*1+0x0]
   32460:	mov    rax,rdx
   32463:	add    rdx,0x1
   32467:	cmp    rax,r12
   3246a:	jae    324b5 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1635>
   3246c:	imul   rax,rdx
   32470:	movapd xmm1,XMMWORD PTR [rsi]
   32474:	shr    rax,1
   32477:	add    rax,rbx
   3247a:	shl    rax,0x4
   3247e:	movapd xmm0,XMMWORD PTR [r13+rax*1+0x1030]
   32488:	movapd xmm3,xmm0
   3248c:	unpcklpd xmm3,xmm0
   32490:	unpckhpd xmm0,xmm0
   32494:	mulpd  xmm3,xmm1
   32498:	shufpd xmm1,xmm1,0x1
   3249d:	mulpd  xmm0,xmm1
   324a1:	movapd xmm1,xmm3
   324a5:	subpd  xmm1,xmm0
   324a9:	addpd  xmm0,xmm3
   324ad:	movsd  xmm0,xmm1
   324b1:	subpd  xmm5,xmm0
   324b5:	add    rsi,0x10
   324b9:	cmp    rdx,0x10
   324bd:	jne    32460 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x15e0>
   324bf:	imul   rdi,rbx
   324c3:	pxor   xmm4,xmm4
   324c7:	shr    rdi,1
   324ca:	lea    r9,[rbx+rdi*1+0x103]
   324d2:	shl    r9,0x4
   324d6:	add    r9,r13
   324d9:	movsd  xmm0,QWORD PTR [r9]
   324de:	movsd  xmm1,QWORD PTR [r9+0x8]
   324e4:	comisd xmm4,xmm0
   324e8:	ja     325b9 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1739>
   324ee:	comisd xmm4,xmm1
   324f2:	movapd xmm3,xmm0
   324f6:	ja     3261d <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x179d>
   324fc:	movapd xmm4,xmm1
   32500:	maxsd  xmm3,xmm4
   32504:	movapd xmm7,xmm3
   32508:	pxor   xmm3,xmm3
   3250c:	ucomisd xmm7,xmm3
   32510:	jp     32518 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1698>
   32512:	je     325f3 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1773>
   32518:	divsd  xmm0,xmm7
   3251c:	divsd  xmm1,xmm7
   32520:	mulsd  xmm0,xmm0
   32524:	mulsd  xmm1,xmm1
   32528:	addsd  xmm0,xmm1
   3252c:	pxor   xmm1,xmm1
   32530:	ucomisd xmm0,xmm1
   32534:	jp     3253d <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x16bd>
   32536:	movapd xmm8,xmm1
   3253b:	je     32562 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x16e2>
   3253d:	movsd  xmm3,QWORD PTR [rip+0x531b]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   32545:	movapd xmm8,xmm3
   3254a:	movapd xmm1,xmm3
   3254e:	maxsd  xmm8,xmm0
   32553:	movapd xmm0,xmm8
   32558:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   3255d:	mulsd  xmm8,xmm0
   32562:	movsd  xmm3,QWORD PTR [rsp+0x98]
   3256b:	mulsd  xmm8,xmm7
   32570:	comisd xmm3,xmm8
   32575:	ja     325f3 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1773>
   32577:	movsd  xmm0,QWORD PTR [rsp+0x98]
   32580:	lea    rdi,[rsp+0x170]
   32588:	mov    rdx,r9
   3258b:	lea    rsi,[rsp+0x160]
   32593:	movaps XMMWORD PTR [rsp+0x160],xmm5
   3259b:	call   30c50 <_ZN4Base6Matrix9Numerical6ScalarINS0_7ComplexIdEEE6divideERKS4_S7_d>
   325a0:	mov    eax,r14d
   325a3:	pxor   xmm6,xmm6
   325a7:	movapd xmm0,XMMWORD PTR [rsp+0x170]
   325b0:	movaps XMMWORD PTR [rbp-0x10],xmm0
   325b4:	jmp    32420 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x15a0>
   325b9:	comisd xmm4,xmm1
   325bd:	movapd xmm3,xmm0
   325c1:	xorpd  xmm3,XMMWORD PTR [rip+0x51d7]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   325c9:	jbe    324fc <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x167c>
   325cf:	comisd xmm1,xmm0
   325d3:	movapd xmm7,xmm1
   325d7:	xorpd  xmm7,XMMWORD PTR [rip+0x51c1]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   325df:	seta   al
   325e2:	test   al,al
   325e4:	je     32518 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1698>
   325ea:	movapd xmm7,xmm3
   325ee:	jmp    32518 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1698>
   325f3:	test   rbx,rbx
   325f6:	je     32e26 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1fa6>
   325fc:	sub    rbx,0x2
   32600:	movaps XMMWORD PTR [rbp-0x20],xmm6
   32604:	cmp    rbx,0xffffffffffffffff
   32608:	je     32e26 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1fa6>
   3260e:	sub    rbp,0x20
   32612:	xor    eax,eax
   32614:	movaps XMMWORD PTR [rbp-0x10],xmm6
   32618:	jmp    32420 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x15a0>
   3261d:	movapd xmm4,xmm1
   32621:	xorpd  xmm4,XMMWORD PTR [rip+0x5177]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   32629:	jmp    32500 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1680>
   3262e:	comisd xmm1,xmm0
   32632:	movapd xmm6,xmm4
   32636:	xorpd  xmm6,XMMWORD PTR [rip+0x5162]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   3263e:	jbe    32309 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1489>
   32644:	comisd xmm0,xmm4
   32648:	movapd xmm1,xmm0
   3264c:	xorpd  xmm1,XMMWORD PTR [rip+0x514c]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   32654:	seta   dl
   32657:	test   dl,dl
   32659:	je     3265f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x17df>
   3265b:	movapd xmm1,xmm6
   3265f:	movapd xmm6,xmm1
   32663:	jmp    32321 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x14a1>
   32668:	comisd xmm1,xmm3
   3266c:	movapd xmm6,xmm1
   32670:	pxor   xmm4,xmm4
   32674:	unpcklpd xmm6,xmm6
   32678:	jb     32686 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1806>
   3267a:	movsd  xmm4,QWORD PTR [rip+0x51a6]        # 37828 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
   32682:	divsd  xmm4,xmm1
   32686:	unpcklpd xmm4,xmm4
   3268a:	lea    rcx,[r11-0x30]
   3268e:	mov    rdx,rbx
   32691:	jmp    326af <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x182f>
   32693:	mulpd  xmm0,xmm4
   32697:	add    rdx,0x10
   3269b:	movaps XMMWORD PTR [rcx+0x130],xmm0
   326a2:	add    rcx,0x10
   326a6:	cmp    rdx,r15
   326a9:	je     32028 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x11a8>
   326af:	comisd xmm1,xmm3
   326b3:	movapd xmm0,XMMWORD PTR [rdx]
   326b7:	jae    32693 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1813>
   326b9:	divpd  xmm0,xmm6
   326bd:	jmp    32697 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1817>
   326bf:	movapd xmm1,xmm0
   326c3:	xorpd  xmm1,XMMWORD PTR [rip+0x50d5]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   326cb:	jmp    3230d <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x148d>
   326d0:	mov    rbx,QWORD PTR [rsp+0x148]
   326d8:	mov    r13,QWORD PTR [rsp+0x120]
   326e0:	mov    r11,QWORD PTR [rsp+0xb8]
   326e8:	mov    rbp,QWORD PTR [rsp+0x128]
   326f0:	mov    r12,QWORD PTR [rsp+0x150]
   326f8:	pxor   xmm5,xmm5
   326fc:	mov    rax,r11
   326ff:	movapd xmm2,xmm5
   32703:	movsd  xmm1,QWORD PTR [rax]
   32707:	add    rax,0x10
   3270b:	movapd xmm3,xmm1
   3270f:	movapd xmm0,xmm1
   32713:	xorpd  xmm0,XMMWORD PTR [rip+0x5085]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   3271b:	cmpltsd xmm1,xmm2
   32720:	andpd  xmm0,xmm1
   32724:	andnpd xmm1,xmm3
   32728:	orpd   xmm1,xmm0
   3272c:	movsd  xmm0,QWORD PTR [rax-0x8]
   32731:	movapd xmm4,xmm0
   32735:	movapd xmm3,xmm0
   32739:	xorpd  xmm3,XMMWORD PTR [rip+0x505f]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   32741:	cmpltsd xmm0,xmm2
   32746:	andpd  xmm3,xmm0
   3274a:	andnpd xmm0,xmm4
   3274e:	orpd   xmm0,xmm3
   32752:	movapd xmm3,xmm1
   32756:	maxsd  xmm3,xmm0
   3275a:	maxsd  xmm3,xmm5
   3275e:	movapd xmm5,xmm3
   32762:	cmp    rax,r13
   32765:	jne    32703 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1883>
   32767:	pxor   xmm0,xmm0
   3276b:	ucomisd xmm3,xmm0
   3276f:	jp     32773 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x18f3>
   32771:	je     327ec <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x196c>
   32773:	movapd xmm3,xmm5
   32777:	pxor   xmm2,xmm2
   3277b:	unpcklpd xmm3,xmm3
   3277f:	movapd xmm0,XMMWORD PTR [r11]
   32784:	movapd xmm1,XMMWORD PTR [r11+0x10]
   3278a:	add    r11,0x20
   3278e:	movlpd xmm1,QWORD PTR [r11-0x18]
   32794:	movhpd xmm0,QWORD PTR [r11-0x10]
   3279a:	divpd  xmm0,xmm3
   3279e:	divpd  xmm1,xmm3
   327a2:	mulpd  xmm0,xmm0
   327a6:	mulpd  xmm1,xmm1
   327aa:	addpd  xmm0,xmm1
   327ae:	addsd  xmm2,xmm0
   327b2:	unpckhpd xmm0,xmm0
   327b6:	addsd  xmm2,xmm0
   327ba:	cmp    r11,r13
   327bd:	jne    3277f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x18ff>
   327bf:	pxor   xmm0,xmm0
   327c3:	ucomisd xmm2,xmm0
   327c7:	jp     327cb <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x194b>
   327c9:	je     327e8 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1968>
   327cb:	movsd  xmm1,QWORD PTR [rip+0x508d]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   327d3:	movapd xmm6,xmm1
   327d7:	maxsd  xmm6,xmm2
   327db:	movapd xmm0,xmm6
   327df:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   327e4:	mulsd  xmm0,xmm6
   327e8:	mulsd  xmm0,xmm5
   327ec:	movsd  xmm1,QWORD PTR [r12+0x28]
   327f3:	comisd xmm1,xmm0
   327f7:	jb     32e96 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2016>
   327fd:	mov    rsi,rbx
   32800:	mov    r13,r15
   32803:	mov    r9,QWORD PTR [rsp+0xe8]
   3280b:	mov    DWORD PTR [r12+0x2158],0x5
   32817:	lea    rax,[rsi+0x1]
   3281b:	mov    rbx,QWORD PTR [rsp+0xe0]
   32823:	movsd  xmm6,QWORD PTR [rsp+0xa8]
   3282c:	mov    r15,QWORD PTR [rsp+0xb0]
   32834:	mov    rbp,QWORD PTR [rsp+0xf8]
   3283c:	movzx  r14d,BYTE PTR [rsp+0x106]
   32845:	movsd  xmm7,QWORD PTR [rsp+0xf0]
   3284e:	cmp    r9,rax
   32851:	je     3286e <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x19ee>
   32853:	add    rsi,0x2
   32857:	cmp    rsi,r9
   3285a:	je     3286e <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x19ee>
   3285c:	lea    rax,[rsi+0x1]
   32860:	cmp    r9,rax
   32863:	je     3286e <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x19ee>
   32865:	add    rsi,0x2
   32869:	cmp    r9,rsi
   3286c:	jne    3285c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x19dc>
   3286e:	movsd  xmm5,QWORD PTR [r15]
   32873:	movsd  xmm8,QWORD PTR [r15-0x8]
   32879:	mov    rax,r13
   3287c:	mov    rdx,r13
   3287f:	pxor   xmm0,xmm0
   32883:	movaps XMMWORD PTR [rdx],xmm0
   32886:	add    rdx,0x20
   3288a:	movaps XMMWORD PTR [rdx-0x10],xmm0
   3288e:	cmp    QWORD PTR [rsp+0x10],rdx
   32893:	jne    32883 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1a03>
   32895:	mov    r8,QWORD PTR [rsp+0x28]
   3289a:	unpcklpd xmm8,xmm8
   3289f:	unpcklpd xmm5,xmm5
   328a3:	lea    rdi,[rsp+0x180]
   328ab:	mov    rcx,r13
   328ae:	movapd xmm3,XMMWORD PTR [rcx]
   328b2:	movapd xmm2,XMMWORD PTR [rcx+0x10]
   328b7:	lea    rdx,[rsp+0x180]
   328bf:	mov    rsi,rbx
   328c2:	movlpd xmm2,QWORD PTR [rcx+0x8]
   328c7:	movhpd xmm3,QWORD PTR [rcx+0x10]
   328cc:	nop    DWORD PTR [rax+rax*1+0x0]
   328d4:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   328df:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   328ea:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   328f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   32900:	movsd  xmm0,QWORD PTR [rdx]
   32904:	movupd xmm1,XMMWORD PTR [rsi]
   32908:	add    rdx,0x10
   3290c:	sub    rsi,0xffffffffffffff80
   32910:	unpcklpd xmm0,xmm0
   32914:	mulpd  xmm0,xmm1
   32918:	addpd  xmm3,xmm0
   3291c:	movsd  xmm0,QWORD PTR [rdx-0x8]
   32921:	unpcklpd xmm0,xmm0
   32925:	mulpd  xmm0,xmm1
   32929:	addpd  xmm2,xmm0
   3292d:	cmp    rdx,r8
   32930:	jne    32900 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1a80>
   32932:	movapd xmm1,XMMWORD PTR [rdi]
   32936:	add    rcx,0x20
   3293a:	add    rdi,0x20
   3293e:	add    rbx,0x10
   32942:	movapd xmm0,XMMWORD PTR [rdi-0x10]
   32947:	movlpd xmm0,QWORD PTR [rdi-0x18]
   3294c:	movapd xmm9,xmm5
   32951:	movhpd xmm1,QWORD PTR [rdi-0x10]
   32956:	mulpd  xmm9,xmm1
   3295b:	movapd xmm4,xmm0
   3295f:	mulpd  xmm1,xmm8
   32964:	mulpd  xmm0,xmm5
   32968:	mulpd  xmm4,xmm8
   3296d:	subpd  xmm1,xmm0
   32971:	addpd  xmm4,xmm9
   32976:	subpd  xmm3,xmm1
   3297a:	subpd  xmm2,xmm4
   3297e:	movapd xmm0,xmm3
   32982:	unpcklpd xmm0,xmm2
   32986:	unpckhpd xmm3,xmm2
   3298a:	movaps XMMWORD PTR [rcx-0x20],xmm0
   3298e:	movaps XMMWORD PTR [rcx-0x10],xmm3
   32992:	cmp    QWORD PTR [rsp+0x10],rcx
   32997:	jne    328ae <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1a2e>
   3299d:	pxor   xmm8,xmm8
   329a2:	mov    rdx,r13
   329a5:	movapd xmm2,xmm8
   329aa:	movsd  xmm1,QWORD PTR [rdx]
   329ae:	add    rdx,0x10
   329b2:	movapd xmm3,xmm1
   329b6:	movapd xmm0,xmm1
   329ba:	xorpd  xmm0,XMMWORD PTR [rip+0x4dde]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   329c2:	cmpltsd xmm1,xmm2
   329c7:	andpd  xmm0,xmm1
   329cb:	andnpd xmm1,xmm3
   329cf:	orpd   xmm1,xmm0
   329d3:	movsd  xmm0,QWORD PTR [rdx-0x8]
   329d8:	movapd xmm4,xmm0
   329dc:	movapd xmm3,xmm0
   329e0:	xorpd  xmm3,XMMWORD PTR [rip+0x4db8]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   329e8:	cmpltsd xmm0,xmm2
   329ed:	andpd  xmm3,xmm0
   329f1:	andnpd xmm0,xmm4
   329f5:	orpd   xmm0,xmm3
   329f9:	movapd xmm3,xmm1
   329fd:	maxsd  xmm3,xmm0
   32a01:	maxsd  xmm3,xmm8
   32a06:	movapd xmm8,xmm3
   32a0b:	cmp    QWORD PTR [rsp+0x10],rdx
   32a10:	jne    329aa <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1b2a>
   32a12:	pxor   xmm0,xmm0
   32a16:	ucomisd xmm3,xmm0
   32a1a:	jp     32a22 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1ba2>
   32a1c:	movapd xmm5,xmm0
   32a20:	je     32a9f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1c1f>
   32a22:	movapd xmm3,xmm8
   32a27:	pxor   xmm2,xmm2
   32a2b:	unpcklpd xmm3,xmm3
   32a2f:	movapd xmm1,XMMWORD PTR [rax]
   32a33:	movapd xmm0,XMMWORD PTR [rax+0x10]
   32a38:	add    rax,0x20
   32a3c:	movlpd xmm0,QWORD PTR [rax-0x18]
   32a41:	movhpd xmm1,QWORD PTR [rax-0x10]
   32a46:	divpd  xmm1,xmm3
   32a4a:	divpd  xmm0,xmm3
   32a4e:	mulpd  xmm1,xmm1
   32a52:	mulpd  xmm0,xmm0
   32a56:	addpd  xmm0,xmm1
   32a5a:	addsd  xmm2,xmm0
   32a5e:	unpckhpd xmm0,xmm0
   32a62:	addsd  xmm2,xmm0
   32a66:	cmp    QWORD PTR [rsp+0x10],rax
   32a6b:	jne    32a2f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1baf>
   32a6d:	pxor   xmm0,xmm0
   32a71:	ucomisd xmm2,xmm0
   32a75:	jp     32a7d <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1bfd>
   32a77:	movapd xmm5,xmm0
   32a7b:	je     32a9a <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1c1a>
   32a7d:	movsd  xmm1,QWORD PTR [rip+0x4ddb]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   32a85:	movapd xmm5,xmm1
   32a89:	maxsd  xmm5,xmm2
   32a8d:	movapd xmm0,xmm5
   32a91:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   32a96:	mulsd  xmm5,xmm0
   32a9a:	mulsd  xmm5,xmm8
   32a9f:	divsd  xmm5,xmm6
   32aa3:	comisd xmm5,QWORD PTR [r12+0x2170]
   32aad:	jbe    32ab9 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1c39>
   32aaf:	movsd  QWORD PTR [r12+0x2170],xmm5
   32ab9:	and    r14b,BYTE PTR [r12+0x215d]
   32ac1:	mov    rdx,QWORD PTR [rsp+0xd8]
   32ac9:	lea    rax,[rsp+0x180]
   32ad1:	mov    BYTE PTR [r12+0x215d],r14b
   32ad9:	mov    rcx,QWORD PTR [rsp+0x28]
   32ade:	movapd xmm0,XMMWORD PTR [rax]
   32ae2:	add    rax,0x10
   32ae6:	add    rdx,0x10
   32aea:	movups XMMWORD PTR [rdx-0x10],xmm0
   32aee:	cmp    rax,rcx
   32af1:	jne    32ade <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1c5e>
   32af3:	add    QWORD PTR [rsp+0xc0],0x1
   32afc:	mov    rax,QWORD PTR [rsp+0xc0]
   32b04:	add    r15,0x10
   32b08:	add    QWORD PTR [rsp+0xd8],0x100
   32b14:	add    QWORD PTR [rsp+0x110],0x10
   32b1d:	cmp    rax,0x10
   32b21:	jne    31034 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1b4>
   32b27:	mov    rax,QWORD PTR [rsp+0xa0]
   32b2f:	mov    BYTE PTR [rax+0x2161],0x1
   32b36:	cmp    BYTE PTR [rax+0x215f],0x0
   32b3d:	je     331a8 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2328>
   32b43:	mov    BYTE PTR [rax+0x215d],0x0
   32b4a:	mov    DWORD PTR [rax+0x2158],0x4
   32b54:	mov    rax,QWORD PTR [rsp+0x35e8]
   32b5c:	sub    rax,QWORD PTR fs:0x28
   32b65:	jne    33224 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x23a4>
   32b6b:	add    rsp,0x35f8
   32b72:	pop    rbx
   32b73:	pop    rbp
   32b74:	pop    r12
   32b76:	pop    r13
   32b78:	pop    r14
   32b7a:	pop    r15
   32b7c:	ret
   32b7d:	mov    rax,QWORD PTR [rsp+0xb8]
   32b85:	mov    rsi,QWORD PTR [rsp+0x120]
   32b8d:	mov    rbx,r15
   32b90:	mov    rdx,r15
   32b93:	movapd xmm0,XMMWORD PTR [rax]
   32b97:	add    rax,0x10
   32b9b:	add    rdx,0x10
   32b9f:	movaps XMMWORD PTR [rdx-0x10],xmm0
   32ba3:	cmp    rax,rsi
   32ba6:	jne    32b93 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1d13>
   32ba8:	mov    r11,QWORD PTR [rsp]
   32bac:	mov    rdi,r15
   32baf:	xor    r8d,r8d
   32bb2:	pxor   xmm8,xmm8
   32bb7:	mov    r9,r11
   32bba:	pxor   xmm9,xmm9
   32bbf:	lea    rsi,[r11+0x1c50]
   32bc6:	mov    rdx,r9
   32bc9:	xor    eax,eax
   32bcb:	movapd xmm7,xmm9
   32bd0:	cmp    rax,rcx
   32bd3:	jae    32c14 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1d94>
   32bd5:	movsd  xmm0,QWORD PTR [rdx+0x30]
   32bda:	movsd  xmm5,QWORD PTR [rdx+0x38]
   32bdf:	movsd  xmm1,QWORD PTR [rsi]
   32be3:	movsd  xmm6,QWORD PTR [rsi+0x8]
   32be8:	movapd xmm3,xmm0
   32bec:	movapd xmm10,xmm5
   32bf1:	mulsd  xmm3,xmm1
   32bf5:	mulsd  xmm10,xmm6
   32bfa:	mulsd  xmm0,xmm6
   32bfe:	mulsd  xmm1,xmm5
   32c02:	subsd  xmm3,xmm10
   32c07:	addsd  xmm0,xmm1
   32c0b:	addsd  xmm7,xmm3
   32c0f:	addsd  xmm9,xmm0
   32c14:	add    rax,0x1
   32c18:	add    rdx,0x100
   32c1f:	add    rsi,0x10
   32c23:	cmp    rax,0x10
   32c27:	jne    32bd0 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1d50>
   32c29:	movapd xmm0,xmm7
   32c2d:	add    r8,0x1
   32c31:	add    rdi,0x10
   32c35:	add    r9,0x10
   32c39:	mulsd  xmm7,xmm8
   32c3e:	movapd xmm1,xmm9
   32c43:	mulsd  xmm1,xmm8
   32c48:	addsd  xmm7,xmm9
   32c4d:	addsd  xmm7,QWORD PTR [rdi-0x8]
   32c52:	subsd  xmm0,xmm1
   32c56:	addsd  xmm0,QWORD PTR [rdi-0x10]
   32c5b:	movsd  QWORD PTR [rdi-0x8],xmm7
   32c60:	movsd  QWORD PTR [rdi-0x10],xmm0
   32c65:	cmp    r8,0x10
   32c69:	jne    32bba <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1d3a>
   32c6f:	mov    r14,QWORD PTR [rsp+0x30]
   32c74:	mov    rdi,QWORD PTR [rsp+0x10]
   32c79:	mov    rdx,r15
   32c7c:	mov    esi,0x10
   32c81:	mov    QWORD PTR [rsp+0x18],r10
   32c86:	mov    rcx,r14
   32c89:	movsd  QWORD PTR [rsp+0x50],xmm4
   32c8f:	call   99c0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0>
   32c94:	movsd  xmm4,QWORD PTR [rsp+0x50]
   32c9a:	mov    r10,QWORD PTR [rsp+0x18]
   32c9f:	mov    rdx,r14
   32ca2:	mov    rcx,QWORD PTR [rsp+0x128]
   32caa:	lea    rax,[rsp+0x180]
   32cb2:	movapd xmm0,XMMWORD PTR [rax]
   32cb6:	subpd  xmm0,XMMWORD PTR [rdx]
   32cba:	add    rax,0x10
   32cbe:	add    rdx,0x10
   32cc2:	movaps XMMWORD PTR [rdx-0x10],xmm0
   32cc6:	cmp    rax,rcx
   32cc9:	jne    32cb2 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1e32>
   32ccb:	pxor   xmm6,xmm6
   32ccf:	mov    rax,QWORD PTR [rsp+0x30]
   32cd4:	movapd xmm3,xmm6
   32cd8:	movsd  xmm1,QWORD PTR [rax]
   32cdc:	add    rax,0x10
   32ce0:	movapd xmm5,xmm1
   32ce4:	movapd xmm0,xmm1
   32ce8:	xorpd  xmm0,XMMWORD PTR [rip+0x4ab0]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   32cf0:	cmpltsd xmm1,xmm3
   32cf5:	andpd  xmm0,xmm1
   32cf9:	andnpd xmm1,xmm5
   32cfd:	orpd   xmm1,xmm0
   32d01:	movsd  xmm0,QWORD PTR [rax-0x8]
   32d06:	movapd xmm7,xmm0
   32d0a:	movapd xmm5,xmm0
   32d0e:	xorpd  xmm5,XMMWORD PTR [rip+0x4a8a]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   32d16:	cmpltsd xmm0,xmm3
   32d1b:	andpd  xmm5,xmm0
   32d1f:	andnpd xmm0,xmm7
   32d23:	orpd   xmm0,xmm5
   32d27:	movapd xmm5,xmm1
   32d2b:	maxsd  xmm5,xmm0
   32d2f:	maxsd  xmm5,xmm6
   32d33:	movapd xmm6,xmm5
   32d37:	cmp    rax,r10
   32d3a:	jne    32cd8 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1e58>
   32d3c:	pxor   xmm0,xmm0
   32d40:	ucomisd xmm5,xmm0
   32d44:	jp     32d4c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1ecc>
   32d46:	je     32dd2 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1f52>
   32d4c:	movapd xmm5,xmm6
   32d50:	mov    rax,QWORD PTR [rsp+0x30]
   32d55:	pxor   xmm3,xmm3
   32d59:	unpcklpd xmm5,xmm5
   32d5d:	movapd xmm1,XMMWORD PTR [rax]
   32d61:	movapd xmm0,XMMWORD PTR [rax+0x10]
   32d66:	add    rax,0x20
   32d6a:	movlpd xmm0,QWORD PTR [rax-0x18]
   32d6f:	movhpd xmm1,QWORD PTR [rax-0x10]
   32d74:	divpd  xmm1,xmm5
   32d78:	divpd  xmm0,xmm5
   32d7c:	mulpd  xmm1,xmm1
   32d80:	mulpd  xmm0,xmm0
   32d84:	addpd  xmm0,xmm1
   32d88:	addsd  xmm3,xmm0
   32d8c:	unpckhpd xmm0,xmm0
   32d90:	addsd  xmm3,xmm0
   32d94:	cmp    rax,r10
   32d97:	jne    32d5d <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1edd>
   32d99:	pxor   xmm0,xmm0
   32d9d:	ucomisd xmm3,xmm0
   32da1:	jp     32da5 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1f25>
   32da3:	je     32dce <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1f4e>
   32da5:	movsd  xmm1,QWORD PTR [rip+0x4ab3]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   32dad:	movsd  QWORD PTR [rsp+0x18],xmm4
   32db3:	movapd xmm5,xmm1
   32db7:	maxsd  xmm5,xmm3
   32dbb:	movapd xmm0,xmm5
   32dbf:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   32dc4:	movsd  xmm4,QWORD PTR [rsp+0x18]
   32dca:	mulsd  xmm0,xmm5
   32dce:	mulsd  xmm0,xmm6
   32dd2:	comisd xmm4,xmm0
   32dd6:	mov    rax,QWORD PTR [rsp+0xb8]
   32dde:	lea    rdx,[r15+0x100]
   32de5:	jb     32e35 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1fb5>
   32de7:	movapd xmm1,XMMWORD PTR [rbx]
   32deb:	add    rbx,0x10
   32def:	add    rax,0x10
   32df3:	movaps XMMWORD PTR [rax-0x10],xmm1
   32df7:	cmp    rbx,rdx
   32dfa:	jne    32de7 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1f67>
   32dfc:	movsd  QWORD PTR [rsp+0x35d0],xmm0
   32e05:	movsd  xmm3,QWORD PTR [rsp+0xc8]
   32e0e:	comisd xmm3,xmm0
   32e12:	setae  BYTE PTR [rsp+0x35e0]
   32e1a:	setb   al
   32e1d:	and    al,BYTE PTR [rsp+0x68]
   32e21:	jmp    323a9 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1529>
   32e26:	mov    r12,QWORD PTR [rsp+0x18]
   32e2b:	mov    r10,QWORD PTR [rsp+0x50]
   32e30:	jmp    323a1 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1521>
   32e35:	mov    rbx,QWORD PTR [rsp+0x30]
   32e3a:	mov    rdi,QWORD PTR [rsp+0x10]
   32e3f:	mov    esi,0x10
   32e44:	mov    r8d,0x10
   32e4a:	mov    rdx,QWORD PTR [rsp+0xb8]
   32e52:	mov    QWORD PTR [rsp+0x18],r10
   32e57:	mov    rcx,rbx
   32e5a:	call   99c0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixINS0_7ComplexIdEELm16ELm16EEES6_Lm16EEES6_Lm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSE_m.isra.0>
   32e5f:	mov    r10,QWORD PTR [rsp+0x18]
   32e64:	lea    rdx,[rsp+0x180]
   32e6c:	mov    rax,rbx
   32e6f:	movapd xmm0,XMMWORD PTR [rdx]
   32e73:	subpd  xmm0,XMMWORD PTR [rax]
   32e77:	add    rax,0x10
   32e7b:	add    rdx,0x10
   32e7f:	movaps XMMWORD PTR [rax-0x10],xmm0
   32e83:	cmp    rax,r10
   32e86:	jne    32e6f <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1fef>
   32e88:	movsd  xmm0,QWORD PTR [rsp+0x35d0]
   32e91:	jmp    32e05 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1f85>
   32e96:	mov    rdx,QWORD PTR [rsp+0x130]
   32e9e:	unpcklpd xmm0,xmm0
   32ea2:	lea    rax,[rsp+0x180]
   32eaa:	movapd xmm1,XMMWORD PTR [rdx]
   32eae:	add    rdx,0x10
   32eb2:	add    rax,0x10
   32eb6:	divpd  xmm1,xmm0
   32eba:	movaps XMMWORD PTR [rax-0x10],xmm1
   32ebe:	cmp    rdx,r13
   32ec1:	jne    32eaa <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x202a>
   32ec3:	mov    rax,r15
   32ec6:	mov    rdx,r15
   32ec9:	pxor   xmm0,xmm0
   32ecd:	movaps XMMWORD PTR [rdx],xmm0
   32ed0:	add    rdx,0x20
   32ed4:	movaps XMMWORD PTR [rdx-0x10],xmm0
   32ed8:	cmp    rdx,QWORD PTR [rsp+0x10]
   32edd:	jne    32ecd <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x204d>
   32edf:	mov    rdi,QWORD PTR [rsp+0xb0]
   32ee7:	mov    rcx,r15
   32eea:	lea    rsi,[rsp+0x180]
   32ef2:	movsd  xmm8,QWORD PTR [rdi-0x8]
   32ef8:	movsd  xmm5,QWORD PTR [rdi]
   32efc:	mov    rdi,QWORD PTR [rsp+0x118]
   32f04:	movapd xmm6,xmm8
   32f09:	movapd xmm4,xmm5
   32f0d:	unpcklpd xmm6,xmm6
   32f11:	unpcklpd xmm4,xmm4
   32f15:	movapd xmm3,XMMWORD PTR [rcx]
   32f19:	movapd xmm2,XMMWORD PTR [rcx+0x10]
   32f1e:	lea    rdx,[rsp+0x180]
   32f26:	mov    r8,rdi
   32f29:	movlpd xmm2,QWORD PTR [rcx+0x8]
   32f2e:	movhpd xmm3,QWORD PTR [rcx+0x10]
   32f33:	xchg   ax,ax
   32f35:	data16 cs nop WORD PTR [rax+rax*1+0x0]
   32f40:	movsd  xmm0,QWORD PTR [rdx]
   32f44:	movupd xmm1,XMMWORD PTR [r8]
   32f49:	add    rdx,0x10
   32f4d:	sub    r8,0xffffffffffffff80
   32f51:	unpcklpd xmm0,xmm0
   32f55:	mulpd  xmm0,xmm1
   32f59:	addpd  xmm3,xmm0
   32f5d:	movsd  xmm0,QWORD PTR [rdx-0x8]
   32f62:	unpcklpd xmm0,xmm0
   32f66:	mulpd  xmm0,xmm1
   32f6a:	addpd  xmm2,xmm0
   32f6e:	cmp    rdx,rbp
   32f71:	jne    32f40 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x20c0>
   32f73:	movapd xmm1,XMMWORD PTR [rsi]
   32f77:	add    rsi,0x20
   32f7b:	add    rcx,0x20
   32f7f:	add    rdi,0x10
   32f83:	movapd xmm0,XMMWORD PTR [rsi-0x10]
   32f88:	movlpd xmm0,QWORD PTR [rsi-0x18]
   32f8d:	movapd xmm9,xmm6
   32f92:	movhpd xmm1,QWORD PTR [rsi-0x10]
   32f97:	mulpd  xmm9,xmm0
   32f9c:	movapd xmm7,xmm1
   32fa0:	mulpd  xmm0,xmm4
   32fa4:	mulpd  xmm1,xmm6
   32fa8:	mulpd  xmm7,xmm4
   32fac:	subpd  xmm1,xmm0
   32fb0:	addpd  xmm7,xmm9
   32fb5:	subpd  xmm3,xmm1
   32fb9:	subpd  xmm2,xmm7
   32fbd:	movapd xmm0,xmm3
   32fc1:	unpcklpd xmm0,xmm2
   32fc5:	unpckhpd xmm3,xmm2
   32fc9:	movaps XMMWORD PTR [rcx-0x20],xmm0
   32fcd:	movaps XMMWORD PTR [rcx-0x10],xmm3
   32fd1:	cmp    rsi,rbp
   32fd4:	jne    32f15 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2095>
   32fda:	pxor   xmm7,xmm7
   32fde:	mov    rdx,r15
   32fe1:	movapd xmm3,xmm7
   32fe5:	movsd  xmm1,QWORD PTR [rdx]
   32fe9:	add    rdx,0x10
   32fed:	movapd xmm2,xmm1
   32ff1:	movapd xmm0,xmm1
   32ff5:	xorpd  xmm0,XMMWORD PTR [rip+0x47a3]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   32ffd:	cmpltsd xmm1,xmm3
   33002:	andpd  xmm0,xmm1
   33006:	andnpd xmm1,xmm2
   3300a:	orpd   xmm1,xmm0
   3300e:	movsd  xmm0,QWORD PTR [rdx-0x8]
   33013:	movapd xmm4,xmm0
   33017:	movapd xmm2,xmm0
   3301b:	xorpd  xmm2,XMMWORD PTR [rip+0x477d]        # 377a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
   33023:	cmpltsd xmm0,xmm3
   33028:	andpd  xmm2,xmm0
   3302c:	andnpd xmm0,xmm4
   33030:	orpd   xmm0,xmm2
   33034:	movapd xmm2,xmm1
   33038:	maxsd  xmm2,xmm0
   3303c:	maxsd  xmm2,xmm7
   33040:	movapd xmm7,xmm2
   33044:	cmp    QWORD PTR [rsp+0x10],rdx
   33049:	jne    32fe5 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2165>
   3304b:	pxor   xmm0,xmm0
   3304f:	ucomisd xmm2,xmm0
   33053:	jp     3305b <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x21db>
   33055:	movapd xmm6,xmm0
   33059:	je     330d6 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2256>
   3305b:	movapd xmm3,xmm7
   3305f:	pxor   xmm2,xmm2
   33063:	unpcklpd xmm3,xmm3
   33067:	movapd xmm1,XMMWORD PTR [rax]
   3306b:	movapd xmm0,XMMWORD PTR [rax+0x10]
   33070:	add    rax,0x20
   33074:	movlpd xmm0,QWORD PTR [rax-0x18]
   33079:	movhpd xmm1,QWORD PTR [rax-0x10]
   3307e:	divpd  xmm1,xmm3
   33082:	divpd  xmm0,xmm3
   33086:	mulpd  xmm1,xmm1
   3308a:	mulpd  xmm0,xmm0
   3308e:	addpd  xmm0,xmm1
   33092:	addsd  xmm2,xmm0
   33096:	unpckhpd xmm0,xmm0
   3309a:	addsd  xmm2,xmm0
   3309e:	cmp    QWORD PTR [rsp+0x10],rax
   330a3:	jne    33067 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x21e7>
   330a5:	pxor   xmm0,xmm0
   330a9:	ucomisd xmm2,xmm0
   330ad:	jp     330b5 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2235>
   330af:	movapd xmm6,xmm0
   330b3:	je     330d2 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2252>
   330b5:	movsd  xmm1,QWORD PTR [rip+0x47a3]        # 37860 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xe0>
   330bd:	movapd xmm6,xmm1
   330c1:	maxsd  xmm6,xmm2
   330c5:	movapd xmm0,xmm6
   330c9:	call   4a80 <_ZN4Base4Math19rsqrt_newton_methodIdLm1EEET_RKS2_S4_.isra.0>
   330ce:	mulsd  xmm6,xmm0
   330d2:	mulsd  xmm6,xmm7
   330d6:	divsd  xmm6,QWORD PTR [rsp+0xa8]
   330df:	movsd  xmm0,QWORD PTR [r12+0x50]
   330e6:	comisd xmm0,xmm6
   330ea:	setae  BYTE PTR [rsp+0x106]
   330f2:	setb   BYTE PTR [rsp+0x105]
   330fa:	cmp    BYTE PTR [rsp+0x35e1],0x0
   33102:	je     33117 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2297>
   33104:	cmp    BYTE PTR [rsp+0x35e0],0x0
   3310c:	jne    33117 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2297>
   3310e:	mov    BYTE PTR [r12+0x215f],0x1
   33117:	add    rbx,0x1
   3311b:	cmp    rbx,QWORD PTR [rsp+0xe8]
   33123:	jne    31544 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x6c4>
   33129:	mov    r13,r15
   3312c:	mov    rbx,QWORD PTR [rsp+0xe0]
   33134:	movsd  xmm6,QWORD PTR [rsp+0xa8]
   3313d:	mov    r15,QWORD PTR [rsp+0xb0]
   33145:	mov    rbp,QWORD PTR [rsp+0xf8]
   3314d:	movzx  r14d,BYTE PTR [rsp+0x106]
   33156:	movsd  xmm7,QWORD PTR [rsp+0xf0]
   3315f:	jmp    32879 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x19f9>
   33164:	mov    rdx,QWORD PTR [rsp+0xd8]
   3316c:	lea    rax,[rsp+0x180]
   33174:	movupd xmm0,XMMWORD PTR [rdx]
   33178:	add    rax,0x10
   3317c:	add    rdx,0x10
   33180:	movaps XMMWORD PTR [rax-0x10],xmm0
   33184:	cmp    rax,QWORD PTR [rsp+0x28]
   33189:	jne    33174 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x22f4>
   3318b:	test   cl,cl
   3318d:	jne    31180 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x300>
   33193:	jmp    31168 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x2e8>
   33198:	nop    DWORD PTR [rax+rax*1+0x0]
   331a0:	xor    r14d,r14d
   331a3:	jmp    3286e <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x19ee>
   331a8:	cmp    BYTE PTR [rsp+0x107],0x0
   331b0:	je     32b54 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1cd4>
   331b6:	mov    rdi,QWORD PTR [rsp+0xa0]
   331be:	cmp    DWORD PTR [rdi+0x2158],0x5
   331c5:	je     32b54 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1cd4>
   331cb:	movzx  r14d,r14b
   331cf:	lea    eax,[r14+0x1]
   331d3:	mov    DWORD PTR [rdi+0x2158],eax
   331d9:	jmp    32b54 <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x1cd4>
   331de:	mov    rsi,rbx
   331e1:	mov    r13,r15
   331e4:	movsd  xmm6,QWORD PTR [rsp+0xa8]
   331ed:	movzx  r14d,BYTE PTR [rsp+0x106]
   331f6:	mov    rbx,QWORD PTR [rsp+0xe0]
   331fe:	mov    r9,QWORD PTR [rsp+0xe8]
   33206:	mov    r15,QWORD PTR [rsp+0xb0]
   3320e:	mov    rbp,QWORD PTR [rsp+0xf8]
   33216:	movsd  xmm7,QWORD PTR [rsp+0xf0]
   3321f:	jmp    3285c <_ZN4Base6Matrix14EigenOperation6SolverINS0_7ComplexIdEEdLm16ELm16EE19solve_eigen_vectorsERKNS0_6MatrixIdLm16ELm16EEE+0x19dc>
   33224:	call   1170 <__stack_chk_fail@plt>

Disassembly of section .fini:
