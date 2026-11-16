
/tmp/matrix_peak_stack_audit/current_b599/current_b599_template:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000015e60 <_ZN4Base6Matrix14GmresOperation13DenseMultiply3RowINS0_7ComplexIdEELm16ELm10ELm8ELb1EE7computeERKNS0_6MatrixIS5_Lm16ELm16EEERKNS0_6VectorIS5_Lm16EEERSC_m.isra.0>:
   15e60:	movsd  xmm3,QWORD PTR [rdi+0xa88]
   15e68:	movsd  xmm0,QWORD PTR [rdi+0xa80]
   15e70:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   15e78:	movsd  xmm2,QWORD PTR [rsi+0xa0]
   15e80:	movapd xmm1,xmm0
   15e84:	movapd xmm5,xmm3
   15e88:	mulsd  xmm5,xmm4
   15e8c:	mulsd  xmm1,xmm2
   15e90:	mulsd  xmm0,xmm4
   15e94:	mulsd  xmm2,xmm3
   15e98:	subsd  xmm1,xmm5
   15e9c:	addsd  xmm1,QWORD PTR [rdx+0x80]
   15ea4:	addsd  xmm0,xmm2
   15ea8:	addsd  xmm0,QWORD PTR [rdx+0x88]
   15eb0:	movsd  QWORD PTR [rdx+0x80],xmm1
   15eb8:	movsd  xmm2,QWORD PTR [rsi+0xa0]
   15ec0:	movsd  QWORD PTR [rdx+0x88],xmm0
   15ec8:	movsd  xmm0,QWORD PTR [rdi+0xa90]
   15ed0:	movsd  xmm3,QWORD PTR [rdi+0xa98]
   15ed8:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   15ee0:	movapd xmm1,xmm0
   15ee4:	movapd xmm5,xmm3
   15ee8:	mulsd  xmm5,xmm4
   15eec:	mulsd  xmm1,xmm2
   15ef0:	mulsd  xmm0,xmm4
   15ef4:	mulsd  xmm2,xmm3
   15ef8:	subsd  xmm1,xmm5
   15efc:	addsd  xmm1,QWORD PTR [rdx+0x90]
   15f04:	addsd  xmm0,xmm2
   15f08:	addsd  xmm0,QWORD PTR [rdx+0x98]
   15f10:	movsd  QWORD PTR [rdx+0x90],xmm1
   15f18:	movsd  xmm2,QWORD PTR [rsi+0xa0]
   15f20:	movsd  QWORD PTR [rdx+0x98],xmm0
   15f28:	movsd  xmm0,QWORD PTR [rdi+0xaa0]
   15f30:	movsd  xmm3,QWORD PTR [rdi+0xaa8]
   15f38:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   15f40:	movapd xmm1,xmm0
   15f44:	movapd xmm5,xmm3
   15f48:	mulsd  xmm5,xmm4
   15f4c:	mulsd  xmm1,xmm2
   15f50:	mulsd  xmm0,xmm4
   15f54:	mulsd  xmm2,xmm3
   15f58:	subsd  xmm1,xmm5
   15f5c:	addsd  xmm1,QWORD PTR [rdx+0xa0]
   15f64:	addsd  xmm0,xmm2
   15f68:	addsd  xmm0,QWORD PTR [rdx+0xa8]
   15f70:	movsd  QWORD PTR [rdx+0xa0],xmm1
   15f78:	movsd  xmm2,QWORD PTR [rsi+0xa0]
   15f80:	movsd  QWORD PTR [rdx+0xa8],xmm0
   15f88:	movsd  xmm0,QWORD PTR [rdi+0xab0]
   15f90:	movsd  xmm3,QWORD PTR [rdi+0xab8]
   15f98:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   15fa0:	movapd xmm1,xmm0
   15fa4:	movapd xmm5,xmm3
   15fa8:	mulsd  xmm5,xmm4
   15fac:	mulsd  xmm1,xmm2
   15fb0:	mulsd  xmm0,xmm4
   15fb4:	mulsd  xmm2,xmm3
   15fb8:	subsd  xmm1,xmm5
   15fbc:	addsd  xmm1,QWORD PTR [rdx+0xb0]
   15fc4:	addsd  xmm0,xmm2
   15fc8:	addsd  xmm0,QWORD PTR [rdx+0xb8]
   15fd0:	movsd  QWORD PTR [rdx+0xb0],xmm1
   15fd8:	movsd  QWORD PTR [rdx+0xb8],xmm0
   15fe0:	movsd  xmm0,QWORD PTR [rdi+0xac0]
   15fe8:	movsd  xmm3,QWORD PTR [rdi+0xac8]
   15ff0:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   15ff8:	movsd  xmm2,QWORD PTR [rsi+0xa0]
   16000:	movapd xmm1,xmm0
   16004:	movapd xmm5,xmm3
   16008:	movupd xmm6,XMMWORD PTR [rdx+0xf0]
   16010:	mulsd  xmm5,xmm4
   16014:	mulsd  xmm1,xmm2
   16018:	mulsd  xmm0,xmm4
   1601c:	mulsd  xmm2,xmm3
   16020:	subsd  xmm1,xmm5
   16024:	addsd  xmm1,QWORD PTR [rdx+0xc0]
   1602c:	addsd  xmm0,xmm2
   16030:	addsd  xmm0,QWORD PTR [rdx+0xc8]
   16038:	movsd  QWORD PTR [rdx+0xc0],xmm1
   16040:	movsd  xmm2,QWORD PTR [rsi+0xa0]
   16048:	movsd  QWORD PTR [rdx+0xc8],xmm0
   16050:	movsd  xmm0,QWORD PTR [rdi+0xad0]
   16058:	movsd  xmm3,QWORD PTR [rdi+0xad8]
   16060:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   16068:	movapd xmm1,xmm0
   1606c:	movapd xmm5,xmm3
   16070:	mulsd  xmm5,xmm4
   16074:	mulsd  xmm1,xmm2
   16078:	mulsd  xmm0,xmm4
   1607c:	mulsd  xmm2,xmm3
   16080:	subsd  xmm1,xmm5
   16084:	addsd  xmm1,QWORD PTR [rdx+0xd0]
   1608c:	addsd  xmm0,xmm2
   16090:	addsd  xmm0,QWORD PTR [rdx+0xd8]
   16098:	movsd  QWORD PTR [rdx+0xd0],xmm1
   160a0:	movsd  xmm2,QWORD PTR [rsi+0xa0]
   160a8:	movsd  QWORD PTR [rdx+0xd8],xmm0
   160b0:	movsd  xmm0,QWORD PTR [rdi+0xae0]
   160b8:	movsd  xmm3,QWORD PTR [rdi+0xae8]
   160c0:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   160c8:	movapd xmm1,xmm0
   160cc:	movapd xmm5,xmm3
   160d0:	mulsd  xmm5,xmm4
   160d4:	mulsd  xmm1,xmm2
   160d8:	mulsd  xmm0,xmm4
   160dc:	mulsd  xmm2,xmm3
   160e0:	subsd  xmm1,xmm5
   160e4:	addsd  xmm1,QWORD PTR [rdx+0xe0]
   160ec:	addsd  xmm0,xmm2
   160f0:	addsd  xmm0,QWORD PTR [rdx+0xe8]
   160f8:	movsd  QWORD PTR [rdx+0xe0],xmm1
   16100:	movsd  xmm1,QWORD PTR [rdi+0xaf0]
   16108:	movsd  xmm2,QWORD PTR [rsi+0xa0]
   16110:	movsd  QWORD PTR [rdx+0xe8],xmm0
   16118:	movsd  xmm3,QWORD PTR [rdi+0xaf8]
   16120:	movapd xmm0,xmm1
   16124:	movsd  xmm4,QWORD PTR [rsi+0xa8]
   1612c:	mulsd  xmm0,xmm2
   16130:	mulsd  xmm1,xmm4
   16134:	movapd xmm5,xmm3
   16138:	mulsd  xmm5,xmm4
   1613c:	mulsd  xmm2,xmm3
   16140:	subsd  xmm0,xmm5
   16144:	addsd  xmm1,xmm2
   16148:	unpcklpd xmm0,xmm1
   1614c:	addpd  xmm0,xmm6
   16150:	movups XMMWORD PTR [rdx+0xf0],xmm0
   16157:	ret

Disassembly of section .fini:
