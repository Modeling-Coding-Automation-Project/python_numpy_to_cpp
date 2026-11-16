
/tmp/matrix_instruction_counts/instructions_for:     file format elf64-x86-64


Disassembly of section .init:

0000000000001000 <_init>:
    1000:	endbr64
    1004:	sub    rsp,0x8
    1008:	mov    rax,QWORD PTR [rip+0xcfd9]        # dfe8 <__gmon_start__>
    100f:	test   rax,rax
    1012:	je     1016 <_init+0x16>
    1014:	call   rax
    1016:	add    rsp,0x8
    101a:	ret

Disassembly of section .plt:

0000000000001020 <.plt>:
    1020:	push   QWORD PTR [rip+0xcf72]        # df98 <_GLOBAL_OFFSET_TABLE_+0x8>
    1026:	jmp    QWORD PTR [rip+0xcf74]        # dfa0 <_GLOBAL_OFFSET_TABLE_+0x10>
    102c:	nop    DWORD PTR [rax+0x0]
    1030:	endbr64
    1034:	push   0x0
    1039:	jmp    1020 <_init+0x20>
    103e:	xchg   ax,ax
    1040:	endbr64
    1044:	push   0x1
    1049:	jmp    1020 <_init+0x20>
    104e:	xchg   ax,ax
    1050:	endbr64
    1054:	push   0x2
    1059:	jmp    1020 <_init+0x20>
    105e:	xchg   ax,ax
    1060:	endbr64
    1064:	push   0x3
    1069:	jmp    1020 <_init+0x20>
    106e:	xchg   ax,ax
    1070:	endbr64
    1074:	push   0x4
    1079:	jmp    1020 <_init+0x20>
    107e:	xchg   ax,ax
    1080:	endbr64
    1084:	push   0x5
    1089:	jmp    1020 <_init+0x20>
    108e:	xchg   ax,ax

Disassembly of section .plt.got:

0000000000001090 <__cxa_finalize@plt>:
    1090:	endbr64
    1094:	jmp    QWORD PTR [rip+0xcf5e]        # dff8 <__cxa_finalize@GLIBC_2.2.5>
    109a:	nop    WORD PTR [rax+rax*1+0x0]

Disassembly of section .plt.sec:

00000000000010a0 <raise@plt>:
    10a0:	endbr64
    10a4:	jmp    QWORD PTR [rip+0xcefe]        # dfa8 <raise@GLIBC_2.2.5>
    10aa:	nop    WORD PTR [rax+rax*1+0x0]

00000000000010b0 <__stack_chk_fail@plt>:
    10b0:	endbr64
    10b4:	jmp    QWORD PTR [rip+0xcef6]        # dfb0 <__stack_chk_fail@GLIBC_2.4>
    10ba:	nop    WORD PTR [rax+rax*1+0x0]

00000000000010c0 <strcmp@plt>:
    10c0:	endbr64
    10c4:	jmp    QWORD PTR [rip+0xceee]        # dfb8 <strcmp@GLIBC_2.2.5>
    10ca:	nop    WORD PTR [rax+rax*1+0x0]

00000000000010d0 <ptrace@plt>:
    10d0:	endbr64
    10d4:	jmp    QWORD PTR [rip+0xcee6]        # dfc0 <ptrace@GLIBC_2.2.5>
    10da:	nop    WORD PTR [rax+rax*1+0x0]

00000000000010e0 <__printf_chk@plt>:
    10e0:	endbr64
    10e4:	jmp    QWORD PTR [rip+0xcede]        # dfc8 <__printf_chk@GLIBC_2.3.4>
    10ea:	nop    WORD PTR [rax+rax*1+0x0]

00000000000010f0 <perror@plt>:
    10f0:	endbr64
    10f4:	jmp    QWORD PTR [rip+0xced6]        # dfd0 <perror@GLIBC_2.2.5>
    10fa:	nop    WORD PTR [rax+rax*1+0x0]

Disassembly of section .text:

0000000000001100 <main.cold>:
    1100:	lea    rdi,[rip+0xaf72]        # c079 <_IO_stdin_used+0x79>
    1107:	call   10f0 <perror@plt>
    110c:	jmp    131b <main+0x1fb>
    1111:	cs nop WORD PTR [rax+rax*1+0x0]
    111b:	nop    DWORD PTR [rax+rax*1+0x0]

0000000000001120 <main>:
    1120:	endbr64
    1124:	cmp    edi,0x3
    1127:	je     112f <main+0xf>
    1129:	mov    eax,0x1
    112e:	ret
    112f:	push   rbx
    1130:	mov    rdi,QWORD PTR [rsi+0x8]
    1134:	mov    rbx,rsi
    1137:	lea    rsi,[rip+0xaf35]        # c073 <_IO_stdin_used+0x73>
    113e:	call   10c0 <strcmp@plt>
    1143:	test   eax,eax
    1145:	je     1322 <main+0x202>
    114b:	mov    rbx,QWORD PTR [rbx+0x10]
    114f:	lea    rsi,[rip+0xaf32]        # c088 <_IO_stdin_used+0x88>
    1156:	mov    rdi,rbx
    1159:	call   10c0 <strcmp@plt>
    115e:	test   eax,eax
    1160:	je     1347 <main+0x227>
    1166:	lea    rsi,[rip+0xaf31]        # c09e <_IO_stdin_used+0x9e>
    116d:	mov    rdi,rbx
    1170:	call   10c0 <strcmp@plt>
    1175:	test   eax,eax
    1177:	je     1350 <main+0x230>
    117d:	lea    rsi,[rip+0xaf31]        # c0b5 <_IO_stdin_used+0xb5>
    1184:	mov    rdi,rbx
    1187:	call   10c0 <strcmp@plt>
    118c:	test   eax,eax
    118e:	je     1359 <main+0x239>
    1194:	lea    rsi,[rip+0xaf31]        # c0cc <_IO_stdin_used+0xcc>
    119b:	mov    rdi,rbx
    119e:	call   10c0 <strcmp@plt>
    11a3:	test   eax,eax
    11a5:	je     1362 <main+0x242>
    11ab:	lea    rsi,[rip+0xaf32]        # c0e4 <_IO_stdin_used+0xe4>
    11b2:	mov    rdi,rbx
    11b5:	call   10c0 <strcmp@plt>
    11ba:	test   eax,eax
    11bc:	je     136b <main+0x24b>
    11c2:	lea    rsi,[rip+0xaf34]        # c0fd <_IO_stdin_used+0xfd>
    11c9:	mov    rdi,rbx
    11cc:	call   10c0 <strcmp@plt>
    11d1:	test   eax,eax
    11d3:	je     137d <main+0x25d>
    11d9:	lea    rsi,[rip+0xaf37]        # c117 <_IO_stdin_used+0x117>
    11e0:	mov    rdi,rbx
    11e3:	call   10c0 <strcmp@plt>
    11e8:	test   eax,eax
    11ea:	je     1386 <main+0x266>
    11f0:	lea    rsi,[rip+0xaf3a]        # c131 <_IO_stdin_used+0x131>
    11f7:	mov    rdi,rbx
    11fa:	call   10c0 <strcmp@plt>
    11ff:	test   eax,eax
    1201:	je     138f <main+0x26f>
    1207:	lea    rsi,[rip+0xaf3e]        # c14c <_IO_stdin_used+0x14c>
    120e:	mov    rdi,rbx
    1211:	call   10c0 <strcmp@plt>
    1216:	test   eax,eax
    1218:	je     1398 <main+0x278>
    121e:	lea    rsi,[rip+0xaf39]        # c15e <_IO_stdin_used+0x15e>
    1225:	mov    rdi,rbx
    1228:	call   10c0 <strcmp@plt>
    122d:	test   eax,eax
    122f:	je     13a1 <main+0x281>
    1235:	lea    rsi,[rip+0xaf35]        # c171 <_IO_stdin_used+0x171>
    123c:	mov    rdi,rbx
    123f:	call   10c0 <strcmp@plt>
    1244:	test   eax,eax
    1246:	je     1374 <main+0x254>
    124c:	lea    rsi,[rip+0xaf31]        # c184 <_IO_stdin_used+0x184>
    1253:	mov    rdi,rbx
    1256:	call   10c0 <strcmp@plt>
    125b:	test   eax,eax
    125d:	je     13aa <main+0x28a>
    1263:	lea    rsi,[rip+0xaf2e]        # c198 <_IO_stdin_used+0x198>
    126a:	mov    rdi,rbx
    126d:	call   10c0 <strcmp@plt>
    1272:	test   eax,eax
    1274:	je     13b3 <main+0x293>
    127a:	lea    rsi,[rip+0xaf2c]        # c1ad <_IO_stdin_used+0x1ad>
    1281:	mov    rdi,rbx
    1284:	call   10c0 <strcmp@plt>
    1289:	test   eax,eax
    128b:	je     13bc <main+0x29c>
    1291:	lea    rsi,[rip+0xaf2b]        # c1c3 <_IO_stdin_used+0x1c3>
    1298:	mov    rdi,rbx
    129b:	call   10c0 <strcmp@plt>
    12a0:	test   eax,eax
    12a2:	je     13c5 <main+0x2a5>
    12a8:	lea    rsi,[rip+0xaf2a]        # c1d9 <_IO_stdin_used+0x1d9>
    12af:	mov    rdi,rbx
    12b2:	call   10c0 <strcmp@plt>
    12b7:	test   eax,eax
    12b9:	je     13ce <main+0x2ae>
    12bf:	lea    rsi,[rip+0xaf2a]        # c1f0 <_IO_stdin_used+0x1f0>
    12c6:	mov    rdi,rbx
    12c9:	call   10c0 <strcmp@plt>
    12ce:	test   eax,eax
    12d0:	je     13da <main+0x2ba>
    12d6:	lea    rsi,[rip+0xaf26]        # c203 <_IO_stdin_used+0x203>
    12dd:	mov    rdi,rbx
    12e0:	call   10c0 <strcmp@plt>
    12e5:	test   eax,eax
    12e7:	je     13e6 <main+0x2c6>
    12ed:	lea    rsi,[rip+0xaf23]        # c217 <_IO_stdin_used+0x217>
    12f4:	mov    rdi,rbx
    12f7:	call   10c0 <strcmp@plt>
    12fc:	test   eax,eax
    12fe:	je     13f2 <main+0x2d2>
    1304:	lea    rsi,[rip+0xaf20]        # c22b <_IO_stdin_used+0x22b>
    130b:	mov    rdi,rbx
    130e:	call   10c0 <strcmp@plt>
    1313:	test   eax,eax
    1315:	je     13fe <main+0x2de>
    131b:	mov    eax,0x1
    1320:	pop    rbx
    1321:	ret
    1322:	xor    ecx,ecx
    1324:	xor    edx,edx
    1326:	xor    esi,esi
    1328:	xor    edi,edi
    132a:	call   10d0 <ptrace@plt>
    132f:	test   rax,rax
    1332:	jne    1100 <main.cold>
    1338:	mov    edi,0x13
    133d:	call   10a0 <raise@plt>
    1342:	jmp    114b <main+0x2b>
    1347:	call   29c0 <_Z3runI12MultiplyCaseIfLm8EEEvv>
    134c:	xor    eax,eax
    134e:	pop    rbx
    134f:	ret
    1350:	call   2d80 <_Z3runI12MultiplyCaseIfLm16EEEvv>
    1355:	xor    eax,eax
    1357:	jmp    134e <main+0x22e>
    1359:	call   3180 <_Z3runI12MultiplyCaseIdLm8EEEvv>
    135e:	xor    eax,eax
    1360:	jmp    134e <main+0x22e>
    1362:	call   3580 <_Z3runI12MultiplyCaseIdLm16EEEvv>
    1367:	xor    eax,eax
    1369:	jmp    134e <main+0x22e>
    136b:	call   39a0 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EEEvv>
    1370:	xor    eax,eax
    1372:	jmp    134e <main+0x22e>
    1374:	call   91e0 <_Z3runI8NormCaseIdLm8EEEvv>
    1379:	xor    eax,eax
    137b:	jmp    134e <main+0x22e>
    137d:	call   3e90 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EEEvv>
    1382:	xor    eax,eax
    1384:	jmp    134e <main+0x22e>
    1386:	call   4360 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv>
    138b:	xor    eax,eax
    138d:	jmp    134e <main+0x22e>
    138f:	call   4880 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EEEvv>
    1394:	xor    eax,eax
    1396:	jmp    134e <main+0x22e>
    1398:	call   6640 <_Z3runI8NormCaseIfLm8EEEvv>
    139d:	xor    eax,eax
    139f:	jmp    134e <main+0x22e>
    13a1:	call   7ac0 <_Z3runI8NormCaseIfLm16EEEvv>
    13a6:	xor    eax,eax
    13a8:	jmp    134e <main+0x22e>
    13aa:	call   a7a0 <_Z3runI8NormCaseIdLm16EEEvv>
    13af:	xor    eax,eax
    13b1:	jmp    134e <main+0x22e>
    13b3:	call   5040 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIfEELm8EEEvv>
    13b8:	xor    eax,eax
    13ba:	jmp    134e <main+0x22e>
    13bc:	call   55b0 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIfEELm16EEEvv>
    13c1:	xor    eax,eax
    13c3:	jmp    134e <main+0x22e>
    13c5:	call   5b50 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIdEELm8EEEvv>
    13ca:	xor    eax,eax
    13cc:	jmp    134e <main+0x22e>
    13ce:	call   60c0 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIdEELm16EEEvv>
    13d3:	xor    eax,eax
    13d5:	jmp    134e <main+0x22e>
    13da:	call   7330 <_Z3runI9GmresCaseIfLm8EEEvv>
    13df:	xor    eax,eax
    13e1:	jmp    134e <main+0x22e>
    13e6:	call   9fa0 <_Z3runI9GmresCaseIdLm8EEEvv>
    13eb:	xor    eax,eax
    13ed:	jmp    134e <main+0x22e>
    13f2:	call   8970 <_Z3runI9GmresCaseIfLm16EEEvv>
    13f7:	xor    eax,eax
    13f9:	jmp    134e <main+0x22e>
    13fe:	call   b750 <_Z3runI9GmresCaseIdLm16EEEvv>
    1403:	xor    eax,eax
    1405:	jmp    134e <main+0x22e>
    140a:	nop    WORD PTR [rax+rax*1+0x0]

0000000000001410 <_start>:
    1410:	endbr64
    1414:	xor    ebp,ebp
    1416:	mov    r9,rdx
    1419:	pop    rsi
    141a:	mov    rdx,rsp
    141d:	and    rsp,0xfffffffffffffff0
    1421:	push   rax
    1422:	push   rsp
    1423:	xor    r8d,r8d
    1426:	xor    ecx,ecx
    1428:	lea    rdi,[rip+0xfffffffffffffcf1]        # 1120 <main>
    142f:	call   QWORD PTR [rip+0xcba3]        # dfd8 <__libc_start_main@GLIBC_2.34>
    1435:	hlt
    1436:	cs nop WORD PTR [rax+rax*1+0x0]

0000000000001440 <deregister_tm_clones>:
    1440:	lea    rdi,[rip+0xcbc9]        # e010 <__TMC_END__>
    1447:	lea    rax,[rip+0xcbc2]        # e010 <__TMC_END__>
    144e:	cmp    rax,rdi
    1451:	je     1468 <deregister_tm_clones+0x28>
    1453:	mov    rax,QWORD PTR [rip+0xcb86]        # dfe0 <_ITM_deregisterTMCloneTable>
    145a:	test   rax,rax
    145d:	je     1468 <deregister_tm_clones+0x28>
    145f:	jmp    rax
    1461:	nop    DWORD PTR [rax+0x0]
    1468:	ret
    1469:	nop    DWORD PTR [rax+0x0]

0000000000001470 <register_tm_clones>:
    1470:	lea    rdi,[rip+0xcb99]        # e010 <__TMC_END__>
    1477:	lea    rsi,[rip+0xcb92]        # e010 <__TMC_END__>
    147e:	sub    rsi,rdi
    1481:	mov    rax,rsi
    1484:	shr    rsi,0x3f
    1488:	sar    rax,0x3
    148c:	add    rsi,rax
    148f:	sar    rsi,1
    1492:	je     14a8 <register_tm_clones+0x38>
    1494:	mov    rax,QWORD PTR [rip+0xcb55]        # dff0 <_ITM_registerTMCloneTable>
    149b:	test   rax,rax
    149e:	je     14a8 <register_tm_clones+0x38>
    14a0:	jmp    rax
    14a2:	nop    WORD PTR [rax+rax*1+0x0]
    14a8:	ret
    14a9:	nop    DWORD PTR [rax+0x0]

00000000000014b0 <__do_global_dtors_aux>:
    14b0:	endbr64
    14b4:	cmp    BYTE PTR [rip+0xcb55],0x0        # e010 <__TMC_END__>
    14bb:	jne    14e8 <__do_global_dtors_aux+0x38>
    14bd:	push   rbp
    14be:	cmp    QWORD PTR [rip+0xcb32],0x0        # dff8 <__cxa_finalize@GLIBC_2.2.5>
    14c6:	mov    rbp,rsp
    14c9:	je     14d7 <__do_global_dtors_aux+0x27>
    14cb:	mov    rdi,QWORD PTR [rip+0xcb36]        # e008 <__dso_handle>
    14d2:	call   1090 <__cxa_finalize@plt>
    14d7:	call   1440 <deregister_tm_clones>
    14dc:	mov    BYTE PTR [rip+0xcb2d],0x1        # e010 <__TMC_END__>
    14e3:	pop    rbp
    14e4:	ret
    14e5:	nop    DWORD PTR [rax]
    14e8:	ret
    14e9:	nop    DWORD PTR [rax+0x0]

00000000000014f0 <frame_dummy>:
    14f0:	endbr64
    14f4:	jmp    1470 <register_tm_clones>
    14f9:	nop    DWORD PTR [rax+0x0]

0000000000001500 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0>:
    1500:	pxor   xmm4,xmm4
    1504:	movaps xmm2,xmm0
    1507:	movaps xmm3,xmm0
    150a:	cmpltss xmm0,xmm4
    150f:	comiss xmm4,xmm1
    1512:	xorps  xmm3,XMMWORD PTR [rip+0xad37]        # c250 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    1519:	andps  xmm3,xmm0
    151c:	andnps xmm0,xmm2
    151f:	orps   xmm0,xmm3
    1522:	ja     16e8 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x1e8>
    1528:	ucomiss xmm1,xmm4
    152b:	jp     1533 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x33>
    152d:	je     16d0 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x1d0>
    1533:	movaps xmm3,xmm1
    1536:	movaps xmm4,xmm3
    1539:	movaps xmm3,xmm1
    153c:	maxss  xmm4,xmm0
    1540:	divss  xmm2,xmm4
    1544:	divss  xmm3,xmm4
    1548:	movaps xmm1,xmm2
    154b:	mulss  xmm1,xmm2
    154f:	movaps xmm0,xmm3
    1552:	mulss  xmm0,xmm3
    1556:	addss  xmm1,xmm0
    155a:	pxor   xmm0,xmm0
    155e:	comiss xmm0,xmm2
    1561:	ja     1700 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x200>
    1567:	ucomiss xmm1,xmm0
    156a:	movaps xmm4,xmm2
    156d:	jp     1579 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x79>
    156f:	pxor   xmm0,xmm0
    1573:	je     16af <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x1af>
    1579:	movss  xmm0,DWORD PTR [rip+0xaa87]        # c008 <_IO_stdin_used+0x8>
    1581:	movaps xmm5,xmm0
    1584:	maxss  xmm5,xmm1
    1588:	movd   eax,xmm5
    158c:	mov    r8d,eax
    158f:	shr    r8d,0x17
    1593:	movzx  ecx,r8b
    1597:	cmp    ecx,0xff
    159d:	je     17c0 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x2c0>
    15a3:	test   ecx,ecx
    15a5:	je     1738 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x238>
    15ab:	and    eax,0x807fffff
    15b0:	mov    edx,0x7e
    15b5:	and    r8d,0x1
    15b9:	or     eax,0x3f000000
    15be:	sub    edx,ecx
    15c0:	movd   xmm1,eax
    15c4:	lea    rax,[rip+0xac75]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    15cb:	sar    edx,1
    15cd:	movss  xmm6,DWORD PTR [rax+r8*4]
    15d3:	mulss  xmm1,DWORD PTR [rip+0xaa31]        # c00c <_IO_stdin_used+0xc>
    15db:	movss  xmm0,DWORD PTR [rip+0xaa2d]        # c010 <_IO_stdin_used+0x10>
    15e3:	subss  xmm0,xmm1
    15e7:	mulss  xmm0,xmm6
    15eb:	movd   ecx,xmm0
    15ef:	mov    eax,ecx
    15f1:	shr    eax,0x17
    15f4:	movzx  eax,al
    15f7:	cmp    eax,0xff
    15fc:	je     1650 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x150>
    15fe:	mov    r8d,ecx
    1601:	mov    r10d,eax
    1604:	and    r8d,0x7fffff
    160b:	or     r10d,r8d
    160e:	je     1650 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x150>
    1610:	mov    r9d,ecx
    1613:	and    r9d,0x80000000
    161a:	test   eax,eax
    161c:	je     17d8 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x2d8>
    1622:	add    eax,edx
    1624:	je     1880 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x380>
    162a:	cmp    eax,0xfe
    162f:	jg     1880 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x380>
    1635:	shl    eax,0x17
    1638:	and    ecx,0x807fffff
    163e:	or     eax,ecx
    1640:	movd   xmm0,eax
    1644:	nop
    1645:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    1650:	movaps xmm1,xmm0
    1653:	movss  xmm7,DWORD PTR [rip+0xa9b9]        # c014 <_IO_stdin_used+0x14>
    165b:	movss  xmm6,DWORD PTR [rip+0xa9b5]        # c018 <_IO_stdin_used+0x18>
    1663:	mulss  xmm1,xmm5
    1667:	mulss  xmm1,xmm0
    166b:	mulss  xmm7,xmm1
    166f:	subss  xmm6,xmm7
    1673:	mulss  xmm6,xmm1
    1677:	movss  xmm1,DWORD PTR [rip+0xac15]        # c294 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x54>
    167f:	subss  xmm1,xmm6
    1683:	movss  xmm6,DWORD PTR [rip+0xa991]        # c01c <_IO_stdin_used+0x1c>
    168b:	mulss  xmm6,xmm5
    168f:	mulss  xmm1,xmm0
    1693:	movss  xmm0,DWORD PTR [rip+0xac39]        # c2d4 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x94>
    169b:	mulss  xmm6,xmm1
    169f:	mulss  xmm6,xmm1
    16a3:	subss  xmm0,xmm6
    16a7:	mulss  xmm0,xmm1
    16ab:	mulss  xmm0,xmm5
    16af:	pxor   xmm1,xmm1
    16b3:	ucomiss xmm4,xmm1
    16b6:	jp     1719 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x219>
    16b8:	jne    1719 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x219>
    16ba:	divss  xmm3,xmm0
    16be:	movaps xmm5,xmm1
    16c1:	movss  DWORD PTR [rdi],xmm5
    16c5:	movss  DWORD PTR [rsi],xmm3
    16c9:	ret
    16ca:	nop    WORD PTR [rax+rax*1+0x0]
    16d0:	movss  xmm5,DWORD PTR [rip+0xa92c]        # c004 <_IO_stdin_used+0x4>
    16d8:	pxor   xmm3,xmm3
    16dc:	movss  DWORD PTR [rdi],xmm5
    16e0:	movss  DWORD PTR [rsi],xmm3
    16e4:	ret
    16e5:	nop    DWORD PTR [rax]
    16e8:	movaps xmm3,xmm1
    16eb:	xorps  xmm3,XMMWORD PTR [rip+0xab5e]        # c250 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    16f2:	jmp    1536 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x36>
    16f7:	nop    WORD PTR [rax+rax*1+0x0]
    1700:	ucomiss xmm1,xmm0
    1703:	movaps xmm4,xmm2
    1706:	xorps  xmm4,XMMWORD PTR [rip+0xab43]        # c250 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    170d:	jp     1579 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x79>
    1713:	jne    1579 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x79>
    1719:	divss  xmm2,xmm4
    171d:	movaps xmm5,xmm4
    1720:	divss  xmm3,xmm0
    1724:	divss  xmm5,xmm0
    1728:	mulss  xmm3,xmm2
    172c:	movss  DWORD PTR [rdi],xmm5
    1730:	movss  DWORD PTR [rsi],xmm3
    1734:	ret
    1735:	nop    DWORD PTR [rax]
    1738:	mov    r8d,eax
    173b:	and    r8d,0x7fffff
    1742:	je     17c0 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x2c0>
    1744:	mov    r9d,eax
    1747:	and    r9d,0x80000000
    174e:	test   eax,0x7f0000
    1753:	mov    eax,r8d
    1756:	je     1860 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x360>
    175c:	shl    eax,0x8
    175f:	mov    edx,0x8
    1764:	cmp    eax,0xfffffff
    1769:	ja     18d0 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x3d0>
    176f:	add    edx,0x4
    1772:	shl    eax,0x4
    1775:	cmp    eax,0x3fffffff
    177a:	ja     1782 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x282>
    177c:	add    edx,0x2
    177f:	shl    eax,0x2
    1782:	test   eax,eax
    1784:	js     1789 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x289>
    1786:	add    edx,0x1
    1789:	lea    ecx,[rdx-0x9]
    178c:	lea    rax,[rip+0xaaad]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    1793:	sar    edx,1
    1795:	shl    r8d,cl
    1798:	not    ecx
    179a:	add    edx,0x3a
    179d:	and    r8d,0x7fffff
    17a4:	and    ecx,0x1
    17a7:	or     r8d,r9d
    17aa:	movss  xmm6,DWORD PTR [rax+rcx*4]
    17af:	or     r8d,0x3f000000
    17b6:	movd   xmm1,r8d
    17bb:	jmp    15d3 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0xd3>
    17c0:	movss  xmm6,DWORD PTR [rip+0xa83c]        # c004 <_IO_stdin_used+0x4>
    17c8:	movaps xmm1,xmm5
    17cb:	xor    edx,edx
    17cd:	jmp    15d3 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0xd3>
    17d2:	nop    WORD PTR [rax+rax*1+0x0]
    17d8:	and    ecx,0x7f0000
    17de:	mov    eax,r8d
    17e1:	je     1840 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x340>
    17e3:	shl    eax,0x8
    17e6:	mov    ecx,0x8
    17eb:	cmp    eax,0xfffffff
    17f0:	ja     18c0 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x3c0>
    17f6:	add    ecx,0x4
    17f9:	shl    eax,0x4
    17fc:	cmp    eax,0x3fffffff
    1801:	ja     1809 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x309>
    1803:	add    ecx,0x2
    1806:	shl    eax,0x2
    1809:	test   eax,eax
    180b:	js     18b0 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x3b0>
    1811:	sub    ecx,0x8
    1814:	mov    eax,edx
    1816:	shl    r8d,cl
    1819:	sub    eax,ecx
    181b:	and    r8d,0x7fffff
    1822:	add    eax,0x1
    1825:	test   eax,eax
    1827:	jle    1900 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x400>
    182d:	or     r8d,r9d
    1830:	shl    eax,0x17
    1833:	or     r8d,eax
    1836:	movd   xmm0,r8d
    183b:	jmp    1650 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x150>
    1840:	shl    eax,0x10
    1843:	mov    ecx,0x10
    1848:	cmp    eax,0xffffff
    184d:	jbe    18ed <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x3ed>
    1853:	cmp    eax,0xfffffff
    1858:	ja     17fc <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x2fc>
    185a:	jmp    17f6 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x2f6>
    185c:	nop    DWORD PTR [rax+0x0]
    1860:	shl    eax,0x10
    1863:	mov    edx,0x10
    1868:	cmp    eax,0xffffff
    186d:	jbe    18e0 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x3e0>
    186f:	cmp    eax,0xfffffff
    1874:	ja     1775 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x275>
    187a:	jmp    176f <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x26f>
    187f:	nop
    1880:	shr    r8d,1
    1883:	or     r8d,r9d
    1886:	or     r9d,0x7f800000
    188d:	or     r8d,0x400000
    1894:	cmp    eax,0xfe
    1899:	cmovg  r8d,r9d
    189d:	movd   xmm0,r8d
    18a2:	jmp    1650 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x150>
    18a7:	nop    WORD PTR [rax+rax*1+0x0]
    18b0:	sub    ecx,0x9
    18b3:	jmp    1814 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x314>
    18b8:	nop    DWORD PTR [rax+rax*1+0x0]
    18c0:	cmp    eax,0x3fffffff
    18c5:	jbe    1803 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x303>
    18cb:	jmp    1811 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x311>
    18d0:	cmp    eax,0x3fffffff
    18d5:	jbe    177c <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x27c>
    18db:	jmp    1786 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x286>
    18e0:	mov    eax,r8d
    18e3:	mov    edx,0x18
    18e8:	shl    eax,0x18
    18eb:	jmp    186f <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x36f>
    18ed:	mov    eax,r8d
    18f0:	mov    ecx,0x18
    18f5:	shl    eax,0x18
    18f8:	jmp    1853 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x353>
    18fd:	nop    DWORD PTR [rax]
    1900:	movd   xmm0,r9d
    1905:	cmp    eax,0xffffffea
    1908:	jl     1650 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x150>
    190e:	or     r8d,0x800000
    1915:	sub    ecx,edx
    1917:	shr    r8d,cl
    191a:	or     r8d,r9d
    191d:	movd   xmm0,r8d
    1922:	jmp    1650 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x150>
    1927:	nop    WORD PTR [rax+rax*1+0x0]

0000000000001930 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0>:
    1930:	pxor   xmm4,xmm4
    1934:	movapd xmm2,xmm0
    1938:	movapd xmm3,xmm0
    193c:	cmpltsd xmm0,xmm4
    1941:	xorpd  xmm3,XMMWORD PTR [rip+0xa917]        # c260 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    1949:	comisd xmm4,xmm1
    194d:	andpd  xmm3,xmm0
    1951:	andnpd xmm0,xmm2
    1955:	orpd   xmm0,xmm3
    1959:	ja     1b30 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x200>
    195f:	ucomisd xmm1,xmm4
    1963:	jp     196b <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x3b>
    1965:	je     1b18 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x1e8>
    196b:	movapd xmm3,xmm1
    196f:	movapd xmm4,xmm3
    1973:	movapd xmm3,xmm1
    1977:	maxsd  xmm4,xmm0
    197b:	divsd  xmm2,xmm4
    197f:	divsd  xmm3,xmm4
    1983:	movapd xmm1,xmm2
    1987:	mulsd  xmm1,xmm2
    198b:	movapd xmm0,xmm3
    198f:	mulsd  xmm0,xmm3
    1993:	addsd  xmm1,xmm0
    1997:	pxor   xmm0,xmm0
    199b:	comisd xmm0,xmm2
    199f:	ja     1b48 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x218>
    19a5:	ucomisd xmm1,xmm0
    19a9:	movapd xmm4,xmm2
    19ad:	jp     19b9 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x89>
    19af:	pxor   xmm0,xmm0
    19b3:	je     1af4 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x1c4>
    19b9:	movsd  xmm0,QWORD PTR [rip+0xa8d7]        # c298 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x58>
    19c1:	movapd xmm5,xmm0
    19c5:	maxsd  xmm5,xmm1
    19c9:	pxor   xmm1,xmm1
    19cd:	cvtsd2ss xmm1,xmm5
    19d1:	movd   eax,xmm1
    19d5:	mov    r8d,eax
    19d8:	shr    r8d,0x17
    19dc:	movzx  ecx,r8b
    19e0:	cmp    ecx,0xff
    19e6:	je     1c10 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x2e0>
    19ec:	test   ecx,ecx
    19ee:	je     1b88 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x258>
    19f4:	and    eax,0x807fffff
    19f9:	mov    edx,0x7e
    19fe:	and    r8d,0x1
    1a02:	or     eax,0x3f000000
    1a07:	sub    edx,ecx
    1a09:	movd   xmm1,eax
    1a0d:	lea    rax,[rip+0xa82c]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    1a14:	sar    edx,1
    1a16:	movss  xmm6,DWORD PTR [rax+r8*4]
    1a1c:	mulss  xmm1,DWORD PTR [rip+0xa5e8]        # c00c <_IO_stdin_used+0xc>
    1a24:	movss  xmm0,DWORD PTR [rip+0xa5e4]        # c010 <_IO_stdin_used+0x10>
    1a2c:	subss  xmm0,xmm1
    1a30:	mulss  xmm0,xmm6
    1a34:	movd   ecx,xmm0
    1a38:	mov    eax,ecx
    1a3a:	shr    eax,0x17
    1a3d:	movzx  eax,al
    1a40:	cmp    eax,0xff
    1a45:	je     1a90 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x160>
    1a47:	mov    r8d,ecx
    1a4a:	mov    r10d,eax
    1a4d:	and    r8d,0x7fffff
    1a54:	or     r10d,r8d
    1a57:	je     1a90 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x160>
    1a59:	mov    r9d,ecx
    1a5c:	and    r9d,0x80000000
    1a63:	test   eax,eax
    1a65:	je     1c20 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x2f0>
    1a6b:	add    eax,edx
    1a6d:	je     1cd0 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x3a0>
    1a73:	cmp    eax,0xfe
    1a78:	jg     1cd0 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x3a0>
    1a7e:	shl    eax,0x17
    1a81:	and    ecx,0x807fffff
    1a87:	or     eax,ecx
    1a89:	movd   xmm0,eax
    1a8d:	nop    DWORD PTR [rax]
    1a90:	cvtss2sd xmm0,xmm0
    1a94:	movapd xmm1,xmm0
    1a98:	movsd  xmm7,QWORD PTR [rip+0xa800]        # c2a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x60>
    1aa0:	movsd  xmm6,QWORD PTR [rip+0xa800]        # c2a8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x68>
    1aa8:	mulsd  xmm1,xmm5
    1aac:	mulsd  xmm1,xmm0
    1ab0:	mulsd  xmm7,xmm1
    1ab4:	subsd  xmm6,xmm7
    1ab8:	mulsd  xmm6,xmm1
    1abc:	movsd  xmm1,QWORD PTR [rip+0xa7ec]        # c2b0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x70>
    1ac4:	subsd  xmm1,xmm6
    1ac8:	movsd  xmm6,QWORD PTR [rip+0xa7e8]        # c2b8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x78>
    1ad0:	mulsd  xmm6,xmm5
    1ad4:	mulsd  xmm1,xmm0
    1ad8:	movsd  xmm0,QWORD PTR [rip+0xa7e0]        # c2c0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
    1ae0:	mulsd  xmm6,xmm1
    1ae4:	mulsd  xmm6,xmm1
    1ae8:	subsd  xmm0,xmm6
    1aec:	mulsd  xmm0,xmm1
    1af0:	mulsd  xmm0,xmm5
    1af4:	pxor   xmm1,xmm1
    1af8:	ucomisd xmm4,xmm1
    1afc:	jp     1b64 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x234>
    1afe:	jne    1b64 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x234>
    1b00:	divsd  xmm3,xmm0
    1b04:	movapd xmm5,xmm1
    1b08:	movsd  QWORD PTR [rdi],xmm5
    1b0c:	movsd  QWORD PTR [rsi],xmm3
    1b10:	ret
    1b11:	nop    DWORD PTR [rax+0x0]
    1b18:	movsd  xmm5,QWORD PTR [rip+0xa770]        # c290 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x50>
    1b20:	pxor   xmm3,xmm3
    1b24:	movsd  QWORD PTR [rdi],xmm5
    1b28:	movsd  QWORD PTR [rsi],xmm3
    1b2c:	ret
    1b2d:	nop    DWORD PTR [rax]
    1b30:	movapd xmm3,xmm1
    1b34:	xorpd  xmm3,XMMWORD PTR [rip+0xa724]        # c260 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    1b3c:	jmp    196f <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x3f>
    1b41:	nop    DWORD PTR [rax+0x0]
    1b48:	ucomisd xmm1,xmm0
    1b4c:	movapd xmm4,xmm2
    1b50:	xorpd  xmm4,XMMWORD PTR [rip+0xa708]        # c260 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    1b58:	jp     19b9 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x89>
    1b5e:	jne    19b9 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x89>
    1b64:	divsd  xmm2,xmm4
    1b68:	movapd xmm5,xmm4
    1b6c:	divsd  xmm3,xmm0
    1b70:	divsd  xmm5,xmm0
    1b74:	mulsd  xmm3,xmm2
    1b78:	movsd  QWORD PTR [rdi],xmm5
    1b7c:	movsd  QWORD PTR [rsi],xmm3
    1b80:	ret
    1b81:	nop    DWORD PTR [rax+0x0]
    1b88:	mov    r8d,eax
    1b8b:	and    r8d,0x7fffff
    1b92:	je     1c10 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x2e0>
    1b94:	mov    r9d,eax
    1b97:	and    r9d,0x80000000
    1b9e:	test   eax,0x7f0000
    1ba3:	mov    eax,r8d
    1ba6:	je     1cb0 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x380>
    1bac:	shl    eax,0x8
    1baf:	mov    edx,0x8
    1bb4:	cmp    eax,0xfffffff
    1bb9:	ja     1d20 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x3f0>
    1bbf:	add    edx,0x4
    1bc2:	shl    eax,0x4
    1bc5:	cmp    eax,0x3fffffff
    1bca:	ja     1bd2 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x2a2>
    1bcc:	add    edx,0x2
    1bcf:	shl    eax,0x2
    1bd2:	test   eax,eax
    1bd4:	js     1bd9 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x2a9>
    1bd6:	add    edx,0x1
    1bd9:	lea    ecx,[rdx-0x9]
    1bdc:	lea    rax,[rip+0xa65d]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    1be3:	sar    edx,1
    1be5:	shl    r8d,cl
    1be8:	not    ecx
    1bea:	add    edx,0x3a
    1bed:	and    r8d,0x7fffff
    1bf4:	and    ecx,0x1
    1bf7:	or     r8d,r9d
    1bfa:	movss  xmm6,DWORD PTR [rax+rcx*4]
    1bff:	or     r8d,0x3f000000
    1c06:	movd   xmm1,r8d
    1c0b:	jmp    1a1c <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0xec>
    1c10:	movss  xmm6,DWORD PTR [rip+0xa3ec]        # c004 <_IO_stdin_used+0x4>
    1c18:	xor    edx,edx
    1c1a:	jmp    1a1c <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0xec>
    1c1f:	nop
    1c20:	and    ecx,0x7f0000
    1c26:	mov    eax,r8d
    1c29:	je     1c90 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x360>
    1c2b:	shl    eax,0x8
    1c2e:	mov    ecx,0x8
    1c33:	cmp    eax,0xfffffff
    1c38:	ja     1d10 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x3e0>
    1c3e:	add    ecx,0x4
    1c41:	shl    eax,0x4
    1c44:	cmp    eax,0x3fffffff
    1c49:	ja     1c51 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x321>
    1c4b:	add    ecx,0x2
    1c4e:	shl    eax,0x2
    1c51:	test   eax,eax
    1c53:	js     1d00 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x3d0>
    1c59:	sub    ecx,0x8
    1c5c:	mov    eax,edx
    1c5e:	shl    r8d,cl
    1c61:	sub    eax,ecx
    1c63:	and    r8d,0x7fffff
    1c6a:	add    eax,0x1
    1c6d:	test   eax,eax
    1c6f:	jle    1d50 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x420>
    1c75:	or     r8d,r9d
    1c78:	shl    eax,0x17
    1c7b:	or     r8d,eax
    1c7e:	movd   xmm0,r8d
    1c83:	jmp    1a90 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x160>
    1c88:	nop    DWORD PTR [rax+rax*1+0x0]
    1c90:	shl    eax,0x10
    1c93:	mov    ecx,0x10
    1c98:	cmp    eax,0xffffff
    1c9d:	jbe    1d3d <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x40d>
    1ca3:	cmp    eax,0xfffffff
    1ca8:	ja     1c44 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x314>
    1caa:	jmp    1c3e <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x30e>
    1cac:	nop    DWORD PTR [rax+0x0]
    1cb0:	shl    eax,0x10
    1cb3:	mov    edx,0x10
    1cb8:	cmp    eax,0xffffff
    1cbd:	jbe    1d30 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x400>
    1cbf:	cmp    eax,0xfffffff
    1cc4:	ja     1bc5 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x295>
    1cca:	jmp    1bbf <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x28f>
    1ccf:	nop
    1cd0:	shr    r8d,1
    1cd3:	or     r8d,r9d
    1cd6:	or     r9d,0x7f800000
    1cdd:	or     r8d,0x400000
    1ce4:	cmp    eax,0xfe
    1ce9:	cmovg  r8d,r9d
    1ced:	movd   xmm0,r8d
    1cf2:	jmp    1a90 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x160>
    1cf7:	nop    WORD PTR [rax+rax*1+0x0]
    1d00:	sub    ecx,0x9
    1d03:	jmp    1c5c <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x32c>
    1d08:	nop    DWORD PTR [rax+rax*1+0x0]
    1d10:	cmp    eax,0x3fffffff
    1d15:	jbe    1c4b <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x31b>
    1d1b:	jmp    1c59 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x329>
    1d20:	cmp    eax,0x3fffffff
    1d25:	jbe    1bcc <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x29c>
    1d2b:	jmp    1bd6 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x2a6>
    1d30:	mov    eax,r8d
    1d33:	mov    edx,0x18
    1d38:	shl    eax,0x18
    1d3b:	jmp    1cbf <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x38f>
    1d3d:	mov    eax,r8d
    1d40:	mov    ecx,0x18
    1d45:	shl    eax,0x18
    1d48:	jmp    1ca3 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x373>
    1d4d:	nop    DWORD PTR [rax]
    1d50:	movd   xmm0,r9d
    1d55:	cmp    eax,0xffffffea
    1d58:	jl     1a90 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x160>
    1d5e:	or     r8d,0x800000
    1d65:	sub    ecx,edx
    1d67:	shr    r8d,cl
    1d6a:	or     r8d,r9d
    1d6d:	movd   xmm0,r8d
    1d72:	jmp    1a90 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0+0x160>
    1d77:	nop    WORD PTR [rax+rax*1+0x0]

0000000000001d80 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>:
    1d80:	push   rbx
    1d81:	mov    r9,rsi
    1d84:	mov    r11,rdi
    1d87:	mov    rbx,rdx
    1d8a:	mov    rsi,rcx
    1d8d:	sub    rsp,0x90
    1d94:	mov    rax,QWORD PTR fs:0x28
    1d9d:	mov    QWORD PTR [rsp+0x88],rax
    1da5:	xor    eax,eax
    1da7:	cmp    r8,0x10
    1dab:	je     1f74 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x1f4>
    1db1:	mov    rdi,rsp
    1db4:	mov    ecx,0x10
    1db9:	mov    r10,rsp
    1dbc:	mov    rdx,rsp
    1dbf:	rep stos QWORD PTR [rdi],rax
    1dc2:	xor    eax,eax
    1dc4:	nop    WORD PTR [rax+rax*1+0x0]
    1dca:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    1dd5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    1de0:	cmp    rax,r8
    1de3:	jae    1dee <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x6e>
    1de5:	movsd  xmm0,QWORD PTR [rbx+rax*8]
    1dea:	movsd  QWORD PTR [rdx],xmm0
    1dee:	add    rax,0x1
    1df2:	add    rdx,0x8
    1df6:	cmp    rax,0x10
    1dfa:	jne    1de0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x60>
    1dfc:	lea    rdi,[rsi+0x8]
    1e00:	mov    rcx,rsi
    1e03:	xor    eax,eax
    1e05:	mov    QWORD PTR [rsi],0x0
    1e0c:	and    rdi,0xfffffffffffffff8
    1e10:	mov    QWORD PTR [rsi+0x78],0x0
    1e18:	sub    rcx,rdi
    1e1b:	sub    ecx,0xffffff80
    1e1e:	shr    ecx,0x3
    1e21:	rep stos QWORD PTR [rdi],rax
    1e24:	cmp    r9,0x10
    1e28:	je     1f17 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x197>
    1e2e:	mov    rdi,r11
    1e31:	xor    edx,edx
    1e33:	xchg   ax,ax
    1e35:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    1e40:	xor    eax,eax
    1e42:	nop    DWORD PTR [rax+0x0]
    1e49:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    1e54:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    1e5f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    1e6a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    1e75:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    1e80:	cmp    rdx,rax
    1e83:	mov    rcx,rax
    1e86:	cmovae rcx,rdx
    1e8a:	cmp    rcx,r9
    1e8d:	jae    1ea3 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x123>
    1e8f:	movsd  xmm0,QWORD PTR [rdi+rax*8]
    1e94:	mulsd  xmm0,QWORD PTR [r10]
    1e99:	addsd  xmm0,QWORD PTR [rsi+rax*8]
    1e9e:	movsd  QWORD PTR [rsi+rax*8],xmm0
    1ea3:	add    rax,0x1
    1ea7:	cmp    rax,0x10
    1eab:	jne    1e80 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x100>
    1ead:	add    rdx,0x1
    1eb1:	sub    rdi,0xffffffffffffff80
    1eb5:	add    r10,0x8
    1eb9:	cmp    rdx,0x10
    1ebd:	jne    1e40 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0xc0>
    1ebf:	xor    eax,eax
    1ec1:	nop    WORD PTR [rax+rax*1+0x0]
    1eca:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    1ed5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    1ee0:	cmp    rax,r8
    1ee3:	jb     1eed <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x16d>
    1ee5:	mov    QWORD PTR [rsi+rax*8],0x0
    1eed:	add    rax,0x1
    1ef1:	cmp    rax,0x10
    1ef5:	jne    1ee0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x160>
    1ef7:	mov    rax,QWORD PTR [rsp+0x88]
    1eff:	sub    rax,QWORD PTR fs:0x28
    1f08:	jne    204f <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x2cf>
    1f0e:	add    rsp,0x90
    1f15:	pop    rbx
    1f16:	ret
    1f17:	mov    rdx,r11
    1f1a:	lea    rcx,[r10+0x80]
    1f21:	nop    DWORD PTR [rax+0x0]
    1f25:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    1f30:	movsd  xmm1,QWORD PTR [r10]
    1f35:	xor    eax,eax
    1f37:	unpcklpd xmm1,xmm1
    1f3b:	nop    DWORD PTR [rax+rax*1+0x0]
    1f40:	movupd xmm0,XMMWORD PTR [rdx+rax*1]
    1f45:	movupd xmm2,XMMWORD PTR [rsi+rax*1]
    1f4a:	mulpd  xmm0,xmm1
    1f4e:	addpd  xmm0,xmm2
    1f52:	movups XMMWORD PTR [rsi+rax*1],xmm0
    1f56:	add    rax,0x10
    1f5a:	cmp    rax,0x80
    1f60:	jne    1f40 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x1c0>
    1f62:	add    r10,0x8
    1f66:	sub    rdx,0xffffffffffffff80
    1f6a:	cmp    r10,rcx
    1f6d:	jne    1f30 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x1b0>
    1f6f:	jmp    1ebf <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x13f>
    1f74:	lea    rdi,[rcx+0x8]
    1f78:	mov    QWORD PTR [rcx],0x0
    1f7f:	and    rdi,0xfffffffffffffff8
    1f83:	mov    QWORD PTR [rcx+0x78],0x0
    1f8b:	sub    rcx,rdi
    1f8e:	sub    ecx,0xffffff80
    1f91:	shr    ecx,0x3
    1f94:	rep stos QWORD PTR [rdi],rax
    1f97:	cmp    r9,0x10
    1f9b:	je     2004 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x284>
    1f9d:	mov    rdi,rdx
    1fa0:	mov    r8,r11
    1fa3:	xor    edx,edx
    1fa5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    1fb0:	xor    eax,eax
    1fb2:	nop    DWORD PTR [rax]
    1fb5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    1fc0:	cmp    rdx,rax
    1fc3:	mov    rcx,rax
    1fc6:	cmovae rcx,rdx
    1fca:	cmp    rcx,r9
    1fcd:	jae    1fe3 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x263>
    1fcf:	movsd  xmm0,QWORD PTR [r8+rax*8]
    1fd5:	mulsd  xmm0,QWORD PTR [rdi]
    1fd9:	addsd  xmm0,QWORD PTR [rsi+rax*8]
    1fde:	movsd  QWORD PTR [rsi+rax*8],xmm0
    1fe3:	add    rax,0x1
    1fe7:	cmp    rax,0x10
    1feb:	jne    1fc0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x240>
    1fed:	add    rdx,0x1
    1ff1:	sub    r8,0xffffffffffffff80
    1ff5:	add    rdi,0x8
    1ff9:	cmp    rdx,0x10
    1ffd:	jne    1fb0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x230>
    1fff:	jmp    1ef7 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x177>
    2004:	mov    rdx,r11
    2007:	mov    rcx,rbx
    200a:	lea    rdi,[r11+0x800]
    2011:	xor    eax,eax
    2013:	xchg   ax,ax
    2015:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2020:	movsd  xmm0,QWORD PTR [rdx+rax*8]
    2025:	mulsd  xmm0,QWORD PTR [rcx]
    2029:	addsd  xmm0,QWORD PTR [rsi+rax*8]
    202e:	movsd  QWORD PTR [rsi+rax*8],xmm0
    2033:	add    rax,0x1
    2037:	cmp    rax,0x10
    203b:	jne    2020 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x2a0>
    203d:	sub    rdx,0xffffffffffffff80
    2041:	add    rcx,0x8
    2045:	cmp    rdx,rdi
    2048:	jne    2011 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x291>
    204a:	jmp    1ef7 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x177>
    204f:	call   10b0 <__stack_chk_fail@plt>
    2054:	cs nop WORD PTR [rax+rax*1+0x0]
    205e:	cs nop WORD PTR [rax+rax*1+0x0]
    2068:	cs nop WORD PTR [rax+rax*1+0x0]
    2072:	cs nop WORD PTR [rax+rax*1+0x0]
    207c:	nop    DWORD PTR [rax+0x0]

0000000000002080 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>:
    2080:	sub    rsp,0x58
    2084:	mov    r9,rsi
    2087:	pxor   xmm0,xmm0
    208b:	mov    r11,rdx
    208e:	mov    rsi,QWORD PTR fs:0x28
    2097:	mov    QWORD PTR [rsp+0x48],rsi
    209c:	mov    rsi,rcx
    209f:	cmp    r8,0x10
    20a3:	je     2219 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x199>
    20a9:	movaps XMMWORD PTR [rsp],xmm0
    20ad:	mov    r10,rsp
    20b0:	mov    rdx,rsp
    20b3:	xor    eax,eax
    20b5:	movaps XMMWORD PTR [rsp+0x10],xmm0
    20ba:	movaps XMMWORD PTR [rsp+0x20],xmm0
    20bf:	movaps XMMWORD PTR [rsp+0x30],xmm0
    20c4:	nop    WORD PTR [rax+rax*1+0x0]
    20ca:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    20d5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    20e0:	cmp    rax,r8
    20e3:	jae    20ef <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x6f>
    20e5:	movss  xmm0,DWORD PTR [r11+rax*4]
    20eb:	movss  DWORD PTR [rdx],xmm0
    20ef:	add    rax,0x1
    20f3:	add    rdx,0x4
    20f7:	cmp    rax,0x10
    20fb:	jne    20e0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x60>
    20fd:	pxor   xmm0,xmm0
    2101:	xor    edx,edx
    2103:	movups XMMWORD PTR [rsi],xmm0
    2106:	movups XMMWORD PTR [rsi+0x10],xmm0
    210a:	movups XMMWORD PTR [rsi+0x20],xmm0
    210e:	movups XMMWORD PTR [rsi+0x30],xmm0
    2112:	cmp    r9,0x10
    2116:	je     21cf <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x14f>
    211c:	nop    DWORD PTR [rax+0x0]
    2120:	xor    eax,eax
    2122:	nop    DWORD PTR [rax+rax*1+0x0]
    212a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2135:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2140:	cmp    rdx,rax
    2143:	mov    rcx,rax
    2146:	cmovae rcx,rdx
    214a:	cmp    rcx,r9
    214d:	jae    2163 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0xe3>
    214f:	movss  xmm0,DWORD PTR [rdi+rax*4]
    2154:	mulss  xmm0,DWORD PTR [r10]
    2159:	addss  xmm0,DWORD PTR [rsi+rax*4]
    215e:	movss  DWORD PTR [rsi+rax*4],xmm0
    2163:	add    rax,0x1
    2167:	cmp    rax,0x10
    216b:	jne    2140 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0xc0>
    216d:	add    rdx,0x1
    2171:	add    rdi,0x40
    2175:	add    r10,0x4
    2179:	cmp    rdx,0x10
    217d:	jne    2120 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0xa0>
    217f:	xor    eax,eax
    2181:	nop    WORD PTR [rax+rax*1+0x0]
    218a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2195:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    21a0:	cmp    rax,r8
    21a3:	jb     21ac <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x12c>
    21a5:	mov    DWORD PTR [rsi+rax*4],0x0
    21ac:	add    rax,0x1
    21b0:	cmp    rax,0x10
    21b4:	jne    21a0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x120>
    21b6:	mov    rax,QWORD PTR [rsp+0x48]
    21bb:	sub    rax,QWORD PTR fs:0x28
    21c4:	jne    230f <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x28f>
    21ca:	add    rsp,0x58
    21ce:	ret
    21cf:	mov    rax,rdi
    21d2:	lea    rcx,[r10+0x40]
    21d6:	cs nop WORD PTR [rax+rax*1+0x0]
    21e0:	movss  xmm1,DWORD PTR [r10]
    21e5:	xor    edx,edx
    21e7:	shufps xmm1,xmm1,0x0
    21eb:	movups xmm0,XMMWORD PTR [rax+rdx*1]
    21ef:	movups xmm2,XMMWORD PTR [rsi+rdx*1]
    21f3:	mulps  xmm0,xmm1
    21f6:	addps  xmm0,xmm2
    21f9:	movups XMMWORD PTR [rsi+rdx*1],xmm0
    21fd:	add    rdx,0x10
    2201:	cmp    rdx,0x40
    2205:	jne    21eb <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x16b>
    2207:	add    r10,0x4
    220b:	add    rax,0x40
    220f:	cmp    r10,rcx
    2212:	jne    21e0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x160>
    2214:	jmp    217f <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0xff>
    2219:	movups XMMWORD PTR [rsi],xmm0
    221c:	movups XMMWORD PTR [rsi+0x10],xmm0
    2220:	movups XMMWORD PTR [rsi+0x20],xmm0
    2224:	movups XMMWORD PTR [rsi+0x30],xmm0
    2228:	cmp    r9,0x10
    222c:	je     22c4 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x244>
    2232:	mov    rdx,rdi
    2235:	mov    r8,r11
    2238:	xor    ecx,ecx
    223a:	nop    WORD PTR [rax+rax*1+0x0]
    2240:	xor    eax,eax
    2242:	nop    DWORD PTR [rax+0x0]
    2249:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2254:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    225f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    226a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2275:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2280:	cmp    rcx,rax
    2283:	mov    rdi,rax
    2286:	cmovae rdi,rcx
    228a:	cmp    rdi,r9
    228d:	jae    22a3 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x223>
    228f:	movss  xmm0,DWORD PTR [rdx+rax*4]
    2294:	mulss  xmm0,DWORD PTR [r8]
    2299:	addss  xmm0,DWORD PTR [rsi+rax*4]
    229e:	movss  DWORD PTR [rsi+rax*4],xmm0
    22a3:	add    rax,0x1
    22a7:	cmp    rax,0x10
    22ab:	jne    2280 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x200>
    22ad:	add    rcx,0x1
    22b1:	add    rdx,0x40
    22b5:	add    r8,0x4
    22b9:	cmp    rcx,0x10
    22bd:	jne    2240 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x1c0>
    22bf:	jmp    21b6 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x136>
    22c4:	mov    rax,rdi
    22c7:	mov    rcx,rdx
    22ca:	lea    rdi,[rdi+0x400]
    22d1:	xor    edx,edx
    22d3:	xchg   ax,ax
    22d5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    22e0:	movss  xmm0,DWORD PTR [rax+rdx*4]
    22e5:	mulss  xmm0,DWORD PTR [rcx]
    22e9:	addss  xmm0,DWORD PTR [rsi+rdx*4]
    22ee:	movss  DWORD PTR [rsi+rdx*4],xmm0
    22f3:	add    rdx,0x1
    22f7:	cmp    rdx,0x10
    22fb:	jne    22e0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x260>
    22fd:	add    rax,0x40
    2301:	add    rcx,0x4
    2305:	cmp    rax,rdi
    2308:	jne    22d1 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x251>
    230a:	jmp    21b6 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x136>
    230f:	call   10b0 <__stack_chk_fail@plt>
    2314:	cs nop WORD PTR [rax+rax*1+0x0]
    231e:	cs nop WORD PTR [rax+rax*1+0x0]
    2328:	cs nop WORD PTR [rax+rax*1+0x0]
    2332:	cs nop WORD PTR [rax+rax*1+0x0]
    233c:	nop    DWORD PTR [rax+0x0]

0000000000002340 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>:
    2340:	sub    rsp,0x38
    2344:	mov    r9,rsi
    2347:	pxor   xmm0,xmm0
    234b:	mov    r11,rdx
    234e:	mov    rsi,QWORD PTR fs:0x28
    2357:	mov    QWORD PTR [rsp+0x28],rsi
    235c:	mov    rsi,rcx
    235f:	cmp    r8,0x8
    2363:	je     24f5 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x1b5>
    2369:	movaps XMMWORD PTR [rsp],xmm0
    236d:	mov    r10,rsp
    2370:	mov    rdx,rsp
    2373:	xor    eax,eax
    2375:	movaps XMMWORD PTR [rsp+0x10],xmm0
    237a:	nop    WORD PTR [rax+rax*1+0x0]
    2380:	cmp    rax,r8
    2383:	jae    238f <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x4f>
    2385:	movss  xmm0,DWORD PTR [r11+rax*4]
    238b:	movss  DWORD PTR [rdx],xmm0
    238f:	add    rax,0x1
    2393:	add    rdx,0x4
    2397:	cmp    rax,0x8
    239b:	jne    2380 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x40>
    239d:	pxor   xmm0,xmm0
    23a1:	xor    edx,edx
    23a3:	movups XMMWORD PTR [rsi],xmm0
    23a6:	movups XMMWORD PTR [rsi+0x10],xmm0
    23aa:	cmp    r9,0x8
    23ae:	je     248f <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x14f>
    23b4:	nop
    23b5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    23c0:	xor    eax,eax
    23c2:	nop    DWORD PTR [rax+0x0]
    23c9:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    23d4:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    23df:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    23ea:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    23f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2400:	cmp    rdx,rax
    2403:	mov    rcx,rax
    2406:	cmovae rcx,rdx
    240a:	cmp    rcx,r9
    240d:	jae    2423 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0xe3>
    240f:	movss  xmm0,DWORD PTR [rdi+rax*4]
    2414:	mulss  xmm0,DWORD PTR [r10]
    2419:	addss  xmm0,DWORD PTR [rsi+rax*4]
    241e:	movss  DWORD PTR [rsi+rax*4],xmm0
    2423:	add    rax,0x1
    2427:	cmp    rax,0x8
    242b:	jne    2400 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0xc0>
    242d:	add    rdx,0x1
    2431:	add    rdi,0x20
    2435:	add    r10,0x4
    2439:	cmp    rdx,0x8
    243d:	jne    23c0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x80>
    243f:	xor    eax,eax
    2441:	nop    WORD PTR [rax+rax*1+0x0]
    244a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2455:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2460:	cmp    rax,r8
    2463:	jb     246c <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x12c>
    2465:	mov    DWORD PTR [rsi+rax*4],0x0
    246c:	add    rax,0x1
    2470:	cmp    rax,0x8
    2474:	jne    2460 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x120>
    2476:	mov    rax,QWORD PTR [rsp+0x28]
    247b:	sub    rax,QWORD PTR fs:0x28
    2484:	jne    25d0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x290>
    248a:	add    rsp,0x38
    248e:	ret
    248f:	movups xmm2,XMMWORD PTR [rsi]
    2492:	movups xmm1,XMMWORD PTR [rsi+0x10]
    2496:	lea    rax,[rdi+0x100]
    249d:	xchg   ax,ax
    249f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    24aa:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    24b5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    24c0:	movups xmm0,XMMWORD PTR [rdi]
    24c3:	movss  xmm3,DWORD PTR [r10]
    24c8:	add    rdi,0x20
    24cc:	add    r10,0x4
    24d0:	shufps xmm3,xmm3,0x0
    24d4:	mulps  xmm0,xmm3
    24d7:	addps  xmm2,xmm0
    24da:	movups XMMWORD PTR [rsi],xmm2
    24dd:	movups xmm0,XMMWORD PTR [rdi-0x10]
    24e1:	mulps  xmm0,xmm3
    24e4:	addps  xmm1,xmm0
    24e7:	movups XMMWORD PTR [rsi+0x10],xmm1
    24eb:	cmp    rdi,rax
    24ee:	jne    24c0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x180>
    24f0:	jmp    243f <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0xff>
    24f5:	movups XMMWORD PTR [rsi],xmm0
    24f8:	xor    edx,edx
    24fa:	movups XMMWORD PTR [rsi+0x10],xmm0
    24fe:	cmp    r9,0x8
    2502:	je     2581 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x241>
    2504:	nop
    2505:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2510:	xor    eax,eax
    2512:	xchg   ax,ax
    2514:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    251f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    252a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2535:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2540:	cmp    rdx,rax
    2543:	mov    rcx,rax
    2546:	cmovae rcx,rdx
    254a:	cmp    rcx,r9
    254d:	jae    2564 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x224>
    254f:	movss  xmm0,DWORD PTR [rdi+rax*4]
    2554:	mulss  xmm0,DWORD PTR [r11+rdx*4]
    255a:	addss  xmm0,DWORD PTR [rsi+rax*4]
    255f:	movss  DWORD PTR [rsi+rax*4],xmm0
    2564:	add    rax,0x1
    2568:	cmp    rax,0x8
    256c:	jne    2540 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x200>
    256e:	add    rdx,0x1
    2572:	add    rdi,0x20
    2576:	cmp    rdx,0x8
    257a:	jne    2510 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x1d0>
    257c:	jmp    2476 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x136>
    2581:	mov    rax,rdi
    2584:	lea    rcx,[rdi+0x100]
    258b:	nop    DWORD PTR [rax+rax*1+0x0]
    2590:	xor    edx,edx
    2592:	nop    DWORD PTR [rax]
    2595:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    25a0:	movss  xmm0,DWORD PTR [rax+rdx*4]
    25a5:	mulss  xmm0,DWORD PTR [r11]
    25aa:	addss  xmm0,DWORD PTR [rsi+rdx*4]
    25af:	movss  DWORD PTR [rsi+rdx*4],xmm0
    25b4:	add    rdx,0x1
    25b8:	cmp    rdx,0x8
    25bc:	jne    25a0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x260>
    25be:	add    rax,0x20
    25c2:	add    r11,0x4
    25c6:	cmp    rax,rcx
    25c9:	jne    2590 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x250>
    25cb:	jmp    2476 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x136>
    25d0:	call   10b0 <__stack_chk_fail@plt>
    25d5:	cs nop WORD PTR [rax+rax*1+0x0]
    25df:	cs nop WORD PTR [rax+rax*1+0x0]
    25e9:	cs nop WORD PTR [rax+rax*1+0x0]
    25f3:	cs nop WORD PTR [rax+rax*1+0x0]
    25fd:	nop    DWORD PTR [rax]

0000000000002600 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>:
    2600:	sub    rsp,0x58
    2604:	mov    r9,rsi
    2607:	pxor   xmm0,xmm0
    260b:	mov    rsi,QWORD PTR fs:0x28
    2614:	mov    QWORD PTR [rsp+0x48],rsi
    2619:	mov    rsi,rdx
    261c:	cmp    r8,0x8
    2620:	je     279d <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x19d>
    2626:	movaps XMMWORD PTR [rsp],xmm0
    262a:	mov    r10,rsp
    262d:	mov    rdx,rsp
    2630:	xor    eax,eax
    2632:	movaps XMMWORD PTR [rsp+0x10],xmm0
    2637:	movaps XMMWORD PTR [rsp+0x20],xmm0
    263c:	movaps XMMWORD PTR [rsp+0x30],xmm0
    2641:	nop    WORD PTR [rax+rax*1+0x0]
    264a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2655:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2660:	cmp    rax,r8
    2663:	jae    266e <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x6e>
    2665:	movsd  xmm0,QWORD PTR [rsi+rax*8]
    266a:	movsd  QWORD PTR [rdx],xmm0
    266e:	add    rax,0x1
    2672:	add    rdx,0x8
    2676:	cmp    rax,0x8
    267a:	jne    2660 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x60>
    267c:	pxor   xmm0,xmm0
    2680:	xor    edx,edx
    2682:	movups XMMWORD PTR [rcx],xmm0
    2685:	movups XMMWORD PTR [rcx+0x10],xmm0
    2689:	movups XMMWORD PTR [rcx+0x20],xmm0
    268d:	movups XMMWORD PTR [rcx+0x30],xmm0
    2691:	cmp    r9,0x8
    2695:	je     2750 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x150>
    269b:	nop    DWORD PTR [rax+rax*1+0x0]
    26a0:	xor    eax,eax
    26a2:	nop    DWORD PTR [rax+rax*1+0x0]
    26aa:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    26b5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    26c0:	cmp    rdx,rax
    26c3:	mov    rsi,rax
    26c6:	cmovae rsi,rdx
    26ca:	cmp    rsi,r9
    26cd:	jae    26e3 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0xe3>
    26cf:	movsd  xmm0,QWORD PTR [rdi+rax*8]
    26d4:	mulsd  xmm0,QWORD PTR [r10]
    26d9:	addsd  xmm0,QWORD PTR [rcx+rax*8]
    26de:	movsd  QWORD PTR [rcx+rax*8],xmm0
    26e3:	add    rax,0x1
    26e7:	cmp    rax,0x8
    26eb:	jne    26c0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0xc0>
    26ed:	add    rdx,0x1
    26f1:	add    rdi,0x40
    26f5:	add    r10,0x8
    26f9:	cmp    rdx,0x8
    26fd:	jne    26a0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0xa0>
    26ff:	xor    eax,eax
    2701:	nop    WORD PTR [rax+rax*1+0x0]
    270a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2715:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2720:	cmp    rax,r8
    2723:	jb     272d <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x12d>
    2725:	mov    QWORD PTR [rcx+rax*8],0x0
    272d:	add    rax,0x1
    2731:	cmp    rax,0x8
    2735:	jne    2720 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x120>
    2737:	mov    rax,QWORD PTR [rsp+0x48]
    273c:	sub    rax,QWORD PTR fs:0x28
    2745:	jne    288f <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x28f>
    274b:	add    rsp,0x58
    274f:	ret
    2750:	lea    rdx,[rdi+0x200]
    2757:	nop    WORD PTR [rax+rax*1+0x0]
    2760:	movsd  xmm1,QWORD PTR [r10]
    2765:	xor    eax,eax
    2767:	unpcklpd xmm1,xmm1
    276b:	movupd xmm0,XMMWORD PTR [rdi+rax*1]
    2770:	movupd xmm2,XMMWORD PTR [rcx+rax*1]
    2775:	mulpd  xmm0,xmm1
    2779:	addpd  xmm0,xmm2
    277d:	movups XMMWORD PTR [rcx+rax*1],xmm0
    2781:	add    rax,0x10
    2785:	cmp    rax,0x40
    2789:	jne    276b <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x16b>
    278b:	add    rdi,0x40
    278f:	add    r10,0x8
    2793:	cmp    rdi,rdx
    2796:	jne    2760 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x160>
    2798:	jmp    26ff <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0xff>
    279d:	movups XMMWORD PTR [rcx],xmm0
    27a0:	xor    edx,edx
    27a2:	movups XMMWORD PTR [rcx+0x10],xmm0
    27a6:	movups XMMWORD PTR [rcx+0x20],xmm0
    27aa:	movups XMMWORD PTR [rcx+0x30],xmm0
    27ae:	cmp    r9,0x8
    27b2:	je     2840 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x240>
    27b8:	nop    DWORD PTR [rax+rax*1+0x0]
    27c0:	xor    eax,eax
    27c2:	nop    DWORD PTR [rax+0x0]
    27c9:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    27d4:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    27df:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    27ea:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    27f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2800:	cmp    rdx,rax
    2803:	mov    r8,rax
    2806:	cmovae r8,rdx
    280a:	cmp    r8,r9
    280d:	jae    2823 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x223>
    280f:	movsd  xmm0,QWORD PTR [rdi+rax*8]
    2814:	mulsd  xmm0,QWORD PTR [rsi+rdx*8]
    2819:	addsd  xmm0,QWORD PTR [rcx+rax*8]
    281e:	movsd  QWORD PTR [rcx+rax*8],xmm0
    2823:	add    rax,0x1
    2827:	cmp    rax,0x8
    282b:	jne    2800 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x200>
    282d:	add    rdx,0x1
    2831:	add    rdi,0x40
    2835:	cmp    rdx,0x8
    2839:	jne    27c0 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x1c0>
    283b:	jmp    2737 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x137>
    2840:	mov    rax,rdi
    2843:	lea    rdi,[rdi+0x200]
    284a:	nop    WORD PTR [rax+rax*1+0x0]
    2850:	xor    edx,edx
    2852:	nop    DWORD PTR [rax]
    2855:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2860:	movsd  xmm0,QWORD PTR [rax+rdx*8]
    2865:	mulsd  xmm0,QWORD PTR [rsi]
    2869:	addsd  xmm0,QWORD PTR [rcx+rdx*8]
    286e:	movsd  QWORD PTR [rcx+rdx*8],xmm0
    2873:	add    rdx,0x1
    2877:	cmp    rdx,0x8
    287b:	jne    2860 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x260>
    287d:	add    rax,0x40
    2881:	add    rsi,0x8
    2885:	cmp    rax,rdi
    2888:	jne    2850 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x250>
    288a:	jmp    2737 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0+0x137>
    288f:	call   10b0 <__stack_chk_fail@plt>
    2894:	cs nop WORD PTR [rax+rax*1+0x0]
    289e:	xchg   ax,ax

00000000000028a0 <instruction_count_begin>:
    28a0:	endbr64
    28a4:	ret
    28a5:	cs nop WORD PTR [rax+rax*1+0x0]
    28af:	nop

00000000000028b0 <instruction_count_end>:
    28b0:	endbr64
    28b4:	ret
    28b5:	cs nop WORD PTR [rax+rax*1+0x0]
    28bf:	nop

00000000000028c0 <_ZN12MultiplyCaseIfLm8EE5applyEPv>:
    28c0:	endbr64
    28c4:	mov    rsi,QWORD PTR [rdi+0x140]
    28cb:	pxor   xmm0,xmm0
    28cf:	movups XMMWORD PTR [rdi+0x120],xmm0
    28d6:	movups XMMWORD PTR [rdi+0x130],xmm0
    28dd:	cmp    rsi,0x8
    28e1:	je     2949 <_ZN12MultiplyCaseIfLm8EE5applyEPv+0x89>
    28e3:	mov    r8,rdi
    28e6:	xor    edx,edx
    28e8:	nop    DWORD PTR [rax+rax*1+0x0]
    28f0:	xor    eax,eax
    28f2:	nop    DWORD PTR [rax]
    28f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2900:	cmp    rdx,rax
    2903:	mov    rcx,rax
    2906:	cmovae rcx,rdx
    290a:	cmp    rcx,rsi
    290d:	jae    2930 <_ZN12MultiplyCaseIfLm8EE5applyEPv+0x70>
    290f:	movss  xmm0,DWORD PTR [r8+rax*4]
    2915:	mulss  xmm0,DWORD PTR [rdi+rdx*4+0x100]
    291e:	addss  xmm0,DWORD PTR [rdi+rax*4+0x120]
    2927:	movss  DWORD PTR [rdi+rax*4+0x120],xmm0
    2930:	add    rax,0x1
    2934:	cmp    rax,0x8
    2938:	jne    2900 <_ZN12MultiplyCaseIfLm8EE5applyEPv+0x40>
    293a:	add    rdx,0x1
    293e:	add    r8,0x20
    2942:	cmp    rdx,0x8
    2946:	jne    28f0 <_ZN12MultiplyCaseIfLm8EE5applyEPv+0x30>
    2948:	ret
    2949:	lea    rcx,[rdi+0x100]
    2950:	xor    edx,edx
    2952:	lea    rsi,[rdi+0x140]
    2959:	nop    DWORD PTR [rax+0x0]
    2960:	lea    rax,[rdi+0x120]
    2967:	nop    DWORD PTR [rax]
    296a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2975:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2980:	movss  xmm0,DWORD PTR [rax+rdx*1-0x120]
    2989:	mulss  xmm0,DWORD PTR [rcx]
    298d:	add    rax,0x4
    2991:	addss  xmm0,DWORD PTR [rax-0x4]
    2996:	movss  DWORD PTR [rax-0x4],xmm0
    299b:	cmp    rsi,rax
    299e:	jne    2980 <_ZN12MultiplyCaseIfLm8EE5applyEPv+0xc0>
    29a0:	add    rdx,0x20
    29a4:	add    rcx,0x4
    29a8:	cmp    rdx,0x100
    29af:	jne    2960 <_ZN12MultiplyCaseIfLm8EE5applyEPv+0xa0>
    29b1:	ret
    29b2:	cs nop WORD PTR [rax+rax*1+0x0]
    29bc:	nop    DWORD PTR [rax+0x0]

00000000000029c0 <_Z3runI12MultiplyCaseIfLm8EEEvv>:
    29c0:	endbr64
    29c4:	push   r14
    29c6:	mov    ecx,0x20
    29cb:	pxor   xmm0,xmm0
    29cf:	xor    r9d,r9d
    29d2:	movabs r8,0xcccccccccccccccd
    29dc:	push   r13
    29de:	push   r12
    29e0:	push   rbp
    29e1:	push   rbx
    29e2:	sub    rsp,0x150
    29e9:	movsd  xmm4,QWORD PTR [rip+0x98d7]        # c2c8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x88>
    29f1:	movsd  xmm1,QWORD PTR [rip+0x98e7]        # c2e0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa0>
    29f9:	mov    rax,QWORD PTR fs:0x28
    2a02:	mov    QWORD PTR [rsp+0x148],rax
    2a0a:	xor    eax,eax
    2a0c:	mov    rdi,rsp
    2a0f:	movsd  xmm3,QWORD PTR [rip+0x98b9]        # c2d0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x90>
    2a17:	mov    rbx,rsp
    2a1a:	movaps XMMWORD PTR [rsp+0x100],xmm0
    2a22:	mov    rbp,rsp
    2a25:	mov    r10,rsp
    2a28:	movsd  xmm2,QWORD PTR [rip+0x98a8]        # c2d8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x98>
    2a30:	mov    QWORD PTR [rsp+0x140],0x8
    2a3c:	movaps XMMWORD PTR [rsp+0x110],xmm0
    2a44:	movaps XMMWORD PTR [rsp+0x120],xmm0
    2a4c:	movaps XMMWORD PTR [rsp+0x130],xmm0
    2a54:	rep stos QWORD PTR [rdi],rax
    2a57:	nop    WORD PTR [rax+rax*1+0x0]
    2a60:	mov    rdi,r9
    2a63:	pxor   xmm0,xmm0
    2a67:	add    r9,0x1
    2a6b:	mov    rsi,r10
    2a6e:	cvtsi2sd xmm0,r9
    2a73:	divsd  xmm0,xmm4
    2a77:	xor    ecx,ecx
    2a79:	cvtsd2ss xmm0,xmm0
    2a7d:	movss  DWORD PTR [r10+0x100],xmm0
    2a86:	jmp    2acf <_Z3runI12MultiplyCaseIfLm8EEEvv+0x10f>
    2a88:	nop    DWORD PTR [rax+rax*1+0x0]
    2a90:	lea    r11,[rcx+rdi*1]
    2a94:	pxor   xmm0,xmm0
    2a98:	mov    rax,r11
    2a9b:	mul    r8
    2a9e:	mov    rax,rdx
    2aa1:	and    rdx,0xfffffffffffffffc
    2aa5:	shr    rax,0x2
    2aa9:	add    rdx,rax
    2aac:	sub    r11,rdx
    2aaf:	cvtsi2sd xmm0,r11
    2ab4:	mulsd  xmm0,xmm1
    2ab8:	add    rcx,0x1
    2abc:	cvtsd2ss xmm0,xmm0
    2ac0:	add    rsi,0x20
    2ac4:	movss  DWORD PTR [rsi-0x20],xmm0
    2ac9:	cmp    rcx,0x8
    2acd:	je     2af0 <_Z3runI12MultiplyCaseIfLm8EEEvv+0x130>
    2acf:	cmp    rcx,rdi
    2ad2:	jne    2a90 <_Z3runI12MultiplyCaseIfLm8EEEvv+0xd0>
    2ad4:	pxor   xmm0,xmm0
    2ad8:	cvtsi2sd xmm0,rcx
    2add:	mulsd  xmm0,xmm3
    2ae1:	addsd  xmm0,xmm2
    2ae5:	jmp    2ab8 <_Z3runI12MultiplyCaseIfLm8EEEvv+0xf8>
    2ae7:	nop    WORD PTR [rax+rax*1+0x0]
    2af0:	add    r10,0x4
    2af4:	cmp    r9,0x8
    2af8:	jne    2a60 <_Z3runI12MultiplyCaseIfLm8EEEvv+0xa0>
    2afe:	mov    rdi,rbx
    2b01:	lea    r13,[rbx+0x100]
    2b08:	lea    r14,[rbx+0x120]
    2b0f:	call   28c0 <_ZN12MultiplyCaseIfLm8EE5applyEPv>
    2b14:	call   28a0 <instruction_count_begin>
    2b19:	mov    rdi,rbx
    2b1c:	call   28c0 <_ZN12MultiplyCaseIfLm8EE5applyEPv>
    2b21:	call   28b0 <instruction_count_end>
    2b26:	lea    rsi,[rip+0x950f]        # c03c <_IO_stdin_used+0x3c>
    2b2d:	mov    edi,0x2
    2b32:	xor    eax,eax
    2b34:	call   10e0 <__printf_chk@plt>
    2b39:	nop    DWORD PTR [rax+0x0]
    2b40:	lea    r12,[r13-0x100]
    2b47:	nop    WORD PTR [rax+rax*1+0x0]
    2b50:	pxor   xmm0,xmm0
    2b54:	lea    rsi,[rip+0x94f2]        # c04d <_IO_stdin_used+0x4d>
    2b5b:	mov    edi,0x2
    2b60:	mov    eax,0x1
    2b65:	cvtss2sd xmm0,DWORD PTR [r12]
    2b6b:	call   10e0 <__printf_chk@plt>
    2b70:	add    r12,0x20
    2b74:	cmp    r12,r13
    2b77:	jne    2b50 <_Z3runI12MultiplyCaseIfLm8EEEvv+0x190>
    2b79:	lea    r13,[r12+0x4]
    2b7e:	cmp    r13,r14
    2b81:	jne    2b40 <_Z3runI12MultiplyCaseIfLm8EEEvv+0x180>
    2b83:	lea    r12,[rbx+0x20]
    2b87:	nop    WORD PTR [rax+rax*1+0x0]
    2b90:	pxor   xmm0,xmm0
    2b94:	lea    rsi,[rip+0x94b2]        # c04d <_IO_stdin_used+0x4d>
    2b9b:	mov    edi,0x2
    2ba0:	mov    eax,0x1
    2ba5:	cvtss2sd xmm0,DWORD PTR [rbx+0x100]
    2bad:	add    rbx,0x4
    2bb1:	call   10e0 <__printf_chk@plt>
    2bb6:	cmp    r12,rbx
    2bb9:	jne    2b90 <_Z3runI12MultiplyCaseIfLm8EEEvv+0x1d0>
    2bbb:	lea    rsi,[rip+0x9480]        # c042 <_IO_stdin_used+0x42>
    2bc2:	mov    edi,0x2
    2bc7:	xor    eax,eax
    2bc9:	call   10e0 <__printf_chk@plt>
    2bce:	xchg   ax,ax
    2bd0:	pxor   xmm0,xmm0
    2bd4:	lea    rsi,[rip+0x9472]        # c04d <_IO_stdin_used+0x4d>
    2bdb:	mov    edi,0x2
    2be0:	mov    eax,0x1
    2be5:	cvtss2sd xmm0,DWORD PTR [rbp+0x120]
    2bed:	add    rbp,0x4
    2bf1:	call   10e0 <__printf_chk@plt>
    2bf6:	cmp    r12,rbp
    2bf9:	jne    2bd0 <_Z3runI12MultiplyCaseIfLm8EEEvv+0x210>
    2bfb:	xor    eax,eax
    2bfd:	lea    rsi,[rip+0x946d]        # c071 <_IO_stdin_used+0x71>
    2c04:	mov    edi,0x2
    2c09:	call   10e0 <__printf_chk@plt>
    2c0e:	mov    rax,QWORD PTR [rsp+0x148]
    2c16:	sub    rax,QWORD PTR fs:0x28
    2c1f:	jne    2c31 <_Z3runI12MultiplyCaseIfLm8EEEvv+0x271>
    2c21:	add    rsp,0x150
    2c28:	pop    rbx
    2c29:	pop    rbp
    2c2a:	pop    r12
    2c2c:	pop    r13
    2c2e:	pop    r14
    2c30:	ret
    2c31:	call   10b0 <__stack_chk_fail@plt>
    2c36:	cs nop WORD PTR [rax+rax*1+0x0]

0000000000002c40 <_ZN12MultiplyCaseIfLm16EE5applyEPv>:
    2c40:	endbr64
    2c44:	mov    r9,QWORD PTR [rdi+0x480]
    2c4b:	pxor   xmm0,xmm0
    2c4f:	movups XMMWORD PTR [rdi+0x440],xmm0
    2c56:	movups XMMWORD PTR [rdi+0x450],xmm0
    2c5d:	movups XMMWORD PTR [rdi+0x460],xmm0
    2c64:	movups XMMWORD PTR [rdi+0x470],xmm0
    2c6b:	cmp    r9,0x10
    2c6f:	je     2d07 <_ZN12MultiplyCaseIfLm16EE5applyEPv+0xc7>
    2c75:	mov    rsi,rdi
    2c78:	lea    rcx,[rdi+0x400]
    2c7f:	xor    edx,edx
    2c81:	nop    DWORD PTR [rax+0x0]
    2c85:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2c90:	xor    eax,eax
    2c92:	xchg   ax,ax
    2c94:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2c9f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2caa:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2cb5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2cc0:	cmp    rdx,rax
    2cc3:	mov    r8,rax
    2cc6:	cmovae r8,rdx
    2cca:	cmp    r8,r9
    2ccd:	jae    2cea <_ZN12MultiplyCaseIfLm16EE5applyEPv+0xaa>
    2ccf:	movss  xmm0,DWORD PTR [rsi+rax*4]
    2cd4:	mulss  xmm0,DWORD PTR [rcx]
    2cd8:	addss  xmm0,DWORD PTR [rdi+rax*4+0x440]
    2ce1:	movss  DWORD PTR [rdi+rax*4+0x440],xmm0
    2cea:	add    rax,0x1
    2cee:	cmp    rax,0x10
    2cf2:	jne    2cc0 <_ZN12MultiplyCaseIfLm16EE5applyEPv+0x80>
    2cf4:	add    rdx,0x1
    2cf8:	add    rsi,0x40
    2cfc:	add    rcx,0x4
    2d00:	cmp    rdx,0x10
    2d04:	jne    2c90 <_ZN12MultiplyCaseIfLm16EE5applyEPv+0x50>
    2d06:	ret
    2d07:	lea    rcx,[rdi+0x400]
    2d0e:	xor    edx,edx
    2d10:	lea    rsi,[rdi+0x480]
    2d17:	nop    WORD PTR [rax+rax*1+0x0]
    2d20:	lea    rax,[rdi+0x440]
    2d27:	nop    DWORD PTR [rax]
    2d2a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2d35:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2d40:	movss  xmm0,DWORD PTR [rax+rdx*1-0x440]
    2d49:	mulss  xmm0,DWORD PTR [rcx]
    2d4d:	add    rax,0x4
    2d51:	addss  xmm0,DWORD PTR [rax-0x4]
    2d56:	movss  DWORD PTR [rax-0x4],xmm0
    2d5b:	cmp    rsi,rax
    2d5e:	jne    2d40 <_ZN12MultiplyCaseIfLm16EE5applyEPv+0x100>
    2d60:	add    rdx,0x40
    2d64:	add    rcx,0x4
    2d68:	cmp    rdx,0x400
    2d6f:	jne    2d20 <_ZN12MultiplyCaseIfLm16EE5applyEPv+0xe0>
    2d71:	ret
    2d72:	cs nop WORD PTR [rax+rax*1+0x0]
    2d7c:	nop    DWORD PTR [rax+0x0]

0000000000002d80 <_Z3runI12MultiplyCaseIfLm16EEEvv>:
    2d80:	endbr64
    2d84:	push   r12
    2d86:	mov    ecx,0x80
    2d8b:	pxor   xmm0,xmm0
    2d8f:	xor    esi,esi
    2d91:	movabs r8,0xcccccccccccccccd
    2d9b:	push   rbp
    2d9c:	push   rbx
    2d9d:	sub    rsp,0x4a0
    2da4:	movsd  xmm4,QWORD PTR [rip+0x953c]        # c2e8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
    2dac:	movsd  xmm1,QWORD PTR [rip+0x952c]        # c2e0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa0>
    2db4:	mov    rax,QWORD PTR fs:0x28
    2dbd:	mov    QWORD PTR [rsp+0x498],rax
    2dc5:	xor    eax,eax
    2dc7:	lea    rdi,[rsp+0x10]
    2dcc:	mov    QWORD PTR [rsp+0x490],0x10
    2dd8:	lea    rbx,[rsp+0x10]
    2ddd:	movsd  xmm3,QWORD PTR [rip+0x950b]        # c2f0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xb0>
    2de5:	movsd  xmm2,QWORD PTR [rip+0x94eb]        # c2d8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x98>
    2ded:	movaps XMMWORD PTR [rsp+0x410],xmm0
    2df5:	mov    r9,rbx
    2df8:	movaps XMMWORD PTR [rsp+0x420],xmm0
    2e00:	movaps XMMWORD PTR [rsp+0x430],xmm0
    2e08:	movaps XMMWORD PTR [rsp+0x440],xmm0
    2e10:	movaps XMMWORD PTR [rsp+0x450],xmm0
    2e18:	rep stos QWORD PTR [rdi],rax
    2e1b:	movaps XMMWORD PTR [rsp+0x460],xmm0
    2e23:	movaps XMMWORD PTR [rsp+0x470],xmm0
    2e2b:	movaps XMMWORD PTR [rsp+0x480],xmm0
    2e33:	xchg   ax,ax
    2e35:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2e40:	mov    r10,rsi
    2e43:	pxor   xmm0,xmm0
    2e47:	add    rsi,0x1
    2e4b:	mov    r11,r9
    2e4e:	cvtsi2sd xmm0,rsi
    2e53:	divsd  xmm0,xmm4
    2e57:	mov    rcx,r10
    2e5a:	xor    edi,edi
    2e5c:	cvtsd2ss xmm0,xmm0
    2e60:	movss  DWORD PTR [r9+0x400],xmm0
    2e69:	jmp    2eb3 <_Z3runI12MultiplyCaseIfLm16EEEvv+0x133>
    2e6b:	nop    DWORD PTR [rax+rax*1+0x0]
    2e70:	mov    rax,rcx
    2e73:	pxor   xmm0,xmm0
    2e77:	mul    r8
    2e7a:	mov    rax,rdx
    2e7d:	and    rdx,0xfffffffffffffffc
    2e81:	shr    rax,0x2
    2e85:	add    rdx,rax
    2e88:	mov    rax,rcx
    2e8b:	sub    rax,rdx
    2e8e:	cvtsi2sd xmm0,rax
    2e93:	mulsd  xmm0,xmm1
    2e97:	add    rdi,0x1
    2e9b:	cvtsd2ss xmm0,xmm0
    2e9f:	add    r11,0x40
    2ea3:	movss  DWORD PTR [r11-0x40],xmm0
    2ea9:	add    rcx,0x1
    2ead:	cmp    rdi,0x10
    2eb1:	je     2ed0 <_Z3runI12MultiplyCaseIfLm16EEEvv+0x150>
    2eb3:	cmp    rdi,r10
    2eb6:	jne    2e70 <_Z3runI12MultiplyCaseIfLm16EEEvv+0xf0>
    2eb8:	pxor   xmm0,xmm0
    2ebc:	cvtsi2sd xmm0,rdi
    2ec1:	mulsd  xmm0,xmm3
    2ec5:	addsd  xmm0,xmm2
    2ec9:	jmp    2e97 <_Z3runI12MultiplyCaseIfLm16EEEvv+0x117>
    2ecb:	nop    DWORD PTR [rax+rax*1+0x0]
    2ed0:	add    r9,0x4
    2ed4:	cmp    rsi,0x10
    2ed8:	jne    2e40 <_Z3runI12MultiplyCaseIfLm16EEEvv+0xc0>
    2ede:	mov    rdi,rbx
    2ee1:	mov    rbp,rbx
    2ee4:	call   2c40 <_ZN12MultiplyCaseIfLm16EE5applyEPv>
    2ee9:	call   28a0 <instruction_count_begin>
    2eee:	mov    rdi,rbx
    2ef1:	call   2c40 <_ZN12MultiplyCaseIfLm16EE5applyEPv>
    2ef6:	call   28b0 <instruction_count_end>
    2efb:	lea    rsi,[rip+0x913a]        # c03c <_IO_stdin_used+0x3c>
    2f02:	mov    edi,0x2
    2f07:	xor    eax,eax
    2f09:	call   10e0 <__printf_chk@plt>
    2f0e:	lea    rdx,[rbx+0x40]
    2f12:	nop    DWORD PTR [rax]
    2f15:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2f20:	mov    r12,rbp
    2f23:	lea    rcx,[rbp+0x400]
    2f2a:	nop    WORD PTR [rax+rax*1+0x0]
    2f30:	pxor   xmm0,xmm0
    2f34:	lea    rsi,[rip+0x9112]        # c04d <_IO_stdin_used+0x4d>
    2f3b:	mov    edi,0x2
    2f40:	mov    eax,0x1
    2f45:	cvtss2sd xmm0,DWORD PTR [r12]
    2f4b:	mov    QWORD PTR [rsp+0x8],rcx
    2f50:	add    r12,0x40
    2f54:	mov    QWORD PTR [rsp],rdx
    2f58:	call   10e0 <__printf_chk@plt>
    2f5d:	mov    rcx,QWORD PTR [rsp+0x8]
    2f62:	mov    rdx,QWORD PTR [rsp]
    2f66:	cmp    r12,rcx
    2f69:	jne    2f30 <_Z3runI12MultiplyCaseIfLm16EEEvv+0x1b0>
    2f6b:	add    rbp,0x4
    2f6f:	cmp    rbp,rdx
    2f72:	jne    2f20 <_Z3runI12MultiplyCaseIfLm16EEEvv+0x1a0>
    2f74:	lea    r12,[rbx+0x400]
    2f7b:	lea    rbp,[rbx+0x440]
    2f82:	nop    DWORD PTR [rax]
    2f85:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2f90:	pxor   xmm0,xmm0
    2f94:	lea    rsi,[rip+0x90b2]        # c04d <_IO_stdin_used+0x4d>
    2f9b:	mov    edi,0x2
    2fa0:	mov    eax,0x1
    2fa5:	cvtss2sd xmm0,DWORD PTR [r12]
    2fab:	call   10e0 <__printf_chk@plt>
    2fb0:	add    r12,0x4
    2fb4:	cmp    r12,rbp
    2fb7:	jne    2f90 <_Z3runI12MultiplyCaseIfLm16EEEvv+0x210>
    2fb9:	lea    rsi,[rip+0x9082]        # c042 <_IO_stdin_used+0x42>
    2fc0:	mov    edi,0x2
    2fc5:	xor    eax,eax
    2fc7:	add    rbx,0x480
    2fce:	call   10e0 <__printf_chk@plt>
    2fd3:	xchg   ax,ax
    2fd5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    2fe0:	pxor   xmm0,xmm0
    2fe4:	lea    rsi,[rip+0x9062]        # c04d <_IO_stdin_used+0x4d>
    2feb:	mov    edi,0x2
    2ff0:	mov    eax,0x1
    2ff5:	cvtss2sd xmm0,DWORD PTR [rbp+0x0]
    2ffa:	call   10e0 <__printf_chk@plt>
    2fff:	add    rbp,0x4
    3003:	cmp    rbp,rbx
    3006:	jne    2fe0 <_Z3runI12MultiplyCaseIfLm16EEEvv+0x260>
    3008:	xor    eax,eax
    300a:	lea    rsi,[rip+0x9060]        # c071 <_IO_stdin_used+0x71>
    3011:	mov    edi,0x2
    3016:	call   10e0 <__printf_chk@plt>
    301b:	mov    rax,QWORD PTR [rsp+0x498]
    3023:	sub    rax,QWORD PTR fs:0x28
    302c:	jne    303a <_Z3runI12MultiplyCaseIfLm16EEEvv+0x2ba>
    302e:	add    rsp,0x4a0
    3035:	pop    rbx
    3036:	pop    rbp
    3037:	pop    r12
    3039:	ret
    303a:	call   10b0 <__stack_chk_fail@plt>
    303f:	nop

0000000000003040 <_ZN12MultiplyCaseIdLm8EE5applyEPv>:
    3040:	endbr64
    3044:	mov    rsi,QWORD PTR [rdi+0x280]
    304b:	pxor   xmm0,xmm0
    304f:	movups XMMWORD PTR [rdi+0x240],xmm0
    3056:	movups XMMWORD PTR [rdi+0x250],xmm0
    305d:	movups XMMWORD PTR [rdi+0x260],xmm0
    3064:	movups XMMWORD PTR [rdi+0x270],xmm0
    306b:	cmp    rsi,0x8
    306f:	je     310d <_ZN12MultiplyCaseIdLm8EE5applyEPv+0xcd>
    3075:	mov    r8,rdi
    3078:	xor    edx,edx
    307a:	nop    WORD PTR [rax+rax*1+0x0]
    3080:	xor    eax,eax
    3082:	nop    DWORD PTR [rax+0x0]
    3089:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    3094:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    309f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    30aa:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    30b5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    30c0:	cmp    rdx,rax
    30c3:	mov    rcx,rax
    30c6:	cmovae rcx,rdx
    30ca:	cmp    rcx,rsi
    30cd:	jae    30f0 <_ZN12MultiplyCaseIdLm8EE5applyEPv+0xb0>
    30cf:	movsd  xmm0,QWORD PTR [r8+rax*8]
    30d5:	mulsd  xmm0,QWORD PTR [rdi+rdx*8+0x200]
    30de:	addsd  xmm0,QWORD PTR [rdi+rax*8+0x240]
    30e7:	movsd  QWORD PTR [rdi+rax*8+0x240],xmm0
    30f0:	add    rax,0x1
    30f4:	cmp    rax,0x8
    30f8:	jne    30c0 <_ZN12MultiplyCaseIdLm8EE5applyEPv+0x80>
    30fa:	add    rdx,0x1
    30fe:	add    r8,0x40
    3102:	cmp    rdx,0x8
    3106:	jne    3080 <_ZN12MultiplyCaseIdLm8EE5applyEPv+0x40>
    310c:	ret
    310d:	lea    rcx,[rdi+0x200]
    3114:	xor    edx,edx
    3116:	lea    rsi,[rdi+0x280]
    311d:	nop    DWORD PTR [rax]
    3120:	lea    rax,[rdi+0x240]
    3127:	nop    DWORD PTR [rax]
    312a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    3135:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    3140:	movsd  xmm0,QWORD PTR [rax+rdx*1-0x240]
    3149:	mulsd  xmm0,QWORD PTR [rcx]
    314d:	add    rax,0x8
    3151:	addsd  xmm0,QWORD PTR [rax-0x8]
    3156:	movsd  QWORD PTR [rax-0x8],xmm0
    315b:	cmp    rsi,rax
    315e:	jne    3140 <_ZN12MultiplyCaseIdLm8EE5applyEPv+0x100>
    3160:	add    rdx,0x40
    3164:	add    rcx,0x8
    3168:	cmp    rdx,0x200
    316f:	jne    3120 <_ZN12MultiplyCaseIdLm8EE5applyEPv+0xe0>
    3171:	ret
    3172:	cs nop WORD PTR [rax+rax*1+0x0]
    317c:	nop    DWORD PTR [rax+0x0]

0000000000003180 <_Z3runI12MultiplyCaseIdLm8EEEvv>:
    3180:	endbr64
    3184:	push   r14
    3186:	mov    ecx,0x40
    318b:	pxor   xmm0,xmm0
    318f:	xor    r9d,r9d
    3192:	movabs r8,0xcccccccccccccccd
    319c:	push   r13
    319e:	push   r12
    31a0:	push   rbp
    31a1:	push   rbx
    31a2:	sub    rsp,0x290
    31a9:	movsd  xmm4,QWORD PTR [rip+0x9117]        # c2c8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x88>
    31b1:	movsd  xmm1,QWORD PTR [rip+0x9127]        # c2e0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa0>
    31b9:	mov    rax,QWORD PTR fs:0x28
    31c2:	mov    QWORD PTR [rsp+0x288],rax
    31ca:	xor    eax,eax
    31cc:	mov    rdi,rsp
    31cf:	movsd  xmm3,QWORD PTR [rip+0x90f9]        # c2d0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x90>
    31d7:	mov    rbx,rsp
    31da:	movaps XMMWORD PTR [rsp+0x200],xmm0
    31e2:	mov    rbp,rsp
    31e5:	mov    r10,rsp
    31e8:	movsd  xmm2,QWORD PTR [rip+0x90e8]        # c2d8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x98>
    31f0:	mov    QWORD PTR [rsp+0x280],0x8
    31fc:	movaps XMMWORD PTR [rsp+0x210],xmm0
    3204:	movaps XMMWORD PTR [rsp+0x220],xmm0
    320c:	movaps XMMWORD PTR [rsp+0x230],xmm0
    3214:	rep stos QWORD PTR [rdi],rax
    3217:	movaps XMMWORD PTR [rsp+0x240],xmm0
    321f:	movaps XMMWORD PTR [rsp+0x250],xmm0
    3227:	movaps XMMWORD PTR [rsp+0x260],xmm0
    322f:	movaps XMMWORD PTR [rsp+0x270],xmm0
    3237:	nop    WORD PTR [rax+rax*1+0x0]
    3240:	mov    rdi,r9
    3243:	pxor   xmm0,xmm0
    3247:	add    r9,0x1
    324b:	mov    rsi,r10
    324e:	cvtsi2sd xmm0,r9
    3253:	divsd  xmm0,xmm4
    3257:	xor    ecx,ecx
    3259:	movsd  QWORD PTR [r10+0x200],xmm0
    3262:	jmp    32ba <_Z3runI12MultiplyCaseIdLm8EEEvv+0x13a>
    3264:	nop    WORD PTR [rax+rax*1+0x0]
    326a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    3275:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    3280:	lea    r11,[rcx+rdi*1]
    3284:	pxor   xmm0,xmm0
    3288:	mov    rax,r11
    328b:	mul    r8
    328e:	mov    rax,rdx
    3291:	and    rdx,0xfffffffffffffffc
    3295:	shr    rax,0x2
    3299:	add    rdx,rax
    329c:	sub    r11,rdx
    329f:	cvtsi2sd xmm0,r11
    32a4:	mulsd  xmm0,xmm1
    32a8:	add    rcx,0x1
    32ac:	movsd  QWORD PTR [rsi],xmm0
    32b0:	add    rsi,0x40
    32b4:	cmp    rcx,0x8
    32b8:	je     32d8 <_Z3runI12MultiplyCaseIdLm8EEEvv+0x158>
    32ba:	cmp    rcx,rdi
    32bd:	jne    3280 <_Z3runI12MultiplyCaseIdLm8EEEvv+0x100>
    32bf:	pxor   xmm0,xmm0
    32c3:	cvtsi2sd xmm0,rcx
    32c8:	mulsd  xmm0,xmm3
    32cc:	addsd  xmm0,xmm2
    32d0:	jmp    32a8 <_Z3runI12MultiplyCaseIdLm8EEEvv+0x128>
    32d2:	nop    WORD PTR [rax+rax*1+0x0]
    32d8:	add    r10,0x8
    32dc:	cmp    r9,0x8
    32e0:	jne    3240 <_Z3runI12MultiplyCaseIdLm8EEEvv+0xc0>
    32e6:	mov    rdi,rbx
    32e9:	lea    r13,[rbx+0x200]
    32f0:	lea    r14,[rbx+0x240]
    32f7:	call   3040 <_ZN12MultiplyCaseIdLm8EE5applyEPv>
    32fc:	call   28a0 <instruction_count_begin>
    3301:	mov    rdi,rbx
    3304:	call   3040 <_ZN12MultiplyCaseIdLm8EE5applyEPv>
    3309:	call   28b0 <instruction_count_end>
    330e:	lea    rsi,[rip+0x8d27]        # c03c <_IO_stdin_used+0x3c>
    3315:	mov    edi,0x2
    331a:	xor    eax,eax
    331c:	call   10e0 <__printf_chk@plt>
    3321:	nop    DWORD PTR [rax+0x0]
    3325:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    3330:	lea    r12,[r13-0x200]
    3337:	nop    WORD PTR [rax+rax*1+0x0]
    3340:	movsd  xmm0,QWORD PTR [r12]
    3346:	lea    rsi,[rip+0x8d00]        # c04d <_IO_stdin_used+0x4d>
    334d:	mov    edi,0x2
    3352:	mov    eax,0x1
    3357:	add    r12,0x40
    335b:	call   10e0 <__printf_chk@plt>
    3360:	cmp    r12,r13
    3363:	jne    3340 <_Z3runI12MultiplyCaseIdLm8EEEvv+0x1c0>
    3365:	lea    r13,[r12+0x8]
    336a:	cmp    r13,r14
    336d:	jne    3330 <_Z3runI12MultiplyCaseIdLm8EEEvv+0x1b0>
    336f:	lea    r12,[rbx+0x40]
    3373:	xchg   ax,ax
    3375:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    3380:	lea    rsi,[rip+0x8cc6]        # c04d <_IO_stdin_used+0x4d>
    3387:	mov    edi,0x2
    338c:	mov    eax,0x1
    3391:	movsd  xmm0,QWORD PTR [rbx+0x200]
    3399:	add    rbx,0x8
    339d:	call   10e0 <__printf_chk@plt>
    33a2:	cmp    r12,rbx
    33a5:	jne    3380 <_Z3runI12MultiplyCaseIdLm8EEEvv+0x200>
    33a7:	lea    rsi,[rip+0x8c94]        # c042 <_IO_stdin_used+0x42>
    33ae:	mov    edi,0x2
    33b3:	xor    eax,eax
    33b5:	call   10e0 <__printf_chk@plt>
    33ba:	nop    WORD PTR [rax+rax*1+0x0]
    33c0:	lea    rsi,[rip+0x8c86]        # c04d <_IO_stdin_used+0x4d>
    33c7:	mov    edi,0x2
    33cc:	mov    eax,0x1
    33d1:	movsd  xmm0,QWORD PTR [rbp+0x240]
    33d9:	add    rbp,0x8
    33dd:	call   10e0 <__printf_chk@plt>
    33e2:	cmp    r12,rbp
    33e5:	jne    33c0 <_Z3runI12MultiplyCaseIdLm8EEEvv+0x240>
    33e7:	xor    eax,eax
    33e9:	lea    rsi,[rip+0x8c81]        # c071 <_IO_stdin_used+0x71>
    33f0:	mov    edi,0x2
    33f5:	call   10e0 <__printf_chk@plt>
    33fa:	mov    rax,QWORD PTR [rsp+0x288]
    3402:	sub    rax,QWORD PTR fs:0x28
    340b:	jne    341d <_Z3runI12MultiplyCaseIdLm8EEEvv+0x29d>
    340d:	add    rsp,0x290
    3414:	pop    rbx
    3415:	pop    rbp
    3416:	pop    r12
    3418:	pop    r13
    341a:	pop    r14
    341c:	ret
    341d:	call   10b0 <__stack_chk_fail@plt>
    3422:	cs nop WORD PTR [rax+rax*1+0x0]
    342c:	cs nop WORD PTR [rax+rax*1+0x0]
    3436:	cs nop WORD PTR [rax+rax*1+0x0]

0000000000003440 <_ZN12MultiplyCaseIdLm16EE5applyEPv>:
    3440:	endbr64
    3444:	mov    QWORD PTR [rdi+0x880],0x0
    344f:	mov    r9,QWORD PTR [rdi+0x900]
    3456:	mov    rdx,rdi
    3459:	xor    eax,eax
    345b:	mov    QWORD PTR [rdi+0x8f8],0x0
    3466:	lea    r8,[rdi+0x880]
    346d:	lea    rdi,[rdi+0x888]
    3474:	and    rdi,0xfffffffffffffff8
    3478:	mov    rcx,r8
    347b:	sub    rcx,rdi
    347e:	sub    ecx,0xffffff80
    3481:	shr    ecx,0x3
    3484:	rep stos QWORD PTR [rdi],rax
    3487:	cmp    r9,0x10
    348b:	je     3507 <_ZN12MultiplyCaseIdLm16EE5applyEPv+0xc7>
    348d:	mov    rdi,rdx
    3490:	lea    rsi,[rdx+0x800]
    3497:	nop    WORD PTR [rax+rax*1+0x0]
    34a0:	xor    eax,eax
    34a2:	nop    DWORD PTR [rax+rax*1+0x0]
    34aa:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    34b5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    34c0:	cmp    rcx,rax
    34c3:	mov    r8,rax
    34c6:	cmovae r8,rcx
    34ca:	cmp    r8,r9
    34cd:	jae    34ea <_ZN12MultiplyCaseIdLm16EE5applyEPv+0xaa>
    34cf:	movsd  xmm0,QWORD PTR [rdi+rax*8]
    34d4:	mulsd  xmm0,QWORD PTR [rsi]
    34d8:	addsd  xmm0,QWORD PTR [rdx+rax*8+0x880]
    34e1:	movsd  QWORD PTR [rdx+rax*8+0x880],xmm0
    34ea:	add    rax,0x1
    34ee:	cmp    rax,0x10
    34f2:	jne    34c0 <_ZN12MultiplyCaseIdLm16EE5applyEPv+0x80>
    34f4:	add    rcx,0x1
    34f8:	sub    rdi,0xffffffffffffff80
    34fc:	add    rsi,0x8
    3500:	cmp    rcx,0x10
    3504:	jne    34a0 <_ZN12MultiplyCaseIdLm16EE5applyEPv+0x60>
    3506:	ret
    3507:	lea    rsi,[rdx+0x800]
    350e:	xor    ecx,ecx
    3510:	add    rdx,0x900
    3517:	nop    WORD PTR [rax+rax*1+0x0]
    3520:	mov    rax,r8
    3523:	nop    DWORD PTR [rax+0x0]
    352a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    3535:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    3540:	movsd  xmm0,QWORD PTR [rax+rcx*1-0x880]
    3549:	mulsd  xmm0,QWORD PTR [rsi]
    354d:	add    rax,0x8
    3551:	addsd  xmm0,QWORD PTR [rax-0x8]
    3556:	movsd  QWORD PTR [rax-0x8],xmm0
    355b:	cmp    rdx,rax
    355e:	jne    3540 <_ZN12MultiplyCaseIdLm16EE5applyEPv+0x100>
    3560:	sub    rcx,0xffffffffffffff80
    3564:	add    rsi,0x8
    3568:	cmp    rcx,0x800
    356f:	jne    3520 <_ZN12MultiplyCaseIdLm16EE5applyEPv+0xe0>
    3571:	ret
    3572:	cs nop WORD PTR [rax+rax*1+0x0]
    357c:	nop    DWORD PTR [rax+0x0]

0000000000003580 <_Z3runI12MultiplyCaseIdLm16EEEvv>:
    3580:	endbr64
    3584:	push   r12
    3586:	mov    ecx,0x100
    358b:	xor    esi,esi
    358d:	movabs r8,0xcccccccccccccccd
    3597:	push   rbp
    3598:	push   rbx
    3599:	sub    rsp,0x920
    35a0:	movsd  xmm4,QWORD PTR [rip+0x8d40]        # c2e8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
    35a8:	movsd  xmm1,QWORD PTR [rip+0x8d30]        # c2e0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa0>
    35b0:	mov    rax,QWORD PTR fs:0x28
    35b9:	mov    QWORD PTR [rsp+0x918],rax
    35c1:	xor    eax,eax
    35c3:	lea    rdi,[rsp+0x10]
    35c8:	mov    QWORD PTR [rsp+0x910],0x10
    35d4:	lea    rbx,[rsp+0x10]
    35d9:	movsd  xmm3,QWORD PTR [rip+0x8d0f]        # c2f0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xb0>
    35e1:	movsd  xmm2,QWORD PTR [rip+0x8cef]        # c2d8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x98>
    35e9:	mov    r9,rbx
    35ec:	rep stos QWORD PTR [rdi],rax
    35ef:	lea    rdi,[rsp+0x810]
    35f7:	mov    ecx,0x10
    35fc:	rep stos QWORD PTR [rdi],rax
    35ff:	lea    rdi,[rsp+0x890]
    3607:	mov    ecx,0x10
    360c:	rep stos QWORD PTR [rdi],rax
    360f:	nop
    3610:	mov    r10,rsi
    3613:	pxor   xmm0,xmm0
    3617:	add    rsi,0x1
    361b:	mov    r11,r9
    361e:	cvtsi2sd xmm0,rsi
    3623:	divsd  xmm0,xmm4
    3627:	mov    rcx,r10
    362a:	xor    edi,edi
    362c:	movsd  QWORD PTR [r9+0x800],xmm0
    3635:	jmp    367e <_Z3runI12MultiplyCaseIdLm16EEEvv+0xfe>
    3637:	nop    WORD PTR [rax+rax*1+0x0]
    3640:	mov    rax,rcx
    3643:	pxor   xmm0,xmm0
    3647:	mul    r8
    364a:	mov    rax,rdx
    364d:	and    rdx,0xfffffffffffffffc
    3651:	shr    rax,0x2
    3655:	add    rdx,rax
    3658:	mov    rax,rcx
    365b:	sub    rax,rdx
    365e:	cvtsi2sd xmm0,rax
    3663:	mulsd  xmm0,xmm1
    3667:	add    rdi,0x1
    366b:	movsd  QWORD PTR [r11],xmm0
    3670:	add    rcx,0x1
    3674:	sub    r11,0xffffffffffffff80
    3678:	cmp    rdi,0x10
    367c:	je     36a0 <_Z3runI12MultiplyCaseIdLm16EEEvv+0x120>
    367e:	cmp    rdi,r10
    3681:	jne    3640 <_Z3runI12MultiplyCaseIdLm16EEEvv+0xc0>
    3683:	pxor   xmm0,xmm0
    3687:	cvtsi2sd xmm0,rdi
    368c:	mulsd  xmm0,xmm3
    3690:	addsd  xmm0,xmm2
    3694:	jmp    3667 <_Z3runI12MultiplyCaseIdLm16EEEvv+0xe7>
    3696:	cs nop WORD PTR [rax+rax*1+0x0]
    36a0:	add    r9,0x8
    36a4:	cmp    rsi,0x10
    36a8:	jne    3610 <_Z3runI12MultiplyCaseIdLm16EEEvv+0x90>
    36ae:	mov    rdi,rbx
    36b1:	mov    rbp,rbx
    36b4:	call   3440 <_ZN12MultiplyCaseIdLm16EE5applyEPv>
    36b9:	call   28a0 <instruction_count_begin>
    36be:	mov    rdi,rbx
    36c1:	call   3440 <_ZN12MultiplyCaseIdLm16EE5applyEPv>
    36c6:	call   28b0 <instruction_count_end>
    36cb:	lea    rsi,[rip+0x896a]        # c03c <_IO_stdin_used+0x3c>
    36d2:	mov    edi,0x2
    36d7:	xor    eax,eax
    36d9:	call   10e0 <__printf_chk@plt>
    36de:	lea    rdx,[rbx+0x80]
    36e5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    36f0:	mov    r12,rbp
    36f3:	lea    rcx,[rbp+0x800]
    36fa:	nop    WORD PTR [rax+rax*1+0x0]
    3700:	movsd  xmm0,QWORD PTR [r12]
    3706:	lea    rsi,[rip+0x8940]        # c04d <_IO_stdin_used+0x4d>
    370d:	mov    edi,0x2
    3712:	mov    eax,0x1
    3717:	mov    QWORD PTR [rsp+0x8],rcx
    371c:	sub    r12,0xffffffffffffff80
    3720:	mov    QWORD PTR [rsp],rdx
    3724:	call   10e0 <__printf_chk@plt>
    3729:	mov    rcx,QWORD PTR [rsp+0x8]
    372e:	mov    rdx,QWORD PTR [rsp]
    3732:	cmp    r12,rcx
    3735:	jne    3700 <_Z3runI12MultiplyCaseIdLm16EEEvv+0x180>
    3737:	add    rbp,0x8
    373b:	cmp    rbp,rdx
    373e:	jne    36f0 <_Z3runI12MultiplyCaseIdLm16EEEvv+0x170>
    3740:	lea    r12,[rbx+0x800]
    3747:	lea    rbp,[rbx+0x880]
    374e:	xchg   ax,ax
    3750:	movsd  xmm0,QWORD PTR [r12]
    3756:	lea    rsi,[rip+0x88f0]        # c04d <_IO_stdin_used+0x4d>
    375d:	mov    edi,0x2
    3762:	mov    eax,0x1
    3767:	add    r12,0x8
    376b:	call   10e0 <__printf_chk@plt>
    3770:	cmp    r12,rbp
    3773:	jne    3750 <_Z3runI12MultiplyCaseIdLm16EEEvv+0x1d0>
    3775:	lea    rsi,[rip+0x88c6]        # c042 <_IO_stdin_used+0x42>
    377c:	mov    edi,0x2
    3781:	xor    eax,eax
    3783:	add    rbx,0x900
    378a:	call   10e0 <__printf_chk@plt>
    378f:	nop
    3790:	movsd  xmm0,QWORD PTR [rbp+0x0]
    3795:	lea    rsi,[rip+0x88b1]        # c04d <_IO_stdin_used+0x4d>
    379c:	mov    edi,0x2
    37a1:	mov    eax,0x1
    37a6:	add    rbp,0x8
    37aa:	call   10e0 <__printf_chk@plt>
    37af:	cmp    rbp,rbx
    37b2:	jne    3790 <_Z3runI12MultiplyCaseIdLm16EEEvv+0x210>
    37b4:	xor    eax,eax
    37b6:	lea    rsi,[rip+0x88b4]        # c071 <_IO_stdin_used+0x71>
    37bd:	mov    edi,0x2
    37c2:	call   10e0 <__printf_chk@plt>
    37c7:	mov    rax,QWORD PTR [rsp+0x918]
    37cf:	sub    rax,QWORD PTR fs:0x28
    37d8:	jne    37e6 <_Z3runI12MultiplyCaseIdLm16EEEvv+0x266>
    37da:	add    rsp,0x920
    37e1:	pop    rbx
    37e2:	pop    rbp
    37e3:	pop    r12
    37e5:	ret
    37e6:	call   10b0 <__stack_chk_fail@plt>
    37eb:	nop    DWORD PTR [rax+rax*1+0x0]

00000000000037f0 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv>:
    37f0:	endbr64
    37f4:	mov    r8,QWORD PTR [rdi+0x280]
    37fb:	pxor   xmm0,xmm0
    37ff:	movups XMMWORD PTR [rdi+0x240],xmm0
    3806:	movups XMMWORD PTR [rdi+0x250],xmm0
    380d:	movups XMMWORD PTR [rdi+0x260],xmm0
    3814:	movups XMMWORD PTR [rdi+0x270],xmm0
    381b:	cmp    r8,0x8
    381f:	je     38df <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0xef>
    3825:	mov    r10,rdi
    3828:	lea    r9,[rdi+0x200]
    382f:	xor    ecx,ecx
    3831:	nop    DWORD PTR [rax+0x0]
    3835:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    3840:	mov    rdx,r10
    3843:	xor    eax,eax
    3845:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    3850:	cmp    rcx,rax
    3853:	mov    rsi,rax
    3856:	cmovae rsi,rcx
    385a:	cmp    rsi,r8
    385d:	jae    38ba <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0xca>
    385f:	movq   xmm1,QWORD PTR [rdx]
    3863:	movq   xmm0,QWORD PTR [r9]
    3868:	movaps xmm2,xmm1
    386b:	movq   xmm3,xmm0
    386f:	shufps xmm0,xmm0,0xe1
    3873:	shufps xmm1,xmm1,0xe5
    3877:	shufps xmm2,xmm2,0xe0
    387b:	movq   xmm2,xmm2
    387f:	mulps  xmm2,xmm3
    3882:	movq   xmm3,xmm0
    3886:	movq   xmm0,xmm1
    388a:	mulps  xmm0,xmm3
    388d:	movq   xmm1,xmm2
    3891:	movq   xmm0,xmm0
    3895:	movaps xmm2,xmm1
    3898:	subps  xmm2,xmm0
    389b:	addps  xmm0,xmm1
    389e:	movq   xmm1,QWORD PTR [rdi+rax*8+0x240]
    38a7:	movss  xmm0,xmm2
    38ab:	movq   xmm0,xmm0
    38af:	addps  xmm0,xmm1
    38b2:	movlps QWORD PTR [rdi+rax*8+0x240],xmm0
    38ba:	add    rax,0x1
    38be:	add    rdx,0x8
    38c2:	cmp    rax,0x8
    38c6:	jne    3850 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x60>
    38c8:	add    rcx,0x1
    38cc:	add    r10,0x40
    38d0:	add    r9,0x8
    38d4:	cmp    rcx,0x8
    38d8:	jne    3840 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x50>
    38de:	ret
    38df:	lea    rcx,[rdi+0x200]
    38e6:	xor    r8d,r8d
    38e9:	lea    r9,[rdi+0x240]
    38f0:	lea    rsi,[rdi+0x280]
    38f7:	nop    WORD PTR [rax+rax*1+0x0]
    3900:	mov    rax,r9
    3903:	mov    rdx,rdi
    3906:	cs nop WORD PTR [rax+rax*1+0x0]
    3910:	movq   xmm1,QWORD PTR [rdx]
    3914:	movq   xmm2,QWORD PTR [rcx]
    3918:	add    rax,0x8
    391c:	add    rdx,0x8
    3920:	movaps xmm0,xmm1
    3923:	movq   xmm3,xmm2
    3927:	shufps xmm2,xmm2,0xe1
    392b:	shufps xmm1,xmm1,0xe5
    392f:	shufps xmm0,xmm0,0xe0
    3933:	movq   xmm0,xmm0
    3937:	movq   xmm2,xmm2
    393b:	movq   xmm1,xmm1
    393f:	mulps  xmm0,xmm3
    3942:	mulps  xmm1,xmm2
    3945:	movq   xmm0,xmm0
    3949:	movq   xmm1,xmm1
    394d:	movaps xmm2,xmm0
    3950:	subps  xmm2,xmm1
    3953:	addps  xmm0,xmm1
    3956:	movq   xmm1,QWORD PTR [rax-0x8]
    395b:	movss  xmm0,xmm2
    395f:	movq   xmm0,xmm0
    3963:	addps  xmm0,xmm1
    3966:	movlps QWORD PTR [rax-0x8],xmm0
    396a:	cmp    rax,rsi
    396d:	jne    3910 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x120>
    396f:	add    r8,0x1
    3973:	add    rdi,0x40
    3977:	add    rcx,0x8
    397b:	cmp    r8,0x8
    397f:	jne    3900 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x110>
    3985:	ret
    3986:	cs nop WORD PTR [rax+rax*1+0x0]
    3990:	cs nop WORD PTR [rax+rax*1+0x0]
    399a:	nop    WORD PTR [rax+rax*1+0x0]

00000000000039a0 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EEEvv>:
    39a0:	endbr64
    39a4:	push   r15
    39a6:	pxor   xmm0,xmm0
    39aa:	push   r14
    39ac:	push   r13
    39ae:	push   r12
    39b0:	push   rbp
    39b1:	push   rbx
    39b2:	sub    rsp,0x298
    39b9:	mov    rbp,QWORD PTR fs:0x28
    39c2:	mov    QWORD PTR [rsp+0x288],rbp
    39ca:	mov    rbp,rsp
    39cd:	lea    rbx,[rsp+0x200]
    39d5:	mov    r12,rbp
    39d8:	mov    rax,rbp
    39db:	nop    DWORD PTR [rax+rax*1+0x0]
    39e0:	movaps XMMWORD PTR [rax],xmm0
    39e3:	add    rax,0x40
    39e7:	movaps XMMWORD PTR [rax-0x30],xmm0
    39eb:	movaps XMMWORD PTR [rax-0x20],xmm0
    39ef:	movaps XMMWORD PTR [rax-0x10],xmm0
    39f3:	cmp    rbx,rax
    39f6:	jne    39e0 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EEEvv+0x40>
    39f8:	mov    QWORD PTR [rsp+0x280],0x8
    3a04:	movsd  xmm7,QWORD PTR [rip+0x88ec]        # c2f8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xb8>
    3a0c:	mov    r13,rbp
    3a0f:	xor    r10d,r10d
    3a12:	movsd  xmm6,QWORD PTR [rip+0x88ae]        # c2c8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x88>
    3a1a:	movsd  xmm2,QWORD PTR [rip+0x88de]        # c300 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xc0>
    3a22:	movaps XMMWORD PTR [rsp+0x200],xmm0
    3a2a:	movabs r14,0xaaaaaaaaaaaaaaab
    3a34:	movsd  xmm3,QWORD PTR [rip+0x88a4]        # c2e0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa0>
    3a3c:	movsd  xmm5,QWORD PTR [rip+0x888c]        # c2d0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x90>
    3a44:	movaps XMMWORD PTR [rsp+0x210],xmm0
    3a4c:	movabs r9,0x4924924924924925
    3a56:	movsd  xmm4,QWORD PTR [rip+0x887a]        # c2d8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x98>
    3a5e:	movaps XMMWORD PTR [rsp+0x220],xmm0
    3a66:	movabs r11,0xcccccccccccccccd
    3a70:	movaps XMMWORD PTR [rsp+0x230],xmm0
    3a78:	movaps XMMWORD PTR [rsp+0x240],xmm0
    3a80:	movaps XMMWORD PTR [rsp+0x250],xmm0
    3a88:	movaps XMMWORD PTR [rsp+0x260],xmm0
    3a90:	movaps XMMWORD PTR [rsp+0x270],xmm0
    3a98:	nop    DWORD PTR [rax+rax*1+0x0]
    3aa0:	mov    rax,r10
    3aa3:	mov    r15,r10
    3aa6:	pxor   xmm0,xmm0
    3aaa:	mov    r8,r13
    3aad:	mul    r14
    3ab0:	pxor   xmm1,xmm1
    3ab4:	mov    rsi,r15
    3ab7:	xor    ecx,ecx
    3ab9:	mov    rax,rdx
    3abc:	and    rdx,0xfffffffffffffffe
    3ac0:	shr    rax,1
    3ac3:	add    rdx,rax
    3ac6:	mov    rax,r10
    3ac9:	add    r10,0x1
    3acd:	cvtsi2sd xmm0,r10
    3ad2:	divsd  xmm0,xmm6
    3ad6:	sub    rax,rdx
    3ad9:	add    rax,0x1
    3add:	cvtsi2sd xmm1,rax
    3ae2:	mulsd  xmm1,xmm7
    3ae6:	unpcklpd xmm0,xmm1
    3aea:	cvtpd2ps xmm0,xmm0
    3aee:	movlps QWORD PTR [r13+0x200],xmm0
    3af6:	jmp    3b47 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EEEvv+0x1a7>
    3af8:	nop    DWORD PTR [rax+rax*1+0x0]
    3b00:	lea    rdi,[rcx+r15*1]
    3b04:	pxor   xmm0,xmm0
    3b08:	mov    rax,rdi
    3b0b:	mul    r11
    3b0e:	mov    rax,rdx
    3b11:	and    rdx,0xfffffffffffffffc
    3b15:	shr    rax,0x2
    3b19:	add    rdx,rax
    3b1c:	sub    rdi,rdx
    3b1f:	cvtsi2sd xmm0,rdi
    3b24:	mulsd  xmm0,xmm3
    3b28:	unpcklpd xmm0,xmm1
    3b2c:	add    rcx,0x1
    3b30:	add    rsi,0x2
    3b34:	add    r8,0x40
    3b38:	cvtpd2ps xmm0,xmm0
    3b3c:	movlps QWORD PTR [r8-0x40],xmm0
    3b41:	cmp    rcx,0x8
    3b45:	je     3b90 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EEEvv+0x1f0>
    3b47:	mov    rax,rsi
    3b4a:	pxor   xmm1,xmm1
    3b4e:	imul   r9
    3b51:	mov    rax,rsi
    3b54:	sar    rax,0x3f
    3b58:	sar    rdx,1
    3b5b:	sub    rdx,rax
    3b5e:	lea    rax,[rdx*8+0x0]
    3b66:	sub    rax,rdx
    3b69:	mov    rdx,rsi
    3b6c:	sub    rdx,rax
    3b6f:	cvtsi2sd xmm1,rdx
    3b74:	mulsd  xmm1,xmm2
    3b78:	cmp    rcx,r15
    3b7b:	jne    3b00 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EEEvv+0x160>
    3b7d:	pxor   xmm0,xmm0
    3b81:	cvtsi2sd xmm0,rcx
    3b86:	mulsd  xmm0,xmm5
    3b8a:	addsd  xmm0,xmm4
    3b8e:	jmp    3b28 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EEEvv+0x188>
    3b90:	add    r13,0x8
    3b94:	cmp    r10,0x8
    3b98:	jne    3aa0 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EEEvv+0x100>
    3b9e:	mov    rdi,rbp
    3ba1:	xor    r14d,r14d
    3ba4:	call   37f0 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv>
    3ba9:	call   28a0 <instruction_count_begin>
    3bae:	mov    rdi,rbp
    3bb1:	call   37f0 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv>
    3bb6:	call   28b0 <instruction_count_end>
    3bbb:	lea    rsi,[rip+0x847a]        # c03c <_IO_stdin_used+0x3c>
    3bc2:	mov    edi,0x2
    3bc7:	xor    eax,eax
    3bc9:	call   10e0 <__printf_chk@plt>
    3bce:	xchg   ax,ax
    3bd0:	lea    r13,[rbx-0x200]
    3bd7:	nop    WORD PTR [rax+rax*1+0x0]
    3be0:	pxor   xmm0,xmm0
    3be4:	pxor   xmm1,xmm1
    3be8:	mov    edi,0x2
    3bed:	mov    eax,0x2
    3bf2:	cvtss2sd xmm0,DWORD PTR [r13+0x0]
    3bf8:	cvtss2sd xmm1,DWORD PTR [r13+0x4]
    3bfe:	lea    rsi,[rip+0x8445]        # c04a <_IO_stdin_used+0x4a>
    3c05:	add    r13,0x40
    3c09:	call   10e0 <__printf_chk@plt>
    3c0e:	cmp    r13,rbx
    3c11:	jne    3be0 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EEEvv+0x240>
    3c13:	add    r14,0x1
    3c17:	lea    rbx,[r13+0x8]
    3c1b:	cmp    r14,0x8
    3c1f:	jne    3bd0 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EEEvv+0x230>
    3c21:	lea    rbx,[rbp+0x40]
    3c25:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    3c30:	pxor   xmm0,xmm0
    3c34:	pxor   xmm1,xmm1
    3c38:	mov    edi,0x2
    3c3d:	mov    eax,0x2
    3c42:	cvtss2sd xmm0,DWORD PTR [rbp+0x200]
    3c4a:	cvtss2sd xmm1,DWORD PTR [rbp+0x204]
    3c52:	lea    rsi,[rip+0x83f1]        # c04a <_IO_stdin_used+0x4a>
    3c59:	add    rbp,0x8
    3c5d:	call   10e0 <__printf_chk@plt>
    3c62:	cmp    rbx,rbp
    3c65:	jne    3c30 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EEEvv+0x290>
    3c67:	lea    rsi,[rip+0x83d4]        # c042 <_IO_stdin_used+0x42>
    3c6e:	mov    edi,0x2
    3c73:	xor    eax,eax
    3c75:	call   10e0 <__printf_chk@plt>
    3c7a:	nop    WORD PTR [rax+rax*1+0x0]
    3c80:	pxor   xmm0,xmm0
    3c84:	pxor   xmm1,xmm1
    3c88:	mov    edi,0x2
    3c8d:	mov    eax,0x2
    3c92:	cvtss2sd xmm0,DWORD PTR [r12+0x240]
    3c9c:	lea    rsi,[rip+0x83a7]        # c04a <_IO_stdin_used+0x4a>
    3ca3:	add    r12,0x8
    3ca7:	cvtss2sd xmm1,DWORD PTR [r12+0x23c]
    3cb1:	call   10e0 <__printf_chk@plt>
    3cb6:	cmp    rbx,r12
    3cb9:	jne    3c80 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EEEvv+0x2e0>
    3cbb:	xor    eax,eax
    3cbd:	lea    rsi,[rip+0x83ad]        # c071 <_IO_stdin_used+0x71>
    3cc4:	mov    edi,0x2
    3cc9:	call   10e0 <__printf_chk@plt>
    3cce:	mov    rax,QWORD PTR [rsp+0x288]
    3cd6:	sub    rax,QWORD PTR fs:0x28
    3cdf:	jne    3cf3 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm8EEEvv+0x353>
    3ce1:	add    rsp,0x298
    3ce8:	pop    rbx
    3ce9:	pop    rbp
    3cea:	pop    r12
    3cec:	pop    r13
    3cee:	pop    r14
    3cf0:	pop    r15
    3cf2:	ret
    3cf3:	call   10b0 <__stack_chk_fail@plt>
    3cf8:	nop    DWORD PTR [rax+rax*1+0x0]

0000000000003d00 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv>:
    3d00:	endbr64
    3d04:	mov    r9,QWORD PTR [rdi+0x900]
    3d0b:	mov    r11,rdi
    3d0e:	mov    r10,rdi
    3d11:	lea    rdx,[rdi+0x900]
    3d18:	lea    rax,[rdi+0x880]
    3d1f:	pxor   xmm0,xmm0
    3d23:	movups XMMWORD PTR [rax],xmm0
    3d26:	add    rax,0x20
    3d2a:	movups XMMWORD PTR [rax-0x10],xmm0
    3d2e:	cmp    rax,rdx
    3d31:	jne    3d23 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x23>
    3d33:	cmp    r9,0x10
    3d37:	je     3de8 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0xe8>
    3d3d:	lea    rdi,[r11+0x800]
    3d44:	xor    esi,esi
    3d46:	cs nop WORD PTR [rax+rax*1+0x0]
    3d50:	lea    rax,[r11+0x880]
    3d57:	mov    rcx,r10
    3d5a:	xor    edx,edx
    3d5c:	nop    DWORD PTR [rax+0x0]
    3d60:	cmp    rsi,rdx
    3d63:	mov    r8,rdx
    3d66:	cmovae r8,rsi
    3d6a:	cmp    r8,r9
    3d6d:	jae    3dbf <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0xbf>
    3d6f:	movq   xmm0,QWORD PTR [rcx]
    3d73:	movq   xmm1,QWORD PTR [rdi]
    3d77:	movaps xmm2,xmm0
    3d7a:	movq   xmm3,xmm1
    3d7e:	shufps xmm1,xmm1,0xe1
    3d82:	shufps xmm0,xmm0,0xe5
    3d86:	shufps xmm2,xmm2,0xe0
    3d8a:	movq   xmm2,xmm2
    3d8e:	mulps  xmm2,xmm3
    3d91:	movq   xmm3,xmm0
    3d95:	movq   xmm0,xmm1
    3d99:	mulps  xmm0,xmm3
    3d9c:	movq   xmm1,xmm2
    3da0:	movq   xmm0,xmm0
    3da4:	movaps xmm2,xmm1
    3da7:	subps  xmm2,xmm0
    3daa:	addps  xmm0,xmm1
    3dad:	movq   xmm1,QWORD PTR [rax]
    3db1:	movss  xmm0,xmm2
    3db5:	movq   xmm0,xmm0
    3db9:	addps  xmm0,xmm1
    3dbc:	movlps QWORD PTR [rax],xmm0
    3dbf:	add    rdx,0x1
    3dc3:	add    rcx,0x8
    3dc7:	add    rax,0x8
    3dcb:	cmp    rdx,0x10
    3dcf:	jne    3d60 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x60>
    3dd1:	add    rsi,0x1
    3dd5:	sub    r10,0xffffffffffffff80
    3dd9:	add    rdi,0x8
    3ddd:	cmp    rsi,0x10
    3de1:	jne    3d50 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x50>
    3de7:	ret
    3de8:	lea    rsi,[r11+0x800]
    3def:	lea    rcx,[r11+0x80]
    3df6:	xor    edi,edi
    3df8:	nop    DWORD PTR [rax+rax*1+0x0]
    3e00:	lea    rdx,[rcx-0x80]
    3e04:	lea    rax,[r11+0x880]
    3e0b:	nop    DWORD PTR [rax+rax*1+0x0]
    3e10:	movq   xmm1,QWORD PTR [rdx]
    3e14:	movq   xmm2,QWORD PTR [rsi]
    3e18:	add    rdx,0x8
    3e1c:	add    rax,0x8
    3e20:	movaps xmm0,xmm1
    3e23:	movq   xmm3,xmm2
    3e27:	shufps xmm2,xmm2,0xe1
    3e2b:	shufps xmm1,xmm1,0xe5
    3e2f:	shufps xmm0,xmm0,0xe0
    3e33:	movq   xmm0,xmm0
    3e37:	movq   xmm2,xmm2
    3e3b:	movq   xmm1,xmm1
    3e3f:	mulps  xmm0,xmm3
    3e42:	mulps  xmm1,xmm2
    3e45:	movq   xmm0,xmm0
    3e49:	movq   xmm1,xmm1
    3e4d:	movaps xmm2,xmm0
    3e50:	subps  xmm2,xmm1
    3e53:	addps  xmm0,xmm1
    3e56:	movq   xmm1,QWORD PTR [rax-0x8]
    3e5b:	movss  xmm0,xmm2
    3e5f:	movq   xmm0,xmm0
    3e63:	addps  xmm0,xmm1
    3e66:	movlps QWORD PTR [rax-0x8],xmm0
    3e6a:	cmp    rdx,rcx
    3e6d:	jne    3e10 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x110>
    3e6f:	add    rdi,0x1
    3e73:	add    rsi,0x8
    3e77:	sub    rcx,0xffffffffffffff80
    3e7b:	cmp    rdi,0x10
    3e7f:	jne    3e00 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x100>
    3e85:	ret
    3e86:	cs nop WORD PTR [rax+rax*1+0x0]

0000000000003e90 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EEEvv>:
    3e90:	endbr64
    3e94:	push   r12
    3e96:	pxor   xmm0,xmm0
    3e9a:	push   rbp
    3e9b:	push   rbx
    3e9c:	sub    rsp,0x920
    3ea3:	mov    rbx,QWORD PTR fs:0x28
    3eac:	mov    QWORD PTR [rsp+0x918],rbx
    3eb4:	lea    rbx,[rsp+0x10]
    3eb9:	lea    rdx,[rsp+0x810]
    3ec1:	mov    rax,rbx
    3ec4:	nop
    3ec5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    3ed0:	lea    rcx,[rax+0x80]
    3ed7:	movaps XMMWORD PTR [rax],xmm0
    3eda:	add    rax,0x20
    3ede:	movaps XMMWORD PTR [rax-0x10],xmm0
    3ee2:	cmp    rax,rcx
    3ee5:	jne    3ed7 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EEEvv+0x47>
    3ee7:	cmp    rax,rdx
    3eea:	jne    3ed0 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EEEvv+0x40>
    3eec:	lea    rdx,[rsp+0x90]
    3ef4:	mov    rax,rbx
    3ef7:	pxor   xmm0,xmm0
    3efb:	movaps XMMWORD PTR [rax+0x800],xmm0
    3f02:	add    rax,0x20
    3f06:	movaps XMMWORD PTR [rax+0x7f0],xmm0
    3f0d:	cmp    rax,rdx
    3f10:	jne    3efb <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EEEvv+0x6b>
    3f12:	mov    rax,rbx
    3f15:	pxor   xmm0,xmm0
    3f19:	movaps XMMWORD PTR [rax+0x880],xmm0
    3f20:	add    rax,0x20
    3f24:	movaps XMMWORD PTR [rax+0x870],xmm0
    3f2b:	cmp    rax,rdx
    3f2e:	jne    3f19 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EEEvv+0x89>
    3f30:	mov    QWORD PTR [rsp+0x910],0x10
    3f3c:	movsd  xmm6,QWORD PTR [rip+0x83b4]        # c2f8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xb8>
    3f44:	mov    r9,rbx
    3f47:	xor    esi,esi
    3f49:	movsd  xmm5,QWORD PTR [rip+0x8397]        # c2e8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
    3f51:	movsd  xmm1,QWORD PTR [rip+0x8387]        # c2e0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa0>
    3f59:	movabs rbp,0xaaaaaaaaaaaaaaab
    3f63:	movabs r8,0xcccccccccccccccd
    3f6d:	movsd  xmm4,QWORD PTR [rip+0x837b]        # c2f0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xb0>
    3f75:	movsd  xmm3,QWORD PTR [rip+0x835b]        # c2d8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x98>
    3f7d:	nop    DWORD PTR [rax]
    3f80:	mov    rax,rsi
    3f83:	mov    r10,rsi
    3f86:	pxor   xmm0,xmm0
    3f8a:	mov    r12,r9
    3f8d:	mul    rbp
    3f90:	pxor   xmm2,xmm2
    3f94:	mov    rcx,r10
    3f97:	mov    r11,r10
    3f9a:	xor    edi,edi
    3f9c:	mov    rax,rdx
    3f9f:	and    rdx,0xfffffffffffffffe
    3fa3:	shr    rax,1
    3fa6:	add    rdx,rax
    3fa9:	mov    rax,rsi
    3fac:	add    rsi,0x1
    3fb0:	cvtsi2sd xmm0,rsi
    3fb5:	divsd  xmm0,xmm5
    3fb9:	sub    rax,rdx
    3fbc:	add    rax,0x1
    3fc0:	cvtsi2sd xmm2,rax
    3fc5:	mulsd  xmm2,xmm6
    3fc9:	unpcklpd xmm0,xmm2
    3fcd:	cvtpd2ps xmm0,xmm0
    3fd1:	movlps QWORD PTR [r9+0x800],xmm0
    3fd9:	jmp    402b <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EEEvv+0x19b>
    3fdb:	nop    DWORD PTR [rax+rax*1+0x0]
    3fe0:	mov    rax,rcx
    3fe3:	pxor   xmm0,xmm0
    3fe7:	mul    r8
    3fea:	mov    rax,rdx
    3fed:	and    rdx,0xfffffffffffffffc
    3ff1:	shr    rax,0x2
    3ff5:	add    rdx,rax
    3ff8:	mov    rax,rcx
    3ffb:	sub    rax,rdx
    3ffe:	cvtsi2sd xmm0,rax
    4003:	mulsd  xmm0,xmm1
    4007:	unpcklpd xmm0,xmm2
    400b:	add    rdi,0x1
    400f:	add    r11,0x2
    4013:	sub    r12,0xffffffffffffff80
    4017:	cvtpd2ps xmm0,xmm0
    401b:	add    rcx,0x1
    401f:	movlps QWORD PTR [r12-0x80],xmm0
    4025:	cmp    rdi,0x10
    4029:	je     4090 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EEEvv+0x200>
    402b:	movabs rax,0x4924924924924925
    4035:	pxor   xmm2,xmm2
    4039:	imul   r11
    403c:	mov    rax,rdx
    403f:	mov    rdx,r11
    4042:	sar    rdx,0x3f
    4046:	sar    rax,1
    4049:	sub    rax,rdx
    404c:	lea    rdx,[rax*8+0x0]
    4054:	sub    rdx,rax
    4057:	mov    rax,r11
    405a:	sub    rax,rdx
    405d:	cvtsi2sd xmm2,rax
    4062:	mulsd  xmm2,QWORD PTR [rip+0x8296]        # c300 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xc0>
    406a:	cmp    rdi,r10
    406d:	jne    3fe0 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EEEvv+0x150>
    4073:	pxor   xmm0,xmm0
    4077:	cvtsi2sd xmm0,rdi
    407c:	mulsd  xmm0,xmm4
    4080:	addsd  xmm0,xmm3
    4084:	jmp    4007 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EEEvv+0x177>
    4086:	cs nop WORD PTR [rax+rax*1+0x0]
    4090:	add    r9,0x8
    4094:	cmp    rsi,0x10
    4098:	jne    3f80 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EEEvv+0xf0>
    409e:	mov    rdi,rbx
    40a1:	mov    rbp,rbx
    40a4:	xor    r12d,r12d
    40a7:	call   3d00 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv>
    40ac:	call   28a0 <instruction_count_begin>
    40b1:	mov    rdi,rbx
    40b4:	call   3d00 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv>
    40b9:	call   28b0 <instruction_count_end>
    40be:	lea    rsi,[rip+0x7f77]        # c03c <_IO_stdin_used+0x3c>
    40c5:	mov    edi,0x2
    40ca:	xor    eax,eax
    40cc:	call   10e0 <__printf_chk@plt>
    40d1:	nop    DWORD PTR [rax+0x0]
    40d5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    40e0:	mov    rdx,rbp
    40e3:	lea    rcx,[rbp+0x800]
    40ea:	nop    WORD PTR [rax+rax*1+0x0]
    40f0:	pxor   xmm0,xmm0
    40f4:	pxor   xmm1,xmm1
    40f8:	mov    edi,0x2
    40fd:	mov    eax,0x2
    4102:	lea    rsi,[rip+0x7f41]        # c04a <_IO_stdin_used+0x4a>
    4109:	mov    QWORD PTR [rsp+0x8],rcx
    410e:	cvtss2sd xmm0,DWORD PTR [rdx]
    4112:	cvtss2sd xmm1,DWORD PTR [rdx+0x4]
    4117:	mov    QWORD PTR [rsp],rdx
    411b:	call   10e0 <__printf_chk@plt>
    4120:	mov    rdx,QWORD PTR [rsp]
    4124:	mov    rcx,QWORD PTR [rsp+0x8]
    4129:	sub    rdx,0xffffffffffffff80
    412d:	cmp    rcx,rdx
    4130:	jne    40f0 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EEEvv+0x260>
    4132:	add    r12,0x1
    4136:	add    rbp,0x8
    413a:	cmp    r12,0x10
    413e:	jne    40e0 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EEEvv+0x250>
    4140:	lea    r12,[rbx+0x800]
    4147:	lea    rbp,[rbx+0x880]
    414e:	xchg   ax,ax
    4150:	pxor   xmm0,xmm0
    4154:	pxor   xmm1,xmm1
    4158:	mov    edi,0x2
    415d:	mov    eax,0x2
    4162:	cvtss2sd xmm0,DWORD PTR [r12]
    4168:	cvtss2sd xmm1,DWORD PTR [r12+0x4]
    416f:	lea    rsi,[rip+0x7ed4]        # c04a <_IO_stdin_used+0x4a>
    4176:	add    r12,0x8
    417a:	call   10e0 <__printf_chk@plt>
    417f:	cmp    r12,rbp
    4182:	jne    4150 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EEEvv+0x2c0>
    4184:	lea    rsi,[rip+0x7eb7]        # c042 <_IO_stdin_used+0x42>
    418b:	mov    edi,0x2
    4190:	xor    eax,eax
    4192:	add    rbx,0x900
    4199:	call   10e0 <__printf_chk@plt>
    419e:	xchg   ax,ax
    41a0:	pxor   xmm0,xmm0
    41a4:	pxor   xmm1,xmm1
    41a8:	mov    edi,0x2
    41ad:	mov    eax,0x2
    41b2:	cvtss2sd xmm0,DWORD PTR [rbp+0x0]
    41b7:	cvtss2sd xmm1,DWORD PTR [rbp+0x4]
    41bc:	lea    rsi,[rip+0x7e87]        # c04a <_IO_stdin_used+0x4a>
    41c3:	add    rbp,0x8
    41c7:	call   10e0 <__printf_chk@plt>
    41cc:	cmp    rbp,rbx
    41cf:	jne    41a0 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EEEvv+0x310>
    41d1:	xor    eax,eax
    41d3:	lea    rsi,[rip+0x7e97]        # c071 <_IO_stdin_used+0x71>
    41da:	mov    edi,0x2
    41df:	call   10e0 <__printf_chk@plt>
    41e4:	mov    rax,QWORD PTR [rsp+0x918]
    41ec:	sub    rax,QWORD PTR fs:0x28
    41f5:	jne    4203 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIfEELm16EEEvv+0x373>
    41f7:	add    rsp,0x920
    41fe:	pop    rbx
    41ff:	pop    rbp
    4200:	pop    r12
    4202:	ret
    4203:	call   10b0 <__stack_chk_fail@plt>
    4208:	nop    DWORD PTR [rax+rax*1+0x0]

0000000000004210 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv>:
    4210:	endbr64
    4214:	mov    r8,QWORD PTR [rdi+0x500]
    421b:	mov    r11,rdi
    421e:	mov    r10,rdi
    4221:	lea    rsi,[rdi+0x500]
    4228:	lea    rax,[rdi+0x480]
    422f:	pxor   xmm0,xmm0
    4233:	movups XMMWORD PTR [rax],xmm0
    4236:	add    rax,0x20
    423a:	movups XMMWORD PTR [rax-0x10],xmm0
    423e:	cmp    rax,rsi
    4241:	jne    4233 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x23>
    4243:	cmp    r8,0x8
    4247:	je     42e1 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0xd1>
    424d:	lea    r9,[r11+0x400]
    4254:	xor    esi,esi
    4256:	cs nop WORD PTR [rax+rax*1+0x0]
    4260:	lea    rdx,[r11+0x480]
    4267:	mov    rcx,r10
    426a:	xor    eax,eax
    426c:	nop    DWORD PTR [rax+0x0]
    4270:	cmp    rsi,rax
    4273:	mov    rdi,rax
    4276:	cmovae rdi,rsi
    427a:	cmp    rdi,r8
    427d:	jae    42bc <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0xac>
    427f:	movupd xmm2,XMMWORD PTR [rcx]
    4283:	movupd xmm0,XMMWORD PTR [r9]
    4288:	movupd xmm1,XMMWORD PTR [rcx]
    428c:	movupd xmm4,XMMWORD PTR [rdx]
    4290:	unpcklpd xmm2,xmm2
    4294:	mulpd  xmm2,xmm0
    4298:	unpckhpd xmm1,xmm1
    429c:	shufpd xmm0,xmm0,0x1
    42a1:	mulpd  xmm0,xmm1
    42a5:	movapd xmm1,xmm2
    42a9:	subpd  xmm1,xmm0
    42ad:	addpd  xmm0,xmm2
    42b1:	movsd  xmm0,xmm1
    42b5:	addpd  xmm0,xmm4
    42b9:	movups XMMWORD PTR [rdx],xmm0
    42bc:	add    rax,0x1
    42c0:	add    rcx,0x10
    42c4:	add    rdx,0x10
    42c8:	cmp    rax,0x8
    42cc:	jne    4270 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x60>
    42ce:	add    rsi,0x1
    42d2:	sub    r10,0xffffffffffffff80
    42d6:	add    r9,0x10
    42da:	cmp    rsi,0x8
    42de:	jne    4260 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x50>
    42e0:	ret
    42e1:	lea    rcx,[r11+0x400]
    42e8:	xor    edi,edi
    42ea:	nop    WORD PTR [rax+rax*1+0x0]
    42f0:	lea    rax,[r11+0x480]
    42f7:	mov    rdx,r10
    42fa:	nop    WORD PTR [rax+rax*1+0x0]
    4300:	movupd xmm0,XMMWORD PTR [rdx]
    4304:	movupd xmm2,XMMWORD PTR [rcx]
    4308:	add    rax,0x10
    430c:	add    rdx,0x10
    4310:	movupd xmm1,XMMWORD PTR [rdx-0x10]
    4315:	movupd xmm3,XMMWORD PTR [rax-0x10]
    431a:	unpcklpd xmm0,xmm0
    431e:	mulpd  xmm0,xmm2
    4322:	unpckhpd xmm1,xmm1
    4326:	shufpd xmm2,xmm2,0x1
    432b:	mulpd  xmm1,xmm2
    432f:	movapd xmm2,xmm0
    4333:	subpd  xmm2,xmm1
    4337:	addpd  xmm0,xmm1
    433b:	movsd  xmm0,xmm2
    433f:	addpd  xmm0,xmm3
    4343:	movups XMMWORD PTR [rax-0x10],xmm0
    4347:	cmp    rax,rsi
    434a:	jne    4300 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0xf0>
    434c:	add    rdi,0x1
    4350:	sub    r10,0xffffffffffffff80
    4354:	add    rcx,0x10
    4358:	cmp    rdi,0x8
    435c:	jne    42f0 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0xe0>
    435e:	ret
    435f:	nop

0000000000004360 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv>:
    4360:	endbr64
    4364:	push   r15
    4366:	pxor   xmm0,xmm0
    436a:	push   r14
    436c:	push   r13
    436e:	push   r12
    4370:	push   rbp
    4371:	push   rbx
    4372:	sub    rsp,0x528
    4379:	mov    r15,QWORD PTR fs:0x28
    4382:	mov    QWORD PTR [rsp+0x518],r15
    438a:	lea    r15,[rsp+0x10]
    438f:	lea    rbx,[rsp+0x90]
    4397:	lea    rcx,[rsp+0x490]
    439f:	mov    rdx,rbx
    43a2:	mov    r12,r15
    43a5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    43b0:	lea    rax,[rdx-0x80]
    43b4:	movaps XMMWORD PTR [rax],xmm0
    43b7:	add    rax,0x20
    43bb:	movaps XMMWORD PTR [rax-0x10],xmm0
    43bf:	cmp    rax,rdx
    43c2:	jne    43b4 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0x54>
    43c4:	lea    rdx,[rax+0x80]
    43cb:	cmp    rdx,rcx
    43ce:	jne    43b0 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0x50>
    43d0:	lea    rbp,[r15+0x400]
    43d7:	mov    rdx,rcx
    43da:	pxor   xmm0,xmm0
    43de:	sub    rdx,rbp
    43e1:	mov    rax,rbp
    43e4:	and    edx,0x10
    43e7:	je     4400 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0xa0>
    43e9:	lea    rax,[r15+0x410]
    43f0:	movaps XMMWORD PTR [rbp+0x0],xmm0
    43f4:	cmp    rax,rcx
    43f7:	je     4410 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0xb0>
    43f9:	nop    DWORD PTR [rax+0x0]
    4400:	movaps XMMWORD PTR [rax],xmm0
    4403:	add    rax,0x20
    4407:	movaps XMMWORD PTR [rax-0x10],xmm0
    440b:	cmp    rax,rcx
    440e:	jne    4400 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0xa0>
    4410:	lea    rax,[r15+0x500]
    4417:	pxor   xmm0,xmm0
    441b:	mov    rdx,rax
    441e:	sub    rdx,rcx
    4421:	and    edx,0x10
    4424:	je     4440 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0xe0>
    4426:	movaps XMMWORD PTR [rcx],xmm0
    4429:	add    rcx,0x10
    442d:	cmp    rcx,rax
    4430:	je     4450 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0xf0>
    4432:	nop    DWORD PTR [rax]
    4435:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    4440:	movaps XMMWORD PTR [rcx],xmm0
    4443:	add    rcx,0x20
    4447:	movaps XMMWORD PTR [rcx-0x10],xmm0
    444b:	cmp    rcx,rax
    444e:	jne    4440 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0xe0>
    4450:	mov    QWORD PTR [rsp+0x8],rbp
    4455:	movsd  xmm7,QWORD PTR [rip+0x7e9b]        # c2f8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xb8>
    445d:	mov    r13,r15
    4460:	xor    r10d,r10d
    4463:	movsd  xmm6,QWORD PTR [rip+0x7e5d]        # c2c8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x88>
    446b:	movsd  xmm2,QWORD PTR [rip+0x7e8d]        # c300 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xc0>
    4473:	mov    QWORD PTR [rsp+0x510],0x8
    447f:	movabs r14,0xaaaaaaaaaaaaaaab
    4489:	movsd  xmm3,QWORD PTR [rip+0x7e4f]        # c2e0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa0>
    4491:	movsd  xmm5,QWORD PTR [rip+0x7e37]        # c2d0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x90>
    4499:	movabs r9,0x4924924924924925
    44a3:	movabs r11,0xcccccccccccccccd
    44ad:	movsd  xmm4,QWORD PTR [rip+0x7e23]        # c2d8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x98>
    44b5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    44c0:	mov    rax,r10
    44c3:	mov    rbp,r10
    44c6:	pxor   xmm0,xmm0
    44ca:	mov    r8,r13
    44cd:	mul    r14
    44d0:	pxor   xmm1,xmm1
    44d4:	mov    rsi,rbp
    44d7:	xor    ecx,ecx
    44d9:	mov    rax,rdx
    44dc:	and    rdx,0xfffffffffffffffe
    44e0:	shr    rax,1
    44e3:	add    rdx,rax
    44e6:	mov    rax,r10
    44e9:	add    r10,0x1
    44ed:	cvtsi2sd xmm0,r10
    44f2:	divsd  xmm0,xmm6
    44f6:	sub    rax,rdx
    44f9:	add    rax,0x1
    44fd:	cvtsi2sd xmm1,rax
    4502:	mulsd  xmm1,xmm7
    4506:	unpcklpd xmm0,xmm1
    450a:	movaps XMMWORD PTR [r13+0x400],xmm0
    4512:	jmp    455b <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0x1fb>
    4514:	nop    DWORD PTR [rax+0x0]
    4518:	lea    rdi,[rcx+rbp*1]
    451c:	pxor   xmm0,xmm0
    4520:	mov    rax,rdi
    4523:	mul    r11
    4526:	mov    rax,rdx
    4529:	and    rdx,0xfffffffffffffffc
    452d:	shr    rax,0x2
    4531:	add    rdx,rax
    4534:	sub    rdi,rdx
    4537:	cvtsi2sd xmm0,rdi
    453c:	mulsd  xmm0,xmm3
    4540:	unpcklpd xmm0,xmm1
    4544:	add    rcx,0x1
    4548:	add    rsi,0x2
    454c:	sub    r8,0xffffffffffffff80
    4550:	movaps XMMWORD PTR [r8-0x80],xmm0
    4555:	cmp    rcx,0x8
    4559:	je     45a8 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0x248>
    455b:	mov    rax,rsi
    455e:	pxor   xmm1,xmm1
    4562:	imul   r9
    4565:	mov    rax,rsi
    4568:	sar    rax,0x3f
    456c:	sar    rdx,1
    456f:	sub    rdx,rax
    4572:	lea    rax,[rdx*8+0x0]
    457a:	sub    rax,rdx
    457d:	mov    rdx,rsi
    4580:	sub    rdx,rax
    4583:	cvtsi2sd xmm1,rdx
    4588:	mulsd  xmm1,xmm2
    458c:	cmp    rcx,rbp
    458f:	jne    4518 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0x1b8>
    4591:	pxor   xmm0,xmm0
    4595:	cvtsi2sd xmm0,rcx
    459a:	mulsd  xmm0,xmm5
    459e:	addsd  xmm0,xmm4
    45a2:	jmp    4540 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0x1e0>
    45a4:	nop    DWORD PTR [rax+0x0]
    45a8:	add    r13,0x10
    45ac:	cmp    r10,0x8
    45b0:	jne    44c0 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0x160>
    45b6:	mov    rdi,r15
    45b9:	mov    rbp,QWORD PTR [rsp+0x8]
    45be:	xor    r14d,r14d
    45c1:	call   4210 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv>
    45c6:	call   28a0 <instruction_count_begin>
    45cb:	mov    rdi,r15
    45ce:	call   4210 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv>
    45d3:	call   28b0 <instruction_count_end>
    45d8:	lea    rsi,[rip+0x7a5d]        # c03c <_IO_stdin_used+0x3c>
    45df:	mov    edi,0x2
    45e4:	xor    eax,eax
    45e6:	call   10e0 <__printf_chk@plt>
    45eb:	nop    DWORD PTR [rax+rax*1+0x0]
    45f0:	lea    r13,[rbp-0x400]
    45f7:	nop    WORD PTR [rax+rax*1+0x0]
    4600:	movsd  xmm0,QWORD PTR [r13+0x0]
    4606:	movsd  xmm1,QWORD PTR [r13+0x8]
    460c:	mov    edi,0x2
    4611:	lea    rsi,[rip+0x7a32]        # c04a <_IO_stdin_used+0x4a>
    4618:	mov    eax,0x2
    461d:	sub    r13,0xffffffffffffff80
    4621:	call   10e0 <__printf_chk@plt>
    4626:	cmp    r13,rbp
    4629:	jne    4600 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0x2a0>
    462b:	add    r14,0x1
    462f:	lea    rbp,[r13+0x10]
    4633:	cmp    r14,0x8
    4637:	jne    45f0 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0x290>
    4639:	nop    DWORD PTR [rax+0x0]
    4640:	lea    rsi,[rip+0x7a03]        # c04a <_IO_stdin_used+0x4a>
    4647:	mov    edi,0x2
    464c:	mov    eax,0x2
    4651:	movsd  xmm0,QWORD PTR [r15+0x400]
    465a:	movsd  xmm1,QWORD PTR [r15+0x408]
    4663:	add    r15,0x10
    4667:	call   10e0 <__printf_chk@plt>
    466c:	cmp    r15,rbx
    466f:	jne    4640 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0x2e0>
    4671:	lea    rsi,[rip+0x79ca]        # c042 <_IO_stdin_used+0x42>
    4678:	mov    edi,0x2
    467d:	xor    eax,eax
    467f:	call   10e0 <__printf_chk@plt>
    4684:	nop
    4685:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    4690:	lea    rsi,[rip+0x79b3]        # c04a <_IO_stdin_used+0x4a>
    4697:	mov    edi,0x2
    469c:	mov    eax,0x2
    46a1:	movsd  xmm0,QWORD PTR [r12+0x480]
    46ab:	movsd  xmm1,QWORD PTR [r12+0x488]
    46b5:	add    r12,0x10
    46b9:	call   10e0 <__printf_chk@plt>
    46be:	cmp    r12,rbx
    46c1:	jne    4690 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0x330>
    46c3:	xor    eax,eax
    46c5:	lea    rsi,[rip+0x79a5]        # c071 <_IO_stdin_used+0x71>
    46cc:	mov    edi,0x2
    46d1:	call   10e0 <__printf_chk@plt>
    46d6:	mov    rax,QWORD PTR [rsp+0x518]
    46de:	sub    rax,QWORD PTR fs:0x28
    46e7:	jne    46fb <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0x39b>
    46e9:	add    rsp,0x528
    46f0:	pop    rbx
    46f1:	pop    rbp
    46f2:	pop    r12
    46f4:	pop    r13
    46f6:	pop    r14
    46f8:	pop    r15
    46fa:	ret
    46fb:	call   10b0 <__stack_chk_fail@plt>

0000000000004700 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv>:
    4700:	endbr64
    4704:	mov    r9,QWORD PTR [rdi+0x1200]
    470b:	mov    r11,rdi
    470e:	mov    r10,rdi
    4711:	lea    rdx,[rdi+0x1200]
    4718:	lea    rax,[rdi+0x1100]
    471f:	pxor   xmm0,xmm0
    4723:	xchg   ax,ax
    4725:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    4730:	movups XMMWORD PTR [rax],xmm0
    4733:	add    rax,0x20
    4737:	movups XMMWORD PTR [rax-0x10],xmm0
    473b:	cmp    rax,rdx
    473e:	jne    4730 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x30>
    4740:	cmp    r9,0x10
    4744:	je     47e7 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0xe7>
    474a:	lea    rdi,[r11+0x1000]
    4751:	xor    esi,esi
    4753:	xchg   ax,ax
    4755:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    4760:	lea    rax,[r11+0x1100]
    4767:	mov    rcx,r10
    476a:	xor    edx,edx
    476c:	nop    DWORD PTR [rax+0x0]
    4770:	cmp    rsi,rdx
    4773:	mov    r8,rdx
    4776:	cmovae r8,rsi
    477a:	cmp    r8,r9
    477d:	jae    47bb <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0xbb>
    477f:	movupd xmm2,XMMWORD PTR [rcx]
    4783:	movupd xmm0,XMMWORD PTR [rdi]
    4787:	movupd xmm1,XMMWORD PTR [rcx]
    478b:	movupd xmm4,XMMWORD PTR [rax]
    478f:	unpcklpd xmm2,xmm2
    4793:	mulpd  xmm2,xmm0
    4797:	unpckhpd xmm1,xmm1
    479b:	shufpd xmm0,xmm0,0x1
    47a0:	mulpd  xmm0,xmm1
    47a4:	movapd xmm1,xmm2
    47a8:	subpd  xmm1,xmm0
    47ac:	addpd  xmm0,xmm2
    47b0:	movsd  xmm0,xmm1
    47b4:	addpd  xmm0,xmm4
    47b8:	movups XMMWORD PTR [rax],xmm0
    47bb:	add    rdx,0x1
    47bf:	add    rcx,0x10
    47c3:	add    rax,0x10
    47c7:	cmp    rdx,0x10
    47cb:	jne    4770 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x70>
    47cd:	add    rsi,0x1
    47d1:	add    r10,0x100
    47d8:	add    rdi,0x10
    47dc:	cmp    rsi,0x10
    47e0:	jne    4760 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x60>
    47e6:	ret
    47e7:	lea    rsi,[r11+0x1000]
    47ee:	lea    rcx,[r11+0x100]
    47f5:	xor    edi,edi
    47f7:	nop    WORD PTR [rax+rax*1+0x0]
    4800:	lea    rdx,[rcx-0x100]
    4807:	lea    rax,[r11+0x1100]
    480e:	xchg   ax,ax
    4810:	movupd xmm0,XMMWORD PTR [rdx]
    4814:	movupd xmm2,XMMWORD PTR [rsi]
    4818:	add    rdx,0x10
    481c:	add    rax,0x10
    4820:	movupd xmm1,XMMWORD PTR [rdx-0x10]
    4825:	movupd xmm3,XMMWORD PTR [rax-0x10]
    482a:	unpcklpd xmm0,xmm0
    482e:	mulpd  xmm0,xmm2
    4832:	unpckhpd xmm1,xmm1
    4836:	shufpd xmm2,xmm2,0x1
    483b:	mulpd  xmm1,xmm2
    483f:	movapd xmm2,xmm0
    4843:	subpd  xmm2,xmm1
    4847:	addpd  xmm0,xmm1
    484b:	movsd  xmm0,xmm2
    484f:	addpd  xmm0,xmm3
    4853:	movups XMMWORD PTR [rax-0x10],xmm0
    4857:	cmp    rdx,rcx
    485a:	jne    4810 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x110>
    485c:	add    rdi,0x1
    4860:	add    rsi,0x10
    4864:	add    rcx,0x100
    486b:	cmp    rdi,0x10
    486f:	jne    4800 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x100>
    4871:	ret
    4872:	cs nop WORD PTR [rax+rax*1+0x0]
    487c:	nop    DWORD PTR [rax+0x0]

0000000000004880 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EEEvv>:
    4880:	endbr64
    4884:	push   r15
    4886:	push   r14
    4888:	push   r13
    488a:	push   r12
    488c:	push   rbp
    488d:	push   rbx
    488e:	sub    rsp,0x1000
    4895:	or     QWORD PTR [rsp],0x0
    489a:	sub    rsp,0x228
    48a1:	pxor   xmm0,xmm0
    48a5:	mov    r14,QWORD PTR fs:0x28
    48ae:	mov    QWORD PTR [rsp+0x1218],r14
    48b6:	lea    r14,[rsp+0x10]
    48bb:	lea    rdx,[rsp+0x1010]
    48c3:	mov    rax,r14
    48c6:	cs nop WORD PTR [rax+rax*1+0x0]
    48d0:	mov    rbx,rax
    48d3:	lea    rax,[rax+0x100]
    48da:	movaps XMMWORD PTR [rbx],xmm0
    48dd:	add    rbx,0x20
    48e1:	movaps XMMWORD PTR [rbx-0x10],xmm0
    48e5:	cmp    rbx,rax
    48e8:	jne    48da <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0x5a>
    48ea:	mov    rax,rbx
    48ed:	cmp    rbx,rdx
    48f0:	jne    48d0 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0x50>
    48f2:	mov    r12,r14
    48f5:	lea    rbp,[rsp+0x1110]
    48fd:	pxor   xmm0,xmm0
    4901:	nop    DWORD PTR [rax+0x0]
    4905:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    4910:	movaps XMMWORD PTR [rax],xmm0
    4913:	add    rax,0x20
    4917:	movaps XMMWORD PTR [rax-0x10],xmm0
    491b:	cmp    rax,rbp
    491e:	jne    4910 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0x90>
    4920:	lea    r13,[r14+0x1200]
    4927:	mov    rax,rbp
    492a:	pxor   xmm0,xmm0
    492e:	mov    rdx,r13
    4931:	sub    rdx,rbp
    4934:	and    edx,0x10
    4937:	je     4950 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0xd0>
    4939:	lea    rax,[rbp+0x10]
    493d:	movaps XMMWORD PTR [rbp+0x0],xmm0
    4941:	cmp    rax,r13
    4944:	je     4960 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0xe0>
    4946:	cs nop WORD PTR [rax+rax*1+0x0]
    4950:	movaps XMMWORD PTR [rax],xmm0
    4953:	add    rax,0x20
    4957:	movaps XMMWORD PTR [rax-0x10],xmm0
    495b:	cmp    rax,r13
    495e:	jne    4950 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0xd0>
    4960:	mov    QWORD PTR [rsp+0x8],rbx
    4965:	movsd  xmm6,QWORD PTR [rip+0x798b]        # c2f8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xb8>
    496d:	mov    r9,r14
    4970:	xor    esi,esi
    4972:	movsd  xmm5,QWORD PTR [rip+0x796e]        # c2e8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
    497a:	movsd  xmm1,QWORD PTR [rip+0x795e]        # c2e0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa0>
    4982:	mov    QWORD PTR [rsp+0x1210],0x10
    498e:	movabs r15,0xaaaaaaaaaaaaaaab
    4998:	movsd  xmm4,QWORD PTR [rip+0x7950]        # c2f0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xb0>
    49a0:	movsd  xmm3,QWORD PTR [rip+0x7930]        # c2d8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x98>
    49a8:	movabs r8,0xcccccccccccccccd
    49b2:	nop    DWORD PTR [rax]
    49b5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    49c0:	mov    rax,rsi
    49c3:	mov    rbx,rsi
    49c6:	pxor   xmm0,xmm0
    49ca:	mov    r11,r9
    49cd:	mul    r15
    49d0:	pxor   xmm2,xmm2
    49d4:	mov    rcx,rbx
    49d7:	mov    r10,rbx
    49da:	xor    edi,edi
    49dc:	mov    rax,rdx
    49df:	and    rdx,0xfffffffffffffffe
    49e3:	shr    rax,1
    49e6:	add    rdx,rax
    49e9:	mov    rax,rsi
    49ec:	add    rsi,0x1
    49f0:	cvtsi2sd xmm0,rsi
    49f5:	divsd  xmm0,xmm5
    49f9:	sub    rax,rdx
    49fc:	add    rax,0x1
    4a00:	cvtsi2sd xmm2,rax
    4a05:	mulsd  xmm2,xmm6
    4a09:	unpcklpd xmm0,xmm2
    4a0d:	movaps XMMWORD PTR [r9+0x1000],xmm0
    4a15:	jmp    4a68 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0x1e8>
    4a17:	nop    WORD PTR [rax+rax*1+0x0]
    4a20:	mov    rax,rcx
    4a23:	pxor   xmm0,xmm0
    4a27:	mul    r8
    4a2a:	mov    rax,rdx
    4a2d:	and    rdx,0xfffffffffffffffc
    4a31:	shr    rax,0x2
    4a35:	add    rdx,rax
    4a38:	mov    rax,rcx
    4a3b:	sub    rax,rdx
    4a3e:	cvtsi2sd xmm0,rax
    4a43:	mulsd  xmm0,xmm1
    4a47:	unpcklpd xmm0,xmm2
    4a4b:	add    rdi,0x1
    4a4f:	add    r10,0x2
    4a53:	add    rcx,0x1
    4a57:	movaps XMMWORD PTR [r11],xmm0
    4a5b:	add    r11,0x100
    4a62:	cmp    rdi,0x10
    4a66:	je     4ac8 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0x248>
    4a68:	movabs rax,0x4924924924924925
    4a72:	pxor   xmm2,xmm2
    4a76:	imul   r10
    4a79:	mov    rax,rdx
    4a7c:	mov    rdx,r10
    4a7f:	sar    rdx,0x3f
    4a83:	sar    rax,1
    4a86:	sub    rax,rdx
    4a89:	lea    rdx,[rax*8+0x0]
    4a91:	sub    rdx,rax
    4a94:	mov    rax,r10
    4a97:	sub    rax,rdx
    4a9a:	cvtsi2sd xmm2,rax
    4a9f:	mulsd  xmm2,QWORD PTR [rip+0x7859]        # c300 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xc0>
    4aa7:	cmp    rdi,rbx
    4aaa:	jne    4a20 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0x1a0>
    4ab0:	pxor   xmm0,xmm0
    4ab4:	cvtsi2sd xmm0,rdi
    4ab9:	mulsd  xmm0,xmm4
    4abd:	addsd  xmm0,xmm3
    4ac1:	jmp    4a47 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0x1c7>
    4ac3:	nop    DWORD PTR [rax+rax*1+0x0]
    4ac8:	add    r9,0x10
    4acc:	cmp    rsi,0x10
    4ad0:	jne    49c0 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0x140>
    4ad6:	mov    rdi,r14
    4ad9:	mov    rbx,QWORD PTR [rsp+0x8]
    4ade:	call   4700 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv>
    4ae3:	call   28a0 <instruction_count_begin>
    4ae8:	mov    rdi,r14
    4aeb:	xor    r14d,r14d
    4aee:	call   4700 <_ZN12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv>
    4af3:	call   28b0 <instruction_count_end>
    4af8:	lea    rsi,[rip+0x753d]        # c03c <_IO_stdin_used+0x3c>
    4aff:	mov    edi,0x2
    4b04:	xor    eax,eax
    4b06:	call   10e0 <__printf_chk@plt>
    4b0b:	nop    DWORD PTR [rax+rax*1+0x0]
    4b10:	mov    r15,r12
    4b13:	lea    rdx,[r12+0x1000]
    4b1b:	nop    DWORD PTR [rax+rax*1+0x0]
    4b20:	movsd  xmm0,QWORD PTR [r15]
    4b25:	movsd  xmm1,QWORD PTR [r15+0x8]
    4b2b:	mov    edi,0x2
    4b30:	lea    rsi,[rip+0x7513]        # c04a <_IO_stdin_used+0x4a>
    4b37:	mov    eax,0x2
    4b3c:	mov    QWORD PTR [rsp+0x8],rdx
    4b41:	add    r15,0x100
    4b48:	call   10e0 <__printf_chk@plt>
    4b4d:	mov    rdx,QWORD PTR [rsp+0x8]
    4b52:	cmp    r15,rdx
    4b55:	jne    4b20 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0x2a0>
    4b57:	add    r14,0x1
    4b5b:	add    r12,0x10
    4b5f:	cmp    r14,0x10
    4b63:	jne    4b10 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0x290>
    4b65:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    4b70:	movsd  xmm0,QWORD PTR [rbx]
    4b74:	movsd  xmm1,QWORD PTR [rbx+0x8]
    4b79:	mov    edi,0x2
    4b7e:	lea    rsi,[rip+0x74c5]        # c04a <_IO_stdin_used+0x4a>
    4b85:	mov    eax,0x2
    4b8a:	add    rbx,0x10
    4b8e:	call   10e0 <__printf_chk@plt>
    4b93:	cmp    rbx,rbp
    4b96:	jne    4b70 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0x2f0>
    4b98:	lea    rsi,[rip+0x74a3]        # c042 <_IO_stdin_used+0x42>
    4b9f:	mov    edi,0x2
    4ba4:	xor    eax,eax
    4ba6:	call   10e0 <__printf_chk@plt>
    4bab:	nop    DWORD PTR [rax+rax*1+0x0]
    4bb0:	movsd  xmm0,QWORD PTR [rbp+0x0]
    4bb5:	movsd  xmm1,QWORD PTR [rbp+0x8]
    4bba:	mov    edi,0x2
    4bbf:	lea    rsi,[rip+0x7484]        # c04a <_IO_stdin_used+0x4a>
    4bc6:	mov    eax,0x2
    4bcb:	add    rbp,0x10
    4bcf:	call   10e0 <__printf_chk@plt>
    4bd4:	cmp    rbp,r13
    4bd7:	jne    4bb0 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0x330>
    4bd9:	xor    eax,eax
    4bdb:	lea    rsi,[rip+0x748f]        # c071 <_IO_stdin_used+0x71>
    4be2:	mov    edi,0x2
    4be7:	call   10e0 <__printf_chk@plt>
    4bec:	mov    rax,QWORD PTR [rsp+0x1218]
    4bf4:	sub    rax,QWORD PTR fs:0x28
    4bfd:	jne    4c11 <_Z3runI12MultiplyCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0x391>
    4bff:	add    rsp,0x1228
    4c06:	pop    rbx
    4c07:	pop    rbp
    4c08:	pop    r12
    4c0a:	pop    r13
    4c0c:	pop    r14
    4c0e:	pop    r15
    4c10:	ret
    4c11:	call   10b0 <__stack_chk_fail@plt>
    4c16:	cs nop WORD PTR [rax+rax*1+0x0]
    4c20:	cs nop WORD PTR [rax+rax*1+0x0]
    4c2a:	cs nop WORD PTR [rax+rax*1+0x0]
    4c34:	cs nop WORD PTR [rax+rax*1+0x0]
    4c3e:	xchg   ax,ax

0000000000004c40 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv>:
    4c40:	endbr64
    4c44:	pxor   xmm4,xmm4
    4c48:	mov    rdx,QWORD PTR [rdi+0x48]
    4c4c:	xor    eax,eax
    4c4e:	movaps xmm2,xmm4
    4c51:	nop    DWORD PTR [rax+0x0]
    4c55:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    4c60:	cmp    rax,rdx
    4c63:	jae    4cb4 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x74>
    4c65:	movss  xmm1,DWORD PTR [rdi+rax*8]
    4c6a:	movaps xmm3,xmm1
    4c6d:	movaps xmm0,xmm1
    4c70:	xorps  xmm0,XMMWORD PTR [rip+0x75d9]        # c250 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    4c77:	cmpltss xmm1,xmm2
    4c7c:	andps  xmm0,xmm1
    4c7f:	andnps xmm1,xmm3
    4c82:	orps   xmm1,xmm0
    4c85:	movss  xmm0,DWORD PTR [rdi+rax*8+0x4]
    4c8b:	movaps xmm5,xmm0
    4c8e:	movaps xmm3,xmm0
    4c91:	xorps  xmm3,XMMWORD PTR [rip+0x75b8]        # c250 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    4c98:	cmpltss xmm0,xmm2
    4c9d:	andps  xmm3,xmm0
    4ca0:	andnps xmm0,xmm5
    4ca3:	orps   xmm0,xmm3
    4ca6:	movaps xmm3,xmm1
    4ca9:	maxss  xmm3,xmm0
    4cad:	maxss  xmm3,xmm4
    4cb1:	movaps xmm4,xmm3
    4cb4:	add    rax,0x1
    4cb8:	cmp    rax,0x8
    4cbc:	jne    4c60 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x20>
    4cbe:	pxor   xmm0,xmm0
    4cc2:	ucomiss xmm4,xmm0
    4cc5:	jp     4cc9 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x89>
    4cc7:	je     4d41 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x101>
    4cc9:	pxor   xmm2,xmm2
    4ccd:	xor    eax,eax
    4ccf:	nop    DWORD PTR [rax+rax*1+0x0]
    4cd4:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    4cdf:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    4cea:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    4cf5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    4d00:	cmp    rax,rdx
    4d03:	jae    4d28 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0xe8>
    4d05:	movss  xmm0,DWORD PTR [rdi+rax*8]
    4d0a:	movss  xmm1,DWORD PTR [rdi+rax*8+0x4]
    4d10:	divss  xmm0,xmm4
    4d14:	divss  xmm1,xmm4
    4d18:	mulss  xmm0,xmm0
    4d1c:	mulss  xmm1,xmm1
    4d20:	addss  xmm0,xmm1
    4d24:	addss  xmm2,xmm0
    4d28:	add    rax,0x1
    4d2c:	cmp    rax,0x8
    4d30:	jne    4d00 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0xc0>
    4d32:	pxor   xmm0,xmm0
    4d36:	ucomiss xmm2,xmm0
    4d39:	jp     4d50 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x110>
    4d3b:	jne    4d50 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x110>
    4d3d:	mulss  xmm0,xmm4
    4d41:	movss  DWORD PTR [rdi+0x40],xmm0
    4d46:	ret
    4d47:	nop    WORD PTR [rax+rax*1+0x0]
    4d50:	movss  xmm1,DWORD PTR [rip+0x72b0]        # c008 <_IO_stdin_used+0x8>
    4d58:	movaps xmm3,xmm1
    4d5b:	maxss  xmm3,xmm2
    4d5f:	maxss  xmm1,xmm3
    4d63:	movd   eax,xmm1
    4d67:	movaps xmm0,xmm1
    4d6a:	mov    esi,eax
    4d6c:	shr    esi,0x17
    4d6f:	movzx  ecx,sil
    4d73:	cmp    ecx,0xff
    4d79:	je     4f08 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x2c8>
    4d7f:	test   ecx,ecx
    4d81:	je     4e88 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x248>
    4d87:	and    eax,0x807fffff
    4d8c:	mov    edx,0x7e
    4d91:	and    esi,0x1
    4d94:	or     eax,0x3f000000
    4d99:	sub    edx,ecx
    4d9b:	movd   xmm2,eax
    4d9f:	lea    rax,[rip+0x749a]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    4da6:	sar    edx,1
    4da8:	movss  xmm5,DWORD PTR [rax+rsi*4]
    4dad:	mulss  xmm2,DWORD PTR [rip+0x7257]        # c00c <_IO_stdin_used+0xc>
    4db5:	movss  xmm1,DWORD PTR [rip+0x7253]        # c010 <_IO_stdin_used+0x10>
    4dbd:	subss  xmm1,xmm2
    4dc1:	mulss  xmm1,xmm5
    4dc5:	movd   ecx,xmm1
    4dc9:	mov    eax,ecx
    4dcb:	shr    eax,0x17
    4dce:	movzx  eax,al
    4dd1:	cmp    eax,0xff
    4dd6:	je     4e20 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x1e0>
    4dd8:	mov    esi,ecx
    4dda:	mov    r9d,eax
    4ddd:	and    esi,0x7fffff
    4de3:	or     r9d,esi
    4de6:	je     4e20 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x1e0>
    4de8:	mov    r8d,ecx
    4deb:	and    r8d,0x80000000
    4df2:	test   eax,eax
    4df4:	je     4f1a <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x2da>
    4dfa:	add    eax,edx
    4dfc:	je     4fca <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x38a>
    4e02:	cmp    eax,0xfe
    4e07:	jg     4fca <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x38a>
    4e0d:	shl    eax,0x17
    4e10:	and    ecx,0x807fffff
    4e16:	or     eax,ecx
    4e18:	movd   xmm1,eax
    4e1c:	nop    DWORD PTR [rax+0x0]
    4e20:	movaps xmm2,xmm0
    4e23:	movss  xmm6,DWORD PTR [rip+0x71e9]        # c014 <_IO_stdin_used+0x14>
    4e2b:	movss  xmm5,DWORD PTR [rip+0x71e5]        # c018 <_IO_stdin_used+0x18>
    4e33:	mulss  xmm2,xmm1
    4e37:	mulss  xmm2,xmm1
    4e3b:	mulss  xmm6,xmm2
    4e3f:	subss  xmm5,xmm6
    4e43:	mulss  xmm5,xmm2
    4e47:	movss  xmm2,DWORD PTR [rip+0x7445]        # c294 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x54>
    4e4f:	subss  xmm2,xmm5
    4e53:	mulss  xmm2,xmm1
    4e57:	movss  xmm1,DWORD PTR [rip+0x71bd]        # c01c <_IO_stdin_used+0x1c>
    4e5f:	mulss  xmm1,xmm0
    4e63:	movss  xmm0,DWORD PTR [rip+0x7469]        # c2d4 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x94>
    4e6b:	mulss  xmm1,xmm2
    4e6f:	mulss  xmm1,xmm2
    4e73:	subss  xmm0,xmm1
    4e77:	mulss  xmm0,xmm2
    4e7b:	mulss  xmm0,xmm3
    4e7f:	jmp    4d3d <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0xfd>
    4e84:	nop    DWORD PTR [rax+0x0]
    4e88:	mov    esi,eax
    4e8a:	and    esi,0x7fffff
    4e90:	je     4f08 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x2c8>
    4e92:	mov    r8d,eax
    4e95:	and    r8d,0x80000000
    4e9c:	test   eax,0x7f0000
    4ea1:	mov    eax,esi
    4ea3:	je     4f9e <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x35e>
    4ea9:	shl    eax,0x8
    4eac:	mov    edx,0x8
    4eb1:	cmp    eax,0xfffffff
    4eb6:	jbe    4fb4 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x374>
    4ebc:	cmp    eax,0x3fffffff
    4ec1:	ja     4ecd <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x28d>
    4ec3:	add    edx,0x2
    4ec6:	shl    eax,0x2
    4ec9:	test   eax,eax
    4ecb:	js     4ed0 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x290>
    4ecd:	add    edx,0x1
    4ed0:	lea    ecx,[rdx-0x9]
    4ed3:	lea    rax,[rip+0x7366]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    4eda:	sar    edx,1
    4edc:	shl    esi,cl
    4ede:	not    ecx
    4ee0:	add    edx,0x3a
    4ee3:	and    esi,0x7fffff
    4ee9:	and    ecx,0x1
    4eec:	or     esi,r8d
    4eef:	movss  xmm5,DWORD PTR [rax+rcx*4]
    4ef4:	or     esi,0x3f000000
    4efa:	movd   xmm2,esi
    4efe:	jmp    4dad <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x16d>
    4f03:	nop    DWORD PTR [rax+rax*1+0x0]
    4f08:	movss  xmm5,DWORD PTR [rip+0x70f4]        # c004 <_IO_stdin_used+0x4>
    4f10:	movaps xmm2,xmm0
    4f13:	xor    edx,edx
    4f15:	jmp    4dad <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x16d>
    4f1a:	and    ecx,0x7f0000
    4f20:	mov    eax,esi
    4f22:	je     4f75 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x335>
    4f24:	shl    eax,0x8
    4f27:	mov    ecx,0x8
    4f2c:	cmp    eax,0xfffffff
    4f31:	jbe    4f8f <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x34f>
    4f33:	cmp    eax,0x3fffffff
    4f38:	ja     4f48 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x308>
    4f3a:	add    ecx,0x2
    4f3d:	shl    eax,0x2
    4f40:	test   eax,eax
    4f42:	js     4ff0 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x3b0>
    4f48:	sub    ecx,0x8
    4f4b:	mov    eax,edx
    4f4d:	shl    esi,cl
    4f4f:	sub    eax,ecx
    4f51:	and    esi,0x7fffff
    4f57:	add    eax,0x1
    4f5a:	test   eax,eax
    4f5c:	jle    4ff8 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x3b8>
    4f62:	or     r8d,esi
    4f65:	shl    eax,0x17
    4f68:	or     r8d,eax
    4f6b:	movd   xmm1,r8d
    4f70:	jmp    4e20 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x1e0>
    4f75:	shl    eax,0x10
    4f78:	mov    ecx,0x10
    4f7d:	cmp    eax,0xffffff
    4f82:	jbe    5028 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x3e8>
    4f88:	cmp    eax,0xfffffff
    4f8d:	ja     4f95 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x355>
    4f8f:	add    ecx,0x4
    4f92:	shl    eax,0x4
    4f95:	cmp    eax,0x3fffffff
    4f9a:	ja     4f40 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x300>
    4f9c:	jmp    4f3a <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x2fa>
    4f9e:	shl    eax,0x10
    4fa1:	mov    edx,0x10
    4fa6:	cmp    eax,0xffffff
    4fab:	jbe    501c <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x3dc>
    4fad:	cmp    eax,0xfffffff
    4fb2:	ja     4fba <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x37a>
    4fb4:	add    edx,0x4
    4fb7:	shl    eax,0x4
    4fba:	cmp    eax,0x3fffffff
    4fbf:	ja     4ec9 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x289>
    4fc5:	jmp    4ec3 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x283>
    4fca:	shr    esi,1
    4fcc:	mov    edx,esi
    4fce:	or     edx,r8d
    4fd1:	or     r8d,0x7f800000
    4fd8:	or     edx,0x400000
    4fde:	cmp    eax,0xfe
    4fe3:	cmovg  edx,r8d
    4fe7:	movd   xmm1,edx
    4feb:	jmp    4e20 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x1e0>
    4ff0:	sub    ecx,0x9
    4ff3:	jmp    4f4b <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x30b>
    4ff8:	movd   xmm1,r8d
    4ffd:	cmp    eax,0xffffffea
    5000:	jl     4e20 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x1e0>
    5006:	or     esi,0x800000
    500c:	sub    ecx,edx
    500e:	shr    esi,cl
    5010:	or     esi,r8d
    5013:	movd   xmm1,esi
    5017:	jmp    4e20 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x1e0>
    501c:	mov    eax,esi
    501e:	mov    edx,0x18
    5023:	shl    eax,0x18
    5026:	jmp    4fad <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x36d>
    5028:	mov    eax,esi
    502a:	mov    ecx,0x18
    502f:	shl    eax,0x18
    5032:	jmp    4f88 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv+0x348>
    5037:	nop    WORD PTR [rax+rax*1+0x0]

0000000000005040 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIfEELm8EEEvv>:
    5040:	endbr64
    5044:	push   rbp
    5045:	pxor   xmm0,xmm0
    5049:	xor    ecx,ecx
    504b:	movabs rdi,0xaaaaaaaaaaaaaaab
    5055:	push   rbx
    5056:	sub    rsp,0x68
    505a:	movsd  xmm3,QWORD PTR [rip+0x7296]        # c2f8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xb8>
    5062:	movsd  xmm2,QWORD PTR [rip+0x725e]        # c2c8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x88>
    506a:	mov    rax,QWORD PTR fs:0x28
    5073:	mov    QWORD PTR [rsp+0x58],rax
    5078:	xor    eax,eax
    507a:	mov    rbp,rsp
    507d:	movaps XMMWORD PTR [rsp],xmm0
    5081:	mov    rbx,rsp
    5084:	mov    QWORD PTR [rsp+0x40],0x0
    508d:	mov    rsi,rsp
    5090:	mov    QWORD PTR [rsp+0x48],0x8
    5099:	movaps XMMWORD PTR [rsp+0x10],xmm0
    509e:	movaps XMMWORD PTR [rsp+0x20],xmm0
    50a3:	movaps XMMWORD PTR [rsp+0x30],xmm0
    50a8:	nop    DWORD PTR [rax+rax*1+0x0]
    50b0:	mov    rax,rcx
    50b3:	pxor   xmm0,xmm0
    50b7:	pxor   xmm1,xmm1
    50bb:	add    rsi,0x8
    50bf:	mul    rdi
    50c2:	mov    rax,rdx
    50c5:	and    rdx,0xfffffffffffffffe
    50c9:	shr    rax,1
    50cc:	add    rdx,rax
    50cf:	mov    rax,rcx
    50d2:	add    rcx,0x1
    50d6:	cvtsi2sd xmm0,rcx
    50db:	divsd  xmm0,xmm2
    50df:	sub    rax,rdx
    50e2:	add    rax,0x1
    50e6:	cvtsi2sd xmm1,rax
    50eb:	mulsd  xmm1,xmm3
    50ef:	unpcklpd xmm0,xmm1
    50f3:	cvtpd2ps xmm0,xmm0
    50f7:	movlps QWORD PTR [rsi-0x8],xmm0
    50fb:	cmp    rcx,0x8
    50ff:	jne    50b0 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIfEELm8EEEvv+0x70>
    5101:	mov    rdi,rbp
    5104:	call   4c40 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv>
    5109:	call   28a0 <instruction_count_begin>
    510e:	mov    rdi,rbp
    5111:	add    rbp,0x40
    5115:	call   4c40 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm8EE5applyEPv>
    511a:	call   28b0 <instruction_count_end>
    511f:	lea    rsi,[rip+0x6f16]        # c03c <_IO_stdin_used+0x3c>
    5126:	mov    edi,0x2
    512b:	xor    eax,eax
    512d:	call   10e0 <__printf_chk@plt>
    5132:	nop    DWORD PTR [rax]
    5135:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    5140:	pxor   xmm0,xmm0
    5144:	pxor   xmm1,xmm1
    5148:	mov    edi,0x2
    514d:	mov    eax,0x2
    5152:	cvtss2sd xmm0,DWORD PTR [rbx]
    5156:	cvtss2sd xmm1,DWORD PTR [rbx+0x4]
    515b:	lea    rsi,[rip+0x6ee8]        # c04a <_IO_stdin_used+0x4a>
    5162:	add    rbx,0x8
    5166:	call   10e0 <__printf_chk@plt>
    516b:	cmp    rbp,rbx
    516e:	jne    5140 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIfEELm8EEEvv+0x100>
    5170:	pxor   xmm0,xmm0
    5174:	mov    edi,0x2
    5179:	mov    eax,0x1
    517e:	lea    rsi,[rip+0x6ecc]        # c051 <_IO_stdin_used+0x51>
    5185:	cvtss2sd xmm0,DWORD PTR [rsp+0x40]
    518b:	call   10e0 <__printf_chk@plt>
    5190:	mov    rax,QWORD PTR [rsp+0x58]
    5195:	sub    rax,QWORD PTR fs:0x28
    519e:	jne    51a7 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIfEELm8EEEvv+0x167>
    51a0:	add    rsp,0x68
    51a4:	pop    rbx
    51a5:	pop    rbp
    51a6:	ret
    51a7:	call   10b0 <__stack_chk_fail@plt>
    51ac:	cs nop WORD PTR [rax+rax*1+0x0]
    51b6:	cs nop WORD PTR [rax+rax*1+0x0]

00000000000051c0 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv>:
    51c0:	endbr64
    51c4:	pxor   xmm4,xmm4
    51c8:	mov    rdx,QWORD PTR [rdi+0x88]
    51cf:	xor    eax,eax
    51d1:	movaps xmm2,xmm4
    51d4:	nop
    51d5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    51e0:	cmp    rax,rdx
    51e3:	jae    5234 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x74>
    51e5:	movss  xmm1,DWORD PTR [rdi+rax*8]
    51ea:	movaps xmm3,xmm1
    51ed:	movaps xmm0,xmm1
    51f0:	xorps  xmm0,XMMWORD PTR [rip+0x7059]        # c250 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    51f7:	cmpltss xmm1,xmm2
    51fc:	andps  xmm0,xmm1
    51ff:	andnps xmm1,xmm3
    5202:	orps   xmm1,xmm0
    5205:	movss  xmm0,DWORD PTR [rdi+rax*8+0x4]
    520b:	movaps xmm5,xmm0
    520e:	movaps xmm3,xmm0
    5211:	xorps  xmm3,XMMWORD PTR [rip+0x7038]        # c250 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    5218:	cmpltss xmm0,xmm2
    521d:	andps  xmm3,xmm0
    5220:	andnps xmm0,xmm5
    5223:	orps   xmm0,xmm3
    5226:	movaps xmm3,xmm1
    5229:	maxss  xmm3,xmm0
    522d:	maxss  xmm3,xmm4
    5231:	movaps xmm4,xmm3
    5234:	add    rax,0x1
    5238:	cmp    rax,0x10
    523c:	jne    51e0 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x20>
    523e:	pxor   xmm0,xmm0
    5242:	ucomiss xmm4,xmm0
    5245:	jp     5249 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x89>
    5247:	je     52c1 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x101>
    5249:	pxor   xmm2,xmm2
    524d:	xor    eax,eax
    524f:	nop    DWORD PTR [rax+rax*1+0x0]
    5254:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    525f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    526a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    5275:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    5280:	cmp    rax,rdx
    5283:	jae    52a8 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0xe8>
    5285:	movss  xmm0,DWORD PTR [rdi+rax*8]
    528a:	movss  xmm1,DWORD PTR [rdi+rax*8+0x4]
    5290:	divss  xmm0,xmm4
    5294:	divss  xmm1,xmm4
    5298:	mulss  xmm0,xmm0
    529c:	mulss  xmm1,xmm1
    52a0:	addss  xmm0,xmm1
    52a4:	addss  xmm2,xmm0
    52a8:	add    rax,0x1
    52ac:	cmp    rax,0x10
    52b0:	jne    5280 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0xc0>
    52b2:	pxor   xmm0,xmm0
    52b6:	ucomiss xmm2,xmm0
    52b9:	jp     52d0 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x110>
    52bb:	jne    52d0 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x110>
    52bd:	mulss  xmm0,xmm4
    52c1:	movss  DWORD PTR [rdi+0x80],xmm0
    52c9:	ret
    52ca:	nop    WORD PTR [rax+rax*1+0x0]
    52d0:	movss  xmm1,DWORD PTR [rip+0x6d30]        # c008 <_IO_stdin_used+0x8>
    52d8:	movaps xmm3,xmm1
    52db:	maxss  xmm3,xmm2
    52df:	maxss  xmm1,xmm3
    52e3:	movd   eax,xmm1
    52e7:	movaps xmm0,xmm1
    52ea:	mov    esi,eax
    52ec:	shr    esi,0x17
    52ef:	movzx  ecx,sil
    52f3:	cmp    ecx,0xff
    52f9:	je     547f <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x2bf>
    52ff:	test   ecx,ecx
    5301:	je     5404 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x244>
    5307:	and    eax,0x807fffff
    530c:	mov    edx,0x7e
    5311:	and    esi,0x1
    5314:	or     eax,0x3f000000
    5319:	sub    edx,ecx
    531b:	movd   xmm2,eax
    531f:	lea    rax,[rip+0x6f1a]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    5326:	sar    edx,1
    5328:	movss  xmm5,DWORD PTR [rax+rsi*4]
    532d:	mulss  xmm2,DWORD PTR [rip+0x6cd7]        # c00c <_IO_stdin_used+0xc>
    5335:	movss  xmm1,DWORD PTR [rip+0x6cd3]        # c010 <_IO_stdin_used+0x10>
    533d:	subss  xmm1,xmm2
    5341:	mulss  xmm1,xmm5
    5345:	movd   ecx,xmm1
    5349:	mov    eax,ecx
    534b:	shr    eax,0x17
    534e:	movzx  eax,al
    5351:	cmp    eax,0xff
    5356:	je     53a0 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x1e0>
    5358:	mov    esi,ecx
    535a:	mov    r9d,eax
    535d:	and    esi,0x7fffff
    5363:	or     r9d,esi
    5366:	je     53a0 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x1e0>
    5368:	mov    r8d,ecx
    536b:	and    r8d,0x80000000
    5372:	test   eax,eax
    5374:	je     5491 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x2d1>
    537a:	add    eax,edx
    537c:	je     5541 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x381>
    5382:	cmp    eax,0xfe
    5387:	jg     5541 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x381>
    538d:	shl    eax,0x17
    5390:	and    ecx,0x807fffff
    5396:	or     eax,ecx
    5398:	movd   xmm1,eax
    539c:	nop    DWORD PTR [rax+0x0]
    53a0:	movaps xmm2,xmm0
    53a3:	movss  xmm6,DWORD PTR [rip+0x6c69]        # c014 <_IO_stdin_used+0x14>
    53ab:	movss  xmm5,DWORD PTR [rip+0x6c65]        # c018 <_IO_stdin_used+0x18>
    53b3:	mulss  xmm2,xmm1
    53b7:	mulss  xmm2,xmm1
    53bb:	mulss  xmm6,xmm2
    53bf:	subss  xmm5,xmm6
    53c3:	mulss  xmm5,xmm2
    53c7:	movss  xmm2,DWORD PTR [rip+0x6ec5]        # c294 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x54>
    53cf:	subss  xmm2,xmm5
    53d3:	mulss  xmm2,xmm1
    53d7:	movss  xmm1,DWORD PTR [rip+0x6c3d]        # c01c <_IO_stdin_used+0x1c>
    53df:	mulss  xmm1,xmm0
    53e3:	movss  xmm0,DWORD PTR [rip+0x6ee9]        # c2d4 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x94>
    53eb:	mulss  xmm1,xmm2
    53ef:	mulss  xmm1,xmm2
    53f3:	subss  xmm0,xmm1
    53f7:	mulss  xmm0,xmm2
    53fb:	mulss  xmm0,xmm3
    53ff:	jmp    52bd <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0xfd>
    5404:	mov    esi,eax
    5406:	and    esi,0x7fffff
    540c:	je     547f <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x2bf>
    540e:	mov    r8d,eax
    5411:	and    r8d,0x80000000
    5418:	test   eax,0x7f0000
    541d:	mov    eax,esi
    541f:	je     5515 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x355>
    5425:	shl    eax,0x8
    5428:	mov    edx,0x8
    542d:	cmp    eax,0xfffffff
    5432:	jbe    552b <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x36b>
    5438:	cmp    eax,0x3fffffff
    543d:	ja     5449 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x289>
    543f:	add    edx,0x2
    5442:	shl    eax,0x2
    5445:	test   eax,eax
    5447:	js     544c <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x28c>
    5449:	add    edx,0x1
    544c:	lea    ecx,[rdx-0x9]
    544f:	mov    eax,esi
    5451:	sar    edx,1
    5453:	shl    eax,cl
    5455:	not    ecx
    5457:	add    edx,0x3a
    545a:	and    eax,0x7fffff
    545f:	and    ecx,0x1
    5462:	or     eax,r8d
    5465:	or     eax,0x3f000000
    546a:	movd   xmm2,eax
    546e:	lea    rax,[rip+0x6dcb]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    5475:	movss  xmm5,DWORD PTR [rax+rcx*4]
    547a:	jmp    532d <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x16d>
    547f:	movss  xmm5,DWORD PTR [rip+0x6b7d]        # c004 <_IO_stdin_used+0x4>
    5487:	movaps xmm2,xmm0
    548a:	xor    edx,edx
    548c:	jmp    532d <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x16d>
    5491:	and    ecx,0x7f0000
    5497:	mov    eax,esi
    5499:	je     54ec <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x32c>
    549b:	shl    eax,0x8
    549e:	mov    ecx,0x8
    54a3:	cmp    eax,0xfffffff
    54a8:	jbe    5506 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x346>
    54aa:	cmp    eax,0x3fffffff
    54af:	ja     54bf <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x2ff>
    54b1:	add    ecx,0x2
    54b4:	shl    eax,0x2
    54b7:	test   eax,eax
    54b9:	js     5567 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x3a7>
    54bf:	sub    ecx,0x8
    54c2:	mov    eax,edx
    54c4:	shl    esi,cl
    54c6:	sub    eax,ecx
    54c8:	and    esi,0x7fffff
    54ce:	add    eax,0x1
    54d1:	test   eax,eax
    54d3:	jle    556f <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x3af>
    54d9:	or     r8d,esi
    54dc:	shl    eax,0x17
    54df:	or     r8d,eax
    54e2:	movd   xmm1,r8d
    54e7:	jmp    53a0 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x1e0>
    54ec:	shl    eax,0x10
    54ef:	cmp    eax,0xffffff
    54f4:	jbe    559f <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x3df>
    54fa:	mov    ecx,0x10
    54ff:	cmp    eax,0xfffffff
    5504:	ja     550c <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x34c>
    5506:	add    ecx,0x4
    5509:	shl    eax,0x4
    550c:	cmp    eax,0x3fffffff
    5511:	ja     54b7 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x2f7>
    5513:	jmp    54b1 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x2f1>
    5515:	shl    eax,0x10
    5518:	cmp    eax,0xffffff
    551d:	jbe    5593 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x3d3>
    551f:	mov    edx,0x10
    5524:	cmp    eax,0xfffffff
    5529:	ja     5531 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x371>
    552b:	add    edx,0x4
    552e:	shl    eax,0x4
    5531:	cmp    eax,0x3fffffff
    5536:	ja     5445 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x285>
    553c:	jmp    543f <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x27f>
    5541:	shr    esi,1
    5543:	mov    edx,esi
    5545:	or     edx,r8d
    5548:	or     r8d,0x7f800000
    554f:	or     edx,0x400000
    5555:	cmp    eax,0xfe
    555a:	cmovg  edx,r8d
    555e:	movd   xmm1,edx
    5562:	jmp    53a0 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x1e0>
    5567:	sub    ecx,0x9
    556a:	jmp    54c2 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x302>
    556f:	movd   xmm1,r8d
    5574:	cmp    eax,0xffffffea
    5577:	jl     53a0 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x1e0>
    557d:	or     esi,0x800000
    5583:	sub    ecx,edx
    5585:	shr    esi,cl
    5587:	or     esi,r8d
    558a:	movd   xmm1,esi
    558e:	jmp    53a0 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x1e0>
    5593:	mov    eax,esi
    5595:	mov    edx,0x18
    559a:	shl    eax,0x18
    559d:	jmp    5524 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x364>
    559f:	mov    eax,esi
    55a1:	mov    ecx,0x18
    55a6:	shl    eax,0x18
    55a9:	jmp    54ff <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv+0x33f>
    55ae:	xchg   ax,ax

00000000000055b0 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIfEELm16EEEvv>:
    55b0:	endbr64
    55b4:	push   r12
    55b6:	pxor   xmm0,xmm0
    55ba:	push   rbp
    55bb:	push   rbx
    55bc:	sub    rsp,0xa0
    55c3:	mov    rbp,QWORD PTR fs:0x28
    55cc:	mov    QWORD PTR [rsp+0x98],rbp
    55d4:	mov    rbp,rsp
    55d7:	lea    rbx,[rsp+0x80]
    55df:	mov    rax,rsp
    55e2:	movaps XMMWORD PTR [rax],xmm0
    55e5:	add    rax,0x20
    55e9:	movaps XMMWORD PTR [rax-0x10],xmm0
    55ed:	cmp    rax,rbx
    55f0:	jne    55e2 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIfEELm16EEEvv+0x32>
    55f2:	mov    DWORD PTR [rsp+0x80],0x0
    55fd:	mov    rsi,rsp
    5600:	xor    ecx,ecx
    5602:	movsd  xmm3,QWORD PTR [rip+0x6cee]        # c2f8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xb8>
    560a:	mov    QWORD PTR [rsp+0x88],0x10
    5616:	movsd  xmm2,QWORD PTR [rip+0x6cca]        # c2e8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
    561e:	movabs rdi,0xaaaaaaaaaaaaaaab
    5628:	nop    DWORD PTR [rax+rax*1+0x0]
    5630:	mov    rax,rcx
    5633:	pxor   xmm0,xmm0
    5637:	pxor   xmm1,xmm1
    563b:	add    rsi,0x8
    563f:	mul    rdi
    5642:	mov    rax,rdx
    5645:	and    rdx,0xfffffffffffffffe
    5649:	shr    rax,1
    564c:	add    rdx,rax
    564f:	mov    rax,rcx
    5652:	add    rcx,0x1
    5656:	cvtsi2sd xmm0,rcx
    565b:	divsd  xmm0,xmm2
    565f:	sub    rax,rdx
    5662:	add    rax,0x1
    5666:	cvtsi2sd xmm1,rax
    566b:	mulsd  xmm1,xmm3
    566f:	unpcklpd xmm0,xmm1
    5673:	cvtpd2ps xmm0,xmm0
    5677:	movlps QWORD PTR [rsi-0x8],xmm0
    567b:	cmp    rcx,0x10
    567f:	jne    5630 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIfEELm16EEEvv+0x80>
    5681:	mov    rdi,rsp
    5684:	call   51c0 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv>
    5689:	call   28a0 <instruction_count_begin>
    568e:	mov    rdi,rsp
    5691:	call   51c0 <_ZN8NormCaseIN4Base6Matrix7ComplexIfEELm16EE5applyEPv>
    5696:	call   28b0 <instruction_count_end>
    569b:	lea    rsi,[rip+0x699a]        # c03c <_IO_stdin_used+0x3c>
    56a2:	mov    edi,0x2
    56a7:	xor    eax,eax
    56a9:	call   10e0 <__printf_chk@plt>
    56ae:	xchg   ax,ax
    56b0:	pxor   xmm0,xmm0
    56b4:	pxor   xmm1,xmm1
    56b8:	mov    edi,0x2
    56bd:	mov    eax,0x2
    56c2:	cvtss2sd xmm0,DWORD PTR [rbp+0x0]
    56c7:	cvtss2sd xmm1,DWORD PTR [rbp+0x4]
    56cc:	lea    rsi,[rip+0x6977]        # c04a <_IO_stdin_used+0x4a>
    56d3:	add    rbp,0x8
    56d7:	call   10e0 <__printf_chk@plt>
    56dc:	cmp    rbp,rbx
    56df:	jne    56b0 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIfEELm16EEEvv+0x100>
    56e1:	pxor   xmm0,xmm0
    56e5:	mov    edi,0x2
    56ea:	mov    eax,0x1
    56ef:	cvtss2sd xmm0,DWORD PTR [rsp+0x80]
    56f8:	lea    rsi,[rip+0x6952]        # c051 <_IO_stdin_used+0x51>
    56ff:	call   10e0 <__printf_chk@plt>
    5704:	mov    rax,QWORD PTR [rsp+0x98]
    570c:	sub    rax,QWORD PTR fs:0x28
    5715:	jne    5723 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIfEELm16EEEvv+0x173>
    5717:	add    rsp,0xa0
    571e:	pop    rbx
    571f:	pop    rbp
    5720:	pop    r12
    5722:	ret
    5723:	call   10b0 <__stack_chk_fail@plt>
    5728:	cs nop WORD PTR [rax+rax*1+0x0]
    5732:	cs nop WORD PTR [rax+rax*1+0x0]
    573c:	nop    DWORD PTR [rax+0x0]

0000000000005740 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv>:
    5740:	endbr64
    5744:	pxor   xmm4,xmm4
    5748:	mov    rsi,QWORD PTR [rdi+0x88]
    574f:	mov    rcx,rdi
    5752:	xor    eax,eax
    5754:	mov    rdx,rdi
    5757:	movapd xmm2,xmm4
    575b:	nop    DWORD PTR [rax+rax*1+0x0]
    5760:	cmp    rax,rsi
    5763:	jae    57c0 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x80>
    5765:	movsd  xmm1,QWORD PTR [rdx]
    5769:	movapd xmm3,xmm1
    576d:	movapd xmm0,xmm1
    5771:	xorpd  xmm0,XMMWORD PTR [rip+0x6ae7]        # c260 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    5779:	cmpltsd xmm1,xmm2
    577e:	andpd  xmm0,xmm1
    5782:	andnpd xmm1,xmm3
    5786:	orpd   xmm1,xmm0
    578a:	movsd  xmm0,QWORD PTR [rdx+0x8]
    578f:	movapd xmm5,xmm0
    5793:	movapd xmm3,xmm0
    5797:	xorpd  xmm3,XMMWORD PTR [rip+0x6ac1]        # c260 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    579f:	cmpltsd xmm0,xmm2
    57a4:	andpd  xmm3,xmm0
    57a8:	andnpd xmm0,xmm5
    57ac:	orpd   xmm0,xmm3
    57b0:	movapd xmm3,xmm1
    57b4:	maxsd  xmm3,xmm0
    57b8:	maxsd  xmm3,xmm4
    57bc:	movapd xmm4,xmm3
    57c0:	add    rax,0x1
    57c4:	add    rdx,0x10
    57c8:	cmp    rax,0x8
    57cc:	jne    5760 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x20>
    57ce:	pxor   xmm0,xmm0
    57d2:	ucomisd xmm4,xmm0
    57d6:	jp     57da <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x9a>
    57d8:	je     5844 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x104>
    57da:	pxor   xmm2,xmm2
    57de:	xor    eax,eax
    57e0:	cs nop WORD PTR [rax+rax*1+0x0]
    57ea:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    57f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    5800:	cmp    rax,rsi
    5803:	jae    5826 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0xe6>
    5805:	movsd  xmm0,QWORD PTR [rcx]
    5809:	movsd  xmm1,QWORD PTR [rcx+0x8]
    580e:	divsd  xmm0,xmm4
    5812:	divsd  xmm1,xmm4
    5816:	mulsd  xmm0,xmm0
    581a:	mulsd  xmm1,xmm1
    581e:	addsd  xmm0,xmm1
    5822:	addsd  xmm2,xmm0
    5826:	add    rax,0x1
    582a:	add    rcx,0x10
    582e:	cmp    rax,0x8
    5832:	jne    5800 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0xc0>
    5834:	pxor   xmm0,xmm0
    5838:	ucomisd xmm2,xmm0
    583c:	jp     5850 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x110>
    583e:	jne    5850 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x110>
    5840:	mulsd  xmm0,xmm4
    5844:	movsd  QWORD PTR [rdi+0x80],xmm0
    584c:	ret
    584d:	nop    DWORD PTR [rax]
    5850:	movsd  xmm0,QWORD PTR [rip+0x6a40]        # c298 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x58>
    5858:	movapd xmm3,xmm0
    585c:	movapd xmm1,xmm0
    5860:	maxsd  xmm3,xmm2
    5864:	pxor   xmm2,xmm2
    5868:	maxsd  xmm1,xmm3
    586c:	cvtsd2ss xmm2,xmm1
    5870:	movd   eax,xmm2
    5874:	mov    esi,eax
    5876:	shr    esi,0x17
    5879:	movzx  ecx,sil
    587d:	cmp    ecx,0xff
    5883:	je     5a18 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x2d8>
    5889:	test   ecx,ecx
    588b:	je     5998 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x258>
    5891:	and    eax,0x807fffff
    5896:	mov    edx,0x7e
    589b:	and    esi,0x1
    589e:	or     eax,0x3f000000
    58a3:	sub    edx,ecx
    58a5:	movd   xmm2,eax
    58a9:	lea    rax,[rip+0x6990]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    58b0:	sar    edx,1
    58b2:	movss  xmm5,DWORD PTR [rax+rsi*4]
    58b7:	mulss  xmm2,DWORD PTR [rip+0x674d]        # c00c <_IO_stdin_used+0xc>
    58bf:	movss  xmm0,DWORD PTR [rip+0x6749]        # c010 <_IO_stdin_used+0x10>
    58c7:	subss  xmm0,xmm2
    58cb:	mulss  xmm0,xmm5
    58cf:	movd   ecx,xmm0
    58d3:	mov    eax,ecx
    58d5:	shr    eax,0x17
    58d8:	movzx  eax,al
    58db:	cmp    eax,0xff
    58e0:	je     5930 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x1f0>
    58e2:	mov    esi,ecx
    58e4:	mov    r9d,eax
    58e7:	and    esi,0x7fffff
    58ed:	or     r9d,esi
    58f0:	je     5930 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x1f0>
    58f2:	mov    r8d,ecx
    58f5:	and    r8d,0x80000000
    58fc:	test   eax,eax
    58fe:	je     5a27 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x2e7>
    5904:	add    eax,edx
    5906:	je     5ad7 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x397>
    590c:	cmp    eax,0xfe
    5911:	jg     5ad7 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x397>
    5917:	shl    eax,0x17
    591a:	and    ecx,0x807fffff
    5920:	or     eax,ecx
    5922:	movd   xmm0,eax
    5926:	cs nop WORD PTR [rax+rax*1+0x0]
    5930:	movapd xmm2,xmm1
    5934:	cvtss2sd xmm0,xmm0
    5938:	movsd  xmm6,QWORD PTR [rip+0x6960]        # c2a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x60>
    5940:	movsd  xmm5,QWORD PTR [rip+0x6960]        # c2a8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x68>
    5948:	mulsd  xmm2,xmm0
    594c:	mulsd  xmm1,QWORD PTR [rip+0x6964]        # c2b8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x78>
    5954:	mulsd  xmm2,xmm0
    5958:	mulsd  xmm6,xmm2
    595c:	subsd  xmm5,xmm6
    5960:	mulsd  xmm5,xmm2
    5964:	movsd  xmm2,QWORD PTR [rip+0x6944]        # c2b0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x70>
    596c:	subsd  xmm2,xmm5
    5970:	mulsd  xmm2,xmm0
    5974:	movsd  xmm0,QWORD PTR [rip+0x6944]        # c2c0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
    597c:	mulsd  xmm1,xmm2
    5980:	mulsd  xmm1,xmm2
    5984:	subsd  xmm0,xmm1
    5988:	mulsd  xmm0,xmm2
    598c:	mulsd  xmm0,xmm3
    5990:	jmp    5840 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x100>
    5995:	nop    DWORD PTR [rax]
    5998:	mov    esi,eax
    599a:	and    esi,0x7fffff
    59a0:	je     5a18 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x2d8>
    59a2:	mov    r8d,eax
    59a5:	and    r8d,0x80000000
    59ac:	test   eax,0x7f0000
    59b1:	mov    eax,esi
    59b3:	je     5aab <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x36b>
    59b9:	shl    eax,0x8
    59bc:	mov    edx,0x8
    59c1:	cmp    eax,0xfffffff
    59c6:	jbe    5ac1 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x381>
    59cc:	cmp    eax,0x3fffffff
    59d1:	ja     59dd <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x29d>
    59d3:	add    edx,0x2
    59d6:	shl    eax,0x2
    59d9:	test   eax,eax
    59db:	js     59e0 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x2a0>
    59dd:	add    edx,0x1
    59e0:	lea    ecx,[rdx-0x9]
    59e3:	lea    rax,[rip+0x6856]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    59ea:	sar    edx,1
    59ec:	shl    esi,cl
    59ee:	not    ecx
    59f0:	add    edx,0x3a
    59f3:	and    esi,0x7fffff
    59f9:	and    ecx,0x1
    59fc:	or     esi,r8d
    59ff:	movss  xmm5,DWORD PTR [rax+rcx*4]
    5a04:	or     esi,0x3f000000
    5a0a:	movd   xmm2,esi
    5a0e:	jmp    58b7 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x177>
    5a13:	nop    DWORD PTR [rax+rax*1+0x0]
    5a18:	movss  xmm5,DWORD PTR [rip+0x65e4]        # c004 <_IO_stdin_used+0x4>
    5a20:	xor    edx,edx
    5a22:	jmp    58b7 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x177>
    5a27:	and    ecx,0x7f0000
    5a2d:	mov    eax,esi
    5a2f:	je     5a82 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x342>
    5a31:	shl    eax,0x8
    5a34:	mov    ecx,0x8
    5a39:	cmp    eax,0xfffffff
    5a3e:	jbe    5a9c <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x35c>
    5a40:	cmp    eax,0x3fffffff
    5a45:	ja     5a55 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x315>
    5a47:	add    ecx,0x2
    5a4a:	shl    eax,0x2
    5a4d:	test   eax,eax
    5a4f:	js     5afd <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x3bd>
    5a55:	sub    ecx,0x8
    5a58:	mov    eax,edx
    5a5a:	shl    esi,cl
    5a5c:	sub    eax,ecx
    5a5e:	and    esi,0x7fffff
    5a64:	add    eax,0x1
    5a67:	test   eax,eax
    5a69:	jle    5b05 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x3c5>
    5a6f:	or     r8d,esi
    5a72:	shl    eax,0x17
    5a75:	or     r8d,eax
    5a78:	movd   xmm0,r8d
    5a7d:	jmp    5930 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x1f0>
    5a82:	shl    eax,0x10
    5a85:	mov    ecx,0x10
    5a8a:	cmp    eax,0xffffff
    5a8f:	jbe    5b35 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x3f5>
    5a95:	cmp    eax,0xfffffff
    5a9a:	ja     5aa2 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x362>
    5a9c:	add    ecx,0x4
    5a9f:	shl    eax,0x4
    5aa2:	cmp    eax,0x3fffffff
    5aa7:	ja     5a4d <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x30d>
    5aa9:	jmp    5a47 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x307>
    5aab:	shl    eax,0x10
    5aae:	mov    edx,0x10
    5ab3:	cmp    eax,0xffffff
    5ab8:	jbe    5b29 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x3e9>
    5aba:	cmp    eax,0xfffffff
    5abf:	ja     5ac7 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x387>
    5ac1:	add    edx,0x4
    5ac4:	shl    eax,0x4
    5ac7:	cmp    eax,0x3fffffff
    5acc:	ja     59d9 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x299>
    5ad2:	jmp    59d3 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x293>
    5ad7:	mov    edx,esi
    5ad9:	shr    edx,1
    5adb:	or     edx,r8d
    5ade:	or     r8d,0x7f800000
    5ae5:	or     edx,0x400000
    5aeb:	cmp    eax,0xfe
    5af0:	cmovg  edx,r8d
    5af4:	movd   xmm0,edx
    5af8:	jmp    5930 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x1f0>
    5afd:	sub    ecx,0x9
    5b00:	jmp    5a58 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x318>
    5b05:	movd   xmm0,r8d
    5b0a:	cmp    eax,0xffffffea
    5b0d:	jl     5930 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x1f0>
    5b13:	or     esi,0x800000
    5b19:	sub    ecx,edx
    5b1b:	shr    esi,cl
    5b1d:	or     esi,r8d
    5b20:	movd   xmm0,esi
    5b24:	jmp    5930 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x1f0>
    5b29:	mov    eax,esi
    5b2b:	mov    edx,0x18
    5b30:	shl    eax,0x18
    5b33:	jmp    5aba <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x37a>
    5b35:	mov    eax,esi
    5b37:	mov    ecx,0x18
    5b3c:	shl    eax,0x18
    5b3f:	jmp    5a95 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv+0x355>
    5b44:	cs nop WORD PTR [rax+rax*1+0x0]
    5b4e:	xchg   ax,ax

0000000000005b50 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIdEELm8EEEvv>:
    5b50:	endbr64
    5b54:	push   r12
    5b56:	pxor   xmm0,xmm0
    5b5a:	push   rbp
    5b5b:	push   rbx
    5b5c:	sub    rsp,0xa0
    5b63:	mov    rbp,QWORD PTR fs:0x28
    5b6c:	mov    QWORD PTR [rsp+0x98],rbp
    5b74:	mov    rbp,rsp
    5b77:	lea    rbx,[rsp+0x80]
    5b7f:	mov    rax,rsp
    5b82:	movaps XMMWORD PTR [rax],xmm0
    5b85:	add    rax,0x20
    5b89:	movaps XMMWORD PTR [rax-0x10],xmm0
    5b8d:	cmp    rax,rbx
    5b90:	jne    5b82 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0x32>
    5b92:	movsd  xmm3,QWORD PTR [rip+0x675e]        # c2f8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xb8>
    5b9a:	mov    rsi,rsp
    5b9d:	xor    ecx,ecx
    5b9f:	mov    QWORD PTR [rsp+0x80],0x0
    5bab:	mov    QWORD PTR [rsp+0x88],0x8
    5bb7:	movsd  xmm2,QWORD PTR [rip+0x6709]        # c2c8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x88>
    5bbf:	movabs rdi,0xaaaaaaaaaaaaaaab
    5bc9:	nop    DWORD PTR [rax+0x0]
    5bd0:	mov    rax,rcx
    5bd3:	pxor   xmm0,xmm0
    5bd7:	pxor   xmm1,xmm1
    5bdb:	add    rsi,0x10
    5bdf:	mul    rdi
    5be2:	mov    rax,rdx
    5be5:	and    rdx,0xfffffffffffffffe
    5be9:	shr    rax,1
    5bec:	add    rdx,rax
    5bef:	mov    rax,rcx
    5bf2:	add    rcx,0x1
    5bf6:	cvtsi2sd xmm0,rcx
    5bfb:	divsd  xmm0,xmm2
    5bff:	sub    rax,rdx
    5c02:	add    rax,0x1
    5c06:	cvtsi2sd xmm1,rax
    5c0b:	mulsd  xmm1,xmm3
    5c0f:	unpcklpd xmm0,xmm1
    5c13:	movaps XMMWORD PTR [rsi-0x10],xmm0
    5c17:	cmp    rcx,0x8
    5c1b:	jne    5bd0 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0x80>
    5c1d:	mov    rdi,rsp
    5c20:	call   5740 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv>
    5c25:	call   28a0 <instruction_count_begin>
    5c2a:	mov    rdi,rsp
    5c2d:	call   5740 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm8EE5applyEPv>
    5c32:	call   28b0 <instruction_count_end>
    5c37:	lea    rsi,[rip+0x63fe]        # c03c <_IO_stdin_used+0x3c>
    5c3e:	mov    edi,0x2
    5c43:	xor    eax,eax
    5c45:	call   10e0 <__printf_chk@plt>
    5c4a:	nop    WORD PTR [rax+rax*1+0x0]
    5c50:	movsd  xmm0,QWORD PTR [rbp+0x0]
    5c55:	movsd  xmm1,QWORD PTR [rbp+0x8]
    5c5a:	mov    edi,0x2
    5c5f:	lea    rsi,[rip+0x63e4]        # c04a <_IO_stdin_used+0x4a>
    5c66:	mov    eax,0x2
    5c6b:	add    rbp,0x10
    5c6f:	call   10e0 <__printf_chk@plt>
    5c74:	cmp    rbp,rbx
    5c77:	jne    5c50 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0x100>
    5c79:	movsd  xmm0,QWORD PTR [rsp+0x80]
    5c82:	mov    edi,0x2
    5c87:	mov    eax,0x1
    5c8c:	lea    rsi,[rip+0x63be]        # c051 <_IO_stdin_used+0x51>
    5c93:	call   10e0 <__printf_chk@plt>
    5c98:	mov    rax,QWORD PTR [rsp+0x98]
    5ca0:	sub    rax,QWORD PTR fs:0x28
    5ca9:	jne    5cb7 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIdEELm8EEEvv+0x167>
    5cab:	add    rsp,0xa0
    5cb2:	pop    rbx
    5cb3:	pop    rbp
    5cb4:	pop    r12
    5cb6:	ret
    5cb7:	call   10b0 <__stack_chk_fail@plt>
    5cbc:	nop    DWORD PTR [rax+0x0]

0000000000005cc0 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv>:
    5cc0:	endbr64
    5cc4:	pxor   xmm4,xmm4
    5cc8:	mov    rsi,QWORD PTR [rdi+0x108]
    5ccf:	mov    rcx,rdi
    5cd2:	xor    eax,eax
    5cd4:	mov    rdx,rdi
    5cd7:	movapd xmm2,xmm4
    5cdb:	nop    DWORD PTR [rax+rax*1+0x0]
    5ce0:	cmp    rax,rsi
    5ce3:	jae    5d40 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x80>
    5ce5:	movsd  xmm1,QWORD PTR [rdx]
    5ce9:	movapd xmm3,xmm1
    5ced:	movapd xmm0,xmm1
    5cf1:	xorpd  xmm0,XMMWORD PTR [rip+0x6567]        # c260 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    5cf9:	cmpltsd xmm1,xmm2
    5cfe:	andpd  xmm0,xmm1
    5d02:	andnpd xmm1,xmm3
    5d06:	orpd   xmm1,xmm0
    5d0a:	movsd  xmm0,QWORD PTR [rdx+0x8]
    5d0f:	movapd xmm5,xmm0
    5d13:	movapd xmm3,xmm0
    5d17:	xorpd  xmm3,XMMWORD PTR [rip+0x6541]        # c260 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    5d1f:	cmpltsd xmm0,xmm2
    5d24:	andpd  xmm3,xmm0
    5d28:	andnpd xmm0,xmm5
    5d2c:	orpd   xmm0,xmm3
    5d30:	movapd xmm3,xmm1
    5d34:	maxsd  xmm3,xmm0
    5d38:	maxsd  xmm3,xmm4
    5d3c:	movapd xmm4,xmm3
    5d40:	add    rax,0x1
    5d44:	add    rdx,0x10
    5d48:	cmp    rax,0x10
    5d4c:	jne    5ce0 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x20>
    5d4e:	pxor   xmm0,xmm0
    5d52:	ucomisd xmm4,xmm0
    5d56:	jp     5d5a <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x9a>
    5d58:	je     5dc4 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x104>
    5d5a:	pxor   xmm2,xmm2
    5d5e:	xor    eax,eax
    5d60:	cs nop WORD PTR [rax+rax*1+0x0]
    5d6a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    5d75:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    5d80:	cmp    rax,rsi
    5d83:	jae    5da6 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0xe6>
    5d85:	movsd  xmm0,QWORD PTR [rcx]
    5d89:	movsd  xmm1,QWORD PTR [rcx+0x8]
    5d8e:	divsd  xmm0,xmm4
    5d92:	divsd  xmm1,xmm4
    5d96:	mulsd  xmm0,xmm0
    5d9a:	mulsd  xmm1,xmm1
    5d9e:	addsd  xmm0,xmm1
    5da2:	addsd  xmm2,xmm0
    5da6:	add    rax,0x1
    5daa:	add    rcx,0x10
    5dae:	cmp    rax,0x10
    5db2:	jne    5d80 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0xc0>
    5db4:	pxor   xmm0,xmm0
    5db8:	ucomisd xmm2,xmm0
    5dbc:	jp     5dd0 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x110>
    5dbe:	jne    5dd0 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x110>
    5dc0:	mulsd  xmm0,xmm4
    5dc4:	movsd  QWORD PTR [rdi+0x100],xmm0
    5dcc:	ret
    5dcd:	nop    DWORD PTR [rax]
    5dd0:	movsd  xmm0,QWORD PTR [rip+0x64c0]        # c298 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x58>
    5dd8:	movapd xmm3,xmm0
    5ddc:	movapd xmm1,xmm0
    5de0:	maxsd  xmm3,xmm2
    5de4:	pxor   xmm2,xmm2
    5de8:	maxsd  xmm1,xmm3
    5dec:	cvtsd2ss xmm2,xmm1
    5df0:	movd   eax,xmm2
    5df4:	mov    esi,eax
    5df6:	shr    esi,0x17
    5df9:	movzx  ecx,sil
    5dfd:	cmp    ecx,0xff
    5e03:	je     5f90 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x2d0>
    5e09:	test   ecx,ecx
    5e0b:	je     5f15 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x255>
    5e11:	and    eax,0x807fffff
    5e16:	mov    edx,0x7e
    5e1b:	and    esi,0x1
    5e1e:	or     eax,0x3f000000
    5e23:	sub    edx,ecx
    5e25:	movd   xmm2,eax
    5e29:	lea    rax,[rip+0x6410]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    5e30:	sar    edx,1
    5e32:	movss  xmm5,DWORD PTR [rax+rsi*4]
    5e37:	mulss  xmm2,DWORD PTR [rip+0x61cd]        # c00c <_IO_stdin_used+0xc>
    5e3f:	movss  xmm0,DWORD PTR [rip+0x61c9]        # c010 <_IO_stdin_used+0x10>
    5e47:	subss  xmm0,xmm2
    5e4b:	mulss  xmm0,xmm5
    5e4f:	movd   ecx,xmm0
    5e53:	mov    eax,ecx
    5e55:	shr    eax,0x17
    5e58:	movzx  eax,al
    5e5b:	cmp    eax,0xff
    5e60:	je     5eb0 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x1f0>
    5e62:	mov    esi,ecx
    5e64:	mov    r9d,eax
    5e67:	and    esi,0x7fffff
    5e6d:	or     r9d,esi
    5e70:	je     5eb0 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x1f0>
    5e72:	mov    r8d,ecx
    5e75:	and    r8d,0x80000000
    5e7c:	test   eax,eax
    5e7e:	je     5f9f <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x2df>
    5e84:	add    eax,edx
    5e86:	je     604f <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x38f>
    5e8c:	cmp    eax,0xfe
    5e91:	jg     604f <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x38f>
    5e97:	shl    eax,0x17
    5e9a:	and    ecx,0x807fffff
    5ea0:	or     eax,ecx
    5ea2:	movd   xmm0,eax
    5ea6:	cs nop WORD PTR [rax+rax*1+0x0]
    5eb0:	movapd xmm2,xmm1
    5eb4:	cvtss2sd xmm0,xmm0
    5eb8:	movsd  xmm6,QWORD PTR [rip+0x63e0]        # c2a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x60>
    5ec0:	movsd  xmm5,QWORD PTR [rip+0x63e0]        # c2a8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x68>
    5ec8:	mulsd  xmm2,xmm0
    5ecc:	mulsd  xmm1,QWORD PTR [rip+0x63e4]        # c2b8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x78>
    5ed4:	mulsd  xmm2,xmm0
    5ed8:	mulsd  xmm6,xmm2
    5edc:	subsd  xmm5,xmm6
    5ee0:	mulsd  xmm5,xmm2
    5ee4:	movsd  xmm2,QWORD PTR [rip+0x63c4]        # c2b0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x70>
    5eec:	subsd  xmm2,xmm5
    5ef0:	mulsd  xmm2,xmm0
    5ef4:	movsd  xmm0,QWORD PTR [rip+0x63c4]        # c2c0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
    5efc:	mulsd  xmm1,xmm2
    5f00:	mulsd  xmm1,xmm2
    5f04:	subsd  xmm0,xmm1
    5f08:	mulsd  xmm0,xmm2
    5f0c:	mulsd  xmm0,xmm3
    5f10:	jmp    5dc0 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x100>
    5f15:	mov    esi,eax
    5f17:	and    esi,0x7fffff
    5f1d:	je     5f90 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x2d0>
    5f1f:	mov    r8d,eax
    5f22:	and    r8d,0x80000000
    5f29:	test   eax,0x7f0000
    5f2e:	mov    eax,esi
    5f30:	je     6023 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x363>
    5f36:	shl    eax,0x8
    5f39:	mov    edx,0x8
    5f3e:	cmp    eax,0xfffffff
    5f43:	jbe    6039 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x379>
    5f49:	cmp    eax,0x3fffffff
    5f4e:	ja     5f5a <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x29a>
    5f50:	add    edx,0x2
    5f53:	shl    eax,0x2
    5f56:	test   eax,eax
    5f58:	js     5f5d <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x29d>
    5f5a:	add    edx,0x1
    5f5d:	lea    ecx,[rdx-0x9]
    5f60:	mov    eax,esi
    5f62:	sar    edx,1
    5f64:	shl    eax,cl
    5f66:	not    ecx
    5f68:	add    edx,0x3a
    5f6b:	and    eax,0x7fffff
    5f70:	and    ecx,0x1
    5f73:	or     eax,r8d
    5f76:	or     eax,0x3f000000
    5f7b:	movd   xmm2,eax
    5f7f:	lea    rax,[rip+0x62ba]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    5f86:	movss  xmm5,DWORD PTR [rax+rcx*4]
    5f8b:	jmp    5e37 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x177>
    5f90:	movss  xmm5,DWORD PTR [rip+0x606c]        # c004 <_IO_stdin_used+0x4>
    5f98:	xor    edx,edx
    5f9a:	jmp    5e37 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x177>
    5f9f:	and    ecx,0x7f0000
    5fa5:	mov    eax,esi
    5fa7:	je     5ffa <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x33a>
    5fa9:	shl    eax,0x8
    5fac:	mov    ecx,0x8
    5fb1:	cmp    eax,0xfffffff
    5fb6:	jbe    6014 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x354>
    5fb8:	cmp    eax,0x3fffffff
    5fbd:	ja     5fcd <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x30d>
    5fbf:	add    ecx,0x2
    5fc2:	shl    eax,0x2
    5fc5:	test   eax,eax
    5fc7:	js     6075 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x3b5>
    5fcd:	sub    ecx,0x8
    5fd0:	mov    eax,edx
    5fd2:	shl    esi,cl
    5fd4:	sub    eax,ecx
    5fd6:	and    esi,0x7fffff
    5fdc:	add    eax,0x1
    5fdf:	test   eax,eax
    5fe1:	jle    607d <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x3bd>
    5fe7:	or     r8d,esi
    5fea:	shl    eax,0x17
    5fed:	or     r8d,eax
    5ff0:	movd   xmm0,r8d
    5ff5:	jmp    5eb0 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x1f0>
    5ffa:	shl    eax,0x10
    5ffd:	cmp    eax,0xffffff
    6002:	jbe    60ad <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x3ed>
    6008:	mov    ecx,0x10
    600d:	cmp    eax,0xfffffff
    6012:	ja     601a <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x35a>
    6014:	add    ecx,0x4
    6017:	shl    eax,0x4
    601a:	cmp    eax,0x3fffffff
    601f:	ja     5fc5 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x305>
    6021:	jmp    5fbf <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x2ff>
    6023:	shl    eax,0x10
    6026:	cmp    eax,0xffffff
    602b:	jbe    60a1 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x3e1>
    602d:	mov    edx,0x10
    6032:	cmp    eax,0xfffffff
    6037:	ja     603f <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x37f>
    6039:	add    edx,0x4
    603c:	shl    eax,0x4
    603f:	cmp    eax,0x3fffffff
    6044:	ja     5f56 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x296>
    604a:	jmp    5f50 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x290>
    604f:	shr    esi,1
    6051:	mov    edx,esi
    6053:	or     edx,r8d
    6056:	or     r8d,0x7f800000
    605d:	or     edx,0x400000
    6063:	cmp    eax,0xfe
    6068:	cmovg  edx,r8d
    606c:	movd   xmm0,edx
    6070:	jmp    5eb0 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x1f0>
    6075:	sub    ecx,0x9
    6078:	jmp    5fd0 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x310>
    607d:	movd   xmm0,r8d
    6082:	cmp    eax,0xffffffea
    6085:	jl     5eb0 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x1f0>
    608b:	or     esi,0x800000
    6091:	sub    ecx,edx
    6093:	shr    esi,cl
    6095:	or     esi,r8d
    6098:	movd   xmm0,esi
    609c:	jmp    5eb0 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x1f0>
    60a1:	mov    eax,esi
    60a3:	mov    edx,0x18
    60a8:	shl    eax,0x18
    60ab:	jmp    6032 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x372>
    60ad:	mov    eax,esi
    60af:	mov    ecx,0x18
    60b4:	shl    eax,0x18
    60b7:	jmp    600d <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv+0x34d>
    60bc:	nop    DWORD PTR [rax+0x0]

00000000000060c0 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIdEELm16EEEvv>:
    60c0:	endbr64
    60c4:	push   r12
    60c6:	pxor   xmm0,xmm0
    60ca:	push   rbp
    60cb:	push   rbx
    60cc:	sub    rsp,0x120
    60d3:	mov    rbp,QWORD PTR fs:0x28
    60dc:	mov    QWORD PTR [rsp+0x118],rbp
    60e4:	mov    rbp,rsp
    60e7:	lea    rbx,[rsp+0x100]
    60ef:	mov    rax,rsp
    60f2:	nop    DWORD PTR [rax]
    60f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    6100:	movaps XMMWORD PTR [rax],xmm0
    6103:	add    rax,0x20
    6107:	movaps XMMWORD PTR [rax-0x10],xmm0
    610b:	cmp    rax,rbx
    610e:	jne    6100 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0x40>
    6110:	movsd  xmm3,QWORD PTR [rip+0x61e0]        # c2f8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xb8>
    6118:	mov    rsi,rsp
    611b:	xor    ecx,ecx
    611d:	mov    QWORD PTR [rsp+0x100],0x0
    6129:	mov    QWORD PTR [rsp+0x108],0x10
    6135:	movsd  xmm2,QWORD PTR [rip+0x61ab]        # c2e8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
    613d:	movabs rdi,0xaaaaaaaaaaaaaaab
    6147:	nop    WORD PTR [rax+rax*1+0x0]
    6150:	mov    rax,rcx
    6153:	pxor   xmm0,xmm0
    6157:	pxor   xmm1,xmm1
    615b:	add    rsi,0x10
    615f:	mul    rdi
    6162:	mov    rax,rdx
    6165:	and    rdx,0xfffffffffffffffe
    6169:	shr    rax,1
    616c:	add    rdx,rax
    616f:	mov    rax,rcx
    6172:	add    rcx,0x1
    6176:	cvtsi2sd xmm0,rcx
    617b:	divsd  xmm0,xmm2
    617f:	sub    rax,rdx
    6182:	add    rax,0x1
    6186:	cvtsi2sd xmm1,rax
    618b:	mulsd  xmm1,xmm3
    618f:	unpcklpd xmm0,xmm1
    6193:	movaps XMMWORD PTR [rsi-0x10],xmm0
    6197:	cmp    rcx,0x10
    619b:	jne    6150 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0x90>
    619d:	mov    rdi,rsp
    61a0:	call   5cc0 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv>
    61a5:	call   28a0 <instruction_count_begin>
    61aa:	mov    rdi,rsp
    61ad:	call   5cc0 <_ZN8NormCaseIN4Base6Matrix7ComplexIdEELm16EE5applyEPv>
    61b2:	call   28b0 <instruction_count_end>
    61b7:	lea    rsi,[rip+0x5e7e]        # c03c <_IO_stdin_used+0x3c>
    61be:	mov    edi,0x2
    61c3:	xor    eax,eax
    61c5:	call   10e0 <__printf_chk@plt>
    61ca:	nop    WORD PTR [rax+rax*1+0x0]
    61d0:	movsd  xmm0,QWORD PTR [rbp+0x0]
    61d5:	movsd  xmm1,QWORD PTR [rbp+0x8]
    61da:	mov    edi,0x2
    61df:	lea    rsi,[rip+0x5e64]        # c04a <_IO_stdin_used+0x4a>
    61e6:	mov    eax,0x2
    61eb:	add    rbp,0x10
    61ef:	call   10e0 <__printf_chk@plt>
    61f4:	cmp    rbp,rbx
    61f7:	jne    61d0 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0x110>
    61f9:	movsd  xmm0,QWORD PTR [rsp+0x100]
    6202:	mov    edi,0x2
    6207:	mov    eax,0x1
    620c:	lea    rsi,[rip+0x5e3e]        # c051 <_IO_stdin_used+0x51>
    6213:	call   10e0 <__printf_chk@plt>
    6218:	mov    rax,QWORD PTR [rsp+0x118]
    6220:	sub    rax,QWORD PTR fs:0x28
    6229:	jne    6237 <_Z3runI8NormCaseIN4Base6Matrix7ComplexIdEELm16EEEvv+0x177>
    622b:	add    rsp,0x120
    6232:	pop    rbx
    6233:	pop    rbp
    6234:	pop    r12
    6236:	ret
    6237:	call   10b0 <__stack_chk_fail@plt>
    623c:	nop    DWORD PTR [rax+0x0]

0000000000006240 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>:
    6240:	endbr64
    6244:	pxor   xmm2,xmm2
    6248:	xor    eax,eax
    624a:	movaps xmm3,xmm2
    624d:	nop    DWORD PTR [rax+0x0]
    6254:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    625f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    626a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    6275:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    6280:	cmp    rax,rsi
    6283:	jae    62ac <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x6c>
    6285:	movss  xmm0,DWORD PTR [rdi+rax*4]
    628a:	movaps xmm4,xmm0
    628d:	movaps xmm1,xmm0
    6290:	xorps  xmm1,XMMWORD PTR [rip+0x5fb9]        # c250 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    6297:	cmpltss xmm0,xmm3
    629c:	andps  xmm1,xmm0
    629f:	andnps xmm0,xmm4
    62a2:	orps   xmm0,xmm1
    62a5:	maxss  xmm0,xmm2
    62a9:	movaps xmm2,xmm0
    62ac:	add    rax,0x1
    62b0:	cmp    rax,0x8
    62b4:	jne    6280 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x40>
    62b6:	pxor   xmm0,xmm0
    62ba:	ucomiss xmm2,xmm0
    62bd:	jp     62c1 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x81>
    62bf:	je     630f <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xcf>
    62c1:	xor    eax,eax
    62c3:	pxor   xmm1,xmm1
    62c7:	nop    DWORD PTR [rax]
    62ca:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    62d5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    62e0:	cmp    rax,rsi
    62e3:	jae    62f6 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xb6>
    62e5:	movss  xmm0,DWORD PTR [rdi+rax*4]
    62ea:	divss  xmm0,xmm2
    62ee:	mulss  xmm0,xmm0
    62f2:	addss  xmm1,xmm0
    62f6:	add    rax,0x1
    62fa:	cmp    rax,0x8
    62fe:	jne    62e0 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xa0>
    6300:	pxor   xmm0,xmm0
    6304:	ucomiss xmm1,xmm0
    6307:	jp     6310 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd0>
    6309:	jne    6310 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd0>
    630b:	mulss  xmm0,xmm2
    630f:	ret
    6310:	movss  xmm3,DWORD PTR [rip+0x5cf0]        # c008 <_IO_stdin_used+0x8>
    6318:	movaps xmm4,xmm3
    631b:	maxss  xmm4,xmm1
    631f:	maxss  xmm3,xmm4
    6323:	movd   eax,xmm3
    6327:	movaps xmm0,xmm3
    632a:	mov    esi,eax
    632c:	shr    esi,0x17
    632f:	movzx  ecx,sil
    6333:	cmp    ecx,0xff
    6339:	je     64c0 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x280>
    633f:	test   ecx,ecx
    6341:	je     6440 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x200>
    6347:	and    eax,0x807fffff
    634c:	mov    edx,0x7e
    6351:	and    esi,0x1
    6354:	or     eax,0x3f000000
    6359:	sub    edx,ecx
    635b:	movd   xmm3,eax
    635f:	lea    rax,[rip+0x5eda]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    6366:	sar    edx,1
    6368:	movss  xmm5,DWORD PTR [rax+rsi*4]
    636d:	mulss  xmm3,DWORD PTR [rip+0x5c97]        # c00c <_IO_stdin_used+0xc>
    6375:	movss  xmm1,DWORD PTR [rip+0x5c93]        # c010 <_IO_stdin_used+0x10>
    637d:	subss  xmm1,xmm3
    6381:	mulss  xmm1,xmm5
    6385:	movd   ecx,xmm1
    6389:	mov    eax,ecx
    638b:	shr    eax,0x17
    638e:	movzx  eax,al
    6391:	cmp    eax,0xff
    6396:	je     63d8 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x198>
    6398:	mov    esi,ecx
    639a:	mov    edi,eax
    639c:	and    esi,0x7fffff
    63a2:	or     edi,esi
    63a4:	je     63d8 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x198>
    63a6:	mov    edi,ecx
    63a8:	and    edi,0x80000000
    63ae:	test   eax,eax
    63b0:	je     64d2 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x292>
    63b6:	add    eax,edx
    63b8:	je     657f <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x33f>
    63be:	cmp    eax,0xfe
    63c3:	jg     657f <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x33f>
    63c9:	shl    eax,0x17
    63cc:	and    ecx,0x807fffff
    63d2:	or     eax,ecx
    63d4:	movd   xmm1,eax
    63d8:	movaps xmm3,xmm0
    63db:	movss  xmm6,DWORD PTR [rip+0x5c31]        # c014 <_IO_stdin_used+0x14>
    63e3:	movss  xmm5,DWORD PTR [rip+0x5c2d]        # c018 <_IO_stdin_used+0x18>
    63eb:	mulss  xmm3,xmm1
    63ef:	mulss  xmm3,xmm1
    63f3:	mulss  xmm6,xmm3
    63f7:	subss  xmm5,xmm6
    63fb:	mulss  xmm5,xmm3
    63ff:	movss  xmm3,DWORD PTR [rip+0x5e8d]        # c294 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x54>
    6407:	subss  xmm3,xmm5
    640b:	mulss  xmm3,xmm1
    640f:	movss  xmm1,DWORD PTR [rip+0x5c05]        # c01c <_IO_stdin_used+0x1c>
    6417:	mulss  xmm1,xmm0
    641b:	movss  xmm0,DWORD PTR [rip+0x5eb1]        # c2d4 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x94>
    6423:	mulss  xmm1,xmm3
    6427:	mulss  xmm1,xmm3
    642b:	subss  xmm0,xmm1
    642f:	mulss  xmm0,xmm3
    6433:	mulss  xmm0,xmm4
    6437:	mulss  xmm0,xmm2
    643b:	ret
    643c:	nop    DWORD PTR [rax+0x0]
    6440:	mov    esi,eax
    6442:	and    esi,0x7fffff
    6448:	je     64c0 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x280>
    644a:	mov    edi,eax
    644c:	and    edi,0x80000000
    6452:	test   eax,0x7f0000
    6457:	mov    eax,esi
    6459:	je     6553 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x313>
    645f:	shl    eax,0x8
    6462:	mov    edx,0x8
    6467:	cmp    eax,0xfffffff
    646c:	jbe    6569 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x329>
    6472:	cmp    eax,0x3fffffff
    6477:	ja     6483 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x243>
    6479:	add    edx,0x2
    647c:	shl    eax,0x2
    647f:	test   eax,eax
    6481:	js     6486 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x246>
    6483:	add    edx,0x1
    6486:	lea    ecx,[rdx-0x9]
    6489:	lea    rax,[rip+0x5db0]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    6490:	sar    edx,1
    6492:	shl    esi,cl
    6494:	not    ecx
    6496:	add    edx,0x3a
    6499:	and    esi,0x7fffff
    649f:	and    ecx,0x1
    64a2:	or     esi,edi
    64a4:	movss  xmm5,DWORD PTR [rax+rcx*4]
    64a9:	or     esi,0x3f000000
    64af:	movd   xmm3,esi
    64b3:	jmp    636d <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x12d>
    64b8:	nop    DWORD PTR [rax+rax*1+0x0]
    64c0:	movss  xmm5,DWORD PTR [rip+0x5b3c]        # c004 <_IO_stdin_used+0x4>
    64c8:	movaps xmm3,xmm0
    64cb:	xor    edx,edx
    64cd:	jmp    636d <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x12d>
    64d2:	and    ecx,0x7f0000
    64d8:	mov    eax,esi
    64da:	je     652a <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2ea>
    64dc:	shl    eax,0x8
    64df:	mov    ecx,0x8
    64e4:	cmp    eax,0xfffffff
    64e9:	jbe    6544 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x304>
    64eb:	cmp    eax,0x3fffffff
    64f0:	ja     6500 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2c0>
    64f2:	add    ecx,0x2
    64f5:	shl    eax,0x2
    64f8:	test   eax,eax
    64fa:	js     65a2 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x362>
    6500:	sub    ecx,0x8
    6503:	mov    eax,edx
    6505:	shl    esi,cl
    6507:	sub    eax,ecx
    6509:	and    esi,0x7fffff
    650f:	add    eax,0x1
    6512:	test   eax,eax
    6514:	jle    65aa <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x36a>
    651a:	or     edi,esi
    651c:	shl    eax,0x17
    651f:	or     edi,eax
    6521:	movd   xmm1,edi
    6525:	jmp    63d8 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x198>
    652a:	shl    eax,0x10
    652d:	mov    ecx,0x10
    6532:	cmp    eax,0xffffff
    6537:	jbe    65d8 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x398>
    653d:	cmp    eax,0xfffffff
    6542:	ja     654a <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x30a>
    6544:	add    ecx,0x4
    6547:	shl    eax,0x4
    654a:	cmp    eax,0x3fffffff
    654f:	ja     64f8 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2b8>
    6551:	jmp    64f2 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2b2>
    6553:	shl    eax,0x10
    6556:	mov    edx,0x10
    655b:	cmp    eax,0xffffff
    6560:	jbe    65cc <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x38c>
    6562:	cmp    eax,0xfffffff
    6567:	ja     656f <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x32f>
    6569:	add    edx,0x4
    656c:	shl    eax,0x4
    656f:	cmp    eax,0x3fffffff
    6574:	ja     647f <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x23f>
    657a:	jmp    6479 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x239>
    657f:	mov    edx,esi
    6581:	shr    edx,1
    6583:	or     edx,edi
    6585:	or     edi,0x7f800000
    658b:	or     edx,0x400000
    6591:	cmp    eax,0xfe
    6596:	cmovg  edx,edi
    6599:	movd   xmm1,edx
    659d:	jmp    63d8 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x198>
    65a2:	sub    ecx,0x9
    65a5:	jmp    6503 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2c3>
    65aa:	movd   xmm1,edi
    65ae:	cmp    eax,0xffffffea
    65b1:	jl     63d8 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x198>
    65b7:	or     esi,0x800000
    65bd:	sub    ecx,edx
    65bf:	shr    esi,cl
    65c1:	or     esi,edi
    65c3:	movd   xmm1,esi
    65c7:	jmp    63d8 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x198>
    65cc:	mov    eax,esi
    65ce:	mov    edx,0x18
    65d3:	shl    eax,0x18
    65d6:	jmp    6562 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x322>
    65d8:	mov    eax,esi
    65da:	mov    ecx,0x18
    65df:	shl    eax,0x18
    65e2:	jmp    653d <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2fd>
    65e7:	nop    WORD PTR [rax+rax*1+0x0]

00000000000065f0 <_ZN8NormCaseIfLm8EE5applyEPv>:
    65f0:	endbr64
    65f4:	push   rbx
    65f5:	mov    rsi,QWORD PTR [rdi+0x28]
    65f9:	mov    rbx,rdi
    65fc:	call   6240 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    6601:	movss  DWORD PTR [rbx+0x20],xmm0
    6606:	pop    rbx
    6607:	ret
    6608:	cs nop WORD PTR [rax+rax*1+0x0]
    6612:	cs nop WORD PTR [rax+rax*1+0x0]
    661c:	cs nop WORD PTR [rax+rax*1+0x0]
    6626:	cs nop WORD PTR [rax+rax*1+0x0]
    6630:	cs nop WORD PTR [rax+rax*1+0x0]
    663a:	nop    WORD PTR [rax+rax*1+0x0]

0000000000006640 <_Z3runI8NormCaseIfLm8EEEvv>:
    6640:	endbr64
    6644:	push   rbp
    6645:	pxor   xmm0,xmm0
    6649:	push   rbx
    664a:	sub    rsp,0x48
    664e:	movsd  xmm1,QWORD PTR [rip+0x5c72]        # c2c8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x88>
    6656:	mov    rax,QWORD PTR fs:0x28
    665f:	mov    QWORD PTR [rsp+0x38],rax
    6664:	xor    eax,eax
    6666:	mov    rbp,rsp
    6669:	movaps XMMWORD PTR [rsp],xmm0
    666d:	mov    rbx,rsp
    6670:	mov    QWORD PTR [rsp+0x20],0x0
    6679:	mov    rdx,rsp
    667c:	mov    QWORD PTR [rsp+0x28],0x8
    6685:	movaps XMMWORD PTR [rsp+0x10],xmm0
    668a:	cs nop WORD PTR [rax+rax*1+0x0]
    6694:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    669f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    66aa:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    66b5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    66c0:	add    rax,0x1
    66c4:	pxor   xmm0,xmm0
    66c8:	add    rdx,0x4
    66cc:	cvtsi2sd xmm0,rax
    66d1:	divsd  xmm0,xmm1
    66d5:	cvtsd2ss xmm0,xmm0
    66d9:	movss  DWORD PTR [rdx-0x4],xmm0
    66de:	cmp    rax,0x8
    66e2:	jne    66c0 <_Z3runI8NormCaseIfLm8EEEvv+0x80>
    66e4:	mov    rdi,rbp
    66e7:	call   65f0 <_ZN8NormCaseIfLm8EE5applyEPv>
    66ec:	call   28a0 <instruction_count_begin>
    66f1:	mov    rdi,rbp
    66f4:	add    rbp,0x20
    66f8:	call   65f0 <_ZN8NormCaseIfLm8EE5applyEPv>
    66fd:	call   28b0 <instruction_count_end>
    6702:	lea    rsi,[rip+0x5933]        # c03c <_IO_stdin_used+0x3c>
    6709:	mov    edi,0x2
    670e:	xor    eax,eax
    6710:	call   10e0 <__printf_chk@plt>
    6715:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    6720:	pxor   xmm0,xmm0
    6724:	lea    rsi,[rip+0x5922]        # c04d <_IO_stdin_used+0x4d>
    672b:	mov    edi,0x2
    6730:	mov    eax,0x1
    6735:	cvtss2sd xmm0,DWORD PTR [rbx]
    6739:	call   10e0 <__printf_chk@plt>
    673e:	add    rbx,0x4
    6742:	cmp    rbx,rbp
    6745:	jne    6720 <_Z3runI8NormCaseIfLm8EEEvv+0xe0>
    6747:	pxor   xmm0,xmm0
    674b:	mov    edi,0x2
    6750:	mov    eax,0x1
    6755:	lea    rsi,[rip+0x58f5]        # c051 <_IO_stdin_used+0x51>
    675c:	cvtss2sd xmm0,DWORD PTR [rsp+0x20]
    6762:	call   10e0 <__printf_chk@plt>
    6767:	mov    rax,QWORD PTR [rsp+0x38]
    676c:	sub    rax,QWORD PTR fs:0x28
    6775:	jne    677e <_Z3runI8NormCaseIfLm8EEEvv+0x13e>
    6777:	add    rsp,0x48
    677b:	pop    rbx
    677c:	pop    rbp
    677d:	ret
    677e:	call   10b0 <__stack_chk_fail@plt>
    6783:	cs nop WORD PTR [rax+rax*1+0x0]
    678d:	cs nop WORD PTR [rax+rax*1+0x0]
    6797:	cs nop WORD PTR [rax+rax*1+0x0]
    67a1:	cs nop WORD PTR [rax+rax*1+0x0]
    67ab:	cs nop WORD PTR [rax+rax*1+0x0]
    67b5:	cs nop WORD PTR [rax+rax*1+0x0]
    67bf:	nop

00000000000067c0 <_ZN9GmresCaseIfLm8EE5applyEPv>:
    67c0:	endbr64
    67c4:	push   r15
    67c6:	pxor   xmm0,xmm0
    67ca:	mov    esi,0x8
    67cf:	push   r14
    67d1:	movaps xmm1,xmm0
    67d4:	push   r13
    67d6:	push   r12
    67d8:	push   rbp
    67d9:	push   rbx
    67da:	sub    rsp,0x138
    67e1:	movss  xmm3,DWORD PTR [rip+0x5837]        # c020 <_IO_stdin_used+0x20>
    67e9:	mov    rbp,QWORD PTR fs:0x28
    67f2:	mov    QWORD PTR [rsp+0x128],rbp
    67fa:	mov    rbp,rdi
    67fd:	add    rdi,0x100
    6804:	lea    r14,[rsp+0xa0]
    680c:	movss  xmm2,DWORD PTR [rbp+0x178]
    6814:	mov    rbx,QWORD PTR [rbp+0x168]
    681b:	mov    QWORD PTR [rbp+0x3c0],0x0
    6826:	cmpnltss xmm1,xmm2
    682b:	comiss xmm0,xmm2
    682e:	andps  xmm3,xmm1
    6831:	andnps xmm1,xmm2
    6834:	movaps xmm2,xmm0
    6837:	setae  BYTE PTR [rbp+0x3ca]
    683e:	orps   xmm1,xmm3
    6841:	xor    eax,eax
    6843:	movss  DWORD PTR [rsp+0x74],xmm1
    6849:	movss  xmm1,DWORD PTR [rbp+0x17c]
    6851:	mov    WORD PTR [rbp+0x3c8],ax
    6858:	xor    eax,eax
    685a:	cmp    QWORD PTR [rbp+0x170],0x1
    6862:	cmpltss xmm2,xmm1
    6867:	seta   al
    686a:	add    rax,0x1
    686e:	mov    QWORD PTR [rsp+0x40],rax
    6873:	andps  xmm1,xmm2
    6876:	movss  DWORD PTR [rsp+0x8],xmm1
    687c:	movss  xmm1,DWORD PTR [rbp+0x180]
    6884:	cmpltss xmm0,xmm1
    6889:	movaps xmm7,xmm1
    688c:	andps  xmm7,xmm0
    688f:	movss  DWORD PTR [rsp+0x48],xmm7
    6895:	call   6240 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    689a:	mov    rcx,r14
    689d:	mov    esi,0x8
    68a2:	mov    rdi,rbp
    68a5:	mulss  xmm0,DWORD PTR [rsp+0x8]
    68ab:	mov    r8d,0x8
    68b1:	mov    QWORD PTR [rsp+0x68],r14
    68b6:	lea    rdx,[rsp+0x80]
    68be:	addss  xmm0,DWORD PTR [rsp+0x48]
    68c4:	mov    QWORD PTR [rsp+0x78],rdx
    68c9:	movss  DWORD PTR [rsp+0x4c],xmm0
    68cf:	movdqu xmm0,XMMWORD PTR [rbp+0x120]
    68d7:	movaps XMMWORD PTR [rsp+0x80],xmm0
    68df:	movdqu xmm0,XMMWORD PTR [rbp+0x130]
    68e7:	movaps XMMWORD PTR [rsp+0x90],xmm0
    68ef:	pxor   xmm0,xmm0
    68f3:	movaps XMMWORD PTR [rsp+0xa0],xmm0
    68fb:	movaps XMMWORD PTR [rsp+0xb0],xmm0
    6903:	call   2340 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    6908:	movups xmm0,XMMWORD PTR [rbp+0x100]
    690f:	mov    esi,0x8
    6914:	mov    rdi,r14
    6917:	subps  xmm0,XMMWORD PTR [rsp+0xa0]
    691f:	movaps XMMWORD PTR [rsp+0xa0],xmm0
    6927:	movups xmm0,XMMWORD PTR [rbp+0x110]
    692e:	subps  xmm0,XMMWORD PTR [rsp+0xb0]
    6936:	movaps XMMWORD PTR [rsp+0xb0],xmm0
    693e:	call   6240 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    6943:	movss  xmm7,DWORD PTR [rsp+0x4c]
    6949:	mov    r11,QWORD PTR [rbp+0x160]
    6950:	movss  DWORD PTR [rbp+0x3bc],xmm0
    6958:	comiss xmm7,xmm0
    695b:	setae  al
    695e:	mov    BYTE PTR [rbp+0x3c8],al
    6964:	test   r11,r11
    6967:	je     7161 <_ZN9GmresCaseIfLm8EE5applyEPv+0x9a1>
    696d:	mov    edx,0x8
    6972:	cmp    rbx,rdx
    6975:	cmovbe rdx,rbx
    6979:	test   rdx,rdx
    697c:	mov    rsi,rdx
    697f:	mov    edx,0x1
    6984:	cmovne rdx,rsi
    6988:	xor    r12d,r12d
    698b:	mov    QWORD PTR [rsp+0x8],rdx
    6990:	test   al,al
    6992:	jne    6ba0 <_ZN9GmresCaseIfLm8EE5applyEPv+0x3e0>
    6998:	cmp    BYTE PTR [rbp+0x3c9],0x0
    699f:	jne    6ba0 <_ZN9GmresCaseIfLm8EE5applyEPv+0x3e0>
    69a5:	movss  xmm2,DWORD PTR [rbp+0x3bc]
    69ad:	comiss xmm2,DWORD PTR [rip+0x5670]        # c024 <_IO_stdin_used+0x24>
    69b4:	pxor   xmm0,xmm0
    69b8:	pxor   xmm1,xmm1
    69bc:	movups XMMWORD PTR [rbp+0x338],xmm0
    69c3:	mov    DWORD PTR [rbp+0x358],0x0
    69cd:	movups XMMWORD PTR [rbp+0x348],xmm0
    69d4:	movss  DWORD PTR [rbp+0x338],xmm2
    69dc:	jb     69ea <_ZN9GmresCaseIfLm8EE5applyEPv+0x22a>
    69de:	movss  xmm1,DWORD PTR [rip+0x561e]        # c004 <_IO_stdin_used+0x4>
    69e6:	divss  xmm1,xmm2
    69ea:	comiss xmm2,DWORD PTR [rip+0x5633]        # c024 <_IO_stdin_used+0x24>
    69f1:	mov    rsi,QWORD PTR [rsp+0x68]
    69f6:	lea    rdx,[rsp+0xc0]
    69fe:	lea    rcx,[rbp+0x184]
    6a05:	movss  xmm0,DWORD PTR [rsi]
    6a09:	mov    rax,rsi
    6a0c:	jb     6bb9 <_ZN9GmresCaseIfLm8EE5applyEPv+0x3f9>
    6a12:	mulss  xmm0,xmm1
    6a16:	add    rax,0x8
    6a1a:	movss  DWORD PTR [rcx],xmm0
    6a1e:	movss  xmm0,DWORD PTR [rax-0x4]
    6a23:	mulss  xmm0,xmm1
    6a27:	movss  DWORD PTR [rcx+0x4],xmm0
    6a2c:	cmp    rax,rdx
    6a2f:	je     6be8 <_ZN9GmresCaseIfLm8EE5applyEPv+0x428>
    6a35:	movss  xmm0,DWORD PTR [rax]
    6a39:	add    rcx,0x8
    6a3d:	jmp    6a12 <_ZN9GmresCaseIfLm8EE5applyEPv+0x252>
    6a3f:	nop
    6a40:	movdqa xmm0,XMMWORD PTR [rsp+0x80]
    6a49:	lea    rdi,[rsp+0x100]
    6a51:	xor    r8d,r8d
    6a54:	movaps XMMWORD PTR [rsp+0x100],xmm0
    6a5c:	movdqa xmm0,XMMWORD PTR [rsp+0x90]
    6a65:	movaps XMMWORD PTR [rsp+0x110],xmm0
    6a6d:	nop    DWORD PTR [rax]
    6a70:	mov    rcx,r13
    6a73:	pxor   xmm1,xmm1
    6a77:	xor    edx,edx
    6a79:	nop    DWORD PTR [rax+0x0]
    6a80:	cmp    rdx,rsi
    6a83:	jae    6a9a <_ZN9GmresCaseIfLm8EE5applyEPv+0x2da>
    6a85:	movss  xmm0,DWORD PTR [rcx+0x184]
    6a8d:	mulss  xmm0,DWORD PTR [rbp+rdx*4+0x39c]
    6a96:	addss  xmm1,xmm0
    6a9a:	add    rdx,0x1
    6a9e:	add    rcx,0x20
    6aa2:	cmp    rdx,0x8
    6aa6:	jne    6a80 <_ZN9GmresCaseIfLm8EE5applyEPv+0x2c0>
    6aa8:	addss  xmm1,DWORD PTR [rdi]
    6aac:	add    r8,0x1
    6ab0:	add    rdi,0x4
    6ab4:	add    r13,0x4
    6ab8:	movss  DWORD PTR [rdi-0x4],xmm1
    6abd:	cmp    r8,0x8
    6ac1:	jne    6a70 <_ZN9GmresCaseIfLm8EE5applyEPv+0x2b0>
    6ac3:	mov    rbx,QWORD PTR [rsp+0x68]
    6ac8:	mov    r8d,0x8
    6ace:	mov    esi,0x8
    6ad3:	mov    rdi,rbp
    6ad6:	lea    rdx,[rsp+0x100]
    6ade:	mov    QWORD PTR [rsp+0x10],r11
    6ae3:	mov    rcx,rbx
    6ae6:	mov    BYTE PTR [rsp+0x48],al
    6aea:	movss  DWORD PTR [rsp+0x18],xmm2
    6af0:	call   2340 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    6af5:	movups xmm7,XMMWORD PTR [rbp+0x100]
    6afc:	mov    esi,0x8
    6b01:	mov    rdi,rbx
    6b04:	movaps xmm0,xmm7
    6b07:	subps  xmm0,XMMWORD PTR [rsp+0xa0]
    6b0f:	movaps XMMWORD PTR [rsp+0x20],xmm7
    6b14:	movups xmm7,XMMWORD PTR [rbp+0x110]
    6b1b:	movaps XMMWORD PTR [rsp+0x30],xmm7
    6b20:	movaps XMMWORD PTR [rsp+0xa0],xmm0
    6b28:	movaps xmm0,xmm7
    6b2b:	subps  xmm0,XMMWORD PTR [rsp+0xb0]
    6b33:	movaps XMMWORD PTR [rsp+0xb0],xmm0
    6b3b:	call   6240 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    6b40:	movss  xmm2,DWORD PTR [rsp+0x18]
    6b46:	movzx  eax,BYTE PTR [rsp+0x48]
    6b4b:	mov    r11,QWORD PTR [rsp+0x10]
    6b50:	comiss xmm2,xmm0
    6b53:	jb     72b8 <_ZN9GmresCaseIfLm8EE5applyEPv+0xaf8>
    6b59:	movdqa xmm1,XMMWORD PTR [rsp+0x100]
    6b62:	movss  DWORD PTR [rbp+0x3bc],xmm0
    6b6a:	movaps XMMWORD PTR [rsp+0x80],xmm1
    6b72:	movdqa xmm1,XMMWORD PTR [rsp+0x110]
    6b7b:	movaps XMMWORD PTR [rsp+0x90],xmm1
    6b83:	movss  xmm7,DWORD PTR [rsp+0x4c]
    6b89:	comiss xmm7,xmm0
    6b8c:	setb   dl
    6b8f:	setae  BYTE PTR [rbp+0x3c8]
    6b96:	and    eax,edx
    6b98:	mov    BYTE PTR [rbp+0x3c9],al
    6b9e:	xchg   ax,ax
    6ba0:	add    r12,0x1
    6ba4:	cmp    r11,r12
    6ba7:	je     7161 <_ZN9GmresCaseIfLm8EE5applyEPv+0x9a1>
    6bad:	movzx  eax,BYTE PTR [rbp+0x3c8]
    6bb4:	jmp    6990 <_ZN9GmresCaseIfLm8EE5applyEPv+0x1d0>
    6bb9:	divss  xmm0,xmm2
    6bbd:	add    rax,0x8
    6bc1:	movss  DWORD PTR [rcx],xmm0
    6bc5:	movss  xmm0,DWORD PTR [rax-0x4]
    6bca:	divss  xmm0,xmm2
    6bce:	movss  DWORD PTR [rcx+0x4],xmm0
    6bd3:	cmp    rdx,rax
    6bd6:	je     6be8 <_ZN9GmresCaseIfLm8EE5applyEPv+0x428>
    6bd8:	movss  xmm0,DWORD PTR [rax]
    6bdc:	add    rcx,0x8
    6be0:	jmp    6bb9 <_ZN9GmresCaseIfLm8EE5applyEPv+0x3f9>
    6be2:	nop    WORD PTR [rax+rax*1+0x0]
    6be8:	pxor   xmm0,xmm0
    6bec:	mov    QWORD PTR [rsp+0x58],r12
    6bf1:	xor    esi,esi
    6bf3:	mov    ecx,0x1
    6bf8:	mov    QWORD PTR [rsp+0x30],rdx
    6bfd:	xor    r12d,r12d
    6c00:	xor    edx,edx
    6c02:	lea    r14,[rbp+0x318]
    6c09:	mov    QWORD PTR [rsp+0x50],r11
    6c0e:	lea    r13,[rbp+0x1a4]
    6c15:	mov    QWORD PTR [rsp+0x60],rbp
    6c1a:	movaps XMMWORD PTR [rsp+0xc0],xmm0
    6c22:	movaps XMMWORD PTR [rsp+0xd0],xmm0
    6c2a:	movaps XMMWORD PTR [rsp+0xe0],xmm0
    6c32:	movaps XMMWORD PTR [rsp+0xf0],xmm0
    6c3a:	movss  DWORD PTR [rsp+0x70],xmm2
    6c40:	jmp    6c95 <_ZN9GmresCaseIfLm8EE5applyEPv+0x4d5>
    6c42:	nop    DWORD PTR [rax+0x0]
    6c49:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    6c54:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    6c5f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    6c6a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    6c75:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    6c80:	mov    r12,rax
    6c83:	add    r14,0x4
    6c87:	add    r13,0x20
    6c8b:	cmp    r12,0x8
    6c8f:	je     6fd0 <_ZN9GmresCaseIfLm8EE5applyEPv+0x810>
    6c95:	cmp    r12,QWORD PTR [rsp+0x8]
    6c9a:	lea    rax,[r12+0x1]
    6c9f:	setb   r15b
    6ca3:	and    r15b,cl
    6ca6:	mov    BYTE PTR [rsp+0x48],r15b
    6cab:	je     6c80 <_ZN9GmresCaseIfLm8EE5applyEPv+0x4c0>
    6cad:	movdqu xmm0,XMMWORD PTR [r13-0x20]
    6cb3:	mov    rdx,QWORD PTR [rsp+0x30]
    6cb8:	mov    esi,0x8
    6cbd:	lea    rcx,[rsp+0xe0]
    6cc5:	mov    r8d,0x8
    6ccb:	mov    rdi,rbp
    6cce:	mov    QWORD PTR [rsp+0x20],rax
    6cd3:	movaps XMMWORD PTR [rdx],xmm0
    6cd6:	movdqu xmm0,XMMWORD PTR [r13-0x10]
    6cdc:	mov    QWORD PTR [rsp+0x10],rcx
    6ce1:	movaps XMMWORD PTR [rdx+0x10],xmm0
    6ce5:	call   2340 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    6cea:	mov    rcx,QWORD PTR [rsp+0x10]
    6cef:	mov    esi,0x8
    6cf4:	mov    rdi,rcx
    6cf7:	mov    QWORD PTR [rsp+0x18],rcx
    6cfc:	call   6240 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    6d01:	mov    rax,QWORD PTR [rsp+0x20]
    6d06:	mov    r8,QWORD PTR [rsp+0x40]
    6d0b:	xor    edi,edi
    6d0d:	mov    DWORD PTR [rbp+0x334],0x0
    6d17:	mov    rcx,QWORD PTR [rsp+0x18]
    6d1c:	movss  DWORD PTR [rsp+0x10],xmm0
    6d22:	pxor   xmm0,xmm0
    6d26:	movups XMMWORD PTR [rbp+0x314],xmm0
    6d2d:	movups XMMWORD PTR [rbp+0x324],xmm0
    6d34:	lea    r15,[rbp+0x314]
    6d3b:	lea    rbx,[rbp+0x184]
    6d42:	mov    rdx,r15
    6d45:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    6d50:	pxor   xmm1,xmm1
    6d54:	xor    esi,esi
    6d56:	movaps xmm0,xmm1
    6d59:	movups xmm4,XMMWORD PTR [rbx+rsi*1]
    6d5d:	add    rsi,0x10
    6d61:	movaps xmm5,xmm4
    6d64:	mulps  xmm5,xmm4
    6d67:	mulps  xmm4,XMMWORD PTR [rsi+rcx*1-0x10]
    6d6c:	addss  xmm1,xmm5
    6d70:	movaps xmm3,xmm5
    6d73:	movaps xmm6,xmm5
    6d76:	addss  xmm0,xmm4
    6d7a:	shufps xmm3,xmm5,0x55
    6d7e:	unpckhps xmm6,xmm5
    6d81:	shufps xmm5,xmm5,0xff
    6d85:	addss  xmm1,xmm3
    6d89:	movaps xmm3,xmm4
    6d8c:	shufps xmm4,xmm4,0x55
    6d90:	addss  xmm0,xmm4
    6d94:	movaps xmm4,xmm3
    6d97:	unpckhps xmm4,xmm3
    6d9a:	shufps xmm3,xmm3,0xff
    6d9e:	addss  xmm1,xmm6
    6da2:	addss  xmm0,xmm4
    6da6:	addss  xmm1,xmm5
    6daa:	addss  xmm0,xmm3
    6dae:	cmp    rsi,0x20
    6db2:	jne    6d59 <_ZN9GmresCaseIfLm8EE5applyEPv+0x599>
    6db4:	divss  xmm0,xmm1
    6db8:	movss  xmm1,DWORD PTR [rdx]
    6dbc:	add    rbx,0x20
    6dc0:	add    rdx,0x4
    6dc4:	addss  xmm1,xmm0
    6dc8:	shufps xmm0,xmm0,0x0
    6dcc:	movss  DWORD PTR [rdx-0x4],xmm1
    6dd1:	movups xmm2,XMMWORD PTR [rbx-0x20]
    6dd5:	movaps xmm1,XMMWORD PTR [rsp+0xe0]
    6ddd:	mulps  xmm2,xmm0
    6de0:	subps  xmm1,xmm2
    6de3:	movaps XMMWORD PTR [rsp+0xe0],xmm1
    6deb:	movups xmm1,XMMWORD PTR [rbx-0x10]
    6def:	mulps  xmm1,xmm0
    6df2:	movaps xmm0,XMMWORD PTR [rsp+0xf0]
    6dfa:	subps  xmm0,xmm1
    6dfd:	movaps XMMWORD PTR [rsp+0xf0],xmm0
    6e05:	cmp    r13,rbx
    6e08:	jne    6d50 <_ZN9GmresCaseIfLm8EE5applyEPv+0x590>
    6e0e:	add    rdi,0x1
    6e12:	cmp    rdi,r8
    6e15:	jne    6d34 <_ZN9GmresCaseIfLm8EE5applyEPv+0x574>
    6e1b:	mov    esi,0x8
    6e20:	mov    rdi,rcx
    6e23:	mov    QWORD PTR [rsp+0x18],rax
    6e28:	mov    QWORD PTR [rsp+0x20],rcx
    6e2d:	call   6240 <_ZN4Base6Matrix9Numerical4normIfLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    6e32:	mov    rax,QWORD PTR [rsp+0x18]
    6e37:	movss  xmm10,DWORD PTR [rip+0x51e8]        # c028 <_IO_stdin_used+0x28>
    6e40:	mulss  xmm10,DWORD PTR [rsp+0x10]
    6e47:	movaps xmm8,xmm0
    6e4b:	movss  DWORD PTR [r14],xmm0
    6e50:	comiss xmm10,xmm0
    6e54:	jae    6e66 <_ZN9GmresCaseIfLm8EE5applyEPv+0x6a6>
    6e56:	cmp    rax,QWORD PTR [rsp+0x8]
    6e5b:	mov    rcx,QWORD PTR [rsp+0x20]
    6e60:	jb     71aa <_ZN9GmresCaseIfLm8EE5applyEPv+0x9ea>
    6e66:	movss  xmm9,DWORD PTR [rip+0x53e1]        # c250 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    6e6f:	test   r12,r12
    6e72:	je     6ee2 <_ZN9GmresCaseIfLm8EE5applyEPv+0x722>
    6e74:	movss  xmm0,DWORD PTR [rbp+0x314]
    6e7c:	mov    rdx,rbp
    6e7f:	lea    rcx,[r14-0x318]
    6e86:	movss  xmm9,DWORD PTR [rip+0x53c1]        # c250 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    6e8f:	nop
    6e90:	movss  xmm1,DWORD PTR [rdx+0x318]
    6e98:	movss  xmm5,DWORD PTR [rdx+0x37c]
    6ea0:	add    rdx,0x4
    6ea4:	movss  xmm3,DWORD PTR [rdx+0x358]
    6eac:	movaps xmm6,xmm1
    6eaf:	movaps xmm4,xmm5
    6eb2:	mulss  xmm1,xmm5
    6eb6:	mulss  xmm4,xmm0
    6eba:	mulss  xmm6,xmm3
    6ebe:	mulss  xmm3,xmm0
    6ec2:	movaps xmm0,xmm1
    6ec5:	addss  xmm4,xmm6
    6ec9:	subss  xmm0,xmm3
    6ecd:	movss  DWORD PTR [rdx+0x310],xmm4
    6ed5:	movss  DWORD PTR [rdx+0x314],xmm0
    6edd:	cmp    rdx,rcx
    6ee0:	jne    6e90 <_ZN9GmresCaseIfLm8EE5applyEPv+0x6d0>
    6ee2:	movss  xmm11,DWORD PTR [r14-0x4]
    6ee8:	movss  xmm1,DWORD PTR [r14]
    6eed:	lea    rsi,[r14+0x44]
    6ef1:	lea    rdi,[r14+0x64]
    6ef5:	mov    QWORD PTR [rsp+0x10],rax
    6efa:	movaps xmm0,xmm11
    6efe:	call   1500 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0>
    6f03:	mulss  xmm11,DWORD PTR [r14+0x64]
    6f09:	movss  xmm0,DWORD PTR [r14+0x44]
    6f0f:	mulss  xmm0,DWORD PTR [r14]
    6f14:	mov    rax,QWORD PTR [rsp+0x10]
    6f19:	mov    rdx,rax
    6f1c:	imul   rdx,r12
    6f20:	addss  xmm0,xmm11
    6f25:	shr    rdx,1
    6f28:	movss  DWORD PTR [r14-0x4],xmm0
    6f2e:	nop    DWORD PTR [rax+0x0]
    6f35:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    6f40:	movss  xmm0,DWORD PTR [r15]
    6f45:	movss  DWORD PTR [r15+rdx*4-0x90],xmm0
    6f4f:	add    r15,0x4
    6f53:	cmp    r14,r15
    6f56:	jne    6f40 <_ZN9GmresCaseIfLm8EE5applyEPv+0x780>
    6f58:	movss  xmm0,DWORD PTR [r14+0x44]
    6f5e:	movss  xmm1,DWORD PTR [r14+0x20]
    6f64:	xorps  xmm0,xmm9
    6f68:	mulss  xmm0,xmm1
    6f6c:	mulss  xmm1,DWORD PTR [r14+0x64]
    6f72:	movss  DWORD PTR [r14+0x24],xmm0
    6f78:	movss  DWORD PTR [r14+0x20],xmm1
    6f7e:	add    QWORD PTR [rbp+0x3c0],0x1
    6f86:	comiss xmm10,xmm8
    6f8a:	jb     70f0 <_ZN9GmresCaseIfLm8EE5applyEPv+0x930>
    6f90:	cmp    rax,0x8
    6f94:	je     7270 <_ZN9GmresCaseIfLm8EE5applyEPv+0xab0>
    6f9a:	cmp    r12,0x6
    6f9e:	je     7294 <_ZN9GmresCaseIfLm8EE5applyEPv+0xad4>
    6fa4:	add    r14,0x8
    6fa8:	add    r13,0x40
    6fac:	add    r12,0x3
    6fb0:	xor    ecx,ecx
    6fb2:	movzx  edx,BYTE PTR [rsp+0x48]
    6fb7:	mov    rsi,rax
    6fba:	add    r14,0x4
    6fbe:	add    r13,0x20
    6fc2:	cmp    r12,0x8
    6fc6:	jne    6c95 <_ZN9GmresCaseIfLm8EE5applyEPv+0x4d5>
    6fcc:	nop    DWORD PTR [rax+0x0]
    6fd0:	movss  xmm2,DWORD PTR [rsp+0x70]
    6fd6:	mov    r11,QWORD PTR [rsp+0x50]
    6fdb:	mov    eax,edx
    6fdd:	mov    r12,QWORD PTR [rsp+0x58]
    6fe2:	mov    r13,QWORD PTR [rsp+0x60]
    6fe7:	movss  xmm5,DWORD PTR [rsp+0x74]
    6fed:	mov    edi,0x7
    6ff2:	mov    ecx,0x1
    6ff7:	pxor   xmm3,xmm3
    6ffb:	jmp    700a <_ZN9GmresCaseIfLm8EE5applyEPv+0x84a>
    6ffd:	nop    DWORD PTR [rax]
    7000:	sub    rdi,0x1
    7004:	jb     70d0 <_ZN9GmresCaseIfLm8EE5applyEPv+0x910>
    700a:	mov    DWORD PTR [rbp+rdi*4+0x39c],0x0
    7015:	cmp    rdi,rsi
    7018:	setb   dl
    701b:	and    dl,cl
    701d:	mov    r8d,edx
    7020:	je     7000 <_ZN9GmresCaseIfLm8EE5applyEPv+0x840>
    7022:	movss  xmm1,DWORD PTR [rbp+rdi*4+0x338]
    702b:	lea    r9,[rdi+0x1]
    702f:	cmp    rdi,0x7
    7033:	je     7072 <_ZN9GmresCaseIfLm8EE5applyEPv+0x8b2>
    7035:	mov    rcx,r9
    7038:	nop    DWORD PTR [rax+rax*1+0x0]
    7040:	mov    rdx,rcx
    7043:	add    rcx,0x1
    7047:	cmp    rdx,rsi
    704a:	jae    706c <_ZN9GmresCaseIfLm8EE5applyEPv+0x8ac>
    704c:	imul   rdx,rcx
    7050:	shr    rdx,1
    7053:	add    rdx,rdi
    7056:	movss  xmm0,DWORD PTR [rbp+rdx*4+0x284]
    705f:	mulss  xmm0,DWORD PTR [rbp+rcx*4+0x398]
    7068:	subss  xmm1,xmm0
    706c:	cmp    rcx,0x8
    7070:	jne    7040 <_ZN9GmresCaseIfLm8EE5applyEPv+0x880>
    7072:	imul   r9,rdi
    7076:	shr    r9,1
    7079:	lea    rdx,[r9+rdi*1]
    707d:	movss  xmm0,DWORD PTR [rbp+rdx*4+0x284]
    7086:	comiss xmm3,xmm0
    7089:	ja     7128 <_ZN9GmresCaseIfLm8EE5applyEPv+0x968>
    708f:	comiss xmm5,xmm0
    7092:	jbe    714c <_ZN9GmresCaseIfLm8EE5applyEPv+0x98c>
    7098:	lea    rdx,[rdi-0x1]
    709c:	test   rdi,rdi
    709f:	je     70d8 <_ZN9GmresCaseIfLm8EE5applyEPv+0x918>
    70a1:	mov    DWORD PTR [rbp+rdx*4+0x39c],0x0
    70ac:	sub    rdi,0x2
    70b0:	cmp    rdi,0xffffffffffffffff
    70b4:	je     70d8 <_ZN9GmresCaseIfLm8EE5applyEPv+0x918>
    70b6:	mov    DWORD PTR [rbp+rdi*4+0x39c],0x0
    70c1:	xor    ecx,ecx
    70c3:	sub    rdi,0x1
    70c7:	jae    700a <_ZN9GmresCaseIfLm8EE5applyEPv+0x84a>
    70cd:	nop    DWORD PTR [rax]
    70d0:	test   cl,cl
    70d2:	jne    6a40 <_ZN9GmresCaseIfLm8EE5applyEPv+0x280>
    70d8:	mov    eax,0x1
    70dd:	mov    BYTE PTR [rbp+0x3c9],al
    70e3:	jmp    6ba0 <_ZN9GmresCaseIfLm8EE5applyEPv+0x3e0>
    70e8:	nop    DWORD PTR [rax+rax*1+0x0]
    70f0:	movaps xmm1,xmm0
    70f3:	xorps  xmm9,xmm0
    70f7:	mov    r13,rbx
    70fa:	mov    r12,rax
    70fd:	cmpltss xmm1,DWORD PTR [rip+0x5152]        # c258 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x18>
    7106:	mov    rsi,rax
    7109:	andps  xmm9,xmm1
    710d:	andnps xmm1,xmm0
    7110:	orps   xmm1,xmm9
    7114:	comiss xmm1,DWORD PTR [rsp+0x4c]
    7119:	seta   cl
    711c:	xor    edx,edx
    711e:	jmp    6c83 <_ZN9GmresCaseIfLm8EE5applyEPv+0x4c3>
    7123:	nop    DWORD PTR [rax+rax*1+0x0]
    7128:	movaps xmm4,xmm0
    712b:	xorps  xmm4,XMMWORD PTR [rip+0x511e]        # c250 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    7132:	comiss xmm5,xmm4
    7135:	ja     7098 <_ZN9GmresCaseIfLm8EE5applyEPv+0x8d8>
    713b:	movaps xmm4,xmm5
    713e:	xorps  xmm4,XMMWORD PTR [rip+0x510b]        # c250 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    7145:	minss  xmm4,xmm0
    7149:	movaps xmm0,xmm4
    714c:	divss  xmm1,xmm0
    7150:	mov    ecx,r8d
    7153:	movss  DWORD PTR [rbp+rdi*4+0x39c],xmm1
    715c:	jmp    7000 <_ZN9GmresCaseIfLm8EE5applyEPv+0x840>
    7161:	movdqa xmm0,XMMWORD PTR [rsp+0x80]
    716a:	movups XMMWORD PTR [rbp+0x140],xmm0
    7171:	movdqa xmm0,XMMWORD PTR [rsp+0x90]
    717a:	movups XMMWORD PTR [rbp+0x150],xmm0
    7181:	mov    rax,QWORD PTR [rsp+0x128]
    7189:	sub    rax,QWORD PTR fs:0x28
    7192:	jne    731f <_ZN9GmresCaseIfLm8EE5applyEPv+0xb5f>
    7198:	add    rsp,0x138
    719f:	pop    rbx
    71a0:	pop    rbp
    71a1:	pop    r12
    71a3:	pop    r13
    71a5:	pop    r14
    71a7:	pop    r15
    71a9:	ret
    71aa:	comiss xmm0,DWORD PTR [rip+0x4e73]        # c024 <_IO_stdin_used+0x24>
    71b1:	pxor   xmm1,xmm1
    71b5:	jb     71c3 <_ZN9GmresCaseIfLm8EE5applyEPv+0xa03>
    71b7:	movss  xmm1,DWORD PTR [rip+0x4e45]        # c004 <_IO_stdin_used+0x4>
    71bf:	divss  xmm1,xmm0
    71c3:	comiss xmm8,DWORD PTR [rip+0x4e59]        # c024 <_IO_stdin_used+0x24>
    71cb:	lea    rsi,[rsp+0x100]
    71d3:	mov    rdx,rbx
    71d6:	movss  xmm0,DWORD PTR [rsp+0xe0]
    71df:	jb     7240 <_ZN9GmresCaseIfLm8EE5applyEPv+0xa80>
    71e1:	mulss  xmm0,xmm1
    71e5:	add    rcx,0x8
    71e9:	movss  DWORD PTR [rdx],xmm0
    71ed:	movss  xmm0,DWORD PTR [rcx-0x4]
    71f2:	mulss  xmm0,xmm1
    71f6:	movss  DWORD PTR [rdx+0x4],xmm0
    71fb:	cmp    rsi,rcx
    71fe:	je     6e66 <_ZN9GmresCaseIfLm8EE5applyEPv+0x6a6>
    7204:	movss  xmm0,DWORD PTR [rcx]
    7208:	add    rdx,0x8
    720c:	jmp    71e1 <_ZN9GmresCaseIfLm8EE5applyEPv+0xa21>
    720e:	nop    WORD PTR [rax+rax*1+0x0]
    7214:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    721f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    722a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    7235:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    7240:	divss  xmm0,xmm8
    7245:	add    rcx,0x8
    7249:	movss  DWORD PTR [rdx],xmm0
    724d:	movss  xmm0,DWORD PTR [rcx-0x4]
    7252:	divss  xmm0,xmm8
    7257:	movss  DWORD PTR [rdx+0x4],xmm0
    725c:	cmp    rcx,rsi
    725f:	je     6e66 <_ZN9GmresCaseIfLm8EE5applyEPv+0x6a6>
    7265:	movss  xmm0,DWORD PTR [rcx]
    7269:	add    rdx,0x8
    726d:	jmp    7240 <_ZN9GmresCaseIfLm8EE5applyEPv+0xa80>
    726f:	nop
    7270:	movss  xmm2,DWORD PTR [rsp+0x70]
    7276:	mov    r11,QWORD PTR [rsp+0x50]
    727b:	mov    esi,0x8
    7280:	mov    r12,QWORD PTR [rsp+0x58]
    7285:	mov    r13,QWORD PTR [rsp+0x60]
    728a:	movzx  eax,BYTE PTR [rsp+0x48]
    728f:	jmp    6fe7 <_ZN9GmresCaseIfLm8EE5applyEPv+0x827>
    7294:	movss  xmm2,DWORD PTR [rsp+0x70]
    729a:	mov    r11,QWORD PTR [rsp+0x50]
    729f:	mov    esi,0x7
    72a4:	mov    r12,QWORD PTR [rsp+0x58]
    72a9:	mov    r13,QWORD PTR [rsp+0x60]
    72ae:	movzx  eax,BYTE PTR [rsp+0x48]
    72b3:	jmp    6fe7 <_ZN9GmresCaseIfLm8EE5applyEPv+0x827>
    72b8:	mov    rcx,QWORD PTR [rsp+0x68]
    72bd:	mov    rdx,QWORD PTR [rsp+0x78]
    72c2:	mov    esi,0x8
    72c7:	mov    rdi,rbp
    72ca:	mov    r8d,0x8
    72d0:	mov    QWORD PTR [rsp+0x10],r11
    72d5:	mov    BYTE PTR [rsp+0x48],al
    72d9:	call   2340 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm8ELm8EEEfLm8EEEfLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    72de:	movaps xmm0,XMMWORD PTR [rsp+0x20]
    72e3:	mov    r11,QWORD PTR [rsp+0x10]
    72e8:	subps  xmm0,XMMWORD PTR [rsp+0xa0]
    72f0:	movzx  eax,BYTE PTR [rsp+0x48]
    72f5:	movaps XMMWORD PTR [rsp+0xa0],xmm0
    72fd:	movaps xmm0,XMMWORD PTR [rsp+0x30]
    7302:	subps  xmm0,XMMWORD PTR [rsp+0xb0]
    730a:	movaps XMMWORD PTR [rsp+0xb0],xmm0
    7312:	movss  xmm0,DWORD PTR [rbp+0x3bc]
    731a:	jmp    6b83 <_ZN9GmresCaseIfLm8EE5applyEPv+0x3c3>
    731f:	call   10b0 <__stack_chk_fail@plt>
    7324:	cs nop WORD PTR [rax+rax*1+0x0]
    732e:	xchg   ax,ax

0000000000007330 <_Z3runI9GmresCaseIfLm8EEEvv>:
    7330:	endbr64
    7334:	push   r14
    7336:	mov    ecx,0x20
    733b:	pxor   xmm0,xmm0
    733f:	movabs r8,0xcccccccccccccccd
    7349:	push   r13
    734b:	push   r12
    734d:	push   rbp
    734e:	push   rbx
    734f:	sub    rsp,0x3e0
    7356:	mov    rax,QWORD PTR fs:0x28
    735f:	mov    QWORD PTR [rsp+0x3d8],rax
    7367:	xor    eax,eax
    7369:	mov    rdi,rsp
    736c:	movaps XMMWORD PTR [rsp+0x100],xmm0
    7374:	mov    rbx,rsp
    7377:	movaps XMMWORD PTR [rsp+0x110],xmm0
    737f:	mov    rbp,rsp
    7382:	mov    r9,rsp
    7385:	mov    QWORD PTR [rsp+0x168],0x8
    7391:	mov    QWORD PTR [rsp+0x178],0x2edbe6ff
    739d:	mov    DWORD PTR [rsp+0x180],0x0
    73a8:	movaps XMMWORD PTR [rsp+0x120],xmm0
    73b0:	rep stos QWORD PTR [rdi],rax
    73b3:	lea    rdi,[rsp+0x188]
    73bb:	mov    ecx,0x1f
    73c0:	mov    QWORD PTR [rsp+0x184],0x0
    73cc:	mov    QWORD PTR [rsp+0x27c],0x0
    73d8:	movaps XMMWORD PTR [rsp+0x130],xmm0
    73e0:	movaps XMMWORD PTR [rsp+0x140],xmm0
    73e8:	movaps XMMWORD PTR [rsp+0x150],xmm0
    73f0:	rep stos QWORD PTR [rdi],rax
    73f3:	lea    rdi,[rsp+0x288]
    73fb:	mov    ecx,0x11
    7400:	mov    QWORD PTR [rsp+0x284],0x0
    740c:	mov    QWORD PTR [rsp+0x30c],0x0
    7418:	rep stos QWORD PTR [rdi],rax
    741b:	xor    eax,eax
    741d:	movups XMMWORD PTR [rsp+0x314],xmm0
    7425:	xor    edi,edi
    7427:	movups XMMWORD PTR [rsp+0x324],xmm0
    742f:	mov    DWORD PTR [rsp+0x334],0x0
    743a:	movss  xmm2,DWORD PTR [rip+0x4bf2]        # c034 <_IO_stdin_used+0x34>
    7442:	mov    WORD PTR [rsp+0x3c8],ax
    744a:	movss  xmm4,DWORD PTR [rip+0x4bda]        # c02c <_IO_stdin_used+0x2c>
    7452:	mov    BYTE PTR [rsp+0x3ca],0x0
    745a:	movss  xmm3,DWORD PTR [rip+0x4bce]        # c030 <_IO_stdin_used+0x30>
    7462:	mov    DWORD PTR [rsp+0x358],0x0
    746d:	mov    DWORD PTR [rsp+0x3bc],0x0
    7478:	mov    QWORD PTR [rsp+0x3c0],0x0
    7484:	movups XMMWORD PTR [rsp+0x338],xmm0
    748c:	movups XMMWORD PTR [rsp+0x348],xmm0
    7494:	movups XMMWORD PTR [rsp+0x35c],xmm0
    749c:	movups XMMWORD PTR [rsp+0x36c],xmm0
    74a4:	movups XMMWORD PTR [rsp+0x37c],xmm0
    74ac:	movups XMMWORD PTR [rsp+0x38c],xmm0
    74b4:	movups XMMWORD PTR [rsp+0x39c],xmm0
    74bc:	movups XMMWORD PTR [rsp+0x3ac],xmm0
    74c4:	nop
    74c5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    74d0:	movss  xmm1,DWORD PTR [r9+0x100]
    74d9:	mov    rsi,r9
    74dc:	xor    ecx,ecx
    74de:	jmp    751e <_Z3runI9GmresCaseIfLm8EEEvv+0x1ee>
    74e0:	lea    r10,[rdi+rcx*1]
    74e4:	pxor   xmm0,xmm0
    74e8:	mov    rax,r10
    74eb:	mul    r8
    74ee:	mov    rax,rdx
    74f1:	and    rdx,0xfffffffffffffffc
    74f5:	shr    rax,0x2
    74f9:	add    rdx,rax
    74fc:	sub    r10,rdx
    74ff:	cvtsi2ss xmm0,r10
    7504:	mulss  xmm0,xmm2
    7508:	add    rcx,0x1
    750c:	movss  DWORD PTR [rsi],xmm0
    7510:	addss  xmm1,xmm0
    7514:	add    rsi,0x20
    7518:	cmp    rcx,0x8
    751c:	je     7540 <_Z3runI9GmresCaseIfLm8EEEvv+0x210>
    751e:	cmp    rdi,rcx
    7521:	jne    74e0 <_Z3runI9GmresCaseIfLm8EEEvv+0x1b0>
    7523:	pxor   xmm0,xmm0
    7527:	cvtsi2ss xmm0,rdi
    752c:	mulss  xmm0,xmm4
    7530:	addss  xmm0,xmm3
    7534:	jmp    7508 <_Z3runI9GmresCaseIfLm8EEEvv+0x1d8>
    7536:	cs nop WORD PTR [rax+rax*1+0x0]
    7540:	add    rdi,0x1
    7544:	add    r9,0x4
    7548:	movss  DWORD PTR [r9+0xfc],xmm1
    7551:	cmp    rdi,0x8
    7555:	jne    74d0 <_Z3runI9GmresCaseIfLm8EEEvv+0x1a0>
    755b:	mov    rdi,rbx
    755e:	mov    QWORD PTR [rsp+0x160],0x1
    756a:	lea    r13,[rbx+0x100]
    7571:	lea    r14,[rbx+0x120]
    7578:	mov    QWORD PTR [rsp+0x170],0x1
    7584:	mov    QWORD PTR [rsp+0x178],0x15f79688
    7590:	mov    DWORD PTR [rsp+0x180],0x0
    759b:	call   67c0 <_ZN9GmresCaseIfLm8EE5applyEPv>
    75a0:	call   28a0 <instruction_count_begin>
    75a5:	mov    rdi,rbx
    75a8:	call   67c0 <_ZN9GmresCaseIfLm8EE5applyEPv>
    75ad:	call   28b0 <instruction_count_end>
    75b2:	lea    rsi,[rip+0x4a83]        # c03c <_IO_stdin_used+0x3c>
    75b9:	mov    edi,0x2
    75be:	xor    eax,eax
    75c0:	call   10e0 <__printf_chk@plt>
    75c5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    75d0:	lea    r12,[r13-0x100]
    75d7:	nop    WORD PTR [rax+rax*1+0x0]
    75e0:	pxor   xmm0,xmm0
    75e4:	lea    rsi,[rip+0x4a62]        # c04d <_IO_stdin_used+0x4d>
    75eb:	mov    edi,0x2
    75f0:	mov    eax,0x1
    75f5:	cvtss2sd xmm0,DWORD PTR [r12]
    75fb:	call   10e0 <__printf_chk@plt>
    7600:	add    r12,0x20
    7604:	cmp    r12,r13
    7607:	jne    75e0 <_Z3runI9GmresCaseIfLm8EEEvv+0x2b0>
    7609:	lea    r13,[r12+0x4]
    760e:	cmp    r13,r14
    7611:	jne    75d0 <_Z3runI9GmresCaseIfLm8EEEvv+0x2a0>
    7613:	lea    r12,[rbx+0x20]
    7617:	nop    WORD PTR [rax+rax*1+0x0]
    7620:	pxor   xmm0,xmm0
    7624:	lea    rsi,[rip+0x4a22]        # c04d <_IO_stdin_used+0x4d>
    762b:	mov    edi,0x2
    7630:	mov    eax,0x1
    7635:	cvtss2sd xmm0,DWORD PTR [rbx+0x100]
    763d:	add    rbx,0x4
    7641:	call   10e0 <__printf_chk@plt>
    7646:	cmp    rbx,r12
    7649:	jne    7620 <_Z3runI9GmresCaseIfLm8EEEvv+0x2f0>
    764b:	lea    rsi,[rip+0x49f0]        # c042 <_IO_stdin_used+0x42>
    7652:	mov    edi,0x2
    7657:	xor    eax,eax
    7659:	call   10e0 <__printf_chk@plt>
    765e:	xchg   ax,ax
    7660:	pxor   xmm0,xmm0
    7664:	lea    rsi,[rip+0x49e2]        # c04d <_IO_stdin_used+0x4d>
    766b:	mov    edi,0x2
    7670:	mov    eax,0x1
    7675:	cvtss2sd xmm0,DWORD PTR [rbp+0x140]
    767d:	add    rbp,0x4
    7681:	call   10e0 <__printf_chk@plt>
    7686:	cmp    rbp,r12
    7689:	jne    7660 <_Z3runI9GmresCaseIfLm8EEEvv+0x330>
    768b:	pxor   xmm0,xmm0
    768f:	lea    rsi,[rip+0x49c7]        # c05d <_IO_stdin_used+0x5d>
    7696:	mov    edi,0x2
    769b:	movzx  ecx,BYTE PTR [rsp+0x3c8]
    76a3:	movzx  r8d,BYTE PTR [rsp+0x3c9]
    76ac:	mov    eax,0x1
    76b1:	mov    rdx,QWORD PTR [rsp+0x3c0]
    76b9:	cvtss2sd xmm0,DWORD PTR [rsp+0x3bc]
    76c2:	call   10e0 <__printf_chk@plt>
    76c7:	mov    rax,QWORD PTR [rsp+0x3d8]
    76cf:	sub    rax,QWORD PTR fs:0x28
    76d8:	jne    76ea <_Z3runI9GmresCaseIfLm8EEEvv+0x3ba>
    76da:	add    rsp,0x3e0
    76e1:	pop    rbx
    76e2:	pop    rbp
    76e3:	pop    r12
    76e5:	pop    r13
    76e7:	pop    r14
    76e9:	ret
    76ea:	call   10b0 <__stack_chk_fail@plt>
    76ef:	cs nop WORD PTR [rax+rax*1+0x0]
    76f9:	nop    DWORD PTR [rax+0x0]

0000000000007700 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>:
    7700:	endbr64
    7704:	pxor   xmm2,xmm2
    7708:	xor    eax,eax
    770a:	movaps xmm3,xmm2
    770d:	nop    DWORD PTR [rax+0x0]
    7714:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    771f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    772a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    7735:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    7740:	cmp    rax,rsi
    7743:	jae    776c <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x6c>
    7745:	movss  xmm0,DWORD PTR [rdi+rax*4]
    774a:	movaps xmm4,xmm0
    774d:	movaps xmm1,xmm0
    7750:	xorps  xmm1,XMMWORD PTR [rip+0x4af9]        # c250 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    7757:	cmpltss xmm0,xmm3
    775c:	andps  xmm1,xmm0
    775f:	andnps xmm0,xmm4
    7762:	orps   xmm0,xmm1
    7765:	maxss  xmm0,xmm2
    7769:	movaps xmm2,xmm0
    776c:	add    rax,0x1
    7770:	cmp    rax,0x10
    7774:	jne    7740 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x40>
    7776:	pxor   xmm0,xmm0
    777a:	ucomiss xmm2,xmm0
    777d:	jp     7781 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x81>
    777f:	je     77cf <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xcf>
    7781:	xor    eax,eax
    7783:	pxor   xmm1,xmm1
    7787:	nop    DWORD PTR [rax]
    778a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    7795:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    77a0:	cmp    rax,rsi
    77a3:	jae    77b6 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xb6>
    77a5:	movss  xmm0,DWORD PTR [rdi+rax*4]
    77aa:	divss  xmm0,xmm2
    77ae:	mulss  xmm0,xmm0
    77b2:	addss  xmm1,xmm0
    77b6:	add    rax,0x1
    77ba:	cmp    rax,0x10
    77be:	jne    77a0 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xa0>
    77c0:	pxor   xmm0,xmm0
    77c4:	ucomiss xmm1,xmm0
    77c7:	jp     77d0 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd0>
    77c9:	jne    77d0 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd0>
    77cb:	mulss  xmm0,xmm2
    77cf:	ret
    77d0:	movss  xmm3,DWORD PTR [rip+0x4830]        # c008 <_IO_stdin_used+0x8>
    77d8:	movaps xmm4,xmm3
    77db:	maxss  xmm4,xmm1
    77df:	maxss  xmm3,xmm4
    77e3:	movd   eax,xmm3
    77e7:	movaps xmm0,xmm3
    77ea:	mov    esi,eax
    77ec:	shr    esi,0x17
    77ef:	movzx  ecx,sil
    77f3:	cmp    ecx,0xff
    77f9:	je     7974 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x274>
    77ff:	test   ecx,ecx
    7801:	je     78fc <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x1fc>
    7807:	and    eax,0x807fffff
    780c:	mov    edx,0x7e
    7811:	and    esi,0x1
    7814:	or     eax,0x3f000000
    7819:	sub    edx,ecx
    781b:	movd   xmm3,eax
    781f:	lea    rax,[rip+0x4a1a]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    7826:	sar    edx,1
    7828:	movss  xmm5,DWORD PTR [rax+rsi*4]
    782d:	mulss  xmm3,DWORD PTR [rip+0x47d7]        # c00c <_IO_stdin_used+0xc>
    7835:	movss  xmm1,DWORD PTR [rip+0x47d3]        # c010 <_IO_stdin_used+0x10>
    783d:	subss  xmm1,xmm3
    7841:	mulss  xmm1,xmm5
    7845:	movd   ecx,xmm1
    7849:	mov    eax,ecx
    784b:	shr    eax,0x17
    784e:	movzx  eax,al
    7851:	cmp    eax,0xff
    7856:	je     7898 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x198>
    7858:	mov    esi,ecx
    785a:	mov    edi,eax
    785c:	and    esi,0x7fffff
    7862:	or     edi,esi
    7864:	je     7898 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x198>
    7866:	mov    edi,ecx
    7868:	and    edi,0x80000000
    786e:	test   eax,eax
    7870:	je     7986 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x286>
    7876:	add    eax,edx
    7878:	je     7a33 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x333>
    787e:	cmp    eax,0xfe
    7883:	jg     7a33 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x333>
    7889:	shl    eax,0x17
    788c:	and    ecx,0x807fffff
    7892:	or     eax,ecx
    7894:	movd   xmm1,eax
    7898:	movaps xmm3,xmm0
    789b:	movss  xmm6,DWORD PTR [rip+0x4771]        # c014 <_IO_stdin_used+0x14>
    78a3:	movss  xmm5,DWORD PTR [rip+0x476d]        # c018 <_IO_stdin_used+0x18>
    78ab:	mulss  xmm3,xmm1
    78af:	mulss  xmm3,xmm1
    78b3:	mulss  xmm6,xmm3
    78b7:	subss  xmm5,xmm6
    78bb:	mulss  xmm5,xmm3
    78bf:	movss  xmm3,DWORD PTR [rip+0x49cd]        # c294 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x54>
    78c7:	subss  xmm3,xmm5
    78cb:	mulss  xmm3,xmm1
    78cf:	movss  xmm1,DWORD PTR [rip+0x4745]        # c01c <_IO_stdin_used+0x1c>
    78d7:	mulss  xmm1,xmm0
    78db:	movss  xmm0,DWORD PTR [rip+0x49f1]        # c2d4 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x94>
    78e3:	mulss  xmm1,xmm3
    78e7:	mulss  xmm1,xmm3
    78eb:	subss  xmm0,xmm1
    78ef:	mulss  xmm0,xmm3
    78f3:	mulss  xmm0,xmm4
    78f7:	mulss  xmm0,xmm2
    78fb:	ret
    78fc:	mov    esi,eax
    78fe:	and    esi,0x7fffff
    7904:	je     7974 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x274>
    7906:	mov    edi,eax
    7908:	and    edi,0x80000000
    790e:	test   eax,0x7f0000
    7913:	mov    eax,esi
    7915:	je     7a07 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x307>
    791b:	shl    eax,0x8
    791e:	mov    edx,0x8
    7923:	cmp    eax,0xfffffff
    7928:	jbe    7a1d <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x31d>
    792e:	cmp    eax,0x3fffffff
    7933:	ja     793f <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x23f>
    7935:	add    edx,0x2
    7938:	shl    eax,0x2
    793b:	test   eax,eax
    793d:	js     7942 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x242>
    793f:	add    edx,0x1
    7942:	lea    ecx,[rdx-0x9]
    7945:	mov    eax,esi
    7947:	sar    edx,1
    7949:	shl    eax,cl
    794b:	not    ecx
    794d:	add    edx,0x3a
    7950:	and    eax,0x7fffff
    7955:	and    ecx,0x1
    7958:	or     eax,edi
    795a:	or     eax,0x3f000000
    795f:	movd   xmm3,eax
    7963:	lea    rax,[rip+0x48d6]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    796a:	movss  xmm5,DWORD PTR [rax+rcx*4]
    796f:	jmp    782d <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x12d>
    7974:	movss  xmm5,DWORD PTR [rip+0x4688]        # c004 <_IO_stdin_used+0x4>
    797c:	movaps xmm3,xmm0
    797f:	xor    edx,edx
    7981:	jmp    782d <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x12d>
    7986:	and    ecx,0x7f0000
    798c:	mov    eax,esi
    798e:	je     79de <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2de>
    7990:	shl    eax,0x8
    7993:	mov    ecx,0x8
    7998:	cmp    eax,0xfffffff
    799d:	jbe    79f8 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2f8>
    799f:	cmp    eax,0x3fffffff
    79a4:	ja     79b4 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2b4>
    79a6:	add    ecx,0x2
    79a9:	shl    eax,0x2
    79ac:	test   eax,eax
    79ae:	js     7a56 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x356>
    79b4:	sub    ecx,0x8
    79b7:	mov    eax,edx
    79b9:	shl    esi,cl
    79bb:	sub    eax,ecx
    79bd:	and    esi,0x7fffff
    79c3:	add    eax,0x1
    79c6:	test   eax,eax
    79c8:	jle    7a5e <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x35e>
    79ce:	or     edi,esi
    79d0:	shl    eax,0x17
    79d3:	or     edi,eax
    79d5:	movd   xmm1,edi
    79d9:	jmp    7898 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x198>
    79de:	shl    eax,0x10
    79e1:	mov    ecx,0x10
    79e6:	cmp    eax,0xffffff
    79eb:	jbe    7a8c <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x38c>
    79f1:	cmp    eax,0xfffffff
    79f6:	ja     79fe <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2fe>
    79f8:	add    ecx,0x4
    79fb:	shl    eax,0x4
    79fe:	cmp    eax,0x3fffffff
    7a03:	ja     79ac <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2ac>
    7a05:	jmp    79a6 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2a6>
    7a07:	shl    eax,0x10
    7a0a:	mov    edx,0x10
    7a0f:	cmp    eax,0xffffff
    7a14:	jbe    7a80 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x380>
    7a16:	cmp    eax,0xfffffff
    7a1b:	ja     7a23 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x323>
    7a1d:	add    edx,0x4
    7a20:	shl    eax,0x4
    7a23:	cmp    eax,0x3fffffff
    7a28:	ja     793b <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x23b>
    7a2e:	jmp    7935 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x235>
    7a33:	mov    edx,esi
    7a35:	shr    edx,1
    7a37:	or     edx,edi
    7a39:	or     edi,0x7f800000
    7a3f:	or     edx,0x400000
    7a45:	cmp    eax,0xfe
    7a4a:	cmovg  edx,edi
    7a4d:	movd   xmm1,edx
    7a51:	jmp    7898 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x198>
    7a56:	sub    ecx,0x9
    7a59:	jmp    79b7 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2b7>
    7a5e:	movd   xmm1,edi
    7a62:	cmp    eax,0xffffffea
    7a65:	jl     7898 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x198>
    7a6b:	or     esi,0x800000
    7a71:	sub    ecx,edx
    7a73:	shr    esi,cl
    7a75:	or     esi,edi
    7a77:	movd   xmm1,esi
    7a7b:	jmp    7898 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x198>
    7a80:	mov    eax,esi
    7a82:	mov    edx,0x18
    7a87:	shl    eax,0x18
    7a8a:	jmp    7a16 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x316>
    7a8c:	mov    eax,esi
    7a8e:	mov    ecx,0x18
    7a93:	shl    eax,0x18
    7a96:	jmp    79f1 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2f1>
    7a9b:	nop    DWORD PTR [rax+rax*1+0x0]

0000000000007aa0 <_ZN8NormCaseIfLm16EE5applyEPv>:
    7aa0:	endbr64
    7aa4:	push   rbx
    7aa5:	mov    rsi,QWORD PTR [rdi+0x48]
    7aa9:	mov    rbx,rdi
    7aac:	call   7700 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    7ab1:	movss  DWORD PTR [rbx+0x40],xmm0
    7ab6:	pop    rbx
    7ab7:	ret
    7ab8:	nop    DWORD PTR [rax+rax*1+0x0]

0000000000007ac0 <_Z3runI8NormCaseIfLm16EEEvv>:
    7ac0:	endbr64
    7ac4:	push   rbp
    7ac5:	pxor   xmm0,xmm0
    7ac9:	push   rbx
    7aca:	sub    rsp,0x68
    7ace:	movsd  xmm1,QWORD PTR [rip+0x4812]        # c2e8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
    7ad6:	mov    rax,QWORD PTR fs:0x28
    7adf:	mov    QWORD PTR [rsp+0x58],rax
    7ae4:	xor    eax,eax
    7ae6:	mov    rbp,rsp
    7ae9:	movaps XMMWORD PTR [rsp],xmm0
    7aed:	mov    rbx,rsp
    7af0:	mov    QWORD PTR [rsp+0x40],0x0
    7af9:	mov    rdx,rsp
    7afc:	mov    QWORD PTR [rsp+0x48],0x10
    7b05:	movaps XMMWORD PTR [rsp+0x10],xmm0
    7b0a:	movaps XMMWORD PTR [rsp+0x20],xmm0
    7b0f:	movaps XMMWORD PTR [rsp+0x30],xmm0
    7b14:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    7b1f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    7b2a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    7b35:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    7b40:	add    rax,0x1
    7b44:	pxor   xmm0,xmm0
    7b48:	add    rdx,0x4
    7b4c:	cvtsi2sd xmm0,rax
    7b51:	divsd  xmm0,xmm1
    7b55:	cvtsd2ss xmm0,xmm0
    7b59:	movss  DWORD PTR [rdx-0x4],xmm0
    7b5e:	cmp    rax,0x10
    7b62:	jne    7b40 <_Z3runI8NormCaseIfLm16EEEvv+0x80>
    7b64:	mov    rdi,rbp
    7b67:	call   7aa0 <_ZN8NormCaseIfLm16EE5applyEPv>
    7b6c:	call   28a0 <instruction_count_begin>
    7b71:	mov    rdi,rbp
    7b74:	add    rbp,0x40
    7b78:	call   7aa0 <_ZN8NormCaseIfLm16EE5applyEPv>
    7b7d:	call   28b0 <instruction_count_end>
    7b82:	lea    rsi,[rip+0x44b3]        # c03c <_IO_stdin_used+0x3c>
    7b89:	mov    edi,0x2
    7b8e:	xor    eax,eax
    7b90:	call   10e0 <__printf_chk@plt>
    7b95:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    7ba0:	pxor   xmm0,xmm0
    7ba4:	lea    rsi,[rip+0x44a2]        # c04d <_IO_stdin_used+0x4d>
    7bab:	mov    edi,0x2
    7bb0:	mov    eax,0x1
    7bb5:	cvtss2sd xmm0,DWORD PTR [rbx]
    7bb9:	call   10e0 <__printf_chk@plt>
    7bbe:	add    rbx,0x4
    7bc2:	cmp    rbx,rbp
    7bc5:	jne    7ba0 <_Z3runI8NormCaseIfLm16EEEvv+0xe0>
    7bc7:	pxor   xmm0,xmm0
    7bcb:	mov    edi,0x2
    7bd0:	mov    eax,0x1
    7bd5:	lea    rsi,[rip+0x4475]        # c051 <_IO_stdin_used+0x51>
    7bdc:	cvtss2sd xmm0,DWORD PTR [rsp+0x40]
    7be2:	call   10e0 <__printf_chk@plt>
    7be7:	mov    rax,QWORD PTR [rsp+0x58]
    7bec:	sub    rax,QWORD PTR fs:0x28
    7bf5:	jne    7bfe <_Z3runI8NormCaseIfLm16EEEvv+0x13e>
    7bf7:	add    rsp,0x68
    7bfb:	pop    rbx
    7bfc:	pop    rbp
    7bfd:	ret
    7bfe:	call   10b0 <__stack_chk_fail@plt>
    7c03:	cs nop WORD PTR [rax+rax*1+0x0]
    7c0d:	cs nop WORD PTR [rax+rax*1+0x0]
    7c17:	cs nop WORD PTR [rax+rax*1+0x0]
    7c21:	cs nop WORD PTR [rax+rax*1+0x0]
    7c2b:	cs nop WORD PTR [rax+rax*1+0x0]
    7c35:	cs nop WORD PTR [rax+rax*1+0x0]
    7c3f:	nop

0000000000007c40 <_ZN9GmresCaseIfLm16EE5applyEPv>:
    7c40:	endbr64
    7c44:	push   r15
    7c46:	pxor   xmm0,xmm0
    7c4a:	mov    esi,0x10
    7c4f:	push   r14
    7c51:	movaps xmm3,xmm0
    7c54:	push   r13
    7c56:	push   r12
    7c58:	push   rbp
    7c59:	push   rbx
    7c5a:	sub    rsp,0x1d8
    7c61:	movss  xmm1,DWORD PTR [rdi+0x4d8]
    7c69:	movss  xmm2,DWORD PTR [rip+0x43af]        # c020 <_IO_stdin_used+0x20>
    7c71:	mov    r14,QWORD PTR fs:0x28
    7c7a:	mov    QWORD PTR [rsp+0x1c8],r14
    7c82:	lea    r14,[rdi+0x400]
    7c89:	mov    rbx,QWORD PTR [rdi+0x4c8]
    7c90:	mov    QWORD PTR [rdi+0xc50],0x0
    7c9b:	lea    r15,[rsp+0xc0]
    7ca3:	cmpnltss xmm3,xmm1
    7ca8:	comiss xmm0,xmm1
    7cab:	mov    QWORD PTR [rsp+0x18],rdi
    7cb0:	andps  xmm2,xmm3
    7cb3:	andnps xmm3,xmm1
    7cb6:	movss  xmm1,DWORD PTR [rdi+0x4dc]
    7cbe:	setae  BYTE PTR [rdi+0xc5a]
    7cc5:	orps   xmm3,xmm2
    7cc8:	movaps xmm2,xmm0
    7ccb:	xor    eax,eax
    7ccd:	cmpltss xmm2,xmm1
    7cd2:	mov    WORD PTR [rdi+0xc58],ax
    7cd9:	xor    eax,eax
    7cdb:	cmp    QWORD PTR [rdi+0x4d0],0x1
    7ce3:	seta   al
    7ce6:	movss  DWORD PTR [rsp+0x20],xmm3
    7cec:	add    rax,0x1
    7cf0:	mov    QWORD PTR [rsp+0x60],r14
    7cf5:	andps  xmm1,xmm2
    7cf8:	mov    QWORD PTR [rsp+0x38],rax
    7cfd:	movss  DWORD PTR [rsp+0x8],xmm1
    7d03:	movss  xmm1,DWORD PTR [rdi+0x4e0]
    7d0b:	mov    rdi,r14
    7d0e:	cmpltss xmm0,xmm1
    7d13:	movaps xmm3,xmm1
    7d16:	andps  xmm3,xmm0
    7d19:	movss  DWORD PTR [rsp+0x10],xmm3
    7d1f:	call   7700 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    7d24:	mulss  xmm0,DWORD PTR [rsp+0x8]
    7d2a:	mov    r10,QWORD PTR [rsp+0x18]
    7d2f:	mov    rcx,r15
    7d32:	addss  xmm0,DWORD PTR [rsp+0x10]
    7d38:	mov    r8d,0x10
    7d3e:	mov    esi,0x10
    7d43:	mov    QWORD PTR [rsp+0x40],r15
    7d48:	lea    rdx,[rsp+0x80]
    7d50:	mov    rdi,r10
    7d53:	movss  DWORD PTR [rsp+0x48],xmm0
    7d59:	movdqu xmm0,XMMWORD PTR [r10+0x440]
    7d62:	movaps XMMWORD PTR [rsp+0x80],xmm0
    7d6a:	movdqu xmm0,XMMWORD PTR [r10+0x450]
    7d73:	movaps XMMWORD PTR [rsp+0x90],xmm0
    7d7b:	movdqu xmm0,XMMWORD PTR [r10+0x460]
    7d84:	movaps XMMWORD PTR [rsp+0xa0],xmm0
    7d8c:	movdqu xmm0,XMMWORD PTR [r10+0x470]
    7d95:	movaps XMMWORD PTR [rsp+0xb0],xmm0
    7d9d:	pxor   xmm0,xmm0
    7da1:	movaps XMMWORD PTR [rsp+0xc0],xmm0
    7da9:	movaps XMMWORD PTR [rsp+0xd0],xmm0
    7db1:	movaps XMMWORD PTR [rsp+0xe0],xmm0
    7db9:	movaps XMMWORD PTR [rsp+0xf0],xmm0
    7dc1:	mov    QWORD PTR [rsp+0x78],rdx
    7dc6:	mov    QWORD PTR [rsp+0x8],r10
    7dcb:	call   2080 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    7dd0:	mov    r10,QWORD PTR [rsp+0x8]
    7dd5:	mov    rcx,r15
    7dd8:	mov    rax,r14
    7ddb:	movss  xmm3,DWORD PTR [rsp+0x20]
    7de1:	lea    rdx,[r10+0x440]
    7de8:	movups xmm0,XMMWORD PTR [rax]
    7deb:	subps  xmm0,XMMWORD PTR [rcx]
    7dee:	add    rax,0x10
    7df2:	add    rcx,0x10
    7df6:	movaps XMMWORD PTR [rcx-0x10],xmm0
    7dfa:	cmp    rax,rdx
    7dfd:	jne    7de8 <_ZN9GmresCaseIfLm16EE5applyEPv+0x1a8>
    7dff:	mov    rdi,QWORD PTR [rsp+0x40]
    7e04:	mov    esi,0x10
    7e09:	mov    QWORD PTR [rsp+0x8],r10
    7e0e:	movss  DWORD PTR [rsp+0x18],xmm3
    7e14:	mov    QWORD PTR [rsp+0x10],rdx
    7e19:	call   7700 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    7e1e:	movss  xmm3,DWORD PTR [rsp+0x48]
    7e24:	mov    r10,QWORD PTR [rsp+0x8]
    7e29:	comiss xmm3,xmm0
    7e2c:	mov    r11,QWORD PTR [r10+0x4c0]
    7e33:	movss  DWORD PTR [r10+0xc4c],xmm0
    7e3c:	setae  al
    7e3f:	mov    BYTE PTR [r10+0xc58],al
    7e46:	test   r11,r11
    7e49:	je     883b <_ZN9GmresCaseIfLm16EE5applyEPv+0xbfb>
    7e4f:	cmp    rbx,0x10
    7e53:	mov    edi,0x10
    7e58:	mov    ecx,0x1
    7e5d:	movss  xmm3,DWORD PTR [rsp+0x18]
    7e63:	cmovbe rdi,rbx
    7e67:	mov    rdx,QWORD PTR [rsp+0x10]
    7e6c:	mov    rbp,r10
    7e6f:	movaps xmm7,xmm3
    7e72:	xorps  xmm7,XMMWORD PTR [rip+0x43d7]        # c250 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    7e79:	test   rdi,rdi
    7e7c:	mov    QWORD PTR [rsp+0x70],rdx
    7e81:	cmovne rcx,rdi
    7e85:	movss  DWORD PTR [rsp+0x6c],xmm7
    7e8b:	xor    ebx,ebx
    7e8d:	mov    QWORD PTR [rsp+0x8],rcx
    7e92:	nop    DWORD PTR [rax]
    7e95:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    7ea0:	test   al,al
    7ea2:	jne    8190 <_ZN9GmresCaseIfLm16EE5applyEPv+0x550>
    7ea8:	cmp    BYTE PTR [rbp+0xc59],0x0
    7eaf:	jne    8190 <_ZN9GmresCaseIfLm16EE5applyEPv+0x550>
    7eb5:	movss  xmm1,DWORD PTR [rbp+0xc4c]
    7ebd:	comiss xmm1,DWORD PTR [rip+0x4160]        # c024 <_IO_stdin_used+0x24>
    7ec4:	pxor   xmm0,xmm0
    7ec8:	pxor   xmm2,xmm2
    7ecc:	movups XMMWORD PTR [rbp+0xb48],xmm0
    7ed3:	mov    DWORD PTR [rbp+0xb88],0x0
    7edd:	movups XMMWORD PTR [rbp+0xb58],xmm0
    7ee4:	movups XMMWORD PTR [rbp+0xb68],xmm0
    7eeb:	movups XMMWORD PTR [rbp+0xb78],xmm0
    7ef2:	movss  DWORD PTR [rbp+0xb48],xmm1
    7efa:	jb     7f08 <_ZN9GmresCaseIfLm16EE5applyEPv+0x2c8>
    7efc:	movss  xmm2,DWORD PTR [rip+0x4100]        # c004 <_IO_stdin_used+0x4>
    7f04:	divss  xmm2,xmm1
    7f08:	comiss xmm1,DWORD PTR [rip+0x4115]        # c024 <_IO_stdin_used+0x24>
    7f0f:	mov    rax,QWORD PTR [rsp+0x40]
    7f14:	lea    r9,[rbp+0x4e4]
    7f1b:	mov    rcx,r9
    7f1e:	movss  xmm0,DWORD PTR [rax]
    7f22:	lea    rdx,[rax+0x40]
    7f26:	jae    7f80 <_ZN9GmresCaseIfLm16EE5applyEPv+0x340>
    7f28:	mov    rax,QWORD PTR [rsp+0x40]
    7f2d:	add    rax,0x8
    7f31:	divss  xmm0,xmm1
    7f35:	movss  DWORD PTR [rcx],xmm0
    7f39:	movss  xmm0,DWORD PTR [rax-0x4]
    7f3e:	divss  xmm0,xmm1
    7f42:	movss  DWORD PTR [rcx+0x4],xmm0
    7f47:	cmp    rax,rdx
    7f4a:	je     81a9 <_ZN9GmresCaseIfLm16EE5applyEPv+0x569>
    7f50:	movss  xmm0,DWORD PTR [rax]
    7f54:	add    rcx,0x8
    7f58:	jmp    7f2d <_ZN9GmresCaseIfLm16EE5applyEPv+0x2ed>
    7f5a:	nop    DWORD PTR [rax+rax*1+0x0]
    7f5f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    7f6a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    7f75:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    7f80:	mulss  xmm0,xmm2
    7f84:	add    rax,0x8
    7f88:	movss  DWORD PTR [rcx],xmm0
    7f8c:	movss  xmm0,DWORD PTR [rax-0x4]
    7f91:	mulss  xmm0,xmm2
    7f95:	movss  DWORD PTR [rcx+0x4],xmm0
    7f9a:	cmp    rax,rdx
    7f9d:	je     81a9 <_ZN9GmresCaseIfLm16EE5applyEPv+0x569>
    7fa3:	movss  xmm0,DWORD PTR [rax]
    7fa7:	add    rcx,0x8
    7fab:	jmp    7f80 <_ZN9GmresCaseIfLm16EE5applyEPv+0x340>
    7fad:	nop    DWORD PTR [rax]
    7fb0:	movdqa xmm0,XMMWORD PTR [rsp+0x80]
    7fb9:	lea    rdx,[rsp+0x180]
    7fc1:	xor    edi,edi
    7fc3:	mov    rsi,rdx
    7fc6:	movaps XMMWORD PTR [rsp+0x180],xmm0
    7fce:	movdqa xmm0,XMMWORD PTR [rsp+0x90]
    7fd7:	movaps XMMWORD PTR [rsp+0x190],xmm0
    7fdf:	movdqa xmm0,XMMWORD PTR [rsp+0xa0]
    7fe8:	movaps XMMWORD PTR [rsp+0x1a0],xmm0
    7ff0:	movdqa xmm0,XMMWORD PTR [rsp+0xb0]
    7ff9:	movaps XMMWORD PTR [rsp+0x1b0],xmm0
    8001:	nop    DWORD PTR [rax+0x0]
    8005:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8010:	mov    r8,r9
    8013:	pxor   xmm2,xmm2
    8017:	xor    eax,eax
    8019:	nop    WORD PTR [rax+rax*1+0x0]
    801f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    802a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8035:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8040:	cmp    rax,rcx
    8043:	jae    8057 <_ZN9GmresCaseIfLm16EE5applyEPv+0x417>
    8045:	movss  xmm0,DWORD PTR [r8]
    804a:	mulss  xmm0,DWORD PTR [rbp+rax*4+0xc0c]
    8053:	addss  xmm2,xmm0
    8057:	add    rax,0x1
    805b:	add    r8,0x40
    805f:	cmp    rax,0x10
    8063:	jne    8040 <_ZN9GmresCaseIfLm16EE5applyEPv+0x400>
    8065:	addss  xmm2,DWORD PTR [rsi]
    8069:	add    rdi,0x1
    806d:	add    rsi,0x4
    8071:	add    r9,0x4
    8075:	movss  DWORD PTR [rsi-0x4],xmm2
    807a:	cmp    rdi,0x10
    807e:	jne    8010 <_ZN9GmresCaseIfLm16EE5applyEPv+0x3d0>
    8080:	mov    r14,QWORD PTR [rsp+0x40]
    8085:	mov    esi,0x10
    808a:	mov    rdi,rbp
    808d:	mov    QWORD PTR [rsp+0x10],r11
    8092:	mov    r8d,0x10
    8098:	movss  DWORD PTR [rsp+0x20],xmm1
    809e:	mov    rcx,r14
    80a1:	movss  DWORD PTR [rsp+0x18],xmm3
    80a7:	call   2080 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    80ac:	mov    rdx,QWORD PTR [rsp+0x60]
    80b1:	movss  xmm3,DWORD PTR [rsp+0x18]
    80b7:	mov    rax,r14
    80ba:	mov    r11,QWORD PTR [rsp+0x10]
    80bf:	mov    rcx,QWORD PTR [rsp+0x70]
    80c4:	movss  xmm1,DWORD PTR [rsp+0x20]
    80ca:	movups xmm0,XMMWORD PTR [rdx]
    80cd:	subps  xmm0,XMMWORD PTR [rax]
    80d0:	add    rdx,0x10
    80d4:	add    rax,0x10
    80d8:	movaps XMMWORD PTR [rax-0x10],xmm0
    80dc:	cmp    rdx,rcx
    80df:	jne    80ca <_ZN9GmresCaseIfLm16EE5applyEPv+0x48a>
    80e1:	mov    rdi,QWORD PTR [rsp+0x40]
    80e6:	mov    esi,0x10
    80eb:	mov    QWORD PTR [rsp+0x10],r11
    80f0:	movss  DWORD PTR [rsp+0x20],xmm1
    80f6:	movss  DWORD PTR [rsp+0x18],xmm3
    80fc:	call   7700 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    8101:	movss  xmm1,DWORD PTR [rsp+0x20]
    8107:	mov    r11,QWORD PTR [rsp+0x10]
    810c:	movss  xmm3,DWORD PTR [rsp+0x18]
    8112:	comiss xmm1,xmm0
    8115:	jb     88f8 <_ZN9GmresCaseIfLm16EE5applyEPv+0xcb8>
    811b:	movdqa xmm1,XMMWORD PTR [rsp+0x180]
    8124:	movss  DWORD PTR [rbp+0xc4c],xmm0
    812c:	movaps XMMWORD PTR [rsp+0x80],xmm1
    8134:	movdqa xmm1,XMMWORD PTR [rsp+0x190]
    813d:	movaps XMMWORD PTR [rsp+0x90],xmm1
    8145:	movdqa xmm1,XMMWORD PTR [rsp+0x1a0]
    814e:	movaps XMMWORD PTR [rsp+0xa0],xmm1
    8156:	movdqa xmm1,XMMWORD PTR [rsp+0x1b0]
    815f:	movaps XMMWORD PTR [rsp+0xb0],xmm1
    8167:	movss  xmm7,DWORD PTR [rsp+0x48]
    816d:	comiss xmm7,xmm0
    8170:	setb   al
    8173:	setae  BYTE PTR [rbp+0xc58]
    817a:	and    eax,r13d
    817d:	mov    BYTE PTR [rbp+0xc59],al
    8183:	xchg   ax,ax
    8185:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8190:	add    rbx,0x1
    8194:	cmp    r11,rbx
    8197:	je     8838 <_ZN9GmresCaseIfLm16EE5applyEPv+0xbf8>
    819d:	movzx  eax,BYTE PTR [rbp+0xc58]
    81a4:	jmp    7ea0 <_ZN9GmresCaseIfLm16EE5applyEPv+0x260>
    81a9:	pxor   xmm0,xmm0
    81ad:	mov    QWORD PTR [rsp+0x18],r9
    81b2:	xor    r13d,r13d
    81b5:	xor    ecx,ecx
    81b7:	mov    QWORD PTR [rsp+0x50],r11
    81bc:	mov    edx,0x1
    81c1:	xor    r15d,r15d
    81c4:	lea    r12,[rbp+0xb08]
    81cb:	mov    QWORD PTR [rsp+0x58],rbx
    81d0:	lea    r9,[rbp+0x524]
    81d7:	movaps XMMWORD PTR [rsp+0x100],xmm0
    81df:	movaps XMMWORD PTR [rsp+0x110],xmm0
    81e7:	movaps XMMWORD PTR [rsp+0x120],xmm0
    81ef:	movaps XMMWORD PTR [rsp+0x130],xmm0
    81f7:	movaps XMMWORD PTR [rsp+0x140],xmm0
    81ff:	movaps XMMWORD PTR [rsp+0x150],xmm0
    8207:	movaps XMMWORD PTR [rsp+0x160],xmm0
    820f:	movaps XMMWORD PTR [rsp+0x170],xmm0
    8217:	movss  DWORD PTR [rsp+0x4c],xmm1
    821d:	movss  DWORD PTR [rsp+0x68],xmm3
    8223:	jmp    8255 <_ZN9GmresCaseIfLm16EE5applyEPv+0x615>
    8225:	nop    DWORD PTR [rax+rax*1+0x0]
    822a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8235:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8240:	mov    r15,r10
    8243:	add    r12,0x4
    8247:	add    r9,0x40
    824b:	cmp    r15,0x10
    824f:	je     85f3 <_ZN9GmresCaseIfLm16EE5applyEPv+0x9b3>
    8255:	cmp    r15,QWORD PTR [rsp+0x8]
    825a:	lea    r10,[r15+0x1]
    825e:	setb   r14b
    8262:	and    r14b,dl
    8265:	je     8240 <_ZN9GmresCaseIfLm16EE5applyEPv+0x600>
    8267:	movdqu xmm0,XMMWORD PTR [r9-0x40]
    826d:	mov    r8d,0x10
    8273:	mov    esi,0x10
    8278:	lea    r13,[rsp+0x140]
    8280:	lea    rdx,[rsp+0x100]
    8288:	mov    rcx,r13
    828b:	mov    rdi,rbp
    828e:	mov    QWORD PTR [rsp+0x28],r10
    8293:	movaps XMMWORD PTR [rsp+0x100],xmm0
    829b:	movdqu xmm0,XMMWORD PTR [r9-0x30]
    82a1:	mov    QWORD PTR [rsp+0x10],r9
    82a6:	movaps XMMWORD PTR [rsp+0x110],xmm0
    82ae:	movdqu xmm0,XMMWORD PTR [r9-0x20]
    82b4:	movaps XMMWORD PTR [rsp+0x120],xmm0
    82bc:	movdqu xmm0,XMMWORD PTR [r9-0x10]
    82c2:	movaps XMMWORD PTR [rsp+0x130],xmm0
    82ca:	call   2080 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    82cf:	mov    esi,0x10
    82d4:	mov    rdi,r13
    82d7:	call   7700 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    82dc:	mov    r10,QWORD PTR [rsp+0x28]
    82e1:	mov    rcx,QWORD PTR [rsp+0x38]
    82e6:	xor    edx,edx
    82e8:	mov    DWORD PTR [rbp+0xb44],0x0
    82f2:	mov    r9,QWORD PTR [rsp+0x10]
    82f7:	movss  DWORD PTR [rsp+0x20],xmm0
    82fd:	pxor   xmm0,xmm0
    8301:	movups XMMWORD PTR [rbp+0xb04],xmm0
    8308:	movups XMMWORD PTR [rbp+0xb14],xmm0
    830f:	movups XMMWORD PTR [rbp+0xb24],xmm0
    8316:	movups XMMWORD PTR [rbp+0xb34],xmm0
    831d:	lea    rbx,[rbp+0xb04]
    8324:	mov    rsi,QWORD PTR [rsp+0x18]
    8329:	lea    rdi,[rbp+0x524]
    8330:	mov    r8,r10
    8333:	mov    r11,rbx
    8336:	cs nop WORD PTR [rax+rax*1+0x0]
    8340:	pxor   xmm2,xmm2
    8344:	xor    eax,eax
    8346:	movaps xmm5,xmm2
    8349:	movups xmm0,XMMWORD PTR [rsi+rax*1]
    834d:	movaps xmm1,xmm0
    8350:	mulps  xmm1,xmm0
    8353:	mulps  xmm0,XMMWORD PTR [r13+rax*1+0x0]
    8359:	add    rax,0x10
    835d:	addss  xmm2,xmm1
    8361:	movaps xmm4,xmm1
    8364:	movaps xmm6,xmm1
    8367:	movaps xmm7,xmm0
    836a:	shufps xmm4,xmm1,0x55
    836e:	unpckhps xmm6,xmm1
    8371:	shufps xmm1,xmm1,0xff
    8375:	addss  xmm7,xmm5
    8379:	addss  xmm2,xmm4
    837d:	movaps xmm4,xmm0
    8380:	shufps xmm4,xmm0,0x55
    8384:	movaps xmm5,xmm4
    8387:	movaps xmm4,xmm0
    838a:	addss  xmm5,xmm7
    838e:	unpckhps xmm4,xmm0
    8391:	shufps xmm0,xmm0,0xff
    8395:	addss  xmm2,xmm6
    8399:	addss  xmm5,xmm4
    839d:	addss  xmm2,xmm1
    83a1:	addss  xmm5,xmm0
    83a5:	cmp    rax,0x40
    83a9:	jne    8349 <_ZN9GmresCaseIfLm16EE5applyEPv+0x709>
    83ab:	divss  xmm5,xmm2
    83af:	movss  xmm0,DWORD PTR [r11]
    83b4:	mov    rax,rsi
    83b7:	mov    r10,r13
    83ba:	mov    QWORD PTR [rsp+0x10],r13
    83bf:	addss  xmm0,xmm5
    83c3:	shufps xmm5,xmm5,0x0
    83c7:	movss  DWORD PTR [r11],xmm0
    83cc:	movups xmm2,XMMWORD PTR [rax]
    83cf:	movaps xmm0,XMMWORD PTR [r10]
    83d3:	add    rax,0x10
    83d7:	add    r10,0x10
    83db:	mulps  xmm2,xmm5
    83de:	subps  xmm0,xmm2
    83e1:	movaps XMMWORD PTR [r10-0x10],xmm0
    83e6:	cmp    rax,rdi
    83e9:	jne    83cc <_ZN9GmresCaseIfLm16EE5applyEPv+0x78c>
    83eb:	add    r11,0x4
    83ef:	add    rsi,0x40
    83f3:	lea    rdi,[rax+0x40]
    83f7:	cmp    r11,r12
    83fa:	jne    8340 <_ZN9GmresCaseIfLm16EE5applyEPv+0x700>
    8400:	add    rdx,0x1
    8404:	mov    r10,r8
    8407:	cmp    rdx,rcx
    840a:	jne    831d <_ZN9GmresCaseIfLm16EE5applyEPv+0x6dd>
    8410:	mov    QWORD PTR [rsp+0x28],r8
    8415:	mov    r8,QWORD PTR [rsp+0x10]
    841a:	mov    esi,0x10
    841f:	mov    rdi,r13
    8422:	mov    QWORD PTR [rsp+0x10],r9
    8427:	mov    QWORD PTR [rsp+0x30],r8
    842c:	call   7700 <_ZN4Base6Matrix9Numerical4normIfLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    8431:	mov    r9,QWORD PTR [rsp+0x10]
    8436:	mov    r10,QWORD PTR [rsp+0x28]
    843b:	movss  xmm10,DWORD PTR [rip+0x3be4]        # c028 <_IO_stdin_used+0x28>
    8444:	movaps xmm8,xmm0
    8448:	movss  DWORD PTR [r12],xmm0
    844e:	mulss  xmm10,DWORD PTR [rsp+0x20]
    8455:	comiss xmm10,xmm0
    8459:	jae    846b <_ZN9GmresCaseIfLm16EE5applyEPv+0x82b>
    845b:	cmp    r10,QWORD PTR [rsp+0x8]
    8460:	mov    r8,QWORD PTR [rsp+0x30]
    8465:	jb     878d <_ZN9GmresCaseIfLm16EE5applyEPv+0xb4d>
    846b:	movss  xmm9,DWORD PTR [rip+0x3ddc]        # c250 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    8474:	test   r15,r15
    8477:	je     84e8 <_ZN9GmresCaseIfLm16EE5applyEPv+0x8a8>
    8479:	movss  xmm0,DWORD PTR [rbp+0xb04]
    8481:	lea    rax,[rbp+0xb04]
    8488:	lea    rdx,[r12-0x4]
    848d:	movss  xmm9,DWORD PTR [rip+0x3dba]        # c250 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    8496:	cs nop WORD PTR [rax+rax*1+0x0]
    84a0:	movss  xmm1,DWORD PTR [rax+0x4]
    84a5:	movss  xmm5,DWORD PTR [rax+0xc8]
    84ad:	add    rax,0x4
    84b1:	movss  xmm2,DWORD PTR [rax+0x84]
    84b9:	movaps xmm6,xmm1
    84bc:	movaps xmm4,xmm5
    84bf:	mulss  xmm1,xmm5
    84c3:	mulss  xmm4,xmm0
    84c7:	mulss  xmm6,xmm2
    84cb:	mulss  xmm2,xmm0
    84cf:	movaps xmm0,xmm1
    84d2:	addss  xmm4,xmm6
    84d6:	subss  xmm0,xmm2
    84da:	movss  DWORD PTR [rax-0x4],xmm4
    84df:	movss  DWORD PTR [rax],xmm0
    84e3:	cmp    rax,rdx
    84e6:	jne    84a0 <_ZN9GmresCaseIfLm16EE5applyEPv+0x860>
    84e8:	movss  xmm11,DWORD PTR [r12-0x4]
    84ef:	movss  xmm1,DWORD PTR [r12]
    84f5:	lea    rsi,[r12+0x84]
    84fd:	lea    rdi,[r12+0xc4]
    8505:	mov    QWORD PTR [rsp+0x20],r9
    850a:	movaps xmm0,xmm11
    850e:	mov    QWORD PTR [rsp+0x10],r10
    8513:	call   1500 <_ZN4Base6Matrix9Numerical6givensIfEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0>
    8518:	mov    r10,QWORD PTR [rsp+0x10]
    851d:	mov    r9,QWORD PTR [rsp+0x20]
    8522:	mulss  xmm11,DWORD PTR [r12+0xc4]
    852c:	movss  xmm0,DWORD PTR [r12+0x84]
    8536:	mulss  xmm0,DWORD PTR [r12]
    853c:	mov    rax,r10
    853f:	imul   rax,r15
    8543:	shr    rax,1
    8546:	addss  xmm0,xmm11
    854b:	movss  DWORD PTR [r12-0x4],xmm0
    8552:	nop    DWORD PTR [rax]
    8555:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8560:	movss  xmm0,DWORD PTR [rbx]
    8564:	movss  DWORD PTR [rbx+rax*4-0x220],xmm0
    856d:	add    rbx,0x4
    8571:	cmp    rbx,r12
    8574:	jne    8560 <_ZN9GmresCaseIfLm16EE5applyEPv+0x920>
    8576:	movss  xmm0,DWORD PTR [r12+0x84]
    8580:	movss  xmm2,DWORD PTR [r12+0x40]
    8587:	xorps  xmm0,xmm9
    858b:	mulss  xmm0,xmm2
    858f:	mulss  xmm2,DWORD PTR [r12+0xc4]
    8599:	movss  DWORD PTR [r12+0x44],xmm0
    85a0:	movss  DWORD PTR [r12+0x40],xmm2
    85a7:	add    QWORD PTR [rbp+0xc50],0x1
    85af:	comiss xmm10,xmm8
    85b3:	jb     8728 <_ZN9GmresCaseIfLm16EE5applyEPv+0xae8>
    85b9:	cmp    r10,0x10
    85bd:	je     88d0 <_ZN9GmresCaseIfLm16EE5applyEPv+0xc90>
    85c3:	cmp    r15,0xe
    85c7:	je     88a8 <_ZN9GmresCaseIfLm16EE5applyEPv+0xc68>
    85cd:	add    r12,0x8
    85d1:	sub    r9,0xffffffffffffff80
    85d5:	add    r15,0x3
    85d9:	mov    r13d,r14d
    85dc:	xor    edx,edx
    85de:	mov    rcx,r10
    85e1:	add    r12,0x4
    85e5:	add    r9,0x40
    85e9:	cmp    r15,0x10
    85ed:	jne    8255 <_ZN9GmresCaseIfLm16EE5applyEPv+0x615>
    85f3:	movss  xmm1,DWORD PTR [rsp+0x4c]
    85f9:	movss  xmm3,DWORD PTR [rsp+0x68]
    85ff:	mov    r9,QWORD PTR [rsp+0x18]
    8604:	mov    r11,QWORD PTR [rsp+0x50]
    8609:	mov    rbx,QWORD PTR [rsp+0x58]
    860e:	mov    esi,0xf
    8613:	mov    eax,0x1
    8618:	pxor   xmm4,xmm4
    861c:	jmp    862a <_ZN9GmresCaseIfLm16EE5applyEPv+0x9ea>
    861e:	xchg   ax,ax
    8620:	sub    rsi,0x1
    8624:	jb     870d <_ZN9GmresCaseIfLm16EE5applyEPv+0xacd>
    862a:	mov    DWORD PTR [rbp+rsi*4+0xc0c],0x0
    8635:	cmp    rsi,rcx
    8638:	setb   dil
    863c:	and    dil,al
    863f:	je     8620 <_ZN9GmresCaseIfLm16EE5applyEPv+0x9e0>
    8641:	movss  xmm2,DWORD PTR [rbp+rsi*4+0xb48]
    864a:	lea    r8,[rsi+0x1]
    864e:	cmp    rsi,0xf
    8652:	je     86b2 <_ZN9GmresCaseIfLm16EE5applyEPv+0xa72>
    8654:	mov    rdx,r8
    8657:	nop    DWORD PTR [rax+rax*1+0x0]
    865f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    866a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8675:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8680:	mov    rax,rdx
    8683:	add    rdx,0x1
    8687:	cmp    rax,rcx
    868a:	jae    86ac <_ZN9GmresCaseIfLm16EE5applyEPv+0xa6c>
    868c:	imul   rax,rdx
    8690:	shr    rax,1
    8693:	add    rax,rsi
    8696:	movss  xmm0,DWORD PTR [rbp+rax*4+0x8e4]
    869f:	mulss  xmm0,DWORD PTR [rbp+rdx*4+0xc08]
    86a8:	subss  xmm2,xmm0
    86ac:	cmp    rdx,0x10
    86b0:	jne    8680 <_ZN9GmresCaseIfLm16EE5applyEPv+0xa40>
    86b2:	imul   r8,rsi
    86b6:	shr    r8,1
    86b9:	lea    rax,[r8+rsi*1]
    86bd:	movss  xmm0,DWORD PTR [rbp+rax*4+0x8e4]
    86c6:	comiss xmm4,xmm0
    86c9:	ja     8759 <_ZN9GmresCaseIfLm16EE5applyEPv+0xb19>
    86cf:	comiss xmm3,xmm0
    86d2:	jbe    8779 <_ZN9GmresCaseIfLm16EE5applyEPv+0xb39>
    86d8:	lea    rax,[rsi-0x1]
    86dc:	test   rsi,rsi
    86df:	je     8715 <_ZN9GmresCaseIfLm16EE5applyEPv+0xad5>
    86e1:	mov    DWORD PTR [rbp+rax*4+0xc0c],0x0
    86ec:	sub    rsi,0x2
    86f0:	cmp    rsi,0xffffffffffffffff
    86f4:	je     8715 <_ZN9GmresCaseIfLm16EE5applyEPv+0xad5>
    86f6:	mov    DWORD PTR [rbp+rsi*4+0xc0c],0x0
    8701:	xor    eax,eax
    8703:	sub    rsi,0x1
    8707:	jae    862a <_ZN9GmresCaseIfLm16EE5applyEPv+0x9ea>
    870d:	test   al,al
    870f:	jne    7fb0 <_ZN9GmresCaseIfLm16EE5applyEPv+0x370>
    8715:	mov    eax,0x1
    871a:	mov    BYTE PTR [rbp+0xc59],al
    8720:	jmp    8190 <_ZN9GmresCaseIfLm16EE5applyEPv+0x550>
    8725:	nop    DWORD PTR [rax]
    8728:	movaps xmm1,xmm0
    872b:	xorps  xmm9,xmm0
    872f:	mov    r15,r10
    8732:	mov    rcx,r10
    8735:	cmpltss xmm1,DWORD PTR [rip+0x3b1a]        # c258 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x18>
    873e:	andps  xmm9,xmm1
    8742:	andnps xmm1,xmm0
    8745:	orps   xmm1,xmm9
    8749:	comiss xmm1,DWORD PTR [rsp+0x48]
    874e:	seta   dl
    8751:	xor    r13d,r13d
    8754:	jmp    8243 <_ZN9GmresCaseIfLm16EE5applyEPv+0x603>
    8759:	movaps xmm5,xmm0
    875c:	xorps  xmm5,XMMWORD PTR [rip+0x3aed]        # c250 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x10>
    8763:	comiss xmm3,xmm5
    8766:	ja     86d8 <_ZN9GmresCaseIfLm16EE5applyEPv+0xa98>
    876c:	movss  xmm7,DWORD PTR [rsp+0x6c]
    8772:	minss  xmm7,xmm0
    8776:	movaps xmm0,xmm7
    8779:	divss  xmm2,xmm0
    877d:	mov    eax,edi
    877f:	movss  DWORD PTR [rbp+rsi*4+0xc0c],xmm2
    8788:	jmp    8620 <_ZN9GmresCaseIfLm16EE5applyEPv+0x9e0>
    878d:	comiss xmm0,DWORD PTR [rip+0x3890]        # c024 <_IO_stdin_used+0x24>
    8794:	pxor   xmm2,xmm2
    8798:	jb     87a6 <_ZN9GmresCaseIfLm16EE5applyEPv+0xb66>
    879a:	movss  xmm2,DWORD PTR [rip+0x3862]        # c004 <_IO_stdin_used+0x4>
    87a2:	divss  xmm2,xmm0
    87a6:	movss  xmm0,DWORD PTR [rsp+0x140]
    87af:	add    r13,0x40
    87b3:	mov    rax,r9
    87b6:	comiss xmm8,DWORD PTR [rip+0x3866]        # c024 <_IO_stdin_used+0x24>
    87be:	jb     8800 <_ZN9GmresCaseIfLm16EE5applyEPv+0xbc0>
    87c0:	mulss  xmm0,xmm2
    87c4:	add    r8,0x8
    87c8:	movss  DWORD PTR [rax],xmm0
    87cc:	movss  xmm0,DWORD PTR [r8-0x4]
    87d2:	mulss  xmm0,xmm2
    87d6:	movss  DWORD PTR [rax+0x4],xmm0
    87db:	cmp    r8,r13
    87de:	je     846b <_ZN9GmresCaseIfLm16EE5applyEPv+0x82b>
    87e4:	movss  xmm0,DWORD PTR [r8]
    87e9:	add    rax,0x8
    87ed:	jmp    87c0 <_ZN9GmresCaseIfLm16EE5applyEPv+0xb80>
    87ef:	nop    WORD PTR [rax+rax*1+0x0]
    87f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8800:	divss  xmm0,xmm8
    8805:	add    r8,0x8
    8809:	movss  DWORD PTR [rax],xmm0
    880d:	movss  xmm0,DWORD PTR [r8-0x4]
    8813:	divss  xmm0,xmm8
    8818:	movss  DWORD PTR [rax+0x4],xmm0
    881d:	cmp    r8,r13
    8820:	je     846b <_ZN9GmresCaseIfLm16EE5applyEPv+0x82b>
    8826:	movss  xmm0,DWORD PTR [r8]
    882b:	add    rax,0x8
    882f:	jmp    8800 <_ZN9GmresCaseIfLm16EE5applyEPv+0xbc0>
    8831:	nop    DWORD PTR [rax+0x0]
    8838:	mov    r10,rbp
    883b:	movdqa xmm0,XMMWORD PTR [rsp+0x80]
    8844:	movups XMMWORD PTR [r10+0x480],xmm0
    884c:	movdqa xmm0,XMMWORD PTR [rsp+0x90]
    8855:	movups XMMWORD PTR [r10+0x490],xmm0
    885d:	movdqa xmm0,XMMWORD PTR [rsp+0xa0]
    8866:	movups XMMWORD PTR [r10+0x4a0],xmm0
    886e:	movdqa xmm0,XMMWORD PTR [rsp+0xb0]
    8877:	movups XMMWORD PTR [r10+0x4b0],xmm0
    887f:	mov    rax,QWORD PTR [rsp+0x1c8]
    8887:	sub    rax,QWORD PTR fs:0x28
    8890:	jne    895c <_ZN9GmresCaseIfLm16EE5applyEPv+0xd1c>
    8896:	add    rsp,0x1d8
    889d:	pop    rbx
    889e:	pop    rbp
    889f:	pop    r12
    88a1:	pop    r13
    88a3:	pop    r14
    88a5:	pop    r15
    88a7:	ret
    88a8:	movss  xmm1,DWORD PTR [rsp+0x4c]
    88ae:	movss  xmm3,DWORD PTR [rsp+0x68]
    88b4:	mov    r13d,r14d
    88b7:	mov    ecx,0xf
    88bc:	mov    r9,QWORD PTR [rsp+0x18]
    88c1:	mov    r11,QWORD PTR [rsp+0x50]
    88c6:	mov    rbx,QWORD PTR [rsp+0x58]
    88cb:	jmp    860e <_ZN9GmresCaseIfLm16EE5applyEPv+0x9ce>
    88d0:	movss  xmm1,DWORD PTR [rsp+0x4c]
    88d6:	movss  xmm3,DWORD PTR [rsp+0x68]
    88dc:	mov    r13d,r14d
    88df:	mov    ecx,0x10
    88e4:	mov    r9,QWORD PTR [rsp+0x18]
    88e9:	mov    r11,QWORD PTR [rsp+0x50]
    88ee:	mov    rbx,QWORD PTR [rsp+0x58]
    88f3:	jmp    860e <_ZN9GmresCaseIfLm16EE5applyEPv+0x9ce>
    88f8:	mov    r14,QWORD PTR [rsp+0x40]
    88fd:	mov    rdx,QWORD PTR [rsp+0x78]
    8902:	mov    esi,0x10
    8907:	mov    rdi,rbp
    890a:	mov    r8d,0x10
    8910:	mov    QWORD PTR [rsp+0x18],r11
    8915:	mov    rcx,r14
    8918:	movss  DWORD PTR [rsp+0x10],xmm3
    891e:	call   2080 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIfLm16ELm16EEEfLm16EEEfLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    8923:	mov    rdx,QWORD PTR [rsp+0x60]
    8928:	mov    r11,QWORD PTR [rsp+0x18]
    892d:	mov    rax,r14
    8930:	movss  xmm3,DWORD PTR [rsp+0x10]
    8936:	movups xmm0,XMMWORD PTR [rdx]
    8939:	subps  xmm0,XMMWORD PTR [rax]
    893c:	add    rdx,0x10
    8940:	add    rax,0x10
    8944:	movaps XMMWORD PTR [rax-0x10],xmm0
    8948:	cmp    rdx,QWORD PTR [rsp+0x70]
    894d:	jne    8936 <_ZN9GmresCaseIfLm16EE5applyEPv+0xcf6>
    894f:	movss  xmm0,DWORD PTR [rbp+0xc4c]
    8957:	jmp    8167 <_ZN9GmresCaseIfLm16EE5applyEPv+0x527>
    895c:	call   10b0 <__stack_chk_fail@plt>
    8961:	cs nop WORD PTR [rax+rax*1+0x0]
    896b:	nop    DWORD PTR [rax+rax*1+0x0]

0000000000008970 <_Z3runI9GmresCaseIfLm16EEEvv>:
    8970:	endbr64
    8974:	push   r12
    8976:	mov    ecx,0x80
    897b:	pxor   xmm0,xmm0
    897f:	xor    esi,esi
    8981:	movabs r8,0xcccccccccccccccd
    898b:	push   rbp
    898c:	push   rbx
    898d:	sub    rsp,0xc80
    8994:	mov    rax,QWORD PTR fs:0x28
    899d:	mov    QWORD PTR [rsp+0xc78],rax
    89a5:	xor    eax,eax
    89a7:	lea    rdi,[rsp+0x10]
    89ac:	lea    rbx,[rsp+0x10]
    89b1:	mov    QWORD PTR [rsp+0x4d8],0x10
    89bd:	mov    QWORD PTR [rsp+0x4e8],0x2edbe6ff
    89c9:	mov    DWORD PTR [rsp+0x4f0],0x0
    89d4:	movaps XMMWORD PTR [rsp+0x410],xmm0
    89dc:	movaps XMMWORD PTR [rsp+0x420],xmm0
    89e4:	movaps XMMWORD PTR [rsp+0x430],xmm0
    89ec:	rep stos QWORD PTR [rdi],rax
    89ef:	lea    rdi,[rsp+0x4f8]
    89f7:	mov    ecx,0x7f
    89fc:	mov    QWORD PTR [rsp+0x4f4],0x0
    8a08:	mov    QWORD PTR [rsp+0x8ec],0x0
    8a14:	movaps XMMWORD PTR [rsp+0x440],xmm0
    8a1c:	movaps XMMWORD PTR [rsp+0x450],xmm0
    8a24:	movaps XMMWORD PTR [rsp+0x460],xmm0
    8a2c:	movaps XMMWORD PTR [rsp+0x470],xmm0
    8a34:	movaps XMMWORD PTR [rsp+0x480],xmm0
    8a3c:	movaps XMMWORD PTR [rsp+0x490],xmm0
    8a44:	movaps XMMWORD PTR [rsp+0x4a0],xmm0
    8a4c:	movaps XMMWORD PTR [rsp+0x4b0],xmm0
    8a54:	movaps XMMWORD PTR [rsp+0x4c0],xmm0
    8a5c:	rep stos QWORD PTR [rdi],rax
    8a5f:	lea    rdi,[rsp+0x8f8]
    8a67:	mov    ecx,0x43
    8a6c:	mov    QWORD PTR [rsp+0x8f4],0x0
    8a78:	mov    QWORD PTR [rsp+0xb0c],0x0
    8a84:	rep stos QWORD PTR [rdi],rax
    8a87:	xor    eax,eax
    8a89:	movups XMMWORD PTR [rsp+0xb14],xmm0
    8a91:	mov    rdi,rbx
    8a94:	mov    DWORD PTR [rsp+0xb54],0x0
    8a9f:	movss  xmm1,DWORD PTR [rip+0x358d]        # c034 <_IO_stdin_used+0x34>
    8aa7:	mov    WORD PTR [rsp+0xc68],ax
    8aaf:	movss  xmm3,DWORD PTR [rip+0x3581]        # c038 <_IO_stdin_used+0x38>
    8ab7:	mov    BYTE PTR [rsp+0xc6a],0x0
    8abf:	movss  xmm2,DWORD PTR [rip+0x3569]        # c030 <_IO_stdin_used+0x30>
    8ac7:	mov    DWORD PTR [rsp+0xb98],0x0
    8ad2:	mov    DWORD PTR [rsp+0xc5c],0x0
    8add:	mov    QWORD PTR [rsp+0xc60],0x0
    8ae9:	movups XMMWORD PTR [rsp+0xb24],xmm0
    8af1:	movups XMMWORD PTR [rsp+0xb34],xmm0
    8af9:	movups XMMWORD PTR [rsp+0xb44],xmm0
    8b01:	movups XMMWORD PTR [rsp+0xb58],xmm0
    8b09:	movups XMMWORD PTR [rsp+0xb68],xmm0
    8b11:	movups XMMWORD PTR [rsp+0xb78],xmm0
    8b19:	movups XMMWORD PTR [rsp+0xb88],xmm0
    8b21:	movups XMMWORD PTR [rsp+0xb9c],xmm0
    8b29:	movups XMMWORD PTR [rsp+0xbac],xmm0
    8b31:	movups XMMWORD PTR [rsp+0xbbc],xmm0
    8b39:	movups XMMWORD PTR [rsp+0xbcc],xmm0
    8b41:	movups XMMWORD PTR [rsp+0xbdc],xmm0
    8b49:	movups XMMWORD PTR [rsp+0xbec],xmm0
    8b51:	movups XMMWORD PTR [rsp+0xbfc],xmm0
    8b59:	movups XMMWORD PTR [rsp+0xc0c],xmm0
    8b61:	movups XMMWORD PTR [rsp+0xc1c],xmm0
    8b69:	movups XMMWORD PTR [rsp+0xc2c],xmm0
    8b71:	movups XMMWORD PTR [rsp+0xc3c],xmm0
    8b79:	movups XMMWORD PTR [rsp+0xc4c],xmm0
    8b81:	nop    DWORD PTR [rax+0x0]
    8b85:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8b90:	movss  xmm4,DWORD PTR [rdi+0x400]
    8b98:	mov    rcx,rsi
    8b9b:	mov    r10,rdi
    8b9e:	xor    r9d,r9d
    8ba1:	jmp    8beb <_Z3runI9GmresCaseIfLm16EEEvv+0x27b>
    8ba3:	nop    DWORD PTR [rax+rax*1+0x0]
    8ba8:	mov    rax,rcx
    8bab:	pxor   xmm0,xmm0
    8baf:	mul    r8
    8bb2:	mov    rax,rdx
    8bb5:	and    rdx,0xfffffffffffffffc
    8bb9:	shr    rax,0x2
    8bbd:	add    rdx,rax
    8bc0:	mov    rax,rcx
    8bc3:	sub    rax,rdx
    8bc6:	cvtsi2ss xmm0,rax
    8bcb:	mulss  xmm0,xmm1
    8bcf:	add    r9,0x1
    8bd3:	addss  xmm4,xmm0
    8bd7:	add    r10,0x40
    8bdb:	add    rcx,0x1
    8bdf:	movss  DWORD PTR [r10-0x40],xmm0
    8be5:	cmp    r9,0x10
    8be9:	je     8c08 <_Z3runI9GmresCaseIfLm16EEEvv+0x298>
    8beb:	cmp    rsi,r9
    8bee:	jne    8ba8 <_Z3runI9GmresCaseIfLm16EEEvv+0x238>
    8bf0:	pxor   xmm0,xmm0
    8bf4:	cvtsi2ss xmm0,rsi
    8bf9:	mulss  xmm0,xmm3
    8bfd:	addss  xmm0,xmm2
    8c01:	jmp    8bcf <_Z3runI9GmresCaseIfLm16EEEvv+0x25f>
    8c03:	nop    DWORD PTR [rax+rax*1+0x0]
    8c08:	add    rsi,0x1
    8c0c:	movss  DWORD PTR [rdi+0x400],xmm4
    8c14:	add    rdi,0x4
    8c18:	cmp    rsi,0x10
    8c1c:	jne    8b90 <_Z3runI9GmresCaseIfLm16EEEvv+0x220>
    8c22:	mov    rdi,rbx
    8c25:	lea    r12,[rbx+0x40]
    8c29:	mov    QWORD PTR [rsp+0x4d0],0x1
    8c35:	mov    QWORD PTR [rsp+0x4e0],0x1
    8c41:	mov    QWORD PTR [rsp+0x4e8],0x15f79688
    8c4d:	mov    DWORD PTR [rsp+0x4f0],0x0
    8c58:	call   7c40 <_ZN9GmresCaseIfLm16EE5applyEPv>
    8c5d:	call   28a0 <instruction_count_begin>
    8c62:	mov    rdi,rbx
    8c65:	call   7c40 <_ZN9GmresCaseIfLm16EE5applyEPv>
    8c6a:	call   28b0 <instruction_count_end>
    8c6f:	lea    rsi,[rip+0x33c6]        # c03c <_IO_stdin_used+0x3c>
    8c76:	mov    edi,0x2
    8c7b:	xor    eax,eax
    8c7d:	call   10e0 <__printf_chk@plt>
    8c82:	mov    rdx,rbx
    8c85:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8c90:	mov    rbp,rdx
    8c93:	lea    rcx,[rdx+0x400]
    8c9a:	nop    WORD PTR [rax+rax*1+0x0]
    8ca0:	pxor   xmm0,xmm0
    8ca4:	lea    rsi,[rip+0x33a2]        # c04d <_IO_stdin_used+0x4d>
    8cab:	mov    edi,0x2
    8cb0:	mov    eax,0x1
    8cb5:	cvtss2sd xmm0,DWORD PTR [rbp+0x0]
    8cba:	mov    QWORD PTR [rsp+0x8],rcx
    8cbf:	add    rbp,0x40
    8cc3:	mov    QWORD PTR [rsp],rdx
    8cc7:	call   10e0 <__printf_chk@plt>
    8ccc:	mov    rcx,QWORD PTR [rsp+0x8]
    8cd1:	mov    rdx,QWORD PTR [rsp]
    8cd5:	cmp    rbp,rcx
    8cd8:	jne    8ca0 <_Z3runI9GmresCaseIfLm16EEEvv+0x330>
    8cda:	add    rdx,0x4
    8cde:	cmp    rdx,r12
    8ce1:	jne    8c90 <_Z3runI9GmresCaseIfLm16EEEvv+0x320>
    8ce3:	lea    rbp,[rbx+0x400]
    8cea:	lea    r12,[rbx+0x440]
    8cf1:	nop    DWORD PTR [rax+0x0]
    8cf5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8d00:	pxor   xmm0,xmm0
    8d04:	lea    rsi,[rip+0x3342]        # c04d <_IO_stdin_used+0x4d>
    8d0b:	mov    edi,0x2
    8d10:	mov    eax,0x1
    8d15:	cvtss2sd xmm0,DWORD PTR [rbp+0x0]
    8d1a:	call   10e0 <__printf_chk@plt>
    8d1f:	add    rbp,0x4
    8d23:	cmp    rbp,r12
    8d26:	jne    8d00 <_Z3runI9GmresCaseIfLm16EEEvv+0x390>
    8d28:	lea    rsi,[rip+0x3313]        # c042 <_IO_stdin_used+0x42>
    8d2f:	mov    edi,0x2
    8d34:	xor    eax,eax
    8d36:	call   10e0 <__printf_chk@plt>
    8d3b:	lea    rbp,[rbx+0x480]
    8d42:	add    rbx,0x4c0
    8d49:	nop    DWORD PTR [rax+0x0]
    8d50:	pxor   xmm0,xmm0
    8d54:	lea    rsi,[rip+0x32f2]        # c04d <_IO_stdin_used+0x4d>
    8d5b:	mov    edi,0x2
    8d60:	mov    eax,0x1
    8d65:	cvtss2sd xmm0,DWORD PTR [rbp+0x0]
    8d6a:	call   10e0 <__printf_chk@plt>
    8d6f:	add    rbp,0x4
    8d73:	cmp    rbp,rbx
    8d76:	jne    8d50 <_Z3runI9GmresCaseIfLm16EEEvv+0x3e0>
    8d78:	pxor   xmm0,xmm0
    8d7c:	lea    rsi,[rip+0x32da]        # c05d <_IO_stdin_used+0x5d>
    8d83:	mov    edi,0x2
    8d88:	movzx  ecx,BYTE PTR [rsp+0xc68]
    8d90:	movzx  r8d,BYTE PTR [rsp+0xc69]
    8d99:	mov    eax,0x1
    8d9e:	mov    rdx,QWORD PTR [rsp+0xc60]
    8da6:	cvtss2sd xmm0,DWORD PTR [rsp+0xc5c]
    8daf:	call   10e0 <__printf_chk@plt>
    8db4:	mov    rax,QWORD PTR [rsp+0xc78]
    8dbc:	sub    rax,QWORD PTR fs:0x28
    8dc5:	jne    8dd3 <_Z3runI9GmresCaseIfLm16EEEvv+0x463>
    8dc7:	add    rsp,0xc80
    8dce:	pop    rbx
    8dcf:	pop    rbp
    8dd0:	pop    r12
    8dd2:	ret
    8dd3:	call   10b0 <__stack_chk_fail@plt>
    8dd8:	cs nop WORD PTR [rax+rax*1+0x0]
    8de2:	cs nop WORD PTR [rax+rax*1+0x0]
    8dec:	cs nop WORD PTR [rax+rax*1+0x0]
    8df6:	cs nop WORD PTR [rax+rax*1+0x0]

0000000000008e00 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>:
    8e00:	endbr64
    8e04:	pxor   xmm2,xmm2
    8e08:	xor    eax,eax
    8e0a:	movapd xmm3,xmm2
    8e0e:	nop    WORD PTR [rax+rax*1+0x0]
    8e14:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8e1f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8e2a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8e35:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8e40:	cmp    rax,rsi
    8e43:	jae    8e73 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x73>
    8e45:	movsd  xmm0,QWORD PTR [rdi+rax*8]
    8e4a:	movapd xmm4,xmm0
    8e4e:	movapd xmm1,xmm0
    8e52:	xorpd  xmm1,XMMWORD PTR [rip+0x3406]        # c260 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    8e5a:	cmpltsd xmm0,xmm3
    8e5f:	andpd  xmm1,xmm0
    8e63:	andnpd xmm0,xmm4
    8e67:	orpd   xmm0,xmm1
    8e6b:	maxsd  xmm0,xmm2
    8e6f:	movapd xmm2,xmm0
    8e73:	add    rax,0x1
    8e77:	cmp    rax,0x8
    8e7b:	jne    8e40 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x40>
    8e7d:	pxor   xmm0,xmm0
    8e81:	ucomisd xmm2,xmm0
    8e85:	jp     8e89 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x89>
    8e87:	je     8ed0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd0>
    8e89:	xor    eax,eax
    8e8b:	pxor   xmm1,xmm1
    8e8f:	nop    WORD PTR [rax+rax*1+0x0]
    8e95:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    8ea0:	cmp    rax,rsi
    8ea3:	jae    8eb6 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xb6>
    8ea5:	movsd  xmm0,QWORD PTR [rdi+rax*8]
    8eaa:	divsd  xmm0,xmm2
    8eae:	mulsd  xmm0,xmm0
    8eb2:	addsd  xmm1,xmm0
    8eb6:	add    rax,0x1
    8eba:	cmp    rax,0x8
    8ebe:	jne    8ea0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xa0>
    8ec0:	pxor   xmm0,xmm0
    8ec4:	ucomisd xmm1,xmm0
    8ec8:	jp     8ed8 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd8>
    8eca:	jne    8ed8 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd8>
    8ecc:	mulsd  xmm0,xmm2
    8ed0:	ret
    8ed1:	nop    DWORD PTR [rax+0x0]
    8ed8:	movsd  xmm0,QWORD PTR [rip+0x33b8]        # c298 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x58>
    8ee0:	pxor   xmm3,xmm3
    8ee4:	movapd xmm4,xmm0
    8ee8:	maxsd  xmm4,xmm1
    8eec:	movapd xmm1,xmm0
    8ef0:	maxsd  xmm1,xmm4
    8ef4:	cvtsd2ss xmm3,xmm1
    8ef8:	movd   eax,xmm3
    8efc:	mov    esi,eax
    8efe:	shr    esi,0x17
    8f01:	movzx  ecx,sil
    8f05:	cmp    ecx,0xff
    8f0b:	je     9090 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x290>
    8f11:	test   ecx,ecx
    8f13:	je     9010 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x210>
    8f19:	and    eax,0x807fffff
    8f1e:	mov    edx,0x7e
    8f23:	and    esi,0x1
    8f26:	or     eax,0x3f000000
    8f2b:	sub    edx,ecx
    8f2d:	movd   xmm3,eax
    8f31:	lea    rax,[rip+0x3308]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    8f38:	sar    edx,1
    8f3a:	movss  xmm5,DWORD PTR [rax+rsi*4]
    8f3f:	mulss  xmm3,DWORD PTR [rip+0x30c5]        # c00c <_IO_stdin_used+0xc>
    8f47:	movss  xmm0,DWORD PTR [rip+0x30c1]        # c010 <_IO_stdin_used+0x10>
    8f4f:	subss  xmm0,xmm3
    8f53:	mulss  xmm0,xmm5
    8f57:	movd   ecx,xmm0
    8f5b:	mov    eax,ecx
    8f5d:	shr    eax,0x17
    8f60:	movzx  eax,al
    8f63:	cmp    eax,0xff
    8f68:	je     8faa <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x1aa>
    8f6a:	mov    esi,ecx
    8f6c:	mov    edi,eax
    8f6e:	and    esi,0x7fffff
    8f74:	or     edi,esi
    8f76:	je     8faa <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x1aa>
    8f78:	mov    edi,ecx
    8f7a:	and    edi,0x80000000
    8f80:	test   eax,eax
    8f82:	je     909f <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x29f>
    8f88:	add    eax,edx
    8f8a:	je     914c <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x34c>
    8f90:	cmp    eax,0xfe
    8f95:	jg     914c <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x34c>
    8f9b:	shl    eax,0x17
    8f9e:	and    ecx,0x807fffff
    8fa4:	or     eax,ecx
    8fa6:	movd   xmm0,eax
    8faa:	movapd xmm3,xmm1
    8fae:	cvtss2sd xmm0,xmm0
    8fb2:	movsd  xmm6,QWORD PTR [rip+0x32e6]        # c2a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x60>
    8fba:	movsd  xmm5,QWORD PTR [rip+0x32e6]        # c2a8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x68>
    8fc2:	mulsd  xmm3,xmm0
    8fc6:	mulsd  xmm1,QWORD PTR [rip+0x32ea]        # c2b8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x78>
    8fce:	mulsd  xmm3,xmm0
    8fd2:	mulsd  xmm6,xmm3
    8fd6:	subsd  xmm5,xmm6
    8fda:	mulsd  xmm5,xmm3
    8fde:	movsd  xmm3,QWORD PTR [rip+0x32ca]        # c2b0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x70>
    8fe6:	subsd  xmm3,xmm5
    8fea:	mulsd  xmm3,xmm0
    8fee:	movsd  xmm0,QWORD PTR [rip+0x32ca]        # c2c0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
    8ff6:	mulsd  xmm1,xmm3
    8ffa:	mulsd  xmm1,xmm3
    8ffe:	subsd  xmm0,xmm1
    9002:	mulsd  xmm0,xmm3
    9006:	mulsd  xmm0,xmm4
    900a:	mulsd  xmm0,xmm2
    900e:	ret
    900f:	nop
    9010:	mov    esi,eax
    9012:	and    esi,0x7fffff
    9018:	je     9090 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x290>
    901a:	mov    edi,eax
    901c:	and    edi,0x80000000
    9022:	test   eax,0x7f0000
    9027:	mov    eax,esi
    9029:	je     9120 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x320>
    902f:	shl    eax,0x8
    9032:	mov    edx,0x8
    9037:	cmp    eax,0xfffffff
    903c:	jbe    9136 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x336>
    9042:	cmp    eax,0x3fffffff
    9047:	ja     9053 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x253>
    9049:	add    edx,0x2
    904c:	shl    eax,0x2
    904f:	test   eax,eax
    9051:	js     9056 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x256>
    9053:	add    edx,0x1
    9056:	lea    ecx,[rdx-0x9]
    9059:	lea    rax,[rip+0x31e0]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    9060:	sar    edx,1
    9062:	shl    esi,cl
    9064:	not    ecx
    9066:	add    edx,0x3a
    9069:	and    esi,0x7fffff
    906f:	and    ecx,0x1
    9072:	or     esi,edi
    9074:	movss  xmm5,DWORD PTR [rax+rcx*4]
    9079:	or     esi,0x3f000000
    907f:	movd   xmm3,esi
    9083:	jmp    8f3f <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x13f>
    9088:	nop    DWORD PTR [rax+rax*1+0x0]
    9090:	movss  xmm5,DWORD PTR [rip+0x2f6c]        # c004 <_IO_stdin_used+0x4>
    9098:	xor    edx,edx
    909a:	jmp    8f3f <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x13f>
    909f:	and    ecx,0x7f0000
    90a5:	mov    eax,esi
    90a7:	je     90f7 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2f7>
    90a9:	shl    eax,0x8
    90ac:	mov    ecx,0x8
    90b1:	cmp    eax,0xfffffff
    90b6:	jbe    9111 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x311>
    90b8:	cmp    eax,0x3fffffff
    90bd:	ja     90cd <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2cd>
    90bf:	add    ecx,0x2
    90c2:	shl    eax,0x2
    90c5:	test   eax,eax
    90c7:	js     916f <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x36f>
    90cd:	sub    ecx,0x8
    90d0:	mov    eax,edx
    90d2:	shl    esi,cl
    90d4:	sub    eax,ecx
    90d6:	and    esi,0x7fffff
    90dc:	add    eax,0x1
    90df:	test   eax,eax
    90e1:	jle    9177 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x377>
    90e7:	or     edi,esi
    90e9:	shl    eax,0x17
    90ec:	or     edi,eax
    90ee:	movd   xmm0,edi
    90f2:	jmp    8faa <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x1aa>
    90f7:	shl    eax,0x10
    90fa:	mov    ecx,0x10
    90ff:	cmp    eax,0xffffff
    9104:	jbe    91a5 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x3a5>
    910a:	cmp    eax,0xfffffff
    910f:	ja     9117 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x317>
    9111:	add    ecx,0x4
    9114:	shl    eax,0x4
    9117:	cmp    eax,0x3fffffff
    911c:	ja     90c5 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2c5>
    911e:	jmp    90bf <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2bf>
    9120:	shl    eax,0x10
    9123:	mov    edx,0x10
    9128:	cmp    eax,0xffffff
    912d:	jbe    9199 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x399>
    912f:	cmp    eax,0xfffffff
    9134:	ja     913c <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x33c>
    9136:	add    edx,0x4
    9139:	shl    eax,0x4
    913c:	cmp    eax,0x3fffffff
    9141:	ja     904f <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x24f>
    9147:	jmp    9049 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x249>
    914c:	mov    edx,esi
    914e:	shr    edx,1
    9150:	or     edx,edi
    9152:	or     edi,0x7f800000
    9158:	or     edx,0x400000
    915e:	cmp    eax,0xfe
    9163:	cmovg  edx,edi
    9166:	movd   xmm0,edx
    916a:	jmp    8faa <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x1aa>
    916f:	sub    ecx,0x9
    9172:	jmp    90d0 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2d0>
    9177:	movd   xmm0,edi
    917b:	cmp    eax,0xffffffea
    917e:	jl     8faa <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x1aa>
    9184:	or     esi,0x800000
    918a:	sub    ecx,edx
    918c:	shr    esi,cl
    918e:	or     esi,edi
    9190:	movd   xmm0,esi
    9194:	jmp    8faa <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x1aa>
    9199:	mov    eax,esi
    919b:	mov    edx,0x18
    91a0:	shl    eax,0x18
    91a3:	jmp    912f <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x32f>
    91a5:	mov    eax,esi
    91a7:	mov    ecx,0x18
    91ac:	shl    eax,0x18
    91af:	jmp    910a <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x30a>
    91b4:	cs nop WORD PTR [rax+rax*1+0x0]
    91be:	xchg   ax,ax

00000000000091c0 <_ZN8NormCaseIdLm8EE5applyEPv>:
    91c0:	endbr64
    91c4:	push   rbx
    91c5:	mov    rsi,QWORD PTR [rdi+0x48]
    91c9:	mov    rbx,rdi
    91cc:	call   8e00 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    91d1:	movsd  QWORD PTR [rbx+0x40],xmm0
    91d6:	pop    rbx
    91d7:	ret
    91d8:	nop    DWORD PTR [rax+rax*1+0x0]

00000000000091e0 <_Z3runI8NormCaseIdLm8EEEvv>:
    91e0:	endbr64
    91e4:	push   rbp
    91e5:	pxor   xmm0,xmm0
    91e9:	push   rbx
    91ea:	sub    rsp,0x68
    91ee:	movsd  xmm1,QWORD PTR [rip+0x30d2]        # c2c8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x88>
    91f6:	mov    rax,QWORD PTR fs:0x28
    91ff:	mov    QWORD PTR [rsp+0x58],rax
    9204:	xor    eax,eax
    9206:	mov    rbp,rsp
    9209:	movaps XMMWORD PTR [rsp],xmm0
    920d:	mov    rbx,rsp
    9210:	mov    QWORD PTR [rsp+0x40],0x0
    9219:	mov    rdx,rsp
    921c:	mov    QWORD PTR [rsp+0x48],0x8
    9225:	movaps XMMWORD PTR [rsp+0x10],xmm0
    922a:	movaps XMMWORD PTR [rsp+0x20],xmm0
    922f:	movaps XMMWORD PTR [rsp+0x30],xmm0
    9234:	nop
    9235:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    9240:	add    rax,0x1
    9244:	pxor   xmm0,xmm0
    9248:	add    rdx,0x8
    924c:	cvtsi2sd xmm0,rax
    9251:	divsd  xmm0,xmm1
    9255:	movsd  QWORD PTR [rdx-0x8],xmm0
    925a:	cmp    rax,0x8
    925e:	jne    9240 <_Z3runI8NormCaseIdLm8EEEvv+0x60>
    9260:	mov    rdi,rbp
    9263:	call   91c0 <_ZN8NormCaseIdLm8EE5applyEPv>
    9268:	call   28a0 <instruction_count_begin>
    926d:	mov    rdi,rbp
    9270:	add    rbp,0x40
    9274:	call   91c0 <_ZN8NormCaseIdLm8EE5applyEPv>
    9279:	call   28b0 <instruction_count_end>
    927e:	lea    rsi,[rip+0x2db7]        # c03c <_IO_stdin_used+0x3c>
    9285:	mov    edi,0x2
    928a:	xor    eax,eax
    928c:	call   10e0 <__printf_chk@plt>
    9291:	nop    DWORD PTR [rax+0x0]
    9295:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    92a0:	movsd  xmm0,QWORD PTR [rbx]
    92a4:	lea    rsi,[rip+0x2da2]        # c04d <_IO_stdin_used+0x4d>
    92ab:	mov    edi,0x2
    92b0:	mov    eax,0x1
    92b5:	add    rbx,0x8
    92b9:	call   10e0 <__printf_chk@plt>
    92be:	cmp    rbx,rbp
    92c1:	jne    92a0 <_Z3runI8NormCaseIdLm8EEEvv+0xc0>
    92c3:	movsd  xmm0,QWORD PTR [rsp+0x40]
    92c9:	mov    edi,0x2
    92ce:	mov    eax,0x1
    92d3:	lea    rsi,[rip+0x2d77]        # c051 <_IO_stdin_used+0x51>
    92da:	call   10e0 <__printf_chk@plt>
    92df:	mov    rax,QWORD PTR [rsp+0x58]
    92e4:	sub    rax,QWORD PTR fs:0x28
    92ed:	jne    92f6 <_Z3runI8NormCaseIdLm8EEEvv+0x116>
    92ef:	add    rsp,0x68
    92f3:	pop    rbx
    92f4:	pop    rbp
    92f5:	ret
    92f6:	call   10b0 <__stack_chk_fail@plt>
    92fb:	nop    DWORD PTR [rax+rax*1+0x0]

0000000000009300 <_ZN9GmresCaseIdLm8EE5applyEPv>:
    9300:	endbr64
    9304:	push   r15
    9306:	pxor   xmm0,xmm0
    930a:	mov    esi,0x8
    930f:	push   r14
    9311:	movapd xmm1,xmm0
    9315:	push   r13
    9317:	push   r12
    9319:	push   rbp
    931a:	push   rbx
    931b:	sub    rsp,0x1d8
    9322:	movsd  xmm2,QWORD PTR [rdi+0x2d8]
    932a:	movsd  xmm3,QWORD PTR [rip+0x2f3e]        # c270 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x30>
    9332:	mov    r15,QWORD PTR fs:0x28
    933b:	mov    QWORD PTR [rsp+0x1c8],r15
    9343:	lea    r15,[rdi+0x200]
    934a:	mov    rbx,QWORD PTR [rdi+0x2c8]
    9351:	mov    QWORD PTR [rdi+0x768],0x0
    935c:	lea    r14,[rsp+0xc0]
    9364:	cmpnltsd xmm1,xmm2
    9369:	comisd xmm0,xmm2
    936d:	mov    QWORD PTR [rsp+0x10],rdi
    9372:	andpd  xmm3,xmm1
    9376:	andnpd xmm1,xmm2
    937a:	movapd xmm2,xmm0
    937e:	setae  BYTE PTR [rdi+0x772]
    9385:	orpd   xmm1,xmm3
    9389:	xor    eax,eax
    938b:	movsd  QWORD PTR [rsp+0x60],xmm1
    9391:	movsd  xmm1,QWORD PTR [rdi+0x2e0]
    9399:	mov    WORD PTR [rdi+0x770],ax
    93a0:	xor    eax,eax
    93a2:	cmp    QWORD PTR [rdi+0x2d0],0x1
    93aa:	cmpltsd xmm2,xmm1
    93af:	seta   al
    93b2:	mov    QWORD PTR [rsp+0x68],r15
    93b7:	add    rax,0x1
    93bb:	mov    QWORD PTR [rsp+0x38],rax
    93c0:	andpd  xmm1,xmm2
    93c4:	movsd  QWORD PTR [rsp],xmm1
    93c9:	movsd  xmm1,QWORD PTR [rdi+0x2e8]
    93d1:	mov    rdi,r15
    93d4:	cmpltsd xmm0,xmm1
    93d9:	movapd xmm7,xmm1
    93dd:	andpd  xmm7,xmm0
    93e1:	movsd  QWORD PTR [rsp+0x8],xmm7
    93e7:	call   8e00 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    93ec:	mulsd  xmm0,QWORD PTR [rsp]
    93f1:	mov    r9,QWORD PTR [rsp+0x10]
    93f6:	mov    rcx,r14
    93f9:	addsd  xmm0,QWORD PTR [rsp+0x8]
    93ff:	mov    r8d,0x8
    9405:	mov    esi,0x8
    940a:	mov    QWORD PTR [rsp+0x30],r14
    940f:	lea    rdx,[rsp+0x80]
    9417:	mov    rdi,r9
    941a:	movsd  QWORD PTR [rsp+0x40],xmm0
    9420:	movdqu xmm0,XMMWORD PTR [r9+0x240]
    9429:	movaps XMMWORD PTR [rsp+0x80],xmm0
    9431:	movdqu xmm0,XMMWORD PTR [r9+0x250]
    943a:	movaps XMMWORD PTR [rsp+0x90],xmm0
    9442:	movdqu xmm0,XMMWORD PTR [r9+0x260]
    944b:	movaps XMMWORD PTR [rsp+0xa0],xmm0
    9453:	movdqu xmm0,XMMWORD PTR [r9+0x270]
    945c:	movaps XMMWORD PTR [rsp+0xb0],xmm0
    9464:	pxor   xmm0,xmm0
    9468:	movaps XMMWORD PTR [rsp+0xc0],xmm0
    9470:	movaps XMMWORD PTR [rsp+0xd0],xmm0
    9478:	movaps XMMWORD PTR [rsp+0xe0],xmm0
    9480:	movaps XMMWORD PTR [rsp+0xf0],xmm0
    9488:	mov    QWORD PTR [rsp+0x78],rdx
    948d:	mov    QWORD PTR [rsp],r9
    9491:	call   2600 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    9496:	mov    r9,QWORD PTR [rsp]
    949a:	mov    rdx,r14
    949d:	mov    rax,r15
    94a0:	lea    rcx,[r9+0x240]
    94a7:	movupd xmm0,XMMWORD PTR [rax]
    94ab:	subpd  xmm0,XMMWORD PTR [rdx]
    94af:	add    rax,0x10
    94b3:	add    rdx,0x10
    94b7:	movaps XMMWORD PTR [rdx-0x10],xmm0
    94bb:	cmp    rax,rcx
    94be:	jne    94a7 <_ZN9GmresCaseIdLm8EE5applyEPv+0x1a7>
    94c0:	mov    rdi,QWORD PTR [rsp+0x30]
    94c5:	mov    esi,0x8
    94ca:	mov    QWORD PTR [rsp],r9
    94ce:	mov    QWORD PTR [rsp+0x8],rcx
    94d3:	call   8e00 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    94d8:	movsd  xmm7,QWORD PTR [rsp+0x40]
    94de:	mov    r9,QWORD PTR [rsp]
    94e2:	comisd xmm7,xmm0
    94e6:	mov    r10,QWORD PTR [r9+0x2c0]
    94ed:	movsd  QWORD PTR [r9+0x760],xmm0
    94f6:	setae  al
    94f9:	mov    BYTE PTR [r9+0x770],al
    9500:	test   r10,r10
    9503:	je     9de9 <_ZN9GmresCaseIdLm8EE5applyEPv+0xae9>
    9509:	mov    edx,0x8
    950e:	mov    rcx,QWORD PTR [rsp+0x8]
    9513:	mov    rbp,r10
    9516:	mov    r12,r9
    9519:	cmp    rbx,rdx
    951c:	cmovbe rdx,rbx
    9520:	mov    QWORD PTR [rsp+0x70],rcx
    9525:	test   rdx,rdx
    9528:	mov    rsi,rdx
    952b:	mov    edx,0x1
    9530:	cmovne rdx,rsi
    9534:	xor    ebx,ebx
    9536:	mov    QWORD PTR [rsp],rdx
    953a:	nop    WORD PTR [rax+rax*1+0x0]
    9540:	test   al,al
    9542:	jne    97c0 <_ZN9GmresCaseIdLm8EE5applyEPv+0x4c0>
    9548:	cmp    BYTE PTR [r12+0x771],0x0
    9551:	jne    97c0 <_ZN9GmresCaseIdLm8EE5applyEPv+0x4c0>
    9557:	movsd  xmm1,QWORD PTR [r12+0x760]
    9561:	comisd xmm1,QWORD PTR [rip+0x2d9f]        # c308 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xc8>
    9569:	pxor   xmm0,xmm0
    956d:	pxor   xmm2,xmm2
    9571:	movups XMMWORD PTR [r12+0x658],xmm0
    957a:	mov    QWORD PTR [r12+0x698],0x0
    9586:	movups XMMWORD PTR [r12+0x668],xmm0
    958f:	movups XMMWORD PTR [r12+0x678],xmm0
    9598:	movups XMMWORD PTR [r12+0x688],xmm0
    95a1:	movsd  QWORD PTR [r12+0x658],xmm1
    95ab:	jb     95b9 <_ZN9GmresCaseIdLm8EE5applyEPv+0x2b9>
    95ad:	movsd  xmm2,QWORD PTR [rip+0x2cdb]        # c290 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x50>
    95b5:	divsd  xmm2,xmm1
    95b9:	comisd xmm1,QWORD PTR [rip+0x2d47]        # c308 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xc8>
    95c1:	mov    rax,QWORD PTR [rsp+0x30]
    95c6:	lea    rcx,[r12+0x2f0]
    95ce:	movsd  xmm0,QWORD PTR [rax]
    95d2:	lea    rdx,[rax+0x40]
    95d6:	jae    97db <_ZN9GmresCaseIdLm8EE5applyEPv+0x4db>
    95dc:	mov    rax,QWORD PTR [rsp+0x30]
    95e1:	add    rax,0x10
    95e5:	divsd  xmm0,xmm1
    95e9:	movsd  QWORD PTR [rcx],xmm0
    95ed:	movsd  xmm0,QWORD PTR [rax-0x8]
    95f2:	divsd  xmm0,xmm1
    95f6:	movsd  QWORD PTR [rcx+0x8],xmm0
    95fb:	cmp    rax,rdx
    95fe:	je     9808 <_ZN9GmresCaseIdLm8EE5applyEPv+0x508>
    9604:	movsd  xmm0,QWORD PTR [rax]
    9608:	add    rcx,0x10
    960c:	jmp    95e1 <_ZN9GmresCaseIdLm8EE5applyEPv+0x2e1>
    960e:	xchg   ax,ax
    9610:	movdqa xmm0,XMMWORD PTR [rsp+0x80]
    9619:	lea    rdx,[rsp+0x180]
    9621:	mov    r8,r12
    9624:	xor    edi,edi
    9626:	mov    rsi,rdx
    9629:	movaps XMMWORD PTR [rsp+0x180],xmm0
    9631:	movdqa xmm0,XMMWORD PTR [rsp+0x90]
    963a:	movaps XMMWORD PTR [rsp+0x190],xmm0
    9642:	movdqa xmm0,XMMWORD PTR [rsp+0xa0]
    964b:	movaps XMMWORD PTR [rsp+0x1a0],xmm0
    9653:	movdqa xmm0,XMMWORD PTR [rsp+0xb0]
    965c:	movaps XMMWORD PTR [rsp+0x1b0],xmm0
    9664:	nop
    9665:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    9670:	mov    r10,r8
    9673:	pxor   xmm2,xmm2
    9677:	xor    eax,eax
    9679:	nop    DWORD PTR [rax+0x0]
    9680:	cmp    rax,rcx
    9683:	jae    969c <_ZN9GmresCaseIdLm8EE5applyEPv+0x39c>
    9685:	movsd  xmm0,QWORD PTR [r10+0x2f0]
    968e:	mulsd  xmm0,QWORD PTR [r12+rax*8+0x720]
    9698:	addsd  xmm2,xmm0
    969c:	add    rax,0x1
    96a0:	add    r10,0x40
    96a4:	cmp    rax,0x8
    96a8:	jne    9680 <_ZN9GmresCaseIdLm8EE5applyEPv+0x380>
    96aa:	addsd  xmm2,QWORD PTR [rsi]
    96ae:	add    rdi,0x1
    96b2:	add    rsi,0x8
    96b6:	add    r8,0x8
    96ba:	movsd  QWORD PTR [rsi-0x8],xmm2
    96bf:	cmp    rdi,0x8
    96c3:	jne    9670 <_ZN9GmresCaseIdLm8EE5applyEPv+0x370>
    96c5:	mov    r15,QWORD PTR [rsp+0x30]
    96ca:	mov    esi,0x8
    96cf:	mov    rdi,r12
    96d2:	mov    BYTE PTR [rsp+0x8],r9b
    96d7:	mov    r8d,0x8
    96dd:	movsd  QWORD PTR [rsp+0x10],xmm1
    96e3:	mov    rcx,r15
    96e6:	call   2600 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    96eb:	mov    rdx,QWORD PTR [rsp+0x68]
    96f0:	mov    rcx,QWORD PTR [rsp+0x70]
    96f5:	mov    rax,r15
    96f8:	movzx  r9d,BYTE PTR [rsp+0x8]
    96fe:	movsd  xmm1,QWORD PTR [rsp+0x10]
    9704:	movupd xmm0,XMMWORD PTR [rdx]
    9708:	subpd  xmm0,XMMWORD PTR [rax]
    970c:	add    rdx,0x10
    9710:	add    rax,0x10
    9714:	movaps XMMWORD PTR [rax-0x10],xmm0
    9718:	cmp    rdx,rcx
    971b:	jne    9704 <_ZN9GmresCaseIdLm8EE5applyEPv+0x404>
    971d:	mov    rdi,QWORD PTR [rsp+0x30]
    9722:	mov    esi,0x8
    9727:	mov    BYTE PTR [rsp+0x8],r9b
    972c:	movsd  QWORD PTR [rsp+0x10],xmm1
    9732:	call   8e00 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    9737:	movsd  xmm1,QWORD PTR [rsp+0x10]
    973d:	movzx  r9d,BYTE PTR [rsp+0x8]
    9743:	comisd xmm1,xmm0
    9747:	jb     9f38 <_ZN9GmresCaseIdLm8EE5applyEPv+0xc38>
    974d:	movdqa xmm1,XMMWORD PTR [rsp+0x180]
    9756:	movsd  QWORD PTR [r12+0x760],xmm0
    9760:	movaps XMMWORD PTR [rsp+0x80],xmm1
    9768:	movdqa xmm1,XMMWORD PTR [rsp+0x190]
    9771:	movaps XMMWORD PTR [rsp+0x90],xmm1
    9779:	movdqa xmm1,XMMWORD PTR [rsp+0x1a0]
    9782:	movaps XMMWORD PTR [rsp+0xa0],xmm1
    978a:	movdqa xmm1,XMMWORD PTR [rsp+0x1b0]
    9793:	movaps XMMWORD PTR [rsp+0xb0],xmm1
    979b:	movsd  xmm6,QWORD PTR [rsp+0x40]
    97a1:	comisd xmm6,xmm0
    97a5:	setae  BYTE PTR [r12+0x770]
    97ae:	setb   al
    97b1:	and    eax,r9d
    97b4:	mov    BYTE PTR [r12+0x771],al
    97bc:	nop    DWORD PTR [rax+0x0]
    97c0:	add    rbx,0x1
    97c4:	cmp    rbp,rbx
    97c7:	je     9de6 <_ZN9GmresCaseIdLm8EE5applyEPv+0xae6>
    97cd:	movzx  eax,BYTE PTR [r12+0x770]
    97d6:	jmp    9540 <_ZN9GmresCaseIdLm8EE5applyEPv+0x240>
    97db:	mulsd  xmm0,xmm2
    97df:	add    rax,0x10
    97e3:	movsd  QWORD PTR [rcx],xmm0
    97e7:	movsd  xmm0,QWORD PTR [rax-0x8]
    97ec:	mulsd  xmm0,xmm2
    97f0:	movsd  QWORD PTR [rcx+0x8],xmm0
    97f5:	cmp    rax,rdx
    97f8:	je     9808 <_ZN9GmresCaseIdLm8EE5applyEPv+0x508>
    97fa:	movsd  xmm0,QWORD PTR [rax]
    97fe:	add    rcx,0x10
    9802:	jmp    97db <_ZN9GmresCaseIdLm8EE5applyEPv+0x4db>
    9804:	nop    DWORD PTR [rax+0x0]
    9808:	pxor   xmm0,xmm0
    980c:	mov    QWORD PTR [rsp+0x50],rbp
    9811:	xor    r9d,r9d
    9814:	xor    ecx,ecx
    9816:	lea    rax,[r12+0x330]
    981e:	mov    QWORD PTR [rsp+0x58],rbx
    9823:	lea    r13,[r12+0x618]
    982b:	mov    edx,0x1
    9830:	mov    QWORD PTR [rsp+0x10],rax
    9835:	lea    r14,[r12+0x370]
    983d:	xor    r15d,r15d
    9840:	movaps XMMWORD PTR [rsp+0x100],xmm0
    9848:	movaps XMMWORD PTR [rsp+0x110],xmm0
    9850:	movaps XMMWORD PTR [rsp+0x120],xmm0
    9858:	movaps XMMWORD PTR [rsp+0x130],xmm0
    9860:	movaps XMMWORD PTR [rsp+0x140],xmm0
    9868:	movaps XMMWORD PTR [rsp+0x150],xmm0
    9870:	movaps XMMWORD PTR [rsp+0x160],xmm0
    9878:	movaps XMMWORD PTR [rsp+0x170],xmm0
    9880:	movsd  QWORD PTR [rsp+0x48],xmm1
    9886:	jmp    98d5 <_ZN9GmresCaseIdLm8EE5applyEPv+0x5d5>
    9888:	nop
    9889:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    9894:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    989f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    98aa:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    98b5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    98c0:	mov    r15,rax
    98c3:	add    r13,0x8
    98c7:	add    r14,0x40
    98cb:	cmp    r15,0x8
    98cf:	je     9c35 <_ZN9GmresCaseIdLm8EE5applyEPv+0x935>
    98d5:	cmp    r15,QWORD PTR [rsp]
    98d9:	lea    rax,[r15+0x1]
    98dd:	setb   bpl
    98e1:	and    bpl,dl
    98e4:	mov    BYTE PTR [rsp+0x8],bpl
    98e9:	je     98c0 <_ZN9GmresCaseIdLm8EE5applyEPv+0x5c0>
    98eb:	movdqu xmm0,XMMWORD PTR [r14-0x80]
    98f1:	mov    r8d,0x8
    98f7:	mov    esi,0x8
    98fc:	lea    rbx,[rsp+0x140]
    9904:	lea    rdx,[rsp+0x100]
    990c:	mov    rcx,rbx
    990f:	mov    rdi,r12
    9912:	mov    QWORD PTR [rsp+0x20],rax
    9917:	movaps XMMWORD PTR [rsp+0x100],xmm0
    991f:	movdqu xmm0,XMMWORD PTR [r14-0x70]
    9925:	movaps XMMWORD PTR [rsp+0x110],xmm0
    992d:	movdqu xmm0,XMMWORD PTR [r14-0x60]
    9933:	movaps XMMWORD PTR [rsp+0x120],xmm0
    993b:	movdqu xmm0,XMMWORD PTR [r14-0x50]
    9941:	movaps XMMWORD PTR [rsp+0x130],xmm0
    9949:	call   2600 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    994e:	mov    esi,0x8
    9953:	mov    rdi,rbx
    9956:	call   8e00 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    995b:	mov    r10,QWORD PTR [rsp+0x38]
    9960:	mov    rax,QWORD PTR [rsp+0x20]
    9965:	xor    esi,esi
    9967:	mov    QWORD PTR [r12+0x650],0x0
    9973:	movsd  QWORD PTR [rsp+0x18],xmm0
    9979:	pxor   xmm0,xmm0
    997d:	movups XMMWORD PTR [r12+0x610],xmm0
    9986:	movups XMMWORD PTR [r12+0x620],xmm0
    998f:	movups XMMWORD PTR [r12+0x630],xmm0
    9998:	movups XMMWORD PTR [r12+0x640],xmm0
    99a1:	lea    rbp,[r12+0x610]
    99a9:	mov    r8,QWORD PTR [rsp+0x10]
    99ae:	lea    rdi,[r12+0x2f0]
    99b6:	mov    rdx,rbp
    99b9:	nop    DWORD PTR [rax+0x0]
    99c0:	pxor   xmm1,xmm1
    99c4:	xor    ecx,ecx
    99c6:	movapd xmm0,xmm1
    99ca:	movupd xmm3,XMMWORD PTR [rdi+rcx*1]
    99cf:	movapd xmm4,xmm3
    99d3:	mulpd  xmm4,xmm3
    99d7:	mulpd  xmm3,XMMWORD PTR [rbx+rcx*1]
    99dc:	add    rcx,0x10
    99e0:	movapd xmm2,xmm4
    99e4:	unpckhpd xmm4,xmm4
    99e8:	addsd  xmm2,xmm1
    99ec:	addsd  xmm0,xmm3
    99f0:	unpckhpd xmm3,xmm3
    99f4:	addsd  xmm4,xmm2
    99f8:	addsd  xmm0,xmm3
    99fc:	movapd xmm1,xmm4
    9a00:	cmp    rcx,0x40
    9a04:	jne    99ca <_ZN9GmresCaseIdLm8EE5applyEPv+0x6ca>
    9a06:	divsd  xmm0,xmm4
    9a0a:	movsd  xmm1,QWORD PTR [rdx]
    9a0e:	mov    r9,rbx
    9a11:	mov    rcx,rdi
    9a14:	mov    r11,rbx
    9a17:	addsd  xmm1,xmm0
    9a1b:	unpcklpd xmm0,xmm0
    9a1f:	movsd  QWORD PTR [rdx],xmm1
    9a23:	movupd xmm3,XMMWORD PTR [rcx]
    9a27:	movapd xmm2,XMMWORD PTR [r11]
    9a2c:	add    rcx,0x10
    9a30:	add    r11,0x10
    9a34:	mulpd  xmm3,xmm0
    9a38:	subpd  xmm2,xmm3
    9a3c:	movaps XMMWORD PTR [r11-0x10],xmm2
    9a41:	cmp    rcx,r8
    9a44:	jne    9a23 <_ZN9GmresCaseIdLm8EE5applyEPv+0x723>
    9a46:	lea    r8,[rcx+0x40]
    9a4a:	add    rdi,0x40
    9a4e:	add    rdx,0x8
    9a52:	cmp    r8,r14
    9a55:	jne    99c0 <_ZN9GmresCaseIdLm8EE5applyEPv+0x6c0>
    9a5b:	add    rsi,0x1
    9a5f:	cmp    rsi,r10
    9a62:	jne    99a1 <_ZN9GmresCaseIdLm8EE5applyEPv+0x6a1>
    9a68:	mov    esi,0x8
    9a6d:	mov    rdi,rbx
    9a70:	mov    QWORD PTR [rsp+0x20],rax
    9a75:	mov    QWORD PTR [rsp+0x28],r9
    9a7a:	call   8e00 <_ZN4Base6Matrix9Numerical4normIdLm8EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    9a7f:	mov    rax,QWORD PTR [rsp+0x20]
    9a84:	movsd  xmm10,QWORD PTR [rip+0x2883]        # c310 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xd0>
    9a8d:	mulsd  xmm10,QWORD PTR [rsp+0x18]
    9a94:	movapd xmm8,xmm0
    9a99:	movsd  QWORD PTR [r13+0x0],xmm0
    9a9f:	comisd xmm10,xmm0
    9aa4:	jae    9ab5 <_ZN9GmresCaseIdLm8EE5applyEPv+0x7b5>
    9aa6:	cmp    rax,QWORD PTR [rsp]
    9aaa:	mov    r9,QWORD PTR [rsp+0x28]
    9aaf:	jb     9e56 <_ZN9GmresCaseIdLm8EE5applyEPv+0xb56>
    9ab5:	movq   xmm9,QWORD PTR [rip+0x27a2]        # c260 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    9abe:	test   r15,r15
    9ac1:	je     9b35 <_ZN9GmresCaseIdLm8EE5applyEPv+0x835>
    9ac3:	movsd  xmm0,QWORD PTR [r12+0x610]
    9acd:	mov    rdx,r12
    9ad0:	lea    rcx,[r13-0x618]
    9ad7:	movq   xmm9,QWORD PTR [rip+0x2780]        # c260 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    9ae0:	movsd  xmm1,QWORD PTR [rdx+0x618]
    9ae8:	movsd  xmm4,QWORD PTR [rdx+0x6e0]
    9af0:	add    rdx,0x8
    9af4:	movsd  xmm2,QWORD PTR [rdx+0x698]
    9afc:	movapd xmm5,xmm1
    9b00:	movapd xmm3,xmm4
    9b04:	mulsd  xmm1,xmm4
    9b08:	mulsd  xmm3,xmm0
    9b0c:	mulsd  xmm5,xmm2
    9b10:	mulsd  xmm2,xmm0
    9b14:	movapd xmm0,xmm1
    9b18:	addsd  xmm3,xmm5
    9b1c:	subsd  xmm0,xmm2
    9b20:	movsd  QWORD PTR [rdx+0x608],xmm3
    9b28:	movsd  QWORD PTR [rdx+0x610],xmm0
    9b30:	cmp    rdx,rcx
    9b33:	jne    9ae0 <_ZN9GmresCaseIdLm8EE5applyEPv+0x7e0>
    9b35:	movsd  xmm11,QWORD PTR [r13-0x8]
    9b3b:	movsd  xmm1,QWORD PTR [r13+0x0]
    9b41:	lea    rsi,[r13+0x88]
    9b48:	lea    rdi,[r13+0xc8]
    9b4f:	mov    QWORD PTR [rsp+0x18],rax
    9b54:	movapd xmm0,xmm11
    9b59:	call   1930 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0>
    9b5e:	mov    rax,QWORD PTR [rsp+0x18]
    9b63:	mulsd  xmm11,QWORD PTR [r13+0xc8]
    9b6c:	movsd  xmm0,QWORD PTR [r13+0x88]
    9b75:	mulsd  xmm0,QWORD PTR [r13+0x0]
    9b7b:	mov    rdx,rax
    9b7e:	imul   rdx,r15
    9b82:	addsd  xmm0,xmm11
    9b87:	shr    rdx,1
    9b8a:	movsd  QWORD PTR [r13-0x8],xmm0
    9b90:	nop    DWORD PTR [rax+rax*1+0x0]
    9b95:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    9ba0:	movsd  xmm0,QWORD PTR [rbp+0x0]
    9ba5:	movsd  QWORD PTR [rbp+rdx*8-0x120],xmm0
    9bae:	add    rbp,0x8
    9bb2:	cmp    rbp,r13
    9bb5:	jne    9ba0 <_ZN9GmresCaseIdLm8EE5applyEPv+0x8a0>
    9bb7:	movsd  xmm0,QWORD PTR [r13+0x88]
    9bc0:	movsd  xmm2,QWORD PTR [r13+0x40]
    9bc6:	xorpd  xmm0,xmm9
    9bcb:	mulsd  xmm0,xmm2
    9bcf:	mulsd  xmm2,QWORD PTR [r13+0xc8]
    9bd8:	movsd  QWORD PTR [r13+0x48],xmm0
    9bde:	movsd  QWORD PTR [r13+0x40],xmm2
    9be4:	add    QWORD PTR [r12+0x768],0x1
    9bed:	comisd xmm10,xmm8
    9bf2:	jb     9d70 <_ZN9GmresCaseIdLm8EE5applyEPv+0xa70>
    9bf8:	cmp    rax,0x8
    9bfc:	je     9ef8 <_ZN9GmresCaseIdLm8EE5applyEPv+0xbf8>
    9c02:	cmp    r15,0x6
    9c06:	je     9f18 <_ZN9GmresCaseIdLm8EE5applyEPv+0xc18>
    9c0c:	add    r13,0x10
    9c10:	sub    r14,0xffffffffffffff80
    9c14:	add    r15,0x3
    9c18:	xor    edx,edx
    9c1a:	movzx  r9d,BYTE PTR [rsp+0x8]
    9c20:	mov    rcx,rax
    9c23:	add    r13,0x8
    9c27:	add    r14,0x40
    9c2b:	cmp    r15,0x8
    9c2f:	jne    98d5 <_ZN9GmresCaseIdLm8EE5applyEPv+0x5d5>
    9c35:	movsd  xmm1,QWORD PTR [rsp+0x48]
    9c3b:	mov    rbp,QWORD PTR [rsp+0x50]
    9c40:	mov    rbx,QWORD PTR [rsp+0x58]
    9c45:	movsd  xmm5,QWORD PTR [rsp+0x60]
    9c4b:	mov    esi,0x7
    9c50:	mov    edx,0x1
    9c55:	pxor   xmm3,xmm3
    9c59:	jmp    9c6a <_ZN9GmresCaseIdLm8EE5applyEPv+0x96a>
    9c5b:	nop    DWORD PTR [rax+rax*1+0x0]
    9c60:	sub    rsi,0x1
    9c64:	jb     9d54 <_ZN9GmresCaseIdLm8EE5applyEPv+0xa54>
    9c6a:	mov    QWORD PTR [r12+rsi*8+0x720],0x0
    9c76:	cmp    rsi,rcx
    9c79:	setb   al
    9c7c:	and    al,dl
    9c7e:	mov    edi,eax
    9c80:	je     9c60 <_ZN9GmresCaseIdLm8EE5applyEPv+0x960>
    9c82:	movsd  xmm2,QWORD PTR [r12+rsi*8+0x658]
    9c8c:	lea    r8,[rsi+0x1]
    9c90:	cmp    rsi,0x7
    9c94:	je     9cf4 <_ZN9GmresCaseIdLm8EE5applyEPv+0x9f4>
    9c96:	mov    rdx,r8
    9c99:	nop    WORD PTR [rax+rax*1+0x0]
    9c9f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    9caa:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    9cb5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    9cc0:	mov    rax,rdx
    9cc3:	add    rdx,0x1
    9cc7:	cmp    rax,rcx
    9cca:	jae    9cee <_ZN9GmresCaseIdLm8EE5applyEPv+0x9ee>
    9ccc:	imul   rax,rdx
    9cd0:	shr    rax,1
    9cd3:	add    rax,rsi
    9cd6:	movsd  xmm0,QWORD PTR [r12+rax*8+0x4f0]
    9ce0:	mulsd  xmm0,QWORD PTR [r12+rdx*8+0x718]
    9cea:	subsd  xmm2,xmm0
    9cee:	cmp    rdx,0x8
    9cf2:	jne    9cc0 <_ZN9GmresCaseIdLm8EE5applyEPv+0x9c0>
    9cf4:	imul   r8,rsi
    9cf8:	shr    r8,1
    9cfb:	lea    rax,[r8+rsi*1]
    9cff:	movsd  xmm0,QWORD PTR [r12+rax*8+0x4f0]
    9d09:	comisd xmm3,xmm0
    9d0d:	ja     9da7 <_ZN9GmresCaseIdLm8EE5applyEPv+0xaa7>
    9d13:	comisd xmm5,xmm0
    9d17:	jbe    9dd1 <_ZN9GmresCaseIdLm8EE5applyEPv+0xad1>
    9d1d:	lea    rax,[rsi-0x1]
    9d21:	test   rsi,rsi
    9d24:	je     9d5c <_ZN9GmresCaseIdLm8EE5applyEPv+0xa5c>
    9d26:	mov    QWORD PTR [r12+rax*8+0x720],0x0
    9d32:	sub    rsi,0x2
    9d36:	cmp    rsi,0xffffffffffffffff
    9d3a:	je     9d5c <_ZN9GmresCaseIdLm8EE5applyEPv+0xa5c>
    9d3c:	mov    QWORD PTR [r12+rsi*8+0x720],0x0
    9d48:	xor    edx,edx
    9d4a:	sub    rsi,0x1
    9d4e:	jae    9c6a <_ZN9GmresCaseIdLm8EE5applyEPv+0x96a>
    9d54:	test   dl,dl
    9d56:	jne    9610 <_ZN9GmresCaseIdLm8EE5applyEPv+0x310>
    9d5c:	mov    eax,0x1
    9d61:	mov    BYTE PTR [r12+0x771],al
    9d69:	jmp    97c0 <_ZN9GmresCaseIdLm8EE5applyEPv+0x4c0>
    9d6e:	xchg   ax,ax
    9d70:	pxor   xmm3,xmm3
    9d74:	movapd xmm1,xmm0
    9d78:	xorpd  xmm9,xmm0
    9d7d:	mov    r15,rax
    9d80:	cmpltsd xmm1,xmm3
    9d85:	mov    rcx,rax
    9d88:	andpd  xmm9,xmm1
    9d8d:	andnpd xmm1,xmm0
    9d91:	orpd   xmm1,xmm9
    9d96:	comisd xmm1,QWORD PTR [rsp+0x40]
    9d9c:	seta   dl
    9d9f:	xor    r9d,r9d
    9da2:	jmp    98c3 <_ZN9GmresCaseIdLm8EE5applyEPv+0x5c3>
    9da7:	movapd xmm4,xmm0
    9dab:	xorpd  xmm4,XMMWORD PTR [rip+0x24ad]        # c260 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    9db3:	comisd xmm5,xmm4
    9db7:	ja     9d1d <_ZN9GmresCaseIdLm8EE5applyEPv+0xa1d>
    9dbd:	movapd xmm4,xmm5
    9dc1:	xorpd  xmm4,XMMWORD PTR [rip+0x2497]        # c260 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    9dc9:	minsd  xmm4,xmm0
    9dcd:	movapd xmm0,xmm4
    9dd1:	divsd  xmm2,xmm0
    9dd5:	mov    edx,edi
    9dd7:	movsd  QWORD PTR [r12+rsi*8+0x720],xmm2
    9de1:	jmp    9c60 <_ZN9GmresCaseIdLm8EE5applyEPv+0x960>
    9de6:	mov    r9,r12
    9de9:	movdqa xmm0,XMMWORD PTR [rsp+0x80]
    9df2:	movups XMMWORD PTR [r9+0x280],xmm0
    9dfa:	movdqa xmm0,XMMWORD PTR [rsp+0x90]
    9e03:	movups XMMWORD PTR [r9+0x290],xmm0
    9e0b:	movdqa xmm0,XMMWORD PTR [rsp+0xa0]
    9e14:	movups XMMWORD PTR [r9+0x2a0],xmm0
    9e1c:	movdqa xmm0,XMMWORD PTR [rsp+0xb0]
    9e25:	movups XMMWORD PTR [r9+0x2b0],xmm0
    9e2d:	mov    rax,QWORD PTR [rsp+0x1c8]
    9e35:	sub    rax,QWORD PTR fs:0x28
    9e3e:	jne    9f95 <_ZN9GmresCaseIdLm8EE5applyEPv+0xc95>
    9e44:	add    rsp,0x1d8
    9e4b:	pop    rbx
    9e4c:	pop    rbp
    9e4d:	pop    r12
    9e4f:	pop    r13
    9e51:	pop    r14
    9e53:	pop    r15
    9e55:	ret
    9e56:	comisd xmm0,QWORD PTR [rip+0x24aa]        # c308 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xc8>
    9e5e:	pxor   xmm2,xmm2
    9e62:	jb     9e70 <_ZN9GmresCaseIdLm8EE5applyEPv+0xb70>
    9e64:	movsd  xmm2,QWORD PTR [rip+0x2424]        # c290 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x50>
    9e6c:	divsd  xmm2,xmm0
    9e70:	add    rbx,0x40
    9e74:	lea    rdx,[r14-0x40]
    9e78:	movsd  xmm0,QWORD PTR [rsp+0x140]
    9e81:	comisd xmm8,QWORD PTR [rip+0x247e]        # c308 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xc8>
    9e8a:	jb     9ec0 <_ZN9GmresCaseIdLm8EE5applyEPv+0xbc0>
    9e8c:	mulsd  xmm0,xmm2
    9e90:	add    r9,0x10
    9e94:	movsd  QWORD PTR [rdx],xmm0
    9e98:	movsd  xmm0,QWORD PTR [r9-0x8]
    9e9e:	mulsd  xmm0,xmm2
    9ea2:	movsd  QWORD PTR [rdx+0x8],xmm0
    9ea7:	cmp    r9,rbx
    9eaa:	je     9ab5 <_ZN9GmresCaseIdLm8EE5applyEPv+0x7b5>
    9eb0:	movsd  xmm0,QWORD PTR [r9]
    9eb5:	add    rdx,0x10
    9eb9:	jmp    9e8c <_ZN9GmresCaseIdLm8EE5applyEPv+0xb8c>
    9ebb:	nop    DWORD PTR [rax+rax*1+0x0]
    9ec0:	divsd  xmm0,xmm8
    9ec5:	add    r9,0x10
    9ec9:	movsd  QWORD PTR [rdx],xmm0
    9ecd:	movsd  xmm0,QWORD PTR [r9-0x8]
    9ed3:	divsd  xmm0,xmm8
    9ed8:	movsd  QWORD PTR [rdx+0x8],xmm0
    9edd:	cmp    r9,rbx
    9ee0:	je     9ab5 <_ZN9GmresCaseIdLm8EE5applyEPv+0x7b5>
    9ee6:	movsd  xmm0,QWORD PTR [r9]
    9eeb:	add    rdx,0x10
    9eef:	jmp    9ec0 <_ZN9GmresCaseIdLm8EE5applyEPv+0xbc0>
    9ef1:	nop    DWORD PTR [rax+0x0]
    9ef8:	movsd  xmm1,QWORD PTR [rsp+0x48]
    9efe:	mov    rbp,QWORD PTR [rsp+0x50]
    9f03:	mov    ecx,0x8
    9f08:	mov    rbx,QWORD PTR [rsp+0x58]
    9f0d:	movzx  r9d,BYTE PTR [rsp+0x8]
    9f13:	jmp    9c45 <_ZN9GmresCaseIdLm8EE5applyEPv+0x945>
    9f18:	movsd  xmm1,QWORD PTR [rsp+0x48]
    9f1e:	mov    rbp,QWORD PTR [rsp+0x50]
    9f23:	mov    ecx,0x7
    9f28:	mov    rbx,QWORD PTR [rsp+0x58]
    9f2d:	movzx  r9d,BYTE PTR [rsp+0x8]
    9f33:	jmp    9c45 <_ZN9GmresCaseIdLm8EE5applyEPv+0x945>
    9f38:	mov    r15,QWORD PTR [rsp+0x30]
    9f3d:	mov    rdx,QWORD PTR [rsp+0x78]
    9f42:	mov    esi,0x8
    9f47:	mov    rdi,r12
    9f4a:	mov    r8d,0x8
    9f50:	mov    BYTE PTR [rsp+0x8],r9b
    9f55:	mov    rcx,r15
    9f58:	call   2600 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm8ELm8EEEdLm8EEEdLm8EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    9f5d:	mov    rdx,QWORD PTR [rsp+0x68]
    9f62:	movzx  r9d,BYTE PTR [rsp+0x8]
    9f68:	mov    rax,r15
    9f6b:	movupd xmm0,XMMWORD PTR [rdx]
    9f6f:	subpd  xmm0,XMMWORD PTR [rax]
    9f73:	add    rdx,0x10
    9f77:	add    rax,0x10
    9f7b:	movaps XMMWORD PTR [rax-0x10],xmm0
    9f7f:	cmp    rdx,QWORD PTR [rsp+0x70]
    9f84:	jne    9f6b <_ZN9GmresCaseIdLm8EE5applyEPv+0xc6b>
    9f86:	movsd  xmm0,QWORD PTR [r12+0x760]
    9f90:	jmp    979b <_ZN9GmresCaseIdLm8EE5applyEPv+0x49b>
    9f95:	call   10b0 <__stack_chk_fail@plt>
    9f9a:	nop    WORD PTR [rax+rax*1+0x0]

0000000000009fa0 <_Z3runI9GmresCaseIdLm8EEEvv>:
    9fa0:	endbr64
    9fa4:	push   r14
    9fa6:	mov    ecx,0x40
    9fab:	pxor   xmm0,xmm0
    9faf:	movabs r8,0xcccccccccccccccd
    9fb9:	push   r13
    9fbb:	push   r12
    9fbd:	push   rbp
    9fbe:	push   rbx
    9fbf:	sub    rsp,0x780
    9fc6:	movq   xmm1,QWORD PTR [rip+0x22a2]        # c270 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x30>
    9fce:	mov    rax,QWORD PTR fs:0x28
    9fd7:	mov    QWORD PTR [rsp+0x778],rax
    9fdf:	xor    eax,eax
    9fe1:	mov    rdi,rsp
    9fe4:	movaps XMMWORD PTR [rsp+0x200],xmm0
    9fec:	mov    rbx,rsp
    9fef:	movaps XMMWORD PTR [rsp+0x210],xmm0
    9ff7:	mov    rbp,rsp
    9ffa:	mov    r9,rsp
    9ffd:	mov    QWORD PTR [rsp+0x2c8],0x8
    a009:	mov    QWORD PTR [rsp+0x2e8],0x0
    a015:	movaps XMMWORD PTR [rsp+0x220],xmm0
    a01d:	movaps XMMWORD PTR [rsp+0x230],xmm0
    a025:	rep stos QWORD PTR [rdi],rax
    a028:	lea    rdi,[rsp+0x2f0]
    a030:	mov    ecx,0x40
    a035:	movaps XMMWORD PTR [rsp+0x240],xmm0
    a03d:	rep stos QWORD PTR [rdi],rax
    a040:	lea    rdi,[rsp+0x4f0]
    a048:	mov    ecx,0x24
    a04d:	movaps XMMWORD PTR [rsp+0x250],xmm0
    a055:	movaps XMMWORD PTR [rsp+0x260],xmm0
    a05d:	movaps XMMWORD PTR [rsp+0x270],xmm0
    a065:	movaps XMMWORD PTR [rsp+0x280],xmm0
    a06d:	movaps XMMWORD PTR [rsp+0x290],xmm0
    a075:	movaps XMMWORD PTR [rsp+0x2a0],xmm0
    a07d:	movaps XMMWORD PTR [rsp+0x2b0],xmm0
    a085:	movups XMMWORD PTR [rsp+0x2d8],xmm1
    a08d:	rep stos QWORD PTR [rdi],rax
    a090:	movsd  xmm2,QWORD PTR [rip+0x2248]        # c2e0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa0>
    a098:	xor    edi,edi
    a09a:	mov    QWORD PTR [rsp+0x650],0x0
    a0a6:	mov    QWORD PTR [rsp+0x698],0x0
    a0b2:	movsd  xmm4,QWORD PTR [rip+0x2216]        # c2d0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x90>
    a0ba:	movsd  xmm3,QWORD PTR [rip+0x2216]        # c2d8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x98>
    a0c2:	movaps XMMWORD PTR [rsp+0x610],xmm0
    a0ca:	mov    QWORD PTR [rsp+0x760],0x0
    a0d6:	mov    QWORD PTR [rsp+0x768],0x0
    a0e2:	mov    QWORD PTR [rsp+0x770],0x0
    a0ee:	movaps XMMWORD PTR [rsp+0x620],xmm0
    a0f6:	movaps XMMWORD PTR [rsp+0x630],xmm0
    a0fe:	movaps XMMWORD PTR [rsp+0x640],xmm0
    a106:	movups XMMWORD PTR [rsp+0x658],xmm0
    a10e:	movups XMMWORD PTR [rsp+0x668],xmm0
    a116:	movups XMMWORD PTR [rsp+0x678],xmm0
    a11e:	movups XMMWORD PTR [rsp+0x688],xmm0
    a126:	movaps XMMWORD PTR [rsp+0x6a0],xmm0
    a12e:	movaps XMMWORD PTR [rsp+0x6b0],xmm0
    a136:	movaps XMMWORD PTR [rsp+0x6c0],xmm0
    a13e:	movaps XMMWORD PTR [rsp+0x6d0],xmm0
    a146:	movaps XMMWORD PTR [rsp+0x6e0],xmm0
    a14e:	movaps XMMWORD PTR [rsp+0x6f0],xmm0
    a156:	movaps XMMWORD PTR [rsp+0x700],xmm0
    a15e:	movaps XMMWORD PTR [rsp+0x710],xmm0
    a166:	movaps XMMWORD PTR [rsp+0x720],xmm0
    a16e:	movaps XMMWORD PTR [rsp+0x730],xmm0
    a176:	movaps XMMWORD PTR [rsp+0x740],xmm0
    a17e:	movaps XMMWORD PTR [rsp+0x750],xmm0
    a186:	cs nop WORD PTR [rax+rax*1+0x0]
    a190:	movsd  xmm1,QWORD PTR [r9+0x200]
    a199:	mov    rsi,r9
    a19c:	xor    ecx,ecx
    a19e:	jmp    a1de <_Z3runI9GmresCaseIdLm8EEEvv+0x23e>
    a1a0:	lea    r10,[rdi+rcx*1]
    a1a4:	pxor   xmm0,xmm0
    a1a8:	mov    rax,r10
    a1ab:	mul    r8
    a1ae:	mov    rax,rdx
    a1b1:	and    rdx,0xfffffffffffffffc
    a1b5:	shr    rax,0x2
    a1b9:	add    rdx,rax
    a1bc:	sub    r10,rdx
    a1bf:	cvtsi2sd xmm0,r10
    a1c4:	mulsd  xmm0,xmm2
    a1c8:	add    rcx,0x1
    a1cc:	movsd  QWORD PTR [rsi],xmm0
    a1d0:	addsd  xmm1,xmm0
    a1d4:	add    rsi,0x40
    a1d8:	cmp    rcx,0x8
    a1dc:	je     a200 <_Z3runI9GmresCaseIdLm8EEEvv+0x260>
    a1de:	cmp    rdi,rcx
    a1e1:	jne    a1a0 <_Z3runI9GmresCaseIdLm8EEEvv+0x200>
    a1e3:	pxor   xmm0,xmm0
    a1e7:	cvtsi2sd xmm0,rdi
    a1ec:	mulsd  xmm0,xmm4
    a1f0:	addsd  xmm0,xmm3
    a1f4:	jmp    a1c8 <_Z3runI9GmresCaseIdLm8EEEvv+0x228>
    a1f6:	cs nop WORD PTR [rax+rax*1+0x0]
    a200:	add    rdi,0x1
    a204:	add    r9,0x8
    a208:	movsd  QWORD PTR [r9+0x1f8],xmm1
    a211:	cmp    rdi,0x8
    a215:	jne    a190 <_Z3runI9GmresCaseIdLm8EEEvv+0x1f0>
    a21b:	mov    rdi,rbx
    a21e:	movq   xmm0,QWORD PTR [rip+0x205a]        # c280 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x40>
    a226:	mov    QWORD PTR [rsp+0x2c0],0x1
    a232:	lea    r13,[rbx+0x200]
    a239:	movups XMMWORD PTR [rsp+0x2d8],xmm0
    a241:	lea    r14,[rbx+0x240]
    a248:	mov    QWORD PTR [rsp+0x2d0],0x1
    a254:	mov    QWORD PTR [rsp+0x2e8],0x0
    a260:	call   9300 <_ZN9GmresCaseIdLm8EE5applyEPv>
    a265:	call   28a0 <instruction_count_begin>
    a26a:	mov    rdi,rbx
    a26d:	call   9300 <_ZN9GmresCaseIdLm8EE5applyEPv>
    a272:	call   28b0 <instruction_count_end>
    a277:	lea    rsi,[rip+0x1dbe]        # c03c <_IO_stdin_used+0x3c>
    a27e:	mov    edi,0x2
    a283:	xor    eax,eax
    a285:	call   10e0 <__printf_chk@plt>
    a28a:	nop    WORD PTR [rax+rax*1+0x0]
    a290:	lea    r12,[r13-0x200]
    a297:	nop    WORD PTR [rax+rax*1+0x0]
    a2a0:	movsd  xmm0,QWORD PTR [r12]
    a2a6:	lea    rsi,[rip+0x1da0]        # c04d <_IO_stdin_used+0x4d>
    a2ad:	mov    edi,0x2
    a2b2:	mov    eax,0x1
    a2b7:	add    r12,0x40
    a2bb:	call   10e0 <__printf_chk@plt>
    a2c0:	cmp    r12,r13
    a2c3:	jne    a2a0 <_Z3runI9GmresCaseIdLm8EEEvv+0x300>
    a2c5:	lea    r13,[r12+0x8]
    a2ca:	cmp    r13,r14
    a2cd:	jne    a290 <_Z3runI9GmresCaseIdLm8EEEvv+0x2f0>
    a2cf:	lea    r12,[rbx+0x40]
    a2d3:	xchg   ax,ax
    a2d5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    a2e0:	lea    rsi,[rip+0x1d66]        # c04d <_IO_stdin_used+0x4d>
    a2e7:	mov    edi,0x2
    a2ec:	mov    eax,0x1
    a2f1:	movsd  xmm0,QWORD PTR [rbx+0x200]
    a2f9:	add    rbx,0x8
    a2fd:	call   10e0 <__printf_chk@plt>
    a302:	cmp    rbx,r12
    a305:	jne    a2e0 <_Z3runI9GmresCaseIdLm8EEEvv+0x340>
    a307:	lea    rsi,[rip+0x1d34]        # c042 <_IO_stdin_used+0x42>
    a30e:	mov    edi,0x2
    a313:	xor    eax,eax
    a315:	call   10e0 <__printf_chk@plt>
    a31a:	nop    WORD PTR [rax+rax*1+0x0]
    a320:	lea    rsi,[rip+0x1d26]        # c04d <_IO_stdin_used+0x4d>
    a327:	mov    edi,0x2
    a32c:	mov    eax,0x1
    a331:	movsd  xmm0,QWORD PTR [rbp+0x280]
    a339:	add    rbp,0x8
    a33d:	call   10e0 <__printf_chk@plt>
    a342:	cmp    rbp,r12
    a345:	jne    a320 <_Z3runI9GmresCaseIdLm8EEEvv+0x380>
    a347:	movzx  ecx,BYTE PTR [rsp+0x770]
    a34f:	mov    rdx,QWORD PTR [rsp+0x768]
    a357:	lea    rsi,[rip+0x1cff]        # c05d <_IO_stdin_used+0x5d>
    a35e:	mov    edi,0x2
    a363:	movsd  xmm0,QWORD PTR [rsp+0x760]
    a36c:	mov    eax,0x1
    a371:	movzx  r8d,BYTE PTR [rsp+0x771]
    a37a:	call   10e0 <__printf_chk@plt>
    a37f:	mov    rax,QWORD PTR [rsp+0x778]
    a387:	sub    rax,QWORD PTR fs:0x28
    a390:	jne    a3a2 <_Z3runI9GmresCaseIdLm8EEEvv+0x402>
    a392:	add    rsp,0x780
    a399:	pop    rbx
    a39a:	pop    rbp
    a39b:	pop    r12
    a39d:	pop    r13
    a39f:	pop    r14
    a3a1:	ret
    a3a2:	call   10b0 <__stack_chk_fail@plt>
    a3a7:	cs nop WORD PTR [rax+rax*1+0x0]
    a3b1:	cs nop WORD PTR [rax+rax*1+0x0]
    a3bb:	nop    DWORD PTR [rax+rax*1+0x0]

000000000000a3c0 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>:
    a3c0:	endbr64
    a3c4:	pxor   xmm2,xmm2
    a3c8:	xor    eax,eax
    a3ca:	movapd xmm3,xmm2
    a3ce:	nop    WORD PTR [rax+rax*1+0x0]
    a3d4:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    a3df:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    a3ea:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    a3f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    a400:	cmp    rax,rsi
    a403:	jae    a433 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x73>
    a405:	movsd  xmm0,QWORD PTR [rdi+rax*8]
    a40a:	movapd xmm4,xmm0
    a40e:	movapd xmm1,xmm0
    a412:	xorpd  xmm1,XMMWORD PTR [rip+0x1e46]        # c260 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    a41a:	cmpltsd xmm0,xmm3
    a41f:	andpd  xmm1,xmm0
    a423:	andnpd xmm0,xmm4
    a427:	orpd   xmm0,xmm1
    a42b:	maxsd  xmm0,xmm2
    a42f:	movapd xmm2,xmm0
    a433:	add    rax,0x1
    a437:	cmp    rax,0x10
    a43b:	jne    a400 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x40>
    a43d:	pxor   xmm0,xmm0
    a441:	ucomisd xmm2,xmm0
    a445:	jp     a449 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x89>
    a447:	je     a490 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd0>
    a449:	xor    eax,eax
    a44b:	pxor   xmm1,xmm1
    a44f:	nop    WORD PTR [rax+rax*1+0x0]
    a455:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    a460:	cmp    rax,rsi
    a463:	jae    a476 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xb6>
    a465:	movsd  xmm0,QWORD PTR [rdi+rax*8]
    a46a:	divsd  xmm0,xmm2
    a46e:	mulsd  xmm0,xmm0
    a472:	addsd  xmm1,xmm0
    a476:	add    rax,0x1
    a47a:	cmp    rax,0x10
    a47e:	jne    a460 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xa0>
    a480:	pxor   xmm0,xmm0
    a484:	ucomisd xmm1,xmm0
    a488:	jp     a498 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd8>
    a48a:	jne    a498 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0xd8>
    a48c:	mulsd  xmm0,xmm2
    a490:	ret
    a491:	nop    DWORD PTR [rax+0x0]
    a498:	movsd  xmm0,QWORD PTR [rip+0x1df8]        # c298 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x58>
    a4a0:	pxor   xmm3,xmm3
    a4a4:	movapd xmm4,xmm0
    a4a8:	maxsd  xmm4,xmm1
    a4ac:	movapd xmm1,xmm0
    a4b0:	maxsd  xmm1,xmm4
    a4b4:	cvtsd2ss xmm3,xmm1
    a4b8:	movd   eax,xmm3
    a4bc:	mov    esi,eax
    a4be:	shr    esi,0x17
    a4c1:	movzx  ecx,sil
    a4c5:	cmp    ecx,0xff
    a4cb:	je     a647 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x287>
    a4d1:	test   ecx,ecx
    a4d3:	je     a5cf <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x20f>
    a4d9:	and    eax,0x807fffff
    a4de:	mov    edx,0x7e
    a4e3:	and    esi,0x1
    a4e6:	or     eax,0x3f000000
    a4eb:	sub    edx,ecx
    a4ed:	movd   xmm3,eax
    a4f1:	lea    rax,[rip+0x1d48]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    a4f8:	sar    edx,1
    a4fa:	movss  xmm5,DWORD PTR [rax+rsi*4]
    a4ff:	mulss  xmm3,DWORD PTR [rip+0x1b05]        # c00c <_IO_stdin_used+0xc>
    a507:	movss  xmm0,DWORD PTR [rip+0x1b01]        # c010 <_IO_stdin_used+0x10>
    a50f:	subss  xmm0,xmm3
    a513:	mulss  xmm0,xmm5
    a517:	movd   ecx,xmm0
    a51b:	mov    eax,ecx
    a51d:	shr    eax,0x17
    a520:	movzx  eax,al
    a523:	cmp    eax,0xff
    a528:	je     a56a <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x1aa>
    a52a:	mov    esi,ecx
    a52c:	mov    edi,eax
    a52e:	and    esi,0x7fffff
    a534:	or     edi,esi
    a536:	je     a56a <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x1aa>
    a538:	mov    edi,ecx
    a53a:	and    edi,0x80000000
    a540:	test   eax,eax
    a542:	je     a656 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x296>
    a548:	add    eax,edx
    a54a:	je     a703 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x343>
    a550:	cmp    eax,0xfe
    a555:	jg     a703 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x343>
    a55b:	shl    eax,0x17
    a55e:	and    ecx,0x807fffff
    a564:	or     eax,ecx
    a566:	movd   xmm0,eax
    a56a:	movapd xmm3,xmm1
    a56e:	cvtss2sd xmm0,xmm0
    a572:	movsd  xmm6,QWORD PTR [rip+0x1d26]        # c2a0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x60>
    a57a:	movsd  xmm5,QWORD PTR [rip+0x1d26]        # c2a8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x68>
    a582:	mulsd  xmm3,xmm0
    a586:	mulsd  xmm1,QWORD PTR [rip+0x1d2a]        # c2b8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x78>
    a58e:	mulsd  xmm3,xmm0
    a592:	mulsd  xmm6,xmm3
    a596:	subsd  xmm5,xmm6
    a59a:	mulsd  xmm5,xmm3
    a59e:	movsd  xmm3,QWORD PTR [rip+0x1d0a]        # c2b0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x70>
    a5a6:	subsd  xmm3,xmm5
    a5aa:	mulsd  xmm3,xmm0
    a5ae:	movsd  xmm0,QWORD PTR [rip+0x1d0a]        # c2c0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x80>
    a5b6:	mulsd  xmm1,xmm3
    a5ba:	mulsd  xmm1,xmm3
    a5be:	subsd  xmm0,xmm1
    a5c2:	mulsd  xmm0,xmm3
    a5c6:	mulsd  xmm0,xmm4
    a5ca:	mulsd  xmm0,xmm2
    a5ce:	ret
    a5cf:	mov    esi,eax
    a5d1:	and    esi,0x7fffff
    a5d7:	je     a647 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x287>
    a5d9:	mov    edi,eax
    a5db:	and    edi,0x80000000
    a5e1:	test   eax,0x7f0000
    a5e6:	mov    eax,esi
    a5e8:	je     a6d7 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x317>
    a5ee:	shl    eax,0x8
    a5f1:	mov    edx,0x8
    a5f6:	cmp    eax,0xfffffff
    a5fb:	jbe    a6ed <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x32d>
    a601:	cmp    eax,0x3fffffff
    a606:	ja     a612 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x252>
    a608:	add    edx,0x2
    a60b:	shl    eax,0x2
    a60e:	test   eax,eax
    a610:	js     a615 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x255>
    a612:	add    edx,0x1
    a615:	lea    ecx,[rdx-0x9]
    a618:	mov    eax,esi
    a61a:	sar    edx,1
    a61c:	shl    eax,cl
    a61e:	not    ecx
    a620:	add    edx,0x3a
    a623:	and    eax,0x7fffff
    a628:	and    ecx,0x1
    a62b:	or     eax,edi
    a62d:	or     eax,0x3f000000
    a632:	movd   xmm3,eax
    a636:	lea    rax,[rip+0x1c03]        # c240 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE>
    a63d:	movss  xmm5,DWORD PTR [rax+rcx*4]
    a642:	jmp    a4ff <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x13f>
    a647:	movss  xmm5,DWORD PTR [rip+0x19b5]        # c004 <_IO_stdin_used+0x4>
    a64f:	xor    edx,edx
    a651:	jmp    a4ff <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x13f>
    a656:	and    ecx,0x7f0000
    a65c:	mov    eax,esi
    a65e:	je     a6ae <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2ee>
    a660:	shl    eax,0x8
    a663:	mov    ecx,0x8
    a668:	cmp    eax,0xfffffff
    a66d:	jbe    a6c8 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x308>
    a66f:	cmp    eax,0x3fffffff
    a674:	ja     a684 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2c4>
    a676:	add    ecx,0x2
    a679:	shl    eax,0x2
    a67c:	test   eax,eax
    a67e:	js     a726 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x366>
    a684:	sub    ecx,0x8
    a687:	mov    eax,edx
    a689:	shl    esi,cl
    a68b:	sub    eax,ecx
    a68d:	and    esi,0x7fffff
    a693:	add    eax,0x1
    a696:	test   eax,eax
    a698:	jle    a72e <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x36e>
    a69e:	or     edi,esi
    a6a0:	shl    eax,0x17
    a6a3:	or     edi,eax
    a6a5:	movd   xmm0,edi
    a6a9:	jmp    a56a <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x1aa>
    a6ae:	shl    eax,0x10
    a6b1:	mov    ecx,0x10
    a6b6:	cmp    eax,0xffffff
    a6bb:	jbe    a75c <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x39c>
    a6c1:	cmp    eax,0xfffffff
    a6c6:	ja     a6ce <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x30e>
    a6c8:	add    ecx,0x4
    a6cb:	shl    eax,0x4
    a6ce:	cmp    eax,0x3fffffff
    a6d3:	ja     a67c <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2bc>
    a6d5:	jmp    a676 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2b6>
    a6d7:	shl    eax,0x10
    a6da:	mov    edx,0x10
    a6df:	cmp    eax,0xffffff
    a6e4:	jbe    a750 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x390>
    a6e6:	cmp    eax,0xfffffff
    a6eb:	ja     a6f3 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x333>
    a6ed:	add    edx,0x4
    a6f0:	shl    eax,0x4
    a6f3:	cmp    eax,0x3fffffff
    a6f8:	ja     a60e <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x24e>
    a6fe:	jmp    a608 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x248>
    a703:	mov    edx,esi
    a705:	shr    edx,1
    a707:	or     edx,edi
    a709:	or     edi,0x7f800000
    a70f:	or     edx,0x400000
    a715:	cmp    eax,0xfe
    a71a:	cmovg  edx,edi
    a71d:	movd   xmm0,edx
    a721:	jmp    a56a <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x1aa>
    a726:	sub    ecx,0x9
    a729:	jmp    a687 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x2c7>
    a72e:	movd   xmm0,edi
    a732:	cmp    eax,0xffffffea
    a735:	jl     a56a <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x1aa>
    a73b:	or     esi,0x800000
    a741:	sub    ecx,edx
    a743:	shr    esi,cl
    a745:	or     esi,edi
    a747:	movd   xmm0,esi
    a74b:	jmp    a56a <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x1aa>
    a750:	mov    eax,esi
    a752:	mov    edx,0x18
    a757:	shl    eax,0x18
    a75a:	jmp    a6e6 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x326>
    a75c:	mov    eax,esi
    a75e:	mov    ecx,0x18
    a763:	shl    eax,0x18
    a766:	jmp    a6c1 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm+0x301>
    a76b:	nop    DWORD PTR [rax+rax*1+0x0]

000000000000a770 <_ZN8NormCaseIdLm16EE5applyEPv>:
    a770:	endbr64
    a774:	push   rbx
    a775:	mov    rsi,QWORD PTR [rdi+0x88]
    a77c:	mov    rbx,rdi
    a77f:	call   a3c0 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    a784:	movsd  QWORD PTR [rbx+0x80],xmm0
    a78c:	pop    rbx
    a78d:	ret
    a78e:	cs nop WORD PTR [rax+rax*1+0x0]
    a798:	nop    DWORD PTR [rax+rax*1+0x0]

000000000000a7a0 <_Z3runI8NormCaseIdLm16EEEvv>:
    a7a0:	endbr64
    a7a4:	push   rbp
    a7a5:	mov    ecx,0x10
    a7aa:	push   rbx
    a7ab:	sub    rsp,0xa8
    a7b2:	movsd  xmm1,QWORD PTR [rip+0x1b2e]        # c2e8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa8>
    a7ba:	mov    rax,QWORD PTR fs:0x28
    a7c3:	mov    QWORD PTR [rsp+0x98],rax
    a7cb:	xor    eax,eax
    a7cd:	mov    rdi,rsp
    a7d0:	mov    rbp,rsp
    a7d3:	mov    rbx,rsp
    a7d6:	mov    QWORD PTR [rsp+0x80],0x0
    a7e2:	mov    rdx,rsp
    a7e5:	mov    QWORD PTR [rsp+0x88],0x10
    a7f1:	rep stos QWORD PTR [rdi],rax
    a7f4:	nop
    a7f5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    a800:	add    rax,0x1
    a804:	pxor   xmm0,xmm0
    a808:	add    rdx,0x8
    a80c:	cvtsi2sd xmm0,rax
    a811:	divsd  xmm0,xmm1
    a815:	movsd  QWORD PTR [rdx-0x8],xmm0
    a81a:	cmp    rax,0x10
    a81e:	jne    a800 <_Z3runI8NormCaseIdLm16EEEvv+0x60>
    a820:	mov    rdi,rbp
    a823:	call   a770 <_ZN8NormCaseIdLm16EE5applyEPv>
    a828:	call   28a0 <instruction_count_begin>
    a82d:	mov    rdi,rbp
    a830:	sub    rbp,0xffffffffffffff80
    a834:	call   a770 <_ZN8NormCaseIdLm16EE5applyEPv>
    a839:	call   28b0 <instruction_count_end>
    a83e:	lea    rsi,[rip+0x17f7]        # c03c <_IO_stdin_used+0x3c>
    a845:	mov    edi,0x2
    a84a:	xor    eax,eax
    a84c:	call   10e0 <__printf_chk@plt>
    a851:	nop    DWORD PTR [rax+0x0]
    a855:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    a860:	movsd  xmm0,QWORD PTR [rbx]
    a864:	lea    rsi,[rip+0x17e2]        # c04d <_IO_stdin_used+0x4d>
    a86b:	mov    edi,0x2
    a870:	mov    eax,0x1
    a875:	add    rbx,0x8
    a879:	call   10e0 <__printf_chk@plt>
    a87e:	cmp    rbx,rbp
    a881:	jne    a860 <_Z3runI8NormCaseIdLm16EEEvv+0xc0>
    a883:	movsd  xmm0,QWORD PTR [rsp+0x80]
    a88c:	mov    edi,0x2
    a891:	mov    eax,0x1
    a896:	lea    rsi,[rip+0x17b4]        # c051 <_IO_stdin_used+0x51>
    a89d:	call   10e0 <__printf_chk@plt>
    a8a2:	mov    rax,QWORD PTR [rsp+0x98]
    a8aa:	sub    rax,QWORD PTR fs:0x28
    a8b3:	jne    a8bf <_Z3runI8NormCaseIdLm16EEEvv+0x11f>
    a8b5:	add    rsp,0xa8
    a8bc:	pop    rbx
    a8bd:	pop    rbp
    a8be:	ret
    a8bf:	call   10b0 <__stack_chk_fail@plt>
    a8c4:	cs nop WORD PTR [rax+rax*1+0x0]
    a8ce:	cs nop WORD PTR [rax+rax*1+0x0]
    a8d8:	cs nop WORD PTR [rax+rax*1+0x0]
    a8e2:	cs nop WORD PTR [rax+rax*1+0x0]
    a8ec:	cs nop WORD PTR [rax+rax*1+0x0]
    a8f6:	cs nop WORD PTR [rax+rax*1+0x0]

000000000000a900 <_ZN9GmresCaseIdLm16EE5applyEPv>:
    a900:	endbr64
    a904:	push   r15
    a906:	pxor   xmm0,xmm0
    a90a:	mov    esi,0x10
    a90f:	push   r14
    a911:	movapd xmm1,xmm0
    a915:	lea    r14,[rdi+0x800]
    a91c:	push   r13
    a91e:	mov    r13,rdi
    a921:	push   r12
    a923:	push   rbp
    a924:	push   rbx
    a925:	sub    rsp,0x348
    a92c:	movsd  xmm2,QWORD PTR [rdi+0x998]
    a934:	movsd  xmm3,QWORD PTR [rip+0x1934]        # c270 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x30>
    a93c:	mov    rbp,QWORD PTR fs:0x28
    a945:	mov    QWORD PTR [rsp+0x338],rbp
    a94d:	xor    ebp,ebp
    a94f:	mov    rbx,QWORD PTR [rdi+0x988]
    a956:	mov    QWORD PTR [rdi+0x1888],0x0
    a961:	lea    r15,[rsp+0x130]
    a969:	cmpnltsd xmm1,xmm2
    a96e:	comisd xmm0,xmm2
    a972:	mov    QWORD PTR [rsp+0x90],r14
    a97a:	andpd  xmm3,xmm1
    a97e:	andnpd xmm1,xmm2
    a982:	movapd xmm2,xmm0
    a986:	setae  BYTE PTR [rdi+0x1892]
    a98d:	orpd   xmm1,xmm3
    a991:	xor    eax,eax
    a993:	movsd  QWORD PTR [rsp+0x88],xmm1
    a99c:	movsd  xmm1,QWORD PTR [rdi+0x9a0]
    a9a4:	mov    WORD PTR [rdi+0x1890],ax
    a9ab:	xor    eax,eax
    a9ad:	cmp    QWORD PTR [rdi+0x990],0x1
    a9b5:	cmpltsd xmm2,xmm1
    a9ba:	seta   al
    a9bd:	add    rax,0x1
    a9c1:	mov    QWORD PTR [rsp+0x48],rax
    a9c6:	andpd  xmm1,xmm2
    a9ca:	movsd  QWORD PTR [rsp+0x8],xmm1
    a9d0:	movsd  xmm1,QWORD PTR [rdi+0x9a8]
    a9d8:	mov    rdi,r14
    a9db:	cmpltsd xmm0,xmm1
    a9e0:	movapd xmm7,xmm1
    a9e4:	andpd  xmm7,xmm0
    a9e8:	movsd  QWORD PTR [rsp+0x10],xmm7
    a9ee:	call   a3c0 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    a9f3:	mulsd  xmm0,QWORD PTR [rsp+0x8]
    a9f9:	mov    rax,rbp
    a9fc:	lea    rdi,[rsp+0x130]
    aa04:	addsd  xmm0,QWORD PTR [rsp+0x10]
    aa0a:	mov    ecx,0x10
    aa0f:	mov    esi,0x10
    aa14:	mov    r8d,0x10
    aa1a:	lea    rdx,[rsp+0xb0]
    aa22:	movsd  QWORD PTR [rsp+0x70],xmm0
    aa28:	movdqu xmm0,XMMWORD PTR [r13+0x880]
    aa31:	movaps XMMWORD PTR [rsp+0xb0],xmm0
    aa39:	movdqu xmm0,XMMWORD PTR [r13+0x890]
    aa42:	movaps XMMWORD PTR [rsp+0xc0],xmm0
    aa4a:	movdqu xmm0,XMMWORD PTR [r13+0x8a0]
    aa53:	movaps XMMWORD PTR [rsp+0xd0],xmm0
    aa5b:	movdqu xmm0,XMMWORD PTR [r13+0x8b0]
    aa64:	movaps XMMWORD PTR [rsp+0xe0],xmm0
    aa6c:	movdqu xmm0,XMMWORD PTR [r13+0x8c0]
    aa75:	movaps XMMWORD PTR [rsp+0xf0],xmm0
    aa7d:	movdqu xmm0,XMMWORD PTR [r13+0x8d0]
    aa86:	movaps XMMWORD PTR [rsp+0x100],xmm0
    aa8e:	movdqu xmm0,XMMWORD PTR [r13+0x8e0]
    aa97:	movaps XMMWORD PTR [rsp+0x110],xmm0
    aa9f:	movdqu xmm0,XMMWORD PTR [r13+0x8f0]
    aaa8:	rep stos QWORD PTR [rdi],rax
    aaab:	mov    rcx,r15
    aaae:	mov    rdi,r13
    aab1:	mov    QWORD PTR [rsp+0x68],r15
    aab6:	mov    QWORD PTR [rsp+0xa0],rdx
    aabe:	movaps XMMWORD PTR [rsp+0x120],xmm0
    aac6:	call   1d80 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    aacb:	lea    r8,[r13+0x880]
    aad2:	mov    rdx,r14
    aad5:	mov    rax,r15
    aad8:	movupd xmm0,XMMWORD PTR [rdx]
    aadc:	subpd  xmm0,XMMWORD PTR [rax]
    aae0:	add    rdx,0x10
    aae4:	add    rax,0x10
    aae8:	movaps XMMWORD PTR [rax-0x10],xmm0
    aaec:	cmp    rdx,r8
    aaef:	jne    aad8 <_ZN9GmresCaseIdLm16EE5applyEPv+0x1d8>
    aaf1:	mov    rdi,QWORD PTR [rsp+0x68]
    aaf6:	mov    esi,0x10
    aafb:	mov    QWORD PTR [rsp+0x8],r8
    ab00:	call   a3c0 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    ab05:	movsd  xmm7,QWORD PTR [rsp+0x70]
    ab0b:	mov    rdi,QWORD PTR [r13+0x980]
    ab12:	movsd  QWORD PTR [r13+0x1880],xmm0
    ab1b:	comisd xmm7,xmm0
    ab1f:	mov    QWORD PTR [rsp+0x50],rdi
    ab24:	setae  al
    ab27:	mov    BYTE PTR [r13+0x1890],al
    ab2e:	test   rdi,rdi
    ab31:	je     b5f8 <_ZN9GmresCaseIdLm16EE5applyEPv+0xcf8>
    ab37:	cmp    rbx,0x10
    ab3b:	mov    edi,0x10
    ab40:	mov    edx,0x1
    ab45:	mov    r8,QWORD PTR [rsp+0x8]
    ab4a:	movsd  xmm6,QWORD PTR [rsp+0x88]
    ab53:	cmovbe rdi,rbx
    ab57:	xorpd  xmm6,XMMWORD PTR [rip+0x1701]        # c260 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    ab5f:	mov    QWORD PTR [rsp+0x98],r8
    ab67:	movsd  QWORD PTR [rsp+0xa8],xmm6
    ab70:	test   rdi,rdi
    ab73:	cmovne rdx,rdi
    ab77:	xor    ebp,ebp
    ab79:	mov    QWORD PTR [rsp+0x18],rdx
    ab7e:	test   al,al
    ab80:	jne    aee8 <_ZN9GmresCaseIdLm16EE5applyEPv+0x5e8>
    ab86:	cmp    BYTE PTR [r13+0x1891],0x0
    ab8e:	jne    aee8 <_ZN9GmresCaseIdLm16EE5applyEPv+0x5e8>
    ab94:	lea    rax,[r13+0x1680]
    ab9b:	pxor   xmm2,xmm2
    ab9f:	mov    QWORD PTR [r13+0x16f8],0x0
    abaa:	movsd  xmm1,QWORD PTR [r13+0x1880]
    abb3:	mov    QWORD PTR [rsp+0x58],rax
    abb8:	and    rax,0xfffffffffffffff8
    abbc:	mov    rdi,rax
    abbf:	mov    eax,r13d
    abc2:	sub    eax,edi
    abc4:	lea    ecx,[rax+0x1700]
    abca:	xor    eax,eax
    abcc:	shr    ecx,0x3
    abcf:	comisd xmm1,QWORD PTR [rip+0x1731]        # c308 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xc8>
    abd7:	rep stos QWORD PTR [rdi],rax
    abda:	movsd  QWORD PTR [r13+0x1678],xmm1
    abe3:	jb     abf1 <_ZN9GmresCaseIdLm16EE5applyEPv+0x2f1>
    abe5:	movsd  xmm2,QWORD PTR [rip+0x16a3]        # c290 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x50>
    abed:	divsd  xmm2,xmm1
    abf1:	comisd xmm1,QWORD PTR [rip+0x170f]        # c308 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xc8>
    abf9:	mov    rax,QWORD PTR [rsp+0x68]
    abfe:	lea    r11,[r13+0x9b0]
    ac05:	mov    rcx,r11
    ac08:	movsd  xmm0,QWORD PTR [rax]
    ac0c:	lea    rdx,[rax+0x80]
    ac13:	jae    ac80 <_ZN9GmresCaseIdLm16EE5applyEPv+0x380>
    ac15:	mov    rax,QWORD PTR [rsp+0x68]
    ac1a:	add    rax,0x10
    ac1e:	divsd  xmm0,xmm1
    ac22:	movsd  QWORD PTR [rcx],xmm0
    ac26:	movsd  xmm0,QWORD PTR [rax-0x8]
    ac2b:	divsd  xmm0,xmm1
    ac2f:	movsd  QWORD PTR [rcx+0x8],xmm0
    ac34:	cmp    rax,rdx
    ac37:	je     af04 <_ZN9GmresCaseIdLm16EE5applyEPv+0x604>
    ac3d:	movsd  xmm0,QWORD PTR [rax]
    ac41:	add    rcx,0x10
    ac45:	jmp    ac1a <_ZN9GmresCaseIdLm16EE5applyEPv+0x31a>
    ac47:	xchg   ax,ax
    ac49:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    ac54:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    ac5f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    ac6a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    ac75:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    ac80:	mulsd  xmm0,xmm2
    ac84:	add    rax,0x10
    ac88:	movsd  QWORD PTR [rcx],xmm0
    ac8c:	movsd  xmm0,QWORD PTR [rax-0x8]
    ac91:	mulsd  xmm0,xmm2
    ac95:	movsd  QWORD PTR [rcx+0x8],xmm0
    ac9a:	cmp    rax,rdx
    ac9d:	je     af04 <_ZN9GmresCaseIdLm16EE5applyEPv+0x604>
    aca3:	movsd  xmm0,QWORD PTR [rax]
    aca7:	add    rcx,0x10
    acab:	jmp    ac80 <_ZN9GmresCaseIdLm16EE5applyEPv+0x380>
    acad:	nop    DWORD PTR [rax]
    acb0:	movdqa xmm0,XMMWORD PTR [rsp+0xb0]
    acb9:	lea    rdx,[rsp+0x2b0]
    acc1:	xor    edi,edi
    acc3:	mov    rsi,rdx
    acc6:	movaps XMMWORD PTR [rsp+0x2b0],xmm0
    acce:	movdqa xmm0,XMMWORD PTR [rsp+0xc0]
    acd7:	movaps XMMWORD PTR [rsp+0x2c0],xmm0
    acdf:	movdqa xmm0,XMMWORD PTR [rsp+0xd0]
    ace8:	movaps XMMWORD PTR [rsp+0x2d0],xmm0
    acf0:	movdqa xmm0,XMMWORD PTR [rsp+0xe0]
    acf9:	movaps XMMWORD PTR [rsp+0x2e0],xmm0
    ad01:	movdqa xmm0,XMMWORD PTR [rsp+0xf0]
    ad0a:	movaps XMMWORD PTR [rsp+0x2f0],xmm0
    ad12:	movdqa xmm0,XMMWORD PTR [rsp+0x100]
    ad1b:	movaps XMMWORD PTR [rsp+0x300],xmm0
    ad23:	movdqa xmm0,XMMWORD PTR [rsp+0x110]
    ad2c:	movaps XMMWORD PTR [rsp+0x310],xmm0
    ad34:	movdqa xmm0,XMMWORD PTR [rsp+0x120]
    ad3d:	movaps XMMWORD PTR [rsp+0x320],xmm0
    ad45:	mov    r8,r11
    ad48:	pxor   xmm2,xmm2
    ad4c:	xor    eax,eax
    ad4e:	nop    WORD PTR [rax+rax*1+0x0]
    ad54:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    ad5f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    ad6a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    ad75:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    ad80:	cmp    rax,rcx
    ad83:	jae    ad98 <_ZN9GmresCaseIdLm16EE5applyEPv+0x498>
    ad85:	movsd  xmm0,QWORD PTR [r8]
    ad8a:	mulsd  xmm0,QWORD PTR [r13+rax*8+0x1800]
    ad94:	addsd  xmm2,xmm0
    ad98:	add    rax,0x1
    ad9c:	sub    r8,0xffffffffffffff80
    ada0:	cmp    rax,0x10
    ada4:	jne    ad80 <_ZN9GmresCaseIdLm16EE5applyEPv+0x480>
    ada6:	addsd  xmm2,QWORD PTR [rsi]
    adaa:	add    rdi,0x1
    adae:	add    rsi,0x8
    adb2:	add    r11,0x8
    adb6:	movsd  QWORD PTR [rsi-0x8],xmm2
    adbb:	cmp    rdi,0x10
    adbf:	jne    ad45 <_ZN9GmresCaseIdLm16EE5applyEPv+0x445>
    adc1:	mov    r14,QWORD PTR [rsp+0x68]
    adc6:	mov    r8d,0x10
    adcc:	mov    esi,0x10
    add1:	mov    rdi,r13
    add4:	movsd  QWORD PTR [rsp+0x8],xmm1
    adda:	mov    rcx,r14
    addd:	call   1d80 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    ade2:	mov    rdx,QWORD PTR [rsp+0x90]
    adea:	mov    rcx,QWORD PTR [rsp+0x98]
    adf2:	mov    rax,r14
    adf5:	movsd  xmm1,QWORD PTR [rsp+0x8]
    adfb:	movupd xmm0,XMMWORD PTR [rdx]
    adff:	subpd  xmm0,XMMWORD PTR [rax]
    ae03:	add    rdx,0x10
    ae07:	add    rax,0x10
    ae0b:	movaps XMMWORD PTR [rax-0x10],xmm0
    ae0f:	cmp    rdx,rcx
    ae12:	jne    adfb <_ZN9GmresCaseIdLm16EE5applyEPv+0x4fb>
    ae14:	mov    rdi,QWORD PTR [rsp+0x68]
    ae19:	mov    esi,0x10
    ae1e:	movsd  QWORD PTR [rsp+0x8],xmm1
    ae24:	call   a3c0 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    ae29:	movsd  xmm1,QWORD PTR [rsp+0x8]
    ae2f:	comisd xmm1,xmm0
    ae33:	jb     b6ed <_ZN9GmresCaseIdLm16EE5applyEPv+0xded>
    ae39:	movdqa xmm1,XMMWORD PTR [rsp+0x2b0]
    ae42:	movsd  QWORD PTR [r13+0x1880],xmm0
    ae4b:	movaps XMMWORD PTR [rsp+0xb0],xmm1
    ae53:	movdqa xmm1,XMMWORD PTR [rsp+0x2c0]
    ae5c:	movaps XMMWORD PTR [rsp+0xc0],xmm1
    ae64:	movdqa xmm1,XMMWORD PTR [rsp+0x2d0]
    ae6d:	movaps XMMWORD PTR [rsp+0xd0],xmm1
    ae75:	movdqa xmm1,XMMWORD PTR [rsp+0x2e0]
    ae7e:	movaps XMMWORD PTR [rsp+0xe0],xmm1
    ae86:	movdqa xmm1,XMMWORD PTR [rsp+0x2f0]
    ae8f:	movaps XMMWORD PTR [rsp+0xf0],xmm1
    ae97:	movdqa xmm1,XMMWORD PTR [rsp+0x300]
    aea0:	movaps XMMWORD PTR [rsp+0x100],xmm1
    aea8:	movdqa xmm1,XMMWORD PTR [rsp+0x310]
    aeb1:	movaps XMMWORD PTR [rsp+0x110],xmm1
    aeb9:	movdqa xmm1,XMMWORD PTR [rsp+0x320]
    aec2:	movaps XMMWORD PTR [rsp+0x120],xmm1
    aeca:	movsd  xmm7,QWORD PTR [rsp+0x70]
    aed0:	comisd xmm7,xmm0
    aed4:	setb   al
    aed7:	setae  BYTE PTR [r13+0x1890]
    aedf:	and    eax,ebx
    aee1:	mov    BYTE PTR [r13+0x1891],al
    aee8:	add    rbp,0x1
    aeec:	cmp    QWORD PTR [rsp+0x50],rbp
    aef1:	je     b5f8 <_ZN9GmresCaseIdLm16EE5applyEPv+0xcf8>
    aef7:	movzx  eax,BYTE PTR [r13+0x1890]
    aeff:	jmp    ab7e <_ZN9GmresCaseIdLm16EE5applyEPv+0x27e>
    af04:	lea    rdx,[rsp+0x1b0]
    af0c:	xor    eax,eax
    af0e:	xor    ebx,ebx
    af10:	xor    r15d,r15d
    af13:	mov    ecx,0x10
    af18:	mov    rdi,rdx
    af1b:	mov    QWORD PTR [rsp+0x20],r11
    af20:	lea    r14,[r13+0xa30]
    af27:	rep stos QWORD PTR [rdi],rax
    af2a:	lea    rdi,[rsp+0x230]
    af32:	mov    ecx,0x10
    af37:	mov    QWORD PTR [rsp+0x80],rbp
    af3f:	mov    QWORD PTR [rsp+0x8],rdi
    af44:	rep stos QWORD PTR [rdi],rax
    af47:	lea    rax,[r13+0x15f8]
    af4e:	movsd  QWORD PTR [rsp+0x78],xmm1
    af54:	mov    QWORD PTR [rsp+0x60],rax
    af59:	mov    r12,rax
    af5c:	mov    eax,0x1
    af61:	jmp    af95 <_ZN9GmresCaseIdLm16EE5applyEPv+0x695>
    af63:	nop    DWORD PTR [rax+0x0]
    af6a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    af75:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    af80:	mov    r15,r9
    af83:	add    r12,0x8
    af87:	sub    r14,0xffffffffffffff80
    af8b:	cmp    r15,0x10
    af8f:	je     b384 <_ZN9GmresCaseIdLm16EE5applyEPv+0xa84>
    af95:	cmp    r15,QWORD PTR [rsp+0x18]
    af9a:	lea    r9,[r15+0x1]
    af9e:	setb   sil
    afa2:	and    sil,al
    afa5:	mov    BYTE PTR [rsp+0x10],sil
    afaa:	je     af80 <_ZN9GmresCaseIdLm16EE5applyEPv+0x680>
    afac:	movdqu xmm0,XMMWORD PTR [r14-0x80]
    afb2:	mov    rbx,QWORD PTR [rsp+0x8]
    afb7:	mov    esi,0x10
    afbc:	mov    rdi,r13
    afbf:	mov    r8d,0x10
    afc5:	mov    QWORD PTR [rsp+0x38],r9
    afca:	movaps XMMWORD PTR [rdx],xmm0
    afcd:	movdqu xmm0,XMMWORD PTR [r14-0x70]
    afd3:	mov    rcx,rbx
    afd6:	mov    QWORD PTR [rsp+0x28],rdx
    afdb:	movaps XMMWORD PTR [rdx+0x10],xmm0
    afdf:	movdqu xmm0,XMMWORD PTR [r14-0x60]
    afe5:	movaps XMMWORD PTR [rdx+0x20],xmm0
    afe9:	movdqu xmm0,XMMWORD PTR [r14-0x50]
    afef:	movaps XMMWORD PTR [rdx+0x30],xmm0
    aff3:	movdqu xmm0,XMMWORD PTR [r14-0x40]
    aff9:	movaps XMMWORD PTR [rdx+0x40],xmm0
    affd:	movdqu xmm0,XMMWORD PTR [r14-0x30]
    b003:	movaps XMMWORD PTR [rdx+0x50],xmm0
    b007:	movdqu xmm0,XMMWORD PTR [r14-0x20]
    b00d:	movaps XMMWORD PTR [rdx+0x60],xmm0
    b011:	movdqu xmm0,XMMWORD PTR [r14-0x10]
    b017:	movaps XMMWORD PTR [rdx+0x70],xmm0
    b01b:	call   1d80 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    b020:	mov    rdi,rbx
    b023:	mov    esi,0x10
    b028:	lea    rbx,[r13+0x15f0]
    b02f:	call   a3c0 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    b034:	mov    rcx,rbx
    b037:	xor    r10d,r10d
    b03a:	movsd  QWORD PTR [rsp+0x30],xmm0
    b040:	mov    QWORD PTR [r13+0x15f0],0x0
    b04b:	mov    rax,QWORD PTR [rsp+0x58]
    b050:	mov    QWORD PTR [rax-0x10],0x0
    b058:	mov    rdi,QWORD PTR [rsp+0x60]
    b05d:	xor    eax,eax
    b05f:	and    rdi,0xfffffffffffffff8
    b063:	sub    rcx,rdi
    b066:	add    ecx,0x88
    b06c:	shr    ecx,0x3
    b06f:	rep stos QWORD PTR [rdi],rax
    b072:	mov    r9,QWORD PTR [rsp+0x38]
    b077:	mov    r11,QWORD PTR [rsp+0x48]
    b07c:	mov    rdx,QWORD PTR [rsp+0x28]
    b081:	mov    rdi,QWORD PTR [rsp+0x20]
    b086:	mov    rbp,rbx
    b089:	mov    rsi,rbx
    b08c:	lea    rcx,[r13+0xa30]
    b093:	xchg   ax,ax
    b095:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    b0a0:	mov    r8,QWORD PTR [rsp+0x8]
    b0a5:	pxor   xmm2,xmm2
    b0a9:	mov    rax,rdi
    b0ac:	movapd xmm1,xmm2
    b0b0:	mov    QWORD PTR [rsp+0x28],r8
    b0b5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    b0c0:	movupd xmm0,XMMWORD PTR [rax]
    b0c4:	add    rax,0x10
    b0c8:	add    r8,0x10
    b0cc:	movapd xmm4,xmm0
    b0d0:	mulpd  xmm4,xmm0
    b0d4:	mulpd  xmm0,XMMWORD PTR [r8-0x10]
    b0da:	movapd xmm3,xmm4
    b0de:	unpckhpd xmm4,xmm4
    b0e2:	addsd  xmm3,xmm2
    b0e6:	addsd  xmm1,xmm0
    b0ea:	unpckhpd xmm0,xmm0
    b0ee:	addsd  xmm4,xmm3
    b0f2:	addsd  xmm1,xmm0
    b0f6:	movapd xmm2,xmm4
    b0fa:	cmp    rax,rcx
    b0fd:	jne    b0c0 <_ZN9GmresCaseIdLm16EE5applyEPv+0x7c0>
    b0ff:	divsd  xmm1,xmm4
    b103:	movsd  xmm0,QWORD PTR [rsi]
    b107:	mov    r8,QWORD PTR [rsp+0x8]
    b10c:	mov    rax,rdi
    b10f:	addsd  xmm0,xmm1
    b113:	unpcklpd xmm1,xmm1
    b117:	movsd  QWORD PTR [rsi],xmm0
    b11b:	nop    DWORD PTR [rax+0x0]
    b11f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    b12a:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    b135:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    b140:	movupd xmm2,XMMWORD PTR [rax]
    b144:	movapd xmm0,XMMWORD PTR [r8]
    b149:	add    rax,0x10
    b14d:	add    r8,0x10
    b151:	mulpd  xmm2,xmm1
    b155:	subpd  xmm0,xmm2
    b159:	movaps XMMWORD PTR [r8-0x10],xmm0
    b15e:	cmp    rax,rcx
    b161:	jne    b140 <_ZN9GmresCaseIdLm16EE5applyEPv+0x840>
    b163:	add    rsi,0x8
    b167:	sub    rdi,0xffffffffffffff80
    b16b:	lea    rcx,[rax+0x80]
    b172:	cmp    rsi,r12
    b175:	jne    b0a0 <_ZN9GmresCaseIdLm16EE5applyEPv+0x7a0>
    b17b:	add    r10,0x1
    b17f:	cmp    r10,r11
    b182:	jne    b081 <_ZN9GmresCaseIdLm16EE5applyEPv+0x781>
    b188:	mov    r8,QWORD PTR [rsp+0x28]
    b18d:	mov    rdi,QWORD PTR [rsp+0x8]
    b192:	mov    esi,0x10
    b197:	mov    QWORD PTR [rsp+0x38],r9
    b19c:	mov    QWORD PTR [rsp+0x28],rdx
    b1a1:	mov    QWORD PTR [rsp+0x40],r8
    b1a6:	call   a3c0 <_ZN4Base6Matrix9Numerical4normIdLm16EEENS1_6ScalarIT_E4RealERKNS0_6VectorIS4_XT0_EEEm>
    b1ab:	mov    rdx,QWORD PTR [rsp+0x28]
    b1b0:	mov    r9,QWORD PTR [rsp+0x38]
    b1b5:	movsd  xmm10,QWORD PTR [rip+0x1152]        # c310 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xd0>
    b1be:	mulsd  xmm10,QWORD PTR [rsp+0x30]
    b1c5:	movapd xmm8,xmm0
    b1ca:	movsd  QWORD PTR [r12],xmm0
    b1d0:	comisd xmm10,xmm0
    b1d5:	jae    b1e7 <_ZN9GmresCaseIdLm16EE5applyEPv+0x8e7>
    b1d7:	cmp    r9,QWORD PTR [rsp+0x18]
    b1dc:	mov    r8,QWORD PTR [rsp+0x40]
    b1e1:	jb     b522 <_ZN9GmresCaseIdLm16EE5applyEPv+0xc22>
    b1e7:	movq   xmm9,QWORD PTR [rip+0x1070]        # c260 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    b1f0:	test   r15,r15
    b1f3:	je     b25b <_ZN9GmresCaseIdLm16EE5applyEPv+0x95b>
    b1f5:	movsd  xmm0,QWORD PTR [r13+0x15f0]
    b1fe:	lea    rax,[r12-0x8]
    b203:	movq   xmm9,QWORD PTR [rip+0x1054]        # c260 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    b20c:	nop    DWORD PTR [rax+0x0]
    b210:	movsd  xmm1,QWORD PTR [rbx+0x8]
    b215:	movsd  xmm4,QWORD PTR [rbx+0x190]
    b21d:	add    rbx,0x8
    b221:	movsd  xmm2,QWORD PTR [rbx+0x108]
    b229:	movapd xmm5,xmm1
    b22d:	movapd xmm3,xmm4
    b231:	mulsd  xmm1,xmm4
    b235:	mulsd  xmm3,xmm0
    b239:	mulsd  xmm5,xmm2
    b23d:	mulsd  xmm2,xmm0
    b241:	movapd xmm0,xmm1
    b245:	addsd  xmm3,xmm5
    b249:	subsd  xmm0,xmm2
    b24d:	movsd  QWORD PTR [rbx-0x8],xmm3
    b252:	movsd  QWORD PTR [rbx],xmm0
    b256:	cmp    rbx,rax
    b259:	jne    b210 <_ZN9GmresCaseIdLm16EE5applyEPv+0x910>
    b25b:	movsd  xmm11,QWORD PTR [r12-0x8]
    b262:	movsd  xmm1,QWORD PTR [r12]
    b268:	lea    rsi,[r12+0x108]
    b270:	lea    rdi,[r12+0x188]
    b278:	mov    QWORD PTR [rsp+0x30],rdx
    b27d:	movapd xmm0,xmm11
    b282:	mov    QWORD PTR [rsp+0x28],r9
    b287:	call   1930 <_ZN4Base6Matrix9Numerical6givensIdEEvRKT_S5_RNS1_6ScalarIS3_E4RealERS3_.isra.0>
    b28c:	mov    r9,QWORD PTR [rsp+0x28]
    b291:	mov    rdx,QWORD PTR [rsp+0x30]
    b296:	mulsd  xmm11,QWORD PTR [r12+0x188]
    b2a0:	movsd  xmm0,QWORD PTR [r12+0x108]
    b2aa:	mulsd  xmm0,QWORD PTR [r12]
    b2b0:	mov    rax,r9
    b2b3:	imul   rax,r15
    b2b7:	shr    rax,1
    b2ba:	addsd  xmm0,xmm11
    b2bf:	movsd  QWORD PTR [r12-0x8],xmm0
    b2c6:	nop    DWORD PTR [rax+0x0]
    b2ca:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    b2d5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    b2e0:	movsd  xmm0,QWORD PTR [rbp+0x0]
    b2e5:	movsd  QWORD PTR [rbp+rax*8-0x440],xmm0
    b2ee:	add    rbp,0x8
    b2f2:	cmp    rbp,r12
    b2f5:	jne    b2e0 <_ZN9GmresCaseIdLm16EE5applyEPv+0x9e0>
    b2f7:	movsd  xmm0,QWORD PTR [r12+0x108]
    b301:	movsd  xmm2,QWORD PTR [r12+0x80]
    b30b:	xorpd  xmm0,xmm9
    b310:	mulsd  xmm0,xmm2
    b314:	mulsd  xmm2,QWORD PTR [r12+0x188]
    b31e:	movsd  QWORD PTR [r12+0x88],xmm0
    b328:	movsd  QWORD PTR [r12+0x80],xmm2
    b332:	add    QWORD PTR [r13+0x1888],0x1
    b33a:	comisd xmm10,xmm8
    b33f:	jb     b4b0 <_ZN9GmresCaseIdLm16EE5applyEPv+0xbb0>
    b345:	cmp    r9,0x10
    b349:	je     b6cb <_ZN9GmresCaseIdLm16EE5applyEPv+0xdcb>
    b34f:	cmp    r15,0xe
    b353:	je     b6a9 <_ZN9GmresCaseIdLm16EE5applyEPv+0xda9>
    b359:	add    r12,0x10
    b35d:	add    r14,0x100
    b364:	xor    eax,eax
    b366:	mov    rcx,r9
    b369:	add    r15,0x3
    b36d:	movzx  ebx,BYTE PTR [rsp+0x10]
    b372:	add    r12,0x8
    b376:	sub    r14,0xffffffffffffff80
    b37a:	cmp    r15,0x10
    b37e:	jne    af95 <_ZN9GmresCaseIdLm16EE5applyEPv+0x695>
    b384:	movsd  xmm1,QWORD PTR [rsp+0x78]
    b38a:	mov    r11,QWORD PTR [rsp+0x20]
    b38f:	mov    rbp,QWORD PTR [rsp+0x80]
    b397:	mov    esi,0xf
    b39c:	mov    eax,0x1
    b3a1:	pxor   xmm3,xmm3
    b3a5:	movsd  xmm5,QWORD PTR [rsp+0x88]
    b3ae:	jmp    b3ca <_ZN9GmresCaseIdLm16EE5applyEPv+0xaca>
    b3b0:	nop    DWORD PTR [rax+rax*1+0x0]
    b3b5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    b3c0:	sub    rsi,0x1
    b3c4:	jb     b494 <_ZN9GmresCaseIdLm16EE5applyEPv+0xb94>
    b3ca:	mov    QWORD PTR [r13+rsi*8+0x1800],0x0
    b3d6:	cmp    rsi,rcx
    b3d9:	setb   dil
    b3dd:	and    dil,al
    b3e0:	je     b3c0 <_ZN9GmresCaseIdLm16EE5applyEPv+0xac0>
    b3e2:	movsd  xmm2,QWORD PTR [r13+rsi*8+0x1678]
    b3ec:	lea    r8,[rsi+0x1]
    b3f0:	cmp    rsi,0xf
    b3f4:	je     b434 <_ZN9GmresCaseIdLm16EE5applyEPv+0xb34>
    b3f6:	mov    rdx,r8
    b3f9:	nop    DWORD PTR [rax+0x0]
    b400:	mov    rax,rdx
    b403:	add    rdx,0x1
    b407:	cmp    rax,rcx
    b40a:	jae    b42e <_ZN9GmresCaseIdLm16EE5applyEPv+0xb2e>
    b40c:	imul   rax,rdx
    b410:	shr    rax,1
    b413:	add    rax,rsi
    b416:	movsd  xmm0,QWORD PTR [r13+rax*8+0x11b0]
    b420:	mulsd  xmm0,QWORD PTR [r13+rdx*8+0x17f8]
    b42a:	subsd  xmm2,xmm0
    b42e:	cmp    rdx,0x10
    b432:	jne    b400 <_ZN9GmresCaseIdLm16EE5applyEPv+0xb00>
    b434:	imul   r8,rsi
    b438:	shr    r8,1
    b43b:	lea    rax,[r8+rsi*1]
    b43f:	movsd  xmm0,QWORD PTR [r13+rax*8+0x11b0]
    b449:	comisd xmm3,xmm0
    b44d:	ja     b4e6 <_ZN9GmresCaseIdLm16EE5applyEPv+0xbe6>
    b453:	comisd xmm5,xmm0
    b457:	jbe    b50d <_ZN9GmresCaseIdLm16EE5applyEPv+0xc0d>
    b45d:	lea    rax,[rsi-0x1]
    b461:	test   rsi,rsi
    b464:	je     b49c <_ZN9GmresCaseIdLm16EE5applyEPv+0xb9c>
    b466:	mov    QWORD PTR [r13+rax*8+0x1800],0x0
    b472:	sub    rsi,0x2
    b476:	cmp    rsi,0xffffffffffffffff
    b47a:	je     b49c <_ZN9GmresCaseIdLm16EE5applyEPv+0xb9c>
    b47c:	mov    QWORD PTR [r13+rsi*8+0x1800],0x0
    b488:	xor    eax,eax
    b48a:	sub    rsi,0x1
    b48e:	jae    b3ca <_ZN9GmresCaseIdLm16EE5applyEPv+0xaca>
    b494:	test   al,al
    b496:	jne    acb0 <_ZN9GmresCaseIdLm16EE5applyEPv+0x3b0>
    b49c:	mov    eax,0x1
    b4a1:	mov    BYTE PTR [r13+0x1891],al
    b4a8:	jmp    aee8 <_ZN9GmresCaseIdLm16EE5applyEPv+0x5e8>
    b4ad:	nop    DWORD PTR [rax]
    b4b0:	movapd xmm1,xmm0
    b4b4:	xorpd  xmm9,xmm0
    b4b9:	mov    r15,r9
    b4bc:	mov    rcx,r9
    b4bf:	cmpltsd xmm1,QWORD PTR [rip+0xd90]        # c258 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x18>
    b4c8:	andpd  xmm9,xmm1
    b4cd:	andnpd xmm1,xmm0
    b4d1:	orpd   xmm1,xmm9
    b4d6:	comisd xmm1,QWORD PTR [rsp+0x70]
    b4dc:	seta   al
    b4df:	xor    ebx,ebx
    b4e1:	jmp    af83 <_ZN9GmresCaseIdLm16EE5applyEPv+0x683>
    b4e6:	movapd xmm4,xmm0
    b4ea:	xorpd  xmm4,XMMWORD PTR [rip+0xd6e]        # c260 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x20>
    b4f2:	comisd xmm5,xmm4
    b4f6:	ja     b45d <_ZN9GmresCaseIdLm16EE5applyEPv+0xb5d>
    b4fc:	movsd  xmm6,QWORD PTR [rsp+0xa8]
    b505:	minsd  xmm6,xmm0
    b509:	movapd xmm0,xmm6
    b50d:	divsd  xmm2,xmm0
    b511:	mov    eax,edi
    b513:	movsd  QWORD PTR [r13+rsi*8+0x1800],xmm2
    b51d:	jmp    b3c0 <_ZN9GmresCaseIdLm16EE5applyEPv+0xac0>
    b522:	comisd xmm0,QWORD PTR [rip+0xdde]        # c308 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xc8>
    b52a:	pxor   xmm2,xmm2
    b52e:	jb     b53c <_ZN9GmresCaseIdLm16EE5applyEPv+0xc3c>
    b530:	movsd  xmm2,QWORD PTR [rip+0xd58]        # c290 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x50>
    b538:	divsd  xmm2,xmm0
    b53c:	comisd xmm8,QWORD PTR [rip+0xdc3]        # c308 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xc8>
    b545:	mov    rax,QWORD PTR [rsp+0x8]
    b54a:	movsd  xmm0,QWORD PTR [rax]
    b54e:	lea    rcx,[rax+0x80]
    b555:	mov    rax,r14
    b558:	jb     b5c0 <_ZN9GmresCaseIdLm16EE5applyEPv+0xcc0>
    b55a:	mulsd  xmm0,xmm2
    b55e:	add    r8,0x10
    b562:	movsd  QWORD PTR [rax],xmm0
    b566:	movsd  xmm0,QWORD PTR [r8-0x8]
    b56c:	mulsd  xmm0,xmm2
    b570:	movsd  QWORD PTR [rax+0x8],xmm0
    b575:	cmp    r8,rcx
    b578:	je     b1e7 <_ZN9GmresCaseIdLm16EE5applyEPv+0x8e7>
    b57e:	movsd  xmm0,QWORD PTR [r8]
    b583:	add    rax,0x10
    b587:	jmp    b55a <_ZN9GmresCaseIdLm16EE5applyEPv+0xc5a>
    b589:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    b594:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    b59f:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    b5aa:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    b5b5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    b5c0:	divsd  xmm0,xmm8
    b5c5:	add    r8,0x10
    b5c9:	movsd  QWORD PTR [rax],xmm0
    b5cd:	movsd  xmm0,QWORD PTR [r8-0x8]
    b5d3:	divsd  xmm0,xmm8
    b5d8:	movsd  QWORD PTR [rax+0x8],xmm0
    b5dd:	cmp    r8,rcx
    b5e0:	je     b1e7 <_ZN9GmresCaseIdLm16EE5applyEPv+0x8e7>
    b5e6:	movsd  xmm0,QWORD PTR [r8]
    b5eb:	add    rax,0x10
    b5ef:	jmp    b5c0 <_ZN9GmresCaseIdLm16EE5applyEPv+0xcc0>
    b5f1:	nop    DWORD PTR [rax+0x0]
    b5f8:	movdqa xmm0,XMMWORD PTR [rsp+0xb0]
    b601:	movups XMMWORD PTR [r13+0x900],xmm0
    b609:	movdqa xmm0,XMMWORD PTR [rsp+0xc0]
    b612:	movups XMMWORD PTR [r13+0x910],xmm0
    b61a:	movdqa xmm0,XMMWORD PTR [rsp+0xd0]
    b623:	movups XMMWORD PTR [r13+0x920],xmm0
    b62b:	movdqa xmm0,XMMWORD PTR [rsp+0xe0]
    b634:	movups XMMWORD PTR [r13+0x930],xmm0
    b63c:	movdqa xmm0,XMMWORD PTR [rsp+0xf0]
    b645:	movups XMMWORD PTR [r13+0x940],xmm0
    b64d:	movdqa xmm0,XMMWORD PTR [rsp+0x100]
    b656:	movups XMMWORD PTR [r13+0x950],xmm0
    b65e:	movdqa xmm0,XMMWORD PTR [rsp+0x110]
    b667:	movups XMMWORD PTR [r13+0x960],xmm0
    b66f:	movdqa xmm0,XMMWORD PTR [rsp+0x120]
    b678:	movups XMMWORD PTR [r13+0x970],xmm0
    b680:	mov    rax,QWORD PTR [rsp+0x338]
    b688:	sub    rax,QWORD PTR fs:0x28
    b691:	jne    b747 <_ZN9GmresCaseIdLm16EE5applyEPv+0xe47>
    b697:	add    rsp,0x348
    b69e:	pop    rbx
    b69f:	pop    rbp
    b6a0:	pop    r12
    b6a2:	pop    r13
    b6a4:	pop    r14
    b6a6:	pop    r15
    b6a8:	ret
    b6a9:	movsd  xmm1,QWORD PTR [rsp+0x78]
    b6af:	mov    r11,QWORD PTR [rsp+0x20]
    b6b4:	mov    ecx,0xf
    b6b9:	mov    rbp,QWORD PTR [rsp+0x80]
    b6c1:	movzx  ebx,BYTE PTR [rsp+0x10]
    b6c6:	jmp    b397 <_ZN9GmresCaseIdLm16EE5applyEPv+0xa97>
    b6cb:	movsd  xmm1,QWORD PTR [rsp+0x78]
    b6d1:	mov    r11,QWORD PTR [rsp+0x20]
    b6d6:	mov    ecx,0x10
    b6db:	mov    rbp,QWORD PTR [rsp+0x80]
    b6e3:	movzx  ebx,BYTE PTR [rsp+0x10]
    b6e8:	jmp    b397 <_ZN9GmresCaseIdLm16EE5applyEPv+0xa97>
    b6ed:	mov    r14,QWORD PTR [rsp+0x68]
    b6f2:	mov    rdx,QWORD PTR [rsp+0xa0]
    b6fa:	mov    esi,0x10
    b6ff:	mov    rdi,r13
    b702:	mov    r8d,0x10
    b708:	mov    rcx,r14
    b70b:	call   1d80 <_ZN4Base6Matrix14GmresOperation14apply_operatorINS1_14MatrixOperatorINS0_6MatrixIdLm16ELm16EEEdLm16EEEdLm16EEEvRKT_RKNS0_6VectorIT0_XT1_EEERSC_m.isra.0>
    b710:	mov    rdx,QWORD PTR [rsp+0x90]
    b718:	mov    rax,r14
    b71b:	movupd xmm0,XMMWORD PTR [rdx]
    b71f:	subpd  xmm0,XMMWORD PTR [rax]
    b723:	add    rdx,0x10
    b727:	add    rax,0x10
    b72b:	movaps XMMWORD PTR [rax-0x10],xmm0
    b72f:	cmp    rdx,QWORD PTR [rsp+0x98]
    b737:	jne    b71b <_ZN9GmresCaseIdLm16EE5applyEPv+0xe1b>
    b739:	movsd  xmm0,QWORD PTR [r13+0x1880]
    b742:	jmp    aeca <_ZN9GmresCaseIdLm16EE5applyEPv+0x5ca>
    b747:	call   10b0 <__stack_chk_fail@plt>
    b74c:	nop    DWORD PTR [rax+0x0]

000000000000b750 <_Z3runI9GmresCaseIdLm16EEEvv>:
    b750:	endbr64
    b754:	push   r12
    b756:	push   rbp
    b757:	push   rbx
    b758:	sub    rsp,0x1000
    b75f:	or     QWORD PTR [rsp],0x0
    b764:	sub    rsp,0x8b0
    b76b:	mov    ecx,0x100
    b770:	movsd  xmm1,QWORD PTR [rip+0xb68]        # c2e0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xa0>
    b778:	xor    esi,esi
    b77a:	movsd  xmm3,QWORD PTR [rip+0xb6e]        # c2f0 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0xb0>
    b782:	movsd  xmm2,QWORD PTR [rip+0xb4e]        # c2d8 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x98>
    b78a:	movq   xmm0,QWORD PTR [rip+0xade]        # c270 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x30>
    b792:	movabs r8,0xcccccccccccccccd
    b79c:	mov    rax,QWORD PTR fs:0x28
    b7a5:	mov    QWORD PTR [rsp+0x18a8],rax
    b7ad:	xor    eax,eax
    b7af:	lea    rdi,[rsp+0x10]
    b7b4:	lea    rbx,[rsp+0x10]
    b7b9:	mov    QWORD PTR [rsp+0x998],0x10
    b7c5:	mov    QWORD PTR [rsp+0x9b8],0x0
    b7d1:	mov    QWORD PTR [rsp+0x1890],0x0
    b7dd:	mov    QWORD PTR [rsp+0x1898],0x0
    b7e9:	mov    QWORD PTR [rsp+0x18a0],0x0
    b7f5:	movups XMMWORD PTR [rsp+0x9a8],xmm0
    b7fd:	rep stos QWORD PTR [rdi],rax
    b800:	lea    rdi,[rsp+0x810]
    b808:	mov    ecx,0x10
    b80d:	rep stos QWORD PTR [rdi],rax
    b810:	lea    rdi,[rsp+0x890]
    b818:	mov    ecx,0x10
    b81d:	rep stos QWORD PTR [rdi],rax
    b820:	lea    rdi,[rsp+0x910]
    b828:	mov    ecx,0x10
    b82d:	rep stos QWORD PTR [rdi],rax
    b830:	lea    rdi,[rsp+0x9c0]
    b838:	mov    ecx,0x100
    b83d:	rep stos QWORD PTR [rdi],rax
    b840:	lea    rdi,[rsp+0x11c0]
    b848:	mov    ecx,0x88
    b84d:	rep stos QWORD PTR [rdi],rax
    b850:	lea    rdi,[rsp+0x1600]
    b858:	mov    ecx,0x11
    b85d:	rep stos QWORD PTR [rdi],rax
    b860:	lea    rdi,[rsp+0x1688]
    b868:	mov    ecx,0x11
    b86d:	rep stos QWORD PTR [rdi],rax
    b870:	lea    rdi,[rsp+0x1710]
    b878:	mov    ecx,0x10
    b87d:	rep stos QWORD PTR [rdi],rax
    b880:	lea    rdi,[rsp+0x1790]
    b888:	mov    ecx,0x10
    b88d:	rep stos QWORD PTR [rdi],rax
    b890:	lea    rdi,[rsp+0x1810]
    b898:	mov    ecx,0x10
    b89d:	rep stos QWORD PTR [rdi],rax
    b8a0:	mov    rdi,rbx
    b8a3:	xchg   ax,ax
    b8a5:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    b8b0:	movsd  xmm4,QWORD PTR [rdi+0x800]
    b8b8:	mov    rcx,rsi
    b8bb:	mov    r10,rdi
    b8be:	xor    r9d,r9d
    b8c1:	jmp    b90b <_Z3runI9GmresCaseIdLm16EEEvv+0x1bb>
    b8c3:	nop    DWORD PTR [rax+rax*1+0x0]
    b8c8:	mov    rax,rcx
    b8cb:	pxor   xmm0,xmm0
    b8cf:	mul    r8
    b8d2:	mov    rax,rdx
    b8d5:	and    rdx,0xfffffffffffffffc
    b8d9:	shr    rax,0x2
    b8dd:	add    rdx,rax
    b8e0:	mov    rax,rcx
    b8e3:	sub    rax,rdx
    b8e6:	cvtsi2sd xmm0,rax
    b8eb:	mulsd  xmm0,xmm1
    b8ef:	add    r9,0x1
    b8f3:	addsd  xmm4,xmm0
    b8f7:	sub    r10,0xffffffffffffff80
    b8fb:	add    rcx,0x1
    b8ff:	movsd  QWORD PTR [r10-0x80],xmm0
    b905:	cmp    r9,0x10
    b909:	je     b928 <_Z3runI9GmresCaseIdLm16EEEvv+0x1d8>
    b90b:	cmp    rsi,r9
    b90e:	jne    b8c8 <_Z3runI9GmresCaseIdLm16EEEvv+0x178>
    b910:	pxor   xmm0,xmm0
    b914:	cvtsi2sd xmm0,rsi
    b919:	mulsd  xmm0,xmm3
    b91d:	addsd  xmm0,xmm2
    b921:	jmp    b8ef <_Z3runI9GmresCaseIdLm16EEEvv+0x19f>
    b923:	nop    DWORD PTR [rax+rax*1+0x0]
    b928:	add    rsi,0x1
    b92c:	movsd  QWORD PTR [rdi+0x800],xmm4
    b934:	add    rdi,0x8
    b938:	cmp    rsi,0x10
    b93c:	jne    b8b0 <_Z3runI9GmresCaseIdLm16EEEvv+0x160>
    b942:	mov    rdi,rbx
    b945:	movq   xmm0,QWORD PTR [rip+0x933]        # c280 <_ZN4Base4MathL17ONE_AND_SQRT2_VECE+0x40>
    b94d:	mov    QWORD PTR [rsp+0x990],0x1
    b959:	lea    r12,[rbx+0x80]
    b960:	movups XMMWORD PTR [rsp+0x9a8],xmm0
    b968:	mov    QWORD PTR [rsp+0x9a0],0x1
    b974:	mov    QWORD PTR [rsp+0x9b8],0x0
    b980:	call   a900 <_ZN9GmresCaseIdLm16EE5applyEPv>
    b985:	call   28a0 <instruction_count_begin>
    b98a:	mov    rdi,rbx
    b98d:	call   a900 <_ZN9GmresCaseIdLm16EE5applyEPv>
    b992:	call   28b0 <instruction_count_end>
    b997:	lea    rsi,[rip+0x69e]        # c03c <_IO_stdin_used+0x3c>
    b99e:	mov    edi,0x2
    b9a3:	xor    eax,eax
    b9a5:	call   10e0 <__printf_chk@plt>
    b9aa:	mov    rdx,rbx
    b9ad:	nop    DWORD PTR [rax]
    b9b0:	mov    rbp,rdx
    b9b3:	lea    rcx,[rdx+0x800]
    b9ba:	nop    WORD PTR [rax+rax*1+0x0]
    b9c0:	movsd  xmm0,QWORD PTR [rbp+0x0]
    b9c5:	lea    rsi,[rip+0x681]        # c04d <_IO_stdin_used+0x4d>
    b9cc:	mov    edi,0x2
    b9d1:	mov    eax,0x1
    b9d6:	mov    QWORD PTR [rsp+0x8],rcx
    b9db:	sub    rbp,0xffffffffffffff80
    b9df:	mov    QWORD PTR [rsp],rdx
    b9e3:	call   10e0 <__printf_chk@plt>
    b9e8:	mov    rcx,QWORD PTR [rsp+0x8]
    b9ed:	mov    rdx,QWORD PTR [rsp]
    b9f1:	cmp    rbp,rcx
    b9f4:	jne    b9c0 <_Z3runI9GmresCaseIdLm16EEEvv+0x270>
    b9f6:	add    rdx,0x8
    b9fa:	cmp    rdx,r12
    b9fd:	jne    b9b0 <_Z3runI9GmresCaseIdLm16EEEvv+0x260>
    b9ff:	lea    rbp,[rbx+0x800]
    ba06:	lea    r12,[rbx+0x880]
    ba0d:	nop    DWORD PTR [rax]
    ba10:	movsd  xmm0,QWORD PTR [rbp+0x0]
    ba15:	lea    rsi,[rip+0x631]        # c04d <_IO_stdin_used+0x4d>
    ba1c:	mov    edi,0x2
    ba21:	mov    eax,0x1
    ba26:	add    rbp,0x8
    ba2a:	call   10e0 <__printf_chk@plt>
    ba2f:	cmp    rbp,r12
    ba32:	jne    ba10 <_Z3runI9GmresCaseIdLm16EEEvv+0x2c0>
    ba34:	lea    rsi,[rip+0x607]        # c042 <_IO_stdin_used+0x42>
    ba3b:	mov    edi,0x2
    ba40:	xor    eax,eax
    ba42:	call   10e0 <__printf_chk@plt>
    ba47:	lea    rbp,[rbx+0x900]
    ba4e:	add    rbx,0x980
    ba55:	data16 cs nop WORD PTR [rax+rax*1+0x0]
    ba60:	movsd  xmm0,QWORD PTR [rbp+0x0]
    ba65:	lea    rsi,[rip+0x5e1]        # c04d <_IO_stdin_used+0x4d>
    ba6c:	mov    edi,0x2
    ba71:	mov    eax,0x1
    ba76:	add    rbp,0x8
    ba7a:	call   10e0 <__printf_chk@plt>
    ba7f:	cmp    rbp,rbx
    ba82:	jne    ba60 <_Z3runI9GmresCaseIdLm16EEEvv+0x310>
    ba84:	movzx  ecx,BYTE PTR [rsp+0x18a0]
    ba8c:	mov    rdx,QWORD PTR [rsp+0x1898]
    ba94:	lea    rsi,[rip+0x5c2]        # c05d <_IO_stdin_used+0x5d>
    ba9b:	mov    edi,0x2
    baa0:	movsd  xmm0,QWORD PTR [rsp+0x1890]
    baa9:	mov    eax,0x1
    baae:	movzx  r8d,BYTE PTR [rsp+0x18a1]
    bab7:	call   10e0 <__printf_chk@plt>
    babc:	mov    rax,QWORD PTR [rsp+0x18a8]
    bac4:	sub    rax,QWORD PTR fs:0x28
    bacd:	jne    badb <_Z3runI9GmresCaseIdLm16EEEvv+0x38b>
    bacf:	add    rsp,0x18b0
    bad6:	pop    rbx
    bad7:	pop    rbp
    bad8:	pop    r12
    bada:	ret
    badb:	call   10b0 <__stack_chk_fail@plt>

Disassembly of section .fini:

000000000000bae0 <_fini>:
    bae0:	endbr64
    bae4:	sub    rsp,0x8
    bae8:	add    rsp,0x8
    baec:	ret
