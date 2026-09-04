# CNRL gate analysis

Rows loaded: **329** from **5** file(s).

## Structural audit

**PASS:** no invalid rows, affinity failures, Bclone invariant failures, or shared/Bclone checksum divergences.

## T0-R: residencia por profundidad

| D | S | R | filas | kernel | A/B mediana | min(A)>max(B) | B one-pass GB/s | lectura |
|---:|---:|---:|---:|---|---:|---|---:|---|
| 1472 | 1 | 4 | 1472 | avx2-fused | 0.993× | no | 81.07 | NO |
| 1472 | 1 | 8 | 1472 | avx2-fused | 1.498× | no | 49.89 | WEAK |

## T0-M: matrixización por slots

| D | filas | R | variante | S | G(S)=fused/fused(S=1) | F(S)=fused/repeat | lectura |
|---:|---:|---:|---|---:|---:|---:|---|
| 1472 | 1472 | 4 | clone | 1 | 1.000× | 0.908× | NO |
| 1472 | 1472 | 4 | clone | 8 | 1.944× | 1.815× | PASS |
| 1472 | 1472 | 4 | clone | 16 | 1.705× | 1.659× | PASS |
| 1472 | 1472 | 4 | shared | 1 | 1.000× | 0.901× | NO |
| 1472 | 1472 | 4 | shared | 8 | 2.296× | 1.552× | PASS_STRONG |
| 1472 | 1472 | 4 | shared | 16 | 2.202× | 1.538× | PASS_STRONG |
| 1472 | 1472 | 8 | clone | 1 | 1.000× | 0.870× | NO |
| 1472 | 1472 | 8 | clone | 8 | 4.048× | 2.090× | PASS_STRONG |
| 1472 | 1472 | 8 | clone | 16 | 3.755× | 1.469× | PASS_STRONG |
| 1472 | 1472 | 8 | shared | 1 | 1.000× | 0.813× | NO |
| 1472 | 1472 | 8 | shared | 8 | 2.280× | 1.623× | PASS_STRONG |
| 1472 | 1472 | 8 | shared | 16 | 2.471× | 1.610× | PASS_STRONG |

## T0-M: intercambio profundidad/slots a R×S constante

| D | filas | variante | R×S | pares medidos (R,S: mediana ms) | más rápido |
|---:|---:|---|---:|---|---|
| 1472 | 1472 | clone | 64 | 4,16: 2.978; 8,8: 2.933 | R=8, S=8 |
| 1472 | 1472 | shared | 64 | 4,16: 2.785; 8,8: 2.878 | R=4, S=16 |

## T0-RM: recurrencia real

| D | S | R | transición | shift | A/B mediana | clipping mediano | validez numérica | lectura física |
|---:|---:|---:|---|---:|---:|---:|---|---|
| 1472 | 1 | 8 | fixed-point | 12 | 2.098× | 25.433% | NO VÁLIDO NUMÉRICAMENTE | separación fuerte |
| 1472 | 1 | 8 | fixed-point | 13 | 2.329× | 3.252% | ADVERTENCIA | separación fuerte |
| 1472 | 1 | 8 | fixed-point | 14 | 2.260× | 0.000% | OK | separación fuerte |
| 1472 | 1 | 8 | fixed-point | 15 | 2.083× | 0.000% | OK | separación fuerte |
| 1472 | 1 | 8 | global-rms | 12 | 1.996× | 0.000% | OK | separación |
| 1472 | 1 | 8 | group-rms | 12 | 2.236× | 0.000% | OK | separación fuerte |
| 1472 | 16 | 8 | fixed-point | 12 | 1.215× | 25.411% | NO VÁLIDO NUMÉRICAMENTE | separación |
| 1472 | 16 | 8 | fixed-point | 13 | 1.023× | 3.083% | ADVERTENCIA | paridad compatible con alta reutilización por slots |
| 1472 | 16 | 8 | fixed-point | 14 | 1.068× | 0.002% | OK | paridad compatible con alta reutilización por slots |
| 1472 | 16 | 8 | fixed-point | 15 | 0.945× | 0.000% | OK | paridad compatible con alta reutilización por slots |
| 1472 | 16 | 8 | global-rms | 12 | 0.693× | 0.004% | OK | paridad compatible con alta reutilización por slots |
| 1472 | 16 | 8 | group-rms | 12 | 1.130× | 0.004% | OK | paridad compatible con alta reutilización por slots |
| 1472 | 8 | 8 | fixed-point | 12 | 0.913× | 25.053% | NO VÁLIDO NUMÉRICAMENTE | paridad compatible con alta reutilización por slots |
| 1472 | 8 | 8 | fixed-point | 13 | 1.164× | 2.900% | ADVERTENCIA | paridad compatible con alta reutilización por slots |
| 1472 | 8 | 8 | fixed-point | 14 | 1.059× | 0.000% | OK | paridad compatible con alta reutilización por slots |
| 1472 | 8 | 8 | fixed-point | 15 | 1.017× | 0.000% | OK | paridad compatible con alta reutilización por slots |
| 1472 | 8 | 8 | global-rms | 12 | 1.196× | 0.003% | OK | paridad compatible con alta reutilización por slots |
| 1472 | 8 | 8 | group-rms | 12 | 0.951× | 0.003% | OK | paridad compatible con alta reutilización por slots |

El clipping del microbenchmark de transición no sustituye esta columna: la validez del gate recurrente se juzga sobre la trayectoria T0-RM de R rondas.

## T0-RM: throughput retenido frente al puente frozen

| D | S | R | variante | transición | recurrente/estático |
|---:|---:|---:|---|---|---:|
| 1472 | 1 | 8 | clone | fixed-point | 0.981× |
| 1472 | 1 | 8 | clone | global-rms | 1.019× |
| 1472 | 1 | 8 | clone | group-rms | 0.858× |
| 1472 | 1 | 8 | shared | fixed-point | 1.171× |
| 1472 | 1 | 8 | shared | global-rms | 1.127× |
| 1472 | 1 | 8 | shared | group-rms | 1.063× |
| 1472 | 16 | 8 | clone | fixed-point | 1.064× |
| 1472 | 16 | 8 | clone | global-rms | 1.001× |
| 1472 | 16 | 8 | clone | group-rms | 0.969× |
| 1472 | 16 | 8 | shared | fixed-point | 0.958× |
| 1472 | 16 | 8 | shared | global-rms | 0.584× |
| 1472 | 16 | 8 | shared | group-rms | 0.922× |
| 1472 | 8 | 8 | clone | fixed-point | 1.025× |
| 1472 | 8 | 8 | clone | global-rms | 0.898× |
| 1472 | 8 | 8 | clone | group-rms | 0.978× |
| 1472 | 8 | 8 | shared | fixed-point | 1.018× |
| 1472 | 8 | 8 | shared | global-rms | 1.056× |
| 1472 | 8 | 8 | shared | group-rms | 0.915× |

## Variabilidad externa

**ALERTA:** 46 condición(es) superan 10% de CV.
- CV=15.7%: ('t0rm', 1472, 1, 8, 1472, 'avx2-fused', 'fixed-point', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=14.1%: ('t0rm', 1472, 1, 8, 1472, 'avx2-fused', 'fixed-point', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=14.7%: ('t0rm', 1472, 8, 8, 1472, 'avx2-fused', 'fixed-point', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=20.0%: ('t0rm', 1472, 16, 8, 1472, 'avx2-fused', 'fixed-point', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=33.6%: ('t0rm', 1472, 1, 8, 1472, 'avx2-fused', 'fixed-point', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 13, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=10.0%: ('t0rm', 1472, 8, 8, 1472, 'avx2-fused', 'fixed-point', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 13, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=21.6%: ('t0rm', 1472, 16, 8, 1472, 'avx2-fused', 'fixed-point', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 13, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=12.9%: ('t0rm', 1472, 1, 8, 1472, 'avx2-fused', 'fixed-point', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 14, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=10.5%: ('t0rm', 1472, 1, 8, 1472, 'avx2-fused', 'fixed-point', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 14, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=32.3%: ('t0rm', 1472, 8, 8, 1472, 'avx2-fused', 'fixed-point', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 14, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=20.9%: ('t0rm', 1472, 16, 8, 1472, 'avx2-fused', 'fixed-point', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 14, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=13.6%: ('t0rm', 1472, 16, 8, 1472, 'avx2-fused', 'fixed-point', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 14, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=18.2%: ('t0rm', 1472, 1, 8, 1472, 'avx2-fused', 'fixed-point', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 15, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=24.5%: ('t0rm', 1472, 8, 8, 1472, 'avx2-fused', 'fixed-point', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 15, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=19.4%: ('t0rm', 1472, 8, 8, 1472, 'avx2-fused', 'fixed-point', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 15, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=12.5%: ('t0m', 1472, 8, 8, 1472, 'avx2-repeat', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=14.0%: ('t0m', 1472, 8, 8, 1472, 'avx2-repeat', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=11.5%: ('t0m', 1472, 8, 8, 1472, 'avx2-fused', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=18.8%: ('t0m', 1472, 16, 8, 1472, 'avx2-repeat', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=16.0%: ('t0m', 1472, 16, 8, 1472, 'avx2-fused', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=14.2%: ('t0m', 1472, 16, 8, 1472, 'avx2-repeat', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=11.9%: ('t0m', 1472, 16, 8, 1472, 'avx2-fused', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=11.5%: ('t0m', 1472, 1, 8, 1472, 'avx2-repeat', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=53.5%: ('t0m', 1472, 1, 8, 1472, 'avx2-fused', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=23.0%: ('t0m', 1472, 1, 8, 1472, 'avx2-repeat', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=14.8%: ('t0m', 1472, 1, 8, 1472, 'avx2-fused', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=11.9%: ('t0m', 1472, 8, 4, 1472, 'avx2-repeat', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=13.3%: ('t0m', 1472, 8, 4, 1472, 'avx2-repeat', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=15.3%: ('t0m', 1472, 16, 4, 1472, 'avx2-repeat', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=18.1%: ('t0m', 1472, 16, 4, 1472, 'avx2-fused', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=20.8%: ('t0m', 1472, 16, 4, 1472, 'avx2-repeat', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=16.5%: ('t0m', 1472, 16, 4, 1472, 'avx2-fused', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=18.2%: ('t0m', 1472, 1, 4, 1472, 'avx2-repeat', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=13.4%: ('t0m', 1472, 1, 4, 1472, 'avx2-fused', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=11.1%: ('t0m', 1472, 1, 4, 1472, 'avx2-fused', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=19.1%: ('t0r', 1472, 1, 4, 1472, 'avx2-fused', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=57.6%: ('t0r', 1472, 1, 4, 1472, 'avx2-fused', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=42.2%: ('t0r', 1472, 1, 8, 1472, 'avx2-fused', 'frozen', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=17.0%: ('t0rm', 1472, 1, 8, 1472, 'avx2-fused', 'group-rms', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=19.0%: ('t0rm', 1472, 1, 8, 1472, 'avx2-fused', 'group-rms', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=15.5%: ('t0rm', 1472, 1, 8, 1472, 'avx2-fused', 'global-rms', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=12.7%: ('t0rm', 1472, 1, 8, 1472, 'avx2-fused', 'global-rms', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=12.1%: ('t0rm', 1472, 8, 8, 1472, 'avx2-fused', 'group-rms', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')
- CV=19.3%: ('t0rm', 1472, 8, 8, 1472, 'avx2-fused', 'group-rms', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=19.0%: ('t0rm', 1472, 8, 8, 1472, 'avx2-fused', 'global-rms', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'clone')
- CV=19.8%: ('t0rm', 1472, 16, 8, 1472, 'avx2-fused', 'global-rms', 'full-repetition', 4, '449;318;353;352', '0;2;4;6', '0;1;2;3', 3221342974, 'false', 'true', 12, 1, 1, 0, 32.0, 1e-06, 2, 4, 1, 'false', '0.4.1', 'shared')

## Metric warning

`mac_per_second / S` equals the one-pass int8 weight-stream rate only for a square/rectangular dot-product that loads each weight once per slot group. The authoritative CSV field is `one_pass_weight_gb_per_second`; do not equate raw GMAC/s with DRAM GB/s when S>1.
