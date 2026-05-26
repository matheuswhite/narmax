# NARMAX-FROLS em Rust

Implementação em Rust de identificação de modelos NARMAX (*Nonlinear AutoRegressive Moving Average with eXogenous inputs*) via FROLS (*Forward Regression Orthogonal Least Squares*), aplicada a benchmarks clássicos da literatura de identificação não-linear.

Trabalho da disciplina de Identificação de Sistemas (Mestrado).

## Estrutura do projeto

```
src/
├── symbol.rs       Symbol { name, index, power } — átomo de um termo
├── regressor.rs    Regressor (produto de Symbols), Mul, eval, eval_at
└── main.rs         build_regressors + FROLS + carregamento dos datasets

res/
├── ballbeam.csv         1000  amostras  (idx, u, y)
├── wienerhammer.csv   188000  amostras
├── snls80.csv         131072  amostras  (Silverbox SNLS)
└── schroeder80.csv    131072  amostras  (Silverbox Schroeder)
```

Os CSVs em `res/` foram derivados dos arquivos originais (`.dat`/`.csv` brutos das pastas `SilverboxFiles/` e `WienerHammerstein2009Files/`) em formato uniforme `idx,u,y`.

## Algoritmo FROLS

Dado um conjunto de candidatos a regressores e dados `(u, y)`, o FROLS escolhe iterativamente o subconjunto que melhor explica `y`:

1. **Φ inicial** de tamanho `n × m`, onde `n = N − k_min` (descartando as primeiras `k_min` amostras que não têm lags suficientes) e `m` é o número de candidatos.
2. A cada iteração `l`:
   - Para cada candidato `j` restante, calcula `w_j = φ_j − Σᵢ αᵢ wᵢ` (Gram-Schmidt contra as colunas já escolhidas).
   - Computa `g_j = wᵀⱼ y / ‖wⱼ‖²` e `ERRⱼ = g²ⱼ ‖wⱼ‖² / σ²`.
   - Escolhe o `j*` com maior ERR.
3. **Critério de parada**: `ESR = 1 − Σ ERRᵢ < ρ`, ou `l = l_max`, ou todos os candidatos são linearmente dependentes.
4. Recupera `θ` no espaço original resolvendo `A θ = g` (triangular superior).

Parâmetros usados nos experimentos: `ρ = 0.001`, `l_max = 10`.

## Conjunto de candidatos

`build_regressors(ny, nu, d)` gera **todos** os monômios `y(k-i)·u(k-j)·...` com:
- lags `i ∈ {1..ny}` e `j ∈ {1..nu}`,
- grau total ≤ `d`,
- deduplicação por igualdade de conjunto (`y(k-1)·y(k-2) ≡ y(k-2)·y(k-1)`).

Nos experimentos abaixo, `ny = nu = 2`. Variamos `d` (chamado `non_lin_len` no código) entre 2 e 3.

| `d` | nº de candidatos |
|---:|---:|
| 2  | 14  (4 lineares + 10 quadráticos) |
| 3  | 34  (14 do `d=2` + 20 cúbicos) |

## Datasets

| Arquivo | Sistema | Amostras | Característica |
|---|---|---:|---|
| `ballbeam.csv` | Ball-and-Beam | 1000 | quasi-linear |
| `wienerhammer.csv` | Wiener-Hammerstein (IFAC SYSID 2009) | 188 000 | linear → NL estática → linear |
| `snls80.csv` | Silverbox (multissinal NLS, 80mV) | 131 072 | oscilador de Duffing (mola cúbica) |
| `schroeder80.csv` | Silverbox (Schroeder, 80mV) | 131 072 | mesma planta, excitação diferente |

## Resultados

### Experimento 1 — `non_lin_len = 2`

14 candidatos (lineares + quadráticos).

**Ballbeam**
```
Selecionados (4 regressores):
  θ[0] =    +1.715737  ERR = 0.996320   y(k-1)
  θ[1] =    -0.716180  ERR = 0.002657   y(k-2)
  θ[2] =    +0.668257  ERR = 0.000016   u(k-2)
  θ[3] =    -0.640466  ERR = 0.000075   u(k-1)
  ESR ≈ 9.3e-4 → parou por ρ
```

**Wiener-Hammerstein**
```
Selecionados (2 regressores):
  θ[0] =    +1.967768  ERR = 0.982353   y(k-1)
  θ[1] =    -0.985342  ERR = 0.017120   y(k-2)
  ESR ≈ 5.3e-4 → parou por ρ
```

**Silverbox SNLS**
```
Selecionados (10 regressores):  — esgotou l_max
  θ[0] =    +1.462073  ERR = 0.581526   y(k-1)
  θ[1] =    -0.936429  ERR = 0.389156   y(k-2)
  θ[2] =    +0.403328  ERR = 0.027283   u(k-1)
  θ[3] =    -0.246173  ERR = 0.000382   y(k-2)^2
  θ[4] =    -0.774258  ERR = 0.000156   u(k-1)^2
  θ[5] =    -0.679316  ERR = 0.000050   u(k-2)^2
  θ[6] =    +0.855530  ERR = 0.000056   u(k-1)*u(k-2)
  θ[7] =    -0.224091  ERR = 0.000018   y(k-1)^2
  θ[8] =    +0.351410  ERR = 0.000056   y(k-1)*y(k-2)
  θ[9] =    +0.011008  ERR = 0.000012   u(k-2)
  ESR ≈ 1.3e-3
```

**Silverbox Schroeder**
```
Selecionados (10 regressores):  — esgotou l_max
  θ[0] =    +1.453145  ERR = 0.577352   y(k-1)
  θ[1] =    -0.934719  ERR = 0.391264   y(k-2)
  θ[2] =    +0.405595  ERR = 0.029464   u(k-1)
  θ[3] =    -0.879436  ERR = 0.000365   u(k-1)^2
  θ[4] =    -0.151025  ERR = 0.000135   y(k-2)^2
  θ[5] =    -0.833173  ERR = 0.000059   u(k-2)^2
  θ[6] =    +0.941378  ERR = 0.000070   u(k-1)*u(k-2)
  θ[7] =    +0.019478  ERR = 0.000043   u(k-2)
  θ[8] =    -0.132048  ERR = 0.000008   y(k-1)^2
  θ[9] =    +0.210248  ERR = 0.000029   y(k-1)*y(k-2)
  ESR ≈ 1.3e-3
```

### Experimento 2 — `non_lin_len = 3`

34 candidatos (lineares + quadráticos + cúbicos).

**Ballbeam** — *idêntico ao experimento 1*. Os termos cúbicos não foram suficientemente correlacionados com o resíduo para deslocar nenhum dos lineares.

**Wiener-Hammerstein** — *idêntico ao experimento 1*. Mesmo com termos de grau 3 disponíveis, o FROLS continua escolhendo apenas `y(k-1)` e `y(k-2)`.

**Silverbox SNLS**
```
Selecionados (10 regressores):  — esgotou l_max
  θ[0] =    +1.476664  ERR = 0.581526   y(k-1)
  θ[1] =    -0.933072  ERR = 0.389156   y(k-2)
  θ[2] =    +0.399352  ERR = 0.027283   u(k-1)
  θ[3] =    -0.117397  ERR = 0.000382   y(k-2)^2
  θ[4] =    -1.026434  ERR = 0.000198   y(k-1)^3
  θ[5] =    -0.887063  ERR = 0.000157   u(k-1)^2
  θ[6] =    -0.897715  ERR = 0.000049   u(k-2)^2
  θ[7] =   +11.096343  ERR = 0.000065   u(k-1)*u(k-2)^2
  θ[8] =    +0.847368  ERR = 0.000035   u(k-1)*u(k-2)
  θ[9] =    -0.790900  ERR = 0.000020   y(k-1)*y(k-2)^2
```

**Silverbox Schroeder**
```
Selecionados (6 regressores):  — parou por ρ
  θ[0] =    +1.469665  ERR = 0.577352   y(k-1)
  θ[1] =    -0.927826  ERR = 0.391264   y(k-2)
  θ[2] =    +0.416098  ERR = 0.029464   u(k-1)
  θ[3] =    -1.828523  ERR = 0.000427   y(k-1)^2*y(k-2)
  θ[4] =    -0.851677  ERR = 0.000369   u(k-1)^2
  θ[5] =    -0.098111  ERR = 0.000126   y(k-2)^2
```

### Experimento 3 — `l_max = 15`, `non_lin_len = 3`

Mesma configuração do experimento 2, mas com mais "espaço" para o FROLS preencher (15 regressores em vez de 10). Objetivo: investigar se o SNLS converge por ρ se for dado mais teto.

- **Ballbeam, Wiener-Hammerstein, Schroeder**: resultados *idênticos* ao experimento 2 (todos já paravam por ρ antes de atingir o limite anterior).

- **Silverbox SNLS**: agora preenche os 15 slots — ainda esgota `l_max`.

```
Selecionados (15 regressores):
  θ[0]  =    +1.476083  ERR = 0.581526   y(k-1)
  θ[1]  =    -0.932467  ERR = 0.389156   y(k-2)
  θ[2]  =    +0.393531  ERR = 0.027283   u(k-1)
  θ[3]  =    -0.236096  ERR = 0.000382   y(k-2)^2
  θ[4]  =    -1.090901  ERR = 0.000198   y(k-1)^3
  θ[5]  =    -0.816127  ERR = 0.000157   u(k-1)^2
  θ[6]  =    -0.753658  ERR = 0.000049   u(k-2)^2
  θ[7]  =    +5.480222  ERR = 0.000065   u(k-1)*u(k-2)^2
  θ[8]  =    +0.753219  ERR = 0.000035   u(k-1)*u(k-2)
  θ[9]  =    -0.899995  ERR = 0.000020   y(k-1)*y(k-2)^2
  θ[10] =    +1.913809  ERR = 0.000022   y(k-2)^2*u(k-2)
  θ[11] =    -0.230827  ERR = 0.000013   y(k-1)^2
  θ[12] =    +0.339165  ERR = 0.000051   y(k-1)*y(k-2)
  θ[13] =    +4.462465  ERR = 0.000011   u(k-1)^3
  θ[14] =    +0.137055  ERR = 0.000009   y(k-1)*u(k-2)
  ESR ≈ 1.0e-3 — ainda acima do ρ
```

**Observação importante — quase-colinearidade evidenciada:**

O coeficiente do termo `u(k-1)·u(k-2)²` mudou de **+11.10** (com `l_max = 10`) para **+5.48** (com `l_max = 15`). Como `θ` é recuperado por substituição reversa em `A θ = g`, os coeficientes dos termos iniciais dependem dos termos posteriores. Ao adicionar 5 regressores extras, parte do "trabalho de correção" que esse termo fazia foi redistribuída, reduzindo seu coeficiente. **Isso é diagnóstico de quase-colinearidade no espaço de candidatos** — esses termos com `|θ|` muito acima da média não representam estrutura física dominante, mas refinamentos numéricos.

Coeficientes que permanecem "grandes" (`|θ|` > 1.5):
- θ[7] = +5.48 em `u(k-1)·u(k-2)²`
- θ[10] = +1.91 em `y(k-2)²·u(k-2)`
- θ[13] = +4.46 em `u(k-1)³`

São candidatos naturais a poda em uma fase de validação out-of-sample (não realizada aqui).

**Por que SNLS não converge por ρ mas Schroeder sim?**

Ambos os datasets usam a mesma planta física (Silverbox / Duffing). A diferença está na **excitação**:

- A **Schroeder** é uma multissenoide projetada para excitar bandas de frequência específicas (apenas harmônicos ímpares) com fase otimizada. Concentra energia espectral de forma "limpa", permitindo ao modelo ajustar-se com poucos termos (6).
- O **SNLS** combina seções de ramp-down, multissinais e ruído — força o modelo a explicar uma gama muito mais ampla de comportamento, esgotando `l_max` em todas as configurações testadas (10 e 15).

Para o SNLS atingir `ESR < ρ = 0.001`, provavelmente seria necessário relaxar `ρ`, ou aumentar lags (a memória do modelo de Duffing pode beneficiar de `ny, nu ≥ 3`).

### Experimento 4 — lags `ny = nu = 3`, `non_lin_len = 3`, `l_max = 15`

Motivação: o SNLS continua esgotando `l_max` no experimento 3. A hipótese é que o modelo de Duffing precisa de **memória maior** do que 2 amostras para fechar o resíduo. Aumentamos os lags de 2 para 3 em `y` e `u`.

O espaço de candidatos cresce para **83 termos** (6 lineares + 21 quadráticos + 56 cúbicos), aproximadamente 2.5× o do experimento anterior.

**Ballbeam — encolheu para 2 regressores**
```
Selecionados (2 regressores):
  θ[0] =    +1.462375  ERR = 0.996320   y(k-1)
  θ[1] =    -0.467408  ERR = 0.002955   y(k-3)
  ESR ≈ 7.3e-4 → parou por ρ
```
O FROLS descobriu que **`y(k-3)` sozinho** substitui o trio `y(k-2) + u(k-1) + u(k-2)` da configuração anterior. Modelo mais parcimonioso, mas com lags mais espaçados — provavelmente capturando a dinâmica de 2ª ordem por um par de pontos mais separados.

**Wiener-Hammerstein — inalterado**
```
Selecionados (2 regressores):
  θ[0] =    +1.967762  ERR = 0.982353   y(k-1)
  θ[1] =    -0.985336  ERR = 0.017120   y(k-2)
```
Mesmos termos, mesmos coeficientes. Confirma que a estrutura W-H **não é capturável** por polinomial NARMAX simples mesmo com mais lags — requer modelagem específica.

**Silverbox SNLS — a grande mudança: 15 → 8 regressores, agora parou por ρ**
```
Selecionados (8 regressores):
  θ[0] =    +2.295940  ERR = 0.581526   y(k-1)
  θ[1] =    -2.147412  ERR = 0.389156   y(k-2)
  θ[2] =    +0.382566  ERR = 0.027283   u(k-1)
  θ[3] =    -0.029412  ERR = 0.000395   y(k-3)^2
  θ[4] =    -0.818677  ERR = 0.000203   y(k-1)^3
  θ[5] =    -0.018394  ERR = 0.000175   y(k-1)^2
  θ[6] =    +0.777085  ERR = 0.000113   y(k-3)
  θ[7] =    -0.315628  ERR = 0.000856   u(k-2)
  ESR ≈ 2.9e-4 → parou por ρ ✓
```

Resultados notáveis:

- **Primeira vez que o SNLS atinge o critério de parada por `ρ`** — não esgota `l_max`.
- **Os coeficientes problemáticos do experimento 3 desapareceram**: os θ's "estourados" `+11.10`, `+5.48` e `+4.46` foram completamente eliminados; o maior coeficiente em magnitude agora é `|θ[1]| = 2.15`, da mesma ordem dos lineares.
- O modelo identificado é coerente com a física do Duffing: dinâmica linear de 3ª ordem (`y(k-1), y(k-2), y(k-3)`) + cubicidade da mola (`y(k-1)³, y(k-3)²`) + termos de entrada.

**Curiosidade — ERR não-monotônico no SNLS:** repare que o ERR do θ[7] (= 8.6e-4) é **maior** que os de θ[3]..θ[6] (1e-4 a 4e-4). Em FROLS isso é possível: o ERR de cada candidato muda a cada iteração porque depende do resíduo já ortogonalizado. Adicionar `y(k-3)` no passo 6 pode ter "limpado" o resíduo numa direção que tornou `u(k-2)` subitamente mais correlacionado. Não é bug — é o comportamento esperado do FROLS clássico.

**Silverbox Schroeder — quase inalterado**
```
Selecionados (6 regressores):
  θ[0] =    +1.469577  ERR = 0.577352   y(k-1)
  θ[1] =    -0.927626  ERR = 0.391265   y(k-2)
  θ[2] =    +0.416182  ERR = 0.029464   u(k-1)
  θ[3] =    -1.844366  ERR = 0.000427   y(k-1)^2*y(k-2)
  θ[4] =    -0.843662  ERR = 0.000369   u(k-1)^2
  θ[5] =    -0.103720  ERR = 0.000141   y(k-3)^2
```
Mesmos 6 regressores, mesma estrutura — só o último termo trocou `y(k-2)²` por **`y(k-3)²`** (aproveitando o lag adicional). Os ERRs e coeficientes são quase idênticos. Confirma que para essa excitação Schroeder a estrutura encontrada já é estável.

### Resumo comparativo — todos os experimentos

| Dataset | exp 1 (2,2,2) `l_max=10` | exp 2 (2,2,3) `l_max=10` | exp 3 (2,2,3) `l_max=15` | exp 4 (3,3,3) `l_max=15` |
|---|---|---|---|---|
| ballbeam | 4 reg, ρ-stop | 4 reg, ρ-stop | 4 reg, ρ-stop | **2 reg, ρ-stop** |
| W-H | 2 reg, ρ-stop | 2 reg, ρ-stop | 2 reg, ρ-stop | 2 reg, ρ-stop |
| **SNLS** | 10 reg, esgotou l_max | 10 reg, esgotou l_max | **15 reg, esgotou l_max** | **8 reg, ρ-stop** ✓ |
| Schroeder | 10 reg, esgotou l_max | 6 reg, ρ-stop | 6 reg, ρ-stop | 6 reg, ρ-stop |

A mudança decisiva para o SNLS foi **aumentar os lags**, não o grau ou o `l_max`. O dataset precisa de memória mais profunda — a dinâmica do oscilador de Duffing nesse nível de excitação está distribuída em pelo menos 3 amostras passadas. Esse experimento valida que o pipeline FROLS é capaz de identificar modelos parcimoniosos *quando o espaço de candidatos contém a estrutura correta*.

### Experimento 5 — Validação out-of-sample (OSA vs free-run)

Até aqui, todas as métricas (ERR, ESR) foram medidas no mesmo dado de identificação. Esse experimento adiciona **validação out-of-sample**: split sequencial 70%/30% (treino/teste), identificação no treino e avaliação no teste com duas métricas:

- **OSA** (*One-Step-Ahead*): predição do próximo `y(k)` usando o histórico **real** nos lags. Mede qualidade de predição de 1 passo.
- **MPO** (*Model Predicted Output*, free-run): simulação de `y(k)` propagando-se com as **próprias predições** nos lags. Mede estabilidade e fidelidade do modelo como simulador.

A ratio `MPO/OSA` indica a robustez: próximo de 1 significa que o modelo simula tão bem quanto prevê; valores altos indicam que erros se acumulam em free-run (e/ou estrutura incorreta).

**Configuração comparada:** `(2, 2, 2)` vs `(3, 3, 3)` com `ρ = 0.001`, `l_max = 15`.

| Dataset | Config | OSA RMSE | MPO RMSE | MPO/OSA |
|---|---|---:|---:|---:|
| ballbeam     | `(2, 2, 2)` | 0.001999 | 0.117202 | 58.6× |
| ballbeam     | `(3, 3, 3)` | 0.001724 | **0.057648** | **33.4×** |
| wienerhammer | `(2, 2, 2)` | 0.005468 | 0.238815 | 43.7× |
| wienerhammer | `(3, 3, 3)` | 0.005468 | 0.238815 | 43.7× |
| snls80       | `(2, 2, 2)` | 0.001780 | **0.008365** | **4.7×** |
| snls80       | `(3, 3, 3)` | 0.000888 | 0.009541 | 10.7× |
| schroeder80  | `(2, 2, 2)` | 0.002381 | 0.022015 | 9.2× |
| schroeder80  | `(3, 3, 3)` | 0.002052 | **0.011324** | **5.5×** |

### Observações por dataset

**Ballbeam**: o `(3, 3, 3)` é **claramente melhor em free-run** (MPO 0.058 vs 0.117) apesar de selecionar só 2 regressores contra 4 do `(2, 2, 2)`. **Refuta a hipótese intuitiva** de que mais regressores = melhor simulação. O termo `y(k-3)` parece carregar informação que substitui efetivamente o trio `y(k-2) + u(k-1) + u(k-2)`, gerando um modelo mais parcimonioso e melhor para extrapolação.

**Wiener-Hammerstein**: **resultado idêntico** nas duas configurações. O modelo identificado é o mesmo (`y(k-1) + y(k-2)`, com mesmos coeficientes) — confirma de uma vez por todas que aumentar lags ou grau não ajuda. A não-linearidade estática do W-H não é capturável por NARMAX polinomial padrão neste conjunto de candidatos, e nem mesmo `ny = nu = 3` muda isso.

**Silverbox SNLS**: surpresa — o `(2, 2, 2)` tem **MPO ligeiramente melhor** (0.008365 vs 0.009541), apesar do `(3, 3, 3)` ter ESR menor no treino. Isso é um sinal clássico de que **o `(3, 3, 3)` está começando a overfittar** no SNLS: ele adiciona estrutura que ajuda em OSA (0.000888 vs 0.001780, 2× melhor) mas não generaliza tão bem para free-run. A ratio `MPO/OSA` de 10.7× do `(3, 3, 3)` vs 4.7× do `(2, 2, 2)` confirma essa interpretação. **Para SNLS, o modelo de 10 termos com `(2, 2, 2)` é estruturalmente mais robusto.**

**Silverbox Schroeder**: o `(3, 3, 3)` é claramente melhor (MPO 0.011 vs 0.022, **2× menos erro**), e com menos regressores (6 vs 10). Aqui o ganho do lag adicional é real e generaliza — confirma a observação do experimento 4. O termo `y(k-3)²` que apareceu no `(3, 3, 3)` traz informação que o `(2, 2, 2)` precisava distribuir entre múltiplos quadráticos.

### Conclusão da validação

O experimento 5 muda algumas conclusões intermediárias:

1. **Sobre o Ballbeam**: o `(3, 3, 3)` não é "pior" como a hipótese inicial (baseada em parcimônia de ERR) sugeria — é **melhor** em free-run. A parcimônia de 2 regressores ajudou, não atrapalhou.

2. **Sobre o SNLS**: aquela "vitória" do experimento 4 (15 → 8 regressores parando por ρ) precisa ser qualificada — em validação out-of-sample, o modelo `(3, 3, 3)` overfitta levemente. O `(2, 2, 2)` esgota `l_max`, mas seu modelo de 10 termos simula melhor. **Os números do ERR de identificação podem mentir sobre generalização**.

3. **Sobre o Schroeder**: a vitória do `(3, 3, 3)` é confirmada — gap MPO/OSA cai pela metade.

4. **Sobre o W-H**: nenhuma das configurações testadas funciona bem em free-run (gap ~44×). Modelagem específica ainda é necessária.

5. **Princípio prático**: **mais lags não é universalmente melhor** — vale por dataset, e só validação out-of-sample distingue os casos. Critérios baseados em ERR de identificação subestimam o risco de overfitting.

### Experimento 6 — lags grandes (`ny = nu = 8`, `d = 3`)

Motivação: investigar se aumentar drasticamente os lags ajuda o W-H — talvez a memória do sistema seja longa o suficiente para que `ny, nu = 3` ainda não comporte sua estrutura. Espaço de candidatos cresce para **968 termos** (16 lineares + 136 quadráticos + 816 cúbicos), ~12× maior que o anterior. Tempo de execução: ~22s para os 4 datasets.

#### Resultados — comparação direta com `(3, 3, 3)`

| Dataset | `(3, 3, 3)` MPO | `(8, 8, 3)` MPO | Mudança |
|---|---:|---:|---|
| ballbeam     | 0.057648 | 0.070772 | **+23% pior** (overfitting) |
| wienerhammer | 0.238815 | 0.238815 | **idêntico** |
| snls80       | 0.009541 | 0.009812 | +3% (~igual) |
| schroeder80  | 0.011324 | 0.020130 | **+78% pior** (overfitting) |

#### A descoberta mais marcante — o W-H não muda

Os números do Wiener-Hammerstein são **exatamente idênticos** entre as duas configurações: OSA = 0.005468, MPO = 0.238815. Mesmo com 8 lags disponíveis e 968 candidatos, o FROLS continua escolhendo *apenas* `y(k-1)` e `y(k-2)` com os mesmos coeficientes.

Conclusão dura: **a estrutura W-H é, no espaço de candidatos polinomiais usado, fundamentalmente inacessível**. A não-linearidade estática (provavelmente íngreme — saturação ou deadzone) não é aproximável por polinômios cúbicos misturados com lags. E o bloco linear pós-NL "filtra" temporalmente a contribuição da NL, espalhando informação que polinômios diretos não recompõem.

Resumindo o que sabemos sobre o W-H após 6 experimentos:
- Mesmo modelo `y(k) ≈ 1.97 y(k-1) − 0.99 y(k-2)` é selecionado em todas as configurações tentadas (`(2,2,2)`, `(2,2,3)`, `(3,3,3)`, `(8,8,3)`).
- O modelo explica 99.95% da variância em OSA mas tem ratio MPO/OSA de ~44×.
- Aumentar grau ou lags não muda nada — o resíduo carrega estrutura *não-polinomial* que o FROLS não consegue capturar com esses candidatos.

#### Schroeder e Ballbeam: overfitting confirmado

Ambos pioram em free-run com `(8, 8, 3)`:

- **Schroeder**: MPO sobe de 0.0113 → 0.0201 (+78%). OSA melhora marginalmente (0.0021 → 0.0016), sinal claro de que o modelo ajusta ruído local no treino.
- **Ballbeam**: MPO sobe de 0.058 → 0.071 (+23%). Mesma dinâmica em escala menor.

O `(3, 3, 3)` continua sendo o ponto sweet para ambos.

#### Resumo dos pontos sweet por dataset

| Dataset | Melhor configuração (até agora) | MPO RMSE |
|---|---|---:|
| ballbeam     | `(3, 3, 3)` | 0.058 |
| wienerhammer | nenhuma resolve (estrutura inacessível) | 0.239 |
| snls80       | `(2, 2, 2)` (leve vantagem) | 0.008 |
| schroeder80  | `(3, 3, 3)` | 0.011 |

### Conclusão geral dos 6 experimentos

**Aumentar lags monotônicamente NÃO ajuda** — há um ponto sweet específico por dataset, e ultrapassá-lo introduz overfitting visível em free-run. ERR e ESR de identificação subestimam esse risco; validação out-of-sample é o que distingue um modelo robusto de um modelo que apenas decorou o treino.

Para o W-H, o NARMAX polinomial puro é inadequado. Os caminhos viáveis para esse benchmark estão fora do escopo deste trabalho:
1. **Identificação estruturada por blocos** (Best Linear Approximation + estimativa não-paramétrica da NL).
2. **Polinômios de grau ≥ 4 ou 5** — caro computacionalmente e ainda assim limitado.
3. **NARMAX com termos de ruído (MA)** — pode ajudar marginalmente se o resíduo for autocorrelacionado, mas não resolve o problema estrutural.

## Discussão — evolução de `d = 2` para `d = 3`

### Resumo comparativo

| Dataset            | `d=2`                          | `d=3`                          | Mudou? |
|--------------------|--------------------------------|--------------------------------|:------:|
| ballbeam           | 4 lineares · ERR 99.91%        | 4 lineares (idêntico)          | não    |
| wienerhammer       | 2 lineares · ERR 99.95%        | 2 lineares (idêntico)          | não    |
| silverbox SNLS     | 10 reg. esgotou l_max          | 10 reg. esgotou l_max          | **sim** — termos cúbicos entram |
| silverbox Schroeder| 10 reg. esgotou l_max          | **6 reg.** parou por ρ         | **sim** — modelo mais enxuto   |

### Observações por sistema

**Ballbeam e Wiener-Hammerstein não se beneficiam de grau 3.** Os termos lineares `y(k-1), y(k-2), u(k-1), u(k-2)` já capturam ≥99.9% da variância. Aumentar o grau de candidatura só amplia o espaço de busca sem mudar o ótimo do FROLS — os termos de ordem superior nunca conseguem ERR suficiente para deslocar os lineares.

No caso do **W-H em particular**, vale notar que a não-linearidade estática do sistema (entre dois blocos lineares) **não pode** ser identificada pelo NARMAX polinomial deste experimento. Capturar essa estrutura exigiria modelagem específica (e.g. métodos Best Linear Approximation + estimativa não-paramétrica da NL), ou pelo menos lags muito maiores nos `u` para acomodar a memória dos filtros lineares.

**Silverbox revela o ganho do grau 3 — é o sistema cuja física tem componente cúbica explícita** (oscilador de Duffing, mola restauradora `F = k·x + α·x³`):

- No **SNLS**, com grau 3, **`y(k-1)³` e `u(k-1)·u(k-2)²` aparecem na seleção** (posições 5 e 8), substituindo termos quadráticos que estavam na versão `d=2`. O ERR cumulativo é similar (~99.87%), mas a *estrutura* identificada agora reflete melhor a física da planta.

- No **Schroeder**, o efeito é mais marcante: o modelo encolhe de 10 para **6 regressores**, com `y(k-1)²·y(k-2)` (coeficiente θ = −1.83) na 4ª posição. O FROLS encontrou que **um único termo cúbico** explica mais variância do que múltiplos quadráticos juntos. É a indicação mais clara de que termos não-lineares de ordem ímpar têm peso físico nesse sistema.

### Sobre o coeficiente +11.09 do termo `u(k-1)·u(k-2)²` (SNLS, d=3)

Esse valor destoa dos demais (`|θ|` típicos < 2). Em FROLS, coeficientes grandes em termos de ordem superior frequentemente indicam **quase-colinearidade** com termos já selecionados — o algoritmo "paga" um numerador alto para corrigir resíduos pequenos. Não é necessariamente erro de implementação; é um sinal de que (a) o termo poderia ser podado a posteriori (ERR é baixo, 6.5e-5), ou (b) seria conveniente experimentar com regularização. Em modelos produtivos, vale a pena marcar termos com `|θ|` muito acima da média como candidatos a revisão.

### Conclusão geral

O experimento valida o pipeline FROLS:

1. **Em sistemas dominantemente lineares (ballbeam, W-H)**, o FROLS não introduz termos não-lineares espúrios — robustez confirmada.
2. **Em sistemas com não-linearidade real (Silverbox)**, aumentar `non_lin_len` permite o algoritmo escolher termos que refletem a estrutura física, e em alguns casos (Schroeder) **reduz** o número de parâmetros necessários — parcimônia confirmada.
3. Os modelos identificados são consistentes entre experimentos diferentes na mesma planta (SNLS vs Schroeder convergem para estruturas similares com coeficientes próximos).

## Como executar

```bash
cargo run --release        # roda os 4 datasets com a configuração atual
cargo test                 # roda os testes unitários
```

A configuração (`y_len`, `u_len`, `non_lin_len`, `ρ`, `l_max`) é fixada nas chamadas a `build_regressors` e `Frols::new` em `main.rs`.

## Limitações conhecidas

- **Construção do modelo é apenas polinomial NARMAX**: não inclui termos de ruído MA (a parte "MA" do NARMAX está ausente). Para incluí-los seria preciso iterar o FROLS com resíduos como entradas adicionais (procedimento *extended least squares*).
- **Sem validação out-of-sample**: o ERR reportado é medido na mesma série usada para identificar. Os datasets têm porções de treino/teste recomendadas na literatura que ainda não foram exploradas.
- **Sem regularização**: termos com coeficiente grande em ERR baixo podem indicar overfitting localizado (ver discussão do `+11.09` acima).
- **Faltam métricas de validação** (RMSE, *one-step-ahead* vs *free-run* simulation).
