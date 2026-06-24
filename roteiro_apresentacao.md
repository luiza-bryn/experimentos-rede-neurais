# Roteiro de Apresentação (10 min) — Trabalho Prático 2: Redes Neurais

Baseado na estrutura sugerida no enunciado e no notebook `rede_neural.ipynb`.

---

## 1. Introdução e objetivo (1 min)

"Este trabalho tem como objetivo classificar imagens do conjunto **CIFAR-10**, que possui 10 classes — airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck — com 60 mil imagens 32×32 coloridas (50 mil treino, 10 mil teste). Separamos 10% do treino (5.000 imagens) como conjunto de validação.

Vamos comparar três abordagens em complexidade crescente:
1. Uma MLP simples, sem camadas convolucionais;
2. Uma CNN com data augmentation;
3. Uma busca de hiperparâmetros sobre a CNN, retreinando o melhor modelo e avaliando no conjunto de teste."

*(Mostrar a célula `load_data` e a grade de amostras do CIFAR-10 — `show_samples`.)*

---

## 2. MLP simples + análise (2 min)

"A MLP recebe a imagem 32×32×3 **achatada** em um vetor de 3072 valores (`Flatten`), perdendo a estrutura espacial. Usamos 3 camadas densas (512, 256, 128 neurônios) com ReLU e Dropout de 0.3, totalizando ~1,7 milhão de parâmetros, treinada com Adam (lr=1e-3) e EarlyStopping monitorando `val_loss`."

*(Mostrar `mlp_build` com o `summary()` e o gráfico de `plot_history`.)*

Pontos de análise a falar (conforme pedido no item 1 do enunciado):

- **Por que acurácia e loss não são proporcionais?** A loss (categorical crossentropy) penaliza a *confiança* da predição — um acerto com baixa probabilidade aumenta bastante a loss, e um erro com margem pequena aumenta pouco. Já a acurácia só olha se a classe de maior probabilidade está certa, ignorando a margem. Por isso a loss pode cair enquanto a acurácia estagna, ou a acurácia subir enquanto a loss sobe.
- **O desempenho foi satisfatório?** A MLP girou em torno de ~35–42% de acurácia de validação nas épocas treinadas — bem acima do acaso (10%), mas insuficiente, porque trata pixels de forma independente, sem captar padrões espaciais (bordas, formas). O modelo tende a apresentar sinais de overfitting/estagnação ao longo das épocas.

---

## 3. CNN + data augmentation (2 min)

"Para capturar a estrutura espacial das imagens, construímos uma CNN com 3 blocos convolucionais (32→64→128 filtros), BatchNormalization, MaxPooling e Dropout crescente (0.2, 0.3, 0.4), finalizando com GlobalAveragePooling e uma camada densa de 256 neurônios. O modelo tem apenas ~176 mil parâmetros — muito menor que a MLP, mas mais eficiente."

*(Mostrar a célula `cnn_augmentation` com a visualização de imagens aumentadas — flip, rotação, zoom, translação — e depois `cnn_build`/`summary()`.)*

"O **data augmentation** é aplicado só durante o treino: gera variações artificiais (flip horizontal, rotação, zoom, translação) das mesmas imagens, simulando um dataset maior sem coletar dados novos."

Pontos de análise (item 2 do enunciado):

- **Indicação do uso de augmentation:** o CIFAR-10 tem só 5.000 imagens por classe no treino; redes convolucionais com muitos parâmetros tendem a memorizar esses poucos exemplos. O augmentation aumenta a diversidade vista pelo modelo a cada época, reduzindo overfitting — é especialmente importante quando há pouca quantidade de dados por categoria.
- **Desempenho satisfatório?** A CNN supera claramente a MLP (na ordem de 70–80% de acurácia de validação), e as curvas de treino/validação convergem de forma mais próxima, indicando melhor generalização que a MLP.

*(Mostrar `cnn_plot` com as curvas de loss/acurácia.)*

---

## 4. Otimização de hiperparâmetros (2–3 min)

"Implementamos uma busca em grade (`itertools.product`) testando 3 hiperparâmetros, 2 valores cada — 8 combinações no total:

- **Learning rate**: 1e-3 e 5e-4
- **Dropout**: 0.2 e 0.4
- **Número de filtros base** (dobrando a cada bloco): 32 e 64

Cada combinação treina uma CNN parametrizada (`build_cnn_hp`) por até 30 épocas, com EarlyStopping."

*(Mostrar a célula `hp_search` e a tabela `df_results` ordenada por `val_accuracy`.)*

Pontos de análise (item 3 do enunciado):

- Mostrar a tabela comparativa com as 8 combinações e indicar a **melhor combinação encontrada** (maior `val_accuracy`, menor `val_loss`).
- Explicar o impacto esperado de cada parâmetro: o **número de filtros** tende a ter o maior impacto, pois define a capacidade de representação do modelo; o **dropout** regula overfitting (dropout excessivo pode causar underfitting); o **learning rate** afeta a estabilidade e velocidade de convergência — valores muito altos desestabilizam o treino, muito baixos exigem mais épocas.
- Concluir indicando qual combinação foi escolhida para a etapa final.

---

## 5. Resultado final no teste + conclusão (1–2 min)

"Com a melhor combinação de hiperparâmetros, retreinamos o modelo usando **treino + validação combinados** e avaliamos no conjunto de **teste**, nunca visto antes."

*(Mostrar `final_eval` com loss/acurácia de teste, e a matriz de confusão `confusion_matrix`.)*

Pontos de análise (item 4 do enunciado):

- Apresentar a acurácia e loss finais de teste.
- **Generalização:** se a acurácia de teste estiver próxima da acurácia de validação obtida durante a busca, o modelo generaliza bem; uma queda grande indicaria viés introduzido pela escolha do modelo com base na validação (uma forma indireta de leakage).
- Comentar a matriz de confusão: classes visualmente parecidas (ex.: cat vs dog, automobile vs truck) tendem a ser mais confundidas — é esperado.
- **Conclusão final:** mostrar a tabela comparativa (`compare_models`) MLP → CNN → CNN otimizada, reforçando que cada etapa trouxe ganho de acurácia, evidenciando o valor das camadas convolucionais, do data augmentation e do ajuste de hiperparâmetros.

---

### Checklist antes de gravar
- [ ] Ter os outputs do notebook já executados e salvos (gráficos, tabelas, matriz de confusão).
- [ ] Cronometrar cada bloco para não passar de 10 minutos.
- [ ] Citar qualquer ferramenta de IA usada como apoio (se aplicável).
