# rede-neural-mlp

Notebook [trabalho_pratico_2_redes_neurais.ipynb]

---

**Estrutura do notebook:**

| Tópico | O que contém |
|---|---|
| **0 — Setup** | Imports, carregamento do CIFAR-10, normalização, split treino/val/teste, visualização de amostras |
| **1 — MLP** | Flatten → Dense(512) → Dense(256) → Dense(128) → Softmax, com Dropout e Early Stopping. Gráficos de loss e acurácia por época. Análise textual sobre a relação loss ↔ acurácia |
| **2 — CNN + Augmentation** | 3 blocos conv + BatchNorm + MaxPool + Dropout, augmentation embutido (flip, rotate, zoom, translate, brightness). ReduceLROnPlateau + Early Stopping. Visualização do efeito do augmentation |
| **3 — Busca HP** | Grid search manual: 2 learning rates × 2 dropouts × 2 tamanhos de filtros = 8 combinações. Tabela comparativa com `pandas`. Análise dos hiperparâmetros mais impactantes |
| **4 — Avaliação final** | Retreino do melhor modelo com treino+val, avaliação no teste, **matriz de confusão**, relatório por classe (`sklearn`), tabela comparativa dos 3 modelos |