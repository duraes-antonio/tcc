# Repositório dedicado ao trabalho de conclusão de curso.

- Curso: Sistemas de Informação
- Instituição: Instituto Federal do Espiríto Santo (IFES) - Campus Serra
- Termos-chave: Segmentação de Imagens, Segmentação semântica, Visão computacional, Aprendizado Profundo, DeepLab V3+, U-Net

Período de testes:
- 2/11 a 8/11 - 25 épocas, 23.33% p/ treino, 6.66% p/ validação, 3.33% p/ teste.
**Objetivo:** descobrir configurações promissoras
- 9/11 a 15/11 - 50 épocas, 70% p/ treino, 20% p/ validação, 10% p/ teste.

## Constatações
- **Taxa de aprendizado menor (valor: 0.0005):**
    - Requer mais épocas para treinar
    - Desempenho cresce mais **lentamente**, mas de forma **constante**

- **Taxa de aprendizado maior (valor: 0.05):**
    - Requer menos épocas para treinar
    - Desempenho cresce rapidamente, mas instável **(saturação do Teste em menos de 10 épocas)**

- **Tamanho do lote maior (máximo 16 - imagens 192x192 pixels):**
    - Treinamento mais rápido por época **(133s / época)**
    - Redução do desempenho

- **Tamanho do lote menor (2 imagens/lote):**
    - Treinamento mais lento **(200s / época)**
    - Aumento do desempenho
    
- **Otimizador - Adam:**
    - X

- **Otimizador - Stochastic gradient descent (SGD):**
    - X

- **Otimizador - RMSProp:**
    - X
    
### Observações gerais:
- A taxa de aprendizado 0.0005 gerou resultados semelhantes entre o desempenho de teste e treino, além de desemepnho crescente constante
- 

### Cenários populares:
- Utilizar learning rate entre 0.001 e 0.0001 (0.005 promissor p/ Adam e 0.0005 p/ SGD)
- Utilizar o otimizador Adam
- Utilizar backbone ResNet50 ou variantes da EfficientNet
- Utilizar lotes menores com 1, 2 ou 4 imagens
- β1 = 0.9, β2 = 0.999, e = 10−8, batch = 2. LR incial = 1e−4 e LR decay of 1e−6