# ACV Satellite Monitoring

Classificação de imagens de satélite para monitoramento de queimadas florestais com redes neurais convolucionais treinadas do zero.

## Visão geral

Este projeto foi desenvolvido para a disciplina **Applied Computer Vision (ACV)** e tem como objetivo resolver um problema de **classificação binária de imagens** no contexto de **sensoriamento remoto e monitoramento ambiental**, aplicado ao cenário da **Indústria Espacial**.

A proposta utiliza imagens de satélite do dataset **Sen2Fire** para identificar se um patch contém ou não ocorrência de fogo.

## Problema abordado

O objetivo é classificar cada patch de imagem em duas classes:

* **0 — sem fogo**
* **1 — com fogo**

A motivação é apoiar o monitoramento de queimadas florestais com visão computacional aplicada a imagens de satélite.

## Dataset

O projeto utiliza o dataset **Sen2Fire**, composto por patches de satélite com:

* **2.466 patches** no total
* imagens com **12 bandas espectrais**
* resolução de **512 × 512** por patch
* rótulo espacial por pixel, convertido neste projeto para **rótulo binário por patch**

### Divisão dos dados

A divisão foi realizada por cena para reduzir vazamento de informação entre conjuntos:

* **Treino:** `scene1` e `scene2`
* **Validação:** `scene3`
* **Teste:** `scene4`

### Distribuição por classe

No conjunto completo:

* **2.117** patches sem fogo
* **349** patches com fogo

O dataset apresenta desbalanceamento entre classes, motivando a avaliação de estratégias específicas para esse cenário.

## Pré-processamento

As imagens foram:

* carregadas dos arquivos `.npz`
* redimensionadas para **128 × 128**
* normalizadas por patch
* convertidas para tensores PyTorch

## Modelos treinados

Foram avaliadas três abordagens.

### 1. BaselineCNN

CNN inicial composta por blocos convolucionais com:

* Convolução
* Batch Normalization
* ReLU
* Max Pooling
* Dropout

### 2. ImprovedCNN

Versão mais profunda e mais estável contendo:

* mais camadas convolucionais
* regularização por Dropout progressivo
* camada `AdaptiveAvgPool2d`

### 3. ImprovedCNN + WeightedRandomSampler

Versão treinada com amostragem balanceada para aumentar a representatividade da classe minoritária.

## Resultados

### Melhor desempenho de validação

| Modelo                              | Best Val Acc |
| ----------------------------------- | ------------ |
| BaselineCNN                         | 0.7738       |
| ImprovedCNN                         | 0.8373       |
| ImprovedCNN + WeightedRandomSampler | 0.7282       |

### Desempenho no conjunto de teste

| Modelo                              | Accuracy | F1 (classe fogo) |
| ----------------------------------- | -------- | ---------------- |
| BaselineCNN                         | 0.68     | 0.33             |
| ImprovedCNN                         | 0.83     | 0.52             |
| ImprovedCNN + WeightedRandomSampler | 0.70     | 0.40             |

### Conclusão técnica

A **ImprovedCNN** apresentou o melhor equilíbrio entre desempenho geral e capacidade de detecção da classe de interesse.

Embora o uso de `WeightedRandomSampler` tenha aumentado o recall da classe fogo, houve perda de desempenho global quando comparado à arquitetura ImprovedCNN sem amostragem balanceada.

## Demonstração funcional

O projeto inclui uma aplicação simples em Streamlit capaz de:

* Carregar arquivos `.npz`
* Visualizar bandas do satélite
* Exibir o rótulo real
* Executar inferência utilizando o melhor modelo treinado
* Apresentar probabilidades e confiança da predição
* A aplicação Streamlit utiliza a ImprovedCNN como modelo final.

## Estrutura do repositório

```text
acv-satellite-monitoring/
├── app/
│   └── streamlit_app.py
├── artifacts/
├── notebooks/
│   └── 01_dataset_exploration.ipynb
├── results/
├── src/
│   └── models/
│       ├── __init__.py
│       └── cnn_models.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Arquivos gerados

- `artifacts/improved_cnn_best.pt`
- `results/model_comparison.csv`
- `results/model_comparison.png`
- `app/streamlit_app.py`
- `src/models/cnn_models.py`

## Como executar

### 1. Criar e ativar o ambiente virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar dependências

```powershell
pip install -r requirements.txt
```

### 3. Abrir o notebook

Abra:

```text
notebooks/01_dataset_exploration.ipynb
```

### 4. Executar as células na ordem

O notebook contém:

* exploração do dataset
* pré-processamento
* treinamento das CNNs
* avaliação dos modelos
* comparação entre arquiteturas
* inferência em novos patches

### 5. Executar a interface Streamlit

```powershell
streamlit run app/streamlit_app.py
```

## Tecnologias utilizadas

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* scikit-learn
* Jupyter Notebook
* Streamlit

## Observações

Projeto desenvolvido com foco em:

* visão computacional aplicada
* sensoriamento remoto
* monitoramento ambiental
* reprodutibilidade experimental
* comparação de arquiteturas de CNN

## Autor(es)

* Sabrina Flores - RM550781
* Gabriel Riqueto - RM98685
* Leonardo Mansur - RM551659

## Licença

Uso acadêmico e portfólio.
