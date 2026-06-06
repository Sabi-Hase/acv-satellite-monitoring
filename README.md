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
* imagens com **12 bandas**
* resolução de **512 x 512** por patch
* rótulo espacial por pixel, convertido aqui em **rótulo binário por patch**

### Distribuição dos dados

A divisão foi feita por cena para reduzir vazamento entre conjuntos:

* **Treino:** `scene1` e `scene2`
* **Validação:** `scene3`
* **Teste:** `scene4`

### Distribuição por classe

No conjunto total:

* **2.117** patches sem fogo
* **349** patches com fogo

O dataset é desbalanceado, então o projeto avalia também estratégias para lidar com esse cenário.

## Pré-processamento

As imagens foram:

* carregadas dos arquivos `.npz`
* redimensionadas para **128 x 128**
* normalizadas por patch
* convertidas para tensores PyTorch

## Modelos treinados

Foram treinadas **3 abordagens**:

### 1. BaselineCNN

CNN inicial com blocos convolucionais, BatchNorm, ReLU, MaxPooling e Dropout.

### 2. ImprovedCNN

Versão mais profunda e mais estável, com mais camadas convolucionais, Dropout progressivo e `AdaptiveAvgPool2d`.

### 3. ImprovedCNN + WeightedRandomSampler

Versão treinada com amostragem balanceada para tentar melhorar a classe minoritária.

## Resultados

### Melhor desempenho de validação

| Modelo                              | Best Val Acc |
| ----------------------------------- | -----------: |
| BaselineCNN                         |       0.7738 |
| ImprovedCNN                         |       0.8373 |
| ImprovedCNN + WeightedRandomSampler |       0.7282 |

### Desempenho no teste

| Modelo                              | Test Accuracy | F1 da classe "com fogo" |
| ----------------------------------- | ------------: | ----------------------: |
| BaselineCNN                         |          0.68 |                    0.33 |
| ImprovedCNN                         |          0.83 |                    0.52 |
| ImprovedCNN + WeightedRandomSampler |          0.70 |                    0.40 |

### Conclusão técnica

A **ImprovedCNN** foi a melhor escolha final por apresentar o melhor equilíbrio entre desempenho geral e capacidade de detectar a classe minoritária.

A versão com `WeightedRandomSampler` aumentou a sensibilidade para fogo, mas reduziu a estabilidade e o desempenho global.

## Demonstração funcional

O notebook inclui uma função de inferência para testar o modelo em novos patches e retornar:

* rótulo real
* predição do modelo
* confiança da predição

## Estrutura do repositório

```text
acv-satellite-monitoring/
├── app/
├── artifacts/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_dataset_exploration.ipynb
├── results/
├── src/
│   ├── inference/
│   ├── models/
│   └── training/
├── .gitignore
├── README.md
└── requirements.txt
```

## Arquivos importantes

Os principais artefatos gerados pelo projeto são:

* `artifacts/baseline_cnn_best.pt`
* `artifacts/improved_cnn_best.pt`
* `artifacts/improved_balanced_cnn_best.pt`
* `results/model_comparison.csv`
* `results/model_comparison.png`

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

Abra o arquivo:

```text
notebooks/01_dataset_exploration.ipynb
```

### 4. Executar as células na ordem

O notebook contém:

* exploração do dataset
* pré-processamento
* treinamento das CNNs
* avaliação final
* comparação entre arquiteturas
* inferência com novas imagens

## Tecnologias utilizadas

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* scikit-learn
* Jupyter Notebook
* FastAPI (preparado para demonstração)

## Observações

Este projeto foi construído do zero sobre uma base anterior de estudo, com foco em:

* organização de repositório
* reprodutibilidade
* documentação clara
* comparação técnica entre modelos
* aplicação prática em um cenário espacial

## Autor(es)

* **Sabrina Flores - RM550781**
* **Gabriel Riqueto - RM98685**
* **Leonardo Mansur - RM551659**

## Licença

Uso acadêmico e portfólio.
