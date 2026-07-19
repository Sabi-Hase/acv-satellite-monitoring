# ACV Satellite Monitoring

**Languages / Idiomas**

- 🇧🇷 [Português](#acv-satellite-monitoring)
- 🇬🇧 [English](#english)

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

## Autora

* Sabrina Flores - RM550781

## Licença

Uso acadêmico e portfólio.

# English

# ACV Satellite Monitoring

Satellite image classification for wildfire monitoring using convolutional neural networks trained from scratch.

## Overview

This project was developed for the **Applied Computer Vision (ACV)** course and aims to solve a **binary image classification** problem in the context of **remote sensing and environmental monitoring**, applied to the **Space Industry**.

The proposed solution uses satellite imagery from the **Sen2Fire** dataset to determine whether an image patch contains wildfire activity.

## Problem Statement

The objective is to classify each image patch into one of two classes:

* **0 — No Fire**
* **1 — Fire**

The motivation is to support wildfire monitoring through computer vision techniques applied to satellite imagery.

## Dataset

The project uses the **Sen2Fire** dataset, which consists of satellite image patches with:

* **2,466 patches** in total
* **12 spectral bands**
* **512 × 512** pixel resolution per patch
* pixel-level spatial labels, converted in this project into **binary patch-level labels**

### Data Split

The dataset was split by scene to reduce information leakage between subsets:

* **Training:** `scene1` and `scene2`
* **Validation:** `scene3`
* **Testing:** `scene4`

### Class Distribution

Across the complete dataset:

* **2,117** patches without fire
* **349** patches with fire

The dataset is imbalanced, motivating the evaluation of techniques specifically designed to handle class imbalance.

## Pre-processing

The images were:

* loaded from `.npz` files
* resized to **128 × 128**
* normalized on a per-patch basis
* converted into PyTorch tensors

## Trained Models

Three different approaches were evaluated.

### 1. BaselineCNN

An initial CNN architecture composed of convolutional blocks including:

* Convolution
* Batch Normalization
* ReLU
* Max Pooling
* Dropout

### 2. ImprovedCNN

A deeper and more stable architecture featuring:

* additional convolutional layers
* progressive Dropout regularization
* an `AdaptiveAvgPool2d` layer

### 3. ImprovedCNN + WeightedRandomSampler

The same ImprovedCNN architecture trained using balanced sampling to improve representation of the minority class.

## Results

### Best Validation Performance

| Model                               | Best Val Acc |
| ----------------------------------- | ------------ |
| BaselineCNN                         | 0.7738       |
| ImprovedCNN                         | 0.8373       |
| ImprovedCNN + WeightedRandomSampler | 0.7282       |

### Test Set Performance

| Model                               | Accuracy | Fire F1-score |
| ----------------------------------- | -------- | ------------- |
| BaselineCNN                         | 0.68     | 0.33          |
| ImprovedCNN                         | 0.83     | 0.52          |
| ImprovedCNN + WeightedRandomSampler | 0.70     | 0.40          |

### Technical Conclusion

The **ImprovedCNN** achieved the best balance between overall performance and the ability to detect the target class.

Although using `WeightedRandomSampler` increased recall for the fire class, it resulted in lower overall performance compared with the standard ImprovedCNN architecture.

## Functional Demonstration

The project includes a simple Streamlit application capable of:

* loading `.npz` files
* visualising satellite spectral bands
* displaying the ground-truth label
* running inference using the best-performing trained model
* presenting prediction probabilities and confidence scores

The Streamlit application uses **ImprovedCNN** as the final deployed model.

## Repository Structure

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

## Generated Files

* `artifacts/improved_cnn_best.pt`
* `results/model_comparison.csv`
* `results/model_comparison.png`
* `app/streamlit_app.py`
* `src/models/cnn_models.py`

## Running the Project

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install the dependencies

```powershell
pip install -r requirements.txt
```

### 3. Open the notebook

Open:

```text
notebooks/01_dataset_exploration.ipynb
```

### 4. Run all notebook cells

The notebook includes:

* dataset exploration
* data pre-processing
* CNN training
* model evaluation
* architecture comparison
* inference on unseen image patches

### 5. Launch the Streamlit application

```powershell
streamlit run app/streamlit_app.py
```

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* scikit-learn
* Jupyter Notebook
* Streamlit

## Notes

This project was developed with an emphasis on:

* applied computer vision
* remote sensing
* environmental monitoring
* experimental reproducibility
* CNN architecture comparison

## Author

* Sabrina Flores – RM550781

## License

Academic and portfolio use.
