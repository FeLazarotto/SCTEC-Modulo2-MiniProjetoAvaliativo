# Detecção de Defeitos em Peças industriais com Redes Neurais Convolucionais (CNN)

Este repositório contém o notebook `SCTEC_Modulo2_MiniProjetoAvaliativo.ipynb` [3], desenvolvido como mini projeto avaliativo do Módulo 2 do SCTEC [3, 4]. O objetivo do projeto é realizar a identificação e classificação automatizada de defeitos de fundição industrial utilizando técnicas de visão computacional clássica (OpenCV) e deep learning com TensorFlow/Keras [4, 5, 6].

---

## 📌 Visão Geral do Projeto

O sistema realiza a leitura de imagens de peças industriais classificadas entre duas categorias [5, 8]:
- **`ok_front`**: Peças sem defeitos de fabricação [8].
- **`def_front`**: Peças com defeitos superficiais/estruturais de fundição [8].

O pipeline contempla desde o pré-processamento clássico de imagens até a construção, treinamento e auditoria de um modelo de Rede Neural Convolucional (CNN) [5, 6, 7].

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

As seguintes ferramentas e bibliotecas Python são empregadas no projeto [4]:
- **TensorFlow / Keras**: Carregamento de datasets (`image_dataset_from_directory`), data augmentation e construção/treinamento da CNN [4, 6].
- **OpenCV (`cv2`)**: Manipulação de imagens, conversão de espaço de cor, filtros de suavização, binarização e detecção de bordas [4, 5, 6].
- **Google Colab (`google.colab.drive`)**: Integração e montagem do Google Drive para acesso aos dados [4, 5].
- **Matplotlib & Seaborn**: Visualização de imagens, plots de métricas de treinamento (Loss e Acurácia) e geração da matriz de confusão [4, 7].
- **Scikit-Learn (`sklearn.metrics`)**: Métricas de avaliação de desempenho (`classification_report` e `confusion_matrix`) [4, 7].
- **NumPy & Random / Pathlib / Shutil**: Manipulação matemática, controle de diretórios e separação aleatória de amostras de teste [4, 5].

---

## 📂 Estrutura do Notebook

O código está estruturado nas seguintes etapas sequenciais [5, 6, 7]:

### 1. Inicialização e Configuração
- Montagem do diretório do Google Drive e extração automatizada do arquivo `casting_512x512.zip` [5].
- Separação automatizada de um conjunto de imagens para o teste cego final na pasta `/content/dados/teste` através da função `criar_dataset_teste` [4, 5].

### 2. Análise Exploratória Clássica (OpenCV)
- Seleção aleatória de amostras das classes `ok_front` e `def_front` [5, 8].
- Conversão das imagens para escala de cinza (`cv2.COLOR_BGR2GRAY`) [5].
- Aplicação do filtro de suavização **Gaussian Blur** para redução de ruído [5].

### 3. Destaque de Características
- **Binarização / Limiarização (Thresholding)**: Aplicação de `cv2.threshold` com limiar 90 e inversão binária [6].
- **Detecção de Bordas (Sobel)**: Filtros Sobel nas direções X e Y combinados para realce dos contornos das peças [6].

### 4. Ingestão de Dados e Data Augmentation (Keras)
- Divisão dos dados em **80% para Treino** e **20% para Validação** com imagens redimensionadas para 130x130 pixels e `batch_size=16` [6].
- Configuração do pipeline de **Data Augmentation** (`augmentation_config`) aplicando variação aleatória de brilho (delta max 0.2), variação de contraste (0,8 a 1,2) e espelhamentos horizontal e vertical [4, 6].

### 5. Arquitetura da CNN e Treinamento
Construção de um modelo sequencial customizado com a seguinte estrutura [6]:
- **Camada de Entrada & Rescaling**: Imagens de entrada (130, 130, 3) com reescalonamento dos pixels (1/255) [6].
- **Bloco Convolucional 1**: `Conv2D` (16 filtros, 3, ReLu) + `BatchNormalization` + `MaxPooling2D` [6].
- **Bloco Convolucional 2**: `Conv2D` (32 filtros, 3, ReLu) + `BatchNormalization` + `MaxPooling2D` [6].
- **Bloco Convolucional 3**: `Conv2D` (64 filtros, 3, ReLu) + `BatchNormalization` + `MaxPooling2D` [6].
- **Classificador Dense**: `Flatten` + `Dense` (32 unidades, ReLu) + `Dropout` (0.4) + `Dense` (1 unidade, ativação Sigmoid) [6].

**Configuração do Treinamento [6, 7]:**
- Otimizador: **Adam** [6].
- Função de Perda: **Binary Crossentropy** [6].
- Métrica: **Accuracy** [6].
- Callback: `EarlyStopping` monitorando a perda de validação (`val_loss`), paciência de 10 épocas e restauração dos melhores pesos [7].
- Épocas máximas: **100** [7].

### 6. Auditoria, Avaliação e Teste Cego
- Exibição de gráficos comparativos das curvas de perda (Loss) e acurácia durante as épocas [7].
- Emissão do Relatório de Classificação (`precision`, `recall`, `f1-score`) nos dados de validação [7].
- Plotagem da **Matriz de Confusão** para verificação de falsos positivos e falsos negativos [7].
- **Teste Cego Final**: Predição em lote das peças reservadas, exibindo visualmente a classe real vs. o diagnóstico da IA acompanhado da porcentagem de certeza da predição [7].

---

## 🚀 Como Executar

1. Abra o arquivo `SCTEC_Modulo2_MiniProjetoAvaliativo.ipynb` no ambiente **Google Colab** [3, 5].
2. Certifique-se de que o arquivo compactado do dataset (`casting_512x512.zip`) do link https://drive.google.com/file/d/1gyRXHrDLCbDzJaGTCaeFW2zV5UZJ99fo/view?usp=sharing esteja disponível no seu Google Drive no caminho especificado [5].
3. Execute as células sequencialmente para extrair o dataset, processar as amostras, treinar a rede CNN e visualizar a auditoria final de predição [5, 6, 7].
