# 🚗 Self-Drive: Estudos em Condução Autônoma e Visão Computacional

Repositório dedicado ao estudo, implementação e experimentação de algoritmos de **Visão Computacional** e **Deep Learning** voltados para sistemas de direção autônoma baseados em cursos práticos da área.

---

## 📌 Visão Geral

Este projeto reúne os conceitos fundamentais, notebooks e implementações desenvolvidos durante a realização de um curso voltado para veículos autônomos. Ao longo dos módulos, foram exploradas etapas essenciais da pipeline de percepção e tomada de decisão de um carro autônomo, desde a identificação visual de faixas até o treinamento de redes neurais convolucionais (CNNs).

---

## 🧠 Principais Aprendizados e Tópicos Abordados

Durante os estudos, foram desenvolvidas e consolidadas competências em:

### 1. Visão Computacional Clássica (OpenCV)
- **Pré-processamento de Imagem:** Conversão de espaços de cores (RGB para escala de cinza e HSV), equalização de histograma e aplicação de filtros Gaussianos para redução de ruído.
- **Detecção de Bordas:** Utilização do algoritmo de Canny para realçar descontinuidades e transições de intensidade.
- **Transformada de Hough:** Extração de retas e padrões geométricos para detecção e segmentação de faixas de tráfego (*lane detection*).
- **Região de Interesse (ROI):** Criação de máscaras e isolamento de áreas relevantes da pista para otimizar processamento computacional.

### 2. Fundamentos de Redes Neurais e Deep Learning
- **Perceptrons e Redes Multicamadas (MLP):** Entendimento de funções de perda, otimizadores e técnicas de retropropagação (*backpropagation*).
- **Classificação de Placas de Trânsito:** Implementação e treinamento de Redes Convolucionais (CNNs) para reconhecimento de sinalizações viárias.
- **Pré-processamento e Aumento de Dados (*Data Augmentation*):** Tratamento de desbalanceamento de classes, rotação, zoom, translação e variações de iluminação para aumento de robustez do modelo.

### 3. Clonagem de Comportamento (*Behavioral Cloning*)
- Conceito de predição contínua (regressão) do ângulo do volante (*steering angle*) a partir de feeds de câmeras.
- Arquiteturas inspiradas em modelos de ponta da indústria (como a arquitetura da NVIDIA para carros autônomos).

---

## ⚠️ Status do Projeto e Observação sobre o Módulo Final

> O projeto final (integração completa com o simulador de direção autônoma em tempo real) **não pôde ser concluído devido à desatualização do material e das bibliotecas do curso**.

### Motivos Técnicos:
- **Dependências e Ambientes Legados:** O curso foi construído sobre versões antigas de frameworks (versões legadas de Python, TensorFlow 1.x/Keras e pacotes de comunicação via WebSockets como `python-socketio` e `eventlet`).
- **Incompatibilidade com o Simulador:** O simulador fornecido dependia de protocolos e bibliotecas que foram descontinuadas ou sofreram *breaking changes*, inviabilizando a comunicação cliente-servidor sem que fosse necessária uma reengenharia completa da infraestrutura do curso.

Apesar do encerramento precoce da etapa do simulador, **todo o conteúdo teórico, matemático e prático de visão computacional e modelagem de redes neurais foi devidamente absorvido, validado e documentado** nos notebooks e scripts presentes neste repositório.

---

## 🛠️ Tecnologias e Ferramentas

- **Linguagem:** Python
- **Visão Computacional:** OpenCV (`cv2`)
- **Manipulação e Análise de Dados:** NumPy, Pandas, Matplotlib
- **Machine Learning & Deep Learning:** TensorFlow / Keras, Scikit-Learn
- **Ambiente de Desenvolvimento:** Jupyter Notebook / Google Colab

---

## 📂 Estrutura Sugerida do Repositório

```text
├── lane_detection/         # Detecção clássica de pistas usando OpenCV
├── traffic_signs/          # Classificação de placas de trânsito com CNN
├── behavioral_cloning/     # Modelagem preditiva para ângulo de direção
├── notebooks/              # Experimentos e anotações dos módulos
└── README.md               # Documentação do projeto
```

---

## 🎯 Próximos Passos

Como evolução deste aprendizado, pretendo aplicar os conceitos adquiridos em ferramentas e simuladores modernos, tais como:
- Ambientes de simulação atuais (ex.: [CARLA Simulator, Donkey Car Sim e MetaDrive]);
- Frameworks atualizados (PyTorch / TensorFlow 2.x);
- Projetos modernos de percepção com modelos baseados em Transformers e YOLO para detecção de múltiplos objetos em trânsito.

---
