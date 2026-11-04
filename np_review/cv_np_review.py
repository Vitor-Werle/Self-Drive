"""
Revisão Prática de NumPy para Visão Computacional em Python
------------------------------------------------------------
Em Visão Computacional (OpenCV, scikit-image, PyTorch, etc.), imagens são
representadas fundamentalmente como arrays multidimensionais (ndarrays) do NumPy.

Estrutura típica de imagens em NumPy:
 - Tons de Cinza (Grayscale): Matriz 2D de formato (H, W) -> (Altura, Largura)
 - Imagem Colorida (RGB/BGR): Tensor 3D de formato (H, W, C) -> (Altura, Largura, Canais)
 - Lote de Imagens (Batches): Tensor 4D de formato (N, H, W, C) ou (N, C, H, W)
 - Tipo de dado padrão: np.uint8 (valores inteiros de 0 a 255)
"""

import numpy as np


# ==============================================================================
# 1. CRIAÇÃO DE IMAGENS SINTÉTICAS E TIPOS DE DADOS (DTYPES)
# ==============================================================================
print("=== 1. Criação e Dtypes ===")

# Uma imagem preta (grayscale) de 100x100 pixels
# uint8 é o tipo canônico: ocupa 1 byte por pixel/canal (intervalo: [0, 255])
img_preta = np.zeros((100, 100), dtype=np.uint8)

# Uma imagem branca
img_branca = np.ones((100, 100), dtype=np.uint8) * 255

# Imagem colorida sintética (H=100, W=100, Canais=3 -> BGR/RGB)
# Criando uma imagem azul pura em padrão RGB (R=0, G=0, B=255)
img_azul = np.zeros((100, 100, 3), dtype=np.uint8)
img_azul[:, :, 2] = 255  # Último canal (B no RGB)

print(f"Shape da imagem colorida: {img_azul.shape}")
print(f"Tipo de dados (dtype): {img_azul.dtype}")
print(f"Total de bytes em memória: {img_azul.nbytes} bytes\n")


# ==============================================================================
# 2. SLICING E INDEXAÇÃO AVANÇADA (ROI - REGION OF INTEREST)
# ==============================================================================
print("=== 2. Slicing e ROI ===")

# Simulando uma imagem colorida (altura=200, largura=300, canais=3)
img = np.random.randint(0, 256, size=(200, 300, 3), dtype=np.uint8)

# Extrair uma ROI (Região de Interesse): linhas 50 a 150, colunas 100 a 250
# No NumPy, a indexação sempre segue a ordem: [Y, X] ou [linhas, colunas]
roi = img[50:150, 100:250]
print(f"Shape da ROI selecionada: {roi.shape}")  # (100, 150, 3)

# Copiar vs. Visualização (View):
# Slicing comum no NumPy gera uma VIEW. Modificar a view altera a imagem original!
# Para manter a original intacta, use .copy():
roi_independente = img[50:150, 100:250].copy()

# Desenhando um retângulo preto na imagem original usando slicing:
img[20:60, 20:60] = 0

# Utilizando Ellipsis (...) para acessar todos os canais de um pixel específico:
pixel_central = img[100, 150, ...]
print(f"Canais do pixel central: {pixel_central}\n")


# ==============================================================================
# 3. SEPARAÇÃO E MANIPULAÇÃO DE CANAIS
# ==============================================================================
print("=== 3. Manipulação de Canais ===")

# Em OpenCV a ordem padrão é BGR, enquanto PIL e Matplotlib esperam RGB.
img_bgr = np.random.randint(0, 256, size=(120, 120, 3), dtype=np.uint8)

# Como converter BGR para RGB puramente com NumPy:
# Invertendo a ordem do último eixo (canais) com slicing com passo negativo:
img_rgb = img_bgr[..., ::-1]

# Isolar canais específicos:
canal_b = img_bgr[:, :, 0]
canal_g = img_bgr[:, :, 1]
canal_r = img_bgr[:, :, 2]

# Mesclar canais de volta (merge):
img_reconstruida = np.stack([canal_b, canal_g, canal_r], axis=-1)
print(f"Canais isolados e reconstruídos com shape: {img_reconstruida.shape}\n")


# ==============================================================================
# 4. MÁSCARAS BOOLEANAS E LIMIARIZAÇÃO (THRESHOLDING)
# ==============================================================================
print("=== 4. Máscaras e Thresholding ===")

cinza = np.array([[50, 120, 200],
                  [10, 180, 240],
                  [90, 130, 210]], dtype=np.uint8)

# Segmentação simples (Threshold): pixels acima de 128
mascara = cinza > 128
print("Máscara Booleana (Pixels > 128):")
print(mascara)

# Aplicando binarização (imagem binária 0 ou 255):
binarizada = np.where(mascara, np.uint8(255), np.uint8(0))
print("\nImagem Binarizada:")
print(binarizada)

# Substituição condicional in-place usando a máscara:
cinza_modificada = cinza.copy()
cinza_modificada[cinza < 100] = 0  # Reduz ruído/tons escuros a zero
print("\nCinza após filtro de corte:")
print(cinza_modificada, "\n")


# ==============================================================================
# 5. NORMALIZAÇÃO E PREVENÇÃO DE OVERFLOW EM OPERAÇÕES ARITMÉTICAS
# ==============================================================================
print("=== 5. Normalização e Cuidados com uint8 ===")

# ATENÇÃO ao Overflow no uint8:
a = np.array([250], dtype=np.uint8)
b = np.array([10], dtype=np.uint8)
print("Overflow comum com uint8 (250 + 10):", a + b)  # Resulta em 4! (260 % 256)

# Boas práticas para operações matemáticas (ajuste de brilho/contraste):
# 1. Converter para float32
# 2. Realizar operações
# 3. Aplicar np.clip
# 4. Converter de volta para uint8
brilho_adicional = 30.0
img_float = cinza.astype(np.float32)
img_brilhante = np.clip(img_float + brilho_adicional, 0, 255).astype(np.uint8)
print("\nAjuste seguro de brilho com clip:")
print(img_brilhante)

# Normalização para Redes Neurais (intervalo [0.0, 1.0] ou padronização z-score):
img_norm_0_1 = cinza.astype(np.float32) / 255.0
media, desvio = img_norm_0_1.mean(), img_norm_0_1.std()
img_padronizada = (img_norm_0_1 - media) / (desvio + 1e-7)
print(f"\nNormalização [0, 1] - min: {img_norm_0_1.min():.2f}, max: {img_norm_0_1.max():.2f}")
print(f"Padronização - média: {img_padronizada.mean():.2f}, std: {img_padronizada.std():.2f}\n")


# ==============================================================================
# 6. OPERAÇÕES ESPACIAIS: FLIP, ROTAÇÃO E TRANSPOSIÇÃO
# ==============================================================================
print("=== 6. Transformações Geométricas Rápidas ===")

amostra = np.arange(9, dtype=np.uint8).reshape(3, 3)
print("Matriz Original:\n", amostra)

# Espelhamento horizontal (flip horizontal):
flip_h = np.fliplr(amostra)  # ou amostra[:, ::-1]
print("Flip Horizontal:\n", flip_h)

# Espelhamento vertical (flip vertical):
flip_v = np.flipud(amostra)  # ou amostra[::-1, :]
print("Flip Vertical:\n", flip_v)

# Rotação de 90 graus no sentido anti-horário:
rotacionada = np.rot90(amostra, k=1)
print("Rotacionada 90°:\n", rotacionada, "\n")


# ==============================================================================
# 7. CONVERSÃO DE FORMATOS PARA DEEP LEARNING (HWC <-> CHW e BATCHES)
# ==============================================================================
print("=== 7. Formatos para Deep Learning (PyTorch / ONNX) ===")

# OpenCV / NumPy usam formato HWC: (Altura, Largura, Canais)
# PyTorch usa formato CHW: (Canais, Altura, Largura)
tensor_hwc = np.zeros((224, 224, 3), dtype=np.float32)

# np.transpose permite reordenar os eixos:
# Eixo 2 (Canais) vai para posição 0
# Eixo 0 (Altura) vai para posição 1
# Eixo 1 (Largura) vai para posição 2
tensor_chw = np.transpose(tensor_hwc, (2, 0, 1))
print(f"Shape HWC original: {tensor_hwc.shape} -> Transposto para CHW: {tensor_chw.shape}")

# Adicionando dimensão de Batch (NCHW):
# np.expand_dims adiciona uma dimensão extra na posição desejada (ex: batch_size=1)
tensor_batch = np.expand_dims(tensor_chw, axis=0)  # ou tensor_chw[np.newaxis, ...]
print(f"Shape pronto para inferência (Batch, C, H, W): {tensor_batch.shape}\n")

print("Revisão concluída com sucesso!")
