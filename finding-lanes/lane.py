import cv2                      # OpenCV: processamento de imagens
import numpy as np              # NumPy: arrays e operações numéricas
import matplotlib.pyplot as plt # Matplotlib: usado só para visualizar a imagem (no final)

def canny_algorithm(image):
    # Converte a imagem de BGR (formato padrão do OpenCV) para tons de cinza.
    # Isso reduz 3 canais de cor para 1, simplificando o cálculo de bordas.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aplica o filtro Gaussiano com kernel 5x5 para suavizar a imagem e reduzir ruído.
    # Sem isso, ruídos pequenos seriam detectados como bordas falsas.
    # O 0 é o desvio padrão (calculado automaticamente a partir do tamanho do kernel).
    blur = cv2.GaussianBlur(gray, [5, 5], 0)
    # Detecta as bordas com o algoritmo de Canny.
    # Ele calcula o gradiente (variação de intensidade) de cada pixel:
    #   - gradiente > 150 (high): é borda com certeza
    #   - gradiente < 50 (low): é descartado
    #   - entre 50 e 150: só é borda se estiver conectado a uma borda "certa"
    canny = cv2.Canny(blur, 50, 150) # parâmetros: (img, low, high)
    return canny

def region_of_interest(image):
    # Pega a altura da imagem (shape[0] = linhas = altura em pixels)
    height = image.shape[0]
    # Define o polígono (triângulo) que cobre a região da pista onde
    # as faixas aparecem. Os pontos são (x, y):
    #   (200, height)  -> canto inferior esquerdo
    #   (1100, height) -> canto inferior direito
    #   (550, 250)     -> topo do triângulo (ponto de fuga da pista)
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    # Cria uma imagem totalmente preta (zeros) com o mesmo tamanho da imagem original
    mask = np.zeros_like(image)
    # Preenche o interior do polígono com branco (255).
    # Resultado: máscara preta fora do triângulo e branca dentro dele.
    # (Obs.: no seu comentário original, dentro = 255 e fora = 0)
    cv2.fillPoly(mask, polygons, 255)
    # O cv2.bitwise_and() faz uma operação AND bit a bit entre os pixels das duas imagens.
    # Onde a máscara é 255 (branco), o pixel da imagem é mantido;
    # onde é 0 (preto), o pixel vira 0. Assim, só as bordas dentro do triângulo sobram.
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    # Cria uma imagem preta (do tamanho da original) para desenhar as linhas nela
    line_img = np.zeros_like(image)
    # Só desenha se alguma linha foi detectada
    if lines is not None:
        # Cada linha vem como [x1, y1, x2, y2] (ponto inicial e ponto final)
        for x1, y1, x2, y2 in lines:
            # Desenha a linha na cor (255, 0, 0) = azul (em BGR) com espessura 10
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_img

def average_slope_intercept(image,lines):
    # Listas para guardar (inclinação, intercepto) das linhas de cada lado da pista
    left_fit = []
    right_fit = []
    # O Hough detecta VÁRIOS segmentos pequenos; aqui vamos juntá-los em apenas 2 linhas
    for x1, y1, x2, y2 in lines:
        # Ajusta uma reta (polinômio de grau 1: y = m*x + b) passando pelos 2 pontos.
        # polyfit retorna [inclinação, intercepto]
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]      # inclinação (m)
        intercept = parameters[1]  # onde a reta cruza o eixo y (b)
        # Como o eixo y da imagem cresce para BAIXO, a faixa esquerda
        # tem inclinação negativa e a direita tem inclinação positiva.
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    # Tira a média das inclinações e interceptos de cada lado (axis=0 = média por coluna)
    left_fit_average = np.average(left_fit, axis= 0)
    right_fit_average = np.average(right_fit, axis=0)
    # Converte cada reta média (m, b) em coordenadas de pixel (x1, y1, x2, y2)
    left_line = make_coodenates(image, left_fit_average)
    right_line = make_coodenates(image, right_fit_average)
    # Retorna as duas linhas finais (esquerda e direita)
    return np.array([left_line, right_line])


def make_coodenates(image, line_parameters):
    # Desempacota inclinação e intercepto da reta média
    slope, intercept = line_parameters
    # y1 = base da imagem (altura total)
    y1 = image.shape[0]
    # y2 = 3/5 da altura: define até onde a linha sobe na imagem
    y2 = int(y1*(3/5))
    # Isolando x da equação y = m*x + b  ->  x = (y - b) / m
    # Calcula os valores de x correspondentes a y1 e y2
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


# Lê a imagem do disco. Retorna um ndarray do NumPy com os valores dos pixels (altura x largura x 3 canais BGR)
image = cv2.imread('../img/test_image.jpg')

# Cria uma cópia da imagem para não alterar a original
lane_image = np.copy(image)

# Chama a função que converte para cinza, suaviza e detecta bordas (Canny)
canny = canny_algorithm(lane_image)

# Aplica a máscara triangular: mantém só as bordas da região da pista
cropped_img = region_of_interest(canny)

# Transformada de Hough probabilística: detecta segmentos de reta nas bordas.
# Parâmetros:
#   cropped_img         -> imagem de bordas (já com a máscara)
#   2                   -> resolução de rho (distância) em pixels
#   np.pi/180           -> resolução de theta (ângulo) em radianos (1 grau)
#   50                  -> threshold: mínimo de votos para considerar uma linha
#   np.array([])        -> placeholder (array vazio exigido pela função)
#   minLineLength=40    -> ignora segmentos menores que 40 pixels
#   maxLineGap=5        -> une segmentos separados por até 5 pixels de distância
lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 50, np.array([]), minLineLength=40, maxLineGap=5)

# Junta os vários segmentos detectados em apenas 2 linhas (esquerda e direita)
average_lines = average_slope_intercept(lane_image, lines)

# Desenha essas 2 linhas em uma imagem preta
lines_image = display_lines(lane_image, average_lines)

# Combina a imagem original com a imagem das linhas (sobreposição):
#   lane_image * 0.8 + lines_image * 1 + 1(gama)
# O 0.8 escurece um pouco a original para as linhas azuis se destacarem
combo_image = cv2.addWeighted(lane_image, 0.8, lines_image, 1, 1)

# Mostra o resultado em uma janela
cv2.imshow('result', combo_image)
# Espera uma tecla ser pressionada (0 = espera indefinidamente) antes de fechar a janela
cv2.waitKey(0)
"""
# Visualize the X and Y to get the coordinates
# Útil para descobrir as coordenadas (x, y) do triângulo da região de interesse,
# já que o matplotlib mostra os eixos com os valores dos pixels
plt.imshow(canny)
plt.show()
"""