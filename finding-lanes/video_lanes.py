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
    if lines is None:
        return None
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

    # Lista que vai guardar só as linhas que realmente existirem neste frame
    result = []

    # Só calcula a média se houver pelo menos uma linha desse lado
    if len(left_fit) > 0:
        # Tira a média das inclinações e interceptos de cada lado (axis=0 = média por coluna)
        left_fit_average = np.average(left_fit, axis= 0)
        # Converte cada reta média (m, b) em coordenadas de pixel (x1, y1, x2, y2)
        result.append(make_coodenates(image, left_fit_average))

    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        # Converte cada reta média (m, b) em coordenadas de pixel (x1, y1, x2, y2)
        result.append(make_coodenates(image, right_fit_average))

    # Pode retornar 0, 1 ou 2 linhas; o display_lines já aceita qualquer quantidade
    return np.array(result)


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

cap = cv2.VideoCapture('../video/test_video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: # acabou o vídeo (ou falhou a leitura)
        break
    canny = canny_algorithm(frame)
    cropped_img = region_of_interest(canny)
    lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 50, np.array([]), minLineLength=40, maxLineGap=5)
    average_lines = average_slope_intercept(frame, lines)
    lines_image = display_lines(frame, average_lines)
    combo_image = cv2.addWeighted(frame, 0.8, lines_image, 1, 1)
    cv2.imshow('result', combo_image)
    # & 0xFF mantém só os 8 bits menos significativos do valor retornado.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()