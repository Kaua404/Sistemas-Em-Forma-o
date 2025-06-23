import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import namedtuple
def detectar_tamanho_objetos(image_path, referencia_largura_mm):
    """
    Detecta e mede o tamanho de objetos em uma imagem usando um objeto de referência.

    Args:
        image_path (str): O caminho para a imagem.
        referencia_largura_mm (float): A largura real do objeto de referência em milímetros.
    """
    # Passo 1: Carregar a imagem
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem em '{image_path}'")
        return

    # Passo 2: Pré-processamento
    # Converter para tons de cinza e aplicar desfoque gaussiano
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Passo 3: Detecção de Bordas (Canny)
    edged = cv2.Canny(gray, 50, 100)
    # Dilatação e Erosão para fechar lacunas nas bordas
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Passo 4: Encontrar Contornos
    # cv2.findContours retorna os contornos e a hierarquia
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Se nenhum contorno foi encontrado, não há o que fazer
    if len(contours) < 2:
        print("Não foram encontrados contornos suficientes na imagem.")
        return

    # Ordenar os contornos da esquerda para a direita
    # Isso nos ajuda a assumir que o primeiro contorno é o nosso objeto de referência
    def get_contour_x(contour):
        # Calcula o momento do contorno para encontrar o centroide
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0
        cx = int(M["m10"] / M["m00"])
        return cx

    sorted_contours = sorted(contours, key=get_contour_x)

    # Variável para armazenar a taxa de conversão (pixels por milímetro)
    pixels_por_metrica = None

    # Loop sobre todos os contornos encontrados
    for contour in sorted_contours:
        # Ignorar contornos muito pequenos para evitar ruído
        if cv2.contourArea(contour) < 100:
            continue

        # Calcula a caixa delimitadora rotacionada do contorno
        # Isso é mais preciso do que uma caixa reta (boundingRect) para objetos inclinados
        box = cv2.minAreaRect(contour)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # Desenha o contorno na imagem original
        cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)

        # Extrai as coordenadas dos cantos da caixa
        (tl, tr, br, bl) = box
        
        # Calcula os pontos médios das arestas
        (tltrX, tltrY) = (tl[0] + tr[0]) * 0.5, (tl[1] + tr[1]) * 0.5
        (blbrX, blbrY) = (bl[0] + br[0]) * 0.5, (bl[1] + br[1]) * 0.5
        (tlblX, tlblY) = (tl[0] + bl[0]) * 0.5, (tl[1] + bl[1]) * 0.5
        (trbrX, trbrY) = (tr[0] + br[0]) * 0.5, (tr[1] + br[1]) * 0.5

        # Calcula a distância euclidiana entre os pontos médios (largura e altura em pixels)
        largura_pixels = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        altura_pixels = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # Passo 5 e 6: Identificar referência e calcular a métrica
        # Se a nossa métrica ainda não foi calculada, este é o nosso objeto de referência
        if pixels_por_metrica is None:
            pixels_por_metrica = largura_pixels / referencia_largura_mm
            # Agora temos nossa "régua": sabemos quantos pixels correspondem a um milímetro

        # Passo 7: Medir os objetos
        # Converter as dimensões de pixels para milímetros
        largura_real = largura_pixels / pixels_por_metrica
        altura_real = altura_pixels / pixels_por_metrica

        # Passo 8: Exibir os resultados
        # Desenha as dimensões na imagem
        cv2.putText(image, f"{largura_real:.1f}mm",
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.putText(image, f"{altura_real:.1f}mm",
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

    # Mostra a imagem final com as medições
    # cv2.imshow("Imagem Medida", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Para uso em ambientes como notebooks, é melhor salvar a imagem
    output_path = "resultado.jpg"
    cv2.imwrite(output_path, image)
    print(f"Imagem com as medições salva em: {output_path}")
    return output_path, image

# --- CONFIGURAÇÃO E EXECUÇÃO ---
if __name__ == "__main__":
    # Crie uma imagem de exemplo para testar
    # Um retângulo branco (referência) e um círculo branco (objeto) em um fundo preto
    img_teste = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Objeto de referência (moeda): retângulo de 100 pixels de largura
    cv2.rectangle(img_teste, (50, 150), (150, 250), (255, 255, 255), -1) 
    
    # Objeto a ser medido: círculo com raio de 75 pixels (diâmetro de 150)
    cv2.circle(img_teste, (350, 200), 75, (255, 255, 255), -1)
    
    caminho_imagem_teste = "imagem_de_teste.jpg"
    cv2.imwrite(caminho_imagem_teste, img_teste)

    # LARGURA REAL DO OBJETO DE REFERÊNCIA (o retângulo)
    # Vamos supor que o retângulo de 100 pixels de largura representa nossa moeda de 27mm
    LARGURA_DA_REFERENCIA_MM = 27.0

    # Chamar a função principal
    detectar_tamanho_objetos(caminho_imagem_teste, LARGURA_DA_REFERENCIA_MM)
# ... outras importações ...
from scipy.spatial import distance as dist # LINHA CORRETA

# ... dentro do loop ...

# Calcula a distância euclidiana entre os pontos médios (largura e altura em pixels)
# Aqui estamos usando a função 'dist' que importamos de scipy.spatial
largura_pixels = dist.euclidean((27,0, 27,0), (40,7, 40,7))
altura_pixels = dist.euclidean((27,0, 27,0), (40,7, 40,7))