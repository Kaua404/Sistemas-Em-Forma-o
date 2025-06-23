import cv2
import numpy as np
from scipy.spatial import distance as dist
import argparse # Importamos a biblioteca para argumentos

# A função principal não muda
def detectar_tamanho_objetos(image_path, referencia_largura_mm):
    """
    Detecta e mede o tamanho de objetos em uma imagem usando um objeto de referência.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem em '{image_path}'")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 1:
        print("Nenhum contorno foi encontrado na imagem.")
        return

    def get_contour_x(contour):
        M = cv2.moments(contour)
        return int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0

    sorted_contours = sorted(contours, key=get_contour_x)

    pixels_por_metrica = None

    for contour in sorted_contours:
        if cv2.contourArea(contour) < 100:
            continue

        box = cv2.minAreaRect(contour)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = (tl[0] + tr[0]) * 0.5, (tl[1] + tr[1]) * 0.5
        (blbrX, blbrY) = (bl[0] + br[0]) * 0.5, (bl[1] + br[1]) * 0.5
        (tlblX, tlblY) = (tl[0] + bl[0]) * 0.5, (tl[1] + bl[1]) * 0.5
        (trbrX, trbrY) = (tr[0] + br[0]) * 0.5, (tr[1] + br[1]) * 0.5

        largura_pixels = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        altura_pixels = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        if pixels_por_metrica is None:
            pixels_por_metrica = largura_pixels / referencia_largura_mm

        largura_real = largura_pixels / pixels_por_metrica
        altura_real = altura_pixels / pixels_por_metrica

        cv2.putText(image, f"{largura_real:.1f}mm",
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.putText(image, f"{altura_real:.1f}mm",
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

    output_path = "resultado.jpg"
    cv2.imwrite(output_path, image)
    print(f"Imagem com as medições salva em: {output_path}")

# --- PONTO DE ENTRADA DO PROGRAMA ---
if __name__ == "__main__":
    # 1. Configurar o parser de argumentos
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imagem", required=True,
                    help="Caminho para a imagem de entrada")
    ap.add_argument("-l", "--largura", type=float, required=True,
                    help="Largura do objeto de referência (o mais à esquerda) em milímetros")
    args = vars(ap.parse_args())

    # 2. Chamar nossa função com os argumentos fornecidos pelo usuário
    detectar_tamanho_objetos(args["imagem"], args["largura"])