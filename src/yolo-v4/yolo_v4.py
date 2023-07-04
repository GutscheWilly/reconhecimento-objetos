import cv2
import time
import numpy as np

colors = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

#Carrega as classes
class_names = []
with open("coco.names.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#Captura do vídeo
cap = cv2.VideoCapture(0)
#Carrega os pesos da rede
net = cv2.dnn.readNet("yolov4weights.weights", "yolocfg.txt")

# Setando os parametros
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale = 1/255)

# Ler os frames do vídeo
while True:
    # Captura do frame
    _, frame = cap.read(0)

    # Começa da contagem do MS para detecção
    start = time.time()

    # Detecção
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    # Fim da contagem do MS
    end = time.time()

    # Percorrer todas as detecções
    for (class_id, score, box) in zip(classes, scores, boxes):

        # Gera a cor para a classe
        color = colors[int(class_id) % len(colors)]

        # Pega o nome pelo ID e o score de acurácia

        label = f'{class_names[class_id]} : {round(score*100, 2)}%'

        # Desenha a box de detecção
        cv2.rectangle(frame, box, color, 2)

        # Escreve o nome da classe em cima da box do objeto
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calcular tempo para fazer a detecção
    fps_label = f'FPS: {round(1.0 / (end - start), 2)}'

    # Escrevendo o FPS na imagem
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Mostrando a imagem
    cv2.imshow("detections", frame)

    # Espera da resposta
    if cv2.waitKey(1) == 27:
        break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()