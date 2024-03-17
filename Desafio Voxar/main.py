'''
Autor: Carlos Gabriel
Data: 16/03/24
Código teste do desafio do processo seletivo da Voxar Labs (2ª Fase)
Detecção de poses humanas em vídeos utilizando Deep Learning.
Códigos feitos e não utilizados na execução foram retirados para melhor 
organização do script, apenas o básico foi deixado.
'''

import cv2
from misc import visualization
from SimpleHigherHRNet import SimpleHigherHRNet

#Caminhos
model_patch1 = "./weights/pose_higher_hrnet_w32_512.pth"
model_patch2 = "./weights/pose_higher_hrnet_w32_640.pth"
model_patch3 = "./weights/pose_higher_hrnet_w48_640.pth"
video_patch = "panoptic.mp4"

capture = cv2.VideoCapture(video_patch)
(widht, height) = 500, 350

#Tipo de articulações escolhidas
joints = visualization.joints_dict()['coco']['skeleton']

#Escolha do modelo
model = SimpleHigherHRNet(32, 17, model_patch1)

while True:
    #Leitura do próximo frame
    sucess, frame = capture.read()

    #Se não tiver sucesso na leitura do próximo frame, sair
    if not sucess:
        break

    #Aplicação do modelo de predição e desenho
    skelletons = model.predict(frame)
    for points in skelletons:
        visualization.draw_points_and_skeleton(frame, points, joints)

    #Tecla para sair do vídeo
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #Mostrar frame redimensinado
    cv2.imshow("Teste", cv2.resize(frame, (widht, height)))

#Encerrar vídeo, fechar janela, encerrar.
capture.release()
cv2.destroyAllWindows()

