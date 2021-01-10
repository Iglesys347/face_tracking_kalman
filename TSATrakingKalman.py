from cv2 import cv2
from FiltreKalman import FiltreKalman
import numpy as np

chemin = "files/haarcascade_frontalface_default.xml"
cascade_de_visages = cv2.CascadeClassifier(chemin)

# Lecture de la video (0 pour la webcam, "video.mp4" pour la video)
# Pour arrêter le traitement par webcan appuyer sur la touche "q"
video = cv2.VideoCapture(0)

# Décommenter les lignes suivantes pour utilier une vidéo en entrée
# Renseigner le chemin vers la vidéo
#video_path = "video.mp4"
#video = cv2.VideoCapture(video_path)

largeur = 0
hauteur = 0

# Initialisation du filtre
filtre_k = FiltreKalman(1/30, 1, 0.1,0.1)

# Boucle infinie pour traiter la vidéo jusqu'à ce qu'elle soit finie
while True:
    # Une liste de couples (x,y) correspondant aux positions des visages
    centres_visages=[]
    # Lecture de la vidéo image par image
    retval, image = video.read()

    # Conversion de l'image en nuances de gris (pour faciliter le traitement)
    image_nvgris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detection des visages dans l'image en nv de gris
    visages = cascade_de_visages.detectMultiScale(
        image_nvgris,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Dessine un rectangle autour de chaque visage
    for (x, y, w, h) in visages:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        centres_visages.append(np.array([[x], [y]]))
        largeur = w #la hauteur et largeur nous sont utiles pour plus tard
        hauteur = h

    # Prediction grâce au filtre de Kalman
    (x_pred, y_pred) = filtre_k.prediction()
    # Dessine un rectangle à la position prédite
    cv2.rectangle(image, (int(x_pred), int(y_pred)), (int(x_pred + largeur), int(y_pred + hauteur)), (255, 0, 0), 2)
    # Ajout d'une légende
    cv2.putText(image, "Position predite", (int(x_pred) + 15, int(y_pred)), 0, 0.5, (255, 0, 0), 2)

    # Fase de maj du filtre Kalman
    if centres_visages != []:
        (x_estim, y_estim) = filtre_k.maj(centres_visages[0])
        # Dessine un rectangle à la position estimée grâce à Kalman
        cv2.rectangle(image, (int(x_estim), int(y_estim)), (int(x_estim + largeur), int(y_estim + hauteur)), (0, 0, 255), 2)
        # Ajout d'une légende
        cv2.putText(image, "Position estimee", (int(x_estim + 15), int(y_estim + 10)), 0, 0.5, (0, 0, 255), 2)

    # Pour sortir de la boucle en appuyant sur 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    # Affichage du résultat
    cv2.imshow('Traitement de la video', image)


# Quand on sort de la boucle, on ferme tout
video.release()
cv2.destroyAllWindows()
