# face_tracking_kalman
Implémentation d'un tracker de visage avec un filtre de Kalman.

Pour plus de détails concernant cette implémentation ou son utilisation, se réferer au rapport.

## FiltreKalman
Le module FiltreKalman regroupe la classe et les méthodes se correspondant au filtre de Kalman.

## TSATrackingKalman
Ce script est en quelque sorte le main de ce projet.

### Tracking de visages avec la webcan
Le script utilise de base le flux vidéo fourni par la webcam de l'ordinateur. Pour mettre fin à l'acquisition via la webcam, il faut appuyer sur la touche "q".

### Tracking de visages sur une vidéo
On peut modifier légèrement le script pour pouvoir effectuer le traitement sur une vidéo dont on doit spécifier le chemin dans le script.

## Bibliothèques
Pour fonctionner, ce projet a besoin de deux bibliohtèques : 
- numpy
- opencv-python
