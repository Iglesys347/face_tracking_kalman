"""
Module du filtre de Kalman
"""
import numpy as np

class FiltreKalman(object):
    def __init__(self, dt, std_acc, x_std_meas, y_std_meas):
        """
        Classe pour l'utilisation du filtre de Kalman
        :param dt: temps d'echantillonnage
        :param std_acc: bruit de transition standard
        :param x_std_meas: bruit de deviation standard en x
        :param y_std_meas: bruit de deviation standard en y
        """

        # On initialise dt
        self.dt = dt

        # L'etat initial (on  l'initialise à 0)
        self.x = np.matrix([[0], [0], [0], [0], [0], [0]])

        # La matrice de transition A
        self.A = np.matrix([[1, 0, self.dt, 0, 0, 0],
                            [0, 1, 0, self.dt, 0, 0],
                            [0, 0, 1, 0, self.dt, 0],
                            [0, 0, 0, 1, 0, self.dt],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])


        # La matrice d'observation H
        self.H = np.matrix([[1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0]])

        # Le bruit de transiton
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2, 0, self.dt**2],
                            [(self.dt**3)/2, 0, self.dt**2, 0, self.dt, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2, 0, self.dt],
                            [self.dt**2, 0, self.dt, 0, 1, 0],
                            [0, self.dt**2, 0, self.dt, 0, 1]]) * std_acc**2

        # Bruit d'observation
        self.R = np.matrix([[x_std_meas**2,0],
                           [0, y_std_meas**2]])

        # La matrice de covariance
        self.P = np.eye(self.A.shape[1])

    def prediction(self):
        """Retourne la position prédite en applicant un filtre de Kalman"""
        # Calcul des positions : x_k = Ax_(k-1)
        self.x = np.dot(self.A, self.x)

        # Mise a jour de P : P_k = A*P_(k-1)*A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return self.x[0:2]

    def maj(self, z):
        """Correction de la prediction grace a l'observation k"""
        # S_k = H*P_k*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calcul du gain du filtre : K_k = P_k * H'* S_k^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Calcul des coordonnes maj
        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))

        I = np.eye(self.H.shape[1])
        # Maj de P
        self.P = (I - (K * self.H)) * self.P
        return self.x[0:2]
