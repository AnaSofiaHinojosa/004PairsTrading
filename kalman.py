import numpy as np

class KalmanFilter():
    def __init__(self, Q_mult: float = 0.01, R_mult: float = 0.01, P_mult = 0.01):
        self.W = np.zeros(2)

        # Transition matrix
        self.A = np.eye(2)

        # Noise in estimation
        self.Q = np.eye(2) * Q_mult

        # Noise in observation
        self.R = np.array([[1]]) * R_mult

        # Error in covariance predictions
        self.P = np.eye(2) * P_mult

    def predict(self):
        # Predict next state => x_t | t-1
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, x, y):
        """
        Update step of Kalman Filter.
        If vecm is provided, it will be used as the observed measurement.
        """
        y_obs = y
        C = np.array([[1, x]])

        # Predicted measurement
        y_hat = C @ self.W

        # Innovation (residual)
        innovation = y_obs - y_hat

        # Innovation covariance
        S = C @ self.P @ C.T + self.R

        # Kalman Gain
        K = self.P @ C.T @ np.linalg.inv(S)

        # Update estimate
        self.W = self.W + (K.flatten() * innovation)

        # Update covariance
        self.P = (np.eye(2) - K @ C) @ self.P

    def update_vecm(self, x1, x2, vecm):
        """
        Update step of Kalman Filter using VECM as observation.
        """
        y_obs = vecm
        C = np.array([[x1, x2]])

        # Predicted measurement
        y_hat = C @ self.W

        # Innovation (residual)
        innovation = y_obs - y_hat

        # Innovation covariance
        S = C @ self.P @ C.T + self.R

        # Kalman Gain
        K = self.P @ C.T @ np.linalg.inv(S)

        # Update estimate
        self.W = self.W + (K.flatten() * innovation)

        # Update covariance
        self.P = (np.eye(2) - K @ C) @ self.P


    @property
    def params(self):
        # Return w0 and w1 of the linear model
        return self.W[0], self.W[1]