import numpy as np


class KalmanFilter():
    def __init__(self, Q_mult: float = 0.01, R_mult: float = 0.01, P_mult=0.01):
        """
        Initialize the Kalman Filter.

        Parameters:
            Q_mult (float): Multiplier for process noise covariance.
            R_mult (float): Multiplier for measurement noise covariance.
            P_mult (float): Multiplier for error covariance.
        """

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
        """
        Predict the next state.
        """

        # Predict next state => x_t | t-1
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, x, y):
        """
        Update step of Kalman Filter using linear observation.

        Parameters:
            x (float): Independent variable.
            y (float): Dependent variable.
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
        Update step of Kalman Filter using VECM observation.

        Parameters:
            x1 (float): First independent variable.
            x2 (float): Second independent variable.
            vecm (float): Dependent variable (VECM value).
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
        """
        Get the parameters of the Kalman Filter.
        """

        # Return w0 and w1 of the linear model
        return self.W[0], self.W[1]
