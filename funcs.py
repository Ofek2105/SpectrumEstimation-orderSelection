import numpy as np


def generate_ar_signal(phis, N, c=0, sigma=1):
  """Generate an AR(p) signal.

  Args:
  - phis (list of float): List of autoregressive coefficients, length determines order p.
  - N (int): Number of points in the signal.
  - c (float): Constant term. Default is 0.
  - sigma (float): Standard deviation of the noise. Default is 1.

  Returns:
  - np.array: Generated AR(p) signal.
  """
  p = len(phis)  # Order of the AR model
  X = np.zeros(N)
  # Initial values can be set to zero or randomized
  X[:p] = np.random.normal(0, sigma, p)

  # Generate the signal
  for t in range(p, N):
    X[t] = c + np.dot(phis, X[t - p:t][::-1]) + np.random.normal(0, sigma)

  return X
