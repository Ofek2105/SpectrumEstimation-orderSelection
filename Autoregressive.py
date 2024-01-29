import numpy as np
import matplotlib.pyplot as plt


def generate_random_AR_signal(order, num_samples, plot):
  theta = np.random.rand(order)
  signal = np.zeros(num_samples)

  for i in range(order, num_samples):
    signal[i] = np.dot(signal[i - order:i][::-1], theta) + np.random.normal(0, 1)

  if plot:
    plt.plot(signal)
    plt.show()

  return signal


def calculate_information_criterion(data, order, criteria_type):
  N = len(data)
  X = np.zeros((N - order, order))
  for i in range(order, N):
    X[i - order] = data[i - order:i]

  theta = np.linalg.lstsq(X, data[order:], rcond=None)[0]

  predicted_values = np.zeros(N)
  for i in range(order, N):
    predicted_values[i] = np.dot(data[i - order:i][::-1], theta)

  residuals = data - predicted_values
  rss = np.sum(residuals[order:] ** 2)

  k = order
  if criteria_type == 'AIC':
    criterion = 2 * k - 2 * np.log(rss)
  elif criteria_type == 'AICc':
    criterion = 2 * k - 2 * np.log(rss) + (2 * k * (k + 1)) / (N - k - 1)
  elif criteria_type == 'BIC':
    criterion = np.log(N) * k - 2 * np.log(rss)
  elif criteria_type == 'GIC':
    gamma_k = 1  # Adjust gamma_k value if needed
    criterion = -2 * np.log(rss) + 2 * k * (1 + gamma_k / N)
  else:
    raise ValueError("Invalid criteria type provided.")

  return criterion


def plot_information_criteria(criteria_values, max_order_to_check_):
  orders = np.arange(1, max_order_to_check_ + 1)
  plt.figure(figsize=(10, 8))
  for criteria_type, values in criteria_values.items():
    plt.plot(orders, values, label=criteria_type, marker='o')

  plt.title('Information Criteria for Different Model Orders')
  plt.xlabel('Model Order')
  plt.ylabel('Criterion Value')
  plt.xticks(orders)
  plt.legend()
  plt.grid(True)
  plt.show()


def get_each_criteria_order_estimation(ar_signal, max_order_to_check):
  criteria_types = ['AIC', 'AICc', 'BIC', 'GIC']
  criteria_values = {criteria: [] for criteria in criteria_types}

  for i in range(1, max_order_to_check + 1):
    for criteria_type in criteria_types:
      criterion = calculate_information_criterion(ar_signal, i, criteria_type)
      criteria_values[criteria_type].append(criterion)

  return criteria_values


def main():
  np.random.seed(1)

  true_order = 2
  num_samples = 100
  max_order_to_check = 10
  ar_signal = generate_random_AR_signal(true_order, num_samples, plot=True)

  criteria_values_per_order = get_each_criteria_order_estimation(ar_signal, max_order_to_check)

  plot_information_criteria(criteria_values_per_order, max_order_to_check)

  optimal_orders = {criteria_type: np.argmin(values) + 1 for criteria_type, values in criteria_values_per_order.items()}

  for criteria_type, order in optimal_orders.items():
    print(f"Optimal model order selected using {criteria_type}: {order}")


if __name__ == "__main__":
  main()
