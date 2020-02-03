import numpy as np


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
  fx = sigmoid(x)
  return fx * (1 - fx)


def mse_loss(y_true, y_pred):
  return ((y_true - y_pred) ** 2).mean()


class NeuralNetwork:
  def __init__(self):
    self.w1 = np.random.normal()
    self.w2 = np.random.normal()
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()

    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def feedforward(self, x):
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

  def train(self, data, all_y_trues):
    learn_rate = 0.1

    for epoch in range(1000):
      for x, y_true in zip(data, all_y_trues):
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        y_pred = o1

        p_l_p_ypred = -2 * (y_true - y_pred)

        p_ypred_p_w5 = h1 * deriv_sigmoid(sum_o1)
        p_ypred_p_w6 = h2 * deriv_sigmoid(sum_o1)
        p_ypred_p_b3 = deriv_sigmoid(sum_o1)

        p_ypred_p_h1 = self.w5 * deriv_sigmoid(sum_o1)
        p_ypred_p_h2 = self.w6 * deriv_sigmoid(sum_o1)

        p_h1_p_w1 = x[0] * deriv_sigmoid(sum_h1)
        p_h1_p_w2 = x[1] * deriv_sigmoid(sum_h1)
        p_h1_p_b1 = deriv_sigmoid(sum_h1)

        p_h2_p_w3 = x[0] * deriv_sigmoid(sum_h2)
        p_h2_p_w4 = x[1] * deriv_sigmoid(sum_h2)
        p_h2_p_b2 = deriv_sigmoid(sum_h2)

        self.w1 -= learn_rate * p_l_p_ypred * p_ypred_p_h1 * p_h1_p_w1
        self.w2 -= learn_rate * p_l_p_ypred * p_ypred_p_h1 * p_h1_p_w2
        self.b1 -= learn_rate * p_l_p_ypred * p_ypred_p_h1 * p_h1_p_b1

        self.w3 -= learn_rate * p_l_p_ypred * p_ypred_p_h2 * p_h2_p_w3
        self.w4 -= learn_rate * p_l_p_ypred * p_ypred_p_h2 * p_h2_p_w4
        self.b2 -= learn_rate * p_l_p_ypred * p_ypred_p_h2 * p_h2_p_b2

        self.w5 -= learn_rate * p_l_p_ypred * p_ypred_p_w5
        self.w6 -= learn_rate * p_l_p_ypred * p_ypred_p_w6
        self.b3 -= learn_rate * p_l_p_ypred * p_ypred_p_b3

      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Тест:", epoch, "Погрешность:", loss)


data = np.array([
  [-2, -1],
  [25, 6],
  [17, 4],
  [-15, -6],
])
all_y_trues = np.array([
  1,
  0,
  0,
  1,
])

network = NeuralNetwork()
network.train(data, all_y_trues)

print("Данная программа опрделит пол человека на основе данных о его весе и росте.")
first_man = np.array([int(input("Введите вес человека: ")) - 60, int(input("Введите рост человека: ")) - 165])
if network.feedforward(first_man) > 0.500:
    print("Человек является женщиной.")
elif network.feedforward(first_man) < 0.500:
    print("Человек является мужчиной.")

