import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.linspace(-2, 2, num=100)
    fun1 = 1/np.exp(x)
    fun2 = x/2
    fun3 = np.abs(x)
    plt.plot(x, fun1, label="SLA")
    plt.plot(x, fun2, label="Rent")
    plt.plot(x, fun3, label="Provision")
    plt.plot(x, fun2 + fun3, label="Sum")
    plt.legend()
    plt.show()
