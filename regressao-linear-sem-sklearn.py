import matplotlib.pyplot as plt

def main():
    x = [3.5, 3.69, 3.44, 3.43, 4.34, 4.42, 2.37]  # peso em 1000s
    y = [18, 15, 18, 16, 15, 14, 24] # milhas por galão

    w = 0
    b = 0

    for iterações in range(10000): # EPOCH
        predict = []
        for feature in x:
            label = b + feature*w
            predict.append(label)

        #calcula o loss utilizando o mse
        soma_mse = 0
        for i in range(len(y)):
            soma_mse += pow((predict[i] - y[i]), 2)
        mean_squared_error = soma_mse / len(x)

        #DERIVADA PARCIAL DE DO PESO 
        soma_parcial_w = 0
        for i in range(len(y)):
            soma_parcial_w += (predict[i] - y[i]) * (2*x[i]) # calculo da parcial de x
        parcial_w = soma_parcial_w / len(x)

        #PARCIAL DO BIAS/INTERCEPT 
        soma_parcial_b = 0
        for i in range(len(y)):
            soma_parcial_b += (predict[i] - y[i]) * 2 # calculo da parcial de x
        parcial_b = soma_parcial_b / len(x)

        w, b = atualizando_parametros(w, b ,parcial_w, parcial_b)
    plot_grafico(x, y, predict, mean_squared_error)
        

def atualizando_parametros(w, b, parcial_w, parcial_b):
    learning_rate = 0.01
    return(w - learning_rate*parcial_w, b - learning_rate*parcial_b)

def plot_grafico(eixox, eixoy, predict, loss):
    plt.plot(eixox, eixoy, 'o')
    plt.grid(True)
    plt.xlabel('Peso em 1000s')
    plt.ylabel('Milhas Por Galão')
    plt.title(f'Peso x Milhas por Galão | Loss: {loss:.3f}')

    plt.plot(eixox, predict)
    plt.legend(['Valor Real',
                'Valor Predictado'])
    plt.show()

main()

