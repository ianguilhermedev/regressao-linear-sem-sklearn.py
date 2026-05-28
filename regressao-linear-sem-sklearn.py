import matplotlib.pyplot as plt


def main():
    x = [3.5, 3.69, 3.44, 3.43, 4.34, 4.42, 2.37]  # peso em 1000s
    y = [18, 15, 18, 16, 15, 14, 24] # milhas por galão

    w = 0
    b = 0

    learning_rate = 0.01
    epochs = 10000
    
    treinar_modelo_regressao_linear(x, y, b, w, epochs, learning_rate)



def treinar_modelo_regressao_linear(x, y, b, w, epochs, learning_rate):
    for iteracoes in range(epochs): # EPOCH
        predicts = calculo_predicts(x, b, w)

        #calcula o loss utilizando o mse
        mean_squared_error = calculo_loss(x, y, predicts)

        #DERIVADA PARCIAL DE DO PESO 
        parcial_w, parcial_b = gradiente_descendente(x, y, predicts)

        w, b = atualizando_parametros(w, b ,parcial_w, parcial_b, learning_rate)
    plot_grafico(x, y, predicts, mean_squared_error)

def calculo_predicts(x, b, w):
    list = []
    for feature in x:
        predict = b + feature*w
        list.append(predict)
    return list

def calculo_loss(x, y, predicts):
    soma_mse = 0
    for i in range(len(y)):
        soma_mse += (predicts[i] - y[i]) ** 2

    return soma_mse / len(x)
        
def gradiente_descendente(x, y, predicts):
    soma_parcial_w = 0
    for i in range(len(y)):
        soma_parcial_w += (predicts[i] - y[i]) * (2*x[i]) # calculo da parcial de x
    parcial_w = soma_parcial_w / len(x)

    soma_parcial_b = 0
    for i in range(len(y)):
        soma_parcial_b += (predicts[i] - y[i]) * 2 # calculo da parcial de x
    parcial_b = soma_parcial_b / len(x)

    return (parcial_w, parcial_b)

def atualizando_parametros(w, b, parcial_w, parcial_b, learning_rate):
    return(w - learning_rate*parcial_w, b - learning_rate*parcial_b)

def plot_grafico(eixox, eixoy, predicts, loss):
    plt.plot(eixox, eixoy, 'o')
    plt.grid(True)
    plt.xlabel('Peso em 1000s')
    plt.ylabel('Milhas Por Galão')
    plt.title(f'Peso x Milhas por Galão | Loss: {loss:.3f}')

    plt.plot(eixox, predicts)
    plt.legend(['Valor Real',
                'Valor Predictado'])
    plt.show()

main()

