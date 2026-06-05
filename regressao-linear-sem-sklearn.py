import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.w = 0
        self.b = 0
        self.loss = 0
        self.predicts = []

    @property
    def coef_(self):
        return self.w
    
    @property
    def intercept_(self):
        return self.b
    
    def fit(self, x:object, y:object, epochs=10000, learning_rate=0.01) -> None:
        for _ in range(epochs):
            predicts = [] # EPOCH
            for feature in x:
                predict = self.b + feature*self.w
                predicts.append(predict)

            #calcula o loss utilizando o mse
            soma_mse = 0
            for i in range(len(y)):
                soma_mse += (self.predicts[i] - y[i]) ** 2

            self.loss = soma_mse / len(x)

            #DERIVADA PARCIAL DE DO PESO 
            soma_parcial_w = 0
            for i in range(len(y)):
                soma_parcial_w += (self.predicts[i] - y[i]) * (2*x[i]) # calculo da parcial de x
            parcial_w = soma_parcial_w / len(x)

            soma_parcial_b = 0
            for i in range(len(y)):
                soma_parcial_b += (self.predicts[i] - y[i]) * 2 # calculo da parcial de x
            parcial_b = soma_parcial_b / len(x)

            self.w, self.b = self.w - learning_rate*parcial_w, self.b - learning_rate*parcial_b
        self.predicts = predicts

    def predict(self, feature_value):
        final_predicts = []
        for x in feature_value:
            predicts = self.b + self.w*x
            final_predicts.append(predicts)
        return final_predicts

    def plot_grafico(self, x,y):
        plt.plot(x, y, 'o')
        plt.grid(True)
        plt.title(f'weight: {self.coef_} | Intercept: {self.intercept_} | Loss: {self.loss:.2f}')
        plt.xlabel('Horas Estudadas Por Semana')
        plt.ylabel('Nota Final')
        plt.plot(x, self.predicts)   # linha da reta ajustada
        plt.legend(['Valores Reais', 'Previsão'])
        plt.show()

    
def main():
    x = [1,   1,   2,   2,   3,   3,   4,   4,   5,   5,   6,   6,
     7,   7,   8,   8,   9,   9,  10,  10,  11,  12,  12,  13,  14]
    # horas de estudo por semana

    y = [1.5, 2.0, 2.0, 2.5, 2.5, 3.5, 3.0, 4.0, 3.8, 4.5, 4.5, 5.2,
     5.0, 5.8, 5.5, 6.5, 6.0, 7.0, 6.8, 7.5, 7.2, 8.0, 8.5, 8.5, 9.0]
    # nota final    

    model = LinearRegression()
    model.fit(x, y)
    predicts = model.predict(x)
    loss = model.loss
    print(predicts)
    print(model.coef_)
    print(model.intercept_)

    model.plot_grafico(x, y)

main()
