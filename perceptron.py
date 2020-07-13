import numpy as np

class perceptron:

    def __init__(self, eta, iterations):

        self.eta = eta;
        self.iterations = iterations;

    #función para realizar el entrenamiento del perceptro.
    def  training(self, data, labels):
    	self.tetha = np.zeros(1 + data.shape[1])
    	self.errors_ = []

    	for _ in range(self.iterations):
    		errors = 0;

    		for xi, target in zip(data,labels):

    			update = self.eta * (target - self.predict(xi))
    			self.tetha[1:] += update * xi
    			self.tetha[0] += update
    			errors += int(update != 0.0)

    		self.errors_.append(errors)

    	return self


    #función de activacion
    def predict(self, data):
    	phi = np.where(self.calculation_valor_z(data) >= 0.0, 1, -1)
    	return phi

    def calculation_valor_z(self, data):

    	z = np.dot(data, self.tetha[1:] + self.tetha[0])

    	return z
