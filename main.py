#
#
#	Author: Edgard Diaz
#	date:	12-07-2020
#
#
#	data classification using a simple layer perceptron
#
#
#

from files import process_file
from perceptron import perceptron
from ploting import graph


if __name__ == '__main__':

	epacas = 10000; #iteraciones
	eta = 0.01;	#factor de aprendizaje
	#path dataset
	path = "iris.data"

	obj_plt = graph();
	obj_files = process_file();
	data, labels = obj_files.load_file(path);


	#initialize perceptron
	obj_perceptron = perceptron(eta, epacas)
	#training perceptron
	model = obj_perceptron.training(data,labels);

	#plot errors
	#obj_plt.plotting_errors(model);

	#plot decision regions
	obj_plt.plotting_decision_regions(data, labels, obj_perceptron);

