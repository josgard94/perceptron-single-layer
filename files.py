#	Author: Edgard Díaz
#	date: 12 - 07 - 2020
#
#	This function upload and read the data set iris 
#
import pandas as pd #importar la libreria panda para  procesar el data set iris.
import numpy as np
class process_file:

	def load_file(self,path):
		
		#Read dataset
		iris_data = pd.read_csv(path, header = None);

		#get sepalo and petal size from columns 0 and 2.
		x = iris_data.iloc[0:100, [0,2]].values;
		
		y = iris_data.iloc[0:100, 4].values;
		
		y = np.where(y == 'Iris-versicolor', -1, 1) #get labels vector
		
		return x, y;
