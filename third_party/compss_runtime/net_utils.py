import numpy as np
from pyeddl.tensor import Tensor as eddlT

def net_getParameters(model):

	print("Entra a net getparameters")

	all_params = []

	i = 0
	for layer in model.layers:

		params = list()
		print("Layer", i)
		print("Size: ", len(layer.params))

		if len(layer.params) > 0:

			print("Weights shape: ", eddlT.getShape(layer.params[0]))

			weights = np.array(layer.params[0], copy=True).astype(np.float32)
			#weights = eddlT.getdata(layer.params[0])
			params.append(weights)

			#print("Weights: ", weights)


			if len(layer.params) > 1:

				print("Bias shape: ", eddlT.getShape(layer.params[1]))
				bias = np.array(layer.params[1], copy=True).astype(np.float32)
				#bias = eddlT.getdata(layer.params[1])
				params.append(bias)

				#print("Bias: ", bias)

		all_params.append(params)
		i += 1

	return all_params


def net_aggregateResults(recv_results):

	NUM_WORKERS = len(recv_results)
	final_results = [0.0, 0.0]

	for i in range(0, NUM_WORKERS):
		for j in range(0, len(final_results)):
			final_results[j] += recv_results[i][j]

	for i in range(0, len(final_results)):
		final_results[i] = final_results[i] / NUM_WORKERS

	return final_results
	

def net_insertParameters(model, final_weights):

	print("net_insertParameters")

	# DESC
	print("Los pesos que voy a enchufar en insertParameters: ", final_weights)

	# DESC
	#print("Teoricamente no enchufados en insertParameters: ", net_getParameters(model))

	var = list()
	var.append("Hola")

	#model.layers[0].params = var

	#print("Model insertParam: ", model.layers[0].params)

	# Params de una layer concreta son:
	#type(model.layers[i].params) == <class 'list'>
	# params[0] = Tensor()
	# params[1] = Tensor()

	print("Numero de layers: ", len(model.layers))

	for i in range(0, len(model.layers)):

		params_aux = list()
		print("Layer", i)
		print("Size: ", len(model.layers[i].params))

		if len(model.layers[i].params) > 0:

			v = eddlT(final_weights[i][0].astype(np.float32))
			v = eddlT.toCPU(v)

			print("Info weights que entra")
			v.info()
			print("Info weights que hay")
			model.layers[i].params[0].info()
			#print("Weights shape: ", eddlT.getShape(v))

			#layers[i].params[0] = v
			params_aux.append(v)

			if len(model.layers[i].params) > 1:

				v = eddlT(final_weights[i][1].astype(np.float32))
				v = eddlT.toCPU(v)

				print("Info bias que entra")
				v.info()
				print("Info bias que hay")
				model.layers[i].params[1].info()
				#print("Bias shape: ", eddlT.getShape(v))

				params_aux.append(v)
				#layers[i].params[1] = v
				#params_aux.append(eddlT.create(final_weights[i][1], eddlT.DEV_CPU))
				#model.layers[i].params[1] = eddlT.create(final_weights[i][1])

		#model.layers[i].params = params_aux

	#model.layers = layers
	# DESC
	print("Teoricamente enchufados en insertParameters: ", net_getParameters(model))
	return model


def net_aggregateParameters(workers_parameters):

	NUM_WORKERS = len(workers_parameters)
	recv_weights = workers_parameters

	final_weights = recv_weights[0]

	for i in range(0, len(final_weights)):

		layer_final = final_weights[i]

		if len(layer_final) > 0:
		
			for j in range(1, NUM_WORKERS):

				layer_recv = recv_weights[j][i]
				layer_final[0] = np.add(layer_final[0], layer_recv[0])

				if len(layer_final) > 1:
					layer_final[1] = np.add(layer_final[1], layer_recv[1])

			layer_final[0] = np.divide(layer_final[0], NUM_WORKERS)

			if len(layer_final) > 1:
				layer_final[1] = np.divide(layer_final[1], NUM_WORKERS)

			#print("Final weights[i]: ", final_weights[i])
			#print("Layer final", layer_final)

	return final_weights

def net_numpyParametersFill(parameters, value):

	for i in range(0, len(parameters)):
		for j in range(0, len(parameters[i])):
			parameters[i][j].fill(value)

	return parameters

def net_parametersToNumpy(parameters):

	np_params = list()

	for i in range(0, len(parameters)):

		params = list()

		for j in range(0, len(parameters[i])):
			
			#print("Voy ahora a pasar:", str(i), "|", str(j))
			#parameters[i][j].info()
			v = np.array(parameters[i][j], copy=False).astype(np.float32)
			#v = np.array(parameters[i][j], copy=True)
			#print("V: ", v)
			#v = eddlT.getdata(parameters[i][j])
			params.append(v)

		np_params.append(params)

	return np_params


def net_parametersToTensor(parameters):

	tensor_params = list()

	for i in range(0, len(parameters)):

		params = list()

		for j in range(0, len(parameters[i])):

			v = eddlT(parameters[i][j].astype(np.float32))
			#v.info()
			#v = eddlT.toCPU(v)
			params.append(v)

		tensor_params.append(params)

	return tensor_params

'''@task(serialized_model=IN, returns=int, is_replicated=True)
def build_prueba(serialized_model):

    if environ.get('EDDL_DIR') is not None:
        print("Inside Esta presente la varrr")
    else:
        print("Inside NO esta la var")
    return 0'''
    
def getFreqFromParameters(p):

	freq = dict()
	freq["-1.0"] = 0
	freq["-0.9"] = 0
	freq["-0.8"] = 0
	freq["-0.7"] = 0
	freq["-0.6"] = 0
	freq["-0.5"] = 0
	freq["-0.4"] = 0
	freq["-0.3"] = 0
	freq["-0.2"] = 0
	freq["-0.1"] = 0
	freq["-0.0"] = 0
	freq["0.0"] = 0
	freq["0.1"] = 0
	freq["0.2"] = 0
	freq["0.3"] = 0
	freq["0.4"] = 0
	freq["0.5"] = 0
	freq["0.6"] = 0
	freq["0.7"] = 0
	freq["0.8"] = 0
	freq["0.9"] = 0
	freq["1.0"] = 0

	# Layers
	for i in range(0, len(p)):

		# Weights j=0, params j=1
		for j in range(0, len(p[i])):

			recursiveTensorFreq(p[i][j], freq)

			'''for k in range(0, len(p[i][j])):

				value = p[i][j][k]

				#print("Que es esto: ", type(value))

				if isinstance(value, np.floating):
					
					print("Puede que aqui")
					value_short = "{:.1f}".format(value)
					
					if value_short not in freq:
						print("Algo pasa con1: ", value)

					freq[value_short] += 1

				else:

					for z in range(0, len(p[i][j][k])):

						value = p[i][j][k][z]
						print("Esto tiene pinta de tensor: ", value)
						print("O puede que aqui")
						value_short = "{:.1f}".format(value)

						if value_short not in freq:
							print("Algo pasa con2: ", value)

						freq[value_short] += 1'''

	return freq

def recursiveTensorFreq(tensor, freq):

	# Base case
	if isinstance(tensor, np.floating):
		
		value_short = "{:.1f}".format(tensor)

		if value_short not in freq:
			print("Algo pasa con1: ", tensor)

		freq[value_short] += 1

	# Recursive case
	else:

		for i in range(0, len(tensor)):
			recursiveTensorFreq(tensor[i], freq)

def individualParamsSave(p, name):

	with open(name, 'w') as writer:

		# Layers
		for i in range(0, len(p)):

			# Weights j=0, params j=1
			for j in range(0, len(p[i])):

				recursiveTensorSave(p[i][j], writer)

				'''for k in range(0, len(p[i][j])):

					value = p[i][j][k]

					if isinstance(value, np.floating):

						writer.write(str(value))
						writer.write("\n")
					else:

						for z in range(0, len(p[i][j][k])):

							value = p[i][j][k][z]

							writer.write(str(value))
							writer.write("\n")'''

def recursiveTensorSave(tensor, writer):

	# Base case
	if isinstance(tensor, np.floating):

		writer.write(str(tensor))
		writer.write("\n")

	# Recursive case
	else:
	
		for i in range(0, len(tensor)):
			recursiveTensorSave(tensor[i], writer)

def save_fnc(fichero, parameters):

    f = open(fichero, "a")

    for i in range(0, len(parameters)):

        for j in range(0, len(parameters[i])):

            v = parameters[i][j]
            np.savetxt(f, v)
