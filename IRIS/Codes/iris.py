import numpy as np
import random 
import math

def read_file(filename):
	data=[]
	with open(filename,'r') as file:
		for line in file:
			temp = line.split(",")
			data.append(temp)

	for i in range(len(data)):
		if(i!=len(data)-1):
			data[i][-1]=data[i][-1][:-1]
	return data		

def convert_to_float(data):
	for row in data:
		for j in range(len(row)-1):
			row[j]=float(row[j].strip())
	return data		

def convert_to_int(data):
	lis =[row[-1] for row in data]
	unique = set(lis)
	lookup = dict()
	for i,val in enumerate(unique):
		lookup[val]=i
	for row in data:
		row[-1]=lookup[row[-1]]
	return data					

def data_filter(data):
	dataset=convert_to_float(data)
	dataset=convert_to_int(dataset)
	return dataset

def min_max(dataset):
	minmax=[[min(column),max(column)] for column in zip(*dataset)]
	return minmax

def normalization(data): 
	minmax = min_max(data)
	for row in data:
		for i in range(len(row)-1):
			row[i]=(row[i]-minmax[i][0])/(minmax[i][1]-minmax[i][0])
	return data		

def initialize_network(no_inputs,no_hidden,no_output):
	weight = list()
	inp = [{'weights':[(random.randint(1,100)/100) for i in range(no_inputs+1)]} for i in range(no_hidden)]
	weight.append(inp)
	hidden = [{'weights':[(random.randint(1,100)/100) for i in range(no_hidden+1)]} for i in range(no_output)]
	weight.append(hidden)
	return weight

def cross_validation(data):
	temp=np.array(data)
	np.random.shuffle(temp)
	data = []
	for i in temp:
		i = list(i)
		data.append(i)
	training_length = int(0.7*len(data))	
	training_data = data[:training_length]
	testing_data = data[training_length:]
	return training_data,testing_data	

def activation(weight,inpute):
	total = weight[-1]
	for i in range(len(inpute)-1):
		total = total + weight[i]*inpute[i]
	return total	

def transfer_func(x):
	return 1/(1+math.exp(-x))	

def feed_forward(network,row):
	inpute = row
	for layer in network:
		new_input=list()
		for neuron in layer:
			val = activation(neuron['weights'],inpute)
			val = transfer_func(val)
			neuron['output'] = val
			new_input.append(val)
		inpute = new_input
	return inpute

def transfer_derivative(output):
	return output * (1.0 - output)
 
# Backpropagate error and store in neurons
def backpropagation(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, row):
	l_rate =0.4
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']				 

def train(network,training_data,no_of_generations,no_of_output):
	
	for i in range(0,no_of_generations):
			for row in training_data:
				result = feed_forward(network,row)
				expected = [0 for i in range(no_of_output)]
				expected[int(row[-1])]=1
				backpropagation(network,expected)
				update_weights(network,row)

def accuracy_testing(network,testing_data):
	expected = [row[-1] for row in testing_data]
	predicted = []
	for row in testing_data:
		val = feed_forward(network,row)
		ind = val.index(max(val))
		predicted.append(ind)
	count=0	
	for i in range(len(testing_data)):
		if(expected[i] == predicted[i]):
			count+=1
	return(count/len(testing_data))			

def algorithm(data,no_inputs,no_hidden,no_output,no_of_generations,no_of_tests):
	network = initialize_network(no_inputs,no_hidden,no_output)
	
	for i in range(no_of_tests):
		training_data,testing_data = cross_validation(data)
		train(network,training_data,no_of_generations,no_output)
		accuracy = accuracy_testing(network,testing_data)
		print(accuracy)

if __name__=="__main__":

	filename = "Iris_dataset1.csv"
	print(filename)
	data = read_file(filename)

	data = data_filter(data)
	data = normalization(data)
	no_inputs = len(data[0]) - 1 
	no_hidden = 5
	no_output = 3
	no_of_generations = 300
	no_of_tests = 5
	accuracy = algorithm(data,no_inputs,no_hidden,no_output,no_of_generations,no_of_tests)