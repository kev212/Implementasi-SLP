from csv import reader
import math

# Make a prediction with weights
def active(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return (1/(1+math.exp(-activation)))

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
 
filename = 'iris.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert string class to integers
str_column_to_int(dataset, len(dataset[0])-1)

weights = [0.5, 0.5, 0.5, 0.5, 0.5]
dweights = [0,0,0,0,0]
dbias = [0]
l_rate = 0.1

#test prediction
for row in dataset:
    activation = active(row, weights)
    if activation > 0.5:
        prediction = 1
    else:
        prediction = 0
    error = pow((row[-1]-activation),2)
    for i in range(len(dweights)-1):
        dweights[i+1] = -(row[-1]-activation)*activation*(1-activation)*row[i]
    dbias[0] = -(row[-1]-prediction)*prediction*(1-prediction)*1
    for i in range(len(weights)-1):
        weights[i+1] = weights[i+1] - l_rate*dweights[i+1]
    weights[0] = weights[0] - l_rate * dbias[0]

    print("Expected=%d, Predicted=%d, Error=%f" % (row[-1], prediction, error))

