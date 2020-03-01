import math
import csv
import random

# --------------------------------------------------------------------------

# Split a dataset into k folds
def Cross_validation_split(dataset, num_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / num_folds)
    for _ in range(num_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def Accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation spilt
def Evaluate_algorithm(dataset, algorithmFunc, num_folds, *args):
    folds = Cross_validation_split(dataset, num_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithmFunc(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = Accuracy_metric(actual, predicted)
        scores.append(accuracy)

    return scores

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

# Calculate the Euclidean distance between two vectors
# Assume that @vec1 and @vec2 have the same dimention
# Assume that the last element of both @vec1 and @vec2 are labels and not included into distance calculation
def Euclidean_distance(vec1, vec2):
    distance = 0.0
    for i in range(len(vec1)-1):
        distance += (vec1[i] - vec2[i])**2

    return math.sqrt(distance)

# Locate the most similar neighbors(@num_neighbors closest instances to @test_vec in @dataset)
def Get_neighbors(dataset, test_vec, num_neighbors):
    distances = list()
    for v in dataset:
        dist = Euclidean_distance(v, test_vec)
        distances.append((v, dist))
    distances.sort(key=lambda tup: tup[1])# Sort by distance
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    
    return neighbors

# Make a classification prediction with neighbors(@num_neighbors closest instances to @test_vec in @dataset)
def Predict_classification(dataset, test_vec, num_neighbors):
    neighbors = Get_neighbors(dataset, test_vec, num_neighbors)
    labelIndices = [neighbor[-1] for neighbor in neighbors]
    prediction = max(set(labelIndices), key=labelIndices.count)

    return prediction

# KNN algorithm
def K_nearest_neighbors(train_set, test_set, num_neighbors):
    predictions = list()
    for row in test_set:
        prediction = Predict_classification(train_set, row, num_neighbors)
        predictions.append(prediction)

    return predictions

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

# Load a csv file
def Load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            if not line:
                continue
            dataset.append(line)
    
    return dataset

# Convert string column to float
def Str_column_to_float(dataset, column):
    for line in dataset:
        line[column] = float(line[column].strip())

# Convert labels from string to integer
def Convert_labels_to_int(dataset, label_map):
    for line in dataset:
        line[-1] = label_map[line[-1]]

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

# Test the kNN on the Iris Flowers dataset
random.seed(1)
dataset = Load_csv("iris.csv")
label_map = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2}

# Deal with imported data
for i in range(len(dataset[0]) - 1):# Convert feature values from string to float
    Str_column_to_float(dataset, i)
Convert_labels_to_int(dataset,label_map)# Convert labels from string to integer

# Perform kNN
num_folds = 5
num_neighbors = 10
scores = Evaluate_algorithm(dataset, K_nearest_neighbors, num_folds, num_neighbors)

# Output validation results
print("Scores: %s" % scores)
print("Mean Accuracy: %.3f%%" % (sum(scores)/float(len(scores))))

# Predict
line = [5.7,2.9,4.2,1.3]
result = Predict_classification(dataset, line, num_neighbors)
print("Prediction: %d" % result)

# -------------------------------------------------------------------------