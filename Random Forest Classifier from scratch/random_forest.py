from decision_tree import DecisionTree
import csv
import numpy as np  
import ast

"""
Here, X is assumed to be a matrix with n rows and d columns where n is the
number of total records and d is the number of features of each record. Also,
y is assumed to be a vector of n labels

XX is similar to X, except that XX also contains the data label for each
record.
"""


class RandomForest(object):
    num_trees = 0
    decision_trees = []

    # the bootstrapping dataset for trees
    bootstraps_datasets = []

    # the true class labels, corresponding to records in the bootstrapping
    # dataset
    bootstraps_labels = []

    def __init__(self, num_trees):
        # TODO: do initialization here.
        self.num_trees = num_trees
        self.decision_trees = [DecisionTree() for i in range(num_trees)]

    def _bootstrapping(self, XX, n):
        XX = np.array(XX)
        ind = np.random.randint(XX.shape[0], size=n)
        return XX[ind, :-1], XX[ind, -1]

    def bootstrapping(self, XX):
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)

    def fitting(self):
        for i in range(self.num_trees):
            self.decision_trees[i].learn(self.bootstraps_datasets[i], self.bootstraps_labels[i])

    def voting(self, X):
        y = np.array([], dtype = int)

        for record in X:
            #find the sets of proper trees that consider the record as an out-of-bag sample, and predict the label(class) for the record.
    		#The majority vote serves as the final label for this record.
            votes = []
            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i]
                if record.tolist() not in dataset[:,:-1]:
                    OOB_tree = self.decision_trees[i]
                    effective_vote = OOB_tree.classify(record)
                    votes.append(effective_vote)
            counts = np.bincount(votes[0])
            if len(counts) == 0:
                y = np.append(y,0)
            else:
                y = np.append(y, np.argmax(counts))

        return y


def main():
    X = list()
    y = list()
    XX = list()  # Contains data features and data labels

    # Load data set
    print 'reading hw4-data'
    with open("data.csv") as f:
        next(f, None)

        for line in csv.reader(f, delimiter=","):
            X.append(line[:-1])
            y.append(line[-1])
            xline = [ast.literal_eval(i) for i in line]
            XX.append(xline[:])

    # Initialize according to your implementation
    forest_size = 10

    # Initialize a random forest
    randomForest = RandomForest(forest_size)

    # Create the bootstrapping datasets
    print 'creating the bootstrap datasets'
    randomForest.bootstrapping(XX)

    # Build trees in the forest
    print 'fitting the forest'
    randomForest.fitting()

    # Provide an unbiased error estimation of the random forest based on out-of-bag (OOB) error estimate.
    # Note that you may need to handle the special case in which every single record in X has used for training by some of the trees in the forest.
    y_truth = np.array(y, dtype=int)
    X = np.array(X, dtype=float)
    y_predicted = randomForest.voting(X)

    #results = [prediction == truth for prediction, truth in zip(y_predicted, y_test)]
    results = [prediction == truth for prediction, truth in zip(y_predicted, y_truth)]

    # Accuracy
    accuracy = float(results.count(True)) / float(len(results))
    oob = 1-accuracy
    print "accuracy: %.4f" % accuracy
    print "OOB error estimate: %.4f" % oob


if __name__ == "__main__":
    main()
