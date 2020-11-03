from util import entropy, information_gain, partition_classes
from math import sqrt
import numpy as np


class DecisionTree(object):

    #create node class

    class Node:

      def __init__(self, pred = None, l = None, r = None, val = None, res = None):
        self.l = l
        self.r = r
        self.val = val
        self.pred = pred
        self.res = res

    #initialize tree to have a root node

    def __init__(self):
      self.tree = self.Node()

    def learn(self, X, y):
    #TODO: train decision tree and store it in self.tree

      X = np.array(X)

      y = np.array(y)

      y.shape = (y.shape[0], 1)

      #combine predictors and labels
      rows = np.append(X,y,1)

      self.tree = self.growTree(rows)

    def growTree(self, rows):
      # this recursive function builds the tree

      if len(rows) == 0:
     		return self.Node(res = np.array([0]))
        #if dataset is empty return tree with just root node set to 0 (failure)

      if len(set(rows[:,-1])) == 1:
        return self.Node(res = np.array([rows[0,-1]]))
        #if all class labels same, return root node set to that label

      n_predictors = len(rows[0])-1

      #randomly choose m = sqrt(n) predictors
      m_predictors =  np.random.choice(n_predictors, size=int(round(sqrt(n_predictors))))

      max_gain = 0.0
      best_pred = None
      best_sets = None

      #iterate over each of the m predictors and calculate information gain from splitting at every value for that predictor

      for pred in m_predictors:
        vals = sorted(set(rows[:,pred]))[:-1]
        prev_classes = rows[:,-1]
        for value in vals:
          part1, part2 = partition_classes(rows, pred, value)
          new_gain = information_gain(prev_classes, [part1[:,-1], part2[:,-1]])
          if new_gain > max_gain:
            max_gain = new_gain
            best_pred = [pred, value] #best predictor and value to split at
            best_sets = [part1, part2] #resulting partitions from best split

      if max_gain > 0.05: #build tree as long as there is significant information gain in partitioning
        lb = self.growTree(best_sets[0])
        rb = self.growTree(best_sets[1])

        return self.Node(pred = best_pred[0], val = best_pred[1], l = lb, r = rb)

      else:
        return self.Node(res = rows[:,-1]) #stop building tree and assign result to leaf node

    def classify(self, record):
      result = self.traverse(record, self.tree)
      return result

    def traverse(self,record, tree):
      if tree.res is not None: #leaf node
        return tree.res 
      else:
        v = record[tree.pred] 
        if v <= tree.val:
          br = tree.l
        else:
          br = tree.r
        return self.traverse(record, br) #recursive function, travelling down the tree

