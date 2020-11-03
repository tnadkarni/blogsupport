from scipy import stats
import numpy as np
from math import log


# This method computes entropy for information gain
def entropy(class_y):

    class_y = np.array(class_y)
    p1 = float(sum(class_y))/len(class_y)
    p0 = 1-p1
    try:
        h = -p1*log(p1,2) - p0*log(p0,2)
    except ValueError:
        h = 0
    return h

def partition_classes(rows, pred, split_point):
    #Partition the dataset by the split point for specified predictor
    
    part1 = rows[rows[:, pred] <= split_point]

    part2 = rows[rows[:, pred] > split_point]

    return [part1, part2]
    
def information_gain(previous_y, current_y):
    """Compute the information gain from partitioning the previous_classes
    into the current_classes using entropy function defined earlier.
    """
    new_ent = 0.0
    size  = len(previous_y)
    for i in range(len(current_y)):
        new_ent = new_ent + entropy(current_y[i])*len(current_y[i])/size
    return entropy(previous_y)-new_ent


