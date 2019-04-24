#!/usr/bin/python
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd
#
# Submission: Zayd Hammoudeh
#
#
from __future__ import division
import sys
import re
import math
# Node class for the decision tree
from collections import Counter

import node


train = None
varnames = None
namehash = None
test = None
testvarnames = None
root = None


# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
    # >>>> YOUR CODE GOES HERE <<<<
    # For now, always return "0":
    if p == 0. or p == 1.0: return 0.
    return -p * math.log(p, 2.) - (1 - p) * math.log(1 - p, 2)


# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
    # >>>> YOUR CODE GOES HERE <<<<
    # For now, always return "0":
    h0 = entropy(py / total)

    # Sometimes the test cases are dumb and give ints
    if isinstance(pxi, int): pxi = [pxi, total - pxi]
    if isinstance(py_pxi, int): py_pxi = [py_pxi, py - py_pxi]

    h1 = 0.
    for y_i, n_i in zip(py_pxi, pxi):
        if n_i == 0: continue
        h1 += n_i / total * entropy(y_i / n_i)
    return h0 - h1


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return data, varnames


# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    with open(modelfile, 'w+') as f:
        root.write(f, 0)


# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames, rem_vars=None):
    # >>>> YOUR CODE GOES HERE <<<<
    if rem_vars is None: rem_vars = list(range(len(varnames) - 1))
    # Check for homogeneous leaf
    y_0 = data[0][-1]
    if all([y_0 == x[-1] for x in data]):
        return node.Leaf(varnames, y_0)
    # No variables remaining
    if not rem_vars:
        _build_most_common_val_leaf(varnames, data)

    # Create a split node
    total, py = len(data), sum(1 for x in data if x[-1] == 1)
    best_ig, best_var, best_data_x = 0., None, None
    for var_id in rem_vars:
        data_x =[[x for x in data if x[var_id] == i] for i in range(2)]
        py_pxi = [sum(1 for x in data_x[i] if x[-1] == i) for i in range(len(data_x))]
        pxi = [len(grp) for grp in data_x]
        if any(l == 0 for l in pxi): continue

        ig = infogain(py_pxi, pxi, py, total)

        if ig > best_ig: best_ig, best_var, best_data_x = ig, var_id, data_x

    # No benefit to more splitting so no information left to gain
    if best_ig == 0.: return _build_most_common_val_leaf(varnames, data)
    # Split and create child trees
    rem_vars = [var_id for var_id in rem_vars if var_id != best_var]
    return node.Split(varnames, best_var,
                      build_tree(best_data_x[0], varnames, rem_vars),
                      build_tree(best_data_x[1], varnames, rem_vars))


def _build_most_common_val_leaf(names, data):
    cntr = Counter([x[-1] for x in data])
    return node.Leaf(names, cntr.most_common(1)[0][0])


# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS,testS,modelS):
    global train
    global varnames
    global test
    global testvarnames
    global root
    (train, varnames) = read_data(trainS)
    (test, testvarnames) = read_data(testS)
    modelfile = modelS

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)


def runTest():
    return sum(1 for x in test if x[-1] == root.classify(x)) / len(test)


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if len(argv) != 3:
        print 'Usage: id3.py <train> <test> <model>'
        sys.exit(2)
    loadAndTrain(argv[0], argv[1], argv[2])

    acc = runTest()
    print "Accuracy: ", acc


if __name__ == "__main__":
    main(sys.argv[1:])
