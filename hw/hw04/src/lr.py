#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import sys
import re
from math import log
from math import exp
from math import sqrt

MAX_ITERS = 100


def dot(a, b):
    r""" Perform dot product operation between vectors a and b """
    return sum(a_i * b_i for (a_i, b_i) in zip(a, b))


def add(a, b, sign):
    r""" Perform element-wise addition in form a_i + sign * b_i """
    return [a_i + sign * b_i for (a_i, b_i) in zip(a, b)]


def sigmoid(x):
    try:
        return 1. / (1. + exp(-x))
    except OverflowError:
        return 1.


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    # namehash = {}
    for l in f:
        example = [int(x) for x in p.split(l.strip())]
        x = example[0:-1]
        y = example[-1]
        data.append((x, y))
    return data, varnames


# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
    numvars = len(data[0][0])
    w = [0.0] * numvars
    b = 0.0

    #
    # YOUR CODE HERE
    #
    for _ in range(MAX_ITERS):
        num_wrong = 0
        for (x, y) in data:
            g_base = y * sigmoid(-y * (dot(w, x) + b))
            w_t = add(w, x, eta * g_base)
            w_t = add(w_t, w, -eta * l2_reg_weight)
            w, b = w_t, b + g_base
        if num_wrong == 0:
            break
    return w, b


# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
    (w, b) = model

    #
    # YOUR CODE HERE
    #
    y = dot(w, x) + b
    return sigmoid(y)


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if len(argv) != 5:
        print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    eta = float(argv[2])
    lam = float(argv[3])
    modelfile = argv[4]

    # Train model
    (w, b) = train_lr(train, eta, lam)

    # Write model file
    f = open(modelfile, "w+")
    f.write('%f\n' % b)
    for i in range(len(w)):
        f.write('%s %f\n' % (varnames[i], w[i]))

    # Make predictions, compute accuracy
    correct = 0
    for (x, y) in test:
        prob = predict_lr((w, b), x)
        print(prob)
        if (prob - 0.5) * y > 0:
            correct += 1
    acc = float(correct) / len(test)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
