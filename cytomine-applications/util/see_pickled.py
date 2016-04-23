# -*- coding: utf-8 -*-

import pickle

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"

if __name__ == "__main__":
    model_path = "/home/mass/GRD/r.mormont/models/cell_model_better.pkl"
    with open(model_path, "rb") as model_file:
        classes = pickle.load(model_file)
        classifier = pickle.load(model_file)
        print repr(classifier)