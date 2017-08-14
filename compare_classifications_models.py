# -*- coding: utf-8 -*-
"""
@author: C
"""

from sklearn.datasets import load_iris

class model():
    def __init__(self):
        self.data_set = load_iris()
        print(self.data_set)
        
if __name__ == '__main__':
    ai = model()