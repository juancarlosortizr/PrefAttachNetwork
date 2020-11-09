#!/usr/bin/env python

import numpy as np

# wrappers for generators
def Beta(a, b): return lambda: np.random.beta(a, b)
def Exp(s): return lambda: np.random.exponential(s)
def Poi(a): return lambda: np.random.poisson(a)
def Unif(a, b): return lambda: np.random.uniform(a, b)
def Choice(arr, p=None): return lambda: np.random.choice(arr, p=p)

class PAN(object):
    def __init__(self, m=1, fgen=lambda:1.0, capacity=10000):
        '''Preferential attachment network
        m: degree for each new node
        fgen: fitness distribution
        capacity(optional): hint for size
        '''
        self.capacity = capacity
        self.m = m
        self.degs = np.array([m], dtype='int32')        # degrees
        self.fgen = fgen
        self.fs = np.array([fgen()], dtype='float32')   # fitnesses
        # self.edges = [[0] * m]                          # edges w/ multiplicity TODO: don't think this is necessary, hard to optimize for performance reason
        self.size = 1                                   # n nodes
        self.tot_edges = m                              # n edges
        self._resize(capacity)

    def _resize(self, size):
        '''Pad arrays with zeros to avoid repetitive appending'''
        self.degs.resize(size)
        self.fs.resize(size)
        self.capacity = size

    def add_node(self):
        size = self.size
        m = self.m
        if size == self.capacity:
            self._resize(self.capacity * 2)
        weights = np.multiply(self.fs[:size], self.degs[:size])
        endpoints = np.random.choice(range(size), m, replace=True, p=weights/np.sum(weights))
        # self.edges.append(endpoints)    # new edges
        self.fs[size] = self.fgen()     # new fitness
        self.degs[size] = m             # deg for new node
        self.degs[endpoints] += 1       # update endpoints
        self.size += 1
        self.tot_edges += m

    def grow_to_size(self, size):
        if size > self.capacity: self._resize(size)
        while (self.size < size):
            self.add_node()

if __name__ == '__main__':
    pan = PAN()
    pan.grow_to_size(1000)
    print(pan.degs)

