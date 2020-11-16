#!/usr/bin/env python

import numpy as np
import math

# wrappers for generators
def Beta(a, b): return lambda: np.random.beta(a, b)
def Exp(s): return lambda: np.random.exponential(s)
def Poi(a): return lambda: np.random.poisson(a)
def Unif(a, b): return lambda: np.random.uniform(a, b)
def Choice(arr, p=None): return lambda: np.random.choice(arr, p=p)

class PAN(object):
    def __init__(self, m=1, fgen=lambda:1.0, capacity=1000):
        '''Preferential attachment network
        m: degree for each new node
        fgen: fitness distribution
        capacity(optional): hint for size
        history: the i-th entry is the list of vertices that the i-th guy connected to when it was created.
        edges: the i-th entry is the list of vertices that the i-th guy connected to when it was created.
        Note that the edges guy doesn't update automatically when we create new vertices.
        '''
        self.capacity = capacity
        self.m = m 
        self.degs = np.array([m], dtype='int32')        # degrees
        self.fgen = fgen
        self.fs = np.array([fgen()], dtype='float32')   # fitnesses
        self.history = [[0] * m]                          # edges w/ multiplicity
        # self.edges = []
        self.size = 1                                   # n nodes
        self.tot_edges = m                              # n edges
        self.weights = self.fs * self.degs
        self.tot_weights = self.weights[0]
        self._resize(capacity)

    def _resize(self, size):
        '''Pad arrays with zeros to avoid repetitive appending'''
        self.degs.resize(size)
        self.fs.resize(size)
        self.weights.resize(size)
        self.capacity = size

    def add_node(self):
        size = self.size
        m = self.m
        if size == self.capacity:
            self._resize(self.capacity * 2)
        endpoints = np.random.choice(range(size), m, replace=True, p=self.weights[:size]/self.tot_weights) # maybe easier if replace=False?
        self.history.append(endpoints)    # new edges
        new_fit = self.fgen()                   # new fitness
        self.fs[size] = new_fit
        self.degs[size] = m                     # deg for new node
        np.add.at(self.degs, endpoints, 1)      # update endpoints TODO: this isn't strictly correct. Juan: Wait why not, multi-edges?
                                                # I think it's fine if we have multi-edges right?
        self.weights[size] = m * new_fit
        endpoints_fit = self.fs[endpoints]
        np.add.at(self.weights, endpoints, endpoints_fit)
        self.tot_weights += m * new_fit + math.fsum(endpoints_fit)
        self.size += 1
        self.tot_edges += m
        # self.edges.append([])

    # def update_edges(self):
    #     """ We update self.edges to its correct values, using self.history."""
    #     size = self.size
    #     for i in range(size):
    #         for j in self.history[i]:
    #             self.edges[i].append(j)
    #             if i!=j:
    #                 self.edges[j].append(i)

    def grow_to_size(self, size):
        if size > self.capacity: self._resize(size)
        while (self.size < size):
            self.add_node()
        # self.update_edges()

    def catchup_time(self, i,j):
        """ returns the catch-up time for vertex i to catchup to vertex j, i.e. minimal time such that deg(i) >= deg(j)
        Assumptions: i>j, fitness(i) > fitness(j), and i<size. If these aren't true, we return None.
        Also assume that self.edges is updated. If it doesn't catchup on time, return False"""
        size = self.size
        if i<=j or i>=size: 
            return
        if self.fs[i] <= self.fs[j]: 
            return
        curr_time = j
        deg_j = m
        deg_i = 0

        while curr_time < size and deg_i < deg_j:
            curr_time += 1
            new_neighbours = np.array(self.history[curr_time]) # this bit of code is easier with replace=False 
            deg_j += np.count_nonzero(new_neighbours == j)
            deg_i += np.count_nonzero(new_neighbours == i)

        if deg_i >= deg_j: return curr_time 
        return False


if __name__ == '__main__':
    pan = PAN()
    pan.grow_to_size(1000)
    print(pan.degs)

