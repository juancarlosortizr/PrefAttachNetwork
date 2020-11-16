#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from PrefAttachNetwork import *

FINAL_SIZE = 1000
xyz=1
NUM_SAMPLES_CATCHUP = 1

plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
plt.rc('legend', fontsize=13)
plt.rc('axes', labelsize=14)
plt.rc('axes', titlesize=13)
plt.rc('figure', figsize=[8,6])

pan_plain = PAN(m=5)
pan_beta = PAN(m=5, fgen=Beta(1,10))
pan_plain.grow_to_size(FINAL_SIZE)
pan_beta.grow_to_size(FINAL_SIZE)

# Degree distribution for beta fitnesses vs flat fitness
plt.hist(np.log(pan_plain.degs), bins=50, alpha=0.8)#, normed=True)
plt.hist(np.log(pan_beta.degs), bins=50, alpha=0.8)#, normed=True)
plt.yscale('log')
plt.xlabel('log degree')
plt.legend(['Flat fitness', 'Beta(1,10) fitness'])
plt.show()

# Degree distribution within each fitness group
fitnesses = [1,2,3]
pan_discrete = PAN(fgen=Choice([1,2,3], p=[0.5, 0.3, 0.2]))
pan_discrete.grow_to_size(100000)
plt.hist([np.log(pan_discrete.degs[pan_discrete.fs==f]) for f in fitnesses], bins=20, label=['f='+str(f) for f in fitnesses])#, normed=True)
plt.yscale('log')
plt.xlabel('log degree')
plt.legend()
plt.show()

# Some catch-up times for the plain model
# Vertices i and j with i>j, fitness[i] > fitness[j], alpha = i/j, beta = fitness[i]/fitness[j], catch-up time t. 
# We expect t/i to be roughly equal to alpha^{1/(beta-1)}. (our "prediction")
# Independetly of the fitness distribution!
pan_plain_1 = PAN(m=1)
pan_beta_1 = PAN(m=1, fgen=Beta(1,10))
pan_plan_1.grow_to_size(FINAL_SIZE)
pan_beta_1.grow_to_size(FINAL_SIZE)
fractions_time_plain = []
predictions_plain = []
fractions_time_beta = []
predictions_beta = []
xyz=0

for _ in range(NUM_SAMPLES_CATCHUP):
	size = pan_plain_1.size()
	while True:
		[i,j] = np.random.choice(size, size=2, replace=False)
		catchup_time = pan_plain_1.catchup_time(i,j)
		if catchup_time != None and catchup_time != False:
			break
	predictions_plain.append(pow(i/j, pan_plain_1.fs[i]/pan_plain_1.fs[j]))
	fractions_time_plain.append(catchup_time/i)

	size = pan_beta_1.size()
	while True:
		[i,j] = np.random.choice(size, size=2, replace=False)
		catchup_time = pan_beta_1.catchup_time(i,j)
		if catchup_time != None and catchup_time != False:
			break
	predictions_beta.append(pow(i/j, pan_plain_1.fs[i]/pan_plain_1.fs[j]))
	fractions_time_beta.append(catchup_time/i)

plt.scatter(fractions_time_plain, predictions_plain)
plt.xlabel('Fractions t/t_A')
plt.ylabel('Predicted catch-up time')
plt.title('Catch-up times, plain distribution')
plt.legend()
plt.show()

plt.scatter(fractions_time_beta, predictions_beta)
plt.xlabel('Fractions t/t_A')
plt.ylabel('Predicted catch-up time')
plt.title('Catch-up times, Beta distribution')
plt.legend()
plt.show()