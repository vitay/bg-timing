# -*- coding: utf-8 -*-
"""
Test the Conditioning phase
"""
from TimingNetwork import *
from TrialDefinition import *

# Compile the network
compile()

# Record result of learning
recorded_areas = [{'pop': BLA, 'var': 'rate', 'as_1D': True} ]

print 'Starting sensitization...'

# Definition of the individual US
sensitization_setup = [
    {'GUS': np.array([1., 1., 0., 0.]), 'duration': 500},
    {'GUS': np.array([1., 0., 1., 0.]), 'duration': 500},
    {'GUS': np.array([0., 0., 1., 1.]), 'duration': 500}
]

# Perform 10 sensitization trials per US
for trial in range(10):
    sensitization_trial(sensitization_setup)

# Stop learning in the LH->BLA pathway
LH_BLA.eta = 1000000.0     

print 'Starting conditioning...'

# Definition of the CS-US associations
conditioning_setup = [
    {'GUS': np.array([1., 1., 0., 0.]), 'VIS': np.array([1., 0., 0.]), 'magnitude': 0.8, 'duration': 2000},
    {'GUS': np.array([1., 0., 1., 0.]), 'VIS': np.array([0., 1., 0.]), 'magnitude': 0.5, 'duration': 3000},
    {'GUS': np.array([0., 0., 1., 1.]), 'VIS': np.array([0., 0., 1.]), 'magnitude': 1.0, 'duration': 4000}
]

# Record one trial before learning
record(recorded_areas)
conditioning_trial(conditioning_setup)
recordings = get_record(recorded_areas)
before_BLA = recordings['BLA']['rate']['data']

# Perform 10 conditioning trials per association
for trial in range(10):
    conditioning_trial(conditioning_setup)

# Record one trial
record(recorded_areas)
conditioning_trial(conditioning_setup)
recordings = get_record(recorded_areas)
after_BLA = recordings['BLA']['rate']['data']

print 'Analysing results...'

# Plot result of learning
import pylab as plt
plt.figure()
ax = plt.subplot2grid((2,2),(0, 0))
ax.imshow(before_BLA, aspect='auto', cmap=plt.cm.hot, interpolation='nearest')
ax.set_title('BLA before conditioning')
ax = plt.subplot2grid((2,2),(0, 1))
ax.plot(np.max(before_BLA, axis=0))
ax.set_title('BLA before conditioning')
ax = plt.subplot2grid((2,2),(1, 0))
ax.imshow(after_BLA, aspect='auto', cmap=plt.cm.hot, interpolation='nearest')
ax.set_title('BLA after conditioning')
ax = plt.subplot2grid((2,2),(1, 1))
ax.plot(np.max(after_BLA, axis=0))
ax.set_title('BLA after conditioning')
plt.show()
