# -*- coding: utf-8 -*-
"""
Test the sensitization phase
"""
from TimingNetwork import *
from TrialDefinition import *

# Compile the network
compile()

# Perform 10 sensitization trials per US
trial_setup = [
    {'GUS': np.array([1., 1., 0., 0.]), 'duration': 500},
    {'GUS': np.array([1., 0., 1., 0.]), 'duration': 500},
    {'GUS': np.array([0., 0., 1., 1.]), 'duration': 500}
]
for trial in range(10):
    sensitization_trial(trial_setup)

# Record result of learning
recorded_areas = [{'pop': BLA, 'var': 'rate', 'as_1D': True},
                  {'pop': CE, 'var': 'rate'},
                  {'pop': PPTN, 'var': 'rate'},
                  {'pop': VTA, 'var': 'rate'} ]
                  
record(recorded_areas)
sensitization_trial(trial_setup)
recordings = get_record(recorded_areas)
recorded_BLA = recordings['BLA']['rate']['data']
recorded_PPTN = recordings['PPTN']['rate']['data']
recorded_CE = recordings['CE']['rate']['data']
recorded_VTA = recordings['VTA']['rate']['data']


# Plot result of learning
import pylab as plt
plt.figure()
ax = plt.subplot2grid((2,2),(0, 0))
ax.imshow(recorded_BLA, aspect='auto', cmap=plt.cm.hot)
ax.set_title('BLA')
ax = plt.subplot2grid((2,2),(0, 1))
ax.plot(recorded_PPTN[0])
ax.set_title('PPTN')
ax = plt.subplot2grid((2,2),(1,0))
ax.plot(recorded_CE[0])
ax.set_title('CE')
ax = plt.subplot2grid((2,2),(1, 1))
ax.plot(recorded_VTA[0])
ax.set_title('VTA')
plt.show()
