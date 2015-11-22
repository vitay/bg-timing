# -*- coding: utf-8 -*-
# Timing and expectation of reward: a neuro-computational model of the afferents to the ventral tegmental area
# Vitay, J. and Hamker, F. (2014) *Frontiers in Neurorobotics* 8(4), doi: 10.3389/fnbot.2014.00004   

from TimingNetwork import *
from TrialDefinition import *

# Compile the network
compile()

# Before running the simulation, we have to define the populations which will be recorded:
monitors = {
    VTA: Monitor(VTA, 'r', start=False),
    BLA: Monitor(BLA, 'r', start=False),
    CE: Monitor(CE, 'r', start=False),
    NAcc: Monitor(NAcc, ['r', 'mp', 'g_mod', 's'], start=False),
    PPTN_US: Monitor(PPTN_US, 'r', start=False),
    PPTN_CS: Monitor(PPTN_CS, 'r', start=False),
    VP: Monitor(VP, 'r', start=False),
    LHb: Monitor(LHb, 'r', start=False),
    RMTg: Monitor(RMTg, 'r', start=False),
    LH: Monitor(LH, 'r', start=False),
    VIS: Monitor(VIS, 'r', start=False) 
}

def start_record(monitors):
    for monitor in monitors.values():
        monitor.start()

def get_record(monitors):
    data = {}
    for name, monitor in monitors.items():
        data[name] = monitor.get()
    return data

# 1.2 Sensitization phase
# In this first phase, we present the three US alone to build the LH -> BLA connections. Learning in this pathway is disabled at the end.

# Definition of the individual US
sensitization_setup = [
    {'GUS': np.array([1., 1., 0., 0.]), 'duration': 500.0},
    {'GUS': np.array([1., 0., 1., 0.]), 'duration': 500.0},
    {'GUS': np.array([0., 0., 1., 1.]), 'duration': 500.0}
]

# Perform 10 sensitization trials per US
print('Sensitization phase')
for trial in range(10):
    sensitization_trial(sensitization_setup)

# Stop learning in the LH->BLA pathway
LH_BLA.disable_learning() 

# 1.3 Conditioning phase
# Once the US are known by the system. we can start the conditioning phase with the three CS defined in the paper.

# Definition of the CS-US associations
conditioning_setup = [
    {'GUS': np.array([1., 1., 0., 0.]), 'VIS': np.array([1., 0., 0.]), 'magnitude': 0.8, 'duration': 2000.0},
    {'GUS': np.array([1., 0., 1., 0.]), 'VIS': np.array([0., 1., 0.]), 'magnitude': 0.5, 'duration': 3000.0},
    {'GUS': np.array([0., 0., 1., 1.]), 'VIS': np.array([0., 0., 1.]), 'magnitude': 1.0, 'duration': 4000.0}
]

print('Conditioning phase')
recordings = []
for trial in range(10):
    start_record(monitors) # Tell the network which population to record
    conditioning_trial(conditioning_setup) # Perform one conditioning trial
    recordings.append(get_record(monitors)) # Save the recordings

# 1.4 Omission trials
# Once conditioning is complete, we omit reward to observe the response of DA cells:

print('Omission trial')
start_record(monitors)
omission_trial(conditioning_setup)
recordings.append(get_record(monitors))

# 1.5 Early trials
# Last, we deliver reward 1s earlier than expected:

print('Earlier trial')
start_record(monitors)
earlier_trial(conditioning_setup)
recordings.append(get_record(monitors))

# 2. Results

import pylab as plt

# 2.1 Conditioning in the amygdala (Fig. 4)
# We extract the maximal firing rate in BLA during the first and last conditioning trials for the three stimuli.

BLA_trial1 = np.max(recordings[0][BLA]['r'], axis=1)
BLA_trial10 = np.max(recordings[9][BLA]['r'], axis=1)


plt.figure(figsize=(10,6))
# Trial 1
ax = plt.subplot2grid((2,3),(0, 0))
ax.plot(BLA_trial1[:5000])
ax.set_title('CS1 - US1')
ax.set_ylabel('Trial 1')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((2,3),(0, 1))
ax.plot(BLA_trial1[5000:11000])
ax.set_title('CS2 - US2')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((2,3),(0, 2))
ax.plot(BLA_trial1[11000:18000])
ax.set_ylim([0.0, 1.2])
ax.set_title('CS3 - US3')
# Trial 10
ax = plt.subplot2grid((2,3),(1, 0))
ax.plot(BLA_trial10[:5000])
ax.set_ylabel('Trial 10')
ax.set_xlabel('Time (ms)')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((2,3),(1, 1))
ax.plot(BLA_trial10[5000:11000])
ax.set_xlabel('Time (ms)')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((2,3),(1, 2))
ax.plot(BLA_trial10[11000:18000])
ax.set_xlabel('Time (ms)')
ax.set_ylim([0.0, 1.2])
plt.show()

# 2.2 Timecourse of the activity of the VTA cell during conditioning (Fig. 5)

VTA_trial1 = recordings[0][VTA]['r'][:, 0]
VTA_trial5 = recordings[4][VTA]['r'][:, 0]
VTA_trial10 = recordings[9][VTA]['r'][:, 0]
VTA_omit = recordings[10][VTA]['r'][:, 0]
VTA_sooner = recordings[11][VTA]['r'][:, 0]

plt.figure(figsize=(12,10))
# Trial 1
ax = plt.subplot2grid((5,3),(0, 0))
ax.plot(VTA_trial1[:5000])
ax.set_title('CS1 - US1')
ax.set_ylabel('Trial 1')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((5,3),(0, 1))
ax.plot(VTA_trial1[5000:11000])
ax.set_title('CS2 - US2')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((5,3),(0, 2))
ax.plot(VTA_trial1[11000:18000])
ax.set_ylim([0.0, 1.2])
ax.set_title('CS3 - US3')
# Trial 5
ax = plt.subplot2grid((5,3),(1, 0))
ax.plot(VTA_trial5[:5000])
ax.set_ylabel('Trial 5')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((5,3),(1, 1))
ax.plot(VTA_trial5[5000:11000])
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((5,3),(1, 2))
ax.plot(VTA_trial5[11000:18000])
ax.set_ylim([0.0, 1.2])
# Trial 10
ax = plt.subplot2grid((5,3),(2, 0))
ax.plot(VTA_trial10[:5000])
ax.set_ylabel('Trial 5')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((5,3),(2, 1))
ax.plot(VTA_trial10[5000:11000])
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((5,3),(2, 2))
ax.plot(VTA_trial10[11000:18000])
ax.set_ylim([0.0, 1.2])
# Omission
ax = plt.subplot2grid((5,3),(3, 0))
ax.plot(VTA_omit[:5000])
ax.set_ylabel('Reward omission')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((5,3),(3, 1))
ax.plot(VTA_omit[5000:11000])
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((5,3),(3, 2))
ax.plot(VTA_omit[11000:18000])
ax.set_ylim([0.0, 1.2])
# Earlier
ax = plt.subplot2grid((5,3),(4, 0))
ax.plot(VTA_sooner[:5000])
ax.set_ylabel('Earlier delivery')
ax.set_ylim([0.0, 1.2])
ax.set_xlabel('Time (ms)')
ax = plt.subplot2grid((5,3),(4, 1))
ax.plot(VTA_sooner[5000:11000])
ax.set_ylim([0.0, 1.2])
ax.set_xlabel('Time (ms)')
ax = plt.subplot2grid((5,3),(4, 2))
ax.plot(VTA_sooner[11000:18000])
ax.set_ylim([0.0, 1.2])
ax.set_xlabel('Time (ms)')
plt.show()


# 2.3 Evolution of the maximal activity in VTA during conditioning (Fig. 6)
# For this figure, we need to extract the maximal activity in VTA during each conditioning trial around CS and US onset:

CS1 = slice(900,1100)
US1 = slice(2900, 3100)
CS2 = slice(5900, 6100)
US2 = slice(8900, 9100)
CS3 = slice(11900, 12100)
US3 = slice(15900, 16100)

VTA_CS1 = []; VTA_CS2 = []; VTA_CS3 = []; VTA_US1 = []; VTA_US2 = []; VTA_US3 = []
for trial in range(10):
    VTA_CS1.append(np.max( recordings[trial][VTA]['r'][CS1, 0]))
    VTA_CS2.append(np.max( recordings[trial][VTA]['r'][CS2, 0]))
    VTA_CS3.append(np.max( recordings[trial][VTA]['r'][CS3, 0]))
    VTA_US1.append(np.max( recordings[trial][VTA]['r'][US1, 0]))
    VTA_US2.append(np.max( recordings[trial][VTA]['r'][US2, 0]))
    VTA_US3.append(np.max( recordings[trial][VTA]['r'][US3, 0]))

plt.figure(figsize=(12,4))
# CS1 - US1 
ax = plt.subplot2grid((1,3),(0, 0))
ax.plot(VTA_CS1)
ax.plot(VTA_US1)
ax.set_xlabel('# Trials')
ax.set_ylabel('Burst amplitude')
ax.set_title('CS1 - US1')
ax.set_ylim([0.2, 1.1])
# CS2 - US2 
ax = plt.subplot2grid((1,3),(0, 1))
ax.plot(VTA_CS2)
ax.plot(VTA_US2)
ax.set_title('CS2 - US2')
ax.set_xlabel('# Trials')
ax.set_ylim([0.2, 1.1])
# CS3 - US3 
ax = plt.subplot2grid((1,3),(0, 2))
ax.plot(VTA_CS3)
ax.plot(VTA_US3)
ax.set_title('CS3 - US3')
ax.set_xlabel('# Trials')
ax.set_ylim([0.2, 1.1])

plt.show()

# 2.4 Timecourse of the internal variables of a single NAcc neuron during a reward omission trial (Fig. 8)
# For this figure, we record different internal variables of the cell in NAcc maximally responding to the CS1-US1 interval in the reward omission condition. 

NAcc_rates = recordings[10][NAcc]['r']
active_cell = 0; max_rate = 0.0
for cell in range(NAcc.size):
    if np.max(NAcc_rates[:5000, cell]) > max_rate:
        max_rate = np.max(NAcc_rates[:5000, cell])
        active_cell = cell
NAcc_mp = recordings[10][NAcc]['mp'][:5000, active_cell]
NAcc_s = recordings[10][NAcc]['s'][:5000, active_cell]*0.5 - 0.8
NAcc_vmpfc = recordings[10][NAcc]['g_mod'][:5000, active_cell]

plt.figure(figsize=(10,8))
plt.plot(NAcc_mp, label='membrane potential')
plt.plot(NAcc_s, label = 'up/down-state')
plt.plot(NAcc_vmpfc, label='cortical input')
plt.ylim([-2.0, 1.5])
plt.xlabel('Time (ms)')
plt.legend(loc=2, frameon=False)
plt.show()

# 2.5 Timecourse of activity in different areas of the model (Fig. 10)
# This figures shows the timecourse activity of different populations during the last conditioning trial and the first omission trial for CS1-US1.

plt.figure(figsize=(16,16))
# Conditioning trial
ax = plt.subplot2grid((8,2),(0, 0))
ax.plot(recordings[9][VIS]['r'][:5000, 0]) # TODO: inputs
ax.set_title('Conditioning trial')
ax.set_ylabel('Inputs')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((8,2),(1, 0))
ax.plot(recordings[9][VTA]['r'][:5000, 0])
ax.set_ylabel('VTA')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((8,2),(2, 0))
ax.plot(recordings[9][CE]['r'][:5000, 0])
ax.set_ylabel('CE')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((8,2),(3, 0))
ax.plot(recordings[9][PPTN_US]['r'][:5000, 0])
ax.plot(recordings[9][PPTN_CS]['r'][:5000, 0])
ax.set_ylabel('PPTN')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((8,2),(4, 0))
ax.plot(recordings[9][NAcc]['r'][:5000, active_cell])
ax.set_ylabel('NAcc')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((8,2),(5, 0))
ax.plot(recordings[9][VP]['r'][:5000, 0])
ax.set_ylabel('VP')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((8,2),(6, 0))
ax.plot(recordings[9][LHb]['r'][:5000, 0])
ax.set_ylabel('LHb')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((8,2),(7, 0))
ax.plot(recordings[9][RMTg]['r'][:5000, 0])
ax.set_ylabel('RMTg')
ax.set_ylim([0.0, 1.2])
ax.set_xlabel('Time (ms)')
# Omission trial
ax = plt.subplot2grid((8,2),(0, 1))
ax.plot(recordings[10][VIS]['r'][:5000, 0]) # TODO: inputs
ax.set_title('Omission trial')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((8,2),(1, 1))
ax.plot(recordings[10][VTA]['r'][:5000, 0])
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((8,2),(2, 1))
ax.plot(recordings[10][CE]['r'][:5000, 0])
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((8,2),(3, 1))
ax.plot(recordings[10][PPTN_US]['r'][:5000, 0])
ax.plot(recordings[10][PPTN_CS]['r'][:5000, 0])
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((8,2),(4, 1))
ax.plot(recordings[10][NAcc]['r'][:5000, active_cell])
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((8,2),(5, 1))
ax.plot(recordings[10][VP]['r'][:5000, 0])
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((8,2),(6, 1))
ax.plot(recordings[10][LHb]['r'][:5000, 0])
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((8,2),(7, 1))
ax.plot(recordings[10][RMTg]['r'][:5000, 0])
ax.set_ylim([0.0, 1.2])
ax.set_xlabel('Time (ms)')

plt.show()


