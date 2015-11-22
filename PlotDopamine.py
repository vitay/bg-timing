from ANNarchy import *
from TimingNetwork import *
from TrialDefinition import *

# Compile the network
compile()

# Record result of learning
monitor = Monitor(VTA, 'r', start=False)

print('Starting sensitization...')

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
LH_BLA.disable_learning()    

print('Starting conditioning...')

# Definition of the CS-US associations
conditioning_setup = [
    {'GUS': np.array([1., 1., 0., 0.]), 'VIS': np.array([1., 0., 0.]), 'magnitude': 0.8, 'duration': 2000},
    {'GUS': np.array([1., 0., 1., 0.]), 'VIS': np.array([0., 1., 0.]), 'magnitude': 0.5, 'duration': 3000},
    {'GUS': np.array([0., 0., 1., 1.]), 'VIS': np.array([0., 0., 1.]), 'magnitude': 1.0, 'duration': 4000}
]

# Record trial 1
monitor.start()
conditioning_trial(conditioning_setup)
recordings = monitor.get()
before_area = recordings['r']

# Perform 4 conditioning trials per association
monitor.pause()
for trial in range(4):
    conditioning_trial(conditioning_setup)

# Record trial 5
monitor.resume()
conditioning_trial(conditioning_setup)
recordings = monitor.get()
during_area = recordings['r']

# Perform 4 conditioning trials per association
monitor.pause()
for trial in range(4):
    conditioning_trial(conditioning_setup)

# Record trial 10
monitor.resume()
conditioning_trial(conditioning_setup)
recordings = monitor.get()
after_area = recordings['r']

print('Starting reward omission...')

# Record one omission trial
omission_trial(conditioning_setup)
recordings = monitor.get()
omit_area = recordings['r']

print('Analysing results...')

# Plot result of learning
import pylab as plt
plt.figure()
# Trial 1
ax = plt.subplot2grid((4,3),(0, 0))
ax.plot(before_area[:5000, 0])
ax.set_title('CS1 - US1')
ax.set_ylabel('Trial 1')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((4,3),(0, 1))
ax.plot(before_area[5000:11000, 0])
ax.set_title('CS2 - US2')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((4,3),(0, 2))
ax.plot(before_area[11000:18000, 0])
ax.set_ylim([0.0, 1.2])
ax.set_title('CS3 - US3')
# Trial 5
ax = plt.subplot2grid((4,3),(1, 0))
ax.plot(during_area[:5000, 0])
ax.set_ylabel('Trial 5')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((4,3),(1, 1))
ax.plot(during_area[5000:11000, 0])
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((4,3),(1, 2))
ax.plot(during_area[11000:18000, 0])
ax.set_ylim([0.0, 1.2])
# Trial 10
ax = plt.subplot2grid((4,3),(2, 0))
ax.plot(after_area[:5000, 0])
ax.set_ylabel('Trial 5')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((4,3),(2, 1))
ax.plot(after_area[5000:11000, 0])
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((4,3),(2, 2))
ax.plot(after_area[11000:18000, 0])
ax.set_ylim([0.0, 1.2])
# Omission
ax = plt.subplot2grid((4,3),(3, 0))
ax.plot(omit_area[:5000, 0])
ax.set_ylabel('Reward omission')
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((4,3),(3, 1))
ax.plot(omit_area[5000:11000, 0])
ax.set_ylim([0.0, 1.2])
ax = plt.subplot2grid((4,3),(3, 2))
ax.plot(omit_area[11000:18000, 0])
ax.set_ylim([0.0, 1.2])
plt.show()



