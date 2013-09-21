################################################################
### Script showing the activity of VTA cells during conditioning
### Figure should be similar to Schultz 1998
################################################################

from TimingNetwork import *

# Create the network
net = TimingNetwork()
net.build()

recorded_areas = ['VTA', 
                  {'population':'NAcc', 'variables': ['rate', 'vmpfc', 'bla']}, 
                  {'population':'NAcc_pred', 'variables': ['rate', 'vmpfc', 'bla']}, 
                  'BLA']

# Habituate the network to the gustatory inputs
for trial in range(10):
    valuation_trial(net, 1) # US1
    valuation_trial(net, 2) # US2
    
# Record VTA before conditioning
net.record(recorded_areas) 
conditioning_trial(net, 1) # CS1, US1
before_CS1 = net.get_recordings()
conditioning_trial(net, 2) # CS2, US2
before_CS2 = net.get_recordings()
        
# Perform timed conditioning for 4 more trials
net.stop_recording()
for trial in range(3):
    conditioning_trial(net, 1) # CS1, US1
    conditioning_trial(net, 2) # CS2, US2

# Record VTA during conditioning
net.record(recorded_areas) 
conditioning_trial(net, 1) # CS1, US1
during_CS1 = net.get_recordings()
conditioning_trial(net, 2) # CS2, US2
during_CS2 = net.get_recordings() 

# Perform timed conditioning for 4 more trials
net.stop_recording()
for trial in range(4):
    conditioning_trial(net, 1) # CS1, US1
    conditioning_trial(net, 2) # CS2, US2 

# Record VTA after conditioning
net.record(recorded_areas) 
conditioning_trial(net, 1) # CS1, US1
after_CS1 = net.get_recordings()
conditioning_trial(net, 2) # CS2, US2
after_CS2 = net.get_recordings() 
 
# Extinction trials    
extinction_trial(net, 1) # CS1
omit_CS1 = net.get_recordings()
extinction_trial(net, 2) # CS2
omit_CS2 = net.get_recordings()

# Content of a single plot
def single_plot(data, ax):
#    ax.set_ylim((0., 1.2))
#    ax.plot(np.array(data['VTA']['rate'][0]), label='VTA')
#    ax.plot(np.max(np.array(data['NAcc']['rate']), axis=0), label='NAcc')
##    ax.plot(np.max(np.array(data['NAcc']['vmpfc']), axis=0), label='NAcc - vmPFC')
##    ax.plot(np.max(np.array(data['NAcc']['bla']), axis=0), label='NAcc - BLA')
#    ax.plot(np.max(np.array(data['BLA']['rate']), axis=0), label='BLA')
    data = np.concatenate((np.array(data['NAcc']['rate']), np.array(data['NAcc_pred']['rate'])), axis=0)
    ax.imshow(data, aspect='auto', cmap=color.hot )
    
# Plot
import pylab as plt
import matplotlib.cm as color

ax = plt.subplot(421)
ax.set_title('CS1 - US1')
ax.set_ylabel('Before conditioning')
single_plot(before_CS1, ax)
#ax.legend(loc=2)#, fontsize='small', frameon=False)
ax = plt.subplot(422)
ax.set_title('CS2 - US2')
single_plot(before_CS2, ax)
ax = plt.subplot(423)
ax.set_ylabel('During conditioning')
single_plot(during_CS1, ax)
ax = plt.subplot(424)
single_plot(during_CS2, ax)
ax = plt.subplot(425)
ax.set_ylabel('After conditioning')
single_plot(after_CS1, ax)
ax = plt.subplot(426)
single_plot(after_CS2, ax)
ax = plt.subplot(427)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Omission of reward')
single_plot(omit_CS1, ax)
ax = plt.subplot(428)
ax.set_xlabel('Time (ms)')
single_plot(omit_CS2, ax)

plt.show()


