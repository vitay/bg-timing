################################################################
### Script showing the activity of BLA cells during conditioning
################################################################

from TimingNetwork import *

# Create the network
net = TimingNetwork()
net.build()

recorded_areas = ['BLA', 'GUS', 'VIS', 'LH', 'CE', 'VTA', 'PPTN']

# Habituate the network to the gustatory inputs
for trial in range(10):
    valuation_trial(net, 1) # US1
    valuation_trial(net, 2) # US2
    
# Stop learning in the LH -> BLA pathway
net.projection(pre="LH", post="BLA", connection_type="exc").set_learning_parameters({'tau': 10000000.0})
    
# Record BLA before conditioning
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

# Record BLA during conditioning
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

# Record BLA after conditioning
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
    ax.plot(np.array(data['PPTN']['rate'][0]), label='PPTN')
    ax.plot(np.array(data['VTA']['rate'][0]), label='VTA')
    ax.plot(np.max(np.array(data['BLA']['rate']), axis=0), label='BLA')
    ax.plot(np.max(np.array(data['CE']['rate']), axis=0), label='CE')
    ax.plot(np.max(np.array(data['LH']['rate']), axis=0), label='LH')
    ax.plot(np.max(np.array(data['GUS']['rate']), axis=0), label='GUS')
    ax.plot(np.max(np.array(data['VIS']['rate']), axis=0), label='VIS')
    
# Plot
import pylab as plt
import matplotlib.cm as cm
#fig = plt.figure(figsize=(8, 6), dpi=1200)
ax = plt.subplot(421)
ax.set_title('CS1 - US1')
ax.set_ylabel('Before conditioning')
single_plot(before_CS1, ax)
ax.legend()
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

plt.savefig('Fig01-Amygdala.jpg')
plt.show()


