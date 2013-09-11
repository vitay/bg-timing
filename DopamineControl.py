################################################################
### Script showing the activity of VTA cells during conditioning
### Figure should be similar to Schultz 1998
################################################################

from TimingNetwork import *

# Create the network
net = TimingNetwork()
net.build()


# Habituate the network to the gustatory inputs
for trial in range(10):
    valuation_trial(net, 1) # US1
    valuation_trial(net, 2) # US2
    
# Record VTA before conditioning
net.record(['VTA']) 
conditioning_trial(net, 1) # CS1, US1
before_CS1 = net.get_recordings()
conditioning_trial(net, 2) # CS2, US2
before_CS2 = net.get_recordings()
        
# Perform timed conditioning for 4 more trials
net.stop_recording()
for trial in range(4):
    conditioning_trial(net, 1) # CS1, US1
    conditioning_trial(net, 2) # CS2, US2

# Record VTA during conditioning
net.record(['VTA']) 
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
net.record(['VTA']) 
conditioning_trial(net, 1) # CS1, US1
after_CS1 = net.get_recordings()
conditioning_trial(net, 2) # CS2, US2
after_CS2 = net.get_recordings() 
 
# Extinction trials    
extinction_trial(net, 1) # CS1
omit_CS1 = net.get_recordings()
extinction_trial(net, 2) # CS2
omit_CS2 = net.get_recordings()

# Plot
import pylab as plt
ax = plt.subplot(421)
ax.set_title('CS1 - US1')
ax.set_ylabel('Before conditioning')
ax.plot(np.array(before_CS1['VTA']['rate'][0]))
ax = plt.subplot(422)
ax.set_title('CS2 - US2')
ax.plot(np.array(before_CS2['VTA']['rate'][0]))
ax = plt.subplot(423)
ax.set_ylabel('During conditioning')
ax.plot(np.array(during_CS1['VTA']['rate'][0]))
ax = plt.subplot(424)
ax.plot(np.array(during_CS2['VTA']['rate'][0]))
ax = plt.subplot(425)
ax.set_ylabel('After conditioning')
ax.plot(np.array(after_CS1['VTA']['rate'][0]))
ax = plt.subplot(426)
ax.plot(np.array(after_CS2['VTA']['rate'][0]))
ax = plt.subplot(427)
ax.set_ylabel('Omission of reward')
ax.plot(np.array(omit_CS1['VTA']['rate'][0]))
ax = plt.subplot(428)
ax.plot(np.array(omit_CS2['VTA']['rate'][0]))

plt.show()


