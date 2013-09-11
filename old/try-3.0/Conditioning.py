from TimingNetwork import *
from Tests import *

# Number of CS stimuli
nb_stim = 4

# Number of tastes recognized
nb_tastes = 4

# Delays for each stimuli
delays = [  1000. * (1 + i) for i in range(nb_stim) ]

# CS-US association
US = [ [1.0, 1.0, 0.0, 0.0],  
       [0.0, 1.0, 1.0, 0.0],
       [0.0, 0.0, 1.0, 1.0],
       [1.0, 0.0, 0.0, 1.0] ]

# Method defining a single conditioning trial
def trial(net, cs=False, stim=None):

    # Set CS
    if not stim:
        stim = np.random.randint(nb_stim)
    if cs:
        net.population('visual').neuron(stim).baseline = 1.0
    # Run the simulation
    net.execute(delays[stim])
    # Set the US
    taste = US[stim]
    for neur in net.population('gustatory') : neur.baseline = taste[neur.rank]
    net.execute(300.)
    # ISI
    for neur in net.population('visual') : neur.baseline = 0.0
    for neur in net.population('gustatory') : neur.baseline = 0.0
    net.execute(100.)


if __name__ == "__main__":
    
    # Build the network
    net = TimingNetwork()
    net.nb_stim = nb_stim
    net.nb_tastes = nb_tastes
    net.build()

    # Habituate the network to the US
    for t in range(20):
        trial(net, False)
        
    plot_us(net)
    
