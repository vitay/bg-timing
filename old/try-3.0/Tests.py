import matplotlib.pylab as plt
import numpy as np


def plot_us(net):
    import Conditioning
    # Record activities
    net.learn = False
    net.record(['BLA'])
    for stim in range(len(Conditioning.US)):
        # Reset the network
        for neur in net.population('visual') : neur.baseline = 0.0
        for neur in net.population('gustatory') : neur.baseline = 0.0
        net.execute(100.)
        # Set the US
        for neur in net.population('gustatory') : neur.baseline = Conditioning.US[stim][neur.rank]
        net.execute(100.)
    net.learn = True
    bla = np.array(net.get_recordings()['BLA']['rate'])

    # Plot the results
    fig = plt.figure()
    plt.imshow(bla, aspect='auto', cmap=plt.cm.hot, vmin=0.0, vmax=1.0)    
    plt.show()
