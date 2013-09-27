################################################################
### Script to generate all figures in the article
################################################################

from TimingNetwork import *

import pylab as plt
import matplotlib.cm as cm

# Flags
save_network = False
save_recordings = False
save_figures = False

# Recorded data
recorded_areas = ['BLA', 'GUS', 'VIS', 'LH_ON', 'CE', 'VTA', 'PPTN', 'RMTg', 'VP', 'NAcc']
valuation_trials = []
conditioning_trials = []
extinction_trials = []
sooner_trials = []

def run_simulation(net, nb_valuation = 10, nb_conditioning = 10, nb_extinction = 1, nb_sooner = 1):
    "Trains the network on the conditioning task"
    net.record(recorded_areas)
    # Habituate the network to the gustatory inputs
    for trial in range(nb_valuation):
        valuation_trial(net, 1) # US1
        valuation_trials.append(net.get_recordings())
        valuation_trial(net, 2) # US2
        valuation_trials.append(net.get_recordings())
        
    # Stop learning in the LH -> BLA pathway
    net.projection(pre="LH_ON", post="BLA", connection_type="exc").set_learning_parameters({'tau': 10000000.0})

    # Start the conditioning phase
    for trial in range(nb_conditioning):
        conditioning_trial(net, 1) # CS1, US1
        conditioning_trials.append(net.get_recordings())
        conditioning_trial(net, 2) # CS2, US2
        conditioning_trials.append(net.get_recordings())
        
    # Sooner trials
    net.learn=False
    for trial in range(nb_sooner):
        sooner_trial(net, 1) # CS1, US1
        sooner_trials.append(net.get_recordings())
        sooner_trial(net, 2) # CS2, US2
        sooner_trials.append(net.get_recordings())
        
    # Extinction trials
    net.learn=True
    for trial in range(nb_extinction):
        extinction_trial(net, 1) # CS1, US1
        extinction_trials.append(net.get_recordings())
        extinction_trial(net, 2) # CS2, US2
        extinction_trials.append(net.get_recordings())

def plot_vta(net):
    "Shows activity of the VTA cell during conditioning (similar to Schultz 1998)."
    def single_plot(data, ax):
        ax.set_ylim((0., 1.2))
        ax.plot(np.array(data['VTA']['rate'][0]), label='VTA')
    ax = plt.subplot(521)
    ax.set_title('CS1 - US1')
    ax.set_ylabel('Before conditioning')
    single_plot(conditioning_trials[0], ax)
    ax = plt.subplot(522)
    ax.set_title('CS2 - US2')
    single_plot(conditioning_trials[1], ax)
    ax = plt.subplot(523)
    ax.set_ylabel('During conditioning')
    single_plot(conditioning_trials[3*2+0], ax)
    ax = plt.subplot(524)
    single_plot(conditioning_trials[3*2+1], ax)
    ax = plt.subplot(525)
    ax.set_ylabel('After conditioning')
    single_plot(conditioning_trials[9*2+0], ax)
    ax = plt.subplot(526)
    single_plot(conditioning_trials[9*2+1], ax)
    ax = plt.subplot(527)
    ax.set_ylabel('Reward delivered sooner')
    single_plot(sooner_trials[0], ax)
    ax = plt.subplot(528)
    single_plot(sooner_trials[1], ax)
    ax = plt.subplot(529)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Omission of reward')
    single_plot(extinction_trials[0], ax)
    ax = plt.subplot(5, 2, 10)
    ax.set_xlabel('Time (ms)')
    single_plot(extinction_trials[1], ax)
    
    if save_figures:
        plt.savefig('figs/VTA_activity.png', dpi=1200)
    else:
        plt.show()
    plt.close()
    
def plot_vta_peaks(net):
    "Shows evolution of maximal DA firing during conditioning."
    def analyse(data):
        CS1 = []; CS2=[]; US1=[]; US2=[]
        stim = 0
        for trial in data:
            vta = np.array(trial['VTA']['rate'][0])
            if stim==0: # CS1
                CS1.append(np.max(vta[900:1300]))
                US1.append(np.max(vta[2900:3300]))
            else: # CS2
                CS2.append(np.max(vta[900:1300]))
                US2.append(np.max(vta[3900:4300]))
            stim = 1 - stim
        return CS1, US1, CS2, US2
    CS1, US1, CS2, US2 = analyse(conditioning_trials)
    ax = plt.subplot(121)
    ax.set_ylim((0., 1.2))
    ax.set_title('CS1 - US1')
    ax.set_xlabel('Number of Trials')
    ax.set_ylabel('Amplitude of VTA bursts')
    ax.plot(CS1, label='CS')
    ax.plot(US1, label='US')
    ax = plt.subplot(122)
    ax.set_ylim((0., 1.2))
    ax.set_title('CS2 - US2')
    ax.set_xlabel('Number of Trials')
    ax.plot(CS2, label='CS')
    ax.plot(US2, label='US')
    
    if save_figures:
        plt.savefig('figs/VTA_peaks.png', dpi=1200)
    else:
        plt.show()
    plt.close()
    
def plot_bla(net):
    "Shows activity of the BLA cells during conditioning."
    def single_plot(data, ax):
        ax.set_ylim((0., 1.2))
        ax.plot(np.max(np.array(data['BLA']['rate']), axis=0), label='BLA max')
    ax = plt.subplot(521)
    ax.set_title('CS1 - US1')
    ax.set_ylabel('Before conditioning')
    single_plot(conditioning_trials[0], ax)
    ax = plt.subplot(522)
    ax.set_title('CS2 - US2')
    single_plot(conditioning_trials[1], ax)
    ax = plt.subplot(523)
    ax.set_ylabel('During conditioning')
    single_plot(conditioning_trials[3*2+0], ax)
    ax = plt.subplot(524)
    single_plot(conditioning_trials[3*2+1], ax)
    ax = plt.subplot(525)
    ax.set_ylabel('After conditioning')
    single_plot(conditioning_trials[9*2+0], ax)
    ax = plt.subplot(526)
    single_plot(conditioning_trials[9*2+1], ax)
    ax = plt.subplot(527)
    ax.set_ylabel('Reward delivered sooner')
    single_plot(sooner_trials[0], ax)
    ax = plt.subplot(528)
    single_plot(sooner_trials[1], ax)
    ax = plt.subplot(529)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Omission of reward')
    single_plot(extinction_trials[0], ax)
    ax = plt.subplot(5, 2, 10)
    ax.set_xlabel('Time (ms)')
    single_plot(extinction_trials[1], ax)
    
    if save_figures:
        plt.savefig('figs/BLA_activity.png', dpi=1200)
    else:
        plt.show()
    plt.close()
    
if __name__=='__main__':

    #############################
    ### Simulation
    #############################
    # Create the network
    net = TimingNetwork()
    net.build()
    
    # Run the simulation
    run_simulation(net)
    
    # Save recordings and net
    if save_network:
        net.save('net.zip')
    if save_recordings:
        recordings = {'valuation': valuation_trials,
                      'conditioning': conditioning_trials,
                      'extinction': extinction_trials,
                      'sooner': sooner_trials }
        import cPickle
        cPickle.dump(recordings, open('recordings.data', 'w')) 
    
    #############################
    ### Analyse the recordings
    #############################
    
    # VTA firing
    plot_vta(net)
    
    # BLA activation
    plot_bla(net)
    
    # Evolution of CS and US-related firing
    plot_vta_peaks(net)
    
