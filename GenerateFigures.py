################################################################
### Script to generate all figures in the article
################################################################

from TimingNetwork import *

import pylab as plt
import matplotlib
import matplotlib.cm as cm
import cPickle

# Network
nb_stim=2

# Flags
learn_network = True # Do not relearn the task
save_figures = True # Save the the figures or display them

# Recorded data
recorded_areas = ['BLA', 'GUS', 'VIS', 'LH_ON', 'CE', 'VTA', 'PPTN', 'RMTg', 'VP', 'NAcc', 'OFC']
valuation_trials = []
conditioning_trials = []
extinction_trials = []
sooner_trials = []

# Tune matplotlib
if save_figures:
    matplotlib.rcParams.update({'font.size': 8})
    
def save_figure(fig, name, width=2, ratio=0.75):
    w = int(float(width) * 90./25.4)
    fig.set_dpi(900)
    fig.set_size_inches((w, w*ratio))
    fig.savefig('figs/'+name+'.svg')
    fig.savefig('figs/'+name+'.eps')
    fig.savefig('figs/'+name+'.jpg', dpi=900)



def run_simulation(nb_stim=2, nb_valuation = 10, nb_conditioning = 10, nb_extinction = 1, nb_sooner = 1):
    "Trains the network on the conditioning task"
    # Create the network
    net = TimingNetwork()
    net.nb_visual_inputs = nb_stim
    net.build()
    net.record(recorded_areas)
    
    # Habituate the network to the gustatory inputs
    for trial in range(nb_valuation):
        for stim in range(nb_stim):
            valuation_trial(net, stim+1) # US1
            valuation_trials.append(net.get_recordings())
        
    # Stop learning in the LH -> BLA pathway
    net.projection(pre="LH_ON", post="BLA", connection_type="exc").set_learning_parameters({'tau': 10000000.0})

    # Start the conditioning phase
    for trial in range(nb_conditioning):
        for stim in range(nb_stim):
            conditioning_trial(net, stim+1) # CS1, US1
            conditioning_trials.append(net.get_recordings())
        
    # Sooner trials
    net.learn=False
    for trial in range(nb_sooner):
        for stim in range(nb_stim):
            sooner_trial(net, stim+1) # CS1, US1
            sooner_trials.append(net.get_recordings())
        
    # Extinction trials
    net.learn=True
    for trial in range(nb_extinction):
        for stim in range(nb_stim):
            extinction_trial(net, stim+1) # CS1, US1
            extinction_trials.append(net.get_recordings())
            
        
    # Save recordings and net
    net.save('net.zip')
    recordings = {'valuation': valuation_trials,
                  'conditioning': conditioning_trials,
                  'extinction': extinction_trials,
                  'sooner': sooner_trials }
    cPickle.dump(recordings, open('recordings.data', 'w')) 

def plot_vta(nb_stim=2):
    "Shows activity of the VTA cell during conditioning (similar to Schultz 1998)."
    def single_plot(data, ax):
        ax.set_ylim((0., 1.2))
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
#        ax.axis["right"].set_visible(False)
#        ax.axis["top"].set_visible(False)
#        ax.spines["right"].set_visible(False)
#        ax.spines["top"].set_visible(False)
        ax.plot(np.array(data['VTA']['rate'][0]), label='VTA')
        
    fig, axes = plt.subplots(nrows=5, ncols=nb_stim,  sharex='col', sharey='row')

    for stim in range(nb_stim):
        title = "CS%(rk)s - US%(rk)s" % {'rk': str(stim+1) }
        ax = axes[0, stim]
        ax.set_title(title)
        if stim == 0:
            ax.set_ylabel('Trial 1')
        single_plot(conditioning_trials[stim], ax)

        ax = axes[1, stim]
        if stim == 0:
            ax.set_ylabel('Trial 5')
        single_plot(conditioning_trials[4*nb_stim + stim], ax)
        

        ax = axes[2, stim]
        if stim == 0:
            ax.set_ylabel('Trial 10')
        single_plot(conditioning_trials[9*nb_stim + stim], ax)
    
        ax = axes[3, stim]
        if stim == 0:
            ax.set_ylabel('Sooner')
        single_plot(sooner_trials[stim], ax)
    
        ax = axes[4, stim]
        ax.set_xlabel('Time (ms)')
        if stim == 0:
            ax.set_ylabel('Omission')
        single_plot(extinction_trials[stim], ax)

    
    if save_figures:
        save_figure(fig, 'VTA_activity')
    else:
        plt.show()
    plt.close()
    
def plot_vta_peaks():
    "Shows evolution of maximal DA firing during conditioning."
    def analyse(data):
        CS1 = []; CS2=[]; US1=[]; US2=[]
        stim = 0
        for trial in data:
            vta = np.array(trial['VTA']['rate'][0])
            if stim==0: # CS1
                CS1.append(np.max(vta[900:1300]))
                US1.append(np.max(vta[1000 + CS_US['1']['duration']-100: 1000 + CS_US['1']['duration']+200]))
            else: # CS2
                CS2.append(np.max(vta[900:1300]))
                US2.append(np.max(vta[1000 + CS_US['2']['duration']-100: 1000 + CS_US['2']['duration']+200]))
            stim = 1 - stim
        return CS1, US1, CS2, US2
    fig = plt.figure()
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
        save_figure(fig, 'VTA_peaks')
    else:
        plt.show()
    plt.close()
    
def plot_bla():
    "Shows activity of the BLA cells during conditioning."
    def single_plot(data, ax):
        ax.set_ylim((0., 1.2))
        ax.plot(np.max(np.array(data['BLA']['rate']), axis=0), label='BLA max')
        #ax.plot(np.max(np.array(data['OFC']['rate']), axis=0), label='OFC max')
    fig = plt.figure()
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
        save_figure(fig, 'BLA_activity')
    else:
        plt.show()
    plt.close()
    
if __name__=='__main__':

    #############################
    ### Simulation
    #############################
    
    if learn_network:
        # Run the simulation
        run_simulation(nb_stim)
    else: # Retrieve saved data
        net = load('net.zip')
        recordings = cPickle.load(open('recordings.data', 'r')) 
        valuation_trials = recordings['valuation']
        conditioning_trials = recordings['conditioning']
        extinction_trials = recordings['extinction']
        sooner_trials = recordings['sooner']
        
    #############################
    ### Analyse the recordings
    #############################
    
    # VTA firing
    plot_vta(nb_stim)
    
    # BLA activation
    plot_bla()
    
    # Evolution of CS and US-related firing
    plot_vta_peaks()
    
