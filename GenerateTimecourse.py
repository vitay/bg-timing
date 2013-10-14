################################################################
### Script to generate all figures with three CS-US
################################################################

from TimingNetwork import *

import pylab as plt
import matplotlib
import matplotlib.cm as cm
import cPickle

# Network
nb_stim=3

# Flags
learn_network = True # Do not relearn the task
save_figures = True # Save the the figures or display them

# Recorded data
recorded_areas = ['BLA', 'VTA']
valuation_trials = []
conditioning_trials = []
extinction_trials = []
sooner_trials = []

    
def save_figure(fig, name, width=2, ratio=0.75):
    matplotlib.rcParams.update({'font.size': 8})
    w = int(float(width) * 90./25.4)
    fig.set_dpi(900)
    fig.set_size_inches((w, w*ratio))
#    fig.savefig('figs/'+name+'.svg')
#    fig.savefig('figs/'+name+'.eps')
    fig.savefig('figs/'+name+'.jpg', dpi=900, bbox_inches="tight")

def find_max_cell(data):
    return np.argmax(np.mean(data, axis=1))

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
            valuation_trial(net, US=stim+1) # US1
            valuation_trials.append(net.get_recordings())
        
    # Stop learning in the LH -> BLA pathway
    net.projection(pre="LH", post="BLA", connection_type="exc").set_learning_parameters({'tau': 10000000.0})

    # Start the conditioning phase
    for trial in range(nb_conditioning):
        for stim in range(nb_stim):
            conditioning_trial(net, CS=stim+1) # CS1, US1
            conditioning_trials.append(net.get_recordings())
        
    # Sooner trials
    net.learn=False
    for trial in range(nb_sooner):
        for stim in range(nb_stim):
            sooner_trial(net, CS=stim+1) # CS1, US1
            sooner_trials.append(net.get_recordings())
        
    # Extinction trials
    net.learn=True
    for trial in range(nb_extinction):
        for stim in range(nb_stim):
            extinction_trial(net, CS=stim+1) # CS1, US1
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
        dopa = np.array(data['VTA']['rate'][0])
        duration = len(dopa)
        ticks = np.linspace(0, duration, int(duration/1000)+1)
        ax.set_xlim((0., duration))
        ax.set_ylim((0., 1.2))
        ax.set_xticks(ticks) 
        ax.set_xticklabels([ int(i) for i in ticks/1000.]) 
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.plot(dopa, color='black', lw=1, label='VTA')
        
    print 'Generate VTA plot'
    fig, axes = plt.subplots(nrows=5, ncols=nb_stim, sharex='col', sharey='row')

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
        ax.set_xlabel('Time (s)')
        if stim == 0:
            ax.set_ylabel('Omission')
        single_plot(extinction_trials[stim], ax)

    
    if save_figures:
        save_figure(fig, 'VTA_activity', width=2, ratio=0.65)
    else:
        plt.show()
    plt.close()
    
def plot_vta_peaks(nb_stim=2):
    "Shows evolution of maximal DA firing during conditioning."
    def analyse(data):
        CS = [ [] for stim in range(nb_stim)]
        US=[[] for stim in range(nb_stim)]
        stim = 0
        for trial in data:
            vta = np.array(trial['VTA']['rate'][0])
            CS[stim].append(np.max(vta[900:1300]))
            US[stim].append(np.max(vta[1000 + CS_US[str(stim+1)]['duration']-100: 1000 + CS_US[str(stim+1)]['duration']+200]))
            stim = (stim + 1)%nb_stim
        return CS, US
        
    print 'Generate evolution of VTA bursts'
    CS, US = analyse(conditioning_trials)
    fig, axes = plt.subplots(nrows=1, ncols=nb_stim)
    
    for stim in range(nb_stim):
        ax = axes[stim]
        ax.set_ylim((0., 1.2))
        title = "CS%(rk)s - US%(rk)s" % {'rk': str(stim+1) }
        ax.set_title(title)
        ax.set_xlabel('Trial')
        if stim == 0:
            ax.set_ylabel('Amplitude of VTA bursts')
        ax.plot(np.arange(len(CS[stim]))+1, CS[stim], color='green', label='CS')
        ax.plot(np.arange(len(US[stim]))+1, US[stim], color='red', label='US')
        if stim == 1:
            ax.legend(loc=2, frameon=False)
    
    if save_figures:
        save_figure(fig, 'VTA_peaks', width=2, ratio=0.4)
    else:
        plt.show()
    plt.close()
    
def plot_bla(nb_stim=2):
    "Shows activity of the BLA cells during conditioning."
    def single_plot(data, ax):
        ax.set_ylim((0., 1.3))
        duration = data['duration']
        ticks = np.linspace(0, duration, int(duration/1000)+1)
        ax.set_xticks(ticks) 
        ax.set_xticklabels([ int(i) for i in ticks/1000.]) 
        cell = find_max_cell(np.array(data['BLA']['rate']))
        ax.plot(np.array(data['BLA']['rate'])[cell], color='black', label='BLA max')

    print 'Generate BLA plot'
    fig, axes = plt.subplots(nrows=2, ncols=nb_stim, sharex=False, sharey=True)

    for stim in range(nb_stim):
        title = "CS%(rk)s - US%(rk)s" % {'rk': str(stim+1) }
        ax = axes[0, stim]
        ax.set_title(title)
        if stim == 0:
            ax.set_ylabel('Trial 1')
        single_plot(conditioning_trials[stim], ax)        

        ax = axes[1, stim]
        if stim == 0:
            ax.set_ylabel('Trial 10')
        ax.set_xlabel('Time (s)')
        single_plot(conditioning_trials[9*nb_stim + stim], ax)
    
    if save_figures:
        save_figure(fig, 'BLA_activity', width=2, ratio=0.5)
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
    plot_bla(nb_stim)
    
    # Evolution of CS and US-related firing
    plot_vta_peaks(nb_stim)
    
