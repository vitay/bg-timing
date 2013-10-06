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
recorded_areas = ['VTA']
extinction_trials = []
    
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
    
    # Habituate the network to the gustatory inputs
    for trial in range(nb_valuation):
        for stim in range(nb_stim):
            valuation_trial(net, US=stim+1) # US1
        
    # Stop learning in the LH -> BLA pathway
    net.projection(pre="LH", post="BLA", connection_type="exc").set_learning_parameters({'tau': 10000000.0})

    # Start the conditioning phase
    for trial in range(nb_conditioning):
        for stim in range(nb_stim):
            # Conditioning trial with learning enabled
            net.learn=True
            conditioning_trial(net, CS=stim+1) # CS1, US1
            # Extinction trial with learning disabled
            net.learn=False
            net.record(recorded_areas)
            extinction_trial(net, CS=stim+1) # CS1, US1
            extinction_trials.append(net.get_recordings())
            
        
    # Save recordings and net
    net.save('net.zip')
    recordings = {'extinction': extinction_trials }
    cPickle.dump(recordings, open('recordings_dips.data', 'w')) 

def plot_vta(nb_stim=2):
    "Shows activity of the VTA cell during conditioning (similar to Schultz 1998)."
    def analyse(data):
        US=[[] for stim in range(nb_stim)]
        stim = 0
        for trial in data:
            vta = np.array(trial['VTA']['rate'][0])
            US[stim].append(np.min(vta))
            stim = (stim + 1)%nb_stim
        return US
        
        
    print 'Generate VTA plot'
    US = analyse(extinction_trials)
    fig, axes = plt.subplots(nrows=1, ncols=nb_stim,  sharex='col', sharey='row')

    for stim in range(nb_stim):
        title = "CS%(rk)s - US%(rk)s" % {'rk': str(stim+1) }
        ax = axes[stim]
        ax.set_title(title)
        ax.set_xlabel('Trial')
        ax.set_ylim((0.0, 0.3))
        if stim == 0:
            ax.set_ylabel('Minimal VTA activity during extinction')
        ax.plot(np.arange(len(US[stim]))+1, US[stim], color='black', label='US')
    
    if save_figures:
        save_figure(fig, 'VTA_dips', width=2, ratio=0.5)
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
        recordings = cPickle.load(open('recordings_dips.data', 'r')) 
        extinction_trials = recordings['extinction']
        
    #############################
    ### Analyse the recordings
    #############################
    
    # VTA firing
    plot_vta(nb_stim)
    
