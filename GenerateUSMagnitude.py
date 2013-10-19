################################################################
### Script to generate all figures in the article
################################################################

from TimingNetwork import *

import pylab as plt
import matplotlib
import matplotlib.cm as cm
import cPickle

# Flags
learn_network = True # Do not relearn the task
save_figures = True # Save the the figures or display them

# Recorded data
recorded_areas = ['BLA', 'VTA']
first_trials = []
last_trials = []

    
def save_figure(fig, name, width=2, ratio=0.75):
    matplotlib.rcParams.update({'font.size': 8})
    w = int(float(width) * 90./25.4)
    fig.set_dpi(900)
    fig.set_size_inches((w, w*ratio))
#    fig.savefig('figs/'+name+'.svg')
#    fig.savefig('figs/'+name+'.eps')
    fig.savefig('figs/'+name+'.jpg', dpi=900, bbox_inches="tight")



def run_simulation(nb_magnitude=10, nb_valuation = 10, nb_conditioning = 15, nb_extinction = 1, nb_sooner = 1, record=None):
    "Trains the network on the conditioning task for different US magnitudes"
    
    for magnitude in range(nb_magnitude):
        # Create the network
        net = TimingNetwork()
        net.nb_visual_inputs = 2
        net.build()
    
        # Define the association
        stim = {'visual': 0,
                'vector': [1.0, 1.0, 0.0, 0.0],
                'magnitude': (magnitude)/float(nb_magnitude-1),
                'duration': 2000 }
    
        # Habituate the network to the gustatory inputs
        for trial in range(nb_valuation):
            valuation_trial(net, stimulus=stim)
        
        # Stop learning in the LH -> BLA pathway
        net.projection(pre="LH", post="BLA", connection_type="exc").set_learning_parameters({'tau': 10000000.0})

        # Start the conditioning phase
        # Record the first trial
        net.record(record)
        conditioning_trial(net, stimulus=stim)
        first_trials.append(net.get_recordings())
        net.stop_recording()
        # Run 8 trials without recording
        for trial in range(nb_conditioning-2):
            conditioning_trial(net, stimulus=stim) 
        # Record the last trial
        net.record(record)
        conditioning_trial(net, stimulus=stim)
        last_trials.append(net.get_recordings())
        net.stop_recording()        
                
#    # Save recordings
#    recordings = {'first': first_trials,
#                  'last': last_trials}
#    cPickle.dump(recordings, open('recordings_multiple.data', 'w')) 
    
def plot_evolution(bla, vta, nb_magnitude=11):    
    
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    
    ax = axes[0]
    ax.set_ylim((0., 1.35))
    xes = np.arange(nb_magnitude)/float(nb_magnitude-1)
    ax.plot(xes, bla[0], color='red', label='CS, trial #1')
    ax.plot(xes, bla[1], color='green', label='US, trial #1')
    ax.plot(xes, bla[2], color='blue', label='CS, trial #15')
    ax.plot(xes, bla[3], color='black', label='US, trial #15')
    ax.set_xlabel('Reward magnitude')
    ax.set_ylabel('Maximal activity in BLA')
    ax.legend(loc=2, frameon=False , prop={'size':8}, labelspacing=0.4)
    
    ax = axes[1]
    ax.set_ylim((0., 1.2))
    xes = np.arange(nb_magnitude)/float(nb_magnitude-1)
    ax.plot(xes, vta[0], color='red', label='CS, trial #1')
    ax.plot(xes, vta[1], color='green', label='US, trial #1')
    ax.plot(xes, vta[2], color='blue', label='CS, trial #15')
    ax.plot(xes, vta[3], color='black', label='US, trial #15')
    ax.set_xlabel('Reward magnitude')
    ax.set_ylabel('Amplitude of VTA bursts')
    ax.legend(loc=2, frameon=False , prop={'size':8}, labelspacing=0.4)
    
    
    fig.text(0.05, 0.95,'(A)', fontweight='bold', fontsize=8)
    fig.text(0.5, 0.95,'(B)', fontweight='bold', fontsize=8)
    
    if save_figures:
        save_figure(fig, 'VTA_evolution', width=2, ratio=0.4)
    else:
        plt.show()
    plt.close()
 
def analyse_vta(data, pop):
    CS=[]; US=[]
    for rec in data:
        vals = np.array(rec[pop]['rate'][0])
        CS.append(np.max(vals[900:1100]))
        US.append(np.max(vals[2900:3100]))
    return CS, US

def analyse_bla(data, pop):
    CS=[]; US=[]
    for rec in data:
        vals = np.max(np.array(rec[pop]['rate']), axis=0)
        CS.append(np.max(vals[900:1100]))
        US.append(np.max(vals[2900:3100]))
    return CS, US 
    
if __name__=='__main__':

    bla = [[], [], [], []]
    vta = [[], [], [], []]
    
    for n in range(1):
        # Run the simulation
        run_simulation(nb_magnitude=11, record=recorded_areas)
        
        # Analyse recordings
        bla_before_CS, bla_before_US = analyse_bla(first_trials, 'BLA')
        bla_after_CS, bla_after_US = analyse_bla(last_trials, 'BLA')
        vta_before_CS, vta_before_US = analyse_vta(first_trials, 'VTA')
        vta_after_CS, vta_after_US = analyse_vta(last_trials, 'VTA')
        
        bla[0].append(bla_before_CS)
        bla[1].append(bla_before_US)
        bla[2].append(bla_after_CS)
        bla[3].append(bla_after_US)
        
        vta[0].append(vta_before_CS)
        vta[1].append(vta_before_US)
        vta[2].append(vta_after_CS)
        vta[3].append(vta_after_US)
        
        # Reset recordings
        first_trials = []
        last_trials = []
        
    # Compute the mean
    bla_mean=[]
    bla_mean.append(np.mean(bla[0], axis=0))
    bla_mean.append(np.mean(bla[1], axis=0))
    bla_mean.append(np.mean(bla[2], axis=0))
    bla_mean.append(np.mean(bla[3], axis=0))
    vta_mean=[]
    vta_mean.append(np.mean(vta[0], axis=0))
    vta_mean.append(np.mean(vta[1], axis=0))
    vta_mean.append(np.mean(vta[2], axis=0))
    vta_mean.append(np.mean(vta[3], axis=0))
    
    # Do the plot
    plot_evolution(bla_mean, vta_mean, nb_magnitude=11)
