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
recorded_areas = ['BLA', 'VTA', 'PPTN', 'LHb', 'RMTg', 'VP']
first_trials = []
last_trials = []

    
def save_figure(fig, name, width=2, ratio=0.75):
    matplotlib.rcParams.update({'font.size': 8})
    w = int(float(width) * 90./25.4)
    fig.set_dpi(900)
    fig.set_size_inches((w, w*ratio))
#    fig.savefig('figs/'+name+'.svg')
#    fig.savefig('figs/'+name+'.eps')
    fig.savefig('figs/'+name+'.jpg', dpi=900)



def run_simulation(nb_magnitude=10, nb_valuation = 10, nb_conditioning = 10, nb_extinction = 1, nb_sooner = 1):
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
        net.record(recorded_areas)
        conditioning_trial(net, stimulus=stim)
        first_trials.append(net.get_recordings())
        net.stop_recording()
        # Run 8 trials without recording
        for trial in range(nb_conditioning-2):
            conditioning_trial(net, stimulus=stim) 
        # Record the last trial
        net.record(recorded_areas)
        conditioning_trial(net, stimulus=stim)
        last_trials.append(net.get_recordings())
        net.stop_recording()        
                
            
        
    # Save recordings
    recordings = {'first': first_trials,
                  'last': last_trials}
    cPickle.dump(recordings, open('recordings_multiple.data', 'w')) 
    
def plot_evolution(nb_magnitude=11):

    def analyse_results(data, pop):
        CS=[]; US=[]
        for rec in data:
            vals = np.array(rec[pop]['rate'][0])
            CS.append(np.max(vals[900:1100]))
            US.append(np.max(vals[2900:3100]))
        return CS, US
    
    before_CS, before_US = analyse_results(first_trials, 'VTA')
    after_CS, after_US = analyse_results(last_trials, 'VTA')
    
    
    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.set_ylim((0., 1.2))
    xes = np.arange(nb_magnitude)/float(nb_magnitude-1)
    axes.plot(xes, before_CS, label='CS, trial #1')
    axes.plot(xes, before_US, label='US, trial #1')
    axes.plot(xes, after_CS, label='CS, trial #10')
    axes.plot(xes, after_US, label='US, trial #10')
    l = axes.legend(loc=2)
    l.draw_frame(False) 
    
    if save_figures:
        save_figure(fig, 'VTA_evolution', width=2)
    else:
        plt.show()
    plt.close()
    
if __name__=='__main__':

    #############################
    ### Simulation
    #############################
    
    if learn_network:
        # Run the simulation
        run_simulation(nb_magnitude=11)
    else: # Retrieve saved data
        net = load('net.zip')
        recordings = cPickle.load(open('recordings_multiple.data', 'r')) 
        first_trials = recordings['first']
        last_trials = recordings['last']
        
    #############################
    ### Analyse the recordings
    #############################
    
    plot_evolution(nb_magnitude=11)
