################################################################
### Script to generate figure on NAcc
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
recorded_areas = [{'population': 'NAcc', 'variables': ['rate', 'mp', 'mp_up', 'vmpfc', 'bla']} ]
extinction_trials = []

    
def find_max_cell(data):
    return np.argmax(np.mean(data, axis=1))
    
def save_figure(fig, name, width=2, ratio=0.75):
    matplotlib.rcParams.update({'font.size': 8})
    w = int(float(width) * 90./25.4)
    fig.set_dpi(900)
    fig.set_size_inches((w, w*ratio))
#    fig.savefig('figs/'+name+'.svg')
#    fig.savefig('figs/'+name+'.eps')
    fig.savefig('figs/'+name+'.jpg', dpi=900, bbox_inches="tight")



def run_simulation(nb_valuation = 10, nb_conditioning = 10, nb_extinction = 1, nb_sooner = 0, record=None):
    "Trains the network on the conditioning task for different US magnitudes"
    
    # Create the network
    net = TimingNetwork()
    net.nb_visual_inputs = 2
    net.build()

    # Habituate the network to the gustatory inputs
    for trial in range(nb_valuation):
        valuation_trial(net, US=2)
    
    # Stop learning in the LH -> BLA pathway
    net.projection(pre="LH", post="BLA", connection_type="exc").set_learning_parameters({'tau': 10000000.0})

    # Start the conditioning phase
    
    # Run 10 trials without recording
    for trial in range(nb_conditioning):
        conditioning_trial(net, CS=2) 
        
    # Record the last trial with omission
    net.record(record)
    extinction_trial(net, CS=2)
    extinction_trials.append(net.get_recordings())
    net.stop_recording()        
                
    # Save recordings
    recordings = {'extinction': extinction_trials}
    cPickle.dump(recordings, open('recordings_nacc.data', 'w')) 
 
    
def plot_nacc():
    "Shows activity of the NAcc cells during conditioning."
    print 'Generate NAcc plot'
    fig, ax = plt.subplots(nrows=1, ncols=1)

    data = extinction_trials[0]['NAcc']
    cell = find_max_cell(np.array(data['rate']))
    duration = extinction_trials[0]['duration']
    ticks = np.linspace(0, duration, int(duration/1000)+1)
    ax.set_xticks(ticks) 
    ax.set_xticklabels([ int(i) for i in ticks/1000.]) 
    #ax.set_title('NAcc during extinction')  
    ax.set_ylabel('Variables of the NAcc cell')  
    ax.set_xlabel('Time (s)')  
    ax.set_ylim((-2.0, 1.5))
    
    #ax.plot(np.array(data['rate'])[cell], color='black', label='rate')
    ax.plot(np.array(data['mp'])[cell], color='red', label='mp')
    ax.plot(np.array(data['mp_up'])[cell] - 0.9, color='green', label='up/down state')
    ax.plot(np.array(data['vmpfc'])[cell], color='blue', label='vmpfc')    
    #ax.plot(np.array(data['bla'])[cell], color='green', label='bla')
    ax.legend(loc=2, frameon=False)
    
    if save_figures:
        save_figure(fig, 'NAcc_activity', width=2, ratio=0.75)
    else:
        plt.show()
    plt.close()   
    
if __name__=='__main__':

    #############################
    ### Simulation
    #############################
    
    if learn_network:
        # Run the simulation
        run_simulation(record=recorded_areas)
    else: # Retrieve saved data
        net = load('net.zip')
        recordings = cPickle.load(open('recordings_nacc.data', 'r')) 
        extinction_trials = recordings['extinction']
        
    #############################
    ### Analyse the recordings
    #############################
    
    # NAcc activation
    plot_nacc()
