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
recorded_areas = ['BLA', 'VTA', 'LH', 'VIS', 'NAcc', 'CE', 'PPTN_US', 'PPTN_CS', 'VP', 'RMTg', 'LHb']
conditioning_trials = []
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

def run_simulation(nb_stim=2, nb_valuation = 15, nb_conditioning = 10, nb_extinction = 1, nb_sooner = 1):
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
            conditioning_trial(net, CS=stim+1) # CS1, US1
        
    # Record one conditioning and one extinction trial
    net.learn=False
    net.record(recorded_areas)
    for stim in range(nb_stim):
        conditioning_trial(net, CS=stim+1) # CS1, US1
        conditioning_trials.append(net.get_recordings())
        extinction_trial(net, CS=stim+1) # CS1, US1
        extinction_trials.append(net.get_recordings())
            
        
    # Save recordings and net
    net.save('net.zip')
    recordings = {'conditioning': conditioning_trials,
                  'extinction': extinction_trials}
    cPickle.dump(recordings, open('recordings.data', 'w')) 

def plot_all(nb_stim=2):
    "Shows activity of the VTA cell during conditioning (similar to Schultz 1998)."
    def set_ticks(ax, ticks):
        ax.set_ylim((-0.05, 1.2))
        ax.set_xticks(ticks) 
        ax.set_xticklabels([ int(i) for i in ticks/1000.])         
        ax.set_yticks([0.0, 0.5, 1.0])  
        
    print 'Generate timecourse plot'
    fig, axes = plt.subplots(nrows=8, ncols=2, sharex=True, sharey=True)

    cond =  conditioning_trials[0]
    ext =  extinction_trials[0]
    duration = len(cond['VTA']['rate'][0])
    ticks = np.linspace(0, duration, int(duration/1000)+1)
    
    ax = axes[0, 0]
    ax.set_title("Conditioning trial")
    ax.set_ylabel('Inputs')
    set_ticks(ax, ticks)
    rec = np.array(cond['LH']['rate'][0])
    ax.plot(rec, color='black')
    rec = np.array(cond['VIS']['rate'][0])
    ax.plot(rec, color='blue')

    ax = axes[0, 1]
    ax.set_title("Omission trial")
    set_ticks(ax, ticks)
    rec = np.array(ext['LH']['rate'][0])
    ax.plot(rec, color='black')
    rec = np.array(cond['VIS']['rate'][0])
    ax.plot(rec, color='blue')
    
    ax = axes[1, 0]
    ax.set_ylabel('VTA')
    set_ticks(ax, ticks)
    rec = np.array(cond['VTA']['rate'][0])
    ax.plot(rec, color='black')

    ax = axes[1, 1]
    set_ticks(ax, ticks)
    rec = np.array(ext['VTA']['rate'][0])
    ax.plot(rec, color='black')
    
    ax = axes[2, 0]
    ax.set_ylabel('CE')
    set_ticks(ax, ticks)
    rec = np.array(cond['CE']['rate'][0])
    ax.plot(rec, color='black')

    ax = axes[2, 1]
    set_ticks(ax, ticks)
    rec = np.array(ext['CE']['rate'][0])
    ax.plot(rec, color='black')
    
    ax = axes[3, 0]
    ax.set_ylabel('PPTN')
    set_ticks(ax, ticks)
    rec = np.array(cond['PPTN_US']['rate'][0])
    ax.plot(rec, color='black')
    rec = np.array(cond['PPTN_CS']['rate'][0])
    ax.plot(rec, color='blue')

    ax = axes[3, 1]
    set_ticks(ax, ticks)
    rec = np.array(ext['PPTN_US']['rate'][0])
    ax.plot(rec, color='black')
    rec = np.array(ext['PPTN_CS']['rate'][0])
    ax.plot(rec, color='blue')
    
    ax = axes[4, 0]
    ax.set_ylabel('NAcc')
    set_ticks(ax, ticks)
    max_cell = find_max_cell(cond['NAcc']['rate'])
    rec = np.array(cond['NAcc']['rate'][max_cell]) 
    ax.plot(rec, color='black')

    ax = axes[4, 1]
    set_ticks(ax, ticks)
    max_cell = find_max_cell(cond['NAcc']['rate'])
    rec = np.array(ext['NAcc']['rate'][max_cell])
    ax.plot(rec, color='black')
    
    ax = axes[5, 0]
    ax.set_ylabel('VP')
    set_ticks(ax, ticks)
    rec = np.array(cond['VP']['rate'][0])
    ax.plot(rec, color='black')

    ax = axes[5, 1]
    set_ticks(ax, ticks)
    rec = np.array(ext['VP']['rate'][0])
    ax.plot(rec, color='black')
    
    ax = axes[6, 0]
    ax.set_ylabel('LHb')
    set_ticks(ax, ticks)
    rec = np.array(cond['LHb']['rate'][0])
    ax.plot(rec, color='black')

    ax = axes[6, 1]
    set_ticks(ax, ticks)
    rec = np.array(ext['LHb']['rate'][0])
    ax.plot(rec, color='black')

    ax = axes[7, 0]
    ax.set_ylabel('RMTg')
    ax.set_xlabel('Time (s)')
    set_ticks(ax, ticks)
    rec = np.array(cond['RMTg']['rate'][0])
    ax.plot(rec, color='black')

    ax = axes[7, 1]
    set_ticks(ax, ticks)
    ax.set_xlabel('Time (s)')
    rec = np.array(ext['RMTg']['rate'][0])
    ax.plot(rec, color='black')
    
    
    fig.text(0.05, 0.95,'(A)', fontweight='bold', fontsize=8)
    fig.text(0.5, 0.95,'(B)', fontweight='bold', fontsize=8)
    
    if save_figures:
        save_figure(fig, 'timecourse_all', width=2, ratio=0.7)
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
        conditioning_trials = recordings['conditioning']
        extinction_trials = recordings['extinction']
        
    #############################
    ### Analyse the recordings
    #############################
    
    # VTA firing
    plot_all(nb_stim)
    
