# -*- coding: utf-8 -*-
# Implementation in ANNarchy4 of the timing model presented in the Frontiers article
 
from ANNarchy4 import *
setup(suppress_warnings=True)

############################
# Parameters
############################
nb_gus = 4
nb_vis = 3

############################
# Definition of the neurons
############################

# Basic leaky neuron
LeakyNeuron = Neuron(
    tau = 10.0,
    noise = Variable(init=0.0, eq=Uniform(-0.1, 0.1)),
    baseline = Variable(init=0.0),
    mp = Variable(
        init=0.0, 
        eq='tau*dmp/dt + mp = sum(exc) - sum(inh) + baseline + noise'
    ),
    rate = Variable(init=0.0, eq='rate = pos(mp)')
)

# Neuron with temporal adaptation of inputs
PhasicNeuron = Neuron(
    tau = 10.0,
    tau_adaptation = 500.0,
    K_adaptation = 1.0,
    noise = Variable(init=0.0, eq=Uniform(-0.1, 0.1)),
    adapted_exc = Variable(
        init=0.0, 
        eq='tau_adaptation*dadapted_exc/dt + adapted_exc =  sum(exc) '
    ),
    dopa = Variable(
        init=0.0, 
        eq='dopa = sum(dopa) '
    ),
    mp = Variable(
        init=0.0, 
        eq='tau*dmp/dt + mp =  pos(sum(exc) - K_adaptation * adapted_exc) - sum(inh) + noise'
    ),
    rate = Variable(init=0.0, eq='rate = pos(mp)'),
    order= ['noise', 'dopa', 'adapted_exc', 'mp', 'rate']
)


############################
# Definition of the synapses
############################
Covariance = Synapse(
    eta = 100.0,
    tau_alpha = 10.0,
    alpha = Variable(init=0.0, eq="tau_alpha * dalpha/dt + alpha = pos(post.rate - 1.0)"),
    value = Variable(eq="""
    eta * dvalue/dt = if (pre.rate > mean(pre.rate) or post.rate > mean(post.rate) ) 
                      then 
                        (pre.rate - mean(pre.rate) ) * (post.rate - mean(post.rate)) - alpha * (post.rate - mean(post.rate))^2 * value 
                      else 
                        0.0
    """, min=0.0),
    order = ['alpha', 'value']
)

DACovariance = Synapse(
    eta = 100.0,
    tau_alpha = 1.0,
    tau_dopa = 300.0,
    K_alpha = 5.0,
    K_LTD = 1.0,
    dopa_threshold_LTP = 0.3,
    dopa_K_LTP = 10.0,
    alpha = Variable(init=0.0, eq="tau_alpha * dalpha/dt + alpha = pos(post.rate - 1.0)"),
    dopa_mean = Variable(init=0.0, eq="tau_dopa*ddopa_mean/dt + dopa_mean = post.dopa"),
    dopa = Variable(init=0.0, eq="""
        dopa = if post.dopa > dopa_threshold_LTP
               then 
                    dopa_K_LTP * pos( post.sum(dopa) - dopa_mean )
               else
                    0.0
    """),
    value = Variable(eq="""
    eta * dvalue/dt = if (pre.rate > mean(pre.rate) and post.rate > mean(post.rate) ) 
                      then 
                        ( dopa * (pre.rate - mean(pre.rate) ) * (post.rate - mean(post.rate)) - K_alpha * alpha * (post.rate - mean(post.rate))^2 * value) 
                      else 
                        ( if (pre.rate > mean(pre.rate) or post.rate > mean(post.rate) )
                          then
                             K_LTD * dopa * (pre.rate - mean(pre.rate) ) * (post.rate - mean(post.rate))
                          else
                             0.0
                        )
    """, min=0.0),
    order = ['dopa_mean', 'dopa', 'alpha', 'value']
)

AntiHebb = Synapse(
    eta = 100.0,
    value = Variable(eq="""
    eta * dvalue/dt = if (pre.rate > mean(pre.rate) and post.rate > mean(post.rate) ) 
                      then 
                        (pre.rate - mean(pre.rate) ) * (post.rate - mean(post.rate))
                      else
                        0.0
    """, min=0.0, max=3.0)  
)


#############################
# Creation of the populations
#############################

# Gustatory inputs
LH = Population(name='LH', geometry=nb_gus, neuron=LeakyNeuron)

# Visual inputs
VIS = Population(name='VIS', geometry=nb_vis, neuron=LeakyNeuron)

# Basolateral amygdala
BLA = Population(name='BLA', geometry=(6, 6), neuron=PhasicNeuron)
BLA.tau_adaptation = 500.0
BLA.K_adaptation = 0.8

# Central nucleus of the amygdala
CE = Population(name='CE', geometry=1, neuron=LeakyNeuron)

# Pedunculopontine nucleus
PPTN = Population(name='PPTN', geometry=1, neuron=PhasicNeuron)
PPTN.tau_adaptation = 50.0
PPTN.K_adaptation = 1.0

# Ventral tegmental area
VTA = Population(name='VTA', geometry=1, neuron=LeakyNeuron)
VTA.tau = 30.0
VTA.baseline = 0.2


#############################
# Creation of the projections
#############################

# Projections to the amygdala

LH_BLA = Projection(
    pre = LH,
    post = BLA,
    target = 'exc',
    connector = All2All(weights=Uniform(0.2, 0.4)),
    synapse = DACovariance
)

BLA_BLA = Projection(
    pre = BLA,
    post = BLA,
    target = 'inh',
    connector = All2All(weights=0.5),
    synapse = AntiHebb
)

BLA_CE = Projection(
    pre = BLA,
    post = CE,
    target = 'exc',
    connector = All2All(weights=1.0)
)

VTA_BLA = Projection(
    pre = VTA,
    post = BLA,
    target = 'dopa',
    connector = All2All(weights=1.0)
)

# Projections to VTA

LH_PPTN = Projection(
    pre = LH,
    post = PPTN,
    target = 'exc',
    connector = All2All(weights=0.75)
)

PPTN_VTA = Projection(
    pre = PPTN,
    post = VTA,
    target = 'exc',
    connector = All2All(weights=1.5)
)

#############################
# Trial definition
#############################

# Sensitization trial
def sensitization_trial(trial_setup):
    # Loop over all US, in ascending order
    for us_def in trial_setup:
        # Reset
        simulate(500)
        # Set inputs
        LH.baseline = us_def['GUS']
        # Simulate for the desired duration
        simulate(us_def['duration'])
        # Reset
        LH.baseline = np.zeros(len(us_def['GUS']))
        simulate(500)
    

#############################
# Main loop
#############################
if __name__ == '__main__':
    # Compile the network
    compile()
    
    # Perform 10 sensitization trials per US
    trial_setup = [
        {'GUS': np.array([1., 1., 0., 0.]), 'duration': 1000},
        {'GUS': np.array([1., 0., 1., 0.]), 'duration': 1000},
        {'GUS': np.array([0., 0., 1., 1.]), 'duration': 1000}
    ]
    for trial in range(10):
        sensitization_trial(trial_setup)
