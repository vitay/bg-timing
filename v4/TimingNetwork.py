# -*- coding: utf-8 -*-
# Implementation in ANNarchy4 of the timing model presented in the Frontiers article
 
from ANNarchy4 import *
setup(suppress_warnings=True, show_time=False, verbose=False)

# Import neuron and synapse definitions
from NeuronDefinition import *
from SynapseDefinition import *
from ConnectorDefinition import *

############################
# Parameters
############################
nb_gus = 4
nb_vis = 3
nb_bla = 6

#############################
# Creation of the populations
#############################

# Gustatory inputs
LH = Population(name='LH', geometry=nb_gus, neuron=LeakyNeuron)

# Visual inputs
VIS = Population(name='VIS', geometry=nb_vis, neuron=LeakyNeuron)

# Inerotemporal cortex
IT = Population(name='IT', geometry=(nb_vis, nb_vis), neuron=LeakyNeuron)

# Basolateral amygdala
BLA = Population(name='BLA', geometry=(nb_bla, nb_bla), neuron=ShuntingPhasicNeuron)
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

# Visual inputs
VIS_IT = Projection(
    pre = VIS,
    post = IT,
    target = 'exc',
    connector = Cluster(weights=1.0, number=3)
)

# Projections to the amygdala
LH_BLA = Projection(
    pre = LH,
    post = BLA,
    target = 'exc',
    connector = All2All(weights=Uniform(0.2, 0.4)),
    synapse = DACovariance
)

IT_BLA = Projection(
    pre = IT,
    post = BLA,
    target = 'mod',
    connector = All2All(weights=0.0),
    synapse = DAShuntingCovariance
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
# Main loop
#############################
if __name__ == '__main__':
    # Compile the network
    compile()
    from TrialDefinition import *
    
    # Perform 10 sensitization trials per US
    trial_setup = [
        {'GUS': np.array([1., 1., 0., 0.]), 'duration': 500},
        {'GUS': np.array([1., 0., 1., 0.]), 'duration': 500},
        {'GUS': np.array([0., 0., 1., 1.]), 'duration': 500}
    ]
    for trial in range(10):
        sensitization_trial(trial_setup)
    
    # Stop learning in the LH->BLA pathway
    LH_BLA.eta = 1000000.0  
    
    # Perform 10 conditioning trials per association
    trial_setup = [
        {'GUS': np.array([1., 1., 0., 0.]), 'VIS': np.array([1., 0., 0.]), 'magnitude': 0.8, 'duration': 2000},
        {'GUS': np.array([1., 0., 1., 0.]), 'VIS': np.array([0., 1., 0.]), 'magnitude': 0.5, 'duration': 3000},
        {'GUS': np.array([0., 0., 1., 1.]), 'VIS': np.array([0., 0., 1.]), 'magnitude': 1.0, 'duration': 4000}
    ]
    for trial in range(10):
        conditioning_trial(trial_setup)
