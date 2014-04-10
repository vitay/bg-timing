# -*- coding: utf-8 -*-
# Implementation in ANNarchy4 of the timing model presented in the Frontiers article
 
from ANNarchy4 import *
setup(dt = 1.0, suppress_warnings=True, show_time=False, verbose=False)

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
nb_nacc = 6

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
PPTN_US = Population(name='PPTN_US', geometry=1, neuron=PhasicNeuron)
PPTN_US.tau_adaptation = 50.0
PPTN_US.K_adaptation = 1.0
PPTN_CS = Population(name='PPTN_CS', geometry=1, neuron=PhasicNeuron)
PPTN_CS.tau_adaptation = 50.0
PPTN_CS.K_adaptation = 1.0

# Ventral tegmental area
VTA = Population(name='VTA', geometry=1, neuron=DopamineNeuron)
VTA.tau = 30.0
VTA.baseline = 0.2
VTA.tau_decrease = 30.0
VTA.tau_modulation = 300.0

# Cortical oscillators in ventromedial prefrontal cortex
vmPFC = Population(name='vmPFC', geometry=(50, nb_vis), neuron=OscillatorNeuron)
vmPFC.freq = np.random.uniform(2.0, 8.0, size=vmPFC.geometry)
vmPFC.phase = np.random.uniform(0.0, np.pi, size=vmPFC.geometry)

# Nucleus accumbens
NAcc = Population(name='NAcc', geometry=(nb_nacc, nb_nacc), neuron=StriatalNeuron)

# Ventral Pallidum 
VP = Population(name='VP', geometry=1, neuron=ShuntingNeuron)
VP.baseline = 0.5

# Rostromedial tegmental nucleus
RMTg = Population(name='RMTg', geometry=1, neuron=LeakyNeuron)
RMTg.description['variables'][2]['bounds']['max'] = 1.1

# Lateral Habenula
LHb = Population(name='LHb', geometry=1, neuron=LeakyNeuron)
LHb.baseline = 1.0
LHb.description['variables'][2]['bounds']['max'] = 1.1


#############################
# Creation of the projections
#############################

# Visual inputs
VIS_IT = Projection(
    pre = VIS,
    post = IT,
    target = 'exc'
).connect_with_func(method=connect_cluster, weight=1.0, number=3)

# Visual inputs to vmPFC
VIS_vmPFC = Projection(
    pre = VIS,
    post = vmPFC,
    target = 'exc'
).connect_with_func(method=connect_stripes, weight=1.0)

# Projections to the amygdala
LH_BLA = Projection(
    pre = LH,
    post = BLA,
    target = 'exc',
    synapse = DACovariance
).connect_all_to_all(weights=Uniform(0.2, 0.4))

IT_BLA = Projection(
    pre = IT,
    post = BLA,
    target = 'mod',
    synapse = DAShuntingCovariance
).connect_all_to_all(weights=0.0)

BLA_BLA = Projection(
    pre = BLA,
    post = BLA,
    target = 'inh',
    synapse = AntiHebb
).connect_all_to_all(weights=0.5)

BLA_CE = Projection(
    pre = BLA,
    post = CE,
    target = 'exc'
).connect_all_to_all(weights=1.0)

VTA_BLA = Projection(
    pre = VTA,
    post = BLA,
    target = 'dopa'
).connect_all_to_all(weights=1.0)

# Projections to VTA

LH_PPTN_US = Projection(
    pre = LH,
    post = PPTN_US,
    target = 'exc'
).connect_all_to_all(weights=0.75)

CE_PPTN_CS = Projection(
    pre = CE,
    post = PPTN_CS,
    target = 'exc'
).connect_all_to_all(weights=1.3)

PPTN_US_VTA = Projection(
    pre = PPTN_US,
    post = VTA,
    target = 'exc'
).connect_all_to_all(weights=1.5)

PPTN_CS_VTA = Projection(
    pre = PPTN_CS,
    post = VTA,
    target = 'exc'
).connect_all_to_all(weights=1.5)

PPTN_PPTN = Projection(
    pre = PPTN_US,
    post = PPTN_CS,
    target = 'inh'
).connect_all_to_all(weights=2.0)

NAcc_VTA = Projection(
    pre = NAcc,
    post = VTA,
    target = 'mod',
    synapse = Hebb
).connect_all_to_all(weights=0.0)


# Projections to NAcc

BLA_NAcc = Projection(
    pre = BLA,
    post = NAcc,
    target = 'exc'
).connect_one_to_one(weights=0.2)

VTA_NAcc = Projection(
    pre = VTA,
    post = NAcc,
    target = 'dopa'
).connect_all_to_all(weights=1.0)


NAcc_NAcc = Projection(
    pre = NAcc,
    post = NAcc,
    target = 'inh',
    synapse = AntiHebb
).connect_all_to_all(weights=0.5)
NAcc_NAcc.tau = 1000.0  

vmPFC_NAcc = Projection(
    pre = vmPFC,
    post = NAcc,
    target = 'mod',
    synapse = DACovariance
).connect_all_to_all(weights=0.0)
vmPFC_NAcc.eta = 50.0
vmPFC_NAcc.tau_alpha = 10.0 
vmPFC_NAcc.tau_dopa = 10.0
vmPFC_NAcc.K_alpha = 10.0
vmPFC_NAcc.K_LTD = 10.0
vmPFC_NAcc.dopa_threshold_LTP = 0.3
vmPFC_NAcc.dopa_K_LTP = 5.0
vmPFC_NAcc.description['variables'][3]['bounds']['min'] = -0.2 # TODO!!!

# Projections within the ventral BG

PPTN_US_VP = Projection(
    pre = PPTN_US,
    post = VP,
    target = 'exc'
).connect_all_to_all(weights=0.5)

PPTN_CS_VP = Projection(
    pre = PPTN_CS,
    post = VP,
    target = 'exc'
).connect_all_to_all(weights=0.5)

NAcc_VP = Projection(
    pre = NAcc,
    post = VP,
    target = 'inh',
    synapse = Hebb
).connect_all_to_all(weights=0.0)
NAcc_VP.eta = 100.0
NAcc_VP.threshold_pre = 0.0
NAcc_VP.threshold_post = 0.5
NAcc_VP.description['variables'][0]['bounds']['max'] = 2.0

VP_RMTg = Projection(
    pre = VP,
    post = RMTg,
    target = 'inh'
).connect_all_to_all(weights=1.0)

VP_LHb = Projection(
    pre = VP,
    post = LHb,
    target = 'inh'
).connect_all_to_all(weights=3.0)
        
LHb_RMTg = Projection(
    pre = LHb,
    post = RMTg,
    target = 'exc'
).connect_all_to_all(weights=1.5)

RMTg_VTA = Projection(
    pre = RMTg,
    post = VTA,
    target = 'inh'
).connect_all_to_all(weights=1.0)


#############################
# Main loop
#############################
if __name__ == '__main__':
    # Compile the network
    compile()
    
    from TrialDefinition import *
    
    print 'Start sensitization'
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
    
    print 'Start conditioning'
    # Perform 10 conditioning trials per association
    trial_setup = [
        {'GUS': np.array([1., 1., 0., 0.]), 'VIS': np.array([1., 0., 0.]), 'magnitude': 0.8, 'duration': 2000},
        {'GUS': np.array([1., 0., 1., 0.]), 'VIS': np.array([0., 1., 0.]), 'magnitude': 0.5, 'duration': 3000},
        {'GUS': np.array([0., 0., 1., 1.]), 'VIS': np.array([0., 0., 1.]), 'magnitude': 1.0, 'duration': 4000}
    ]
    for trial in range(10):
        conditioning_trial(trial_setup)
        
    print 'Finished'
