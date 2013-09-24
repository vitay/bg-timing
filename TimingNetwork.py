"""
Control of dopaminergic firing during conditioning.
"""

import ANNarchy
compiler=ANNarchy.Compiler()
compiler.add_module_path(['./modules'])
compiler.neurons = ['LinearNeuron',
                    'DopamineNeuron',
                    'GatedNeuron',
                    'PhasicNeuron',
                    'ModulatedPhasicNeuron',
                    'OscillatorNeuron',
                    'StriatalNeuron',
                    'ShuntingExcitationNeuron']
compiler.learning_rules = ['DA_Covariance',
                           'AntiHebb',
                           'Hebb']
compiler.precision = 'double'
compiler.openMP = False
#compiler.clean()
compiler.build()

from ANNarchy.core import *

class stripes(Projection):
    ''' Connection pattern for vmPFC'''
    def __init__(self, pre, post, connection_type, value, var_value=0.0, delay=0):
        Projection.__init__(self, pre, post, connection_type)
        self.value=value
        self.var_value=var_value
        self.delay=delay

    def connect(self):
        pre=self.net.population(self.pre)
        post=self.net.population(self.post)
        for rk in range(post.height):
            for v in range(post.width):
                ranks = [v]
                values= [self.value + self.var_value * (2.*np.random.random()*1) ]
                delays= [self.delay]
                post.neuron(v, rk).connect((self.synapse)(None, 0), self.pre, vector_int(ranks), self.connection_type, vector_float(values), vector_int(delays) )

class TimingNetwork(Network):
    '''Network for the learning of CS-US intervals.'''
    def __init__(self):
        Network.__init__(self)
        # Parameters
        self.noise = 0.1
        # Number of neurons
        self.nb_visual_inputs = 2
        self.nb_gustatory_inputs = 4
        self.nb_bla = 6
        self.nb_visual = 3
        self.nb_oscillators = 30
        self.nb_nacc = 6
        # Frequencies of the oscillators
        self.min_freq = 2.0
        self.max_freq = 8.0

    def build(self):

        self.create_populations()
        self.connect_populations()

    def create_populations(self):

        #######################
        # Inputs
        #######################

        # Visual Input
        self.add(name="VIS", width=self.nb_visual_inputs,
                 neuron=LinearNeuron)
        self.population("VIS").set_parameters({
            'tau': 10.0,
            'threshold': 0.0,
            'noise': self.noise
        })

        # Gustatory Input
        self.add(name="GUS", width=self.nb_gustatory_inputs,
                 neuron=LinearNeuron)
        self.population("GUS").set_parameters({
            'tau': 10.0,
            'noise': 0.0,
            'threshold': 0.0,
            'noise': self.noise
        })

        # Drive inputs
        self.add(name="DRIVE", width=self.nb_gustatory_inputs,
                 neuron=LinearNeuron)
        self.population("DRIVE").set_parameters({
            'tau': 10.0,
            'noise': 0.0,
            'threshold': 0.0,
            'noise': self.noise
        })
        self.population("DRIVE").set_variables({
            'baseline': 1.0
        })

        # IT representation of visual inputs
        self.add(name="IT", width=self.nb_visual, height=self.nb_visual,
                 neuron=LinearNeuron)
        self.population("IT").set_parameters({
            'tau': 10.0,
            'noise': 0.2,
            'threshold': 0.0 ,
            'noise': self.noise
        })


        #######################
        # Lateral hypothalamus
        #######################

        # ON channel
        self.add(name="LH_ON", width=self.nb_gustatory_inputs,
                 neuron=GatedNeuron)
        self.population("LH_ON").set_parameters({
            'tau': 10.0,
            'noise': self.noise,
            'threshold': 0.0,
#            'tau_adaptation_input': 2000.0,
#            'tau_adaptation_drive': 400.0,
#            'tau_adaptation_rate': 200.0
        })
        self.population("LH_ON").set_variables({ 'baseline': 0.0 })


        #######################
        # Amygdala
        #######################

        # BLA
        self.add(name="BLA", width=self.nb_bla, height=self.nb_bla,
                 neuron=ModulatedPhasicNeuron)
        self.population("BLA").set_parameters({
            'tau': 10.0,
            'noise': self.noise,
            'fb_mod': 0.1,
            'fb_exc': 0.6,
            'tau_adaptation': 500.0
        })
        self.population("BLA").set_variables({
            'baseline': -0.2
        })

        # CE
        self.add(name="CE", width=1,
                 neuron=LinearNeuron)
        self.population("CE").set_parameters({
            'tau': 10.0,
            'noise': self.noise,
            'threshold': 0.0
        })
        self.population("CE").set_variables({
            'baseline': 0.0
        })

        #######################
        # Basal Ganglia
        #######################

        # Ventromedial prefrontal cortex
        self.add(name="vmPFC", width=self.nb_visual_inputs, height=self.nb_oscillators,
                 neuron=OscillatorNeuron)
        self.population("vmPFC").set_parameters({
            'tau': 1.0,
            'noise': 0.0,
            'start_oscillate': 0.8,
            'stop_oscillate': 0.2
        })
        self.population("vmPFC").set_variables({
            'freq': self.min_freq + (self.max_freq- self.min_freq)* np.random.random(self.population("vmPFC").geometry),
            'phase': np.pi * np.random.random(self.population("vmPFC").geometry)
        })

        # Nucleus accumbens
        self.add(name="NAcc", width=self.nb_nacc, height=self.nb_nacc,
                 neuron=StriatalNeuron)
        self.population("NAcc").set_parameters({
            'tau': 5.0,
            'noise': self.noise,
            'threshold_up': 0.1,
            'threshold_down': 0.6,
            'tau_state': 400.0,
            'threshold_exc': 0.8,
            'threshold_dopa': 0.7
        })
        self.population("NAcc").set_variables({
            'baseline': -0.3
        })
        
        # Ventral Pallidum
        self.add(name="VP", width=1,
                 neuron=ShuntingExcitationNeuron)
        self.population("VP").set_parameters({
            'tau': 10.0,
            'noise': self.noise
        })
        self.population("VP").set_variables({
            'baseline': 0.5
        })

        #####################
        # Dopamine cells
        #####################

        # DA neurons in VTA
        self.add(name="VTA", width=1,
                 neuron=DopamineNeuron)
        self.population("VTA").set_parameters({
            'tau': 40.0,
            'tau_decrease': 40.0,
            'tau_modulation': 500.0,
            'noise': self.noise,
            'threshold': 0.0,
            'max_rate': 1.1
        })
        self.population("VTA").set_variables({
            'baseline': 0.5
        })


        # Rostromedial tegmental nucleus
        self.add(name="RMTg", width=1,
                 neuron=LinearNeuron)
        self.population("RMTg").set_parameters({
            'tau': 10.0,
            'noise': self.noise,
            'threshold': 0.5,
            'max_rate': 1.1
        })
        self.population("RMTg").set_variables({
            'baseline': 0.0
        })

        # Lateral Habenula
        self.add(name="LHb", width=1,
                 neuron=LinearNeuron)
        self.population("LHb").set_parameters({
            'tau': 10.0,
            'noise': self.noise,
            'threshold': 0.0,
            'max_rate': 1.1
        })
        self.population("LHb").set_variables({
            'baseline': 1.0
        })

        # Pedunculopontine nucleus
        self.add(name="PPTN", width=1,
                 neuron=PhasicNeuron)
        self.population("PPTN").set_parameters({
            'tau': 10.0,
            'noise': self.noise,
            'tau_adaptation': 100.0
        })
        self.population("PPTN").set_variables({
            'baseline': 0.0
        })

    def connect_populations(self):

        #######################
        # INPUTS
        #######################

        # Visual inputs to IT
        self.connect(fixed_number_pre(pre="VIS", post="IT", connection_type="exc",
                                      value=1.0, number=1, delay=0 ) )
                                      
        # Visual input to vmPFC
        self.connect(stripes(pre="VIS", post="vmPFC", connection_type="exc",
                             value=1.0, delay=0))

        # Gustatory input to PPTN
        self.connect(all2all(pre="LH_ON", post="PPTN", connection_type="exc",
                             value=0.5, delay=0))

        #######################
        # VTA and RMTg
        #######################

        # CE -> PPTN, exc
        self.connect(all2all(pre="CE", post="PPTN", connection_type="exc",
                             value=1.5, delay=0) )
        # PPTN -> VTA, exc
        self.connect(all2all(pre="PPTN", post="VTA", connection_type="exc",
                             value=1.5, delay=0) )

        # PPTN -> VP, exc
        self.connect(all2all(pre="PPTN", post="VP", connection_type="exc",
                             value=0.2, delay=0))
        # VP -> RMTg, inh
        self.connect(all2all(pre="VP", post="RMTg", connection_type="inh",
                             value=1.0, delay=0))
        # VP -> LHb, inh
        self.connect(all2all(pre="VP", post="LHb", connection_type="inh",
                             value=3.0, delay=0))

        # LHb -> RMTg, exc
        self.connect(all2all(pre="LHb", post="RMTg", connection_type="exc",
                             value=1.5, delay=0))

        # RMTg -> VTA, inh
        self.connect(all2all(pre="RMTg", post="VTA", connection_type="inh",
                             value=1.0, delay=0))


        #######################
        # Lateral hypothalamus
        #######################

        # Gustatory input to the ON channel
        self.connect(one2one(pre="GUS", post="LH_ON", connection_type="exc",
                             value=1.0, delay=0))
        # Drive to the ON channel
        self.connect(one2one(pre="DRIVE", post="LH_ON", connection_type="drive",
                             value=0.5, delay=0))

        #######################
        # Amygdala
        #######################

        # DA to BLA, dopa
        self.connect(all2all(pre="VTA", post="BLA", connection_type="dopa",
                             value=1.0,  delay=0) )

        # LH_ON to BLA, exc (US)
        proj = self.connect(all2all(pre="LH_ON", post="BLA", connection_type="exc",
                                    value=0.5, var_value= 0.1, delay=0),
                            learning_rule=DA_Covariance )
        proj.set_learning_parameters({
            'tau': 100.0,
            'min_value': 0.0,
            'K_alpha': 10.0,
            'tau_alpha': 1.0,
            'regularization_threshold': 1.0,
            'DA_threshold_positive': 0.6,
            'DA_threshold_negative': 0.1,
            'DA_K_positive': 10.0,
            'DA_K_negative': 10.0
        })

        # IT to BLA, mod (CS)
        proj = self.connect(all2all(pre="IT", post="BLA", connection_type="mod",
                                    value=0.0, delay=0),
                            learning_rule=DA_Covariance )
        proj.set_learning_parameters({
            'tau': 500.0,
            'min_value': 0.0,
            'K_alpha': 10.0,
            'tau_alpha': 1.0,
            'regularization_threshold': 1.0,
            'DA_threshold_positive': 0.6,
            'DA_threshold_negative': 0.1,
            'DA_K_positive': 3.0,
            'DA_K_negative': 1.0
        })

        # Competition in BLA, inh
        proj = self.connect(all2all(pre="BLA", post="BLA", connection_type="inh",
                                    value=0.6, var_value= 0.1, delay=0),
                            learning_rule=AntiHebb )
        proj.set_learning_parameters({
            'tau': 100.0,
            'theta': 0.001,
            'min_value': 0.0,
            'max_value': 2.0
        })

        # BLA to CE, exc
        self.connect(all2all(pre="BLA", post="CE", connection_type="exc",
                             value=1.0, delay=0))


        #######################
        # Basal Ganglia
        #######################

        # Dopaminergic modulation of NAcc
        self.connect(all2all(pre="VTA", post="NAcc", connection_type="dopa",
                             value=1.0, delay=0))

        # Competition in NAcc
        proj = self.connect(all2all(pre="NAcc", post="NAcc", connection_type="inh",
                             value=0.6, delay=0),
                             learning_rule=AntiHebb )
        proj.set_learning_parameters({
            'tau': 1000.0,
            'theta': 0.01,
            'min_value': 0.0,
            'max_value': 1.0
        })

        # Timing information from vmPFC to NAcc
        proj = self.connect(all2all(pre="vmPFC", post="NAcc", connection_type="mod",
                                    value=0.0, var_value=0.05,  delay=0),
                            learning_rule=DA_Covariance)
        proj.set_learning_parameters({
            'tau': 20.0,
            'K_LTD': 10.0,
            'min_value': -0.2,
            'K_alpha': 10.0,
            'tau_alpha': 10.0,
            'tau_dopa': 10.0,
            'regularization_threshold': 1.0,
            'DA_threshold_positive': 0.6,
            'DA_threshold_negative': 0.1,
            'DA_K_positive': 5.0,
            'DA_K_negative': 1.0
        })
        
#        # Reward information from BLA to NAcc
#        proj = self.connect(all2all(pre="BLA", post="NAcc", connection_type="mod",
#                                    value=0.0, var_value=0.0,  delay=0),
#                            learning_rule=DA_Covariance)
#        proj.set_learning_parameters({
#            'tau': 500.0,
#            'min_value': 0.0,
#            'K_alpha': 5.0,
#            'tau_alpha': 1.0,
#            'regularization_threshold': 0.8,
#            'DA_threshold_positive': 0.7,
#            'DA_threshold_negative': 0.3,
#            'DA_K_positive': 4.0,
#            'DA_K_negative': 1.0
#        })
        

        # Inhibitory projection from NAcc to VP
        proj = self.connect(all2all(pre="NAcc", post="VP", connection_type="inh",
                                    value=0.1, var_value=0.0, delay=0),
                            learning_rule=Hebb)
        proj.set_learning_parameters({
            'tau': 20.0,
            'min_value': 0.0,
            'max_value': 1.0,
            'threshold_pre' : 0.0,
            'threshold_post' : 0.5,
        })

        # NAcc -> VTA, mod
        proj = self.connect(all2all(pre="NAcc", post="VTA", connection_type="mod",
                                    value=0.0, var_value=0.0, delay=0),
                            learning_rule = Hebb )
        proj.set_learning_parameters({
            'tau': 100.0,
            'min_value': 0.0,
            'max_value': 10.0,
            'threshold_pre' : 0.0,
            'threshold_post' : 0.5
        })



# Habituate the network to gustatory inputs
def valuation_trial(net, US=None):
    # Reset the network for 500ms
    for neur in net.population('GUS'):
        neur.baseline = 0.0
    net.execute(500)
    # Select the US randomly if not given
    if not US:
        US = np.random.randint(2) + 1
    GUS = [1.0, 0.0, 0.0, 0.0]
    GUS[US] = 1.0
    print 'Valuation: ', GUS
    # Let the network learn for 1s
    net.population('GUS').set_variables({'baseline': GUS})
    net.execute(1000)
    # Reset the network for 500ms
    for neur in net.population('GUS'):
        neur.baseline = 0.0
    net.execute(500)

# Perform timed conditioning
def conditioning_trial(net, CS=None):
    # Reset for 1s
    for neur in net.population('VIS'):
        neur.baseline = 0.0
    for neur in net.population('GUS'):
        neur.baseline = 0.0
    net.execute(1000)
    # Select the CS randomly if not given
    if not CS:
        CS = np.random.randint(2) + 1
    VIS = [0.0, 0.0]
    GUS = [1.0, 0.0, 0.0, 0.0]
    VIS[CS-1] = 1.0
    GUS[CS] = 1.0
    print 'Conditioning: ', VIS, GUS
    # Present CS1 for 2s, CS2 for 3s
    net.population('VIS').set_variables({'baseline': VIS})
    net.execute(1000*(CS+1))
    # Present the US for 1s
    net.population('GUS').set_variables({'baseline': GUS})
    net.execute(1000)
    # Reset for 1s
    for neur in net.population('VIS'):
        neur.baseline = 0.0
    for neur in net.population('GUS'):
        neur.baseline = 0.0
    net.execute(1000)

# Extinction
def extinction_trial(net, CS=None):
    # Reset for 1s
    for neur in net.population('VIS'):
        neur.baseline = 0.0
    for neur in net.population('GUS'):
        neur.baseline = 0.0
    net.execute(1000)
    # Select the CS randomly if not given
    if not CS:
        CS = np.random.randint(2) + 1
    VIS = [0.0, 0.0]
    GUS = [1.0, 0.0, 0.0, 0.0]
    VIS[CS-1] = 1.0
    GUS[CS] = 1.0
    print 'Conditioning: ', VIS, GUS
    # Present CS1 for 2s, CS2 for 3s
    net.population('VIS').set_variables({'baseline': VIS})
    net.execute(1000*(CS+1))
    # DO NOT Present the US for 1s
    #net.population('GUS').set_variables({'baseline': GUS})
    net.execute(1000)
    # Reset for 1s
    for neur in net.population('VIS'):
        neur.baseline = 0.0
    for neur in net.population('GUS'):
        neur.baseline = 0.0
    net.execute(1000)

# Reward is delivered sooner than expected
def sooner_trial(net, CS=None):
    # Reset for 1s
    for neur in net.population('VIS'):
        neur.baseline = 0.0
    for neur in net.population('GUS'):
        neur.baseline = 0.0
    net.execute(1000)
    # Select the CS randomly if not given
    if not CS:
        CS = np.random.randint(2) + 1
    VIS = [0.0, 0.0]
    GUS = [1.0, 0.0, 0.0, 0.0]
    VIS[CS-1] = 1.0
    GUS[CS] = 1.0
    print 'Conditioning: ', VIS, GUS
    # Present CS1 for 1s, CS2 for 2s: sooner than what was learned
    net.population('VIS').set_variables({'baseline': VIS})
    net.execute(1000*(CS))
    # Present the US for 1s
    net.population('GUS').set_variables({'baseline': GUS})
    net.execute(1000)
    # Reset for 1s
    for neur in net.population('VIS'):
        neur.baseline = 0.0
    for neur in net.population('GUS'):
        neur.baseline = 0.0
    net.execute(1000)
