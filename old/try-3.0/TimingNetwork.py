import ANNarchy
compiler=ANNarchy.Compiler()
compiler.add_module_path(['./modules'])
compiler.neurons = ['InputNeuron', 'LinearNeuron']
compiler.learning_rules = ['BCM', 'AntiHebb']
compiler.OMP_NUM_THREADS = 1
compiler.build()

from ANNarchy.core import *


class TimingNetwork(Network):
    '''Network for the learning of CS-US intervals.'''
    
    def __init__(self):
        Network.__init__(self)
        # Default parameter values
        self.nb_stim = 4
        self.nb_tastes = 4
        self.nb_bla = 50
    
    def build(self):
    
        self.create_populations()
        self.connect_populations()
        
    def create_populations(self):
    
        # Visual Input
        self.add(name="visual", neuron=InputNeuron, width=self.nb_stim)
        self.population("visual").set_parameters({'tau': 10.0})
        # Gustatory Input
        self.add(name="gustatory", neuron=InputNeuron, width=self.nb_tastes)
        self.population("gustatory").set_parameters({'tau': 10.0})       
        # BLA
        self.add(name="BLA", neuron=LinearNeuron, width=self.nb_bla)
        self.population("BLA").set_parameters({'tau': 10.0, 'threshold': 0.0, 'noise': 0.2})
        # BLA inhibitory interneurons
        self.add(name="BLA_inh", neuron=LinearNeuron, width=self.nb_bla/5)
        self.population("BLA_inh").set_parameters({'tau': 10.0, 'threshold': 0.0, 'noise': 0.2})
        
    
    def connect_populations(self):
    
        # gustatory -> BLA for US learning
        proj = self.connect(projection=all2all(pre='gustatory', post='BLA', connection_type='AMPA', value=RandomDistribution('uniform', [0.0, 0.5])),
                            learning_rule = BCM )
        proj.set_learning_parameters({'tau': 10., 'tau_theta': 1000., 'threshold_pre': 0.0})
    
        # BLA <-> BLA_inh for competition
        proj = self.connect(projection=all2all(pre='BLA', post='BLA_inh', connection_type='AMPA', value=0.0),
                            learning_rule = BCM)
        proj.set_learning_parameters({'tau': 10., 'tau_theta': 1000.0, 'threshold_pre': 0.0})
        proj = self.connect(projection=all2all(pre='BLA_inh', post='BLA', connection_type='GABA', value=0.0),
                            learning_rule = AntiHebb)
        proj.set_learning_parameters({'tau': 10., 'theta': 0.00001, 'max_value': 0.5, 'threshold_post': 0.0, 'threshold_pre': 0.0})
                    
                    
                    
                    
                    
