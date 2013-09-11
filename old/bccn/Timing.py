import ANNarchy, ANNarchy_LIB
from ANNarchy import *
from ANNarchy_LIB import *
from ConnectionPatterns import *

# Modules that should be imported in the C++ library
ANNarchy.annarchy_import = ['LinearNeuron', 'FeedbackNeuron', 'DopamineNeuron', 'StriatalNeuron', 'Oscillator', 'DA_Covariance', 'AntiHebb', 'ConditioningTask']


class network(annarNetwork):
    """
    Main network class.
    """

    def build(self):
        """
        Creates the different populations and connections of the network.
        """
        nb_visual_inputs=2
        nb_gustatory_inputs=4
        nb_oscillators=50
        nb_nacc=5
        
        # Visual inputs
        self.add(name="VIS", width=nb_visual_inputs, height=1, neuron=LinearNeuron)
        for neur in self.neurons(population="VIS")  :
            neur.tau=10.0
            neur.noise = 0.0

        # Gustatory inputs
        self.add(name="GUS", width=nb_gustatory_inputs, height=1, neuron=LinearNeuron)
        for neur in self.neurons(population="GUS")  :
            neur.tau=10.0
            neur.noise = 0.0
       
        # DA neurons
        self.add(name="VTA", width=1, height=1, neuron=DopamineNeuron)
        for neur in self.neurons(population="VTA")  :
            neur.tau=30.0
            neur.tau_decrease=30.0
            neur.noise = 0.1
            neur.baseline = 0.5

        # Cortical oscillators
        self.add(name="vmPFC", width=nb_visual_inputs, height=nb_oscillators, neuron=Oscillator)
        min_freq=8.0
        max_freq=12.0
        for neur in self.neurons(population="vmPFC")  :
            neur.tau = 1.0
            neur.noise = 0.0
            neur.delay= 10
            neur.freq= min_freq + (max_freq- min_freq)* np.random.random()

        # DA neurons
        self.add(name="NAcc", width=nb_nacc, height=nb_nacc, neuron=StriatalNeuron)
        for neur in self.neurons(population="NAcc")  :
            neur.tau=10.0
            neur.baseline=-0.2
            neur.noise=0.2
            neur.threshold_up=0.0
            neur.threshold_down=0.5
            neur.tau_drive=500.0
            neur.mod_drive=0.7
            neur.threshold_nmda=1.2
            neur.threshold_dopa=0.8
       
        # Connect the populations
        self.connect(vertical_stripes(self, pre="VIS", post="vmPFC", connection_type="FF", value=1.0, delay=0))
        self.connect(all2all(self, pre="GUS", post="VTA", connection_type="FF", value=0.7, delay=0))
        self.connect(all2all(self, pre="VTA", post="NAcc", connection_type="DOPA", value=1.0, delay=0))
        self.connect(all2all(self, pre="NAcc", post="NAcc", connection_type="LAT", value=0.2, var_value=0.0, delay=0))
        self.connect(all2all(self, pre="vmPFC", post="NAcc", connection_type="FF", value=0.0, var_value=0.1,  delay=0)).set_learning_rule(learning_rule=DA_Covariance)
        
        # Define the learning rules
        for rule in self.learning_rules(post="NAcc", connection_type="FF"):
            rule.tau=10.0
            rule.min_value=-0.2
            rule.K_alpha=20.0
            rule.tau_alpha=10.0
            rule.DA_threshold_positive=0.75
            rule.DA_threshold_negative=0.25
            rule.DA_K_positive=1.0
            rule.DA_K_negative=1.0
            
        print 'Network created.'
        return self


def main():
    """
    Default method called when running annarchy with the -r option.
    """
    # Create the network
    net=network().build()
    env=ConditioningTask(net)
    net.perceive(env)
    #net.show()
    
    # Let it learn a couple of trials
    net.run(15*net.world().trial_duration)
    net.wait()
    
    # Record activities
    import CustomTests
    CustomTests.test_timing(net)  
    
    exit(0)       
    return net
    
 
    
