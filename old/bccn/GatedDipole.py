import ANNarchy, ANNarchy_LIB
from ANNarchy import *
from ANNarchy_LIB import *
from ConnectionPatterns import *

# Modules that should be imported in the C++ library
ANNarchy.annarchy_import = ['LinearNeuron', 'DopamineNeuron', 'FeedbackNeuron', 'GatedDipole', 'ConditioningTask']


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
        nb_bla=5
        
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

        # Drive inputs
        self.add(name="DRIVE", width=nb_gustatory_inputs, height=1, neuron=LinearNeuron)
        for neur in self.neurons(population="DRIVE")  :
            neur.tau=10.0
            neur.noise = 0.0
            neur.baseline = 1.0
       
        # DA neurons
        self.add(name="VTA", width=1, height=1, neuron=DopamineNeuron)
        for neur in self.neurons(population="VTA")  :
            neur.tau=30.0
            neur.tau_decrease=30.0
            neur.noise = 0.1
            neur.baseline = 0.5
       
        # LH-ON neurons
        self.add(name="LH_ON", width=nb_gustatory_inputs, height=1, neuron=GatedDipole)
        for neur in self.neurons(population="LH_ON")  :
            neur.tau=10.0
            neur.noise = 0.1
            neur.baseline = 0.0
            neur.threshold = 0.0
            neur.fb_exc = 0.5
            neur.fb_mod = 0.5
            neur.tau_adaptation_input = 2000.0
            neur.tau_adaptation_drive = 400.0
            neur.tau_adaptation_rate = 200.0
       
        # LH-OFF neurons
        self.add(name="LH_OFF", width=nb_gustatory_inputs, height=1, neuron=GatedDipole)
        for neur in self.neurons(population="LH_OFF")  :
            neur.tau=10.0
            neur.noise = 0.1
            neur.baseline = 0.0
            neur.threshold = 0.0
            neur.fb_exc = 0.5
            neur.fb_mod = 0.5
            neur.tau_adaptation_input = 2000.0
            neur.tau_adaptation_drive = 400.0
            neur.tau_adaptation_rate = 200.0

            
        # Connect the populations
        # Gated dipoles in LH
        self.connect(one2one(self, pre="GUS", post="LH_ON", connection_type="FF", value=1.0, delay=0))
        self.connect(one2one(self, pre="DRIVE", post="LH_ON", connection_type="DRIVE", value=0.5, delay=0))
        self.connect(one2one(self, pre="DRIVE", post="LH_OFF", connection_type="DRIVE", value=0.5, delay=0))
        self.connect(one2one(self, pre="LH_ON", post="LH_OFF", connection_type="LAT", value=2.0, delay=0))
        self.connect(one2one(self, pre="LH_OFF", post="LH_ON", connection_type="LAT", value=2.0, delay=0))

        print 'Network created.'
        return self


def main():
    """
    Default method called when running annarchy with the -r option.
    """
    # Create the network
    net=network().build()
    
    # Record activities
    import matplotlib.pyplot as plt
    lh=[]
    for t in range(500): # record
        net.step()
        lh.append( [net.population('LH_ON').neuron(0).rate, 
                    net.population('LH_ON').neuron(1).rate,  
                    net.population('LH_ON').neuron(2).rate, 
                    net.population('LH_OFF').neuron(0).rate, 
                    net.population('LH_OFF').neuron(1).rate,  
                    net.population('LH_OFF').neuron(2).rate]) 
    
    # Set US
    net.population('GUS').neuron(0).baseline=1.0
    net.population('GUS').neuron(1).baseline=1.0
    for t in range(500): # record
        net.step()
        lh.append( [net.population('LH_ON').neuron(0).rate, 
                    net.population('LH_ON').neuron(1).rate,  
                    net.population('LH_ON').neuron(2).rate, 
                    net.population('LH_OFF').neuron(0).rate, 
                    net.population('LH_OFF').neuron(1).rate,  
                    net.population('LH_OFF').neuron(2).rate]) 
    # Reset US
    net.population('GUS').neuron(0).baseline=0.0
    net.population('GUS').neuron(1).baseline=0.0
    for t in range(500): # record
        net.step()
        lh.append( [net.population('LH_ON').neuron(0).rate, 
                    net.population('LH_ON').neuron(1).rate,  
                    net.population('LH_ON').neuron(2).rate, 
                    net.population('LH_OFF').neuron(0).rate, 
                    net.population('LH_OFF').neuron(1).rate,  
                    net.population('LH_OFF').neuron(2).rate]) 
        
    # Set US
    net.population('GUS').neuron(0).baseline=1.0
    net.population('GUS').neuron(2).baseline=1.0
    for t in range(500): # record
        net.step()
        lh.append( [net.population('LH_ON').neuron(0).rate, 
                    net.population('LH_ON').neuron(1).rate,  
                    net.population('LH_ON').neuron(2).rate, 
                    net.population('LH_OFF').neuron(0).rate, 
                    net.population('LH_OFF').neuron(1).rate,  
                    net.population('LH_OFF').neuron(2).rate]) 
    # Reset US
    net.population('GUS').neuron(0).baseline=0.0
    net.population('GUS').neuron(2).baseline=0.0
    for t in range(500): # record
        net.step()
        lh.append( [net.population('LH_ON').neuron(0).rate, 
                    net.population('LH_ON').neuron(1).rate,  
                    net.population('LH_ON').neuron(2).rate, 
                    net.population('LH_OFF').neuron(0).rate, 
                    net.population('LH_OFF').neuron(1).rate,  
                    net.population('LH_OFF').neuron(2).rate]) 
                    
    for t in range(500): # record
        net.step()
        lh.append( [net.population('LH_ON').neuron(0).rate, 
                    net.population('LH_ON').neuron(1).rate,  
                    net.population('LH_ON').neuron(2).rate, 
                    net.population('LH_OFF').neuron(0).rate, 
                    net.population('LH_OFF').neuron(1).rate,  
                    net.population('LH_OFF').neuron(2).rate]) 
        
        
    plt.plot(lh)
    plt.legend(('Sugar ON', 'Salt ON', 'Fat ON', 'Sugar OFF', 'Salt OFF', 'Fat OFF'),
           'upper right', shadow=True, fancybox=True)
    plt.show()   
    
    
    exit(0)       
    return net
    
 
    
