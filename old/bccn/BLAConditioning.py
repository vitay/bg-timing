import ANNarchy, ANNarchy_LIB
from ANNarchy import *
from ANNarchy_LIB import *
from ConnectionPatterns import *
from CustomTests import *

# Modules that should be imported in the C++ library
ANNarchy.annarchy_import = ['LinearNeuron', 'DopamineNeuron', 'FeedbackNeuron', 'PhasicNeuron', 'GatedDipole', 'ConditioningTask', 'ValuationTask', 'DA_Covariance', 'Hebb', 'AntiHebb']


class network(annarNetwork):
    """
    Main network class.
    """

    def build(self):
        """
        Creates the different populations and connections of the network.
        """
        nb_visual_inputs = 2
        nb_gustatory_inputs = 4
        nb_bla = 5
        nb_visual = 3
        
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
      
        # IT representation of visual inputs
        self.add(name="IT", width=nb_visual, height=nb_visual, neuron=LinearNeuron)
        for neur in self.neurons(population="IT")  :
            neur.tau=10.0
            neur.noise = 0.3
            
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
            neur.baseline = - 0.2
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
            neur.baseline = - 0.2
            neur.threshold = 0.0
            neur.fb_exc = 0.5
            neur.fb_mod = 0.5
            neur.tau_adaptation_input = 2000.0
            neur.tau_adaptation_drive = 400.0
            neur.tau_adaptation_rate = 200.0

        # CS_US association in BLA amygdala
        self.add(name="BLA", width=nb_bla, height=nb_bla, neuron=PhasicNeuron)
        for neur in self.neurons(population="BLA")  :
            neur.tau=10.0
            neur.baseline=-0.2
            neur.noise=0.1
            neur.fb_mod=0.1
            neur.fb_exc=0.5
            neur.tau_adaptation=10000.0
            
        # Bodily reactions in CE amygdala
        self.add(name="CE", width=1, height=1, neuron=LinearNeuron)
        for neur in self.neurons(population="CE")  :
            neur.tau=10.0
            neur.baseline=0.0
            neur.noise=0.1
            
        # Connect the populations
        # Gated dipoles in LH
        self.connect(one2one(self, pre="GUS", post="LH_ON", connection_type="FF", value=1.0, delay=0))
        self.connect(one2one(self, pre="DRIVE", post="LH_ON", connection_type="DRIVE", value=0.5, delay=0))
        self.connect(one2one(self, pre="DRIVE", post="LH_OFF", connection_type="DRIVE", value=0.5, delay=0))
        self.connect(one2one(self, pre="LH_ON", post="LH_OFF", connection_type="LAT", value=2.0, delay=0))
        self.connect(one2one(self, pre="LH_OFF", post="LH_ON", connection_type="LAT", value=2.0, delay=0))
        
        # Pavlovian conditioning in BLA
        self.connect(cluster(self, pre="VIS", post="IT", connection_type="FF", value=1.0, cluster_size=-1, delay=0))
        self.connect(all2all(self, pre="VTA", post="BLA", connection_type="DOPA", value=1.0,  delay=0)).set_learning_rule(learning_rule=DA_Covariance)
        self.connect(all2all(self, pre="GUS", post="BLA", connection_type="FF", value=0.5, var_value= 0.1,  delay=0)).set_learning_rule(learning_rule=DA_Covariance)
        self.connect(all2all(self, pre="BLA", post="BLA", connection_type="LAT", value=0.4, var_value= 0.1, delay=0)).set_learning_rule(learning_rule=AntiHebb)        
        self.connect(all2all(self, pre="IT", post="BLA", connection_type="FB", min_value=0.0, max_value=0.0, delay=0)).set_learning_rule(learning_rule=DA_Covariance) 
        self.connect(all2all(self, pre="BLA", post="CE", connection_type="FF", value=0.9, delay=0))#.set_learning_rule(learning_rule=Hebb)
        
        # VTA control
        self.connect(all2all(self, pre="GUS", post="VTA", connection_type="FF", value=1.0, delay=0))#.set_learning_rule(learning_rule=Hebb)
        #self.connect(all2all(self, pre="CE", post="VTA", connection_type="FF", value=1.8, delay=0))#.set_learning_rule(learning_rule=Hebb)

        
        # Define the learning rules
        for rule in self.learning_rules(post="BLA", connection_type="FF"):
            rule.tau=10.0
            rule.min_value=-0.2
            rule.K_alpha=100.0
            rule.tau_alpha=10.0
            rule.DA_threshold_positive=0.6
            rule.DA_threshold_negative=0.4
            rule.DA_K_positive=10.0
            rule.DA_K_negative=10.0
            
        for rule in self.learning_rules(post="BLA", connection_type="FB"):
            rule.tau=100.0
            rule.min_value=-0.2
            rule.K_alpha=100.0
            rule.tau_alpha=10.0
            rule.DA_threshold_positive=0.6
            rule.DA_threshold_negative=0.4
            rule.DA_K_positive=3.0
            rule.DA_K_negative=10.0
            
        for rule in self.learning_rules(post="BLA", connection_type="LAT"):
            rule.tau=100.0
            rule.theta=0.001
            rule.max=2.0
            
        for rule in self.learning_rules(post="CE", connection_type="FF"):
            rule.tau=1000000.0
            rule.K_alpha=100000.0
            rule.tau_alpha=10.0
            
#        for rule in self.learning_rules(post="VTA", connection_type="FF"):
#            rule.tau=10000.0
#            rule.K_alpha=10.0
#            rule.tau_alpha=1000.0
                
        print 'Network created.'
        return self


def main():
    """
    Default method called when running annarchy with the -r option.
    """
    # Create the network
    net=network().build()
    
    # Habituate to the US
    env_us=ValuationTask(net)
    net.perceive(env_us)
    print 'Starting habituation to the US...'
    net.run(10*net.world().trial_duration)
    net.wait()
    
    # Store results
    bla_before=[]
    bla_after=[]
    
    # Reset inputs
    net.learn=False
    net.world().stop()
    net.world().remove_inputs()
    for t in range(1000): # record for 500ms
        net.step()
        bla_before.append( [neur.rate for neur in net.neurons('BLA')] ) 
    # Do one ensemble of trials
    #net.world().start()
    net.population('VIS').neuron(0).baseline=1.0
    for t in range(2000): # record for 5s
        net.step()
        bla_before.append( [neur.rate for neur in net.neurons('BLA')] ) 
    net.population('GUS').neuron(0).baseline=1.0
    net.population('GUS').neuron(1).baseline=1.0
    for t in range(1000): # record for 5s
        net.step()
        bla_before.append( [neur.rate for neur in net.neurons('BLA')] ) 
    net.population('VIS').neuron(0).baseline=0.0
    net.population('GUS').neuron(0).baseline=0.0
    net.population('GUS').neuron(1).baseline=0.0
    for t in range(1000): # record for 5s
        net.step()
        bla_before.append( [neur.rate for neur in net.neurons('BLA')] ) 
    
    
    
    # Associate with the CS
    env_cs=ConditioningTask(net)
    net.perceive(env_cs)
    net.learn=True
    print 'Starting conditioning with the CS...'
    net.run(5*net.world().trial_duration)
    net.wait()
    
    # Reset inputs
    net.learn=False
    net.world().stop()
    net.world().remove_inputs()
    for t in range(1000): # record for 500ms
        net.step()
        bla_after.append( [neur.rate for neur in net.neurons('BLA')] ) 
    # Do one ensemble of trials
    #net.world().start()
    net.population('VIS').neuron(0).baseline=1.0
    for t in range(2000): # record for 5s
        net.step()
        bla_after.append( [neur.rate for neur in net.neurons('BLA')] ) 
    net.population('GUS').neuron(0).baseline=1.0
    net.population('GUS').neuron(1).baseline=1.0
    for t in range(1000): # record for 5s
        net.step()
        bla_after.append( [neur.rate for neur in net.neurons('BLA')] ) 
    net.population('VIS').neuron(0).baseline=0.0
    net.population('GUS').neuron(0).baseline=0.0
    net.population('GUS').neuron(1).baseline=0.0
    for t in range(1000): # record for 5s
        net.step()
        bla_after.append( [neur.rate for neur in net.neurons('BLA')] ) 
    
        
    # Plot everything
    plt.subplot(121)
    imgplot = plt.imshow(np.array(bla_before).T, aspect='auto')
    #plt.colorbar()
    plt.subplot(122)
    imgplot = plt.imshow(np.array(bla_after).T, aspect='auto')
    plt.colorbar()
    
    plt.show()  
    
    exit(0)       
    return net
    
 
    
