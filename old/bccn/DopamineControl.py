import ANNarchy, ANNarchy_LIB
from ANNarchy import *
from ANNarchy_LIB import *
from ConnectionPatterns import *
from CustomTests import *

# Modules that should be imported in the C++ library
ANNarchy.annarchy_import = ['LinearNeuron', 'DopamineNeuron', 'FeedbackNeuron', 'PhasicNeuron', 'TonicNeuron', 'GatedDipole', 'GatedNeuron', 'StriatalNeuron', 'DiffNeuron', 'Oscillator', 'ConditioningTask', 'ValuationTask', 'Covariance', 'DA_Covariance', 'GABA_Tonic', 'Hebb', 'AntiHebb']

nb_visual_inputs = 2
nb_gustatory_inputs = 4
nb_bla = 5
nb_visual = 3
nb_oscillators = 30
nb_nacc = 6

class network(annarNetwork):
    """
    Main network class.
    """

    def build(self):
        """
        Creates the different populations and connections of the network.
        """
        #####################
        # Creates inputs
        #####################
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

        #####################
        # Dopamine cells
        #####################
            
        # DA neurons
        self.add(name="VTA", width=1, height=1, neuron=DopamineNeuron)
        for neur in self.neurons(population="VTA")  :
            neur.tau=30.0
            neur.tau_decrease=30.0
            neur.noise = 0.1
            neur.baseline = 0.5

        # GABergic cells in VTA
        self.add(name="VTA_GABA", width=1, height=1, neuron=DiffNeuron)
        for neur in self.neurons(population="VTA_GABA")  :
            neur.tau=10.0
            neur.baseline=0.0
            neur.noise=0.1
            neur.threshold=0.4
            
        #####################
        # Gated dipoles in LH
        #####################
        self.build_LH()

        #####################
        # CS_US association in amygdala
        #####################
        self.build_amygdala()
        
        ####################
        # Timed control in NAcc
        ####################
        self.build_timing()
        
        ####################
        # Interconnections
        ####################     
        self.connect(cluster(self, pre="VIS", post="IT", connection_type="FF", value=1.0, cluster_size=-1, delay=0))
       
        # VTA control
        self.connect(all2all(self, pre="LH_ON", post="VTA", connection_type="FF", value=1.0, delay=0))
        self.connect(all2all(self, pre="CE", post="VTA", connection_type="FF", value=1.8, delay=0))
        self.connect(all2all(self, pre="NAcc", post="VTA", connection_type="GABA", value=0.0, var_value=0.0, delay=0)).set_learning_rule(learning_rule=Hebb)
        
        self.connect(all2all(self, pre="LHb", post="VTA_GABA", connection_type="FF", value=1.0, var_value=0.0, delay=0))
        self.connect(all2all(self, pre="PPTN", post="VTA_GABA", connection_type="GABA", value=1.0, var_value=0.0, delay=0))
        self.connect(all2all(self, pre="VP", post="VTA_GABA", connection_type="GABA", value=1.0, var_value=0.0, delay=0))
        self.connect(all2all(self, pre="VTA_GABA", post="VTA", connection_type="LAT", value=2.0, var_value=0.0, delay=0))
            
        for rule in self.learning_rules(post="VTA", connection_type="GABA"):
            rule.tau=2000.0
            rule.K_alpha=0.0
            rule.tau_alpha=1.0
                
        print 'Network created.'
        return self


    def build_LH(self):
        """
        Lateral hypothalamus
        """              
        # LH-ON neurons
        self.add(name="LH_ON", width=nb_gustatory_inputs, height=1, neuron=GatedDipole)
        for neur in self.neurons(population="LH_ON")  :
            neur.tau=10.0
            neur.noise = 0.1
            neur.baseline = -0.2
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

        # Gated dipoles in LH
        self.connect(one2one(self, pre="GUS", post="LH_ON", connection_type="FF", value=1.0, delay=0))
        self.connect(one2one(self, pre="DRIVE", post="LH_ON", connection_type="DRIVE", value=0.5, delay=0))
        self.connect(one2one(self, pre="DRIVE", post="LH_OFF", connection_type="DRIVE", value=0.5, delay=0))
        self.connect(one2one(self, pre="LH_ON", post="LH_OFF", connection_type="LAT", value=2.0, delay=0))
        #self.connect(one2one(self, pre="LH_OFF", post="LH_ON", connection_type="LAT", value=2.0, delay=0))
        
    def build_amygdala(self):
        """
        Amygdala = BLA + CE
        """
        # CS-US association in BLA    
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
            
        # Connections
        self.connect(all2all(self, pre="VTA", post="BLA", connection_type="DOPA", value=1.0,  delay=0)).set_learning_rule(learning_rule=DA_Covariance)
        self.connect(all2all(self, pre="LH_ON", post="BLA", connection_type="FF", value=0.5, var_value= 0.1,  delay=0)).set_learning_rule(learning_rule=DA_Covariance)
        self.connect(all2all(self, pre="BLA", post="BLA", connection_type="LAT", value=0.6, var_value= 0.1, delay=0)).set_learning_rule(learning_rule=AntiHebb)        
        self.connect(all2all(self, pre="IT", post="BLA", connection_type="FB", min_value=0.0, max_value=0.0, delay=0)).set_learning_rule(learning_rule=DA_Covariance) 
        self.connect(all2all(self, pre="BLA", post="CE", connection_type="FF", value=0.9, delay=0))#.set_learning_rule(learning_rule=Hebb)
        
        # Define the learning rules
        for rule in self.learning_rules(post="BLA", connection_type="FF"):
            rule.tau=10.0
            rule.min_value=-0.2
            rule.K_alpha=100.0
            rule.tau_alpha=10.0
            rule.DA_threshold_positive=0.7
            rule.DA_threshold_negative=0.4
            rule.DA_K_positive=10.0
            rule.DA_K_negative=10.0
            
        for rule in self.learning_rules(post="BLA", connection_type="FB"):
            rule.tau=100.0
            rule.min_value=-0.2
            rule.K_alpha=50.0
            rule.tau_alpha=10.0
            rule.DA_threshold_positive=0.6
            rule.DA_threshold_negative=0.4
            rule.DA_K_positive=3.0
            rule.DA_K_negative=10.0
            
        for rule in self.learning_rules(post="BLA", connection_type="LAT"):
            rule.tau=100.0
            rule.theta=0.001
            rule.max=2.0
            
        #for rule in self.learning_rules(post="CE", connection_type="FF"):
        #    rule.tau=1000000.0
        #    rule.K_alpha=100000.0
        #    rule.tau_alpha=10.0
        
    def build_timing(self):
        """
        Creates the different populations and connections of the network.
        """

        # Cortical oscillators
        self.add(name="vmPFC", width=nb_visual_inputs, height=nb_oscillators, neuron=Oscillator)
        min_freq=0.5
        max_freq=2.0
        for neur in self.neurons(population="vmPFC")  :
            neur.tau = 1.0
            neur.noise = 0.0
            neur.delay= 0
            neur.freq= min_freq + (max_freq- min_freq)* np.random.random()

        # Nucleus accumbens
        self.add(name="NAcc", width=nb_nacc, height=nb_nacc, neuron=StriatalNeuron)
        for neur in self.neurons(population="NAcc")  :
            neur.tau=10.0
            neur.baseline=-0.2
            neur.noise=0.0
            neur.threshold_up=0.0
            neur.threshold_down=0.5
            neur.tau_drive=200.0
            neur.threshold_nmda=1.2
            neur.threshold_dopa=0.8

        # Ventral Pallidum
        self.add(name="VP", width=1, height=1, neuron=TonicNeuron)
        for neur in self.neurons(population="VP")  :
            neur.tau=10.0
            neur.baseline=1.0
            neur.noise=0.0

        # Lateral Habenula
        self.add(name="LHb", width=1, height=1, neuron=TonicNeuron)
        for neur in self.neurons(population="LHb")  :
            neur.tau=10.0
            neur.baseline=1.0
            neur.noise=0.1

        # Pedunculopontine nucleus
        self.add(name="PPTN", width=1, height=1, neuron=GatedNeuron)
        for neur in self.neurons(population="PPTN")  :
            neur.tau=10.0
            neur.baseline=-0.1
            neur.noise=0.0
            neur.tau_adaptation=100000.0
       
        # Connect the populations
        self.connect(vertical_stripes(self, pre="VIS", post="vmPFC", connection_type="FF", value=1.0, delay=0))
        self.connect(all2all(self, pre="VTA", post="NAcc", connection_type="DOPA", value=1.0, delay=0))
        self.connect(all2all(self, pre="NAcc", post="NAcc", connection_type="LAT", value=0.2, var_value=0.0, delay=0))
        self.connect(all2all(self, pre="vmPFC", post="NAcc", connection_type="FF", value=0.0, var_value=0.2,  delay=0)).set_learning_rule(learning_rule=DA_Covariance)
        self.connect(all2all(self, pre="NAcc", post="VP", connection_type="GABA", value=0.1, var_value=0.03, delay=0)).set_learning_rule(learning_rule=GABA_Tonic)
        self.connect(all2all(self, pre="VP", post="LHb", connection_type="GABA", value=1.0, var_value=0.0, delay=0))
        self.connect(all2all(self, pre="VP", post="PPTN", connection_type="GABA", value=1.0, var_value=0.0, delay=0))
        self.connect(all2all(self, pre="GUS", post="PPTN", connection_type="FF", value=0.8, var_value=0.0, delay=0))
        
        # Define the learning rules
        for rule in self.learning_rules(post="NAcc", connection_type="FF"):
            rule.tau=20.0
            rule.min_value=-0.5
            rule.K_alpha=5.0
            rule.tau_alpha=10.0
            rule.DA_threshold_positive=0.75
            rule.DA_threshold_negative=0.25
            rule.DA_K_positive=1.0
            rule.DA_K_negative=1.0
            
        for rule in self.learning_rules(post="VP", connection_type="GABA"):
            rule.tau=500.0
            rule.min_value=0.0
            rule.max_value=1.0



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
    
    # Associate with the CS
    env_cs=ConditioningTask(net)
    net.perceive(env_cs)
    print 'Starting conditioning with the CS...'
    
    # Main experiment with 9 CS-US trials and 1 extinction
    tentrials(net)
        
    # Test learning in BLA
    #test_bla(net)
    
    # Test Gated dipoles
    #test_gated_dipole(net)
    
    # Test oscillators
    #test_oscillators(net)
    
    # Test reward amplitude and drive
    #test_reward(net)
    
    exit(0)       
    return net
    
def tentrials(net):

    
    # Record CS activities before learning
    da=[]
    da_inh=[]
    nacc=[]
    vp=[]
    lhb=[]
    pptn=[]
    vtagaba=[]
    bla=[]
    for t in range(9): # 9 CS-US pairings
        dat, dainht, nacct, vpt, lhbt, pptnt, vtagabat, lhont, blat = record_trial(net) 
        da.append(dat)
        da_inh.append(dainht)
        vp.append(vpt)
        lhb.append(lhbt)
        pptn.append(pptnt)
        vtagaba.append(vtagabat)
        nacc.append(np.sum(np.array(nacct), axis=1))
        bla.append(np.sum(np.array(blat), axis=1))
        
    net.world().us(False)
    for t in range(1): # 1 CS extinction
        dat, dainht, nacct, vpt, lhbt, pptnt, vtagabat, lhont, blat = record_trial(net) 
        da.append(dat)
        da_inh.append(dainht)
        vp.append(vpt)
        lhb.append(lhbt)
        pptn.append(pptnt)
        vtagaba.append(vtagabat)
        nacc.append(np.sum(np.array(nacct), axis=1))
        bla.append(np.sum(np.array(blat), axis=1))
    
    # Plot everything
    import matplotlib.pyplot as plt
#    for t in range(5):
#        plt.subplot(2, 5, t+1)
#        plt.plot(da[t]) 
#        #plt.plot(da_inh[t]) 
#        #plt.plot(nacc[t]) 
#        #plt.plot(vp[t]) 
#        #plt.plot(lhb[t]) 
#        #plt.plot(pptn[t]) 
#        #plt.plot(vtagaba[t]) 
#        #plt.plot(bla[t]) 
#        plt.ylim((0.0, 1.2))  
#    for t in range(5):
#        plt.subplot(2, 5, t+6)
#        plt.plot(da[t+5]) 
#        #plt.plot(da_inh[t+5]) 
#        #plt.plot(nacc[t+5])  
#        #plt.plot(vp[t+5]) 
#        #plt.plot(lhb[t+5]) 
#        #plt.plot(pptn[t+5]) 
#        #plt.plot(vtagaba[t+5]) 
#        #plt.plot(bla[t+5]) 
#        plt.ylim((0.0, 1.2))  
#    #plt.savefig('DA.svg', transparency=True)
#    plt.show()
 
     
    plt.subplot(2, 2, 1)
    plt.plot(da[0]) 
    plt.title('Before learning (1th trial)')
    plt.ylabel('Activity in VTA')
    plt.ylim((0.0, 1.2))
    plt.subplot(2, 2, 2)
    plt.plot(da[3]) 
    plt.title('During learning (5th trial)')
    plt.ylabel('Activity in VTA')
    plt.ylim((0.0, 1.2))
    plt.subplot(2, 2, 3)
    plt.plot(da[8])
    plt.title('After learning (9th trial)')
    plt.ylabel('Activity in VTA')
    plt.xlabel('Time (ms)')
    plt.ylim((0.0, 1.2))
    plt.subplot(2, 2, 4)
    plt.plot(da[9])  
    plt.title('Reward omitted (10th trial)')
    plt.ylabel('Activity in VTA')
    plt.xlabel('Time (ms)')
    plt.ylim((0.0, 1.2))
    plt.show()
    
def test_bla(net):
        
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
    net.run(3*net.world().trial_duration)
    net.wait()
    
    # Reset inputs
    net.learn=False
    net.world().stop()
    net.world().remove_inputs()
    for t in range(1000): # record for 500ms
        net.step()
        bla_after.append( [neur.rate for neur in net.neurons('BLA')] ) 
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
    
    # Select the maximally active cell
    bla_before=np.array(bla_before)
    bla_after=np.array(bla_after)
    before=bla_before[:,(np.sum(bla_before, axis=0)).argmax()]
    after=bla_after[:,(np.sum(bla_after, axis=0)).argmax()]

    # Plot everything
    plt.subplot(121)
    plt.plot(before)
    plt.title('After 10 exposures to the US alone.')
    plt.xlabel('Time (ms)')
    plt.ylabel('Activity of a selective cell in BLA')
    plt.ylim((0.0, 1.4))  
    plt.subplot(122)
    plt.plot(after)
    plt.title('After 5 CS-US associations.')
    plt.xlabel('Time (ms)')
    plt.ylabel('Activity of a selective cell in BLA')
    plt.ylim((0.0, 1.4))
    
    plt.savefig('BLA.pdf', transparency=True)
    plt.show()  
  
def test_oscillators(net):

    # Store results
    vmpfc=[]
    nacc=[]
    
    # learn
    net.run(10*net.world().trial_duration)
    net.wait()
    
    # Reset inputs
    net.learn=False
    net.world().stop()
    net.world().remove_inputs()
    for t in range(500): # record for 500ms
        net.step()
        vmpfc.append( [neur.rate for neur in net.neurons('vmPFC')] )
        nacc.append( [neur.rate for neur in net.neurons('NAcc')] )
    net.population('VIS').neuron(0).baseline=1.0  
    for t in range(3000): # record for 500ms
        net.step()
        vmpfc.append( [neur.rate for neur in net.neurons('vmPFC')] )  
        nacc.append( [neur.rate for neur in net.neurons('NAcc')] )
        
    plt.subplot(121)    
    plt.imshow(np.array(vmpfc).T, aspect='auto', cmap=plt.cm.hot)
    plt.title('Activity in vmPFC')
    plt.xlabel('Time (ms)')
    plt.ylabel('# neurons')
    plt.subplot(122)    
    plt.imshow(np.array(nacc).T, aspect='auto', cmap=plt.cm.hot)
    plt.title('Activity in NAcc. Learned delay = 2s')
    plt.xlabel('Time (ms)')
    plt.ylabel('# neurons')
    plt.show() 
     
def test_reward(net):

    # Store results
    da_amp=[]
    da_drive=[]
    
    
    # Reset inputs
    net.learn=False
    net.world().stop()
    for r in range(11):
        net.world().remove_inputs()
        for t in range(500): # record for 500ms
            net.step()
        net.population('GUS').neuron(0).baseline=r/10.0
        net.population('GUS').neuron(1).baseline=r/10.0
        damax=0.0
        for t in range(500): # record for 500ms
            net.step()
            da=net.population('VTA').neuron(0).rate  
            if da>damax:
                damax=da
        da_amp.append(damax)
    for r in range(11):
        net.world().remove_inputs()
        for t in range(500): # record for 500ms
            net.step()
        net.population('GUS').neuron(0).baseline=1.0
        net.population('GUS').neuron(1).baseline=1.0
        net.population('DRIVE').neuron(0).baseline=r/10.0
        net.population('DRIVE').neuron(1).baseline=r/10.0
        net.population('DRIVE').neuron(2).baseline=r/10.0
        net.population('DRIVE').neuron(3).baseline=r/10.0
        damax=0.0
        for t in range(500): # record for 500ms
            net.step()
            da=net.population('VTA').neuron(0).rate  
            if da>damax:
                damax=da
        da_drive.append(damax)
               
    plt.subplot(121)    
    plt.plot(0.1*np.arange(0, 11), da_amp, linewidth=2.0)
    plt.title('Influence of reward amplitude on VTA')
    plt.xlabel('Reward amplitude')
    plt.ylabel('Peak activity in VTA')
    plt.ylim((0.5, 1.0))
    plt.subplot(122)    
    plt.plot(0.1*np.arange(0, 11), da_drive, linewidth=2.0)
    plt.title('Influence of drive on VTA')
    plt.xlabel('Drive')
    plt.ylabel('Peak activity in VTA')
    plt.ylim((0.5, 1.0))
    plt.show()
  
    
def record_trial(net):
    da=[]
    da_inh=[]
    nacc=[]
    vp=[]
    lhb=[]
    pptn=[]
    vtagaba=[]
    lhon=[]
    bla=[]
    
    # Reset inputs
    net.world().stop()
    net.world().remove_inputs()
    for t in range(1000): # record for 5s
        net.step()
        da.append(net.population('VTA').neuron(0).rate)
        da_inh.append(net.population('VTA').neuron(0).inhibition)
        nacc.append([neur.rate for neur in net.neurons('NAcc')])
        vp.append(net.population('VP').neuron(0).rate)
        lhb.append(net.population('LHb').neuron(0).rate)
        pptn.append(net.population('PPTN').neuron(0).rate)
        vtagaba.append(net.population('VTA_GABA').neuron(0).rate)
        lhon.append([neur.rate for neur in net.neurons('LH_ON')])
        bla.append([neur.rate for neur in net.neurons('BLA')])
    
    # Do one ensemble of trials
    net.world().start()
    for t in range(net.world().trial_duration): # record for 5s
        net.step()
        da.append(net.population('VTA').neuron(0).rate)
        da_inh.append(net.population('VTA').neuron(0).inhibition)
        nacc.append([neur.rate for neur in net.neurons('NAcc')])
        vp.append(net.population('VP').neuron(0).rate)
        lhb.append(net.population('LHb').neuron(0).rate)
        pptn.append(net.population('PPTN').neuron(0).rate)
        vtagaba.append(net.population('VTA_GABA').neuron(0).rate)
        lhon.append([neur.rate for neur in net.neurons('LH_ON')])
        bla.append([neur.rate for neur in net.neurons('BLA')])
        
    return da, da_inh, nacc, vp, lhb, pptn, vtagaba, lhon, bla
 
    
