import ANNarchy
import numpy as np
import matplotlib.pyplot as plt
#import scitools.easyviz as plt


def test_timing(net):
    """
    Present only the CSs and measure responses in NAcc.
    """
    
    # Reset inputs
    net.world().stop()
    net.world().remove_inputs()
    net.run(100)
    net.wait()
    
    # Set first CS
    net.population('VIS').neuron(0).baseline=1.0
    nacc_recordings_1=[]
    for t in range(4000): # record for 5s
        net.step()
        nacc_recordings_1.append( [neur.rate for neur in net.neurons('NAcc')] ) 
    nacc_recordings_1=np.array(nacc_recordings_1)
    nacc_mean_1=np.sum(nacc_recordings_1, axis=1)
    nacc_recordings_1[nacc_recordings_1<0.1]=0.0
    
    
    # Reset inputs
    net.world().remove_inputs()
    net.run(100)
    net.wait()
    
    # Set second CS
    net.population('VIS').neuron(1).baseline=1.0
    nacc_recordings_2=[]
    for t in range(4000): # record for 5s
        net.step()
        nacc_recordings_2.append( [neur.rate for neur in net.neurons('NAcc')] ) 
    nacc_recordings_2=np.array(nacc_recordings_2)
    nacc_mean_2=np.sum(nacc_recordings_2, axis=1)
    nacc_recordings_2[nacc_recordings_2<0.1]=0.0
     
    # Plot everything
    plt.subplot(221)
    imgplot = plt.imshow(nacc_recordings_1, aspect='auto')
    plt.colorbar()
    plt.subplot(222)
    imgplot = plt.imshow(nacc_recordings_2, aspect='auto')
    plt.colorbar()
    plt.subplot(223)
    plt.plot(nacc_mean_1)
    plt.subplot(224)
    plt.plot(nacc_mean_2)
    
    plt.show()   

def test_dopamine(net)  :
    """ 
    Test dopamine firing patterns
    """
    # Reset inputs
    net.world().stop()
    net.world().remove_inputs()
    net.run(100)
    net.wait()

    # Set US
    net.population('GUS').neuron(0).baseline=1.0
    net.population('GUS').neuron(1).baseline=1.0
    vta=[]
    for t in range(1000): # record
        net.step()
        vta.append( net.population('VTA').neuron(0).rate) 
    plt.plot(vta)
    plt.show()   
    
def test_gated_dipole(net):
    """
    Test behaviour of gated dipoles.
    """
    # Reset inputs
    net.run(500)
    net.wait()   
    
    # Set US
    net.population('GUS').neuron(0).baseline=1.0
    net.population('GUS').neuron(1).baseline=1.0
    lh=[]
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
        
        
        
    plt.plot(lh)
    plt.show()   
    
    
def test_bla_learning(net):
    """
    Present only the USs and measure responses in BLA.
    """
    # Store results
    bla=[]
    ce=[]
    da=[]
    
    # Reset inputs
    net.world().stop()
    net.world().remove_inputs()
    for t in range(500): # record for 500ms
        net.step()
        bla.append( [neur.rate for neur in net.neurons('BLA')] ) 
        ce.append(net.population('CE').neuron(0).rate)
        da.append(net.population('VTA').neuron(0).rate)
    
    # Do one ensemble of trials
    net.world().start()
    for t in range(net.world().trial_duration): # record for 5s
        net.step()
        bla.append( [neur.rate for neur in net.neurons('BLA')] ) 
        ce.append(net.population('CE').neuron(0).rate)
        da.append(net.population('VTA').neuron(0).rate)
        
    # Plot everything
    plt.subplot(221)
    imgplot = plt.imshow(bla, aspect='auto')
    plt.colorbar()
    plt.subplot(222)
    plt.plot(np.sum(np.array(bla), axis=1))    
    plt.subplot(223)
    plt.plot(ce)
    plt.subplot(224)
    plt.plot(da)
    
    plt.show()   
    
    
def record_vta(net):
    da=[]
    
    # Reset inputs
    net.world().stop()
    net.world().remove_inputs()
    for t in range(1000): # record for 5s
        net.step()
        da.append(net.population('VTA').neuron(0).rate)
    
    # Do one ensemble of trials
    net.world().start()
    for t in range(net.world().trial_duration): # record for 5s
        net.step()
        da.append(net.population('VTA').neuron(0).rate)
        
    return da   
    
        
