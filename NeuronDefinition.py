from ANNarchy import *

############################
# Definition of the neurons
############################

# Basic leaky neuron
LeakyNeuron = RateNeuron(
    parameters = """
    tau = 10.0 : population
    baseline = 0.0
    """,
    equations = """
    noise = Uniform(-0.1, 0.1)
    tau * dmp/dt + mp = sum(exc) - sum(inh) + baseline + noise
    r = pos(mp) : max=2.0
    """
)

# Neuron with temporal adaptation of excitatory inputs
PhasicNeuron = RateNeuron(
    parameters = """
    tau = 10.0 : population
    tau_adaptation = 500.0 : population
    K_adaptation = 1.0 : population
    """,
    equations = """
    g_dopa = sum(dopa) 
    noise = Uniform(-0.1, 0.1)
    tau_adaptation * dadapted_exc/dt + adapted_exc =  sum(exc) 
    tau*dmp/dt + mp =  pos(sum(exc) - K_adaptation * adapted_exc) - sum(inh) + noise
    r = pos(mp)
    """
)

# Neuron with shunting of inhibitory inputs
ShuntingNeuron = RateNeuron(
    parameters = """
    tau = 10.0 : population
    baseline = 0.5 : population
    """,
    equations = """
    noise = Uniform(-0.1, 0.1)
    inhibition = if sum(exc) < 0.1: sum(inh) else : 0.0
    tau*dmp/dt + mp =  sum(exc) - inhibition + baseline + noise
    r = pos(mp)
    """
)

# Neuron with temporal adaptation and shunting of excitatory inputs
ShuntingPhasicNeuron = RateNeuron(
    parameters = """
    tau = 10.0 : population
    tau_adaptation = 500.0 : population
    K_adaptation = 1.0 : population
    """,
    equations = """
    g_exc = sum(exc) 
    g_mod = sum(mod) 
    g_dopa = sum(dopa) 
    noise = Uniform(-0.1, 0.1)

    has_exc = if g_exc > 0.1 : 0.0 else: 1.0

    tau_adaptation * dadapted_exc/dt + adapted_exc =  g_exc 
    tau_adaptation * dadapted_mod/dt + adapted_mod =  g_mod 

    tau*dmp/dt + mp =   pos(sum(exc) - K_adaptation * adapted_exc)
        + pos(g_mod - K_adaptation * adapted_mod) * has_exc 
        - sum(inh) + noise
    
    r = pos(mp)
    """
)

# Oscillator neuron
OscillatorNeuron = RateNeuron(
    parameters = """
    tau = 1.0 : population
    freq = 1.0
    phase = 0.0 
    start_oscillate = 0.8 : population
    stop_oscillate = 0.2 : population
    """,
    equations = """
    g_exc = sum(exc) 
    oscillating = if g_exc > start_oscillate:
                    1
                  else:
                    if (g_exc > start_oscillate) & (oscillating > 0) :
                        1
                    else:
                        0 : int  
    time = if oscillating > 0:
                time + 1 
           else:
                0  : int, init=0 

    r = if oscillating > 0 : 
            (1.0 - exp(-time/500.0)) * (sin(2.0*pi*freq*time/1000.0 + phase) + 1.0)/2.0
          else:
            0.0 : min=0.0
    """
)

# Striatal Neuron
StriatalNeuron = RateNeuron(
    parameters = """
    tau = 10.0 : population
    baseline = -0.9 : population
    delta_up = 0.5 : population
    tau_state = 400.0 : population
    T_exc = 1.0 : population
    T_dopa = 0.3 : population
    K_dopa = 0.5 : population
    """,
    equations = """
    g_exc = sum(exc)
    g_mod = sum(mod)
    g_inh = sum(inh)
    g_dopa = sum(dopa)
    noise = Uniform(-0.1, 0.1)
    
    transition_up_down = if (s > 0.5) & ( (s_time > 0.95) | ((s_time > 0.8)&(g_dopa < T_dopa)) ):
                             1
                          else:
                             0  : int, init=0
                            
    transition_down_up = if (s < 0.5) & ( (g_dopa > T_dopa) | (g_exc + g_mod > T_exc) | (s_time < 0.05) ):
                             1
                         else:
                             0 : int, init=0
                            
    s_time = if (transition_up_down > 0):
                1.0
             else:
                if (transition_down_up > 0) :
                    0.0
                else:
                    s_time + dt / tau_state * (s - s_time)   : init = 0.0 

    s = if (transition_up_down > 0) :
            0.0
        else : 
            if (transition_down_up > 0) :
                1.0
            else:
                s 
                
    tau*dmp/dt + mp = g_exc + g_mod - g_inh + g_dopa * K_dopa + delta_up * s + baseline + noise

    r = pos(mp) : max = 1.1
    """
)


# Dopamine Neuron
DopamineNeuron = RateNeuron(
    parameters = """
    tau = 30.0 : population
    tau_decrease = 30.0 : population
    baseline = 0.2 : population
    tau_modulation = 300.0 : population
    """,
    equations = """
    g_exc  = sum(exc)
    g_mod = sum(mod)
    g_inh = sum(inh)
    noise = Uniform(-0.1, 0.1)
    
    dip = if mean_exc < 0.1 :
            positive(g_inh - mean_inh)
          else:
            0.0
    
    mean_mod = if g_mod > mean_mod:
                    g_mod
                 else:
                    mean_mod + dt/tau_modulation * (g_mod - mean_mod)
                    
    tau*dmp/dt + mp = g_exc * (1.0 - mean_mod) - dip + baseline + noise
        
    tau_decrease * dmean_exc/dt + mean_exc  = g_exc
    tau_decrease * dmean_inh/dt + mean_inh  = g_inh
    
    r = pos(mp) : max = 1.2
    """
)
