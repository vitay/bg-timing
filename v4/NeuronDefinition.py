from ANNarchy4 import *

############################
# Definition of the neurons
############################

# Basic leaky neuron
LeakyNeuron = Neuron(
    tau = 10.0,
    noise = Variable(init=0.0, eq=Uniform(-0.1, 0.1)),
    baseline = Variable(init=0.0),
    mp = Variable(
        init=0.0, 
        eq='tau*dmp/dt + mp = sum(exc) - sum(inh) + baseline + noise'
    ),
    rate = Variable(init=0.0, eq='rate = pos(mp)')
)

# Neuron with temporal adaptation of excitatory inputs
PhasicNeuron = Neuron(
    tau = 10.0,
    tau_adaptation = 500.0,
    K_adaptation = 1.0,
    noise = Variable(init=0.0, eq=Uniform(-0.1, 0.1)),
    adapted_exc = Variable(
        init=0.0, 
        eq='tau_adaptation*dadapted_exc/dt + adapted_exc =  sum(exc) '
    ),
    dopa = Variable(
        init=0.0, 
        eq='dopa = sum(dopa) '
    ),
    mp = Variable(
        init=0.0, 
        eq='tau*dmp/dt + mp =  pos(sum(exc) - K_adaptation * adapted_exc) - sum(inh) + noise'
    ),
    rate = Variable(init=0.0, eq='rate = pos(mp)'),
    order= ['noise', 'dopa', 'adapted_exc', 'mp', 'rate']
)

# Neuron with temporal adaptation of excitatory inputs
ShuntingPhasicNeuron = Neuron(
    tau = 10.0,
    tau_adaptation = 500.0,
    K_adaptation = 1.0,
    noise = Variable(init=0.0, eq=Uniform(-0.1, 0.1)),
    sum_exc = Variable(
        init=0.0, 
        eq='sum_exc = sum(exc) '
    ),
    sum_mod = Variable(
        init=0.0, 
        eq='sum_mod = sum(mod) '
    ),
    dopa = Variable(
        init=0.0, 
        eq='dopa = sum(dopa) '
    ),
    adapted_exc = Variable(
        init=0.0, 
        eq='tau_adaptation*dadapted_exc/dt + adapted_exc =  sum_exc '
    ),
    adapted_mod = Variable(
        init=0.0, 
        eq='tau_adaptation*dadapted_mod/dt + adapted_mod =  sum_mod '
    ),
    mp = Variable(
        init=0.0, 
        eq="""
        tau*dmp/dt + mp =   pos(sum_exc - K_adaptation * adapted_exc) 
                            + pos(sum_mod - K_adaptation * adapted_mod) * (if sum_exc>0.1 then 0.0 else 1.0)
                            - sum(inh) 
                            + noise
        
        """
    ),
    rate = Variable(init=0.0, eq='rate = pos(mp)'),
    order= ['noise', 'dopa', 'sum_exc', 'sum_mod', 'adapted_exc', 'adapted_mod', 'mp', 'rate']
)
