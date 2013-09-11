from ANNarchy4 import *

# Set the simulation parameters
setup(dt=1.0)

# Define the neurons
InputNeuron = Neuron(
    tau = 10.0,
    baseline = Variable(init=0.0),
    rate = Variable(
        init=0.0,
        eq="tau * drate/dt + rate = baseline",
        min=0.0
    )
)

LinearNeuron = Neuron(
    tau = 10.0,
    mp = Variable(
        init=0.0,
        eq="tau * dmp/dt + mp = sum(exc) - sum(inh)"
    ),
    rate = Variable(
        init=0.0,
        eq="rate = mp",
        min=0.0
    )
)

DopamineNeuron = Neuron(
    tau = 30.0,
    tau_decrease=30.0,
    reward = Variable(
        init=0.0,
        eq="reward = sum(exc)",
        max=100.0
    ),
    prediction = Variable(
        init=0.0,
        eq="prediction = sum(inh)"
    ),
    mean_reward = Variable(
        init=0.0,
        eq="tau_decrease* dmean_reward/dt + mean_reward = reward"
    ),
    rate = Variable(
        init=0.0,
        eq="tau * drate/dt + rate = sum(exc) - sum(inh)",
        min=0.0, max=1.1
    )
)

# Define the synapses
Oja = Synapse(
    tau = 2000,
    alpha = 8.0,
    value = Variable(
        init=0.0,
        eq="""
  tau * dvalue/dt = (pre.rate - mean(pre.rate)) * (post.rate - mean(post.rate))
                      - alpha * (post.rate - mean(post.rate))^2 * value
 """
    )
)

AntiHebb = Synapse(
    tau = 2000,
    alpha = 0.3,
    value = Variable(
        init=0.0,
        eq="tau * dvalue/dt = pre.rate * post.rate - alpha * post.rate^2 * value",
        min=0.0
    )
)

# Create the populations
visual = Population(name='visual', geometry=(2, ), neuron=InputNeuron)
gustatory = Population(name='gustatory', geometry=(4, ), neuron=InputNeuron)
vta = Population(name='vta', geometry=(1, ), neuron=DopamineNeuron)

visual.tau = 15.0

# Create the projections
proj = Projection(
    pre=visual, post=gustatory, target='exc',
    synapse=Oja,
    connector=Connector('All2All', weights=RandomDistribution('constant', [0.]))
)


# Environment
def trial():
    pass


if __name__ == '__main__':

    compile()

