import ANNarchy, ANNarchy_LIB
from ANNarchy import *
from ANNarchy_LIB import *


class vertical_stripes(annarConnection):

    def __init__(self, net, pre, post, connection_type, delay=0, value=0.0, var_value=0.0):
        annarConnection.__init__(self, net, "vertical_stripes", pre, post, connection_type, delay, value=value, var_value=var_value)
        
        if net.population(post).width==net.population(pre).width:
            for w in range(net.population(post).width)  :
                for h in range(net.population(post).height)  :
                    postneur=net.population(post).neuron(w,h)
                    for k in range(net.population(pre).height):
                        preneur=net.population(pre).neuron(w,k)
                        val=value+var_value*(2.0*random.random()-1.0)
                        postneur.connect(preneur, connection_type, val, delay)
        else: 
            print 'Error while creating the stripes connections between', pre, 'and', post, ': width sizes do not match'
            
class cluster(annarConnection):
    """ Clustered connection pattern between two populations. Subclass of annarConnection.

    Each neuron in the presynaptic population sends a connection to a given number of neurons of the postsynaptic population.

    Parameters:
    - net : the network to which this pattern belong.
    - pre : the name of the presynaptic population.
    - post : the name of the postsynaptic population.
    - connection_type : the type of the connection
    - delay : the delay in ms of information transmission at the synapse level (default=0).
    - value: the initial value for the connection weights (default=1.0).
    - var_value: the variation on the value for the connection weights (default=0.0). In the postsynaptic neuron, connections weights will vary uniformly between (value - var_value) and (value + var_value).
    - cluster_size: the number of postsynaptic neurons receiving inputs from the same presynaptic neuron (default=-1). When -1 is given, the postsynaptic population will be filled with clusters of the maximum possible size.
    - var_cluster_size: the variation of the number of neurons in a cluster (default=0).

    Either (value), (value, var_value) or (min_value, max_value) should be defined, but not other combinations.
    """

    def __init__(self, net, pre, post, connection_type, delay=0, value=0.0, var_value=0.0, cluster_size=1, var_cluster_size=0):
        annarConnection.__init__(self, net, "cluster", pre, post, connection_type, delay, value=value, var_value=var_value, cluster_size=cluster_size,var_cluster_size=var_cluster_size)
        
        pre_size=self.net.population(pre).size
        post_size=self.net.population(post).size

        order=range(post_size)
        random.shuffle(order)

        if cluster_size==-1:
            mean_cluster_size=int(post_size/pre_size)
        else:
            mean_cluster_size=cluster_size

        used_neurons=0
        for preneur in self.net.neurons(pre): # for each cluster
            c_size=random.randint(mean_cluster_size - var_cluster_size, mean_cluster_size + var_cluster_size)
            if (used_neurons+c_size)<post_size: # enough neurons left
                c_size=c_size
            else: # not enough neurons to build this cluster
                c_size=post_size - used_neurons
            for i in range(c_size):
                postneur=self.net.population(post).neuron(order[used_neurons+i])
                postneur.connect(preneur, connection_type, value, delay)
            used_neurons+=c_size
