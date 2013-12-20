from ANNarchy4 import *

class Cluster(Connector):
    """ Connector to form cluster patterns: each presynaptic neuron contacts exactly `number` different postsynaptic neurons.
    """
    def __init__(self, weights, delays=0, **parameters):
        
#        self.weights = weights
#        self.delays = delays
#        self.parameters = parameters
        
        Connector.__init__(self, weights, delays, **parameters)
        
    def connect(self):
        " Method to create the synapses."         
        dendrites = []
        post_ranks = []
        
        # Get the number parameter
        if not 'number' in self.parameters.keys():
            number = 1
        else:
            number = self.parameters['number']
            
        # Test if there is enough space in the post population to make clusters
        if number*self.proj.pre.size > self.proj.post.size:
            print 'Error when creating the Cluster connector between', self.proj.pre.name, 'and', self.proj.post.name
            print 'There are not enough neurons (' + str(self.proj.post.size)+ ') in', self.proj.post.name, 'to create', self.proj.pre.size, 'clusters of', number, 'neurons ('+str(number*self.proj.pre.size)+').'
            exit(0)
            
        # Create the connections
        order = range(self.proj.post.size)
        np.random.shuffle(order)
        index = 0
        for pre in range(self.proj.pre.size):
            for k in range(number):
                dendrite = Dendrite(self.proj, post_rank=order[index], ranks=[pre], weights=self.weights.get_values(1))
                dendrites.append(dendrite)
                post_ranks.append(order[index])
                index += 1

        return dendrites, post_ranks
