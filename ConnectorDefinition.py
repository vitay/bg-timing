import numpy as np
from ANNarchy import *

def connect_cluster(pre, post, weight, number):
    " Connector to form cluster patterns: each presynaptic neuron contacts exactly `number` different postsynaptic neurons."       
    matrix = CSR()

    # Test if there is enough space in the post population to make clusters
    if number*pre.size > post.size:
        print('Error when creating the Cluster connector between', pre.name, 'and', post.name)
        print('There are not enough neurons (' + str(post.size)+ ') in', post.name, 'to create', pre.size, 'clusters of', number, 'neurons ('+str(number*pre.size)+').')
        exit(0)
        
    # Create the connections
    order = list(range(post.size))
    np.random.shuffle(order)
    index = 0
    for pre in range(pre.size):
        for k in range(number):
            matrix.add(order[index], [pre], [weight], [0])            
            index += 1

    return matrix
    
def connect_stripes(pre, post, weight):
    " Connector to connect each row of post (2D) to one neuron of pre (1D). "
 
    matrix = CSR()
    (height, width) = post.geometry
    for rk in range(post.size):
        h, w = post.coordinates_from_rank(rk)
        matrix.add(rk, [w], [weight], [0]) 
                
    return matrix
