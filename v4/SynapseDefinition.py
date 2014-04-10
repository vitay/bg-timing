from ANNarchy4 import *

############################
# Definition of the synapses
############################

# Covariance learning rule with homeostatic regularization (dynamic Oja)
Covariance = Synapse(
    eta = 100.0,
    tau_alpha = 10.0,
    alpha = Variable(init=0.0, eq="tau_alpha * dalpha/dt + alpha = pos(post.rate - 1.0)"),
    value = Variable(eq="""
    eta * dvalue/dt = if (pre.rate > mean(pre.rate) or post.rate > mean(post.rate) ) 
                      then 
                        (pre.rate - mean(pre.rate) ) * (post.rate - mean(post.rate)) - alpha * (post.rate - mean(post.rate))^2 * value 
                      else 
                        0.0
    """, min=0.0),
    order = ['alpha', 'value']
)

# Covariance learning rule modulated by dopamine
DACovariance = Synapse(
    eta = 100.0,
    tau_alpha = 1.0,
    tau_dopa = 300.0,
    K_alpha = 5.0,
    K_LTD = 1.0,
    dopa_threshold_LTP = 0.3,
    dopa_K_LTP = 10.0,
    alpha = Variable(init=0.0, eq="tau_alpha * dalpha/dt + alpha = pos(post.rate - 1.0)"),
    dopa_mean = Variable(init=0.0, eq="tau_dopa*ddopa_mean/dt + dopa_mean = post.sum(dopa)"),
    dopa = Variable(init=0.0, eq="""
        dopa = if post.sum(dopa) > dopa_threshold_LTP
               then 
                    dopa_K_LTP * pos( post.sum(dopa) - dopa_mean )
               else
                    0.0
    """),
    value = Variable(eq="""
    eta * dvalue/dt = if (pre.rate > mean(pre.rate) and post.rate > mean(post.rate) ) 
                      then 
                        dopa * (pre.rate - mean(pre.rate) ) * (post.rate - mean(post.rate)) - K_alpha * alpha * (post.rate - mean(post.rate))^2 * value  
                      else 
                        ( if (pre.rate > mean(pre.rate) or post.rate > mean(post.rate) )
                          then
                             K_LTD * dopa * (pre.rate - mean(pre.rate) ) * (post.rate - mean(post.rate))
                          else
                             0.0
                        )
    """, min=0.0),
    order = ['dopa_mean', 'dopa', 'alpha', 'value']
)

# Covariance learning rule modulated by dopamine with shunting excitation
DAShuntingCovariance = Synapse(
    eta = 2000.0,
    tau_dopa = 300.0,
    K_LTD = 5.0,
    dopa_threshold_LTP = 0.3,
    dopa_K_LTP = 4.0,
    dopa = Variable(init=0.0, eq="""
        dopa = if post.sum(dopa) > dopa_threshold_LTP
               then 
                    dopa_K_LTP
               else
                    0.0
    """),
    value = Variable(eq="""
    eta * dvalue/dt = if (pre.rate > mean(pre.rate) and post.rate > mean(post.rate) ) 
                      then 
                        dopa * (pre.rate - mean(pre.rate) ) * (post.rate - mean(post.rate)) * pos(post.sum(exc) - post.sum(mod))
                      else 
                        ( if (pre.rate > mean(pre.rate) or post.rate > mean(post.rate) )
                          then
                             K_LTD * dopa * (pre.rate - mean(pre.rate) ) * (post.rate - mean(post.rate))
                          else
                             0.0
                        )
    """, min=0.0),
    order = ['dopa', 'value']
)

# Anti-hebbian learning rule
AntiHebb = Synapse(
    eta = 100.0,
    value = Variable(eq="""
    eta * dvalue/dt = if (pre.rate > mean(pre.rate) and post.rate > mean(post.rate) ) 
                      then 
                        (pre.rate - mean(pre.rate) ) * (post.rate - mean(post.rate))
                      else
                        0.0
    """, min=0.0, max=3.0)  
)
