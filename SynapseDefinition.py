from ANNarchy4 import *

############################
# Definition of the synapses
############################


# Covariance learning rule with homeostatic regularization (dynamic Oja)
Covariance = RateSynapse(
    parameters = """
    eta = 100.0 : postsynaptic
    tau_alpha = 10.0 : postsynaptic
    """,
    equations = """
    tau_alpha * dalpha/dt + alpha = pos(post.rate - 1.0) : postsynaptic
    eta * dvalue/dt = if (pre.rate > mean(pre.rate) | post.rate > mean(post.rate) ) : 
                          (pre.rate - mean(pre.rate) ) * (post.rate - mean(post.rate)) - alpha * (post.rate - mean(post.rate))^2 * value 
                      else :
                            0.0 : min=0.0
    """
)

# Covariance learning rule modulated by dopamine
DACovariance = RateSynapse(
    parameters = """
    eta = 100.0 : postsynaptic
    tau_alpha = 1.0 : postsynaptic
    tau_dopa = 300.0 : postsynaptic
    K_alpha = 5.0 : postsynaptic
    K_LTD = 1.0 : postsynaptic
    dopa_threshold_LTP = 0.3 : postsynaptic
    dopa_K_LTP = 10.0 : postsynaptic
    """,
    equations = """
    tau_alpha * dalpha/dt + alpha = pos(post.rate - 1.0)  : postsynaptic
    tau_dopa *ddopa_mean/dt + dopa_mean = post.g_dopa  : postsynaptic
    dopa = if post.g_dopa > dopa_threshold_LTP :
               dopa_K_LTP * pos( post.g_dopa - dopa_mean )
           else :
                0.0
    eta * dvalue/dt = if (pre.rate > mean(pre.rate)) & ( post.rate > mean(post.rate) ) : 
                         dopa * (pre.rate - mean(pre.rate) ) * (post.rate - mean(post.rate)) - K_alpha * alpha * (post.rate - mean(post.rate))^2 * value
                      else :
                        if (pre.rate > mean(pre.rate) ) |( post.rate > mean(post.rate) ) : 
                          K_LTD * dopa * (pre.rate - mean(pre.rate) ) * (post.rate - mean(post.rate)) 
                        else:
                          0.0  :  min=0.0
    """
)

# Covariance learning rule modulated by dopamine with shunting excitation
DAShuntingCovariance = RateSynapse(
    parameters = """
    eta = 2000.0 : postsynaptic
    tau_dopa = 300.0 : postsynaptic
    K_LTD = 5.0 : postsynaptic
    dopa_threshold_LTP = 0.3 : postsynaptic
    dopa_K_LTP = 4.0 : postsynaptic
    """,
    equations = """
    dopa = if post.g_dopa > dopa_threshold_LTP : 
               dopa_K_LTP
           else : 
               0.0
    eta * dvalue/dt = if (pre.rate > mean(pre.rate)) & (post.rate > mean(post.rate) ) : 
                          dopa * (pre.rate - mean(pre.rate) ) * (post.rate - mean(post.rate)) * pos(post.g_exc - post.g_mod) 
                      else : 
                          if (pre.rate > mean(pre.rate)) | (post.rate > mean(post.rate) ) : 
                              K_LTD * dopa * (pre.rate - mean(pre.rate) ) * (post.rate - mean(post.rate)) 
                          else: 
                             0.0 : min=0.0
    """
)

# Anti-hebbian learning rule
AntiHebb = RateSynapse(
    parameters = """
    eta = 100.0 : postsynaptic
    """,
    equations = """
    eta * dvalue/dt = if (pre.rate > mean(pre.rate)) & (post.rate > mean(post.rate) ) : 
                          (pre.rate - mean(pre.rate) ) * (post.rate - mean(post.rate))
                      else : 
                          0.0 : min=0.0, max=3.0
    """
)

# Hebbian learning rule
Hebb = RateSynapse(
    parameters = """
    eta = 500.0 : postsynaptic
    threshold_pre = 0.0 : postsynaptic
    threshold_post = 0.0 : postsynaptic
    """,
    equations = """
    eta * dvalue/dt = positive(pre.rate - threshold_pre) * positive(post.rate - threshold_post) : min=0.0, max=20.0
    """
)
