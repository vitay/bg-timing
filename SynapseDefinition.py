from ANNarchy import *

############################
# Definition of the synapses
############################


# Covariance learning rule with homeostatic regularization (dynamic Oja)
Covariance = Synapse(
    parameters = """
    eta = 100.0 : postsynaptic
    tau_alpha = 10.0 : postsynaptic
    """,
    equations = """
    tau_alpha * dalpha/dt + alpha = pos(post.r - 1.0) : postsynaptic
    eta * dw/dt = ite(  pre.r > mean(pre.r) | post.r > mean(post.r),
                        (pre.r - mean(pre.r) ) * (post.r - mean(post.r)) - alpha * (post.r - mean(post.r))^2 * w,
                        0.0) : min=0.0
    """
)

# Covariance learning rule modulated by dopamine
DACovariance = Synapse(
    parameters = """
    eta = 100.0 : postsynaptic
    tau_alpha = 1.0 : postsynaptic
    tau_dopa = 300.0 : postsynaptic
    K_alpha = 5.0 : postsynaptic
    K_LTD = 1.0 : postsynaptic
    dopa_threshold_LTP = 0.3 : postsynaptic
    dopa_K_LTP = 10.0 : postsynaptic
    wmin = 0.0 : postsynaptic
    """,
    equations = """

    dopa = ite (post.g_dopa > dopa_threshold_LTP,
                dopa_K_LTP * pos( post.g_dopa - dopa_mean ),
                0.0 ) : postsynaptic

    tau_alpha * dalpha/dt + alpha = pos(post.r - 1.0)  : postsynaptic
    tau_dopa *ddopa_mean/dt + dopa_mean = post.g_dopa  : postsynaptic

    eta * dw/dt = ite( (pre.r > mean(pre.r)) & ( post.r > mean(post.r) ),
                       dopa * (pre.r - mean(pre.r) ) * (post.r - mean(post.r)) - K_alpha * alpha * (post.r - mean(post.r))^2 * w,
                       ite( (pre.r > mean(pre.r) ) | ( post.r > mean(post.r) ),
                             K_LTD * dopa * (pre.r - mean(pre.r) ) * (post.r - mean(post.r)),
                             0.0
                        )
                    )  :  min=wmin
    """
)

# Covariance learning rule modulated by dopamine with shunting excitation
DAShuntingCovariance = Synapse(
    parameters = """
    eta = 2000.0 : postsynaptic
    tau_dopa = 300.0 : postsynaptic
    K_LTD = 5.0 : postsynaptic
    dopa_threshold_LTP = 0.3 : postsynaptic
    dopa_K_LTP = 4.0 : postsynaptic
    """,
    equations = """
    dopa = ite (post.g_dopa > dopa_threshold_LTP, dopa_K_LTP, 0.0)

    eta * dw/dt = ite(
                    (pre.r > mean(pre.r)) & (post.r > mean(post.r) ),
                    dopa * (pre.r - mean(pre.r) ) * (post.r - mean(post.r)) * pos(post.g_exc - post.g_mod),
                    ite (
                        (pre.r > mean(pre.r)) | (post.r > mean(post.r) ),
                        K_LTD * dopa * (pre.r - mean(pre.r) ) * (post.r - mean(post.r)),
                        0.0
                        )
                    ) : min=0.0
    """
)

# Anti-hebbian learning rule
AntiHebb = Synapse(
    parameters = """
    eta = 100.0 : postsynaptic
    """,
    equations = """
    eta * dw/dt = ite(
                    (pre.r > mean(pre.r)) & (post.r > mean(post.r) ),
                    (pre.r - mean(pre.r) ) * (post.r - mean(post.r)),
                    0.0
                    ) : min=0.0, max=3.0
    """
)

# Hebbian learning rule
Hebb = Synapse(
    parameters = """
    eta = 500.0 : postsynaptic
    threshold_pre = 0.0 : postsynaptic
    threshold_post = 0.0 : postsynaptic
    wmin = 0.0  : postsynaptic
    wmax = 20.0 : postsynaptic
    """,
    equations = """
    eta * dw/dt = positive(pre.r - threshold_pre) * positive(post.r - threshold_post) : min=wmin, max=wmax
    """
)
