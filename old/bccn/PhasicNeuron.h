#ifndef PHASIC_NEURON_H
#define PHASIC_NEURON_H

#include "ANNarchy.h"

ANNARCHY_EXPORT(fb_exc, double, fb_exc_, "Excitatory influence of feedback connections.")
ANNARCHY_EXPORT(fb_mod, double, fb_mod_, "Modulatory influence of feedback connections on feedforward connections.")
ANNARCHY_EXPORT(dopa_mod, double, dopa_mod_, "Modulatory influence of dopaminergic connections on feedforward connections.")
ANNARCHY_EXPORT(threshold, double, threshold_, "Firing threshold for the membrane potential.")
ANNARCHY_EXPORT(tau_adaptation, double, tau_adaptation_, "Adaptation rate of the inputs.")


// Subclass of annarNeuron
class PhasicNeuron : public annarNeuron
{

public:

    PhasicNeuron(class annarPopulation* population, int rank):  annarNeuron(population, rank){
        tau_=10.0;
        fb_mod_=1.0;
        fb_exc_=0.0;
        dopa_mod_=0.0;
        threshold_=0.0;
        tau_adaptation_=1000.0;
        adapted_inputs_=0.0;
    };

    virtual void step(){
    
        adapted_inputs_=1.0/tau_adaptation_*(sum("FF") - adapted_inputs_);
        
        mp_+= 1.0 /tau_ * (-mp_ + positive(sum("FF") - adapted_inputs_) * (1.0 + fb_mod_*sum("FB") ) * (1.0 + dopa_mod_ *(sum("DOPA")-0.5) ) + fb_exc_*sum("FB") -  sum("LAT") + baseline_ + noise_*(2.0*rand_num-1.0));
        
        rate_=positive(mp_-threshold_);
    };


protected:
    double fb_exc_, fb_mod_, threshold_, dopa_mod_;
    double tau_adaptation_, adapted_inputs_;

};




#endif
