#ifndef GATED_NEURON_H
#define GATED_NEURON_H

#include "ANNarchy.h"

ANNARCHY_EXPORT(threshold, double, threshold_, "Firing threshold for the membrane potential.")
ANNARCHY_EXPORT(tau_adaptation, double, tau_adaptation_, "Adaptation rate of the inputs.")


// Subclass of annarNeuron
class GatedNeuron : public annarNeuron
{

public:

    GatedNeuron(class annarPopulation* population, int rank):  annarNeuron(population, rank){
        tau_=10.0;
        threshold_=0.0;
        tau_adaptation_=1000.0;
        adapted_inputs_=0.0;
    };

    virtual void step(){
    
        adapted_inputs_=1.0/tau_adaptation_ * (sum("FF") - adapted_inputs_);
    
        mp_+= 1.0 /tau_ * (-mp_ + 2.0*positive( sum("FF") - 0.5 ) * (sum("GABA") < 0.5? 1.0 : 0.0 ) + baseline_ + noise_*(2.0*rand_num-1.0));
        
        rate_=positive(mp_-threshold_);
        if(rate_>1.1) rate_=1.1;
    };


protected:
    double threshold_;
    double tau_adaptation_, adapted_inputs_;

};




#endif
