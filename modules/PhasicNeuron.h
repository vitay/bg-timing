#ifndef PHASIC_NEURON_H
#define PHASIC_NEURON_H

#include "ANNarchy.h"

// Subclass of annarNeuron: leaky integrator with positive transfer function
class PhasicNeuron : public annarNeuron
{
    public:

        PhasicNeuron(class annarPopulation* population, int rank):  annarNeuron(population, rank){

            tau_ = 10.0;
            threshold_ = 0.0;
            tau_adaptation_ = 1000.0;
            adapted_input_ = 0.0;
            adapted_mod_ = 0.0;
            max_rate_ = 1.1;

        };

        virtual void step(){

            adapted_input_ += dt_/tau_adaptation_ * (sum("exc") - adapted_input_);
            adapted_mod_ += dt_/tau_adaptation_ * (sum("mod") - adapted_mod_);
            
            mp_+= dt_/tau_ * (-mp_ + positive(sum("exc") - adapted_input_) 
                                   + positive(sum("mod") - adapted_mod_) * (positive(sum("exc") - adapted_input_) > 0.1 ? 0.0 : 1.0)
                                   + baseline_ 
                                   + noise_*(2.0*rand_num-1.0) );
            
            rate_=positive(mp_-threshold_);
            if(rate_ > max_rate_)
                rate_ = max_rate_;

        };

    protected:

        @PARAMETER FLOAT threshold_; // Threshold for the firing rate
        @PARAMETER FLOAT tau_adaptation_; // time constant for the adaptation of exc inputs
        @VARIABLE FLOAT adapted_input_; // Adapted input
        @VARIABLE FLOAT adapted_mod_; // Adapted input
        @PARAMETER FLOAT max_rate_; // max firing rate
};

#endif
