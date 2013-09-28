#ifndef MODULATEDPHASIC_NEURON_H
#define MODULATEDPHASIC_NEURON_H

#include "ANNarchy.h"

// Subclass of annarNeuron: leaky integrator with positive transfer function
class ModulatedPhasicNeuron : public annarNeuron
{
    public:

        ModulatedPhasicNeuron(class annarPopulation* population, int rank):  annarNeuron(population, rank){

            tau_=10.0;
            fb_mod_=1.0;
            fb_exc_=0.0;
            dopa_mod_=0.0;
            threshold_=0.0;
            tau_adaptation_=1000.0;
            adapted_input_=0.0;
            adapted_modulation_=0.0;

        };

        virtual void step(){

            adapted_input_ += dt_/tau_adaptation_*(sum("exc") - adapted_input_);
            adapted_modulation_ += dt_/tau_adaptation_*(sum("mod") - adapted_modulation_);
            
            mp_+= dt_/tau_ * (-mp_ + positive(sum("exc") - fb_mod_*adapted_input_) 
                                        //* positive(1.0 - fb_mod_*sum("mod") ) 
                                        //* (1.0 + dopa_mod_ *(sum("dopa") - 0.5) ) 
                                   + fb_exc_ * positive(sum("mod") - fb_mod_*adapted_modulation_) * (sum("exc")>0.1? 0.0: 1.0)
                                   - sum("inh") 
                                   + baseline_ 
                                   + noise_*(2.0*rand_num-1.0) );
            
            rate_=positive(mp_-threshold_);

        };

    protected:

        @PARAMETER FLOAT threshold_; // Threshold for the firing rate
        @PARAMETER FLOAT fb_exc_; // Excitatory effect of modulatory inputs
        @PARAMETER FLOAT fb_mod_; // Modulatory effect of modulatory inputs
        @PARAMETER FLOAT dopa_mod_; // Modulatory effect of dopamine
        @PARAMETER FLOAT tau_adaptation_; // time constant for the adaptation of exc inputs
        @VARIABLE FLOAT adapted_input_; // Adapted input
        @VARIABLE FLOAT adapted_modulation_; // Adapted modulation
};

#endif
