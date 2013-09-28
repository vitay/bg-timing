#ifndef DOPAMINE_NEURON_H
#define DOPAMINE_NEURON_H

#include "ANNarchy.h"

// Subclass of annarNeuron: leaky integrator with positive transfer function
class DopamineNeuron : public annarNeuron
{
    public:

        DopamineNeuron(class annarPopulation* population, int rank):  annarNeuron(population, rank){

            tau_ = 30.0;
            tau_decrease_ = 30.0;
            tau_modulation_ = 500.0;
            threshold_ = 0.0;
            max_rate_ = 1.1;
            
            input_ = 0.0;
            mean_input_ = 0.0;
            inhibition_ = 0.0;
            dip_ = 0.0;

        };

        virtual void step(){

            input_ = sum("exc");
            
            if(sum("mod") > inhibition_)
                inhibition_ +=  dt_ / 1.0 * (sum("mod") - inhibition_);
            else
                inhibition_ += dt_/tau_modulation_ * ( sum("mod") - inhibition_);

            mean_input_+= dt_/tau_decrease_* (input_-mean_input_);
            
            dip_= sum("inh");       
            
            mp_+= dt_/tau_ * (-mp_ 
                                + input_ * positive(1.0 - inhibition_ ) 
                                //+ positive(input_ - mean_input_) * positive(1.0 - sum("mod") ) 
                                - (mean_input_< 0.1? dip_ : 0.0) //* positive(mean_input_ - input_ + 0.05)  
                                + baseline_ + noise_*(2.0*rand_num-1.0));
            
            rate_ = positive(mp_ - threshold_);
            if(rate_ > max_rate_)
                rate_ = max_rate_;

        };

    protected:

        @PARAMETER FLOAT threshold_; // threshold for the firing rate
        @PARAMETER FLOAT max_rate_; // max firing rate
        @PARAMETER FLOAT tau_decrease_; // Time constant for the alpha function.
        @PARAMETER FLOAT tau_modulation_; // Time constant for the inhibitory modulation from NAcc.
        @VARIABLE FLOAT input_; // Received reward
        @VARIABLE FLOAT mean_input_; // Mean received reward (for phasic activation)
        @VARIABLE FLOAT inhibition_; // Inhibition received from NAcc
        @VARIABLE FLOAT dip_; // Inhibition from RMTg
};

#endif
