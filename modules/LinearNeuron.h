#ifndef LINEAR_NEURON_H
#define LINEAR_NEURON_H

#include "ANNarchy.h"

// Subclass of annarNeuron
class LinearNeuron : public annarNeuron {

    public:

        LinearNeuron(class annarPopulation* population, int rank):  annarNeuron(population, rank){
        
            tau_ = 10.0;
            baseline_ = 0.0;
            
            threshold_ = 0.0;
            max_rate_ = 1.1;
            
            
        };

        virtual void step(){
        
            mp_+= dt_ /tau_ * (-mp_ + sum("exc") - sum("inh") + baseline_ + noise_*(2.0*rand_num-1.0) );
            
            rate_=positive(mp_-threshold_);
            if(rate_>max_rate_) 
                rate_=max_rate_;
        };


    protected:

        @PARAMETER FLOAT threshold_; // threshold for the firing rate
        @PARAMETER FLOAT max_rate_; // max  firing rate

};




#endif
