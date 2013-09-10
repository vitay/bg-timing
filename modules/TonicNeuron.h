#ifndef TONIC_NEURON_H
#define TONIC_NEURON_H

#include "ANNarchy.h"

// Subclass of annarNeuron
class TonicNeuron : public annarNeuron {

    public:

        TonicNeuron(class annarPopulation* population, int rank):  annarNeuron(population, rank){
        
            tau_ = 10.0;
            threshold_ = 0.0;
            baseline_ = 1.0;
            
        };

        virtual void step(){
        
            mp_+= dt_ /tau_ * (-mp_ + sum("exc") - sum("inh") + baseline_ + noise_*(2.0*rand_num-1.0) );
            
            rate_=positive(mp_-threshold_);
            if(rate_>1.1) 
                rate_=1.1;
        };


    protected:

        @PARAMETER FLOAT threshold_; // threshold for the firing rate

};




#endif
