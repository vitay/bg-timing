#ifndef THRESHOLD_NEURON_H
#define THRESHOLD_NEURON_H

#include "ANNarchy.h"

// Subclass of annarNeuron
class ThresholdNeuron : public annarNeuron {

    public:

        ThresholdNeuron(class annarPopulation* population, int rank):  annarNeuron(population, rank){
        
            tau_=10.0;
            threshold_=0.0;
            
        };

        virtual void step(){
        
        
            mp_+= 1.0 /tau_ * (-mp_ + 2.0*positive(sum("exc") - 0.5 ) * (sum("inh") < 0.5? 1.0 : 0.0 ) 
                                    + baseline_ 
                                    + noise_*(2.0*rand_num-1.0) );
            
            rate_=positive(mp_-threshold_);
            if(rate_>1.1) 
                rate_=1.1;
        };


    protected:
        double threshold_;

};




#endif
