#ifndef DIFF_NEURON_H
#define DIFF_NEURON_H

#include "ANNarchy.h"

// Subclass of annarNeuron: leaky integrator with positive transfer function
class DiffNeuron : public annarNeuron
{
    public:

        DiffNeuron(class annarPopulation* population, int rank):  annarNeuron(population, rank){

            tau_ = 10.0;
            threshold_ = 0.0;

        };

        virtual void step(){

            FLOAT lhb = sum("exc");
            FLOAT pptn = sum("inh");
            FLOAT diff = 0.0;
            if(pptn > 0.3){
                diff = 0.0;
            } else{
                diff = lhb;
            }
                
        
            mp_+= 1.0 /tau_ * (-mp_ + diff + baseline_ + noise_*(2.0*rand_num-1.0) );
            
            rate_ = 2.0*positive(mp_-threshold_);
            if(rate_ > 1.1) 
                rate_ = 1.1;

        };

    protected:

        @PARAMETER FLOAT threshold_; // Threshold for the firing rate
};

#endif
