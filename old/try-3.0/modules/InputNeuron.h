#ifndef INPUT_NEURON_H
#define INPUT_NEURON_H

#include "ANNarchy.h"

// Subclass of annarNeuron: leaky integrator with positive transfer function
class InputNeuron : public annarNeuron
{
    public:

        InputNeuron(class annarPopulation* population, int rank):  annarNeuron(population, rank){
        
            tau_ = 1.0;
        
        };

        virtual void step(){
        
            // Update the membrane potential
            mp_ += dt_ /tau_ * (-mp_ + baseline_);
            
            // Update the instantaneous firing rate
            rate_ = positive(mp_);
        
        }

    protected:

};

#endif
