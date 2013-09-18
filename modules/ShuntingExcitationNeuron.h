#ifndef SHUNTING_EXCITATION_NEURON_H
#define SHUNTING_EXCITATION_NEURON_H

#include "ANNarchy.h"

// Subclass of annarNeuron: leaky integrator with positive transfer function
class ShuntingExcitationNeuron : public annarNeuron
{
    public:

        ShuntingExcitationNeuron(class annarPopulation* population, int rank):  annarNeuron(population, rank){

            tau_ = 10.0;
            threshold_ = 0.0;

        };

        virtual void step(){

            // Update the membrane potential
            mp_ += dt_ /tau_ * (-mp_ + sum("exc") - positive(1.0 - sum("exc")) * sum("inh") + baseline_ + noise_*(2.0*rand_num -1.0));

            // Update the instantaneous firing rate
            rate_ = positive(mp_ - threshold_);

        };

    protected:

        @PARAMETER FLOAT threshold_; // Threshold for the firing rate
};

#endif
