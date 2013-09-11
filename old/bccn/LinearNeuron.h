#ifndef LINEAR_NEURON_H
#define LINEAR_NEURON_H

#include "ANNarchy.h"


ANNARCHY_EXPORT(threshold, double, threshold_, "Threshold over the membrane potential.")

// Subclass of annarNeuron
class LinearNeuron : public annarNeuron
{

public:

    LinearNeuron(class annarPopulation* population, int rank):  annarNeuron(population, rank){
        tau_=10.0;
        noise_=0.0;
        threshold_=0.0;
    };

    virtual void step(){
        mp_+= 1.0 /tau_ * (-mp_ + sum() + baseline_ + noise_*(2.0*rand_num-1.0));
        rate_=(mp_ > threshold_ ? mp_ - threshold_: 0.0);
    }


protected:
    double threshold_;

};




#endif
