#ifndef DOPAMINE_NEURON_H
#define DOPAMINE_NEURON_H

#include "ANNarchy.h"

ANNARCHY_EXPORT(tau_decrease, double, tau_decrease_, "Decrease rate of the phasic response.")
ANNARCHY_EXPORT_VARIABLE(inhibition, double, inhibition_, "Modulatory inhibition received by the cell.")
ANNARCHY_EXPORT_VARIABLE(dip, double, dip_, "Pausing inhibition received by the cell.")


// Subclass of annarNeuron
class DopamineNeuron : public annarNeuron
{

public:

    DopamineNeuron(class annarPopulation* population, int rank):  annarNeuron(population, rank){
        
        tau_=50.0;
        tau_decrease_=50.0;
        noise_=0.1;
        mean_input_=0.0;
        baseline_=0.5;
        inhibition_=0.0;
        input_=0.0;
        dip_=0.0;
    };

    virtual void step(){
    
        input_=min(sum("FF"), 100.0);
        
        inhibition_ = sum("GABA");
        
        dip_=(input_<mean_input_+0.05? 3.0*positive(sum("LAT")-0.1):0.0);
        
        mean_input_+=1.0/tau_decrease_* (input_-mean_input_);
        
        mp_+= 1.0 /tau_ * (-mp_ + positive(input_-mean_input_) * positive(1.0 - inhibition_) - dip_ + baseline_ + noise_*(2.0*rand_num-1.0));
        
        rate_=positive(mp_);
        if(rate_>1.1)
            rate_=1.1;
    };


protected:
    double input_;
    double tau_decrease_;
    double mean_input_;
    double inhibition_;
    double dip_;

};




#endif
