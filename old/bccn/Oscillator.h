#ifndef OSCILLATOR_NEURON_H
#define OSCILLATOR_NEURON_H

#include "ANNarchy.h"
#include "math.h"

ANNARCHY_EXPORT_VARIABLE(freq, double, freq_, "Frequency of the oscillation.")
ANNARCHY_EXPORT_VARIABLE(delay, int, delay_, "Maximal phase of the oscillator between each trial.")


// Subclass of annarNeuron
class Oscillator : public annarNeuron
{

public:

    Oscillator(class annarPopulation* population, int rank):  annarNeuron(population, rank){
        tau_=10.0;
        baseline_=0.0;
        noise_=0.1;
        freq_=1.0;
        t_=0;
        skew_=0;
        delay_=0;
        oscillating_=false;
        reward_modulation_=1.0;
    };

    virtual void step(){
           
        if(!oscillating_&&(sum("FF")>0.8)){
            oscillating_=true;
            skew_=0;//int(delay_*rand_num); // Skew is randomly chosen between 0 and delay_
            //cout<< delay_ << " " << tz_ << endl;
        }else if (oscillating_&&(sum("FF")<0.2)){
            oscillating_=false;
            t_=0; // Reset internal time
        }
        
        //reward_modulation_=min(1.0, sum("FB"));
        
        mp_+= 1.0 /tau_ * (-mp_ + (oscillating_?reward_modulation_*(1.0-exp(-t_/500.0))*threshold(sin(2.0*3.1415*freq_*double(t_+ skew_)/1000.0)):0.0) + baseline_ + noise_*(2.0*rand_num-1.0));
        rate_=positive(mp_);
        if(oscillating_)
            t_++;

    };

    double threshold(double x){
        return (x>0.5? x : 0.0);
    }

protected:
    double freq_;
    int t_, skew_, delay_;
    bool oscillating_;
    double reward_modulation_;

};




#endif
