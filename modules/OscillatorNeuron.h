#ifndef OSCILLATOR_NEURON_H
#define OSCILLATOR_NEURON_H

#include "ANNarchy.h"

class OscillatorNeuron : public annarNeuron
{

    public:

        OscillatorNeuron(class annarPopulation* population, int rank): annarNeuron(population, rank){
        
            tau_=10.0;
            baseline_=0.0;
            noise_=0.1;
            
            freq_=1.0;
            t_=0;
            oscillating_=false;
            
            start_oscillate_ = 0.8;
            stop_oscillate_ = 0.2;
            
        };

        virtual void step(){
               
            if(!oscillating_ && (sum("exc") > start_oscillate_)){
                oscillating_ = true;
            }
            else if (oscillating_ && (sum("exc") < stop_oscillate_)){
                oscillating_=false;
                t_=0; 
            }
            
            mp_+= 1.0 /tau_ * ( -mp_ 
                + (oscillating_? 
                        (1.0 - exp(-t_ / 500.0)) * above(sin(2.0*M_PI*freq_*FLOAT(t_)/1000.0))
                        : 0.0) 
                + baseline_ + noise_*(2.0*rand_num-1.0));
            
            rate_=positive(mp_);
            
            if(oscillating_)
                t_++;

        };

        FLOAT above(FLOAT x){
            return (x>0.5? x : 0.0);
        }

    protected:
        
        @PARAMETER FLOAT start_oscillate_; // value of input where it starts oscillating
        @PARAMETER FLOAT stop_oscillate_; // value of input where it stops oscillating
        
        @VARIABLE FLOAT freq_; // Frequency of the oscillator
        
        int t_; // Internal time of the oscillator
        bool oscillating_; // State of the oscillator

};


#endif
