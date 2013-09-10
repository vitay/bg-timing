#ifndef GATED_NEURON_H
#define GATED_NEURON_H

#include "ANNarchy.h"

// Subclass of annarNeuron: leaky integrator with positive transfer function
class GatedNeuron : public annarNeuron
{
    public:

        GatedNeuron(class annarPopulation* population, int rank):  annarNeuron(population, rank){

            tau_=10.0;
            threshold_=0.0;
            baseline_=0.5;
            noise_=0.1;
            
            drive_=0.0;
            input_=0.0;
            inhib_=0.0;
            fb_=0.0;
            
            adapted_input_=0.0;
            adapted_drive_=0.0;
            adapted_rate_=0.0;
            
            tau_adaptation_input_=500.0;
            tau_adaptation_drive_=500.0;
            tau_adaptation_rate_=500.0;

        };

        virtual void step(){

            drive_=sum("drive");
            input_=sum("exc");
            inhib_=sum("inh");
            fb_=sum("mod");
            
            adapted_input_ += 1.0/tau_adaptation_input_ * (input_ - adapted_input_);
            adapted_rate_ += 1.0/tau_adaptation_rate_ * (rate_ - adapted_rate_);
            adapted_drive_ += 1.0/tau_adaptation_drive_ * (adapted_rate_*drive_ - adapted_drive_);
            
            mp_+= 1.0 /tau_ * (-mp_ + positive(input_ - adapted_input_)*positive(drive_-adapted_drive_) 
                                    - adapted_rate_* inhib_ 
                                    + positive(drive_-adapted_drive_) 
                                    + baseline_ + noise_*(2.0*rand_num-1.0));
            
            rate_=positive(mp_-threshold_);

        };

    protected:

        @PARAMETER FLOAT threshold_; // Threshold for the firing rate
        @PARAMETER FLOAT tau_adaptation_input_; // Time constant adaptation input
        @PARAMETER FLOAT tau_adaptation_drive_; // Time constant adaptation drive
        @PARAMETER FLOAT tau_adaptation_rate_; // Time constant adaptation rate
        
        @VARIABLE FLOAT drive_; // Drive to this channel
        @VARIABLE FLOAT input_; // Gustatory input
        @VARIABLE FLOAT inhib_; // inhibition from the other channel
        @VARIABLE FLOAT fb_; // Feedback from the amygdala
        
        @VARIABLE FLOAT adapted_input_; // Adapted input
        @VARIABLE FLOAT adapted_drive_; // Adapted drive
        @VARIABLE FLOAT adapted_rate_; // Adapted rate
};

#endif
