#ifndef GATED_DIPOLE_H
#define GATED_DIPOLE_H

#include "ANNarchy.h"

ANNARCHY_EXPORT(fb_exc, double, fb_exc_, "Excitatory influence of feedback connections.")
ANNARCHY_EXPORT(fb_mod, double, fb_mod_, "Modulatory influence of feedback connections on feedforward connections.")
ANNARCHY_EXPORT(threshold, double, threshold_, "Firing threshold for the membrane potential.")
ANNARCHY_EXPORT(tau_adaptation_drive, double, tau_adaptation_drive_, "Rate adaptation for drive.")
ANNARCHY_EXPORT(tau_adaptation_input, double, tau_adaptation_input_, "Rate adaptation for input.")
ANNARCHY_EXPORT(tau_adaptation_rate, double, tau_adaptation_rate_, "Rate adaptation for rate.")


// Subclass of annarNeuron
class GatedDipole : public annarNeuron
{

public:

    GatedDipole(class annarPopulation* population, int rank):  annarNeuron(population, rank){
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
        
        fb_exc_=0.5;
        fb_mod_=0.5;
    };

    virtual void step(){
    
        drive_=sum("DRIVE");
        input_=sum("FF");
        inhib_=sum("LAT");
        fb_=sum("FB");
        
        adapted_input_+= 1.0/tau_adaptation_input_ * (input_ - adapted_input_);
        adapted_rate_+= 1.0/tau_adaptation_rate_ * (rate_ - adapted_rate_);
        adapted_drive_+= 1.0/tau_adaptation_drive_ * (adapted_rate_*drive_ - adapted_drive_);
        
        mp_+= 1.0 /tau_ * (-mp_ + positive(input_ - adapted_input_)*positive(drive_-adapted_drive_) 
                                - adapted_rate_* inhib_ 
                                + positive(drive_-adapted_drive_) 
                                + baseline_ + noise_*(2.0*rand_num-1.0));
        
        //  * (1.0 + fb_mod_*fb_ ) + fb_exc_*fb_
        
        rate_=positive(mp_-threshold_);
    };


protected:
    double threshold_, drive_, input_, inhib_, fb_;
    double fb_mod_, fb_exc_;
    double adapted_input_, tau_adaptation_input_;
    double adapted_drive_, tau_adaptation_drive_;
    double adapted_rate_, tau_adaptation_rate_;

};




#endif
