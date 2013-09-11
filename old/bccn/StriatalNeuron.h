#ifndef STRIATAL_NEURON_H
#define STRIATAL_NEURON_H

#include "ANNarchy.h"

ANNARCHY_EXPORT(dopa_mod, double, dopa_mod_, "Modulatory influence of dopaminergic connections on feedforward connections.")
ANNARCHY_EXPORT(threshold_up, double, threshold_up_, "Firing threshold for the membrane potential in the up-state.")
ANNARCHY_EXPORT(threshold_down, double, threshold_down_, "Firing threshold for the membrane potential in the down_state.")
ANNARCHY_EXPORT(tau_drive, double, tau_drive_, "Time constant of the up/down states.")
ANNARCHY_EXPORT(mod_drive, double, mod_drive_, "Amplitude of the up/down states.")
ANNARCHY_EXPORT_VARIABLE(upstate, double, upstate_, "Variable representing the up/down states.")
ANNARCHY_EXPORT_VARIABLE(threshold_nmda, double, threshold_nmda_, "Minimum NMDA input to drive the cell into the up-state.")
ANNARCHY_EXPORT_VARIABLE(threshold_dopa, double, threshold_dopa_, "Dopamine threshold to drive the cell into the up-state.")


// Subclass of annarNeuron
class StriatalNeuron : public annarNeuron
{

public:

    StriatalNeuron(class annarPopulation* population, int rank):  annarNeuron(population, rank){
        tau_=10.0;
        dopa_mod_=0.0;
        threshold_up_=0.0;
        threshold_down_=0.5;
        tau_drive_=1000.0;
        up_down_=false;
        upstate_=0.0;
        downstate_=0.0;
        mod_drive_=1.0;
        threshold_nmda_=1.0;
        threshold_dopa_=0.7;
    };

    virtual void step(){
    
        // Updating the state of the neuron
        if(up_down_){ // up-state
            upstate_+= 1.0/tau_drive_* ( 0.0 -upstate_); 
            if(upstate_<0.05){ // transition to the down-state
                up_down_=false;
                upstate_=0.0;
            }  
        }
        else{ // down-state
            upstate_+= 1.0/tau_drive_* ( 1.0 -upstate_);
            if(((sum("DOPA")>threshold_dopa_) || (sum("FF")>threshold_nmda_) ) && (upstate_>0.95)){ // transition to the up-state
                up_down_=true;
                upstate_=1.0;
            }
        }
            
    
        // Firing rate of the neuron
        mp_+= 1.0 /tau_ * (-mp_ + sum("FF") * (1.0 + dopa_mod_ *(sum("DOPA")-0.5) ) -  sum("LAT") + baseline_ + noise_*(2.0*rand_num-1.0));
        if(up_down_){
            rate_=positive(mp_-threshold_up_);
        }else{
            rate_=positive(mp_-threshold_down_);
        }
        if(rate_>1.2)
            rate_=1.2;        
    };


protected:
    double threshold_up_, threshold_down_, dopa_mod_, tau_drive_, upstate_, downstate_, mod_drive_, threshold_nmda_, threshold_dopa_;
    bool up_down_;

};




#endif
