#ifndef STRIATAL_NEURON_H
#define STRIATAL_NEURON_H

#include "ANNarchy.h"


// Subclass of annarNeuron
class StriatalNeuron : public annarNeuron
{

    public:

        StriatalNeuron(class annarPopulation* population, int rank):  annarNeuron(population, rank){
        
            tau_=10.0;
            
            dopa_mod_=0.0;
            threshold_up_=0.0;
            threshold_down_=0.5;
            tau_state_=1000.0;
            max_rate_ = 1.1;
            
            up_down_=false;
            upstate_=0.0;
            downstate_=0.0;
            
            threshold_exc_=1.0;
            threshold_dopa_=0.7;
            
        };

        virtual void step(){
        
            // Updating the state of the neuron
            if(up_down_){ // up-state
                upstate_+= dt_/tau_state_* ( 0.0 - upstate_); 
                if( (upstate_<0.05) || (upstate_ < 0.5 && rate_ < 0.1) || (sum("dopa")< 0.1) ){ // transition to the down-state
                    up_down_=false;
                    upstate_=0.0;
                }  
            }
            else{ // down-state
                upstate_+= dt_/tau_state_* ( 1.0 -upstate_);
                if( (sum("dopa")>threshold_dopa_) || (sum("exc")>threshold_exc_) || (upstate_>0.95)  ){ // transition to the up-state
                    up_down_=true;
                    upstate_=1.0;
                }
            }
                
        
            // Firing rate of the neuron
            mp_+= dt_ /tau_ * (-mp_ + sum("exc") -  sum("inh") + baseline_ + noise_*(2.0*rand_num-1.0));
            
            if(up_down_){
                rate_=positive(mp_-threshold_up_);
            }else{
                rate_=positive(mp_-threshold_down_);
            }
            if(rate_> max_rate_)
                rate_= max_rate_;        
        };


    protected:
        @PARAMETER FLOAT threshold_up_; // Threshold for the transition to up-state
        @PARAMETER FLOAT threshold_down_; // Threshold for the transition to down-state
        @PARAMETER FLOAT dopa_mod_; // Modulatory influence of dopamine
        @PARAMETER FLOAT tau_state_; // Time constant of the up/down state variable
        @PARAMETER FLOAT max_rate_; // Maximum firing rate

        @PARAMETER FLOAT threshold_exc_; // Threshold for the excitatory inputs
        @PARAMETER FLOAT threshold_dopa_; // Threshold for the dopamine
        
        @VARIABLE FLOAT upstate_; // Up-state
        @VARIABLE FLOAT downstate_; // Down-state
              
        // Internals
        bool up_down_;

};




#endif
