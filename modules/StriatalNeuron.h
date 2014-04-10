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
            threshold_ = 0.0;
            membrane_up_ = 0.5;
            tau_state_=1000.0;
            max_rate_ = 1.1;
            
            up_down_=false;
            upstate_=0.0;
            mp_up_=0.0;
            
            threshold_exc_=1.0;
            threshold_dopa_=0.7;
            
            vmpfc_ = 0.0;
            bla_ = 0.0;
        };

        virtual void step(){
        
            // Store weigthed sums
            vmpfc_ = sum("mod");
            bla_ = sum("exc");
        
            // Updating the state of the neuron
            if(up_down_){ // up-state
                upstate_+= dt_/tau_state_* ( 0.0 - upstate_); 
                if( (upstate_<0.05)  || (upstate_ < 0.2 && sum("dopa") < threshold_dopa_)  ){ // transition to the down-state
                    up_down_=false;
                    upstate_=0.0;
                    mp_up_ = 0.0;
                }  
            }
            else{ // down-state
                upstate_+= dt_/tau_state_* ( 1.0 -upstate_);
                if( (sum("dopa")>threshold_dopa_) || (vmpfc_ + bla_ > threshold_exc_) || (upstate_>0.95)  ){ // transition to the up-state
                    up_down_=true;
                    upstate_=1.0;
                    mp_up_ = membrane_up_;
                }
            }
                
        
            // Firing rate of the neuron
            mp_+= dt_ /tau_ * (-mp_ + vmpfc_ + bla_ - sum("inh") + 0.5* sum("dopa") + mp_up_  + baseline_ + noise_*(2.0*rand_num-1.0));
            
            rate_=positive(mp_-threshold_);
            
            if(rate_> max_rate_)
                rate_= max_rate_;        
        };


    protected:
        @PARAMETER FLOAT threshold_; // Threshold for the mp
        @PARAMETER FLOAT membrane_up_; // Increase of mp when in up-state
        @PARAMETER FLOAT dopa_mod_; // Modulatory influence of dopamine
        @PARAMETER FLOAT tau_state_; // Time constant of the up/down state variable
        @PARAMETER FLOAT max_rate_; // Maximum firing rate

        @PARAMETER FLOAT threshold_exc_; // Threshold for the excitatory inputs
        @PARAMETER FLOAT threshold_dopa_; // Threshold for the dopamine
        
        @VARIABLE FLOAT upstate_; // Up-state
        @VARIABLE FLOAT mp_up_; // Increase of mp when in up-state
              
        @VARIABLE FLOAT vmpfc_; // input from vmPFC
        @VARIABLE FLOAT bla_; // input from BLA
        
        // Internals
        bool up_down_;

};




#endif
