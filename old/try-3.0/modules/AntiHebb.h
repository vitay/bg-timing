#ifndef ANTIHEBB_H
#define ANTIHEBB_H

#include "ANNarchy.h"

// Subclass of annarLearningRule: anti-hebbian learning rule
class AntiHebb : public annarLearningRule
{
    public:
        AntiHebb():  annarLearningRule(){
        
            tau_=10000.0;
            theta_=0.001;
            max_value_=1.0;
            threshold_pre_=0.0;
            threshold_post_=0.0;
            
        };

        // Global method called before learn()
        virtual void step() {
		    // Postsynaptic firing rate
            post_rate= neuron_->getRate();
	    }	

        // Update each connection of the same type
        virtual void learn(annarSynapse *synapse){
        
            // Retrieve synaptic information
		    FLOAT value	= synapse->getValue();
            FLOAT pre_rate = synapse->preRate();

            // Compute weight variation
            FLOAT delta	= positive(post_rate - threshold_post_ ) * positive(pre_rate - threshold_pre_) - theta_*value;
                
            // Update synaptic efficiency
            if(value + delta*dt_/tau_ <0.0)
                synapse->setValue(0.0);
            else if(value + delta*dt_/tau_ > max_value_)
                synapse->setValue(max_value_);
            else
                synapse->incValue(delta*dt_/tau_);
                
        }


    protected:
    
	    @VARIABLE FLOAT post_rate; // Firing rate of the post-synaptic neuron

        @PARAMETER FLOAT tau_; // Rate of learning.
        @PARAMETER FLOAT theta_; // Decrease rate of the connection when both cells are uncorrelated.
        @PARAMETER FLOAT max_value_; // Maximum value of the weight.
        @PARAMETER FLOAT threshold_pre_; // Threshold on presynaptic rate
        @PARAMETER FLOAT threshold_post_; // Threshold on postsynaptic rate
};




#endif
