#ifndef BCM_H
#define BCM_H
#include "ANNarchy.h"

class BCM : public annarLearningRule {

    public:
        BCM():  annarLearningRule(){

            theta_ = 0.0;
            tau_theta_ = 10000.0;
            tau_ = 1000.0;
            threshold_pre_ = 0.0;

        };

        virtual void step() {

            theta_ += dt_ / tau_theta_ * ( neuron_->getRate() - theta_);
        }

        virtual void learn(annarSynapse* synapse) {

            // Access pre- and post-synaptic firing rates
            FLOAT post_rate = neuron_->getRate();
            FLOAT pre_rate = synapse->preRate();
            FLOAT value = synapse->getValue();

            // Compute the synaptic variation
            FLOAT delta = post_rate * ( post_rate - theta_ ) * (1.0 - value) * pre_rate * (pre_rate - threshold_pre_) ;

            // Update the synaptic strength
            if(value + delta*dt_/tau_ < 0.0){
                synapse->setValue(0.0);
            } 
            else{
                synapse->incValue(delta*dt_/tau_);
            }

        }
    protected:

        @VARIABLE FLOAT theta_; // Sliding threshold
        @PARAMETER FLOAT tau_theta_; // Time constant for the sliding threshold
        @PARAMETER FLOAT tau_; // Time constant for the learning rule
        @PARAMETER FLOAT threshold_pre_; // Threshold on presynaptic activity

};
#endif
