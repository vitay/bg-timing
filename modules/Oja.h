#ifndef OJA_H
#define OJA_H

#include "ANNarchy.h"


class Oja : public annarLearningRule
{
    public:
        Oja():  annarLearningRule(){
        
            tau_=1000.0;
            K_alpha_=10.0;
            tau_alpha_=10.0;
            
            alpha_=0.0;
        
        };


        virtual void learn(annarSynapse* synapse){

            // Pre- and post-synaptic firing rates
            FLOAT post_rate= neuron_->getRate();        
            FLOAT weight=synapse->getValue();
            FLOAT pre_rate= synapse->preRate();
            
            // Regularization constraint
            alpha_=positive(alpha_+dt_/tau_alpha_*(positive(post_rate - 1.0) - alpha_));

            weight += dt_/tau_ *(post_rate * pre_rate - K_alpha_ * alpha_ * post_rate * post_rate * weight );
            
            if(weight < 0.0)
                weight = 0.0;
            synapse->setValue(weight);
        }


    protected:
        @PARAMETER FLOAT tau_; // Rate of learning
        @PARAMETER FLOAT K_alpha_; // Regularization coefficient
        @PARAMETER FLOAT tau_alpha_; // Time constant regularization
        @VARIABLE FLOAT alpha_; // Regularization variable
    };




#endif
