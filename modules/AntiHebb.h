#ifndef ANTIHEBB_H
#define ANTIHEBB_H

#include "ANNarchy.h"


// Subclass of annarLearningRule
class AntiHebb : public annarLearningRule
{
    public:
    
        AntiHebb():  annarLearningRule(){
        
            tau_=10000.0;
            theta_=0.01;
            min_value_=0.0;
            max_value_=1.0;
            
        };

        virtual void step(){

            post_rate= neuron_->getRate();
            mean_post=neuron_->getPopulation()->getMean();
        }

        virtual void learn(annarSynapse* synapse){

            FLOAT weight=synapse->getValue();
            FLOAT pre_rate= synapse->preRate();
            FLOAT mean_pre=synapse->getPre()->getPopulation()->getMean();


            FLOAT delta=0.0;
            if((post_rate>mean_post)&&(pre_rate>mean_pre)) {
                delta= (post_rate - mean_post)* (pre_rate - mean_pre) - theta_*weight;
            }
            
            weight += dt_/tau_ * delta;
            if(weight < min_value_)
                weight = min_value_;
            else if(weight > max_value_)
                weight = max_value_;
            synapse->setValue(weight);

        }


    protected:
    
        @PARAMETER FLOAT tau_; // Learning rate
        @PARAMETER FLOAT theta_; // Decay rate
        @PARAMETER FLOAT min_value_; // Minimal value
        @PARAMETER FLOAT max_value_; // Maximal value
        
        // Internal
        FLOAT post_rate, mean_post;
};




#endif
