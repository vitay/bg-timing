#ifndef GABA_TONIC_H
#define GABA_TONIC_H

#include "ANNarchy.h"

class GABATonic : public annarLearningRule
{
    public:
    
        GABATonic():  annarLearningRule(){
        
            tau_=1000.0;
            min_value_=0.0;
            max_value_=1.0;
            
        };

        virtual void learn(class annarSynapse* synapse){

            FLOAT post_rate= neuron_->getRate();
            FLOAT post_baseline= neuron_->getBaseline();

            FLOAT weight=synapse->getValue();
            FLOAT pre_rate= synapse->preRate();

            FLOAT delta= positive(post_baseline - post_rate )* post_rate* pre_rate ;

            weight += dt_/tau_ * delta;
            if(weight < min_value_)
                weight = min_value_;
            else if(weight > max_value_)
                weight = max_value_;
            synapse->setValue(weight);

        }

    protected:
    
        @PARAMETER FLOAT tau_; // Learning rate
        @PARAMETER FLOAT min_value_; // Minimal value
        @PARAMETER FLOAT max_value_; // Maximal value
};

#endif
