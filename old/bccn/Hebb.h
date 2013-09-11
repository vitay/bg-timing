#ifndef HEBB_H
#define HEBB_H

#include "ANNarchy.h"

ANNARCHY_EXPORT(tau, double, tau_, "Time constant of learning.")
ANNARCHY_EXPORT(K_alpha, double, K_alpha_, "Constant multiplier for the regularization constraint.")
ANNARCHY_EXPORT_VARIABLE(alpha, double, alpha_, "Variable used for regularization.")
ANNARCHY_EXPORT(tau_alpha, double, tau_alpha_, "Time constant for alpha.")


// Subclass of annarLearningRule
class Hebb : public annarLearningRule
{
public:
    Hebb():  annarLearningRule(){
        tau_=1000.0;
        K_alpha_=10.0;
        alpha_=0.0;
        tau_alpha_=10.0;
    };

    double positive(double x) {return (x>0.0?x:0.0);};

    virtual void learn(vector<class annarWeight*> weights){

        double post_rate= neuron_->getRate();
        alpha_=positive(alpha_+1.0/tau_alpha_*(positive(post_rate-1.0)-alpha_));

        double delta=0.0;

        for(int i=0; i<weights.size(); i++){

            double weight=weights[i]->getValue();
            double pre_rate= weights[i]->preRate();

            delta=  (post_rate )* (pre_rate) - K_alpha_*alpha_* post_rate* post_rate * weight;

            weights[i]->incValue(delta/tau_);
            if(weights[i]->getValue()<0.0)
                weights[i]->setValue(0.0);
        }
    }


protected:
    double tau_;
    double K_alpha_;
    double alpha_, tau_alpha_;
};




#endif
