#ifndef COVARIANCE_H
#define COVARIANCE_H

#include "ANNarchy.h"
#include "FeedbackNeuron.h"

ANNARCHY_EXPORT(tau, double, tau_, "Time constant of learning.")
ANNARCHY_EXPORT(K_alpha, double, K_alpha_, "Constant multiplier for the regularization constraint.")
ANNARCHY_EXPORT_VARIABLE(alpha, double, alpha_, "Variable used for regularization.")
ANNARCHY_EXPORT(tau_alpha, double, tau_alpha_, "Time constant for alpha.")
ANNARCHY_EXPORT(min_value, double, min_value_, "Minimum value of a weight.")


// Subclass of annarLearningRule
class Covariance : public annarLearningRule
{
public:
    Covariance():  annarLearningRule(){
        tau_=1000.0;
        K_alpha_=10.0;
        alpha_=0.0;
        tau_alpha_=10.0;
        min_value_=0.0;
    };

    double positive(double x) {return (x>0.0?x:0.0);};

    virtual void learn(vector<class annarWeight*> weights){

        double post_rate= neuron_->getRate();
        alpha_=positive(alpha_+1.0/tau_alpha_*(positive(post_rate-1.0)-alpha_));

        double mean_post=neuron_->getPopulation()->getMean();

        double delta=0.0;

        for(int i=0; i<weights.size(); i++){

            double weight=weights[i]->getValue();
            double pre_rate= weights[i]->preRate();
            double mean_pre=weights[i]->getPre()->getPopulation()->getMean();

            if((post_rate<mean_post)&&(pre_rate<mean_pre)) { // no learning
                delta=0.0;
            }else{ // LTD or LTP
                delta= (post_rate - mean_post)* (pre_rate - mean_pre) - K_alpha_*alpha_* positive(post_rate - mean_post)* positive(post_rate - mean_post) * weight;
            }
            
            weights[i]->incValue(delta/tau_); // Increase the weight
            if(weights[i]->getValue()<min_value_) // Lower threshold
                weights[i]->setValue(min_value_);
        }
    }


protected:
    double tau_;
    double K_alpha_;
    double alpha_, tau_alpha_;
    double min_value_;
};




#endif
