#ifndef ANTIHEBB_H
#define ANTIHEBB_H

#include "ANNarchy.h"

ANNARCHY_EXPORT(tau, double, tau_, "Time constant of learning.")
ANNARCHY_EXPORT(theta, double, theta_, "Decrease rate of the connection when both cells are uncorrelated.")
ANNARCHY_EXPORT(max, double, max_, "Maximum value of the weight.")


// Subclass of annarLearningRule
class AntiHebb : public annarLearningRule
{
public:
    AntiHebb():  annarLearningRule(){
        tau_=10000.0;
        theta_=0.01;
        max_=1.0;
    };

    double positive(double x) {return (x>0.0?x:0.0);};

    virtual void learn(vector<class annarWeight*> weights){

        double post_rate= neuron_->getRate();

        double mean_post=neuron_->getPopulation()->getMean();

        double delta=0.0;

        for(int i=0; i<weights.size(); i++){

            double weight=weights[i]->getValue();
            double pre_rate= weights[i]->preRate();
            double mean_pre=weights[i]->getPre()->getPopulation()->getMean();

            if((post_rate>mean_post)&&(pre_rate>mean_pre)) {
                delta= (post_rate - mean_post)* (pre_rate - mean_pre) - theta_*weight;
            }
            weights[i]->incValue(delta/tau_);
            if(weights[i]->getValue()<0.0)
                weights[i]->setValue(0.0);
            else if(weights[i]->getValue()>max_)
                weights[i]->setValue(max_);
        }
    }


protected:
    double tau_;
    double theta_;
    double max_;
};




#endif
