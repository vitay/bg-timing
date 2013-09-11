#ifndef DATIMECOVARIANCE_H
#define DATIMECOVARIANCE_H

#include "ANNarchy.h"

ANNARCHY_EXPORT(tau, double, tau_, "Time constant of learning.")
ANNARCHY_EXPORT(min_value, double, min_value_, "Minimum value of a weight.")
ANNARCHY_EXPORT(K_alpha, double, K_alpha_, "Constant multiplier for the regularization constraint.")
ANNARCHY_EXPORT_VARIABLE(alpha, double, alpha_, "Variable used for regularization.")
ANNARCHY_EXPORT(tau_alpha, double, tau_alpha_, "Time constant for alpha.")
ANNARCHY_EXPORT(DA_threshold_positive, double, DA_threshold_positive_, "Threshold where DA induces LTP.")
ANNARCHY_EXPORT(DA_threshold_negative, double, DA_threshold_negative_, "Threshold where DA induces LTD.")
ANNARCHY_EXPORT(DA_K_positive, double, DA_K_positive_, "Coefficient of DA-induced LTP.")
ANNARCHY_EXPORT(DA_K_negative, double, DA_K_negative_, "Coefficient of DA-induced LTD.")


// Subclass of annarLearningRule
class DA_TimedCovariance : public annarLearningRule
{
public:
    DA_TimedCovariance():  annarLearningRule(){
        tau_=1000.0;
        min_value_=0.0;
        K_alpha_=10.0;
        alpha_=0.0;
        tau_alpha_=10.0;
        DA_threshold_positive_=0.65;
        DA_threshold_negative_=0.35;
        DA_K_positive_=1.0;
        DA_K_negative_=1.0;
        
        
    };

    double positive(double x) {return (x>0.0?x:0.0);};

    virtual void learn(vector<class annarWeight*> weights){

        double post_rate= neuron_->getRate();
        alpha_=positive(alpha_+1.0/tau_alpha_*(positive(post_rate-1.0)-alpha_));

        double dopa=neuron_->sum("DOPA");
        double dopa_mod= (dopa>DA_threshold_positive_? DA_K_positive_*(dopa - DA_threshold_positive_) : (dopa < DA_threshold_negative_? DA_K_negative_*(dopa -DA_threshold_negative_) : 0.0));

        double mean_post=neuron_->getPopulation()->getMean();

        double delta=0.0;

        vector<annarWeight*>::iterator w_it;
        for(w_it=weights.begin(); w_it!=weights.end();w_it++){

            double weight=(*w_it)->getValue();
            double pre_rate= (*w_it)->preRate();
            double mean_pre=(*w_it)->getPre()->getPopulation()->getMean();

            if((post_rate>mean_post)&&(pre_rate>mean_pre)) { // Both cells are active:  LTP with DA modulation and regularization
                delta= dopa_mod * (post_rate - mean_post)* (pre_rate - mean_pre) - K_alpha_*alpha_* positive(post_rate - mean_post)* positive(post_rate - mean_post) * positive(weight);
            }
            else{
                if((post_rate<mean_post)&&(pre_rate<mean_pre)) { // No cell is active: no learning
                    delta=0.0;
                }
                else{ // Only one of the cells is active: LTD
                    delta= dopa_mod * (post_rate - mean_post)* (pre_rate - mean_pre);
                }
            }

            (*w_it)->incValue(delta/tau_); // Increase the weight
            if((*w_it)->getValue()<min_value_) // Lower threshold
                (*w_it)->setValue(min_value_);
        }
    }

protected:
    double tau_, min_value_;
    double K_alpha_;
    double alpha_, tau_alpha_;
    double DA_threshold_positive_, DA_threshold_negative_;
    double DA_K_positive_, DA_K_negative_;
};




#endif
