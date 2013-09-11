#ifndef DAHEBB_H
#define DAHEBB_H

#include "ANNarchy.h"
#include "FeedbackNeuron.h"

ANNARCHY_EXPORT(tau, double, tau_, "Time constant of learning.")
ANNARCHY_EXPORT(K_alpha, double, K_alpha_, "Constant multiplier for the regularization constraint.")
ANNARCHY_EXPORT_VARIABLE(alpha, double, alpha_, "Variable used for regularization.")
ANNARCHY_EXPORT(tau_alpha, double, tau_alpha_, "Time constant for alpha.")
ANNARCHY_EXPORT(DA_threshold_positive, double, DA_threshold_positive_, "Threshold where DA induces LTP.")
ANNARCHY_EXPORT(DA_threshold_negative, double, DA_threshold_negative_, "Threshold where DA induces LTD.")
ANNARCHY_EXPORT(DA_K_positive, double, DA_K_positive_, "Coefficient of DA-induced LTP.")
ANNARCHY_EXPORT(DA_K_negative, double, DA_K_negative_, "Coefficient of DA-induced LTD.")


// Subclass of annarLearningRule
class DA_Hebb : public annarLearningRule
{
public:
    DA_Hebb():  annarLearningRule(){
        tau_=1000.0;
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


        double delta=0.0;

        for(int i=0; i<weights.size(); i++){

            double weights[i]->getValue();
            double pre_rate= weights[i]->preRate();

            delta= dopa_mod * (post_rate )* (pre_rate) - K_alpha_*alpha_* post_rate* post_rate * weight;

            weights[i]->incValue(delta/tau_);
            if(weights[i]->getValue()<0.0)
                weights[i]->setValue(0.0);
        }
    }


protected:
    double tau_;
    double K_alpha_;
    double alpha_, tau_alpha_;
    double DA_threshold_positive_, DA_threshold_negative_;
    double DA_K_positive_, DA_K_negative_;
};




#endif
