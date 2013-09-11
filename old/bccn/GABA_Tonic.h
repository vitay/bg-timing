#ifndef GABA_TONIC_H
#define GABA_TONIC_H

#include "ANNarchy.h"

ANNARCHY_EXPORT(tau, double, tau_, "Time constant of learning.")
ANNARCHY_EXPORT(min_value, double, min_value_, "Minimum value of a weight.")
ANNARCHY_EXPORT(max_value, double, max_value_, "Maximum value of a weight.")
ANNARCHY_EXPORT(DA_threshold_positive, double, DA_threshold_positive_, "Threshold where DA induces LTP.")
ANNARCHY_EXPORT(DA_threshold_negative, double, DA_threshold_negative_, "Threshold where DA induces LTD.")
ANNARCHY_EXPORT(DA_K_positive, double, DA_K_positive_, "Coefficient of DA-induced LTP.")
ANNARCHY_EXPORT(DA_K_negative, double, DA_K_negative_, "Coefficient of DA-induced LTD.")


// Subclass of annarLearningRule
class GABA_Tonic : public annarLearningRule
{
public:
    GABA_Tonic():  annarLearningRule(){
        tau_=1000.0;
        min_value_=0.0;
        max_value_=0.0;
        DA_threshold_positive_=0.65;
        DA_threshold_negative_=0.35;
        DA_K_positive_=1.0;
        DA_K_negative_=1.0;
    };

    double positive(double x) {return (x>0.0?x:0.0);};

    virtual void learn(vector<class annarWeight*> weights){

        double post_rate= neuron_->getRate();
        double post_baseline= neuron_->getBaseline();

/*        double dopa=neuron_->sum("DOPA");
        double dopa_mod= (dopa>DA_threshold_positive_? DA_K_positive_*(dopa - DA_threshold_positive_) : (dopa < DA_threshold_negative_? DA_K_negative_*(dopa -DA_threshold_negative_) : 0.0));
*/
        double delta=0.0;

        for(int i=0; i<weights.size(); i++){

            double weight=weights[i]->getValue();
            double pre_rate= weights[i]->preRate();

            delta= positive(post_baseline - post_rate )* post_rate* (pre_rate ) ;

            weights[i]->incValue(delta/tau_); // Increase the weight
            if(weights[i]->getValue()<min_value_) // Lower threshold
                weights[i]->setValue(min_value_);
            else if(weights[i]->getValue()>max_value_)
                weights[i]->setValue(max_value_);
        }
    }

protected:
    double tau_, min_value_, max_value_;
    double DA_threshold_positive_, DA_threshold_negative_;
    double DA_K_positive_, DA_K_negative_;
};




#endif
