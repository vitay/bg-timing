#ifndef DACOPY_H
#define DACOPY_H

#include "ANNarchy.h"

class DA_Copy : public annarLearningRule
{
    public:
        DA_Copy():  annarLearningRule(){
        
            tau_=1000.0;
            min_value_=0.0;
            
            K_LTD_=1.0;
            tau_dopa_=300.0;
            
            K_alpha_=10.0;
            tau_alpha_=10.0;
            regularization_threshold_=1.0;
            
            DA_threshold_positive_=0.65;
            DA_threshold_negative_=0.35;
            DA_K_positive_=1.0;
            DA_K_negative_=1.0;
            
            weighted_sum = 0.0;
            
            alpha_=0.0;
            dopa_mod = 0.0;
            post_rate = 0.0;
            mean_post = 0.0;
            dopa = 0.0;
            dopa_mean = 0.5;
            
        };
        
        virtual void step(){
              
            // Retrieve values
            post_rate = neuron_->getRate();
            mean_post = neuron_->getPopulation()->getMean();
            dopa = neuron_->sum("dopa");
            weighted_sum = neuron_->sum(connection_type_);
            activation = neuron_->sum("exc");
            
            // Regularization constraint
            alpha_ += dt_/tau_alpha_*((activation > 0.1? positive(weighted_sum - activation) : 0.0) - alpha_);

            // Dopaminergic modulation
            dopa_mean += dt_/tau_dopa_ * (dopa - dopa_mean);
            dopa_mod = (dopa > DA_threshold_positive_? 
                            DA_K_positive_*(dopa - DA_threshold_positive_)
                            //DA_K_positive_*positive(dopa - dopa_mean) 
                       : (dopa < DA_threshold_negative_? 
                            DA_K_negative_*(dopa -DA_threshold_negative_) 
                          : 0.0)
                        );      
        
        }


        virtual void learn(annarSynapse* synapse){

            FLOAT weight = synapse->getValue();
            FLOAT pre_rate= synapse->preRate();
            FLOAT mean_pre = synapse->getPre()->getPopulation()->getMean();

            FLOAT delta = 0.0;
            if((post_rate>mean_post)&&(pre_rate>mean_pre)) { 
                // Both cells are active:  LTP with DA modulation and regularization
                //delta = dopa_mod * (post_rate - mean_post)* (pre_rate - mean_pre) - K_alpha_*alpha_* positive(post_rate - mean_post)* positive(post_rate - mean_post) * weight;
                delta = dopa_mod * (post_rate - mean_post)* (pre_rate - mean_pre) * positive(activation - weighted_sum);
            }
            else{
                if((post_rate<mean_post)&&(pre_rate<mean_pre)) { 
                    // No cell is active: no learning
                    delta=0.0;
                }
                else{ 
                    // Only one of the cells is active: LTD
                    delta= K_LTD_ * dopa_mod * (post_rate - mean_post)* (pre_rate - mean_pre);
                }
            }
            
            weight += dt_/tau_*delta;
            if(weight < min_value_)
                weight = min_value_;
            synapse->setValue(weight);

        }

    protected:
    
        @PARAMETER FLOAT tau_; // Learning rate
        @PARAMETER FLOAT min_value_; // Minimal value
        @PARAMETER FLOAT regularization_threshold_; // Regularization threshold
        @PARAMETER FLOAT K_alpha_; // regularization coefficient
        @PARAMETER FLOAT tau_alpha_; // Time constant coefficient
        @PARAMETER FLOAT tau_dopa_; // Time constant coefficient
        @PARAMETER FLOAT DA_threshold_positive_; // DA threshold
        @PARAMETER FLOAT DA_threshold_negative_; // DA threshold
        @PARAMETER FLOAT DA_K_positive_; // DA coefficient
        @PARAMETER FLOAT DA_K_negative_; // DA coefficient
        @PARAMETER FLOAT K_LTD_; // LTD ratio

        // Internal
        FLOAT alpha_, post_rate, mean_post, dopa, dopa_mod, dopa_mean, weighted_sum, activation;

};




#endif
