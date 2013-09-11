#ifndef VALUATION_TASK_H
#define VALUATION_TASK_H

#include "ANNarchy.h"
#include <math.h>
#include "ConditioningTask.h"

ANNARCHY_EXPORT(duration, int, duration_, "Duration of a single stimulus presentation.")
ANNARCHY_EXPORT(trial_duration, int, trial_duration_, "Duration of a trial.")
ANNARCHY_EXPORT_VARIABLE(time, int, time_, "Local time for the environment.")
ANNARCHY_FUNCTION(reset, reset, "Reinitializes the environment.")
ANNARCHY_FUNCTION(remove_inputs, remove_inputs, "Removes inputs to the network.")
ANNARCHY_FUNCTION(pause, pause, "Pauses the environment.")
ANNARCHY_FUNCTION(stop, stop, "Stops the environment.")
ANNARCHY_FUNCTION(start, start, "Starts the environment.")
ANNARCHY_FUNCTION(transition, transition, "Switches to the next event in the sequence.")
ANNARCHY_EXPORT(gustatory, string, gus_, "Name of the population receiving gustatory inputs.")


// Subclass of annarWorld
class ValuationTask : public annarWorld
{
public:

    // Constructor
    ValuationTask(annarNetwork* net)   :  annarWorld(net){

        time_=0;
        run_=true;
        duration_=1000;
        gus_="GUS";

        sequence_.push_back(US1);
        sequence_.push_back(ITI);
        sequence_.push_back(ITI);
        sequence_.push_back(US2);
        sequence_.push_back(ITI);
        sequence_.push_back(ITI);
        rank_=0;
        
        trial_duration_=sequence_.size()*duration_;
     
    };

    // Function called at each timestep before the neurons are updated
    virtual void step(){

        if(run_){
            if(time_%duration_==0){
                transition();
            }
            time_++;
        }

    }
    
    void remove_inputs(){
        class annarPopulation* gus = net_->getPopulation(gus_);

        for(int i=0; i<gus->nbNeurons(); i++){
             gus->getNeuron(i)->setBaseline(0.0);
        }   
    }

    void transition(){
        class annarPopulation* gus = net_->getPopulation(gus_);

        switch(sequence_[rank_]){
            case ITI:
                remove_inputs();
                break;
            case US1:
                gus->getNeuron(0)->setBaseline(1.0);
                gus->getNeuron(1)->setBaseline(1.0);
                gus->getNeuron(2)->setBaseline(0.0);
                gus->getNeuron(3)->setBaseline(0.0);
                break;
            case US2:
                gus->getNeuron(0)->setBaseline(1.0);
                gus->getNeuron(1)->setBaseline(0.0);
                gus->getNeuron(2)->setBaseline(1.0);
                gus->getNeuron(3)->setBaseline(0.0);
                break;
            default:
                break;
        }
        rank_++;
        if(rank_==sequence_.size())
            rank_=0;
    };

    void reset(){
        time_=0;
    };

    void pause(){
        run_=!run_;
    };
    
    void stop(){
        run_=false;
    };
    
    void start(){
        run_=true;
    };


protected:

    int trial_duration_, duration_, time_;
    string gus_;

    int rank_;
    vector<int> sequence_;

    bool run_;
};




#endif
