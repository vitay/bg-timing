# Timing and expectation of reward

Code for the model described in:

Vitay, J. and Hamker, F. (2014). Timing and expectation of reward: a neuro-computational model of the afferents to the ventral tegmental area. *Frontiers in Neurorobotics* 8(4), [doi:10.3389/fnbot.2014.00004](http://dx.doi.org/10.3389/fnbot.2014.00004)  

## Run the code

The code currently works with ANNarchy 4.5.3 [https://bitbucket.org/annarchy/annarchy](https://bitbucket.org/annarchy/annarchy).

To display the main figures of the article, simply run:

~~~~ {.python}
python GenerateFigures.py
~~~~

## Content

* `GenerateFigures.py`: main script to generate the figures. 
* `PlotDopamine.py`: generates only the figure for dopamine.
* `TimingNetwork.py`: definition of the network. 
* `NeuronDefinition.py`: definition of the neurons. 
* `SynapseDefinition.py`: definition of the synapses. 
* `ConnectorDefinition.py`: definition of the custom connection patterns. 
* `TrialDefinition.py`: definition of the trial procedures. 
