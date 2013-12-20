#############################
# Trial definition
#############################
from ANNarchy4 import *

# Sensitization trial
def sensitization_trial(trial_setup):
    # Loop over all US, in ascending order
    for us_def in trial_setup:
        # Reset
        simulate(500)
        # Set inputs
        get_population('LH').baseline = us_def['GUS']
        # Simulate for the desired duration
        simulate(us_def['duration'])
        # Reset
        get_population('LH').baseline = np.zeros(len(us_def['GUS']))
        simulate(500)

# Conditioning trial
def conditioning_trial(trial_setup):
    # Loop over all US, in ascending order
    for us_def in trial_setup:
        # Reset
        simulate(1000)
        # Set the CS input
        get_population('VIS').baseline = us_def['VIS']
        # Simulate for the desired duration
        simulate(us_def['duration'])
        # Set the US for 1s
        get_population('LH').baseline = us_def['magnitude']* us_def['GUS']
        simulate(1000)
        # Reset
        get_population('VIS').baseline = np.zeros(get_population('VIS').size)
        get_population('LH').baseline = np.zeros(get_population('LH').size)
        simulate(1000)
