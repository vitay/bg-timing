#############################
# Trial definition
#############################
from ANNarchy import *

# Sensitization trial
def sensitization_trial(trial_setup):
    # Loop over all US, in ascending order
    for us_def in trial_setup:
        # Reset
        simulate(500.0)
        # Set inputs
        get_population('LH').baseline = us_def['GUS']
        # Simulate for the desired duration
        simulate(us_def['duration'])
        # Reset
        get_population('LH').baseline = 0.0
        simulate(500.0)

# Conditioning trial
def conditioning_trial(trial_setup):
    # Loop over all US, in ascending order
    for us_def in trial_setup:
        # Reset
        simulate(1000.0)
        # Set the CS input
        get_population('VIS').baseline = us_def['VIS']
        # Simulate for the desired duration
        simulate(us_def['duration'])
        # Set the US for 1s
        get_population('LH').baseline = us_def['magnitude']* us_def['GUS']
        simulate(1000.0)
        # Reset
        get_population('VIS').baseline = 0.0
        get_population('LH').baseline = 0.0
        simulate(1000.0)


# Omission trial
def omission_trial(trial_setup):
    # Loop over all US, in ascending order
    for us_def in trial_setup:
        # Reset
        simulate(1000.0)
        # Set the CS input
        get_population('VIS').baseline = us_def['VIS']
        # Simulate for the desired duration + 1s for the absent US
        simulate(us_def['duration'] + 1000.0)
        # Reset
        get_population('VIS').baseline = 0.0
        get_population('LH').baseline = 0.0
        simulate(1000.0)
        
        
# Earlier trial
def earlier_trial(trial_setup):
    # Loop over all US, in ascending order
    for us_def in trial_setup:
        # Reset
        simulate(1000.0)
        # Set the CS input
        get_population('VIS').baseline = us_def['VIS']
        # Simulate for the desired duration minus 1s
        simulate(us_def['duration'] - 1000.0)
        # Set the US for 1s
        get_population('LH').baseline = us_def['magnitude']* us_def['GUS']
        simulate(1000.0)
        # Reset
        get_population('VIS').baseline = 0.0
        get_population('LH').baseline = 0.0
        simulate(2000.0)
