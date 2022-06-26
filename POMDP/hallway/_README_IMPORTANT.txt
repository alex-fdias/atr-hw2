This folder was created to run the POMDP problem 'hallway.POMDP' (57-state navigation problem).
The basic structure and contents of this folder replicates the folder 'pomdpSoftware-0.1/problems/hallway2'
 of the Perseus MATLAB toolbox ('pomdpSoftware_0.1.tar.gz'). While there are instructions for
 such a replication in the toolbox README file ('pomdpSoftware-0.1/README', Section 4. RUNNING
 YOUR OWN POMDP PROBLEM), the instructions here are specific for the 'hallway.POMDP' problem.
 

1) To make MATLAB aware of the base toolbox .m files, the path of the toolbox 'generic' folder
 ('pomdpSoftware-0.1/generic') must be added to the MATLAB path. This is taken care of in the
 main.m file (addpath statement) automatically, every time MATLAB is started and thus is still
 not aware of the path (unless it has been added permanently).

2) Copy the initProblem.m file from the 'problems/tag' or 'problems/hallway2' folder and adjust
 the naming to 'Hallway' or 'hallway' (problem.description, problem.unixName). You can name the
 actions, as done in the 'initProblem.m' file of 'problems/hallway2' (not required).

3) The default 'episodeEnded.m' (in the 'generic' folder) should work (NOT tested), as the reward
 is the same for all goal states, or the file 'episodeEnded.m' in folder 'problems/hallway2' can
 be customized (the goal states are the states 57 to 60, instead of 69 to 72)

4) The 'getDefaultMaxTime.m' script specifies the duration for which the algorithm is allowed
   to run (a few minutes should be enough)

5) The 'main.m' file included customizes the algorithm execution and results simulation:
   i)   Number of points sampling the belief space, convergence criterion (params.epsilon),
        and maximum allowed run time
   iii) Run randomized point-based approximate value iteration (Perseus algorithm, 'runvi' function)
   iv)  After loading the generated .mat file with the algorithm execution results, the custom
        files 'getDefaultSamplingParamsAllStates.m' and 'sampleRewardsAllStates.m' allow running
		simulations of the value function approximation of Perseus to obtain a distribution of
		the value function for each state.
   v)   The resulting .mat file can be read by the 'HW2.py' script and the results of the Perseus
        algorithm (value function approximation) can then be compared to those of Value Iteration
   
 