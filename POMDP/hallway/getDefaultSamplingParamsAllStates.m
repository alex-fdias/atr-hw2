function [nrRewardStarts,maxSteps,nrRewardRuns] = ...
    getDefaultSamplingParamsAllStates
% $Id: getDefaultSamplingParamsAllStates.m,v 1.0 2022/06/30 alex-fdias Exp $

global problem
%nrRewardStarts=100;
% retrieve the number of terminal/goal states
nrRewardStarts=min(find(abs(problem.startCum - 1) < 1e-10));
maxSteps=251;
nrRewardRuns=1000;
