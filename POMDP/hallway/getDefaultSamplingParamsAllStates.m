function [nrRewardStarts,maxSteps,nrRewardRuns] = ...
    getDefaultSamplingParamsAllStates
% $Id: getDefaultSamplingParams.m,v 1.1 2004/07/13 13:58:15 mtjspaan Exp $

global problem
%nrRewardStarts=100;
% retrieve the number of terminal/goal states
nrRewardStarts=min(find(abs(problem.startCum - 1) < 1e-10));
maxSteps=251;
nrRewardRuns=1000;
