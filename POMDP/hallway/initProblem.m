function initProblem
% Problem specific initialization function for Hallway.
% $Id: initProblem.m,v 1.0 2022/06/26 00:00:00 alex-fdias Exp $

clear global pomdp;
global problem;
global pomdp;

% String describing the problem.
problem.description='Hallway';

% String used for creating filenames etc.
problem.unixName='hallway';

% Use sparse matrix computation.
problem.useSparse=1;

% Load the (cached) .POMDP, defaults to unixName.POMDP.
initPOMDP;

% Generic POMDP initialization code. Should be called after initPOMDP.
initProblemGeneric;

problem.actions=char('Stay in place','Move forward','Turn right',['Turn ' ...
                    'around'],'Turn left');
