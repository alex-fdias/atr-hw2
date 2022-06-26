% adjust to your specific folder path
perseusToolboxFolderPath = 'C:\Users\Assus\Documents\MATLAB\pomdpSoftware-0.1';

addpath(strcat(perseusToolboxFolderPath,'\','generic'))
initProblem;

% number of points sampling the belief space (belief points set size, |B|)
num_belief_set_points=1000;

S=sampleBeliefs(num_belief_set_points);

% declare struct 'params' with fields 'epsilon' and 'maxTime'
params.epsilon=1e-6;
params.maxTime=3600;
runvi(S, params);


% after execution of runvi(), run this code
%load('hallwayvi20220620T135853_60_42_converged_72.mat'); % file saved *after* function runvi() completes execution
close all
% print runtime
disp(vi.backupStats.Time{end}-vi.backupStats.startTime)
clear global problem;
initProblem;
global problem;
num_iters = size(vi.backupStats.V,2);
iter_min = num_iters;
iter_max = num_iters;
[nrStarts,maxNrSteps,nrRuns] = getDefaultSamplingParamsAllStates;
successes_iters    = zeros(nrStarts*nrRuns, iter_max-iter_min+1);
num_steps_iters    = zeros(nrStarts*nrRuns, iter_max-iter_min+1);
disc_rewards_iters = zeros(nrStarts*nrRuns, iter_max-iter_min+1);
for i=1:size(iter_min:iter_max,2)
    disp(i)
    R=sampleRewardsAllStates(vi.backupStats.V{iter_min+i-1});
    successes_iters(:,i)    = R(:,2);
    num_steps_iters(:,i)    = R(:,3);
    disc_rewards_iters(:,i) = R(:,4);
end

disc_rewards_iters_startState = zeros(nrStarts,nrRuns);
for j=1:nrStarts
    disc_rewards_iters_startState(j,:) = disc_rewards_iters((j-1)*nrRuns+1:j*nrRuns,1);
end
save(strcat('disc_rewards_iters_startState_',int2str(nrRuns),'.mat'),'disc_rewards_iters_startState','-mat')
figure
axis square
boxplot(disc_rewards_iters_startState','Notch','on')
set(gca,'YLim',[0 1])

%error('end')

%value_vecs = vi.backupStats.V;
%value_vec = vi.backupStats.V{1,end}.v;

%num_states = size(value_vecs{1}.v,2);
%value_vec_all = vertcat(vi.backupStats.V{1,end}.v);

%num_iters_trunc = vi.iter;
%figure
%axis square
%plot(1:num_iters_trunc, sum(successes_iters(:,1:num_iters_trunc),1)/(nrStarts*nrRuns)*100)

%figure
%axis square
%boxplot(successes_iters(:,1:num_iters_trunc),'Notch','on')

%figure
%axis square
%boxplot(disc_rewards_iters(:,1:num_iters_trunc),'Notch','on')

%prctile(disc_rewards_iters(:,end),[5,25,50,75,95],"all")