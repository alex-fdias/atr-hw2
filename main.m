initProblem;

% number of points sampling the belief space (belief points set size, |B|)
num_belief_set_points=10;

S=sampleBeliefs(num_belief_set_points);

params.epsilon=1e-3;
params.maxTime=3600;
runvi(S, params);


% after execution of runvi(), run this code
%load('hallwayvi20220620T135853_60_42_converged_72.mat'); % file saved *after* function runvi() completes execution
close all
% print runtime
disp(vi.backupStats.Time{end}-vi.backupStats.startTime)
clear global problem;
initProblem;
num_iters = size(vi.backupStats.V,2);
[nrStarts,maxNrSteps,nrRuns] = getDefaultSamplingParams;
successes_iters    = zeros(nrStarts*nrRuns, num_iters);
num_steps_iters    = zeros(nrStarts*nrRuns, num_iters);
disc_rewards_iters = zeros(nrStarts*nrRuns, num_iters);
for i=1:num_iters
    disp(i)
    R=sampleRewards(vi.backupStats.V{i});
    successes_iters(:,i)    = R(:,2);
    num_steps_iters(:,i)    = R(:,3);
    disc_rewards_iters(:,i) = R(:,4);
end

value_vecs = vi.backupStats.V;
value_vec = vi.backupStats.V{1,end}.v;

num_states = size(value_vecs{1}.v,2);
value_vec_all = vertcat(vi.backupStats.V{1,end}.v);

num_iters_trunc = vi.iter;
figure
axis square
plot(1:num_iters_trunc, sum(successes_iters(:,1:num_iters_trunc),1)/(nrStarts*nrRuns)*100)

figure
axis square
boxplot(successes_iters(:,1:num_iters_trunc),'Notch','on')

figure
axis square
boxplot(disc_rewards_iters(:,1:num_iters_trunc),'Notch','on')

prctile(disc_rewards_iters(:,end),[5,25,50,75,95],"all")