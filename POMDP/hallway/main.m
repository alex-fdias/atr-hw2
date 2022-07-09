clear all
close all
drawnow
clc
% adjust to your specific folder path
perseusToolboxFolderPath = 'C:\Users\Assus\Documents\MATLAB\pomdpSoftware-0.1';

addpath(strcat(perseusToolboxFolderPath,'\','generic'))
initProblem;

num_belief_set_points_exponents = [1:4]';
runtimes_vi   = zeros(size(num_belief_set_points_exponents), 'double');
filenames     = cell(size(num_belief_set_points_exponents));
global problem;
global vi; % make MATLAB aware of this global struct
for i=1:numel(num_belief_set_points_exponents)
    % number of points sampling the belief space (belief points set size, |B|)
    num_belief_set_points=10^num_belief_set_points_exponents(i);
    
    S=sampleBeliefs(num_belief_set_points);
    
    % declare struct 'params' with fields 'epsilon' and 'maxTime'
    params.epsilon=1e-3;
    params.maxTime=3600;
    filename=runvi(S, params);
    
    idx_frwd_slash = strfind(filename,'/');
    if ~isempty(idx_frwd_slash)
        filenames{i} = strcat(filename(idx_frwd_slash+1:end),'.mat');
    else
        error('something bad happened, check code')
    end
    
    
    runtimes_vi(i) = vi.backupStats.Time{end}-vi.backupStats.startTime;
    
    % print runtime
    disp(runtimes_vi(i))
    num_iters = size(vi.backupStats.V,2);
    iter_min = num_iters;
    iter_max = num_iters;
    [nrStarts,maxNrSteps,nrRuns] = getDefaultSamplingParamsAllStates;
    successes_iters    = zeros(nrStarts*nrRuns, iter_max-iter_min+1, 'int32' );
    num_steps_iters    = zeros(nrStarts*nrRuns, iter_max-iter_min+1, 'int32' );
    disc_rewards_iters = zeros(nrStarts*nrRuns, iter_max-iter_min+1, 'double');
    runtimes_sims      = zeros(iter_max-iter_min+1, 'double');
    for j=1:size(iter_min:iter_max,2)
        disp(j)
        time_aux=tic;
        R=sampleRewardsAllStates(vi.backupStats.V{iter_min+j-1});
        runtimes_sims(j)        = toc(time_aux);
        successes_iters(:,j)    = R(:,2);
        num_steps_iters(:,j)    = R(:,3);
        disc_rewards_iters(:,j) = R(:,4);
    end
    
    % rearrange the array (equivalent to reshape in Python/NumPy)
    successes_iters_StartState    = zeros(nrStarts,nrRuns, 'int32');
    num_steps_iters_startState    = zeros(nrStarts,nrRuns, 'int32');
    disc_rewards_iters_startState = zeros(nrStarts,nrRuns, 'double');
    for k=1:nrStarts
        successes_iters_StartState   (k,:) = successes_iters   ((k-1)*nrRuns+1:k*nrRuns,1);
        num_steps_iters_startState   (k,:) = num_steps_iters   ((k-1)*nrRuns+1:k*nrRuns,1);
        disc_rewards_iters_startState(k,:) = disc_rewards_iters((k-1)*nrRuns+1:k*nrRuns,1);
    end
    save(filenames{i},'successes_iters_StartState'   ,'-mat','-append')
    save(filenames{i},'num_steps_iters_startState'   ,'-mat','-append')
    save(filenames{i},'disc_rewards_iters_startState','-mat','-append')
    save(filenames{i},'runtimes_sims'                ,'-mat','-append')
    figure
    axis square
    boxplot(disc_rewards_iters_startState','Notch','on')
    set(gca,'YLim',[0 1])
end

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