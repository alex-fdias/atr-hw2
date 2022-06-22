import numpy             as np
import matplotlib.pyplot as plt

from pathlib   import Path
from functools import partial


np.random.seed(42)

def POMDP_file_parser(fname):
    '''
    POMDP file parser (incomplete, working for file \'hallway.POMDP\')

    Parameters
    ----------
    fname : TYPE
        DESCRIPTION.

    Raises
    ------
    KeyError
        DESCRIPTION.
    ValueError
        DESCRIPTION.
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    discount_factor : TYPE
        DESCRIPTION.
    num_states : TYPE
        DESCRIPTION.
    num_actions : TYPE
        DESCRIPTION.
    num_observations : TYPE
        DESCRIPTION.
    transition_probab_arr : TYPE
        DESCRIPTION.
    observation_probab_arr : TYPE
        DESCRIPTION.
    rewards_arr : TYPE
        DESCRIPTION.

    '''

    # POMDP file specification:
    # http://cs.brown.edu/research/ai/pomdp/examples/pomdp-file-spec.html
    
    with open(fname, encoding='utf-8') as f:
        # .strip: remove '\n' (new line) and ' ' (whitespace) characters at the end of each line
        file_lines = [line.strip('\n ') for line in f]
    
    iterator_file_lines = iter(file_lines)
    for line in iterator_file_lines:
        if len(line)>0 and not line.startswith('#'): # line is not empty and is not a comment line
            line_split = line.split(':')
            #print(line_split)
            
            if line_split[0] == 'discount':
                discount_factor    = float(line_split[1])
            elif line_split[0] == 'values':
                # what is the use of this?!
                values_type      = line_split[1]
            elif line_split[0] == 'states':
                num_states       = int(line_split[1])
            elif line_split[0] == 'actions':
                num_actions      = int(line_split[1])
                
                if 'num_states' not in locals().keys():
                    raise KeyError('num_states variable should have been defined by now')
                
                transition_probab_arr = np.zeros(
                                                 shape=(
                                                        num_states,
                                                        num_actions,
                                                        num_states,
                                                       ),
                                                 dtype=float,
                                                )
                
            elif line_split[0]=='observations':
                num_observations = int(line_split[1])
                
                if 'num_states'  not in locals().keys():
                    raise KeyError('num_states variable should have been defined by now')
                if 'num_actions' not in locals().keys():
                    raise KeyError('num_actions variable should have been defined by now')
                
                observation_probab_arr = np.empty(
                                                  shape=(
                                                         num_actions,
                                                         num_states,
                                                         num_observations,
                                                        ),
                                                  dtype=float,
                                                 )
                
                # R: <action> : <start-state> : <end-state> : <observation> %f
                rewards_arr = np.zeros(
                                       shape=(
                                              num_states,
                                              num_actions,
                                              num_states,
                                              num_observations,
                                              ),
                                       dtype=float,
                                      )
                
            elif line_split[0]=='start':
                # probabilities of starting in each of the 'num_states' states
                
                # go to next line
                line = next(iterator_file_lines)
                line_split = line.split()
                if len(line_split)!=num_states:
                    raise ValueError('number of splitted values should equal the number of states')
                    
                start_probab_vec = np.array(line_split, dtype=float)
                
            elif line_split[0]=='T':
                # transition probabilities
                #print(line_split, len(line_split))
                if len(line_split)==4:
                    if ' * ' in line_split:
                        raise NotImplementedError
                        
                    line_split_aux = line_split[3].split()
                    
                    transition_probab_arr[int(line_split[2]), int(line_split[1]), int(line_split_aux[0])] = float(line_split_aux[1])
                elif len(line_split)==3:
                    if line_split[1]==' * ' and line_split[2]!=' * ':
                        start_state_aux = int(line_split[2])
                        
                        line = next(iterator_file_lines)
                        
                        # note: the right side of the array should have dimension 'num_states',
                        # while the left side has dimension 'num_actions' x 'num_states'
                        # NumPy does broadcasting (implicit expansion in MATLAB) so that the
                        # right-side array (row vector) is copied/replicated for the first
                        # dimension (of size 'num_actions') in the resulting array
                        transition_probab_arr[start_state_aux, :, :] = np.array(line.split(), dtype=float)
                        
                        del start_state_aux
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            elif line_split[0]=='O':
                # observation probabilities
                #print(line_split, len(line_split))
                if len(line_split)==3:
                    if line_split[1]==' * ':
                        end_state_aux = int(line_split[2])
                        
                        line = next(iterator_file_lines)
                        observation_probab_arr[:, end_state_aux:end_state_aux+1, :] = np.array(line.split(), dtype=float)
                        
                        del end_state_aux
                else:
                    raise NotImplementedError
                pass
            elif line_split[0]=='R':
                # rewards            
                #print(line_split, len(line_split))
                if len(line_split)==5:
                    # R: <action> : <start-state> : <end-state> : <observation> %f
                    if line_split[1]==' * ' and line_split[2]==' * ' and line_split[4].split()[0]=='*':
                        end_state_aux = int(line_split[3])
                        
                        rewards_arr[:, :, end_state_aux:end_state_aux+1, :] = float(line_split[4].split()[1])
                        
                        del end_state_aux
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
    
    ###################
    ## sanity checks ##
    ###################
    
    # this checks that, for each state-action pair, the system
    # transitions to each of the 'num_states' states with probabilities
    # that must sum to one
    if np.abs(transition_probab_arr.sum() - np.prod(transition_probab_arr.shape[:2])) > 1e-60:
        raise ValueError('shouldn\'t happen')
    
    # this checks that, for each action-(end-)state pair, the probabilities
    # of observing each of the 'num_observations' observations must sum to one
    if np.abs(observation_probab_arr.sum() - np.prod(observation_probab_arr.shape[:2])) > 1e-60:
        raise ValueError('shouldn\'t happen')

    # this checks that the probabilities of starting in each of the 'num_states'
    # states sum to one
    if np.abs(start_probab_vec.sum() - 1) > 1e-60:
        raise ValueError('shouldn\'t happen')
    
    return discount_factor, num_states, num_actions, num_observations, transition_probab_arr, observation_probab_arr, rewards_arr, start_probab_vec

def compute_optimal_policy(q_fun, print_flag=False):
    '''
    Receives a state-action value function matrix Q(s,a) and returns the
    corresponding (stochastic) optimal policy; if there is more than one
    optimal action for a given state, a probability of 1/N is assigned
    to each of the N optimal actions (equiprobability apportioning scheme)  

    Parameters
    ----------
    q_fun : TYPE \\ numpy 2d array
        state-action value function matrix (number of states, number of actions)
    print_flag : TYPE, optional
        Whether to print the optimal policy actions for each state. The default is False.

    Returns
    -------
    policy : TYPE
        An optimal stochastic policy corresponding to the input action-value function matrix.

    '''
    
    max_idxs = np.abs(q_fun - np.max(q_fun, axis=1, keepdims=True)) < 1e-10
    policy            = np.empty(shape=q_fun.shape)
    policy[~max_idxs] = 0.0
    policy[ max_idxs] = np.repeat(1.0/max_idxs.sum(axis=1), max_idxs.sum(axis=1))
    
    if print_flag:
        idx = 0
        state_curr = -1
        new_state_flag = True
        print('Optimal policy (state: optimal actions): ')
        for action in np.argwhere(policy):
            if state_curr!=action[0]:
                state_curr=action[0]
                print('\nstate ' + str(action[0]+1) + ':', end='')
            
            print(' ' + str(action[1]+1), end='')
            if policy[action[0],action[1]] < 1.0:
                # multiple optimal actions for the same state, print probability
                print('(' + '{:02f}'.format(policy[action[0],action[1]]) + ')', end='')
                
        print(end='\n')
    
    return policy

def compute_policy_diff(policy_1, policy_2):
    return np.abs(policy_1 - policy_2).sum()/2

def print_algorithm_name_message(alg_name_message):
    print('#'*(6+len(alg_name_message)))
    print('## ' + alg_name_message + ' ##')
    print('#'*(6+len(alg_name_message)))

def policy_iteration(discount_factor, num_states, num_actions, transition_probab_arr, rewards_arr, start_probab_vec, goal_states_bool, plot_result=True, plot_output_dir='plots', plot_output_fname='plot_policy_iteration', plot_output_ext='.pdf', print_optimal_policy=True, stay_action_policy_initialization=False, equiprobability_policy_initialization=True):
    print_algorithm_name_message('Policy Iteration')
    
    policy_policyiteration = np.empty(shape=(num_states,num_actions))
    if stay_action_policy_initialization:
        policy_policyiteration[:,0 ] = 1.0
        policy_policyiteration[:,1:] = 0.0
    elif equiprobability_policy_initialization:
        policy_policyiteration[...] = 1.0/num_actions
    else:
        raise NotImplementedError
    
    
    policy_policyiteration_prev      = np.empty(shape=(num_states,num_actions))
    policy_policyiteration_prev[...] = policy_policyiteration[...]
    
    zeros_initialization = False
    if not zeros_initialization:
        value_fun_vec                    = np.empty(shape=(num_states), dtype=float)
        value_fun_vec[ goal_states_bool] = 0
        value_fun_vec[~goal_states_bool] = np.random.random(size=(num_states-goal_states_bool.sum(),))
    else:
        value_fun_vec                    = np.zeros(shape=(num_states), dtype=float)
    
    
    value_fun_vec_new                = np.empty(shape=(num_states), dtype=float)
    q_fun_aux                        = np.empty(shape=(num_states, num_actions), dtype=float)
    
    vector_norm = partial(np.linalg.norm, ord=2) # partial function, to make code more readable
    value_fun_vec_diff = []
    
    theta = 1e-6
    iter_curr = 0
    while True:
        iter_curr += 1
        
        #################################
        ## iterative Policy Evaluation ##
        #################################
        while True:
            #print(policy_policyiteration*(transition_probab_arr*(rewards_arr[:,:,:, 0] + discount_factor*np.tile(value_fun_vec.reshape(1,1,-1), reps=[num_states,1,1]))).sum(axis=2))
            #print((transition_probab_arr*(rewards_arr[:,:,:, 0] + discount_factor*np.tile(value_fun_vec.reshape(1,1,-1), reps=[num_states,1,1]))))
            #raise ValueError
            value_fun_vec_new[...]              = (policy_policyiteration*(transition_probab_arr*(rewards_arr[:,:,:, 0] + discount_factor*np.tile(value_fun_vec.reshape(1,1,-1), reps=[num_states,1,1]))).sum(axis=2)).sum(axis=1)
            
            #print(value_fun_vec_new)
            #raise ValueError
            
            value_fun_vec_new[goal_states_bool] = 0
            #q_value_fun[...]           = (transition_probab_arr*(rewards_arr[..., 0] + discount_factor*np.tile(value_fun_vec_.reshape(1,1,-1), reps=[num_states,num_actions,1]))).sum(axis=2)
            
            # check relative change of the norm of V(s) from previous iteration
            value_fun_vec_diff.append((vector_norm(value_fun_vec_new - value_fun_vec)/vector_norm(value_fun_vec)))
            value_fun_vec[...] = value_fun_vec_new
            if value_fun_vec_diff[-1] < theta:
                print('Policy Evaluation took {} iterations'.format(iter_curr))
                break
        
        q_fun_aux[...]                = policy_policyiteration*(transition_probab_arr*(rewards_arr[:,:,:, 0] + discount_factor*np.tile(value_fun_vec.reshape(1,1,-1), reps=[num_states,1,1]))).sum(axis=2)
        q_fun_aux[goal_states_bool,:] = 0
        
        policy_policyiteration = compute_optimal_policy(q_fun=q_fun_aux, print_flag=False)
        if compute_policy_diff(policy_policyiteration, policy_policyiteration_prev) == 0:
            if print_optimal_policy:
                compute_optimal_policy(q_fun=q_fun_aux, print_flag=True)
            break
        else:
            policy_policyiteration_prev[...] = policy_policyiteration
    
    if plot_result:
        #iterations = np.arange(1,iter_curr+1)
        states_ = np.arange(1,num_states-goal_states_bool.sum()+1)
        num_subplots = 1
        fig, axs = plt.subplots(num_subplots, 1, figsize=(6,2*num_subplots))
        axs.plot(states_, q_fun_aux[~goal_states_bool,:].max(axis=1), label=r'$\theta$=' + str(theta))
        axs.set_xlabel('state')
        axs.set_ylabel('value function')
        axs.set_xticks(np.arange(0, (num_states-goal_states_bool.sum())+1, step=4))  # Set label locations.
        axs.set_xlim([0, (num_states-goal_states_bool.sum())])
        axs.set_ylim([0.20, 1.00])
        axs.legend(loc='upper right')
        
        fig.tight_layout()
        plt.savefig(
                    fname = plot_output_dir + \
                            '/' + \
                            plot_output_fname + \
                            plot_output_ext,
                    format = 'pdf',
                   )
        plt.show()
        
    return q_fun_aux, policy_policyiteration

def value_iteration(discount_factor, num_states, num_actions, transition_probab_arr, rewards_arr, start_probab_vec, goal_states_bool, policy_ref, plot_result=True, plot_output_dir='plots', plot_output_fname='plot_value_iteration', plot_output_ext='.pdf', print_optimal_policy=True):
    print_algorithm_name_message('Value Iteration')
    
    value_fun_vec_ = np.empty(shape=(num_states,), dtype=float)
    value_fun_vec_[ goal_states_bool] = 0
    
    value_fun_vec_[~goal_states_bool] = np.random.random(size=(num_states-goal_states_bool.sum(),))
    #value_fun_vec_[~goal_states_bool] = 0
    #value_fun_vec_[~goal_states_bool] = -100*np.ones(shape=(num_states-goal_states_bool.sum(),))

    value_fun_vec_prev                = np.empty(shape=(num_states,), dtype=float)
    value_fun_vec_prev[...]           = value_fun_vec_[...]

    value_fun_vec_history = []

    vector_norm = partial(np.linalg.norm, ord=2) # partial function, to make code more readable
    theta = 1e-6
    q_value_fun = np.empty(shape=(num_states,num_actions), dtype=float)
    value_fun_vec_diff = []
    value_policy_diff  = []
    iter_curr = 0
    num_iters = -1
    num_iters_max = 1000
    while True:
        iter_curr += 1
        
        # notes about the next line (major computation of Value Iteration)
        # transition_probab_arr:
        #   number of current states x number of actions x number of next states
        # rewards_arr:
        #   number of current states x number of actions x number of next states x number of observations
        # value_fun_vec_:
        #   number of current states
        #
        # Computation:
        #   1) since the rewards (rewards_arr) only depend on the next state and not
        #   on the observation (last index/dimension), rewards_arr[..., 0]
        #   is a 3-dimensional array of
        #   number of current states x number of actions x number of next states
        #   (i.e., the observation index/dimension is discarded)
        #
        #   2) in 'rewards_arr[..., 0] + discount_factor*value_fun_vec_', 
        #   value_fun_vec_ gets expanded (implicit expansion) to dimensions
        #   number of current states x number of actions x number of next states
        #   to ensure compatible, element-wise sum with rewards_arr[..., 0]
        #
        #   3) in 2), the resulting temporary array has dimensions 
        #   number of current states x number of actions x number of next states,
        #   which are the same dimensions of the transition probabilities array,
        #   transition_probab_arr; therefore, the [element-wise] multiplication
        #   can take place
        #
        #   4) by summing with respect to the dimension of 'number of next states',
        #   the weighted average (weighted by the transition probabilities) over
        #   the next states is obtained, i.e., an array of dimensions 
        #   number of current states x number of actions
        #   is obtained; there is no need to sum over the rewards, as indicated by
        #   the general expression of the Value Iteration algorithm, as the reward
        #   of coming from a state s by taking action and arriving at the next
        #   state s' is always the same (check the file parsing code above); more
        #   accurately, it only depends on the next state itself
        #
        #   5) the resulting array being of dimensions
        #   number of current states x number of actions, one only needs to take
        #   the maximum over the actions (over the 2nd dimension) in order to get
        #   the updated values of the state-value function V(s)
        q_value_fun[...]           = (transition_probab_arr*(rewards_arr[..., 0] + discount_factor*np.tile(value_fun_vec_.reshape(1,1,-1), reps=[num_states,num_actions,1]))).sum(axis=2)
        q_value_fun[goal_states_bool,:] = 0
        value_fun_vec_             = q_value_fun.max(axis=1) # maximization over actions (2nd dimension)
        
        # calculate optimal policy for the current iteration
        policy_valueiteration = compute_optimal_policy(q_value_fun, print_flag=False)
        value_policy_diff.append(compute_policy_diff(policy_valueiteration, policy_ref))
        value_fun_vec_history.append(value_fun_vec_)

        # check relative change of the norm of V(s) from previous iteration
        value_fun_vec_diff.append((vector_norm(value_fun_vec_ - value_fun_vec_prev)/vector_norm(value_fun_vec_prev)))
        if value_fun_vec_diff[-1] < theta:
            num_iters = iter_curr
            print('Value Iteration took {} iterations'.format(num_iters))
            break
        else:
            value_fun_vec_prev[...] = value_fun_vec_[...]
        
        
        if iter_curr==num_iters_max:
            num_iters = iter_curr
            print('Value Iteration did NOT converge in {} iterations'.format(num_iters))
            break

    if print_optimal_policy:
        compute_optimal_policy(q_value_fun, print_flag=True)

    if plot_result:
        iterations = np.arange(1,iter_curr+1)
    
        num_subplots = 3
        fig, axs = plt.subplots(3, 1, figsize=(6,2*num_subplots))
        axs[0].plot(iterations, value_policy_diff)
        #axs[0].plot(episodes, policy_diff, label=r'$\epsilon$=' + str(eps              ))
        #axs[0].plot(episodes, policy_diff, label='min='         + str(policy_diff.min()))
        #axs[0].set_ylim(0, policy_diff.max())
        axs[0].set_xlabel('iteration')
        axs[0].set_ylabel('policy diff')
        axs[0].set_xlim([0, iter_curr])
        axs[0].set_ylim([0, 45])
    
        axs[1].plot(iterations, np.log10(value_fun_vec_diff), label=r'$\theta$='   + str(theta))
        axs[1].set_ylabel('vector norm diff (log$_{10}$)')
        axs[1].legend(loc='upper right')
        axs[1].set_xlabel('iteration')
        axs[1].set_xlim([0, iter_curr])
        axs[1].set_ylim([-6.5, 0])
        
        states_ = np.arange(1,num_states-goal_states_bool.sum()+1)
        for i in range(iter_curr):
            if i==0:
                axs[2].plot(states_, value_fun_vec_history[i][~goal_states_bool], label=r'iteration ' + str(1))
            elif i==9-1:
                axs[2].plot(states_, value_fun_vec_history[i][~goal_states_bool], label=r'iteration ' + str(9))
            elif i==iter_curr-1:
                axs[2].plot(states_, value_fun_vec_history[i][~goal_states_bool], label=r'iteration ' + str(iter_curr))
            else:
                axs[2].plot(states_, value_fun_vec_history[i][~goal_states_bool])
            #axs[2].plot(states_, value_fun_vec_history[i][~goal_states_bool])
            axs[2].set_xlabel('state')
            axs[2].set_ylabel('value function')
            axs[2].set_xticks(np.arange(0, (num_states-goal_states_bool.sum())+1, step=4))  # Set label locations.
            axs[2].set_ylim([0.20, 1.00])
            axs[2].set_xlim([0, num_states-goal_states_bool.sum()])
        axs[2].legend(loc='upper right')
            
            #if i>0:
            #    print(value_fun_vec_history[i] - value_fun_vec_history[i-1])
            #    if (value_fun_vec_history[i] - value_fun_vec_history[i-1] < 0).any():
            #        print('whoops')
            #        raise ValueError
    
    
        fig.tight_layout()
        plt.savefig(
                    fname = plot_output_dir + \
                            '/' + \
                            plot_output_fname + \
                            plot_output_ext,
                    format = 'pdf',
                   )
        plt.show()
    
    
    return q_value_fun, policy_valueiteration

def q_learning(discount_factor, num_states, num_actions, goal_states_bool, transition_probab_arr, rewards_arr, start_probab_vec, q_value_fun_ref, policy_ref, plot_output_dir='plots'):
    print_algorithm_name_message('Q-learning')
    
    q_fun_vec_                 = np.empty(shape=(num_states,num_actions), dtype=float)
    q_fun_vec_[ goal_states_bool,:] = 0
    q_fun_vec_[~goal_states_bool,:] = 0*np.ones(shape=(num_states-goal_states_bool.sum(),1))
    q_fun_vec_prev             = np.empty(shape=(num_states,num_actions), dtype=float)
    q_fun_vec_prev[...]        = q_fun_vec_[...]
    q_fun_vec_new              = np.empty(shape=(num_states,num_actions), dtype=float)
    q_fun_vec_new[...]         = q_fun_vec_[...]
    
    num_episodes          = 10000
    num_steps_episode_max = 10000
    alpha = 200
    eps   = 0.8
    goal_states_nums = np.argwhere(goal_states_bool).reshape(-1)
    rewards          = np.zeros(shape=(num_episodes))
    successes        = np.zeros(shape=(num_episodes))
    steps            = np.zeros(shape=(num_episodes), dtype=int  )
    policy_diff      = np.empty(shape=(num_episodes), dtype=float)
    q_fun_rmse       = np.empty(shape=(num_episodes), dtype=float)
    q_fun_norm_chng  = np.empty(shape=(num_episodes), dtype=float)
    episode_policy_convergence_flag = False
    num_steps         = np.zeros(shape=(), dtype=int)
    cumulative_reward = np.zeros(shape=(num_episodes), dtype=int )
    q_update_steps    = 1000
    episode_policy_convergence = -1
    for episode in range(num_episodes):
        if (episode % 100) == 0:
            if episode > 0:
                print('episode: ', episode, ', policy_diff: ', policy_diff[episode-1], q_fun_rmse[episode-1], q_fun_norm_chng[episode-1])
            else:
                print('alpha: ' + str(alpha) + ', epsilon: ' + str(eps))
                print('episode: ', episode)
        
        #if episode > 0:
        ##    if q_fun_norm_chng[episode] < 1e-2:
        #    if (episode % q_update_steps) == 0:
        #        q_fun_vec_[...] = q_fun_vec_new[...]
        
        state_curr = np.random.randint(num_states-goal_states_bool.sum())
        while True:
            if np.random.random() < eps:
                # choose one of the possible actions with equal probability
                action = np.random.randint(num_actions)
            else:
                # greedy
                actions_max_idxs = np.argwhere(np.abs(q_fun_vec_[state_curr,:] - q_fun_vec_[state_curr,:].max()) < 1e-10)
                action           = actions_max_idxs[np.random.randint(actions_max_idxs.shape[0]),0]
            
            state_next  = np.argwhere(transition_probab_arr[state_curr, action, :].cumsum() >= np.random.random())[0][0]
            reward_next = rewards_arr[state_curr, action, state_next, 0]
            rewards[episode] += reward_next
            
            num_steps += 1
            #q_fun_vec_[state_curr,action] = q_fun_vec_[state_curr,action] + alpha          *(reward_next + discount_factor*q_fun_vec_[state_next,:].max() - q_fun_vec_[state_curr,action])
            q_fun_vec_[state_curr,action] = q_fun_vec_[state_curr,action] + alpha/num_steps*(reward_next + discount_factor*q_fun_vec_[state_next,:].max() - q_fun_vec_[state_curr,action])
            
            steps[episode] += 1
            if steps[episode] == num_steps_episode_max:
                # maximum allowed number of steps reached, end episode
                break
            
            if state_next in goal_states_nums:
                # goal state reached, mark success and end episode
                successes[episode] = 1
                break
            
            state_curr = state_next
        
        # determine optimal policy for the action-value functions achieved
        # at the end of the episode
        # verify if optimal policy is good
        policy_qlearning     = compute_optimal_policy(q_fun_vec_)
        policy_diff[episode] = compute_policy_diff(policy_ref, policy_qlearning)
        
        q_fun_rmse[episode] = (((q_value_fun_ref - q_fun_vec_)**2).sum()/num_states)**1/2
        q_fun_norm_chng[episode] = np.linalg.norm(q_fun_vec_prev - q_fun_vec_, ord=2)/np.linalg.norm(q_fun_vec_, ord=2)
        q_fun_vec_prev[...] = q_fun_vec_[...]
        
        if policy_diff[episode] < 1e-6:
            if not episode_policy_convergence_flag:
                episode_policy_convergence_flag = True
                episode_policy_convergence = episode+1
                print('Q-learning achieved the optimal policy in episode {}'.format(episode+1))
    
    if not episode_policy_convergence_flag:
        episode_policy_convergence_flag = True
        episode_policy_convergence = episode+1
    
    
    
    # plots
    format_str = "{:0" + str(np.floor(np.log10(num_episodes)).astype(int)+1) + "d}"
    
    fname_str_aux = plot_output_dir + '/plot_Qlearning_' + \
                    'a=' + '{:.2f}'.format(alpha) + '_' \
                    'e=' + '{:.2f}'.format(eps) + '_' \
                    'min=' + '{:.2f}'.format(policy_diff.min()) + '_'
    
    
    episodes = np.arange(1, episode_policy_convergence+1, 1)
    
    num_subplots = 4
    fig, axs = plt.subplots(num_subplots, 1, figsize=(6,2*num_subplots))
    axs[0].plot(episodes, policy_diff[:episode_policy_convergence], label=r'$\alpha$='   + str(alpha            )            )
    axs[0].plot(episodes, policy_diff[:episode_policy_convergence], label=r'$\epsilon$=' + str(eps              )            )
    axs[0].plot(episodes, policy_diff[:episode_policy_convergence], label='min='         + '{:.2f}'.format(policy_diff.min()))
    axs[0].set_ylim(0, policy_diff.max())
    axs[0].set_ylabel('policy diff')
    #axs[0].grid(True)
    axs[0].legend(loc='upper right')
    
    axs[1].plot(episodes, np.log10(q_fun_rmse[:episode_policy_convergence]))
    axs[1].set_ylabel('Q RMSE (log$_{10}$)')
    
    axs[2].plot(episodes, np.log10(q_fun_norm_chng[:episode_policy_convergence]))
    axs[2].set_ylabel('Q norm chng (log$_{10}$)')
    
    axs[3].plot(episodes, steps[:episode_policy_convergence], label=r'total=' + str(steps[:episode_policy_convergence].sum()))
    axs[3].set_ylabel('steps')
    axs[3].legend(loc='upper right')
    
    #axs[4].plot(episodes, successes[:episode_policy_convergence], label='successes')
    #axs[4].set_xlabel('episode')
    #axs[4].set_ylabel(r'success')
    #axs[4].plot(episodes, successes[:episode_policy_convergence].cumsum()/episode_policy_convergence, label='successes (cum., rel.)', color='tab:red')
    #axs[4].legend(loc='lower right')
    
    fig.tight_layout()
    plt.savefig(
                fname = fname_str_aux + \
                        'episode_' + \
                        format_str.format(episode_policy_convergence) + \
                        '.pdf',
                format = 'pdf',
               )
    plt.show()
    
    
    episodes = np.arange(1, num_episodes+1, 1)
    
    num_subplots = 4
    fig, axs = plt.subplots(num_subplots, 1, figsize=(6,2*num_subplots))
    axs[0].plot(episodes, policy_diff, label=r'$\alpha$='   + str(alpha            ))
    axs[0].plot(episodes, policy_diff, label=r'$\epsilon$=' + str(eps              ))
    axs[0].plot(episodes, policy_diff, label='min='         + str(policy_diff.min()))
    axs[0].set_ylim(0, policy_diff.max())
    axs[0].set_ylabel('policy diff')
    #axs[0].grid(True)
    axs[0].legend(loc='upper right')

    axs[1].plot(episodes, np.log10(q_fun_rmse))
    axs[1].set_ylabel('Q RMSE (log$_{10}$)')

    axs[2].plot(episodes, np.log10(q_fun_norm_chng))
    axs[2].set_ylabel('Q norm chng (log$_{10}$)')

    axs[3].plot(episodes                             , steps                             , label=r'total: '       + str(steps.sum())                             )
    axs[3].plot(episodes[:episode_policy_convergence], steps[:episode_policy_convergence], label=r'convergence: ' + str(steps[:episode_policy_convergence].sum()))
    axs[3].set_xlabel('episode')
    axs[3].set_ylabel('steps')
    axs[3].legend(loc='upper right')
    
    #axs[3].plot(episodes, successes, label='successes')
    #axs[3].set_xlabel('episode')
    #axs[3].set_ylabel(r'success')
    #axs[3].plot(episodes, successes.cumsum()/num_episodes, label='successes (cum., rel.)', color='tab:red')
    #axs[3].legend(loc='lower right')
    
    fig.tight_layout()
    plt.savefig(
                fname = fname_str_aux + \
                        'episodes_' + \
                        format_str.format(num_episodes) + \
                        '.pdf',
                format = 'pdf',
               )
    plt.show()

def q_learning_step_history(discount_factor, num_states, num_actions, goal_states_bool, transition_probab_arr, rewards_arr, start_probab_vec, q_value_fun_ref, policy_ref, plot_output_dir='plots'):
    print_algorithm_name_message('Q-learning (step history)')
    
    q_fun_vec_                 = np.empty(shape=(num_states,num_actions), dtype=float)
    q_fun_vec_[ goal_states_bool,:] = 0
    q_fun_vec_[~goal_states_bool,:] = 0*np.ones(shape=(num_states-goal_states_bool.sum(),1))
    q_fun_vec_prev             = np.empty(shape=(num_states,num_actions), dtype=float)
    q_fun_vec_prev[...]        = q_fun_vec_[...]
    q_fun_vec_new              = np.empty(shape=(num_states,num_actions), dtype=float)
    q_fun_vec_new[...]         = q_fun_vec_[...]
    
    num_episodes          = 50000
    num_steps_episode_max = 10000
    alpha = 0.5
    eps   = 0.9
    goal_states_nums = np.argwhere(goal_states_bool).reshape(-1)
    rewards          = np.zeros(shape=(num_episodes))
    successes        = np.zeros(shape=(num_episodes))
    steps            = np.zeros(shape=(num_episodes), dtype=int  )
    policy_diff      = np.empty(shape=(num_episodes), dtype=float)
    q_fun_rmse       = np.empty(shape=(num_episodes), dtype=float)
    q_fun_norm_chng  = np.empty(shape=(num_episodes), dtype=float)
    episode_policy_convergence_flag = False
    num_steps         = np.zeros(shape=(num_states-goal_states_bool.sum(),num_actions), dtype=int)
    cumulative_reward = np.zeros(shape=(num_episodes), dtype=int )
    q_update_steps    = 1000
    episode_policy_convergence = -1
    for episode in range(num_episodes):
        if (episode % 100) == 0:
            if episode > 0:
                print('episode: ', episode, ', policy_diff: ', policy_diff[episode-1], q_fun_rmse[episode-1], q_fun_norm_chng[episode-1])
            else:
                print('alpha: ' + str(alpha) + ', epsilon: ' + str(eps))
                print('episode: ', episode)
        
        #if episode > 0:
        ##    if q_fun_norm_chng[episode] < 1e-2:
        #    if (episode % q_update_steps) == 0:
        #        q_fun_vec_[...] = q_fun_vec_new[...]
        
        state_curr = np.random.randint(num_states-goal_states_bool.sum())
        while True:
            if np.random.random() < eps:
                # choose one of the possible actions with equal probability
                action = np.random.randint(num_actions)
            else:
                # greedy
                actions_max_idxs = np.argwhere(np.abs(q_fun_vec_[state_curr,:] - q_fun_vec_[state_curr,:].max()) < 1e-10)
                action           = actions_max_idxs[np.random.randint(actions_max_idxs.shape[0]),0]
            
            state_next  = np.argwhere(transition_probab_arr[state_curr, action, :].cumsum() >= np.random.random())[0][0]
            reward_next = rewards_arr[state_curr, action, state_next, 0]
            rewards[episode] += reward_next
            
            num_steps[state_curr,action] += 1
            #print(num_steps[state_curr,action])
            #q_fun_vec_[state_curr,action] = q_fun_vec_[state_curr,action] + alpha          *(reward_next + discount_factor*q_fun_vec_[state_next,:].max() - q_fun_vec_[state_curr,action])
            q_fun_vec_[state_curr,action] = q_fun_vec_[state_curr,action] + alpha/num_steps[state_curr,action]*(reward_next + discount_factor*q_fun_vec_[state_next,:].max() - q_fun_vec_[state_curr,action])
            #print(q_fun_vec_[state_curr,action])
            #raise ValueError
            
            steps[episode] += 1
            if steps[episode] == num_steps_episode_max:
                # maximum allowed number of steps reached, end episode
                break
            
            if state_next in goal_states_nums:
                # goal state reached, mark success and end episode
                successes[episode] = 1
                break
            
            state_curr = state_next
        
        # determine optimal policy for the action-value functions achieved
        # at the end of the episode
        # verify if optimal policy is good
        policy_qlearning     = compute_optimal_policy(q_fun_vec_)
        policy_diff[episode] = compute_policy_diff(policy_ref, policy_qlearning)
        
        q_fun_rmse[episode] = (((q_value_fun_ref - q_fun_vec_)**2).sum()/num_states)**1/2
        q_fun_norm_chng[episode] = np.linalg.norm(q_fun_vec_prev - q_fun_vec_, ord=2)/np.linalg.norm(q_fun_vec_, ord=2)
        q_fun_vec_prev[...] = q_fun_vec_[...]
        
        if policy_diff[episode] < 1e-6:
            if not episode_policy_convergence_flag:
                episode_policy_convergence_flag = True
                episode_policy_convergence = episode+1
                print('Q-learning achieved the optimal policy in episode {}'.format(episode+1))
    
    if not episode_policy_convergence_flag:
        episode_policy_convergence_flag = True
        episode_policy_convergence = episode+1
    
    
    
    # plots
    format_str = "{:0" + str(np.floor(np.log10(num_episodes)).astype(int)+1) + "d}"
    
    fname_str_aux = plot_output_dir + '/plot_Qlearning_' + \
                    'a=' + '{:.2f}'.format(alpha) + '_' \
                    'e=' + '{:.2f}'.format(eps) + '_' \
                    'min=' + '{:.2f}'.format(policy_diff.min()) + '_'
    
    
    episodes = np.arange(1, episode_policy_convergence+1, 1)
    
    num_subplots = 4
    fig, axs = plt.subplots(num_subplots, 1, figsize=(6,2*num_subplots))
    axs[0].plot(episodes, policy_diff[:episode_policy_convergence], label=r'$\alpha$='   + str(alpha            )            )
    axs[0].plot(episodes, policy_diff[:episode_policy_convergence], label=r'$\epsilon$=' + str(eps              )            )
    axs[0].plot(episodes, policy_diff[:episode_policy_convergence], label='min='         + '{:.2f}'.format(policy_diff.min()))
    axs[0].set_ylim(0, policy_diff.max())
    axs[0].set_ylabel('policy diff')
    #axs[0].grid(True)
    axs[0].legend(loc='upper right')
    
    axs[1].plot(episodes, np.log10(q_fun_rmse[:episode_policy_convergence]))
    axs[1].set_ylabel('Q RMSE (log$_{10}$)')
    
    axs[2].plot(episodes, np.log10(q_fun_norm_chng[:episode_policy_convergence]))
    axs[2].set_ylabel('Q norm chng (log$_{10}$)')
    
    axs[3].plot(episodes, steps[:episode_policy_convergence], label=r'total=' + str(steps[:episode_policy_convergence].sum()))
    axs[3].set_ylabel('steps')
    axs[3].legend(loc='upper right')
    
    #axs[4].plot(episodes, successes[:episode_policy_convergence], label='successes')
    #axs[4].set_xlabel('episode')
    #axs[4].set_ylabel(r'success')
    #axs[4].plot(episodes, successes[:episode_policy_convergence].cumsum()/episode_policy_convergence, label='successes (cum., rel.)', color='tab:red')
    #axs[4].legend(loc='lower right')
    
    fig.tight_layout()
    plt.savefig(
                fname = fname_str_aux + \
                        'episode_' + \
                        format_str.format(episode_policy_convergence) + \
                        '.pdf',
                format = 'pdf',
               )
    plt.show()
    
    
    episodes = np.arange(1, num_episodes+1, 1)
    
    num_subplots = 4
    fig, axs = plt.subplots(num_subplots, 1, figsize=(6,2*num_subplots))
    axs[0].plot(episodes, policy_diff, label=r'$\alpha$='   + str(alpha            ))
    axs[0].plot(episodes, policy_diff, label=r'$\epsilon$=' + str(eps              ))
    axs[0].plot(episodes, policy_diff, label='min='         + str(policy_diff.min()))
    axs[0].set_ylim(0, policy_diff.max())
    axs[0].set_ylabel('policy diff')
    #axs[0].grid(True)
    axs[0].legend(loc='upper right')

    axs[1].plot(episodes, np.log10(q_fun_rmse))
    axs[1].set_ylabel('Q RMSE (log$_{10}$)')

    axs[2].plot(episodes, np.log10(q_fun_norm_chng))
    axs[2].set_ylabel('Q norm chng (log$_{10}$)')

    axs[3].plot(episodes                             , steps                             , label=r'total: '       + str(steps.sum())                             )
    axs[3].plot(episodes[:episode_policy_convergence], steps[:episode_policy_convergence], label=r'convergence: ' + str(steps[:episode_policy_convergence].sum()))
    axs[3].set_xlabel('episode')
    axs[3].set_ylabel('steps')
    axs[3].legend(loc='upper right')
    
    #axs[3].plot(episodes, successes, label='successes')
    #axs[3].set_xlabel('episode')
    #axs[3].set_ylabel(r'success')
    #axs[3].plot(episodes, successes.cumsum()/num_episodes, label='successes (cum., rel.)', color='tab:red')
    #axs[3].legend(loc='lower right')
    
    fig.tight_layout()
    plt.savefig(
                fname = fname_str_aux + \
                        'episodes_' + \
                        format_str.format(num_episodes) + \
                        '.pdf',
                format = 'pdf',
               )
    plt.show()
    
def main():
    plots_output_dir = 'plots'
    plot_output_ext = '.pdf'
    
    discount_factor, num_states, num_actions, num_observations, transition_probab_arr, observation_probab_arr, rewards_arr, start_probab_vec = POMDP_file_parser('hallway.POMDP')
    goal_states_bool = np.abs(rewards_arr[0, 0, :, 0]-1.0) < 1e-6
    
    Path(Path.cwd() / plots_output_dir).mkdir(parents=True, exist_ok=True)
    
    q_value_fun_policyiteration, policy_policyiteration = policy_iteration(
                                              discount_factor       = discount_factor,
                                              num_states            = num_states,
                                              num_actions           = num_actions,
                                              transition_probab_arr = transition_probab_arr,
                                              rewards_arr           = rewards_arr,
                                              start_probab_vec      = start_probab_vec,
                                              goal_states_bool      = goal_states_bool,
                                              plot_output_dir       = plots_output_dir,
                                              print_optimal_policy  = True,                                              
                                             )

    # if you would like to try the special case where the optimal policy
    # returned by Policy Iteration is the action 'stay' for all non-terminal
    # states, set the boolean variable 'optimal_policy_policyiteration_stay'
    # to True
    # Important: running Policy Iteration again WILL change the results of
    # Value Iteration (because the random numbers generated will change due
    # to the random number generation involved in a new run of Policy
    # Iteration)
    optimal_policy_policyiteration_stay = False
    if optimal_policy_policyiteration_stay:
        policy_iteration(
                         discount_factor       = 1.0,
                         num_states            = num_states,
                         num_actions           = num_actions,
                         transition_probab_arr = transition_probab_arr,
                         rewards_arr           = rewards_arr,
                         start_probab_vec      = start_probab_vec,
                         goal_states_bool      = goal_states_bool,
                         plot_output_dir       = plots_output_dir,
                         print_optimal_policy  = True,
                         stay_action_policy_initialization = True,
                         plot_result           = False,
                        )

    q_value_fun_valueiteration, policy_valueiteration = value_iteration(
                                                                        discount_factor       = discount_factor,
                                                                        num_states            = num_states,
                                                                        num_actions           = num_actions,
                                                                        transition_probab_arr = transition_probab_arr,
                                                                        rewards_arr           = rewards_arr,
                                                                        start_probab_vec      = start_probab_vec,
                                                                        goal_states_bool      = goal_states_bool,
                                                                        policy_ref            = policy_policyiteration,
                                                                        print_optimal_policy  = False, 
                                                                       )
    
    # why are the action-value functions Q(s,a) not the same for
    # Policy Iteration and Value Iteration, except for the optimal
    # action in each state?
    # Q(s,a) for Policy Iteration is all zeros except for the optimal action
    # Q(s,a) for Value Iteration is all non-zeros
    #print(q_value_fun_policyiteration)
    #print(q_value_fun_valueiteration)
    #print(q_value_fun_policyiteration - q_value_fun_valueiteration)
    #print(np.linalg.norm(q_value_fun_policyiteration - q_value_fun_valueiteration))
    #raise ValueError
    # check if the policies determined by policy iteration and value iteration
    # are the same
    if compute_policy_diff(policy_valueiteration, policy_policyiteration) < 1e-6:
        print('[OK] Policy Iteration and Value Iteration converged to the same policy')
    else:
        raise ValueError('shouldn\'t happen')
    
    q_learning( #q_learning_step_history 
               discount_factor       = discount_factor,
               num_states            = num_states,
               num_actions           = num_actions,
               goal_states_bool      = goal_states_bool,
               transition_probab_arr = transition_probab_arr,
               rewards_arr           = rewards_arr,
               start_probab_vec      = start_probab_vec,
               q_value_fun_ref       = q_value_fun_valueiteration,
               policy_ref            = policy_valueiteration,
               plot_output_dir       = plots_output_dir,
              )
    
    raise ValueError    
    
    
    #######################
    ## Double Q-learning ##
    #######################
    q_value_fun_ref = q_value_fun_valueiteration # should be an argument of the function
    
    q_fun_vec_1                      = np.empty(shape=(num_states,num_actions), dtype=float)
    q_fun_vec_1[ goal_states_bool,:] = 0
    q_fun_vec_1[~goal_states_bool,:] = np.random.random(size=(num_states-goal_states_bool.sum(),1))
    
    q_fun_vec_2                      = np.empty(shape=(num_states,num_actions), dtype=float)
    q_fun_vec_2[ goal_states_bool,:] = 0
    q_fun_vec_2[~goal_states_bool,:] = np.random.random(size=(num_states-goal_states_bool.sum(),1))
    
    q_fun_vec_prev1             = np.empty(shape=(num_states,num_actions), dtype=float)
    q_fun_vec_prev1[...]        = q_fun_vec_1[...]
    
    q_fun_vec_prev2             = np.empty(shape=(num_states,num_actions), dtype=float)
    q_fun_vec_prev2[...]        = q_fun_vec_2[...]
    
    #q_fun_vec_new              = np.empty(shape=(num_states,num_actions), dtype=float)
    #q_fun_vec_new[...]         = q_fun_vec_[...]
    
    num_episodes          = 10000
    num_steps_episode_max = 10000
    alpha = 1
    eps   = 0.90
    goal_states_nums = np.argwhere(goal_states_bool).reshape(-1)
    rewards          = np.zeros(shape=(num_episodes))
    successes        = np.zeros(shape=(num_episodes))
    steps            = np.zeros(shape=(num_episodes), dtype=int  )
    policy_diff1     = np.empty(shape=(num_episodes), dtype=float)
    policy_diff2     = np.empty(shape=(num_episodes), dtype=float)
    q_fun_rmse1      = np.empty(shape=(num_episodes), dtype=float)
    q_fun_rmse2      = np.empty(shape=(num_episodes), dtype=float)
    q_fun_norm_chng1 = np.empty(shape=(num_episodes), dtype=float)
    q_fun_norm_chng2 = np.empty(shape=(num_episodes), dtype=float)
    episode_policy_convergence_flag = False
    num_steps         = np.zeros(shape=(), dtype=int)
    cumulative_reward = np.zeros(shape=(num_episodes), dtype=int  )
    q_update_steps    = 1000
    episode_policy_convergence = -1
    print_algorithm_name_message('Double Q-learning')
    for episode in range(num_episodes):
        if (episode % 100) == 0:
            if episode > 0:
                print('episode: ', episode, ', policy_diff: ', policy_diff1[episode-1], q_fun_rmse1[episode-1], q_fun_norm_chng1[episode-1])
            else:
                print('alpha: ' + str(alpha) + ', epsilon: ' + str(eps))
                print('episode: ', episode)
        
        #if episode > 0:
        ##    if q_fun_norm_chng[episode] < 1e-2:
        #    if (episode % q_update_steps) == 0:
        #        q_fun_vec_[...] = q_fun_vec_new[...]
        
        state_curr = np.random.randint(num_states-goal_states_bool.sum())
        while True:
            if np.random.random() < eps:
                # choose one of the possible actions with equal probability
                action = np.random.randint(num_actions)
            else:
                # greedy
                actions_max_idxs = np.argwhere(np.abs((q_fun_vec_1[state_curr,:]+q_fun_vec_2[state_curr,:]) - (q_fun_vec_1[state_curr,:]+q_fun_vec_2[state_curr,:]).max()) < 1e-10)
                action           = actions_max_idxs[np.random.randint(actions_max_idxs.shape[0]),0]
            
            state_next  = np.argwhere(transition_probab_arr[state_curr, action, :].cumsum() >= np.random.random())[0][0]
            reward_next = rewards_arr[state_curr, action, state_next, 0]
            rewards[episode] += reward_next
            
            #raise ValueError
            if np.random.random() < 0.5:
                q_fun_vec_1[state_curr,action] = q_fun_vec_1[state_curr,action] + alpha/((num_steps+1)**0.4)*(reward_next + discount_factor*q_fun_vec_2[state_next,q_fun_vec_1[state_next,:].argmax()] - q_fun_vec_1[state_curr,action])
            else:
                q_fun_vec_2[state_curr,action] = q_fun_vec_2[state_curr,action] + alpha/((num_steps+1)**0.4)*(reward_next + discount_factor*q_fun_vec_1[state_next,q_fun_vec_2[state_next,:].argmax()] - q_fun_vec_2[state_curr,action])
            
            num_steps      += 1
            steps[episode] += 1
            if steps[episode] == num_steps_episode_max:
                # maximum allowed number of steps reached, end episode
                break
            
            if state_next in goal_states_nums:
                # goal state reached, mark success and end episode
                successes[episode] = 1
                break
            
            state_curr = state_next
        
        # determine optimal policy for the action-value functions achieved
        # at the end of the episode
        # verify if optimal policy is good
        policy_qlearning1     = compute_optimal_policy(q_fun_vec_1)
        policy_diff1[episode] = compute_policy_diff(policy_valueiteration, policy_qlearning1)
    
        policy_qlearning2     = compute_optimal_policy(q_fun_vec_2)
        policy_diff2[episode] = compute_policy_diff(policy_valueiteration, policy_qlearning2)
        
        q_fun_rmse1[episode]      = (((q_value_fun_ref - q_fun_vec_1)**2).sum()/num_states)**1/2
        q_fun_norm_chng1[episode] = np.linalg.norm(q_fun_vec_prev1 - q_fun_vec_1, ord=2)/np.linalg.norm(q_fun_vec_1, ord=2)
        q_fun_vec_prev1[...]      = q_fun_vec_1[...]
    
        q_fun_rmse2[episode]      = (((q_value_fun_ref - q_fun_vec_2)**2).sum()/num_states)**1/2
        q_fun_norm_chng2[episode] = np.linalg.norm(q_fun_vec_prev2 - q_fun_vec_2, ord=2)/np.linalg.norm(q_fun_vec_2, ord=2)
        q_fun_vec_prev2[...]      = q_fun_vec_2[...]
        
        if policy_diff1[episode] < 1e-6 and policy_diff2[episode] < 1e-6:
            if not episode_policy_convergence_flag:
                episode_policy_convergence_flag = True
                episode_policy_convergence = episode+1
    
    if not episode_policy_convergence_flag:
        episode_policy_convergence_flag = True
        episode_policy_convergence = episode+1
    
    
    # plots
    format_str = "{:0" + str(np.floor(np.log10(num_episodes)).astype(int)+1) + "d}"
    
    fname_str_aux = plots_output_dir + '/plot_double_Qlearning_' + \
                    'a=' + '{:.2f}'.format(alpha) + '_' \
                    'e=' + '{:.2f}'.format(eps) + '_' \
                    'min=' + '{:.2f}'.format(policy_diff1.min()) + '_'
    
    
    episodes = np.arange(1, episode_policy_convergence+1, 1)
    
    num_subplots = 5
    fig, axs = plt.subplots(5, 1, figsize=(6,2*num_subplots))
    axs[0].plot(episodes, policy_diff1[:episode_policy_convergence], label=r'$\alpha$='   + str(alpha            )            )
    axs[0].plot(episodes, policy_diff1[:episode_policy_convergence], label=r'$\epsilon$=' + str(eps              )            )
    axs[0].plot(episodes, policy_diff1[:episode_policy_convergence], label='min='         + '{:.2f}'.format(policy_diff1.min()))
    axs[0].plot(episodes, policy_diff2[:episode_policy_convergence])
    axs[0].plot(episodes, policy_diff2[:episode_policy_convergence])
    axs[0].plot(episodes, policy_diff2[:episode_policy_convergence])
    axs[0].set_ylim(0, np.max([policy_diff1.max(), policy_diff2.max()]))
    axs[0].set_ylabel('policy diff')
    axs[0].legend(loc='upper right')
    
    axs[1].plot(episodes, np.log10(q_fun_rmse1[:episode_policy_convergence]), label='1')
    axs[1].plot(episodes, np.log10(q_fun_rmse2[:episode_policy_convergence]), label='2')
    axs[1].set_ylabel('Q RMSE (log$_{10}$)')
    axs[1].legend(loc='upper right')
    
    axs[2].plot(episodes, np.log10(q_fun_norm_chng1[:episode_policy_convergence]), label='1')
    axs[2].plot(episodes, np.log10(q_fun_norm_chng2[:episode_policy_convergence]), label='2')
    axs[2].set_ylabel('Q norm chng (log$_{10}$)')
    
    axs[3].plot(episodes, steps[:episode_policy_convergence], label=r'total=' + str(steps[:episode_policy_convergence].sum()))
    axs[3].set_ylabel('steps')
    axs[3].legend(loc='upper right')
    
    axs[4].plot(episodes, successes[:episode_policy_convergence], label='successes')
    axs[4].set_xlabel('episode')
    axs[4].set_ylabel(r'success')
    axs[4].plot(episodes, successes[:episode_policy_convergence].cumsum()/episode_policy_convergence, label='successes (cum., rel.)', color='tab:red')
    axs[4].legend(loc='lower right')
    
    fig.tight_layout()
    plt.savefig(
                fname = fname_str_aux + \
                        'doubleQ_episode_' + \
                        format_str.format(episode_policy_convergence) + \
                        '.png',
                format = 'png',
               )
    plt.show()
    
    
    episodes = np.arange(1, num_episodes+1, 1)
    
    num_subplots = 4
    fig, axs = plt.subplots(4, 1, figsize=(6,2*num_subplots))
    axs[0].plot(episodes, policy_diff1, label=r'$\alpha$='   + str(alpha            ))
    axs[0].plot(episodes, policy_diff1, label=r'$\epsilon$=' + str(eps              ))
    axs[0].plot(episodes, policy_diff1, label='min='         + '{:.2f}'.format(policy_diff1.min()))
    axs[0].plot(episodes, policy_diff2)
    axs[0].plot(episodes, policy_diff2)
    axs[0].plot(episodes, policy_diff2)
    axs[0].set_ylim(0, np.max([policy_diff1.max(), policy_diff2.max()]))
    axs[0].set_ylabel('policy diff')
    axs[0].legend(loc='upper right')
    
    axs[1].plot(episodes, np.log10(q_fun_rmse1), label='1')
    axs[1].plot(episodes, np.log10(q_fun_rmse2), label='2')
    axs[1].set_ylabel('Q RMSE (log$_{10}$)')
    axs[1].legend(loc='upper right')
    
    axs[2].plot(episodes, steps, label=r'total: '       + str(steps.sum())                             )
    axs[2].plot(episodes, steps, label=r'convergence: ' + str(steps[:episode_policy_convergence].sum()))
    axs[2].set_ylabel('steps')
    axs[2].legend(loc='upper right')
    
    axs[3].plot(episodes, successes, label='successes')
    axs[3].set_xlabel('episode')
    axs[3].set_ylabel(r'success')
    axs[3].plot(episodes, successes.cumsum()/num_episodes, label='successes (cum., rel.)', color='tab:red')
    axs[3].legend(loc='lower right')
    
    fig.tight_layout()
    plt.savefig(
                fname = fname_str_aux + \
                        'episodes_' + \
                        format_str.format(num_episodes) + \
                        '.png',
                format = 'png',
               )
    plt.show()

if __name__ == "__main__":
    main()