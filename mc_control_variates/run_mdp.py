from mdp_env import RandomMDP
from montecarlo import OnPolicyFirstVisitMonteCarlo, OnPolicyEveryVisitMonteCarlo, OffPolicyMC, evaluate_agent
from plot import plot_comparative
from operator import itemgetter
import numpy as np

test, save = False, False
#test, save = False, True

# Random MDP settings
nstates = 10
nactions = 2

# Algorithms settings
offevmc_pi_policy='esoft'
#offevmc_pi_policy='greedy'
epsilon = 2e-1
if test:
    nb_episode = 50
    num_mdp = 2
    max_iter = 100
else :
    nb_episode = 100
    num_mdp = 50
    max_iter = 100

results = []
greedy_results = []
labels = ["On Policy First-visit MC", "On Policy Every-visit MC", "Off Policy Every-visit MC"]


for i in range(num_mdp):

    env = RandomMDP(nstates,nactions)
    results_fvmc, Q1, valid_results_fvmc = OnPolicyFirstVisitMonteCarlo(env, epsilon=epsilon, nb_episode=nb_episode, max_iter=max_iter)
    results_evmc, Q2, valid_results_evmc = OnPolicyEveryVisitMonteCarlo(env, epsilon=epsilon, nb_episode=nb_episode, max_iter=max_iter)
    results_off_evmc, Q3,valid_results_off_evmc = OffPolicyMC(env, pi_policy=offevmc_pi_policy, epsilon=epsilon, nb_episode=nb_episode, max_iter=max_iter)
    print(i)
    print(map(lambda x: np.sum(x[2]),results_fvmc))
    print(map(lambda x: np.sum(x[2]),results_evmc))
    print(map(lambda x: np.sum(x[2]),results_off_evmc))
    results.append((results_fvmc, results_evmc, results_off_evmc, valid_results_fvmc,valid_results_evmc,valid_results_off_evmc))

    g_res = []
    for Qi in (Q1,Q2,Q3):
        g_res.append(evaluate_agent(env,pi=Qi, max_iter=max_iter, pi_policy='greedy',nb_episode=100))
    greedy_results.append(g_res)


print("After training greedy evaluation of Q matrix")
for i in (0,1,2):
    print(labels[i]+" : ")
    algo_results = map(itemgetter(i),greedy_results)
    algo_reward = map(itemgetter(1),algo_results)
    algo_iter_taken = map(itemgetter(0),algo_results)
    print('average reward',np.average(algo_reward))
    print('stdreward',np.std(algo_reward))
    print('average nb iter ',np.average(algo_iter_taken))
    print('std',np.std(algo_iter_taken))

if save :
    import pickle
    pickle.dump(results,open('results1.pkl','wb'))
    print('save done')

plot_comparative(map(lambda x: x[0:3],results), labels, moving_avg_window_size=None, plot_iter_average=False,all_lines=True)
labels += ["Greedy First-visit MC","Greedy Every-Visit MC", "Greedy OP EV MC"]
plot_comparative(results, labels, moving_avg_window_size=None, plot_iter_average=False)
