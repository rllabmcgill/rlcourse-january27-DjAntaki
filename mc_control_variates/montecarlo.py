
import numpy as np
from utils import egreedy_probs,sample, esoft_probs, egreedy_sample, esoft_sample

def make_episode(env ,pi, pi_policy='sample',max_iter = 1000,**kwargs):

    A ,R = [] ,[]
    S = [env.reset()]

    i = 0
    s = S[0]
    while True :
        i += 1
        if pi_policy == 'greedy':
            action = np.argmax(pi[s])
        elif pi_policy == 'sample':
            action = sample(pi[s])
        elif pi_policy == "egreedy":
            action = egreedy_sample(pi[s], kwargs['epsilon'])
        elif pi_policy == "esoft":
            action = esoft_sample(pi[s], kwargs['epsilon'])

        s, r_t1, isterminal, _ = env.step(action)
        A.append(action)
        S.append(s)
        R.append(r_t1)

        if isterminal or i > max_iter:
            break

    return S ,A ,R

def make_batch_episode(env, pi, pi_policy='sample',max_iter = 1000, nb_episode=1000,**kwargs):
    ep_count = 0
    episodes = []

    while True:
        ep_count += 1
        episodes.append(make_episode(env,pi,pi_policy,max_iter, *kwargs))
        if ep_count == nb_episode:
            break
    return episodes

def evaluate_agent(env, pi, pi_policy, max_iter, nb_episode=1000,verbose=False):
    e = make_batch_episode(env, pi, pi_policy, max_iter, nb_episode=nb_episode)
    return evaluate(e,verbose)

def evaluate(episodes,verbose=True):
    nb_episodes = len(episodes)
    iter_taken = np.zeros((nb_episodes,))
    nb_decisions = np.zeros((nb_episodes,))
    total_reward = np.zeros((nb_episodes,))
    for n in range(nb_episodes):
        S,A,R = episodes[n]
        iter_taken[n] = len(S)-1
        total_reward[n] = np.sum(R)
    if verbose :
        print('average time lasted', np.average(iter_taken))
        print('average reward', np.average(total_reward))
        print('standard deviation reward', np.std(total_reward))
    return iter_taken, total_reward

def OnPolicyFirstVisitMonteCarlo(env, pi_policy='egreedy', gamma=0.9, epsilon=1e-1, nb_episode=50, max_iter=1000):
    Q = np.ones((env.num_states, env.num_actions))
    returns = [[[] for a in range(env.num_actions)] for i in range(env.num_states)]
    pi = 1.0/ env.num_actions * np.ones((env.num_states, env.num_actions))

    ep_count = 0
    episodes = []
    valid_episodes = []
    valid_episodes.append(make_episode(env, pi=Q, pi_policy='greedy', max_iter=max_iter))

    while True:
        ep_count += 1

        S, A, R = make_episode(env,pi,max_iter=max_iter)
        episodes.append([S,A,R])

        for e, (s, a, r) in enumerate(zip(S, A, R)):
            G = reduce(lambda x,y: gamma*x+y, reversed(R[e:-1]), R[-1])
            returns[s][a].append(G)
            Q[s, a] = np.average(returns[s][a])

        # Policy update
        for s in set(S):
            if pi_policy == "egreedy":
                pi[s] = egreedy_probs(Q[s],epsilon)
            elif pi_policy == 'esoft':
                pi[s] = esoft_probs(Q[s],epsilon)
        valid_episodes.append(make_episode(env, pi=Q, pi_policy='greedy', max_iter=max_iter))

        if ep_count == nb_episode:
            break

    return episodes, Q,valid_episodes

def OnPolicyEveryVisitMonteCarlo(env, pi_policy='egreedy',gamma=0.9, epsilon=1e-2, nb_episode=100, max_iter=1000):
    Q = np.ones((env.num_states, env.num_actions))
    returns = [[[] for a in range(env.num_actions)] for i in range(env.num_states)]
    pi = 1.0/ env.num_actions * np.ones((env.num_states, env.num_actions))

    ep_count = 0
    episodes = []
    valid_episodes = []
    valid_episodes.append(make_episode(env, pi=Q, pi_policy='greedy', max_iter=max_iter))

    while True:
        ep_count += 1

        S, A, R = make_episode(env,pi,max_iter=max_iter)
        episodes.append([S,A,R])

        l = len(S)

        def find_next(e, s, a):
            if e + 1 == l :
                return 0
            i = 1
            while True :
#                print(i,(e+i +1 == l),(S[e+i]==s and A[e+i]==a))
#                if (e+i +1 == l) or (S[e+i]==s and A[e+i]==a) :
                if (e + i + 1 == l) or S[e + i] == s:
                    i -= 1
                    break
                i += 1
            return i

        for e, (s, a, r) in enumerate(zip(S, A, R)):
            i = find_next(e,s,a)
 #           print(R)
 #           print(e,i)
 #           print(R[e:e+i], R[e+i])
            G = reduce(lambda x,y: gamma*x+y, reversed(R[e:e+i]), R[e+i])
            returns[s][a].append(G)
            Q[s, a] = np.average(returns[s][a])

        # Policy update


        for s in set(S):
            if pi_policy == "egreedy":
                pi[s] = egreedy_probs(Q[s],epsilon)
            elif pi_policy == 'esoft':
                pi[s] = esoft_probs(Q[s],epsilon)


        valid_episodes.append(make_episode(env, pi=Q, pi_policy='greedy', max_iter=max_iter))

        if ep_count == nb_episode:
            break

    return episodes, Q, valid_episodes

def OffPolicyMC(env, mu_policy='esoft', pi_policy='greedy', gamma=0.9, max_iter=100, nb_episode=100,epsilon=2e-1):
    """ Off-policy every visit MC policy evaluation
        with importance sampling

    Richard Sutton's book p.119
    """

    Q = np.ones((env.num_states, env.num_actions))
    C = np.zeros((env.num_states, env.num_actions))
    pi = 1.0 / env.num_actions * np.ones((env.num_states, env.num_actions))


    if mu_policy == 'uniform':
        mu = 1.0 / env.num_actions * np.ones((env.num_states, env.num_actions))

    ep_count = 0
    train_episodes = []
    valid_episodes = []
    valid_episodes2 = []

    if pi_policy == "greedy":
        valid_episodes.append(make_episode(env, pi=Q, pi_policy='greedy', max_iter=max_iter))
    elif pi_policy == "esoft":
        valid_episodes.append(make_episode(env, pi=Q, pi_policy='esoft', max_iter=max_iter, epsilon=0.025))

    valid_episodes2.append(make_episode(env, pi=Q, pi_policy='greedy', max_iter=max_iter))


    while True:
        ep_count += 1

        if mu_policy == 'esoft':
            mu = np.array([esoft_probs(Q[i],epsilon) for i in range(env.num_states)])

        S, A, R = make_episode(env,mu,max_iter=max_iter)
        train_episodes.append([S,A,R])

        if pi_policy == "esoft":
            for s in set(S):
                pi[s] = np.array([esoft_probs(Q[s],0.025)])

        T,G,W = len(S),0,1

        for t in range(T-2,-1,-1):
            G = gamma*G+R[t]
            s_t, a_t = S[t], A[t]
            C[s_t,a_t] = C[s_t,a_t] + W
            Q[s_t,a_t] = Q[s_t,a_t] + float(W)/C[s_t,a_t] * (G - Q[s_t,a_t])
            if pi_policy == "greedy":
                pi_s_t = np.argmax(Q[s_t, a_t])
                if not a_t == pi_s_t:
                    break
                W = float(W) / mu[s_t, a_t]
            elif pi_policy == "esoft":
                W = float(W)*pi[s_t,a_t]/mu[s_t,a_t]

        if pi_policy == "greedy":
            valid_episodes.append(make_episode(env,pi=Q,pi_policy='greedy',max_iter=max_iter))
        elif pi_policy == "esoft":
            valid_episodes.append(make_episode(env,pi=Q,pi_policy='esoft',max_iter=max_iter,epsilon=0.025))

        valid_episodes2.append(make_episode(env,pi=Q,pi_policy='greedy',max_iter=max_iter))

        if ep_count == nb_episode:
            break

    return valid_episodes, Q, valid_episodes2