from matplotlib import pyplot as plt
from utils import movingaverage, weighted_moving_average
import numpy as np
from operator import itemgetter

def plot_comparative(results, labels, moving_avg_window_size=None, plot_iter_average=False,all_lines=False):


    if plot_iter_average:
        ax1 = plt.subplot(2,1,1)
    else :
        ax1 = plt.subplot(1,1,1)
    ax1.set_ylabel("Avg. Reward")
    ax1.set_xlabel("# of episodes completed")
    ax1.set_title("Average reward in fct. of # of episodes completed ")

    if plot_iter_average:
        ax2 = plt.subplot(2,1,2)
        ax2.set_title("Average reward in fct of # iteration completed")
        ax2.set_ylabel("Reward")
        ax2.set_xlabel("# of iterations completed")

    if len(results[0]) == 3 :
        colors = ['b-','r-','g-']
    else :
        colors = ['b:','r:','g:','b-','r-','g-']



    for i, lbl in enumerate(labels):

        c = colors[i]

        num_episodes = len(results[0][i])
        avg_r_episode = np.zeros((num_episodes,))
        iter_reward_list = []

        for res in map(itemgetter(i),results):
            l,r = zip(*list(map(lambda x: (len(x[2]),np.sum(x[2])),res)))
            assert len(l) == num_episodes
            avg_r_episode += r
            xpoints = np.array(range(num_episodes))

            if not (moving_avg_window_size is None):
                r = movingaverage(r, moving_avg_window_size)
                xpoints = xpoints[moving_avg_window_size - 1:]
            #    xpoints2 = xpoints2[moving_avg_window_size - 1:]

            if all_lines :
                ax1.plot(xpoints, r, c,alpha=0.12)

            if plot_iter_average:
                xpoints2 = reduce(lambda x, y: (x[0] + [x[1] + y], x[1] + y), l, ([], 0))[0]
                step_size = float(xpoints2[-1] - xpoints2[0]) / num_episodes
                width = 50
                xpoints2, avg_r_iter_taken = weighted_moving_average(xpoints2, r, step_size, width=width)
                if all_lines :
                    ax2.plot(xpoints2, avg_r_iter_taken, c, alpha=0.12)
                iter_reward_list += zip(xpoints2,avg_r_iter_taken)
            #ax2.plot(xpoints2, r, c,alpha=0.1)
            #iter_reward_list += zip(xpoints2,r)

        avg_r_episode = 1.0/len(results)* avg_r_episode
        xpoints = np.array(range(num_episodes))
        if not (moving_avg_window_size is None):
            avg_r_episode = movingaverage(avg_r_episode, moving_avg_window_size)
            xpoints = xpoints[moving_avg_window_size - 1:]

        ax1.plot(xpoints, avg_r_episode, c,label=lbl)

        if plot_iter_average:
            iter_reward_list = sorted(iter_reward_list, key=itemgetter(0))
            step_size = float(iter_reward_list[-1][0] - iter_reward_list[0][0]) / num_episodes
            x, y = zip(*iter_reward_list)
            width = 205
            xpoints2, avg_r_iter_taken = weighted_moving_average(x, y, step_size, width=width)
            ax2.plot(xpoints2, avg_r_iter_taken, c, label=lbl)


    legend = ax1.legend(loc='lower right', shadow=True)

    if plot_iter_average:
        legend = ax2.legend(loc='lower right', shadow=True)

    plt.show()


if __name__ == "__main__":
    import pickle
    results = pickle.load(open("results1.pkl",'rb'))
    labels = ["On Policy First-visit MC", "On Policy Every-visit MC", "Off Policy Every-visit MC with IS"]
    plot_comparative(map(lambda x: x[0:3], results), labels, moving_avg_window_size = None,plot_iter_average = False, all_lines = True)
    labels += ["Greedy pi evaluation - On Policy First-visit MC", "Greedy pi evaluation - On Policy Every-visit MC","Greedy pi evaluation - Off Policy EV MC with IS"]

    plot_comparative(results, labels,moving_avg_window_size=None,plot_iter_average=False)
