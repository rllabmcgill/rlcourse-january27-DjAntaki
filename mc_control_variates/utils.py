import numpy as np

def weighted_moving_average(x,y,step_size,width=10):
    #http://stackoverflow.com/questions/18517722/weighted-moving-average-in-python
    bin_centers  = np.arange(np.min(x),np.max(x)-0.5*step_size,step_size)+0.5*step_size
    bin_avg = np.zeros(len(bin_centers))

    #We're going to weight with a Gaussian function
    def gaussian(x,amp=1,mean=0,sigma=1):
        return amp*np.exp(-(x-mean)**2/(2*sigma**2))

    for index in range(0,len(bin_centers)):
        bin_center = bin_centers[index]
        weights = gaussian(x,mean=bin_center,sigma=width)
        bin_avg[index] = np.average(y,weights=weights)

    return (bin_centers,bin_avg)

#
#

def movingaverage(values, window):
    #taken from https://gordoncluster.wordpress.com/2014/02/13/python-numpy-how-to-generate-moving-averages-efficiently-part-2/
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def onehot(num_el,index):
    return np.eye(num_el,num_el)[index]

#
#

def uniform_probs(shape):
    size = np.product(shape)
    return 1.0/size * np.ones(shape)

def softmax(x,temp=1):
    e_x = np.exp(x/temp)
    return e_x / e_x.sum(axis=0)

def sample(weights):
    return np.random.choice(range(len(weights)),p=weights)

def egreedy_sample(policy, epsilon=1e-2):
    """ returns argmax with prob (1-epsilon), else returns a random index"""
    if np.random.binomial(1,1-epsilon) == 1:
        return np.argmax(policy)
    else :
        return np.random.choice(range(len(policy)))

def egreedy_probs(weights, epsilon=1e-2):
    a_star = np.argmax(weights)
    num_action = len(weights)
    return (1-epsilon)*onehot(num_action,a_star) + epsilon * uniform_probs((num_action,))


def esoft_probs(weights, epsilon=1e-2,temp=1):
    """ returns probability of sample for :
    argmax gets prob (1-epsilon) and the rest of the probabilies are distributed with regards to softmax values """
    a_star = np.argmax(weights)
    num_action = len(weights)
    soft_probs = epsilon/num_action * softmax(weights,temp=temp)
    return (1-np.sum(soft_probs))*onehot(num_action,a_star) + soft_probs

def esoft_sample(weights, epsilon=1e-2,temp=1):
    return sample(esoft_probs(weights, epsilon=epsilon,temp=temp))
    if np.random.binomial(1,1-epsilon) == 1:
        return np.argmax(weights)
    else :
        num_action = len(weights)
        soft_probs = epsilon / num_action * softmax(weights, temp=temp)
        return sample(soft_probs)

#
#
# functions
#

def get_fixed_value_func(v):
    def get_value(*args, **kwargs):
        return v
    return get_value

def get_scheduled_value_func(times,values,key=None):
    """ times : the scheduled time at which the returned value change.
        values : the values to be returned
        key : if none then looks at first argument

        TODO : less sketchy code"""

    assert sorted(times) == times
    times_len = len(times)
    def get_value(*args, **kwargs):
        t = kwargs[key]
        sched_value_index = np.max(filter(lambda x:t >= times[x],range(times_len)))
        return values[sched_value_index]
    return get_value

##Stuff

class Bunch:
    def __init__(self,**kwargs):
        for k, arg in kwargs.items():
            setattr(self,k,arg)

