import numpy as np
import random


class rand_gen(object):
    def __init__(self):
        n = 0
        rands = []

    def rand(self):
        # sample a random value
        r = np.random.rand()
        # add these rands into our list of rands
        self.rands.append(r)
        # increment the number of rands used
        self.n = self.n + 1
        return r

    def check_randomness(self):
        # check the average and std of the rands used
        r_bar = np.average(self.rands)
        r_std = np.std(self.rands)
        return (r_bar, r_std)


class rand(object):
    def __init__(self, seed=6):
        random.seed(seed)

    def insphere(self, r):
        x = r * random.random()
        y = r * random.random()
        z = r * random.random()
        while (np.sqrt(x*x + y*y + z*z) > r):
            x = r * random.random()
            y = r * random.random()
            z = r * random.random()
        return x, y, z
