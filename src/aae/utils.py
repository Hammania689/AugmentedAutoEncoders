import time


class TimerManager(object):

    def __init__(self):
        self.name = None
        self._prev = []
        self.reset()

    def reset(self):
        self._t_start = None

    def __call__(self, name):

        if isinstance(self.name, type(None)):
            self.name = name

        assert name == self.name, (f"Manager is set to track {self.name}.\n"
                                       + f"Was given {name}.")


        if not isinstance(self._t_start, type(None)):
            cur_tick = time.time() - self._t_start
            self._prev.append(cur_tick)

            self._t_start = None
        else:
            self._t_start = time.time()


    def __repr__(self, prec: float=.4):

        import numpy as np

        if len(self._prev) == 0:
            mu = 0.0
            std =  0.0
        else:
            mu = np.mean(self._prev)
            std = np.std(self._prev)

        return (  f"{self.name}\n{'='*50}\n"
                + f"Avg: {mu:{prec}} | std: {std:{prec}} | # of ticks {len(self._prev)}")



