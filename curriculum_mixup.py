# Curriculum Mixup
# Performs Manifold Mixup and Output Mixup with an increasing alpha to get a gradual increase in difficulty.

from torch.distributions.beta import Beta
from fastai2.basics import *
from fastai2.callback.schedule import *
from manifold_mixup import *

__all__ = ['CManifoldMixup', 'COutputMixup',
           'SchedLin', 'SchedCos', 'SchedNo', 'SchedExp', 'SchedPoly', 'combine_scheds', 'combined_cos'] # reexport for conveniance

class CManifoldMixup(ManifoldMixup):
    "Implements a scheduling policy on top of manifold mixup, letting you increase the difficulty progressively."
    def __init__(self, alpha_max:float=0.4, scheduler=SchedCos, **kwargs):
        """
        `alpha_max` is the parameter for the beta law, we recommend a value between 0. and 1.
        `scheduler` is the scheduling function (such as SchedLin, SchedCos, SchedNo, SchedExp or SchedPoly)

        See the [Annealing](http://dev.fast.ai/callback.schedule#Annealing) section of fastai2's documentation for a list of available schedulers, ways to combine them and provide your own.
        Note that you can pass a raw scheduler (`SchedCos`), that will go from 0 to alpha_max, but also a partially applied scheduler to have full control over the minimum and maximum values (`SchedCos(0.,0.8)`)
        """
        # insures that the user is using alpha_max
        if 'alpha' in kwargs:
            raise Exception('`alpha` parameter detected, please use `alpha_max` when calling a curriculum based mixup callback.')
        # init ManifoldMixup
        super().__init__(alpha=0., **kwargs)
        # discriminate between raw scheduler and partially applied schedular (that already have a minimum and maximum defined) 
        if isinstance(scheduler, functools.partial):
            print("Partially applied scheduler detected, ignoring alpha_max parameter.")
            self.scheduler = scheduler
        else:
            self.scheduler = scheduler(0., alpha_max)

    def begin_batch(self):
        "Updates alpha as a function of the training percentage."
        alpha = self.scheduler(self.pct_train)
        self.distrib = Beta(tensor(alpha), tensor(alpha))
        super().begin_batch()

class COutputMixup(OutputMixup):
    "Implements a scheduling policy on top of output mixup, letting you increase the difficulty progressively."
    def __init__(self, alpha_max:float=0.4, scheduler=SchedCos, **kwargs):
        """
        `alpha_max` is the parameter for the beta law, we recommend a value between 0. and 1.
        `scheduler` is the scheduling function (such as SchedLin, SchedCos, SchedNo, SchedExp or SchedPoly)

        See the [Annealing](http://dev.fast.ai/callback.schedule#Annealing) section of fastai2's documentation for a list of available schedulers, ways to combine them and provide your own.
        Note that you can pass a raw scheduler (`SchedCos`), that will go from 0 to alpha_max, but also a partially applied scheduler to have full control over the minimum and maximum values (`SchedCos(0.,0.8)`)
        """
        # insures that the user is using alpha_max
        if 'alpha' in kwargs:
            raise Exception('`alpha` parameter detected, please use `alpha_max` when calling a curriculum based mixup callback.')
        # init ManifoldMixup
        super().__init__(alpha=0., **kwargs)
        # discriminate between raw scheduler and partially applied schedular (that already have a minimum and maximum defined) 
        if isinstance(scheduler, functools.partial):
            print("Partially applied scheduler detected, ignoring alpha_max parameter.")
            self.scheduler = scheduler
        else:
            self.scheduler = scheduler(0., alpha_max)

    def begin_batch(self):
        "Updates alpha as a function of the training percentage."
        alpha = self.scheduler(self.pct_train)
        self.distrib = Beta(tensor(alpha), tensor(alpha))
