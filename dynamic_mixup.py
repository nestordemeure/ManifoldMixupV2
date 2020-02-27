# Curriculum Mixup
# Performs Manifold Mixup and Output Mixup with an increasing alpha to get a gradual increase in difficulty.
# source: https://github.com/nestordemeure/ManifoldMixupV2/blob/master/dynamic_mixup.py

from torch.distributions.beta import Beta
from fastai2.basics import *
from fastai2.callback.schedule import *
from manifold_mixup import *

__all__ = ['DynamicManifoldMixup', 'DynamicOutputMixup']

class DynamicManifoldMixup(ManifoldMixup):
    "Implements a scheduling policy on top of manifold mixup, letting you increase the difficulty progressively."
    def __init__(self, alpha_min=0., alpha_max:float=0.6, scheduler=SchedCos, **kwargs):
        """
        `alpha_min` is the minimum value of the parameter for the beta law, we recommand keeping it to 0.
        `alpha_max` is the parameter for the beta law, we recommend a value between 0. and 1.
        `scheduler` is the scheduling function (such as SchedLin, SchedCos, SchedNo, SchedExp or SchedPoly)

        See the [Annealing](http://dev.fast.ai/callback.schedule#Annealing) section of fastai2's documentation for a list of available schedulers, ways to combine them and provide your own.
        Note that you can pass a raw scheduler (`SchedCos`), that will go from 0 to alpha_max, but also a partially applied scheduler to have full control over the minimum and maximum values (`SchedCos(0.,0.8)`)
        """
        if 'alpha' in kwargs:
            # insures that the user is using alpha_max
            raise Exception('`alpha` parameter detected, please use `alpha_max` (and optionally `alpha_min`) when calling a curriculum based mixup callback.')
        print("Scheduler detected, growing alpha from", alpha_min, "to", alpha_max)
        super().__init__(alpha=0., **kwargs)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.scheduler = scheduler

    def begin_batch(self):
        "Updates alpha as a function of the training percentage."
        # we do the partial application here (and not in the constructor) to avoid a pickle ambiguity error on learn.export
        # due to the fact that the partially applied function as the same name as the original function
        alpha = self.scheduler(self.alpha_min, self.alpha_max)(self.pct_train)
        self.distrib = Beta(tensor(alpha), tensor(alpha))
        super().begin_batch()

class DynamicOutputMixup(OutputMixup):
    "Implements a scheduling policy on top of output mixup, letting you increase the difficulty progressively."
    def __init__(self, alpha_min:float=0.0, alpha_max:float=0.6, scheduler=SchedCos, **kwargs):
        """
        `alpha_min` is the minimum value of the parameter for the beta law, we recommand keeping it to 0.
        `alpha_max` is the parameter for the beta law, we recommend a value between 0. and 1.
        `scheduler` is the scheduling function (such as SchedLin, SchedCos, SchedNo, SchedExp or SchedPoly)

        See the [Annealing](http://dev.fast.ai/callback.schedule#Annealing) section of fastai2's documentation for a list of available schedulers, ways to combine them and provide your own.
        Note that you can pass a raw scheduler (`SchedCos`), that will go from 0 to alpha_max, but also a partially applied scheduler to have full control over the minimum and maximum values (`SchedCos(0.,0.8)`)
        """
        if 'alpha' in kwargs:
            # insures that the user is using alpha_max
            raise Exception('`alpha` parameter detected, please use `alpha_max` (and optionally `alpha_min`) when calling a curriculum based mixup callback.')
        print("Scheduler detected, growing alpha from", alpha_min, "to", alpha_max)
        super().__init__(alpha=0., **kwargs)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.scheduler = scheduler

    def begin_batch(self):
        "Updates alpha as a function of the training percentage."
        # we do the partial application here (and not in the constructor) to avoid a pickle ambiguity error on learn.export
        # due to the fact that the partially applied function as the same name as the original function
        alpha = self.scheduler(self.alpha_min, self.alpha_max)(self.pct_train)
        self.distrib = Beta(tensor(alpha), tensor(alpha))
