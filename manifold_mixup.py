"Implements [ManifoldMixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf) training method"
import warnings
import torch.nn as nn
from torch.utils.data import Dataset
from numpy import random

__all__ = ["ManifoldMixupDataset", "ManifoldMixupModule", "ManifoldMixupModel", "ManifoldMixupLoss"]

class ManifoldMixupDataset(Dataset):
    "Wrap a dataset with this class in order to produce mixup compatible input*output pairs."
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        # TODO we could avoid using the same x2 twice with a shuffle
        new_idx = random.randint(0, len(self.dataset))
        x1, y1 = self.dataset[index]
        x2, y2 = self.dataset[new_idx]
        return [x1, x2], [y1, y2]

    def __len__(self):
        return len(self.dataset)

class ManifoldMixupModule(nn.Module):
    """
    Wrap a module with this class to indicate that you whish to apply manifold mixup to the output of this module.
    Note that this has no effect and is just used to locate modules of interest when wrapping a model with ManifoldMixupModel 
    """
    def __init__(self, module):
        super(ManifoldMixupModule, self).__init__()
        self.module = module
    def forward(self, x, *args, **kwargs):
        return self.module(x, *args, **kwargs)

class ManifoldMixupModel(nn.Module):
    "Wrap a model with this class in order to apply manifold mixup."
    def __init__(self, model, alpha=0.4, mixup_all=True, use_input_mixup=True):
        """
        `alpha` is the parameter for the beta law.

        If `mixup_all` is set to true, mixup will be applied to any random module.
        Oherwise it will only be applied to a random ManifoldMixupModule.

        If `use_input_mixup` is set to True, mixup will also be applied to the inputs.
        """
        super(ManifoldMixupModel, self).__init__()
        self.use_input_mixup = use_input_mixup
        self.model = model
        if not mixup_all:
            self.module_list = list(filter(lambda module: isinstance(module, ManifoldMixupModule), list(self.model.modules())))
        else:
            self.module_list = list(self.model.modules())
        if len(self.module_list) == 0:
            raise ValueError('No eligible layer found for mixup. Try passing mixup_all=True or wrap one of your modules with a ManifoldMixupModule')
        print(f'{len(self.module_list)} modules eligible for mixup')
        self.alpha = alpha
        self.intermediate_other = None
        self.lam = None
        self.hooked = None
        self._warning_raised = False

    # input is a pair (x=(x0,x1))
    # output is a classical ouput
    # lam is computed on the fly
    # outputs a pair output,lam
    def forward(self, x):
        x1, x2 = x
        self.lam = random.beta(self.alpha, self.alpha)
        # selects a module to apply mixup
        minimum_module_index = -1 if self.use_input_mixup else 0
        k = random.randint(minimum_module_index, len(self.module_list))
        if k == -1: # applies mixup to an input
            mixed_x = self.lam * x1 + (1 - self.lam) * x2
            output = self.model(mixed_x)
        else: # applies mixup to an inner module
            # applies model to x1 and extracts output of target module
            self.hooked = False
            fetcher_hook = self.module_list[k].register_forward_hook(self.hook_fetch)
            self.model(x1)
            fetcher_hook.remove()
            # applies model to x2 and injects x1's output at the target module
            self.hooked = False
            modifier_hook = self.module_list[k].register_forward_hook(self.hook_modify)
            output = self.model(x2)
            modifier_hook.remove()
        self.hooked = None
        return output, self.lam

    # TODO we might want to kill the run once we get the information we want in order to improve runtime
    # TODO if we can suspend a run we could even switch outputs between x1 and x2 in order to avoid a x2 slowdown
    def hook_fetch(self, module, input, output):
        "Intercepts the output of a module diring a run."
        if not self.hooked:
            self.intermediate_other = output
            self.hooked = True
        elif not self._warning_raised:
            warnings.warn('One of the manifold mixup modules defined in the model is used more than once in forward pass. Mixup will happen only at first call.', Warning)
            self._warning_raised = True
    
    def hook_modify(self, module, input, output):
        "Mix the output of this batch with the output of another batch previously saved."
        if not self.hooked:
            output = self.lam * self.intermediate_other + (1 - self.lam) * output
            self.hooked = True

class ManifoldMixupLoss(nn.Module):
    "Wrap a loss with this class in order to take mixup into account."
    def __init__(self, originalLoss):
        super(ManifoldMixupLoss, self).__init__()
        self.originalLoss = originalLoss

    def forward(self, outputs, targets):
        output, lam = outputs
        target1, target2 = targets
        loss1, loss2 = self.originalLoss(output, target1), self.originalLoss(output, target2)
        return lam * loss1 + (1 - lam) * loss2
