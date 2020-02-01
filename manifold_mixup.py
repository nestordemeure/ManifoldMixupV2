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
        new_idx = random.randint(0, len(self.dataset))
        x_0, y_0 = self.dataset[index]
        x_1, y_1 = self.dataset[new_idx]
        return [x_0, x_1], [y_0, y_1]

    def __len__(self):
        return len(self.dataset)

class ManifoldMixupModule(nn.Module):
    " Wrap a module with this class to indicate that you whish to use manifold mixup with this module only."
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

    def forward(self, x):
        x_0, x_1 = x
        self.lam = random.beta(self.alpha, self.alpha)
        l_l = -1 if self.use_input_mixup else 0
        k = random.randint(l_l, len(self.module_list))
        if k == -1:
            x_ = self.lam * x_0 + (1 - self.lam) * x_1
            out = self.model(x_)
        else:
            self._update_hooked(False)
            fetcher_hook = self.module_list[k].register_forward_hook(self.hook_fetch)
            self.model(x_1)
            fetcher_hook.remove()
            self._update_hooked(False)
            modifier_hook = self.module_list[k].register_forward_hook(self.hook_modify)
            out = self.model(x_0)
            modifier_hook.remove()
        self._update_hooked(None)
        return out, self.lam

    def hook_modify(self, module, input, output):
        if not self.hooked:
            output = (1 - self.lam) * self.intermediate_other + self.lam * output
            self._update_hooked(True)

    def hook_fetch(self, module, input, output):
        if not self.hooked:
            self.intermediate_other = output
            self._update_hooked(True)
        else:
            if not self._warning_raised:
                warnings.warn('One of the mixup modules defined in the model is used more than once in forward pass. Mixup will happen only at first call.',
                              Warning)
                self._warning_raised = True

    def _update_hooked(self, flag):
        self.hooked = flag

class ManifoldMixupLoss(nn.Module):
    "Wrap a loss with this class in order to take mixup into account."
    def __init__(self, originalLoss):
        super(ManifoldMixupLoss, self).__init__()
        self.originalLoss = originalLoss

    def forward(self, outs, y):
        out, lam = outs
        y_0, y_1 = y
        loss_0, loss_1 = self.originalLoss(out, y_0), self.originalLoss(out, y_1)
        return lam * loss_0 + (1 - lam) * loss_1
        