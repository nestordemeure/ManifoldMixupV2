"Implements a fastai callback for the [Manifold Mixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf) training method."
from fastai.torch_core import *
from fastai.callback import *
from fastai.basic_train import Learner, LearnerCallback
from fastai.text import models

__all__ = ["ManifoldMixupModule", "ManifoldMixupLoss", "ManifoldMixupCallback", "non_mixable_module_types", "manifold_mixup", "output_mixup"]

def _adapt_dim(t, t_target):
    """
    Takes a tensor and adds trailing dimensions until it fits the dimension of the target tensor
    This function is useful to multiply tensors of arbitrary size
    implementation inspired by: https://github.com/pytorch/pytorch/issues/9410#issuecomment-552786888
    """
    # this might be implementable with `view` (?)
    nb_current_dim = t.dim()
    nb_target_dim = t_target.dim()
    t = t[(..., )*nb_current_dim + (None, ) * (nb_target_dim-nb_current_dim)]
    return t

# classes of modules that should be avoided when using mixup
non_mixable_module_types = [nn.Sequential, nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout,
                            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
                            nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell, models.AWD_LSTM,
                            nn.RNN, nn.RNNBase, nn.RNNCell, nn.RNNCellBase]

def _is_mixable(m):
    "Checks wether the module m is an instance of a module that is allowed for mixup."
    return not any(isinstance(m, non_mixable_class) for non_mixable_class in non_mixable_module_types)

class ManifoldMixupModule(Module):
    """
    Wrap a module with this class to indicate that you wish to apply manifold mixup to the output of this module.
    Note that this, by itself, has no effect and is just used to locate modules of interest when using the ManifoldMixupCallback.
    """
    def __init__(self, module):
        super(ManifoldMixupModule, self).__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        return self.module(x, *args, **kwargs)

def _get_mixup_module_list(model, use_only_mixup_modules):
    "returns all the modules that can be used for mixup"
    if use_only_mixup_modules:
        module_list = list(filter(lambda module: isinstance(module, ManifoldMixupModule), list(model.modules())))
    else:
        module_list = list(filter(_is_mixable, list(model.modules())))
    if len(module_list) == 0:
        raise ValueError('No eligible layer found for mixup. Try passing use_only_mixup_modules=False or wrap one of your modules with a ManifoldMixupModule')
    print(f'{len(module_list)} modules eligible for mixup')
    return module_list

class ManifoldMixupCallback(LearnerCallback):
    "Callback that creates the mixed-up input and target."
    def __init__(self, learn:Learner, alpha:float=0.4, use_input_mixup:bool=True,
                 use_only_mixup_modules:bool=False, module_list:Collection=None):
        """
        `alpha` is the parameter for the beta law.

        If `use_input_mixup` is set to True, mixup might also be applied to the inputs.

        If `use_only_mixup_modules` is set to false, mixup will be applied to a random valid module.
        Oherwise it will only be applied to the modules wrapped with ManifoldMixupModule.

        You can also hardcode the modules you want to use by passing them with `module_list`.
        Doing so will bypass `use_only_mixup_modules` but not `use_input_mixup`.
        """
        super().__init__(learn)
        # parameters describing the mixup
        self.alpha = alpha
        self.use_only_mixup_modules = use_only_mixup_modules
        self.use_input_mixup = use_input_mixup
        self.module_list = _get_mixup_module_list(learn.model, use_only_mixup_modules) if module_list is None else module_list
        # temporary variables storing intermediate states
        self.lambd = None
        self.shuffled_index = None
        self.mixup_hook = None
        self.is_input_mixup = None # are we using simple input mixup
        self.mixup_is_done = False # has the mixup step already been done
        self._warning_raised = False

    def on_train_begin(self, **kwargs):
        "Injects ManifoldMixupLoss on top of the current loss function."
        self.learn.loss_func = ManifoldMixupLoss(self.learn.loss_func)

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Selects a module to apply mixup and modifies the target accordingly."
        if not train: return
        # creates tensor filled with the random ponderation drawn from a beta distribution of parameter alpha
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        self.lambd = torch.from_numpy(lambd).float().to(last_input.device)
        # decides on a way to shuffle inputs
        self.shuffled_index = torch.randperm(last_target.size(0)).to(last_input.device)
        # selects a module to apply mixup
        minimum_module_index = -1 if self.use_input_mixup else 0
        k = np.random.randint(minimum_module_index, len(self.module_list))
        if k == -1: # applies mixup to an input
            self.is_input_mixup = True
            input_lambd = _adapt_dim(self.lambd, last_input)
            last_input = last_input * (1 - input_lambd) + last_input[self.shuffled_index] * input_lambd
        else: # applies mixup to an inner module
            self.is_input_mixup = False
            self.mixup_hook = self.module_list[k].register_forward_hook(self.hook_mixup)
        # stores the target but also a properly shuffle copy and the lambda to combine them
        new_target = [last_target, last_target[self.shuffled_index], self.lambd]
        return {'last_input': last_input, 'last_target': new_target}

    def hook_mixup(self, module, input, output):
        "Interupt one run to use its intermediate results with a second model call."
        if not self.mixup_is_done: # performs mixup
            lambd = _adapt_dim(self.lambd, output)
            output = output * (1 - lambd) + output[self.shuffled_index] * lambd
            self.mixup_is_done = True
            return output
        elif not self._warning_raised:
            warnings.warn('One of the mixup modules defined in the model is used more than once in forward pass. Mixup will happen only at first call.', Warning)
            self._warning_raised = True

    def on_loss_begin(self, train, **kwargs):
        "Removes hook and stacks outputs if needed"
        if (not train) or self.is_input_mixup: return
        self.mixup_hook.remove()
        self.mixup_is_done = False

    def on_train_end(self, **kwargs):
        "Restores the original loss function"
        self.learn.loss_func = self.learn.loss_func.get_old()

# TODO can we use the loss from input mixup ?
class ManifoldMixupLoss(Module):
    "Adapts the loss function `criterion` to be used with manifold mixup."
    def __init__(self, criterion, reduction='mean'):
        super().__init__()
        if hasattr(criterion, 'reduction'):
            self.criterion = criterion
            self.old_red = criterion.reduction
            setattr(self.criterion, 'reduction', 'none')
        else:
            self.criterion = partial(criterion, reduction='none')
            self.old_crit = criterion
        self.reduction = reduction

    def forward(self, output, *target):
        if len(target) != 3:
            finalLoss = self.criterion(output, target[0])
        else:
            (target1, target2, lam) = target
            loss1 = self.criterion(output,target1)
            loss2 = self.criterion(output,target2)
            lam = _adapt_dim(lam, loss1)
            finalLoss = loss1 * (1-lam) + loss2 * lam
        if self.reduction == 'mean':  return finalLoss.mean()
        if self.reduction == 'sum':   return finalLoss.sum()
        return finalLoss

    def get_old(self):
        "Returns the original loss function"
        if hasattr(self, 'old_crit'):
            return self.old_crit
        elif hasattr(self, 'old_red'):
            setattr(self.criterion, 'reduction', self.old_red)
            return self.criterion

def manifold_mixup(learn:Learner, alpha:float=0.4, use_input_mixup:bool=True, use_only_mixup_modules:bool=False, module_list:Collection=None) -> Learner:
    "Adds manifold-mixup http://proceedings.mlr.press/v97/verma19a/verma19a.pdf to `learn`."
    learn.callback_fns.append(partial(ManifoldMixupCallback, alpha=alpha, use_input_mixup=use_input_mixup, use_only_mixup_modules=use_only_mixup_modules, module_list=module_list))
    return learn

def output_mixup(learn:Learner, alpha:float=0.4, use_only_mixup_modules:bool=False) -> Learner:
    "Adds a variant of manifold-mixup, that is only applied to the last viable module, to `learn`."
    module_list = [_get_mixup_module_list(learn.model, use_only_mixup_modules)[-1]]
    learn.callback_fns.append(partial(ManifoldMixupCallback, alpha=alpha, use_input_mixup=False, use_only_mixup_modules=use_only_mixup_modules, module_list=module_list))
    return learn

# adds manifold_mixup to Learner's available methods
Learner.manifold_mixup = manifold_mixup
Learner.output_mixup = output_mixup
