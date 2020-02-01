"Implements a fastai callback for the [ManifoldMixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf) training method"
from fastai.layers import *
from fastai.torch_core import *
from fastai.callback import *
from fastai.basic_train import Learner, LearnerCallback

__all__ = ["ManifoldMixupModule", "ManifoldMixupModel", "ManifoldMixupLoss", "ManifoldMixupCallback", "interleaved_manifold_mixup"]

#--------------------------------------------------------------------------------------------------
# Various functions

def adapt_dim(t, t_target):
    """
    Takes a 1D tensor and adds trailing dimensions until it fits the dimension of the target tensor
    This function is useful to multiply tensors of arbitrary size
    implementation inspired by: https://github.com/pytorch/pytorch/issues/9410#issuecomment-552786888
    """
    # this might be implementable with view()
    nb_current_dim = t.dim()
    nb_target_dim = t_target.dim()
    t = t[(..., )*nb_current_dim + (None, ) * (nb_target_dim-nb_current_dim)]
    return t

#--------------------------------------------------------------------------------------------------
# Manifold mixup

class ManifoldMixupModule(Module):
    """
    Wrap a module with this class to indicate that you whish to apply manifold mixup to the output of this module.
    Note that this has no effect and is just used to locate modules of interest when wrapping a model with ManifoldMixupModel
    """
    def __init__(self, module):
        super(ManifoldMixupModule, self).__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        return self.module(x, *args, **kwargs)

class ManifoldMixupModel(Module):
    "Wrap a model with this class in order to apply manifold mixup."
    def __init__(self, model, mixup_all=True, use_input_mixup=True):
        """
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
        self.lam = None
        self.intermediate_output = None
        self.input = None
        self.kwargs = None
        self.output = None

    def forward(self, *x, **kwargs):
        "Takes a triplet (x1,x2,lam) and returns an output."
        if len(x) != 3: return self.model(x[0], **kwargs)
        x1, x2, self.lam = x
        # selects a module to apply mixup
        minimum_module_index = -1 if self.use_input_mixup else 0
        k = np.random.randint(minimum_module_index, len(self.module_list))
        if k == -1: # applies mixup to an input
            self.lam = adapt_dim(self.lam, x1)
            mixed_x = (1 - self.lam) * x1 + self.lam * x2
            output = self.model(mixed_x, **kwargs)
        else: # applies mixup to an inner module
            self.input = x2
            self.kwargs = kwargs
            mixup_hook = self.module_list[k].register_forward_hook(self.hook_mixup)
            output1 = self.model(x1, **kwargs)
            mixup_hook.remove()
            output2 = self.output
            output = torch.cat((output1, output2), dim=0)
        return output

    def hook_mixup(self, module, input, output):
        "stores intermediate result, restart model with different input, "
        if self.intermediate_output is None:
            # stores intermediate output
            self.intermediate_output = output
            # restarts model with different input (using stored output in the mixup and producing a new intermediate output)
            self.output = self.model(self.input, **self.kwargs)
            # performs mixup with new intermediate output
            new_output = output * (1 - self.lam) + self.intermediate_output * self.lam
            self.intermediate_output = None
        else:
            # performs mixup with intermediate output computed with a different input
            self.lam = adapt_dim(self.lam, output) # resize lam
            new_output = output * (1 - self.lam) + self.intermediate_output * self.lam
            # stores ouput
            self.intermediate_output = output
        return new_output

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
        # computes loss
        if len(target) != 3:
            finalLoss = self.criterion(output, target[0])
        else:
            (target1, target2, lam) = target
            if output.size(0) != target1.size(0):
                # if we have not done the mixup step on the input, the output is twice as long
                extendedTarget1 = torch.cat((target1, target2), dim=0)
                extendedTarget2 = torch.cat((target2, target1), dim=0)
                target1 = extendedTarget1
                target2 = extendedTarget2
                lam = torch.cat((lam,lam), dim=0)
            loss1 = self.criterion(output,target1)
            loss2 = self.criterion(output,target2)
            lam = adapt_dim(lam, loss1)
            finalLoss = loss1 * (1-lam) + loss2 * lam
        # applies a reduction to the loss if needed
        if self.reduction == 'mean':  return finalLoss.mean()
        if self.reduction == 'sum':   return finalLoss.sum()
        return finalLoss

    def get_old(self):
        "returns the original loss function"
        if hasattr(self, 'old_crit'):
            return self.old_crit
        elif hasattr(self, 'old_red'):
            setattr(self.criterion, 'reduction', self.old_red)
            return self.criterion

#--------------------------------------------------------------------------------------------------
# Callback

class ManifoldMixupCallback(LearnerCallback):
    "Callback that creates the mixed-up input and target."
    def __init__(self, learn:Learner, alpha:float=0.4, mixup_all:bool=True, use_input_mixup:bool=True):
        """
        `alpha` is the parameter for the beta law.

        If `mixup_all` is set to true, mixup will be applied to any random module.
        Oherwise it will only be applied to a random ManifoldMixupModule.

        If `use_input_mixup` is set to True, mixup might also be applied to the inputs.
        """
        super().__init__(learn)
        self.alpha = alpha
        self.mixup_all = mixup_all
        self.use_input_mixup = use_input_mixup

    def on_train_begin(self, **kwargs):
        "Injects ManifoldMixupLoss and ManifoldMixupModel on top of the current loss function and model."
        self.learn.loss_func = ManifoldMixupLoss(self.learn.loss_func)
        self.learn.model = ManifoldMixupModel(self.learn.model, mixup_all=self.mixup_all, use_input_mixup=self.use_input_mixup)

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies manifold-mixup to `last_input` and `last_target` if `train`."
        if not train: return
        # creates tensor filled with the random ponderation drawed from a beta distribution of parameter alpha
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        lambd = torch.from_numpy(lambd).float().to(last_input.device)
        # builds inputs and ouputs of the form: (batch, batch[shuffle], lambd)
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        new_input = [last_input, last_input[shuffle], lambd]
        new_target = [last_target, last_target[shuffle], lambd]
        return {'last_input': new_input, 'last_target': new_target}

    def on_train_end(self, **kwargs):
        "Restores original loss function and model"
        self.learn.model = self.learn.model.model
        self.learn.loss_func = self.learn.loss_func.get_old()

def interleaved_manifold_mixup(learn:Learner, alpha:float=0.4, mixup_all:bool=True, use_input_mixup:bool=True) -> Learner:
    "Adds manifold-mixup http://proceedings.mlr.press/v97/verma19a/verma19a.pdf to `learn`."
    learn.callback_fns.append(partial(ManifoldMixupCallback, alpha=alpha, mixup_all=mixup_all, use_input_mixup=use_input_mixup))
    return learn

# adds manifold_mixup to Learner's methods
Learner.interleaved_manifold_mixup = interleaved_manifold_mixup
