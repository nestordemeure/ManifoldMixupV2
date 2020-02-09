# Manifold Mixup
# Implements a fastai callback for the Manifold Mixup training method.
# source: https://github.com/nestordemeure/ManifoldMixup
# reference: http://proceedings.mlr.press/v97/verma19a/verma19a.pdf

from torch.distributions.beta import Beta
from fastai.torch_core import *
from fastai.callback import *
from fastai.basic_train import Learner, LearnerCallback
from fastai.text.models import AWD_LSTM
from fastai.vision.models import UnetBlock
from fastai.callbacks.mixup import MixUpLoss

__all__ = ["ManifoldMixupModule", "ManifoldMixupCallback", "non_mixable_module_types", "manifold_mixup", "output_mixup"]

#------------------------------------------------------------------------------
# Module selection

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

# classes of modules that should be avoided when using mixup
# mostly modules that are just propagating their inputs and recurent layers
non_mixable_module_types = [nn.Sequential, nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout,
                            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
                            nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell, AWD_LSTM,
                            nn.RNN, nn.RNNBase, nn.RNNCell, nn.RNNCellBase]

def _is_mixable(m):
    "Checks wether the module m is an instance of a module that is allowed for mixup."
    return not any(isinstance(m, non_mixable_class) for non_mixable_class in non_mixable_module_types)

def _is_block_module(m):
    "Checks wether a module is a Block (typically a kind of resBlock)"
    return "block" in str(type(m)).lower()

def is_suitable_output_module(m):
    "Returns true if the module is a suitable output module"
    moduleName = str(type(m)).lower()
    return not (("loss" in moduleName) or ("max" in moduleName))

def _get_mixup_module_list(model, module_list=None):
    "returns all the modules that can be used for mixup"
    # checks for a user defined module list
    if module_list is not None:
        print(f'Manifold mixup: user defined module list detected, {len(module_list)} modules will be used for mixup.')
        return module_list
    module_list = list(model.modules())
    # checks for modules wrapped with ManifoldMixupModule
    user_wrapped_modules = list(filter(lambda module: isinstance(module, ManifoldMixupModule), module_list))
    if len(user_wrapped_modules) != 0:
        print(f'Manifold mixup: ManifoldMixupModule modules detected, {len(user_wrapped_modules)} modules will be used for mixup.')
        return user_wrapped_modules
    # checks for UnetBlock to only instrument the decoder part of a U-Net
    # following the recommendations of: `Prostate Cancer Segmentation using Manifold Mixup U-Net`
    ublock_modules = list(filter(lambda module: isinstance(module, UnetBlock), module_list))
    if len(ublock_modules) != 0:
        print(f'Manifold mixup: U-Net structure detected, {len(ublock_modules)} modules will be used for mixup.')
        return ublock_modules
    # checks for blocks
    block_modules = list(filter(_is_block_module, module_list))
    if len(block_modules) != 0:
        print(f'Manifold mixup: Block structure detected, {len(block_modules)} modules will be used for mixup.')
        return block_modules
    # checks for any module that is mixable
    mixable_modules = list(filter(_is_mixable, module_list))
    if len(mixable_modules) != 0:
        print(f'Manifold mixup: no known network structure detected, {len(mixable_modules)} modules will be used for mixup.')
        return mixable_modules
    # no module has been found
    raise ValueError('No eligible layer found for mixup. Try wrapping candidate modules with ManifoldMixupModule or passing an explicit list of targets with module_list')

def _get_output_module(model):
    "returns the last module suitable for output mixup (neither loss nor softmax)"
    module_list = list(model.modules())
    output_module = next((m for m in reversed(module_list) if is_suitable_output_module(m)), None)
    if output_module is not None: return output_module
    raise ValueError('No eligible layer found for mixup.')

#------------------------------------------------------------------------------
# Manifold Mixup

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

class ManifoldMixupCallback(LearnerCallback):
    "Callback that creates the mixed-up input and target."
    def __init__(self, learn:Learner, alpha:float=0.4, use_input_mixup:bool=True,
                 module_list:Collection=None, stack_y:bool=True):
        """
        `alpha` is the parameter for the beta law.

        If `use_input_mixup` is set to True, mixup might also be applied to the inputs.

        If `stack_y` is set to false, the target outputs will be directly linearly combined (good for regression).
        Otherwise they will be stacked and forwarded to MixUpLoss which works under the hypothesis that the output is a long and performs the combinaison after having evaluated the loss (good for classification).

        The algorithm tries to establish a sensible list of modules on which to apply mixup:
        - it uses a user provided `module_list` if possible
        - otherwise it uses only the modules wrapped with `ManifoldMixupModule`
        - if none are found, it defaults to modules with `Block` in their name (targetting mostly resblocks)
        - finaly, if needed, it defaults to all modules that are not included in the `non_mixable_module_types` list
        """
        super().__init__(learn)
        # parameters describing the mixup
        self.distrib = Beta(tensor(alpha), tensor(alpha))
        self.use_input_mixup = use_input_mixup
        self.module_list = _get_mixup_module_list(learn.model, module_list)
        self.stack_y = stack_y
        # temporary variables storing intermediate states
        self._lambd = None
        self._shuffled_index = None
        self._mixup_hook = None
        self._is_input_mixup = None # are we using simple input mixup
        self._mixup_is_done = False # has the mixup step already been done
        self._warning_raised = False

    def on_train_begin(self, **kwargs):
        "Injects MixupLoss on top of the current loss function."
        if self.stack_y: self.learn.loss_func = MixUpLoss(self.learn.loss_func)

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Selects a module to apply mixup and modifies the target accordingly."
        if not train: return
        # creates tensor filled with the random ponderation drawn from a beta distribution of parameter alpha
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        self._lambd = torch.from_numpy(lambd).float().to(last_input.device)
        # decides on a way to shuffle inputs
        self._shuffled_index = torch.randperm(last_target.size(0)).to(last_input.device)
        # selects a module to apply mixup
        minimum_module_index = -1 if self.use_input_mixup else 0
        k = np.random.randint(minimum_module_index, len(self.module_list))
        if k == -1: # applies mixup to an input
            self._is_input_mixup = True
            input_lambd = _adapt_dim(self._lambd, last_input)
            last_input = last_input * input_lambd + last_input[self._shuffled_index] * (1 - input_lambd)
        else: # applies mixup to an inner module
            self._is_input_mixup = False
            self._mixup_hook = self.module_list[k].register_forward_hook(self.hook_mixup)
        # process the target
        output_lambd = _adapt_dim(self._lambd, last_target)
        last_target2 = last_target[self._shuffled_index]
        if self.stack_y:
            # stores the target but also a properly shuffled copy and the lambda to combine them
            new_target = torch.cat([last_target[:,None].float(), last_target2[:,None].float(), output_lambd[:,None].float()], 1)
        else:
            # mixes the targets
            new_target = last_target.float() * output_lambd + last_target2.float() * (1-output_lambd)
        return {'last_input': last_input, 'last_target': new_target}

    def hook_mixup(self, module, input, output):
        "Interupt one run to use its intermediate results with a second model call."
        if not self._mixup_is_done: # performs mixup
            lambd = _adapt_dim(self._lambd, output)
            output = output * lambd + output[self._shuffled_index] * (1 - lambd)
            self._mixup_is_done = True
            return output
        elif not self._warning_raised:
            warnings.warn(f"One of the mixup modules ({ type(module) }) defined in the model is used more than once in forward pass.\n" \
                          "Mixup will happen only at first call. This warning might be due to :\n" \
                          "- a recurent modules being intrumented or a single module being aplied to different inputs (you should add those modules to `non_mixable_module_types` as they might interfere with mixup),\n" \
                          "- a module being applied to its own output in a loop (in which case you can safely ignore this warning).",
                          Warning)
            self._warning_raised = True

    def on_loss_begin(self, train, **kwargs):
        "Removes hook and stacks outputs if needed"
        if (not train) or self._is_input_mixup: return
        self._mixup_hook.remove()
        self._mixup_is_done = False

    def on_train_end(self, **kwargs):
        "Restores the original loss function"
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()

def manifold_mixup(learn:Learner, alpha:float=0.4, use_input_mixup:bool=True, module_list:Collection=None, stack_y:bool=True) -> Learner:
    """
    Adds manifold-mixup http://proceedings.mlr.press/v97/verma19a/verma19a.pdf to `learn`.

    `alpha` is the parameter for the beta law.

    If `use_input_mixup` is set to True, mixup might also be applied to the inputs.

    If `stack_y` is set to false, the target outputs will be directly linearly combined (good for regression).
    Otherwise they will be stacked and forwarded to MixUpLoss which works under the hypothesis that the output is a long and performs the combinaison after having evaluated the loss (good for classification).

    The algorithm tries to establish a sensible list of modules on which to apply mixup:
    - it uses a user provided `module_list` if possible
    - otherwise it uses only the modules wrapped with `ManifoldMixupModule`
    - if none are found, it defaults to modules with `Block` in their name (targetting mostly resblocks)
    - finaly, if needed, it defaults to all modules that are not included in the `non_mixable_module_types` list
    """
    learn.callback_fns.append(partial(ManifoldMixupCallback, alpha=alpha, use_input_mixup=use_input_mixup, module_list=module_list, stack_y=stack_y))
    return learn

def output_mixup(learn:Learner, alpha:float=0.4, stack_y:bool=True) -> Learner:
    """
    Adds a variant of manifold-mixup, that is only applied to the last viable module, to `learn`.

    `alpha` is the parameter for the beta law.

    If `stack_y` is set to false, the target outputs will be directly linearly combined (good for regression).
    Otherwise they will be stacked and forwarded to MixUpLoss which works under the hypothesis that the output is a long and performs the combinaison after having evaluated the loss (good for classification).
    """
    module_list = [_get_output_module(learn.model)]
    learn.callback_fns.append(partial(ManifoldMixupCallback, alpha=alpha, use_input_mixup=False, module_list=module_list, stack_y=stack_y))
    return learn

# adds manifold_mixup to Learner's available methods
Learner.manifold_mixup = manifold_mixup
Learner.output_mixup = output_mixup
