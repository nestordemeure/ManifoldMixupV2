# Manifold Mixup
# Implements a fastai-V2 callback for the Manifold Mixup and Output Mixup training methods.
# source: https://github.com/nestordemeure/ManifoldMixupV2/blob/master/manifold_mixup.py
# reference: http://proceedings.mlr.press/v97/verma19a/verma19a.pdf

from torch.distributions.beta import Beta
from fastai2.basics import *
from fastai2.callback.mixup import reduce_loss
from fastai2.text.models import AWD_LSTM
from fastai2.vision.models.unet import UnetBlock
from fastai2.tabular.model import TabularModel

__all__ = ['ManifoldMixupModule', 'ManifoldMixup', 'OutputMixup', 'non_mixable_module_types']

#------------------------------------------------------------------------------
# Module selection

class ManifoldMixupModule(Module):
    """
    Wrap a module with this class to indicate that you wish to apply manifold mixup to the output of this module.
    Note that this, by itself, has no effect and is just used to locate modules of interest when using the ManifoldMixupCallback.
    """
    def __init__(self, module):
        super().__init__()
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

def _get_mixup_module_list(model):
    "returns all the modules that can be used for mixup"
    module_list = list(model.modules())
    # checks for modules wrapped with ManifoldMixupModule
    user_wrapped_modules = list(filter(lambda module: isinstance(module, ManifoldMixupModule), module_list))
    if len(user_wrapped_modules) != 0:
        print(f'Manifold mixup: ManifoldMixupModule modules detected, {len(user_wrapped_modules)} modules will be used for mixup.')
        return user_wrapped_modules
    # checks for tabular model in which case we get only linear layers
    if isinstance(model, TabularModel):
        linear_modules = list(filter(lambda module: isinstance(module, nn.Linear), module_list))
        print(f'Manifold mixup: TabularModel detected, {len(linear_modules)} modules will be used for mixup.')
        return linear_modules
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

#------------------------------------------------------------------------------
# Manifold Mixup

class ManifoldMixup(Callback):
    "Callback that mixes a random inner layer and the target."
    run_after,run_valid = [Normalize],False
    def __init__(self, alpha:float=0.4, use_input_mixup:bool=True, module_list:Collection=None):
        """
        `alpha` is the parameter for the beta law.

        If `use_input_mixup` is set to True, mixup might also be applied to the inputs.

        The algorithm tries to establish a sensible list of modules on which to apply mixup:
        - it uses a user provided `module_list` if possible
        - otherwise it uses only the modules wrapped with `ManifoldMixupModule`
        - if none are found, it defaults to modules with `Block` in their name (targetting mostly resblocks)
        - finaly, if needed, it defaults to all modules that are not included in the `non_mixable_module_types` list
        """
        alpha = float(alpha) # insures that alpha is a float as an int would crash Beta
        self.distrib = Beta(tensor(alpha), tensor(alpha))
        self.use_input_mixup = use_input_mixup
        self.module_list = module_list
        self.mixup_hook_handle = None

    def begin_fit(self):
        "replace the loss function with one that is adapted to mixup"
        self.warning_raised = False
        # lists the modules that can be used for mixup
        if self.module_list is None:
            self.module_list = _get_mixup_module_list(self.learn.model)
        # if the ouput is integer based (classification), we stack the losses otherwise we combine them
        self.stack_y = getattr(self.learn.loss_func, 'y_int', False)
        if self.stack_y:
            self.old_lf = self.learn.loss_func
            self.learn.loss_func = self.lf

    def begin_batch(self):
        "mixes inputs and stores mixed output and lambda"
        self.shuffle = torch.randperm(self.y.size(0)).to(self.y.device)
        # lambda used for linear combinaison
        lam = self.distrib.sample((self.y.size(0),)).squeeze().to(self.y.device)
        lam = torch.stack([lam, 1-lam], 1)
        self.lam = lam.max(1)[0]
        # selects a module to apply mixup
        minimum_module_index = -1 if self.use_input_mixup else 0
        k = np.random.randint(minimum_module_index, len(self.module_list))
        if k == -1: # applies mixup to an input
            assert (not isinstance(self.x, tuple)), "Manifold mixup: Your input type does not seem compatible with input mixup, please set `use_input_mixup=False`."
            xb1 = tuple(L(self.xb).itemgot(self.shuffle))
            nx_dims = len(self.x.size())
            self.learn.xb = tuple(L(xb1,self.xb).map_zip(torch.lerp,weight=unsqueeze(self.lam, n=nx_dims-1)))
        else: # applies mixup to an inner module
            self.mixup_hook_handle = self.module_list[k].register_forward_hook(self.hook_mixup)
        # replaces y with a linear combinaison of y and yb1
        self.yb1 = tuple(L(self.yb).itemgot(self.shuffle))
        if not self.stack_y:
            ny_dims = len(self.y.size())
            self.learn.yb = tuple(L(self.yb1,self.yb).map_zip(torch.lerp,weight=unsqueeze(self.lam, n=ny_dims-1)))
        # flags used to control that everything ran properly
        self.mixup_has_been_applied = False

    def hook_mixup(self, module, input, output):
        "Interupt one run to use its intermediate results with a second model call."
        if not self.mixup_has_been_applied: # performs mixup
            output_dims = len(output.size())
            output = torch.lerp(output[self.shuffle], output, weight=unsqueeze(self.lam, n=output_dims-1))
            self.mixup_has_been_applied = True
            return output
        elif not self.warning_raised:
            warnings.warn(f"One of the mixup modules ({ type(module) }) defined in the model is used more than once in forward pass.\n" \
                          "Mixup will happen only at first call. This warning might be due to :\n" \
                          "- a recurent modules being intrumented or a single module being aplied to different inputs (you should add those modules to `non_mixable_module_types` as they might interfere with mixup),\n" \
                          "- a module being applied to its own output in a loop (in which case you can safely ignore this warning).",
                          Warning)
            self.warning_raised = True

    def lf(self, pred, *yb):
        "loss function adapted to mixup"
        if not self.training: return self.old_lf(pred, *yb)
        with NoneReduce(self.old_lf) as lf:
            # lam*loss(yb1) + (1-lam)*loss(yb)
            loss = torch.lerp(lf(pred,*self.yb1), lf(pred,*yb), self.lam)
        return reduce_loss(loss, getattr(self.old_lf, 'reduction', 'mean'))

    def after_batch(self):
        "Removes hook if needed"
        if self.mixup_hook_handle is not None:
            self.mixup_hook_handle.remove()
            self.mixup_hook_handle = None

    def after_fit(self):
        "restores the original loss function"
        if self.stack_y: self.learn.loss_func = self.old_lf

#------------------------------------------------------------------------------
# Output Mixup

class OutputMixup(Callback):
    """
    Callback that mixes the output of the last layer and the target.
    NOTE: this callback is not suitable for regression problems
    """
    run_after,run_valid = [Normalize],False
    def __init__(self, alpha:float=0.4):
        "`alpha` is the parameter for the beta law."
        alpha = float(alpha) # insures that alpha is a float as an int would crash Beta
        self.distrib = Beta(tensor(alpha), tensor(alpha))

    def begin_fit(self):
        "Injects the new loss function"
        if getattr(self.learn.loss_func, 'y_int', False):
            # classification type of output
            self.old_loss_func = self.learn.loss_func
            self.learn.loss_func = self.mixed_loss
            print(f'Output mixup: the loss function is now properly wrapped.')
        else:
            # the output type seem unfit for instrumentation
            raise Exception("You cannot use output mixup for regression problems.")

    def after_fit(self):
        "Restores the original loss function."
        self.learn.loss_func = self.old_loss_func

    def mixed_loss(self, pred, *yb):
        """
        Loss function that mixes the prediction before computing the loss and weighting it.
        This requires that the softmax / loss function is done fully inside the loss and not in the network.
        """
        if not self.training: return self.old_loss_func(pred, *yb)
        with NoneReduce(self.old_loss_func) as lf:
            # shuffles used to match batch elements
            shuffle = torch.randperm(len(*yb)).to(pred.device)
            # lambda used for linear combinaison
            lam = self.distrib.sample((len(*yb),)).squeeze().to(pred.device)
            lam = torch.stack([lam, 1-lam], 1)
            lam = lam.max(1)[0]
            # shuffled prediction
            pred_dims = len(pred.size())
            pred_mixed = torch.lerp(pred[shuffle], pred, weight=unsqueeze(lam, n=pred_dims-1))
            # shuffled targets
            yb_shuffled = tuple(L(yb).itemgot(shuffle))
            # final loss
            loss = torch.lerp(lf(pred_mixed,*yb_shuffled), lf(pred_mixed,*yb), lam)
        return reduce_loss(loss, getattr(self.old_loss_func, 'reduction', 'mean'))
