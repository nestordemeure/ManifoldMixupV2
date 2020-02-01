"Implements a fastai callback for the [ManifoldMixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf) training method"
from fastai.torch_core import *
from fastai.callback import *
from fastai.basic_train import Learner, LearnerCallback
from manifold_mixup import *

__all__ = ["ManifoldMixUpLoss", "ManifoldMixUpCallback", "manifold_mixup"]

# TODO can we get rid of one of the two losses ?
class ManifoldMixUpLoss(Module):
    "Adapt the loss function `criterion` to go with mixup."
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

    def forward(self, output, target):
        if len(target.size()) == 2:
            #y1,y2,lam = target
            loss1, loss2 = self.criterion(output,target[:,0].long()), self.criterion(output,target[:,1].long())
            finalLoss = loss1 * target[:,2] + loss2 * (1-target[:,2])
        else:
            finalLoss = self.criterion(output, target)
        if self.reduction == 'mean':  return finalLoss.mean()
        if self.reduction == 'sum':   return finalLoss.sum()
        return finalLoss

    def get_old(self):
        if hasattr(self, 'old_crit'):  return self.old_crit
        elif hasattr(self, 'old_red'): 
            setattr(self.criterion, 'reduction', self.old_red)
            return self.criterion

class ManifoldMixUpCallback(LearnerCallback):
    "Callback that creates the mixed-up input and target."
    def __init__(self, learn:Learner, alpha:float=0.4, stack_x:bool=False, stack_y:bool=True):
        super().__init__(learn)
        self.alpha,self.stack_x,self.stack_y = alpha,stack_x,stack_y

    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = ManifoldMixUpLoss(self.learn.loss_func)

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies manifold-mixup to `last_input` and `last_target` if `train`."
        if not train: return
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        lambd = last_input.new(lambd)
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        if self.stack_x:
            new_input = [last_input, last_input[shuffle], lambd]
        else: 
            out_shape = [lambd.size(0)] + [1 for _ in range(len(x1.shape) - 1)]
            new_input = (last_input * lambd.view(out_shape) + x1 * (1-lambd).view(out_shape))
        if self.stack_y:
            new_target = torch.cat([last_target[:,None].float(), y1[:,None].float(), lambd[:,None].float()], 1)
        else:
            if len(last_target.shape) == 2:
                lambd = lambd.unsqueeze(1).float()
            new_target = last_target.float() * lambd + y1.float() * (1-lambd)
        return {'last_input': new_input, 'last_target': new_target}

    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()

def manifold_mixup(learn:Learner, alpha:float=0.4) -> Learner:
    "Adds manifold-mixup http://proceedings.mlr.press/v97/verma19a/verma19a.pdf to `learn`."
    learn.callback_fns.append(partial(ManifoldMixUpCallback, alpha=alpha))
    return learn