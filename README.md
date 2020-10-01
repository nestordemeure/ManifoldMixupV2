# Manifold Mixup

Unofficial implementation of [ManifoldMixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf) (Proceedings of ICML 19) for [fast.ai (V2)](https://docs.fast.ai/) based on [Shivam Saboo](https://github.com/shivamsaboo17)'s [pytorch implementation](https://github.com/shivamsaboo17/ManifoldMixup) of manifold mixup, fastai's input mixup [implementation](https://docs.fast.ai/callback.mixup) plus some improvements/variants that I developped with [lessw2020](https://github.com/lessw2020).

This package provides four additional callbacks to the fastai learner :
- `ManifoldMixup` which implements [ManifoldMixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf)
- `OutputMixup` which implements a variant that does the mixup only on the output of the last layer (this was shown to be more performant on a [benchmark](https://forums.fast.ai/t/mixup-data-augmentation/22764/72) and an independant [blogpost](https://medium.com/analytics-vidhya/better-result-with-mixup-at-final-layer-e9ba3a4a0c41))
- `DynamicManifoldMixup` which lets you use manifold mixup with a schedule to increase difficulty progressively
- `DynamicOutputMixup` which lets you use manifold mixup with a schedule to increase difficulty progressively

## Usage

For a minimal demonstration of the various callbacks and their parameters, see the [Demo notebook](https://github.com/nestordemeure/ManifoldMixupV2/blob/master/Demo.ipynb).

### Mixup

To use manifold mixup, you need to import `manifold_mixup` and pass the corresponding callback to the `cbs` argument of your learner :

```python
learner = Learner(data, model, cbs=ManifoldMixup())
learner.fit(8)
```

The `ManifoldMixup` callback takes three parameters :
- `alpha=0.4` parameter of the beta law used to sample the interpolation weight
- `use_input_mixup=True` do you want to apply mixup to the inputs
- `module_list=None` can be used to pass an explicit list of target modules

The `OutputMixup` variant takes only the `alpha` parameters.

### Dynamic mixup

Dynamic callbackss, which are available via `dynamic_mixup`, take three parameters instead of the single `alpha` parameter :
- `alpha_min=0.0` the initial, minimum, value for the parameter of the beta law used to sample the interpolation weight (we recommend keeping it to 0)
- `alpha_max=0.6` the final, maximum, value for the parameter of the beta law used to sample the interpolation weight
- `scheduler=SchedCos` the scheduling function to describe the evolution of `alpha` from `alpha_min` to `alpha_max`

The default schedulers are `SchedLin`, `SchedCos`, `SchedNo`, `SchedExp` and `SchedPoly`.
See the [Annealing](http://dev.fast.ai/callback.schedule#Annealing) section of fastai2's documentation for more informations on available schedulers, ways to combine them and provide your own.

## Notes

### Which modules will be intrumented by ManifoldMixup ?

`ManifoldMixup` tries to establish a sensible list of modules on which to apply mixup:
- it uses a user provided `module_list` if possible
- otherwise it uses only the modules wrapped with `ManifoldMixupModule`
- if none are found, it defaults to modules with `Block` or `Bottleneck` in their name (targetting mostly resblocks)
- finaly, if needed, it defaults to all modules that are not included in the `non_mixable_module_types` list

The `non_mixable_module_types` list contains mostly recurrent layers but you can add elements to it in order to define module classes that should not be used for mixup (*do not hesitate to create an issue or start a PR to add common modules to the default list*).

### When can I use OutputMixup ?

`OutputMixup` applies the mixup directly to the output of the last layer.
This only works if the loss function contains something like a softmax (and not when it is directly used as it is for regression).

Thus, `OutputMixup` **cannot be used for regression**.

### A note on skip-connections / residual-blocks

`ManifoldMixup` (this does not apply to `OutputMixup`) is greatly degraded when applied *inside* a residual block.
This is due to the mixed-up values becoming incoherent with the output of the skip connection (which have not been mixed).

While this implementation is equiped to work around the problem for U-Net and ResNet like architectures, you might run into problems (negligeable improvements over the baseline) with other network structures.
In which case, the best way to apply manifold mixup would be to manually select the modules to be instrumented.

*For more unofficial fastai extensions, see the [Fastai Extensions Repository](https://github.com/nestordemeure/fastai-extensions-repository).*
