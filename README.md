# Manifold Mixup V2

Unofficial implementation of [ManifoldMixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf) (Proceedings of ICML 19) for [fast.ai V2](http://dev.fast.ai/) based on [Shivam Saboo](https://github.com/shivamsaboo17)'s [pytorch implementation](https://github.com/shivamsaboo17/ManifoldMixup) of manifold mixup, fastai's input mixup [implementation](https://dev.fast.ai/callback.mixup) plus some personnal improvements/variants.

This package provides two additional callbacks to the fastai learner :
- `ManifoldMixUp` which implements [ManifoldMixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf)
- `OutputMixUp` which implements a variant that does the mixup on the last viable layer only (this was shown to be more performant on a [benchmark](https://forums.fast.ai/t/mixup-data-augmentation/22764/72) and an independant [blogpost](https://medium.com/analytics-vidhya/better-result-with-mixup-at-final-layer-e9ba3a4a0c41))

**Warning:** As fastai V2 is still in its alpha stage, this code might become invalid due to internal changes.
If you notice any error of this kind, please report it. We should be able to fix it within 24 hours.

## Usage

To use manifold mixup, you can either pass the corresponding callback with the `cbs` argument of your learner 
or just call one of the `manifold_mixup` and `output_mixup` methods on your learner (for a minimal demonstration, see the [Demo notebook](https://github.com/nestordemeure/ManifoldMixup/blob/V2/Demo.ipynb)):

```python
learner = Learner(data, model).manifold_mixup()
learner.fit(8)
```

The `ManifoldMixUp` callback takes three parameters :
- `alpha=0.4` parameter of the beta law used for sampling the interpolation weight
- `use_input_mixup=True` do you want to apply mixup to the inputs
- `module_list=None` can be used to pass an explicit list of target modules

The `OutputMixUp` variant takes only the `alpha` parameters.

## Modules

### Which modules will be intrumented ?

`ManifoldMixUp` tries to establish a sensible list of modules on which to apply mixup:
- it uses a user provided `module_list` if possible
- otherwise it uses only the modules wrapped with `ManifoldMixupModule`
- if none are found, it defaults to modules with `Block` in their name (targetting mostly resblocks)
- finaly, if needed, it defaults to all modules that are not included in the `non_mixable_module_types` list

`OutputMixUp` is slightly different in that it will simply use the last layer that is neither a loss nor a softmax.

The `non_mixable_module_types` list contains mostly recurrent layers but you can add elements to it in order to define module classes that should not be used for mixup (*do not hesitate to create an issue or start a PR to add common modules to the default list*).

### A note on skip-connections / residual-blocks

`ManifoldMixUp` (this does not apply to `OutputMixUp`) is greatly degraded when applied *inside* a residual block.
This is due to the mixed-up values becoming incoherent with the output of the skip connection (which have not been mixed).

While this implementation is equiped to work around the problem for U-Net and ResNet like architectures, you might run into problems (negligeable improvements over the baseline) with other network structures.
In which case, the best way to apply manifold mixup would be to manually select the modules to be instrumented.

*For more unofficial fastai extensions, see the [Fastai Extensions Repository](https://github.com/nestordemeure/fastai-extensions-repository).*