# Manifold Mixup V2

Unofficial implementation of [ManifoldMixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf) (Proceedings of ICML 19) for [fast.ai V2](http://dev.fast.ai/) based on [Shivam Saboo](https://github.com/shivamsaboo17)'s [pytorch implementation](https://github.com/shivamsaboo17/ManifoldMixup) of manifold mixup, fastai's input mixup [implementation](https://dev.fast.ai/callback.mixup) plus some personnal improvements/variants.

This package provides two additional callbacks to the fastai learner :
- `ManifoldMixup` which implements [ManifoldMixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf)
- `OutputMixup` which implements a variant that does the mixup only on the output of the last layer (this was shown to be more performant on a [benchmark](https://forums.fast.ai/t/mixup-data-augmentation/22764/72) and an independant [blogpost](https://medium.com/analytics-vidhya/better-result-with-mixup-at-final-layer-e9ba3a4a0c41))

**Warning:** As fastai V2 is still in its alpha stage, this code might become invalid due to internal changes.
If you notice any error of this kind, please report it. We should be able to fix it within 24 hours.

## Usage

To use manifold mixup, you need to pass the corresponding callback to the `cbs` argument of your learner (for a minimal demonstration, see the [Demo notebook](https://github.com/nestordemeure/ManifoldMixupV2/blob/master/Demo.ipynb)):

```python
learner = Learner(data, model, cbs=ManifoldMixup())
learner.fit(8)
```

The `ManifoldMixup` callback takes three parameters :
- `alpha=0.4` parameter of the beta law used for sampling the interpolation weight
- `use_input_mixup=True` do you want to apply mixup to the inputs
- `module_list=None` can be used to pass an explicit list of target modules

The `OutputMixup` variant takes only the `alpha` parameters.

## Notes

### Which modules will be intrumented by ManifoldMixup ?

`ManifoldMixup` tries to establish a sensible list of modules on which to apply mixup:
- it uses a user provided `module_list` if possible
- otherwise it uses only the modules wrapped with `ManifoldMixupModule`
- if none are found, it defaults to modules with `Block` in their name (targetting mostly resblocks)
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
