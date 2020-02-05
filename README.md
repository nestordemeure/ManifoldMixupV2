# Manifold Mixup

Unofficial implementation of [ManifoldMixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf) (Proceedings of ICML 19) for [fast.ai V1](https://docs.fast.ai/index.html) based on [Shivam Saboo](https://github.com/shivamsaboo17)'s [pytorch implementation](https://github.com/shivamsaboo17/ManifoldMixup) of manifold mixup, fastai's input mixup [implementation](https://docs.fast.ai/callbacks.mixup.html) plus some personnal improvements/variants.

This package provides two additional methods to the fastai learner :
- `.manifold_mixup()` which implements [ManifoldMixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf)
- `.output_mixup()` which implements a variant that does the mixup on the last viable layer only (this was shown to be more performant on a [benchmark](https://forums.fast.ai/t/mixup-data-augmentation/22764/64) and an independant [blogpost](https://medium.com/analytics-vidhya/better-result-with-mixup-at-final-layer-e9ba3a4a0c41))

## Usage

To use manifold mixup, you just need to call a method, either `manifold_mixup` or `output_mixup`, on your learner (for a minimal demonstration, see the [Demo notebook](https://github.com/nestordemeure/ManifoldMixup/blob/master/Demo.ipynb)):

```python
learner = Learner(data, model).manifold_mixup()
learner.fit(8)
```

The `manifold_mixup` method takes four parameters :
- `alpha=0.4` parameter of the beta law used for sampling the interpolation weight
- `use_input_mixup=True` do you want to apply mixup to the inputs
- `module_list=None` can be used to pass an explicit list of target modules
- `stack_y=True` do you want to perform the combinaison after the evaluation of the loss function (good for classification) or directly on the raw targets (good for regression)

The `output_mixup` variant takes only the `alpha` and `stack_y` parameters.

## Mixup compatible modules

`manifold_mixup` tries to establish a sensible list of modules on which to apply mixup:
- it uses a user provided `module_list` if possible
- otherwise it uses only the modules wrapped with `ManifoldMixupModule`
- if none are found, it defaults to modules with `Block` in their name (targetting mostly resblocks)
- finaly, if needed, it defaults to all modules that are not included in the `non_mixable_module_types` list

`output_mixup` is slightly different in that it will simply use the last layer that is neither a loss nor a softmax.

The `non_mixable_module_types` list contains mostly recurrent layers but you can add elements to it in order to define module classes that should not be used for mixup (*do not hesitate to create an issue or start a PR to add common modules to the default list*).

## Todo

This repository will be updated to [fast.ai V2](http://dev.fast.ai/) once it gets out of alpha stage.
In the meantime, I might create a dedicated branch.
