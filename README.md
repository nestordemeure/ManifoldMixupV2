# Manifold Mixup

Unofficial implementation of [ManifoldMixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf) (Proceedings of ICML 19) for [fast.ai V1](https://docs.fast.ai/index.html) based on [Shivam Saboo](https://github.com/shivamsaboo17)'s [pytorch implementation](https://github.com/shivamsaboo17/ManifoldMixup) of manifold mixup, fastai's input mixup [implementation](https://docs.fast.ai/callbacks.mixup.html) plus some personnal improvements/variants.

This package provides two additional methods to the fastai learner :
- `.manifold_mixup()` which implements [ManifoldMixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf)
- `.output_mixup()` which implements a variant that does the mixup on the last viable layer only (this was shown to be more performant on a [benchmark](https://forums.fast.ai/t/mixup-data-augmentation/22764/64))

## Usage

To use manifold mixup, you just need to call the `manifold_mixup` method on your learner (for a minimal demonstration, see the [Demo notebook](https://github.com/nestordemeure/ManifoldMixup/blob/master/Demo.ipynb)):

```python
learner = Learner(data, model).manifold_mixup()
learner.fit(8)
```

The `manifold_mixup` method takes five parameters :
- `alpha=0.4` parameter of the beta law used for sampling the interpolation weight
- `use_input_mixup=True` do you want to apply mixup to the inputs
- `module_list=None` can be used to pass an explicit list of target modules.
- `stack_y=True` do you want to perform the combinaison after the evaluation of the loss function (good for classification) or directly on the raw targets (good for regression).

The `output_mixup` variant takes only the `alpha`, and `stack_y` parameters.

## Mixup compatible modules

By default most modules can be used for mixup, notable exceptions include `Batchnorm` layers and most recurent layers. 
You can add classes to the `non_mixable_module_types` list in order to define module classes that should not be used for mixup (*do not hesitate to create an issue or start a PR to add common modules to the default list*).

If you want to target only a subset of the modules used in your model, you can either wrap them with a `ManifoldMixupModule` and set `use_only_mixup_modules` to `True` or pass them directly with the `module_list` parameter.

## Todo

This repository will be updated to [fast.ai V2](http://dev.fast.ai/) once it gets out of alpha stage.
In the meantime, I might create a dedicated branch.
