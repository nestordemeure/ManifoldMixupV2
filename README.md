# Manifold Mixup

Unofficial implementation of [ManifoldMixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf) (Proceedings of ICML 19) for [fast.ai V1](https://docs.fast.ai/index.html) heavily based on [Shivam Saboo](https://github.com/shivamsaboo17)'s great [pytorch implementation](https://github.com/shivamsaboo17/ManifoldMixup) of manifold mixup and fastai's input mixup [implementation](https://docs.fast.ai/callbacks.mixup.html).

## Usage

Just call the `.manifold_mixup()` method of a learner (as you would with classical [mixup](https://docs.fast.ai/callbacks.mixup.html)):

```python
learner = Learner(data, model, metrics=[accuracy]).manifold_mixup()
learner.fit(8)
```

The `manifold_mixup` method takes tree parameters :
- `alpha=0.4` parameter of the beta law used for sampling the interpolation weight
- `mixup_all=True` do you want to apply mixup to any random module or only modules wrapped with a `ManifoldMixupModule`
- `use_input_mixup=True` do you want to apply mixup to the inputs

By default, any module can be used for mixup but, if you want to restrict it to a subset of modules, you can wrap them with a `ManifoldMixupModule`. 

## TODO

This repository will be updated to [fast.ai V2](http://dev.fast.ai/) once it gets out of alpha stage.
In the meantime, I might create a dedicated branch.

Use only callbacks and no module wrapping for the model to insure compatibility with fastai (currently the lr finder does not work).
This seem to be particularly suited for the interleaved variation.
