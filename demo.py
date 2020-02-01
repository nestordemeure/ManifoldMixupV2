"Short demo inspired by https://docs.fast.ai/callbacks.mixup.html"
from fastai.vision import *
from manifold_mixup import *

# gets the data
path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)

# manifold mixup
model = simple_cnn((3,16,16,2))
learn = Learner(data, model, metrics=[accuracy]).manifold_mixup()
learn.fit(8)
learn.recorder.plot_losses()

# no mixup
model = models.resnet18()# simple_cnn((3,16,16,2))
learn = Learner(data, model, metrics=[accuracy])
learn.fit(8)
learn.recorder.plot_losses()
# 0.987242 accuracy, 1s per iter

# input mixup
model = simple_cnn((3,16,16,2))
learn = Learner(data, model, metrics=[accuracy]).mixup()
learn.fit(8)
learn.recorder.plot_losses()
# 0.992640 accuracy, 1s per iter
