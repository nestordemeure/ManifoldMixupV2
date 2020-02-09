"Short demo inspired by https://docs.fast.ai/callbacks.mixup.html"
from fastai.vision import *
from manifold_mixup import *

# gets the data
path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)

# no mixup
model = simple_cnn((3,16,16,2))
learn = Learner(data, model, metrics=[accuracy])
learn.fit(8)
learn.recorder.plot_losses()

# input mixup
model = simple_cnn((3,16,16,2))
learn = Learner(data, model, metrics=[accuracy]).mixup()
learn.fit(8)
learn.recorder.plot_losses()

# manifold mixup
model = simple_cnn((3,16,16,2))
learn = Learner(data, model, metrics=[accuracy]).manifold_mixup()
learn.fit(8)
learn.recorder.plot_losses()

# output mixup
model = simple_cnn((3,16,16,2))
learn = Learner(data, model, metrics=[accuracy]).output_mixup()
learn.fit(8)
learn.recorder.plot_losses()
