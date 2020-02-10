"Short demo inspired by http://dev.fast.ai/callback.mixup"
from fastai2.vision.all import *
from manifold_mixup import *

# gets the data
path = untar_data(URLs.MNIST_TINY)
items = get_image_files(path)
datasets = Datasets(items, tfms=[PILImageBW.create, [parent_label, Categorize]], splits=GrandparentSplitter()(items))
databunch = datasets.dataloaders(after_item=[ToTensor(), IntToFloatTensor()])

# model definition
def conv(ni, nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)
# TODO doable with a one liner ? fastai2.layers.SimpleCNN((3,16,16,2))
def simple_cnn():
    return nn.Sequential(conv(1, 3), nn.BatchNorm2d(3), nn.ReLU(inplace=False),
                         conv(3, 16), nn.BatchNorm2d(16), nn.ReLU(inplace=False),
                         conv(16,16), nn.BatchNorm2d(16), nn.ReLU(inplace=False),
                         conv(16, 2), nn.BatchNorm2d(2), Flatten() )

# no mixup
model = simple_cnn()
learn = Learner(databunch, model, metrics=accuracy)
learn.fit(8)
learn.recorder.plot_loss()

# input mixup
model = simple_cnn()
learn = Learner(databunch, model, metrics=accuracy, cbs=MixUp())
learn.fit(8)
learn.recorder.plot_loss()

# manifold mixup
model = simple_cnn()
learn = Learner(databunch, model, metrics=accuracy, cbs=ManifoldMixup(alpha=1.))
learn.fit(8)
learn.recorder.plot_loss()

# output mixup
model = simple_cnn()
learn = Learner(databunch, model, metrics=accuracy, cbs=OutputMixup())
learn.fit(8)
learn.recorder.plot_loss()
