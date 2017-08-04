## Minimal neural networks library for educational purposes
[![Build Status](https://travis-ci.org/inejc/nnlib.svg?branch=master)](https://travis-ci.org/inejc/nnlib)
[![codecov](https://codecov.io/gh/inejc/nnlib/branch/master/graph/badge.svg)](https://codecov.io/gh/inejc/nnlib)
[![codebeat badge](https://codebeat.co/badges/6bb37624-a748-4c41-bfd0-a3a7e787f212)](https://codebeat.co/projects/github-com-inejc-nnlib-master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e1aa9012832442e8849a125ae917f1a0)](https://www.codacy.com/app/inejc/nnlib?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=inejc/nnlib&amp;utm_campaign=Badge_Grade)

A pure Python and NumPy implementation of a neural networks library developed for educational purposes.

It focuses on readability rather than speed and thus aims at providing an easily understandable toy code, as opposed to the real production grade libraries.

##### Gradient checks
All analytical solutions are gradient checked with a numerical method (a center divided difference formula).

### Currently implemented
* [Fully connected layer](nnlib/layers/fully_connected.py)
* [Softmax with cross entropy loss layer](nnlib/layers/softmax.py)
* [ReLU family activations](nnlib/layers/relu.py)
* [Vanilla stochastic gradient descent optimizer + variants](nnlib/optimizers/sgd.py)
* [L2 regularization](nnlib/regularizers.py)

### Future plans
* Dropout
* Batch normalization
* Different initializations
* Tanh, Sigmoid activations
* Convolutional layer
* Additional optimizers (RMSprop, Adagrad, Adam)

### Example usage
A computational graph that maintains the connectivity of the layers is called a `Model` ([see model.py](nnlib/model.py)).
```python
from nnlib import Model

model = Model()
```
New layers are added to the model with the `add()` method
```python
from nnlib.layers import FullyConnected, ReLU, SoftmaxWithCrossEntropy

model.add(FullyConnected(num_input_neurons=20, num_neurons=50))
model.add(ReLU())
model.add(FullyConnected(num_input_neurons=50, num_neurons=3))
model.add(SoftmaxWithCrossEntropy())
```
An optimizer needs to be passed to the `compile()` method to prepare everything for training
```python
from nnlib.optimizers import SGD

model.compile(SGD(lr=0.01))
```
To train the model call the `fit()` method and to make predictions call the `predict_proba()` or the `predict()` method
```python
model.fit(X_train, y_train, batch_size=32, num_epochs=100)

y_probs = model.predict_proba(X_test)
y_pred = model.predict(X_test)
```
### Resources
- [Stanford's CS231n](https://github.com/cs231n)
- [Keras](https://github.com/fchollet/keras)
- [Lasagne](https://github.com/Lasagne/Lasagne)
