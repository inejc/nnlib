## Minimal neural networks library for educational purposes
[![Build Status](https://travis-ci.org/inejc/nnlib.svg?branch=master)](https://travis-ci.org/inejc/nnlib)
[![codecov](https://codecov.io/gh/inejc/nnlib/branch/master/graph/badge.svg)](https://codecov.io/gh/inejc/nnlib)

A pure Python and NumPy implementation of a neural networks library developed for educational purposes.

It focuses on readability rather than speed and thus aims at providing an easily understandable toy code, as opposed to the real production grade libraries.

##### Gradient checks
All analytical solutions are gradient checked with a numerical method (a center divided difference formula).

### Currently implemented
* [Fully connected layer] (nnlib/layers/fully_connected.py)
* [Softmax with cross entropy loss layer] (nnlib/layers/softmax.py)
* [ReLU family activations] (nnlib/layers/relu.py)
* [Stochastic gradient descent optimizer] (nnlib/optimizers/sgd.py)

### Future plans
* Dropout
* Batch normalization
* Different initializations
* L2 regularization
* Tanh, Sigmoid activations
* Convolutional layer
* Additional optimizers (SGD with momentum, RMSprop, Adagrad, Adam)

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
