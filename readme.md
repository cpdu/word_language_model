This is a class-based version of [pytorch official language model example](https://github.com/pytorch/examples/tree/master/word_language_model) following the paper [Extensions of recurrent neural network language model.2011.Tomáš Mikolov](http://ieeexplore.ieee.org/document/5947611/). In the original model, the last layer is full connected, so its size grows with the vocabulary size and slows down the training. You can use --cls option to train your model in class-based mode.