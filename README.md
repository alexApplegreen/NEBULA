
<img src="https://github.com/user-attachments/assets/f489a7ea-0b4c-4a2a-8cb6-0326dade6a80" height="500px">

# NEBULA (Neural Error Bit Upset and Learning Adaptation)
NEBULA is a library providing tools for testing out the impact of radiation-induced biterrors
on the performance of neural networks.
There are 2 main features:
- static error injection
- error injection during training

For static errorinjection the `injector` class provides a multithreaded implementation
leveraging the multiprocessing library to time-efficiently alter a potentially large number of weights
inside the model.


For Fault Aware Training applications, the `TrainingInjector` class can be used to inject biterrors into
the model during training. The Traininginjector will add a pertubating layer to a given model which
is inactive during inference, but will add statistic noise in the form of bit errors to the networks weights.


## Setup
create virtual env (optional but recommended):
```(bash)
python3 -m venv venv
```

activate venv:
```(bash)
source venv/bin/activate
```

install NEBULA locally:
```(bash)
python3 -m pip install -e .
```

execute unittests:
```(bash)
python3 -m unittest discover -s test
```

### Logging
NEBULA uses the python logging library. The default log level is INFO.
The log level can be set by:
```bash
export NEBULA_LOG_LEVEL=DEBUG
```


### Samples
There are some usage examples in the samples folder.
The benchmark.py file executes one injection in every weight of every layer of the NN and prints out the runtime
The benchmark can be executed with (assuming inside the samples folder):

```bash
python3 benchmark.py
```

inside the mnist_example.py the framework is used to inject errors into an existing mnist network.
The sample is implemented such that a model can be read in via a .h5 file, also the images to handwritten
numbers must be supplied by the user. The images must have dimensions of 28x28 pixels and be
binary greyscaled.
