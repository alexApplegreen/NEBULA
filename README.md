# NEBULA (Neural Error Bit Upset and Learning Adaptation)

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
