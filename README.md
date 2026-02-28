# Modeling Chaucer's Meter

## Initial set-up

After cloning this repository, change into the new directory and invoke:

```
python3 -m venv local
source local/bin/activate
pip install --group all
cp custom.py.example custom.py
```

Thereafter, you can leave the environment with:

```
deactivate
```

and re-enter the environment with:

```
source local/bin/activate
```

## Running experiments

While within the environment, invoking:

```
steamroller -n
```

will print the commands that would be run to perform all defined experiments. Removing "-n" actually runs the commands.

## Defining experiments

Experiments are described in the `SConstruct` file, in terms of *build rules* each implemented as a Python script under `scripts/`, and how they are strung together. Variables can be overridden locally by adding or changing entries in the `custom.py` file (which is *not* under version control).
