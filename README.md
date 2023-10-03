# Master's Thesis: The Expressive Power of Variants of the Weisfeiler-Leman Algorithm applied to Graph Neural Networks
This is the repository containing the code for my masters thesis: "The Expressive Power of Variants of the Weisfeiler-Leman Algorithm applied to Graph Neural Networks" supervised by Prof. Christopher Morris.

## Installing all dependencies
To install all dependencies needed for the code, please install poetry as explained in poetrys [documentation](https://python-poetry.org/docs/).
In the home directory of the repository, execute:
```console
poetry install
```

If you want to use the kernel and neural baselines, you will need
- `eigen3`,
- `g++/gcc` 
- `pybind11`

When installing on MacOS, we suggest using [homebrew](https://brew.sh). For instance, installing `eigen3`, `g++` and `pybind11` is done by 
```console
brew install eigen
brew install gcc
brew install pybind11
```

## Running Kernel Baselines
In order to run the kernelp baselines, please navigate to the folder that contains all auxiliary methods and execute 
```console
cd code/main_methods/kernel_models
```

For linux users, run 
```console
g++ -O3 -shared -std=c++11 -fPIC `python3 -m pybind11 eigen --includes`  kernel_models.cpp src/*cpp -o ../kernel_models`python3-config --extension-suffix`
```

and MacOS users, use: 
```console
g++ -O3 -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes`  kernel_models.cpp src/*cpp -o ../kernel_models`python3-config --extension-suffix`
```

To run the script for the kernel baselines, execute: 
```console
cd ..
python3 main_kernel.py
```

## Running Neural Baselines

For all neural models, navigate to the neural models from the root directory by 
```console
cd code/main_methods/preprocessing
```

To integrate `pybind11`, for MacOS execute:
```console
g++ -O3 -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes`  preprocessing.cpp src/*cpp -o ../preprocessing`python3-config --extension-suffix`
``` 

For linux users, this becomes
```
g++ -O3 -shared -std=c++11 -fPIC `python3 -m pybind11 eigen --includes`  kernel_models.cpp src/*cpp -o ../kernel_models`python3-config --extension-suffix`
```

Navigate back to the main methods folder by ```cd ..```, that contains all main methods executing the GNN models.
For this simply run, e.g.,
```
python3 main_M-2-GNN.py
```