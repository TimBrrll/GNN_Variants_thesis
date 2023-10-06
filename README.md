# Master's Thesis: The Expressive Power of Variants of the Weisfeiler-Leman Algorithm applied to Graph Neural Networks
This is the repository containing the code for my masters thesis: "The Expressive Power of Variants of the Weisfeiler-Leman Algorithm applied to Graph Neural Networks" supervised by Prof. Christopher Morris.

# Running by using Docker
As it is the simplest way to install all dependencies, we recommend deploying and running the code via Docker. We refer to the [Docker installation site](https://docs.docker.com/engine/install/) for further information.  
In the `Dockerfile` found at the root of this repository, you can adjust which main method you would like to run. This can be found in line 53 and can be adjusted freely before building the image.
In the home directory of this repository, you can build the image for the main method you would like to run, by executing: 
```console
docker build -t <your_tag> .
```
Inserting your own tag within the spaceholder will generate an image assigned to that tag. 
You can run the container via the following command: 

```console
docker run <your_tag>
```

# Running locally
This option involves more finetuning and adjusting of dependency-paths and is thus not recommended. 

## Installing all dependencies
To install all dependencies needed for the code, please install poetry as explained in poetry's [documentation](https://python-poetry.org/docs/).
In the home directory of the repository, execute:
```console
poetry install
```

If you want to use the kernel and neural baselines, it is crucial to install
- `eigen3`,
- `g++/gcc` 
- `pybind11`

When installing on MacOS, we suggest using [homebrew](https://brew.sh) for this. For instance, installing `eigen3`, `g++` and `pybind11` is done via
```console
brew install eigen
brew install gcc
brew install pybind11
```

## Running Kernel Baselines

The code and its path dependencies is set to run via `Docker`. Therefore you need to adjust some dependencies within the code for it to work locally.
As a result, open `code/main_methods/kernel_models/src/AuxiliaryMethods.cpp` and change the variable `path` to `path = ".";` in lines 33 and 175.
In order to run the kernel baselines, please navigate to the folder that contains all auxiliary methods from the root directory by 
```console
cd code/main_methods/kernel_models
```
To include the C++ sript as Python runnable code, as a linux user, run 
```console
g++ -O3 -shared -std=c++11 -fPIC `python3 -m pybind11 eigen --includes`  kernel_models.cpp src/*cpp -o ../kernel_models`python3-config --extension-suffix`
```

and as a MacOS users, use: 
```console
g++ -O3 -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes`  kernel_models.cpp src/*cpp -o ../kernel_models`python3-config --extension-suffix`
```

In order for this to work, you will need to adjust the paths to `Pybind11` and `Eigen` in some of the included files. 

To run the script for the kernel baselines by navigating to the `main_methods`-folder through: 
```console
cd ..
python3 main_kernel.py
```

## Running Neural Baselines

As mentioned previously, the code is programmed to work as a Docker container.
Therefore, for it to work locally, change the `path` variable in `code/main_methods/preprocessing/src/AuxiliaryMethods.cpp` to `path = ".";` in lines 42, 254 and 284.

For all neural models, navigate to the neural models from the root directory by 
```console
cd code/main_methods/preprocessing
```

To integrate `pybind11`, for linux platforms execute:
```
g++ -O3 -shared -std=c++11 -fPIC `python3 -m pybind11 eigen --includes`  kernel_models.cpp src/*cpp -o ../kernel_models`python3-config --extension-suffix`
```

For MacOS users, this is
```console
g++ -O3 -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes`  preprocessing.cpp src/*cpp -o ../preprocessing`python3-config --extension-suffix`
``` 


Navigate back to the main methods folder by ```cd ..```, that contains all main methods executing the GNN models. For this simply run, e.g.,
```
python3 main_M-2-GNN.py
```
