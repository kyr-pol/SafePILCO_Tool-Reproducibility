# SafePILCO_Tool-Reproducibility
This repo reproduces the experiments with SafePILCO, for the tool paper submitted to QEST 2020.

The main [PILCO](https://github.com/nrontsis/PILCO) library needs to be installed first. 
This package provides the exact hyperparameters used for the paper and includes scripts to run multiple random seeds, save results, 
and process the data to obtain the plots and tables of results presented in the paper.

## Installation
To install PILCO in a clean conda environment:
- download from [git](https://github.com/nrontsis/PILCO)
- Create a new Python 3.7 env:
```
conda create - -name pilco python=3.7 pip
```
- Install requirements (after activating the conda environment):
```
pip install -r requirements
```
- Install the package itself by running `python setup.py develop`.

To install this package, clone and run `python setup.py install`. Seaborn is used for plotting, so install this via pip too.

## Experiments and further dependencies
Several experiments use mujoco enviromnets through OpenAI gym. For instructions on installing mujoco 
(which is a proprietary library but free trials are available) see [mujoco_py](https://github.com/openai/mujoco-py).

The Building Automation experiments use Matlab, and to call Matlab from Python the matlab engine for Python is used, as provided by Mathworks.
See [here](https://uk.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) for further insructions.

## Running the experiments
### Plain Pilco
Experiments for the inverted pendulum, mountain car, pendulum swing up, and double inverted pendulum tasks are run by the script `PlainPilco/experiments.py`.
We note here that running all experiments sequentially for 10 random seeds, as is the default, can take a long time so feel free to 
adjust the `number_of_random_seeds` variable as needed. Roughly each run of the algorithm, on an early 2015 MacBook Pro takes:
- < 30 minutes for the mountain car and inverted pendulum environments
- 1-2 hours for the pendulum swing up 
- several hours (3-5) for the double inverted pendulum task.

Results are saved in the `PlainPilco/results` folder.

The swimmer experiments are run separately by the script `PlainPilco/swimmer_experiments.sh`. 
We run this on a server with an Nvidia Tesla v100 graphics card and each run took about an hour.

After the experiments are run, results can be plotted by running `PlainPilco/post_process.py`. 
The plots show the performance of a random policy at each task too, which can be estimated by running `PlainPILCO/get_random_baselines.py`.
Plots appear one by one on the desktop and are saved in `PlainPilco/plots/`. 
These correspond to Figure 3 from the QEST paper.

### Safe Pilco
Three experiments are run, logged and the output data are post processed here, with all three actions happening independently for each one.

To run the safe cars scenario experiments use `SafePilco/linear_cars/experiments.py` and `SafePilco/linear_cars/post_process.py`
to calculate the relevant statistics.

The environment for the BAS experiments comes from [this](https://gitlab.com/natchi92/BASBenchmarks) Matlab repository.
It has to be cloned by the user, and the path to its source folder should be given as an input argument to
`SafePilco/BAS_experiments/experiments.sh`, so the command that runs the BAS experiments should look like: `./experiments.sh /Users/XXXX/BASBenchmarks/src`. To post process the results, run `SafePilco/BAS_experiments/post_process.py`.

Similarly for the same swimmer task, `SafePilco/safe_swimmer/experiments.sh` and `SafePilco/safe_swimmer/post_process.py` run and analyse the experiments.

The post process scripts print in the standard output the statistics that are reported in Table 2 of the paper.
Experiments for the safe swimmer can take a long time to run, so using a GPU or cloud compute is a good idea, 
at least if the experiments are to be repeated for multiple random seeds.

## Some notes and troubleshooting 
- On Linux, the oct2py installation might fail through pip, running apt-get install octave should fix it.
- When running the swimmer experiment on a MacBook Pro, a tensorflow related error can occur that looks like 
what is described [here](https://github.com/tensorflow/tensorflow/issues/23780).
What fixed it for us was disabling the tensorflow meta-optimiser, adding 
`tf.config.optimizer.set_experimental_options({"disable_meta_optimizer":True})` before any optimisation run in the swimmer experiment.
- Experiments that need mujoco: inverted pendulum, double pendulum, swimmer and safe swimmer.




