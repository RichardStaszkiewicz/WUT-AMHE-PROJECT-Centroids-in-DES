import Python_Benchmark_Test_Optimization_Function_Single_Objective.pybenchfunction as bench
from modules.trial_runner import *

default_params = {
    "upper": np.array([101, 101]),
    "lower": np.array([-101, -101]),
    "stopfitness": 1e-9,
    "lambda": 20,
    "diag": True,
    "time": 0.5
}
repetitions = 5
dims = [10, 50, 100]

def filter_func(func):
    return func.is_dim_compatible(max(dims))

n_dimentional_funcs = list(filter(
        filter_func,
        bench.get_functions(None)
))

config = TrialConfig.from_product(
    n_dimentional_funcs,
    dims,
    ALL_CENTROIDS,
    default_params,
    repetitions,
)
print(f"Config prepared, n_experiments={len(config.experiments)}")

runner = TrialRunner(config)
runner.run_all()