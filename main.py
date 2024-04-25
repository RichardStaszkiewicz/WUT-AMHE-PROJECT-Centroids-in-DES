import Python_Benchmark_Test_Optimization_Function_Single_Objective.pybenchfunction as bench
from modules.trial_runner import *
from modules.default_config import get_default_config

if __name__ == "__main__":
    config = get_default_config()

    runner = TrialRunner(config)
    runner.run_all()
