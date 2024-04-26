from modules.trial_runner import *
import Python_Benchmark_Test_Optimization_Function_Single_Objective.pybenchfunction as bench

DEFAULT_PARAMS = {
    "upper": np.array([-5, -23.3, 14, 11]),
    "lower": np.array([-101, -101, -101, -150]),
    "stopfitness": 1e-10,
    "lambda": 5,
    "time": 5,
    "centroid_fc": mean_centroid,
    "diag": True,
}


class TestTrials:
    def test_trial_init_from_raw(self):
        bench_funcs = bench.get_functions(None)[0:3]
        trial = Trial.from_raw_f(
            bench_funcs[0], 2, ALL_CENTROIDS[0], 0, DEFAULT_PARAMS
        )
        assert str(trial) == 'Ackley @ dim=2, centr=vanila_centroid, rep=0'

    def test_run_example_experiment(self, tmpdir):
        bench_funcs = bench.get_functions(None)[0:3]
        trial = Trial.from_raw_f(
            bench_funcs[0], 2, ALL_CENTROIDS[0], 0, DEFAULT_PARAMS
        )
        path = tmpdir.mkdir("res")
        runner = TrialRunner(TrialConfig([trial]), results_folder=path)
        runner.run_one_experiment(trial)


class TestConfig:
    def test_init_from_product(self):
        bench_funcs = bench.get_functions(None)

        config = TrialConfig.from_product(
            list(filter((lambda func: func.is_dim_compatible(50)), bench_funcs))[
                0:3],
            [2, 10, 50],
            ALL_CENTROIDS[0:3],
            DEFAULT_PARAMS,
            5
        )

        assert len(config.experiments) == 3*3*3*5
