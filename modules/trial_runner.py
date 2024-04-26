from modules.DES import des_classic
from modules.centroids import *
import numpy as np
import os
from dataclasses import dataclass, field
import multiprocessing
from copy import copy


RESULTS_FOLDER = './results'


@dataclass
class Trial:
    """Holds information needed do perform a single separated trial
    """
    f: object
    dim: int
    centroid: callable
    params: dict
    x_min: np.array
    m: float
    constraints: np.array
    repetition: int

    def __str__(self) -> str:
        return f"{self.f.name} @ dim={self.dim}, centr={self.centroid.__qualname__}, rep={self.repetition}"

    @classmethod
    def from_raw_f(cls, f_raw: object, dim: int, centroid: callable, rep: int, default_params: dict):
        """Creates a single proper trial

        Args:
            f_raw (object): function object, not initialized with any dimention
            dim (int): dimentionality
            centroid (callable): centroid function
            rep (int): repetition NUMBER (which repetion is this?)
            default_params (dict): default params
        """
        f = f_raw(dim)
        x_min, m = f.get_global_minimum(dim)

        params = copy(default_params)
        params["lower"] = f.input_domain.T[0]
        params["upper"] = f.input_domain.T[1]
        params["budget"] = params["lambda"] * 1000 if not "budget" in params or params["budget"] is None else params["budget"]

        constraints = f.input_domain.T[0]

        params_centroid = copy(params)
        params_centroid['centroid_fc'] = centroid

        return cls(
            f,
            dim,
            centroid,
            params_centroid,
            x_min,
            m,
            constraints,
            rep
        )


@dataclass
class TrialConfig:
    """Holds experiments to be run in parallel by the trial runner
    """
    experiments: list[Trial] = field(default_factory=list)

    @classmethod
    def from_product(cls, funcs: list[object], dims: list[int], centroids: list[callable], default_params: dict, repetitions: int):
        """Creates a config object, which constains a list of experiments which are all possible combinations of given functions, dimentions and centroids. Each combination is multiplied repetitions times.

        Args:
            funcs (list[object]): list of functions to consider
            dims (list[int]): list of dimentions
            centroids (list[callable]): list of centroid funcs
            default_params (dict): default parameters for DES
            repetitions (int): number of repetitions of each combination

        Returns:
            TrialConfig: the resulting trial config
        """
        config = cls()
        for f_raw in funcs:
            for dim in dims:
                # print("Config", f_raw.__qualname__, dim)
                for rep in range(repetitions):
                    for centroid in centroids:
                        try:
                            config.experiments.append(
                                Trial.from_raw_f(
                                    f_raw,
                                    dim,
                                    centroid,
                                    rep,
                                    default_params
                                ))
                        except Exception as e:
                            print(f"ERROR: {e}")
        return config


@dataclass
class TrialResult:
    """Holds trial results
    """
    trial: Trial
    logs: object

    def get_dump_path(self, results_folder: str = RESULTS_FOLDER):
        return f'{results_folder}/{self.get_descriptor()}'

    def get_descriptor(self):
        return f"{self.trial.centroid.__qualname__};{self.trial.f.name};{self.trial.dim};{self.trial.repetition}"

    def get_key_ommit_repetition(self):
        """Key for identifying experiments. Does not include the repetition number.
        key = (centroid name, function name, dimentionality)
        """
        return self.trial.centroid.__qualname__, self.trial.f.name, self.trial.dim

    @classmethod
    def load_from_trial(cls, results_folder: str, trial: Trial):
        data = np.load(cls.get_path_from_parts(results_folder, trial.centroid, trial.f, trial.dim, trial.repetition)+".npy", allow_pickle=True).tolist()
        return cls(trial, data)

    @staticmethod
    def get_path_from_parts(results_folder: str, centroid: callable, f: object, dim: int, repetition: int):
        """Creates a dump path from component parts to read from

        Args:
            centroid (callable): centroid function
            f (object): function object
            dim (int): dimentionality
            repetition (int): repetition number
        """
        descriptor = f"{centroid.__qualname__};{f.name};{dim};{repetition}"
        return f'{results_folder}/{descriptor}'


class TrialRunner:
    BASE_SALT: int = 999

    def __init__(self, trial_config: TrialConfig, max_procs: int = multiprocessing.cpu_count()) -> None:
        self._max_procs = min(multiprocessing.cpu_count(), max_procs)
        self._config = trial_config

    def run_all(self):
        try:
            os.mkdir(RESULTS_FOLDER)
        except FileExistsError as _:
            pass

        print(f"Running experiments on {self._max_procs} cores")
        with multiprocessing.Pool(self._max_procs) as pool:
            pool.map(self.run_one_experiment, self._config.experiments)

    @classmethod
    def run_one_experiment(cls, trial: Trial):
        print(f"Processing {trial}\n", end="")
        np.random.seed(cls.BASE_SALT + trial.repetition)
        result = des_classic(trial.constraints, trial.f, **trial.params)
        cls._save_result(TrialResult(trial, result))

    @classmethod
    def _save_result(cls, result: TrialResult):
        np.save(result.get_dump_path(), result.logs, allow_pickle=True)
