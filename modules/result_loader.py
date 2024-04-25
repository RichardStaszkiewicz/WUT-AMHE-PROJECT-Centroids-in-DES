from modules.trial_runner import *
import numpy as np
from Python_Benchmark_Test_Optimization_Function_Single_Objective.pybenchfunction.function import XinSheYang
from modules.centroids import ALL_CENTROIDS


def get_result(processed_results_folder):
    path = TrialResult.get_path_from_parts(
        processed_results_folder,
        ALL_CENTROIDS[0],
        XinSheYang(10),
        10, 0
    )+".npy"
    log = np.load(path, allow_pickle=True)
    return log


def extract_bestVals_averages_means_and_save(processed_results_folder: str, save_folder: str, trials: list[Trial]):
    data_processed = {}
    for trial in trials:
        try:
            # Load saved data
            trial_result = TrialResult.load_from_trial(processed_results_folder, trial)
            # print(f"Loading {trial}")

            key = trial_result.get_key_ommit_repetition()
            if key not in data_processed:
                data_processed[key] = {
                    "bestVals": [],
                    "counts": [],
                    "par": []
                }
            # Save relevant data
            data_processed[key]["bestVals"].append(trial_result.logs["diagnostic"]["bestVal"])
            data_processed[key]["counts"].append(trial_result.logs["counts"])
            data_processed[key]["par"].append(trial_result.logs["par"])
        except Exception as e:
            print(f"ERROR: {trial}\n {e}")

    # Process releavant data
    for key in data_processed:
        try:
            # Save count and par statistics
            reps = len(data_processed[key]["counts"])
            min_count = 0  # count for finding the min_par TODO: MINIMALIZACJA?
            avg_count = sum(data_processed[key]["counts"])/reps
            min_par = 1e9
            avg_par = sum(data_processed[key]["par"])/reps

            # TODO: masked array numpy mean!!!

            for rep in range(reps):
                if data_processed[key]["par"][rep] < min_par:
                    min_par = data_processed[key]["par"][rep]
                    min_count = data_processed[key]["counts"][rep]

            data_processed.pop("counts")
            data_processed.pop("par")
            data_processed["min_count"] = min_count
            data_processed["min_par"] = min_par
            data_processed["avg_count"] = avg_count
            data_processed["avg_par"] = avg_par

        except Exception as e:
            print(f"ERROR: {key}\n{e}")

    np.save(f"{save_folder}/averages.npy", data_processed, allow_pickle=True)
    # return data_processed


def get_preprocessed_bestvals(save_folder: str):
    return np.load(f"{save_folder}/averages.npy", allow_pickle=True).tolist()
