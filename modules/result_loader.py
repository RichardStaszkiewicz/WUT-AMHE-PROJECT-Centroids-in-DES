from modules.trial_runner import *
import numpy as np
from Python_Benchmark_Test_Optimization_Function_Single_Objective.pybenchfunction.function import (
    XinSheYang,
)
from modules.centroids import ALL_CENTROIDS
import matplotlib.pyplot as plt


class RelevantDataProcessor:
    def __init__(self, logs_foler: str, trials: list[Trial]) -> None:
        self._trials = trials
        self._logs_folder = logs_foler

    def _extraction(self):
        """Extracts population bestval stats, counts and total best values."""
        data_raw = {}
        print("Extracting relevant data")
        for trial in self._trials:
            try:
                trial_result = TrialResult.load_from_trial(self._logs_folder, trial)
                key = trial_result.get_key_ommit_repetition()

                if key not in data_raw:
                    data_raw[key] = {"bestVals": [], "counts": [], "value": []}
                # Save relevant data
                data_raw[key]["bestVals"].append(
                    np.squeeze(trial_result.logs["diagnostic"]["bestVal"])
                )
                data_raw[key]["counts"].append(trial_result.logs["counts"])
                data_raw[key]["value"].append(trial_result.logs["value"])
            except Exception as e:
                print(f"ERROR: {trial}\n {e}")
        return data_raw

    def _preprocessing(self, data_raw: dict):
        """Processes bestval statistics, counts and values into bestval, counts and values averaged over 5 runs, as well as total minium value reached and the count it was reached at

        Args:
            data_raw (dict): extracted data dict
        Returns:
            dict with all the fields
        """
        data_processed = {}
        # Process releavant data
        print("Preprocessing relevant data")
        for key in data_raw:
            try:
                # Save count and par statistics
                reps = len(data_raw[key]["counts"])

                min_count = 0  # count for finding the min_par TODO: MINIMALIZACJA?
                min_value = 1e9

                avg_count = sum(d["function"] for d in data_raw[key]["counts"]) / reps
                avg_value = sum(data_raw[key]["value"]) / reps

                best_vals_raw = data_raw[key]["bestVals"]
                max_length = max(len(pop) for pop in best_vals_raw)
                masked_vectors = np.ma.masked_all(
                    (len(best_vals_raw), max_length), dtype=np.float64
                )

                for i, vec in enumerate(best_vals_raw):
                    masked_vectors[i, : len(vec)] = vec

                avg_best_vals = np.ma.mean(masked_vectors, axis=0)
                std_best_vals = np.ma.std(masked_vectors, axis=0)
                assert len(avg_best_vals) == max_length

                for rep in range(reps):
                    if data_raw[key]["value"][rep] < min_value:
                        min_value = data_raw[key]["value"][rep]
                        min_count = data_raw[key]["counts"][rep]["function"]

                if key not in data_processed:
                    data_processed[key] = {}
                data_processed[key]["count_at_min_value"] = min_count
                data_processed[key]["total_min_value"] = min_value
                data_processed[key]["avg_count"] = avg_count
                data_processed[key]["avg_min_value"] = avg_value
                data_processed[key]["best_vals_avg"] = avg_best_vals
                data_processed[key]["best_vals_std"] = std_best_vals

            except Exception as e:
                print(f"ERROR: {key}\n{e}")

        return data_processed

    def get_extract_relevant_data_from_logs(self):
        data_raw = self._extraction()
        return self._preprocessing(data_raw)

    def extract_and_save_relevant_data(self, path: str):
        data_processed = self.get_extract_relevant_data_from_logs()
        print("Saving data")
        np.save(path, data_processed, allow_pickle=True)

    @staticmethod
    def load_prepared_relevant_data(path: str):
        """Loads previously processed relevant data, look at _preprocessing to see what fields are available"""
        return np.load(path, allow_pickle=True).tolist()


class RelevantDataStatistics:
    def __init__(self, relevant_data: dict) -> None:
        self._relevant_data = relevant_data

    def get_centroids_for_each_function_sorted_by(self, field: str):
        """Returns a dict with sorted names of centroids along with the used value

        ret[function name][dimentionality][i] = (used value, centroid name)

        Args:
            field (str): Which dict key should i use to sort the values?
        """
        result = {}
        for key in self._relevant_data:
            centroid_name, function_name, dimention = key

            if function_name not in result:
                result[function_name] = {}
            if dimention not in result[function_name]:
                result[function_name][dimention] = []

            result[function_name][dimention].append(
                (self._relevant_data[key][field], centroid_name)
            )

        for fun_name in result:
            for dim in result[fun_name]:
                result[fun_name][dim] = sorted(result[fun_name][dim])

        return result

    def filter_graph_data(self, constraints: tuple):
        """
        constraints:    touple ([str], [str], [int]) which defines the graph to build. It filters the trials to inclued in the research.
                        Setting it to None includes all observations of such category
        """
        filtered_data = dict()
        for tid in self._relevant_data.keys():
            x = [i is None for i in constraints]  # [1 0 1]
            if not sum(
                [0 if (x[i]) else tid[i] not in constraints[i] for i in range(len(x))]
            ):
                filtered_data[tid] = self._relevant_data[tid]
        return filtered_data

    def generate_convergence_curves_for_fc_x_dim(
        self, save="./results/graphics/convergence", verbose=False, every=(10, 50)
    ):
        """
        data - dict() the dictionary of trials named with the (centroid name, function name, dimensionality) convention
        """
        for fc in np.unique([x[1] for x in list(self._relevant_data.keys())]):
            for dim in np.unique([x[2] for x in list(self._relevant_data.keys())]):
                graph_data = self.filter_graph_data((None, [fc], [dim]))
                fig, ax = plt.subplots()
                for centroid, col, x in zip([x.__qualname__ for x in ALL_CENTROIDS], ['r', 'g', 'b', 'y', 'm'], list(range(5))):
                    try:
                        ax.errorbar(
                            list(range(
                                len(graph_data[(centroid, fc, dim)]["best_vals_avg"])
                            )),
                            graph_data[(centroid, fc, dim)]["best_vals_avg"],
                            fmt=col,
                            yerr=graph_data[(centroid, fc, dim)]["best_vals_std"],
                            label=f"{centroid}",
                            errorevery=(int(every[0] + x*0.2*every[1]), every[1])

                        )
                    except Exception as e:
                        if verbose:
                            print(f"could not find {centroid} in {fc, dim}... Error: {e}")
                ax.legend()
                fig.suptitle(f"{fc}, dim={dim}")
                ax.set_yscale('log')
                fig.savefig(f"{save}/{fc}-{dim}-convergence.png")
                fig.clear()
                plt.close()