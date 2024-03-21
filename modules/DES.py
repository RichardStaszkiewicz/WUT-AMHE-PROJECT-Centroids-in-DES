import numpy as np
import random
import time
import pandas as pd


def des_classic(par, fn, lower=None, upper=None, **kwargs):
    """
    Differential Evolution Strategy with classic control parameters.

    Parameters:
    - par (list): Initial parameter values.
    - fn (function): Objective function to be minimized.
    - lower (float or list, optional): Lower bounds for each parameter. If None, default is -100 for each parameter.
    - upper (float or list, optional): Upper bounds for each parameter. If None, default is 100 for each parameter.
    - **kwargs (dict): Additional control parameters for the algorithm.

    Returns:
    - dict: Result dictionary containing the following keys:
        - "par": List of the best parameter values found.
        - "value": The objective function value corresponding to the best parameters.
        - "counts": Dictionary containing the count of function evaluations (`"function"`).
        - "resets": Number of restarts performed.
        - "convergence": Flag indicating convergence (1 if reached maximum iterations, 0 otherwise).
        - "time": Total execution time in seconds.
        - "message": String message indicating the reason for termination.
        - "diagnostic": Dictionary containing diagnostic logs if enabled.

    Notes:
    - This implementation uses a classic differential evolution strategy with control parameters.
    - The objective function `fn` should take a parameter vector and return a scalar value to be minimized.
    - Control parameters can be passed through the `**kwargs` dictionary.
    """

    def control_param(name, default):
        """
        Get the value of a control parameter from keyword arguments or return the default.

        Parameters:
        - name (str): Name of the control parameter.
        - default: Default value to return if the parameter is not present in the keyword arguments.

        Returns:
        - Value of the specified control parameter if present, otherwise the default value.

        Notes:
        - This function is used to retrieve a control parameter from keyword arguments (`kwargs`).
        - If the parameter with the given name is present in the `kwargs`, its value is returned.
        - If the parameter is not present, the default value is returned.
        """
        v = kwargs[name] if name in kwargs.keys() else None
        return v if v is not None else default

    def sample_from_history(history, history_sample, lmbda):
        """
        Sample indices from a historical population based on given history and sample indices.

        Parameters:
        - history (list): List containing historical populations.
        - history_sample (list): List of indices indicating which historical populations to sample from.
        - lmbda (int): Number of samples to generate.

        Returns:
        - list: List of sampled indices from historical populations.

        Notes:
        - This function is used to randomly sample indices from historical populations based on provided history
        and sample indices.
        - It returns a list of 'lmbda' indices, each representing a random index from a selected historical population.
        """
        ret = []
        for _ in range(lmbda):
            ret.append(random.randint(0, len(history[history_sample[_]].T) - 1))
        return ret

    def delete_infs_nans(x):
        """
        Replace NaN and Inf values in the given array with the maximum finite float value.

        Parameters:
        - x (numpy.ndarray): Input array containing numerical values.

        Returns:
        - numpy.ndarray: Array with NaN and Inf values replaced by the maximum finite float value.

        Notes:
        - This function is used to handle NaN (Not a Number) and Inf (Infinity) values in an input array.
        - It replaces all NaN values with the maximum finite float value.
        - It replaces all Inf values with the maximum finite float value.

        Examples:
        ```python
        # Example Usage:
        arr = np.array([1.0, 2.0, np.nan, np.inf, -np.inf])
        result = delete_infs_nans(arr)
        print(result)
        # Output: array([ 1.,  2., 1.7976931348623157e+308, 1.7976931348623157e+308, 1.7976931348623157e+308])
        ```
        """
        x[np.isnan(x)] = np.finfo(float).max
        x[np.isinf(x)] = np.finfo(float).max
        return x

    def fn_(x):
        """
        Evaluate the objective function for a given input vector within specified bounds.

        Parameters:
        - x (numpy.ndarray): Input vector to be evaluated.

        Returns:
        - float: Objective function value for the input vector or maximum finite float value if outside bounds.

        Notes:
        - This function evaluates the objective function 'fn' for a given input vector 'x'.
        - It checks whether all elements of 'x' are within the specified lower and upper bounds.
        - If the input vector is within bounds, it increments the global counteval variable and returns the
        result of evaluating 'fn' on the input vector.
        - If the input vector is outside bounds, it returns the maximum finite float value.

        Examples:
        ```python
        # Example Usage:
        result = fn_(np.array([1.0, 2.0, 3.0]))
        print(result)
        ```
        """
        if all(x >= lower) and all(x <= upper):
            nonlocal counteval
            counteval += 1
            return fn(x)
        else:
            return np.finfo(float).max

    def fn_l(P):
        """
        Evaluate the objective function for a population matrix within specified bounds.

        Parameters:
        - P (numpy.ndarray): Population matrix where each column represents an input vector.

        Returns:
        - numpy.ndarray: Array of objective function values for the input vectors or maximum finite float values if outside bounds.

        Notes:
        - This function evaluates the objective function 'fn_' for a population matrix 'P'.
        - If 'P' is a 2D array (matrix), it applies 'fn_' to each column (individual) of the matrix.
        - It checks whether the cumulative counteval plus the number of columns in 'P' is within the specified budget.
        - If within budget, it returns an array of objective function values for each individual.
        - If outside budget, it returns an array with maximum finite float values for individuals beyond the budget.

        Examples:
        ```python
        # Example Usage:
        result = fn_l(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        print(result)
        ```
        """
        if P.ndim > 1:
            if counteval + P.shape[1] <= budget:
                return np.apply_along_axis(fn_, 0, P)
            else:
                bud_left = budget - counteval
                ret = np.zeros(bud_left)
                if bud_left > 0:
                    for i in range(bud_left):
                        ret[i] = fn_(P[:, i])
                return np.concatenate(
                    (ret, np.repeat(np.finfo(float).max, P.shape[1] - bud_left))
                )
        else:
            if counteval < budget:
                return fn_(P)
            else:
                return np.finfo(float).max

    def fn_d(P, P_repaired, fitness):
        """
        Evaluate the fitness of a population matrix considering repaired individuals.

        Parameters:
        - P (numpy.ndarray): Original population matrix where each column represents an input vector.
        - P_repaired (numpy.ndarray): Repaired population matrix with the same structure as 'P'.
        - fitness (numpy.ndarray): Original fitness values corresponding to the individuals in 'P'.

        Returns:
        - numpy.ndarray: Array of fitness values considering repaired individuals.

        Notes:
        - This function calculates the fitness of a population matrix 'P' while considering repaired individuals.
        - It checks for inconsistencies between the original and repaired populations, applying penalties if repairs were made.
        - If both 'P' and 'P_repaired' are 2D arrays (matrices), it calculates the fitness for each corresponding pair of individuals.
        - If inconsistencies are detected, penalties are applied to the fitness values accordingly.

        Examples:
        ```python
        # Example Usage:
        result = fn_d(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                    np.array([[1.5, 2.0, 3.0], [4.0, 5.5, 6.0]]),
                    np.array([10.0, 15.0]))
        print(result)
        ```
        """
        P = delete_infs_nans(P)
        P_repaired = delete_infs_nans(P_repaired)

        if P.ndim > 1 and P_repaired.ndim > 1:
            repaired_ind = np.all(P != P_repaired, axis=0)
            P_fit = fitness.copy()
            vec_dist = np.sum((P - P_repaired) ** 2, axis=0)
            P_fit[repaired_ind] = worst_fit + vec_dist[repaired_ind]
            P_fit = delete_infs_nans(P_fit)
            return P_fit
        else:
            P_fit = fitness.copy()
            if not np.array_equal(P, P_repaired):
                P_fit = worst_fit + np.sum((P - P_repaired) ** 2)
                P_fit = delete_infs_nans(P_fit)
            return P_fit

    def bounce_back_boundary2(x):
        """
        Bounce back the elements of a vector within specified lower and upper bounds.

        Parameters:
        - x (numpy.ndarray): Input vector to be bounded.

        Returns:
        - numpy.ndarray: Bounded vector after applying bounce-back conditions.

        Notes:
        - This function ensures that each element of the input vector 'x' is within the specified lower and upper bounds.
        - If any element in 'x' exceeds the bounds, it is adjusted by bouncing back from the boundaries.
        - The bounce-back adjustment involves determining the excess distance beyond the bounds and applying it in the opposite direction.

        Examples:
        ```python
        # Example Usage:
        result = bounce_back_boundary2(np.array([1.5, 0.5, 2.5]), lower=np.array([1.0, 0.0, 2.0]), upper=np.array([2.0, 1.0, 3.0]))
        print(result)
        ```
        """
        if all(x >= lower) and all(x <= upper):
            return x
        elif any(x < lower):
            for i in np.where(x < lower)[0]:
                x[i] = lower[i] + abs(lower[i] - x[i]) % (upper[i] - lower[i])
        elif any(x > upper):
            for i in np.where(x > upper)[0]:
                x[i] = upper[i] - abs(upper[i] - x[i]) % (upper[i] - lower[i])
        x = delete_infs_nans(x)
        return bounce_back_boundary2(x)

    N = len(par)
    if lower is None:
        lower = np.full(N, -100)
    elif isinstance(lower, (int, float)):
        lower = np.full(N, lower)

    if upper is None:
        upper = np.full(N, 100)
    elif isinstance(upper, (int, float)):
        upper = np.full(N, upper)

    # Algorithm parameters
    Ft = control_param("Ft", 1)
    initFt = control_param("initFt", 1)
    stopfitness = control_param("stopfitness", -np.inf)
    budget = control_param("budget", 10000 * N)
    initlambda = control_param("lambda", 4 * N)
    lambda_ = initlambda
    mu = np.int64(control_param("mu", np.floor(lambda_ / 2)))
    weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    weights_sum_s = np.sum(weights**2)
    mueff = control_param("mueff", (np.sum(weights) ** 2) / weights_sum_s)
    cc = control_param("ccum", mu / (mu + 2))
    path_length = control_param("pathLength", 6)
    cp = control_param("cp", 1 / np.sqrt(N))
    maxiter = control_param("maxit", np.floor(budget / (lambda_ + 1)).astype(np.int64))
    maxtime = control_param("time", np.Inf)
    c_Ft = control_param("c_Ft", 0)
    path_ratio = control_param("pathRatio", np.sqrt(path_length))
    hist_size = np.int64(control_param("history", np.ceil(6 + np.ceil(3 * np.sqrt(N)))))
    Ft_scale = ((mueff + 2) / (N + mueff + 3)) / (
        1
        + 2 * np.maximum(0, np.sqrt((mueff - 1) / (N + 1)) - 1)
        + (mueff + 2) / (N + mueff + 3)
    )
    tol = control_param("tol", 1e-12)
    counteval = 0
    sqrt_N = np.sqrt(N)

    log_all = control_param("diag", False)
    log_Ft = control_param("diag.Ft", log_all)
    log_value = control_param("diag.value", log_all)
    log_mean = control_param("diag.mean", log_all)
    log_mean_cord = control_param("diag.meanCords", log_all)
    log_pop = control_param("diag.pop", log_all)
    log_best_val = control_param("diag.bestVal", log_all)
    log_worst_val = control_param("diag.worstVal", log_all)
    log_eigen = control_param("diag.eigen", log_all)

    Lamarckism = control_param("Lamarckism", False)

    # Safety checks
    assert len(upper) == N
    assert len(lower) == N
    assert np.all(lower < upper)

    # Initialize variables
    best_fit = np.inf
    best_par = None
    worst_fit = None
    last_restart = 0
    restart_length = 0
    restart_number = 0

    # Preallocate logging structures
    if log_Ft:
        Ft_log = np.zeros(1)
    if log_value:
        value_log = np.zeros((0, lambda_))
    if log_mean:
        mean_log = np.zeros(1)
    if log_mean_cord:
        mean_cords_log = np.zeros(N)
    if log_pop:
        pop_log = np.zeros((N, lambda_, maxiter))
    if log_best_val:
        best_val_log = np.array(np.Infinity)
    if log_worst_val:
        worst_val_log = np.zeros(1)
    if log_eigen:
        eigen_log = np.zeros(lambda_)

    # Allocate buffers
    d_mean = np.zeros((N, hist_size))
    Ft_history = np.zeros(hist_size)
    pc = np.zeros((N, hist_size))

    # Initialize internal strategy parameters
    msg = None
    restart_number = -1
    time_start = time.time()

    while counteval < budget and ((time.time() - time_start) < maxtime):
        restart_number += 1
        mu = np.int64(np.floor(lambda_ / 2))
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        weights_pop = np.log(lambda_ + 1) - np.log(np.arange(1, lambda_ + 1))
        weights_pop /= np.sum(weights_pop)
        hist_head = 0
        iter_ = 0
        history = [None] * hist_size
        Ft = initFt

        # Create first population
        population = np.random.uniform(0.8 * lower, 0.8 * upper, size=(lambda_, N)).T

        cum_mean = (upper + lower) / 2
        population_repaired = np.apply_along_axis(bounce_back_boundary2, 0, population)

        if Lamarckism:
            population = population_repaired

        selection = np.zeros(mu, dtype=int)
        selected_points = np.zeros((N, mu))
        fitness = fn_l(population)
        old_mean = np.zeros(N)
        new_mean = np.copy(par)
        limit = 0
        worst_fit = np.max(fitness)

        ############################POI##########################
        #########################################################
        # Store population and selection means
        pop_mean = np.matmul(population, weights_pop)
        #########################################################
        #########################################################

        mu_mean = np.copy(new_mean)

        # Matrices for creating diffs
        diffs = np.zeros((N, lambda_))
        x1_sample = np.zeros(lambda_, dtype=int)
        x2_sample = np.zeros(lambda_, dtype=int)

        chi_N = np.sqrt(N)
        hist_norm = 1 / np.sqrt(2)
        counter_repaired = 0
        stoptol = False

        while (
            counteval < budget
            and not stoptol
            and ((time.time() - time_start) < maxtime)
        ):
            iter_ += 1
            hist_head = (hist_head % hist_size) + 1
            mu = np.int64(np.floor(lambda_ / 2))
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)

            if log_Ft:
                Ft_log = np.vstack((Ft_log, Ft))
            if log_value:
                value_log = np.vstack((value_log, fitness))
            if log_mean:
                mean_log = np.vstack((mean_log, fn_l(bounce_back_boundary2(new_mean))))
            if log_mean_cord:
                mean_cords_log = np.vstack((mean_cords_log, new_mean))
            if log_pop:
                pop_log[:, :, iter_ - 1] = population
            if log_best_val:
                best_val_log = np.vstack(
                    (best_val_log, np.min([np.min(best_val_log), np.min(fitness)]))
                )
            if log_worst_val:
                worst_val_log = np.vstack(
                    (worst_val_log, np.max([np.max(worst_val_log), np.max(fitness)]))
                )
            if log_eigen:
                cov_matrix = np.cov(np.transpose(population))
                eigen_values = np.linalg.eigvals(cov_matrix)
                eigen_values = np.flip(np.sort(eigen_values))
                eigen_log = np.vstack((eigen_log, eigen_values))

            # Select best 'mu' individuals of population
            selection = np.argsort(fitness)[:mu]
            selected_points = population[:, selection]

            # Save selected population in the history buffer
            history[hist_head - 1] = selected_points * hist_norm / Ft

            # Calculate weighted mean of selected points
            old_mean = np.copy(new_mean)


            #######################POI####################
            ##############################################
            new_mean = np.matmul(selected_points, weights)
            ##############################################
            ##############################################






            # Write to buffers
            mu_mean = np.copy(new_mean)
            d_mean[:, hist_head - 1] = (mu_mean - pop_mean) / Ft

            step = (new_mean - old_mean) / Ft

            # Update Ft
            Ft_history[hist_head - 1] = Ft
            old_Ft = Ft

            # Update parameters
            if hist_head == 1:
                pc[:, hist_head - 1] = (1 - cp) * np.zeros(N) / np.sqrt(N) + np.sqrt(
                    mu * cp * (2 - cp)
                ) * step
            else:
                pc[:, hist_head - 1] = (1 - cp) * pc[:, hist_head - 2] + np.sqrt(
                    mu * cp * (2 - cp)
                ) * step

            # Sample from history with uniform distribution
            limit = hist_head if iter_ < hist_size else hist_size
            history_sample = np.random.choice(
                np.arange(0, limit), size=lambda_, replace=True
            )
            history_sample2 = np.random.choice(
                np.arange(0, limit), size=lambda_, replace=True
            )

            x1_sample = sample_from_history(history, history_sample, lambda_)
            x2_sample = sample_from_history(history, history_sample, lambda_)

            # Make diffs
            for i in range(lambda_):
                x1 = history[history_sample[i]][:, x1_sample[i]]
                x2 = history[history_sample[i]][:, x2_sample[i]]
                diffs[:, i] = (
                    np.sqrt(cc)
                    * (
                        (x1 - x2)
                        + np.random.randn(1) * d_mean[:, history_sample[i] - 1]
                    )
                    + np.sqrt(1 - cc)
                    * np.random.randn(1)
                    * pc[:, history_sample2[i] - 1]
                )

            # New population
            population = new_mean[:, np.newaxis] + Ft * diffs
            population += (
                tol
                * (max(1 - 2 / N**2, 0)) ** (iter_ / 2)
                * np.random.randn(*diffs.shape)
                / chi_N
            )
            population = delete_infs_nans(population)

            # Check constraints violations and repair the individual if necessary
            population_temp = population.copy()
            population_repaired = np.apply_along_axis(
                bounce_back_boundary2, 0, population
            )

            counter_repaired = np.sum(
                np.any(population_temp != population_repaired, axis=0)
            )

            if Lamarckism:
                population = population_repaired
            ########################################################
            #######################POI##############################
            pop_mean = np.matmul(population, weights_pop)
            ########################################################
            ########################################################

            # Evaluation
            fitness = fn_l(population)
            if not Lamarckism:
                fitness_non_lamarckian = fn_d(population, population_repaired, fitness)

            # Break if fit
            wb = np.argmin(fitness)

            if fitness[wb] < best_fit:
                best_fit = fitness[wb]
                if Lamarckism:
                    best_par = population[:, wb]
                else:
                    best_par = population_repaired[:, wb]

            # Check worst fit
            ww = np.argmax(fitness)
            if fitness[ww] > worst_fit:
                worst_fit = fitness[ww]

            # Fitness with penalty for nonLamarckian approach
            if not Lamarckism:
                fitness = fitness_non_lamarckian

            # Check if the middle point is the best found so far
            cum_mean = 0.8 * cum_mean + 0.2 * new_mean
            cum_mean_repaired = bounce_back_boundary2(cum_mean)
            fn_cum = fn_l(cum_mean_repaired)

            if fn_cum < best_fit:
                best_fit = fn_cum
                best_par = cum_mean_repaired

            if fitness[0] <= stopfitness:
                msg = "Stop fitness reached."
                break

    exe_time = time.time() - time_start
    if exe_time > maxtime:
        msg = "Time limit reached"
        exe_time = maxtime

    cnt = {"function": int(counteval)}

    log = {}

    if log_Ft:
        log["Ft"] = Ft_log
    if log_value:
        log["value"] = value_log[:iter_, :]
    if log_mean:
        log["mean"] = mean_log[:iter_]
    if log_mean_cord:
        log["meanCord"] = mean_cords_log
    if log_pop:
        log["pop"] = pop_log[:, :, :iter_]
    if log_best_val:
        log["bestVal"] = best_val_log
    if log_worst_val:
        log["worstVal"] = worst_val_log
    if log_eigen:
        log["eigen"] = eigen_log

    res = {
        "par": best_par.tolist(),
        "value": best_fit,
        "counts": cnt,
        "resets": restart_number,
        "convergence": 1 if iter_ >= maxiter else 0,
        "time": exe_time,
        "message": msg,
        "diagnostic": log,
    }

    return res


class des_tuner_wrapper(object):
    """
    Wrapper class for tuning hyperparameters using Differential Evolution Strategy.

    Parameters:
    - evaluation_fc (function): Objective function to be minimized.
    - start_config (dict): Initial configuration for hyperparameters {HP_name: value}.
    - search_config (dict): Search space configuration for hyperparameters {HP_name: (lower, upper)}.

    Methods:
    - __init__(self, evaluation_fc, start_config, search_config) -> None:
        Initializes the wrapper with the given parameters.

    - fit(self, kwargs: dict):
        Optimizes hyperparameters using Differential Evolution Strategy.

        Parameters:
        - kwargs (dict): Additional control parameters for the optimization.

        Returns:
        - dict: Result dictionary containing optimization details.

    Notes:
    - The objective function `evaluation_fc` should take a parameter vector and return a scalar value to be minimized.
    - The search space for hyperparameters is defined by the `search_config` dictionary.
    - The initial configuration for hyperparameters is provided by the `start_config` dictionary.
    - Additional control parameters for the optimization can be passed through the `kwargs` dictionary.
    """

    def __init__(self, evaluation_fc, start_config: dict, search_config: dict) -> None:
        """
        Initializes the wrapper with the given parameters.

        Parameters:
        - evaluation_fc (function): Objective function to be minimized.
        - start_config (dict): Initial configuration for hyperparameters {HP_name: value}.
        - search_config (dict): Search space configuration for hyperparameters {HP_name: (lower, upper)}.
        """
        self.eval_fc = evaluation_fc
        self.search_config = search_config
        self.default_config = start_config
        self.hp_tuned = search_config.keys()

    def fit(self, kwargs: dict):
        """
        Optimizes hyperparameters using Differential Evolution Strategy.

        Parameters:
        - kwargs (dict): Additional control parameters for the optimization.

        Returns:
        - dict: Result dictionary containing optimization details.
        """
        result = des_classic(
            np.array([self.default_config[hp] for hp in self.hp_tuned]),
            self.eval_fc,
            upper=np.array([self.search_config[hp][1] for hp in self.hp_tuned]),
            lower=np.array([self.search_config[hp][0] for hp in self.hp_tuned]),
            **kwargs
        )
        result["hp_names"] = list(self.hp_tuned)
        return result


# Example usage:
if __name__ == "__main__":
    par = [-100, -100, -100, -100]
    fn = (
        lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2
    )  # Example fitness function
    kwargs = {
        "upper": np.array([-5, -23.3, 14, 11]),
        "lower": np.array([-101, -101, -101, -150]),
        "stopfitness": 1e-10,
        "lambda": 5,
        "time": 5,
        "diag": True,
    }
    result = des_classic(par, fn, **kwargs)
    print(result)