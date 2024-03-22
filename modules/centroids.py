import numpy as np


def vanila_centroid(population, args=None):
    """
    args[0] = weights_pop
    """
    return np.matmul(args[0], population)


def mean_centroid(population, args=None):
    return population.mean(axis=0)


def median_centroid(population, args=None):
    return np.median(population, axis=0)


def interquartile_centroid(population, args=None):
    """
    args[0] = percentile to cut (int)
    """
    iqr = np.percentile(
        population, 100-args[0], axis=0) - np.percentile(population, args[0], axis=0)
    return np.percentile(population, args[0], axis=0) + iqr/2


def windsor_centroid(population, args=None):
    """
    Calculate the windsorized mean for a given numpy array.

    Parameters:
        arr (numpy.ndarray): Input array.
        args[0]: percentile (float, optional): Percentile to use for winsorizing. Default is 10.

    Returns:
        numpy.ndarray: Windsorized mean of the input array along the 0 axis.
    """
    def windsor_mean(population):
        lower_percentile = np.percentile(population, args[0])
        upper_percentile = np.percentile(population, 100 - args[0])

        # Replace elements below lower percentile with the lower percentile
        population[population < lower_percentile] = lower_percentile

        # Replace elements above upper percentile with the upper percentile
        population[population > upper_percentile] = upper_percentile

        # Calculate the mean
        return np.mean(population)

    return np.apply_along_axis(windsor_mean, 0, population)
