import numpy as np


def create_tiling_grid(low, high, bins=(10, 10), offsets=(0.0, 0.0)):
    """Define a uniformly-spaced grid that can be used for tile-coding a space.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins or tiles along each corresponding dimension.
    offsets : tuple
        Split points for each dimension should be offset by these values.

    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    # TODO: Implement this
    grid = []
    for i in range(len(low)):
        grid.append(tuple(np.linspace(low[i] + offsets[i], high[i] + offsets[i], bins[i] + 1)[1:-1]))
    return grid


def create_tilings(low, high, tiling_specs):
    """Define multiple tilings using the provided specifications.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    tiling_specs : list of tuples
        A sequence of (bins, offsets) to be passed to create_tiling_grid().

    Returns
    -------
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    """
    grids = []
    for i in range(len(tiling_specs)):
        grids.append(create_tiling_grid(low, high, tiling_specs[i][0], tiling_specs[i][1]))
    return grids


def discretize(sample, grid):
    """Discretize a sample as per given grid.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.

    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    # TODO: Implement this
    discretized_sample = []
    for i in range(len(sample)):
        discretized_sample.append(np.sum((np.asarray(grid[i]) < sample[i]).astype(int)))
    return tuple(discretized_sample)


def tile_encode(sample, tilings, flatten=False):
    """Encode given sample using tile-coding.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    flatten : bool
        If true, flatten the resulting binary arrays into a single long vector.

    Returns
    -------
    encoded_sample : list or array_like
        A list of binary vectors, one for each tiling, or flattened into one.
    """
    # TODO: Implement this
    encoded_sample = []
    for i in range(len(tilings)):
        d = discretize(sample, tilings[i])
        encoded_sample.append(d)
    if flatten is True:
        flat_sample = [i for tup in encoded_sample for i in tup]
        return flat_sample
    return encoded_sample
