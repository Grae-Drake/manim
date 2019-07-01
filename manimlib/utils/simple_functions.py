from functools import reduce
import inspect
import numpy as np
import operator as op


def sigmoid(x):
    """
    Maps the real numbers onto the interval -1 to 1.
    
    Parameters
    ----------
    x : int
        The real number to be mapped onto the -1 to 1 interval.

    Examples
    --------
    >>> sigmoid(-2)
    0.11920292202211755
    >>> sigmoid(0)
    0.5
    >>> sigmoid(1.5)
    0.8175744761936437
    """
    return 1.0 / (1 + np.exp(-x))


CHOOSE_CACHE = {}


def choose_using_cache(n, r):
    """
    Helper function for choose() for efficient caching.
    """
    if n not in CHOOSE_CACHE:
        CHOOSE_CACHE[n] = {}
    if r not in CHOOSE_CACHE[n]:
        CHOOSE_CACHE[n][r] = choose(n, r, use_cache=False)
    return CHOOSE_CACHE[n][r]


def choose(n, r, use_cache=True):
    """
    Calculate the number of k-combinations.
    
    Also known as "n choose r" for given n and r. This function defaults to
    caching results for effeciently calling multiple times.

    Parameters
    ----------
    n : int
        The size of the set you're choosing combinations from.
    r : int
        The size of the subset you're choosing.
    use_cach : bool
        Whether to look up results from and store results to a cache. Defaults to True.
    
    Examples
    --------
    >>> choose(5, 3)
    10
    >>> choose(2, 2)
    1
    >>> choose(3, 4)
    0
    >>> choose(4, 0)
    1
    """
    if use_cache:
        return choose_using_cache(n, r)
    if n < r:
        return 0
    if r == 0:
        return 1
    denom = reduce(op.mul, range(1, r + 1), 1)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    return numer // denom


def get_num_args(function):
    return len(get_parameters(function))


def get_parameters(function):
    return inspect.signature(function).parameters


def clip_in_place(array, min_val=None, max_val=None):
    """
    Clip all values of an array in place to optional min and max values.

    Parameters
    ----------
    array : numpy.ndarray
        The Numpy array to modify in place.
    min_val : Union[None, int]
        The minimum value for the clipped array. Defaults to None.
    max_val : Union[None, int]
        The maximum value for the clipped array. Defaults to None.

    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4])
    >>> arr_1 = np.array([1, 2, 3, 4])
    >>> clip_in_place(arr_1, 2, 3)
    array([2, 2, 3, 3])
    >>> arr_1
    array([2, 2, 3, 3])
    >>> arr_2 = np.array([-1, 0, 0.5, -2, 1, 2])
    >>> clip_in_place(arr_2, 0, 1)
    array([0. , 0. , 0.5, 0. , 1. , 1. ])
    >>> arr_2
    array([0. , 0. , 0.5, 0. , 1. , 1. ])
    >>> arr_3 = np.array([1, 100, 1000, -10])
    >>> clip_in_place(arr_3, max_val=42)
    array([  1,  42,  42, -10])
    >>> arr_3
    array([  1,  42,  42, -10])
    """
    if max_val is not None:
        array[array > max_val] = max_val
    if min_val is not None:
        array[array < min_val] = min_val
    return array


def fdiv(a, b, zero_over_zero_value=None):
    """
    Lightweight name for this extremely common operation.

    Todo: We may wish to have more fine-grained control over division by zero
    behavior in the future (separate specifiable values for 0/0 and x/0 with
    x != 0), but for now, we just allow the option to handle indeterminate 0/0.

    Parameters
    ----------
    a : int
        Numerator.
    b : int
        Denominator.
    zero_over_zero_value : 
        Todo: document this parameter and behavior.

    Examples
    --------
    >>> fdiv(10, 5)
    2.0
    >>> fdiv(1, 4)
    0.25
    """
    if zero_over_zero_value is not None:
        out = np.full_like(a, zero_over_zero_value)
        where = np.logical_or(a != 0, b != 0)
    else:
        out = None
        where = True

    return np.true_divide(a, b, out=out, where=where)


def binary_search(function,
                  target,
                  lower_bound,
                  upper_bound,
                  tolerance=1e-4):
    """
    Binary search for the input yielding a target output for a given function.
    
    Be careful with discontinuous functions and local extrema.

    Parameters
    ----------
    function : function
        The function yielding your target output.
    target : Union[int, float, complex]
        The result of function(x) for the x you're searching for.
    lower_bound : Union[int, float, complex]
        The lower bound of your binary search.
    upper_bound : Union[int, float, complex]
        The upper bound of your binary search.
    tolerance : Union[int, float, complex]
        The relative error to tolerate. Search completes when the difference
        between the lower bound and upper bound is less than this number.
        Defaults to 1e-4.

    Examples
    --------
    >>> round(binary_search(lambda x: x ** 2, 9, 0, 100), 5)
    2.99997
    >>> round(binary_search(sigmoid, 0.15, -100, 100), 5)
    -1.73464
    """
    
    lh = lower_bound
    rh = upper_bound
    while abs(rh - lh) > tolerance:
        mh = np.mean([lh, rh])
        lx, mx, rx = [function(h) for h in (lh, mh, rh)]
        if lx == target:
            return lx
        if rx == target:
            return rx

        if lx <= target and rx >= target:
            if mx > target:
                rh = mh
            else:
                lh = mh
        elif lx > target and rx < target:
            lh, rh = rh, lh
        else:
            return None
    return mh
