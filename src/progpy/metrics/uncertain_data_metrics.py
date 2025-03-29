# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

"""
This file includes functions for calculating general metrics (i.e. mean, std, percentiles, etc.) on any distribution of type UncertainData (e.g. states, event_states, an EOL distribution, etc.)
"""
from typing import Iterable, Union, Dict, List, Optional, Any
from numpy import isscalar, mean, std, array, percentile
from scipy import stats
from warnings import warn

from ..uncertain_data import UncertainData, UnweightedSamples

def calc_metrics(data: Union[UncertainData, Iterable[float]], 
                ground_truth: Optional[Union[float, Dict[str, float]]] = None, 
                **kwargs) -> Dict[str, Any]:
    """Calculate metrics for uncertain data

    Args:
        data: Data from a single event. Can be UncertainData or array of floats
        ground_truth: Ground truth value(s). Can be float or dict of floats
        **kwargs: Configuration parameters including:
            n_samples (int): Number of samples for metrics calculation. Default 10,000
            keys (List[str]): Keys to calculate metrics for. Default all keys

    Returns:
        Dict[str, Any]: Collection of calculated metrics

    Raises:
        ValueError: If data is empty or invalid
        TypeError: If data type is not supported
    """
    # Input validation
    if data is None:
        raise ValueError("Data cannot be None")

    params = {
        'n_samples': 10000,
    }
    params.update(kwargs)

    if isinstance(data, UncertainData):
        # Handle UncertainData type
        keys = params.get('keys', data.keys())
        keys = [keys] if isinstance(keys, str) else keys

        # Validate ground truth
        if ground_truth is not None and isscalar(ground_truth):
            ground_truth = {key: ground_truth for key in keys}

        samples = data if isinstance(data, UnweightedSamples) else data.sample(params['n_samples'])

        if len(samples) == 0:
            raise ValueError('Data must not be empty')

        # Calculate metrics for each key
        result = {
            key: calc_metrics(
                samples.key(key),
                None if ground_truth is None else ground_truth.get(key),
                **kwargs
            ) for key in keys
        }

        # Update with distribution-specific values
        for key in keys:
            result[key].update({
                'mean': data.mean[key],
                'median': data.median[key],
                'percentiles': {'50': data.median[key]}
            })

        return result

    # Handle array/list of numbers
    if isinstance(data, Iterable):
        if len(data) == 0:
            raise ValueError('Data must not be empty')

        data_array = array([d for d in data if d is not None])
        if len(data_array) == 0:
            raise ValueError('All samples were None')
        
        if len(data_array) < len(data):
            warn("Some samples were None, metrics calculated only for non-None samples")

        data_array.sort()
        data_mean = mean(data_array)
        data_median = data_array[len(data_array)//2]

        # Calculate percentiles more efficiently
        metrics = {
            'min': data_array[0],
            'percentiles': {
                str(p): percentile(data_array, p) if len(data_array) >= 100/p else None
                for p in [0.01, 0.1, 1, 10, 25, 50, 75]
            },
            'median': data_median,
            'mean': data_mean,
            'std': std(data_array),
            'max': data_array[-1],
            'median absolute deviation': mean(abs(data_array - data_median)),
            'mean absolute deviation': mean(abs(data_array - data_mean)),
            'number of samples': len(data_array)
        }

        # Add ground truth metrics if provided
        if ground_truth is not None:
            abs_errors = abs(data_array - ground_truth)
            metrics.update({
                'mean absolute error': mean(abs_errors),
                'mean absolute percentage error': mean(abs_errors) / ground_truth,
                'relative accuracy': 1 - abs(ground_truth - data_mean) / ground_truth,
                'ground truth percentile': stats.percentileofscore(data_array, ground_truth)
            })

        return metrics

    raise TypeError(f"Data must be UncertainData or array of numbers, got {type(data)}")
