# -*- coding: utf-8 -*-
"""
@author: Josh Kemppainen
Revision 1.0
October 28, 2025
Michigan Technological University
1400 Townsend Dr.
Houghton, MI 49931

    ****************************************************************
    * Requirements:                                                *
    *   python 3.7+                                                *
    *                                                              *
    *   python numpy module:                                       *
    *    - pip3 install numpy (if pip manager is installed)        *
    ****************************************************************
"""
##############################
# Import Necessary Libraries #
##############################
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
import numpy as np




# RFR compute
def compute_fringe_slope_R2(x_data, y_data, min_x_data=None, max_x_data=None, direction='forward'):
    """Calculate cumulative linear-fit slopes and R-squared values.

    Starting at the first data point in the requested direction, this function
    calculates a least-squares linear fit for each growing data prefix. Results
    are returned only for prefix endpoints whose x values lie within the
    inclusive bounds and whose prefixes contain at least four points.

    For the RFR method and its interpretation, see Kemppainen et al.,
    https://doi.org/10.1038/s44172-026-00669-6.

    Parameters
    ----------
    x_data : array-like
        One-dimensional x coordinates, such as strain. Must have the same
        length as ``y_data`` and be suitable for numeric NumPy operations.
    y_data : array-like
        One-dimensional response values, such as stress, aligned with
        ``x_data``.
    min_x_data : float or None, optional
        Inclusive lower bound for returned prefix endpoints. Defaults to the
        minimum value in ``x_data``.
    max_x_data : float or None, optional
        Inclusive upper bound for returned prefix endpoints. Defaults to the
        maximum value in ``x_data``.
    direction : {'forward', 'reverse', 'backward', True, False}, optional
        Traversal direction. ``'forward'`` or ``True`` uses the input order;
        ``'reverse'``, ``'backward'`` or ``False`` flips both arrays first.

    Returns
    -------
    fringe : numpy.ndarray
        X coordinate at the endpoint of each accepted cumulative fit.
    slopes : numpy.ndarray
        Least-squares slope for each accepted cumulative fit.
    r2 : numpy.ndarray
        Coefficient of determination for each accepted cumulative fit.
    """
    # Set direction
    if direction == 'forward' or direction is True:
        x_data = np.asarray(x_data, dtype=float)
        y_data = np.asarray(y_data, dtype=float)
    elif direction == 'reverse' or direction == 'backward' or direction is False:
        x_data = np.flip(np.asarray(x_data, dtype=float))
        y_data = np.flip(np.asarray(y_data, dtype=float))
    else:
        raise Exception(f'ERROR direction={direction} not supported')

    # Set defaults if min_x_data or max_x_data are None
    if min_x_data is None: min_x_data = min(x_data)
    if max_x_data is None: max_x_data = max(x_data)

    # 1-indexed count: at index i we've seen i+1 points
    n = np.arange(1, len(x_data) + 1)
    slice_ = (min_x_data <= x_data) & (x_data <= max_x_data) & (n >= 4)

    sum_xi = np.cumsum(x_data)
    sum_yi = np.cumsum(y_data)
    sum_xi_2 = np.cumsum(x_data * x_data)
    sum_yi_2 = np.cumsum(y_data * y_data)
    sum_xi_yi = np.cumsum(x_data * y_data)

    sum_xi = sum_xi[slice_]
    sum_yi = sum_yi[slice_]
    sum_xi_2 = sum_xi_2[slice_]
    sum_yi_2 = sum_yi_2[slice_]
    sum_xi_yi = sum_xi_yi[slice_]
    n = n[slice_]
    fringe = x_data[slice_]

    SSxy = sum_xi_yi - (sum_xi * sum_yi / n)
    SSxx = sum_xi_2 - (sum_xi * sum_xi / n)
    slopes = SSxy / SSxx

    SStotal = sum_yi_2 - (sum_yi * sum_yi / n)
    SSreg = slopes * slopes * SSxx
    R2 = SSreg / SStotal

    return fringe, slopes, R2


# Function to window data
def window(x, y, xlo=None, xhi=None):
    """Return paired samples whose x coordinates fall within a range.

    The bounds are inclusive. Samples containing NaN in ``x`` are excluded
    because comparisons with NaN are false.
    
    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional x-coordinate array.
    y : numpy.ndarray
        One-dimensional y-value array aligned with ``x``.
    xlo : float or None, optional
        Inclusive lower x bound. Defaults to the smallest non-NaN value in
        ``x``.
    xhi : float or None, optional
        Inclusive upper x bound. Defaults to the largest non-NaN value in
        ``x``.
    
    Returns
    -------
    win_x : numpy.ndarray
        X values within the requested interval.
    win_y : numpy.ndarray
        Corresponding y values, selected using the same mask as ``win_x``.
    """
    if xlo is None: xlo = np.nanmin(x)
    if xhi is None: xhi = np.nanmax(x)
    in_window = (x >= xlo) & (x <= xhi)  # NaN samples compare False -> dropped here
    win_x, win_y = x[in_window], y[in_window]
    
    return win_x, win_y


# The RFR-mulit pass for finding linear region
def rfr_multi_pass(x_data, y_data, sequence='FBF', min_span=0, ic=None):
    """Search for linear-region bounds using alternating forward/backward passes.

    Each pass calculates cumulative slopes and R-squared values with
    :func:`compute_fringe_slope_R2`, then uses the endpoint with the largest
    R-squared value as the next pass's initial condition. The returned bounds
    are the sorted endpoints from the final two passes, or from the sole pass
    and initial condition when only one pass is requested.

    Parameters
    ----------
    x_data : array-like
        One-dimensional x coordinates. Must be aligned with ``y_data``.
    y_data : array-like
        One-dimensional response values aligned with ``x_data``.
    sequence : str, optional
        Non-empty sequence of alternating pass directions: ``'F'`` for
        forward and ``'B'`` for backward. For example, ``'FBF'`` or ``'BFB'``.
    min_span : float, optional
        Minimum x distance from the current initial condition excluded from
        candidate fit endpoints. Defaults to zero.
    ic : float or None, optional
        Starting x coordinate. Defaults to the minimum x value when the
        sequence starts with ``'F'`` or the maximum when it starts with ``'B'``.

    Returns
    -------
    passes : list of dict
        One entry per pass, containing ``'fringe'`` (candidate endpoints),
        ``'r2'`` (their R-squared values), ``'max_fringe'`` (the selected
        endpoint), and ``'max_r2'`` (its R-squared value).
    bounds : numpy.ndarray, shape (2,)
        Sorted x coordinates delimiting the selected region.

    Raises
    ------
    ValueError
        If ``sequence`` is empty, contains characters other than ``F``/``B``,
        does not alternate directions, or a pass has no valid candidates.
    """
    
    # Error checking
    sequence = sequence.upper()
    if not sequence or any(c not in "FB" for c in sequence):
        raise ValueError("sequence must contain only F and B.")
    
    if any(a == b for a, b in zip(sequence, sequence[1:])):
        raise ValueError("sequence must alternate between F and B.")
    
    # Set starting point (initial condition) for multi-pass
    min_x_data = min(x_data)
    max_x_data = max(x_data)
    if sequence.upper().startswith('F'):
        if ic is None: ic = min_x_data
    elif sequence.upper().startswith('B'):
        if ic is None: ic = max_x_data
    else:
        raise Exception(f'sequence={sequence} not supported, must be FBF or BFB')
        
    # Start multi-pass sequence
    passes = [] # [(fringe, slopes), ... Npasses]
    prior = None
    initial_ic = ic
    for direction in sequence:
        # Ensure direction is alternating
        if prior is not None and prior == direction.upper():
            raise Exception(f'sequence={sequence} does not alternate F and B')
            
        # Compute a pass
        if direction.upper() == 'F':
            win_x, win_y = window(x_data, y_data, xlo=ic, xhi=max_x_data)
            fringe, slopes, r2 = compute_fringe_slope_R2(win_x, win_y, min_x_data=ic+min_span, max_x_data=max_x_data, direction='forward') 
        elif direction.upper() == 'B':
            win_x, win_y = window(x_data, y_data, xlo=min_x_data, xhi=ic)
            fringe, slopes, r2 = compute_fringe_slope_R2(win_x, win_y, min_x_data=min_x_data, max_x_data=ic-min_span, direction='reverse')   
        else:
            raise Exception(f'direction={direction} not supported, must be F or B')
            
        if fringe.size == 0:
            raise ValueError(f'No valid fringe candidates for direction={direction}, ic={ic}, min_span={min_span}.')
            
        # Maximize the r2
        max_r2_index = int(np.argmax(r2))
        max_fringe = fringe[max_r2_index]
        max_r2     = r2[max_r2_index]
        
        # Log pass and reset ic
        tmp = {'fringe':fringe, 'r2':r2, 'max_fringe':max_fringe, 'max_r2':max_r2}
        passes.append(tmp)
        prior = direction
        ic = max_fringe

    # Set bounds based on number of passes
    if len(passes) > 1:  # Use last two FB or BF
        bounds = sorted([passes[-1]['max_fringe'], passes[-2]['max_fringe']])
    elif len(passes) == 1:
        bounds = sorted([passes[0]['max_fringe'], initial_ic])
    else: # Default to min and max of series
        bounds = sorted([min_x_data, max_x_data])
    
    return passes, np.array(bounds)


# Compute a linear regression within bounds
def linear_regression(x, y, xlo=None, xhi=None):
    """Fit a first-degree polynomial to data within optional x bounds.

    The fit is performed on samples selected by :func:`window` and uses
    ``numpy.polynomial.polynomial.polyfit`` coefficient ordering.

    Parameters
    ----------
    x : array-like
        One-dimensional x coordinates.
    y : array-like
        One-dimensional y values aligned with ``x``.
    xlo : float or None, optional
        Inclusive lower x bound. Defaults to the minimum value in ``x``.
    xhi : float or None, optional
        Inclusive upper x bound. Defaults to the maximum value in ``x``.

    Returns
    -------
    numpy.ndarray, shape (2,)
        Linear-fit coefficients ``[intercept, slope]`` for
        ``y = intercept + slope*x``.
    """
    reduced_x, reduced_y = window(x, y, xlo=xlo, xhi=xhi)
    regression_coeffs = np.polynomial.polynomial.polyfit(reduced_x, reduced_y, 1) # [intercept, slope]
    return regression_coeffs


# Compute the intersection between to regression coeffs
def line_intersection(coeffs1, coeffs2):
    """Calculate the intersection of two lines given polynomial coefficients.

    Each line is represented in the coefficient order returned by
    :func:`linear_regression`: ``[intercept, slope]``.

    Parameters
    ----------
    coeffs1 : array-like, shape (2,)
        ``[intercept, slope]`` coefficients for the first line.
    coeffs2 : array-like, shape (2,)
        ``[intercept, slope]`` coefficients for the second line.

    Returns
    -------
    intersection : list of float
        Coordinates ``[x, y]`` where the lines intersect. When the slopes are
        equal, there is no unique intersection and these coordinates are not
        meaningful.
    does_intersect : bool
        ``True`` when the slopes differ; ``False`` when the lines are parallel
        or identical.
    """
    b1, m1 = coeffs1
    b2, m2 = coeffs2
    does_intersect = True
    if m1 == m2:
        does_intersect = False
        x_intersect = 0
        y_intersect = 0
    else:
        x_intersect = (b2 - b1) / (m1 - m2)
        y_intersect = b1 + m1 * x_intersect

    return [x_intersect, y_intersect], does_intersect


# Compute the KDE and set Tg based on the selection method
def compute_gaussian_kde(xdata, tgs, ic_increment, peak_index, weighting, slopes, widths, bound_weights, glassy_slope, kernel_width=None):
    """Estimate a KDE of candidate Tg values and select a peak by probability.

    A Gaussian KDE is evaluated on a temperature grid. The PDF is partitioned
    at local minima between neighboring peaks, and each region's probability
    is estimated by numerical integration. Peaks are ranked by decreasing
    probability and ``peak_index`` selects the requested rank. Optional
    weights can be supplied through a NumPy-compatible expression.

    Parameters
    ----------
    xdata : array-like
        Temperature data used to clamp the PDF evaluation grid to its minimum
        and maximum values.
    tgs : array-like
        Candidate glass-transition temperatures, in kelvin, used to fit the
        KDE. At least two distinct values are required for KDE evaluation.
    ic_increment : float
        IC-grid temperature increment, in kelvin. Used as the kernel width
        when ``kernel_width`` is None.
    peak_index : int
        Zero-based rank of the peak to select after sorting by decreasing
        integrated probability. Negative values use Python-style indexing.
    weighting : str or None
        Optional expression evaluated element-wise to produce one KDE weight
        per candidate Tg. Available variables are ``rww`` (rubbery widths
        normalized by their maximum), ``rsw`` (rubbery slopes divided by
        ``glassy_slope``), ``bpw`` (bound weights), ``kw`` (effective kernel
        width), and ``np`` (NumPy). For example, ``'0.5*rww + 0.5*rsw'``.
        Use None for an unweighted KDE.
    slopes : array-like
        Candidate rubbery-region slopes used by the ``rsw`` weighting variable.
    widths : array-like
        Candidate rubbery-region widths used by the ``rww`` weighting variable.
    bound_weights : array-like
        Candidate bound-penetration weights, available as ``bpw`` in the
        weighting expression.
    glassy_slope : float
        Glassy-region slope used to normalize ``slopes`` for ``rsw``.
    kernel_width : float or None, optional
        Effective Gaussian kernel width, in kelvin, defined as the
        approximately ``±3σ`` span. Defaults to ``ic_increment``.

    Returns
    -------
    tuple
        ``(temp, tg_kde, xpeaks, ypeaks, peak_info,
        selected_tg, kernel_effective_width)``. ``temp`` and ``tg_kde`` are
        None when a KDE cannot be formed. ``xpeaks`` and ``ypeaks`` contain
        detected peak temperatures and densities. ``peak_info`` is a list of
        dictionaries with each peak's PDF index, temperature, density, integrated
        probability, region bounds, selection status, label, and rank.
        ``selected_tg`` is the selected Tg in kelvin (or a fallback value if
        the KDE cannot be formed). ``kernel_effective_width`` is the width
        used to determine the KDE bandwidth.

    Raises
    ------
    IndexError
        If ``peak_index`` is outside the available peak ranks.
    """
    # Over-ride default kernel widths
    if kernel_width is None:
        kernel_effective_width = ic_increment
    else:
        kernel_effective_width = kernel_width

    # A KDE requires at least two different Tg values.
    if len(tgs) < 2 or np.std(tgs, ddof=1) == 0:
        selected_tg = tgs[0] if len(tgs) else (max(xdata) + min(xdata)) / 2
        print('WARNING - not enough distinct Tg values to compute a Gaussian KDE.')
        return (None, None, np.array([]), np.array([]), [],
                selected_tg, kernel_effective_width)

    # Convert the requested kernel width (approximately ±3σ) to gaussian_kde's
    # bandwidth scaling factor.
    tg_std = np.std(tgs, ddof=1)
    kernel_std = kernel_effective_width / 6.0
    bw_method = kernel_std / tg_std

    # Build the PDF evaluation grid, clamped to the xdata range.
    xmin, xmax = min(tgs), max(tgs)
    padding = max(0.05*(xmax - xmin), 4*kernel_std)
    xmin_pdf = max(min(xdata), xmin - padding)
    xmax_pdf = min(max(xdata), xmax + padding)

    dx_pdf = 0.01
    temp = np.arange(xmin_pdf, xmax_pdf + dx_pdf, dx_pdf)
    
    # Compute optional weights for the Gaussian KDE. The user may supply an
    # arbitrary NumPy-compatible mathematical expression through:
    #
    #     weighting = "<expression>"
    #
    # where the expression is evaluated element-wise and must return one weight
    # per candidate Tg. The following variables are available:
    #
    #  rww = rubbery width weight. Normalized to one (divided by maximum width)
    #  rsw = rubbery slope weight. Divided by glassy slope
    #  bpw = bound penetration weight.
    #
    # Example expressions:
    #
    #     weights = "rww"
    #     weights = "rsw"
    #     weights = "0.5*rww + 0.5*rsw"
    #     weights = "rww**2"
    #     weights = "np.sqrt(rww)"
    #     weights = "np.exp(-rsw)"
    weights = None
    if weighting is not None:
        # Compute possible weighting options, where the following are supported
        rww = widths/max(widths)   # Set width-weights via a normalized width of regions
        rsw = slopes/glassy_slope  # Set slope-weights via a normalized cte relative to glassy slope
        bpw = bound_weights        # Set bound penalty where no-penetration=1 and larger-penetraions approach zero
        kw = kernel_effective_width # Set scalar value for kernel width
        
        # Setup the globals namespace to limit scope of what eval() can do
        allowed_builtins = ['min','max','sum','abs','len','map','range','reversed']
        copied_builtins = globals()['__builtins__'].copy()
        globals_dict = {'rww':rww, 'rsw':rsw, 'bpw':bpw, 'kw': kw}
        globals_dict['__builtins__'] = {key:copied_builtins[key] for key in allowed_builtins}
        globals_dict['np'] = globals()['np']

        # Try computing weights with vectorized numpy and eval
        try:
            weights = np.asarray(eval(weighting, globals_dict), dtype=float)
            weights = np.maximum(weights, 0)   # clamp negatives to zero
            weights = np.minimum(weights, 1e4) # clamp large values to 1e4
            weights = np.nan_to_num(weights, nan=1.0) # set Nan's to 1.0
            
            # If all weights evaluate to zero reset weights to ones (this is
            # useful to shut-off weighting with zeros like "0.0*rww + 0.0*rsw",
            # but still keep the syntax for setting weights for testing)
            weight_set = set(weights)
            if len(weight_set) == 1 and sum(weight_set) == 0:
                weights = np.ones_like(weights)
            
            if weights.ndim != 1:
                raise ValueError("weights expression must return a 1-D array")
            if len(weights) != len(tgs):
                raise ValueError("weights expression returned the wrong length")
            if np.any(weights < 0):
                raise ValueError("weights must be non-negative")
            print('Successfully evaluated weighting compute = "{}"'.format(weighting))
        except:
            print('Failed to evaulate weighting compute = "{}". Using weights=None'.format(weighting))

    # Compute the KDE and locate its peaks.
    tg_kde_func = gaussian_kde(tgs, bw_method=bw_method, weights=weights)
    tg_kde      = tg_kde_func(temp)     
    peaks, _    = find_peaks(tg_kde)

    # If the PDF has no interior peak, use its global maximum as one peak.
    if len(peaks) == 0:
        peaks = np.array([np.argmax(tg_kde)])

    # Partition the PDF at the local minima between neighboring peaks.
    xpeaks = temp[peaks]
    ypeaks = tg_kde[peaks]
    left_bases = np.zeros(len(peaks), dtype=int)
    right_bases = np.zeros(len(peaks), dtype=int)
    for i, peak in enumerate(peaks):
        if i == 0:
            left_bases[i] = 0
        else:
            previous_peak = peaks[i - 1]
            left_bases[i] = previous_peak + np.argmin(tg_kde[previous_peak:peak + 1])

        if i == len(peaks) - 1:
            right_bases[i] = len(tg_kde) - 1
        else:
            next_peak = peaks[i + 1]
            right_bases[i] = peak + np.argmin(tg_kde[peak:next_peak + 1])

    # Integrate the PDF within each peak's region.
    peak_prob = np.array([
        np.trapz(tg_kde[lb:rb + 1], temp[lb:rb + 1])
        for lb, rb in zip(left_bases, right_bases)
    ])

    # Rank peaks by decreasing probability and select the requested rank.
    ranked_indices = np.argsort(peak_prob)[::-1]
    if not -len(ranked_indices) <= peak_index < len(ranked_indices):
        raise IndexError(f'peak_index={peak_index} is out of range for {len(ranked_indices)} peaks.')

    selected_rank = peak_index % len(ranked_indices)
    selected_peak_index = ranked_indices[selected_rank]
    selected_tg = xpeaks[selected_peak_index]

    peak_info = []
    for rank, peak_idx in enumerate(ranked_indices):
        i = int(peak_idx)
        peak_info.append({
            'peak_index': i,
            'pdf_index': int(peaks[i]),
            'temp': xpeaks[i],
            'density': ypeaks[i],
            'prob': peak_prob[i],
            'left_base': int(left_bases[i]),
            'right_base': int(right_bases[i]),
            'selected': i == selected_peak_index,
            'label': 'prob[{}]\n{:.3g}'.format(rank, peak_prob[i]),
            'label_rank': rank,
        })

    return (temp, tg_kde, xpeaks, ypeaks, peak_info,
            selected_tg, kernel_effective_width)