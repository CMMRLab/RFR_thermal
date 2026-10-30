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
import modules.signals as signals
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib as mpl  
import numpy as np

# General plot settings
plt.rcParams.update({
    # Main text
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,

    # Plot text
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,

    # Mathematical text
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',

    # Keep fonts editable when exported
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})




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

# Full stand alone RFR-thermal implementation
def full_rfr_thermal(temperature, volume, wn, order, quadrant_mirror, min_span, min_tg, max_tg, weighting, peak_index, plot_orientation=''):
    #---------------------------------------------#
    # The filtering and FBF/BFB assume increasing #
    # xdata, so flip the lists if needed          #
    #---------------------------------------------#
    if temperature[-1] < temperature[0]:
        volume.reverse()
        temperature.reverse()
    
    # We will be using NumPy, thus convert to NumPy
    volume      = np.array(volume)
    temperature = np.array(temperature)
    
    
    #------------------------#
    # Filter the volume data #
    #------------------------#
    # Compute the PSD if wn is set to 'op'
    if str(wn).startswith('op'):
        # Ensure closests y-data point to zero is moved to be zero, such
        # that the DC-offset value makes sense for the mean crossing method
        closests_zero_index = np.argmin(np.abs(volume - 0))
        yshift = -volume[closests_zero_index]
        shifted_volume = volume + yshift
        
        # Compute the PSD for the shifted volume data
        wns, psd = signals.compute_PSD(temperature, shifted_volume)
        mean_psd = np.mean(psd)
        
        # Find where each PSD crosses the mean
        wn_frequency = wns[np.min(np.where(psd < mean_psd)[0])]
        wn_power = psd[np.min(np.where(wns == wn_frequency)[0])]
        print('{:<50} {}'.format('Computed normalized cutoff frequency for volume: ', wn_frequency))
        
        # Filter the data with an optimized wn value
        filtered_volume, qm_volume = signals.butter_lowpass_filter(temperature, volume, wn_frequency, order, quadrant_mirror)
    else:
        # Filter the data with a user defined wn value
        wn_frequency = wn
        filtered_volume, qm_volume = signals.butter_lowpass_filter(temperature, volume, wn_frequency, order, quadrant_mirror)
        
        
    #----------------------------------------------#
    # Determine the glassy region using a FBF pass #
    #----------------------------------------------#
    # Find the glassy region
    glassy_passes, glassy_bounds = rfr_multi_pass(temperature, filtered_volume, sequence='FBF', min_span=min_span, ic=None)
    gf1, gr1, gf2 = glassy_passes # Extract out glassy passes and set colors (g=glassy, f=forward or r=reverse, int=pass-number)
    gf1['color'] = 'tab:cyan'
    gr1['color'] = 'tab:blue'
    gf2['color'] = 'darkblue'
    
    # Apply a linear regression in the glassy region
    glassy_coeffs = linear_regression(temperature, filtered_volume, xlo=min(glassy_bounds), xhi=max(glassy_bounds))
    glassy_volume = glassy_coeffs[1]*glassy_bounds + glassy_coeffs[0]       # glassy volume data from glassy bounds (for plotting)
    
    
    #---------------------------------------------#
    # Rubbery IC Backward-forward-backward checks #
    #---------------------------------------------#
    # Setup IC grid to search over
    ic_increment = min_span/2 # couple ic_increment to min_span parameter
    ic_start     = (max(glassy_bounds) // ic_increment + 1) * ic_increment  # round up on glassy upper, 1st will fail and be ignored
    ic_end       = max(temperature)
    ic_steps     = int( (ic_end - ic_start) // ic_increment + 1 )
    ic_list      = [float(ic_start + n*ic_increment) for n in range(ic_steps)]
    
    # Set some global parameters for the IC-search
    tau = min_span/4
    if min_tg is None: min_tg = -np.inf # possible restriction based on settings
    if max_tg is None: max_tg =  np.inf # possible restriction based on settings
    
    # Trim out the glassy region
    reduced_x, reduced_y = window(temperature, filtered_volume, xlo=max(glassy_bounds), xhi=max(temperature))
    
    # Start guess linear sections from different ICs
    xlo = max(glassy_bounds)
    guesses = {} # { float(ic): [ [rfr_pass1, ... rfr_passN], [lo_bound, hi_bound], regression, guessed_tg], ... Nic's }
    lowers, uppers, tgs, ics, slopes, bound_weights, distance_from_glassy, bound_penetrations = [], [], [], [], [], [], [], []
    for ic in ic_list:
        # Compute FBF or BFB from IC
        try:
            passes, bounds = rfr_multi_pass(reduced_x, reduced_y, sequence='BFB', min_span=min_span, ic=ic)
            if len(passes) != 3: continue # skip incomplete FBF or BFB searches
        except: continue # skip any failed attempts
    
        # Compute the linear regression and get the interseciton. Skip over meaningles results
        guessed_coeffs = linear_regression(temperature, filtered_volume, xlo=min(bounds), xhi=max(bounds))
        tg_guess, does_intersect = line_intersection(glassy_coeffs, guessed_coeffs)
        if not does_intersect: continue # skip over non-intersecting guesses
        if tg_guess[0] < min(temperature): continue # Tg should be within xdata-span (lower-side)
        if tg_guess[0] > max(temperature): continue # Tg should be within xdata-span (upper-side)
        if tg_guess[0] < min_tg: continue # Skip over too small of Tg based on user inputs 
        if tg_guess[0] > max_tg: continue # Skip over too large of Tg based on user inputs 
        
        # Compute bound penertaion penality
        bound_penetration = 0.0
        bound_penetration += max(0.0, max(glassy_bounds) - tg_guess[0])  # Tg should not be within the glassy region (upper bound)
        bound_penetration += max(0.0, tg_guess[0] - min(bounds)) # Tg should not be within the rubbery region (lower bound)
        bound_weight = np.exp(-bound_penetration / tau)
        
        # Log the results
        guesses[ic] = [passes, bounds, guessed_coeffs, tg_guess]
        bound_penetrations.append( bound_penetration )
        bound_weights.append( bound_weight )
        lowers.append( min(bounds) )
        uppers.append( max(bounds) )
        slopes.append( guessed_coeffs[1] )
        tgs.append( tg_guess[0] )
        ics.append( ic )
                
        
    #---------------------------------------#
    # Generate rubbery canidate ranking KDE #
    #---------------------------------------#
    # Setup the inputs
    distance_from_glassy = np.array( distance_from_glassy )
    bound_weights = np.array( bound_weights )
    uppers   = np.array( uppers )
    lowers   = np.array( lowers )
    widths   = uppers - lowers
    tgs      = np.array( tgs )
    slopes   = np.array( slopes )
    glassy_slope = glassy_coeffs[1]
    
    # Compute the KDE
    try:
        outputs = compute_gaussian_kde(temperature, tgs, ic_increment, peak_index, weighting, slopes, widths, bound_weights, glassy_slope)
        temp, tg_kde, xpeaks, ypeaks, peak_info, selected_tg, kernel_width = outputs
    except: # it is unlikely above will fail, but if it does set Tg at the center and warn the user
        temp, tg_kde, xpeaks, ypeaks, peak_info = None, None, None, None, None
        selected_tg = (max(temperature) - min(temperature))/2
        print('WARNING - the generation of the PDF failed, setting the selected Tg value at the center of the X-data')
        
    # Go and find closests Tg in guesses (we will update based on widest in group if possible)
    closests_ic = {ic: float( abs(selected_tg - tg_guess[0]) ) for ic, (passes, bounds, guessed_lr, tg_guess) in guesses.items()}
    rubbery_ic = min(closests_ic, key=closests_ic.get) # Find key of lowest value (minimized match of selected Tg)

    # Use closests IC from Tristans group elimation for rubbery domain
    passes, rubbery_bounds, rubbery_lr, intersection = guesses[rubbery_ic]
    rr1, rf1, rr2 = passes # Extract out rubbery passes and set colors (r=rubbery, f=forward or r=reverse, int=pass-number)
    rr1['color'] = 'tab:orange'
    rf1['color'] = 'tab:red'
    rr2['color'] = 'darkred'
    
    # Apply a linear regression in the rubbery region
    rubbery_coeffs = linear_regression(temperature, filtered_volume, xlo=min(rubbery_bounds), xhi=max(rubbery_bounds))
    rubbery_volume = rubbery_coeffs[1]*rubbery_bounds + rubbery_coeffs[0] # rubbery volume data from rubbery bounds (for plotting)
    
    
    #--------------------------------#
    # Compute the thermal properties #
    #--------------------------------#
    # Compute Tg
    tg, does_intersect = line_intersection(glassy_coeffs, rubbery_coeffs)
    
    # Average room temperature data to compute the CTE
    room_temp = 300
    delta     = 10  # Average room_temp += 10 K
    reduced_x, reduced_y = window(temperature, filtered_volume, xlo=room_temp-delta, xhi=room_temp+delta)
    
    # Compute CTE's based on room temperature average
    vol_avg     = np.mean(reduced_y)
    glassy_cte  = (1/(3*vol_avg))*glassy_coeffs[1]
    rubbery_cte = (1/(3*vol_avg))*rubbery_coeffs[1]
    if not plot_orientation:
        fig = None # set fig variable as a None return
        return tg, glassy_cte, rubbery_cte, fig
    
    
    #---------------------------------#
    # Plot the results of this method #
    #---------------------------------#
    # General plot settings
    fs = 12   # Set fontsize
    ms = 6.0  # General marker size (there maybe scaling of these in the plotting code)
    lw = 2.5  # General line-width (there maybe scaling of these in the plotting code)
    plot_dim = 4 # Sub-figure dimensions
    legend_fs_scale = 0.75 # scales legend font size
    label_rel_pos   = (0.02, 0.98)
    
    # Coloring
    tg_color      = '#ff9d3aff'
    raw_color     = '#bbbbbbff'
    filter_color  = 'dimgray'
    glassy_color  = '#0097A7'
    rubbery_color = 'tab:red'
    
    # Set axis label mapping (if not present code will default to LAMMPS column names)
    axis_labels = {} # {'Xdata-colum':'desired X-axis label', 'Ydata-colum':'desired Y-axis label'}
    axis_labels['xunits'] = 'K'
    axis_labels['yunits'] = '$Å^3$'
    
    # Extract axis labels
    xlabel = 'Temperature ({})'.format(axis_labels['xunits'])
    ylabel = 'Volume ({})'.format(axis_labels['yunits'])
    
    # Set-up plot orientation
    if plot_orientation == 'wide':
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(3.5*plot_dim, 2*plot_dim))
    elif plot_orientation == 'long':
        fig, ((ax1, ax4), (ax2, ax3), (ax5, ax6)) = plt.subplots(3, 2, figsize=(2.5*plot_dim, 3*plot_dim))
    else:
        raise Exception('ERROR plot_orientation="{}", but only "wide" and "long" are supported'.format(plot_orientation))
    
    
    # Plot ax1 data
    #--------------
    ax1.plot(temperature, volume,          '.', ms=ms, color=raw_color,    label='LAMMPS data')
    ax1.plot(temperature, filtered_volume, '-', lw=lw, color=filter_color, label='Filtered data')
    
    ax1.plot(glassy_bounds, glassy_volume,    marker='|', ls='--', ms=2*ms, lw=lw, color=glassy_color,  label='Glassy region\n * CTE={:.2e}'.format(glassy_cte))
    ax1.plot(rubbery_bounds, rubbery_volume,  marker='|', ls='--', ms=2*ms, lw=lw, color=rubbery_color, label='Rubbery region\n * CTE={:.2e}'.format(rubbery_cte))
    
    # Generate "v-connecting lines"
    vx = [glassy_bounds[-1], tg[0], rubbery_bounds[0]]
    vy = [glassy_volume[-1], tg[1], rubbery_volume[0]]
    ax1.plot(vx, vy, '--', lw=lw/2, color='black')
    ax1.plot(*tg, 'p', ms=2*ms, color=tg_color, mec='black', mew=1.2, label='$T_g$ ({:.2f})'.format(tg[0]))
    
    ax1.legend(loc='upper left', bbox_to_anchor=(0.1, 1), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
    ax1.set_xlabel(xlabel, fontsize=fs)
    ax1.set_ylabel(ylabel, fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs)
    ax1.text(*label_rel_pos, '(a)', transform=ax1.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')
    
    ax1.ticklabel_format(
    axis='y',
    style='sci',
    scilimits=(5, 5),   # force a 10^5 multiplier
    useOffset=False,
    useMathText=False,  # displays "1e5"
    )
    ax1.yaxis.get_offset_text().set_fontsize(0.8*fs)
    
    
    # Plot ax2 data
    #--------------
    ax2.plot(gf1['fringe'], gf1['r2'], '-', lw=lw, color=gf1['color'], label='Glassy: 1st pass (forward)')
    ax2.plot(gr1['fringe'], gr1['r2'], '-', lw=lw, color=gr1['color'], label='Glassy: 2nd pass (backward)')
    ax2.plot(gf2['fringe'], gf2['r2'], '-', lw=lw, color=gf2['color'], label='Glassy: 3rd pass (forward)')
    ax2.plot(gf1['max_fringe'], gf1['max_r2'], '<', ms=ms, mec=gf1['color'], mfc='white', mew=lw)#, label='max(1st pass)')
    ax2.plot(gr1['max_fringe'], gr1['max_r2'], '>', ms=ms, mec=gr1['color'], mfc='white', mew=lw)#, label='max(2nd pass)')
    ax2.plot(gf2['max_fringe'], gf2['max_r2'], '<', ms=ms, mec=gf2['color'], mfc='white', mew=lw)#, label='max(3rd pass)')
    
    ax2.plot(rr1['fringe'], rr1['r2'], '-', lw=lw, color=rr1['color'], label='Rubbery: 1st pass (backward)')
    ax2.plot(rf1['fringe'], rf1['r2'], '-', lw=lw, color=rf1['color'], label='Rubbery: 2nd pass (forward)')
    ax2.plot(rr2['fringe'], rr2['r2'], '-', lw=lw, color=rr2['color'], label='Rubbery: 3rd pass (backward)')
    ax2.plot(rr1['max_fringe'], rr1['max_r2'], '>', ms=ms, mec=rr1['color'], mfc='white', mew=lw)#, label='max(1st pass)')
    ax2.plot(rf1['max_fringe'], rf1['max_r2'], '<', ms=ms, mec=rf1['color'], mfc='white', mew=lw)#, label='max(2nd pass)')
    ax2.plot(rr2['max_fringe'], rr2['max_r2'], '>', ms=ms, mec=rr2['color'], mfc='white', mew=lw)#, label='max(3rd pass)')
    
    ax2.legend(loc='lower right', bbox_to_anchor=(1, 0), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
    ax2.set_xlabel(xlabel, fontsize=fs)
    ax2.set_ylabel('$R^2$', fontsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=fs)
    if plot_orientation == 'wide':
        ax2.text(*label_rel_pos, '(b)', transform=ax2.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')
    elif plot_orientation == 'long':
        ax2.text(*label_rel_pos, '(c)', transform=ax2.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')
        
    all_r2  = np.concatenate([gf1['r2'], gr1['r2'],  gf2['r2'], rr1['r2'], rf1['r2'], rr2['r2']])
    min_r2  = min(all_r2)
    max_r2  = max(all_r2)
    tol_r2  = 25/100 # scale by 10%
    span_r2 = max_r2 - min_r2
    ylo = min_r2 - tol_r2*span_r2*4.5
    yhi = max_r2 + tol_r2*span_r2
    ax2.set_ylim((ylo, yhi))
    
    
    # Plot ax3 data
    #--------------
    norm = mpl.colors.Normalize(vmin=min(ic_list), vmax=max(ic_list))
    cmap = plt.cm.coolwarm  # blue -> red
    zorders = []
    for ic, (tmp_passes, tmp_bounds, tmp_coeffs, tg_guess) in guesses.items():
        color = cmap(norm(ic))
        tmp_volume = tmp_coeffs[1]*tmp_bounds + tmp_coeffs[0] # tmp volume data from tmp bounds (for plotting)
        span = max(tmp_bounds) - min(tmp_bounds) 
        zorder = 1/span # Let smaller line segments have a larger zorder
        zorders.append(zorder)
        
        ax3.plot(tmp_bounds, tmp_volume,  marker='|', ls='-', ms=1.5*ms, lw=lw, color=color, zorder=zorder, alpha=1.0)
        ax3.plot(*tg_guess, 'p', ms=1.5*ms, mec='black', mew=1.2, zorder=100*zorder, color=color)
        
    ax3.plot(temperature, filtered_volume, '-', lw=1.5*lw, color=filter_color, zorder=min(zorders)/10, label='Filtered data')
    ax3.plot(glassy_bounds, glassy_volume, marker='|', ls='--', ms=2*ms, lw=lw, color=glassy_color, label='Glassy region')

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3)
    cbar.set_label('Initial-condition Temperature ({})'.format(axis_labels['xunits']), fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)
    
    ax3.legend(loc='lower right', bbox_to_anchor=(1, 0), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
    ax3.set_xlabel(xlabel, fontsize=fs)
    ax3.set_ylabel(ylabel, fontsize=fs)
    ax3.tick_params(axis='both', which='major', labelsize=fs)
    if plot_orientation == 'wide':
        ax3.text(*label_rel_pos, '(c)', transform=ax3.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')
    elif plot_orientation == 'long':
        ax3.text(*label_rel_pos, '(d)', transform=ax3.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')
        
    ax3.ticklabel_format(
    axis='y',
    style='sci',
    scilimits=(5, 5),   # force a 10^5 multiplier
    useOffset=False,
    useMathText=False,  # displays "1e5"
    )
    ax3.yaxis.get_offset_text().set_fontsize(0.8*fs)


    # Plot ax4 data
    #--------------
    if str(wn).startswith('op'):
        psd_color = 'tab:blue' #filter_color
        mean_color = '#ff9d3aff'
        wn_color = 'tab:green'
        ax4.stem(wns, psd, linefmt=psd_color, basefmt=psd_color, markerfmt='o', label='$|X(f)|^2/N$')
        ax4.axhline(mean_psd, color=mean_color, ls='--', lw=2, label='Average power={:.2f}'.format(mean_psd))
        ax4.plot(wn_frequency, wn_power, 'o', ms=8, color=wn_color , label='Cutoff frequency\n({:.4f}, {:.2f})'.format(wn_frequency, wn_power))
        
        ax4.set_xlabel('Normalized Frequencies (unitless)', fontsize=fs)
        ax4.set_ylabel('Power Spectral Density', fontsize=fs)
        ax4.legend(loc='upper right', bbox_to_anchor=(1, 1), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
        ax4.tick_params(axis='both', which='major', labelsize=fs)
        
        ax4.set_xlim((-0.0013, 0.0125)) # Comment/uncomment for xlimits
        ax4.set_ylim((-1*mean_psd, 25*mean_psd)) # Comment/uncomment for xlimits
        
        if plot_orientation == 'wide':
            ax4.text(*label_rel_pos, '(d)', transform=ax4.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')
        elif plot_orientation == 'long':
            ax4.text(*label_rel_pos, '(b)', transform=ax4.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')
        
        
    # Plot ax5 data
    #--------------
    #ax5.plot(ics, lowers, '-v', lw=lw, ms=ms, color='tab:blue',   label='Lower bound')
    #ax5.plot(ics, uppers, '-^', lw=lw, ms=ms, color='tab:red',    label='Upper bound')
    ax5.plot(ics, bound_penetrations, '-o', lw=lw, ms=ms, color='tab:blue', label='Bound Penetrations ({})'.format(axis_labels['xunits']))
    ax5.plot(ics, uppers-lowers, '-s', lw=lw, ms=ms, color='tab:grey', label='Rubbery widths ({})'.format(axis_labels['xunits']))
    ax5.plot(ics, tgs,    '-p', lw=lw, ms=ms, color='tab:purple', label='Possible $T_g$s ({})'.format(axis_labels['xunits']))
    ax5.axhline(selected_tg, color='black', ls='--', lw=lw, zorder=0.5, label='Selected Tg = {:.2f}'.format(selected_tg))
    
    ax5.set_xlabel('Initial-condition Temperature ({})'.format(axis_labels['xunits']), fontsize=fs, color='black')
    ax5.set_ylabel('Possible Rubbery states ({})'.format(axis_labels['xunits']), fontsize=fs, color='black')
    ax5.tick_params(axis='x', labelsize=fs, colors='black')
    ax5.tick_params(axis='y', labelsize=fs, colors='black')

    all_ks  = np.concatenate([bound_penetrations, uppers-lowers, tgs])
    min_ks  = min(all_ks)
    max_ks  = max(all_ks)
    tol_ks  = 55/100
    span_ks = max_ks - min_ks
    ylo = max([min_ks-tol_ks*span_ks, -10])
    yhi = max_ks + tol_ks*span_ks
    ax5.set_ylim((ylo, yhi))
    
    twin_color = 'tab:cyan'
    ax5_twin = ax5.twinx()
    ax5_twin.plot(ics, slopes, '-d', lw=lw, ms=ms, color=twin_color)
    ax5_twin.set_ylabel('Possible rubbery slopes ({}/{})'.format(axis_labels['yunits'], axis_labels['xunits']),
        fontsize=fs, color=twin_color)
    ax5_twin.tick_params(axis='y', labelsize=fs, colors=twin_color)
    
    if plot_orientation == 'wide':
        ax5.text(*label_rel_pos, '(e)', transform=ax5.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')
        handles, labels = ax5.get_legend_handles_labels()
        ax5_twin.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.1, 1), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
        
    elif plot_orientation == 'long':
        ax5.text(*label_rel_pos, '(e)', transform=ax5.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')
        handles, labels = ax5.get_legend_handles_labels()
        ax5_twin.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.075, 1), fancybox=True, ncol=2, fontsize=legend_fs_scale*fs*0.95)
    
    min_slopes = min(slopes)
    max_slopes = max(slopes)
    span_slopes = max_slopes - min_slopes
    tol_slopes  = 10/100
    ylo = min_slopes - tol_slopes*span_slopes
    yhi = max_slopes + tol_slopes*span_slopes*4.5
    ax5_twin.set_ylim((ylo, yhi))
    
    
    # Plot ax6 data
    #--------------
    if tg_kde is not None:
        # Ensure this is the largest dx-span in data to reduce number of points in plots for speed of rendering
        plot_dx = 0.5
        
        # Resample the KDE onto a uniform plotting grid that is smaller in dx to reduce size of plot
        temp_plot = np.arange(temp[0], temp[-1] + plot_dx, plot_dx)
        tg_kde_plot = np.interp(temp_plot, temp, tg_kde)
        
        ax6.plot(temp_plot, tg_kde_plot, lw=2, color='tab:purple', label='Convergence kernel\ndensity estimate (KDE)')
        ax6.fill_between(temp_plot, tg_kde_plot, alpha=0.25, color='tab:purple')
    if xpeaks is not None:
        ax6.plot(xpeaks, ypeaks, '*', ms=2*ms, mfc='tab:purple', mec='black', mew=1.2, label='KDE - Peaks')
        
        # Annotate peaks with selection criterion labels
        ymax = max(tg_kde)
        xrange = max(temp) - min(temp)
        ytext_hi = 1.40*ymax
        ytext_lo = 1.10*ymax
        max_labeled_peak = 3
        for n, info in enumerate(peak_info):
            label_rank = info.get('label_rank', n)
            if label_rank > max_labeled_peak and not info['selected']:
                continue
            
            # Alternate small left/right x-offsets
            xt = info['temp']
            yd = info['density']
            dx = (-1)**n * 0.05*xrange
            xtxt = xt + dx

            # Alternate between two label heights
            ytext = ytext_hi if n % 2 == 0 else ytext_lo
            if info['selected']:
                bbox = dict(boxstyle='round,pad=0.25', fc='white', ec='black', lw=1.2)
                arrow_color = 'black'
                weight = 'bold'
                
                base_color = 'black'
                base_lw = 1.2
                base_alpha = 0.8
                shade_alpha = 0.55
            else:
                bbox = dict(boxstyle='round,pad=0.25', fc='white', ec='black', lw=0.8, alpha=0.9)
                arrow_color = '0.35'
                weight = 'normal'
                
                base_color = '0.35'
                base_lw = 0.8
                base_alpha = 0.45
                shade_alpha = 0.03

            arrowprops = dict(arrowstyle='->', color=arrow_color, lw=1.0, shrinkA=0, shrinkB=4)
            ax6.annotate(info['label'], xy=(xt, yd), xytext=(xtxt, ytext), textcoords='data', ha='center', va='bottom',
                         fontsize=0.65*fs, fontweight=weight, bbox=bbox, arrowprops=arrowprops, zorder=10)
            
            # Peak bases
            lb = info['left_base']
            rb = info['right_base']
            xlb = temp[lb]
            xrb = temp[rb]
            ylb = tg_kde[lb]
            yrb = tg_kde[rb]
            #ax6.axvline(xlb, color=base_color, ls=':', lw=base_lw, alpha=base_alpha, zorder=3)
            #ax6.axvline(xrb, color=base_color, ls=':', lw=base_lw, alpha=base_alpha, zorder=3)

            ax6.fill_between(temp[lb:rb+1], tg_kde[lb:rb+1], alpha=shade_alpha, color='tab:purple', zorder=2)
            ax6.plot(xlb, ylb, '|', ms=1.8*ms, color=base_color, alpha=base_alpha, zorder=6)
            ax6.plot(xrb, yrb, '|', ms=1.8*ms, color=base_color, alpha=base_alpha, zorder=6)

        ax6.set_ylim(top=2.3*ymax)
        ax6.axvline(selected_tg, color='black', ls='--', lw=lw, label='Selected Tg = {:.2f}'.format(selected_tg))
        
        ax6.legend(loc='upper left', bbox_to_anchor=(0.1, 1), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
        ax6.set_xlabel("Possible $T_g$s ({})".format(axis_labels['xunits']), fontsize=fs)
        ax6.set_ylabel('Estimated probabilty density', fontsize=fs)
        ax6.tick_params(axis='both', which='major', labelsize=fs)
        if plot_orientation == 'wide':
            ax6.text(*label_rel_pos, '(f)', transform=ax6.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')
        elif plot_orientation == 'long':
            ax6.text(*label_rel_pos, '(f)', transform=ax6.transAxes, fontsize=fs, fontweight='bold', va='top', ha='left')

    fig.tight_layout()
    return tg, glassy_cte, rubbery_cte, fig