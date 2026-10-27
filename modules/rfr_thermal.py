# -*- coding: utf-8 -*-
"""
@author: Josh Kemppainen
Revision 1.0
March 24, 2025
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
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import gaussian_kde
import numpy as np


# RFR compute
def compute_fringe_slope_R2(x_data, y_data, min_x_data=None, max_x_data=None, direction='forward'):
    """
    Compute fringe slope vectorized

    out returns other internally computed values

    For more information read (and Cite)
    https://doi.org/10.1038/s44172-026-00669-6

    :param x_data: Strain
    :type x_data: Array1D
    :param y_data: Stress
    :type y_data: Array1D
    :param min_x_data: Minimum x_data, defaults to None to use minimum index
    :type min_x_data: Optional(Number)
    :param max_x_data: Maximum x_data, defaults to None to use maximum index
    :type max_x_data: Optional(Number)
    :param direction: Direction, Must be 'forward' or 'backward', flips both vectors, defaults to 'forward'.
    :type direction: str

    :return: Fringe  X, slopes, and R2
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
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
    if xlo is None: xlo = np.nanmin(x)
    if xhi is None: xhi = np.nanmax(x)
    in_window = (x >= xlo) & (x <= xhi)  # NaN samples compare False -> dropped here
    win_x, win_y = x[in_window], y[in_window]
    
    return win_x, win_y


# The RFR-mulit pass for finding linear region
def rfr_multi_pass(x_data, y_data, sequence='FBF', min_span=0, ic=None):
    
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
    reduced_x, reduced_y = window(x, y, xlo=xlo, xhi=xhi)
    regression_coeffs = np.polynomial.polynomial.polyfit(reduced_x, reduced_y, 1) # [intercept, slope]
    return regression_coeffs

# Compute the intersection between to regression coeffs
def line_intersection(coeffs1, coeffs2):
    b1, m1 = coeffs1
    b2, m2 = coeffs2
    does_intersect = True
    if m1 == m2:
        does_intersect = False
        #raise ValueError("The lines are parallel or identical; no unique intersection.")

    x_intersect = (b2 - b1) / (m1 - m2)
    y_intersect = b1 + m1 * x_intersect

    return [x_intersect, y_intersect], does_intersect

# Compute the KDE and set Tg based on the selection method
def compute_gaussian_kde(xdata, tgs, ic_increment, peak_index, weighting, slopes, widths, bound_weights, glassy_slope, kernel_width=None):
    if kernel_width is None:
        kernel_effective_width = ic_increment
    else:
        kernel_effective_width = kernel_width
    tg_selection = 'prob'
    selected_label = ''

    # A KDE requires at least two different Tg values.
    if len(tgs) < 2 or np.std(tgs, ddof=1) == 0:
        selected_tg = tgs[0] if len(tgs) else (max(xdata) + min(xdata)) / 2
        print('WARNING - not enough distinct Tg values to compute a Gaussian KDE.')
        return (None, None, tg_selection, np.array([]), np.array([]), [],
                selected_tg, selected_label, kernel_effective_width)

    # Convert the requested kernel width (approximately ±3σ) to gaussian_kde's
    # bandwidth scaling factor.
    tg_std = np.std(tgs, ddof=1)
    kernel_std = kernel_effective_width / 6.0
    bw_method = kernel_std / tg_std

    # Build the PDF evaluation grid, clamped to the xdata range.
    xmin, xmax = min(tgs), max(tgs)
    padding = max(0.05 * (xmax - xmin), 4 * kernel_std)
    xmin_pdf = max(min(xdata), xmin - padding)
    xmax_pdf = min(max(xdata), xmax + padding)

    dx_pdf = 0.01
    temp = np.arange(xmin_pdf, xmax_pdf + dx_pdf, dx_pdf)
    
    # Compute optional weights for the Gaussian KDE. The user may supply an
    # arbitrary NumPy-compatible mathematical expression through:
    #
    #     weights = "<expression>"
    #
    # where the expression is evaluated element-wise and must return one weight
    # per candidate Tg. The following variables are available:
    #
    #   rw : Normalized rubbery-region width
    #        rw = widths / max(widths)
    #
    #   rs : Rubbery-region slope normalized by the glassy slope
    #        rs = ctes / glassy_slope
    #
    # Supported NumPy functions are available through the np namespace.
    #
    # Example expressions:
    #
    #     weights = "rw"
    #     weights = "rs"
    #     weights = "0.5*rw + 0.5*rs"
    #     weights = "rw**2"
    #     weights = "np.sqrt(rw)"
    #     weights = "np.exp(-rs)"
    weights = None
    if weighting is not None:
        # Compute possible weighting options, where the following are supported
        #  rw = rubbery width normalized to one (divided by maximum width)
        #  rs = rubbery slope divided by glassy slope
        rw = widths/max(widths)   # Set width-weights via a normalized width of regions
        rs = slopes/glassy_slope  # Set slope-weights via a normalized cte relative to glassy slope
        bw = bound_weights        # Set bound penalty where no-penetration=1 and larger-penetraions approach zero
        kw = kernel_effective_width # Set scalar value for kernel width
        
        # Setup the globals namespace to limit scope of what eval() can do
        allowed_builtins = ['min','max','sum','abs','len','map','range','reversed']
        copied_builtins = globals()['__builtins__'].copy()
        globals_dict = {'rw':rw, 'rs':rs, 'kw': kw, 'bw':bw}
        globals_dict['__builtins__'] = {key:copied_builtins[key] for key in allowed_builtins}
        globals_dict['np'] = globals()['np']

        # Try computing weights with vectorized numpy and eval
        try:
            weights = np.asarray(eval(weighting, globals_dict), dtype=float)
            weights = np.maximum(weights, 0)   # clamp negatives to zero
            weights = np.minimum(weights, 1e4) # clamp large values to 1e4
            weights = np.nan_to_num(weights, nan=1.0) # set Nan's to 1.0
            
            # If all weights evaluate to zero reset weights to ones (this is
            # useful to shut-off weighting with zeros like "0.0*rw + 0.0*rs",
            # but still keep the syntax for setting weights for testing)
            weight_set = set(weights)
            if len(weight_set) == 1 and sum(weight_set) == 0:
                weights = np.ones_like(weights)
                print(weights)
            
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

    xpeaks = temp[peaks]
    ypeaks = tg_kde[peaks]

    # Partition the PDF at the local minima between neighboring peaks.
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
    selected_label = f'PDF(prob,{peak_index})'

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

    return (temp, tg_kde, tg_selection, xpeaks, ypeaks, peak_info,
            selected_tg, selected_label, kernel_effective_width)