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
    * Dependencies:                                                *
    *   python matplotlib module:                                  *
    *    - pip3 install matplotlib (if pip manager is installed)   *
    *                                                              *
    *   python numpy module:                                       *
    *    - pip3 install numpy (if pip manager is installed)        *
    *                                                              *
    *   python scipy module:                                       *
    *    - pip3 install scipy (if pip manager is installed)        *
    ****************************************************************
    * Notes for Anaconda Spyder IDE users:                         *
    *   - If running this from the Anaconda Spyder IDE, before     *
    *     running you will have to turn on the interactive plot    *
    *     if you want the interactive plot to pop-up. You may do   *
    *     this by executing the following command in the console:  *
    *        %matplotlib qt                                        *
    *     Then to turn back on the inline plotting, where the plots*
    *     will appear in the Plots tab execute the following       *
    *     command in the console:                                  *
    *        %matplotlib inline                                    *
    ****************************************************************
"""
##############################
# Import Necessary Libraries #
##############################
import modules.read_log as read_log
import modules.signals as signals
import modules.rfr_thermal as rfr
import matplotlib.pyplot as plt
import matplotlib as mpl  
import numpy as np


# General plot settings
plt.close('all')
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




###############
# Data Inputs #
###############
# logfile to read in
logfile = 'logfiles/heating1/iso/EPON_862_rep_3_Tg_CTE_hrate_75_temps_175-875K.log.lammps'
logfile = 'logfiles/heating1/aniso/EPON_862_rep_3_Tg_CTE_hrate_75_temps_175-875K.log.lammps'

# Set the sections of the logfile to get data from. Sections are counted
# starting from 1 and end at N-sections. The following options are available
# in example format:
#    'all'    get data for all sections from logfile
#    '1'      get data for only section 1 from log file
#    '1,3'    get data for sections 1 and 3 from log file
#    '1-3'    get data for sections 1, 2, and 3 from log file
#    '1-3,5'  get data for sections 1, 2, 3, and 5 from log file
# *NOTE if reading multiple sections from the log file and a column
#  is missing in a certain section, the get_data function will create
#  zeros.*
sections = '4-end' # earlier section are setting up the temperature ramp

# Plot orientation setup for different reasons. Here are the following options
#   plot_orientation = 'long' # makes plot long vertically (good for papers)
#   plot_orientation = 'wide' # makes plot wide vertically (good for presentations)
plot_orientation = 'long'
#plot_orientation = 'wide'


######################
# RFR-thermal inputs #
######################
# KDE weighting expression. Available variables are:
#   bpw = bound-penetration weight
#   rww = normalized rubbery-region width weight
#   rsw = rubbery-region slope divided by the glassy-region slope
#   kw  = effective KDE kernel width
# Combine these variables with NumPy-compatible operations to weight each Tg
# candidate. Set weighting=None to compute an unweighted KDE.
weighting = None
weighting = 'bpw*rww*rsw'

# Select a peak after sorting PDF peak regions by decreasing integrated
# probability: 0 selects the most probable region, 1 the next most probable,
# and negative values use Python-style indexing.
peak_index = 0

# Minimum temperature span used when searching for candidate linear regions,
# in the same units as the temperature data. The IC-grid increment is set to
# half this value below.
min_span = 50

# Optional Tg limits, in the same units as the temperature data. Candidate Tg
# values outside these inclusive limits are skipped. Use None to disable a
# limit. The None assignment is shown as an example; the active lower limit is
# set to 350 below.
min_tg = 350
min_tg = None
max_tg = None


####################
# Filtering inputs #
####################
# Filter settings
# order=<int>, where <int> is the order of the low pass Butterworth filter. Examples:
#   order=2
#   order=3
#
# wn=<float or string>, where <float> is a floating point number between 0 and 1 to set critical frequency, which is the point at which the gain
# drops to 1/sqrt(2) that of the passband (the "-3 dB point"). If you want to control the "true cutoff frequency" instead of the "normalized
# cutoff frequency" you may compute "wn" in the following way:
#     wn = cutoff/nf; where "wn" is the input to the filter, "nf" is the Nyquist frequency, and "cutoff" is the cutoff frequency.
#     nf = 0.5*fs; where "fs" is the sampling frequency (1/delta-x).
#     wn = cutoff/(0.5*fs)
#
# If wn=<string>, the string can be 'op' where 'op' means optimize wn automatically by computing the Power Spectral Density (PSD) and finding where
# the power crosses the mean of the PSD. The following order values are recommend for wn='op':
#   wn='op' and order=2
# Examples:
#   wn=0.1
#   wn=0.5
#   wn=op
#
# quadrant_mirror=<string>, where <string> is in the format 'lo,hi', where lo and hi are integer values (i.e. '1,1', '2,1', ...). The purpose of 
# quadrant_mirror (qm) is to perform quadrant mirroring to reduce the transient edge effects of the start-up and shut-down cost of the convolution
# on either the forward or backwards pass. By mirroring the data at each end, the start-up and shut-down costs of the convolution can be "pushed"
# into data that does not matter. The data is mirrored prior to filtering and then sectioned back to the original data after filtering. The integer
# meanings for lo and hi are described below:
#   +---------+-----------------------------------------------------+-----------------------------------------------------+
#   | integer |                    lo X-data end                    |                    hi X-data end                    |
#   +---------+-----------------------------------------------------+-----------------------------------------------------+
#   |    1    | shuts off mirroring operation at "lo" X-data end    | shuts off mirroring operation at "hi" X-data end    |
#   |    2    | mirrors data "head-to-head" into quadrant 2         | mirrors data "tail-to-tail" and leaves Y-data alone |
#   |    3    | mirrors data "head-to-head" into quadrant 3         | mirrors data "tail-to-tail" and flips Y-data        |
#   |    4    | mirrors data "head-to-tail" and leaves Y-data alone | mirrors data "tail-to-head" and leaves Y-data alone |
#   +---------+-----------------------------------------------------+-----------------------------------------------------+
# When the MD stress-strain response has "slack" it is best to mirror the data into quadrant 2, and when the MD stress-strain response has "no
# slack" it is best to mirror the data into quadrant 3 for the "lo" integer. For the "hi" integer majority of the time the optimal integer is
# 2 and will be rare to deviate to other integers. Additionally the "-p" flag can be added to plot the quadrant mirroring of the data for
# inspection. The default is '1,1' and will be used if not specified. Examples:
#   qm=1,2
#   qm=2,2
#   qm=2,3-p
#   qm=1,4-p
# Additionally the string can be set to "msr" or "msr-p", where msr stands for minimum sum of residuals squared. The sum of residuals
# squared is determined for all permuations and combinations of lo and hi integer values; where which ever Butterworth filtered data has the
# minimum sum of residuals squared between permuations, is the "best quadrants" to use as the mirror (or to not mirror the data at all for
# quadrant 1). This is because the "edge" effects cause the low and high x-range residuals to be larger, when mirrored in the incorrect
# direction. Additionally the "-p" flag can be added to plot the optimized quadrant position and the quadrant mirroring of the data for
# inspection. Examples:
#   qm=msr
#   qm=msr-p
wn    = 'op'
order = 2
quadrant_mirror = 'msr'




#############################
# Read log and show example #
#############################
if __name__ == "__main__":
    
    #-----------------------------------------------------------------------#
    # Set column variables based on strain_direction (dependent on LAMMPS   #
    # logged variables - will need to update based on user setup in LAMMPS) #
    #-----------------------------------------------------------------------#
    volume_column = 'f_vol_ave'  # Time averaged volume
    temperature_column = 'v_ramp_temp'   # Ramp temperature (if you did not log this, use a numpy.linspace() to create it)
    
    
    #-------------------------------------------#
    # Read logfile and get numpy arrays of data #
    #-------------------------------------------#
    # Read logfile
    keywords = [volume_column,temperature_column]
    log = read_log.file(logfile, keywords=keywords, pflag=False)
    data = log.get_data(sections, remove_duplicates=True, pflag=False) # {column-name:[list of data]}
    
    # For cooling we need to reverse the lists
    volume      = data[volume_column]
    temperature = data[temperature_column]
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
    glassy_passes, glassy_bounds = rfr.rfr_multi_pass(temperature, filtered_volume, sequence='FBF', min_span=min_span, ic=None)
    gf1, gr1, gf2 = glassy_passes # Extract out glassy passes and set colors (g=glassy, f=forward or r=reverse, int=pass-number)
    gf1['color'] = 'tab:cyan'
    gr1['color'] = 'tab:blue'
    gf2['color'] = 'darkblue'
    
    # Apply a linear regression in the glassy region
    glassy_coeffs = rfr.linear_regression(temperature, filtered_volume, xlo=min(glassy_bounds), xhi=max(glassy_bounds))
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
    reduced_x, reduced_y = rfr.window(temperature, filtered_volume, xlo=max(glassy_bounds), xhi=max(temperature))
    
    # Start guess linear sections from different ICs
    xlo = max(glassy_bounds)
    guesses = {} # { float(ic): [ [rfr_pass1, ... rfr_passN], [lo_bound, hi_bound], regression, guessed_tg], ... Nic's }
    lowers, uppers, tgs, ics, slopes, bound_weights, distance_from_glassy, bound_penetrations = [], [], [], [], [], [], [], []
    for ic in ic_list:
        # Compute FBF or BFB from IC
        try:
            passes, bounds = rfr.rfr_multi_pass(reduced_x, reduced_y, sequence='BFB', min_span=min_span, ic=ic)
            if len(passes) != 3: continue # skip incomplete FBF or BFB searches
        except: continue # skip any failed attempts
    
        # Compute the linear regression and get the interseciton. Skip over meaningles results
        guessed_coeffs = rfr.linear_regression(temperature, filtered_volume, xlo=min(bounds), xhi=max(bounds))
        tg_guess, does_intersect = rfr.line_intersection(glassy_coeffs, guessed_coeffs)
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
        outputs = rfr.compute_gaussian_kde(temperature, tgs, ic_increment, peak_index, weighting, slopes, widths, bound_weights, glassy_slope)
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
    rubbery_coeffs = rfr.linear_regression(temperature, filtered_volume, xlo=min(rubbery_bounds), xhi=max(rubbery_bounds))
    rubbery_volume = rubbery_coeffs[1]*rubbery_bounds + rubbery_coeffs[0] # rubbery volume data from rubbery bounds (for plotting)
    
    
    #--------------------------------#
    # Compute the thermal properties #
    #--------------------------------#
    # Compute Tg
    tg, does_intersect = rfr.line_intersection(glassy_coeffs, rubbery_coeffs)
    
    # Average room temperature data to compute the CTE
    room_temp = 300
    delta     = 10  # Average room_temp += 10 K
    reduced_x, reduced_y = rfr.window(temperature, filtered_volume, xlo=room_temp-delta, xhi=room_temp+delta)
    
    # Compute CTE's based on room temperature average
    vol_avg     = np.mean(reduced_y)
    glassy_cte  = (1/(3*vol_avg))*glassy_coeffs[1]
    rubbery_cte = (1/(3*vol_avg))*rubbery_coeffs[1]
    
    
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
    axis_labels[volume_column] = 'Volume'
    axis_labels[temperature_column] = 'Temperature'
    
    # Extract axis labels
    xlabel = '{} ({})'.format(axis_labels.get(temperature_column, temperature_column), axis_labels['xunits'])
    ylabel = '{} ({})'.format(axis_labels.get(volume_column, volume_column), axis_labels['yunits'])
    
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
    plt.show()
    basename = logfile[:logfile.rfind('.')]
    figname  = '{}_{}.jpeg'.format(basename, plot_orientation)
    fig.savefig(figname, dpi=300)