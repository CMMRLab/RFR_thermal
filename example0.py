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




##########
# Inputs #
##########
# logfile to read in
logfile = 'logfiles/heating1/iso/EPON_862_rep_1_Tg_CTE_hrate_75_temps_175-875K.log.lammps'

# Plot orientation setup for different reasons. Here are the following options
#   plot_orientation = 'long' # makes plot long vertically (good for papers)
#   plot_orientation = 'wide' # makes plot wide vertically (good for presentations)
plot_orientation = 'long'
plot_orientation = 'wide'


weighting = 'bw*rw*rs'
peak_index = 0
min_span = 50
min_tg = None
max_tg = None




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
sections = '4-end'


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
order = 2
wn = 'op'
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
    keywords = [volume_column,temperature_column]
    log = read_log.file(logfile, keywords=keywords, pflag=False)
    data = log.get_data(sections, remove_duplicates=True, pflag=False) # {column-name:[list of data]}
    volume      = np.array(data[volume_column])
    temperature = np.array(data[temperature_column])
    
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
        
        # Find the mean value of the different PSD's
        mean_psd = np.mean(psd)
        
        # Find where each PSD crosses the mean
        wn_critical = wns[np.min(np.where(psd < mean_psd)[0])]
        print('{:<50} {}'.format('Computed normalized cutoff frequency for volume: ', wn_critical))
        
        # Compute the corresponding power value for the crossing of the mean
        power = psd[np.min(np.where(wns == wn_critical)[0])]
    
        # Filter the data with an optimized wn value
        filtered_volume, qm_volume = signals.butter_lowpass_filter(temperature, volume, wn_critical, order, quadrant_mirror)
    else:
        # Filter the data with a user defined wn value
        wn_critical = wn
        filtered_volume, qm_volume = signals.butter_lowpass_filter(temperature, volume, wn_critical, order, quadrant_mirror)
        
        
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
    tg_vals  = np.array( tgs )
    slope_vals = np.array( slopes )
    glassy_slope = glassy_coeffs[1]
    
    # Compute the KDE
    try:
        outputs = rfr.compute_gaussian_kde(temperature, tgs, ic_increment, peak_index, weighting, slopes, widths, bound_weights, glassy_slope)
        temp, tg_kde, tg_selection, xpeaks, ypeaks, peak_info, selected_tg, selected_label, kernel_width = outputs
    except: # it is unlikely above will fail, but if it does set Tg at the center and warn the user
        temp, tg_kde, tg_selection, xpeaks, ypeaks, peak_info = None, None, None, None, None, None
        selected_tg = (max(temperature) - min(temperature))/2
        selected_label = ''
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
    tg, does_intersect = rfr.line_intersection(glassy_coeffs, rubbery_coeffs)
    
    
    
    #---------------------------------#
    # Plot the results of this method #
    #---------------------------------#
    # General plot settings
    fs = 12   # Set fontsize
    ms = 5.0  # General marker size (there maybe scaling of these in the plotting code)
    lw = 2.5  # General line-width (there maybe scaling of these in the plotting code)
    plot_dim = 4 # Sub-figure dimensions
    legend_fs_scale = 0.8 # scales legend font size
    
    # Coloring
    ref_color     = 'tab:grey'
    tg_color      = '#ff9d3aff'
    exp_color     = '#EE3377'
    raw_color     = '#bbbbbbff'
    filter_color  = '#2c7fb8ff'
    glassy_color  = '#0097A7'
    rubbery_color = 'tab:red'
    
    filter_color  = 'black'
    
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
        fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2, figsize=(2.5*plot_dim, 3*plot_dim))
    else:
        raise Exception('ERROR plot_orientation="{}", but only "wide" and "long" are supported'.format(plot_orientation))
    
    
    # Plot ax1 data
    ax1.plot(temperature, volume,          '.', ms=ms, color=raw_color,    label='LAMMPS data')
    ax1.plot(temperature, filtered_volume, '-', lw=lw, color=filter_color, label='Filtered data')
    
    ax1.plot(glassy_bounds, glassy_volume,    marker='|', ls='--', ms=ms, lw=lw, color=glassy_color,  label='Glassy region')
    ax1.plot(rubbery_bounds, rubbery_volume,  marker='|', ls='--', ms=ms, lw=lw, color=rubbery_color, label='Rubbery region')
    
    # Generate "v-connecting lines"
    vx = [glassy_bounds[-1], tg[0], rubbery_bounds[0]]
    vy = [glassy_volume[-1], tg[1], rubbery_volume[0]]
    ax1.plot(vx, vy, '--', lw=lw/2, color='black')
    ax1.plot(*tg, 'p', ms=2*ms, color=tg_color, mec='black', mew=1.2, label='$T_g$ ({:.6f})'.format(tg[0]))
    
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
    ax1.set_xlabel(xlabel, fontsize=fs)
    ax1.set_ylabel(ylabel, fontsize=fs)
    ax1.tick_params(axis='both', which='major', labelsize=fs)
    
    
    # Plot ax2 data
    ax2.plot(gf1['fringe'], gf1['r2'], '-', lw=lw, color=gf1['color'], label='1st pass (forward)')
    ax2.plot(gr1['fringe'], gr1['r2'], '-', lw=lw, color=gr1['color'], label='2nd pass (reverse)')
    ax2.plot(gf2['fringe'], gf2['r2'], '-', lw=lw, color=gf2['color'], label='3rd pass (forward)')
    ax2.plot(gf1['max_fringe'], gf1['max_r2'], '<', ms=ms, mec=gf1['color'], mfc='white', mew=1.25*lw, label='max(1st pass)')
    ax2.plot(gr1['max_fringe'], gr1['max_r2'], '>', ms=ms, mec=gr1['color'], mfc='white', mew=1.25*lw, label='max(2nd pass)')
    ax2.plot(gf2['max_fringe'], gf2['max_r2'], '<', ms=ms, mec=gf2['color'], mfc='white', mew=1.25*lw, label='max(3rd pass)')
    
    ax2.plot(rr1['fringe'], rr1['r2'], '-', lw=lw, color=rr1['color'], label='1st pass (reverse)')
    ax2.plot(rf1['fringe'], rf1['r2'], '-', lw=lw, color=rf1['color'], label='2nd pass (forward)')
    ax2.plot(rr2['fringe'], rr2['r2'], '-', lw=lw, color=rr2['color'], label='3rd pass (reverse)')
    
    ax2.plot(rr1['max_fringe'], rr1['max_r2'], '>', ms=ms, mec=rr1['color'], mfc='white', mew=1.25*lw, label='max(1st pass)')
    ax2.plot(rf1['max_fringe'], rf1['max_r2'], '<', ms=ms, mec=rf1['color'], mfc='white', mew=1.25*lw, label='max(2nd pass)')
    ax2.plot(rr2['max_fringe'], rr2['max_r2'], '>', ms=ms, mec=rr2['color'], mfc='white', mew=1.25*lw, label='max(3rd pass)')
    
    #ax2.legend(loc='upper left', bbox_to_anchor=(0, 1), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
    ax2.set_xlabel(xlabel, fontsize=fs)
    ax2.set_ylabel('$R^2$', fontsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=fs)
    
    
    # Plot ax3 data
    ax3.plot(temperature, filtered_volume, '.', ms=ms, color=filter_color, alpha=0.35, label='LAMMPS data')
    ax3.plot(glassy_bounds, glassy_volume, marker='|', ls='--', ms=ms, lw=lw, color=glassy_color,  label='Glassy region')
    
    norm = mpl.colors.Normalize(vmin=min(ic_list), vmax=max(ic_list))
    cmap = plt.cm.coolwarm  # blue -> red
    for ic, (tmp_passes, tmp_bounds, tmp_coeffs, tg_guess) in guesses.items():
        color = cmap(norm(ic))
        tmp_volume = tmp_coeffs[1]*tmp_bounds + tmp_coeffs[0] # tmp volume data from tmp bounds (for plotting)
        ax3.plot(tmp_bounds, tmp_volume,  marker='|', ls='-', ms=1.5*ms, lw=lw, color=color, alpha=1.0)
        ax3.plot(*tg_guess, 'p', ms=1.5*ms, color=color)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3)
    cbar.set_label('Initial-condition Temperature ({})'.format(axis_labels['xunits']), fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)
    
    ax3.legend(loc='upper left', bbox_to_anchor=(0, 1), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
    ax3.set_xlabel(xlabel, fontsize=fs)
    ax3.set_ylabel(ylabel, fontsize=fs)
    ax3.tick_params(axis='both', which='major', labelsize=fs)
    

    # Plot ax4 data
    if str(wn).startswith('op'):
        psd_color = filter_color
        mean_color = '#ff9d3aff'
        wn_color = 'tab:green'
        ax4.stem(wns, psd, linefmt=psd_color, basefmt=psd_color, markerfmt='.', label='$|X(f)|^2/N$')
        ax4.plot(wn_critical, power, 'o', ms=8, color=wn_color , label='Critical frequency\n({:.4f}, {:.4f})'.format(wn_critical, power))
        ax4.axhline(mean_psd, color=mean_color, ls='--', lw=2, label='Mean power={:.4f}'.format(mean_psd))
        ax4.set_xlabel('Normalized Frequencies (unitless)', fontsize=fs)
        ax4.set_ylabel('Power Spectral Density', fontsize=fs)
        ax4.legend(loc='upper right', bbox_to_anchor=(1, 1), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
        ax4.tick_params(axis='both', which='major', labelsize=fs)
        ax4.set_xlim((-0.001, 0.0125)) # Comment/uncomment for xlimits
        ax4.set_ylim((-1*mean_psd, 25*mean_psd)) # Comment/uncomment for xlimits
        
        
    # Plot ax5 data
    ax5.plot(ics, lowers, '-v', lw=lw, ms=ms, color='tab:blue',   label='Lower bound')
    ax5.plot(ics, uppers, '-^', lw=lw, ms=ms, color='tab:red',    label='Upper bound')
    ax5.plot(ics, tgs,    '-p', lw=lw, ms=ms, color='tab:purple', label='Tg')
    
    ax5.set_xlabel('Initial-condition Temperature ({})'.format(axis_labels['xunits']), fontsize=fs, color='black')
    ax5.set_ylabel('Possible Rubbery states ({})'.format(axis_labels['xunits']), fontsize=fs, color='black')
    ax5.tick_params(axis='x', labelsize=fs, colors='black')
    ax5.tick_params(axis='y', labelsize=fs, colors='black')
    
    twin_color = 'tab:cyan'
    ax5_twin = ax5.twinx()
    ax5_twin.plot(ics, slopes, '-d', lw=lw, ms=ms, color=twin_color)
    ax5_twin.set_ylabel('Possible rubbery slopes ({}/{})'.format(axis_labels['yunits'], axis_labels['xunits']),
        fontsize=fs, color=twin_color)
    ax5_twin.tick_params(axis='y', labelsize=fs, colors=twin_color)
    
    handles, labels = ax5.get_legend_handles_labels()
    ax5_twin.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, 1), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
    
    # Plot ax6 data
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
        ytext_lo = 1.20*ymax
        max_labeled_peak = 3
        for n, info in enumerate(peak_info):
            label_rank = info.get('label_rank', n)
            if label_rank > max_labeled_peak and not info['selected']:
                continue
            
            xt = info['temp']
            yd = info['density']
            
            # Alternate small left/right x-offsets
            dx = (-1)**n * 0.02*xrange
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
            ax6.axvline(xlb, color=base_color, ls=':', lw=base_lw, alpha=base_alpha, zorder=3)
            ax6.axvline(xrb, color=base_color, ls=':', lw=base_lw, alpha=base_alpha, zorder=3)

            ax6.fill_between(temp[lb:rb+1], tg_kde[lb:rb+1], alpha=shade_alpha, color='tab:purple', zorder=2)
            ax6.plot(xlb, ylb, '|', ms=1.8*ms, color=base_color, alpha=base_alpha, zorder=6)
            ax6.plot(xrb, yrb, '|', ms=1.8*ms, color=base_color, alpha=base_alpha, zorder=6)

        ax6.set_ylim(top=2.25*ymax)
        ax6.axvline(selected_tg, color='black', ls='--', lw=lw, label='Selected Tg = {:.2f}'.format(selected_tg))
        
        ax6.legend(loc='upper left', bbox_to_anchor=(0, 1), fancybox=True, ncol=1, fontsize=legend_fs_scale*fs)
        ax6.set_xlabel("Possible $T_g$s ({})".format(axis_labels['xunits']), fontsize=fs)
        ax6.set_ylabel('Estimated probabilty density', fontsize=fs)
        ax6.tick_params(axis='both', which='major', labelsize=fs)

    
    fig.tight_layout()
    plt.show()
    # basename = logfile[:logfile.rfind('.')]
    # fig.savefig(basename+'.jpeg', dpi=300)