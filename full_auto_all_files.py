# -*- coding: utf-8 -*-
"""
@author: Josh Kemppainen
Revision 1.0
November 3, 2026
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
import modules.rfr_thermal as rfr
import matplotlib.pyplot as plt
import numpy as np
import traceback
import difflib
import glob
import time
import os

plt.rcParams.update({'figure.max_open_warning': 0}) # Will suppress warnings about number of figures
from matplotlib.backends.backend_pdf import PdfPages


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
# logfiles to read in using wildcard matching. See second logfole variable
# for how to set wildcars.
logfile = 'logfiles/heating1/aniso/PBZ_rep_*_Tg_CTE_hrate_75_temps_175-875K.log.lammps'
logfile = 'logfiles/*/*/*_rep_*_Tg_CTE_*_*_temps_*-*.log.lammps'


# Name for curated results (.csv for tg and CTEs and .pdf for all plots)
array_file = 'full_auto_rfr_thermal'


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




#################################
# Wildcard extraction functions #
#################################
# Function to convert string to float or int and will default to string if needed
def string2digit(string):
    digit = string
    try: 
        digit = float(string)
        if digit.is_integer():
            digit = int(digit)
    except: pass
    return digit


# Function to get wildcards between a glob_string and a string glob matched with. Inspired from:
# https://stackoverflow.com/questions/17904097/python-difference-between-two-strings
def get_glob_wildcards(glob_string, matched_string):
    # Force both strings to follow the same format rules
    glob_string = os.path.normpath(glob_string)
    matched_string = os.path.normpath(matched_string)
    
    # Find all strings that diff sets as adds "+"
    wildcards = []; add = False; tmp = ''
    for i, s in enumerate(difflib.ndiff(glob_string, matched_string)):
        if s[0] == ' ':
            add = False
        if s[0] == '-' or s[0] == ' ':
            if tmp != '': wildcards.append(tmp)
            add = False; tmp = ''
        if s[0] == '+':
            add = True 
        if add:
            tmp += s[-1]
    if tmp != '': wildcards.append(tmp)
            
    # Try converting from a string to float or int
    wildcards = [string2digit(i) for i in wildcards]# if i not in glob_string]
    return wildcards




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
    
    
    #-------------------------------------#
    # Start full-auto wildcard processing #
    #-------------------------------------#
    files = glob.glob(logfile)
    try: # Try sorting files
        from natsort import natsorted
        files = natsorted(files)
        print('Read in logfiles were sorted using natsort, thus they should be iterated through')
        print('in a sequence that represents evolution or progression and logged in that order.')
    except:
        files = sorted(files)
        print('WARNING natsort was not installed and read in logfiles were sorted using pythons')
        print('sorted function. The file iteration sequence may be unorder from a evolution or')
        print('progression stand point. Depending on the names given to the read in files. You may')
        print('intall natsort with:  pip3 install natsort   (if pip manager is installed)')
    if not files:
        print('WARNING logfile="{}" did not match with any files. Skipping Full-Auto run.'.format(logfile))
    if files:
        # Full-Auto run
        #--------------
        array_data = {} # {'filename':[[wildcards], tg, glassy_cte, rubbery_cte]}
        figures = {} # {'filename': fig_instance}
        processed_files, failed_files = [], []
        loop_time = time.time()
        for n, file in enumerate(files, 1):
            print('\n\nStarting analysis:')
            print(' - logfile                        : {}'.format(logfile))
            print(' - matched file                   : {}'.format(file))
            print(' - elapsed time                   : {:.2f} (seconds)'.format(time.time() - loop_time))
            print(' - progress                       : {} of {} ({:.2f}%)'.format(n, len(files), 100*(n/len(files))))
            
            try: # There are various reasons this could crash (no file, no data ....)
                # Get wildcards for logging
                wildcards = get_glob_wildcards(logfile, file)
                
                # Read logfile
                keywords = [volume_column,temperature_column]
                log = read_log.file(file, keywords=keywords, pflag=False)
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
                
                # Analyze the volume-temperature data
                tg, glassy_cte, rubbery_cte, fig = rfr.full_rfr_thermal(temperature, volume, wn, order, quadrant_mirror, 
                                                                        min_span, min_tg, max_tg, weighting, peak_index,
                                                                        plot_orientation='wide')
                processed_files.append(file)
                print('Successfully processed')
            except: 
                wildcards, tg, glassy_cte, rubbery_cte, fig = [], [None, None], None, None, None
                failed_files.append(file)
                
                stack_trace_string = traceback.format_exc()
                print('Failed to be processed\n', stack_trace_string)
            
            # Log the results
            array_data[file] = [wildcards, {'tg':tg[0], 'glassy_cte':glassy_cte, 'rubbery_cte':rubbery_cte}]
            figures[file] = fig
            
            
        # Let user know which ones failed
        #--------------------------------
        ns = len(processed_files)
        nf = len(files)
        percentage = 100*ns/nf
        print('Sucessfully processed {} of {} ({:.2f} %)'.format(ns, nf, percentage))
        if failed_files:
            print('\n\nFailed files for unknown reasons:')
            for file in failed_files:
                print('- {}'.format(file))
        
        
        # Log all results to .csv file
        #-----------------------------
        # Find all columns to log
        wildcard_columns, thermal_columns = set(), set()
        for file, (wildcards, thermal) in array_data.items():

            # Set wildcard columns
            for n, wildcard in enumerate(wildcards):
                wildcard_columns.add('wildcards[{}]'.format(n))
                
            # Set thermal columns
            for key in thermal:
                thermal_columns.add(key)
            
        # Set None defaults
        columns  = sorted(wildcard_columns) + sorted(thermal_columns)
        csv_data = {} # {'filename':{'wildcards[0]':'wildcard', 'tg':value, ... Nproperties},  ... Nfiles}
        for file in array_data:
            csv_data[file] = {i:None for i in columns}
        
        # Add data to columns
        for file, (wildcards, thermal) in array_data.items():
            # Upate wildcards
            for n, wildcard in enumerate(wildcards):
                name = 'wildcards[{}]'.format(n)
                csv_data[file][name] = wildcard
                
            # Update thermal properties
            for key, value in thermal.items():
                csv_data[file][key] = value
                
        # Write results to csv file
        csvfile = '{}.csv'.format(array_file)
        try:
            print('Writing "{}" ...'.format(csvfile))
            with open(csvfile, 'w') as f:
                # Join with comma's and write titles
                titles = ', '.join(['filename'] + columns)
                f.write('{}\n'.format(titles))
                
                # Write rows
                for file, data in csv_data.items():
                    outputs = [str(file)]
                    for column in columns:
                        outputs.append( str(data.get(column, None)) )
                    f.write('{}\n'.format(', '.join(outputs)))
        except: 
            stack_trace_string = traceback.format_exc()
            print('Failed to write "{}". Most likely cause is another program has it open'.format(csvfile))
            print(stack_trace_string)
        

        # Put all figures in a single .pdf file
        #--------------------------------------
        pdffile = '{}.pdf'.format(array_file)
        try:
            pdf = PdfPages(pdffile)
            print('Writing "{}" ...'.format(pdffile))
            for n, (filename, fig) in enumerate(figures.items()):
                try:
                    fig.suptitle(filename, fontsize=12)
                    fig.tight_layout()
                    pdf.savefig(fig)
                except: 
                    stack_trace_string = traceback.format_exc()
                    print(f'\nAn error occurred: {stack_trace_string}')
                finally:
                    # free memory now
                    if n + 1 < len(figures): plt.close(fig)
        except: 
            stack_trace_string = traceback.format_exc()
            print('Failed to write "{}". Most likely cause is another program has it open'.format(pdffile))
            print(stack_trace_string)
        finally:
            pdf.close()
            figures.clear()
            
    # Full run time
    #--------------
    execution_time = (time.time() - loop_time)
    log.out('\n\nArray time in seconds: ' + str(execution_time))