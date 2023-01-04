import numpy as np
import sys, os, glob, pickle, gc
import matplotlib.pyplot as plt
import numpy as np
import warnings
from run_search import run_search
from  astropy import time
from astropy.io import fits



n_cores = 5

v_min = 100. # Pixels/day
v_max = 620.

v_steps = 50 #for full range of rates
ang_steps = 15 #for half angles # 15 for full angles

num_obs = 10


chipNum = 0
visit = '2022-08-01'
ref_im_num = '2773118'

if len(sys.argv)>3:
    chipNum = int(float(sys.argv[1]))
    visit = sys.argv[2]
    ref_im_num = None if sys.argv[3]=='None' else '{:07d}'.format(int(float(sys.argv[3])))

"""
Instructions for Charles:

1. copy the DIFF*fits images to a directory. Have NO OTHER FILES in that
   directory, or KBmod will puke. For example, if you download the plantList
   files in the arc location, move them elsewhere.

2. set the string inside the () braces of im_filepath to the directory
   containing the DIF*fits images.

3. set the string inside the () braces of res_filepath to some place that you
   want to store the results. The entire path will be created if it doesn't
   exist.

4. set the string inside the () braces of time_file to the location of
   times_c00.dat which I attached in the email along with this code. Make sure
   the end of the path still reads as times_c00.dat! :)

5. Load the version of python that kbmod was compiled with, and source the
   kbmod/setup.bash file. If you haven't edited that file yet, the important
   entries are:

     export PYTHONPATH=$PWD/analysis:$PWD/search/pybinds:$PYTHONPATH
     # Disable python multiprocessing
     export OPENBLAS_NUM_THREADS=1
     export MKL_NUM_THREADS="1"
     export MKL_DYNAMIC="FALSE"
     export OMP_NUM_THREADS=1

   and comment out the rest of the setup contents.

6. Finally, run the program as > python  run_kbmod_CFHT.py !!!
"""


im_filepath=(
    "/media/fraserw/SecondStage/Projects/kbmod/DATA/rerun/diff_warpCompare/deepDiff/" +
    f"{visit}/warps/{str(chipNum).zfill(2)}/")

res_filepath=(
    f"/media/fraserw/rocketdata/Projects/kbmod/kbmod_results/{visit}/results_{str(chipNum).zfill(2)}")
results_suffix = ""
time_file=(
    f"/media/fraserw/rocketdata/Projects/kbmod/times_files/{visit}/times_c{str(chipNum).zfill(2)}.dat")


psf_val = 1.5
eps = 0.0008
sigmaG_lims = [25, 75]
mask_num_images = 20 ## should be a large fraction of the input images, ~2/3rds seems good
if visit == '2022-08-21':
    likelihood_limit = 5.
elif visit == '2022-08-22':
    likelihood_limit = 5.
elif visit == '2022-08-23':
    likelihood_limit = 5.
elif visit == '2022-08-24':
    likelihood_limit = 5.
elif visit == '2022-08-04':
    likelihood_limit = 5.
elif visit == '2022-08-05':
    likelihood_limit = 5.
    #psf_val = 2.5
    #eps = 0.0005
    #sigmaG_lims = [25, 75]
    #mask_num_images = 40
elif visit == '2022-08-01-ISISTEST':
    likelihood_limit = 5.
    #psf_val = 1.4
    #eps = 0.0016
elif visit == '2022-08-01':
    likelihood_limit = 5.



try:
    os.makedirs(res_filepath)
except:
    pass

ang_below = -np.pi + 0.35 #Angle below ecliptic
ang_above = np.pi + 0.35 # Angle above ecliptic

v_arr = [v_min, v_max, v_steps]
ang_arr = [ang_below, ang_above, ang_steps]

input_parameters = {
            'custom_bit_mask': {
                'BAD': 0, 'SAT': 1, 'INTRP': 2, 'EDGE': 4, 'DETECTED': 5,
                'DETECTED_NEGATIVE': 6, 'SUSPECT': 7, 'NO_DATA': 8, 'CROSSTALK': 9,
                'NOT_BLENDED': 10, 'UNMASKEDNAN': 11, 'BRIGHT_OBJECT': 12,
                'CLIPPED': 13, 'INEXACT_PSF': 14, 'REJECTED': 15,
                'SENSOR_EDGE': 16},
            'custom_flag_keys': ['EDGE', 'NO_DATA', 'SAT', 'INTRP', 'REJECTED'],
            'im_filepath':im_filepath,
            'res_filepath':res_filepath,
            'time_file':time_file,
            'output_suffix':results_suffix,
            'v_arr':v_arr,
            'ang_arr':ang_arr,

            'num_cores': n_cores,

            'num_obs':num_obs, # min number of individual frames include in stack of a candidate to call it a detection
            'do_mask':True, # check performance on vs. off
            'lh_level':likelihood_limit,
            'sigmaG_lims':sigmaG_lims, # maybe try [15,60]
            'mom_lims':[50.5,50.5,3.5,3.0,3.0],#[37.5,37.5,2.5,2.0,2.0],
            'psf_val': psf_val,
            'peak_offset':[3.0,3.0],
            'chunk_size':1000000,
            'stamp_type':'parallel_sum', #can be cpp_median or parallel_sum
            'do_stamp_filter': True,
            'do_clustering': True,
            'eps': eps,
            'cluster_type':'mid_position',
            'gpu_filter':True, #nominally True. on GPU lightcurve filter, verify that having this off makes things worse.
            'clip_negative':True,
            'sigmaG_filter_type':'both',
            'file_format': 'DIFFEXP-{:07d}-'+str(chipNum).zfill(2)+'.fits' if ref_im_num is None else  'DIFFEXP-{:07d}-'+ref_im_num+'-'+str(chipNum).zfill(2)+'.fits',
            'visit_in_filename':[8,15],
            'mask_num_images': mask_num_images,
            'chunk_start_index': 0,
            'chunks_to_consider': 40,
            }


rs = run_search(input_parameters)
rs.run_search()

del rs
gc.collect()
