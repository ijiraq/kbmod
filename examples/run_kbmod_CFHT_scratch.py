import numpy as np
import sys, os, glob, gc
import numpy as np
from  astropy import time
from astropy.io import fits
from astropy import units
import argparse
import shutil
import logging


def create_times_file(img_path, file_fn, pattern="DIFFEXP*.fits"):
    """
    Get list of MJD at mid exposure for all files in img_path matching pattern and store in file_fn.

    """

    files = glob.glob(os.path.join(img_path, pattern))
    files.sort()
    logging.debug(f"Getting MJD for file list: {files} and storing to {file_fn}")
    with open(file_fn, 'w') as outhan:
        for fn in files:
            with fits.open(fn) as hdulist:
                mjd = hdulist[0].header['MJD-OBS']
                exptime = hdulist[0].header['EXPTIME']*units.second
                t = time.Time(mjd, format='mjd') + exptime/2.0
                expnum = int(os.path.basename(fn).split('-')[1])
                # mjd += exptime/(24.*3600.)
                # t = time.Time(mjd+2400000.5, format='jd')
                print(f'{expnum} {t.mjd:12.6f}', file=outhan)
                
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('visit', type=time.Time, help="The visit date (e.g. 2022-08-01)")
    parser.add_argument('chip', type=int, help="CCD to process")
    parser.add_argument('--reference-expnum', default=None, type=int, help="Reference CCD number")
    parser.add_argument('--input-path', default=os.getcwd(), type=str, help="Directory with input DIFFEXP files, store results here")
    parser.add_argument('--results-suffix', default="", type=str, help="Suffix to put on result file")
    parser.add_argument('--likelihood', default=5.0, type=float, help="Likelihood threshold for detection")
    parser.add_argument('--scratch-path', default="/scratch", help="Base directory for scratch files on container")
    parser.add_argument('--dry-run', default=False, action='store_true', help="Just test to see if script builds")
    parser.add_argument('--log-level', choices=['INFO', 'ERROR', 'DEBUG'], default='ERROR')
    parser.add_argument('--mask-flags', nargs='*', choices=['BRIGHT_OBJECT','CLIPPED', 'CR', 'DETECTED', 'DETECTED_NEGATIVE', 'EDGE',
                                                            'INEXACT_PSF',
                                                            'INTRP', 
                                                            'NOT_DEBLENDED', 'NO_DATA', 'REJECTED', 'SAT',
                                                            'SENSOR_EDGE', 'SUSPECT', 'UNMASKEDNAN'],
                        default=['EDGE', 'NO_DATA', 'SAT', 'INTRP', 'REJECTED'])

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    visit = args.visit.isot.split('T')[0]
    chipNum = args.chip

    ref_im_num = args.reference_expnum
    file_format_parts = ["DIFFEXP", "{:07d}"]
    if ref_im_num is not None:
        file_format_parts.append(f"{ref_im_num:07d}")
    file_format_parts.append(f"{chipNum:02d}")
    file_format_parts.append("0000")
    file_format_parts.append("0800")
    file_format_parts.append("0000")
    file_format_parts.append("0800")
    file_format = "-".join(file_format_parts)+".fits"

    logging.debug(f"File format string: {file_format}")
    res_filepath = args.input_path  # this is also where the DIFFEXP files we were process are located.
    if not os.access(res_filepath, os.R_OK & os.X_OK):
        raise FileNotFoundError(res_filepath)
    
    # scratch space on container (copy data here before processsing) 
    scratch_path = args.scratch_path  # this is where we will copy DIFFEXP files to

    if not os.access(scratch_path, os.R_OK & os.W_OK & os.X_OK):
        raise FileNotFoundError(scratch_path)
    im_filepath=(os.path.join(scratch_path, 'kbmod', f'{visit}', f'{chipNum:03d}'))
    os.makedirs(im_filepath, exist_ok=True)
    
    results_suffix = args.results_suffix
    likelihood_limit = args.likelihood

    time_file=os.path.join(res_filepath, '../times.dat')    

    # copy the DIFF files to the scratch area.
    logging.debug(f"copying DIFFEXP files from {res_filepath} to {im_filepath}")
    for filename in glob.glob(os.path.join(res_filepath,"DIFFEXP*.fits")):
        logging.debug(f"Copying {filename} to {im_filepath}")
        if not os.access(os.path.join(im_filepath, os.path.basename(filename)), os.F_OK):
            shutil.copy2(filename, im_filepath)

    # get the bit mask values from those requested by user, just use the last file copied over
    flag_hdu = fits.open(filename)[2]
    custom_flag_keys = []
    custom_flag_mask = {}
    for keyword in args.mask_flags:
        logging.debug(f"Looking up bit value for {keyword} in {filename}")
        bit = flag_hdu.header.get(f'MP_{keyword}', None)
        if bit is None:
            logging.warning(f"Keyworkd {keyword} not found in mask for {filename}")
            continue
        custom_flag_mask[keyword] = bit
        custom_flag_keys.append(keyword)
    custom_flag_mask = None

    # create the file containing the list of times.
    create_times_file(im_filepath, time_file)

    n_cores = 1  # How many cores to request
    v_min = 100. # minimum rate of motion pixels/day
    v_max = 620. # maximum rate of motion pixels/day
    v_steps = 50 # set to 50 for full range of rates
    ang_steps = 15 # ?? for half angles # 15 for full angles
    num_obs = 10 # minimum number of images reqquired for source detection
    psf_val = 1.5
    mask_num_images = 20
    sigmaG_lims = [25,75]
    eps = 0.0008
    ang_below = -np.pi + 0.35 # Angle below ecliptic
    ang_above = np.pi + 0.35 # Angle above ecliptic
    v_arr = [v_min, v_max, v_steps]
    ang_arr = [ang_below, ang_above, ang_steps]

    input_parameters = {
        'custom_bit_mask': custom_flag_mask,
        'custom_flag_keys': custom_flag_keys,
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
        'sigmaG_lims':[25,75], # maybe try [15,60]
        'mom_lims':[50.5,50.5,3.5,3.0,3.0],#[37.5,37.5,2.5,2.0,2.0],
        'psf_val':1.5,
        'peak_offset':[3.0,3.0],
        'chunk_size':1000000,
        'stamp_type':'parallel_sum', #can be cpp_median or parallel_sum
        'eps':0.0008,
        'do_clustering': True,
        'gpu_filter':True, #nominally True. on GPU lightcurve filter, verify that having this off makes things worse.
        'clip_negative':True,
        'sigmaG_filter_type':'both',
        'file_format': file_format,
        'visit_in_filename':[8,15],
        'cluster_type':'mid_position',
        'mask_num_images': mask_num_images,
        'chunk_start_index': 0,
        'chunks_to_consider': 40,
    }

    logging.debug(f"input_parameters:")
    for key in input_parameters:
        logging.debug(f"{key} : {input_parameters[key]}")
        
    if args.dry_run:
        sys.exit()
    else:
        run(input_parameters, scratch_path)

def run(input_parameters, scratch_path):
    # from create_stamps import CreateStamps, CNNFilter, VisualizeResults, load_stamps
    from run_search import run_search
    rs = run_search(input_parameters)
    rs.run_search()
    del rs
    gc.collect()
    shutil.rmtree(scratch_path, ignore_errors=True)


if __name__ == '__main__':
    main()
