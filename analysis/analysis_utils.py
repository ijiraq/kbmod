import os
import sys
import pdb
import shutil
import pandas as pd
import numpy as np
import mpmath
import time
import multiprocessing as mp
import csv
import astropy.coordinates as astroCoords
import astropy.units as u
import heapq
import pickle
from kbmodpy import kbmod as kb
from astropy.io import fits
from astropy.wcs import WCS
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from skimage import measure
from collections import OrderedDict

class SharedTools():
    """
    This class manages tools that are shared by the classes Interface and
    PostProcess.
    """
    def __init__(self):

        return

    def gen_results_dict(self):
        """
        Return an empty results dictionary. All values needed for a results
        dictionary should be added here. This dictionary gets passed into and
        out of most Interface and PostProcess methods, getting altered and/or
        replaced by filtering along the way.
        """
        keep = {'stamps': [], 'new_lh': [], 'results': [], 'times': [],
                'lc': [], 'lc_index':[], 'all_stamps':[], 'psi_curves':[],
                'phi_curves':[], 'final_results': ...}
        return(keep)

class Interface(SharedTools):
    """
    This class manages the KBMOD interface with the local filesystem, the cpp
    KBMOD code, and the PostProcess python filtering functions. It is
    responsible for loading in data from .fits files, initializing the kbmod
    object, loading results from the kbmod object into python, and saving
    results to file.
    """
    def __init__(self):

        return


    def return_filename(self, visit_num, file_format):
        """
        This function returns the filename of a single fits file that will be
        loaded into KBMOD.
        INPUT-
            visit_num : int
                The visit ID of the visit to be ingested into KBMOD.
            file_format : string
                An unformatted string to be passed to return_filename(). When
                str.format() passes a visit ID to file_format, file_format
                should return the name of a file corresponding to that visit
                ID.
        OUTPUT-
            fits_file : string
                The file name for a visit given by visit_num
        """
        fits_file = file_format.format(int(visit_num))
        return(fits_file)

    def get_folder_visits(self, folder_visits, visit_in_filename):
        """
        This function generates the visit IDs for the search using the folder
        containing the visit images.
        INPUT-
            folder_visits : string
                The path to the folder containing the visits that KBMOD will
                search over.
            visit_in_filename : int list
                A list containg the first and last character of the visit ID
                contained in the filename. By default, the first six characters
                of the filenames in this folder should contain the visit ID.
        OUTPUT-
            patch_visit_ids : numpy array
                A numpy array containing the visit IDs for the files contained
                in the folder given by folder_visits.
        """
        start = visit_in_filename[0]
        end = visit_in_filename[1]
        patch_visit_ids = np.array([str(visit_name[start:end])
                                    for visit_name in folder_visits])
        return(patch_visit_ids)

    def load_images(
        self, im_filepath, time_file, mjd_lims, visit_in_filename,
        file_format):
        """
        This function loads images and ingests them into a search object.
        INPUT-
            im_filepath : string
                Image file path from which to load images
            time_file : string
                File name containing image times
            mjd_lims : int list
                Optional MJD limits on the images to search.
            visit_in_filename : int list
                A list containg the first and last character of the visit ID
                contained in the filename. By default, the first six characters
                of the filenames in this folder should contain the visit ID.
            file_format : string
                An unformatted string to be passed to return_filename(). When
                str.format() passes a visit ID to file_format, file_format
                should return the name of a vile corresponding to that visit
                ID.
        OUTPUT-
            search : kbmod.stack_search object
            image_params : dictionary
                Contains the following image parameters:
                Julian day, x size of the images, y size of the images,
                ecliptic angle of the images.
        """
        image_params = {}
        print('---------------------------------------')
        print("Loading Images")
        print('---------------------------------------')
        visit_nums, visit_times = np.genfromtxt(time_file, unpack=True)
        image_time_dict = OrderedDict()
        for visit_num, visit_time in zip(visit_nums, visit_times):
            image_time_dict[str(int(visit_num))] = visit_time
        patch_visits = sorted(os.listdir(im_filepath))
        patch_visit_ids = self.get_folder_visits(patch_visits,
                                                 visit_in_filename)
        patch_visit_times = np.array([image_time_dict[str(int(visit_id))]
                                      for visit_id in patch_visit_ids])
        if mjd_lims is None:
            use_images = patch_visit_ids
        else:
            visit_only = np.where(((patch_visit_times >= mjd_lims[0])
                                   & (patch_visit_times <= mjd_lims[1])))[0]
            print(visit_only)
            use_images = patch_visit_ids[visit_only].astype(int)

        image_params['mjd'] = np.array([image_time_dict[str(int(visit_id))]
                                        for visit_id in use_images])
        times = image_params['mjd'] - image_params['mjd'][0]
        file_name = self.return_filename(int(use_images[0]),file_format)
        file_path = os.path.join(im_filepath,file_name)
        hdulist = fits.open(file_path)
        wcs = WCS(hdulist[1].header)

        ### get the bit mask directly from the header and store in image_params
        mask_header = hdulist[2].header
        default_bit_mask = {}
        # get a list of possible bit mask keywords
        # header keywords are all proceeded by 'MP_' so grab that list of keywords
        for keyword in mask_header['MP_*']:
            flag = keyword.replace("MP_","")
            default_bit_mask[flag] = mask_header[keyword]
        # set a default list of keywords to flag
        default_flag_keys = ['EDGE', 'NO_DATA', 'SAT', 'INTRP', 'REJECTED']
        for k in list(default_bit_mask.keys()):
            default_bit_mask[k] = mask_header[f"MP_{k}"]

        # setup array for default_flag_keys that might be missing from header
        missing_bit_mask_keywords = []
        for k in default_flag_keys:
            if k not in default_bit_mask:
                missing_bit_mask_keywords.append(k)

        image_params['missing_bit_mask_keywords'] = missing_bit_mask_keywords
        image_params['bit_mask'] = default_bit_mask
        image_params['flag_keys'] = default_flag_keys

        image_params['ec_angle'] = self._calc_ecliptic_angle(wcs)
        del(hdulist)

        images = [kb.layered_image('{0:s}/{1:s}'.format(
            im_filepath, self.return_filename(f,file_format)))
            for f in np.sort(use_images)]

        print('Loaded {0:d} images'.format(len(images)))
        stack = kb.image_stack(images)

        stack.set_times(times)
        print("Times set", flush=True)

        image_params['x_size'] = stack.get_width()
        image_params['y_size'] = stack.get_height()
        image_params['times']  = stack.get_times()
        return(stack, image_params)


    def save_results(self, res_filepath, out_suffix, keep):
        """
        This function saves results from a given search method (either region
        search or grid search)
        INPUT-
            res_filepath : string
            out_suffix : string
                Suffix to append to the output file name
            keep : dictionary
                Dictionary that contains the values to keep and print to file.
                It is a standard results dictionary generated by
                self.gen_results_dict().
        """

        print('---------------------------------------')
        print("Saving Results")
        print('---------------------------------------', flush=True)
        np.savetxt('%s/results_%s.txt' % (res_filepath, out_suffix),
                   np.array(keep['results'])[keep['final_results']], fmt='%s')
        with open('%s/lc_%s.txt' % (res_filepath, out_suffix), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(keep['lc'])[keep['final_results']])
        if len(keep['psi_curves'])>0:
            with open('%s/psi_%s.txt' % (res_filepath, out_suffix), 'w') as f:
                writer = csv.writer(f)
                writer.writerows(
                    np.array(keep['psi_curves'])[keep['final_results']])
        if len(keep['phi_curves'])>0:
            with open('%s/phi_%s.txt' % (res_filepath, out_suffix), 'w') as f:
                writer = csv.writer(f)
                writer.writerows(
                    np.array(keep['phi_curves'])[keep['final_results']])
        with open('%s/lc_index_%s.txt' % (res_filepath, out_suffix), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(keep['lc_index'])[keep['final_results']])
        with open('%s/times_%s.txt' % (res_filepath, out_suffix), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(keep['times'])[keep['final_results']])
        np.savetxt(
            '%s/filtered_likes_%s.txt' % (res_filepath, out_suffix),
            np.array(keep['new_lh'])[keep['final_results']], fmt='%.4f')
        np.savetxt(
            '%s/ps_%s.txt' % (res_filepath, out_suffix),
            np.array(keep['stamps']).reshape(
                len(np.array(keep['stamps'])), 441), fmt='%.4f')
        print('not writing the stamps .np output file or the residuals file')
        #stamps_to_save = np.array(keep['all_stamps'])
        #with open('{}/res_per_px_stats_{}.pkl'.format(res_filepath, out_suffix), 'wb') as f:
        #    pickle.dump({'min_LH_per_px':keep['min_LH_per_px'],
        #             'num_res_per_px':keep['num_res_per_px']}, f)
        #np.save(
        #    '%s/all_ps_%s.npy' % (res_filepath, out_suffix), stamps_to_save)

    def _calc_ecliptic_angle(self, test_wcs, angle_to_ecliptic=0.):
        """
        This function calculates the ecliptic angle of an image using the
        World Coordinate System (WCS). However, it is degenerate with respect
        to PI. This is to say, the returned answer may be off by PI (180
        degrees)
        INPUT-
            test_wcs : astropy WCS
                The WCS from a fits file, as loaded in with astropy.wcs.WCS()
            angle_to_ecliptic : [NEEDS DOC STRING]
        OUTPUT-
            eclip_angle : float
                The angle to the ecliptic for the fits file corresponding to
                test_wcs. NOTE- May be off by +/- PI (180 degrees)
        """
        wcs = [test_wcs]
        pixel_coords = [[],[]]
        pixel_start = [[1000, 2000]]
        angle = np.float(angle_to_ecliptic)
        vel_array = np.array([[6.*np.cos(angle), 6.*np.sin(angle)]])
        time_array = [0.0, 1.0, 2.0]
        vel_par_arr = vel_array[:, 0]
        vel_perp_arr = vel_array[:, 1]

        if type(vel_par_arr) is not np.ndarray:
            vel_par_arr = [vel_par_arr]
        if type(vel_perp_arr) is not np.ndarray:
            vel_perp_arr = [vel_perp_arr]
        for start_loc, vel_par, vel_perp in zip(
            pixel_start, vel_par_arr, vel_perp_arr):

            start_coord = astroCoords.SkyCoord.from_pixel(
                start_loc[0], start_loc[1], wcs[0])
            eclip_coord = start_coord.geocentrictrueecliptic
            eclip_l = []
            eclip_b = []
            for time_step in time_array:
                eclip_l.append(eclip_coord.lon + vel_par*time_step*u.arcsec)
                eclip_b.append(eclip_coord.lat + vel_perp*time_step*u.arcsec)
            eclip_vector = astroCoords.SkyCoord(
                eclip_l, eclip_b, frame='geocentrictrueecliptic')
            pixel_coords_set = astroCoords.SkyCoord.to_pixel(
                eclip_vector, wcs[0])
            pixel_coords[0].append(pixel_coords_set[0])
            pixel_coords[1].append(pixel_coords_set[1])

        pixel_coords = np.array(pixel_coords)
        x_dist = pixel_coords[0, 0, -1] - pixel_coords[0, 0, 0]
        y_dist = pixel_coords[1, 0, -1] - pixel_coords[1, 0, 0]
        eclip_angle = np.arctan2(y_dist,x_dist)
        return(eclip_angle)

class PostProcess(SharedTools):
    """
    This class manages the post-processing utilities used to filter out and
    otherwise remove false positives from the KBMOD search. This includes,
    for example, kalman filtering to remove outliers, stamp filtering to remove
    results with non-Gaussian postage stamps, and clustering to remove similar
    results.
    """
    def __init__(self, config):
        self.coeff = None
        self.num_cores = config['num_cores']
        self.sigmaG_lims = config['sigmaG_lims']
        self.eps = config['eps']
        self.cluster_type = config['cluster_type']
        self.cluster_function = config['cluster_function']
        self.clip_negative = config['clip_negative']
        return

    def apply_mask(self, stack, mask_num_images=2, mask_threshold=120.,
                   bit_mask = None, flag_keys = None):
        """
        This function applys a mask to the images in a KBMOD stack. This mask
        sets a high variance for masked pixels
        INPUT-
            stack : kbmod.image_stack object
                The stack before the masks have been applied.
            mask_num_images : int
                The minimum number of images in which a masked pixel must
                appear in order for it to be masked out. E.g. if
                masked_num_images=2, then an object must appear in the same
                place in at least two images in order for the variance at that
                location to be increased.
            mask_threshold : float
                Any pixel with a flux greater than mask_threshold has the
                variance increased at that pixel location.
        OUTPUT-
            stack : kbmod.image_stack object
                The stack after the masks have been applied.
        """
        # mask pixels with any flags
        #flags = ~0
        # Only valid for LSST latest difference images. Use with caution
        mask_bits_dict_v22 = {
            'BAD': 0, 'CLIPPED': 9, 'CR': 3, 'CROSSTALK': 10, 'DETECTED': 5,
            'DETECTED_NEGATIVE': 6, 'EDGE': 4, 'INEXACT_PSF': 11, 'INTRP': 2,
            'NOT_DEBLENDED': 12, 'NO_DATA': 8, 'REJECTED': 13, 'SAT': 1,
            'SENSOR_EDGE': 14, 'SUSPECT': 7, 'UNMASKEDNAN': 15}
        mask_bits_dict_v20 = {
            'BAD': 0, 'CLIPPED': 9, 'CR': 3, 'DETECTED': 5,
            'DETECTED_NEGATIVE': 6, 'EDGE': 4, 'INEXACT_PSF': 10, 'INTRP': 2,
            'NOT_DEBLENDED': 11, 'NO_DATA': 8, 'REJECTED': 12, 'SAT': 1,
            'SENSOR_EDGE': 13, 'SUSPECT': 7}
        # for FraserW+KavelaarsJ processed warpdiffs using LSST Pipe v19.1
        mask_bits_dict_HSC = {
            'BAD': 0, 'SAT': 1, 'INTRP': 2, 'EDGE': 4, 'DETECTED': 5,
            'DETECTED_NEGATIVE': 6, 'SUSPECT': 7, 'NO_DATA': 8, 'CROSSTALK': 9,
            'NOT_BLENDED': 10, 'UNMASKEDNAN': 11, 'BRIGHT_OBJECT': 12,
            'CLIPPED': 13, 'INEXACT_PSF': 14, 'REJECTED': 15,
            'SENSOR_EDGE': 16}
        # for YTC's processed warpdiffs (FOSSIL) using HSC Pipe
        mask_bits_dict_HSC_YTC = {
            'BAD': 0, 'SAT': 1, 'INTRP': 2, 'EDGE': 4, 'DETECTED': 5,
            'DETECTED_NEGATIVE': 6, 'SUSPECT': 7, 'NO_DATA': 8, 'CROSSTALK': 11,
            'NOT_BLENDED': 13, 'UNMASKEDNAN': 16, 'BRIGHT_OBJECT': 9,
            'CLIPPED': 10, 'INEXACT_PSF': 12, 'REJECTED': 14,
            'SENSOR_EDGE': 15, 'CR': 3}

        if bit_mask is None:
            print('Using bit_mask in image header\n')
            mask_bits_dict = bit_mask
        elif type(bit_mask) == type({}):
            print('Using user supplied bit_mask\n')
            mask_bits_dict = bit_mask
        else:
            print('Using hard coded bit_mask mask_bits_dict_HSC\n')
            mask_bits_dict = mask_bits_dict_HSC

        if type(flag_keys) == type([]):
            print('Using user supplied flag_keys dict.\n')
        elif flag_keys is None: # not sure this condition will ever occur...
            print("Using hardcoded flag_keys 'EDGE', 'NO_DATA', 'SAT', 'INTRP', 'REJECTED'\n")
            flag_keys = ['EDGE', 'NO_DATA', 'SAT', 'INTRP', 'REJECTED']

        vals, keys = list(mask_bits_dict.values()), list(mask_bits_dict.keys())
        for l in range(len(vals)):
            print('  ',keys[l],':',vals[l])
        print()
        # Mask the following pixels: DEEP
        #flag_keys = ['BAD','EDGE','NO_DATA','SUSPECT','UNMASKEDNAN']
        #master_flag_keys = ['DETECTED','REJECTED']

        # Mask the following pixels: Fraser HSC
        #flag_keys = ['EDGE', 'NO_DATA', 'SAT', 'INTRP', 'REJECTED']#, 'BRIGHT_OBJECT']


        # for FOSSIL
        #flag_keys = ['EDGE', 'NO_DATA', 'SAT']
        # for CLASSY ISIS
        #flag_keys = ['EDGE', 'NO_DATA', 'SAT', 'BAD', 'NO_DATA', 'INTRP', 'BRIGHT_OBJECT']

        master_flag_keys = [] #['DETECTED']  for fossil, [] for NH

        flags = 0
        for bit in flag_keys:
            flags += 2**mask_bits_dict[bit]

        #flags = 1011
        # unless it has one of these special combinations of flags
        #flag_exceptions = [32,39]
        flag_exceptions = [0]
        # mask any pixels which have any of these flags
        #master_flags = int('100111', 2)
        master_flags = 0
        for bit in master_flag_keys:
            master_flags += 2**mask_bits_dict[bit]

        # Apply masks
        stack.apply_mask_flags(flags, flag_exceptions)
        stack.apply_master_mask(master_flags, mask_num_images)

        for i in range(10):
            stack.grow_mask()

        # This applies a mask to pixels with more than 120 counts
        #stack.apply_mask_threshold(mask_threshold)
        return(stack)

    def load_results(
        self, search, image_params, lh_level, filter_type='clipped_sigmaG',
        chunk_size=500000, max_lh=1e9, chunk_start_index = 0, chunks_to_consider = 1000):
        """
        This function loads results that are output by the gpu grid search.
        Results are loaded in chunks and evaluated to see if the minimum
        likelihood level has been reached. If not, another chunk of results is
        fetched.
        INPUT-
            search : kbmod search object
            image_params : dictionary
                Contains the following image parameters:
                Julian day, x size of the images, y size of the images,
                ecliptic angle of the images.
            lh_level : float
                The minimum likelihood theshold for an acceptable result.
                Results below this likelihood level will be discarded.
            filter_type : string
                The type of initial filtering to apply. Acceptable values are
                'clipped_average'
                'kalman'
            chunk_size : int
                The number of results to load at a given time from search.
            max_lh : float
                The maximum likelihood threshold for an acceptable results.
                Results ABOVE this likelihood level will be discarded.
            chunk_start_index : int
                The chunk starting index. The sorting will start with
                Chunk Start = chunk_start_index*chunk_size. Default to 0.
            chunks_to_consider : int
                The max number of chunks to consider.
                Default to 1000 (a very large number).
        OUTPUT-
            keep : dictionary
                Dictionary containing values from trajectories. When output,
                it should have at least 'psi_curves', 'phi_curves', and
                'results'. It is a standard results dictionary generated by
                self.gen_results_dict().
        """
        if filter_type=='clipped_sigmaG':
            filter_func = self.apply_clipped_sigmaG
        elif filter_type=='clipped_average':
            filter_func = self.apply_clipped_average
        elif filter_type=='kalman':
            filter_func = self.apply_kalman_filter
        keep = self.gen_results_dict()
        tmp_results = self.gen_results_dict()
        likelihood_limit = False
        res_num = chunk_start_index*chunk_size
        psi_curves = []
        phi_curves = []
        all_results = []
        keep['min_LH_per_px'] = 9999*np.ones([image_params['x_size'],image_params['y_size']])
        keep['num_res_per_px'] = np.zeros([image_params['x_size'],image_params['y_size']])
        print('---------------------------------------')
        print("Retrieving Results")
        print('---------------------------------------')
        total_results_num = chunk_size*chunks_to_consider ## set this here as the while-loop can exit without setting the correct value
        while likelihood_limit is False and res_num<chunk_size*(chunks_to_consider + chunk_start_index):
            print('Getting results...')
            tmp_psi_curves = []
            tmp_phi_curves = []
            results = search.get_results(res_num, chunk_size)
            print('---------------------------------------')
            chunk_headers = ("Chunk Start", "Chunk Max Likelihood",
                             "Chunk Min. Likelihood")
            chunk_values = (res_num, results[0].lh, results[-1].lh)
            for header, val, in zip(chunk_headers, chunk_values):
                if type(val) == np.int:
                    print('%s = %i' % (header, val))
                else:
                    print('%s = %.2f' % (header, val))
            print('---------------------------------------')
            # Find the size of the psi phi curves and preallocate arrays
            foo_psi,_=search.lightcurve(results[0])
            curve_len = len(foo_psi.flatten())
            curve_shape = [len(results),curve_len]
            for i,line in enumerate(results):
                if line.lh < max_lh:
                    if keep['min_LH_per_px'][line.x,line.y] > line.lh:
                        keep['min_LH_per_px'][line.x,line.y] = line.lh
                    keep['num_res_per_px'][line.x,line.y] += 1
                    curve_index = i+res_num
                    psi_curve, phi_curve = search.lightcurve(line)
                    tmp_psi_curves.append(psi_curve)
                    tmp_phi_curves.append(phi_curve)
                    all_results.append(line)
                    if line.lh < lh_level:
                        likelihood_limit = True
                        total_results_num = res_num+i
                        break
            if len(tmp_psi_curves)>0:
                tmp_results['psi_curves'] = tmp_psi_curves
                tmp_results['phi_curves'] = tmp_phi_curves
                tmp_results['results'] = results
                keep_idx_results = filter_func(
                    tmp_results, search, image_params, lh_level)
                keep = self.read_filter_results(
                    keep_idx_results, keep, search, tmp_psi_curves,
                    tmp_phi_curves, results, image_params, lh_level)
                #Wes Hacking
                keep['psi_curves'] = []
                keep['phi_curves'] = []
                #end Wes hacking
            res_num+=chunk_size
        print('Keeping {} of {} total results'.format(
            np.shape(keep['psi_curves'])[0], total_results_num), flush=True)
        return(keep,likelihood_limit)

    def read_filter_results(
        self, keep_idx_results, keep, search, psi_curves, phi_curves, results,
        image_params, lh_level):
        """
        This function reads the results from level 1 filtering like
        apply_clipped_average() and appends the results to a 'keep' dictionary.
        INPUT-
            keep_idx_results : list
                list of tuples containing the index of a results, the
                indices of the passing values in the lightcurve, and the
                new likelihood for the lightcurve.
            keep : dictionary
                Dictionary containing values from trajectories. When output,
                it should have at least 'psi_curves', 'phi_curves', and
                'results'. It is a standard results dictionary generated by
                self.gen_results_dict().
            search : kbmod search object
            psi_curves : list
                List of psi_curves from kbmod search.
            phi_curves : list
                List of phi_curves from kbmod search.
            results : list
                List of results from kbmod search.
            image_params : dictionary
                Contains the following image parameters:
                Julian day, x size of the images, y size of the images,
                ecliptic angle of the images.
            lh_level : float
                The minimum likelihood theshold for an acceptable result.
                Results below this likelihood level will be discarded.
        OUTPUT-
            keep : dictionary
                Dictionary containing values from trajectories. When output,
                it should have at least 'psi_curves', 'phi_curves', and
                'results'. It is a standard results dictionary generated by
                self.gen_results_dict().
        """
        masked_phi_curves = np.copy(phi_curves)
        masked_phi_curves[masked_phi_curves==0] = 1e9
        num_good_results = 0
        if len(keep_idx_results[0]) < 3:
            keep_idx_results = [(0, [-1], 0.)]
        for result_on in range(len(psi_curves)):
            if keep_idx_results[result_on][1][0] == -1:
                continue
            elif len(keep_idx_results[result_on][1]) < 3:
                continue
            elif keep_idx_results[result_on][2] < lh_level:
                continue
            else:
                keep_idx = keep_idx_results[result_on][1]
                new_likelihood = keep_idx_results[result_on][2]
                keep['results'].append(results[result_on])
                keep['new_lh'].append(new_likelihood)
                keep['lc'].append(
                    psi_curves[result_on]/masked_phi_curves[result_on])
                keep['lc_index'].append(keep_idx)
                keep['psi_curves'].append(psi_curves[result_on])
                keep['phi_curves'].append(phi_curves[result_on])
                keep['times'].append(image_params['mjd'][keep_idx])
                num_good_results+=1
        print('Keeping {} results'.format(num_good_results))
        return(keep)

    def get_coadd_stamps(self, results, search, keep, stamp_type='sum'):
        """
        Get the coadded stamps for the initial results from a kbmod search.
        INPUT-
            keep : dictionary
                Dictionary containing values from trajectories. When input,
                it should have at least 'psi_curves', 'phi_curves', and
                'results'. These are populated in Interface.load_results().
            search : kbmod.stack_search object
        OUTPUT-
            keep : dictionary
                Dictionary containing values from trajectories. When input,
                it should have at least 'psi_curves', 'phi_curves', and
                'results'. These are populated in Interface.load_results().
        """
        start = time.time()
        if stamp_type=='cpp_median':
            num_images = len(keep['psi_curves'][0])
            boolean_idx = []
            for keep in keep['lc_index']:
                bool_row = np.zeros(num_images)
                bool_row[keep] = 1
                boolean_idx.append(bool_row.astype(int).tolist())
            #boolean_idx = np.array(boolean_idx)
            #pdb.set_trace()
            coadd_stamps = [np.array(stamp) for stamp in
                              search.median_stamps(results, boolean_idx, 10)]
        elif stamp_type=='parallel_sum':
            coadd_stamps = [np.array(stamp) for stamp in search.summed_stamps(results, 10)]
        else:
            coadd_stamps = []
            for i,result in enumerate(results):
                if stamp_type=='sum':
                    stamps = np.array(search.stacked_sci(result, 10)).astype(np.float32)
                    coadd_stamps.append(stamps)
                elif stamp_type=='median':
                    stamps = search.sci_stamps(result, 10)
                    stamp_arr = np.array(
                        [np.array(stamps[s_idx]) for s_idx in keep['lc_index'][i]])
                    stamp_arr[np.isnan(stamp_arr)]=0
                    coadd_stamps.append(np.median(stamp_arr, axis=0))
        print('Loaded {} coadded stamps. {:.3f}s elapsed'.format(
            len(coadd_stamps), time.time()-start), flush=True)
        return(coadd_stamps)

    def get_all_stamps(self, keep, search):
        """
        Get the stamps for the final results from a kbmod search.
        INPUT-
            keep : dictionary
                Dictionary containing values from trajectories. When input,
                it should have at least 'psi_curves', 'phi_curves', and
                'results'. These are populated in Interface.load_results().
            search : kbmod.stack_search object
        OUTPUT-
            keep : dictionary
                Dictionary containing values from trajectories. When input,
                it should have at least 'psi_curves', 'phi_curves', and
                'results'. These are populated in Interface.load_results().
        """
        final_results = keep['final_results']

        for result in np.array(keep['results'])[final_results]:
            stamps = search.sci_stamps(result, 10)
            all_stamps = np.array(
                [np.array(stamp).reshape(21,21) for stamp in stamps])
            keep['all_stamps'].append(all_stamps)
        return(keep)

    def apply_clipped_sigmaG(
        self, old_results, search, image_params, lh_level):
        """
        This function applies a clipped median filter to the results of a KBMOD
        search using sigmaG as a robust estimater of standard deviation.
            INPUT-
                keep : dictionary
                    Dictionary containing values from trajectories. When input,
                    it should have at least 'psi_curves', 'phi_curves', and
                    'results'. These are populated in Interface.load_results().
                search : kbmod.stack_search object
                image_params : dictionary
                    Dictionary containing parameters about the images that were
                    searched over. apply_kalman_filter only uses MJD
                lh_level : float
                    Minimum likelihood to search
            OUTPUT-
                keep_idx_results : list
                    list of tuples containing the index of a results, the
                    indices of the passing values in the lightcurve, and the
                    new likelihood for the lightcurve.
        """
        print("Applying Clipped-sigmaG Filtering")
        self.lc_filter_type = image_params['sigmaG_filter_type']
        start_time = time.time()
        # Make copies of the values in 'old_results' and create a new dict
        psi_curves = np.copy(old_results['psi_curves'])
        phi_curves = np.copy(old_results['phi_curves'])
        masked_phi_curves = np.copy(phi_curves)
        masked_phi_curves[masked_phi_curves==0] = 1e9

        results = old_results['results']
        keep = self.gen_results_dict()
        if self.coeff is None:
            if self.sigmaG_lims is not None:
                self.percentiles = self.sigmaG_lims
            else:
                self.percentiles = [25,75]
            self.coeff = self._find_sigmaG_coeff(self.percentiles)
        print('Starting pooling...')
        pool = mp.Pool(processes=self.num_cores)
        num_curves = len(psi_curves)
        index_list = [j for j in range(num_curves)]
        zipped_curves = zip(
            psi_curves, phi_curves, index_list)
        keep_idx_results = pool.starmap_async(
            self._clipped_sigmaG, zipped_curves)
        pool.close()
        pool.join()
        keep_idx_results = keep_idx_results.get()
        end_time = time.time()
        time_elapsed = end_time-start_time
        print('{:.2f}s elapsed'.format(time_elapsed))
        print('Completed filtering.', flush=True)
        print('---------------------------------------')
        return(keep_idx_results)

    def apply_clipped_average(
        self, old_results, search, image_params, lh_level):
        """
        This function applies a clipped median filter to the results of a KBMOD
        search.
            INPUT-
                keep : dictionary
                    Dictionary containing values from trajectories. When input,
                    it should have at least 'psi_curves', 'phi_curves', and
                    'results'. These are populated in Interface.load_results().
                search : kbmod.stack_search object
                image_params : dictionary
                    Dictionary containing parameters about the images that were
                    searched over. apply_kalman_filter only uses MJD
                lh_level : float
                    Minimum likelihood to search
            OUTPUT-
                keep_idx_results : list
                    list of tuples containing the index of a results, the
                    indices of the passing values in the lightcurve, and the
                    new likelihood for the lightcurve.
        """
        print("Applying Clipped-Average Filtering")
        start_time = time.time()
        # Make copies of the values in 'old_results' and create a new dict
        psi_curves = np.copy(old_results['psi_curves'])
        phi_curves = np.copy(old_results['phi_curves'])
        masked_phi_curves = np.copy(phi_curves)
        masked_phi_curves[masked_phi_curves==0] = 1e9

        results = old_results['results']
        keep = self.gen_results_dict()

        print('Starting pooling...')
        pool = mp.Pool(processes=self.num_cores)
        zipped_curves = zip(
            psi_curves, phi_curves, [j for j in range(len(psi_curves))])
        keep_idx_results = pool.starmap_async(
            self._clipped_average, zipped_curves)
        pool.close()
        pool.join()
        keep_idx_results = keep_idx_results.get()
        end_time = time.time()
        time_elapsed = end_time-start_time
        print('{:.2f}s elapsed'.format(time_elapsed))
        print('Completed filtering.', flush=True)
        print('---------------------------------------')
        return(keep_idx_results)

    def _find_sigmaG_coeff(self, percentiles):
        z1 = percentiles[0]/100
        z2 = percentiles[1]/100

        x1 = self._invert_Gaussian_CDF(z1)
        x2 = self._invert_Gaussian_CDF(z2)
        coeff = 1/(x2-x1)
        print('sigmaG limits: [{},{}]'.format(percentiles[0],percentiles[1]))
        print('sigmaG coeff: {:.4f}'.format(coeff), flush=True)
        return(coeff)

    def _invert_Gaussian_CDF(self, z):
        if z < 0.5:
            sign = -1
        else:
            sign = 1
        x = sign*np.sqrt(2)*mpmath.erfinv(sign*(2*z-1))
        return(float(x))

    def _clipped_sigmaG(
        self, psi_curve, phi_curve, index, n_sigma=2):
        """
        This function applies a clipped median filter to a set of likelihood
        values. Points are eliminated if they are more than n_sigma*sigmaG away
        from the median.
        INPUT-
            psi_curve : numpy array
                A single Psi curve, likely a single row of a larger matrix of
                psi curves, such as those that are loaded in from
                Interface.load_results() and stored in keep['psi_curves'].
            phi_curve : numpy array
                A single Phi curve, likely a single row of a larger matrix of
                phi curves, such as those that are loaded in from
                Interface.load_results() and stored in keep['phi_curves'].
            index : integer
                The row value of the larger Psi and Phi matrixes from which
                psi_values and phi_values are extracted. Used to keep track
                of processing while running multiprocessing.
            n_sigma : integer
                The number of standard deviations away from the median that
                the largest likelihood values (N=num_clipped) must be in order
                to be eliminated.
        OUTPUT-
            index : integer
                The row value of the larger Psi and Phi matrixes from which
                psi_values and phi_values are extracted. Used to keep track
                of processing while running multiprocessing.
            good_index : numpy array
                The indices that pass the filtering for a given set of curves.
            new_lh : float
                The new maximum likelihood of the set of curves, after
                max_lh_index has been applied.
        """
        masked_phi = np.copy(phi_curve)
        masked_phi[masked_phi==0] = 1e9
        if self.lc_filter_type=='lh':
            lh = psi_curve/np.sqrt(masked_phi)
            good_index = self._exclude_outliers(lh, n_sigma)
        elif self.lc_filter_type=='flux':
            flux = psi_curve/masked_phi
            good_index = self._exclude_outliers(flux, n_sigma)
        elif self.lc_filter_type=='both':
            lh = psi_curve/np.sqrt(masked_phi)
            good_index_lh = self._exclude_outliers(lh, n_sigma)
            flux = psi_curve/masked_phi
            good_index_flux = self._exclude_outliers(flux, n_sigma)
            good_index = np.intersect1d(good_index_lh, good_index_flux)
        else:
            print('Invalid filter type, defaulting to likelihood', flush=True)
            lh = psi_curve/np.sqrt(masked_phi)
            good_index = self._exclude_outliers(lh, n_sigma)

        if len(good_index)==0:
            new_lh = 0
            good_index=[-1]
        else:
            new_lh = self._compute_lh(
                psi_curve[good_index], phi_curve[good_index])
        return(index,good_index,new_lh)

    def _exclude_outliers(self, lh, n_sigma):
        if self.clip_negative:
            lower_per, median, upper_per = np.percentile(
                lh[lh>0], [self.percentiles[0], 50, self.percentiles[1]])
            sigmaG = self.coeff*(upper_per-lower_per)
            nSigmaG = n_sigma*sigmaG
            ### HACK DEBUG
            #if len(np.where(np.isnan(lh) | np.isinf(lh) )[0])>0 or np.isinf(median) or np.isinf(nSigmaG) or np.isnan(median) or np.isnan(nSigmaG):
            #    print(lh, median, nSigmaG)
            #    exit()
            #exit()
            ###
            good_index = np.where(np.logical_and(lh>0,np.logical_and(
                lh>median-nSigmaG, lh<median+nSigmaG)))[0]
            #good_index = np.where(np.logical_and(lh!=0,np.logical_and(
            #    lh>median-nSigmaG, lh<median+nSigmaG)))[0]
        else:
            lower_per, median, upper_per = np.percentile(
                lh, [self.percentiles[0], 50, self.percentiles[1]])
            sigmaG = self.coeff*(upper_per-lower_per)
            nSigmaG = n_sigma*sigmaG
            good_index = np.where(np.logical_and(
                lh>median-nSigmaG, lh<median+nSigmaG))[0]
        return(good_index)

    def _clipped_average(
        self, psi_curve, phi_curve, index, num_clipped=5, n_sigma=4,
        lower_lh_limit=-100):
        """
        This function applies a clipped median filter to a set of likelihood
        values. The largest likelihood values (N=num_clipped) are eliminated if
        they are more than n_sigma*standard deviation away from the median,
        which is calculated excluding the largest values.
        INPUT-
            psi_curve : numpy array
                A single Psi curve, likely a single row of a larger matrix of
                psi curves, such as those that are loaded in from
                Interface.load_results() and stored in keep['psi_curves'].
            phi_curve : numpy array
                A single Phi curve, likely a single row of a larger matrix of
                phi curves, such as those that are loaded in from
                Interface.load_results() and stored in keep['phi_curves'].
            index : integer
                The row value of the larger Psi and Phi matrixes from which
                psi_values and phi_values are extracted. Used to keep track
                of processing while running multiprocessing.
            num_clipped : integer
                The number of likelihood values to consider eliminating. Only
                considers the largest N=num_clipped values.
            n_sigma : integer
                The number of standard deviations away from the median that
                the largest likelihood values (N=num_clipped) must be in order
                to be eliminated.
            lower_lh_limit : float
                Likelihood values lower than lower_lh_limit are automatically
                eliminated from consideration.
        OUTPUT-
            index : integer
                The row value of the larger Psi and Phi matrixes from which
                psi_values and phi_values are extracted. Used to keep track
                of processing while running multiprocessing.
            max_lh_index : numpy array
                The indices that pass the filtering for a given set of curves.
            new_lh : float
                The new maximum likelihood of the set of curves, after
                max_lh_index has been applied.
        """
        masked_phi = np.copy(phi_curve)
        masked_phi[masked_phi==0] = 1e9
        lh = psi_curve/np.sqrt(masked_phi)
        max_lh = np.array(heapq.nlargest(num_clipped,lh))
        clipped_lh_index = np.where(np.logical_and(
            lh>lower_lh_limit,np.in1d(lh, max_lh, invert=True)))[0]
        if len(clipped_lh_index)==0:
            return(index,[-1],0)
        clipped_lh = lh[clipped_lh_index]
        median = np.median(clipped_lh)
        sigma = np.sqrt(np.var(clipped_lh))
        outlier_index = np.where(lh > median+n_sigma*sigma)
        if len(outlier_index[0])>0:
            outliers = np.min(lh[outlier_index])
            max_lh_index = np.where(
                np.logical_and(lh>lower_lh_limit,lh<outliers))[0]
            new_lh = self._compute_lh(
                psi_curve[max_lh_index], phi_curve[max_lh_index])
            return(index,max_lh_index,new_lh)
        else:
            max_lh_index = np.where(
                np.logical_and(lh>lower_lh_limit,lh<np.max(lh)+1))[0]
            new_lh = self._compute_lh(
                psi_curve[max_lh_index], phi_curve[max_lh_index])
            return(index,max_lh_index,new_lh)

    def apply_kalman_filter(self, old_results, search, image_params, lh_level):
        """
        This function applies a kalman filter to the results of a KBMOD search
            INPUT-
                keep : dictionary
                    Dictionary containing values from trajectories. When input,
                    it should have at least 'psi_curves', 'phi_curves', and
                    'results'. These are populated in Interface.load_results().
                search : kbmod.stack_search object
                image_params : dictionary
                    Dictionary containing parameters about the images that were
                    searched over. apply_kalman_filter only uses MJD
                lh_level : float
                    Minimum likelihood to search
            OUTPUT-
                keep_idx_results : list
                    list of tuples containing the index of a results, the
                    indices of the passing values in the lightcurve, and the
                    new likelihood for the lightcurve.
        """
        print("Applying Kalman Filtering")
        # Make copies of the values in 'old_results' and create a new dict
        psi_curves = np.copy(old_results['psi_curves'])
        phi_curves = np.copy(old_results['phi_curves'])
        masked_phi_curves = np.copy(phi_curves)
        masked_phi_curves[masked_phi_curves==0] = 1e9
        results = old_results['results']
        keep = self.gen_results_dict()

        print('Starting pooling...')
        pool = mp.Pool(processes=self.num_cores)
        zipped_curves = zip(
            psi_curves, phi_curves, [j for j in range(len(psi_curves))])
        keep_idx_results = pool.starmap_async(
            self._return_indices, zipped_curves)
        pool.close()
        pool.join()
        keep_idx_results = keep_idx_results.get()
        print('---------------------------------------')
        return(keep_idx_results)

    def apply_stamp_filter(
        self, keep, search, center_thresh=0.03, peak_offset=[2., 2.],
        mom_lims=[35.5, 35.5, 1., .25, .25], chunk_size=1000000,
        stamp_type='sum'):
        """
        This function filters result postage stamps based on their Gaussian
        Moments. Results with stamps that are similar to a Gaussian are kept.
        INPUT-
            keep : dictionary
                Contains the values of which results were kept from the search
                algorithm
            image_params : dictionary
                Contains values concerning the image and search initial
                settings
        OUTPUT-
            keep : dictionary
                Contains the values of which results were kept from the search
                algorithm
        """
        self.center_thresh = center_thresh
        self.peak_offset = peak_offset
        self.mom_lims = mom_lims
        #lh_sorted_idx = np.argsort(np.array(keep['new_lh']))[::-1]
        print('---------------------------------------')
        print("Applying Stamp Filtering")
        print('---------------------------------------', flush=True)
        i = 0
        passing_stamps_idx = []
        num_results = len(keep['results'])
        counter = 0
        if num_results > 0:
            print("Stamp filtering %i results" % num_results)
            while i<num_results:
                if i+chunk_size < num_results:
                    end_idx = i+chunk_size
                else:
                    end_idx = num_results
                stamps_slice = self.get_coadd_stamps(
                    np.array(keep['results'])[i:end_idx], search, keep, stamp_type)
                pool = mp.Pool(processes=self.num_cores, maxtasksperchild=1000)
                stamp_filt_pool = pool.map_async(
                    self._stamp_filter_parallel, np.copy(stamps_slice))
                pool.close()
                pool.join()
                stamp_filt_results = stamp_filt_pool.get()
                passing_stamps_chunk = np.where(
                    np.array(stamp_filt_results) == 1)[0]
                passing_stamps_idx.append(passing_stamps_chunk+i)
                #Wes Nuke# keep['stamps'].append(np.array(stamps_slice)[passing_stamps_chunk]) ####MAYBE NUKE
                i+=chunk_size
                counter+=1
            del(stamp_filt_results)
            del(stamp_filt_pool)
        if counter>0:#len(keep['stamps']) > 0:   ###change the length, adding a count variable
            #Wes Nuke# keep['stamps'] = np.concatenate(keep['stamps'], axis=0) #####MYABE NUKE
            keep['final_results'] = np.unique(np.concatenate(passing_stamps_idx))
        print('Keeping %i results' % len(keep['final_results']), flush=True)
        return(keep)

    def apply_clustering(self, keep, image_params):
        """
        This function clusters results that have similar trajectories.
        INPUT-
            keep : Dictionary
                Contains the values of which results were kept from the search
                algorithm
            image_params : dictionary
                Contains values concerning the image and search initial
                settings
        OUTPUT-
            keep : Dictionary
                Contains the values of which results were kept from the search
                algorithm
        """
        results_indices = keep['final_results']
        if np.any(results_indices==...):
            results_indices = np.linspace(0, len(keep['results'])-1, len(keep['results']), dtype=np.int)

        print("Clustering %i results" % len(results_indices), flush=True)
        if len(results_indices)>0:
            cluster_idx = self._cluster_results(
                np.array(keep['results'])[results_indices],
                image_params['x_size'], image_params['y_size'],
                image_params['vel_lims'], image_params['ang_lims'], image_params['mjd'])
            keep['final_results'] = results_indices[cluster_idx]
            if len(keep['stamps'])>0:
                keep['stamps'] = keep['stamps'][cluster_idx]
            del(cluster_idx)
        print('Keeping %i results' % len(keep['final_results']))
        return(keep)

    def _kalman_filter(self, obs, var):
        """
        This function applies a Kalman filter to a given set of flux values.
        INPUT-
            obs : numpy array
                obs should be a flux value computed by Psi/Phi for each data
                point. It should have the same length as keep['psi_curves'],
                unless points where flux=0 have been masked out.
            var : numpy array
                The inverse Phi values, with Phi<-999 masked.
        OUTPUT-
            xhat : numpy array
                The kalman flux
            P : numpy array
                The kalman error
        """
        xhat = np.zeros(len(obs))
        P = np.zeros(len(obs))
        xhatminus = np.zeros(len(obs))
        Pminus = np.zeros(len(obs))
        K = np.zeros(len(obs))

        Q = 1.
        R = np.copy(var)

        xhat[0] = obs[0]
        P[0] = R[0]

        for k in range(1,len(obs)):
            xhatminus[k] = xhat[k-1]
            Pminus[k] = P[k-1] + Q

            K[k] = Pminus[k] / (Pminus[k] + R[k])
            xhat[k] = xhatminus[k] + K[k]*(obs[k]-xhatminus[k])
            P[k] = (1-K[k])*Pminus[k]
        return xhat, P

    def _return_indices(self, psi_values, phi_values, val_on):
        """
        This function returns the indices of the Psi and Phi values that pass
        Kalman filtering.
        INPUT-
            psi_values : numpy array
                A single Psi curve, likely a single row of a larger matrix of
                psi curves, such as those that are loaded in from
                Interface.load_results() and stored in keep['psi_curves'].
            phi_values : numpy array
                A single Phi curve, likely a single row of a larger matrix of
                phi curves, such as those that are loaded in from
                Interface.load_results() and stored in keep['phi_curves'].
            val_on : int
                The row value of the larger Psi and Phi matrixes from which
                psi_values and phi_values are extracted. Used to keep track
                of processing while running multiprocessing.
        OUTPUT-
            val_on : int
                The row value of the larger Psi and Phi matrixes from which
                psi_values and phi_values are extracted. Used to keep track
                of processing while running multiprocessing.
            flux_idx : numpy array
                The indices corresponding to the phi and psi values that pass
                kalman filtering.
            new_lh : float
                The likelihood that there is a source at the given location.
                This likelihood is computed using only the values of phi and
                psi that PASS kalman filtering.
        """
        masked_phi_values = np.copy(phi_values)
        masked_phi_values[masked_phi_values==0] = 1e9
        flux_vals = psi_values/masked_phi_values
        flux_idx = np.where(flux_vals > 0.)[0]
        if len(flux_idx) < 2:
            return ([], [-1], [])
        fluxes = flux_vals[flux_idx]
        inv_flux = np.array(masked_phi_values[flux_idx])
        inv_flux[inv_flux < -999.] = 9999999.
        f_var = (1./inv_flux)

        ## 1st pass
        #f_var = #var_curve[flux_idx]#np.var(fluxes)*np.ones(len(fluxes))
        kalman_flux, kalman_error = self._kalman_filter(fluxes, f_var)
        if np.min(kalman_error) < 0.:
            return ([], [-1], [])
        deviations = np.abs(kalman_flux - fluxes) / kalman_error**.5

        #print(deviations, fluxes)
        # keep_idx = np.where(deviations < 500.)[0]
        keep_idx = np.where(deviations < 5.)[0]

        ## Second Pass (reverse order in case bright object is first datapoint)
        kalman_flux, kalman_error = self._kalman_filter(
            fluxes[::-1], f_var[::-1])
        if np.min(kalman_error) < 0.:
            return ([], [-1], [])
        deviations = np.abs(kalman_flux - fluxes[::-1]) / kalman_error**.5
        #print(fluxes, f_var, kalman_flux, kalman_error**.5, deviations)
        # keep_idx_back = np.where(deviations < 500.)[0]
        keep_idx_back = np.where(deviations < 5.)[0]

        if len(keep_idx) >= len(keep_idx_back):
            new_psi = psi_values[flux_idx[keep_idx]]
            new_phi = phi_values[flux_idx[keep_idx]]
            new_lh = self._compute_lh(new_psi,new_phi)
            return (val_on, flux_idx[keep_idx], new_lh)
        else:
            reverse_idx = len(flux_idx)-1 - keep_idx_back
            new_psi = psi_values[flux_idx[reverse_idx]]
            new_phi = phi_values[flux_idx[reverse_idx]]
            new_lh = self._compute_lh(new_psi,new_phi)
            return (val_on, flux_idx[reverse_idx], new_lh)

    def _compute_lh(self, psi_values, phi_values):
        """
        This function computes the likelihood that there is a source along
        a given trajectory with the input Psi and Phi curves.
        INPUT-
            psi_values : numpy array
                The Psi values along a trajectory.
            phi_values : numpy array
                The Phi values along a trajectory.
        OUTPUT-
            lh : float
                The likelihood that there is a source along the given
                trajectory.
        """
        if (psi_values==0).all():
            lh = 0
        else:
            lh = np.sum(psi_values)/np.sqrt(np.sum(phi_values))
        return(lh)

    def _cluster_results(
        self, results, x_size, y_size, v_lim, ang_lim, mjd_times, cluster_args=None):
        """
        This function clusters results and selects the highest-likelihood
        trajectory from a given cluster.
        INPUT-
            results : kbmod results
                A list of kbmod trajectory results such as are stored in
                keep['results'].
            x_size : list
                The width of the images used in the kbmod stack, such as are
                stored in image_params['x_size'].
            y_size : list
                The height of the images used in the kbmod stack, such as are
                stored in image_params['y_size'].
            v_lim : list
                The velocity limits of the search, such as are stored in
                image_params['v_lim'].
            ang_lim : list
                The angle limits of the search, such as are stored in
                image_params['ang_lim']
            cluster_args : dictionary
                Arguments to pass to dbscan or OPTICS.
        OUTPUT-
            top_vals : numpy array
                An array of the indices for the best trajectories of each
                individual cluster.
        """
        if self.cluster_function == 'DBSCAN':
            default_cluster_args = dict(eps=self.eps, min_samples=-1, n_jobs=-1)
        elif self.cluster_function == 'OPTICS':
            default_cluster_args = dict(max_eps=self.eps, min_samples=2, n_jobs=-1)

        if cluster_args is not None:
            default_cluster_args.update(cluster_args)
        cluster_args = default_cluster_args


        """
        #######
        # Wes hacking

        #setup the smaller memory footprint by making an array of only the size we want
        if self.cluster_type == 'all': #cluster type 'all' np.array([scaled_x, scaled_y, scaled_vel, scaled_ang], dtype=np.float)
            clust_arr = np.zeros((len(results), 4), dtype=np.float)

        elif self.cluster_type == 'position': #cluster type 'position' np.array([scaled_x, scaled_y], dtype=np.float)
            clust_arr = np.zeros((len(results), 2), dtype=np.float)

        elif self.cluster_type == 'mid_position': #cluster type 'mid_position' np.array([scaled_mid_x, scaled_mid_y], dtype=np.float)
            clust_arr = np.zeros((len(results), 2), dtype=np.float)

            #x_arr = []
            #y_arr = []
            #times =
            median_time = np.median(mjd_times - mjd_times[0])
            #
            #for line in results:
            #    x_arr.append(line.x)
            #    y_arr.append(line.y)
            #
            #mid_x_arr = x_arr + median_time * vx_arr
            #mid_y_arr = y_arr + median_time * vy_arr
            #
            #scaled_mid_x = mid_x_arr/x_size
            #scaled_mid_y = mid_y_arr/y_size

            for line in results:
                clust_arr[:, 0] = (line.x + median_time*line.x_v)/x_size
                clust_arr[:, 1] = (line.y + median_time*line.y_v)/y_size
            cluster = DBSCAN(**cluster_args)
            cluster.fit(clust_arr.T)

        #####
        """
        x_arr = []
        y_arr = []
        vx_arr = []
        vy_arr = []
        vel_arr = []
        ang_arr = []
        times = mjd_times - mjd_times[0]
        median_time = np.median(times)

        for line in results:
            x_arr.append(line.x)
            y_arr.append(line.y)
            vx_arr.append(line.x_v)
            vy_arr.append(line.y_v)
            vel_arr.append(np.sqrt(line.x_v**2. + line.y_v**2.))
            ang_arr.append(np.arctan2(line.y_v,line.x_v))

        x_arr = np.array(x_arr)
        y_arr = np.array(y_arr)
        vx_arr = np.array(vx_arr)
        vy_arr = np.array(vy_arr)
        vel_arr = np.array(vel_arr)
        ang_arr = np.array(ang_arr)

        mid_x_arr = x_arr + median_time * vx_arr
        mid_y_arr = y_arr + median_time * vy_arr

        scaled_x = x_arr/x_size
        scaled_y = y_arr/y_size
        scaled_vel = (vel_arr - v_lim[0])/(v_lim[1] - v_lim[0])
        scaled_ang = (ang_arr - ang_lim[0])/(ang_lim[1] - ang_lim[0])

        if self.cluster_function == 'DBSCAN':
            cluster = DBSCAN(**cluster_args)
        elif self.cluster_function == 'OPTICS':
            cluster = OPTICS(**cluster_args)

        if self.cluster_type == 'all':
            cluster.fit(np.array([
                scaled_x, scaled_y, scaled_vel, scaled_ang], dtype=np.float).T)
        elif self.cluster_type == 'position':
            cluster.fit(np.array([
                scaled_x, scaled_y], dtype=np.float).T)
        elif self.cluster_type == 'mid_position':
            scaled_mid_x = mid_x_arr/x_size
            scaled_mid_y = mid_y_arr/y_size
            cluster.fit(np.array([scaled_mid_x, scaled_mid_y], dtype=np.float).T)

        top_vals = []
        for cluster_num in np.unique(cluster.labels_):
            cluster_vals = np.where(cluster.labels_ == cluster_num)[0]
            top_vals.append(cluster_vals[0])

        del(cluster)

        return top_vals

    def _stamp_filter_parallel(self, stamps):
        """
        This function filters an individual stamp and returns a true or false
        value for the index.
        INPUT-
            stamps : numpy array
                The stamps for a given trajectory. Stamps will be accepted if
                they are sufficiently similar to a Gaussian.
        OUTPUT-
            keep_stamps : int (boolean)
                A 1 (True) or 0 (False) value on whether or not to keep the
                trajectory.
        """
        center_thresh = self.center_thresh
        x_peak_offset, y_peak_offset = self.peak_offset
        mom_lims = self.mom_lims
        s = np.copy(stamps)
        s[np.isnan(s)] = 0.0
        s = s - np.min(s)
        stamp_sum = np.sum(s)
        if stamp_sum != 0:
            s /= stamp_sum
        s = np.array(s, dtype=np.dtype('float64')).reshape(21, 21)
        mom = measure.moments_central(s, center=(10,10))
        mom_list = [mom[2, 0], mom[0, 2], mom[1, 1], mom[1, 0], mom[0, 1]]
        peak_1, peak_2 = np.where(s == np.max(s))

        if len(peak_1) > 1:
            peak_1 = np.max(np.abs(peak_1-10.))

        if len(peak_2) > 1:
            peak_2 = np.max(np.abs(peak_2-10.))
        if ((mom_list[0] < mom_lims[0]) & (mom_list[1] < mom_lims[1])
            & (np.abs(mom_list[2]) < mom_lims[2])
            & (np.abs(mom_list[3]) < mom_lims[3])
            & (np.abs(mom_list[4]) < mom_lims[4])
            & (np.abs(peak_1 - 10.) < x_peak_offset)
            & (np.abs(peak_2 - 10.) < y_peak_offset)):
            if center_thresh != 0:
                if np.max(stamps/np.sum(stamps)) > center_thresh:
                    keep_stamps = 1
                else:
                    keep_stamps = 0
            else:
                keep_stamps = 1
        else:
            keep_stamps = 0
        del(s)
        del(mom_list)
        del(peak_1)
        del(peak_2)
        return(keep_stamps)

    def _stamp_filter_parallel_fraser(self, stamps):
        """
        This function filters an individual stamp and returns a true or false
        value for the index.
        INPUT-
            stamps : numpy array
                The stamps for a given trajectory. Stamps will be accepted if
                they are sufficiently similar to a Gaussian.
        OUTPUT-
            keep_stamps : int (boolean)
                A 1 (True) or 0 (False) value on whether or not to keep the
                trajectory.
        """
        center_thresh = self.center_thresh
        x_peak_offset, y_peak_offset = self.peak_offset
        mom_lims = self.mom_lims

        w = np.where(np.isnan(stamps))
        stamps[w] = 0.0
        stamp_min = np.min(stamps)
        stamps -= stamp_min
        #s = np.copy(stamps)
        #s[np.isnan(s)] = 0.0
        #s = s - np.min(s)
        stamp_sum = np.sum(stamps)
        if stamp_sum != 0:
            stamps /= stamp_sum
            #s /= stamp_sum
        #s = np.array(s, dtype=np.dtype('float64')).reshape(21, 21)
        stamps = np.array(stamps, dtype=np.dtype('float64')).reshape(21, 21)
        mom = measure.moments_central(stamps, center=(10,10))
        mom_list = [mom[2, 0], mom[0, 2], mom[1, 1], mom[1, 0], mom[0, 1]]
        peak_1, peak_2 = np.where(stamps == np.max(stamps))

        if len(peak_1) > 1:
            peak_1 = np.max(np.abs(peak_1-10.))

        if len(peak_2) > 1:
            peak_2 = np.max(np.abs(peak_2-10.))

        # redo all the modifications to stamps done above
        if stamp_sum != 0:
            stamps *= stamp_sum
        stamps += stamp_min

        if ((mom_list[0] < mom_lims[0]) & (mom_list[1] < mom_lims[1])
            & (np.abs(mom_list[2]) < mom_lims[2])
            & (np.abs(mom_list[3]) < mom_lims[3])
            & (np.abs(mom_list[4]) < mom_lims[4])
            & (np.abs(peak_1 - 10.) < x_peak_offset)
            & (np.abs(peak_2 - 10.) < y_peak_offset)):
            if center_thresh != 0:
                if np.max(stamps/np.sum(stamps)) > center_thresh:
                    keep_stamps = 1
                else:
                    keep_stamps = 0
            else:
                keep_stamps = 1
        else:
            keep_stamps = 0
        #del(s)
        del(mom_list)
        del(peak_1)
        del(peak_2)
        return(keep_stamps)
