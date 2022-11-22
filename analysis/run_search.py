import os
import warnings
import pdb
import sys
import shutil
import pandas as pd
import numpy as np
import time
import multiprocessing as mp
import astropy.coordinates as astroCoords
import astropy.units as u
from kbmodpy import kbmod as kb
from astropy.io import fits
from astropy.wcs import WCS
from sklearn.cluster import DBSCAN
from skimage import measure
from analysis_utils import Interface, PostProcess
import gc, pickle
#from keras_filter_resnet import *

class region_search:
    """
    CLASS CURRENTLY DOES NOT WORK
    """
    def __init__(self,v_guess,radius,num_obs):
        """
        INPUT-
            v_guess : float array
                Initial object velocity guess. Given as an array or tuple.
                Algorithm will search velocities within 'radius' of 'v_guess'
            radius : float
                radius in velocity space to search, centered around 'v_guess'
            num_obs : int
                The minimum number of observations required to keep the object
        """
        self.v_guess = v_guess
        self.radius = radius
        self.num_obs = num_obs
        return

    def run_search(self, im_filepath, res_filepath, out_suffix, time_file,
                   likelihood_level=10., mjd_lims=None):
        # Initialize some values
        start = time.time()

        memory_error = False
        # Load images to search
        search,image_params = self.load_images(
            im_filepath, time_file, mjd_lims=mjd_lims)

        # Run the region search
        # Save values in image_params for use in filter_results

        print("Starting Search")
        print('---------------------------------------')
        param_headers = ("X Velocity Guess","Y Velocity Guess",
                         "Radius in velocity space")
        param_values = (*self.v_guess,self.radius)
        for header, val in zip(param_headers, param_values):
            print('%s = %.4f' % (header, val))
        results = search.region_search(
            *self.v_guess, self.radius, likelihood_level, int(self.num_obs))
        duration = image_params['times'][-1]-image_params['times'][0]
        # Convert the results to the grid formatting
        grid_results = kb.region_to_grid(results,duration)
        # Process the search results
        keep = self.process_region_results(
            search, image_params, res_filepath, likelihood_level, grid_results)
        del(search)

        # Cluster the results
        #keep = self.filter_results(keep,image_params)

        # Save the results
        self.save_results(res_filepath, out_suffix, keep)

        end = time.time()

        del(keep)
        return

    def process_region_results(
        self,search,image_params,res_filepath,likelihood_level,results):
        """
        Processes results that are output by the gpu search.
        """

        keep = {'stamps': [], 'new_lh': [], 'results': [], 'times': [],
                'lc': [], 'final_results': []}

        print('---------------------------------------')
        print("Processing Results")
        print('---------------------------------------')
        print('Starting pooling...')
        pool = mp.Pool(processes=16)
        print('Getting results...')

        psi_curves = []
        phi_curves = []
        # print(results)
        for line in results:
            psi_curve, phi_curve = search.lightcurve(line)
            psi_curves.append(np.array(psi_curve).flatten())
            phi_curve = np.array(phi_curve).flatten()
            phi_curve[phi_curve == 0.] = 99999999.
            phi_curves.append(phi_curve)

        keep_idx_results = pool.starmap_async(
            return_indices,
            zip(psi_curves, phi_curves, [j for j in range(len(psi_curves))]))
        pool.close()
        pool.join()
        keep_idx_results = keep_idx_results.get()
        if (len(keep_idx_results) < 1):
            keep_idx_results = [(0,[-1],0.)]

        if (len(keep_idx_results[0]) < 3):
            keep_idx_results = [(0, [-1], 0.)]

        for result_on in range(len(psi_curves)):

            if keep_idx_results[result_on][1][0] == -1:
                continue
            elif len(keep_idx_results[result_on][1]) < 3:
                continue
            elif keep_idx_results[result_on][2] < likelihood_level:
                continue
            else:
                keep_idx = keep_idx_results[result_on][1]
                new_likelihood = keep_idx_results[result_on][2]
                keep['results'].append(results[result_on])
                keep['new_lh'].append(new_likelihood)
                stamps = search.sci_stamps(results[result_on], 10)
                stamp_arr = np.array(
                    [np.array(stamps[s_idx]) for s_idx in keep_idx])
                keep['stamps'].append(np.sum(stamp_arr, axis=0))
                keep['lc'].append(
                    (psi_curves[result_on]/phi_curves[result_on])[keep_idx])
                keep['times'].append(image_params['mjd'][keep_idx])
        print(len(keep['results']))
        # Needed for compatibility with grid_search save functions
        keep['final_results'] = range(len(keep['results']))

        return(keep)

class run_search:
    """
    This class runs the grid search for kbmod.
    """
    def __init__(self, input_parameters):

        """
        INPUT-
            input_parameters : dictionary
                Dictionary containing input parameters. Merged with the
                defaults dictionary. MUST include 'im_filepath',
                'res_filepath', and 'time_file'. These are the filepaths to the
                image directory, results directory, and time file,
                respectively. Should contain 'v_arr', and 'ang_arr', which are
                lists containing the lower and upper velocity and angle limits.
        """

        defaults = { # Mandatory values
            'im_filepath':None, 'res_filepath':None, 'time_file':None,
            # Suggested values
            'v_arr':[92.,526.,256], 'ang_arr':[np.pi/15,np.pi/15,128],
            # Optional values
            'output_suffix':'search', 'mjd_lims':None, 'average_angle':None,
            'do_mask':True, 'mask_num_images':2, 'mask_threshold':120.,
            'lh_level':10., 'psf_val':1.4, 'num_obs':10, 'num_cores':30,
            'visit_in_filename':[0,6], 'file_format':'{0:06d}.fits',
            'sigmaG_lims':[25,75], 'chunk_size':500000, 'max_lh':1000.,
            'filter_type':'clipped_sigmaG', 'center_thresh':0.00,
            'peak_offset':[2.,2.], 'mom_lims':[35.5,35.5,2.0,0.3,0.3],
            'stamp_type':'sum', 'eps':0.03, 'gpu_filter':False,
            'do_clustering':True, 'do_stamp_filter':True,
            'clip_negative':False, 'sigmaG_filter_type':'lh',
            'cluster_type':'all', 'cluster_function':'DBSCAN',
            'chunk_start_index': 0, 'chunks_to_consider': 1000,
            'custom_bit_mask': None,
            }
        # Make sure input_parameters contains valid input options
        for key, val in input_parameters.items():
            if key in defaults:
                defaults[key] = val
            else:
                warnings.warn('Key "{}" is not a valid option. It is being ignored.'.format(key))
        self.config = defaults
        #self.config = {**defaults, **input_parameters}
        if (self.config['im_filepath'] is None):
            raise ValueError('Image filepath not set')
        if (self.config['res_filepath'] is None):
            raise ValueError('Results filepath not set')
        if (self.config['time_file'] is None):
            raise ValueError('Time filepath not set')
        return

    def do_gpu_search(self, search, image_params, post_process):

        # Run the grid search
        # Set min and max values for angle and velocity
        if self.config['average_angle'] == None:
            average_angle = image_params['ec_angle']
        else:
            average_angle = self.config['average_angle']
        ang_min = average_angle - self.config['ang_arr'][0]
        ang_max = average_angle + self.config['ang_arr'][1]
        vel_min = self.config['v_arr'][0]
        vel_max = self.config['v_arr'][1]
        image_params['ang_lims'] = [ang_min, ang_max]
        image_params['vel_lims'] = [vel_min, vel_max]

        search_start = time.time()
        print("Starting Search")
        print('---------------------------------------')
        param_headers = ("Ecliptic Angle", "Min. Search Angle",
                         "Max Search Angle", "Min Velocity", "Max Velocity")
        param_values = (image_params['ec_angle'], *image_params['ang_lims'],
                        *image_params['vel_lims'])
        for header, val in zip(param_headers, param_values):
            print('%s = %.4f' % (header, val))
        print('')
        if self.config['gpu_filter']:
            print('Using in-line GPU filtering methods', flush=True)
            self.config['sigmaG_coeff'] = post_process._find_sigmaG_coeff(
                self.config['sigmaG_lims'])
            search.gpuFilter(
                int(self.config['ang_arr'][2]), int(self.config['v_arr'][2]),
                *image_params['ang_lims'], *image_params['vel_lims'],
                int(self.config['num_obs']),
                np.array(self.config['sigmaG_lims'])/100.0,
                self.config['sigmaG_coeff'], self.config['mom_lims'],
                self.config['lh_level'])
        else:
            search.gpu(
                int(self.config['ang_arr'][2]), int(self.config['v_arr'][2]),
                *image_params['ang_lims'], *image_params['vel_lims'],
                int(self.config['num_obs']))
        print(
            'Search finished in {0:.3f}s \n'.format(time.time()-search_start),
            flush=True)
        return(search, image_params)

    def run_search(self):
        """
        This function serves as the highest-level python interface for starting
        a KBMOD search.
        INPUT-
            im_filepath : string
                Path to the folder containing the images to be ingested into
                KBMOD and searched over.
            res_filepath : string
                Path to the folder that will contain the results from the
                search.
            out_suffix : string
                Suffix to append to the output files. Used to differentiate
                between different searches over the same stack of images.
            time_file : string
                Path to the file containing the image times.
            lh_level : float
                Minimum acceptable likelihood level for a trajectory.
                Trajectories with likelihoods below this value will be
                discarded.
            psf_val : float
                Determines the size of the psf generated by the kbmod stack.
            mjd_lims : numpy array
                Limits the search to images taken within the limits input by
                mjd_lims.
            average_angle : float
                Overrides the ecliptic angle calculation and instead centers
                the average search around average_angle.
        """

        start = time.time()
        kb_interface = Interface()
        kb_post_process = PostProcess(self.config)

        # Load images to search
        stack,image_params = kb_interface.load_images(
            self.config['im_filepath'], self.config['time_file'],
            self.config['mjd_lims'], self.config['visit_in_filename'],
            self.config['file_format'])

        # Save values in image_params for later use
        if self.config['do_mask']:
            bit_mask = self.config['custom_bit_mask'] if self.config['custom_bit_mask'] is not None else image_params['bit_mask']
            stack = kb_post_process.apply_mask(
                stack, mask_num_images=self.config['mask_num_images'],
                mask_threshold=self.config['mask_threshold'],
                bit_mask = bit_mask)
        psf = kb.psf(self.config['psf_val'])
        #help(psf)
        print('\n The adopted PSF:')
        print(psf.print_psf())
        #exit()
        search = kb.stack_search(stack, psf) # stack_search is defined in pybind11 classBindings.cpp

        search, image_params = self.do_gpu_search(
            search, image_params, kb_post_process)

        # Load the KBMOD results into Python and apply a filter based on
        # 'filter_type'
        image_params['sigmaG_filter_type'] = self.config['sigmaG_filter_type']
        chunk_count, filt_count = 0, 0
        like_lim = False
        while chunk_count<1000 and not like_lim: #1000 is just a lazy very large number
            (keep, like_lim) = kb_post_process.load_results(
                            search, image_params, self.config['lh_level'],
                            chunk_size=self.config['chunk_size'],
                            filter_type=self.config['filter_type'],
                            max_lh=self.config['max_lh'],
                            chunks_to_consider = self.config['chunks_to_consider'],
                            chunk_start_index = self.config['chunk_start_index'] + chunk_count)

            # Wes HACK memory reduction
            keep['psi_curves'] = []
            keep['phi_curves'] = []
            # Wes Hack end
            if self.config['do_stamp_filter']:
                keep = kb_post_process.apply_stamp_filter(
                    keep, search, center_thresh=self.config['center_thresh'],
                    peak_offset=self.config['peak_offset'],
                    mom_lims=self.config['mom_lims'],
                    stamp_type=self.config['stamp_type'])

            #print(keep['final_results'])
            #for k in list(keep.keys()):
            #    print(k,type(keep[k]))
            #    try:
            #        print(len(keep[k]))
            #    except:
            #        pass
            #    print()
            #exit()

            makeStacks = False
            if makeStacks:
                from convenience_utils import makeStack, badflags
                from numpy import ma
                import pylab as pyl

                t1 = time.time()
                sci_imgs = stack.sciences()
                sci_masks = stack.masks()


                ### setup parameters needed for Wes's stack maker
                av_mode = 'single_rate'
                cutout_size = 43
                hcos = int(cutout_size/2)
                pad = int(image_params['vel_lims'][1]*(image_params['mjd'][-1] - image_params['mjd'][0]))+1 + hcos
                ra_grid = [] # for the single_rate cutout mode, ra_grid is not actually used.
                kb_xy = []
                for i in keep['final_results']:
                    res = keep['results'][i]
                    kb_xy.append([res.x, res.y, res.x_v, res.y_v])
                kb_xy = np.array(kb_xy)

                cutout = ma.masked_array(np.zeros((1,len(image_params['mjd']), cutout_size, cutout_size), dtype = sci_imgs[0].dtype),
                                         mask = np.zeros((1,len(image_params['mjd']), cutout_size, cutout_size)))#, dtype = sci_imgs[0].dtype))
                stamps = np.zeros((len(kb_xy), cutout_size, cutout_size), dtype = sci_imgs[0].dtype)
                ###

                im_data = np.pad(sci_imgs, ((0, 0), (pad, pad), (pad, pad)), mode = 'constant', constant_values = 0.0 )
                mask_data = np.pad(sci_masks, ((0, 0), (pad, pad), (pad, pad)), mode = 'constant', constant_values = 4096 ) #
                del sci_imgs
                del sci_masks

                w = np.where(((mask_data.astype('int') & badflags) == 0) & ~np.isnan(im_data))
                W = np.where( ~(((mask_data.astype('int') & badflags) == 0) & ~np.isnan(im_data)))
                mask_data[w] = 0.0 # good values
                mask_data[W] = 1.0 # bad values

                # actually produce the stacks. Single threaded, super inefficient...
                print(f'Producing {len(kb_xy)} stacks.')
                for i in range(0, len(kb_xy), 1):
                    makeStack(i, cutout, stamps,
                              im_data, mask_data,
                              kb_xy, image_params['mjd'], ra_grid,
                              self.config['v_arr'][0], self.config['v_arr'][1], self.config['v_arr'][2],
                              pad, hcos, c_ind=0,
                              mode='single_rate', av_mode='mean')
                t2 = time.time()

                print('Total stamp making time {:.2f}s. \n'.format(t2-t1))

                showExampleCutouts = False
                if showExampleCutouts:
                    for i in range(10,2000,100):
                        pyl.imshow(stamps[i,:,:])
                        pyl.show()

                runML = True
                if runML:


                    rots_dict = {'000': 0, '001': 0, '002': 0, '003': 0, '004': 0, '005': 0, '006': 0, '007': 0, '008': 0, '010': 0,
                                 '011': 0, '012': 0, '013': 0, '014': 0, '015': 0, '016': 2, '017': 2, '018': 2, '019': 0, '020': 0,
                                 '021': 0, '022': 2, '023': 2, '024': 2, '025': 2, '026': 0, '027': 0, '028': 0, '029': 0, '030': 2,
                                 '031': 2, '032': 2, '033': 2, '034': 0, '035': 0, '036': 0, '037': 0, '038': 2, '039': 2, '040': 2,
                                 '041': 2, '042': 0, '043': 0, '044': 0, '045': 0, '046': 2, '047': 2, '048': 2, '049': 2, '050': 0,
                                 '051': 0, '052': 0, '053': 0, '054': 2, '055': 2, '056': 2, '057': 2, '058': 0, '059': 0, '060': 0,
                                 '061': 0, '062': 2, '063': 2, '064': 2, '065': 2, '066': 0, '067': 0, '068': 0, '069': 0, '070': 2,
                                 '071': 2, '072': 2, '073': 2, '074': 0, '075': 0, '076': 0, '077': 0, '078': 2, '079': 2, '080': 2,
                                 '081': 0, '082': 0, '083': 0, '084': 2, '085': 2, '086': 2, '087': 2, '088': 2, '089': 2, '090': 2,
                                 '091': 2, '092': 2, '093': 2, '094': 2, '095': 2, '096': 2, '097': 2, '098': 2, '099': 2, '100': 1,
                                 '101': 1, '102': 3, '103': 3}

                    models = ['RNML_KBmod_modelSave_10.0_27.0_wRotAug_4',
                              'RNML_KBmod_modelSave_10.0_27.0_wRotAug_5',
                              'RNML_KBmod_modelSave_10.0_27.0_wRotAug_8',
                                 ]
                    for i in range(len(models)):
                        models[i] = '../ML_SNS/'+models[i]

                    conf_thresholds = [[0.4, 0.4, 0.4],]
                    if chunk_count==0:
                        keras_filter = filter_stamps_rn(models)

                    f = np.clip(stamps, -3500.0, np.max(stamps))
                    w = np.where(np.isnan(f))
                    f[w] = 0.0

                    if rots_dict['051']!=0:
                        f = np.rot90(f, k=-rots_dict[chip], axes=(1, 2))
                    f = np.expand_dims(f, axis=-1)

                    classes = keras_filter.filter_stamps(f, conf_thresholds, class_w_triplets=True)
                    print(classes)
                    keep['final_results'] = keep['final_results'][np.where(classes)]
                    #for k in ['stamps', 'results', 'times', 'lc', 'lc_index', 'all_stamps', 'psi_curves', 'phi_curves']:
                    #    if len(keep[k])>0:
                    #        new = []
                    #        for j in range(len(classes)):
                    #            if classes[j]:
                    #                new.append(keep[k][j])
                    #        keep[j] = new
                    #for k in ['final_results', 'min_LH_per_px', 'num_res_per_px']:
                    #    if len(keep[k])>0:
                    #        print(k,len(keep[k]), len(w[0]))
                    #        keep[k] = keep[k][w]
                    #
                    #exit()
            chunk_count += self.config['chunks_to_consider']
            filt_count+=1

        if self.config['do_clustering']:
            keep = kb_post_process.apply_clustering(keep, image_params)

        print('getting stamps')
        ## HACK
        print('NOT ACTUALLY RETRIEVING STAMPS.')
        #keep = kb_post_process.get_all_stamps(keep, search)
        #could try keep['all_stamps'] = [] and skip the above
        keep['all_stamps'] = []
        ## END HACK
        print('saving')
        kb_interface.save_results(self.config['res_filepath'], self.config['output_suffix'], keep)
        del(search)
        del(keep)
        end = time.time()
        print("Time taken for patch: ", end-start)
