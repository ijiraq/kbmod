#! /usr/bin/env python

from astropy.io import fits
from astropy import wcs, time
from analysis_utils import Interface
import numpy as np, pylab as pyl
import glob
from numpy import ma

badflags = 1+512+8+64+16+4+256+2+128+2048+4096

def calc_ecliptic_angle(fn):
    #fn = '/media/fraserw/rocketdata/Projects/kbmod/DATA/rerun/diff_warpCompare/deepDiff/03093/HSC-R2/warps/000/DIFFEXP-0220262-0220380-000.fits'

    with fits.open(fn) as han:
        WCS = wcs.WCS(han[1].header)


    interface = Interface()
    return(interface._calc_ecliptic_angle(WCS))



def makeStack(i, cutout, stamps, im_data, mask_data, kb_xy, times, ra_grid, v_min, v_max, n_rates, pad, hcos, c_ind, mode = 'full_grid', av_mode='median'):
    #global v_min, v_max, n_rates
    if mode == 'full_grid':
        print(i+1, len(kb_xy))
        for r, rag in enumerate(ra_grid):
            for j in range(len(times)):
                x,y = kb_xy[i, :2] + (times[j] - times[0])*rag
                x_int, y_int = int(x)+pad, int(y)+pad

                try:
                    cutout[c_ind,j,:,:] = im_data[j, y_int-hcos:y_int+hcos+1, x_int-hcos:x_int+hcos+1]
                    cutout.mask[c_ind,j,:,:] = mask_data[j, y_int-hcos:y_int+hcos+1, x_int-hcos:x_int+hcos+1]
                except:
                    print(x_int, y_int, x, y, pad, hcos)
                    print(rag)
                    exit()
            if av_node == 'mean':
                stamp = np.mean(cutout[c_ind],axis=0)
            elif av_node == 'median':
                stamp = np.median(cutout[c_ind],axis=0)
            stamps[i, r, :, :] = stamp
    elif mode == 'triplet_grid':
        print(i+1, len(kb_xy))
        r_step = np.zeros((2)) + (v_max-v_min)/n_rates
        best_rate = kb_xy[i, 2:]
        trip_ra_grid = np.array([[-1, -1], [-1, 0], [-1, 1], \
                                 [0, -1], [0, 0], [0, 1], \
                                 [1, -1], [1, 0], [1, 1],])*r_step + best_rate

        for r, rag in enumerate(trip_ra_grid):
            for j in range(len(times)):
                x,y = kb_xy[i, :2] + (times[j] - times[0])*rag
                x_int, y_int = int(x)+pad, int(y)+pad

                try:
                    cutout[c_ind,j,:,:] = im_data[j, y_int-hcos:y_int+hcos+1, x_int-hcos:x_int+hcos+1]
                    cutout.mask[c_ind,j,:,:] = mask_data[j, y_int-hcos:y_int+hcos+1, x_int-hcos:x_int+hcos+1]
                except:
                    print(x_int, y_int, x, y, pad, hcos)
                    print(rag)
                    exit()
            if av_mode=='mean':
                stamp = ma.mean(cutout[c_ind],axis=0)
            elif av_mode=='median':
                stamp = ma.median(cutout[c_ind],axis=0)
            stamps[i, r, :, :] = stamp

    elif mode == 'single_rate':
        for j in range(len(times)):
            x,y = kb_xy[i, :2] + (times[j] - times[0])*kb_xy[i, 2:]
            x_int, y_int = int(x)+pad, int(y)+pad

            try:
                cutout[c_ind,j,:,:] = im_data[j, y_int-hcos:y_int+hcos+1, x_int-hcos:x_int+hcos+1]
                cutout.mask[c_ind,j,:,:] = mask_data[j, y_int-hcos:y_int+hcos+1, x_int-hcos:x_int+hcos+1]
            except:
                print(x_int, y_int, x, y, pad, hcos)
                print(rag)
                exit()
        if av_mode == 'mean':
            stamp = np.mean(cutout[c_ind],axis=0)
        elif av_mode == 'median':
            stamp = np.median(cutout[c_ind],axis=0)
        stamps[i, :, :] = stamp

def create_times_file(img_path, file_fn):
    files = glob.glob(f'{img_path}/DIFFEXP*fits')
    files.sort()

    outhan = open(file_fn, 'w+')
    for ii, fn in enumerate(files):
        with fits.open(fn) as han:
            t = time.Time(han[0].header['DATE-AVG'], format='isot')
        n = fn.split('DIFFEXP-')[1].split('-')[0]
        print('{} {:12.6f}'.format(int(float(n)),t.mjd),file=outhan)
    outhan.close()

def visualize_mask_plane(mask, bits):
    if type(bits) == float:
        pyl.imshow(mask&bits, origin='lower')
        pyl.show()
    else:
        keys = list(bits.keys())
        values = list(bits.values())

        n = int(len(bits)**0.5)+1
        fig = pyl.figure('Mask')
        for i in range(n):
            for j in range(n):
                c = i*n + j
                if c>=len(bits):
                    break
                if c == 0:
                    sp = fig.add_subplot(n,n,c+1)
                    sp.imshow(mask&2**values[c], origin='lower')
                    sp.set_title(keys[c])
                    pyl.xticks(color='w')
                    pyl.yticks(color='w')
                else:
                    SP = fig.add_subplot(n,n,c+1, sharex=sp, sharey=sp)
                    SP.imshow(mask&2**values[c], origin='lower')
                    SP.set_title(keys[c])
                    pyl.xticks(color='w')
                    pyl.yticks(color='w')
        pyl.show()
