import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import math
from scipy.signal import correlate2d
from scipy.stats import norm
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter
import os
from os.path import join, basename
import pickle
import plenopticam as pcam
from matplotlib import animation
from IPython.display import HTML


def convert():
    """face_path = '\\15463\\assgn5\\data\\input_'
    my_path = '../data/my_data_new/DSC_'

    with open('../data/lfp_img_align.pkl', 'rb') as f:
        lfp_img_align = pickle.load(f)
        
    print("My shape is:", lfp_img_align.shape)
    plt.figure()
    plt.imshow(lfp_img_align/lfp_img_align.max(), interpolation='none')
    plt.grid(False)
    plt.title('Aligned Illum image')
    plt.show()"""

    cfg = pcam.cfg.PlenopticamConfig()
    cfg.default_values()
    cfg.params[cfg.lfp_path] = '../data/raw_lfr.lfr'
    cfg.params[cfg.cal_path] = '../data/caldata-B5144703440.tar'
    cfg.params[cfg.opt_cali] = True
    cfg.params[cfg.ptc_leng] = 13
    cfg.params[cfg.cal_meth] = pcam.cfg.constants.CALI_METH[3]
    sta = pcam.misc.PlenopticamStatus()

    reader = pcam.lfp_reader.LfpReader(cfg, sta)
    reader.main()
    lfp_img = reader.lfp_img

    """    plt.figure()
    plt.imshow(lfp_img, cmap='gray', interpolation='none')
    plt.grid(False)
    plt.title('Raw Illum image')
    plt.show()"""

    cal_finder = pcam.lfp_calibrator.CaliFinder(cfg, sta)
    ret = cal_finder.main()
    wht_img = cal_finder.wht_bay

    """plt.figure()
    plt.imshow(wht_img, cmap='gray', interpolation='none')
    plt.grid(False)
    plt.title('Raw white calibration image')
    plt.show()"""

    cal_obj = pcam.lfp_calibrator.LfpCalibrator(wht_img, cfg, sta)
    ret = cal_obj.main()
    cfg = cal_obj.cfg

    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['pdf.fonttype'] = 42

    y_coords = [row[0] for row in cfg.calibs[cfg.mic_list]]
    x_coords = [row[1] for row in cfg.calibs[cfg.mic_list]]

    s = 3
    h, w, c = wht_img.shape if len(wht_img.shape) == 3 else wht_img.shape + (1,)
    hp, wp = [39]*2
    fig, axs = plt.subplots(s, s, facecolor='w', edgecolor='k', figsize=(9, 9))
    markers = ['r.', 'b+', 'gx']
    labels = [r'$\bar{\mathbf{c}}_{j,h}$', 
            r'$\tilde{\mathbf{c}}_{j,h}$ with $\beta=0$', 
            r'$\tilde{\mathbf{c}}_{j,h}$ with $\beta=1$']
    m = 2

    """for i in range(s):
        for j in range(s):
            # plot cropped image part
            k = h//2 + (i-s//2) * int(h/2.05) - hp // 2
            l = w//2 + (j-s//2) * int(w/2.05) - wp // 2
            axs[i, j].imshow(wht_img[k:k+hp, l:l+wp, ...], cmap='gray')

            # plot centroids in cropped area
            coords_crop = [(y, x) for y, x in zip(y_coords, x_coords) 
                        if k <= y <= k+hp-.5 and l <= x <= l+wp-.5]
            y_centroids = [row[0] - k for row in coords_crop]
            x_centroids = [row[1] - l for row in coords_crop]
            axs[i, j].plot(x_centroids, y_centroids, markers[m], 
                        markersize=10, label=labels[m])
            axs[i, j].grid(False)

            if j == 0 or i == s-1:
                if j == 0 and i == s-1:
                    axs[i, j].tick_params(top=False, bottom=True, left=True, right=False,
                                        labelleft=True, labelbottom=True)
                    axs[i, j].set_yticklabels([str(k), str(k+hp//2), str(k+hp)])
                    axs[i, j].set_xticklabels([str(l), str(l+wp//2), str(l+wp)])
                elif j == 0:
                    axs[i, j].tick_params(top=False, bottom=True, left=True, right=False,
                                        labelleft=True, labelbottom=False)
                    axs[i, j].set_yticklabels([str(k), str(k+hp//2), str(k+hp)])
                elif i == s-1:
                    axs[i, j].tick_params(top=False, bottom=True, left=True, right=False,
                                        labelleft=False, labelbottom=True)
                    axs[i, j].set_xticklabels([str(l), str(l+wp//2), str(l+wp)])

            else:
                axs[i, j].tick_params(top=False, bottom=True, left=True, right=False,
                                    labelleft=False, labelbottom=False)

            axs[i, j].set_yticks(range(0, hp+1, hp//2))
            axs[i, j].set_xticks(range(0, wp+1, wp//2))

    plt.legend(loc='upper left', bbox_to_anchor=(-2.8, -.1), fancybox=True, fontsize=14)
    # set common labels
    fig.text(0.52, 0.05, 'Horizontal dimension [px]', ha='center', va='center', fontsize=18)
    fig.text(0.06, 0.5, 'Vertical dimension [px]', ha='center', va='center', rotation='vertical', fontsize=18)

    fig.tight_layout()
    plt.savefig('centroid_LMA_fits+regular_div.pdf', bbox_inches="tight")
    plt.show()"""

    ret = cfg.load_cal_data()
    aligner = pcam.lfp_aligner.LfpAligner(lfp_img, cfg, sta, wht_img)
    ret = aligner.main()
    lfp_img_align = aligner.lfp_img

    with open(join(cfg.exp_path, 'lfp_img_align.pkl'), 'rb') as f:
        lfp_img_align = pickle.load(f)
        
    """plt.figure()
    plt.imshow(lfp_img_align/lfp_img_align.max(), interpolation='none')
    plt.grid(False)
    plt.title('Aligned Illum image')
    plt.show()"""

    try:
        from plenopticam.lfp_reader import LfpDecoder
        # try to load json file (if present)
        json_dict = cfg.load_json(cfg.exp_path, basename(cfg.exp_path)+'.json')
        cfg.lfpimg = LfpDecoder.filter_lfp_json(json_dict, cfg.lfpimg)
    except FileNotFoundError:
        pass

    extractor = pcam.lfp_extractor.LfpExtractor(lfp_img_align, cfg, sta)
    ret = extractor.main()
    vp_img_arr = extractor.vp_img_arr

    view_obj = pcam.lfp_extractor.LfpViewpoints(vp_img_arr=vp_img_arr)
    vp_view = view_obj.central_view

    """plt.figure()
    plt.imshow(vp_view/vp_view.max(), interpolation='none')
    plt.grid(False)
    plt.title('Central sub-aperture image view')
    plt.show()"""

    view_obj = pcam.lfp_extractor.LfpViewpoints(vp_img_arr=vp_img_arr)
    vp_stack = view_obj.views_stacked_img

    plt.figure()
    plt.imshow(vp_stack/vp_stack.max(), interpolation='none')
    plt.grid(False)
    plt.title('All sub-aperture images view')
    plt.show()

    print("To clarify, my shape is:", vp_stack.shape)
    skimage.io.imsave('../data/png_img.png', vp_stack/vp_stack.max())
    skimage.io.imsave('../data/tif_img.tif', vp_stack/vp_stack.max())

im = skimage.io.imread('../data/png_img.png')
skimage.io.imshow(im)
plt.show()

height, width, _ = im.shape
height //= 13
width //= 13
all_ims = np.zeros((13,13,height,width,3))
for i in range(13):
    for j in range(13):
        all_ims[]

"""with open('../data/lfp_img_align.pkl', 'rb') as f:
    lfp_img_align = pickle.load(f)
    
plt.figure()
plt.imshow(lfp_img_align/lfp_img_align.max(), interpolation='none')
plt.grid(False)
plt.title('Aligned Illum image')
plt.show()

try:
    from plenopticam.lfp_reader import LfpDecoder
    # try to load json file (if present)
    json_dict = cfg.load_json(cfg.exp_path, basename(cfg.exp_path)+'.json')
    cfg.lfpimg = LfpDecoder.filter_lfp_json(json_dict, cfg.lfpimg)
except FileNotFoundError:
    pass"""