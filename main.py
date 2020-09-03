import numpy as np
import nibabel as nib
import argparse

# concept:
#   LGN_L tif generated from PT.img with certain threshold
#       (default at 0.001 * PT_MAX) - want to draw DL_L tif
#   b0 - draw bounding box -> DL_L_bb.img/hdr

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--probtractx", required=True, help="path to PT.img")
ap.add_argument("-f", "--filledboundingbox", required=True,
                help="path to filled bounding box")
ap.add_argument("-t", "--thresholdfactor", required=True,
                help="threshold factor * pt_max")
ap.add_argument("-o", "--outputname", required=True, help="output name")
args = vars(ap.parse_args())

# get pixel data of pt
pt = nib.load(args["probtractx"])
pt_pixel_data = pt.get_fdata()

# apply threshold
thres = np.max(pt_pixel_data) * args["thresholdfactor"]
thres_mask = pt_pixel_data < thres
pt_pixel_data[thres_mask] = 0

# get pixel data of filled bb
fbb = nib.load(args["filledboundingbox"])

fbb_pixel_data = fbb.get_fdata()
# change this to mask
fbb_pixel_mask = np.where(fbb_pixel_data > 0, 1, 0)
# use mask to intersect

pt_pixel_roi = pt_pixel_data * fbb_pixel_mask
# 1 or 100?-verify with uint8 .img/.hdr

pt_pixel_roi = np.where(pt_pixel_roi > 0, 1, 0)
# export output as .img .hdr pair
