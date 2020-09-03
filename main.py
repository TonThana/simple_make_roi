import numpy as np
import nibabel as nib
import argparse
import overlayhelper
# concept:
#   LGN_L tif generated from PT.img with certain threshold
#       (default at 0.001 * PT_MAX) - want to draw DL_L tif
#   b0 - draw bounding box -> DL_L_bb.img/hdr

# example ---
# python main.py -p /data1/cl/Project_Panitha/DISEASED_CONVERTED_3/
# BENCHAPHON_NIAMSIN/DTI/LGN_L.pt/PT.img -f /data1/cl/Project_Panitha/
# DISEASED_CONVERTED_3/BENCHAPHON_NIAMSIN/DTI/filledBBox/ldl-l-fbb.img -t
# 0.001 -o dl-l

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--probtractx", required=True, help="path to PT.img")
ap.add_argument("-f", "--filledboundingbox", required=True,
                help="path to filled bounding box i.e. dl-l-fbb.img")
ap.add_argument("-t", "--thresholdfactor", default=0.001,
                help="threshold factor * pt_max", type=float)
ap.add_argument("-o", "--outputname", required=True, help="output name")
args = vars(ap.parse_args())

# get pixel data of pt
pt = nib.load(args["probtractx"])

pt_pixel_data = pt.get_fdata()

# apply threshold
thres = np.max(pt_pixel_data) * args["thresholdfactor"]
# print(np.max(pt_pixel_data), args["thresholdfactor"])
thres_mask = pt_pixel_data < thres
# print(thres_mask)
pt_pixel_data[thres_mask] = 0

# # get pixel data of filled bb
fbb = nib.load(args["filledboundingbox"])

fbb_pixel_data = fbb.get_fdata()
fbb_pixel_data = np.squeeze(fbb_pixel_data)
# overlayhelper.overlay_helper(
#     roiData=fbb_pixel_data, imageData=pt_pixel_data, title="pt")

# # change this to mask
fbb_pixel_mask = np.where(fbb_pixel_data > 0, 1, 0)
# print(np.max(fbb_pixel_mask))
# # use mask to intersect

pt_roi = pt_pixel_data * fbb_pixel_mask
print(np.max(pt_roi))
pt_roi = np.where(pt_roi > 0, 1, 0).astype("int16")
overlayhelper.overlay_helper(imageData=pt_roi, title=args["outputname"])
pt_roi_a75 = nib.AnalyzeImage(pt_roi, affine=np.eye(4))
nib.save(pt_roi_a75, args["outputname"])
# # export output as .img .hdr pair
