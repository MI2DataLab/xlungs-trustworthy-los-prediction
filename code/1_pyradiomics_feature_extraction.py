import pathlib
from pathlib import Path

import pydicom
from pydicom.pixel_data_handlers import apply_voi_lut, apply_rescale

import radiomics
from radiomics import featureextractor

import os
import cv2
import json
import hashlib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

path_orig_dicoms = ''
path_orig_masks = ''
path_transf_dicoms = ''
path_transf_masks = ''

path_orig_dicoms = Path(path_orig_dicoms)
path_orig_masks = Path(path_orig_masks)
path_transf_dicoms = Path(path_transf_dicoms)
path_transf_masks = Path(path_transf_masks)

label = 1
channel = 0

hash_f = 'deleted_for_safety'
extractor = featureextractor.RadiomicsFeatureExtractor(correctMask = True, label = label)

to_pop = [
    'diagnostics_Image-original_Spacing',
    'diagnostics_Configuration_Settings',
    'diagnostics_Configuration_EnabledImageTypes'
]

to_transf = [
    'diagnostics_Mask-original_BoundingBox',
    'diagnostics_Mask-original_CenterOfMassIndex',
    'diagnostics_Mask-original_CenterOfMass',
    'diagnostics_Image-original_Size',
    'diagnostics_Mask-original_Spacing', 
    'diagnostics_Mask-original_Size'
]

results = dict()
idxs = []

jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
n_jobs = 200
paths = list(path_orig_masks.iterdir())
n_rows = len(paths)
n_per_job = int(n_rows / n_jobs)
paths.sort()
paths = paths[(jobid * n_per_job):((jobid + 1) * n_per_job)]

for i, patient_id in enumerate(paths):
    # get patient_id
    idx = patient_id.parts[-1][:-4]
    idxs.append(idx)
    print(idx, flush = True)
    # hash it
    idx_hash = hash_f(idx)
    # get paths for the mask and corresponding dicom
    path_mask = path_orig_masks / patient_id.parts[-1]
    path_dicom = path_orig_dicoms / idx_hash
    path_output_dicom = path_transf_dicoms / (idx + '.dcm')
    path_output_mask = path_transf_masks / (idx + '.dcm')
    try:
        path_dicom = list(path_dicom.glob('**/*.dcm'))[0]
        print(path_dicom, flush = True)
    except IndexError:
        print('File not found', flush = True)
        continue
    mask = sitk.ReadImage(path_mask)
    image = sitk.ReadImage(path_dicom)

    # get size for resize
    img_size = image.GetSize()[:2]
    
    # prepare mask
    selector = sitk.VectorIndexSelectionCastImageFilter()
    selector.SetIndex(channel)
    mask = selector.Execute(mask)
    mask = sitk.GetArrayFromImage(mask)
    mask[mask > 0] = label
    mask = cv2.resize(
        mask,
        img_size,
        interpolation = cv2.INTER_NEAREST)
    mask = mask[None, ...]
    mask = sitk.GetImageFromArray(mask)
    mask.SetSpacing(image.GetSpacing())

    result = extractor.execute(image, mask, label = label)
    for t in to_transf:
        for j in range(len(result[t])):
            result[f'{t}_{j}'] = result[t][j]
    for t in to_transf:
        result.pop(t)
    for p in to_pop:
        result.pop(p)

    if i == 0:
        for k, v in result.items():
            results[k] = [v]
    else:
        for k, v in result.items():
            results[k].append(v)
    
    # save extracted features
    df = pd.DataFrame.from_dict(results, orient = 'columns')
    df.index = idxs
    df.to_csv(f'data/features/_{jobid}.csv')
