import json
import numpy
import math
from PIL import Image 
import matplotlib.pyplot as plt

import shutil
import os
import bisect


import argparse
import zipfile
from tqdm import tqdm

import copy

inzip = 'ffhq_dataset.json'
outzip = 'output_dataset.json'

with open(inzip) as f:
    dataset = json.load(f)

#--------------------------------------------------------------------------------------------------

# Get OpenCV's function to extract [pitch, yaw, roll] from a 4x4 pose matrix

# Straight copy-pasta'd from the opencv source.
# https://github.com/egonSchiele/OpenCV/blob/master/tests/python/transformations.py

_EPS = numpy.finfo(float).eps * 4.0

_NEXT_AXIS = [1, 2, 0, 1]

_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}


def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.
    axes : One of 24 axis sequences as string or encoded tuple
    Note that many Euler angle triplets can describe one matrix.
    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print axes, "failed"
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

#--------------------------------------------------------------------------------------------------

# this is a long and convoluted script which rebalances ffhq by:
# 1: getting the min and max yaw over the dataset
# 2: splitting the dataset into N=9 uniform size arcs across the range
#     (with possibly differing number of images in each arc)
# 3: adding duplicates of images in each ark,
#     with arcs further from the center receiving more duplicates
# 
# The new dataset is still biased towards frontal facing images
# but much less so than before.

l = []

for label in dataset['labels']:
    m = numpy.array(label[1][:16]).reshape(4,4)
    angles = euler_from_matrix(m)
    l.append(angles)

indices = list(range(0, len(l)))
sorted_by_yaw = sorted(indices, key=lambda x:l[x][1])

min_yaw = l[sorted_by_yaw[0]][1]
max_yaw = l[sorted_by_yaw[-1]][1]
min_yaw, max_yaw

bins = 9
bin_indices = []
for x in numpy.linspace(min_yaw, max_yaw, bins+1):
    num_images = bisect.bisect_left(list(l[x][1] for x in sorted_by_yaw), x)
    bin_indices.append(num_images)
bin_indices

for a, b in zip(bin_indices[:-1], bin_indices[1:]):
    print(b-a)


print("Initial dataset size:", len(dataset['labels']))
num_multiplier = [16, 8, 4, 2, 1, 2, 4, 8, 16]
num_replicas = {}
for i, x in enumerate(sorted_by_yaw):
    num_multiplier_i = bisect.bisect_left(bin_indices, i) - 1
    num_replicas[x] = num_multiplier[num_multiplier_i]

with open('intermediate_duplicates.json', 'w') as f:
    json.dump(num_replicas, f)