import sys
import pyredner
import torch
import os
import time
import math
import numpy
import random

# For loop progress bar.
from tqdm import tqdm

#from chamferdist.chamferdist import ChamferDistance


# GROUND TRUTH.
meshfile1 = sys.argv[1]
# RECONSTRUCTED.
meshfile2 = sys.argv[2]


#torch.cuda.init()


########################## LOAD TARGET #########################

print("Loading target: " + meshfile1)
_, mesh_list1, _ = pyredner.load_obj(meshfile1)


print("Loading target: " + meshfile2)
_, mesh_list2, _ = pyredner.load_obj(meshfile2)

_, mesh1 = mesh_list1[0]
_, mesh2 = mesh_list2[0]

print("Putting vertices on GPU...")
verts1 = mesh1.vertices.cuda().contiguous()
verts2 = mesh2.vertices.cuda().contiguous()
print("Done")

print("Computing chamfer distance from ground truth -> reconstructed")
v12_mins = torch.zeros(verts1.shape[0])
for i in tqdm(range(verts1.shape[0])):
    # Vertex from mesh 1.
    v = verts1[i]
    distances = torch.sum((v - verts2) ** 2, dim=1)
    v12_mins[i] = torch.min(distances, dim=0)[0] ** 0.5

print("Computing chamfer distance from reconstructed -> ground truth")
v21_mins = torch.zeros(verts2.shape[0])
for i in tqdm(range(verts2.shape[0])):
    # Vertex from mesh 2.
    v = verts2[i]
    distances = torch.sum((v - verts1) ** 2, dim=1)
    v21_mins[i] = torch.min(distances, dim=0)[0] ** 0.5

print("DISTANCE 1->2:", torch.sum(v12_mins))
print("DISTANCE 2->1:", torch.sum(v21_mins))

# F1 distance is the harmonic mean of the precision and the recall.
threshold = 10e-4
precision = 100.0 * (v21_mins[v21_mins < threshold].shape[0]) / verts2.shape[0]
recall = 100.0 * (v12_mins[v12_mins < threshold].shape[0]) / verts1.shape[0]
print("precision:", precision)
print("recall:", recall)
print("F1 DISTANCE:", (2.0 * precision * recall) / (precision + recall))

'''
# Verts 1 -> verts 2
v1_stack = ((torch.unsqueeze(verts1, 1)).repeat(1, verts2.shape[0], 1))
v2_stack = ((torch.unsqueeze(verts2, 0)).repeat(verts1.shape[0], 1, 1))
print("V2 shape:", v2_stack.shape)
print("V1 shape:", v1_stack.shape)
v12_dist = torch.sum((v1_stack - v2_stack) ** 2, dim=2)
min_dist_12 = torch.min(v12_dist, dim=1)[0]


# Verts 2 -> verts 1
v2_stack = ((torch.unsqueeze(verts2, 1)).repeat(1, verts1.shape[0], 1))
v1_stack = ((torch.unsqueeze(verts1, 0)).repeat(verts2.shape[0], 1, 1))
print("V2 shape:", v2_stack.shape)
print("V1 shape:", v1_stack.shape)
v21_dist = torch.sum((v1_stack - v2_stack) ** 2, dim=2)
min_dist_21 = torch.min(v21_dist, dim=1)[0]
'''

'''

print("DISTANCE 1->2:", torch.sum(min_dist_12))
print("DISTANCE 2->1:", torch.sum(min_dist_21))
print("TOTAL DISTANCE:", torch.sum(min_dist_21) + torch.sum(min_dist_12))
'''

# c = ChamferDistance()
# d1, d2, _, _ = c(verts1, verts2)
# print("DISTANCE:", torch.sum(d1) + torch.sum(d2))
