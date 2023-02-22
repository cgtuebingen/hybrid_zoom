#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:23:28 2019

@author: Jieen Chen

This is the test code for slm_camera_ray_tracing scripts.
"""
# 1. slm_phase = slm.phase_function()
# 2. slm_profile = slm.phase_profile() # 3d plot slm phase function
# 3. boolean = intersect(object, ray) # intersection of the ray and the object/slm
# 4. object_texture = object.texture()
# 5. image = render(object, slm, camera)

import slm_camera_ray_tracing as slm_c
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Test slm
slm_mirror = slm_c.slm()
#slm_mirror.lens(15, 550*10**(-6))
slm_mirror.grating(15, 550*10**(-6))
plt.imshow(slm_mirror.phase_function)

# Test object plane
object_plane_image = cv2.imread("test_images/ISO_12233-reschart.png",cv2.IMREAD_GRAYSCALE)
object_depth_0 = slm_c.object_plane(object_plane_image, np.array([2952*10**4, 4724*10**4]), 10**3)
print(np.all(object_depth_0.resolution==object_plane_image.shape))
assert(object_depth_0.dist_plane_cam==10**3)
print(np.all(object_depth_0.image==object_plane_image))
print(np.all(object_depth_0.pitch==np.array([2952*10**4, 4724*10**4])/np.array(object_plane_image.shape)))

# Test camera
camera = slm_c.camera(100, np.array([220, 320]), np.array([9.936,14.472])) # camera class
assert(camera.focal_length==100)
print(np.all(camera.O==np.array([0,0,-100])))
print(np.all(camera.resolution==np.array([220,320])))
print(np.all(camera.sensor_size==np.array([9.936,14.472])))

# Test method: physical_coordinate
for y in [0, 110, 220]:
    for x in [0, 160, 320]:
        index = np.array([y,x])
        physical_coordinates = slm_c.physical_coordinate(index, camera)
        print(physical_coordinates)
        
# Test method: trace_ray
img = np.zeros([220,320])
slm_center = slm_mirror.resolution/2
slm_center = slm_center.astype(int)

for y in np.arange(0,220):
    for x in np.arange(0,320):
        index_sensor = np.array([y, x])
        # index_sensor = np.array([1104,1608])
        # index_sensor = np.array([0,0])
        physical_index = slm_c.physical_coordinate(index_sensor, camera)
        # print("Physical index is ")
        # print(physical_index)
        Q = np.array([physical_index[0],physical_index[1],camera.Q0[2]])
        rayD = Q-camera.O
        # print("The ray is ")
        # print(rayD)
        traced, pixel_slm, pixel_slm_physical = slm_c.trace_ray(camera.O, rayD, slm_mirror, camera)
        # print(traced)
        # print(pixel_slm)
        if traced == True:
            img[y,x] = 1
        else:
            continue
        #    print([pixel_slm[0]+slm_center[0], pixel_slm[1]+slm_center[1]])
        
        # Test method: compute_original_point
        # =============================================================================
        # This method computes the original point once the ray is defracted. 
        # If there is no modulation of the SLM, the original points should be [0,0,0]
        # =============================================================================
        rayR = rayD
        
        # Compute original point of the diffracted ray
        O_diffracted_ray = slm_c.compute_original_point(rayR, pixel_slm_physical, camera, slm_mirror)
        print(O_diffracted_ray)
plt.imshow(img)
