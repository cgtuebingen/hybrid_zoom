#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:42:35 2019

@author: chenjiee

Image formation model based on image-order raytracing. The camera is a
camera with a thin lens. The aperture of the thin lens collects the light field
 generated after the SLM. The SLM is simplified as a pure refractive device. We
 render the image through a two-step raytracer. Firstly, we trace the sensor 
 pixel to all SLM pixels. Secondly, we trace each SLM pixel to the object irra-
diance. The camera is focused on infinity.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

class slm():
    def __init__(self, phase_function=np.zeros([1080,1920]), size=np.array([8.64,15.36]), dist_slm_cam = 100):
        """This initializes the SLM class. It describes the SLM's parameters, phase
        functions, physical size, and image resolution."""
        self.dist_plane_cam = dist_slm_cam # distance to the project plane
        self.O = np.array([0,0,self.dist_plane_cam])
        self.phase_function = phase_function
        self.physical_size = size
        self.resolution = np.array(phase_function.shape)
        self.pitch = np.array(size)/np.array(self.resolution)
        
    def lens(self, focal_length, wavelength):
        """This function generates a fresnel lens with entered focal length."""
        y = np.linspace(-self.resolution[0]/2,self.resolution[0]/2+1, self.resolution[0])
        x = np.linspace(-self.resolution[1]/2,self.resolution[1]/2+1, self.resolution[1])
        X,Y = np.meshgrid(x,y)
        Y *= self.pitch[0]
        X *= self.pitch[1]
        wave_number = 2*np.pi/wavelength
        self.phase_function = wave_number*(X**2+Y**2)/(2*focal_length)
        
    def grating(self, shift, wavelength):
        """This method generates a grating that has a slope defined by 'shift'."""
        y = np.linspace(-self.resolution[0]/2,self.resolution[0]/2+1, self.resolution[0])
        x = np.linspace(-self.resolution[1]/2,self.resolution[1]/2+1, self.resolution[1])
        X,Y = np.meshgrid(x,y)
        Y *= self.pitch[0]
        X *= self.pitch[1]
        wave_number = 2*np.pi/wavelength
        self.phase_function = wave_number*X/(2*shift)
        
    def nurb_freefrom(self, par0, par1):
        """This method generates a freeform surface profile based on the NURBS Python module."""
        # nurb_surface = nurbs(par0,par1)
        # phase_function = flatten(nurb_surfcae)
        # self.phase_function = phase_function
        
    def load_phase_function(self, filename):
        """This method loads the phase function and convert it to the ray modulation."""
        phase_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        phase_img = phase_img / 255.0
        self.phase_function = phase_img
        
    def cubic(self):
        """This method generates a cubic surface. It follows the formula f = x^3 + y^3"""
        y = np.linspace(-self.resolution[0]/2,self.resolution[0]/2+1, self.resolution[0])
        x = np.linspace(-self.resolution[1]/2,self.resolution[1]/2+1, self.resolution[1])
        X,Y = np.meshgrid(x,y)
        Y *= self.pitch[0]
        X *= self.pitch[1]
        self.phase_function = (X**3 + Y**3)
        
    def modulation(self, pixel_slm, ray, factor):
        """This method generates the new ray through the modulation function."""
        temp_ray_z = ray[2] - factor * self.phase_function[pixel_slm[0], pixel_slm[1]]
        ray = np.array([ray[0], ray[1], temp_ray_z])
        return ray
        
class camera():
    def __init__(self,focal_length, resolution, sensor_size):
        """This initializes the camera class. It describes the camera's parameters,
        such as focal length, original point, and resolution."""
        self.focal_length = focal_length
        self.O = np.array([0,0,-self.focal_length])
        self.resolution = resolution
        self.sensor_size = sensor_size
        self.pitch = np.array(sensor_size)/np.array(resolution)
        self.Q0 = np.array([0,0,0])
    
class object_plane():
    def __init__(self, image, physical_size, dist_plane_cam):
        """This initializes the object plane class. It describes the plane at certain
        depth with respect to the sensor. """
        self.dist_plane_cam = dist_plane_cam
        self.image = image
        self.resolution = image.shape
        self.physical_size = physical_size
        self.O = np.array([0,0,self.dist_plane_cam])
        self.pitch = np.array(physical_size)/np.array(image.shape)
    
def normalizeZ(x):
    return x/x[2]

def physical_coordinate(index, planes):
    """This method computes the physical coordinates of one pixel"""
    center = np.array(planes.resolution)/2
    center = center.astype(int)
    return np.array((index-center)*planes.pitch)

def compute_original_point(ray, intersection_pt_index, camera, slm):
    """This method computes the original point of the angular modulated ray."""
    ray_normalized = normalizeZ(ray)
    O = intersection_pt_index - (slm.dist_plane_cam + camera.focal_length) * ray_normalized
    return O
    
def trace_ray(O, ray, plane, camera):
    """This method computes the intersection point between the object/slm and the ray.
    It returns the boolean value, and the pixel index."""
    ray_normalized = normalizeZ(ray) # The normalized ray for tracing the intersect 
    intersection_point = ray_normalized * (plane.dist_plane_cam + camera.focal_length) + O # Compute intersection point 
    
    # Compute range of physical size of the SLM or object
    zero_point = np.array([0,0])
    physical_index = physical_coordinate(zero_point, plane)
    physical_range = np.array([physical_index[0], physical_index[1], -physical_index[0], -physical_index[1]])
    
    if intersection_point[0] <= physical_range[0] or intersection_point[0] > physical_range[2] or intersection_point[1] <= physical_range[1] or intersection_point[1] > physical_range[3]:
        return False, False, False
    else:
        # intersection_point to intersection_index
        intersection_index = intersection_point[:2]/plane.pitch
        
        # normalize the index
        plane_center = np.array(plane.resolution)/2
        plane_center = plane_center.astype(int)
        intersection_index = intersection_index.astype(int) + plane_center
        
        return True, intersection_index, intersection_point
    
def render():
    """This method renders the ray-traced image. It contains the following procedures:
        initialize camera, image_canvas
        for all pixels on the camera:
            define the ray
            trace_ray
            --1. intersect
            --2. angle_modulation
            generate_reflect_ray
            reflection
            --1. intersect
            --2. pixel color"""
            
    # Read in object plane image
    # image_file=input("Please enter object plane image file name\n")
    image_file = "C:/Users/Jieen/Documents/PhD Thesis/code/hybrid_zoom/simulation/ISO_12233-reschart.png"
    # image_file = "C:/Users/Jieen/Documents/PhD Thesis/code/hybrid_zoom/simulation/window.png"
    # image_file = "C:/Users/Jieen/Documents/PhD Thesis/code/hybrid_zoom/simulation/toolbox.png"
    object_plane_image = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
    object_plane_image = object_plane_image.astype(float) / 255.
    object_plane_0 = object_plane(object_plane_image, np.array(object_plane_image.shape)/10, 10**3)

    # Initialize parameters
    # cam = camera(100, np.array([2208, 3216]), np.array([9.936,14.472])) # camera class
    # cam = camera(100, np.array([440, 640]), np.array([9.936,14.472])) # camera class
    cam = camera(100, np.array([880, 1280]), np.array([9.936, 14.472]))
    # cam = camera(100, np.array([1760, 2560]), np.array([9.936, 14.472]))
    h, w = cam.resolution
    slm_mirror = slm(np.zeros([1080,1920]), np.array([8.64,15.36]), 50) # slm class
    
    #fresnel_lens_focal_length = 15 # focal length for the phase function
    # wavelength = 550*10**(-6) # wavelength for the phase function
    # slm_mirror.lens(fresnel_lens_focal_length, wavelength)
    # slm_mirror.cubic()
    
    # filename = input("Please enter the SLM image file\n")
    # slm_filename = "C:/Users/Jieen/Documents/PhD Thesis/code/hybrid_zoom/simulation/moving_sine_stripes_frequencies/moving_sine_stripes_frequencies10.png"
    slm_filename = "C:/Users/Jieen/Documents/PhD Thesis/code/hybrid_zoom/simulation/slm_patterns/cubic_phase.png"
    # slm_filename = "C:/Users/Jieen/Documents/PhD Thesis/code/hybrid_zoom/simulation/radial_wave_animation/radial_wave_animation.0001.png"
    slm_mirror.load_phase_function(slm_filename)
    
    # Modulation mode
    modulation_yn = input("Please enter if you'd like to modulate the image (y/n):\n")
    if modulation_yn == "y":
        factor = input("Please enter factor:\n")
        factor = float(factor)
    # factor = 0.8
        
    # Scan all pixels
    # Pixel coordinates
    img = np.zeros(cam.resolution)
    for x in np.arange(0, w):
        for y in np.arange(0,h):
            # Define ray
            index = np.array([y,x])
            physical_index = physical_coordinate(index, cam)
            Q = np.array([physical_index[0],physical_index[1],cam.Q0[2]])
            rayD = Q-cam.O
            O = cam.O
            
            # Trace_ray
            traced, pixel_slm, pixel_slm_physical = trace_ray(O, rayD, slm_mirror, cam)
            if traced == False:
                continue
            
            # Generate diffracted ray
            if modulation_yn == "y":
                rayR = slm_mirror.modulation(pixel_slm, rayD, factor)
                img_filename = "modulated_" + replace(str(factor)) + ".png"
            else:
                rayR = rayD
                img_filename = "non_modulated.png"
            
            # Compute original point of the diffracted ray
            O_diffracted_ray = compute_original_point(rayR, pixel_slm_physical, cam, slm_mirror)
            
            # Trace reflected ray
            traced_ref, pixel_object, pixel_object_physical = trace_ray(O_diffracted_ray, rayR, object_plane_0, cam)
            if traced_ref == False:
                continue
            
            # Record color
            pixel_object = pixel_object.astype(int)
            pix_color = object_plane_0.image[pixel_object[0], pixel_object[1]]
            img[y, x] = pix_color
    return img, img_filename

def replace(str1): 
    maketrans = str1.maketrans 
    final = str1.translate(maketrans('.', '_')) 
    return final 

if __name__=='__main__':
    image, img_filename = render()
    plt.imshow(image, cmap="gray")
    directory = "" ##"./results"
    resolution = "1k"
    # modulation = "frequency_3"
    modulation = "spiral"
    # modulation = "radial_1"
    img_filename = directory + "/" + resolution + "_" + modulation + "_" + img_filename
    plt.imsave(img_filename, image, cmap="gray")
