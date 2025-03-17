def DiSpect_SepEcho(ksp):
    import numpy as np
    import math
    new_shp=ksp.shape
    mult = 1
    for i in new_shp[1:]:
        mult *= i
    E=np.array([[1,np.exp(1j*math.pi*2/3),np.exp(-1j*math.pi*2/3)],[1,np.exp(-1j*math.pi*2/3),np.exp(1j*math.pi*2/3)],[1,1,1]])
    ksp=np.dot(E,np.reshape(ksp,(3,mult)))

    kspDconj=np.squeeze(np.reshape(ksp[0,:],new_shp[1:]))
    kspD=np.squeeze(np.reshape(ksp[1,:],new_shp[1:]))
    kspT1=np.squeeze(np.reshape(ksp[2,:],new_shp[1:]))

    return kspDconj, kspT1, kspD



def ProcessSpectra(spectrum, thresh, resize_factor):    
    from scipy.ndimage import label
    import numpy as np
    import cv2 as cv
    spectrum = np.abs(spectrum)

    # Threshold based on mean 
    binary_mask = (spectrum>thresh*np.mean(spectrum)).astype(np.uint8)

    # Remove single voxels (noise)
    labeled_array, num_features = label(binary_mask, structure=np.ones((3,3)))
    unique, counts = np.unique(labeled_array, return_counts=True)
    valid_components = {comp for comp, count in zip(unique[1:], counts[1:]) if count > 2}
    cleaned_mask = np.isin(labeled_array, list(valid_components)).astype(np.uint8)

    cleaned_spectrum = cleaned_mask*spectrum
    
    # Resize to QSM
    resized = cv.resize(np.roll(np.flip(np.flip(np.transpose(cleaned_spectrum)),1),-1,0), dsize=(int(resize_factor[0]*cleaned_spectrum.shape[0]), int(resize_factor[1]*cleaned_spectrum.shape[1])), interpolation=cv.INTER_CUBIC)
    resized=resized/np.max(resized)
    return resized


def ProcessSpectra_comp(spectrum, norm, thresh, resize_factor):    
    from scipy.ndimage import label
    import numpy as np
    import cv2 as cv
    norm = np.abs(norm)

    # Threshold based on mean 
    binary_mask = (norm>thresh*np.max(norm)).astype(np.uint8)
    cleaned_spectrum = binary_mask*spectrum
    
    # Resize to QSM
    resized = cv.resize(np.roll(np.flip(np.flip(np.transpose(cleaned_spectrum)),1),0,0), dsize=(int(resize_factor[0]*cleaned_spectrum.shape[0]), int(resize_factor[1]*cleaned_spectrum.shape[1])), interpolation=cv.INTER_CUBIC)

    return resized


def OverlayColor(overlayed, resized, offset, color):
    if color=="yellow":
        overlayed[offset[0]:offset[0]+resized.shape[0],offset[1]:offset[1]+resized.shape[1],0] += resized
        overlayed[offset[0]:offset[0]+resized.shape[0],offset[1]:offset[1]+resized.shape[1],1] += resized
    elif color=="blue":
        overlayed[offset[0]:offset[0]+resized.shape[0],offset[1]:offset[1]+resized.shape[1],1] += resized
        overlayed[offset[0]:offset[0]+resized.shape[0],offset[1]:offset[1]+resized.shape[1],2] += resized
    elif color=="green":
        overlayed[offset[0]:offset[0]+resized.shape[0],offset[1]:offset[1]+resized.shape[1],1] += resized
    else:
        overlayed[offset[0]:offset[0]+resized.shape[0],offset[1]:offset[1]+resized.shape[1],0] += resized
    return overlayed
    

def mean_filter(signal, window_size):
    import numpy as np
    """
    Apply a mean filter to a 1D signal.

    Parameters:
    signal (np.ndarray): Input 1D signal.
    window_size (int): Size of the moving window.

    Returns:
    np.ndarray: The filtered signal.
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    
    # Pad the signal at the beginning and end to handle the borders
    padded_signal = np.pad(signal, (window_size//2, window_size-1-window_size//2), mode='edge')
    
    # Create an array to store the filtered signal
    filtered_signal = np.zeros_like(signal)
    
    # Apply the mean filter
    for i in range(len(signal)):
        filtered_signal[i] = np.mean(padded_signal[i:i+window_size])
    
    return filtered_signal

def get_linear_colors_from_colormap(colormap_name, num_colors):
    import numpy as np
    import matplotlib.pyplot as plt
    """
    Get a linear set of colors from an existing colormap in Matplotlib.

    Parameters:
    colormap_name (str): The name of the colormap to sample from.
    num_colors (int): The number of colors to sample.

    Returns:
    list: A list of colors sampled from the colormap.
    """
    # Get the colormap
    colormap = plt.get_cmap(colormap_name)
    
    # Generate a set of linearly spaced values between 0 and 1
    values = np.linspace(0, 1, num_colors)
    
    # Sample colors from the colormap
    colors = [colormap(value) for value in values]
    
    return colors

def get_thresholding_sum(thresholding, scale):
    
    import numpy as np
    
    thresholding_sum = np.zeros((thresholding.shape[0],thresholding.shape[1]))
    for i in range(thresholding.shape[2]):
        thresh = scale/np.exp(-(i*150+550)/3500)
        thresholding_sum += thresholding[:,:,i]>thresh*np.max(thresholding[:,:,i])
    return thresholding_sum


def get_deltat(s1,s2,thresh):
    
    from scipy import signal
    from scipy.interpolate import interp1d
    import matplotlib.colors as mcolors
    import numpy as np
    
    time = np.linspace(0,s1.shape[2]-1,s1.shape[2])*150+250
    new_time = np.linspace(0,s1.shape[2]-1,s1.shape[2]*10)*150+250
    t1_calib = np.exp(-(np.linspace(0,s1.shape[2]-1,s1.shape[2])*150+250)/1584)
    
    step = new_time[1]-new_time[0]
    res = np.zeros((s1.shape[0],s1.shape[1]))
    for i in range(s1.shape[0]):
        for j in range(s1.shape[1]):
            data1 = s1[i,j,:]/t1_calib.copy()
            data2 = s2[i,j,:]/t1_calib.copy()
            interpolator1 = interp1d(time, mean_filter(data1,3), kind='cubic') 
            interpolated_data1 = interpolator1(new_time)     

            interpolator2 = interp1d(time, mean_filter(data2,3), kind='cubic') 
            interpolated_data2 = interpolator2(new_time)     

            thresh1 = thresh*np.max(interpolated_data1)
            thresh2 = thresh*np.max(interpolated_data2)
            stop1=0
            stop2=0
            for ind in range(len(interpolated_data1)):
                if interpolated_data1[ind]>thresh1 and stop1==0:
                    t1 = ind
                    stop1=1
                if interpolated_data2[ind]>thresh2 and stop2==0:
                    t2 = ind
                    stop2=1
            res[i,j] = (t1-t2)*step
    return res


def OverlayColor_cmap(resized,imgc,cmap,offset,limit,centered):
    
    import numpy as np
    if centered == 1:
        rgba_img = cmap((resized+limit)/(2*limit))
    else:  
        rgba_img = cmap((-(resized))/(limit))
    overlayed=np.zeros((imgc.shape[0],imgc.shape[1],4))
    overlayed[offset[0]:offset[0]+resized.shape[0],offset[1]:offset[1]+resized.shape[1],:]=rgba_img
    overlayed = np.abs(overlayed)/np.max(np.abs(overlayed))
    
    return overlayed

def Plot_SourceROIs(s1, s2, source_voxel):

    import numpy as np
    t1_calib = np.exp(-(np.linspace(0,s1.shape[2]-1,s1.shape[2])*150+250)/1584)
    time = np.linspace(0,s1.shape[2]-1,s1.shape[2])*0.15+0.25
    
    baseline = mean_filter(s1[source_voxel[0],source_voxel[1],:]/t1_calib,3)
    caffeine = mean_filter(s2[source_voxel[0],source_voxel[1],:]/t1_calib,3)

    maxval = np.maximum(np.max(baseline),np.max(caffeine))
    baseline = baseline/np.max(baseline)
    caffeine = caffeine/np.max(caffeine)
    
    return baseline, caffeine, time

def Calculate_Tstatistic(spectrum_baseline, spectrum_task):
    import scipy.stats as stats
    import numpy as np
    import numpy.matlib

    group_before = np.reshape(np.transpose(np.reshape(spectrum_baseline[:,:,3:12,:,:],(14*10,9,40*40)),(0,2,1)),(14*10*40*40,9))
    group_after = np.reshape(np.transpose(np.reshape(spectrum_task[:,:,3:12,:,:],(14*10,9,40*40)),(0,2,1)),(14*10*40*40,9))
    t1_calib = np.matlib.repmat(np.exp(-(np.linspace(0,8,9)*150+550)/1584),224000,1)

    # Perform the paired t-test
    t_statistic, p_value = stats.ttest_ind(group_before, group_after, axis=1)
    
    # Reshape
    t_statistic = np.reshape(t_statistic,(14,10,40,40))
    p_value = np.reshape(p_value,(14,10,40,40))
    
    return t_statistic, p_value