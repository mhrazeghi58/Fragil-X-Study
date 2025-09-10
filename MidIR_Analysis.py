#!/usr/bin/env python
# coding: utf-8

# In[1]:


import  os                                     # To Define the current working directory 
from    IPython.display import Image, display    # To display images  
import  re                                     # Split File name 
from    scipy.io import loadmat                  # To load multiple .mat files
import  numpy as np 
import  matplotlib.pyplot as plt
import  numpy.ma as ma
from    sklearn.preprocessing import StandardScaler
from    sklearn.cluster import KMeans
from    sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
os.chdir(r'D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)')      # Directory for Windows 
print(os.getcwd())                             # Check we are in the working directory we want
print(*os.listdir(), sep='\n')   


# In[3]:


T1=np.load('HP_WT_0.npy')
T2=np.load('HP_WT.npy')
T3=np.load('HP_WT_2.npy')
T4=np.load('HP_KO_0.npy')
T5=np.load('HP_KO_1.npy')
T6=np.load('HP_KO_2.npy')
T_ref=np.genfromtxt("braintissue.txt", delimiter="\t", skip_header=0, filling_values=np.nan) # Change delimiter if needed
T0=T_ref[:,0]
#print(T_ref[:,0])
#T1=T1[2850:,200:,:]
#T2=np.load('PFC_KO_1.npy')
#T2=T2[3350:,200:,:]
#T3=np.load('PFC_KO_2.npy')
#T3=T3[3500:7100,:,:]
#T4=np.load('PFC_WT.npy')
#T4=T4[2800:,:,:]
#T5=np.load('PFC_WT_1.npy')
#T5=T5[2800:,:,:]
#T6=np.load('PFC_WT_2.npy')
#T6=T6[2800:5900,50:3700,:]


# In[4]:


#print(T0)
T0.shape


# In[5]:


T0=T_ref[:,0]
data_min = np.min(T0)

# Shift the entire data towards zero by subtracting the global minimum
T0 = T0 - data_min

subset_min = np.min(T0[100:380])
shifted_data = T0.copy()
shifted_data[100:380] -= subset_min
T0=shifted_data


# In[ ]:


wn  = np.linspace(950,1800,426)
get_ipython().run_line_magic('matplotlib', '')
plt.rcParams.update({'font.size': 15}) # darkgrey tomato\n
fig, ax = plt.subplots(1,1, figsize=(12,8))
plt.plot(wn, T0, linewidth=3, color='black', label="Reference")


# In[5]:


T_data = {"T1": T1, "T2": T2, "T3": T3, "T4": T4, "T5": T5, "T6": T6}
low_threshold = 0.0
masked_cubes = {}
reshaped_data = {}

# Loop through datasets
for i, (key, T) in enumerate(T_data.items(), 1):
    # Define exc based on percentiles
    exc = np.percentile(T[:, :, 353], 99.999)  # Upper threshold

    # Create mask and apply it
    maskspec = ma.masked_where(np.logical_or(T[:, :, 353] <= low_threshold, T[:, :, 353] >= exc), T[:, :, 353])
    maskspec3d = np.zeros(T.shape, dtype=bool)
    maskspec3d[:, :, :] = maskspec[:, :, np.newaxis].mask
    maskspec3dcube = ma.array(T, mask=maskspec3d)

    # Store the masked 3D cube
    masked_cubes[key] = maskspec3dcube

    # Reshape the unmasked data into 2D array
    reshaped_data[f"{key}_m2d"] = T[~maskspec3dcube.mask].reshape([-1, T.shape[-1]])

# Example: Accessing the results
#print("Masked cube for T1:", masked_cubes["T1"])
#print("Reshaped data for T1:", reshaped_data["T1_m2d"])


# In[ ]:


wns  = np.linspace(950,1000,50)
get_ipython().run_line_magic('matplotlib', '')
plt.rcParams.update({'font.size': 15}) # darkgrey tomato\n",
fig, ax = plt.subplots(1,1, figsize=(12,8))
ax.imshow(T6[:,:,353], cmap='twilight', vmin=T6[:,:,353].min(), vmax= T6[:,:,353].max())
ax.set_ylabel('pixel')
ax.set_xlabel('pixel')
#ax[1].plot(wns, spectral[3000,2000,:])
#ax[1].plot(wns, spectral[3380:3420,2780:2820,:].mean(axis=(1,0))) # substrate
#ax[1].plot(wns, spectral[966,1370,:] )
ax.set_xlabel('pixel')
ax.set_ylabel('pixel')
plt.show()


# In[ ]:


plt.figure()
plt.imshow(maskspec3dcube[:,:,353], cmap='twilight_shifted')
plt.xlabel('pixel')
plt.ylabel('pixel')
plt.show()


# In[ ]:


maskspec3dcube =masked_cubes["T1"]
maskspec3dcube1=masked_cubes["T2"]
maskspec3dcube2=masked_cubes["T3"]
maskspec3dcube3=masked_cubes["T4"]
maskspec3dcube4=masked_cubes["T5"]
maskspec3dcube5=masked_cubes["T6"]
f, axarr = plt.subplots(1,6) 
width  = 27
height = 27
plt.rcParams['figure.figsize'] = [width, height]
axarr[0].imshow(maskspec3dcube[:,:,353], cmap='twilight_shifted')
axarr[0].set_title('WT1')
axarr[1].imshow(maskspec3dcube1[:,:,353], cmap='twilight_shifted')
axarr[1].set_title('WT2')
axarr[2].imshow(maskspec3dcube2[:,:,353], cmap='twilight_shifted')
axarr[2].set_title('WT3')
axarr[3].imshow(maskspec3dcube3[:,:,353], cmap='twilight_shifted')
axarr[3].set_title('KO1')
axarr[4].imshow(maskspec3dcube4[:,:,353], cmap='twilight_shifted')
axarr[4].set_title('KO2')
axarr[5].imshow(maskspec3dcube5[:,:,353], cmap='twilight_shifted')
axarr[5].set_title('KO3')
#axarr[2].axhline(y=720, color='r', linestyle='-')
#axarr[2].axhline(y=1200, color='r', linestyle='-')
#axarr[2].axvline(x=1200, color='r', linestyle='-')
#axarr[2].axvline(x=1680, color='r', linestyle='-')
#plt.xticks(np.linspace(0.0,3840,17))
#plt.yticks(np.linspace(0.0,3840,17))
#axarr[3].imshow(maskspec3dcube[1200:1680,720:1200,353].squeeze().T, cmap='twilight_shifted', vmin=exc2, vmax=exc1)
plt.show()


# In[ ]:


t1m2d = reshaped_data["T1_m2d"]
t1m2d1 = reshaped_data["T2_m2d"]
t1m2d2 = reshaped_data["T3_m2d"]
t1m2d3 = reshaped_data["T4_m2d"]
t1m2d4 = reshaped_data["T5_m2d"]
t1m2d5 = reshaped_data["T6_m2d"]
# Calculate means and standard deviations
WT_mean = t1m2d.mean(axis=0)
#WT_mean=WT_mean-WT_mean[0]
WT_mean=WT_mean[0:150]
#WT_std = t1m2d.std(axis=0)

WT1_mean = t1m2d1.mean(axis=0)
#WT1_mean=WT1_mean-abs(WT1_mean[0])
#WT1_std = t1m2d1.std(axis=0)
WT1_mean=WT1_mean[0:150]

WT2_mean = t1m2d2.mean(axis=0)
#WT2_mean=WT2_mean-WT2_mean[0]
#WT2_std = t1m2d2.std(axis=0)
WT2_mean=WT2_mean[0:150]

KO_mean = t1m2d3.mean(axis=0)
#KO_mean=KO_mean-KO_mean[0]
#KO_std = t1m2d3.std(axis=0)
KO_mean=KO_mean[0:150]

KO1_mean = t1m2d4.mean(axis=0)
#KO1_mean=KO1_mean-KO1_mean[0]
#KO1_std = t1m2d4.std(axis=0)
KO1_mean=KO1_mean[0:150]

KO2_mean = t1m2d5.mean(axis=0)
#KO2_mean=KO2_mean-KO2_mean[0]
#KO2_std = t1m2d5.std(axis=0)
KO2_mean=KO2_mean[0:150]

# Plot with shaded regions for error
plt.figure(figsize=(10, 6))
# Plotting
wn = np.linspace(950,1100,150)

get_ipython().run_line_magic('matplotlib', '')
plt.rcParams.update({'font.size': 15}) # darkgrey tomato\n
fig, ax = plt.subplots(1,1, figsize=(12,8))
plt.plot(wn, WT_mean, linewidth=3, color='orange', label="WT")
plt.plot(wn, WT1_mean, linewidth=3, color='orange')
plt.plot(wn, WT2_mean, linewidth=3, color='orange')
plt.plot(wn, KO_mean, linewidth=3, color='green')
plt.plot(wn, KO1_mean, linewidth=3, color='green')
plt.plot(wn, KO2_mean, linewidth=3, color='green', label="KO")


# Add labels, legend, and set axis limits
plt.ylabel('Absorbance')
plt.xlabel('Wavenumber (cm$^{-1}$)')
plt.legend(loc="best")
plt.xlim([950,1000])
plt.show()


# In[ ]:


wn = np.linspace(950, 1800, 426)

t1m2d = reshaped_data["T1_m2d"]
t1m2d1 = reshaped_data["T2_m2d"]
t1m2d2 = reshaped_data["T3_m2d"]
t1m2d3 = reshaped_data["T4_m2d"]
t1m2d4 = reshaped_data["T5_m2d"]
t1m2d5 = reshaped_data["T6_m2d"]
# Calculate means and standard deviations
WT_mean = t1m2d.mean(axis=0)
WT_mean=WT_mean-WT_mean[0]
WT_std = t1m2d.std(axis=0)

WT1_mean = t1m2d1.mean(axis=0)
WT1_mean=WT1_mean-abs(WT1_mean[0])
WT1_std = t1m2d1.std(axis=0)

WT2_mean = t1m2d2.mean(axis=0)
WT2_mean=WT2_mean-WT2_mean[0]
WT2_std = t1m2d2.std(axis=0)

KO_mean = t1m2d3.mean(axis=0)
KO_mean=KO_mean-KO_mean[0]
KO_std = t1m2d3.std(axis=0)

KO1_mean = t1m2d4.mean(axis=0)
KO1_mean=KO1_mean-KO1_mean[0]
KO1_std = t1m2d4.std(axis=0)

KO2_mean = t1m2d5.mean(axis=0)
KO2_mean=KO2_mean-KO2_mean[0]
KO2_std = t1m2d5.std(axis=0)

# Plot with shaded regions for error
plt.figure(figsize=(10, 6))

# WT main data
plt.plot(wn, WT_mean, linewidth=2, color='orange', label="WT")
plt.fill_between(wn, WT_mean - WT_std, WT_mean + WT_std, color='orange', alpha=0.22)

# KO datasets with different shading
plt.plot(wn, WT1_mean, linewidth=2, color='orange', alpha=0.99)
plt.fill_between(wn, WT1_mean - WT1_std, WT1_mean + WT1_std, color='orange', alpha=0.22)

plt.plot(wn, WT2_mean, linewidth=2, color='orange', alpha=0.99)
plt.fill_between(wn, WT2_mean - WT2_std, WT2_mean + WT2_std, color='orange', alpha=0.22)

plt.plot(wn, KO_mean, linewidth=2, color='green', label="KO", alpha=0.99)
plt.fill_between(wn, KO_mean - KO_std, KO_mean + KO_std, color='green', alpha=0.22)

# WT additional dataset
plt.plot(wn, KO1_mean, linewidth=2, color='green', alpha=0.99)
plt.fill_between(wn, KO1_mean - KO1_std, KO1_mean + KO1_std, color='green', alpha=0.22)

# WT additional dataset
plt.plot(wn, KO2_mean, linewidth=2, color='green', alpha=0.99)
plt.fill_between(wn, KO2_mean - KO2_std, KO2_mean + KO2_std, color='green', alpha=0.22)

# Labels, legend, and title
plt.ylabel('Absorbance')
plt.xlabel('Wavenumber (cm$^{-1}$)')
plt.legend(loc="best")
plt.xlim([950, 1800])
plt.title("Spectral Data with Shaded Error Regions")
plt.show()


# In[ ]:


from scipy.signal import savgol_filter

wn = np.linspace(950, 1800, 426)

# Calculate means and standard deviations
WT_mean = t1m2d.mean(axis=0)
WT_mean=WT_mean-WT_mean[0]
WT_std = t1m2d.std(axis=0)

WT1_mean = t1m2d1.mean(axis=0)
WT1_mean=WT1_mean-WT1_mean[0]
WT1_std = t1m2d1.std(axis=0)

WT2_mean = t1m2d2.mean(axis=0)
WT2_mean=WT2_mean-WT2_mean[0]
WT2_std = t1m2d2.std(axis=0)

KO_mean = t1m2d3.mean(axis=0)
KO_mean=KO_mean-KO_mean[0]
KO_std = t1m2d3.std(axis=0)

KO1_mean = t1m2d4.mean(axis=0)
KO1_mean=KO1_mean-KO1_mean[0]
KO1_std = t1m2d4.std(axis=0)

KO2_mean = t1m2d5.mean(axis=0)
KO2_mean=KO2_mean-KO2_mean[0]
KO2_std = t1m2d5.std(axis=0)

# Function to calculate the derivative and propagate the error
def calculate_derivative_and_error(data, error, wn):
    first_derivative = np.gradient(data, wn)
    second_derivative = savgol_filter(np.gradient(first_derivative, wn),5,2)
    
    # Error propagation: error for derivatives
    first_derivative_error = np.gradient(error, wn)
    second_derivative_error = savgol_filter(np.gradient(first_derivative, wn),5,2)
    
    return first_derivative, first_derivative_error, second_derivative, second_derivative_error

# Compute derivatives and errors for each group
WT_first, WT_first_error, WT_second, WT_second_error = calculate_derivative_and_error(WT_mean, WT_std, wn)
WT1_first, WT1_first_error, WT1_second, WT1_second_error = calculate_derivative_and_error(WT1_mean, WT1_std, wn)
WT2_first, WT2_first_error, WT2_second, WT2_second_error = calculate_derivative_and_error(WT2_mean, WT2_std, wn)
KO_first, KO_first_error, KO_second, KO_second_error = calculate_derivative_and_error(KO_mean, KO_std, wn)
KO1_first, KO1_first_error, KO1_second, KO1_second_error = calculate_derivative_and_error(KO1_mean, KO1_std, wn)
KO2_first, KO2_first_error, KO2_second, KO2_second_error = calculate_derivative_and_error(KO2_mean, KO2_std, wn)

# Plot first derivatives with error ribbons
plt.figure(figsize=(12, 8))

# First Derivatives Plot
plt.subplot(2, 1, 1)
plt.plot(wn, WT_first, linewidth=2, color='orange', label="WT First Derivative")
plt.fill_between(wn, WT_first - WT_first_error, WT_first + WT_first_error, color='orange', alpha=0.2)
plt.plot(wn, WT1_first, linewidth=2, color='orange', label="WT 1 First Derivative", alpha=0.99)
plt.fill_between(wn, WT1_first - WT1_first_error, WT1_first + WT1_first_error, color='orange', alpha=0.2)
plt.plot(wn, WT2_first, linewidth=2, color='orange', label="WT 2 First Derivative", alpha=0.99)
plt.fill_between(wn, WT2_first - WT2_first_error, WT2_first + WT2_first_error, color='orange', alpha=0.2)
plt.plot(wn, KO_first, linewidth=2, color='green', label="KO First Derivative", alpha=0.99)
plt.fill_between(wn, KO_first - KO_first_error, KO_first + KO_first_error, color='green', alpha=0.2)
plt.plot(wn, KO1_first, linewidth=2, color='green', label="KO 1 First Derivative", alpha=0.99)
plt.fill_between(wn, KO1_first - KO1_first_error, KO1_first + KO1_first_error, color='green', alpha=0.2)
plt.plot(wn, KO2_first, linewidth=2, color='green', label="KO 2 First Derivative", alpha=0.99)
plt.fill_between(wn, KO2_first - KO2_first_error, KO2_first + KO2_first_error, color='green', alpha=0.2)
plt.ylabel('First Derivative')
plt.legend(loc="best")
plt.title("First Derivatives with Error Ribbons")

# Second Derivatives Plot
plt.subplot(2, 1, 2)
plt.plot(wn, WT_second, linewidth=2, color='orange', label="WT Second Derivative", linestyle='-')
plt.fill_between(wn, WT_second - WT_second_error, WT_second + WT_second_error, color='orange', alpha=0.2)
plt.plot(wn, WT1_second, linewidth=2, color='orange', label="WT 1 Second Derivative", linestyle='-', alpha=0.99)
plt.fill_between(wn, WT1_second - WT1_second_error, WT1_second + WT1_second_error, color='orange', alpha=0.2)
plt.plot(wn, WT2_second, linewidth=2, color='orange', label="WT 2 Second Derivative", linestyle='-', alpha=0.99)
plt.fill_between(wn, WT2_second - WT2_second_error, WT2_second + WT2_second_error, color='orange', alpha=0.2)
plt.plot(wn, KO_second, linewidth=2, color='green', label="KO Second Derivative", linestyle='-', alpha=0.99)
plt.fill_between(wn, KO_second - KO_second_error, KO_second + KO_second_error, color='green', alpha=0.2)
plt.plot(wn, KO1_second, linewidth=2, color='green', label="KO 1 Second Derivative", linestyle='-', alpha=0.99)
plt.fill_between(wn, KO1_second - KO1_second_error, KO1_second + KO1_second_error, color='green', alpha=0.2)
plt.plot(wn, KO2_second, linewidth=2, color='green', label="KO 2 Second Derivative", linestyle='-', alpha=0.99)
plt.fill_between(wn, KO2_second - KO2_second_error, KO2_second + KO2_second_error, color='green', alpha=0.2)
plt.ylabel('Second Derivative')
plt.legend(loc="best")
plt.title("Second Derivatives with Error Ribbons")
plt.tight_layout()
plt.show()


# In[ ]:


wn.shape


# In[ ]:


from scipy.signal import savgol_filter

maskspec3dcube =masked_cubes["T1"]
maskspec3dcube1=masked_cubes["T2"]
maskspec3dcube2=masked_cubes["T3"]
maskspec3dcube3=masked_cubes["T4"]
maskspec3dcube4=masked_cubes["T5"]
maskspec3dcube5=masked_cubes["T6"]

#wn = np.linspace(1076, 1190, 57)
wn=np.linspace(950,1800,426)
# Region CC
#WT_mean = maskspec3dcube[2115:2483,441:675,63:120].mean(axis=(0,1))
#WT_mean=WT_mean[146:206]
#WT_std = maskspec3dcube[2115:2483,441:675,63:120].std(axis=(0,1))

#WT1_mean =maskspec3dcube1[2253:3040,482:861,63:120].mean(axis=(0,1))
#WT1_mean=WT1_mean-WT1_mean[0]
#WT1_std = maskspec3dcube1[2253:3040,482:861,63:120].std(axis=(0,1))

#WT2_mean = maskspec3dcube2[2230:2669,489:670,63:120].mean(axis=(0,1))
#WT2_mean=WT2_mean-WT2_mean[0]
#WT2_std = maskspec3dcube2[2230:2669,489:670,63:120].std(axis=(0,1))

#WT3_mean = maskspec3dcube3[2123:2591,316:724,63:120].mean(axis=(0,1))
#WT3_mean=WT3_mean-WT3_mean[0]
#WT3_std = maskspec3dcube3[2123:2591,316:724,63:120].std(axis=(0,1))

#WT4_mean = maskspec3dcube4[2201:2532,316:573,63:120].mean(axis=(0,1))
#WT4_mean=WT4_mean-WT4_mean[0]
#WT4_std = maskspec3dcube4[2201:2532,316:573,63:120].std(axis=(0,1))

#WT5_mean = maskspec3dcube5[2168:2662,398:708,63:120].mean(axis=(0,1))
#WT5_mean=WT5_mean-WT5_mean[0]
#WT5_std  = maskspec3dcube5[2168:2662,398:708,63:120].std(axis=(0,1))

# Region CA-1  
WT_mean = maskspec3dcube[1071:1431,303:485,:].mean(axis=(0,1))
#WT_mean=WT_mean-WT_mean[0]
WT_std = maskspec3dcube[1071:1431,303:485,:].std(axis=(0,1))

WT1_mean =maskspec3dcube1[1288:1429,485:583,:].mean(axis=(0,1))
#WT1_mean=WT1_mean-WT1_mean[0]
WT1_std = maskspec3dcube1[1288:1429,485:583,:].std(axis=(0,1))

WT2_mean = maskspec3dcube2[1062:1470,365:539,:].mean(axis=(0,1))
#WT2_mean=WT2_mean-WT2_mean[0]
WT2_std = maskspec3dcube2[1062:1470,365:539,:].std(axis=(0,1))

WT3_mean = maskspec3dcube3[1257:1640,315:538,:].mean(axis=(0,1))
#WT3_mean=WT3_mean-WT3_mean[0]
WT3_std = maskspec3dcube3[1257:1640,315:538,:].std(axis=(0,1))

WT4_mean = maskspec3dcube4[1051:1561,163:378,:].mean(axis=(0,1))
#WT4_mean=WT4_mean-WT4_mean[0]
WT4_std = maskspec3dcube4[1051:1561,163:378,:].std(axis=(0,1))

WT5_mean = maskspec3dcube5[1038:1475,337:560,:].mean(axis=(0,1))
#WT5_mean=WT5_mean-WT5_mean[0]
WT5_std  = maskspec3dcube5[1038:1475,337:560,:].std(axis=(0,1))


# Function to calculate the derivative and propagate the error
def calculate_derivative_and_error(data, error, wn):
    first_derivative = np.gradient(data, wn)
    second_derivative = savgol_filter(np.gradient(first_derivative, wn),15,2)
    
    # Error propagation: error for derivatives
    first_derivative_error = np.gradient(error, wn)
    second_derivative_error = np.gradient(first_derivative_error, wn)
    
    return first_derivative, first_derivative_error, second_derivative, second_derivative_error

# Compute derivatives and errors for each group
WT_first, WT_first_error, WT_second, WT_second_error = calculate_derivative_and_error(WT_mean, WT_std, wn)
WT1_first, WT1_first_error, WT1_second, WT1_second_error = calculate_derivative_and_error(WT1_mean, WT1_std, wn)
WT2_first, WT2_first_error, WT2_second, WT2_second_error = calculate_derivative_and_error(WT2_mean, WT2_std, wn)
KO_first, KO_first_error, KO_second, KO_second_error = calculate_derivative_and_error(WT3_mean, WT3_std, wn)
KO1_first, KO1_first_error, KO1_second, KO1_second_error = calculate_derivative_and_error(WT4_mean, WT4_std, wn)
KO2_first, KO2_first_error, KO2_second, KO2_second_error = calculate_derivative_and_error(WT5_mean, WT5_std, wn)

# Plot first derivatives with error ribbons
plt.figure(figsize=(10, 8))

# First Derivatives Plot
#plt.subplot(1, 2, 1)
plt.plot(wn, WT_first+.001, linewidth=2, color='orange', label="WT First Derivative")
#plt.fill_between(wn, WT_first - WT_first_error, WT_first + WT_first_error, color='orange', alpha=0.2)
#plt.plot(wn, WT1_first+.002, linewidth=2, color='orange', label="WT 1 First Derivative")
#plt.fill_between(wn, WT1_first - WT1_first_error, WT1_first + WT1_first_error, color='orange', alpha=0.2)
plt.plot(wn, WT2_first+0.003, linewidth=2, color='orange', label="WT 2 First Derivative")
#plt.fill_between(wn, WT2_first - WT2_first_error, WT2_first + WT2_first_error, color='orange', alpha=0.2)
plt.plot(wn, KO_first-.001, linewidth=2, color='green', label="KO First Derivative")
#plt.fill_between(wn, KO_first - KO_first_error, KO_first + KO_first_error, color='green', alpha=0.2)
plt.plot(wn, KO1_first-.0011, linewidth=2, color='green', label="KO 1 First Derivative")
#plt.fill_between(wn, KO1_first - KO1_first_error, KO1_first + KO1_first_error, color='green', alpha=0.2)
plt.plot(wn, KO2_first-.0012, linewidth=2, color='green', label="KO 2 First Derivative")
#plt.fill_between(wn, KO2_first - KO2_first_error, KO2_first + KO2_first_error, color='green', alpha=0.2)
plt.ylabel('First Derivative')
#plt.legend(loc="best")
plt.title("First Derivatives")

# Second Derivatives Plot
plt.figure(figsize=(10, 8))
#plt.subplot(1, 2, 2)
plt.plot(wn, WT_second+0.0006, linewidth=2, color='orange', label="WT Second Derivative", linestyle='-')
#plt.fill_between(wn, WT_second - WT_second_error, WT_second + WT_second_error, color='orange', alpha=0.2)
#plt.plot(wn, WT1_second+0.0008, linewidth=2, color='orange', label="WT 1 Second Derivative", linestyle='-')
#plt.fill_between(wn, WT1_second - WT1_second_error, WT1_second + WT1_second_error, color='orange', alpha=0.2)
plt.plot(wn, WT2_second+0.0005, linewidth=2, color='orange', label="WT 2 Second Derivative", linestyle='-')
#plt.fill_between(wn, WT2_second - WT2_second_error, WT2_second + WT2_second_error, color='orange', alpha=0.2)
plt.plot(wn, KO_second-0.0004, linewidth=2, color='green', label="KO Second Derivative", linestyle='-')
#plt.fill_between(wn, KO_second - KO_second_error, KO_second + KO_second_error, color='green', alpha=0.2)
plt.plot(wn, KO1_second-0.00043, linewidth=2, color='green', label="KO 1 Second Derivative", linestyle='-')
#plt.fill_between(wn, KO1_second - KO1_second_error, KO1_second + KO1_second_error, color='green', alpha=0.2)
plt.plot(wn, KO2_second-0.00046, linewidth=2, color='green', label="KO 2 Second Derivative", linestyle='-')
#plt.fill_between(wn, KO2_second - KO2_second_error, KO2_second + KO2_second_error, color='green', alpha=0.2)
plt.ylabel('Second Derivative')
#plt.legend(loc="best")
plt.title("Second Derivatives")
plt.tight_layout()
plt.show()


# In[ ]:


wn = np.linspace(950, 1800, 426)

maskspec3dcube =masked_cubes["T1"]
maskspec3dcube1=masked_cubes["T2"]
maskspec3dcube2=masked_cubes["T3"]
maskspec3dcube3=masked_cubes["T4"]
maskspec3dcube4=masked_cubes["T5"]
maskspec3dcube5=masked_cubes["T6"]

# Region CA-1  
#WT_mean = maskspec3dcube[1071:1431,303:485,:].mean(axis=(0,1))
#WT_mean=WT_mean-WT_mean[0]
#WT_std = maskspec3dcube[1071:1431,303:485,:].std(axis=(0,1))

#WT1_mean =maskspec3dcube1[1288:1429,485:583,:].mean(axis=(0,1))
#WT1_mean=WT1_mean-WT1_mean[0]
#WT1_std = maskspec3dcube1[1288:1429,485:583,:].std(axis=(0,1))

#WT2_mean = maskspec3dcube2[1062:1470,365:539,:].mean(axis=(0,1))
#WT2_mean=WT2_mean-WT2_mean[0]
#WT2_std = maskspec3dcube2[1062:1470,365:539,:].std(axis=(0,1))

#WT3_mean = maskspec3dcube3[1257:1640,315:538,:].mean(axis=(0,1))
#WT3_mean=WT3_mean-WT3_mean[0]
#WT3_std = maskspec3dcube3[1257:1640,315:538,:].std(axis=(0,1))

#WT4_mean = maskspec3dcube4[1051:1561,163:378,:].mean(axis=(0,1))
#WT4_mean=WT4_mean-WT4_mean[0]
#WT4_std = maskspec3dcube4[1051:1561,163:378,:].std(axis=(0,1))

#WT5_mean = maskspec3dcube5[1038:1475,337:560,:].mean(axis=(0,1))
#WT5_mean=WT5_mean-WT5_mean[0]
#WT5_std  = maskspec3dcube5[1038:1475,337:560,:].std(axis=(0,1))

# Region CC
WT_mean = maskspec3dcube[2115:2483,441:675,:].mean(axis=(0,1))
#WT_mean=WT_mean-WT_mean[0]
WT_std = maskspec3dcube[2115:2483,441:675,:].std(axis=(0,1))

WT1_mean =maskspec3dcube1[2253:3040,482:861,:].mean(axis=(0,1))
#WT1_mean=WT1_mean-WT1_mean[0]
WT1_std = maskspec3dcube1[2253:3040,482:861,:].std(axis=(0,1))

WT2_mean = maskspec3dcube2[2230:2669,489:670,:].mean(axis=(0,1))
#WT2_mean=WT2_mean-WT2_mean[0]
WT2_std = maskspec3dcube2[2230:2669,489:670,:].std(axis=(0,1))

WT3_mean = maskspec3dcube3[2123:2591,316:724,:].mean(axis=(0,1))
#WT3_mean=WT3_mean-WT3_mean[0]
WT3_std = maskspec3dcube3[2123:2591,316:724,:].std(axis=(0,1))

WT4_mean = maskspec3dcube4[2201:2532,316:573,:].mean(axis=(0,1))
#WT4_mean=WT4_mean-WT4_mean[0]
WT4_std = maskspec3dcube4[2201:2532,316:573,:].std(axis=(0,1))

WT5_mean = maskspec3dcube5[2168:2662,398:708,:].mean(axis=(0,1))
#WT5_mean=WT5_mean-WT5_mean[0]
WT5_std  = maskspec3dcube5[2168:2662,398:708,:].std(axis=(0,1))

# Region CA-2
#WT_mean = maskspec3dcube[314:598,676:912,:].mean(axis=(0,1))
#WT_mean=WT_mean-WT_mean[0]
#WT_std = maskspec3dcube[314:598,676:912,:].std(axis=(0,1))

#WT1_mean =maskspec3dcube1[815:1085,730:923,:].mean(axis=(0,1))
#WT1_mean=WT1_mean-WT1_mean[0]
#WT1_std = maskspec3dcube1[815:1085,730:923,:].std(axis=(0,1))

#WT2_mean = maskspec3dcube2[321:636,940:1230,:].mean(axis=(0,1))
#WT2_mean=WT2_mean-WT2_mean[0]
#WT2_std = maskspec3dcube2[321:636,940:1230,:].std(axis=(0,1))

#WT3_mean = maskspec3dcube3[587:962,738:994,:].mean(axis=(0,1))
#WT3_mean=WT3_mean-WT3_mean[0]
#WT3_std = maskspec3dcube3[587:962,738:994,:].std(axis=(0,1))

#WT4_mean = maskspec3dcube4[313:667,667:944,:].mean(axis=(0,1))
#WT4_mean=WT4_mean-WT4_mean[0]
#WT4_std = maskspec3dcube4[313:667,667:944,:].std(axis=(0,1))

#WT5_mean = maskspec3dcube5[384:729,837:1128,:].mean(axis=(0,1))
#WT5_mean=WT5_mean-WT5_mean[0]
#WT5_std  = maskspec3dcube5[384:729,837:1128,:].std(axis=(0,1))

# Region CA-3
#WT_mean = maskspec3dcube[803:1048,854:1004,:].mean(axis=(0,1))
#WT_mean=WT_mean-WT_mean[0]
#WT_std = maskspec3dcube[803:1048,854:1004,:].std(axis=(0,1))

#WT1_mean =maskspec3dcube1[4079:4219,1162:1287,:].mean(axis=(0,1))
#WT1_mean=WT1_mean-WT1_mean[0]
#WT1_std = maskspec3dcube1[4079:4219,1162:1287,:].std(axis=(0,1))

#WT2_mean = maskspec3dcube2[874:1282,1126:1287,:].mean(axis=(0,1))
#WT2_mean=WT2_mean-WT2_mean[0]
#WT2_std = maskspec3dcube2[874:1282,1126:1287,:].std(axis=(0,1))

#WT3_mean = maskspec3dcube3[3506:3871,883:1064,:].mean(axis=(0,1))
#WT3_mean=WT3_mean-WT3_mean[0]
#WT3_std = maskspec3dcube3[3506:3871,883:1064,:].std(axis=(0,1))

#WT4_mean = maskspec3dcube4[828:1036,978:1077,:].mean(axis=(0,1))
#WT4_mean=WT4_mean-WT4_mean[0]
#WT4_std = maskspec3dcube4[828:1036,978:1077,:].std(axis=(0,1))

#WT5_mean = maskspec3dcube5[3750:4037,1051:1195,:].mean(axis=(0,1))
#WT5_mean=WT5_mean-WT5_mean[0]
#WT5_std  = maskspec3dcube5[3750:4037,1051:1195,:].std(axis=(0,1))

# Region DG
#WT_mean = maskspec3dcube[2903:3072,1090:1154,:].mean(axis=(0,1))
#WT_mean=WT_mean-WT_mean[0]
#WT_std = maskspec3dcube[2903:3072,1090:1154,:].std(axis=(0,1))

#WT1_mean =maskspec3dcube1[3119:3318,1166:1255,:].mean(axis=(0,1))
#WT1_mean=WT1_mean-WT1_mean[0]
#WT1_std = maskspec3dcube1[3119:3318,1166:1255,:].std(axis=(0,1))

#WT2_mean = maskspec3dcube2[1746:2114,1068:1270,:].mean(axis=(0,1))
#WT2_mean=WT2_mean-WT2_mean[0]
#WT2_std = maskspec3dcube2[1746:2114,1068:1270,:].std(axis=(0,1))

#WT3_mean = maskspec3dcube3[2701:2866,1115:1182,:].mean(axis=(0,1))
#WT3_mean=WT3_mean-WT3_mean[0]
#WT3_std = maskspec3dcube3[2701:2866,1115:1182,:].std(axis=(0,1))

#WT4_mean = maskspec3dcube4[1713:1893,974:1032,:].mean(axis=(0,1))
#WT4_mean=WT4_mean-WT4_mean[0]
#WT4_std = maskspec3dcube4[1713:1893,974:1032,:].std(axis=(0,1))

#WT5_mean = maskspec3dcube5[3063:3460,973:1102,:].mean(axis=(0,1))
#WT5_mean=WT5_mean-WT5_mean[0]
#WT5_std  = maskspec3dcube5[3063:3460,973:1102,:].std(axis=(0,1))

WT = np.mean([WT_mean, WT2_mean], axis=0)
KO = np.mean([WT3_mean, WT4_mean, WT5_mean], axis=0)

# Region ls
#WT_mean = maskspec3dcube[1004:1407,600:751,:].mean(axis=(0,1))
#WT_mean=WT_mean-WT_mean[0]
#WT_std = maskspec3dcube[1004:1407,600:751,:].std(axis=(0,1))

#WT1_mean =maskspec3dcube1[1438:1807,678:782,:].mean(axis=(0,1))
#WT1_mean=WT1_mean-WT1_mean[0]
#WT1_std = maskspec3dcube1[1438:1807,678:782,:].std(axis=(0,1))

#WT2_mean = maskspec3dcube2[1077:1529,662:826,:].mean(axis=(0,1))
#WT2_mean=WT2_mean-WT2_mean[0]
#WT2_std = maskspec3dcube2[1077:1529,662:826,:].std(axis=(0,1))

#WT3_mean = maskspec3dcube3[1273:1596,643:711,:].mean(axis=(0,1))
#WT3_mean=WT3_mean-WT3_mean[0]
#WT3_std = maskspec3dcube3[1273:1596,643:711,:].std(axis=(0,1))

#WT4_mean = maskspec3dcube4[1028:1655,471:676,:].mean(axis=(0,1))
#WT4_mean=WT4_mean-WT4_mean[0]
#WT4_std = maskspec3dcube4[1028:1655,471:676,:].std(axis=(0,1))

#WT5_mean = maskspec3dcube5[1063:1771,572:758,:].mean(axis=(0,1))
#WT5_mean=WT5_mean-WT5_mean[0]
#WT5_std  = maskspec3dcube5[1063:1771,572:758,:].std(axis=(0,1))



# Plot with shaded regions for error
get_ipython().run_line_magic('matplotlib', '')
plt.rcParams.update({'font.size': 15}) # darkgrey tomato\n
fig, ax = plt.subplots(1,1, figsize=(12,8))

# WT main data
plt.plot(wn, WT, linewidth=2, color='orange', label="WT")
#plt.fill_between(wn, WT_mean - WT_std, WT_mean + WT_std, color='orange', alpha=0.22)

# KO datasets with different shading
#plt.plot(wn, WT1_mean, linewidth=2, color='orange')
#plt.fill_between(wn, WT1_mean - WT1_std, WT1_mean + WT1_std, color='orange', alpha=0.22)

plt.plot(wn, KO, linewidth=2, color='green', label="KO")
#plt.fill_between(wn, WT2_mean - WT2_std, WT2_mean + WT2_std, color='orange', alpha=0.22)

#plt.plot(wn, WT3_mean, linewidth=2, color='green', label="KO")
#plt.fill_between(wn, WT3_mean - WT3_std, WT3_mean + WT3_std, color='green', alpha=0.22)

# WT additional dataset
#plt.plot(wn, WT4_mean, linewidth=2, color='green', alpha=0.99)
#plt.fill_between(wn, WT4_mean - WT4_std, WT4_mean + WT4_std, color='green', alpha=0.22)

# WT additional dataset
#plt.plot(wn, WT5_mean, linewidth=2, color='green', alpha=0.99)
#plt.fill_between(wn, WT5_mean - WT5_std, WT5_mean + WT5_std, color='green', alpha=0.22)

# Labels, legend, and title
plt.ylabel('Absorbance')
plt.xlabel('Wavenumber (cm$^{-1}$)')
plt.legend(loc="best")
plt.xlim([950, 1800])
plt.title("Spectral Data with Shaded Error Regions")
plt.show()


# In[ ]:


wn = np.linspace(950, 1800, 426)

maskspec3dcube =masked_cubes["T1"]
#maskspec3dcube1=masked_cubes["T2"]
#maskspec3dcube2=masked_cubes["T3"]
#maskspec3dcube3=masked_cubes["T4"]
#maskspec3dcube4=masked_cubes["T5"]
#maskspec3dcube5=masked_cubes["T6"]


WT_mean = maskspec3dcube[1245:1345,507:623,:].mean(axis=(0,1))
#WT_mean=WT_mean-WT_mean[0]
WT_std = maskspec3dcube[1245:1345,507:623,:].std(axis=(0,1))

WT1_mean = maskspec3dcube[834:1034,706:897,:].mean(axis=(0,1))
#WT1_mean=WT1_mean-WT1_mean[0]
WT1_std = maskspec3dcube[834:1034,706:897,:].std(axis=(0,1))

WT2_mean = maskspec3dcube[1372:1522,1073:1179,:].mean(axis=(0,1))
#WT2_mean=WT2_mean-WT2_mean[0]
WT2_std = maskspec3dcube[1372:1522,1073:1179,:]].std(axis=(0,1))

WT3_mean = maskspec3dcube[1937:2214,1096:1238,:].mean(axis=(0,1))
#WT3_mean=WT3_mean-WT3_mean[0]
WT3_std = maskspec3dcube[1937:2214,1096:1238,:].std(axis=(0,1))

WT4_mean = maskspec3dcube[1679:1875,698:779,:].mean(axis=(0,1))
#WT4_mean=WT4_mean-WT4_mean[0]
WT4_std = maskspec3dcube[1679:1875,698:779,:].std(axis=(0,1))

#KO_mean = maskspec3dcube3[2020:2515,366:510,:].mean(axis=(0,1))
#KO_mean=KO_mean-KO_mean[0]
#KO_std = maskspec3dcube3[2020:2515,366:510,:].std(axis=(0,1))

#KO1_mean = maskspec3dcube4[2136:2532,410:569,:].mean(axis=(0,1))
#KO1_mean=KO1_mean-KO1_mean[0]
#KO1_std = maskspec3dcube4[2136:2532,410:569,:].std(axis=(0,1))

#KO2_mean = maskspec3dcube5[2148:2586,506:662,:].mean(axis=(0,1))
#KO2_mean=KO2_mean-KO2_mean[0]
#KO2_std = maskspec3dcube5[2148:2586,506:662,:].std(axis=(0,1))
# Function to calculate the derivative and propagate the error
def calculate_derivative_and_error(data, error, wn):
    first_derivative = np.gradient(data, wn)
    second_derivative = np.gradient(first_derivative, wn)
    
    # Error propagation: error for derivatives
    first_derivative_error = np.gradient(error, wn)
    second_derivative_error = np.gradient(first_derivative_error, wn)
    
    return first_derivative, first_derivative_error, second_derivative, second_derivative_error

# Compute derivatives and errors for each group
WT_first, WT_first_error, WT_second, WT_second_error = calculate_derivative_and_error(WT_mean, WT_std, wn)
WT1_first, WT1_first_error, WT1_second, WT1_second_error = calculate_derivative_and_error(WT1_mean, WT1_std, wn)
WT2_first, WT2_first_error, WT2_second, WT2_second_error = calculate_derivative_and_error(WT2_mean, WT2_std, wn)
#KO_first, KO_first_error, KO_second, KO_second_error = calculate_derivative_and_error(KO_mean, KO_std, wn)
#KO1_first, KO1_first_error, KO1_second, KO1_second_error = calculate_derivative_and_error(KO1_mean, KO1_std, wn)
#KO2_first, KO2_first_error, KO2_second, KO2_second_error = calculate_derivative_and_error(KO2_mean, KO2_std, wn)

# Plot first derivatives with error ribbons
plt.figure(figsize=(12, 8))

# First Derivatives Plot
plt.subplot(2, 1, 1)
plt.plot(wn, WT_first, linewidth=2, color='orange', label="WT First Derivative")
plt.fill_between(wn, WT_first - WT_first_error, WT_first + WT_first_error, color='orange', alpha=0.1)
plt.plot(wn, WT1_first, linewidth=2, color='orange', label="WT 1 First Derivative", alpha=0.9)
plt.fill_between(wn, WT1_first - WT1_first_error, WT1_first + WT1_first_error, color='orange', alpha=0.1)
plt.plot(wn, WT2_first, linewidth=2, color='orange', label="WT 2 First Derivative", alpha=0.9)
plt.fill_between(wn, WT2_first - WT2_first_error, WT2_first + WT2_first_error, color='orange', alpha=0.3)
plt.plot(wn, KO_first, linewidth=2, color='green', label="KO First Derivative", alpha=0.9)
plt.fill_between(wn, KO_first - KO_first_error, KO_first + KO_first_error, color='green', alpha=0.3)
plt.plot(wn, KO1_first, linewidth=2, color='green', label="KO 1 First Derivative", alpha=0.9)
plt.fill_between(wn, KO1_first - KO1_first_error, KO1_first + KO1_first_error, color='green', alpha=0.3)
plt.plot(wn, KO2_first, linewidth=2, color='green', label="KO 2 First Derivative", alpha=0.9)
plt.fill_between(wn, KO2_first - KO2_first_error, KO2_first + KO2_first_error, color='green', alpha=0.3)
plt.ylabel('First Derivative')
plt.legend(loc="best")
plt.title("First Derivatives with Error Ribbons")

# Second Derivatives Plot
plt.subplot(2, 1, 2)
plt.plot(wn, WT_second, linewidth=2, color='orange', label="WT Second Derivative", linestyle='--')
plt.fill_between(wn, WT_second - WT_second_error, WT_second + WT_second_error, color='orange', alpha=0.3)
plt.plot(wn, WT1_second, linewidth=2, color='orange', label="WT 1 Second Derivative", linestyle='--', alpha=0.9)
plt.fill_between(wn, WT1_second - WT1_second_error, WT1_second + WT1_second_error, color='orange', alpha=0.3)
plt.plot(wn, WT2_second, linewidth=2, color='orange', label="WT 2 Second Derivative", linestyle='--', alpha=0.9)
plt.fill_between(wn, WT2_second - WT2_second_error, WT2_second + WT2_second_error, color='orange', alpha=0.3)
plt.plot(wn, KO_second, linewidth=2, color='green', label="KO Second Derivative", linestyle='--', alpha=0.9)
plt.fill_between(wn, KO_second - KO_second_error, KO_second + KO_second_error, color='green', alpha=0.3)
plt.plot(wn, KO1_second, linewidth=2, color='green', label="KO 1 Second Derivative", linestyle='--', alpha=0.9)
plt.fill_between(wn, KO1_second - KO1_second_error, KO1_second + KO1_second_error, color='green', alpha=0.3)
plt.plot(wn, KO2_second, linewidth=2, color='green', label="KO 2 Second Derivative", linestyle='--', alpha=0.9)
plt.fill_between(wn, KO2_second - KO2_second_error, KO2_second + KO2_second_error, color='green', alpha=0.3)
plt.ylabel('Second Derivative')
plt.legend(loc="best")
plt.title("Second Derivatives with Error Ribbons")
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Extract datasets from the dictionary
reshaped_data = {
    "T1_m2d": t1m2d,
    "T2_m2d": t1m2d1,
    "T3_m2d": t1m2d2,
    "T4_m2d": t1m2d3,
    "T5_m2d": t1m2d4,
    "T6_m2d": t1m2d5,
}

# Initialize the scaler
scaler = StandardScaler().set_output(transform="default")

# Number of principal components
n_components = 10

# Initialize an empty list to store explained variance ratios
explained_variance_ratios = []

# Loop through the datasets in the dictionary
for key, data in reshaped_data.items():
    # Step 1: Scale the data
    scaled_data = scaler.fit_transform(data.T)  # Transpose if samples are columns
    
    # Step 2: Apply PCA
    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit(scaled_data)
    
    # Step 3: Extract explained variance ratio and its sum
    variance_ratio = pca.explained_variance_ratio_
    variance_sum = variance_ratio.sum()
    
    # Store results
    explained_variance_ratios.append(variance_ratio)
    
    # Print results for this dataset
    print(f"Dataset {key} - Explained Variance Ratio: {variance_ratio}")
    print(f"Dataset {key} - Total Variance Explained: {variance_sum:.4f}")

# Final Output: `explained_variance_ratios` contains variance ratios for all datasets


# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Extract datasets from the dictionary
reshaped_data = {
    "T1_m2d": t1m2d,
    "T2_m2d": t1m2d1,
    "T3_m2d": t1m2d2,
    "T4_m2d": t1m2d3,
    "T5_m2d": t1m2d4,
    "T6_m2d": t1m2d5,
}

# Initialize the scaler
scaler = StandardScaler().set_output(transform="default")

# Number of principal components
n_components = 10

# KMeans parameters
n_clusters = 8  # Number of clusters
n_init = 4  # Number of initializations for KMeans

# Initialize storage for clustering and mask results
clustering_results = {}
masked_results = {}

# Loop through datasets
for key, data in reshaped_data.items():
    # Step 1: Scale the data
    scaled_data = scaler.fit_transform(data.T)  # Transpose if samples are columns
    
    # Step 2: Apply PCA
    pca = PCA(n_components=n_components, svd_solver='full')
    X_reduced = pca.fit_transform(scaled_data)  # PCA-transformed data
    
    # Step 3: Apply KMeans
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=n_init, random_state=42)
    km_labels = kmeans.fit_predict(X_reduced)  # Fit and predict cluster labels
    
    # Step 4: Map KMeans cluster labels to a 2D array using the mask
    mask_k = np.zeros((maskspec.shape[0], maskspec.shape[1]))
    count = 0
    
    for r in range(maskspec.shape[0]):  # Iterate over rows
        for s in range(maskspec.shape[1]):  # Iterate over columns
            if maskspec.mask[r, s] == False:  # If not masked
                mask_k[r, s] = km_labels[count]  # Assign cluster label
                count += 1
            else:  # If masked
                mask_k[r, s] = 9.0  # Fill with 9
    
    # Store results
    clustering_results[key] = {
        "kmeans_model": kmeans,  # Fitted KMeans model
        "labels": km_labels,  # Cluster labels
        "cluster_centers": kmeans.cluster_centers_,  # Cluster centers
    }
    masked_results[key] = mask_k  # Save the reconstructed mask
    
    # Print results for this dataset
    print(f"Dataset {key} - KMeans Cluster Labels Shape: {km_labels.shape}")
    print(f"Dataset {key} - Masked Array Shape: {mask_k.shape}")

# Final Output:
# - `clustering_results` contains KMeans models and labels for all datasets.
# - `masked_results` contains the 2D cluster label arrays (with mask applied) for all datasets.


# In[ ]:


import matplotlib.pyplot as plt

# Create the figure
f = plt.figure(figsize=(8, 6))  # Adjust size as needed

# Plot the mask
i = plt.imshow(mask_k, cmap='Dark2')  # Use 'Dark2' colormap

# Add a colorbar
cbar = f.colorbar(i)
cbar.set_ticks([])  # Remove ticks from the colorbar

# Add labels, title, etc. for clarity (optional)
plt.title("Clustered Mask Visualization", fontsize=14)
plt.axis('off')  # Turn off axis for a cleaner image

# Display the plot
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Extract datasets from the dictionary
reshaped_data = {
    "T1_m2d": t1m2d,
    "T2_m2d": t1m2d1,
    "T3_m2d": t1m2d2,
    "T4_m2d": t1m2d3,
    "T5_m2d": t1m2d4,
    "T6_m2d": t1m2d5,
}

# Initialize the scaler
scaler = StandardScaler().set_output(transform="default")

# Number of principal components
n_components = 20

# Loop through datasets
for key, data in reshaped_data.items():
    # Step 1: Scale the data
    scaled_data = scaler.fit_transform(data.T)  # Transpose if samples are columns
    
    # Step 2: Apply PCA (n_components set to 20)
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(scaled_data)  # Apply PCA
    
    # Step 3: Create the 3D scatter plot for the first three components
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

    # Scatter plot using the first three components
    ax.scatter(
        X_reduced[:, 0],  # First component
        X_reduced[:, 1],  # Second component
        X_reduced[:, 2]   # Third component
    )

    # Set titles and labels
    ax.set_title(f"First Three PCA Dimensions - {key}")
    ax.set_xlabel("1st Eigenvector")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("2nd Eigenvector")
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel("3rd Eigenvector")
    ax.zaxis.set_ticklabels([])

    # Display the plot
    plt.show()


# In[ ]:


# plot for clusters separately

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Initialize the number of clusters (adjust this based on your KMeans setup)
n_clusters = 8  # Example: 8 clusters

# Assuming `X_reduced` is already the PCA-transformed data and `km_labels` are the cluster labels from KMeans
# Create a dictionary to store individual cluster masks
cluster_masks = {}

# Loop through each cluster (from 0 to n_clusters-1)
for cluster_id in range(n_clusters):
    # Create a mask where 1 represents the current cluster and 0 otherwise
    cluster_mask = np.zeros_like(mask_k)  # Assuming mask_k is your original mask
    cluster_mask[km_labels == cluster_id] = 1  # Assign 1 for pixels belonging to the current cluster

    # Store the mask in the dictionary
    cluster_masks[f"Cluster_{cluster_id}"] = cluster_mask

    # Plot the mask for the current cluster
    plt.figure(figsize=(8, 6))
    plt.imshow(cluster_mask, cmap="Blues")  # Blue color map for visualization
    plt.title(f"Mask for Cluster {cluster_id}")
    plt.axis('off')  # Hide axes for cleaner plot
    plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path

# Use this magic command to enable interactive plotting in Jupyter
get_ipython().run_line_magic('matplotlib', '')

# Extract datasets from the dictionary
t1m2d0 = reshaped_data["T1_m2d"]
t1m2d1 = reshaped_data["T2_m2d"]
t1m2d2 = reshaped_data["T3_m2d"]
t1m2d3 = reshaped_data["T4_m2d"]
t1m2d4 = reshaped_data["T5_m2d"]
t1m2d5 = reshaped_data["T6_m2d"]

# Global list to store ROI coordinates
roi_coords_all_samples = []

# Class to manage ROI selection
class ROISelector:
    def __init__(self, ax):
        self.ax = ax
        self.roi_coords = None  # Store ROI coordinates
        self.selector = PolygonSelector(ax, self.onselect, useblit=True)
        plt.connect('key_press_event', self.on_key_press)

    def onselect(self, verts):
        """Callback function to store polygon vertices."""
        self.roi_coords = np.array(verts)
        print(f"ROI selected with vertices: {self.roi_coords}")

    def on_key_press(self, event):
        """Press Enter to finalize the selection."""
        if event.key == "enter":
            plt.close()  # Close the plot when user confirms selection

    def get_roi(self):
        """Wait for user input and return ROI coordinates."""
        plt.show()  # Show plot and wait for interaction
        return self.roi_coords

# Function to select ROI
def select_roi_for_sample(sample_image, sample_index):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(sample_image, cmap='gray')
    ax.set_title(f"Select ROI for Sample {sample_index + 1}\nPress Enter to Confirm")

    roi_selector = ROISelector(ax)
    roi_coords = roi_selector.get_roi()  # Wait for user selection

    return roi_coords if roi_coords is not None else np.array([])  # Return empty if no selection

# Iterate through samples for ROI selection
for i in range(6):
    sample_image = globals()[f"t1m2d{i}"]

    print(f"\nSelecting ROI for Sample {i + 1}...")
    roi_coords = select_roi_for_sample(sample_image, i)

    if roi_coords.size > 0:
        roi_coords_all_samples.append(roi_coords)
        print(f"ROI selected for Sample {i + 1}: {roi_coords.shape}")
    else:
        print(f"No ROI selected for Sample {i + 1}.")


# In[70]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

maskspec3dcube =masked_cubes["T1"]
maskspec3dcube1=masked_cubes["T2"]
maskspec3dcube2=masked_cubes["T3"]
maskspec3dcube3=masked_cubes["T4"]
maskspec3dcube4=masked_cubes["T5"]
maskspec3dcube5=masked_cubes["T6"]
# Function to apply Extended Multiplicative Signal Correction (EMSC)
def emsc_correction(spectrum, reference, poly_order=4):
    """
    Apply an improved Extended Multiplicative Signal Correction (EMSC) to a given spectrum.

    Parameters:
        spectrum (array): The input spectrum to be corrected.
        reference (array): The reference spectrum (from PCA or mean spectrum).
        poly_order (int): Order of polynomial baseline correction (default=2).

    Returns:
        corrected_spectrum (array): The EMSC-corrected spectrum.
    """
    # Construct the EMSC model matrix
    X = [np.ones_like(reference), reference]  # Offset and scaling

    # Add polynomial baseline terms
    for i in range(1, poly_order + 1):
        X.append(np.linspace(0, 1, len(reference)) ** i)

    # Convert to a matrix and solve using least squares
    X = np.vstack(X).T
    coeffs, _, _, _ = np.linalg.lstsq(X, spectrum, rcond=None)

    # Compute the corrected spectrum
    baseline = np.dot(X[:, 2:], coeffs[2:])  # Baseline contribution
    corrected_spectrum = (spectrum - (coeffs[0] + baseline)) / coeffs[1]

    return corrected_spectrum

# Function for iterative RMieS-EMSC correction
def rmies_emsc(spectrum, reference, iterations=5):
    """
    Apply iterative RMieS-EMSC correction to a given spectrum.

    Parameters:
        spectrum (array): The input spectrum to be corrected.
        reference (array): The reference spectrum from PCA.

    Returns:
        corrected_spectrum (array): The RMieS-EMSC corrected spectrum.
    """
    corrected_spectrum = spectrum.copy()
    
    for i in range(iterations):
        corrected_spectrum = emsc_correction(corrected_spectrum, reference)
        #print(f"Iteration {i+1}: Correction applied.")

    return corrected_spectrum

# Step 1: Load Spectral Data
# (Replace this with actual spectral data)
wavelengths = np.linspace(950, 1800, 426)  # Example wavenumber range
np.random.seed(42)

# Generate a synthetic dataset with 10 distorted spectra
#num_spectra = 10
#all_spectra = np.array([
#    np.exp(-((wavelengths - 1400) / 150)**2) * (1 + 0.2 * np.exp(-wavelengths / 1200)) + np.random.normal(0, 0.02, len(wavelengths))
#    for _ in range(num_spectra)
#])
all_spectra= maskspec3dcube[1379:1875,498:779,:]
original_shape = all_spectra.shape
all_spectra=all_spectra.astype(np.float32).reshape(1, -1)
# Step 2: Prepare PCA-Based Reference Spectrum
scaler = StandardScaler()
scaled_spectra = scaler.fit_transform(all_spectra)  # Normalize spectra before PCA

#pca = PCA(n_components=1)  # Extract first principal component
#pc1 = pca.fit_transform(scaled_spectra.T)[:, 0]

# Rescale PC1 back to original range
#reference_spectrum = scaler.inverse_transform(pc1.reshape(1, -1)).flatten()
#reference_spectrum=reference_spectrum.reshape(original_shape)
#reference_spectrum=reference_spectrum.mean(axis=(0,1))
# Step 3: Apply RMieS-EMSC Correction
#all_spectra=all_spectra.astype(np.float32)
all_spectra = all_spectra.reshape(original_shape)
reference_spectrum = T0
# Select the subset of the 3D spectral data
# Get the shape of the spectra array
num_rows, num_cols, num_bands = all_spectra.shape

# Initialize an array to store corrected spectra
corrected_spectra = np.zeros_like(all_spectra)

# Apply RMie-EMSC correction to each spectrum
for i in range(num_rows):
    for j in range(num_cols):
        distorted_spectrum = all_spectra[i, j, :]
        corrected_spectrum = rmies_emsc(distorted_spectrum, reference_spectrum, iterations=5)
        corrected_spectra[i, j, :] = corrected_spectrum

# Compute the mean spectrum across all pixels
mean_spectrum = np.mean(corrected_spectra, axis=(0, 1))
corrected_spectrum=mean_spectrum
distorted_spectrum=np.mean(all_spectra, axis=(0, 1))

# The `mean_spectrum` contains the averaged spectrum over the entire region

#corrected_spectrum[corrected_spectrum < 0] = 0

# Shift all values so the minimum value in the spectrum becomes zero
#corrected_spectrum -= np.min(corrected_spectrum)

# Step 4: Plot Results
get_ipython().run_line_magic('matplotlib', '')
plt.rcParams.update({'font.size': 15}) # darkgrey tomato\n",
plt.figure()
plt.plot(wavelengths, distorted_spectrum, label="Distorted Spectrum", linestyle="dashed", linewidth=2)
#plt.plot(wavelengths, reference_spectrum, label="Standard Reference", linestyle="dotted", linewidth=2)
plt.plot(wavelengths, corrected_spectrum, label="Corrected Spectrum", linewidth=3)
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Intensity")
plt.title("RMieS-EMSC Correction Using PCA Reference")
plt.legend()
plt.show()

get_ipython().run_line_magic('matplotlib', '')
#plt.figure()
fig, ax = plt.subplots(1,2, figsize=(16,12))
ax[0].imshow(all_spectra[:,:,353], cmap='hot', vmin=all_spectra[:,:,352].min(), vmax= all_spectra[:,:,352].max())
ax[1].imshow(corrected_spectra[:,:,353], cmap='hot', vmin=corrected_spectra[:,:,352].min(), vmax=corrected_spectra[:,:,352].max())
#ax.set_ylabel('pixel')
#ax.set_xlabel('pixel')
#ax[1].plot(wns, spectral[3000,2000,:])
#ax[1].plot(wns, spectral[3380:3420,2780:2820,:].mean(axis=(1,0))) # substrate
#ax[1].plot(wns, spectral[966,1370,:] )
#ax.set_xlabel('pixel')
#ax.set_ylabel('pixel')
plt.show()


# In[98]:


get_ipython().run_line_magic('matplotlib', '')
#plt.figure()
fig, ax = plt.subplots(1,2, figsize=(16,12))
ax[0].imshow(all_spectra[:,:,353], cmap='hot', vmin=1.12*all_spectra[:,:,353].min(), vmax= 1.02*all_spectra[:,:,353].max())
ax[1].imshow(corrected_spectra[:,:,353], cmap='hot', vmin=1.12*corrected_spectra[:,:,353].min(), vmax=1.02*corrected_spectra[:,:,353].max())
#ax.set_ylabel('pixel')
#ax.set_xlabel('pixel')
#ax[1].plot(wns, spectral[3000,2000,:])
#ax[1].plot(wns, spectral[3380:3420,2780:2820,:].mean(axis=(1,0))) # substrate
#ax[1].plot(wns, spectral[966,1370,:] )
#ax.set_xlabel('pixel')
#ax.set_ylabel('pixel')
plt.show()


# In[48]:


import matplotlib.pyplot as plt

# Create figure and subplots
fig, ax = plt.subplots(1, 2, figsize=(16, 12))

# Display images
im = ax[0].imshow(all_spectra[:, :, 353], cmap='jet')
ax[1].imshow(corrected_spectra[:, :, 353], cmap='jet')

# Create a single colorbar for both subplots
cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.07, pad=0.02)
cbar.set_label('Intensity')

# Show plot
plt.show()



# In[ ]:


WT_mean=WT_mean.reshape(-1,1)


# In[ ]:


c=t1m2d


# In[ ]:


maskspec3dcube[2115:2483,441:675,:].shape


# In[ ]:




