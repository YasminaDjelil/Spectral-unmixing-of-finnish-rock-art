import os
import numpy as np
import matplotlib.pyplot as plt
import re
from osgeo import gdal
from sklearn.decomposition import NMF
import pysptools.eea as eea
from sklearn.cluster import KMeans

def extract_envi_info_from_file(file_path):
    # Extract (samples, lines, bands)
    sample_pattern = re.compile(r'samples\s*=\s*(\d+)', re.IGNORECASE)
    band_pattern = re.compile(r'bands\s*=\s*(\d+)', re.IGNORECASE)
    line_pattern = re.compile(r'lines\s*=\s*(\d+)', re.IGNORECASE)

    with open(file_path, 'r') as file:
        header_text = file.read()

    samples_match = sample_pattern.search(header_text)
    bands_match = band_pattern.search(header_text)
    lines_match = line_pattern.search(header_text)

    samples = int(samples_match.group(1)) if samples_match else None
    bands = int(bands_match.group(1)) if bands_match else None
    lines = int(lines_match.group(1)) if lines_match else None
    # Extract wavelenght
    wavelengths = re.findall(r'\d+\.\d+', header_text)
    wavelengths = [float(wavelength) for wavelength in wavelengths]


    return samples, bands, lines, wavelengths

def load_raw_hyperspectral_data(raw_path, hdr_path, wavelengths, bands, lines, samples, crop_region):

    raw_data = np.empty((bands, lines, samples), dtype=np.uint16)

    raw_dataset = gdal.Open(raw_path)

    for i in range(bands):
        band_data = raw_dataset.GetRasterBand(i+1).ReadAsArray()
        raw_data[i, :, :] = band_data

    raw_data = np.transpose(raw_data, (1, 2, 0))
    print("Shape of raw data:", raw_data.shape)
    print("start wavelength =", wavelengths[0])
    print("end wavelength =", wavelengths[-1])
    print("step wavelength =", wavelengths[1]-wavelengths[0])
    num_bands = len(wavelengths)

    # cropping
    if crop_region:
        x, y, width, height = crop_region
        cropped_data = raw_data[y:y+height, x:x+width, :]
        return cropped_data
    else:
        # Reshape
        raw_data = raw_data.reshape( lines, samples, num_bands)
        return raw_data

def plot_abundance_maps(abundance_maps, save_dir=None, base_filename='abundance_map'):
    num_endmembers = abundance_maps.shape[2]
    plt.figure(figsize=(10, 6))

    for i in range(num_endmembers):
        plt.subplot(1, num_endmembers, i + 1)
        im = plt.imshow(abundance_maps[:, :, i], cmap='gray')

        if save_dir:
            plt.axis('off')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f'{base_filename}_endmember_{i + 1}.png')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.clf()
        else:
            plt.colorbar(im, aspect=5)
            plt.xlabel('Column')
            plt.ylabel('Row')
            plt.title(f'Abundance Map for Endmember {i + 1}')
            plt.axis('on')

    if save_dir:
        print(f'Abundance maps saved to {save_dir}')
    else:
        plt.tight_layout(pad=1)
        plt.show()

   def Unmix(raw_path, hdr_path, EM_method='NFINDR', q=2, abun_method='FCLSU', crop_region=None, normalize=True):
    # 1: extract
    samples, bands, lines, wavelengths = extract_envi_info_from_file(hdr_path)
    
    # é: load data
    raw_data = load_raw_hyperspectral_data(raw_path, hdr_path, wavelengths, bands, lines, samples, crop_region)
    print(f"Data loaded with shape: {raw_data.shape}")
    
    # 3: endmember extraction method
    if EM_method == 'NFINDR':
        EM_spectra = eea.NFINDR().extract(M=raw_data, q=q, transform=None, maxit=100, normalize=normalize, mask=None)
    elif EM_method == 'PPI':
        EM_spectra = eea.PPI().extract(M=raw_data, q=q, numSkewers=1000, normalize=normalize, mask=None)
    elif EM_method == 'KMeans':
        kmeans = KMeans(n_clusters=q, random_state=0)
        kmeans.fit(raw_data.reshape(-1, 204))
        EM_kmeans = kmeans.cluster_centers_
        EM_spectra, _ = (EM_kmeans - np.min(EM_kmeans)) / (np.max(EM_kmeans) - np.min(EM_kmeans))
    else:
        raise ValueError(f"Unknown endmember extraction method: {EM_method}")

    # 4: abundance maps reconstruction method

    num_rows, num_columns, num_bands = raw_data.shape
    reshaped_data = raw_data.reshape(-1, num_bands)
    if abun_method == 'FCLSU':
        A = EM_spectra.T
        A_pseudo_inv = np.linalg.pinv(A)
        abundance_maps = np.dot(A_pseudo_inv, reshaped_data.T)
        abundance_maps = abundance_maps.T.reshape(num_rows, num_columns, -1)

    elif abun_method == 'NMF': #unsupervised method
        nmf = NMF(n_components=q, max_iter=600, init='nndsvd', random_state=0)  # some init might improve the results
        W = nmf.fit_transform(reshaped_data)  # W = abundance maps
        H = nmf.components_  # H = endmember spectra
        abundance_maps = W.reshape(num_rows, num_columns, -1)
    else:
        raise ValueError(f"Unknown abundance method: {abun_method}")

    
    abundance_maps = (abundance_maps  - np.min(abundance_maps)) / (np.max(abundance_maps )- np.min(abundance_maps))

    plot_abundance_maps(abundance_maps)
    
    return abundance_maps
