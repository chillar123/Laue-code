##################### IMPORT OF NEEDED PACKAGES #####################

from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from skued import diffread, autocenter, nfold
from itertools import chain
import matplotlib.image

##################### END #####################


##################### SMALLER USED FUNCTIONS #####################
# - replace_boundary, only makes the lower_bound and upper_bound appear in a dataset in matrix/array
def replace_boundary(matrix, lower_bound, upper_bound):
    result_matrix = matrix.copy()

    # Replace lower boundary values
    result_matrix[result_matrix > upper_bound] = 1
    result_matrix[result_matrix < lower_bound] = 0

    """
    Returns matrix without values above upper bound and without values below lower bound.

    matrix: np.array of data matrix.
    lower_bound: lower value bound for input matrix.
    upper_bound: upper value bound for input matrix.
    returns: clusters and the amount of clustering
    """
    return result_matrix


# - listOfLists, removes nested "(" "," ")"  for a list
def flatten(listoflists):
    'Flatten one level of nesting lists'
    return chain.from_iterable(listoflists)


def find_clusters(array):
    """
    Returns connected data points in a matrix.

    array: np.array of data matrix.
    returns: clusters and the amount of clustering
    """
    clustered = np.empty_like(array)
    unique_vals = np.unique(array)
    cluster_count = 0
    for val in unique_vals:
        labelling, label_count = ndimage.label(array == val)
        for k in range(1, label_count + 1):
            clustered[labelling == k] = cluster_count
            cluster_count += 1
    return clustered, cluster_count


# - filter_isolated_cells, looks at matrix elements and only keeps them if there are other values in the 8 elements around,it in a matrix (compares it to a 3x3 ones-matrix)
def filter_isolated_cells(array, struct):
    """
    Returns array with isolated single cells removed from matrix.

    array: np.array of data for single cell removal
    struct: structure array for generating unique regions
    returns: Array with minimum region size > 1
    """

    filtered_array = np.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0

    return filtered_array


def fixed_size_subset(a, x, y, size):
    """
    Gets a subset of 2D array given x and y coordinates
    and an output size. If the slices exceed the bounds
    of the input array, the non overlapping values
    are filled with NaNs.

    a: np.array, 2D array from which to take a subset
    x, y: int, coordinates of the center of the subset
    size: int, size of the output array
    returns: np.array, the subset of the input array
    """
    o, r = np.divmod(size, 2)
    l = (x - (o + r - 1)).clip(0)
    u = (y - (o + r - 1)).clip(0)
    a_ = a[l: x + o + 1, u:y + o + 1]
    out = np.full((size, size), np.nan, dtype=a.dtype)
    out[:a_.shape[0], :a_.shape[1]] = a_
    return out


##################### END #####################


##################### FIRST DENOISED IMAGES #####################
def denoised_images(lower_limit, upper_limit):
    # Main variables/parameters:

    # In replace_boundary:
    # - upper_limit, matrix element value upper bound (anything above gets set to 0)
    # - lower_limit, matrix element value lower bound (anything below gets set to 0)

    im = data
    # Sets a max value of the dataset which is visible within the wanted range
    maxvalue = np.mean(data) + np.median(data)
    # A 3x3 kernel reduces some noise the image
    im_kernel1 = ndimage.median_filter(im, size=(3, 3))
    # A 50x50 kernel mimics the noise of the background
    im_kernel2 = ndimage.median_filter(im, size=(50, 50))

    # Removing background noise
    im_kernel_denoised = im_kernel1 - im_kernel2

    # Plot of 3x3 kernel denoised graph
    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(im_kernel1, vmin=0, vmax=maxvalue, cmap='inferno')
    ax1.set_title('Data (3x3 kernel)\nClose this figure to proceed!')

    plt.show()

    # If the denoised image looks good, we use this from now on
    im = im_kernel_denoised

    plt.show()
    # To always have values in the matrix which are >>1:
    offset = 1000
    im = im + offset

    # Set a threshold for the Bragg peaks to background

    im = replace_boundary(im, lower_limit + offset, upper_limit + offset)

    # The values within the threshold is set to 1 (needs upper/lower bound threshold to have vales to be >>1)
    im = np.divide(im, im + np.ones_like(im))
    im = im.round(decimals=0, out=None)

    # Filters away data which has no adjacent other data
    im = filter_isolated_cells(im, struct=np.ones((3, 3)))


    return im


##################### END #####################


##################### MASK FOR REMOVING LOUD OR ARTIFACT DATA #####################
def square_center_mask(y1, y2, x1, x2, lower_limit, upper_limit):
    # Main variables/parameters:
    # In y1,y2,x1,x2
    # - Coordinates for the whole beam block. Set a y1:y2,x1:x2 rectangle where bool will be false, removes these datapoints


    # Makes a bool type same size matrix as the image
    mask = np.ones_like(im, dtype=bool)

    # mask for beamblock given as coordinates like (top-right2,bottom-left2,bottom-left1,top-right1) from the plots
    mask[y1:y2, x1:x2] = False

    # Merges the loud pixel mask and image
    im1 = mask * im

    # Finds high symmetry center from inversion symmetry around the image center
    rc, cc = autocenter(im, mask)

    # plots the merged loud pixel mask and image with a red dot at center
    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(im1, vmin=0, vmax=1, cmap='inferno')
    ax1.scatter(cc, rc, color='g')
    ax1.set_title('Image with pre-made beam block mask and center dot\nClose this figure to proceed!')
    plt.show()

    return rc, cc, mask, im


##################### END #####################


##################### N-FOLD SYMMETRY #####################
def n_fold(y1, y2, x1, x2, nfolds, smoothening):
    # Main variables/parameters:
    # In nfold:
    # - nfolds, set int as fold symmetry

    # In y1,y2,x1,x2
    # - Coordinates for the whole beam block. Set a y1:y2,x1:x2 rectangle where bool will be false, removes these datapoints

    # In smoothening
    # - smoothening, median kernel averaging if threshold left some noise in the image

    # Mirrors the set amount of symmetry folds of the data by first removing beamblock with a mask
    mask[y1:y2, x1:x2] = False
    im_fold = im * mask
    av = nfold(im_fold, mod=nfolds, center=(cc, rc), mask=mask)
    av = np.ceil(av)

    # A more thorough removal of small noise clusters
    sat = np.pad(av, pad_width=1, mode='constant', constant_values=0)
    sat = np.cumsum(np.cumsum(sat, axis=0), axis=1)
    sat = np.pad(sat, ((1, 0), (1, 0)), mode='constant', constant_values=0)

    # These are all the possible overlapping 3x3 windows sums
    sum3x3 = sat[3:, 3:] + sat[:-3, :-3] - sat[3:, :-3] - sat[:-3, 3:]

    # This takes away the central pixel value
    sum3x3 -= av

    # This zeros all the isolated pixels
    av[sum3x3 == 0] = 0
    av = replace_boundary(av, 0.999, 1)

    # If any noise is left, another kernel averaging can be done
    av = ndimage.median_filter(av, size=(smoothening, smoothening))
    av = np.round(av)

    # Plots the image, beamblock mask, removed beam block with center and then the image with nfold symmetry
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 3))

    ax1.imshow(im, vmin=0, vmax=1, cmap='inferno')

    ax2.imshow(np.logical_not(mask) * im, vmin=0, vmax=1, cmap='inferno')

    ax3.imshow((im * mask), vmin=0, vmax=1, cmap='inferno')
    ax3.scatter(cc, rc, color='g')

    ax4.imshow(av, vmin=0, vmax=1, cmap='inferno')

    ax4.scatter(cc, rc, color='g')

    ax1.set_title('Filtered data with beam block')
    ax2.set_title('Beam block mask')
    ax3.set_title('Removal of beam block')
    ax4.set_title('Applied n-fold symmetry')
    plt.tight_layout()
    plt.show()

    return av


##################### END #####################


##################### BRAGG PEAK DETECTION, IMAGING AND RETURNING LOCATION #####################
def bragg_peak_location(minarea):
    # Main variables/parameters:
    # In if (for cluster sizes)
    # - minarea, the smallest area of a wanted cluster

    # Finds clusters in data
    clusters, cluster_count = find_clusters(av)

    # Empty list for append of center coordinates for all found clusters
    bragg_peak_location = []

    # Finds a suitable max size of clusters (we want all the large clusters if filtering is done correctly)
    maxarea = len(av) * len(av)
    # Finds clusters between a threshold (pixel area of cluster) and appends it into an array
    ones = np.ones_like(av, dtype=int)
    cluster_sizes = ndimage.sum(ones, labels=clusters, index=range(cluster_count)).astype(int)
    com = ndimage.center_of_mass(ones, labels=clusters, index=range(cluster_count))
    for i, (size, center) in enumerate(zip(cluster_sizes, com)):
        if minarea < size < maxarea:
            bragg_peak_location.append(center)

    # Formats found cluster locations (x,y values) from [(x,y),(x,y),...] to [x,y,x,y,...] and rounds them to a pixel
    my_list = bragg_peak_location
    my_flattened_list = list(flatten(my_list))
    my_flattened_list = np.array(np.float_(my_flattened_list))
    my_flattened_list = np.round(my_flattened_list)

    # Separates x from y coordinates of cluster locations
    bragg_peak_location_y = my_flattened_list[::2]  # Start at first element, then every other
    bragg_peak_location_x = my_flattened_list[1::2]  # Start at second element, then every other

    # Empty list for append of distance from center of data to found clusters
    bragg_peak_distance_to_center = []

    # Finds distance from each cluster to center
    for i in range(len(bragg_peak_location_x)):
        bragg_peak_distance_to_center.append(
            np.sqrt((abs(bragg_peak_location_x[i] - cc) ** 2 + abs(bragg_peak_location_y[i] - rc) ** 2)))

    # Makes distance list into an array
    bragg_peak_distance_to_center = np.array(bragg_peak_distance_to_center)

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))

    for i, txt in enumerate(np.round(bragg_peak_distance_to_center, decimals=1)):
        ax1.annotate(txt, (bragg_peak_location_x[i], bragg_peak_location_y[i]), color='m', fontsize=8)

    # Sorts distance list from the closest cluster to the furthest away (needs to be an array not a list... unlucky)
    bragg_peak_distance_to_center = sorted(bragg_peak_distance_to_center, key=lambda x: float(x))

    # Prints all the cluster to center distances
    print('Ascending cluster to center distances in pixels:')
    print(bragg_peak_distance_to_center)

    # Plots the data with a red dot on the found clusters
    ax1.imshow(av, cmap='inferno', interpolation='nearest')
    ax1.scatter(cc, rc, color='g')
    ax1.set_title('Clusters to center distance in pixels\nClose this figure to proceed!')
    plt.plot(bragg_peak_location_x, bragg_peak_location_y, 'o', color='r')
    plt.savefig(filename+' Clusters to center plot.png')

    plt.show()
    matplotlib.image.imsave(filename+' Output image.png', data * av)

    return bragg_peak_distance_to_center, bragg_peak_location_x, bragg_peak_location_y


##################### END #####################


##################### BRAGG PEAK RELATIVE INTESITY #####################

def Relativ_Intensities():
    average_intensity_bragg_peak = []
    ### ROUND BRAGG PEAK LOCATION x,y
    for i in range(len(bragg_peak_location_x)):
        average_intensity_bragg_peak.append(np.mean(
            fixed_size_subset(data, bragg_peak_location_y.astype(int)[i], bragg_peak_location_x.astype(int)[i], 10)))

    fig, ax = plt.subplots(figsize=(8, 8))

    im_kernel1 = ndimage.median_filter(data, size=(3, 3))
    maxvalue = np.mean(data) + np.median(data)

    ax.imshow(im_kernel1, vmin=0, vmax=maxvalue, cmap='inferno')
    ax.set_title('Data (3x3 kernel)\nClose this figure to proceed!')
    ax.scatter(bragg_peak_location_x, bragg_peak_location_y)
    for i, txt in enumerate(np.round(average_intensity_bragg_peak / max(average_intensity_bragg_peak), decimals=2)):
        ax.annotate(txt, (bragg_peak_location_x[i], bragg_peak_location_y[i]), fontsize=8)

    ax.set_title('Cluster relative intensities')
    plt.savefig(filename+' Relative intensity plot.png')
    plt.show()

    return


##################### END #####################


##################### USER INTERFACE #####################

# Global statement to load data before processing data.
Globalstatement = False

while True:

    print(" \n"
          "--Main menu--\n"
          "1: Load data\n"
          "2: Find Bragg Peaks\n"
          "3: Quit\n")
    userinput = input("Choose number corresponding to a menu item: ")
    print(" ")

    # Preventing user to process data without loading data.
    if userinput != "1" and userinput != "3" and Globalstatement == False:
        print("\n-----------------------Error!-------------------------")
        print("Error: Data must be loaded first.")

    elif userinput == "3":
        print("\n-----------------------Quitting!-------------------------")
        quit()
    else:

        # Opens the data file from path
        if userinput == "1":
            while True:
                print("\n-----------------------Write Path-------------------------")
                print("Input 0 to go back to main menu.")
                filename = input("Write the path a .tif Laue image: ")
                # Return to main menu.

                try:
                    if filename == "0":
                        break
                    else:
                        data = diffread(filename)
                        print("\nData load complete!\n")
                        Globalstatement = True
                        break

                except FileNotFoundError:
                    print("\n\n-----------------------Error!-------------------------")
                    print("Error: Data file is not of correct format or is not found.")
                    pass



        # finds bragg peaks
        elif userinput == "2":

            while True:
                print("\n-----------------------Bragg Peak Value Threshold-------------------------")
                print("Generating images...")
                print(
                    'Hover Bragg peaks on the shown figure with your cursor and note the maximum and minimum intensity values of the Bragg peaks in the bottom right of the figure window.'
                    '\nZoom in with the magnifying glass icon if needed.')
                print("\nClose the figure window to proceed.")
                im = data
                # Sets a max value of the dataset which is visible within the wanted range
                maxvalue = np.mean(data) + np.median(data)
                # A 3x3 kernel reduces some noise the image
                im_kernel1 = ndimage.median_filter(im, size=(3, 3))
                # A 50x50 kernel mimics the noise of the background
                im_kernel2 = ndimage.median_filter(im, size=(50, 50))

                # Removing background noise
                im_kernel_denoised = im_kernel1 - im_kernel2

                fig, ax1 = plt.subplots(figsize=(6, 6))
                ax1.imshow(im_kernel_denoised, vmin=0, vmax=maxvalue, cmap='inferno')
                ax1.set_title('Background removed from data\nClose this figure to proceed!')
                plt.show()
                print("\n")
                print("\n------------------------------------------------")
                upperlimit = int(input("Write an upper limit pixel value for the Bragg peaks: "))
                lowerlimit = int(input("Write a lower limit pixel value for the Bragg peaks: "))

                ####
                print("\n------------------------------------------------")
                print("Generating images...")
                im = replace_boundary(im_kernel_denoised, lowerlimit, upperlimit)
                rc, cc, notused, notused2 = square_center_mask(0, 360, 280, 400, lowerlimit, upperlimit)
                fig, ax1 = plt.subplots(figsize=(6, 6))
                ax1.imshow(im, vmin=lowerlimit, vmax=upperlimit, cmap='inferno')
                ax1.scatter(cc, rc, color='g')
                ax1.set_title('Are the Bragg peaks and the center of the high symmetry point shown?\nClose this figure to proceed!')
                plt.show()

                print("Are you satisfied with the bounds and the center?"
                      "\n1: Yes"
                      "\n2: No, let me try again")
                satisfied = input("\nChoose number corresponding to a menu item: ")


                if satisfied == "2":
                    pass
                elif satisfied == "1":
                ###

                    while True:
                        print("\n------------------------------------------------")
                        print("Generating images... Might take a few seconds, in the mean time, read below!")
                        print(
                            "\nA standard beam block is available, but if you want to make your own mask for the beam block: Find pixel positions of a square covering the beam block. "
                            "\nHover curser on current image and find top-left x1,y1 and bottom-right x2,y2 coordinates and note them."
                            "\n\nClose the first figure window to proceed to the pre-made beam block mask figure.")

                        im = denoised_images(lowerlimit, upperlimit)
                        print("\n-----------------------Mask For Beam Block-------------------------")
                        print(
                            "This image is with the pre-made beam block, the green dot should be at the high symmetry point.")
                        print("Close the figure window to proceed. Don't worry about the `FutureWarning´ :-)")
                        rc, cc, mask, im = square_center_mask(0, 360, 280, 400, lowerlimit, upperlimit)
                        # rc, cc, mask, im = square_center_mask(y1, y2, x1, x2)

                        print("\n------------------------------------------------")
                        print("Have you noted some mask coordinates and do you want to make your own mask? (remember that y1<y2 and x1<x2)"
                              "\nThe pre-made beam block mask is: x1=280, y1=0, x2=400, y2=360."
                              "\n1: Yes"
                              "\n2: No, use the pre-made beam block mask")
                        satisfied = input("\nChoose number corresponding to a menu item: ")


                        if satisfied == "1":
                            print("\n------------------------------------------------")
                            print("Input the mask coordinates below.")

                            x1 = int(input("The top-left x1 value: "))
                            y1 = int(input("The top-left y1 value: "))

                            x2 = int(input("The bottom-right x2 value: "))
                            y2 = int(input("The bottom-right y2 value: "))
                            print("\n------------------------------------------------")
                            print("Generating images... Don't worry about the `FutureWarning´ :-)"
                                  "\nClose the figure window to proceed.")
                            rc, cc, mask, im = square_center_mask(y1, y2, x1, x2, lowerlimit, upperlimit)

                            print("Are you satisfied with your own beam block mask?"
                                  "\n1: Yes"
                                  "\n2: No, let me try again")
                            satisfied = input("\nChoose number corresponding to a menu item: ")

                        elif satisfied == "2":
                            y1 = int(0)
                            y2 = int(360)
                            x1 = int(280)
                            x2 = int(400)
                            satisfied = "1"

                        if satisfied == "2":
                            pass
                        elif satisfied == "1":
                            while True:
                                print("\n-----------------------Fold Symmetry-------------------------")
                                n = int(
                                    input("Input a folding symmetry of the Laue image, input 1 if no folding is wanted: "))
                                f = int(input(
                                    "Input a size of a median kernel filter to apply on the Laue image, input 1 if no filter is wanted: "))
                                print("\n\nGenerating images... Might take a few seconds.")
                                print("\nClose the figure to proceed.")
                                av = n_fold(y1, y2, x1, x2, n, f)
                                print("\n------------------------------------------------")
                                print("Are you satisfied with the fold symmetry and the filtering?"
                                      "\n1: Yes"
                                      "\n2: No, let me try again")
                                satisfied = input("\nChoose number corresponding to a menu item: ")

                                if satisfied == "2":
                                    pass
                                elif satisfied == "1":

                                    while True:
                                        print("\n-----------------------Finding Bragg peaks-------------------------")
                                        print(
                                            "Estimate the smallest area of wanted Bragg peak (in the unit squared pixels) on the opened figure, use the zoom tool in the plot toolbar."
                                            "\nClose the figure window to proceed.")
                                        fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
                                        ax1.imshow(av, vmin=0, vmax=1, cmap='inferno')
                                        ax1.scatter(cc, rc, color='g')
                                        ax1.set_title("Note the minimum allowed Bragg peak pixels' area\nClose this figure to proceed!")
                                        plt.show()

                                        print("\n------------------------------------------------")
                                        ma = int(input("Input the smallest area of one of the found Bragg peak: "))
                                        bragg_peak_distance_to_center, bragg_peak_location_x, bragg_peak_location_y = bragg_peak_location(
                                            ma)

                                        print("\n------------------------------------------------")
                                        print("Are you satisfied with the found Bragg peaks?"
                                              "\n1: Yes"
                                              "\n2: No, let me try again")
                                        satisfied = input("\nChoose number corresponding to a menu item: ")

                                        if satisfied == "2":
                                            pass
                                        elif satisfied == "1":
                                            print(
                                                "\n-----------------------Relative Intensities of Bragg peaks-------------------------")
                                            print("This is a plot of the relative intensities of the found Bragg peaks."
                                                  "\nClose the figure window to quit. Again, don't worry about the `RuntimeWarning´ :-)")
                                            Relativ_Intensities()

                                            print("\nThe data is saved in the same folder as the input data with the filename attached.")

                                            quit()
##################### END #####################