import numpy as np
from datetime import date
from scipy.optimize import curve_fit
from os.path import join, isdir
from os import makedirs
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.colors import LogNorm
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle
from PIL import Image
import cv2

def gaussian(x, x0, sigma, area, const):
    """
    Defines a normal distribution with given parameters.

    Inputs:
        x (array)       : x-values to calculate distribution at
        x0 (float)      : Center of distribution
        sigma (float)   : width of distribution
        area (float)    : integrated total of distribution
        const (float)   : constant background

    Output:
        gaussian(array) : Normal distribution evaluated at points x
    """
    prefactor = area / np.sqrt(2 * np.pi * sigma * sigma)
    exponent = -0.5 * np.power(((x - x0) / sigma), 2)
    gaussian = prefactor * np.exp(exponent) + const

    return gaussian

def load_tiff(filename):
    """"
    Loads a tiff image from the VAXSTER instrument into a numpy array.
    Uses pillow package (PIL) method Image()

    Inputs:
        filename (pathlike) :   Path to inputfile

    Output:
        im (array)          :   2D-numpy.array of tiff image
    """
    im = Image.open(filename)
    im = np.array(im)
    return np.array(im)

def find_DB(twoD_data):
    """"
    Finds the position of the direct beam on a 2D detector image.
    Projects 2D data onto x- and y- axis and fits 1D gaussian for each direction.

    Inputs:
        twoD_data (np.array)    :   2D numpay.array of float values describing the detector image

    Output:
        DB (tuple)              :   (1,1)-tuple containing the x- and y- position of the direct beam
    """
    #Project image data onto x- and y-axis
    x_data = np.sum(twoD_data,0)
    y_data = np.sum(twoD_data,1)

    #Define initial guesses for the fit and constrain fit-range to +/- 5 pixels around highest intensity
    x_guess = np.argmax(x_data)
    y_guess = np.argmax(y_data)
    x_data_fit = x_data[x_guess-5:x_guess+6]
    y_data_fit = y_data[y_guess-5:y_guess+6]
    x_p0 = [5,0.5,np.amax(x_data),100]
    y_p0 = [5,0.5,np.amax(y_data),100]

    x_opt,x_cov = curve_fit(gaussian,range(len(x_data_fit)),x_data_fit,p0=x_p0)
    y_opt,y_cov = curve_fit(gaussian,range(len(y_data_fit)),y_data_fit,p0=y_p0)
    DB = (x_opt[0]+x_guess-5-0.5,y_opt[0]+y_guess-5+0.5)

    return DB

def calculate_Q(delta_x,delta_y,dist,k_in,wavelength):
    """"
    Calculate Q-vector for single pixel center.

    Inputs:
        delta_x (float)     :   distance of pixel center from direct beam in x-direction, given in mm
        delta_y (float)     :   distance of pixel center from direct beam in y-direction, given in mm
        dist (float)        :   Sample-detector-distance in mm
        k_in (tuple)        :   3-tuple defining the k-vector of the incident x-ray beam, given in Angström^-1
        wavelength (float)  :   Wavelength of used x-ray source, given in Angström

    Output:
        Q (tuple)           :   3-tuple defining the Q-vector for a scattered x-ray beam impinging on the pixel
    """

    alpha_out = -1*np.atan(delta_y / dist)
    phi = np.atan(delta_x / dist)
    k_out = (2 * np.pi * np.cos(phi) * np.cos(alpha_out) / wavelength,
             2 * np.pi * np.sin(phi) / wavelength,
             2 * np.pi * np.cos(phi) * np.sin(alpha_out) / wavelength)
    Q = (k_out[0] - k_in[0],
         k_out[1] - k_in[1],
         k_out[2] - k_in[2])
    return Q

def calibrate_q_scale_qz_qy(data,DB,dist,px_size,alpha,wavelength):
    """"
    Converts detector image from pixels to Qz vs. Qy scale. Transforms coordinates from pixel center to pixel vertices.

    Inputs:
        data (array)        :   2D numpy array containing scattering data
        DB (tuple)          :   2-tuple containing x- and y- coordinates of DB
        dist (float)        :   Sample-detector distance, given in mm
        px_size (float)     :   Pixel size, given in mm
        alpha (float)       :   Incident angle, given in degrees
        wavelength (float)  :   Wavelength of x-rays, given in Angström

    Output:
        x_values (array)    :   1D numpy.array of Qy values
        y_values (array)    :   1D numpy.array of Qz values
        data (array)        :   Scattering Intensities
    """
    shape = (data.shape[0]+1,data.shape[1]+1)
    x_values = np.zeros(shape)
    y_values = np.zeros(shape)
    k_in = (2*np.pi*np.cos(np.deg2rad(alpha))/wavelength,
            0,
            -2*np.pi*np.sin(np.deg2rad(alpha))/wavelength)
    for i in range(shape[0]):
        for j in range(shape[1]):
            index = (i, j)
            delta_y = (i - DB[1]) * px_size - px_size / 2
            delta_x = (j - DB[0]) * px_size - px_size / 2
            Q = calculate_Q(delta_x,delta_y,dist,k_in,wavelength)
            y_values[index]=Q[2]
            x_values[index]=Q[1]

    return x_values,y_values,data

def calibrate_q_scale_missing_wedge(data,DB,dist,px_size,alpha,wavelength):
    """"
    Converts detector image from pixels to Q_parralel vs. Q_perpendicular scale.
    Transforms coordinates from pixel center to pixel vertices and inserts an empty pixel row
    to account for the missing wedge. For this the central pixel column is split into
    3 pixel columns, splitting the measured intensity according to solid angle ratio.

    Inputs:
        data (array)        :   2D numpy array containing scattering data
        DB (tuple)          :   2-tuple containing x- and y- coordinates of DB
        dist (float)        :   Sample-detector distance, given in mm
        px_size (float)     :   Pixel size, given in mm
        alpha (float)       :   Incident angle, given in degrees
        wavelength (float)  :   Wavelength of x-rays, given in Angström

    Output:
        x_values (array)    :   1D numpy.array of Q_parallel values
        y_values (array)    :   1D numpy.array of Q_perpendicular values
        data (array)        :   Scattering Intensities
    """
    #The new shape has a shape of (data+1,data+3) to account for vertices and the missing wedge insertion.
    shape = (data.shape[0]+1,data.shape[1]+3)
    new_data = np.zeros((data.shape[0],data.shape[1]+2))
    x_values = np.zeros(shape)
    y_values = np.zeros(shape)
    k_in = (2*np.pi*np.cos(np.deg2rad(alpha))/wavelength,
            0,
            -2*np.pi*np.sin(np.deg2rad(alpha))/wavelength)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]+1):
            index = (i, j)
            delta_y_down = (i - DB[1]) * px_size - px_size / 2 #Calculate y-distance of lower vertice from DB

            delta_x_center = (j - DB[0]) * px_size #Calculate x-distance of pixel center from DB
            delta_x_left = delta_x_center - px_size / 2
            delta_x_right = delta_x_center + px_size / 2

            #If true, left and right vertices are on different sides of the DB
            if delta_x_left*delta_x_right < 0:
                ratio_left = 0.5 - (j - DB[0])
                # for plotting the size of the missing wedge will be calculated as +-5% of pixel size left/right of center
                missing_wedge_x_left = -0.05 * (ratio_left * px_size)
                missing_wedge_x_right = 0.05 * (1 - ratio_left) *px_size
                if j < new_data.shape[1] and i < new_data.shape[0]:
                    #splitting central pixel into 3 parts, left of centre(j), right of centre(j+2) and missing wedge(j+1).
                    #dividing counts according to position of central pixel
                    new_data[index]=data[(i,j)]*ratio_left
                    new_data[(i,j+1)] = -1
                    new_data[(i,j+2)]=data[(i,j)]*(1-ratio_left)
                #For the last pixel the upper vertices are calculated additionally
                if i < 606:
                    k_max = 2
                else:
                    k_max = 1
                for k in range(k_max):
                    delta_y = delta_y_down+k*px_size
                    Q_mW_left = calculate_Q(missing_wedge_x_left,delta_y,dist,k_in,wavelength)
                    Q_mW_right = calculate_Q(missing_wedge_x_right,delta_y,dist,k_in,wavelength)
                    Q = calculate_Q(delta_x_left,delta_y,dist,k_in,wavelength)
                    x_k0 = np.sqrt(np.power(Q[0],2)+np.power(Q[1],2))*np.sign(Q[1])
                    y_k0 = Q[2]
                    x_k1 = np.sqrt(np.power(Q_mW_left[0],2)+np.power(Q_mW_left[1],2))*np.sign(Q_mW_left[1])
                    y_k1 = Q_mW_left[2]
                    x_k2 = np.sqrt(np.power(Q_mW_right[0],2)+np.power(Q_mW_right[1],2))*np.sign(Q_mW_right[1])
                    y_k2 = Q_mW_right[2]
                    x_values[(i+k,j)]=x_k0
                    x_values[(i+k,j+1)]=x_k1
                    x_values[(i+k,j+2)]=x_k2
                    y_values[(i+k,j)]=y_k0
                    y_values[(i+k,j+1)]=y_k1
                    y_values[(i+k,j+2)]=y_k2
            #Otherwise all vertices of the pixel are either left or right of the DB
            else:
                #Right of the DB the x-Index has to be incremented by 2 to account for the inserted missing wedge columns
                if delta_x_center > 0:
                    index = (i, j+2)
                try:
                    new_data[index]=data[(i,j)]
                except:
                    pass

                Q = calculate_Q(delta_x_left,delta_y_down,dist,k_in,wavelength)
                x_values[index]=np.sqrt(np.power(Q[0],2)+np.power(Q[1],2))*np.sign(Q[1])
                y_values[index]=Q[2]

    return x_values,y_values,new_data

def get_filepair(folder,start):
    """"
    Creates the filepaths for two consecutive file numbers.
    """
    files = np.array(range(2))+start
    files = [join(folder,f"latest_{i:07d}_craw.tiff") for i in files]
    return files

def merge_images(files):
    """"
    Fits the DB position of two images and merges the data according to the DB shift.
    The shift is performed in whole pixel values.
    """
    data1 = load_tiff(files[0])
    data2 = load_tiff(files[1])

    DB1 = find_DB(data1)
    DB2 = find_DB(data2)

    delta_y = round(DB2[1]-DB1[1])
    shape = data1.shape
    new_data = np.zeros((shape[0]+np.abs(delta_y),shape[1]))
    with np.nditer(new_data,flags=["multi_index"],op_flags=['writeonly']) as iterator:
        for x in iterator:
            index = iterator.multi_index
            new_int = -1
            i1 = -1
            i2 = -1
            if index[0] < shape[0]:
                i1 = float(data1[index])
            if index[0] + delta_y >= 0:
                i2 = float(data2[(index[0]+delta_y,index[1])])

            if i1*i2 >= 0:
                x[...] = (i1 + i2)/2
            else:
                x[...] = max(i1, i2)

    new_data = new_data[:round(DB1[1])+1,:]
    return DB1, new_data

def plot_2d(ax,x_values,y_values,data,vmin=1,vmax=1e3,colorbar=True,xlim=None,ylim=None):
    quadmesh = ax.pcolormesh(x_values,y_values,data,norm=LogNorm(vmin=vmin,vmax=vmax))
    # ax.axis('equal')
    # ax.set_adjustable("box")
    if type(xlim)==type(None):
        ax.set_xlim(auto=xlim)
    else:
        ax.set_xlim(xlim)

    if type(ylim)==type(None):
        ax.set_ylim(auto=xlim)
    else:
        ax.set_xlim(xlim)

    if colorbar:
        plt.gcf().colorbar(quadmesh,ax=ax,shrink=0.6,label="Intensity [cts.]")

def get_qz_qy_cuts(qz,qy,data,DB,dqz=5,dqy=5):
    x_pos = int(DB[0])
    y_pos = int(DB[1])
    qz = qz[1:,:-1]
    qy = qy[1:,:-1]
    new_qy = qz[y_pos]
    new_qz = np.transpose(qy)[x_pos]

    data_qz =  np.transpose(data)[x_pos-int(dqy/2):x_pos+int(dqy/2)+int(dqy%2)][:]
    data_qy =  data[y_pos-int(dqz/2):y_pos+int(dqz/2)+int(dqz%2)][:]

    data_qz = np.sum(data_qz,0)
    data_qy = np.sum(data_qy,0)

    pixel_size_y =np.abs(new_qy[x_pos]-new_qy[x_pos+1])
    pixel_size_z =np.abs(new_qz[y_pos]-new_qz[y_pos+1])

    rz_ypos = new_qy[x_pos]-int(dqy/2)*pixel_size_y
    ry_zpos = new_qz[y_pos]-int(dqz/2)*pixel_size_z

    rect_z = Rectangle((rz_ypos,new_qz.min()),dqz*pixel_size_y,np.abs(new_qz[0]-new_qz[-1]),linewidth=1,edgecolor="r",facecolor="none")
    rect_y = Rectangle((new_qy.min(),ry_zpos),np.abs(new_qy[0]-new_qy[-1]),dqy*pixel_size_z,linewidth=1,edgecolor="r",facecolor="none")
    return new_qz,new_qy,data_qz,data_qy,rect_z,rect_y

def rotate_axis(axis,degree):
    r = Affine2D().rotate_deg(degree)
    for x in axis.images + axis.lines + axis.collections:
        trans = x.get_transform()
        x.set_transform(r + trans)
        if isinstance(x, PathCollection):
            transoff = x.get_offset_transform()
            x._transOffset = r + transoff

    old = axis.axis()
    axis.axis(old[2:4]+old[0:2])
    return axis

def prepare_for_writing(y_values,x_values,data):
    y_low = []
    y_high = []
    x_low = []
    x_high =[]
    new_data = []
    with np.nditer(data,flags=["multi_index"]) as iterator:
        for int in iterator:
            new_data.append(int)
            index=iterator.multi_index
            y_low.append(y_values[index])
            x_low.append(x_values[index])
            high_index = tuple(np.array(index)+1)
            y_high.append(y_values[high_index])
            x_high.append(x_values[high_index])
    return np.array(y_low),np.array(y_high),np.array(x_low),np.array(x_high),np.array(new_data)

def create_folders(outdir,specifier=""):
    """
    Creates a folder named the current date + an optional specifier

    """
    dirname = f"{date.today()}{specifier}"
    if not isdir(join(outdir,dirname)):
        makedirs(join(outdir,dirname))

    return join(outdir,dirname)

def make_mp4(path, video_path,nbrs):
    images = []
    for i in nbrs:
        images.append(join(path, "{:05d}_overview_plots.png".format(i)))
    frame = cv2.imread(images[0])
    height, width, channels = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(join(video_path, "full_rotation.mp4"), fourcc, 2.0, (width, height))

    for ctr, image in enumerate(images):
        if ctr % 10 == 0:
            print(ctr)
        frame = cv2.imread(image)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

def main():
    pass

if __name__=='__main__':
    main()
