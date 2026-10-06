import numpy as np
from scipy.optimize import curve_fit
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.colors import LogNorm
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle
from matplotlib.layout_engine import ConstrainedLayoutEngine
from PIL import Image
from pandas import ExcelWriter, DataFrame

def gaussian(x, x0, sigma, area, const):
    prefactor = area / np.sqrt(2 * np.pi * sigma * sigma)
    exponent = -0.5 * np.power(((x - x0) / sigma), 2)
    gaussian = prefactor * np.exp(exponent) + const

    return gaussian

def load_tiff(filename):
    im = Image.open(filename)
    im = np.array(im)
    # plt.imshow(im,norm=LogNorm())
    # plt.show()
    return np.array(im)

def find_DB(twoD_data):
    x_data = np.sum(twoD_data,0)
    y_data = np.sum(twoD_data,1)

    x_guess = np.argmax(x_data)
    y_guess = np.argmax(y_data)
    x_data_fit = x_data[x_guess-5:x_guess+6]
    y_data_fit = y_data[y_guess-5:y_guess+6]
    x_p0 = [5,0.5,np.amax(x_data),100]
    y_p0 = [5,0.5,np.amax(y_data),100]

    x_opt,x_cov = curve_fit(gaussian,range(len(x_data_fit)),x_data_fit,p0=x_p0)
    y_opt,y_cov = curve_fit(gaussian,range(len(y_data_fit)),y_data_fit,p0=y_p0)
    DB = (x_opt[0]+x_guess-5-0.5,y_opt[0]+y_guess-5+0.5)

    # im_ax = plt.subplot2grid((3,3),(0,0),2,2)
    # x_ax = plt.subplot2grid((3,3),(2,0),1,2)
    # y_ax = plt.subplot2grid((3,3),(0,2),2,1)
    # im_ax.imshow(twoD_data,norm=LogNorm())
    # x_ax.semilogy(x_data)
    # x_ax.semilogy(gaussian(range(len(x_data)),DB[0],*x_opt[1:]))
    # y_ax.semilogy(y_data)
    # y_ax.semilogy(gaussian(range(len(y_data)),DB[1],*y_opt[1:]))
    # plt.tight_layout()
    # plt.show()
    return DB

def calculate_q(x_pix,y_pix,dist,wavelength=1.3414):
    tth = np.atan(np.sqrt(x_pix^2+y_pix^2)/(dist))
    q_total = 4*np.pi*np.sin(tth/2)/wavelength
    q_x = 1

def calculate_Q(alpha,delta_x,delta_y,dist,k_in,wavelength):
    alpha_out = -1*np.atan(delta_y / dist)
    phi = np.atan(delta_x / dist)
    k_out = (2 * np.pi * np.cos(phi) * np.cos(alpha_out) / wavelength,
             2 * np.pi * np.sin(phi) / wavelength,
             2 * np.pi * np.cos(phi) * np.sin(alpha_out) / wavelength)
    Q = (k_out[0] - k_in[0],
         k_out[1] - k_in[1],
         k_out[2] - k_in[2])
    # if np.rad2deg(alpha_out)>5:
        # print(f"Alpha in: {alpha}, Alpha out:{np.rad2deg(alpha_out)}, phi:{np.rad2deg(phi)}")
        # print(f"k_out: {k_out}")
        # print(f"k_in:{k_in}")
        # print(f"Q:{Q}")
    return Q

def calibrate_q_scale_qz_qy(data,DB,dist,px_size,alpha,wavelength):
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
            Q = calculate_Q(alpha,delta_x,delta_y,dist,k_in,wavelength)
            y_values[index]=Q[2]
            x_values[index]=Q[1]

    return x_values,y_values,data

def calibrate_q_scale_missing_wedge(data,DB,dist,px_size,alpha,wavelength,delta=1):
    shape = (data.shape[0]+1,data.shape[1]+3)
    new_data = np.zeros((data.shape[0],data.shape[1]+2))
    x_values = np.zeros(shape)
    y_values = np.zeros(shape)
    k_in = (2*np.pi*np.cos(np.deg2rad(alpha))/wavelength,
            0,
            -2*np.pi*np.sin(np.deg2rad(alpha))/wavelength)

    x_db = int(DB[0])
    delta=delta
    for i in range(data.shape[0]):
        for j in range(data.shape[1]+1):
            index = (i, j)
            delta_y_down = (i - DB[1]) * px_size - px_size / 2

            delta_x_center = (j - DB[0]) * px_size
            delta_x_left = delta_x_center - px_size / 2
            delta_x_right = delta_x_center + px_size / 2

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
                if i < 606:
                    k_max = 2
                else:
                    k_max = 1
                for k in range(k_max):
                    delta_y = delta_y_down+k*px_size
                    Q_mW_left = calculate_Q(alpha,missing_wedge_x_left,delta_y,dist,k_in,wavelength)
                    Q_mW_right = calculate_Q(alpha,missing_wedge_x_right,delta_y,dist,k_in,wavelength)
                    Q = calculate_Q(alpha,delta_x_left,delta_y,dist,k_in,wavelength)
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
                    if i == 0:
                        print(Q)
                        print(Q_mW_left)
                        print(Q_mW_right)
                        print((x_k0,x_k1,x_k2))
                        print((y_k0,y_k1,y_k2))

            else:
                if delta_x_center > 0:
                    index = (i, j+2)
                try:
                    new_data[index]=data[(i,j)]
                except:
                    pass

                Q = calculate_Q(alpha,delta_x_left,delta_y_down,dist,k_in,wavelength)
                x_values[index]=np.sqrt(np.power(Q[0],2)+np.power(Q[1],2))*np.sign(Q[1])
                y_values[index]=Q[2]


    # plt.imshow(x_values)
    # plt.show()
    # plt.imshow(y_values)
    # plt.show()
    # plt.close("all")
    return x_values,y_values,new_data


def get_filepair(folder,start):
    files = np.array(range(2))+start
    files = [join(folder,f"latest_{i:07d}_craw.tiff") for i in files]
    return files

def merge_images(files):
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
                i1 = data1[index]
            if index[0] + delta_y >= 0:
                i2 = data2[(index[0]+delta_y,index[1])]

            if i1*i2 >= 0:
                x[...] = i1+i2
            else:
                x[...] = max(i1,i2)*2

            # x[...] = -1 if index[0] >= shape[0] else data1[index]
            # x[...] = -1 if index[0]+delta_y < 0 else data2[(index[0]+delta_y,index[1])]

    new_data = new_data[:round(DB1[1])+1,:]
    # plt.imshow(new_data, norm=LogNorm())
    # plt.show()
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

def get_qz_qy_cuts(qz,qy,data,DB,dqz=5,dqy=5,ax1=None,ax2=None):
    x_pos = int(DB[0])
    y_pos = int(DB[1])
    qz = qz[1:,:-1]
    qy = qy[1:,:-1]
    new_qy = qz[y_pos]
    new_qz = np.transpose(qy)[x_pos]

    data_qz =  np.transpose(data)[x_pos-int(dqy/2):x_pos+int(dqy/2)+int(dqy%2)][:]
    data_qy =  data[y_pos-int(dqz/2):y_pos+int(dqz/2)+int(dqz%2)][:]

    if type(ax1) != type(None):
        pass
    if type(ax1) != type(None):
        pass
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

def GIWAXS():
    plt.close("all")
    folder = r"Z:\Klaus\Data\Kooperationen\FAU Prof. R. Fink\VaxsterRawData"
    output_folder = r"Z:\Klaus\Data\Kooperationen\FAU Prof. R. Fink\Fabian Streller\C13BTBT\260709_GIWAXS_GISAXS_Evaluated"
    numbers = [100634,100644,100670]
    names = ["S15", "S25", "S13"]
    names = ["C8BTBTC8", "BTBT_nochain"]
    numbers = [100746,100748]
    # numbers = [100630,100640,100666]
    # numbers = [100634]
    dist1960 = 1592.042 #mm
    dist600 =  232.663 #mm
    dist550 = 189.065 #mm
    dist = dist600
    px_size = 0.172 #mm
    alpha = 0.2 #degree
    wavelength = 1.34012 # Angström
    for ctr,nbr in enumerate(numbers):
        files = get_filepair(folder,nbr)
        DB, data = merge_images(files)
        DB = (DB[0],DB[1]-8)
        x_values, y_values, qz_qy_data = calibrate_q_scale_qz_qy(data,DB,dist,px_size,alpha,wavelength)
        xpp_values, ypp_values, qpp_data = calibrate_q_scale_missing_wedge(data,DB,dist,px_size,alpha,wavelength)
        fig = plt.figure(figsize=(14,8),layout="constrained")
        subfigs = fig.subfigures(2,1,hspace=0.01,height_ratios=[3,1])
        axes_up = subfigs[0].subplots(1,3,width_ratios=[1,3,3],height_ratios=[1])
        a = 0.175
        b = 0.352
        c = 1-a-b
        axes_down = subfigs[1].subplots(1,3,width_ratios=[a,b,c])
        # ax1 = plt.subplot2grid((4,56),(0,0),colspan=8,rowspan=3)
        # ax2 = plt.subplot2grid((4,56),(3,11),rowspan=1,colspan=18)
        # ax_i1 = plt.subplot2grid((4,56),(0,7),rowspan=3,colspan=24)
        # ax_i2 = plt.subplot2grid((4,56),(0,31),rowspan=3,colspan=24)
        ax1 = axes_up[0]
        ax2 = axes_down[1]
        ax_i1 = axes_up[1]
        ax_i2 = axes_up[2]
        axes_down[0].axis('off')
        axes_down[2].axis('off')
        qz_cut, qy_cut, qz_cut_int, qy_cut_int,rect_z,rect_y = get_qz_qy_cuts(x_values,y_values,qz_qy_data,DB,10,10,ax1,ax2)

        xlim = (rect_z.xy[0],rect_y.xy[0]+rect_y.get_width())
        ylim = (rect_y.xy[1],rect_z.xy[1]+rect_z.get_height())
        plot_2d(ax_i1,x_values,y_values,qz_qy_data,colorbar=False,xlim=xlim,ylim=ylim)
        ax_i1.add_patch(rect_z)
        ax_i1.add_patch(rect_y)
        plot_2d(ax_i2,xpp_values,ypp_values,qpp_data,xlim=(xlim[0],xpp_values.max()),ylim=(ylim[0],ypp_values.max()))
        ax1.plot(qz_cut_int,qz_cut)
        ax1.set_xscale("log")
        ax1.set_ylim(ylim)
        ax1.set_xlim((qz_cut_int.min(),1e6))
        ax1.xaxis.set_inverted(True)
        ax2.plot(qy_cut,qy_cut_int)
        ax2.set_yscale("log")
        ax2.set_xlim(xlim)
        ax2.set_ylim((qz_cut_int.min(),1e3))
        ax2.yaxis.set_inverted(True)

        ax1.set_ylabel("Q${_z}$ [Å$^{-1}$]")
        ax1.set_xlabel("Intensity [arb. u.]")
        ax2.set_xlabel("Q${_y}$ [Å$^{-1}$]")
        ax2.set_ylabel("Intensity [arb. u.]")
        ax_i1.set_xlabel("Q${_y}$ [Å$^{-1}$]")
        ax_i1.set_ylabel("Q${_z}$ [Å$^{-1}$]")
        ax_i2.set_xlabel("Q${_\parallel}$ [Å$^{-1}$]")
        ax_i2.set_ylabel("Q${_\perp}$ [Å$^{-1}$]")

        fig.get_layout_engine().set(w_pad=2/72, h_pad=2/72,wspace=0.15,hspace=0.05)
        fig.suptitle(f"{nbr} {names[ctr]}")
        # plt.show()
        output_fn = f"{nbr}_overview_plots.png"
        plt.savefig(join(output_folder,output_fn))
        plt.close("all")
        output_excel = f"{nbr}_{names[ctr]}_"
        qz_low, qz_high, qy_low, qy_high, qzy_int = prepare_for_writing(y_values,x_values,qz_qy_data)
        qperp_low, qperp_high, qpar_low, qpar_high, qpp_int = prepare_for_writing(ypp_values,xpp_values,qpp_data)
        DataFrame({"Qz_low [1/AA]": qz_low, "Qz_high [1/AA]":qz_high,
                       "Qy_low [1/AA]": qy_low, "Qy_high [1/AA]":qy_high,
                       "Intensity [cts]": qzy_int}).to_csv(join(output_folder,f"{output_excel}_qzqy.csv"))
        DataFrame({"Qperp_low [1/AA]": qperp_low, "Qperp_high [1/AA]":qperp_high,
                       "Qpar_low [1/AA]": qpar_low, "Qpar_high [1/AA]":qpar_high,
                       "Intensity [cts]": qpp_int}).to_csv(join(output_folder,f"{output_excel}_qpp.csv"))
        DataFrame({"Qz [1/AA]": qz_cut, "Intensity [cts]": qz_cut_int}).to_csv(join(output_folder,f"{output_excel}_qzcut.csv"))
        DataFrame({"Qy [1/AA]": qy_cut, "Intensity [cts]": qy_cut_int}).to_csv(join(output_folder,f"{output_excel}_qycut.csv"))

def GISAXS():
    plt.close("all")
    folder = r"Z:\Klaus\Data\Kooperationen\FAU Prof. R. Fink\VaxsterRawData"
    output_folder = r"Z:\Klaus\Data\Kooperationen\FAU Prof. R. Fink\Fabian Streller\C13BTBT\260709_GIWAXS_GISAXS_Evaluated"
    names = ["C8BTBTC8", "BTBT_nochain"]
    numbers = [100744,100750]
    dist1960 = 1592.042 #mm
    dist600 =  232.663 #mm
    dist550 = 189.065 #mm
    dist = dist1960
    px_size = 0.172 #mm
    alpha = 0.2 #degree
    wavelength = 1.34012 # Angström
    for ctr,nbr in enumerate(numbers):
        for i in range(2):
            filename = join(folder, f"latest_{nbr + i:07d}_craw.tiff")
            if i==0:
                data = load_tiff(filename)
            else:
                data += load_tiff(filename)

        DB = find_DB(data)
        # DB = (DB[0],DB[1]-8)
        x_values, y_values, qz_qy_data = calibrate_q_scale_qz_qy(data,DB,dist,px_size,alpha,wavelength)
        xpp_values, ypp_values, qpp_data = calibrate_q_scale_missing_wedge(data,DB,dist,px_size,alpha,wavelength)
        fig = plt.figure(figsize=(14,8),layout="constrained")
        subfigs = fig.subfigures(2,1,hspace=0.01,height_ratios=[3,1])
        axes_up = subfigs[0].subplots(1,3,width_ratios=[1,3,3],height_ratios=[1])
        a = 0.175
        b = 0.352
        c = 1-a-b
        axes_down = subfigs[1].subplots(1,3,width_ratios=[a,b,c])
        # ax1 = plt.subplot2grid((4,56),(0,0),colspan=8,rowspan=3)
        # ax2 = plt.subplot2grid((4,56),(3,11),rowspan=1,colspan=18)
        # ax_i1 = plt.subplot2grid((4,56),(0,7),rowspan=3,colspan=24)
        # ax_i2 = plt.subplot2grid((4,56),(0,31),rowspan=3,colspan=24)
        ax1 = axes_up[0]
        ax2 = axes_down[1]
        ax_i1 = axes_up[1]
        ax_i2 = axes_up[2]
        axes_down[0].axis('off')
        axes_down[2].axis('off')
        qz_cut, qy_cut, qz_cut_int, qy_cut_int,rect_z,rect_y = get_qz_qy_cuts(x_values,y_values,qz_qy_data,DB,20,20,ax1,ax2)

        xlim = (rect_z.xy[0],rect_y.xy[0]+rect_y.get_width())
        ylim = (rect_y.xy[1],rect_z.xy[1]+rect_z.get_height())
        plot_2d(ax_i1,x_values,y_values,qz_qy_data,colorbar=False,xlim=xlim,ylim=ylim)
        ax_i1.add_patch(rect_z)
        ax_i1.add_patch(rect_y)
        plot_2d(ax_i2,xpp_values,ypp_values,qpp_data,xlim=(xlim[0],xpp_values.max()),ylim=(ylim[0],ypp_values.max()))
        ax1.plot(qz_cut_int,qz_cut)
        ax1.set_xscale("log")
        ax1.set_ylim(ylim)
        ax1.set_xlim((qz_cut_int.min(),1e9))
        ax1.xaxis.set_inverted(True)
        ax2.plot(qy_cut,qy_cut_int)
        ax2.set_yscale("log")
        ax2.set_xlim(xlim)
        ax2.set_ylim((qz_cut_int.min(),1e3))
        ax2.yaxis.set_inverted(True)

        ax1.set_ylabel("Q${_z}$ [Å$^{-1}$]")
        ax1.set_xlabel("Intensity [arb. u.]")
        ax2.set_xlabel("Q${_y}$ [Å$^{-1}$]")
        ax2.set_ylabel("Intensity [arb. u.]")
        ax_i1.set_xlabel("Q${_y}$ [Å$^{-1}$]")
        ax_i1.set_ylabel("Q${_z}$ [Å$^{-1}$]")
        ax_i2.set_xlabel("Q${_\parallel}$ [Å$^{-1}$]")
        ax_i2.set_ylabel("Q${_\perp}$ [Å$^{-1}$]")

        fig.get_layout_engine().set(w_pad=2/72, h_pad=2/72,wspace=0.15,hspace=0.05)
        fig.suptitle(f"{nbr} {names[ctr]}")
        # plt.show()
        output_fn = f"{nbr}_overview_plots.png"
        plt.savefig(join(output_folder,output_fn))
        plt.close("all")
        output_excel = f"{nbr}_{names[ctr]}_"
        qz_low, qz_high, qy_low, qy_high, qzy_int = prepare_for_writing(y_values,x_values,qz_qy_data)
        qperp_low, qperp_high, qpar_low, qpar_high, qpp_int = prepare_for_writing(ypp_values,xpp_values,qpp_data)
        DataFrame({"Qz_low [1/AA]": qz_low, "Qz_high [1/AA]":qz_high,
                   "Qy_low [1/AA]": qy_low, "Qy_high [1/AA]":qy_high,
                   "Intensity [cts]": qzy_int}).to_csv(join(output_folder,f"{output_excel}_qzqy.csv"))
        DataFrame({"Qperp_low [1/AA]": qperp_low, "Qperp_high [1/AA]":qperp_high,
                   "Qpar_low [1/AA]": qpar_low, "Qpar_high [1/AA]":qpar_high,
                   "Intensity [cts]": qpp_int}).to_csv(join(output_folder,f"{output_excel}_qpp.csv"))
        DataFrame({"Qz [1/AA]": qz_cut, "Intensity [cts]": qz_cut_int}).to_csv(join(output_folder,f"{output_excel}_qzcut.csv"))
        DataFrame({"Qy [1/AA]": qy_cut, "Intensity [cts]": qy_cut_int}).to_csv(join(output_folder,f"{output_excel}_qycut.csv"))

def main():
    GIWAXS()

if __name__=='__main__':
    main()