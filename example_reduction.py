import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from os.path import join
from data_handling import get_filepair, merge_images, calibrate_q_scale_qz_qy, calibrate_q_scale_missing_wedge
from data_handling import get_qz_qy_cuts, prepare_for_writing, plot_2d, create_folders, make_mp4
from local_variables import _OUTDIR,_DATADIR

def main():
    GIWAXS()

def GIWAXS():
    """"
    Example evaluation for GIWAXS data
    """

    #Define path structure
    rawdata_folder = _DATADIR
    output_folder = create_folders(_OUTDIR,specifier="")

    wavelength = 1.34012  # Angström
    px_size = 0.172  # mm pixel size

    numbers = np.arange(101190,101225,2)            #list of file numbers
    names = [f"{i}_alpha_{(((i-101190)/2)+1)*0.02}deg" for i in numbers] #list of sample names

    dist = 174.93466570943617 #mm sample detector distance
    alphas = [(((i-101190)/2)+1)*0.02 for i in numbers] #degree alpha incidence angle

    for ctr,nbr in enumerate(numbers):
        print(f"Processing image {nbr}\n")
        alpha = alphas[ctr]
        files = get_filepair(rawdata_folder,nbr)
        DB, data = merge_images(files)
        DB = (DB[0],DB[1]-8)
        x_values, y_values, qz_qy_data = calibrate_q_scale_qz_qy(data,DB,dist,px_size,alpha,wavelength)
        xpp_values, ypp_values, qpp_data = calibrate_q_scale_missing_wedge(data,DB,dist,px_size,alpha,wavelength)
        qz_cut, qy_cut, qz_cut_int, qy_cut_int, rect_z, rect_y = get_qz_qy_cuts(x_values, y_values, qz_qy_data,
                                                                                DB, 10,10)

        plot_GIS_dataset(rect_z, rect_y, x_values, y_values, qz_qy_data,
                         xpp_values, ypp_values, qpp_data, qz_cut_int, qz_cut,
                         qy_cut, qy_cut_int,nbr, names, ctr, output_folder)

        save_data(nbr, names, ctr, output_folder, y_values, x_values, qz_qy_data,
                  ypp_values, xpp_values, qpp_data, qz_cut, qz_cut_int, qy_cut, qy_cut_int)

    video_folder = output_folder
    make_mp4(output_folder, video_folder, numbers)

def save_data(nbr, names, ctr, output_folder, y_values,x_values,qz_qy_data,
              ypp_values,xpp_values,qpp_data, qz_cut, qz_cut_int, qy_cut, qy_cut_int):
    output_excel = f"{nbr}_{names[ctr]}_"
    qz_low, qz_high, qy_low, qy_high, qzy_int = prepare_for_writing(y_values, x_values, qz_qy_data)
    qperp_low, qperp_high, qpar_low, qpar_high, qpp_int = prepare_for_writing(ypp_values, xpp_values, qpp_data)
    DataFrame({"Qz_low [1/AA]": qz_low, "Qz_high [1/AA]": qz_high,
               "Qy_low [1/AA]": qy_low, "Qy_high [1/AA]": qy_high,
               "Intensity [cts]": qzy_int}).to_csv(join(output_folder, f"{output_excel}_qzqy.csv"))
    DataFrame({"Qperp_low [1/AA]": qperp_low, "Qperp_high [1/AA]": qperp_high,
               "Qpar_low [1/AA]": qpar_low, "Qpar_high [1/AA]": qpar_high,
               "Intensity [cts]": qpp_int}).to_csv(join(output_folder, f"{output_excel}_qpp.csv"))
    DataFrame({"Qz [1/AA]": qz_cut, "Intensity [cts]": qz_cut_int}).to_csv(
        join(output_folder, f"{output_excel}_qzcut.csv"))
    DataFrame({"Qy [1/AA]": qy_cut, "Intensity [cts]": qy_cut_int}).to_csv(
        join(output_folder, f"{output_excel}_qycut.csv"))

def plot_GIS_dataset(rect_z, rect_y, x_values, y_values, qz_qy_data,
                     xpp_values, ypp_values, qpp_data, qz_cut_int, qz_cut,
                     qy_cut, qy_cut_int,nbr, names, ctr, output_folder):
    plt.close("all")
    fig = plt.figure(figsize=(14,8),layout="constrained")
    subfigs = fig.subfigures(2,1,hspace=0.01,height_ratios=[3,1])
    axes_up = subfigs[0].subplots(1,3,width_ratios=[1,3,3],height_ratios=[1])
    a = 0.175
    b = 0.352
    c = 1-a-b
    axes_down = subfigs[1].subplots(1,3,width_ratios=[a,b,c])
    ax1 = axes_up[0]
    ax2 = axes_down[1]
    ax_i1 = axes_up[1]
    ax_i2 = axes_up[2]
    axes_down[0].axis('off')
    axes_down[2].axis('off')


    xlim = (rect_z.xy[0],rect_y.xy[0]+rect_y.get_width())
    ylim = (rect_y.xy[1],rect_z.xy[1]+rect_z.get_height())
    plot_2d(ax_i1,x_values,y_values,qz_qy_data,colorbar=False,xlim=xlim,ylim=ylim)
    ax_i1.add_patch(rect_z)
    ax_i1.add_patch(rect_y)
    plot_2d(ax_i2,xpp_values,ypp_values,qpp_data,xlim=(xlim[0],xpp_values.max()),ylim=(ylim[0],ypp_values.max()))
    ax1.plot(qz_cut_int,qz_cut)
    ax1.set_xscale("log")
    ax1.set_ylim(ylim)
    ax1.set_xlim((max(qz_cut_int.min()*0.95,1e0),min(qz_cut_int.max(),1e4)))
    ax1.xaxis.set_inverted(True)
    ax2.plot(qy_cut,qy_cut_int)
    ax2.set_yscale("log")
    ax2.set_xlim(xlim)
    ax2.set_ylim((max(qy_cut_int.min()*0.95,1e0),min(qy_cut_int.max(),1e3)))
    ax2.yaxis.set_inverted(True)

    ax1.set_ylabel(r"Q${_z}$ [Å$^{-1}$]")
    ax1.set_xlabel(r"Intensity [arb. u.]")
    ax2.set_xlabel(r"Q${_y}$ [Å$^{-1}$]")
    ax2.set_ylabel(r"Intensity [arb. u.]")
    ax_i1.set_xlabel(r"Q${_y}$ [Å$^{-1}$]")
    ax_i1.set_ylabel(r"Q${_z}$ [Å$^{-1}$]")
    ax_i2.set_xlabel(r"Q${_\parallel}$ [Å$^{-1}$]")
    ax_i2.set_ylabel(r"Q${_\perp}$ [Å$^{-1}$]")

    fig.get_layout_engine().set(w_pad=2/72, h_pad=2/72,wspace=0.15,hspace=0.05)
    fig.suptitle(f"{nbr} {names[ctr]}")
    # plt.show()
    output_fn = f"{nbr}_overview_plots.png"
    plt.savefig(join(output_folder,output_fn))
    plt.close("all")

if __name__=='__main__':
    main()