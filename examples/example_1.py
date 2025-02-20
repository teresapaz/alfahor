from alfahor import alfahor

#This example uses data from MAPS Large program. To obtain the fits files please go to their repository. The masks are found in the indicated folder

PA = -133.3 #sign to ensure that bottom disk side is towards the south in t.plot_interactive()
inc = 46.7
dist = 101. #in pc
vsys = 5.8 #in km/s
mstar = 2.0

fits_file = 'HD_163296_CO_220GHz.robust_0.5.JvMcorr.image.fits'
folder_masks = 'hd163296_masks/12CO_masks'

#if you have a folder with your prior masks, use_folder_mask = True
#change the sign of PA so that when using plot_interactive() the bottom side is to the south
#if you do not have the masks, you can use ang_limit=NUMBER , where NUMBER is the extent of the image in arcsec. Default is 5.0

try:
    t = alfahor(fits_file, PA, inc, dist, vsys, mstar, folder_masks, use_folder_masks=True)
except FileNotFoundError:
    import subprocess
    _ = subprocess.run(["bash", "download_MAPS.sh"])
    t = alfahor(fits_file, PA, inc, dist, vsys, mstar, folder_masks, use_folder_masks=True)

#if you don't have the masks first create them by using t.plot_interactive()
#when running calculate_vertical_structure a file is saved containing the information on radial and height position, as well as the input parameters for the calculations

t.calculate_vertical_structure()

#plot_height_prof allows you to check the vertical profile, but does not save the figures

t.plot_height_prof()

#check_plot shows the masks and location of emission maxima. It is set to do a 5x6 grid with 30 channels around the center channel (at the systemic velocity).  You can change the imaging parameters and the amount of channels by changing nrows or ncols.

t.check_plot(true_PA = True, ang_extent=3.0)
