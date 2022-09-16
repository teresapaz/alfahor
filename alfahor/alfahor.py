# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Cursor, Slider
from astropy.io import fits
from scipy import ndimage
import matplotlib.gridspec as gridspec
import math
import matplotlib.colors as colors
from matplotlib.patches import Circle
import glob
import scipy.constants as sc
import cv2
import os

# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------
def main():
    alfahor()


# --------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------

class alfahor(object):

    """
    ``alfahor`` is a class that reads a spectral cube (FITS file format) and handles
    channel masks that can be created interactively to obtain the vertical structure of
    a certain molecule.
    """

    cid = None
    cidmotion=None
    cidsave = None
    coords_far = []
    lines_far = []
    coords_near = []
    lines_near = []

    def __init__(self, fits_file, pa, inc, dist, vsys, folder, ang_limit=5.0, use_folder_masks=False, dx=0, dy=0):
        """
        Initialise alfahor

        Parameters
        ----------
        fits_file (str) : Path to the fits file.

        pa (float) : position angle of source [degrees]. Can be positive or negative, sign
        must be defined such that the lower side of the emission is towards the south once
        the system has been rotated.

        inc (float) : inclination angle of source [degrees].

        dist (float) : distance of source [parsec].

        vsys (float) : system velocity [km/s].

        folder (string) : folder where masks are or will be saved.

        ang_limit (Optional[float]) : If specified, clip data to a square field of view with
        sides given by 'ang_limit' [arcsec]. Default is 5.0.

        use_folder_masks (Optional[bool]) : If masks have already been saved for this source
        in 'folder' set to ''True''. This will ensure the data size is equal to the previous
        mask size. Default assumes no previous masks exist ''False''.

        dx (Optional[float]) : in case the data is not centered, 'dx' allows to shift the data
        in the horizontal axis [pixel]. Default is 0, data is assumed to be centered.

        dy (Optional[float]) : in case the data is not centered, 'dx' allows to shift the data
        in the vertical axis [pixel]. Default is 0, data is assumed to be centered.
        """

        open_fits = fits.open(fits_file)
        self.data_all = np.squeeze(np.squeeze(open_fits[0].data))
        self.len = len(self.data_all[0])
        self.center = self.len/2.
        self.pix_scale = np.round(abs(open_fits[0].header['CDELT1'])*3600,6)
        self.bmaj = np.radians(open_fits[0].header['bmaj']) * 206265
        self.bmin = np.radians(open_fits[0].header['bmin']) * 206265
        self.unit_axis3 = open_fits[0].header['CUNIT3']
        if self.unit_axis3 == 'Hz':
            self.freq_0 = open_fits[0].header['RESTFRQ']
            self.freq_init = open_fits[0].header['CRVAL3']
            self.freq_delta = open_fits[0].header['CDELT3']
        if self.unit_axis3 == 'm/s':

            self.vel_init = open_fits[0].header['CRVAL3']
            self.vel_delta = open_fits[0].header['CDELT3']

        self.vsys = vsys*1e3

        if not os.path.isdir(folder):
            print ('Creating folder %s'%folder)
            os.system('mkdir %s'%folder)

        if not use_folder_masks:
            self.limits = ang_limit/self.pix_scale

        if use_folder_masks:
            masks = glob.glob(folder + '/mask_*')
            self.limits = len(np.load(masks[0], allow_pickle=True, encoding='latin1')[0])/2.


        if self.limits > self.center:
            self.limits = self.center


        self.data_all = self.data_all[:, int(self.center- self.limits):int(self.center+self.limits), int(self.center-self.limits):int(self.center+self.limits)]
        self.data_all = np.roll(self.data_all, int(dx), axis=2)
        self.data_all = np.roll(self.data_all, int(dy), axis=1)

        self.len = len(self.data_all[0])
        self.center = self.len/2.

        self.PA = pa #here + or - to get bottom surface in south
        self.inc = inc
        self.dist = dist
        self.fits_file = fits_file

        self.chan = 0
        self.folder = folder
        self.pa_sign = pa/abs(pa)
        self.dx = dx
        self.dy = dy

        if self.pa_sign>0:
            self.angle_for_rot = self.PA - 90
        if self.pa_sign<0:
            self.angle_for_rot = -self.PA + 90  #negative PA because the angle is negative!


        self.x_near = None
        self.x_far = None
        self.y_near = None
        self.y_far = None

        self.vert_struct_r = None
        self.vert_struct_h = None

        self.im_power = 1


    def plot_interactive(self):

        """
        Open plotting interface to check for data rotation and create interactive masks.
        """

        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(2,3,(2,6))
        self.fig.subplots_adjust(bottom=0.25)

        self.axbutton_far = self.fig.add_subplot(5,3,1)
        self.axbutton_near = self.fig.add_subplot(5,3,7)
        self.axbutton_delete = self.fig.add_subplot(5,3,13)

        self.axbutton_far_save = self.fig.add_subplot(8,6,13)
        self.axbutton_near_save = self.fig.add_subplot(8,6,31)

        self.ax_chan_slider = plt.axes([0.25, 0.15, 0.65, 0.03])

        self.ax_power_im_slider = plt.axes([0.25, 0.1, 0.65, 0.03])


        self.chan_slider = Slider(self.ax_chan_slider, "Channel", 0, len(self.data_all)+1,
                                  valinit=self.chan,
                                  valstep=1,
                                  #valstep=np.arange(0, len(self.data_all)+1),
                                  color="green",
                                  #initcolor='none' #--> Available in mpl>v3.5 only
        )

        self.power_im_slider = Slider(self.ax_power_im_slider, "Scale Image", 0, 2,
                                      valinit=self.im_power,
                                      valstep=0.1,
                                      #valstep=np.linspace(0,2,21),
                                      color="blue",
                                      #initcolor='none'
        )


        self.chan_slider.on_changed(self.update_chan_slider)

        self.power_im_slider.on_changed(self.update_power_im_slider)

        data_chan = self.data_all[self.chan]
        rot_im = self.rotate_channel_images(data_chan)
        self.chan_image = self.ax.imshow(rot_im, origin='lower', cmap='jet', norm=colors.PowerNorm(gamma=self.im_power))

        #self.ax.add_patch(Circle((self.center, self.center), radius=3.5, color='white'))
        self.ax.tick_params(bottom = False, left = False, labelbottom = False, labelleft = False)

        self.ln=0
        self.color = 'yellow'

        button_far = Button(self.axbutton_far, 'Far Side', color = 'yellow')
        button_far_save = Button(self.axbutton_far_save, 'Save', color = 'yellow')


        button_far.on_clicked(self.button_action_far)
        button_far_save.on_clicked(self.button_save_masks_far)


        button_near = Button(self.axbutton_near, 'Near Side', color = 'magenta')
        button_near_save = Button(self.axbutton_near_save, 'Save', color = 'magenta')

        button_near.on_clicked(self.button_action_near)
        button_near_save.on_clicked(self.button_save_masks_near)

        button_delete = Button(self.axbutton_delete, 'Delete', color = 'grey')
        button_delete.on_clicked(self.button_delete)

        self.cursor = Cursor(self.ax, linewidth=0)
        self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.onmove)


        plt.show()

    def button_save_masks_far(self, event):

        """
        Button action for saving far side mask
        """

        if self.cidsave == None:
            self.side = 'far'
            self.cidsave = self.fig.canvas.mpl_connect('button_press_event', self.save_masks)

    def button_save_masks_near(self, event):

        """
        Button action for saving near side mask
        """

        if self.cidsave == None:
            self.side = 'near'
            self.cidsave = self.fig.canvas.mpl_connect('button_press_event', self.save_masks)

    def button_delete(self, event):

        """
        Button action for deleting all masks
        """

        self.fig.clf()
        self.plot_interactive()
        self.coords_near = []
        self.coords_far = []

    def save_masks(self, event):

        """
        Button action for saving masks
        """

        if(self.cidsave is None):
            return

        if self.side == 'near':
            coords = self.coords_near
            lines = self.lines_near

        if self.side == 'far':
            coords = self.coords_far
            lines = self.lines_far


        pix_coord = coords
        a3 = np.array( [pix_coord], dtype=np.int32 )
        mask_chan = np.zeros([self.len, self.len],dtype=np.uint8)
        cv2.fillPoly(mask_chan, a3, True )

        if self.side == 'near':
            self.mask_near = mask_chan
            np.save(self.folder + '/mask_near_chan_' + str(self.chan), np.array([self.mask_near]+ [self.chan], dtype=object))

        if self.side == 'far':
            self.mask_far = mask_chan
            np.save(self.folder + '/mask_far_chan_' + str(self.chan), np.array([self.mask_far]+ [self.chan], dtype=object))

        self.cidsave = None

    def onclick_makemask(self, event):

        """
        Button action for making masks
        """

        if(self.cid is None):
            return

        ix, iy = event.xdata, event.ydata

        if self.side == 'near':

            if(len(self.coords_near) > 1 and (ix, iy) == self.coords_near[-1]):
                self.coords_near.append(self.coords_near[0])
                self.fig.canvas.mpl_disconnect(self.cid)
                self.cid = None
                self.cidmotion = None

            else:
                if(len(self.coords_near) > 1000):
                    raise ValueError('Too many coordinates were selected.')

                self.coords_near.append((ix, iy))

            if(len(self.coords_near) > 1):
                xy = np.array(self.coords_near[-2:])
                new_line, = self.ax.plot(xy[:,0], xy[:,1], color=self.color)
                self.lines_near.append(new_line)
                self.fig.canvas.draw()

        if self.side == 'far':

            if(len(self.coords_far) > 1 and (ix, iy) == self.coords_far[-1]):
                self.coords_far.append(self.coords_far[0])
                self.fig.canvas.mpl_disconnect(self.cid)
                self.cid = None
                self.cidmotion = None

            else:
                if(len(self.coords_far) > 1000):
                    raise ValueError('Too many coordinates were selected.')

                self.coords_far.append((ix, iy))

            if(len(self.coords_far) > 1):
                xy = np.array(self.coords_far[-2:])
                new_line, = self.ax.plot(xy[:,0], xy[:,1], color=self.color)
                self.lines_far.append(new_line)
                self.fig.canvas.draw()

    def onmove(self, event):

        """
        Cursor action for drawing masks
        """

        if(self.cid is None):
            return

        if self.side == 'near':
            coords = self.coords_near
            lines = self.lines_near
        if self.side == 'far':
            coords = self.coords_far
            lines = self.lines_far

        if (len(coords)>=1):
            ix, iy = event.xdata, event.ydata
            if self.ln!=0:
                self.ln.remove()
            self.ln, = self.ax.plot([coords[-1:][0][0], ix], [coords[-1:][0][1], iy], color=self.color)
            self.fig.canvas.draw()

    def button_action_near(self, event):

        """
        Button action for making near side mask
        """
        if(self.cid is None):
            self.color = 'magenta'
            self.ln=0
            self.coords_near = []
            self.lines_near = []
            self.side = 'near'
            self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick_makemask)
        else:
            pass

    def button_action_far(self, event):

        """
        Button action for making far side mask
        """
        if(self.cid is None):
            self.color = 'yellow'
            self.ln=0
            self.coords_far = []
            self.lines_far = []
            self.side = 'far'
            self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick_makemask)
        else:
            pass

    def update_chan_slider(self, chan):

        """
        Channel slider
        """

        data_chan = self.data_all[chan]
        rot_im = self.rotate_channel_images(data_chan)
        self.chan_image = self.ax.imshow(rot_im, zorder=1, origin='lower', cmap='jet', norm=colors.PowerNorm(gamma=self.im_power))
        self.fig.canvas.draw()
        self.chan = chan

    def update_power_im_slider(self, power):

        """
        Slider to regulate plotting color scale.
        """

        self.im_power = power
        data_chan = self.data_all[self.chan]
        rot_im = self.rotate_channel_images(data_chan)
        self.chan_image = self.ax.imshow(rot_im, zorder=1, origin='lower', cmap='jet', norm=colors.PowerNorm(gamma=self.im_power))
        self.fig.canvas.draw()

    def rotate_channel_images(self, data_chan):

        """
        Rotate data, important to define position angle sign to get bottom surface
        towards the south.

        Args:
          data_chan (array): channel emission to be rotated.

        Returns:
          rotated data (array)
        """

        return ndimage.rotate(data_chan, (self.angle_for_rot), reshape=False)

    def channel_velocity(self, chan):

        """
        Obtain velocity of a certain channel.

        Args:
          chan (float): channel number.

        Returns:
          vchan (float): channel velocity.
        """
        if self.unit_axis3 == 'Hz':
            vchan = sc.c*(1 - ((self.freq_init + chan*self.freq_delta)/self.freq_0))

            v_init = sc.c*(1 - ((self.freq_init + self.freq_delta)/self.freq_0))
            v_end = sc.c*(1 - ((self.freq_init + len(self.data_all)*self.freq_delta)/self.freq_0))

        if self.unit_axis3 == 'm/s':
            vchan = (self.vel_init + chan*self.vel_delta)
            v_init = self.vel_init
            v_end = (self.vel_init + len(self.data_all)*self.vel_delta )

        if v_init>v_end:
            return -(vchan - self.vsys)

        if v_init<v_end:
            return vchan - self.vsys

    def get_center_chan_info(self):

        """
        Obtains the number of the channel closest to the systemic velocity and the maximum
        emission flux in that channel.

        Returns:
          ctrl_chan (float): channel number of emission closest to the systemic velocity.
          vmax (float): maximum emission value in ctrl_chan [Jy/beam]
        """
        if self.unit_axis3 == 'Hz':
            ctrl_chan = int(np.round( (self.freq_0 * (1- self.vsys/sc.c) - self.freq_init) / self.freq_delta))

        if self.unit_axis3 == 'm/s':
            ctrl_chan = int(np.round((self.vsys - self.vel_init)/self.vel_delta))

        vmax = max(np.concatenate(self.data_all[ctrl_chan]))
        return ctrl_chan, vmax

    def tracing_maxima(self):

        """
        Determines the location of emission maxima in a rotated channel within the far
        and near side masks using cartesian coordinates.

        Returns:
          x_near_max_all (list): positions of maxima along x axis, for near side emission
          y_near_max_all (list): positions of maxima along y axis, for near side emission
          x_far_max_all (list): positions of maxima along x axis, for far side emission
          y_far_max_all (list): positions of maxima along y axis, for far side emission
        """

        x_far_max_all = []
        x_near_max_all = []
        y_far_max_all = []
        y_near_max_all = []

        x_dir = np.arange(0, self.len)
        XX = np.meshgrid(x_dir,x_dir)[0]

        chan_array = self.get_mask_array()
        ctrl_chan, vmax = self.get_center_chan_info()
        for chan in chan_array:


            vchan = self.channel_velocity(chan)
            if (4.0/self.pix_scale) >= self.limits: #going up to 4 arcseconds from center or less if image is smaller
                max_x = (self.limits) -1


            if (4.0/self.pix_scale) < self.limits:
                max_x = 4.0/self.pix_scale


            sample_per_beam = self.bmaj / 4. / self.pix_scale #sampling 4 times within beam
            amount_points_x = int(max_x / sample_per_beam)

            x_far_max = []
            x_near_max = []
            y_far_max = []
            y_near_max = []

            chan_data = self.rotate_channel_images(self.data_all[chan])
            mask_chan_far = np.load(self.folder + '/mask_far_chan_'+str(chan)+'.npy', allow_pickle=True, encoding='latin1')[0]
            mask_chan_near = np.load(self.folder + '/mask_near_chan_'+str(chan)+'.npy', allow_pickle=True, encoding='latin1')[0]

            #depends on where it opens
            direction = np.mean(XX[mask_chan_far==1])

            if direction>=self.center:
                array_x = np.linspace(self.center, int(self.center+max_x), amount_points_x)
            if direction<self.center:
                array_x = np.linspace(int(self.center-max_x), self.center, amount_points_x)


            for x in array_x:
                y_far_value = []
                y_near_value = []
                y_far_pos = []
                y_near_pos = []

                #array_y = np.linspace(0,  int(2*self.center)-1, int(2*self.center))
                array_y = np.linspace(int(self.center-max_x), int(self.center+max_x), int(3*max_x))
                for y_f in array_y:

                    mask_value = mask_chan_far[int(y_f), int(x)]

                    if mask_value==1:
                        y_far_value.append(chan_data[int(y_f), int(x)])
                        y_far_pos.append(y_f)
                        #self.data_all[chan][int(y_f), int(x)] = 100

                if len(y_far_value)!=0:
                    index_max = y_far_value.index(max(y_far_value))
                    y_far_max.append(y_far_pos[index_max])
                    x_far_max.append(x)


                for y_n in array_y:

                    mask_value = mask_chan_near[int(y_n), int(x)]

                    if mask_value==1:
                        y_near_value.append(chan_data[int(y_n), int(x)])
                        y_near_pos.append(y_n)
                        #self.data_all[chan][int(y_n), int(x)] = 100

                if len(y_near_value)!=0:
                    index_max = y_near_value.index(max(y_near_value))
                    y_near_max.append(y_near_pos[index_max])
                    x_near_max.append(x)


            x_far_max_all.append(x_far_max)
            x_near_max_all.append(x_near_max)
            y_far_max_all.append(y_far_max)
            y_near_max_all.append(y_near_max)

        self.x_near = x_near_max_all
        self.x_far = x_far_max_all
        self.y_near = y_near_max_all
        self.y_far = y_far_max_all

        return x_near_max_all, y_near_max_all, x_far_max_all, y_far_max_all

    def check_plot(self, true_PA, ang_extent=0., all_chans=[], vmax=0, vmin=0, gamma=1.0, ncols=6, nrows=5, put_masks=True, put_dots=True):

        """
        Creates and saves a plot with channel maps, overlaying if specified the used masks and
        maxima location.

        Args:
          true_PA (bool): ''True'' indicates the channels must be imaged as observed in the
          original data ''False'' indicates the channels are imaged rotated, such that the
          bottom side is towards the south as used in the analysis.
          ang_extent (Optional[float])
          all_chans (Optional[list])
          vmax (Optional[float])
          vmin (Optional[float])
          gamma (Optional[float])
          ncols (Optional[int])
          nrows (Optional[int])
          put_masks (Optional[bool])
          put_dots (Optional[bool])

        Returns:
          Saves pdf channel grid.
        """
        if vmax==0:
            ctrl_chan, vmax = self.get_center_chan_info()
        if all_chans==[]:
            ctrl_chan, vmax_0 = self.get_center_chan_info()
            from_ctr = int(np.round(ncols*nrows/2))
            all_chans = np.arange(ctrl_chan-from_ctr, ctrl_chan+from_ctr)
            if len(all_chans)> (ncols*nrows):
                all_chans = np.arange(ctrl_chan-from_ctr, ctrl_chan+from_ctr-1)


        chan_array = self.get_mask_array()

        if self.x_near == None:
            self.tracing_maxima()


        fig = plt.figure(figsize=(ncols*2, nrows*2))
        spec = gridspec.GridSpec(ncols= ncols, nrows = nrows , figure =fig, wspace = 0.01, hspace=0.01)

        first = 0
        row = 0

        for channel in range(len(all_chans)):
            if channel!=0:
                if math.floor(channel/ncols)==first:
                    row+=1
                if math.floor(channel/ncols)!=first:
                    first = math.floor(channel/ncols)
                    row=0

            ax = fig.add_subplot(spec[int(math.floor(channel/ncols)), row ])

            if not true_PA:
                img_plot = self.rotate_channel_images(self.data_all[all_chans[channel]])

            if true_PA:
                img_plot = self.data_all[all_chans[channel]]

            ax.imshow(img_plot, origin = 'lower', cmap = 'magma',
                        norm=colors.PowerNorm(gamma=gamma, vmax=vmax, vmin=vmin))

            if put_masks:
                if all_chans[channel] in chan_array:
                    mask_far_plot = np.load(self.folder + '/mask_far_chan_'+str(all_chans[channel])+'.npy', allow_pickle=True, encoding='latin1')[0]
                    mask_near_plot = np.load(self.folder + '/mask_near_chan_'+str(all_chans[channel])+'.npy', allow_pickle=True, encoding='latin1')[0]
                    if not true_PA:
                        ax.contour(mask_far_plot, origin = 'lower', colors = ['white'], linewidths = [0.5])
                        ax.contour(mask_near_plot, origin = 'lower', colors = ['cyan'], linewidths = [0.5])
                    if true_PA:
                        mask_far_plot = (np.round(ndimage.rotate(mask_far_plot, -self.angle_for_rot, reshape=False)))
                        mask_near_plot =  (np.round(ndimage.rotate(mask_near_plot, -self.angle_for_rot, reshape=False)))
                        ax.contour(mask_far_plot , origin = 'lower', colors = ['white'], linewidths = [0.5])
                        ax.contour( mask_near_plot, origin = 'lower', colors = ['cyan'], linewidths = [0.5])

            if put_dots:
                if all_chans[channel] in chan_array:
                    chan = list(chan_array).index(all_chans[channel])
                    x_far_max = np.array(self.x_far[chan])
                    x_near_max = np.array(self.x_near[chan])
                    y_far_max = np.array(self.y_far[chan])
                    y_near_max = np.array(self.y_near[chan])

                    patches_near = []
                    patches_far = []
                    if not true_PA:
                        for i in range(len(x_near_max)):
                            patches_near.append(Circle((x_near_max[i], y_near_max[i]), radius=1., color='cyan'))

                        for i in range(len(x_far_max)):
                            patches_far.append(Circle((x_far_max[i], y_far_max[i]), radius=1., color='white'))

                        patches_far.append(Circle((self.center, self.center), radius=3., color='yellow'))
                        patches_far.append(Circle((self.center, self.center), radius=3., color='yellow'))

                    if true_PA:

                        angle = np.radians(self.angle_for_rot)

                        x_far_max += -self.center
                        x_near_max += -self.center
                        y_far_max += -self.center
                        y_near_max += -self.center

                        x_far_rot = x_far_max*np.cos(angle) - y_far_max*np.sin(angle) + self.center
                        y_far_rot = x_far_max*np.sin(angle) + y_far_max*np.cos(angle) + self.center

                        x_near_rot = x_near_max*np.cos(angle) - y_near_max*np.sin(angle) + self.center
                        y_near_rot = x_near_max*np.sin(angle) + y_near_max*np.cos(angle) + self.center

                        for i in range(len(x_near_max)):
                            patches_near.append(Circle((x_near_rot[i], y_near_rot[i]), radius=1.5, color='cyan'))

                        for i in range(len(x_far_max)):
                            patches_far.append(Circle((x_far_rot[i], y_far_rot[i]), radius=1.5, color='white'))

                        patches_far.append(Circle((self.center, self.center), radius=3.5, color='yellow'))
                        patches_far.append(Circle((self.center, self.center), radius=3.5, color='yellow'))

                    for p in patches_far+patches_near:
                        ax.add_patch(p)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            if ang_extent != 0:
                region = ang_extent/self.pix_scale
                if region <= self.center:
                    ax.set_xlim(self.center - region, self.center + region)
                    ax.set_ylim(self.center - region, self.center + region)

        if true_PA:
            plt.savefig(self.folder +'/channel_maps.pdf')

        if not true_PA:
            plt.savefig(self.folder +'/channel_maps_rotated.pdf')

        plt.show()

    def calculate_vertical_structure(self):

        """
        Obtains vertical structure values (height of emission at a given radius) based on
        Pinte et al. 2018. Values are saved in a txt file, together with the relevant
        parameters used for the calculations.
        """

        chan_array = self.get_mask_array()
        if self.x_near == None:
            self.tracing_maxima()
        pix_to_au = self.pix_scale * self.dist


        r_per_chan = []
        h_per_chan = []
        v_per_chan = []

        r_all_values = []
        h_all_values = []
        v_all_values = []

        for num in range(len(chan_array)):

            x_far_max = self.x_far[num]
            x_near_max = self.x_near[num]
            y_far_max = self.y_far[num]
            y_near_max = self.y_near[num] #this is all in pixels

            chan = chan_array[num]

            r_chan = []
            h_chan = []


            for x_t in x_far_max:
                if x_t in x_near_max:

                    index_far = x_far_max.index(x_t)
                    index_near = x_near_max.index(x_t)

                    y_far = y_far_max[index_far]
                    y_near = y_near_max[index_near]

                    y_c = (y_far + y_near)/2.
                    r_x = (x_t-self.center)
                    r_y = (y_far - y_c)/(np.cos(np.radians(self.inc)))
                    r = np.sqrt((r_x**2) + (r_y**2))
                    h = -(self.center- y_c)/np.sin(np.radians(self.inc))

                    if h>=0 :
                        r_chan.append(r * pix_to_au)
                        h_chan.append(h * pix_to_au)


            if len(r_chan)==0:
                new_r_chan, new_h_chan =  [], []
            #this is to order the values per channel
            if len(r_chan)!=0:
                new_r_chan, new_h_chan= (list(t) for t in zip(*sorted(zip(r_chan, h_chan))))

            r_per_chan.append(new_r_chan)
            h_per_chan.append(new_h_chan)

            #here all the values from all channels in single list
            r_all_values += r_chan
            h_all_values += h_chan

        new_r, new_h = (list(t) for t in zip(*sorted(zip(r_all_values, h_all_values))))
        self.vert_struct_r = np.array(new_r)
        self.vert_struct_h = np.array(new_h)

        if self.dx==0 and self.dy==0:
            np.savetxt(self.folder + '/vertical_structure_values.txt',
                        np.c_[self.vert_struct_r, self.vert_struct_h],
                        delimiter = ' ',
                        header = 'data from file ' + self.fits_file + '\nparameters used:\nPA: ' + str(self.PA) + '\ninc: ' + str(self.inc) + '\ndist: ' + str(self.dist) + ' \nvsys: ' + str(self.vsys) + '\nr[au] h[au]')
        if self.dx!=0 or self.dy!=0:
            np.savetxt(self.folder + '/vertical_structure_values.txt',
                        np.c_[self.vert_struct_r, self.vert_struct_h],
                        delimiter = ' ',
                        header = 'data from file ' + self.fits_file + '\nparameters used:\nPA: ' + str(self.PA) + '\ninc: ' + str(self.inc) + '\ndist: ' + str(self.dist) + ' \nvsys: ' + str(self.vsys) +  ' \ndx: ' + str(self.dx) + ', dy: ' + str(self.dy) +'\nr[au] h[au]')


        self.bin_values()

    def plot_height_prof(self):

        """
        Plots vertical profile.
        """

        if self.vert_struct_r is None:
            raise ValueError('Do calculate_vertical_structure()')
            return
        r_plot_lines = np.arange(1.1*(max(self.vert_struct_r)))

        plt.figure(1)
        plt.plot(self.vert_struct_r, self.vert_struct_h, '.')
        plt.plot(r_plot_lines, 0.1*r_plot_lines, '-', color='grey', linewidth=0.7, alpha=0.5)
        plt.plot(r_plot_lines, 0.3*r_plot_lines, '-', color='grey', linewidth=0.7, alpha=0.5)
        plt.plot(r_plot_lines, 0.5*r_plot_lines, '-', color='grey', linewidth=0.7, alpha=0.5)

        plt.xlabel('Radial Distance [au]', fontsize=13)
        plt.ylabel('Height [au]', fontsize=13)

        plt.xlim(0, 1.05*max(self.vert_struct_r))
        plt.ylim(0, 1.05*max(self.vert_struct_h))

        plt.figure(2)

        plt.fill_between(self.vert_r_bin, self.vert_h_bin-self.vert_h_bin_std , self.vert_h_bin+self.vert_h_bin_std, color='blue', alpha=0.1)
        plt.plot(self.vert_r_bin, self.vert_h_bin, '-', color='blue')


        plt.plot(r_plot_lines, 0.1*r_plot_lines, '-', color='grey', linewidth=0.7, alpha=0.5)
        plt.plot(r_plot_lines, 0.3*r_plot_lines, '-', color='grey', linewidth=0.7, alpha=0.5)
        plt.plot(r_plot_lines, 0.5*r_plot_lines, '-', color='grey', linewidth=0.7, alpha=0.5)

        plt.xlabel('Radial Distance [au]', fontsize=13)
        plt.ylabel('Height [au]', fontsize=13)
        plt.xlim(0, 1.05*max(self.vert_struct_r))
        plt.ylim(0, 1.05*max(self.vert_struct_h))
        plt.show()

    def bin_values(self):
        """
        Bins vertical profile values considering radial bins as wide as the semi-major beam axis.
        """
        bin_size = self.bmin * self.dist
        r_bin = np.arange(0, 1.2*(max(self.vert_struct_r)), bin_size)

        r_bin_low = r_bin[:-1]
        r_bin_up = r_bin[1:]

        r_bin_val = []
        h_bin_val = []
        r_bin_std = []
        h_bin_std = []
        for i in np.arange(len(r_bin_up)):
            r_low = r_bin_low[i]
            r_up = r_bin_up[i]

            if any(self.vert_struct_r>=r_low) and any(self.vert_struct_r<r_up):
                all_r = self.vert_struct_r[(self.vert_struct_r>=r_low)*(self.vert_struct_r<r_up)]
                all_h = self.vert_struct_h[(self.vert_struct_r>=r_low)*(self.vert_struct_r<r_up)]

                if len(all_h)>2:
                    r_bin_val.append(np.mean([r_low, r_up]))
                    h_bin_val.append(np.mean(all_h))
                    h_bin_std.append(np.std(all_h))
                if len(all_h)<=2:
                    r_bin_val.append(np.nan)
                    h_bin_val.append(np.nan)
                    h_bin_std.append(np.nan)

        self.vert_r_bin = np.array(r_bin_val)
        self.vert_h_bin = np.array(h_bin_val)
        self.vert_h_bin_std = np.array(h_bin_std)

    def get_mask_array(self):
        """
        Returns an array with the numbers of the masked channels.
        """
        mask_array = glob.glob(self.folder + '/mask_far_*')
        num_array = []
        for mask in mask_array:
            num = mask[len(self.folder)+len('/mask_far_chan_'):-4]
            num_array.append(int(num))


        return list(np.sort(np.array(num_array)))
