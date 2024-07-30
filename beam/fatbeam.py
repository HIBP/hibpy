# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 23:26:37 2024

@author: Ammosov, Krokhalev
"""
#%%
try:
    from joblib import delayed
    from hibpy.misc.par import _Parallel
    JOBLIB_AVAILABLE = True
except Exception as e:
    JOBLIB_AVAILABLE = False
    print('WARNING: joblib import failed', e)
#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ..misc.profiles import gauss
from ..phys.constants import SI_1keV
from ..misc.plot_funcs import plot_fan
from ..beam.stopper import CountStopper
from ..geom.geom import plot_point, vec3D
from ..beam.optimizer import stopped_at_aim
from ..beam.trajectory import Trajectory, pass_fan
#%%
class Fatbeam():
    """
    Fatbeam class allows to trace several trajectories to aim, calculate and
    analyze sample volumes (ionization zones) parameters.
    
    Initial parameters:
        tr : Trajectory class object
            Trajectory with valid rrvv.
            
        lattice : object of child class of AbstractLatticeMarkup class
            Contains fatbeam initial points parameters such as amount of
            filaments, beam diameter, divergency angle, etc.
        
        profile : callable, optional
            Defines initial ion current distribution profile for fatbeam.
            e.g. gauss, bell (see profile.py). The default is gauss(1., 1./3.).
    
    Example commands:
        Creating lattice:
            lattice = CircleLatticeMarkup(d_beam, div_angle*mRad, n, m)
        
        Create fatbeam:
            fb = Fatbeam(traj, lattice)
            
        Strat calculation:
            fb.calc(prim_beamline, line, prim_stopper, fan_stopper,
            precise_fan_stopper, B, prim_dt, sec_dt, pt_in_plasma, fan_density=4,
            timestep_divider=50, parallel=calc_with_parallel)
        
        Saving:
            fb.save(path)
            
        Loading:
            fb = Fatbeam.from_file(path)
        
        Ionization zone (Sample Volume):
            fb.plot_sv(n) - plot sample volume (ionization zone) of nth slit.
            n : int
            
            fb.info_sv(n, coord_converter) - get information about sample volume
            (ionization zone) of nth slit. n : int; coord_converter : object of
            child class of AbstractCoordConverter class.
            
            fb.get_sv(n) - get coordinates of sample volume (ionization zone) of nth slit. n : int
        
        Misc. commands:
            fb[0] - get first trajectory (filament)
            
            len(fb) - get amount of filaments
            
            fb.lattice.plot() - plot lattice markup.
            
            fb.lattice.plot3d() - plot lattice in 3D axes with visualization of
            divergency angle.
    
    """
    def __init__(self, tr, lattice, profile=gauss(1., 1./3.)):
        """
        Parameters
        ----------
        tr : Trajectory class object
            Trajectory with valid rrvv.
        lattice : object of child class of AbstractLatticeMarkup class
        profile : callable, optional
            Defines initial ion current distribution profile for fatbeam.
            e.g. gauss, bell (see profile.py). The default is gauss(1., 1./3.).

        Returns
        -------
        None.

        """
        self.trajectories = [Trajectory(tr.q, tr.m, tr.Ebeam, rv0, U=tr.U)
                             for rv0 in lattice(tr.rrvv[0])]
        self.lattice = lattice
        self._profile = profile
        self.weights = self.lattice.get_weights(self._profile)
    
    def calc(self, prim_beamline, line, prim_stopper, fan_stopper, precise_fan_stopper,
             B, prim_dt, sec_dt, pt_in_plasma, fan_density=4, timestep_divider=10, 
             parallel=True):
        """
        
        Method for conduct calculation.

        Parameters
        ----------
        Parameters are identical to pass_filament method.

        Returns
        -------
        None.

        """
        
        if parallel:
            self.trajectories = _Parallel(package_size=None, n_jobs=-2, verbose=10) (delayed(self.pass_filament_with_return)(tr, prim_beamline, line, prim_stopper,
                                                  fan_stopper, precise_fan_stopper, B,
                                                  prim_dt, sec_dt, pt_in_plasma, timestep_divider=timestep_divider) for tr in self.trajectories)
        
        else:
            for tr in self.trajectories:
                # print_traj(traj)
                self.pass_filament(tr, prim_beamline, line, prim_stopper,
                                              fan_stopper, precise_fan_stopper, B,
                                              prim_dt, sec_dt, pt_in_plasma, timestep_divider=timestep_divider,
                                              parallel=True)
        
    def pass_filament(self, tr, prim_beamline, line, prim_stopper, fan_stopper, precise_fan_stopper,
             B, prim_dt, sec_dt, pt_in_plasma, fan_density=4, timestep_divider=10, 
             parallel=True):
        
        """
        Passes precise fan to aim (slits) and sets partial_dense_fan for trajectory.
        Modifies entry Trajectory object.
        
        Parameters
        ----------         
        prim_beamline : Group3D[FlatPlates; FlaredPlates]
            Beamline e.g. line_A or line_B.
            
        line : Group3D[FlatPlates; FlaredPlates]
            Beamline e.g. line_A or line_B.
            
        prim_stopper : CollisionStopper
            Stopper for primary trajectory.
            
        fan_stopper : CollisionStopper
            Stopper for rough fan of secondaries.
            
        precise_fan_stopper : CollisionStopper
            Stopper for precise fan (partial_dense_fan).
            
        B : RegularGridVectorInterpolator3D
            Magnetic field interpolator.
            
        prim_dt : float
            Time step dt for calculation primary trajectory.
            
        sec_dt : float
            Time step dt for calculation secondary trajectory.
            
        pt_in_plasma : Callable
            Function for determining if point is inside of plasma.
            
        fan_density : int, optional
            Step on fan. For pass_fan function.
            The default is 4. Take each 4th trajectory.
            
        timestep_divider : int, optional
            Factor for increasing points density of partial primary trajectory. 
            Sets new dt for running Trajecory: prim_dt/timestep_divider.
            The default is 10.
        
        parallel : bool, optional
            Conduct calculations in parallel using joblib library.
            The default is True.
            
        Returns
        -------
        None.

        """
        
        # set U from trajectory to primary and secondary beamlines
        self.set_U_to_beamline(tr.U, prim_beamline)
        self.set_U_to_beamline(tr.U, line)
        
        # pass primary trajectories
        tr.run(prim_beamline.E, B, prim_stopper, prim_dt)
        pass_fan(tr, prim_beamline.E, B, pt_in_plasma, fan_stopper, sec_dt, 
                      fan_density=fan_density)
        
        # create precise primary temporary trajectories
        start_idx, stop_idx = self.fan_to_aim_indexes(tr, fan_density)
        count_stopper = CountStopper((stop_idx - start_idx)*timestep_divider)
        trajectory_tmp = Trajectory(tr.q, tr.m, tr.Ebeam*SI_1keV, tr.rrvv[start_idx], U=tr.U)
        trajectory_tmp.run(prim_beamline.E, B, count_stopper, prim_dt/timestep_divider)
        
        # Pass precise secondary fan
        pass_fan(trajectory_tmp, line.E, B, pt_in_plasma, precise_fan_stopper, sec_dt, 
                      fan_density=1, parallel=parallel)
        
        # set partial_dense_fan which goes to precise_fan_stopper        
        tr.partial_dense_fan = trajectory_tmp.fan
        
        # calculate lambda and put in lambdas attribute
        tr.lambdas = self.calc_lambda(tr)
        
    def pass_filament_with_return(self, tr, *args, **kwargs):
        """
        Same as pass_filament method but returns the trajectory with precise_fan_stopper.
        Needed for joblib Parallel calculations.

        Parameters
        ----------
        tr : Trajectoty
            Object of Trajectory class.
        *args :
            args from pass_filament method.
        **kwargs :
            kwargs from pass_filament method.

        Returns
        -------
        tr : Trajectoty
            Object of Trajectory class with updated partial_dense_fan.
        """
        self.pass_filament(tr, *args, **kwargs)
        return tr
    
    def plot(self, slit_numbers=None, geometry_group=None, title_on=True,
             legend_on=True, axes_code=None, **kwargs):
        """
        
        Plot fatbeam secondary trajectories which goes to slits.
        
        Parameters
        ----------
        slit_numbers : list/range, optional
            Example "slit_numbers=[2, 3, 5]" - take 2, 3 and 5 slits.
            The default is None (all slits).
        geometry_group : Group3D([...]), optional
            A Group3D object contains different geometry objects with plot method.
            The default is None.
        title_on : bool, optional
            Show title toggle. The default is True.
        legend_on : bool, optional
            Show legend toggle. The default is True.
        axes_code : str, optional
            Show different axes: "XY", "YZ", etc. The default is None (shows "XY" axes).
        **kwargs :
            Additional plot settings.

        Returns
        -------
        None.

        """
        
        for tr in self.trajectories:
            if len(tr.partial_dense_fan) > 0:
                fans_to_slits = tr.slit_bins
                
                if geometry_group is not None:
                    geometry_group.plot(axes_code=axes_code, **kwargs)
                
                if slit_numbers is None:
                    slit_numbers_ = fans_to_slits.keys()
                    
                for slit_number in slit_numbers_:
                    plot_fan(fans_to_slits[slit_number], color='C%d'%slit_number, 
                              axes_code=axes_code, label=slit_number, **kwargs)
                    
                    for tr_in_fan in fans_to_slits[slit_number]:
                        plot_point(tr_in_fan.obstacle.intersect_with_segment(tr_in_fan.rrvv[-1, :3], tr_in_fan.rrvv[-2, :3]), 
                                   color='C%d'%slit_number, axes_code=axes_code, **kwargs)
                
        if legend_on:
            handles, labels = plt.gca().get_legend_handles_labels()
            l_h_dict = dict(sorted(dict(zip(labels, handles)).items()))
            handles, labels = l_h_dict.values(), l_h_dict.keys()
            plt.legend(handles, labels, loc='best')
        
        if title_on:
            plt.title(f'Ebeam = {round(tr.Ebeam/SI_1keV)} keV, UA2 = {round(tr.U["A2"])} kV')
    
    def set_U_to_beamline(self, U_dict, beamline):
        """
        Set U for each plate of beamline.

        Parameters
        ----------
        U_dict : dict
            Dictionary of U e.g. trajectory.U.
        beamline : Group3D[FlatPlates; FlaredPlates]
            Beamline e.g. line_A or line_B.

        Returns
        -------
        None.

        """
        for plates in beamline:
            if plates.name in U_dict.keys():
                plates.U = U_dict[plates.name]   
    
    def fan_to_aim_indexes(self, trajectory, fan_density):
        """
        Returns first and last indexes of trajectories in fan which goes to aim.
        Trajectory must have fan.

        Parameters
        ----------
        trajectory : Trajecoty
            Object of Trajectory class.
        fan_density : int
            Skipping step for passing fan.

        Returns
        -------
        int (or None)
            Index of first trajectory point which intersected with aim.
        int (or None)
            Index of last trajectory point which intersected with aim.
        """
        fan_to_aim_indexes = [tr.prim_index for tr in trajectory.fan if stopped_at_aim(tr)]
        if fan_to_aim_indexes:
            return fan_to_aim_indexes[0] - fan_density, fan_to_aim_indexes[-1] + fan_density
        else:
            return None, None
    
    def calc_lambda(self, tr):
        """
        Calcs lambda [m] (length along primary trajectory which goes to Slits) 
        AND SETS lambda_dict : dict{int : float}
              Dictionary: keys - slit numbers : int; values - lambdas [m] : float.   
        TO Traectory.lambdas.
        
        Parameters
        ----------
        tr : Trajectory
            Trajectory class object; must have partial_dense_fan.

        Returns
        -------
        lambda_dict : dict
            Keys: slit number: int; Values: 1D ionization zone size: float.
        
        """
        lambda_dict = {}
        if len(tr.partial_dense_fan) > 0:
            fans_to_slits = tr.slit_bins
            if fans_to_slits:
                for slit_number in fans_to_slits.keys():
                    summ = 0
                    last_diff = 1.
                    for tr1, tr2 in zip(fans_to_slits[slit_number][:-1], fans_to_slits[slit_number][1:]):
                        diff = tr2.rrvv[0, :3] - tr1.rrvv[0, :3]
                        diff = np.linalg.norm(diff)
                        if not diff > 5*last_diff:
                            summ += diff
                            last_diff = diff
                        # if diff > 5*last_diff:
                        #     break #!!!
                    lambda_dict[slit_number] = summ
        return lambda_dict
    
    def get_sv(self, slit_number):
        """
        Provides list of SV points by slit_number.

        Parameters
        ----------
        slit_number : int
            Slit number of energy analyzer aperture.

        Returns
        -------
        ionization_coords : list[list[float, float, float], ...] or None
            List of points-vertices of sample volume (SV) or None if there is no SV.

        """
        
        ionization_coords = []
        for tr in self.trajectories:
            if (tr.partial_dense_fan) and (slit_number in tr.slit_bins.keys()):
                secondaries_to_slits = tr.slit_bins[slit_number]
                
                # write first points of secondaries which go to slits
                for tr2 in secondaries_to_slits:
                    pt = tr2.rrvv[0, :3]
                    ionization_coords.append(list(pt))
        
        ionization_coords = np.asarray(ionization_coords)
        # print(f"IZ: {ionization_coords}")
        
        if len(ionization_coords) > 0:
            return ionization_coords
        else:
            return None
    
    def _centroid_poly(self, poly):
        """
        Calculate central point from set of points.

        Parameters
        ----------
        poly : list[[float, float, float], ...]
            List of points.

        Returns
        -------
        list [float, float, float]
            Point - list of coordinates.

        """
        T = Delaunay(poly).simplices
        n = T.shape[0]
        W = np.zeros(n)
        C = 0
        
        for m in range(n):
            sp = poly[T[m, :], :]
            W[m] = ConvexHull(sp).volume
            C += W[m] * np.mean(sp, axis=0)  
        return C / np.sum(W)
    
    def centroid_weights(self, slit_number):
        """
        Calculates central point of sample volume using filaments weights.
        Weights calculated using initial current profile e.g. gauss, bell.

        Parameters
        ----------
        slit_number : int
            Slit number of energy analyzer aperture.

        Returns
        -------
        list[float, float, float]
            Point - center of sample volume.

        """
        weights = 0.
        weighted_sum = vec3D(0, 0, 0)
        for tr, weight in zip(self.trajectories, self.weights):
            if (tr.partial_dense_fan) and (slit_number in tr.slit_bins.keys()):
                secondaries_to_slits = tr.slit_bins[slit_number]
                
                # write first points of secondaries which go to slits
                for tr2 in secondaries_to_slits:
                    pt = tr2.rrvv[0, :3]
                    weighted_sum += pt*weight
                    weights += weight
        
        return weighted_sum/weights
    
    def info_sv(self, slit_number, coord_converter):
        """
        Provides information about sample volume (SV). Info: SV size,
        SV position in cartesian [m] and flux coordinates, volume [m^3].

        Parameters
        ----------
        slit_number : int
            Slit number of energy analyzer aperture.
        coord_converter : object of child class of AbstractCoordConverter class
            Converts coordinates from cartesian (x, y, z) to flux coordinate
            system (rho, theta, phi).

        Returns
        -------
        dict_info : dict
            Dictionary contains info about SV: "size_cartesian" [m],
            "size_flux", "pos_cartesian"[m], "pos_flux", "volume" [m^3].
            
        """
        ionization_coords = self.get_sv(slit_number)
        if ionization_coords is not None:
            
            hull = ConvexHull(ionization_coords)

            coord_plasma = []
            # ionization zone bounds
            # cartesian
            x_min, x_max = min(ionization_coords[:, 0]), max(ionization_coords[:, 0])
            y_min, y_max = min(ionization_coords[:, 1]), max(ionization_coords[:, 1])
            z_min, z_max = min(ionization_coords[:, 2]), max(ionization_coords[:, 2])
            # plasma flux coordinates
            for r in ionization_coords:
                rho, theta, phi = coord_converter(r)
                coord_plasma.append([rho, theta, phi])
            coord_plasma = np.array(coord_plasma)
            rho_min, rho_max = min(coord_plasma[:, 0]), max(coord_plasma[:, 0])
            theta_min, theta_max = min(coord_plasma[:, 1]), max(coord_plasma[:, 1])
            phi_min, phi_max = min(coord_plasma[:, 2]), max(coord_plasma[:, 2])
            
            # calculate central dot "center using weights"
            pos_cartesian = list(self.centroid_weights(slit_number))
            pos_flux = coord_converter(pos_cartesian)
            
            # calculate central dot "smart center"
            # pos_cartesian2 = self._centroid_poly(ionization_coords)
            # pos_flux2 = self._centroid_poly(coord_plasma)
            
            dict_info = {"size_cartesian": [x_max-x_min, y_max-y_min, z_max-z_min],
                         "size_flux": [rho_max-rho_min,theta_max-theta_min, phi_max-phi_min],
                         "pos_cartesian": pos_cartesian,
                         "pos_flux": pos_flux,
                         "volume": hull.volume, # [m^3]
                         }
        
            return dict_info
    
    def set_aspect_equal_3d(self, ax):
        """
        Fix equal aspect bug for 3D plots.

        Parameters
        ----------
        ax : matplotlib Axes
            axes with projection='3d'.

        Returns
        -------
        None.

        """
    
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
    
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        zmean = np.mean(zlim)
    
        plot_radius = max([abs(lim - mean_)
                           for lims, mean_ in ((xlim, xmean),
                                               (ylim, ymean),
                                               (zlim, zmean))
                           for lim in lims])
    
        ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
        ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])
    
    def plot_sv(self, slit_number, ax=None, axes_code=None, facecolor='black',
                facealpha=0.2, centermarker='o', verticesmarker='o',
                centercolor='red', verticescolor='blue', equal_on=False):
        """
        Plot sample volume (SV) in 3D by specific slit number.

        Parameters
        ----------
        slit_number : int
            Slit number of energy analyzer aperture.
        ax : matplotlib Axes, optional
            Axes with projection='3d'.
        axes_code : str, optional
            Axes to plot e.g. "XY", "YZ", "ZX", etc. The default is None ("XY").
        facecolor : str, optional
            Ionization zone 3D model faces color. The default is 'black'.
        facealpha : str, optional
            Ionization zone 3D model faces transparency. The default is 0.2.
        centermarker : str, optional
            Ionization zone 3D model central point marker. The default is 'o'.
        verticesmarker : str, optional
            Ionization zone 3D model vertices points marker. The default is 'o'.
        centercolor : str, optional
            Ionization zone 3D model central point color. The default is 'red'.
        verticescolor : str, optional
            Ionization zone 3D model vertices points color. The default is 'blue'.
        equal_on : bool, optional
            Set equal axes. The default is False.

        Returns
        -------
        None.

        """

        ionization_coords = self.get_sv(slit_number)

        if ionization_coords is not None:
            if ax is None or not hasattr(ax, 'add_collection3d'):
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
            
            hull = ConvexHull(ionization_coords)
            # draw the polygons of the convex hull
            for s in hull.simplices:
                tri = Poly3DCollection([ionization_coords[s]])
                tri.set_color(facecolor)
                tri.set_alpha(facealpha)
                ax.add_collection3d(tri)
                
            # draw vertices
            ax.scatter(ionization_coords[:, 0], ionization_coords[:, 1],
                       ionization_coords[:, 2], marker=verticesmarker,
                       color=verticescolor)
            
            # central point
            center_weighted = self.centroid_weights(slit_number)
            ax.scatter(center_weighted[0], center_weighted[1], center_weighted[2],
                       color=centercolor, marker=centermarker)
            
            # center_smart = self._centroid_poly(ionization_coords)
            # ax.scatter(center_smart[0], center_smart[1], center_smart[2], color='green')
            
            # ionization zone bounds (cartesian)
            x_min, x_max = min(ionization_coords[:, 0]), max(ionization_coords[:, 0])
            y_min, y_max = min(ionization_coords[:, 1]), max(ionization_coords[:, 1])
            z_min, z_max = min(ionization_coords[:, 2]), max(ionization_coords[:, 2])
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            
            if equal_on:
                self.set_aspect_equal_3d(ax)
            
            plt.show()
        
    
    def save(self, path):
        """
        Allows to save fatbeam as file.

        Parameters
        ----------
        path : str
            Path to save fatbeam as file including filename.

        Returns
        -------
        None.

        """
        data = []
        data.append(self.trajectories)
        data.append(self.lattice)
        data = np.asanyarray(data, dtype=object)
        np.save(path, data)
        print(f"fatbeam saved: \"{path}\".\n")
    
    @classmethod
    def from_file(cls, path, silent=True):
        """
        Allows to load saved fatbeam files.

        Parameters
        ----------
        path : str
            File path to saved fatbeam.

        Returns
        -------
        fb : Fatbeam class
            Fatebeam class object loaded from file.
            
        """
        if os.path.exists(path):
                # unpack data from file
                data = np.load(path, allow_pickle=True)
                if data is not None:
                    trajectories = data[0]
                    lattice = data[1]
                    # create new fatbeam and rewrite loaded data
                    fb = cls(trajectories[-1], lattice)
                    fb.trajectories = trajectories
                    print("fatbeam loaded.\n")
                    return fb
        else:
            raise FileNotFoundError(f"No such file \"{path}\"")
    
    @property
    def profile(self):
        return self._profile
    
    @profile.setter
    def profile(self, profile):
        self._profile = profile
        self.weights = self.lattice.get_weights(profile)
    
    def __getitem__(self, item):
        return self.trajectories[item]
    
    def __len__(self):
        return len(self.trajectories)
    
    def __repr__(self):
        Ebeam = round(self.trajectories[-1].Ebeam/SI_1keV)
        tr_info = f"Ebeam = {Ebeam}\nUA2 = {round(self.trajectories[-1].U['A2'])}\n"
        fb_info = f"{self.lattice}"
        contains = ";\n".join([f"{tr}" for tr in self.trajectories])
        repr_str = contains + '\n' + tr_info + fb_info
        return repr_str