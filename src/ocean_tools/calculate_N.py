# calculating N

import numpy as np
from typing import Dict
from ocean_tools.geostrophy import geostrophic_surface_currents, relative_vorticity
from ocean_tools.derivatives import directional_derivative
from ocean_tools.geodesy import (
    projection_zonal_meridional,
    track_orientation,
    distances_along_axis, 
)

def zonal_merid_derivatives(
    ds,
    target_field: str,
    longitude: str = "longitude",
    latitude: str = "latitude",
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Compute distances in the along and across track directions.
    
    Parameters
    ----------
    longitude
        Swath longitudes
    latitude
        Swath latitudes
    kwargs
        Additional arguments for the derivation
    """
    longitude = ds[longitude]
    latitude = ds[latitude]
    var = ds[target_field]

    # Angle de projection
    angles_zonal_along = track_orientation(
        latitude,
        longitude,
        along_track_dim="num_lines")

    #Distance along track et across track
    distances_across_track = distances_along_axis(
        longitude,
        latitude,
        dim="num_pixels")

    distances_along_track = distances_along_axis(
        longitude,
        latitude,
        dim="num_lines")

    #Dérivée dans le repère de la fauchée
    deriv_var_along = directional_derivative(
        var,
        distances_along_track,
        dim="num_lines",
        **kwargs)

    deriv_var_across = -directional_derivative(
        var,
        distances_across_track,
        dim="num_pixels",
        **kwargs)

    #Transformation dans le repère lon/lat
    (
        deriv_var_zonal,
        deriv_var_meridional) = projection_zonal_meridional(
            deriv_var_along,
            deriv_var_across,
            angles_zonal_along)

    #Vitesse dans le repère lon/lat
    #(
    #    deriv_zonal,
    #    deriv_meridional,) = geostrophic_surface_currents(
    #        deriv_var_zonal,
    #        deriv_var_meridional,
    #        ds.latitude)
    #deriv_zonal.attrs["short_name"] = "U"
    #deriv_meridional.attrs["short_name"] = "V"
    
    #Vitesse dans le repère de la fauchée
    (
        deriv_along,
        deriv_across,) = geostrophic_surface_currents(
            deriv_var_along,
            deriv_var_across,
            ds.latitude)

    
    ds.update({
        #"deriv_across2" : deriv_across,
        #"deriv_along2" : deriv_along,
        #"distance_across_track2" : distances_across_track,
        #"distance_along_track2" : distances_along_track,
        target_field + "_dx" : deriv_var_zonal,
        target_field + "_dy": deriv_var_meridional,
    })

    return ds