# coding: utf-8

# ## Simulate non-isoplanatic imaging for LOW at 100MHz.
# 
# ### A set of model components are drawn from GLEAM. An ionospheric screen model is used to calculate the pierce points of the two stations in an interferometer for a given component. The model visibilities are calculated directly, and screen phase applied to obtain the corrupted visibility.


import logging

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from data_models.data_model_helpers import export_blockvisibility_to_hdf5
from data_models.memory_data_models import SkyModel
from data_models.polarisation import PolarisationFrame
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.execution_support.dask_init import get_dask_Client
from wrappers.arlexecute.visibility.coalesce import convert_blockvisibility_to_visibility
from wrappers.serial.image.operations import import_image_from_fits, smooth_image, qa_image, export_image_to_fits
from wrappers.serial.imaging.base import create_image_from_visibility, advise_wide_field
from wrappers.serial.imaging.primary_beams import create_low_test_beam
from wrappers.serial.simulation.mpc import create_gaintable_from_screen
from wrappers.serial.simulation.testing_support import create_named_configuration, \
    create_low_test_skycomponents_from_gleam
from wrappers.serial.skycomponent.operations import apply_beam_to_skycomponent, insert_skycomponent
from wrappers.serial.skycomponent.operations import filter_skycomponents_by_flux
from wrappers.serial.visibility.base import create_blockvisibility, copy_visibility

from workflows.arlexecute.skymodel.skymodel_arlexecute import predict_skymodel_list_arlexecute_workflow

if __name__ == '__main__':
    def init_logging():
        logging.basicConfig(filename='low-sims-mpc.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
    
    
    init_logging()
    log = logging.getLogger()
    
    
    def lprint(*args):
        log.info(*args)
        print(*args)
    
    # In[4]:
    
    n_workers = 128
    
    c = get_dask_Client(
        memory_limit=64 * 1024 * 1024 * 1024, n_workers=n_workers, threads_per_worker=1)
    arlexecute.set_client(c)
    # Initialise logging on the workers. This appears to only work using the process scheduler.
    arlexecute.run(init_logging)
    
    # ### Set up the observation: 10 minutes at transit, with 10s integration. Skip 5/6 points to avoid out station redundancy
    
    # In[6]:
    
    
    nfreqwin = 1
    ntimes = 61
    rmax = 50000.0
    dec = -40.0 * u.deg
    frequency = numpy.linspace(1e8, 1.3e8, nfreqwin)
    if nfreqwin > 1:
        channel_bandwidth = numpy.array(nfreqwin * [frequency[1] - frequency[0]])
    else:
        channel_bandwidth = [0.3e8]
    times = numpy.linspace(-300, 300.0, ntimes) * numpy.pi / (3600.0 * 12.0)
    
    phasecentre = SkyCoord(ra=+0.0 * u.deg, dec=dec, frame='icrs', equinox='J2000')
    low = create_named_configuration('LOWBD2', rmax=rmax)
    print('Configuration has %d stations' % len(low.data))
    centre = numpy.mean(low.xyz, axis=0)
    distance = numpy.hypot(low.xyz[:, 0] - centre[0],
                           low.xyz[:, 1] - centre[1],
                           low.xyz[:, 2] - centre[2])
    lowouter = low.data[distance > 1000.0][::6]
    lowcore = low.data[distance < 1000.0][::3]
    low.data = numpy.hstack((lowcore, lowouter))
    print('Thinned configuration has %d stations' % len(lowcore.data))
    
    blockvis = create_blockvisibility(
        low,
        times,
        frequency=frequency,
        channel_bandwidth=channel_bandwidth,
        weight=1.0,
        phasecentre=phasecentre,
        polarisation_frame=PolarisationFrame("stokesI"),
        zerow=True)
    
    # ### Find sampling, image size, etc
    wprojection_planes = 1
    vis = convert_blockvisibility_to_visibility(blockvis)
    advice = advise_wide_field(vis, guard_band_image=2.0, delA=0.02)
    
    cellsize = advice['cellsize']
    vis_slices = advice['vis_slices']
    npixel = advice['npixels2']
    
    # ### Generate the model from the GLEAM catalog, including application of the primary beam.
    flux_limit = 0.1
    beam = create_image_from_visibility(
        blockvis,
        npixel=npixel,
        frequency=frequency,
        nchan=nfreqwin,
        cellsize=cellsize,
        phasecentre=phasecentre)
    beam = create_low_test_beam(beam)
    
    original_gleam_components = create_low_test_skycomponents_from_gleam(
        flux_limit=flux_limit,
        phasecentre=phasecentre,
        frequency=frequency,
        polarisation_frame=PolarisationFrame('stokesI'),
        radius=0.2)
    
    all_components = apply_beam_to_skycomponent(original_gleam_components,
                                                beam)
    
    all_components = filter_skycomponents_by_flux(all_components, flux_min=flux_limit)
    
    lprint("Number of components %d" % len(all_components))
    
    # ### Fill the gaintables from the ionospheric screen model
    screen = import_image_from_fits('low_screen_15000.0r0_0.100rate.fits')
    all_gaintables = create_gaintable_from_screen(blockvis, all_components, screen, scale=1.0)
    
    # ### Now assemble the skymodels
    empty = create_image_from_visibility(
        blockvis,
        npixel=128,
        frequency=frequency,
        nchan=nfreqwin,
        cellsize=cellsize,
        phasecentre=phasecentre)
    
    gleam_skymodel_noniso = [SkyModel(components=[all_components[i]], image=empty,
                                      gaintable=all_gaintables[i])
                             for i, sm in enumerate(all_components)]
    
    gleam_skymodel_iso = [SkyModel(components=all_components, image=empty,
                                   gaintable=all_gaintables[0])]
    
    from data_models.data_model_helpers import export_skymodel_to_hdf5
    
    export_skymodel_to_hdf5(gleam_skymodel_noniso, 'low-sims-50000m-noniso_skymodel.hdf5')
    export_skymodel_to_hdf5(gleam_skymodel_iso, 'low-sims-50000m-iso_skymodel.hdf5')
    
    from wrappers.arlexecute.visibility.coalesce import convert_visibility_to_blockvisibility
    
    all_skymodel_iso_blockvis = copy_visibility(blockvis, zero=True)
    all_skymodel_iso_vis = convert_blockvisibility_to_visibility(all_skymodel_iso_blockvis)
    
    future_vis = arlexecute.scatter(all_skymodel_iso_vis)
    result = predict_skymodel_list_arlexecute_workflow(future_vis,
                                                       gleam_skymodel_iso,
                                                       context='2d', docal=True)
    all_skymodel_iso_vis = arlexecute.compute(result, sync=True)[0]
    
    all_skymodel_iso_blockvis = convert_visibility_to_blockvisibility(all_skymodel_iso_vis)
    
    export_blockvisibility_to_hdf5(all_skymodel_iso_blockvis,
                                   'low-sims-50000m-skymodel_iso_blockvis_rmax%.1f.hdf5' % rmax)
    
    # Now predict the visibility for each skymodel and apply the gaintable for that skymodel, returning a list of
    # visibilities, one for each skymodel. We then sum these to obtain the total predicted visibility. All images and
    # skycomponents in the same skymodel get the same gaintable applied which means that in this case each skycomponent
    # has a separate gaintable.
    all_skymodel_noniso_blockvis = copy_visibility(blockvis, zero=True)
    all_skymodel_noniso_vis = convert_blockvisibility_to_visibility(all_skymodel_noniso_blockvis)

    ngroup=n_workers
    future_vis = arlexecute.scatter(all_skymodel_noniso_vis)
    chunks = [gleam_skymodel_noniso[i:i + ngroup] for i in range(0, len(gleam_skymodel_noniso), ngroup)]
    for chunk in chunks:
        result = predict_skymodel_list_arlexecute_workflow(future_vis, chunk, context='2d', docal=True)
        work_vis = arlexecute.compute(result, sync=True)
        for w in work_vis:
            all_skymodel_noniso_vis.data['vis'] += w.data['vis']

    all_skymodel_noniso_blockvis = convert_visibility_to_blockvisibility(all_skymodel_noniso_vis)
    
    export_blockvisibility_to_hdf5(all_skymodel_noniso_blockvis,
                                   'low-sims-50000m-skymodel_noniso_blockvis_rmax%.1f.hdf5' % rmax)

    smodel = create_image_from_visibility(
        blockvis,
        npixel=npixel,
        frequency=frequency,
        nchan=nfreqwin,
        cellsize=cellsize,
        phasecentre=phasecentre)
    smodel = insert_skycomponent(smodel, all_components)
    smodel = smooth_image(smodel, 3.0)
    lprint(qa_image(smodel))
    export_image_to_fits(smodel, 'low-sims-50000m-smodel_rmax%.1f.fits' % rmax)
