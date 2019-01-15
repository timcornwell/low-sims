# coding: utf-8

# ## Simulate non-isoplanatic imaging for LOW at 100MHz.
# 
# ### The input data have been simulated in low-sims-mpc-10000m-prep
#
# ### Continuum Imaging and then ICAL are run to obtain an image and a set of components.

# In[1]:

import sys
import logging

import numpy

from data_models.data_model_helpers import import_blockvisibility_from_hdf5
from processing_library.image.operations import copy_image
from workflows.arlexecute.pipelines.pipeline_arlexecute import continuum_imaging_list_arlexecute_workflow, \
    ical_list_arlexecute_workflow
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.execution_support.dask_init import get_dask_Client
from wrappers.arlexecute.visibility.coalesce import convert_blockvisibility_to_visibility
from wrappers.serial.calibration.calibration_control import create_calibration_controls
from wrappers.serial.image.operations import export_image_to_fits
from wrappers.serial.image.operations import show_image, qa_image
from wrappers.serial.imaging.base import create_image_from_visibility, advise_wide_field

# In[2]:

if __name__ == "__main__":
    
    log = logging.getLogger()

    log.setLevel(logging.DEBUG)
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.addHandler(logging.StreamHandler(sys.stderr))
    
    def lprint(*args):
        log.info(*args)
        print(*args)
    
    n_workers = 16
    c = get_dask_Client(
        memory_limit=64 * 1024 * 1024 * 1024, n_workers=n_workers, threads_per_worker=4)
    arlexecute.set_client(c)
    
    rmax = 50000.0
    blockvis = import_blockvisibility_from_hdf5('results/low-sims-mpc-skymodel-iso-blockvis_rmax50000.0.hdf5')
 
    nfreqwin = len(blockvis.frequency)
    ntimes = len(blockvis.time)
    frequency = blockvis.frequency
    times = blockvis.time
    phasecentre = blockvis.phasecentre
    
    # ### Find sampling, image size, etc
    vis = convert_blockvisibility_to_visibility(blockvis)
    advice = advise_wide_field(vis, guard_band_image=2.0, delA=0.02)
    
    cellsize = advice['cellsize']
    vis_slices = advice['vis_slices']
    npixel = advice['npixels2']
    
    model = create_image_from_visibility(
        blockvis,
        npixel=npixel,
        frequency=frequency,
        nchan=nfreqwin,
        cellsize=cellsize,
        phasecentre=phasecentre)
    
    Vobs = convert_blockvisibility_to_visibility(blockvis)

    vis_list = arlexecute.scatter([vis])
    model_list = arlexecute.scatter([model])

    cimg_list = continuum_imaging_list_arlexecute_workflow(
        vis_list,
        model_imagelist=model_list,
        context='timeslice',
        vis_slices=n_workers,
        algorithm='msclean',
        scales=[0, 3, 10],
        niter=1000,
        fractional_threshold=0.5,
        threshold=0.5,
        nmajor=5,
        gain=0.25,
        psf_support=512,
        deconvolve_facets=8,
        deconvolve_overlap=32,
        deconvolve_taper='tukey')
    
    cimg_deconvolved, cimg_residual, cimg_restored = arlexecute.compute(cimg_list, sync=True)
    
    from processing_components.skycomponent.operations import find_skycomponents
    
    recovered_cimg_components = find_skycomponents(cimg_restored[0], fwhm=2, threshold=0.5, npixels=12)
    print(len(recovered_cimg_components))
    print(recovered_cimg_components[0])
    
    lprint(qa_image(cimg_restored[0]))
    
    export_image_to_fits(cimg_deconvolved[0],
                         'results/low-sims-mpc-iso_cimg_deconvolved_rmax%.1f.fits' % rmax)
    export_image_to_fits(cimg_restored[0],
                         'results/low-sims-mpc-iso-cimg-restored_rmax%.1f.fits' % rmax)
    export_image_to_fits(cimg_residual[0][0],
                         'results/low-sims-mpc-iso-cimg-residual_rmax%.1f.fits' % rmax)
    
    ical_model = copy_image(cimg_deconvolved[0])
    ical_model_list = arlexecute.scatter([ical_model])
    
    # In[14]:
    
    controls = create_calibration_controls()
    
    controls['T']['first_selfcal'] = 0
    controls['T']['phase_only'] = True
    controls['T']['timescale'] = 'auto'
    
    ical_list = ical_list_arlexecute_workflow(
        vis_list,
        model_imagelist=ical_model_list,
        context='timeslice',
        vis_slices=n_workers,
        algorithm='msclean',
        scales=[0, 3, 10],
        niter=1000,
        fractional_threshold=0.5,
        threshold=0.1,
        nmajor=10,
        gain=0.25,
        psf_support=512,
        deconvolve_facets=8,
        deconvolve_overlap=32,
        deconvolve_taper='tukey',
        timeslice='auto',
        global_solution=False,
        do_selfcal=True,
        calibration_context='T',
        controls=controls)
    
    ical_deconvolved, ical_residual, ical_restored, gt_list = arlexecute.compute(ical_list, sync=True)
    
    recovered_ical_components = find_skycomponents(ical_restored[0], fwhm=2, threshold=0.35, npixels=12)
    print(len(recovered_ical_components))
    print(recovered_ical_components[0])
    
    # In[16]:
    
    from data_models.data_model_helpers import export_skycomponent_to_hdf5, export_gaintable_to_hdf5
    
    export_skycomponent_to_hdf5(recovered_ical_components,
                                'results/low-sims-mpc-iso-ical-components_rmax%.1f.hdf5' % rmax)
    export_image_to_fits(ical_deconvolved[0],
                         'results/low-sims-mpc-iso_ical_deconvolved_rmax%.1f.fits' % rmax)
    export_gaintable_to_hdf5(gt_list[0]['T'],
                             'results/low-sims-mpc-iso-ical-gaintable_rmax%.1f.hdf5' % rmax)
    lprint(qa_image(ical_restored[0]))
    export_image_to_fits(ical_restored[0],
                         'results/low-sims-mpc-iso-ical-restored_rmax%.1f.fits' % rmax)
    export_image_to_fits(ical_residual[0][0],
                         'results/low-sims-mpc-iso-ical-residual_rmax%.1f.fits' % rmax)
    
    exit()