# coding: utf-8

# ## Model partition calibration example

# In[ ]:


import logging

import numpy

from data_models.data_model_helpers import export_skymodel_to_hdf5
from data_models.data_model_helpers import import_blockvisibility_from_hdf5
from data_models.data_model_helpers import import_skymodel_from_hdf5
from processing_library.image.operations import copy_image
from workflows.arlexecute.calibration.calibration_arlexecute import calibrate_list_arlexecute_workflow
from workflows.arlexecute.imaging.imaging_arlexecute import deconvolve_list_arlexecute_workflow, \
    restore_list_arlexecute_workflow
from workflows.arlexecute.imaging.imaging_arlexecute import invert_list_arlexecute_workflow
from workflows.arlexecute.skymodel.skymodel_arlexecute import convolve_skymodel_list_arlexecute_workflow
from workflows.arlexecute.skymodel.skymodel_arlexecute import extract_datamodels_skymodel_list_arlexecute_workflow
from workflows.arlexecute.skymodel.skymodel_arlexecute import invert_skymodel_list_arlexecute_workflow
from workflows.arlexecute.skymodel.skymodel_arlexecute import predict_skymodel_list_arlexecute_workflow
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.execution_support.dask_init import get_dask_Client
from wrappers.arlexecute.visibility.coalesce import convert_blockvisibility_to_visibility
from wrappers.serial.image.operations import import_image_from_fits, export_image_to_fits
from wrappers.serial.image.operations import qa_image
from wrappers.arlexecute.skymodel.operations import calculate_skymodel_equivalent_image, \
    update_skymodel_from_image, update_skymodel_from_gaintables
from wrappers.arlexecute.skycomponent.operations import find_skycomponents

from processing_components.simulation.ionospheric_screen import grid_gaintable_to_screen


if __name__ == "__main__":
    
    # In[ ]:
    def init_logging():
        logging.basicConfig(filename='results/low-sims-mpc.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
    
    
    init_logging()
    log = logging.getLogger()
    
    
    def lprint(*args):
        log.info(*args)
        print(*args)
    
    
    c = get_dask_Client(memory_limit=48 * 1024 * 1024 * 1024, n_workers=16)
    arlexecute.set_client(c)
#    arlexecute.run(init_logging)
    
    # ### Read the previously prepared observation
    
    # In[ ]:
    
    rmax = 50000.0
    blockvis = import_blockvisibility_from_hdf5('results/low-sims-skymodel-noniso-blockvis_rmax10000.0.hdf5')
    Vobs = convert_blockvisibility_to_visibility(blockvis)
    
    # ### Initialization phase
    
    # In[ ]:
    
    
    nsources = 10
    
    theta_list = import_skymodel_from_hdf5("results/mpc-skymodel_%dsources_iteration0.hdf5" % (nsources))
    model = copy_image(theta_list[0].image)
    model.data[...] = 0.0
    
    psf_obs = invert_list_arlexecute_workflow([Vobs], [model], context='2d', dopsf=True)
    psf_obs = arlexecute.compute(psf_obs, sync=True)
    
    for iteration in range(1, 10):
        
        print('Iteration %d' % iteration)
        
        future_Vobs = arlexecute.scatter(Vobs)
        Vdatamodel_list = predict_skymodel_list_arlexecute_workflow(future_Vobs, theta_list, context='2d', docal=True)
        Vdatamodel_list = extract_datamodels_skymodel_list_arlexecute_workflow(future_Vobs, Vdatamodel_list)
        Vdatamodel_list = arlexecute.compute(Vdatamodel_list, sync=True)
        
        dirty_all_conv = convolve_skymodel_list_arlexecute_workflow(future_Vobs, theta_list, context='2d', docal=True)
        dirty_all_conv = arlexecute.compute(dirty_all_conv, sync=True)
        dirty_all_cal = invert_skymodel_list_arlexecute_workflow(Vdatamodel_list, theta_list, context='2d', docal=True)
        dirty_all_cal = arlexecute.compute(dirty_all_cal, sync=True)
        
        for i, d in enumerate(dirty_all_cal):
            d[0].data -= dirty_all_conv[i][0].data
        
        peak = 0.0
        combined_dirty_image = copy_image(model)
        combined_dirty_image.data[...] = 0.0
        for i, d in enumerate(dirty_all_cal):
            peak = max(peak, numpy.max(numpy.abs(d[0].data)))
            print(i, numpy.max(numpy.abs(d[0].data)))
            combined_dirty_image.data += d[0].data
        print("Peak residual :", peak)
        
        for ism, sm in enumerate(theta_list):
            deconvolved_list, _ = deconvolve_list_arlexecute_workflow([dirty_all_cal[ism]], [psf_obs[0]],
                                                                      [model], mask=sm.mask, algorithm='msclean',
                                                                      scales=[0, 3, 10], niter=100,
                                                                      fractional_threshold=0.3, threshold=0.3 * peak,
                                                                      gain=0.1,
                                                                      psf_support=128, deconvolve_facets=8,
                                                                      deconvolve_overlap=16,
                                                                      deconvolve_taper='tukey')
            
            deconvolved_list = arlexecute.compute(deconvolved_list, sync=True)
            print(qa_image(deconvolved_list[0], context='skymodel %d' % ism))
            sm.image.data += sm.mask.data * deconvolved_list[0].data
        
        combined_model = calculate_skymodel_equivalent_image(theta_list)
        print(qa_image(combined_model, context='Combined model'))
        
        Vpredicted_list = predict_skymodel_list_arlexecute_workflow(future_Vobs, theta_list, context='2d', docal=True)
        result = calibrate_list_arlexecute_workflow(Vdatamodel_list, Vpredicted_list,
                                                    calibration_context='T',
                                                    iteration=0, global_solution=False)
        Vcalibrated, gaintable_list = arlexecute.compute(result, sync=True)
        
        theta_list = update_skymodel_from_gaintables(theta_list, gaintable_list, calibration_context='T')
        
        export_skymodel_to_hdf5(theta_list,
                                "results/mpc-skymodel_%dsources_iteration%d_rmax%.1f.hdf5" % (nsources, iteration, rmax))
        
        # In[ ]:
        
        export_image_to_fits(combined_model, 'results/low-mpc_%dsources-deconvolved.fits' % nsources)
        
        # In[ ]:
        
    result = restore_list_arlexecute_workflow([combined_model], psf_obs, [(combined_dirty_image, 0.0)])
    result = arlexecute.compute(result, sync=True)
        
        # In[ ]:
        
    print(qa_image(result[0], context='MPC restored image'))
        
        # In[ ]:
        
    recovered_mpccal_components = find_skycomponents(result[0], fwhm=2, threshold=0.14, npixels=12)
    print(len(recovered_mpccal_components))
    print(recovered_mpccal_components[0])
    from data_models.data_model_helpers import export_skycomponent_to_hdf5
        
    export_skycomponent_to_hdf5(recovered_mpccal_components, 'results/low-mpc_%dsources-components.hdf5' % nsources)
    export_image_to_fits(result[0], 'low-mpc_%dsources-restored.fits' % nsources)

    gleam_skymodel_iso = import_skymodel_from_hdf5('results/low-sims-mpc-iso_skymodel_rmax%.1f.hdf5' % rmax)
    from processing_components.skycomponent.operations import filter_skycomponents_by_flux
    
    bright_gleam = filter_skycomponents_by_flux(gleam_skymodel_iso.components, flux_min=0.4)
    
    
    def max_flux(elem):
        return numpy.max(elem.flux)
    
    
    sorted_bright_gleam = sorted(bright_gleam, key=max_flux, reverse=True)
    
    from wrappers.serial.skycomponent.operations import find_skycomponent_matches
    
    # In[ ]:
    
    
    oldscreen = import_image_from_fits('low_screen_5000.0r0_0.100rate.fits')
    from processing_components.image.operations import create_empty_image_like
    
    newscreen = create_empty_image_like(oldscreen)
    gaintables = [th.gaintable for th in theta_list]
    newscreen, weights = grid_gaintable_to_screen(blockvis, gaintables, newscreen)
    print(qa_image(newscreen))
    export_image_to_fits(newscreen, 'mpccal_%dsources_screen.fits' % nsources)
    export_image_to_fits(weights, 'mpccal_%dsources_screen_weights.fits' % nsources)
    
    # In[ ]:
    
    
    gaintables[0].gain
