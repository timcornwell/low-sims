# coding: utf-8

# ## Model partition calibration example

# In[1]:


import logging

import numpy
from data_models.data_model_helpers import export_skymodel_to_hdf5
from data_models.data_model_helpers import import_blockvisibility_from_hdf5
from data_models.data_model_helpers import import_skymodel_from_hdf5
from processing_components.simulation.ionospheric_screen import grid_gaintable_to_screen
from processing_components.skymodel.operations import calculate_skymodel_equivalent_image, \
    update_skymodel_from_gaintables
from processing_library.image.operations import copy_image
from workflows.arlexecute.calibration.calibration_arlexecute import calibrate_list_arlexecute_workflow
from workflows.arlexecute.imaging.imaging_arlexecute import deconvolve_list_arlexecute_workflow
from workflows.serial.imaging.imaging_serial import deconvolve_list_serial_workflow
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
from wrappers.serial.skycomponent.operations import find_skycomponents

if __name__ == '__main__':
    
    n_workers = 128
    c = get_dask_Client(memory_limit=48 * 1024 * 1024 * 1024, n_workers=n_workers)
    arlexecute.set_client(c)
    
    # ### Read the previously prepared observation
    
    blockvis = import_blockvisibility_from_hdf5('low-sims-50000m-skymodel_noniso_blockvis_rmax50000.0.hdf5')
    Vobs = convert_blockvisibility_to_visibility(blockvis)
    
    # ### Initialization phase
    
    # #### Read the previous iteration of skymodels, $\theta_p^{(n)}$. Each skymodel will contain a mask based on the decomposition, an image and gaintable derived from ICAL
    
    # In[6]:
    
    
    nsources = 10
    
    theta_list = import_skymodel_from_hdf5("mpc-skymodel_%dsources_iteration0.hdf5" % (nsources))
    model = copy_image(theta_list[0].image)
    model.data[...] = 0.0
    
    psf_obs = invert_list_arlexecute_workflow([Vobs], [model], context='2d', dopsf=True)
    psf_obs = arlexecute.compute(psf_obs, sync=True)
    
    for iteration in range(1, 100):
        
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
        
        export_image_to_fits(combined_dirty_image, 'low-mpc-residual_image_%dsources_iteration%d.fits' %
                             (nsources, iteration))

        deconvolved_list = [arlexecute.execute(deconvolve_list_serial_workflow)([dirty_all_cal[ism]], [psf_obs[0]],
                                                                                [model], mask=sm.mask,
                                                                                algorithm='msclean',
                                                                                scales=[0, 3, 10], niter=100,
                                                                                fractional_threshold=0.3,
                                                                                threshold=0.3 * peak,
                                                                                gain=0.1, psf_support=128)
                            for ism, sm in enumerate(theta_list)]

        deconvolved_list = arlexecute.compute(deconvolved_list, sync=True)

        combined_model = calculate_skymodel_equivalent_image(theta_list)
        print(qa_image(combined_model, context='Combined model'))
        
        Vpredicted_list = predict_skymodel_list_arlexecute_workflow(future_Vobs, theta_list, context='2d', docal=True)
        result = calibrate_list_arlexecute_workflow(Vdatamodel_list, Vpredicted_list,
                                                    calibration_context='T',
                                                    iteration=0, global_solution=False)
        Vcalibrated, gaintable_list = arlexecute.compute(result, sync=True)
        
        theta_list = update_skymodel_from_gaintables(theta_list, gaintable_list, calibration_context='T')
        
        export_skymodel_to_hdf5(theta_list, "mpc-skymodel_%dsources_iteration%d.hdf5" % (nsources, iteration))
    
    # In[40]:
    
    
    export_image_to_fits(combined_model, 'low-mpc_%dsources-deconvolved.fits' % nsources)
    
    # In[7]:
    
    
    from workflows.arlexecute.imaging.imaging_arlexecute import restore_list_arlexecute_workflow
    
    result = restore_list_arlexecute_workflow([combined_model], psf_obs, [(combined_dirty_image, 0.0)])
    
    # In[8]:
    
    
    result = arlexecute.compute(result, sync=True)
    
    # In[9]:
    
    
    print(qa_image(result[0], context='MPC restored image'))
    
    # In[39]:
    
    
    recovered_mpccal_components = find_skycomponents(result[0], fwhm=2, threshold=0.14, npixels=12)
    print(len(recovered_mpccal_components))
    print(recovered_mpccal_components[0])
    from data_models.data_model_helpers import export_skycomponent_to_hdf5
    
    export_skycomponent_to_hdf5(recovered_mpccal_components, 'low-mpc_%dsources-components.hdf5' % nsources)
    export_image_to_fits(result[0], 'low-mpc_%dsources-restored.fits' % nsources)
    
    # In[37]:
    
    
    from data_models.data_model_helpers import import_skymodel_from_hdf5
    
    gleam_skymodel_iso = import_skymodel_from_hdf5('low-sims-iso_skymodel.hdf5')
    from processing_components.skycomponent.operations import filter_skycomponents_by_flux
    
    bright_gleam = filter_skycomponents_by_flux(gleam_skymodel_iso.components, flux_min=0.4)
    
    
    def max_flux(elem):
        return numpy.max(elem.flux)
    
    
    sorted_bright_gleam = sorted(bright_gleam, key=max_flux, reverse=True)
    
    from wrappers.serial.skycomponent.operations import find_skycomponent_matches
    
    matches = find_skycomponent_matches(recovered_mpccal_components, sorted_bright_gleam, tol=1e-3)
    oldscreen = import_image_from_fits('low_screen_5000.0r0_0.100rate.fits')
    from processing_components.image.operations import create_empty_image_like
    
    newscreen = create_empty_image_like(oldscreen)
    gaintables = [th.gaintable for th in theta_list]
    newscreen, weights = grid_gaintable_to_screen(blockvis, gaintables, newscreen)
    print(qa_image(newscreen))
    export_image_to_fits(newscreen, 'mpccal_%dsources_screen.fits' % nsources)
    export_image_to_fits(weights, 'mpccal_%dsources_screen_weights.fits' % nsources)
    
