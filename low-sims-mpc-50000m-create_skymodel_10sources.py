# coding: utf-8

# ## Model partition calibration example

# In[1]:


import logging

import numpy
from matplotlib import pylab as pylab
from matplotlib import pyplot as plt

from data_models.data_model_helpers import import_blockvisibility_from_hdf5
from processing_library.image.operations import copy_image
from wrappers.serial.image.operations import import_image_from_fits

# In[2]:
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


# In[3]:


pylab.rcParams['figure.figsize'] = (14.0, 14.0)
pylab.rcParams['image.cmap'] = 'rainbow'

# ### Read the previously prepared observation: 10 minutes at transit, with 10s integration.

# In[4]:


rmax = 10000.0
blockvis = import_blockvisibility_from_hdf5('results/low-sims-skymodel-noniso-blockvis_rmax10000.0.hdf5')
from data_models.data_model_helpers import import_skycomponent_from_hdf5, import_gaintable_from_hdf5

recovered_ical_components = import_skycomponent_from_hdf5(
    'results/low-sims-mpc-noniso-ical-components_rmax%.1f.hdf5' % rmax)
ical_deconvolved0 = import_image_from_fits('results/low-sims-mpc-noniso-ical-deconvolved_rmax%.1f.fits' % rmax)
gaintable = import_gaintable_from_hdf5('results/low-sims-mpc-noniso-ical-gaintable_rmax%.1f.hdf5' % rmax)

# In[5]:


nfreqwin = len(blockvis.frequency)
ntimes = len(blockvis.time)
frequency = blockvis.frequency
times = blockvis.time
phasecentre = blockvis.phasecentre

# ### Remove weaker of components that are too close (0.02 rad)

# In[6]:


from wrappers.arlexecute.skycomponent.operations import remove_neighbouring_components, voronoi_decomposition

idx, filtered_ical_components = remove_neighbouring_components(recovered_ical_components, 0.2)
print(len(filtered_ical_components))
nsources = len(filtered_ical_components)

vor, vor_array = voronoi_decomposition(ical_deconvolved0, filtered_ical_components)
vor_image = copy_image(ical_deconvolved0)
vor_image.data[...] = vor_array

iteration = 0
from wrappers.arlexecute.skymodel.operations import initialize_skymodel_voronoi

theta_list = initialize_skymodel_voronoi(ical_deconvolved0, filtered_ical_components, gt=gaintable)
from data_models.data_model_helpers import export_skymodel_to_hdf5

export_skymodel_to_hdf5(theta_list, "mpc-skymodel_%dsources_iteration%d_rmax%.1f.hdf5" % (nsources, iteration, rmax))
