# coding: utf-8

# ## Simulate non-isoplanatic imaging for LOW at 100MHz.
# 
# ### A set of model components are drawn from GLEAM. An ionospheric screen model is used to calculate the pierce points of the two stations in an interferometer for a given component. The model visibilities are calculated directly, and screen phase applied to obtain the corrupted visibility.

# In[1]:


import logging

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from data_models.polarisation import PolarisationFrame
from wrappers.arlexecute.image.operations import smooth_image
from wrappers.arlexecute.skycomponent.operations import insert_skycomponent
from wrappers.arlexecute.visibility.coalesce import convert_blockvisibility_to_visibility
from wrappers.serial.image.operations import export_image_to_fits
from wrappers.serial.image.operations import qa_image
from wrappers.serial.imaging.base import create_image_from_visibility, advise_wide_field
from wrappers.serial.imaging.primary_beams import create_low_test_beam
from wrappers.serial.simulation.testing_support import create_named_configuration, \
    create_low_test_skycomponents_from_gleam
from wrappers.serial.skycomponent.operations import apply_beam_to_skycomponent
from wrappers.serial.skycomponent.operations import filter_skycomponents_by_flux
from wrappers.serial.visibility.base import create_blockvisibility


# In[2]:
def init_logging():
    logging.basicConfig(filename='low-sims-mpc-smodel.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)


init_logging()
log = logging.getLogger()


def lprint(*args):
    log.info(*args)
    print(*args)

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
lowcore = create_named_configuration('LOWBD2', rmax=rmax)

blockvis = create_blockvisibility(
    lowcore,
    times,
    frequency=frequency,
    channel_bandwidth=channel_bandwidth,
    weight=1.0,
    phasecentre=phasecentre,
    polarisation_frame=PolarisationFrame("stokesI"),
    zerow=True)

# ### Find sampling, image size, etc

# In[5]:


wprojection_planes = 1
vis = convert_blockvisibility_to_visibility(blockvis)
advice = advise_wide_field(vis, guard_band_image=2.0, delA=0.02)

cellsize = advice['cellsize']
vis_slices = advice['vis_slices']
npixel = advice['npixels2']

# ### Generate the model from the GLEAM catalog, including application of the primary beam.

# In[6]:


flux_limit = 0.2
dft_threshold = 10.0
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

all_components = apply_beam_to_skycomponent(original_gleam_components, beam)
all_components = filter_skycomponents_by_flux(all_components, flux_min=flux_limit)

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
export_image_to_fits(smodel, 'low-sims-smodel_rmax%.1f.fits' % rmax)
