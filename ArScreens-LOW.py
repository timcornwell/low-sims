import numpy as numpy

from ArScreens import ArScreens
from data_models.polarisation import PolarisationFrame
from processing_components.image.operations import export_image_to_fits, create_image_from_array

if __name__ == '__main__':
    filename = 'low_screen.fits'
    r0 = 7000.0
    hiono = 3e5
    bmax = 3000.0
    diameter = 35.0
    
    bigD = 100000.0
    m = int(bmax / diameter)
    n = int(bigD / bmax)
    npixel = n * m
    print("Number of pixels %d" % (npixel))
    pscale = bigD / (n * m)
    print("Pixel size %.3f (m)" % pscale)
    print("Field of view %.1f (m)" % (npixel * pscale))
    speed = 150000.0 / 3600.0
    direction = 0.0
    print("Ionospheric velocity %.3f (m/s) at direction %.1f (degrees)" % (speed, direction))
    rate = 0.01
    alpha_mag = 0.9999
    ntimes = int(3000 * rate)
    lowparamcube = numpy.array([(r0, speed, direction, hiono)])
    
    my_screens = ArScreens(n, m, pscale, rate, lowparamcube, alpha_mag)
    my_screens.run(ntimes, verbose=True)
    
    from astropy.wcs import WCS
    
    nfreqwin = 1
    npol = 1
    frequency = [1e8]
    channel_bandwidth = [0.1e8]
    
    w = WCS(naxis=4)
    cellsize = pscale
    w.wcs.cdelt = [cellsize, cellsize, 1.0 / rate, channel_bandwidth[0]]
    w.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, ntimes // 2 + 1, 1.0]
    w.wcs.ctype = ['XX', 'YY', 'TIME', 'FREQ']
    w.wcs.crval = [0.0, 0.0, 0.0, frequency[0]]
    w.naxis = 4
    w.wcs.radesys = 'ICRS'
    w.wcs.equinox = 2000.0
    data = numpy.zeros([nfreqwin, ntimes, npixel, npixel])
    for i, screen in enumerate(my_screens.screens[0]):
        data[:, i, ...] = screen[numpy.newaxis, ...] - screen[numpy.newaxis, npixel//2, npixel//2]
    
    im = create_image_from_array(data, wcs=w, polarisation_frame=PolarisationFrame("stokesI"))
    print(im)
    export_image_to_fits(im, filename)
