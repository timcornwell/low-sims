# Simulation of non-isoplanatic self-calibration in ARL

This performs the calculations in 
[SDP memo 97](https://confluence.ska-sdp.org/download/attachments/201294049/Direction%20Dependent%20Self%20Calibration%20in%20ARL%20-%20signed.pdf?version=1&modificationDate=1547631494000&api=v2).

- A phase screen at the height of the ionosphere (300km) moving at 150 km/s is 
simulated. 
- A sky model is calculated from GLEAM. 
- Nonisoplanatic visibilities from the screen are calculated taking into account the two pierce points for a given 
baseline. 
- Isoplanatic visibilities are calculated using just a single pierce point for all stations.
- ICAL is run on both isoplanatic and non-isoplanatic data.
- MPCCAL is run on the non-isoplanatic data, starting from the model and gaintable from the
ICAL.

#### Setup of required packages
- Install ARL from https://github.com/SKA-ScienceDataProcessor/algorithm-reference-library
- Install ARatmospy from https://github.com/shrieks/ARatmospy

#### Steps

The processing steps are in a python file and jupyter notebooks:
- To initialise the phase screen file: cd screens;python ArScreens-LOW.py
- To simulate the data: low-sims-mpc-10000-prep.ipynb
- To perform isoplanatic selfcal (ICAL) of isoplanatic data: low-sims-mpc-10000-iso.ipynb
- To perform isoplanatic selfcal (ICAL) of nonisoplanatic data: low-sims-mpc-10000-noniso.ipynb
- To create skymodel for mpc: low-sims-mpc-10000-create_skymodel.ipynb
- To perform non-isoplanatic selfcal (MPCCAL) of nonisoplanatic data: low-sims-mpc-10000-mpccal.ipynb

The output files are placed in a sub-directory results. The images are standard FITS files, and
all the other files are HDF5. Figures are placed in the figures sub-directory.

#### Notes

- Change the number of Dask workers (n_workers) in each file to a value suitable for your machine. 
The default is 16 workers.
