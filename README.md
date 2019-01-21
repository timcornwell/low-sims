##This performs the calculations in SDP memo 97 _"Direction Dependent Self Calibration in ARL"_

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

The processing steps are in a set of jupyter notebooks:
- To initialise the phase screen file: cd screens;python ArScreens-LOW.py
- To simulate the data: low-sims-mpc-10000-prep.ipynb
- To perform isoplanatic selfcal of isoplanatic data: low-sims-mpc-10000-iso.ipynb
- To perform isoplanatic selfcal of nonisoplanatic datalow-sims-mpc-10000-noniso.ipynb
- To create skymodel for mpc: low-sims-mpc-10000-create_skymodel.ipynb
- To run MPCCal: low-sims-mpc-10000-mpccal.ipynb

The output files are placed in a sub-directory results. The images are standard FITS files, and
all the other files are HDF5. Figures are placed in the figures sub-directory.


