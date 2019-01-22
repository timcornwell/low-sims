JUPYTER ?= jupyter
TIMEOUT = 43200

NBs = low-sims-mpc-10000m-prep.ipynb \
    low-sims-mpc-10000m-iso-ICAL.ipynb \
    low-sims-mpc-10000m-noniso-ICAL.ipynb \
    low-sims-mpc-10000m-create_skymodel.ipynb \
    low-sims-mpc-10000m-noniso-MPCCAL.ipynb

SCREEN = screens/low_screen_5000.0r0_0.100rate.fits

HTMLs = $(NBs:.ipynb=.html)

PDFs = $(NBs:.ipynb=.pdf)

.PHONY: html
html: $(HTMLs)

.PHONY: pdf
pdf: $(PDFs)

%fits:
	cd screens;python ArScreens-LOW.py

%.html: %.ipynb ${SCREEN}
	jupyter nbconvert --execute --ExecutePreprocessor.timeout=$(TIMEOUT) --to html $<

%.pdf: %.ipynb ${SCREEN}
	jupyter nbconvert --execute --ExecutePreprocessor.timeout=$(TIMEOUT) --to pdf $<


.PHONY: clean

clean:
	rm -f ${PDFs} ${HTMLs} ${SCREEN}
