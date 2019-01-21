# Template for making Jupyter notebooks. Will get included
# by Makefiles in subdirectories.

PDFs=  $(NBs:.ipynb=.pdf)

JUPYTER ?= jupyter
TIMEOUT = 43200

.PHONY: pdf
pdf: $(PDFs)

%.pdf: %.ipynb
	jupyter nbconvert --execute --ExecutePreprocessor.timeout=$(TIMEOUT) --to pdf $<

.PHONY: clean

clean:
	rm -f ${PDFs}
