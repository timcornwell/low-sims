# Template for making Jupyter notebooks. Will get included
# by Makefiles in subdirectories.

RSTs=  $(NBs:.ipynb=.rst)

PDFs=  $(NBs:.ipynb=.pdf)

JUPYTER ?= jupyter
TIMEOUT = 43200

.PHONY: rst
rst: $(RSTs)

.PHONY: pdf
pdf: $(PDFs)

%.rst: %.ipynb
	jupyter nbconvert --execute --ExecutePreprocessor.timeout=$(TIMEOUT) --to rst $<

%.pdf: %.ipynb
	jupyter nbconvert --execute --ExecutePreprocessor.timeout=$(TIMEOUT) --to pdf $<

.PHONY: clean

clean:
	rm -f ${RSTs} ${PDFs}
