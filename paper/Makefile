# Build paper.md into PDF / HTML with pandoc (not installed on the machine this was written on).
#
#   brew install pandoc          # or see https://pandoc.org/installing.html
#   brew install --cask basictex # optional, only for the PDF target (provides xelatex)
#   make -C paper html           # -> paper/paper.html (no LaTeX needed)
#   make -C paper pdf            # -> paper/paper.pdf
#
# Citations in paper.md are written out in author-year form and the reference list is part of paper.md,
# so no citeproc step is needed; references.bib holds the same works as BibTeX for reuse.
# Figures are referenced as ../reports/figures/*.png, so pandoc runs from this directory.
# paper.md contains one Devanagari phrase (Section 3.2): for the PDF, MAINFONT must cover Devanagari,
# e.g. `make pdf MAINFONT="Kohinoor Devanagari"` on macOS, or the phrase will render as blank boxes.

MAINFONT ?= Noto Sans

PANDOC ?= pandoc

.PHONY: all html pdf check clean
all: html

check:
	@command -v $(PANDOC) >/dev/null || { echo "pandoc not found: install it first (see the top of this Makefile)"; exit 1; }

html: check
	$(PANDOC) paper.md --standalone --embed-resources --mathml \
	  --metadata title="Language and Script Gaps in Local Language Models for Indian Users" \
	  -o paper.html

pdf: check
	$(PANDOC) paper.md --pdf-engine=xelatex -V geometry:margin=2.2cm -V fontsize=10pt \
	  -V mainfont="$(MAINFONT)" -V monofont="Menlo" \
	  -o paper.pdf

clean:
	rm -f paper.html paper.pdf
