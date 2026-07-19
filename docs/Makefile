# Minimal makefile for Sphinx documentation
#
# Usage:
#   make html
#
# Uses `python3 -m sphinx` so you don't need sphinx-build on PATH.

SPHINXOPTS    ?=
SPHINXBUILD   ?= python3 -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help cleanhtml Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Full rebuild — use this after changing the toctree (new/renamed pages).
# Incremental builds can leave stale sidebars on older pages.
cleanhtml:
	rm -rf "$(BUILDDIR)"
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
