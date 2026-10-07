# One command reproduces every number: `make all`.
PY := .venv/bin/python
export PYTHONPATH := src

.PHONY: all data panels analysis report examples test clean-derived

all: data panels analysis report examples

test:
	$(PY) -m unittest discover -s tests

data:
	$(PY) -m threeleaks.download

panels:
	$(PY) -m threeleaks.panels
	$(PY) -m threeleaks.validate

analysis:
	$(PY) -m threeleaks.commission
	$(PY) -m threeleaks.skill
	$(PY) -m threeleaks.style
	$(PY) -m threeleaks.timing
	$(PY) -m threeleaks.selection
	$(PY) -m threeleaks.decay
	$(PY) -m threeleaks.closet
	$(PY) -m threeleaks.external
	$(PY) -m threeleaks.synthesis
	$(PY) tool/build_scorecard.py

report:
	$(PY) -m threeleaks.numbers
	$(PY) -m threeleaks.report_tables
	cd report && pdflatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null

# Each method by hand on two real funds, checked against the pipeline (results/examples.json).
examples:
	$(PY) -m threeleaks.examples > /dev/null

clean-derived:
	rm -f results/tables/*.csv results/figures/*.png results/numbers.json report/numbers.tex
