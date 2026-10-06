# One command reproduces every number: `make all`.
PY := .venv/bin/python
export PYTHONPATH := src

.PHONY: all data panels analysis report clean-derived

all: data panels analysis report

data:
	$(PY) -m threeleaks.download

panels:
	$(PY) -m threeleaks.panels

analysis:
	$(PY) -m threeleaks.commission
	$(PY) -m threeleaks.skill
	$(PY) -m threeleaks.timing
	$(PY) -m threeleaks.selection
	$(PY) -m threeleaks.synthesis

report:
	$(PY) -m threeleaks.numbers
	cd report && pdflatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null

clean-derived:
	rm -f results/tables/*.csv results/figures/*.png results/numbers.json report/numbers.tex
