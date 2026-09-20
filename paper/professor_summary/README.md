# Professor summary (2026-09-20)

`summary.pdf` — what HeROsim does, how the labels are made, the model, the evaluation design
and where the answer stands, with figures. Rebuild:

```bash
PIPENV_IGNORE_VIRTUALENVS=1 PYTHONPATH=$PWD pipenv run python3 paper/professor_summary/make_figures.py
cd paper/professor_summary && pdflatex summary.tex && pdflatex summary.tex
```

`data/` holds the gate summaries the figures are computed from (fetched from datalab, stripped to
scalars); the statistics are computed by the registered readers under `scripts_cosim/`.
