# fpp3-python-readalong

These notes are a Python-centered read-along of the excellent [Forecasting: Principles and Practice](https://otexts.com/fpp3/index.html) by Rob J Hyndman and George Athanasopoulos [1].

Please find the [table of contents](https://nbviewer.jupyter.org/github/zgana/fpp3-python-readalong/blob/master/Contents.ipynb) on Jupyter nbviewer.


[1]  Hyndman, R.J., & Athanasopoulos, G. (2019) Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia. OTexts.com/fpp3. Accessed on 2020-07-20.


## Running the code in 2024+

I've long wanted to rework this for clarity and completeness (the book has been updated since 2020) as well as improved Python style.  Unfortunately, while I got started at some point, I never followed all the way through on the rewrite.

In the meantime, I've occasionally been asked how the code can be run, or [where to find the data](https://github.com/zgana/fpp3-python-readalong/issues/2).  So now (2024-Sep) I'm posting a minimal update to make the notebooks easily runnable.  Just follow these steps:

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Install a copy of Python 3.8: `uv python install 3.8`
3. Set up a venv: `uv venv --python 3.8`
4. Install the dependencies: `uv pip install -r requirements.txt`
5. Run Jupyter: `.venv/bin/jupyter-lab`
