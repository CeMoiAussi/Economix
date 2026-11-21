# Economix

Interactive Plotly chart of the German wealth distribution, built from `verteilungsbasierte-vermoegensbilanzen-xls-data.xlsx`.

## Files

- `plot_wealth.py` — builds the chart from the Excel data.
- `wealth_distribution.html` — generated interactive chart.
- `verteilungsbasierte-vermoegensbilanzen-xls-data.xlsx` — source data.
- `requirements.txt` — pinned Python dependencies.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

From Windows PowerShell:

```powershell
wsl -d Ubuntu bash -lc 'cd ~/Economix && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt'
```

## Generate the chart

```bash
.venv/bin/python plot_wealth.py
```

From Windows PowerShell:

```powershell
wsl -d Ubuntu bash -lc 'cd ~/Economix && .venv/bin/python plot_wealth.py'
```

Open `wealth_distribution.html` in a browser.

## Method

- Uses the latest available quarter from the spreadsheet.
- Values are per household and shown in EUR.
- Groups are `Quin.1`, `Quin.2`, `Decile D5`, and `Decile D6`–`Decile D10`; `Decile D5` is derived from `Quin.3` and `Decile D6`.
- The current-distribution curve is solved with a convex quadratic program: each group's integral matches its reported mean, the curve is monotone and convex, and an exponential preference is applied with a stronger weight in the top decile.
