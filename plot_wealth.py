from math import exp, isclose, log
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.graph_objects as go

DATA_FILE = Path(__file__).with_name("verteilungsbasierte-vermoegensbilanzen-xls-data.xlsx")
OUTPUT_FILE = Path(__file__).with_name("index.html")

GROUPS = [
    ("Quin.1", 0.20),
    ("Quin.2", 0.20),
    ("Decile D5", 0.10),
    ("Decile D6", 0.10),
    ("Decile D7", 0.10),
    ("Decile D8", 0.10),
    ("Decile D9", 0.10),
    ("Decile D10", 0.10),
]

ASSET_COMPONENTS = [
    "Housing Wealth",
    "Deposits",
    "Life insurance and annuity entitlements",
    "Investment Fund Shares",
    "Listed Shares",
    "Debt Securities",
    "Non Financial Business Wealth",
    "Financial Business Wealth",
    "Non-life insurance technical provisions",
]

LIABILITY_COMPONENTS = ["Mortgages", "Other Debt"]

ASSET_COLORS = [
    "#264653",
    "#2a9d8f",
    "#8ab17d",
    "#e9c46a",
    "#f4a261",
    "#e76f51",
    "#6d597a",
    "#b56576",
    "#84a59d",
]

LIABILITY_COLORS = ["#b56576", "#8d626f"]

DERIVED_GROUPS = {"Decile D5": ("Quin.3", "Decile D6", 2.0, -1.0)}

EVEN_LINE_GRADIENTS = (1.0,)
EVEN_LINE_COLORS = ("#2a9d8f",)
EVEN_LINE_DASHES = ("dot",)

WIDTH_SCALE = 5.0
GROUP_GAP = 0.0
GRID_STEP = 0.025
LAMBDA_SMOOTHNESS = 1.0
LAMBDA_EXPONENTIAL = 1.0
TOP_BETA_MULTIPLIER = 1.5


def read_table(path=DATA_FILE, quarter=None):
    frame = pd.read_excel(path)
    quarters = [column for column in frame.columns if isinstance(column, str) and " Q" in column]
    selected = [quarter] if quarter else quarters
    table = {}
    for row in frame.to_dict("records"):
        key = (row["Item"], row["Breakdown"], row["Level"], row["Unit"])
        table[key] = {
            "type": row["Type"],
            "demographics": row["Demographics"],
            "series_key": row["series_key"],
            "values": {q: row[q] for q in selected},
        }
    return table


def value(table, item, breakdown, quarter, level="Per household", unit="EUR Thousand"):
    amount = table[(item, breakdown, level, unit)]["values"][quarter]
    return amount * 1000 if unit == "EUR Thousand" else amount


def group_value(table, item, group, quarter):
    if group in DERIVED_GROUPS:
        first, second, first_weight, second_weight = DERIVED_GROUPS[group]
        return first_weight * value(table, item, first, quarter) + second_weight * value(
            table, item, second, quarter
        )
    return value(table, item, group, quarter)


def stacked_values(table, quarter, items, names):
    values = {item: [group_value(table, item, name, quarter) for name in names] for item in items}
    bases = {}
    running = [0.0] * len(names)
    for item in items:
        bases[item] = list(running)
        running = [total + change for total, change in zip(running, values[item])]
    return values, bases


def check_components(table, quarter):
    for breakdown, _ in GROUPS:
        assets = sum(group_value(table, item, breakdown, quarter) for item in ASSET_COMPONENTS)
        liabilities = sum(group_value(table, item, breakdown, quarter) for item in LIABILITY_COMPONENTS)
        total_assets = group_value(table, "Total Assets", breakdown, quarter)
        total_liabilities = group_value(table, "Total Liabilities", breakdown, quarter)
        if not isclose(assets, total_assets, rel_tol=1e-6):
            raise ValueError(f"{breakdown}: components {assets} != Total Assets {total_assets}")
        if not isclose(liabilities, total_liabilities, rel_tol=1e-6):
            raise ValueError(f"{breakdown}: components {liabilities} != Total Liabilities {total_liabilities}")


def group_positions():
    widths = [share * WIDTH_SCALE for _, share in GROUPS]
    left_edges = []
    centers = []
    cursor = 0.0
    for width in widths:
        left_edges.append(cursor)
        centers.append(cursor + width / 2)
        cursor += width + GROUP_GAP
    return widths, left_edges, centers, cursor


def current_curve(table, quarter, names, shares, widths, centers):
    amounts = [group_value(table, "Net Wealth", name, quarter) for name in names]
    count = len(amounts)
    total = sum(widths)
    steps = int(round(total / GRID_STEP))
    grid = np.linspace(0.0, total, steps + 1)
    size = len(grid)

    integral_matrix = np.zeros((count, size))
    left_edges = [center - width / 2 for center, width in zip(centers, widths)]
    for index, (left, width) in enumerate(zip(left_edges, widths)):
        start = int(round(left / GRID_STEP))
        end = int(round((left + width) / GRID_STEP))
        integral_matrix[index, start] = GRID_STEP / 2
        integral_matrix[index, start + 1 : end] = GRID_STEP
        integral_matrix[index, end] = GRID_STEP / 2
    targets = np.array([width * amount for width, amount in zip(widths, amounts)], dtype=float)

    derivative_matrix = np.diff(np.eye(size), axis=0)
    second_matrix = np.diff(derivative_matrix, axis=0)
    midpoint_matrix = 0.5 * (np.eye(size)[:-1] + np.eye(size)[1:])
    smoothness_matrix = second_matrix.T @ second_matrix

    shift = max(0.0, -min(amounts)) + 1.0
    shifted = np.array(amounts, dtype=float) + shift
    log_slopes = [
        (np.log(shifted[index + 1]) - np.log(shifted[index])) / (centers[index + 1] - centers[index])
        for index in range(count - 1)
    ]
    center_slopes = [log_slopes[0]]
    center_slopes.extend((log_slopes[index - 1] + log_slopes[index]) / 2 for index in range(1, count - 1))
    center_slopes.append(log_slopes[-1])
    midpoints = (grid[:-1] + grid[1:]) / 2
    beta = np.interp(midpoints, centers, center_slopes) * GRID_STEP
    beta[midpoints >= left_edges[-1]] *= TOP_BETA_MULTIPLIER

    scale = float(np.max(np.abs(targets)))
    coefficients = cp.Variable(size)
    problem = cp.Problem(
        cp.Minimize(
            LAMBDA_SMOOTHNESS * cp.quad_form(coefficients, cp.psd_wrap(smoothness_matrix))
            + LAMBDA_EXPONENTIAL
            * cp.sum_squares(
                derivative_matrix @ coefficients - cp.multiply(beta, midpoint_matrix @ coefficients)
            )
        ),
        [
            integral_matrix @ coefficients == targets / scale,
            derivative_matrix @ coefficients >= 0,
            second_matrix @ coefficients >= 0,
        ],
    )
    problem.solve(solver=cp.CLARABEL)
    if coefficients.value is None:
        raise RuntimeError(f"Curve optimization failed: {problem.status}")

    y_points = (coefficients.value * scale).tolist()
    x_points = grid.tolist()
    group_indexes = np.clip(np.searchsorted(left_edges, x_points, side="right") - 1, 0, count - 1)
    custom = [(names[index], shares[index]) for index in group_indexes]
    return x_points, y_points, custom


def build_figure(table, quarter):
    check_components(table, quarter)

    names = [name for name, _ in GROUPS]
    widths, left_edges, centers, cursor = group_positions()

    custom = list(zip(names, [share for _, share in GROUPS]))
    figure = go.Figure()

    asset_values, asset_bases = stacked_values(table, quarter, ASSET_COMPONENTS, names)
    liability_values, liability_bases = stacked_values(table, quarter, LIABILITY_COMPONENTS, names)

    for color, item in zip(ASSET_COLORS, ASSET_COMPONENTS):
        figure.add_trace(
            go.Bar(
                name=item,
                x=left_edges,
                y=asset_values[item],
                base=asset_bases[item],
                width=widths,
                offset=0,
                customdata=custom,
                marker_color=color,
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    + item
                    + ": €%{y:,.0f} per household<br>Population: %{customdata[1]:.0%}<extra></extra>"
                ),
            )
        )

    for color, item in zip(LIABILITY_COLORS, LIABILITY_COMPONENTS):
        figure.add_trace(
            go.Bar(
                name=item,
                x=left_edges,
                y=liability_values[item],
                base=liability_bases[item],
                width=widths,
                offset=0,
                customdata=custom,
                marker_color=color,
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    + item
                    + ": €%{y:,.0f} per household<br>Population: %{customdata[1]:.0%}<extra></extra>"
                ),
            )
        )

    figure.add_trace(
        go.Bar(
            name="Net Wealth",
            x=left_edges,
            y=[group_value(table, "Net Wealth", name, quarter) for name in names],
            base=0.0,
            width=widths,
            offset=0,
            customdata=custom,
            marker=dict(color="rgba(0,0,0,0)", line=dict(color="#111111", width=2)),
            hovertemplate=(
                "%{customdata[0]}<br>Net Wealth: €%{y:,.0f} per household"
                "<br>Population: %{customdata[1]:.0%}<extra></extra>"
            ),
        )
    )

    mean_wealth = value(table, "Net Wealth", "Total", quarter)
    total_width = sum(widths)
    line_x = [0.0, total_width]
    for gradient, color, dash in zip(EVEN_LINE_GRADIENTS, EVEN_LINE_COLORS, EVEN_LINE_DASHES):
        figure.add_trace(
            go.Scatter(
                name=f"Even distribution ±{gradient:.0%}",
                x=line_x,
                y=[mean_wealth * (1 - gradient + 2 * gradient * x / total_width) for x in line_x],
                mode="lines",
                line=dict(color=color, width=2, dash=dash),
                hovertemplate=f"Even ±{gradient:.0%}<br>€%{{y:,.0f}}<extra></extra>",
            )
        )

    shares = [share for _, share in GROUPS]
    current_x, current_y, current_custom = current_curve(
        table, quarter, names, shares, widths, centers
    )
    figure.add_trace(
        go.Scatter(
            name="Current distribution",
            x=current_x,
            y=current_y,
            mode="lines",
            line=dict(color="#b00020", width=3),
            customdata=current_custom,
            hovertemplate=(
                "%{customdata[0]}<br>Net Wealth: €%{y:,.0f} per household"
                "<br>Population: %{customdata[1]:.0%}<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            name="Arithmetic mean (50%)",
            x=[total_width / 2],
            y=[mean_wealth],
            mode="markers",
            marker=dict(color="#111111", size=8),
            hovertemplate="Arithmetic mean<br>€%{y:,.0f}<extra></extra>",
        )
    )

    figure.update_layout(
        barmode="overlay",
        title=dict(
            text=f"Net wealth per household by population group, {quarter} (Germany)",
            x=0.5,
            xanchor="center",
            y=1.0,
            yanchor="bottom",
        ),
        margin=dict(t=80, b=220, l=70, r=40),
        xaxis_title="Population group (bar width = share of population)",
        yaxis=dict(title="EUR per household", tickprefix="€"),
        xaxis=dict(
            tickmode="array",
            tickvals=centers,
            ticktext=[f"{name}<br>{share:.0%}" for name, (_, share) in zip(names, GROUPS)],
            range=[-0.3, cursor + 0.3],
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
        template="plotly_white",
    )

    return figure


def main():
    table = read_table()
    quarter = list(next(iter(table.values()))["values"])[-1]
    build_figure(table, quarter).write_html(OUTPUT_FILE, include_plotlyjs=True)
    print(f"Wrote {OUTPUT_FILE} ({quarter})")


if __name__ == "__main__":
    main()
