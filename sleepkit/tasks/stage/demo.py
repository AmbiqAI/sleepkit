import datetime
import random

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from ...defines import SKDemoParams
from ...rpc import BackendFactory
from ...utils import setup_logger
from .metrics import compute_sleep_stage_durations
from .utils import load_dataset

logger = setup_logger(__name__)


def get_stage_color_map(num_classes) -> dict[int, str]:
    """Get color map for sleep stages

    Args:
        num_classes (int): Number of sleep stages

    Returns:
        dict[int, str]: Color map
    """
    gray, blue, navy, purple, red = "gray", "#11acd5", "#1f1054", "#ce6cff", "#d62728"
    if num_classes == 2:
        return {0: gray, 1: blue}
    if num_classes == 3:
        return {0: gray, 1: blue, 2: red}
    if num_classes == 4:
        return {0: gray, 1: blue, 2: purple, 3: red}
    if num_classes == 5:
        return {0: gray, 1: navy, 2: blue, 3: purple, 4: red}
    raise ValueError("Invalid number of classes")


def demo(params: SKDemoParams):
    """Run sleep stage classification demo.

    Args:
        params (SKDemoParams): Demo parameters
    """
    bg_color = "rgba(38,42,50,1.0)"
    plotly_template = "plotly_dark"

    sleep_classes = sorted(list(set(params.class_map.values())))
    class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]
    class_colors = get_stage_color_map(params.num_classes)

    logger.info("Setting up")

    # Load backend inference engine
    runner = BackendFactory.create(params.backend, params=params)

    # Load data handler
    ds = load_dataset(
        ds_path=params.ds_path,
        frame_size=params.frame_size,
        dataset=params.dataset,
    )

    # Load entire subject's features
    subject_id = random.choice(ds.subject_ids)
    logger.info(f"Loading subject {subject_id} data")
    features, _, _ = ds.load_subject_data(subject_id=subject_id, normalize=False)
    x, y_true, y_mask = ds.load_subject_data(subject_id=subject_id, normalize=True)

    # max_xlen = int((4*24*1)/params.sampling_rate)
    # features = features[:max_xlen]
    # x = x[:max_xlen]
    # y_true = y_true[:max_xlen]
    # y_mask = y_mask[:max_xlen]

    y_true = np.vectorize(params.class_map.get)(y_true)

    # Run inference
    runner.open()
    logger.info("Running inference")
    y_pred = np.zeros_like(y_true)
    for i in tqdm(range(0, x.shape[0], params.frame_size), desc="Inference"):
        if i + params.frame_size > x.shape[0]:
            start, stop = x.shape[0] - params.frame_size, x.shape[0]
        else:
            start, stop = i, i + params.frame_size
        runner.set_inputs(x[start:stop, :])
        runner.perform_inference()
        yy = runner.get_outputs()
        y_pred[start:stop] = np.argmax(yy, axis=-1).flatten()
    # END FOR
    runner.close()

    # Mask out bad data
    y_pred = y_pred[y_mask == 1]
    y_true = y_true[y_mask == 1]

    tod = datetime.datetime(2025, 5, 24, random.randint(12, 23), 00)
    ts = [tod + datetime.timedelta(seconds=(1.0 / params.sampling_rate) * i) for i in range(y_pred.size)]

    # Report
    logger.info("Generating report")
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"colspan": 2, "type": "xy", "secondary_y": True}, None],
            [{"type": "domain"}, {"type": "bar"}],
        ],
        subplot_titles=(None, None),
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )

    pred_sleep_durations = compute_sleep_stage_durations(y_pred)
    class_durations = [30 * pred_sleep_durations.get(c, 0) / 60 for c in sleep_classes]
    # pred_sleep_eff = compute_sleep_efficiency(pred_sleep_durations, class_mapping)

    # Sleep Stage Plot
    sleep_bounds = np.concatenate(([0], np.diff(y_pred).nonzero()[0] + 1))
    legend_groups = set()
    for i in range(1, len(sleep_bounds)):
        start, stop = sleep_bounds[i - 1], sleep_bounds[i]
        label = y_pred[start]
        name = class_names[label]
        color = class_colors.get(label, None)
        fig.add_trace(
            go.Scatter(
                x=[ts[start], ts[stop]],
                y=[label, label],
                mode="lines",
                line_shape="hv",
                name=name,
                legendgroup=name,
                showlegend=name not in legend_groups,
                line_color=color,
                line_width=4,
                fill="tozeroy",
                opacity=0.7,
            ),
            row=1,
            col=1,
        )
        legend_groups.add(name)
    # END FOR

    fig.update_yaxes(
        autorange=False,
        range=[max(sleep_classes) + 0.25, min(sleep_classes) - 0.25],
        ticktext=class_names,
        tickvals=list(range(len(class_names))),
        row=1,
        col=1,
        secondary_y=False,
    )

    # Data Plot
    for f in range(features.shape[1]):
        name = f"FEAT{f+1}"
        feat_y = np.where(y_mask == 1, features[:, f], np.nan)
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=feat_y,
                name=name,
                opacity=0.5,
                legendgroup="Features",
                legendgrouptitle_text="Features",
                visible="legendonly",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
    # END FOR

    # Cycle Plot | Efficiency Plot
    fig.add_trace(
        go.Pie(
            name="",
            labels=class_names,
            values=[pred_sleep_durations.get(c, 0) for c in sleep_classes],
            textinfo="label+percent",
            # texttemplate = "%{label}: %{percent}",
            textfont_size=15,
            hole=0.3,
            hoverinfo="none",  # "label+percent",
            showlegend=False,
            marker_colors=[class_colors.get(c, None) for c in sleep_classes],
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=class_durations,
            y=class_names,
            marker_color=[class_colors.get(c, "red") for c in sleep_classes],
            showlegend=False,
            text=[f"{t:0.0f} min" for t in class_durations],
            textposition="auto",
            hoverinfo="none",
            # hovertemplate="%{y}: %{x:0.2}H",
            orientation="h",
            name="",
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(title="Duration (min)", row=2, col=2)

    fig.update_layout(
        template=plotly_template,
        height=800,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(l=10, r=10, t=40, b=20),
        legend=dict(groupclick="toggleitem"),
        title=f"Sleep {'Detect' if params.num_classes == 2 else 'Stage'} Demo (subject {subject_id})",
    )
    fig.write_html(params.job_dir / "demo.html", include_plotlyjs="cdn", full_html=False)
    fig.show()

    logger.info(f"Report saved to {params.job_dir / 'demo.html'}")
