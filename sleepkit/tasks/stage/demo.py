import datetime
import random

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import neuralspot_edge as nse

from ...defines import TaskParams
from ...backends import BackendFactory
from ...features import H5Dataloader
from .metrics import compute_sleep_stage_durations
from .utils import subject_data_preprocessor


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


def demo(params: TaskParams):
    """Run sleep stage classification demo.

    Args:
        params (TaskParams): Demo parameters
    """
    logger = nse.utils.setup_logger(__name__)

    bg_color = "rgba(38,42,50,1.0)"
    plotly_template = "plotly_dark"

    sleep_classes = sorted(set(params.class_map.values()))
    class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]
    class_colors = get_stage_color_map(params.num_classes)

    logger.debug("Setting up")

    # Load backend inference engine
    runner = BackendFactory.get(params.backend)(params=params)

    # Load features
    dataloader = H5Dataloader(
        path=params.feature.save_path,
        frame_size=params.frame_size,
        feat_key=params.feature.feat_key,
        label_key=params.feature.label_key,
        mask_key=params.feature.mask_key,
        feat_cols=params.feature.feat_cols,
        class_map=params.class_map,
    )

    # Load entire subject's features
    subject_id = random.choice(dataloader.subject_ids)
    logger.debug(f"Loading subject {subject_id} data")
    features, _, _ = dataloader.load_subject_data(subject_id=subject_id, preprocessor=None)
    x, y_true, y_mask = dataloader.load_subject_data(subject_id=subject_id, preprocessor=subject_data_preprocessor)

    # Run inference
    runner.open()
    logger.debug("Running inference")
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
    logger.debug("Generating report")
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
        name = f"FEAT{f + 1}"
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
    if params.display_report:
        fig.show()
    logger.debug(f"Report saved to {params.job_dir / 'demo.html'}")

    fig = plt.figure(layout="constrained", figsize=(10, 6))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1])

    # ax1.set_title("Sleep Stage Plot")
    ax1.set_xlabel("Time")
    ax1.set_yticks(range(len(class_names)))
    ax1.set_yticklabels(class_names)
    ax1.set_ylim(len(class_names) - 0.5, -0.5)
    ax1.grid(True)
    for i in range(1, len(sleep_bounds)):
        start, stop = sleep_bounds[i - 1], sleep_bounds[i]
        label = y_pred[start]
        color = class_colors.get(label, None)
        ax1.plot(ts[start:stop], y_pred[start:stop], color=color, linewidth=8)
    # END FOR
    # Rotate x-axis labels 45 degrees
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # # Data Plot
    # for f in range(features.shape[1]):
    #     name = f"FEAT{f+1}"
    #     feat_y = np.where(y_mask == 1, features[:, f], np.nan)
    #     ax1.plot(ts, feat_y, label=name, alpha=0.5)
    # # END FOR

    # Cycle Plot
    # ax21.set_title("Sleep Stage Distribution")
    ax21.pie(
        [pred_sleep_durations.get(c, 0) for c in sleep_classes],
        labels=class_names,
        autopct="%1.1f%%",
        colors=[class_colors.get(c, None) for c in sleep_classes],
    )

    # Efficiency Plot
    # ax22.set_title("Sleep Stage Durations")
    ax22.barh(class_names, class_durations, color=[class_colors.get(c, "red") for c in sleep_classes])
    for i, duration in enumerate(class_durations):
        duration_str = f"{duration:0.0f} min"
        if duration > 60:
            duration_str = f"{duration / 60:0.1f} hr"
        ax22.text(duration, i, duration_str, ha="left", va="center")

    plt.suptitle(f"Sleep {'Detect' if params.num_classes == 2 else 'Stage'} Demo (subject {subject_id})")

    fig.tight_layout()
    fig.savefig(params.job_dir / "demo.png")
