import datetime
import random

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import neuralspot_edge as nse

from ...defines import TaskParams
from ...backends import BackendFactory
from ...features import H5Dataloader
from .metrics import compute_apnea_hypopnea_index
from .utils import subject_data_preprocessor

logger = nse.utils.setup_logger(__name__)


def get_apnea_color_map(num_classes) -> dict[int, str]:
    """Get color map for apnea classes

    Args:
        num_classes (int): Number of classes

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
    """Sleep Apnea Demo

    Args:
        params (TaskParams): Task parameters

    """

    bg_color = "rgba(38,42,50,1.0)"
    plotly_template = "plotly_dark"

    apnea_classes = sorted(list(set(params.class_map.values())))
    # class_names = params.class_names or [f"Class {i}" for i in range(params.num_classes)]
    class_colors = get_apnea_color_map(params.num_classes)

    logger.debug("Setting up")

    # Load backend inference engine
    runner = BackendFactory.create(params.backend, params=params)

    # Load features
    dataloader = H5Dataloader(
        path=params.feature.save_path,
        frame_size=params.frame_size,
        feat_key=params.feature.feat_key,
        label_key=params.feature.label_key,
        mask_key=params.feature.mask_key,
        feat_cols=params.feature.feat_cols,
    )

    # Load entire subject's features
    subject_id = random.choice(dataloader.subject_ids)
    logger.debug(f"Loading subject {subject_id} data")
    features, _, _ = dataloader.load_subject_data(subject_id=subject_id, preprocessor=subject_data_preprocessor)
    x, y_true, y_mask = dataloader.load_subject_data(subject_id=subject_id, preprocessor=subject_data_preprocessor)

    y_true = np.vectorize(params.class_map.get)(y_true)

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

    tod = datetime.datetime(2025, 5, 24, random.randint(18, 22), 00)
    ts = [tod + datetime.timedelta(seconds=i / params.sampling_rate) for i in range(y_pred.size)]

    act_ahi = compute_apnea_hypopnea_index(
        y_true,
        min_duration=int(10 * params.sampling_rate),
        sample_rate=params.sampling_rate,
    )
    pred_ahi = compute_apnea_hypopnea_index(
        y_pred,
        min_duration=int(10 * params.sampling_rate),
        sample_rate=params.sampling_rate,
    )

    logger.debug(f"Actual AHI: {act_ahi:0.2f}")
    logger.debug(f"Predicted AHI: {pred_ahi:0.2f}")

    # Report
    logger.debug("Generating report")
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"colspan": 2, "type": "xy", "secondary_y": True}, None],
            [{"colspan": 2, "type": "bar"}, None],
        ],
        subplot_titles=(None, None),
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )

    # Plot predicted as positive and actual as negative
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=y_pred,
            mode="lines",
            name="Predicted",
            # #11acd5 with 50% alpha
            line_color="rgba(17, 172, 213, 0.5)",
            fill="tozeroy",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=-y_true,
            mode="lines",
            name="Actual",
            # #ce6cff with 50% alpha
            line_color="rgba(206, 108, 255, 0.5)",
            fill="tozeroy",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    fig.update_yaxes(
        autorange=True,
        range=[-min(apnea_classes) + 0.25, max(apnea_classes) - 0.25],
        row=1,
        col=1,
        secondary_y=False,
    )

    # Data Plot
    for f in range(features.shape[1]):
        name = f"FEAT{f + 1}"
        feat_y = np.where(y_mask == 1, x[:, f], np.nan)
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=feat_y,
                name=name,
                opacity=0.5,
                # legendgroup="Features",
                # legendgrouptitle_text="Features",
                # visible="legendonly",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
    # END FOR

    ahi_bins = [5, 15, 30, 100]
    ahi_group_names = ["MILD", "MODERATE", "SEVERE"]

    ahi_bars = [act_ahi, pred_ahi]
    fig.add_trace(
        go.Bar(
            x=[act_ahi, pred_ahi],
            y=["Actual", "Predicted"],
            marker_color=[class_colors.get(c, "red") for c in apnea_classes],
            showlegend=False,
            text=[f"{t:0.1f}" for t in ahi_bars],
            textposition="auto",
            hoverinfo="none",
            # hovertemplate="%{y}: %{x:0.2}H",
            orientation="h",
            name="AHI",
        ),
        row=2,
        col=1,
    )

    for ahi_bin, ahi_name in zip(ahi_bins, ahi_group_names):
        fig.add_shape(
            type="line",
            x0=ahi_bin,
            y0=-0.5,
            x1=ahi_bin,
            y1=1.5,
            line=dict(
                color="white",
                width=2,
                dash="dashdot",
            ),
            row=2,
            col=1,
        )
        fig.add_annotation(
            x=ahi_bin,
            y=0.5,
            textangle=-90,
            text=ahi_name,
            showarrow=False,
            xshift=10,
            row=2,
            col=1,
        )
    # END FOR
    fig.update_xaxes(title="Apnea Hypopnea Index (AHI)", row=2, col=1)

    fig.update_layout(
        template=plotly_template,
        height=800,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(l=10, r=10, t=40, b=20),
        legend=dict(groupclick="toggleitem"),
        title=f"Sleep Apnea Demo (subject {subject_id})",
    )

    fig.write_html(params.job_dir / "demo.html", include_plotlyjs="cdn", full_html=False)
    fig.show()

    logger.info(f"Report saved to {params.job_dir / 'demo.html'}")
