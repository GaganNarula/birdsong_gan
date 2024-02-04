"""A Gradio app to inspect spectrograms and listen to audio samples."""
import os
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from birdsong_gan.audio_utils import (
    make_spectrogram,
    plot_spectrogram,
    play_audio_sounddevice,
)
from birdsong_gan.data.data_utils import DataExplorer


# create a DataExplorer instance
# set the path to the dataset later
de = DataExplorer()


with gr.Blocks(theme="soft") as demo:

    # we need two columns, parameters and selection boxes, dropdowns, etc. in the first column,
    # and the output in the second column
    with gr.Row(equal_height=False):

        # first column
        with gr.Column(scale=4):

            # need a file explorer to select dataset
            pathtods = gr.FileExplorer(label="Select Dataset", type="folder", root_dir="/home/"
            # input parameters
            bird = gr.Dropdown(
                label="Bird",
                choices=["bird1", "bird2", "bird3"],
                default="bird1",
            )
            age_lower = gr.Slider(
                label="Lower Age Bound",
                minimum=0,
                maximum=100,
                default=0,
                step=1,
            )
            age_upper = gr.Slider(
                label="Upper Age Bound",
                minimum=0,
                maximum=100,
                default=100,
                step=1,
            )
            recording_date_lower = gr.DatePicker(
                label="Lower Recording Date Bound", default="2021-01-01"
            )
            recording_date_upper = gr.DatePicker(
                label="Upper Recording Date Bound", default="2021-12-31"
            )

        # second column
        with gr.Column(scale=1):
            # action button
            action_btn = gr.Button("Go!")