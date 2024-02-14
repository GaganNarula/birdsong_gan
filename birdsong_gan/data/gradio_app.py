"""A Gradio app to inspect spectrograms and listen to audio samples."""

import os
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from birdsong_gan.utils.audio_utils import (
    make_spectrogram,
    plot_spectrogram,
    play_audio_sounddevice,
)
from birdsong_gan.data.data_utils import DataExplorer

HOME = os.getenv("HOME")

DEFAULT_PATH = "/media/gagan/Gagan_external/songbird_data/age_resampled_hfdataset"


# create a DataExplorer instance
# set the path to the dataset later
de = DataExplorer(path_to_dataset=DEFAULT_PATH)


with gr.Blocks(theme="soft") as demo:

    gr.Markdown("## Spectrogram Explorer")
    # we need two columns, parameters and selection boxes, dropdowns, etc. in the first column,
    # and the output in the second column
    with gr.Row(equal_height=False):

        # first column
        with gr.Column(scale=6):

            # need a file explorer to select dataset
            select_dataset = gr.FileExplorer(label="Select Dataset", root_dir=HOME)
            submit_btn = gr.Button("Select")

            def get_dir(path):
                de.load_dataset(path)
                print(f"Loaded dataset from {path}")

            submit_btn.click(fn=get_dir, inputs=[select_dataset])

            all_unique_birds = de.get_all_unique_values("bird_name")
            all_unique_birds = [str(b) for b in all_unique_birds]

            # input parameters
            bird = gr.Dropdown(
                label="Bird",
                choices=all_unique_birds,
            )
            age_lower = gr.Slider(
                label="Lower Age Bound",
                minimum=20,
                maximum=190,
                value=30,
                step=1,
            )
            age_upper = gr.Slider(
                label="Upper Age Bound",
                minimum=0,
                maximum=200,
                value=90,
                step=1,
            )
            # recording_date_lower = gr.Textbox(
            #     label="Lower Recording Date Bound",
            #     value="2021-01-01",
            #     placeholder="Format: YYYY-MM-DD",
            # )
            # recording_date_upper = gr.Textbox(
            #     label="Upper Recording Date Bound",
            #     value="2021-12-31",
            #     placeholder="Format: YYYY-MM-DD",
            # )

        # second column
        with gr.Column(scale=1):
            # action button
            action_btn = gr.Button("Plot random spectrogram")

            def plot_random_spectrogram(
                bird: str = None,
                age_lower: int = None,
                age_upper: int = None,
                fig_size: tuple[int, int] = (12, 5),
            ):
                """Plot a random spectrogram from the dataset,"""
                if bird is None:
                    fig, _ = plt.subplots(figsize=fig_size, nrows=1, ncols=1)
                    return fig

                # get a subset of the dataset
                print("Getting bird subset from the dataset")
                subset = de.get_bird_subset(bird)

                # get the age range subset
                print("Getting age range subset from the dataset")
                subset = de.get_age_range_subset(subset, age_lower, age_upper)

                # get a random sample
                print("Getting a random sample from the dataset")
                sample = de.get_random_sample(ds=subset, seed=0, n=1)

                # get the audio
                audio = np.array(sample["audio"])

                print("Plotting the spectrogram")
                # make the spectrogram
                spectrogram = make_spectrogram(audio)

                # plot the spectrogram
                fig = plot_spectrogram(spectrogram, figsize=fig_size)

                return fig

            spect = gr.Plot(label="Spectrogram")
        # audio = gr.Audio(play_audio_sounddevice, type="file", label="Audio")

        action_btn.click(
            fn=plot_random_spectrogram,
            inputs=[bird, age_lower, age_upper],
            outputs=[spect],
        )


# launch the app
demo.launch(share=False)  # Local development
