import streamlit as st
from dataset import *

BIRDHDFPATH = '/media/songbird/datapartition/mdgan_training_input_with_age_HDF'

st.title('Birdsong spectrogram explorer')

st.markdown(''' This dashboard is used to explore birdsong spectrograms in the dataset''')

# find all birds
available_birds = os.listdir(BIRDHDFPATH)

birdselect = st.sidebar.selectbox('BIRDS',available_birds)