import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.set_page_config(
    page_title="Графики",
)

st.sidebar.write('Графики построены по датасету "Вино"')

df = pd.read_csv('wine.csv', sep=';')
fig, axs = plt.subplots(2, 2)
df.plot.scatter(x='alcohol', y='residual sugar', ax = axs[0,0])
a = df.corr()
df.hist('quality', ax = axs[1,0])
df.boxplot('alcohol', ax = axs[0,1])
axs[1,1] = sns.heatmap(a)
plt.subplots_adjust(
        wspace=1,
        hspace=1
        )
plt.savefig('graphs.png')
img = Image.open('graphs.png')
st.image(img)
