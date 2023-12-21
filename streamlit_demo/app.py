import tempfile
import os
import pysinsy
import numpy as np
import soundfile as sf
import streamlit as st
from nnmnkwii.io import hts
from nnsvs.pretrained import create_svs_engine

st.title("NNSVS Demo")
st.markdown("Upload your .xml music file with text as input to make it sing.")

models = {
    "yoko": "r9y9/yoko_latest",
    "namine ritsu": "r9y9/namine_ritsu_diffusion"
}

voice_option = st.selectbox("Select the voice", models.keys())
uploaded_file = st.file_uploader("Choose a .xml music file", type=["xml", "musicxml", "lab"])
# ----------ここはオリジナル---------#
# 生成したい音声の個数
num = st.number_input("Input number that you want to create", value=1)
st.markdown("### Choose elements for fluc")
# 揺らぎ成分の選択
col = st.columns(3)
fluc = [False, False, False]
fluc_elem = ["lag", "duration", "f0", "mgc", "bap"]
timelag = col[0].checkbox(label=fluc_elem[0])
duration = col[1].checkbox(label=fluc_elem[1])
st.markdown("##### acoustic")
col1 = st.columns(3)
f0 = col1[0].checkbox(label=fluc_elem[2])
mgc = col1[1].checkbox(label=fluc_elem[3])
bap = col1[2].checkbox(label=fluc_elem[4])

fluc = [timelag, duration, f0, mgc, bap]
# print(fluc)
if st.button("synthesis") and uploaded_file:
    True_num = fluc.count(True)
    if True_num == 0:
        st.write(f"揺らぎなしの歌声を{num}個出力します")
    else:
        text = ""
        for i in range(len(fluc)):
            if fluc[i]:
                text += fluc_elem[i]
                text += "・"
        text = text[0:-1]
        st.write(f"{text}が揺らぎありの歌声を{num}個出力します")
    for i in range(num):
        with st.spinner(f"Synthesizing to wav {i + 1}/{num}"):
            # synthesize
            input_name, ext = os.path.splitext(uploaded_file.name)
            if ext == ".xml" or ext == ".musicxml":
                with tempfile.NamedTemporaryFile(suffix=".xml") as f:
                    f.write(uploaded_file.getbuffer())
                    contexts = pysinsy.extract_fullcontext(f.name)

                labels = hts.HTSLabelFile.create_from_contexts(contexts)

            elif ext == ".lab":
                with tempfile.NamedTemporaryFile(suffix=".lab") as f:
                    f.write(uploaded_file.getbuffer())
                    labels = hts.load(f.name)

            engine = create_svs_engine(models[voice_option])
            wav, sr = engine.svs(labels, fluc=fluc)

            # show audio player
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                name = f.name[:-4] + "_" + input_name
                if all(fluc):
                    if name[-1] == "":
                        name = name[0:-1]
                    for i in range(len(fluc)):
                        if fluc[i]:
                            name += "_"
                            name += fluc_elem[i]
                else:
                    if name[-1] == "_":
                        name += "no_fluc"
                    else:
                        name += "_no_fluc"
                name += ".wav"
                sf.write(name, wav.astype(np.int16), sr)
                with open(name, "rb") as wav_file:
                    st.audio(wav_file.read(), format="audio/wav")
