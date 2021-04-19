import streamlit as st
from summarizer import T5Summarizer
import os


def setup_ui():
    # パラメーターの設定
    hparams = {
        "max_input_length": 512,
        "max_target_length": 64,
        "model_dir": os.path.join(os.path.dirname(__file__), "model")
    }

    # メイン部分を作成
    body = st.text_area(label="文字を入力してください", value="")
    start_btn = st.button("解析開始")
    if start_btn and len(body) > 0:
        with st.spinner("解析中..."):
            response = inference(text=body, hparams=hparams)
        st.success("解析完了.")

        st.write("要約文 : {}".format(response))


def inference(text:str, hparams:dict) -> str:
    summarizer = T5Summarizer(hparams=hparams)
    outputs = summarizer.predict(inputs=[text])
    return outputs[0]


def main():
    setup_ui()


if __name__ == "__main__":
    main()
