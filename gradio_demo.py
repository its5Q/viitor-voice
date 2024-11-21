import gradio as gr
import torchaudio
from io import BytesIO
from viitor_voice.utils.offline_inference import OfflineInference

# 初始化 TTS 引擎
tts_engine = OfflineInference(
    model_path='ZzWater/highres-voice-en',
    config_path='viitor_voice/inference_configs/en.json'
)

# 获取有效说话人列表
valid_speakers = list(tts_engine.prompt_map.keys())


# 定义生成音频函数
def generate_audio(text, speaker, speed):
    # 检查说话人是否有效
    if speaker not in valid_speakers:
        return f"Error: Invalid speaker. Please select one from {valid_speakers}."

    # 使用 OfflineInference 生成音频
    audios = tts_engine.batch_infer(text_list=[text], speaker=[speaker], speed=int(speed))

    return audios[0]


# Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## Qwen2 based streaming tts")

    with gr.Row():
        text_input = gr.Textbox(label="text", lines=5, placeholder="请输入要转换为语音的文本")
        speaker_input = gr.Dropdown(label="speaker", choices=valid_speakers, value=valid_speakers[0])
        speed_input = gr.Slider(label="speed", minimum=0, maximum=7, step=1, value=2)

    output_audio = gr.Audio(label="生成的音频", type="numpy")

    generate_button = gr.Button("生成语音")

    generate_button.click(
        fn=generate_audio,
        inputs=[text_input, speaker_input, speed_input],
        outputs=output_audio,
    )

# 启动服务
if __name__ == "__main__":
    demo.launch()
