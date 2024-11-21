import gradio as gr
import torchaudio
from io import BytesIO
from viitor_voice.utils.offline_inference import OfflineInference

tts_engine = OfflineInference(
    model_path='ZzWater/viitor-voice-en',
    config_path='viitor_voice/inference_configs/en.json'
)

valid_speakers = list(tts_engine.prompt_map.keys())


def generate_audio(text, speaker, speed):
    # check if speaker is valid
    if speaker not in valid_speakers:
        return f"Error: Invalid speaker. Please select one from {valid_speakers}."

    # Use OfflineInference to generate audio
    audios = tts_engine.batch_infer(text_list=[text], speaker=[speaker], speed=int(speed))

    return 24000, audios[0].numpy()[0]


with gr.Blocks() as demo:
    gr.Markdown("## VIITOR VOICE: LLM based streaming tts")

    with gr.Row():
        text_input = gr.Textbox(label="text", lines=5, placeholder="input text")
        speaker_input = gr.Dropdown(label="speaker", choices=valid_speakers, value=valid_speakers[0])
        speed_input = gr.Slider(label="speed", minimum=1, maximum=3, step=1, value=2)

    output_audio = gr.Audio(label="audio", type="numpy")

    generate_button = gr.Button("generate")

    generate_button.click(
        fn=generate_audio,
        inputs=[text_input, speaker_input, speed_input],
        outputs=output_audio,
    )

# Launch the service on localhost:5005
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=5005)
