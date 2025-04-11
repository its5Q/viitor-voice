import gradio as gr
import sys, os

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from viitor_voice.inference.vllm_engine import VllmEngine
from viitor_voice.inference.transformers_engine import TransformersEngine

if __name__ == '__main__':
    # Initialize your OfflineInference class with the appropriate paths
    offline_inference = TransformersEngine("./checkpoints/poc2/checkpoint-25200")


    def clone_batch(prompt_text):
        try:
            audios = offline_inference.batch_infer(
                text_list=[prompt_text]
            )
            print(audios)
            return 24000, audios[0].cpu().numpy()[0].astype('float32')
        except Exception as e:
            return str(e)


    with gr.Blocks() as demo:
        gr.Markdown("# TTS Inference Interface")
        with gr.Tab("Batch TTS"):
            gr.Markdown("### Batch TTS")

            prompt_text = gr.Textbox(label="Prompt Text", placeholder="Enter the prompt text")

            tts_button = gr.Button("Run TTS")
            tts_output = gr.Audio(label="Generated Audios", type="numpy")

            tts_button.click(
                fn=clone_batch,
                inputs=[prompt_text],
                outputs=tts_output
            )

    demo.launch(server_name="0.0.0.0")
