import os
from phonemizer.backend.espeak.wrapper import EspeakWrapper
_ESPEAK_LIBRARY = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
EspeakWrapper.set_library(_ESPEAK_LIBRARY)
import torch
import torchaudio
import gradio as gr
from pathlib import Path
import subprocess
import shutil
import tempfile

# Set CUDA for GPU acceleration
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load checkpoint and tokenizer models
ckpt_fn = "pretrained_models/giga830M.pth"
encodec_fn = "pretrained_models/encodec_4cb2048_giga.th"
ckpt = torch.load(ckpt_fn, map_location="cpu")
device = "cuda" if torch.cuda.is_available() else "cpu"

from models import voicecraft
model = voicecraft.VoiceCraft(ckpt["config"])
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()  # Set the model to evaluation mode

from data.tokenizer import AudioTokenizer, TextTokenizer
text_tokenizer = TextTokenizer(backend="espeak-ng")
audio_tokenizer = AudioTokenizer(signature=encodec_fn)  # will also put the neural codec model on GPU

# Define MFA command and execute alignment
def run_mfa(temp_folder, orig_audio, orig_transcript):
    filename = Path(orig_audio).stem
    audio_dest = temp_folder / f"{filename}.wav"
    if not audio_dest.exists():
        shutil.copy(orig_audio, audio_dest)
    
    transcript_dest = temp_folder / f"{filename}.txt"
    with open(transcript_dest, "w") as f:
        f.write(orig_transcript)
    
    align_temp = temp_folder / "mfa_alignments"
    align_temp.mkdir(parents=True, exist_ok=True)

    english_us_arpa_dict = "pretrained_models/english_us_arpa.dict" 
    english_us_arpa_model = "pretrained_models/english_us_arpa"
    mfa_command = f'mfa align -j 1 --output_format csv "{temp_folder}" "{english_us_arpa_dict}" "{english_us_arpa_model}" "{align_temp}"'
    result = subprocess.run(mfa_command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        retry_command = f'mfa align -j 1 --output_format csv "{temp_folder}" "{english_us_arpa_dict}" "{english_us_arpa_model}" "{align_temp}" --beam 1000 --retry_beam 2000'
        subprocess.run(retry_command, shell=True)
    return align_temp / f"{filename}.csv"

# Define the Gradio interface function
def tts_concatenate(original_audio, original_transcript, target_transcript, top_k=0, top_p=0.8, temperature=1, stop_repetition=3, inverse_offset=0, kvcache=1):
    # Processing logic remains the same; ensure to use normalization and handle audio saving/loading as demonstrated.

# Setup Gradio Interface
iface = gr.Interface(
    fn=tts_concatenate,
    inputs=[
        gr.Audio(label="Original Audio", type="filepath"),
        gr.Textbox(label="Original Transcript"),
        gr.Textbox(label="Target Transcript"),
        gr.Number(label="Top K", default=0),
        gr.Number(label="Top P", default=0.8),
        gr.Number(label="Temperature", default=1),
        gr.Number(label="Stop Repetition", default=3),
        gr.Number(label="Inverse Offset", default=0),
        gr.Number(label="KVCache", default=1)
    ],
    outputs=[
        gr.Audio(label="Generated Audio", type="filepath"),
        gr.Audio(label="Combined Audio", type="filepath")
    ]
)

# Launch the interface
iface.launch(share=False)
