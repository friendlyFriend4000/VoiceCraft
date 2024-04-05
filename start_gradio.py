import os
import gradio as gr
import pandas as pd
import torch
import torchaudio
from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
import whisperx
import time
import gc
import shutil
import subprocess
import GPUtil # pip install GPUtil

audio_fn = ""
transcript_fn = ""
align_fn = ""

model_loaded = False

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import getpass
# Get the current username (cross-platform solution)
username = getpass.getuser()
# Set the 'USER' environment variable to the current username
os.environ['USER'] = username


# This function is modified to correctly handle and utilize the transcription result.
def transcribe_btn_click(model_choice, audio_choice, transcribed_text):

    device = "cuda"
    batch_size = 1  # Adjust based on your GPU memory availability
    compute_type = "float16"  # Consider using "int8" for lower memory usage, though it may impact accuracy

    model = whisperx.load_model(model_choice, device, compute_type=compute_type)
    pre_result = model.transcribe(audio_choice, batch_size=batch_size)
    
    # Correctly handle the transcription result based on its structure
    if 'segments' in pre_result:
        result = " ".join([segment['text'] for segment in pre_result['segments']])
    else:
        result = pre_result.get('text', '')

    print("Transcribe text: " + result)  # Directly print the result as it is now a string

    # remove model to save VRAM
    gc.collect(); torch.cuda.empty_cache(); del model
    
    # Handling file and transcript saving correctly
    orig_audio = audio_choice
    global orig_transcript
    orig_transcript = result  # Directly use the result string
    temp_folder = "./demo/temp"
    os.makedirs(temp_folder, exist_ok=True)
    # os.system(f"cp '{orig_audio}' '{temp_folder}'")  # Ensure the command works for paths with spaces
    # use cross-platform shutill instead of linux cp
    shutil.copy(orig_audio, os.path.join(temp_folder, os.path.basename(orig_audio)))
    filename = os.path.splitext(os.path.basename(orig_audio))[0]
    
    # Define paths to the dictionary and acoustic model
    english_us_arpa_dict = "pretrained_models/english_us_arpa.dict"
    english_us_arpa_model = "pretrained_models/english_us_arpa"

    with open(os.path.join(temp_folder, f"{filename}.txt"), "w") as f:
        f.write(orig_transcript)
    # run MFA to get the alignment
    align_temp = f"{temp_folder}/mfa_alignments"
    os.makedirs(align_temp, exist_ok=True)

    if os.path.exists(f"{align_temp}/{filename}.csv"):
        pass
        print("mfa.cvs file exists already")
    else:
        print(align_temp + " is None")
        # os.system(f"mfa align -j 1 --output_format csv {temp_folder} english_us_arpa english_us_arpa {align_temp}")
        # Construct the MFA command
        mfa_command = f'mfa align -j 1 --output_format csv --clean "{temp_folder}" "{english_us_arpa_dict}" "{english_us_arpa_model}" "{align_temp}"'

        # Execute MFA alignment
        work = subprocess.run(mfa_command, shell=True, capture_output=True, text=True)
        if work.returncode != 0:
            # Retry MFA with increased beam size on failure
            retry_command = f'mfa align -j 1 --output_format csv "{temp_folder}" "{english_us_arpa_dict}" "{english_us_arpa_model}" "{align_temp}" --beam 1000 --retry_beam 2000'
            subprocess.run(retry_command, shell=True)


    # print("yes")
    global audio_fn
    audio_fn = f"{temp_folder}/{filename}.wav"
    
    global transcript_fn
    transcript_fn = f"{temp_folder}/{filename}.txt"
    global align_fn
    align_fn = f"{align_temp}/{filename}.csv"

    df = pd.read_csv(align_fn)
    # Select the first three columns
    df = df.iloc[:, :3]

    # Convert DataFrame to HTML
    html = df.to_html(index=False)

    return result, html

def run(seed, stop_repetition, sample_batch_size, left_margin, right_margin, codec_audio_sr, codec_sr, top_k, top_p,
        temperature, kvcache, cutoff_value, target_transcript, silence_tokens):
    # Cleanup memory at the beginning
    torch.cuda.empty_cache()
    gc.collect()
    
    # Get the GPU with the most available memory
    gpus = GPUtil.getGPUs()
    gpu = gpus[0]  # Assuming you want to use the first GPU

    # take a look at demo/temp/mfa_alignment, decide which part of the audio to use as prompt
    #cut_off_sec = cutoff_value  # NOTE: according to forced-alignment file, the word "common" stop as 3.01 sec, this should be different for different audio
    #target_transcript = target_transcript
    #info = torchaudio.info(audio_fn)
    #audio_dur = info.num_frames / info.sample_rate

    #assert cut_off_sec < audio_dur, f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"
    #prompt_end_frame = int(cut_off_sec * info.sample_rate)


    cut_off_sec = cutoff_value  # NOTE: according to forced-alignment file, the word "common" stop as 3.01 sec, this should be different for different audio
    target_transcript =  orig_transcript + target_transcript
    info = torchaudio.info(audio_fn)
    cut_off_sec = info.num_frames / info.sample_rate
    audio_dur = info.num_frames / info.sample_rate
    assert cut_off_sec <= audio_dur, f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"
    prompt_end_frame = int(cut_off_sec * info.sample_rate) 
    print(f"prompt_end_frame:",prompt_end_frame)


    # # load model, tokenizer, and other necessary files
    # # original file loaded it each time. here we load it only once
    # global model_loaded
    # f model_loaded==False:
    from models import voicecraft
    voicecraft_name = "giga830M.pth"
    ckpt_fn = f"./pretrained_models/{voicecraft_name}"
    encodec_fn = "./pretrained_models/encodec_4cb2048_giga.th"
    if not os.path.exists(ckpt_fn):
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/{voicecraft_name}\?download\=true")
        os.system(f"mv {voicecraft_name}\?download\=true ./pretrained_models/{voicecraft_name}")
    if not os.path.exists(encodec_fn):
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th")
        os.system(f"mv encodec_4cb2048_giga.th ./pretrained_models/encodec_4cb2048_giga.th")

    ckpt = torch.load(ckpt_fn, map_location="cpu")
    model = voicecraft.VoiceCraft(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()


    text_tokenizer = TextTokenizer(backend="espeak-ng")
    audio_tokenizer = AudioTokenizer(signature=encodec_fn)  # will also put the neural codec model on gpu
    # # run the model to get the output
    decode_config = {'top_k': top_k, 'top_p': top_p, 'temperature': temperature, 'stop_repetition': stop_repetition,
                     'kvcache': kvcache, "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr,
                     "silence_tokens": silence_tokens, "sample_batch_size": sample_batch_size}
    from inference_tts_scale import inference_one_sample


    
    # Monitor VRAM usage
    while True:
        # Get current VRAM usage
        vram_usage = gpu.memoryUsed
        total_vram = gpu.memoryTotal

        # Check if VRAM usage exceeds a threshold (e.g., 90% of total VRAM)
        print("Total VRAM: ", total_vram)
        print("VRAM usage: ", vram_usage)
        if vram_usage > 0.9 * total_vram:
            print("VRAM usage is high. Taking appropriate actions...")
            # Take actions to reduce VRAM usage, such as reducing batch size or clearing memory
            sample_batch_size = max(1, sample_batch_size // 2)
            torch.cuda.empty_cache()
            gc.collect()

        concated_audio, gen_audio = inference_one_sample(model, ckpt["config"], ckpt['phn2num'], text_tokenizer, audio_tokenizer,
                                                        audio_fn, target_transcript, device, decode_config,
                                                        prompt_end_frame)


        # save segments for comparison
        concated_audio, gen_audio = concated_audio[0].cpu(), gen_audio[0].cpu()
        torchaudio.save(f"gen.wav", gen_audio, 16000)


        output_dir = "./demo/generated_tts"
        os.makedirs(output_dir, exist_ok=True)
        seg_save_fn_gen = f"{output_dir}/{os.path.basename(audio_fn)[:-4]}_gen_seed{seed}.wav"
        seg_save_fn_concat = f"{output_dir}/{os.path.basename(audio_fn)[:-4]}_concat_seed{seed}.wav"


        torchaudio.save(seg_save_fn_gen, gen_audio, int(codec_audio_sr))
        torchaudio.save(seg_save_fn_concat, concated_audio, int(codec_audio_sr))

        # Break the loop if generation is successful
        break

    return [seg_save_fn_concat, seg_save_fn_gen]


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(label="Input Audio", type="filepath")
            transcribe_btn_model = gr.Radio(value="base.en", interactive=True, label="what whisper model to download",
                                            choices=["tiny.en (72MB)", "base.en", "small.en", "medium.en", "large"],
                                            info="VRAM usage: tiny.en 1 GB, base.en 1GB, small.en 2GB, medium.en 5GB, large 10GB.")
            transcribed_text = gr.Textbox(label="transcibed text + mfa",
                                          info="write down the transcript for the file, or run whisper model to get the transcript. Takes time to download whisper models on first run")
            transcribe_info_text = gr.TextArea(label="How to use",
                                               value="running everything for the first time will download necessary models (4GB for main encoder + model) \n load a voice and choose your whisper model, base works most of the time. \n transcription and mfa takes ~50s on a 3090 for a 7s audio clip, rerun this when uploading a new audio clip only\nchoose the END value of the cut off word \n")
            transcribe_btn = gr.Button(value="transcribe and create mfa")
            seed = gr.Number(label='seed', interactive=True, value=1)
            stop_repitition = gr.Radio(label="stop_repitition", interactive=True, choices=[1, 2, 3], value=3,
                                       info="if there are long silence in the generated audio, reduce the stop_repetition to 3, 2 or even 1")
            sample_batch_size = gr.Radio(label="sample_batch_size", interactive=True, choices=[4, 3, 2], value=4,
                                         info="if there are long silence or unnaturally strecthed words, increase sample_batch_size to 2, 3 or even 4")
            left_margin = gr.Number(label='left_margin', interactive=True, value=0.08, step=0.01,
                                    info=" not used for TTS, only for speech editing")
            right_margin = gr.Number(label='right_margin', interactive=True, value=0.08, step=0.01,
                                     info=" not used for TTS, only for speech editing")
            codecaudio_sr = gr.Number(label='codec_audio_sr', interactive=True, value=16000)
            codec_sr = gr.Number(label='codec', interactive=True, value=50)
            top_k = gr.Number(label='top_k', interactive=True, value=0)
            top_p = gr.Number(label='top_p', interactive=True, value=0.8)
            temperature = gr.Number(label='temperature', interactive=True, value=1)
            kvcache = gr.Number(label='kvcache', interactive=True, value=1,
                                info='set to 0 to use less VRAM, results may be worse and slower inference')
            silence_tokens = gr.Textbox(label="silence tokens", value="[1388,1898,131]")

        with gr.Column():
            output_audio_con = gr.Audio(label="Output Audio concatenated")
            output_audio_gen = gr.Audio(label="Output Audio generated")
            cutoff_value = gr.Number(label="cutoff_time", value=3.01, interactive=True, step=0.01)
            run_btn = gr.Button(value="run")
            target_transcript = gr.Textbox(label="target transcript")
            cvs_file_html = gr.HTML()

    transcribe_btn.click(fn=transcribe_btn_click, inputs=[transcribe_btn_model, input_audio, transcribed_text],
                         outputs=[transcribed_text, cvs_file_html])
    run_btn.click(fn=run,
                  inputs=[
                      seed,
                      stop_repitition,
                      sample_batch_size,
                      left_margin,
                      right_margin,
                      codecaudio_sr,
                      codec_sr,
                      top_k,
                      top_p,
                      temperature,
                      kvcache,
                      cutoff_value,
                      target_transcript,
                      silence_tokens],
                  outputs=[
                      output_audio_con,
                      output_audio_gen
                  ])

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    demo.launch()
