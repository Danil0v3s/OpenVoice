import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter, BaseSpeakerTTS

ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints/converter'
device="cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_test2'

base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)

source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)
reference_speaker = 'resources/samuel_original.mp3' # This is the voice you want to clone
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='processed', vad=False)

save_path = f'{output_dir}/output_en_default.wav'

# Run the base speaker tts
text = "Shrek first meets Donkey when Lord Farquaad's guards are rounding up fairytale creatures. Donkey, trying to escape capture, bumps into Shrek, who scares off the guards with his intimidating presence. Instead of running away, Donkey is fascinated by Shrek and insists on following him home, much to Shrek's annoyance. Despite Shrek's attempts to get rid of him, Donkey cheerfully tags along, marking the beginning of their unlikely friendship"
src_path = f'{output_dir}/tmp.wav'

#  friendly, cheerful, excited, sad, angry, terrified, shouting, whispering
base_speaker_tts.tts(text, src_path, speaker='cheerful', language='English', speed=1.1)

# Run the tone color converter
encode_message = "@MyShell"
tone_color_converter.convert(
    audio_src_path=src_path, 
    src_se=source_se, 
    tgt_se=target_se, 
    output_path=save_path,
    message=encode_message
)