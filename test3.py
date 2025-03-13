import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter, BaseSpeakerTTS

ckpt_converter = 'checkpoints_v2/converter'
device="cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_test3'

print("\n# load base speaker\n")
base_speaker_tts = BaseSpeakerTTS('models/EN_V2/config.json', device=device)
base_speaker_tts.load_ckpt('models/EN_V2/checkpoint.pth')

print("\n# load tone color converter\n")
tone_color_converter = ToneColorConverter('models/EN_V2/config.json', device=device)
tone_color_converter.load_ckpt('models/EN_V2/checkpoint.pth')

os.makedirs(output_dir, exist_ok=True)

print("\n# load se\n")
source_se = torch.load('checkpoints_v2/base_speakers/ses/en-us.pth').to(device)

reference_speaker = 'resources/samuel_original.mp3'
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='processed', vad=False)

save_path = f'{output_dir}/output_teste3_default.wav'

text = "This audio is from someone who loves Shrek"
src_path = f'{output_dir}/tmp.wav'

#  friendly, cheerful, excited, sad, angry, terrified, shouting, whispering
base_speaker_tts.tts(text, src_path, speaker='default', language='English', speed=1.0)

encode_message = "@MyShell"
tone_color_converter.convert(
    audio_src_path=src_path, 
    src_se=source_se, 
    tgt_se=target_se, 
    output_path=save_path,
    message=encode_message
)