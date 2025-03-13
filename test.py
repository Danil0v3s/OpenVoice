import os
import torch
import nltk
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from openvoice.tts import TTS


ckpt_converter = 'checkpoints_v2/converter'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_v2'
src_path = f'{output_dir}/tmp.wav'

nltk.download('averaged_perceptron_tagger_eng')

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

reference_speaker = 'resources/minion.mp3' # This is the voice you want to clone
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)

os.makedirs(output_dir, exist_ok=True)

text = "Shrek first meets Donkey when Lord Farquaad's guards are rounding up fairytale creatures. Donkey, trying to escape capture, bumps into Shrek, who scares off the guards with his intimidating presence. Instead of running away, Donkey is fascinated by Shrek and insists on following him home, much to Shrek's annoyance. Despite Shrek's attempts to get rid of him, Donkey cheerfully tags along, marking the beginning of their unlikely friendship"

# Speed is adjustable
speed = 1.0
speaker_id = 0
noise_scale = 0.0
noise_scale_w = 0.0

print("creating model")
model = TTS(language='EN_V2', device=device)

source_se = torch.load(f'checkpoints_v2/base_speakers/ses/en-newest.pth', map_location=device)
model.tts_to_file(text, speaker_id, src_path, speed=speed, noise_scale=noise_scale, noise_scale_w=noise_scale_w)

save_path = f'{output_dir}/output_v2_{speaker_id}.wav'

# Run the tone color converter
encode_message = "@MyShell"
tone_color_converter.convert(
    audio_src_path=src_path, 
    src_se=source_se, 
    tgt_se=target_se, 
    output_path=save_path,
    tau=0.3,
    message=encode_message
)

# for language, text in texts.items():
#     model = TTS(language=language, device=device)
#     speaker_ids = model.hps.data.spk2id
    
#     for speaker_key in speaker_ids.keys():
#         speaker_id = speaker_ids[speaker_key]
#         speaker_key = speaker_key.lower().replace('_', '-')

#         source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
#         model.tts_to_file(text, speaker_id, src_path, speed=speed)
#         save_path = f'{output_dir}/output_v2_{speaker_key}.wav'

#         # Run the tone color converter
#         encode_message = "@MyShell"
#         tone_color_converter.convert(
#             audio_src_path=src_path, 
#             src_se=source_se, 
#             tgt_se=target_se, 
#             output_path=save_path,
#             message=encode_message
#         )