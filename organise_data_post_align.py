'''
Script for organising the aligned data into separate folders for each clip, extract the audio from 
the video files and check that all the expected data is present. 
'''

import os
import shutil
import re
import logging
from moviepy.video.io.VideoFileClip import VideoFileClip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = '/Volumes/upfHD/MY_DATA/VOICE_DATA_CLEAN'


def extract_audio_from_video(input_video_path):
    output_audio_path = re.sub(r'(?i)\.(mp4|mov)$', '', input_video_path) + '.wav'

    if os.path.isfile(output_audio_path):
        logger.info("Audio already extracted in %s", output_audio_path)
    else:
        video_clip = VideoFileClip(input_video_path)
        audio_clip = video_clip.audio
        
        audio_clip.write_audiofile(output_audio_path)
        video_clip.close()
        logger.info("Audio extracted and saved to %s", output_audio_path)

    return output_audio_path

def get_end_index(file):
    match = re.search(r'[_-](\d+)\.', file)
    return match.group(1).lstrip('0') if match else None  # Strip leading zeros

def check_folder_contents_not_organised(path):
    # Gather all end indices from the files in the directory
    end_indices = set()
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            end_index = get_end_index(file)
            if end_index:
                end_indices.add(end_index)
    
    # Check if all files have the same end index
    if len(end_indices) == 1:
        # print("All files already have the same end index. No need to organize.")
        return False
    return True

def organise_files(path):
    # Organize files by their end index
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            end_index = get_end_index(file)
            if end_index:
                target_dir = os.path.join(path, end_index)
                os.makedirs(target_dir, exist_ok=True)
                shutil.move(os.path.join(path, file), os.path.join(target_dir, file))

def rename_files_with_leading_zero(start_path):
    for root, dirs, files in os.walk(start_path):
        if dirs and not files:
            continue 
        for file in files:
            match = re.search(r'([_-])(\d+)\.', file)
            if match:
                prefix, number = match.groups()
                normalized_number = number.lstrip('0')
                if number != normalized_number:  # Only rename if there was a leading zero
                    new_name = file.replace(f"{prefix}{number}", f"{prefix}{normalized_number}")
                    os.rename(os.path.join(root, file), os.path.join(root, new_name))
                    print(f"Renamed '{file}' to '{new_name}'")


def check_folder_contents(start_path):
    error_log_path = os.path.abspath(os.path.join(DATA_PATH, 'data_contents_assert_errors.txt'))
    all_files_present = True 

    with open(error_log_path, "w") as error_log:
        for root, dirs, files in os.walk(start_path):
            if root is DATA_PATH:
                continue
            if dirs and not files:
                continue  # Skip directories that have subdirectories but no files
            if dirs:
                continue
            
            logging.info(f'Checking data in directory: {root}')
            
            # Initialize variables to track the existence of required files
            json_file_exists = False
            computer_mp4_exists = False
            computer_wav_exists = False
            mic_wav_exists = False
            phone_mp4_exists = False
            phone_wav_exists = False
            
            # Check for the existence of required files
            for file in files:
                if file.endswith(".json"):
                    if not json_file_exists:  
                        json_file_exists = True
                    else:
                        if not file.startswith('._'):
                            error_log.write(f"EXTRA FILES: Extra json file in {root}: {file}\n")
                elif file.endswith(".mp4"):
                    if "phone" in file.lower():
                        if not phone_mp4_exists:
                            phone_mp4_exists = True
                        else:
                            if not file.startswith('._'):
                                error_log.write(f"EXTRA FILES: Extra mp4 file in {root}: {file}\n")
                    elif "computer" in file.lower():
                        if not computer_mp4_exists:
                            computer_mp4_exists = True
                        else:
                            if not file.startswith('._'):
                                error_log.write(f"EXTRA FILES: Extra mp4 file in {root}: {file}\n")
                    else:
                        error_log.write(f"EXTRA FILES: Extra mp4 file in {root}: {file}\n")
                elif file.endswith(".wav"):
                    if "phone" in file.lower():
                        if not phone_wav_exists:
                            phone_wav_exists = True
                        else:
                            if not file.startswith('._'):
                                error_log.write(f"EXTRA FILES: Extra wav file in {root}: {file}\n")
                    elif "computer" in file.lower():
                        if not computer_wav_exists:
                            computer_wav_exists = True
                        else:
                            if not file.startswith('._'):
                                error_log.write(f"EXTRA FILES: Extra wav file in {root}: {file}\n")
                    elif "mic" in file.lower():
                        if not mic_wav_exists:
                            mic_wav_exists = True
                        else:
                            if not file.startswith('._'):
                                error_log.write(f"EXTRA FILES: Extra wav file in {root}: {file}\n")
                    else:
                        error_log.write(f"EXTRA FILES: Extra wav file in {root}: {file}\n")

            
            # Assert the existence of required files for each directory
            try:
                assert json_file_exists, f"No .json file found in {root}"
            except AssertionError as e:
                all_files_present = False
                logging.info(f"WARNING: Missing .json file in {root}")
                error_log.write(str(e) + "\n")

            try:
                assert computer_mp4_exists, f"No computer video file found in {root}"
            except AssertionError as e:
                all_files_present = False
                logging.info(f"WARNING: Missing computer video file in {root}")
                error_log.write(str(e) + "\n")

            try:
                assert computer_wav_exists, f"No computer audio file found in {root}"
            except AssertionError as e:
                all_files_present = False
                logging.info(f"WARNING: Missing computer audio file in {root}")
                error_log.write(str(e) + "\n")

            try:
                assert phone_mp4_exists, f"No phone video file found in {root}"
            except AssertionError as e:
                all_files_present = False
                logging.info(f"WARNING: Missing phone video file in {root}")
                error_log.write(str(e) + "\n")

            try:
                assert phone_wav_exists, f"No phone audio file found in {root}"
            except AssertionError as e:
                all_files_present = False
                logging.info(f"WARNING: Missing phone audio file in {root}")
                error_log.write(str(e) + "\n")

            try:
                assert mic_wav_exists, f"No mic audio file found in {root}"
            except AssertionError as e:
                all_files_present = False
                logging.info(f"WARNING: Missing mic audio file in {root}")
                error_log.write(str(e) + "\n")

    logging.info(f"All data present: {all_files_present}")


def find_dirs_with_only_files_and_organise(start_path):
    for root, dirs, files in os.walk(start_path):
        if dirs and not files:
            continue
        if files and not dirs:
            if check_folder_contents_not_organised(root):
                organise_files(root)


def get_audio_from_video_files(path):
    for root, dirs, files in os.walk(path):
        if dirs and not files:
            continue
        if files and not dirs:
            for file in os.listdir(root):
                if (not file.startswith("._")) and file.endswith(".mp4"):
                    extract_audio_from_video(os.path.join(root, file))


# find_dirs_with_only_files_and_organise(DATA_PATH)
# rename_files_with_leading_zero(DATA_PATH)
# get_audio_from_video_files(DATA_PATH)
check_folder_contents(DATA_PATH)