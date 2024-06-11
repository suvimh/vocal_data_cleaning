
'''
FUNCTIONS USED FOR DATA CLEANING
'''

import os
import re
from datetime import datetime
import logging
import json
from tqdm import tqdm
from h5py import File
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from moviepy.video.io.VideoFileClip import VideoFileClip

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_h5_data(biosignals_data_dict, file_name):
    """
    Plots the raw biosignal data from the given biosignals_data_dict.

    Args:
        biosignals_data_dict (dict): A dictionary containing the biosignal data.
            The keys of the dictionary represent the biosignal types, and the values
            are dictionaries containing the channel names and corresponding information.
            The channel information should include the "Signal Data" (numpy array),
            and the "Sample Rate" (float).
        file_name (str): The name of the file being plotted.
    """

    num_subplots = sum(len(channels) for channels in biosignals_data_dict.values())
    fig = make_subplots(rows=num_subplots, cols=1, shared_xaxes=True)

    subplot_index = 1
    for _, channels in biosignals_data_dict.items():
        for channel_name, channel_info in channels.items():
            data_list = np.array(channel_info["Signal Data"]).flatten()
            sampling_rate = channel_info["Sample Rate"]
            time = np.arange(len(data_list)) / sampling_rate

            # Plot only the first quarter of the data (zoom in view)
            quarter_index = len(time) // 4
            data_list = data_list[:quarter_index]
            time = time[:quarter_index]

            # Add trace to the subplot with the corresponding index
            fig.add_trace(go.Scatter(x=time, y=data_list, mode='lines', name=channel_name), row=subplot_index, col=1)
            fig.update_yaxes(title_text=channel_name, row=subplot_index, col=1)

            subplot_index += 1

    fig.update_layout(title=f"Raw Biosignal Data: {file_name}", height=400 * num_subplots)

    # Add tick marks in min:sec
    tick_indices = np.linspace(0, len(time) - 1, dtype=int)
    tick_time = [f"{int(t // 60):02d}:{int(t % 60):02d}.{int((t % 1) * 1000):03d}" for t in time[tick_indices]]

    fig.update_xaxes(
        tickmode='array',
        tickvals=[time[t] for t in tick_indices],
        ticktext=tick_time
    )

    pio.show(fig)


def plot_aligned_data(aligned_data, h5_file):
    """
    Plots aligned biosignal data.

    Args:
        aligned_data (dict): A dictionary containing aligned biosignal data.
            The keys are signal types and the values are dictionaries containing
            channel names and corresponding signal data.
        h5_file (str): The name of the HDF5 file.

    """

    fig = make_subplots(rows=len(aligned_data), cols=1, shared_xaxes=True, subplot_titles=list(aligned_data.keys()))

    for i, (signal_type, channels) in enumerate(aligned_data.items(), start=1):
        for channel_name, channel_info in channels.items():
            signal_data = np.array(channel_info['Signal Data']).flatten()
            sampling_rate = channel_info['Sample Rate']
            time = np.arange(len(signal_data)) / sampling_rate
            
            # Plot only the first quarter of the data (zoom in view)
            quarter_index = len(time) // 4
            data_list = signal_data[:quarter_index]
            time = time[:quarter_index]

            fig.add_trace(go.Scatter(x=time, y=data_list, mode='lines', name=f'{channel_name}'), row=i, col=1)
            fig.update_yaxes(title_text='Signal Value', row=i, col=1)

    fig.update_layout(title=f'Biosignal Data: aligned {h5_file}', xaxis_title='Time (s)', height=600)
    
    # Add tick marks in min:sec
    tick_indices = np.linspace(0, len(time) - 1, dtype=int)
    tick_time = [f"{int(t // 60):02d}:{int(t % 60):02d}.{int((t % 1) * 1000):03d}" for t in time[tick_indices]]

    fig.update_xaxes(
        tickmode='array',
        tickvals=[time[t] for t in tick_indices],
        ticktext=tick_time
    )

    fig.show()

def get_aligned_biosignal_data(biosignals_data_dict, start_time):
    """
    Retrieves aligned biosignal data from the given biosignals_data_dict starting from the specified start_time.

    Args:
        biosignals_data_dict (dict): A dictionary containing biosignal data.
        start_time (float): The start time in seconds from which to retrieve the aligned data.

    Returns:
        dict: A dictionary containing the aligned biosignal data.

    """
    aligned_data = {}
    sampling_rate = get_biosignal_sample_rate(biosignals_data_dict)

    start_index = int(sampling_rate * start_time)

    for signal_type, channels in biosignals_data_dict.items():
        aligned_data[signal_type] = {}
        for channel_name, channel_info in channels.items():
            data_array = channel_info['Signal Data']
            aligned_data[signal_type][channel_name] = {
                'Sample Rate': sampling_rate,
                'Acquisition Time': channel_info['Acquisition Time'],
                'Resolutions': channel_info['Resolutions'],
                'Signal Type': channel_info['Signal Type'],
                'Signal Data': data_array[start_index:]
            }

    return aligned_data

def time_to_milliseconds(time_str):
    '''
    Convert a time string in the format 'HH:MM:SS:frames' to milliseconds.

    Parameters:
    - time_str (str): A time string in the format 'HH:MM:SS:frames'.

    Returns:
    - total_milliseconds (float): The time in milliseconds.

    Raises:
    - ValueError: If the time string is not in the correct format 'HH:MM:SS:frames'.
    '''
    match = re.match(r'(\d+):(\d+):(\d+):(\d+)', time_str)
    if match:
        hours, minutes, seconds, frames = map(int, match.groups())
        total_seconds = hours * 3600 + minutes * 60 + seconds + frames / 24.0
        total_milliseconds = total_seconds * 1000
        return total_milliseconds
    else:
        raise ValueError(f"Time string {time_str} is not in the correct format 'HH:MM:SS:frames'.")



def extract_data_from_range(aligned_data, start_time, end_time):
    '''
    Extracts data from aligned biosignals within the specified time range in a JSON serializable way.

    Parameters:
        aligned_data (dict): A dictionary containing aligned biosignals.
        start_time (str): The start time of the desired data range in the format 'HH:MM:SS'.
        end_time (str): The end time of the desired data range in the format 'HH:MM:SS'.

    Returns:
        tuple: A tuple containing two dictionaries. The first dictionary contains the extracted data, and the second dictionary contains the indices of the extracted data.

    '''
    start_ms = time_to_milliseconds(start_time)
    end_ms = time_to_milliseconds(end_time)

    extracted_data = {}
    extracted_indices = {}
    for signal_type, channels in aligned_data.items():
        extracted_data[signal_type] = {}
        extracted_indices[signal_type] = {}
        for channel_name, channel_info in channels.items():
            sampling_rate = channel_info['Sample Rate']
            data_array = channel_info['Signal Data']
            start_index = int(sampling_rate * start_ms / 1000)
            end_index = int(sampling_rate * end_ms / 1000)
            
            extracted_indices[signal_type][channel_name] = {'start_idx': start_index, 'end_idx': end_index}
            
            # Convert nested NumPy arrays to nested lists
            data_array = data_array[start_index:end_index].tolist()
            # Convert any remaining NumPy data types to native Python types
            data_array = np.array(data_array).tolist()
            extracted_data[signal_type][channel_name] = {
                'Sample Rate': int(sampling_rate),
                'Acquisition Time': str(channel_info['Acquisition Time']),
                'Resolutions': [int(res) for res in channel_info['Resolutions']],
                'Signal Type': str(channel_info['Signal Type']),
                'Signal Data': data_array
            }
    return extracted_data, extracted_indices


def highlight_extracts_in_aligned_data(aligned_data, all_indices, h5_file):
    """
    Plots the aligned biosignal data with highlighted extracted sections to mark
    data extracted for each clip.

    Args:
        aligned_data (dict): A dictionary containing the aligned biosignal data.
        all_indices (dict): A dictionary containing the indices of the extracted sections.
        h5_file (str): The name of the H5 file.
    """

    fig = make_subplots(rows=len(aligned_data), cols=1, shared_xaxes=True, subplot_titles=list(aligned_data.keys()))

    for i, (signal_type, channels) in enumerate(aligned_data.items(), start=1):
        for channel_name, channel_info in channels.items():
            signal_data = np.array(channel_info['Signal Data']).flatten()
            sampling_rate = channel_info['Sample Rate']
            time = np.arange(len(signal_data)) / sampling_rate
            fig.add_trace(go.Scatter(x=time, y=signal_data, mode='lines', name=f'{channel_name}'), row=i, col=1)
            fig.update_yaxes(title_text='Signal Value', row=i, col=1)

            # Highlight extracted sections
            for idx, indices in all_indices.items():
                if signal_type in indices and channel_name in indices[signal_type]:
                    start_idx = indices[signal_type][channel_name]['start_idx']
                    end_idx = indices[signal_type][channel_name]['end_idx']
                    extracted_signal_data = signal_data[start_idx:end_idx]
                    extracted_time = time[start_idx:end_idx]
                    
                    fig.add_trace(go.Scatter(x=extracted_time, y=extracted_signal_data, mode='lines', name=f'Extracted Data {idx} - {channel_name}'), row=i, col=1)

    fig.update_layout(title=f'Biosignal Data: aligned {h5_file}', xaxis_title='Time (s)', height=600)

    # Add tick marks in min:sec
    tick_indices = np.linspace(0, len(time) - 1, num=10, dtype=int)
    tick_time = [f"{int(t // 60):02d}:{int(t % 60):02d}.{int((t % 1) * 1000):03d}" for t in time[tick_indices]]
    fig.update_xaxes(tickmode='array', tickvals=[time[t] for t in tick_indices], ticktext=tick_time)

    fig.show()


def save_useful_data_from_biosignals(aligned_data, output_dir, session, h5_file, data_ranges, scale_type, graph_extracted_data=True):
    """
    Extracts and saves useful data from biosignals.

    Args:
        aligned_data (list): List of aligned data.
        output_dir (str): Output directory path.
        session (str): Data collection session.
        h5_file (str): H5 file path.
        data_ranges (list): List of tuples representing start and end times of data ranges.
        scale_type (str): Scale type.
        graph_extracted_data (bool, optional): Whether to generate a graph of the extracted data. Defaults to False.
    """

    folder_path = os.path.join(output_dir, session)

    # Create session folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    all_data = {}
    all_indices = {}

    # Iterate over data ranges to extract and save data
    for idx, (start_time, end_time) in enumerate(tqdm(data_ranges, desc=f"Processing Data for {scale_type}")):
        extracted_data, extracted_indices = extract_data_from_range(aligned_data, start_time, end_time)
        all_data[f"Data_{idx}"] = extracted_data
        all_indices[f"Indices_{idx}"] = extracted_indices

        # Create folder for the scale type if it doesn't exist
        scale_type_folder = os.path.join(folder_path, scale_type)
        if not os.path.exists(scale_type_folder):
            os.makedirs(scale_type_folder)

        # Write the extracted data to a JSON file
        output_file_name = f"{scale_type}_{idx+1}.json"
        output_file_name = re.sub(r'^\._', '', output_file_name)
        output_file_path = os.path.join(scale_type_folder, output_file_name)
        output_file_path = re.sub(r'^\._', '', output_file_path)
        with open(output_file_path, 'w') as outfile:
            json.dump(extracted_data, outfile, indent=4)

    # Generate the graph if graph_extracted_data is True
    if graph_extracted_data:
        highlight_extracts_in_aligned_data(aligned_data, all_indices, h5_file)


def get_h5_and_wav_files_from_input_directory(directory_path):
    """
    Retrieves the path and name of an h5 file from the specified directory.

    Args:
        directory_path (str): The path to the directory containing the files.

    Returns:
        tuple: A tuple containing the path and name of the h5 file found in the directory.
               If no h5 file is found, an empty string is returned for both path and name.
    """
    files = os.listdir(directory_path)

    h5_file = ''
    h5_path = ''

    if not files:
        print("No files found in the directory.")
    else:
        for file in files:
            if file.endswith(".h5"):
                print(f"h5 file: {file}")
                h5_path = os.path.join(directory_path, file)
                h5_file = file

    return h5_path, h5_file
    

def get_biosignal_data_from_h5(path):
    """
    Adapted from biosignalsnotebooks Github repo 
    https://github.com/pluxbiosignals/biosignalsnotebooks

    Function to read a .h5 file with a structure provided by OpenSignals and construct a dictionary with the more
    relevant information about each signal grouped by signal type.
    
    Parameters:
        path (str) : Absolute or relative path to the .h5 file to be read.
    
    Returns:
        signal_types_dict (dict) : Dictionary with signal types as keys and channels grouped by signal type.
    """
    file = File(path)
    signal_types_dict = {}
    channel_counters = {}  # To keep track of numerical values for channel names
    for mac in list(file.keys()):
        device = file[mac]
        sampling_rate = device.attrs["sampling rate"]
        samples = device.attrs["nsamples"]
        time_sec = samples / sampling_rate
        time = datetime.fromtimestamp(time_sec).strftime("%H:%M:%S.") + str(datetime.fromtimestamp(time_sec).microsecond//1000)[:1]
        for channel in list(device["raw"].keys()):
            if "channel" in channel:
                sensor = device["raw"][channel].attrs["sensor"]
                if sensor == "RAW":
                    print("The type of signal is set to RAW. You need to specify the type of signal (sensor): ")
                    sensor = input("Enter sensor type: ")
                if sensor not in signal_types_dict:
                    signal_types_dict[sensor] = {}
                    channel_counters[sensor] = 1
                else:
                    channel_counters[sensor] += 1
                channel_name = f"{sensor}_{channel_counters[sensor]}"
                channel_info = {}
                channel_info["Sample Rate"] = sampling_rate
                channel_info["Acquisition Time"] = time
                channel_info["Resolutions"] = [res for res in device.attrs["resolution"]]
                channel_info["Signal Type"] = sensor
                channel_info["Signal Data"] = device["raw"][channel][()]  # Read all data from the channel
                signal_types_dict[sensor][channel_name] = channel_info
    
    return signal_types_dict



def get_biosignal_sample_rate(biosignal_data):
    """
    Function to check if the sample rate is the same for all channels within each signal type.

    Parameters:
        biosignal_data (dict): Dictionary containing signal types as keys and channels grouped by signal type.

    Returns:
        int or None: The sample rate if it's the same for all channels, otherwise returns None.

    Raises: 
        ValueError: If the sample rates are not consistent among all channels.

    """
    sample_rate = None
    for _, channels in biosignal_data.items():
        for channel_info in channels.values():
            if sample_rate is None:
                sample_rate = channel_info["Sample Rate"]
            elif sample_rate != channel_info["Sample Rate"]:
                raise ValueError("Sample rates are not consistent among all channels.")
    return sample_rate


def extract_audio_from_video(input_video_path):
    """
    Extracts the audio from a video file and saves it as a separate WAV file.

    Args:
        input_video_path (str): The path to the input video file.

    Returns:
        str: The path to the extracted audio file.

    """
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