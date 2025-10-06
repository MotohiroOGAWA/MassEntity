import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
import warnings
from typing import Tuple
from enum import IntEnum
from datetime import datetime

import re

from .ItemParser import ItemParser
from ..core import MSDataset, PeakSeries

class ErrorLogLevel(IntEnum):
    NONE = 0    # Do not write any error log
    BASIC = 1   # Write line number and error message only
    DETAIL = 2  # Write BASIC info + record content that caused the error

def read_msp(filepath, 
             encoding='utf-8', 
             return_header_map=False, 
             set_idx_ori=False, 
             error_log_level: ErrorLogLevel = ErrorLogLevel.NONE,
             error_log_file=None) -> MSDataset:
    file_size = os.path.getsize(filepath)
    processed_size = 0
    line_count = 1
    item_parser = ItemParser()

    cols = {} # Create a data list for each column
    all_peak = []
    offsets = [0]
    peak = []
    max_peak_cnt = 0
    record_cnt = 1
    success_cnt = 0
    text = ""
    error_text = ""
    error_flag = False

    header_map = {}

    if error_log_file is None:
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        error_filename = os.path.splitext(filepath)[0] + f"_error_{now}"
        cnt = 1
        if os.path.exists(error_filename + ".txt"):
            while True:
                if not os.path.exists(f'{error_filename}_{cnt}.txt'):
                    error_filename = f'{error_filename}_{cnt}'
                    break
                else:
                    cnt += 1
        error_file_path = error_filename + ".txt"
    elif not os.path.exists(os.path.dirname(filepath)):
        raise ValueError(f"Error: Directory '{os.path.dirname(filepath)}' does not exist.")
    else:
        error_file_path = error_log_file
    
    with open(filepath, 'r', encoding=encoding) as f:
        peak_flag = False
        with tqdm(total=file_size, desc="Read msp file", mininterval=0.5) as pbar:
            for line in f.readlines():
                try:
                    if not peak_flag and line == '\n':
                        continue

                    text += line

                    if peak_flag and line == '\n':
                        peak_flag = False

                        if not error_flag:
                            all_peak.extend(peak)
                            offsets.append(len(all_peak))
                            max_peak_cnt = max(max_peak_cnt, len(peak))
                            success_cnt += 1
                            for k in cols:
                                cols[k].append("")
                        else:
                            if error_log_level != ErrorLogLevel.NONE:
                                error_text = _get_error_text(record_cnt, line_count, text, cols, error_log_level)
                                
                                if not os.path.exists(error_file_path):
                                    with open(error_file_path, "w") as ef:
                                        ef.write('')
                                with open(error_file_path, "a") as ef:
                                    ef.write(error_text)
                            error_text = ""
                            error_flag = False
                            for k in cols:
                                if len(cols[k]) >= success_cnt + 1:
                                    cols[k][-1] = ""
                        text = ""
                        peak = []
                        record_cnt += 1
                        for k in cols:
                            if len(cols[k]) < success_cnt+1:
                                cols[k] = cols[k] + [""] * (success_cnt+1 - len(cols[k]))
                        pbar.set_postfix_str({"Success": f'{success_cnt}/{record_cnt}'})
                    elif peak_flag:
                        # Handling cases where peaks are tab-separated or space-separated
                        if len(line.strip().split('\t')) == 2:
                            mz, intensity = line.strip().split('\t')
                        elif len(line.strip().split(' ')) == 2:
                            mz, intensity = line.strip().split(' ')
                        else:
                            raise ValueError(f"Error: '{line.strip()}' was not split correctly.")
                        mz, intensity = float(mz), float(intensity)
                        peak.append([mz, intensity])
                    else:
                        k,v = item_parser.parse(line)
                        ori_k = line.split(":", 1)[0].strip()
                        if k not in cols:
                            header_map[k] = ori_k
                            cols[k] = [""] * record_cnt

                        if k == "Comments":
                            # Extract computed SMILES from comments
                            pattern = r'"computed SMILES=([^"]+)"'
                            match = re.search(pattern, v)
                            if match:
                                if "SMILES" not in cols:
                                    cols["SMILES"] = [""] * record_cnt
                                cols["SMILES"][-1] = match.group(1)
                        else:
                            cols[k][-1] = v
                        if k == "NumPeaks":
                            peak_flag = True
                    
                    line_count += 1
                    processed_size = len(line.encode(encoding)) + 1
                    pbar.update(processed_size)
                except Exception as e:
                    text = 'Error: ' + str(e).replace('\n', '\\n') + '\n' + text
                    error_flag = True
                    pass

            # Remove last empty rows in metadata
            for k in cols:
                if cols[k][-1] != "":
                    break
            else:
                for k in cols:
                    del cols[k][-1]
            row_cnt = len(cols[list(cols.keys())[0]])

            # Append last peak data if file doesn't end with a blank line
            if line != '\n' and (len(offsets) - 1 < row_cnt):
                all_peak.extend(peak)
                offsets.append(len(all_peak))
                max_peak_cnt = max(max_peak_cnt, len(peak))
                record_cnt += 1
                success_cnt += 1

            pbar.set_postfix_str({"Success": f'{success_cnt}/{record_cnt}'})
            
        if set_idx_ori:
            cols['IdxOri'] = list(range(row_cnt))

        if (error_log_level != ErrorLogLevel.NONE) and error_text != '':
            error_text = _get_error_text(record_cnt, line_count, text, cols, error_log_level)
            if not os.path.exists(error_file_path):
                with open(error_file_path, "w") as ef:
                    ef.write('')
            with open(error_file_path, "a") as ef:
                ef.write(error_text)

    peaks = torch.tensor(all_peak)
    offsets = torch.tensor(offsets)
    peak_series = PeakSeries(peaks, offsets)
    spectrum_meta = pd.DataFrame(cols)
    ms_dataset = MSDataset(spectrum_meta, peak_series)

    if return_header_map:
        return ms_dataset, header_map
    else:
        return ms_dataset
    
def _get_error_text(record_cnt, line_count, text, cols, error_log_level):
    error_text = f"Record: {record_cnt}\n" + f"Rows: {line_count}\n"
    for k in cols:
        if len(cols[k]) == record_cnt:
            cols[k].pop()
        elif len(cols[k]) > record_cnt:
            error_text += f"Error: '{k}' has more data than the record count.\n"
    if error_log_level == ErrorLogLevel.DETAIL:
        error_text += text + '\n\n'
    else:
        error_text += '\n\n'
    return error_text

def write_msp(dataset: MSDataset, path: str, headers=None, header_map={}, encoding='utf-8'):
    """
    Save MSDataset to MSP file.

    Args:
        path (str): Output MSP file path.
    """
    df = dataset.meta_copy
    if headers is None:
        headers = dataset._columns

    _headers = []
    for c in headers:
        if c not in df.columns:
            continue
        if c == "IdxOri":
            continue
        if c == "NumPeaks":
            continue
        _headers.append(c)
    
    headers = _headers.copy()
    for c in headers:
        if c not in header_map.keys():
            header_map[c] = c
    with open(path, "w", encoding=encoding) as outfile:
        for record in dataset:
            for key in headers:
                value = str(record[key])
                if value == "nan":
                    value = ""
                outfile.write(f"{header_map[key]}: {value}\n")

            outfile.write(f"NumPeaks: {record.n_peaks}\n")
            for peak in record.peaks:
                outfile.write(f"{peak.mz} {peak.intensity}\n")
            outfile.write("\n")
    
def normalize_column(value:str) -> str:
    result = value.replace("_", "").replace("/", "").lower()
    return result

# MSP column mappings for parsing and column naming
msp_column_names = {
    "Name": [],
    "Formula": [],
    "InChIKey": [],
    "PrecursorMZ": [],
    "AdductType": ["PrecursorType"],
    "SpectrumType": [],
    "InstrumentType": [],
    "Instrument": [],
    "IonMode": [],
    "CollisionEnergy": [],
    "ExactMass": [],
}

# Column data types for processing
msp_column_types = {
    "PrecursorMZ": "Float32",
    "ExactMass": "Float32",
}


# Convert columns to their predefined data types (strings if not specified)
def convert_to_types_str(columns):
    column_types = {}
    for c in columns:
        if c in msp_column_types:
            column_types[c] = msp_column_types[c]
        else:
            column_types[c] = "str"
    return column_types

# AdductType column data mapping 
precursor_type_data = {
    "[M]+" : ["M", "[M]"],
    "[M+H]+": ["M+H", "[M+H]"],
    "[M-H]-": ["M-H", "[M-H]"],
    "[M+Na]+": ["M+Na", "[M+Na]"],
    "[M+K]+": ["M+K", "[M+K]"],
    "[M+NH4]+": ["M+NH4", "[M+NH4]"],
    }
to_precursor_type = {}
for precursor_type, data in precursor_type_data.items():
    to_precursor_type[precursor_type] = precursor_type
    for aliases in data:
        to_precursor_type[aliases] = precursor_type


if __name__ == "__main__":
    pass
