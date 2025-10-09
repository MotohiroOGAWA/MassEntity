import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
import warnings
from typing import Tuple
from datetime import datetime

import re

from .IOContext import ReaderContext
from .constants import ErrorLogLevel
from ..core import MSDataset, PeakSeries
from ..utils.annotate import set_spec_id


def read_msp(filepath, 
             encoding='utf-8', 
             return_header_map=False, 
             spec_id_prefix=None, 
             error_log_level: ErrorLogLevel = ErrorLogLevel.NONE,
             error_log_file=None,
             show_progress=True,
             ) -> MSDataset:
    
    msp_reader = ReaderContext(
        filepath, 
        error_log_level=error_log_level, 
        error_log_file=error_log_file,
        encoding=encoding,
        show_progress=show_progress,
        )
    msp_reader.file_type_name = "msp"
    
    with open(filepath, 'r', encoding=encoding) as f:
        peak_flag = False
        peak_columns = []
        for line in f.readlines():
            msp_reader.update(line)
            try:
                if not peak_flag and line == '\n':
                    pass

                elif peak_flag and line == '\n':
                    msp_reader.update_record()
                    peak_flag = False
                    peak_columns = []
                elif peak_flag:
                    # Check if the line contains peak data
                    if len(peak_columns) == 0:
                        items = line.strip().split()
                        if len(items) >= 2:
                            if(items[0].lower() == 'mz' and items[1].lower() == 'intensity'):
                                peak_columns = items.copy()
                                continue
                        if len(peak_columns) == 0:
                            peak_columns = ['mz', 'intensity']


                    items = line.strip().split()
                    if len(items) >= 3:
                        mz_item = items[0]
                        intensity_item = items[1]
                        meta_items = "".join(items[2:]).split(';')
                    elif len(items) == 2:
                        mz_item = items[0]
                        intensity_item = items[1]
                        meta_items = []
                    else:
                        raise ValueError(f"Error: Peak line '{line.strip()}' does not have m/z and intensity values.")
                    
                    if len(meta_items) > len(peak_columns) - 2:
                        raise ValueError(f"Error: Peak line '{line.strip()}' has more metadata items than expected based on header.")

                    peak_entry = {'mz': float(mz_item), 'intensity': float(intensity_item)}
                    for i in range(len(meta_items)):
                        col = peak_columns[i+2]
                        m = meta_items[i]
                        if m != '':
                            peak_entry[col] = m
                    msp_reader.add_peak(**peak_entry)
                else:
                    k,v = line.split(":", 1)
                        
                    parsed_k, parsed_v = msp_reader.add_meta(k,v)

                    if parsed_k == "NumPeaks":
                        peak_flag = True
                
            except Exception as e:
                msp_reader.add_error_message(str(e), line_text=line)
            finally:
                pass

    ms_dataset = msp_reader.get_dataset()

    if spec_id_prefix is not None:
        set_spec_id(ms_dataset, spec_id_prefix)

    if return_header_map:
        return ms_dataset, msp_reader.header_map
    else:
        return ms_dataset

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
