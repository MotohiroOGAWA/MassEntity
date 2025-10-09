import os
import re
from tqdm import tqdm
from datetime import datetime
from typing import Tuple

import pandas as pd
import numpy as np
import torch

from .IOContext import ReaderContext
from .constants import ErrorLogLevel
from ..core import MSDataset, PeakSeries
from ..utils.annotate import set_spec_id


def read_mgf(filepath,
             encoding: str = "utf-8",
             return_header_map: bool = False,
             spec_id_prefix: str = None,
             error_log_level: ErrorLogLevel = ErrorLogLevel.NONE,
             error_log_file=None,
             show_progress: bool = True,
             ) -> MSDataset:
    """
    Read MGF (Mascot Generic Format) file and return as MSDataset.

    Args:
        filepath (str): Path to MGF file.
        encoding (str): File encoding.
        return_header_map (bool): Return header map along with dataset.
        spec_id_prefix (str): Prefix for spectrum ID (optional).
        error_log_level (ErrorLogLevel): Logging level for errors.
        show_progress (bool): Display tqdm progress bar.

    Returns:
        MSDataset
    """

    mgf_reader = ReaderContext(
        filepath,
        error_log_level=error_log_level,
        error_log_file=error_log_file,
        encoding=encoding,
        show_progress=show_progress,
    )
    mgf_reader.file_type_name = "mgf"

    with open(filepath, "r", encoding=encoding) as f:
        peak_flag = False
        for line in f:
            mgf_reader.update(line)
            try:
                if not peak_flag and line == '\n':
                    continue

                if '=' not in line and not peak_flag:
                    peak_flag = True

                # --- MGF block delimiters ---
                if line.upper().startswith("BEGIN IONS"):
                    peak_flag = False
                    peak_columns = []
                elif line.upper().startswith("END IONS"):
                    mgf_reader.update_record()
                    peak_flag = False

                # --- Inside an IONS block ---
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
                    mgf_reader.add_peak(**peak_entry)
                else:
                    # --- Metadata lines ---
                    if '=' in line:
                        k,v = line.split("=", 1)
                        parsed_k, parsed_v = mgf_reader.add_meta(k,v)
                    else:
                        raise ValueError(f"Error: Metadata line '{line.strip()}' does not contain '=' character.")

            except Exception as e:
                mgf_reader.add_error_message(str(e), line_text=line)
            finally:
                pass

    ms_dataset = mgf_reader.get_dataset()

    # Assign spectrum ID prefix if requested
    if spec_id_prefix is not None:
        set_spec_id(ms_dataset, spec_id_prefix)

    if return_header_map:
        return ms_dataset, mgf_reader.header_map
    else:
        return ms_dataset

def write_mgf(dataset: MSDataset, path: str, headers=None, header_map={}, peak_headers=None, encoding='utf-8', delimiter='\t'):
    """
    Save MSDataset to MGF file.
    """
    df = dataset.meta_copy

    if headers is None:
        headers = dataset._columns
    _headers = []
    for c in headers:
        if c not in df.columns:
            continue
        if c == "NumPeaks":
            continue
        _headers.append(c)
    headers = _headers.copy()
    for c in headers:
        if c not in header_map.keys():
            header_map[c] = c

    if peak_headers is None:
        peak_headers = dataset.peaks._metadata.columns.tolist() if dataset.peaks._metadata is not None else []
    _peak_headers = []
    for c in peak_headers:
        if c not in dataset.peaks._metadata.columns:
            continue
        if c in ("mz", "intensity"):
            continue
        _peak_headers.append(c)
    peak_headers = _peak_headers.copy()


    with open(path, "w", encoding=encoding) as wf:
        for record in dataset:
            wf.write("BEGIN IONS\n")
            for key in headers:
                value = str(record[key])
                if value == "nan":
                    value = ""
                wf.write(f"{header_map[key]}={value}\n")

            peak_meta_columns = set()
            mz_inten_pairs = []
            peak_meta_items = []
            peak_meta_empties = []
            for peak in record.peaks:
                mz_inten_pair = f"{peak.mz}{delimiter}{peak.intensity}"
                items = []
                meta_empty = True
                for col in peak_headers:
                    item = str(peak.metadata.get(col, ""))
                    if item == "nan":
                        item = ""
                    if item != '':
                        peak_meta_columns.add(col)
                        meta_empty = False
                    items.append(item)
                mz_inten_pairs.append(mz_inten_pair)
                peak_meta_items.append(items)
                peak_meta_empties.append(meta_empty)
            
            valid_peak_header_idxs = []
            if len(peak_meta_columns) > 0:
                peak_columns = ['mz', 'intensity'] + [col for col in peak_headers if col in peak_meta_columns]
                valid_peak_header_idxs = [i for i, col in enumerate(peak_headers) if col in peak_meta_columns]
                wf.write(delimiter.join(peak_columns) + '\n')
            for i, items in enumerate(peak_meta_items):
                valid_items = [items[j] for j in valid_peak_header_idxs]
                if all((item == '') for item in valid_items):
                    wf.write(f"{mz_inten_pairs[i]}\n")
                else:
                    meta_items_text = ' ; '.join(valid_items)
                    wf.write(f"{mz_inten_pairs[i]}{delimiter}{meta_items_text}\n")
            wf.write("END IONS\n\n")
            wf.write("\n")

