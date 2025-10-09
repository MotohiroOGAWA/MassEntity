import os
from typing import List, Dict
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime
from .ItemParser import ItemParser
from .constants import ErrorLogLevel
from ..core import MSDataset, PeakSeries

class ReaderContext:
    def __init__(self, file_path: str, error_log_level: ErrorLogLevel = ErrorLogLevel.NONE, error_log_file: str = None, encoding: str = 'utf-8', show_progress: bool = True):
        self.file_type_name = ''
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)
        self.processed_size = 0
        self.encoding = encoding
        self.item_parser = ItemParser()
        self.error_log_level = error_log_level
        self.header_map = {}

        self.all_cols:dict[str,list] = {} # Create a data list for each column
        self.all_col_names = []
        self.all_peak:dict[list] = {'mz': [], 'intensity': []} # List of all peaks
        self.all_peak_meta_names = []
        self.offsets = [0]

        # update per record
        self._reset_record()
        self.record_cnt = 0
        self.success_cnt = 0

        # update per line
        self.line_count = 0
        

        if error_log_file is None:
            now = datetime.now().strftime("%Y%m%d%H%M%S")
            error_filename = os.path.splitext(file_path)[0] + f"_error_{now}"
            cnt = 1
            if os.path.exists(error_filename + ".txt"):
                while True:
                    if not os.path.exists(f'{error_filename}_{cnt}.txt'):
                        error_filename = f'{error_filename}_{cnt}'
                        break
                    else:
                        cnt += 1
            self.error_file_path = error_filename + ".txt"
        elif not os.path.exists(os.path.dirname(file_path)):
            raise ValueError(f"Error: Directory '{os.path.dirname(file_path)}' does not exist.")
        else:
            self.error_file_path = error_log_file

        self.pbar = self.progress_bar if show_progress else None

    def get_dataset(self) -> MSDataset:
        all_cols, all_col_names, all_peak, offsets, all_peak_meta_names = self.get_record_data()

        spectrum_meta = pd.DataFrame(all_cols, columns=all_col_names)

        ms_tensor = torch.tensor(all_peak['mz'])
        intensity_tensor = torch.tensor(all_peak['intensity'])
        peak_tensor = torch.stack((ms_tensor, intensity_tensor), dim=1)
        offsets_tensor = torch.tensor(offsets)
        if all_peak_meta_names == []:
            peak_meta_df = None
        else:
            peak_meta_df = pd.DataFrame({k: all_peak[k] for k in all_peak_meta_names}, columns=all_peak_meta_names)
        peak_series = PeakSeries(peak_tensor, offsets_tensor, peak_meta_df)

        ms_dataset = MSDataset(spectrum_meta, peak_series)

        return ms_dataset


    def get_record_data(self) -> dict:
        self.update_record()
        
        return self.all_cols, self.all_col_names, self.all_peak, self.offsets, self.all_peak_meta_names

    @property
    def progress_bar(self):
        return tqdm(total=self.file_size, desc=f"[Reading {self.file_type_name}]{os.path.basename(self.file_path)}", mininterval=0.5)
    
    def update(self, line: str):
        self.line_count += 1
        self.record_text += line

    def update_record(self):
        is_error_written = self._try_write_errors()

        if self.meta != {} or (self.peak['mz'] != [] and self.peak['intensity'] != []):
            if not self.error_flag:
                self._set_record()
                self.success_cnt += 1
            
            self.record_cnt += 1

        tmp = []
        for k in self.all_col_names:
            if k not in self.all_cols:
                tmp.append(k)
        for k in tmp:
            self.all_col_names.remove(k)

        tmp = []
        for k in self.header_map:
            if k not in self.all_col_names:
                tmp.append(k)
        for k in tmp:
            del self.header_map[k]

        tmp = []
        for k in self.all_peak_meta_names:
            if k not in self.all_peak:
                tmp.append(k)
        for k in tmp:
            self.all_peak_meta_names.remove(k)

            
        size = len(self.record_text.encode(self.encoding)) + 1
        self.processed_size += size
        if self.pbar is not None:
            self.pbar.update(size)
            self.pbar.set_postfix_str({"Success": f'{self.success_cnt}/{self.record_cnt}'})

        self._reset_record()

    def _reset_record(self):
        self.meta = {}
        self.peak:dict[list] = {'mz': [], 'intensity': []}
        self.record_text = ""
        self.error_text_list = []
        self.error_flag = False

    def _set_record(self):
        for k in self.meta:
            if k not in self.all_cols:
                self.all_cols[k] = [""] * self.success_cnt
        for k in self.all_cols:
            self.all_cols[k].append(self.meta.get(k, ""))

        for k in self.peak:
            if k not in self.all_peak:
                self.all_peak[k] = [""] * len(self.all_peak['mz'])
                self.all_peak_meta_names.append(k)
        
        peak_length = len(self.peak['mz'])
        for k in self.all_peak:
            self.all_peak[k].extend(self.peak.get(k, [""] * peak_length))
        self.offsets.append(len(self.all_peak['mz']))

    def add_meta(self, key: str, value):
        parsed_key, parsed_value = self.item_parser.parse_item_pair(key, value)
        if parsed_key not in self.all_cols:
            self.header_map[parsed_key] = key
            self.all_col_names.append(parsed_key)
        if parsed_key in self.meta:
            raise ValueError(f"Duplicate meta key: ({key} & {self.meta[parsed_key]}) -> {parsed_key}")
        self.meta[parsed_key] = parsed_value
        return parsed_key, parsed_value
    
    def add_peak(self, mz: float, intensity: float, **metadata):
        peak_entry = {
            "mz": mz,
            "intensity": intensity,
            **metadata 
        }
        for k in peak_entry:
            if k not in self.peak:
                self.peak[k] = [""] * len(self.peak['mz'])
        
        for k in self.peak:
            self.peak[k].append(peak_entry.get(k, ""))
        return peak_entry

    def add_error_message(self, message: str, line_text: str):
        self.error_flag = True
        message = message.strip().replace('\n', '\\n')
        line_text = line_text.strip().replace('\n', '\\n')
        self.error_text_list.append(f"[ERROR] Line ({self.line_count:05d}){line_text} | {message}")

    def _try_write_errors(self) -> bool:
        if self.error_flag and self.error_log_level != ErrorLogLevel.NONE:
            if self.error_log_level == ErrorLogLevel.DETAIL:
                self.error_text_list.append(self.record_text.strip())
            error_text = '\n'.join(self.error_text_list)
            if not os.path.exists(self.error_file_path):
                with open(self.error_file_path, "w") as ef:
                    ef.write('')
            with open(self.error_file_path, "a") as ef:
                ef.write(error_text + '\n\n')
            return True
        return False
    

def parse_peak_text(peak_text: str, auto_col_prefix: str = "column") -> List[Dict]:
    """
    Parse peak text into a list of peak dictionaries.
    """
    peak_columns = []
    peak_entry_list = []
    lines = peak_text.strip().split('\n')
    if len(lines) == 0:
        return peak_entry_list
    
    items = lines[0].strip().split()
    if len(items) >= 2:
        if(items[0].lower() == 'mz' and items[1].lower() == 'intensity'):
            peak_columns = items.copy()
            lines = lines[1:]
    if len(peak_columns) == 0:
        peak_columns = ['mz', 'intensity']

    for line in lines:
        items = line.strip().split(maxsplit=2)
        if len(items) == 3:
            mz_item = items[0]
            intensity_item = items[1]
            semi_split_fields = items[2].split(';')

            quote_start_idx = -1
            quote_end_idx = -1
            quote_char = ''
            meta_items = []
            merged_item = ''
            for i, item in enumerate(semi_split_fields):
                if item.strip().startswith('"') and quote_start_idx == -1:
                    quote_start_idx = i
                    quote_char = '"'
                if item.strip().startswith("'") and quote_start_idx == -1:
                    quote_start_idx = i
                    quote_char = "'"

                if quote_start_idx != -1 and item.strip().endswith(quote_char):
                    quote_end_idx = i

                if quote_start_idx != -1 and quote_end_idx != -1:
                    merged_item = ";".join(semi_split_fields[quote_start_idx:quote_end_idx+1])
                    merged_item = merged_item.strip().strip(quote_char)
                    quote_start_idx = -1
                    quote_end_idx = -1
                    quote_char = ''
                elif quote_start_idx != -1 and quote_end_idx == -1:
                    continue
                else:
                    merged_item = item
                
                if quote_start_idx == -1:
                    meta_items.append(merged_item.strip())
                    merged_item = ''
            if merged_item != '':
                raise ValueError(f"Error: Peak line '{line.strip()}' has unmatched quotes in metadata.")
                
        elif len(items) == 2:
            mz_item = items[0]
            intensity_item = items[1]
            meta_items = []

        else:
            raise ValueError(f"Error: Peak line '{line.strip()}' does not have m/z and intensity values.")
        
        # if len(meta_items) > len(peak_columns) - 2:
        #     raise ValueError(f"Error: Peak line '{line.strip()}' has more metadata items than expected based on header.")

        peak_entry = {'mz': float(mz_item), 'intensity': float(intensity_item)}
        for i in range(len(meta_items)):
            if i+2 >= len(peak_columns):
                col = f"{auto_col_prefix}{i+3-len(peak_columns)}"
            else:
                col = peak_columns[i+2]
            m = meta_items[i]
            if m != '':
                if col in peak_entry:
                    raise ValueError(f"Error: Duplicate peak metadata column '{col}' in line '{line.strip()}'.")
                peak_entry[col] = m
        peak_entry_list.append(peak_entry)
    return peak_entry_list
