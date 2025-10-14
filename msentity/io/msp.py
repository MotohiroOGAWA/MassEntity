from typing import Optional, Callable, Dict, List

from .IOContext import ReaderContext, parse_peak_text
from .constants import ErrorLogLevel
from ..core import MSDataset
from ..utils.annotate import set_spec_id


def read_msp(filepath, 
             encoding='utf-8', 
             return_header_map=False, 
             spec_id_prefix=None, 
             error_log_level: ErrorLogLevel = ErrorLogLevel.NONE,
             error_log_file=None,
             allow_duplicate_cols=False,
             show_progress=True,
             peak_parser: Optional[Callable[[str], List[Dict]]] = None,
             auto_peak_col_prefix: str = "column",
             ) -> MSDataset:
    
    msp_reader = ReaderContext(
        filepath, 
        error_log_level=error_log_level, 
        error_log_file=error_log_file,
        encoding=encoding,
        allow_duplicate_cols=allow_duplicate_cols,
        show_progress=show_progress,
        )
    msp_reader.file_type_name = "msp"
    
    with open(filepath, 'r', encoding=encoding) as f:
        peak_flag = False
        peak_text = ""
        for line in f.readlines():
            msp_reader.update(line)
            try:
                if not peak_flag and line == '\n':
                    pass

                elif peak_flag and line == '\n':
                    try:
                        if peak_parser is None:
                            peaks: List[Dict] = parse_peak_text(peak_text, auto_col_prefix=auto_peak_col_prefix)
                        else:
                            peaks: List[Dict] = peak_parser(peak_text)
                        for peak_entry in peaks:
                            msp_reader.add_peak(**peak_entry)
                    except Exception as e:
                        msp_reader.add_error_message(str(e), line_text=peak_text)

                    msp_reader.update_record()
                    peak_flag = False
                    peak_text = ""

                elif peak_flag:
                    peak_text += line
                    
                else:
                    k,v = line.split(":", 1)
                        
                    parsed_k, parsed_v = msp_reader.add_meta(k,v)

                    if parsed_k == "NumPeaks":
                        peak_flag = True
                
            except Exception as e:
                msp_reader.add_error_message(str(e), line_text=line)
            finally:
                pass
    
    if peak_text != "":
        if peak_parser is None:
            peaks: List[Dict] = parse_peak_text(peak_text)
        else:
            peaks: List[Dict] = peak_parser(peak_text)
        for peak_entry in peaks:
            msp_reader.add_peak(**peak_entry)
        msp_reader.update_record()
        peak_flag = False
        peak_text = ""

    ms_dataset = msp_reader.get_dataset()

    if spec_id_prefix is not None:
        set_spec_id(ms_dataset, spec_id_prefix)

    if return_header_map:
        return ms_dataset, msp_reader.header_map
    else:
        return ms_dataset

def write_msp(dataset: MSDataset, path: str, headers=None, header_map={}, peak_headers=None, encoding='utf-8', delimiter='\t'):
    """
    Save MSDataset to MSP file.
    """
    df = dataset.meta

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
        peak_headers = dataset.peaks.metadata.columns.tolist() if dataset.peaks.metadata is not None else []
    _peak_headers = []
    for c in peak_headers:
        if c not in dataset.peaks.metadata.columns:
            continue
        if c in ("mz", "intensity"):
            continue
        _peak_headers.append(c)
    peak_headers = _peak_headers.copy()


    with open(path, "w", encoding=encoding) as wf:
        for record in dataset:
            for key in headers:
                value = str(record[key])
                if value == "nan":
                    value = ""
                wf.write(f"{header_map[key]}: {value}\n")
            wf.write(f"NumPeaks: {record.n_peaks}\n")

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
                    meta_items_text = '" ; "'.join(valid_items)
                    wf.write(f'{mz_inten_pairs[i]}{delimiter}"{meta_items_text}"\n')
            wf.write("\n")

if __name__ == "__main__":
    pass
