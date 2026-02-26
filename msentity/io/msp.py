from typing import Optional, Callable, Dict, List
from tqdm import tqdm

from .IOContext import ReaderContext, parse_peak_text
from .constants import ErrorLogLevel
from ..core import MSDataset
from ..utils.annotate import set_spec_id


def read_msp(
    filepath,
    encoding="utf-8",
    return_header_map=False,
    spec_id_prefix=None,
    error_log_level: ErrorLogLevel = ErrorLogLevel.NONE,
    error_log_file=None,
    allow_duplicate_cols=False,
    show_progress=True,
    peak_parser: Optional[Callable[[str], List[Dict]]] = None,
    auto_peak_col_prefix: str = "column",
    *,
    peak_chunk_lines: int = 500,
) -> MSDataset:

    msp_reader = ReaderContext(
        filepath,
        file_type_name="msp",
        error_log_level=error_log_level,
        error_log_file=error_log_file,
        encoding=encoding,
        allow_duplicate_cols=allow_duplicate_cols,
        show_progress=show_progress,
    )

    with open(filepath, "r", encoding=encoding) as f:
        peak_flag = False

        # Buffer peak text in chunks to avoid large incremental string concatenation
        peak_chunks: List[str] = []   # list of chunk-strings
        peak_lines: List[str] = []    # current line buffer

        def _flush_peak_lines() -> None:
            """Flush buffered lines into peak_chunks as a single string."""
            if peak_lines:
                peak_chunks.append("".join(peak_lines))
                peak_lines.clear()

        def _get_peak_text_and_reset() -> str:
            """Finalize peak text for a record and reset buffers."""
            _flush_peak_lines()
            text = "".join(peak_chunks)
            peak_chunks.clear()
            return text

        for line in f:  # <-- stream, not readlines()
            msp_reader.update(line)
            try:
                if (not peak_flag) and (line == "\n"):
                    continue

                elif peak_flag and line == "\n":
                    try:
                        peak_text = _get_peak_text_and_reset()

                        if peak_parser is None:
                            peaks: List[Dict] = parse_peak_text(
                                peak_text,
                                auto_col_prefix=auto_peak_col_prefix,
                            )
                        else:
                            peaks: List[Dict] = peak_parser(peak_text)

                        for peak_entry in peaks:
                            msp_reader.add_peak(**peak_entry)

                    except Exception as e:
                        # Log the full peak_text for this record (may be large but only on error)
                        msp_reader.add_error_message(str(e), line_text=peak_text)

                    msp_reader.update_record()
                    peak_flag = False

                elif peak_flag:
                    peak_lines.append(line)
                    if peak_chunk_lines > 0 and len(peak_lines) >= peak_chunk_lines:
                        _flush_peak_lines()

                else:
                    k, v = line.split(":", 1)
                    parsed_k, parsed_v = msp_reader.add_meta(k, v)
                    if parsed_k == "NumPeaks":
                        peak_flag = True
                        peak_chunks.clear()
                        peak_lines.clear()

            except Exception as e:
                msp_reader.add_error_message(str(e), line_text=line)

    # Handle EOF without trailing blank line (if we are still in peak section)
    if peak_flag and (peak_chunks or peak_lines):
        peak_text = _get_peak_text_and_reset()
        try:
            if peak_parser is None:
                peaks: List[Dict] = parse_peak_text(peak_text, auto_col_prefix=auto_peak_col_prefix)
            else:
                peaks: List[Dict] = peak_parser(peak_text)
            for peak_entry in peaks:
                msp_reader.add_peak(**peak_entry)
            msp_reader.update_record()
        except Exception as e:
            msp_reader.add_error_message(str(e), line_text=peak_text)
            msp_reader.update_record()

    ms_dataset = msp_reader.get_dataset()

    if spec_id_prefix is not None:
        set_spec_id(ms_dataset, spec_id_prefix)

    if return_header_map:
        return ms_dataset, msp_reader.header_map
    return ms_dataset

def write_msp(
        dataset: MSDataset, 
        path: str, 
        headers=None, 
        header_map={}, 
        peak_headers=None, 
        encoding='utf-8', 
        delimiter='\t',
        show_progress=True,
        ):
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

    if show_progress:
        pbar = tqdm(total=len(dataset), desc="[Writing MSP]", unit="record", mininterval=1.0)
    else:
        pbar = None
    success_count = 0
    processed_count = 0

    with open(path, "w", encoding=encoding) as wf:
        for record in dataset:
            try:
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
                success_count += 1
            except Exception as e:
                print(f"Error writing record {processed_count}: {str(e)}")
            finally:
                processed_count += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({"Success": f'{success_count}/{processed_count}({success_count/processed_count*100:.1f}%)'})
    if pbar is not None:
        pbar.close()

if __name__ == "__main__":
    pass
