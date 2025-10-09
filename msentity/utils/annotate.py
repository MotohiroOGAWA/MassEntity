from ..core.MSDataset import MSDataset

def set_spec_id(dataset:MSDataset, prefix:str = "") -> bool:
    if not isinstance(prefix, str):
        raise ValueError("Prefix must be string.")
    if 'SpecID' in dataset._spectrum_meta_ref.columns:
        print("Warning: 'SpecID' column already exists in the dataset.")
        return False
    
    n = len(dataset)
    width = len(str(n))
    spec_ids = [f"{prefix}{i+1:0{width}d}" for i in range(n)]
    dataset['SpecID'] = spec_ids
    return True