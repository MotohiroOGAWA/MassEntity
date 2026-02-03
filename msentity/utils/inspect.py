import h5py

def print_hdf5_structure(
    path: str,
    *,
    show_attrs: bool = True,
    show_datasets: bool = True,
    max_depth: int | None = None,
) -> None:
    """
    Print the structure of an HDF5 file.

    Args:
        path (str): Path to the HDF5 file.
        show_attrs (bool): Whether to print attributes.
        show_datasets (bool): Whether to print dataset info (shape, dtype).
        max_depth (int | None): Maximum depth to traverse (None = unlimited).
    """

    def _visit(name: str, obj, depth: int):
        if max_depth is not None and depth > max_depth:
            return

        indent = "  " * depth

        # Group
        if isinstance(obj, h5py.Group):
            print(f"{indent}[Group] {name or '/'}")

            if show_attrs and obj.attrs:
                for k, v in obj.attrs.items():
                    print(f"{indent}  @attr {k}: {v}")

        # Dataset
        elif isinstance(obj, h5py.Dataset):
            if show_datasets:
                print(
                    f"{indent}[Dataset] {name} "
                    f"shape={obj.shape}, dtype={obj.dtype}"
                )
            else:
                print(f"{indent}[Dataset] {name}")

            if show_attrs and obj.attrs:
                for k, v in obj.attrs.items():
                    print(f"{indent}  @attr {k}: {v}")

    with h5py.File(path, "r") as f:
        print(f"HDF5 file: {path}")
        f.visititems(lambda name, obj: _visit(name, obj, name.count("/")))
