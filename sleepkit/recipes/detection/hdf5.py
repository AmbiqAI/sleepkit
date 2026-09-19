"""Require HDF5 payloads to live in the file covered by its content hash."""


def require_self_contained(stream):
    """Reject indirect storage while allowing ordinary internal hard links.

    A file hash cannot bind linked files, virtual sources, or externally stored
    dataset bytes. Soft links are also excluded to keep the supported source
    contract explicit. Track object addresses because hard-linked groups may cycle.
    """
    import h5py

    visited = set()

    def inspect(obj):
        address = h5py.h5o.get_info(obj.id).addr
        if address in visited:
            return
        visited.add(address)
        if isinstance(obj, h5py.Dataset):
            if obj.is_virtual or obj.external:
                raise ValueError("HDF5 datasets must use self-contained storage; external storage and VDS are unsupported")
        elif isinstance(obj, h5py.Group):
            for name in obj:
                if not isinstance(obj.get(name, getlink=True), h5py.HardLink):
                    raise ValueError("HDF5 must be self-contained; soft and external links are unsupported")
                inspect(obj[name])

    inspect(stream)
