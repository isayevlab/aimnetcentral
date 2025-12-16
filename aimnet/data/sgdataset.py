import os
from collections.abc import Callable, Sequence
from glob import glob
from typing import Any

import h5py
import numpy as np
from torch.utils.data.dataloader import DataLoader, default_collate


class DataGroup:
    """Dict-like container for data arrays with consistent shape.

    Args:

    `data (str | Dict[str, np.ndarray] | None)`: The data to be used in the dataset.
        It can be a string representing the path to the data file in NPZ format or a dictionary where
        keys are strings and values are numpy arrays.

    `keys (List[str] | None)`: A list of keys to be used from the data dictionary.

    `shard (Tuple[int, int] | None)`: A tuple representing the shard index and total shards.
    """

    def __init__(
        self,
        data: str | dict[str, np.ndarray] | h5py.Group | None = None,
        keys: list[str] | None = None,
        shard: tuple[int, int] | None = None,
    ):
        # main container for data
        self._data: dict[str, np.ndarray] = {}

        if data is None:
            data = {}

        s = slice(shard[0], None, shard[1]) if shard is not None else slice(None)

        # load  to dict
        if isinstance(data, str):
            if not os.path.isfile(data):
                raise FileNotFoundError(f"{data} does not exist or not a file.")
            data = np.load(data, mmap_mode="r")
            if not hasattr(data, "files"):
                raise ValueError(f"Data file {data} does not contain named arrays.")

        # take only keys
        if keys is None:
            keys = data.keys()  # type: ignore[union-attr]
        data = {k: v[s] for k, v in data.items() if k in keys}  # type: ignore[union-attr]

        # check data
        _n = None
        for k, v in data.items():
            if not isinstance(k, str):
                raise TypeError(f"Expected key to be of type str, but got {type(k).__name__}")
            if keys is not None and k not in keys:
                continue
            if _n is None:
                _n = len(v)
            if len(v) != _n:
                raise ValueError(f"Inconsistent data shape for key {k}. Expected first dimension {_n}, got {len(v)}.")
            self._data[k] = v

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError(f"Failed to set key of type {type(key)}, expected str.")
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Failed to set item of wrong type. Expected {type(np.ndarray)}, got {type(value)}.")
        if len(self) and len(value) != len(self):
            raise ValueError(f"Failed to set item of wrong shape. Expected {len(self)}, got {len(value)}.")
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __len__(self):
        return len(next(iter(self.values()))) if self._data else 0

    def to_dict(self):
        return self._data

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def keys(self):
        return self._data.keys()

    def pop(self, key):
        return self._data.pop(key)

    def rename_key(self, old, new):
        self[new] = self.pop(old)

    def sample(self, idx, keys=None) -> "DataGroup":
        """Return a new `DataGroup` with the data indexed by `idx`."""
        if keys is None:
            keys = self.keys()
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        return self.__class__({k: self[k][idx] for k in keys})

    def random_split(self, *fractions, seed=None):
        if not (0 < sum(fractions) <= 1):
            raise ValueError("Sum of fractions must be between 0 and 1.")
        if not all(f > 0 for f in fractions):
            raise ValueError("All fractions must be greater than 0.")
        idx = np.arange(len(self))
        np.random.seed(seed)
        np.random.shuffle(idx)
        sections = np.around(np.cumsum(fractions) * len(self)).astype(np.int64)
        return [self.sample(sidx) if len(sidx) else self.__class__() for sidx in np.array_split(idx, sections)]

    def cv_split(self, cv: int = 5, seed=None):
        """Return list of `cv` tuples containing train and val `DataGroup`s"""
        fractions = [1 / cv] * cv
        parts = self.random_split(*fractions, seed=seed)
        splits = []
        for icv in range(cv):
            val = parts[icv]
            _idx = [_i for _i in range(cv) if _i != icv]
            train = parts[_idx[0]]
            train.cat(*[parts[_i] for _i in _idx[1:]])
            splits.append((train, val))
        return splits

    def save(self, filename, compress=False):
        op = np.savez_compressed if compress else np.savez
        if len(self):
            op(filename, **self._data)

    def shuffle(self, seed=None):
        idx = np.arange(len(self))
        np.random.seed(seed)
        np.random.shuffle(idx)
        for k, v in self.items():
            self[k] = v[idx]

    def cat(self, *others):
        _n = set(self.keys())
        for other in others:
            if set(other.keys()) != _n:
                raise ValueError("Inconsistent data keys.")
        for k, v in self.items():
            self._data[k] = np.concatenate([v, *[other[k] for other in others]], axis=0)

    def iter_batched(self, batch_size=128, keys=None):
        idx = np.arange(len(self))
        idxs = np.array_split(idx, np.ceil(len(self) / batch_size))
        if keys is None:
            keys = self.keys()
        for idx in idxs:
            yield {k: v[idx] for k, v in self.items() if k in keys}

    def merge(self, other, strict=True):
        if strict:
            if set(self.keys()) != set(other.keys()):
                raise ValueError("Data keys do not match between the datasets.")
            keys = self.keys()
        else:
            keys = set(self.keys()) & set(other.keys())
        for k in list(self.keys()):
            if k in keys:
                self._data[k] = np.concatenate([self[k], other[k]], axis=0)
            else:
                self.pop(k)

    def apply_peratom_shift(self, sap_dict, key_in="energy", key_out="energy", numbers_key="numbers"):
        ntyp = max(sap_dict.keys()) + 1
        sap = np.zeros(ntyp) * np.nan
        for k, v in sap_dict.items():
            sap[k] = v
        self._data[key_out] = self[key_in] - sap[self[numbers_key]].sum(axis=-1)


class SizeGroupedDataset:
    def __init__(
        self,
        data: str | list[str] | dict[int, dict[str, np.ndarray]] | dict[int, DataGroup] | None = None,
        keys: list[str] | None = None,
        shard: tuple[int, int] | None = None,
    ):
        # main containers
        self._data: dict[int, DataGroup] = {}
        self._meta: dict[str, str] = {}

        # load data
        if isinstance(data, str):
            if os.path.isdir(data):
                self.load_datadir(data, keys=keys, shard=shard)
            else:
                self.load_h5(data, keys=keys, shard=shard)
        elif isinstance(data, (list, tuple)):
            self.load_files(data, shard=shard)
        elif isinstance(data, dict):
            self.load_dict(data)
        self.loader_mode = False
        self.x: list[str] = []
        self.y: list[str] = []

    def load_datadir(self, path, keys=None, shard: tuple[int, int] | None = None):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} does not exist or not a directory.")
        for f in glob(os.path.join(path, "???.npz")):
            k = int(os.path.basename(f)[:3])
            self[k] = DataGroup(f, keys=keys, shard=shard)

    def load_files(self, files, keys=None, shard: tuple[int, int] | None = None):
        for fil in files:
            if not os.path.isfile(fil):
                raise FileNotFoundError(f"{fil} does not exist or not a file.")
            k = int(os.path.splitext(os.path.basename(fil))[0])
            self[k] = DataGroup(fil, keys=keys, shard=shard)

    def load_dict(self, data, keys=None):
        for k, v in data.items():
            self[k] = DataGroup(v, keys=keys)

    def load_h5(self, data, keys=None, shard: tuple[int, int] | None = None):
        with h5py.File(data, "r") as f:
            for k, g in f.items():
                k = int(k)
                self[k] = DataGroup(g, keys=keys, shard=shard)
            self._meta = dict(f.attrs)  # type: ignore[attr-defined]

    def keys(self) -> list[int]:
        return sorted(self._data.keys())

    def values(self) -> list:
        return [self[k] for k in self.keys()]

    def items(self) -> list[tuple[int, Any]]:
        return [(k, self[k]) for k in self.keys()]

    def datakeys(self):
        return next(iter(self._data.values())).keys() if self._data else set()

    @property
    def groups(self) -> list[DataGroup]:
        return self.values()

    def __len__(self):
        return sum(len(d) for d in self.values())

    def __setitem__(self, key: int, value: DataGroup):
        if not isinstance(key, int):
            raise TypeError(f"Failed to set key of type {type(key)}, expected int.")
        if not isinstance(value, DataGroup):
            raise TypeError(f"Failed to set item of wrong type. Expected DataGroup, got {type(value)}.")
        if self._data and set(self.datakeys()) != set(value.keys()):
            raise ValueError("Wrong set of data keys.")
        self._data[key] = value

    def __getitem__(self, item: int | tuple[int, Sequence]) -> dict | tuple[dict, dict]:
        if isinstance(item, int):
            ret = self._data[item]
        else:
            grp, idx = item
            if self.loader_mode:
                ret = (
                    {k: v[idx] for k, v in self[grp].items() if k in self.x},  # type: ignore[union-attr, assignment]
                    {k: v[idx] for k, v in self[grp].items() if k in self.y},  # type: ignore[union-attr, assignment]
                )
            else:
                ret = {k: v[idx] for k, v in self[grp].items()}  # type: ignore[union-attr, assignment]
        return ret  # type: ignore[return-value]

    def __contains__(self, value):
        return value in self.keys()

    def rename_datakey(self, old, new):
        for g in self.groups:
            g.rename_key(old, new)

    def apply(self, fn):
        for grp in self.groups:
            fn(grp)

    def merge(self, other, strict=True):
        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        if strict:
            if set(other.datakeys()) != set(self.datakeys()):
                raise ValueError("Data keys do not match between the datasets.")
        else:
            keys = set(other.datakeys()) & set(self.datakeys())
            for k in list(self.datakeys()):
                if k not in keys:
                    for g in self.groups:
                        g.pop(k)
            for k in list(other.datakeys()):
                if k not in keys:
                    for g in other.groups:
                        g.pop(k)
        for k in other:
            if k in self:
                self[k].cat(other[k])  # type: ignore[attr-defined]
            else:
                self[k] = other[k]  # type: ignore[attr-defined]

    def random_split(self, *fractions, seed=None):
        splitted_groups = {}
        for k, v in self.items():
            splitted_groups[k] = v.random_split(*fractions, seed=seed)
        datasets = []
        for i in range(len(fractions)):
            datasets.append(
                self.__class__({k: splitted_groups[k][i] for k in splitted_groups if len(splitted_groups[k][i]) > 0})
            )
        return datasets

    def cv_split(self, cv: int = 5, seed=None):
        splitted_groups = {}
        for k, v in self.items():
            splitted_groups[k] = v.cv_split(cv, seed)
        datasets = []
        for i in range(cv):
            train = self.__class__({k: splitted_groups[k][i][0] for k in splitted_groups})
            val = self.__class__({k: splitted_groups[k][i][1] for k in splitted_groups})
            datasets.append((train, val))
        return datasets

    def shuffle(self, seed=None):
        for v in self.values():
            v.shuffle(seed)

    def save(self, dirname, namemap_fn: Callable | None = None, compress: bool = False):
        os.makedirs(dirname, exist_ok=True)
        if namemap_fn is None:
            namemap_fn = lambda x: f"{x:03d}.npz"
        for k, v in self.items():
            fname = os.path.join(dirname, namemap_fn(k))
            v.save(fname, compress=compress)

    def save_h5(self, filename):
        with h5py.File(filename, "w") as f:
            for n, g in self.items():
                n = f"{n:03d}"
                h5g = f.create_group(n)
                for k, v in g.items():
                    h5g.create_dataset(k, data=v)
                for k, v in self._meta.items():
                    f.attrs[k] = v

    def merge_groups(self, min_size=1, mode_atoms=False, atom_key="numbers"):
        # create list of supergroups
        sgroups = []
        n = 0
        sg = []
        for k, v in self.items():
            _n = len(v)
            if mode_atoms:
                _n *= v[atom_key].shape[1]
            n += _n
            sg.append(k)
            if n >= min_size:
                sgroups.append(sg)
                n = 0
                sg = []
        sgroups[-1].extend(sg)

        # merge
        keys = self.datakeys()
        for sg in sgroups:
            for k in keys:
                arrs = [self[n][k] for n in sg]  # type: ignore
                arrs = self._collate(arrs)
                self[sg[-1]]._data[k] = arrs  # type: ignore
            for n in sg[:-1]:
                del self._data[n]

    @staticmethod
    def _collate(arrs, pad_value=0):
        n = sum(a.shape[0] for a in arrs)
        shape = np.stack([a.shape[1:] for a in arrs], axis=0).max(axis=0)
        arr = np.full((n, *shape), pad_value, dtype=arrs[0].dtype)
        i = 0
        for a in arrs:
            n = a.shape[0]
            slices = tuple([slice(i, i + n)] + [slice(0, x) for x in a.shape[1:]])
            arr[slices] = a
            i += n
        return arr

    def concatenate(self, key):
        try:
            c = np.concatenate([g[key] for g in self.values() if len(g)], axis=0)
        except ValueError:
            c = np.concatenate([g[key].flatten() for g in self.values() if len(g)], axis=0)
        return c

    def apply_peratom_shift(self, key_in="energy", key_out="energy", numbers_key="numbers", sap_dict=None):
        if sap_dict is None:
            E = self.concatenate(key_in)
            ntyp = max(g[numbers_key].max() for g in self.groups) + 1
            eye = np.eye(ntyp, dtype=np.min_scalar_type(ntyp))
            F = np.concatenate([eye[g[numbers_key]].sum(-2) for g in self.values()])
            sap = np.linalg.lstsq(F, E, rcond=None)[0]
            present_elements = np.nonzero(F.sum(0))[0]
        else:
            ntyp = max(sap_dict.keys()) + 1
            sap = np.zeros(ntyp) * np.nan
            for k, v in sap_dict.items():
                sap[k] = v
            present_elements = sap_dict.keys()

        def fn(g):
            g[key_out] = g[key_in] - sap[g[numbers_key]].sum(axis=-1)

        self.apply(fn)

        return {i: sap[i] for i in present_elements}

    def apply_pertype_logratio(self, key_in="volumes", key_out="volumes", numbers_key="numbers", sap_dict=None):
        if sap_dict is None:
            numbers = self.concatenate("numbers")
            present_elements = sorted(np.unique(numbers))
            x = self.concatenate(key_in)
            sap_dict = {}
            for n in present_elements:
                sap_dict[n] = np.median(x[numbers == n])
        sap = np.zeros(max(sap_dict.keys()) + 1)
        for n, v in sap_dict.items():
            sap[n] = v

        def fn(g):
            g[key_out] = np.log(g[key_in] / sap[g[numbers_key]])

        self.apply(fn)
        return sap_dict

    def numpy_batches(self, batch_size=128, keys=None):
        for g in self.values():
            yield from g.iter_batched(batch_size, keys)

    def get_loader(self, sampler, x: list[str], y: list[str] | None = None, **loader_kwargs):
        self.loader_mode = True
        self.x = x
        self.y = y or []

        loader = DataLoader(self, batch_sampler=sampler, **loader_kwargs)  # type: ignore

        def _squeeze(t):
            for d in t:
                for v in d.values():
                    v.squeeze_(0)
            return t

        loader.collate_fn = lambda x: _squeeze(default_collate(x))
        return loader


class SizeGroupedSampler:
    def __init__(
        self,
        ds: SizeGroupedDataset,
        batch_size: int,
        batch_mode: str = "molecules",
        shuffle: bool = False,
        batches_per_epoch: int = -1,
    ):
        self.ds = ds
        self.batch_size = batch_size
        if batch_mode not in ["molecules", "atoms"]:
            raise ValueError(f"Unknown batch_mode {batch_mode}")
        self.batch_mode = batch_mode
        self.shuffle = shuffle
        self.batches_per_epoch = batches_per_epoch

    def __len__(self):
        if self.batches_per_epoch > 0:
            return self.batches_per_epoch
        else:
            return sum(self._get_num_batches_for_group(g) for g in self.ds.groups)

    def __iter__(self):
        return iter(self._samples_list())

    def _get_num_batches_for_group(self, g):
        if self.batch_mode == "molecules":
            return int(np.ceil(len(g) / self.batch_size))
        elif self.batch_mode == "atoms":
            return int(np.ceil(len(g) * g["numbers"].shape[1] / self.batch_size))
        else:
            raise ValueError(f"Unknown batch_mode: {self.batch_mode}")

    def _samples_list(self):
        samples = []
        for group_key, g in self.ds.items():
            n = len(g)
            if n == 0:
                continue
            idx = np.arange(n)
            if self.shuffle:
                np.random.shuffle(idx)
            n_batches = self._get_num_batches_for_group(g)
            samples.extend(((group_key, idx_batch),) for idx_batch in np.array_split(idx, n_batches))
        if self.shuffle:
            np.random.shuffle(samples)
        if self.batches_per_epoch > 0:
            if len(samples) > self.batches_per_epoch:
                samples = samples[: self.batches_per_epoch]
            else:
                # add some random duplicates
                idx = np.arange(len(samples))
                np.random.shuffle(idx)
                n = self.batches_per_epoch - len(samples)
                samples.extend([samples[i] for i in np.random.choice(idx, n, replace=True)])
        return samples
