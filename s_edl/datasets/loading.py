"""Utilities for loading datasets from local files or the HuggingFace Hub."""
from datasets import Dataset
from datasets import load_dataset as _hf_load_dataset


def load_dataset_split(source_cfg, split: str) -> Dataset:
    """Load a dataset split from a local JSONL file or the HuggingFace Hub.

    The ``split`` key works consistently for both sources:

    - **HuggingFace Hub**: selects the named split (e.g. ``train``, ``test``).
    - **Local JSONL**: filters rows where the ``split`` column equals the value.

    An optional ``filters`` dict provides additional field-value filtering
    (applied after split selection for both sources).

    Args:
        source_cfg: ConfigDict with either:
            - ``path``: path to a local JSONL file.
            - ``dataset_name`` + optional ``dataset_config``: HuggingFace Hub dataset.
            Both modes support an optional ``split`` and ``filters`` dict.
        split: Default split name when ``source_cfg`` does not specify one.

    Returns:
        The loaded dataset as a :class:`datasets.Dataset`.

    Raises:
        ValueError: If ``split`` is specified both at the top level and
            inside ``filters``.
    """
    filters = source_cfg.get('filters')
    if 'split' in source_cfg and filters and 'split' in filters:
        raise ValueError(
            "Cannot specify 'split' both at the top level and inside 'filters'. "
            'Use one or the other.')

    if 'path' in source_cfg:
        ds = Dataset.from_json(source_cfg.path)
        # Filter by split column if specified in config
        if 'split' in source_cfg:
            split_val = source_cfg.split
            ds = ds.filter(lambda x, s=split_val: x['split'] == s)
    else:
        actual_split = source_cfg.get('split', split)
        ds = _hf_load_dataset(
            source_cfg.dataset_name,
            source_cfg.get('dataset_config'),
            split=actual_split,
        )

    if filters:
        for field, value in filters.items():
            ds = ds.filter(lambda x, f=field, v=value: x[f] == v)

    max_samples = source_cfg.get('max_samples')
    if max_samples is not None and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    return ds
