"""Read prepared train/dev/test files while preserving original task ordering."""
import json
from pathlib import Path

from .multi_task import prepared_groups


def load_prepared_data(directory):
    directory = Path(directory)
    indexed = []
    split = {}
    seed = ratios = None
    for name in ('train', 'dev', 'test'):
        data = json.loads((directory / f'{name}.json').read_text())
        if data.get('partition') != name or type(data.get('seed')) is not int:
            raise ValueError(f'Invalid partition metadata: {name}')
        if seed is None:
            seed, ratios = data['seed'], data['split_ratio']
            if not isinstance(ratios, list) or len(ratios) != 3 or any(type(v) is not int or v <= 0 for v in ratios):
                raise ValueError('Invalid split_ratio')
        if data['seed'] != seed or data['split_ratio'] != ratios:
            raise ValueError('Inconsistent partition seeds or ratios')
        tasks, indices = data['tasks'], data['task_indices']
        if len(tasks) != len(indices) or any(type(i) is not int for i in indices):
            raise ValueError('Invalid original task indices')
        indexed.extend(zip(indices, tasks))
        split[name] = [t['id'] for t in tasks]
    if sorted(i for i, _ in indexed) != list(range(len(indexed))):
        raise ValueError('Task indices must uniquely cover the dataset')
    tasks = [t for _, t in sorted(indexed, key=lambda item: item[0])]
    split['seed'] = seed
    prepared_groups(tasks, split)
    return {'tasks': tasks, 'split': split, 'split_ratio': ratios}
