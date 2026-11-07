"""Validated task contract and explicit SimpleVLN manifest conversion.

No assumptions about unpublished/version-dependent parquet column nesting:
--mapping maps canonical names to source dotted field paths.
"""
import argparse
import json
import math
from pathlib import Path


def scene_key(value):
    name = Path(str(value)).name
    for suffix in ('.basis.glb', '.glb', '.basis.navmesh', '.navmesh'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    if '-' in name and name.split('-', 1)[0].isdigit():
        name = name.split('-', 1)[1]
    return name


def resolve_scene(root, task):
    root = Path(root)
    if task.get('scene_path'):
        path = root / task['scene_path']
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    sid = task['scene_id']
    key = scene_key(sid)
    candidates = [root / sid, root / sid / (key + '.basis.glb'),
                  root / sid / (key + '.glb'), root / 'mp3d' / key / (key + '.glb'),
                  root / 'hm3d' / 'train' / sid / (key + '.basis.glb')]
    found = {p.resolve() for p in candidates if p.is_file() and p.suffix == '.glb'}
    if len(found) != 1:
        raise ValueError(f'Expected one scene asset for {sid}; set scene_path explicitly (found {len(found)})')
    return found.pop()


def vector(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length or not all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value
    ):
        raise ValueError(f'{name} must contain {length} finite numbers')


def load_tasks(path, training=False):
    tasks = json.loads(Path(path).read_text())
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('tasks must be a nonempty JSON list')
    for task in tasks:
        if not isinstance(task.get('scene_id'), str) or not task['scene_id']:
            raise ValueError('scene_id required')
        if not training:
            continue
        if task.get('split') != 'train':
            raise ValueError('Every training task must explicitly declare split=train')
        kind = task.get('task_type', 'eqa')
        if kind not in ('vln', 'eqa'):
            raise ValueError('task_type must be vln or eqa')
        vector(task.get('start_position'), 3, 'start_position')
        vector(task.get('start_rotation'), 4, 'start_rotation (Habitat xyzw)')
        if abs(sum(x*x for x in task['start_rotation']) - 1) > 1e-3:
            raise ValueError('start_rotation must be a unit xyzw quaternion')
        if kind == 'vln':
            if not isinstance(task.get('instruction'), str) or not task['instruction'].strip():
                raise ValueError('VLN instruction required')
            if not isinstance(task.get('goals'), list) or not task['goals']:
                raise ValueError('VLN requires goals=[{"position": [x,y,z]}]')
            for goal in task['goals']:
                vector(goal.get('position'), 3, 'goal position')
            threshold = task.get('success_distance', 3.0)
            if not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or threshold <= 0:
                raise ValueError('success_distance must be positive and finite')
            if task.get('evidence_sets'):
                raise ValueError('VLN must not use EQA evidence_sets')
        else:
            for key in ('question', 'type', 'answer'):
                if key not in task:
                    raise ValueError(f'EQA requires {key}')
            if 'evidence_sets' in task and (not task['evidence_sets'] or not all(task['evidence_sets'])):
                raise ValueError('evidence_sets must contain nonempty alternatives')
    return tasks


def check_scenes(tasks, held_out):
    def keys(rows):
        return {scene_key(t[k]) for t in rows for k in ('scene_id', 'scene_path') if t.get(k)}
    overlap = keys(tasks) & keys(held_out)
    if overlap:
        raise ValueError(f'Training and held-out scenes overlap: {sorted(overlap)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='JSON list or parquet; parquet requires pyarrow')
    parser.add_argument('--mapping', required=True, help='JSON mapping canonical keys to source dotted paths')
    parser.add_argument('--holdout-tasks', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    mapping = json.loads(Path(args.mapping).read_text())
    required = {'scene_id', 'instruction', 'start_position', 'start_rotation', 'goals'}
    allowed = required | {'episode_id', 'scene_path', 'success_distance'}
    if not required <= mapping.keys() or not mapping.keys() <= allowed:
        raise ValueError(f'Mapping requires {sorted(required)}; optional {sorted(allowed-required)}')
    if Path(args.input).suffix == '.parquet':
        import pyarrow.parquet as pq
        rows = pq.read_table(args.input).to_pylist()
    else:
        rows = json.loads(Path(args.input).read_text())
    converted = []
    for row in rows:
        task = {'task_type': 'vln', 'split': 'train', 'source': 'simplevln'}
        for key, dotted in mapping.items():
            value = row
            for part in dotted.split('.'):
                value = value[part]
            task[key] = value
        converted.append(task)
    # Validate before publishing output, including scene isolation.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory) / 'tasks.json'
        temp.write_text(json.dumps(converted))
        load_tasks(temp, training=True)
    check_scenes(converted, load_tasks(args.holdout_tasks))
    with Path(args.output).open('x') as stream:
        json.dump(converted, stream, indent=2)


if __name__ == '__main__':
    main()
