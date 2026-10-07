"""Capacity gates for complete, unpruned generation cohorts.

Estimates reserve space; they are not measurements of final corpus size.
The gate never deletes samples or changes their representation/count.
"""
import json
import shutil
from pathlib import Path


def capacity_reason(spec):
    destination=Path(spec['directory'])
    ancestor=destination
    while not ancestor.exists() and ancestor!=ancestor.parent:ancestor=ancestor.parent
    completed=0
    marker=spec.get('progress')
    if marker and marker.get('kind')=='family_draw_markers':
        for path in destination.glob('*/draw_*.json'):
            try:
                files=json.loads(path.read_text()).get('files',{})
                if len(files)==4 and all((path.parent/name).is_file() for name in files):completed+=4
            except (OSError,ValueError,TypeError):pass
        marker=None
    if marker and Path(marker['path']).is_file():
        try:
            value=json.loads(Path(marker['path']).read_text())
            completed=int(value.get(marker['field'],0))*int(marker.get('multiplier',1))
        except (OSError,ValueError,TypeError):pass
    remaining=max(0,int(spec['expected_pairs'])-completed)
    required=int(spec.get('reserve_bytes',25*2**30))+remaining*int(spec['estimated_bytes_per_pair'])
    free=shutil.disk_usage(ancestor).free
    if free<required:
        return 'storage: %.1f GiB free; %.1f GiB estimated for %d remaining paired clips plus reserve (%s)'%(free/2**30,required/2**30,remaining,destination)
    return None


def attach_generation_capacity(jobs,split_sizes):
    """Add explicit estimates to native/retained/external generation nodes.

    RGB-D estimate: measured 86.2 MB proposal, rounded up to 110 MB with
    room for the shared history. RGB uses its uncompressed RGB upper size,
    plus metadata/history allowance. Compression differs by output method.
    """
    for job in jobs:
        command=job.get('command',[])
        if '-m' not in command or '--output' not in command:continue
        module=command[command.index('-m')+1]
        if module in ('experiments.generation_exports.windows','experiments.external_baselines.sample_windows',
                      'experiments.retained_controls.core.export_windows'):
            count=int(command[command.index('--count')+1])
            draws=int(command[command.index('--draws')+1])
            pairs=count*draws;field='proposals';multiplier=1;marker='completion.json'
            rgb_only=module.startswith('experiments.external_baselines.')
        elif module in ('experiments.paired_interventions.sample','experiments.retained_controls.core.sample',
                         'experiments.retained_controls.diagnostic_sample') or (
                             module=='experiments.external_baselines.run' and 'sample' in command):
            split=command[command.index('--split')+1]
            draws=int(command[command.index('--draws')+1]) if '--draws' in command else 8
            pairs=int(split_sizes[split])*4*draws
            field='completed_family_draws';multiplier=4;marker='status.json'
            rgb_only=module.startswith('experiments.external_baselines.')
        else:continue
        directory=Path(command[command.index('--output')+1])
        job['storage_capacity']=dict(directory=str(directory),expected_pairs=pairs,
            estimated_bytes_per_pair=40_000_000 if rgb_only else 110_000_000,
            reserve_bytes=25*2**30,progress=dict(path=str(directory/marker),field=field,multiplier=multiplier),
            estimate_only=True,policy='Block insufficient capacity before loading a GPU model; preserve the entire cohort.')
        if module=='experiments.retained_controls.diagnostic_sample':
            job['storage_capacity']['progress']['path']=str(directory/'completion.json')
        if module=='experiments.external_baselines.run':
            job['storage_capacity']['progress']=dict(kind='family_draw_markers')
    return jobs
