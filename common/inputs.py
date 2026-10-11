"""Training inputs contain only public initialization, Train, and Validation."""
from dataclasses import dataclass
from pathlib import Path
import json
from .ranking import order
from .runtime import read, sha256


@dataclass(frozen=True)
class FitInputs:
    root: Path

    @property
    def backbone(self):
        return self.root/'mpnet'

    def project_dir(self, project):
        return self.root/project

    def verify(self, project):
        manifest = read(self.root/'manifest.json')
        record = manifest['projects'][project]
        for name, expected in record['files'].items():
            if name not in ('train.jsonl', 'val.jsonl', 'train_queries.json', 'val_queries.json', 'mining.json'):
                raise ValueError('Unexpected training input')
            if sha256(self.project_dir(project)/name) != expected:
                raise ValueError(f'Input checksum mismatch: {project}/{name}')
        for name, expected in manifest['public_model']['files'].items():
            if sha256(self.backbone/name) != expected:
                raise ValueError(f'Public model checksum mismatch: {name}')
        return record

    def reports(self, project, phase):
        if phase not in ('train', 'val'):
            raise ValueError('FitInputs accepts only Train and Validation')
        with (self.project_dir(project)/f'{phase}.jsonl').open() as stream:
            reports = [json.loads(line) for line in stream]
        if not reports or any(r['phase'] != phase for r in reports):
            raise ValueError('Split membership mismatch')
        return sorted(reports, key=order)

    def queries(self, project, phase):
        if phase not in ('train', 'val'):
            raise ValueError('FitInputs accepts only Train and Validation')
        return read(self.project_dir(project)/f'{phase}_queries.json')

    def mining(self, project, reports, queries):
        value = read(self.project_dir(project)/'mining.json')
        if value['label_scope'] != 'train_only':
            raise ValueError('Negative sampling requires Train labels')
        if value['fit_ids'] != [r['id'] for r in sorted(reports, key=order)]:
            raise ValueError('Negative pool report identity mismatch')
        if [r['query'] for r in value['queries']] != queries:
            raise ValueError('Negative pool query identity mismatch')
        return value
