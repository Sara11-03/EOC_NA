"""Archive an already constructed Pulser Sequence; does not submit a job.

Use archive_sequence(seq, out_dir, qubit_to_municipality=..., emulation_config=...).
The caller must provide the configuration actually used, not present-day defaults.
"""
from pathlib import Path
import json
import importlib.metadata


def archive_sequence(sequence, out_dir, *, qubit_to_municipality=None, emulation_config=None, run_metadata=None):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    text=sequence.to_abstract_repr(json_dumps_options={'indent':2})
    obj=json.loads(text)
    (out/'sequence.json').write_text(text+'\n',encoding='utf8')
    if 'device' in obj:
        (out/'device_snapshot.json').write_text(json.dumps(obj['device'],indent=2)+'\n',encoding='utf8')
    payload={'pulser_version':obj.get('pulser_version'),'qubit_to_municipality':qubit_to_municipality,
             'emulation_configuration_as_supplied_by_caller':emulation_config,'run_metadata':run_metadata,
             'note':'Archive of current object. Do not label this as a historical submission without a provenance match.'}
    (out/'execution_metadata.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    return out/'sequence.json'
