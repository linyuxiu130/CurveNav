"""Fixed40 must evaluate current code and one checkpoint, never mix stale shards."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from curvenav.evaluation import bspline40


def test_snapshot_includes_model_and_pinned_controller(tmp_path,monkeypatch):
    package=tmp_path/'package/curvenav'
    sources=['evaluation/bspline40.py','models/decoder.py','deployment/bspline_bundle.py']
    for name in sources:
        path=package/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
    runtime=tmp_path/'runtime.json';runtime.write_text('{}')
    output=tmp_path/'out';output.mkdir()
    monkeypatch.setattr(bspline40,'__file__',str(package/sources[0]))
    records=bspline40.snapshot_sources(output,runtime)
    assert {r['relative_path'] for r in records}=={
        *('curvenav/'+name for name in sources),'runtime_assets/runtime.json'}
    for record in records:
        assert Path(record['path']).read_bytes()==Path(record['snapshot_path']).read_bytes()
        assert bspline40.sha(Path(record['snapshot_path']))==record['sha256']


@pytest.mark.parametrize('mismatch',['checkpoint','source'])
def test_summary_rejects_mixed_checkpoint_or_source_before_reporting_success(tmp_path,mismatch):
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'weights')
    cases=tmp_path/'cases.json';cases.write_text('{"cases": []}')
    for rank in range(2):
        folder=tmp_path/f'shard_{rank}';folder.mkdir()
        source=folder/'decoder.py';source.write_text('different' if rank==1 and mismatch=='source' else 'same')
        manifest=dict(complete=True,input_sha256=bspline40.sha(cases),
            checkpoint_sha256='wrong' if rank==1 and mismatch=='checkpoint' else bspline40.sha(checkpoint),
            deterministic_algorithms=True,results=[],source_snapshot=[dict(
                relative_path='curvenav/models/decoder.py',snapshot_path=str(source),sha256=bspline40.sha(source))])
        (folder/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(AssertionError,match='checkpoint differs|different source code'):
        bspline40.summarize(SimpleNamespace(input=cases,checkpoint=checkpoint,output=tmp_path,shards=2))
    assert not (tmp_path/'summary.json').exists()
def test_installed_evaluator_builds_runtime_paths_without_parent_pythonpath(tmp_path,monkeypatch):
    import os
    from curvenav.evaluation.run_bspline40 import runtime_env
    monkeypatch.delenv('PYTHONPATH',raising=False)
    runtime=dict(python_paths=[str(tmp_path)])
    assert runtime_env(runtime)['PYTHONPATH']==str(tmp_path)
    monkeypatch.setenv('PYTHONPATH','/existing/source')
    assert runtime_env(runtime)['PYTHONPATH']==os.pathsep.join(['/existing/source',str(tmp_path)])
