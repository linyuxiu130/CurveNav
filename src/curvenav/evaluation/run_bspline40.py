"""Run the forty cases with the ideal omnidirectional geometric executor."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue
import argparse
import json
import hashlib
import os
import subprocess
import sys
import time

GPU_COUNT=8
SHARD_COUNT=24


def runtime_env(runtime):
    python_paths=[str(Path(p).resolve(strict=True)) for p in runtime['python_paths']]
    if 'PYTHONPATH' in os.environ:python_paths.insert(0,os.environ['PYTHONPATH'])
    return dict(os.environ,PYTHONPATH=os.pathsep.join(python_paths),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    inputs=Path('data/curvenav_expert/inputs')
    parser.add_argument('--input',type=Path,default=inputs/'cases.json')
    parser.add_argument('--runtime-assets',type=Path,default=inputs/'runtime_assets.json')
    parser.add_argument('--runtime',type=Path,default=inputs/'runtime.json')
    parser.add_argument('--gpus',type=int,nargs='+',default=list(range(GPU_COUNT)))
    args=parser.parse_args()
    if len(set(args.gpus))!=len(args.gpus) or any(gpu<0 for gpu in args.gpus):parser.error('GPU IDs must be unique and nonnegative')
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    runtime=json.loads(args.runtime.read_text())
    env=runtime_env(runtime)
    command=[sys.executable,'-u','-m','curvenav.evaluation.bspline40']
    flags=['--output',str(args.output.resolve()),'--shards',str(SHARD_COUNT),
        '--input',str(args.input.resolve()),'--runtime-assets',str(args.runtime_assets.resolve())]
    if args.checkpoint:flags+=['--checkpoint',str(args.checkpoint.resolve())]
    pending=Queue()
    for shard in range(SHARD_COUNT):pending.put(shard)
    def worker(gpu):
        records=[]
        while True:
            try:shard=pending.get_nowait()
            except Empty:return records
            began=time.monotonic()
            with (args.output/f'job_{shard}.log').open('x') as log:
                result=subprocess.run(command+['run',*flags,'--shard',str(shard),'--device','0'],
                    env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
            record=dict(job_index=shard,gpu=gpu,shard=shard,exit_code=result.returncode,
                seconds=time.monotonic()-began)
            records.append(record);print(json.dumps(record),flush=True)
    jobs=[]
    with ThreadPoolExecutor(len(args.gpus)) as pool:
        for future in as_completed([pool.submit(worker,gpu) for gpu in args.gpus]):
            jobs.extend(future.result())
    jobs.sort(key=lambda job:job['shard'])
    assert len(jobs)==SHARD_COUNT and [job['shard'] for job in jobs]==list(range(SHARD_COUNT))
    (args.output/'jobs.json').write_text(json.dumps(dict(
        schedule=dict(original_jobs=24,gpu_workers=len(args.gpus),shards=SHARD_COUNT,
            actor_cold_starts=SHARD_COUNT,previous_static_actor_cold_starts=GPU_COUNT,
            tradeoff='More actor/source-snapshot cold starts for one original job per shard and dynamic GPU work assignment.'),
        runtime=dict(path=str(args.runtime.resolve()),sha256=hashlib.sha256(args.runtime.read_bytes()).hexdigest(),settings=runtime),
        jobs=jobs),indent=2)+'\n')
    assert all(job['exit_code']==0 for job in jobs),'Inspect retained per-job log'
    subprocess.run(command+['summarize',*flags],env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True)

if __name__=='__main__':main()
