"""Prepare native memory and recorded candidates from the same causal observations."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import torch
from curvenav.deployment.bspline_bundle import load_bundle
from curvenav.data.bspline_replay import sha,require_native_geometry
from curvenav.data.six_frame_memory import MEMORY_CONTRACT
from curvenav.training.bspline_objective import live_condition


@torch.no_grad()
def prepare(manifests,actor,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    families=[];by_raw={}
    for family,path in enumerate(manifests):
        manifest=json.loads(path.read_text());records=[]
        for row,record in enumerate(manifest['items']):
            if sha(record['path'])!=record['sha256']:raise ValueError('label SHA mismatch')
            item=torch.load(record['path'],map_location='cpu',weights_only=False)
            key=item['raw_path'];by_raw.setdefault(key,[]).append((item,family,row))
            records.append(item)
        families.append((path,manifest,records))
    snapshots={}
    for raw_path,entries in by_raw.items():
        raw=torch.load(raw_path,map_location='cpu',weights_only=False,mmap=True)
        digest=sha(raw_path)
        times=np.asarray(raw['timestamps'],dtype=np.float64)
        for item,_,_ in entries:
            if item['clock']!=times[item['acquisitions']-1]:raise ValueError('label timestamp differs from raw acquisition clock')
        if any(item['raw_sha256']!=digest for item,_,_ in entries):raise ValueError('raw sensor SHA mismatch')
        assert raw['input_contract']==MEMORY_CONTRACT
        writes={write['acquisitions']:write['indices'] for write in raw['memory_writes']}
        cutoffs={item['acquisitions'] for item,_,_ in entries}
        if not cutoffs<=writes.keys():raise ValueError('label cutoff lacks a recorded memory write')
        memory=actor.new_memory()
        native_projection=memory.device.type=='cuda'
        folder=output/'memory'/digest;folder.mkdir(parents=True)
        for index in range(max(cutoffs)):
            pose=raw['body_to_world'][index]
            memory.update(raw['depths'][index],raw['camera_intrinsics'][index],raw['camera_to_body'][index],pose,float(times[index]))
            if index+1 in writes:memory.plan_write(writes[index+1])
            if index+1 in cutoffs:
                plan=memory.read_plan();plan['raw_sha256']=digest
                path=folder/f'{index+1:06d}.pt';torch.save(plan,path);snapshots[raw_path,index+1]=path
        print(json.dumps(dict(raw=digest,rows=len(entries),acquisitions=max(cutoffs))),flush=True)
    outputs=[]
    for family,(parent,manifest,items) in enumerate(families):
        folder=output/f'family_{family:03d}';(folder/'items').mkdir(parents=True)
        records=[];banks=[];raw_cache={};maps={}
        for item in items:
            item['memory_snapshot_path']=str(snapshots[item['raw_path'],item['acquisitions']].resolve())
            item['memory_plan_sha256']=sha(item['memory_snapshot_path'])
            if native_projection:
                key=item['raw_path']
                if key not in raw_cache:raw_cache[key]=torch.load(key,map_location='cpu',weights_only=False,mmap=True)
                raw=raw_cache[key];device=next(actor.policy.parameters()).device
                def provider(index):
                    cache_key=(key,index)
                    if cache_key not in maps:
                        depth=torch.as_tensor(np.asarray(raw['depths'][index],np.float32),device=device)[None,None]
                        maps[cache_key]=actor.policy.depth_encoder.image_features(depth)
                    return maps[cache_key]
                item['memory_plan']=torch.load(item['memory_snapshot_path'],map_location='cpu',weights_only=False)
                condition,_=live_condition(actor,item,raw,feature_provider=provider)
                banks.append(actor.native_bank(condition)['paths'][0].cpu())
                del item['memory_plan']
        if native_projection:
            candidates=folder/'candidates.npz';np.savez_compressed(candidates,paths=torch.stack(banks).numpy())
            preparation=dict(complete=True,parent_manifest=str(parent.resolve()),parent_manifest_sha256=sha(parent),
                model_sha256=actor.provenance['bundle_sha256'],collector_source_sha256=sha(Path(__file__)),
                sensor_data_and_expert_labels_changed=False,recorded_candidates_recomputed=True)
            result=folder/'candidate_preparation.json';result.write_text(json.dumps(preparation,indent=2)+'\n')
            binding=dict(result_path=str(result.resolve()),result_sha256=sha(result),
                candidates_path=str(candidates.resolve()),candidates_sha256=sha(candidates))
        for row,item in enumerate(items):
            item['input_contract']=MEMORY_CONTRACT
            if native_projection:
                item['critic_replay_paths']=banks[row]
                item['critic_replay_provenance'].update(binding,native_paths_float32_sha256=
                    hashlib.sha256(banks[row].contiguous().numpy().tobytes()).hexdigest())
            path=folder/'items'/f'{row:04d}.pt';torch.save(item,path)
            records.append(dict(manifest['items'][row],path=str(path.resolve()),sha256=sha(path)))
        updated=dict(manifest,items=records,input_contract=MEMORY_CONTRACT,
            geometry_input_contract='cuda_native_projection_v1' if native_projection else 'cpu_collection_pending_native_replay_v1',
            training_ready=native_projection,
            memory_preparation=dict(source_sha256=sha(Path(__file__)),parent_manifest=str(parent.resolve()),parent_sha256=sha(parent),
                labels_changed=False))
        if native_projection:updated['candidate_preparation']=preparation
        path=folder/'manifest.json';path.write_text(json.dumps(updated,indent=2)+'\n');outputs.append(str(path.resolve()))
    (output/'manifest.json').write_text(json.dumps(dict(complete=True,input_contract=MEMORY_CONTRACT,training_ready=native_projection,rows=sum(len(items) for _,_,items in families),
        manifests=outputs,labels_changed=False),indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests',type=Path,nargs='+',required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();require_native_geometry(args.device);torch.set_num_threads(1)
    actor=load_bundle(args.bundle,args.device);actor.provenance['bundle_sha256']=sha(args.bundle)
    prepare(args.manifests,actor,args.output)

if __name__=='__main__':main()
