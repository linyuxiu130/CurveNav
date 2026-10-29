"""Exact B1 image maps bound to a frozen CNN and immutable raw sensor files."""
from pathlib import Path
import argparse,hashlib,inspect,json
import numpy as np
import torch
from curvenav.data.bspline_replay import sha
from curvenav.data.six_frame_memory import MEMORY_CONTRACT


def image_modules(actor):
    encoder=actor.policy.depth_encoder
    return (encoder.backbone,encoder.spatial_projection)


def cnn_identity(actor):
    digest=hashlib.sha256()
    for index,module in enumerate(image_modules(actor)):
        for name,value in module.state_dict().items():
            value=value.detach().cpu().contiguous()
            digest.update(str((index,name,str(value.dtype),tuple(value.shape))).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    encoder=actor.policy.depth_encoder
    return dict(weights_sha256=digest.hexdigest(),encoder_source_sha256=sha(inspect.getfile(type(encoder))),
        torch_version=torch.__version__,input_contract=MEMORY_CONTRACT,execution='B1',dtype=str(encoder.neural_dtype),deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def required_frames(records):
    frames={}
    for item,raw,*_ in records:
        entry=frames.setdefault(item['raw_sha256'],dict(raw=raw,indices=set()))
        entry['indices'].update(map(int,np.unique(item['memory_plan']['frame_ids'])))
    return frames


@torch.no_grad()
def write_cache(actor,records,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    device=next(actor.policy.parameters()).device
    identity=cnn_identity(actor);files={}
    for raw_sha,entry in required_frames(records).items():
        indices=sorted(entry['indices']);maps=[]
        for index in indices:
            depth=torch.as_tensor(np.asarray(entry['raw']['depths'][index],dtype=np.float32),device=device)[None,None]
            maps.append(actor.policy.depth_encoder.image_features(depth).cpu())
        path=output/f'{raw_sha}.pt'
        torch.save(dict(indices=indices,maps=torch.cat(maps)),path)
        files[raw_sha]=dict(path=str(path.resolve()),sha256=sha(path),frames=len(indices))
    (output/'manifest.json').write_text(json.dumps(dict(complete=True,cnn=identity,files=files),indent=2)+'\n')


def load_cache(actor,records,path):
    manifest=json.loads(Path(path).read_text())
    if not manifest['complete'] or manifest['cnn']!=cnn_identity(actor):
        raise ValueError('CNN cache does not match frozen weights, encoder source, precision or runtime')
    maps={}
    for raw_sha,entry in required_frames(records).items():
        record=manifest['files'][raw_sha]
        if sha(record['path'])!=record['sha256']:raise ValueError('CNN cache file SHA mismatch')
        data=torch.load(record['path'],map_location='cpu',weights_only=False,mmap=True)
        positions={index:position for position,index in enumerate(data['indices'])}
        for index in entry['indices']:maps[raw_sha,index]=data['maps'][positions[index]:positions[index]+1]
    for module in image_modules(actor):module.eval().requires_grad_(False)
    return maps


def main():
    from curvenav.deployment.bspline_bundle import load_bundle
    from curvenav.data.bspline_dataset import load_items
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--manifests',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True,warn_only=True)
    actor=load_bundle(args.bundle,args.device)
    records,*_=load_items(args.manifests,actor)
    write_cache(actor,records,args.output)

if __name__=='__main__':main()
