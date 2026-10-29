"""The six-frame spatial-memory B-spline model and strict checkpoint I/O."""
from dataclasses import dataclass, replace
import torch
from torch import nn
from curvenav.config_io import config_from_mapping
from curvenav.factory import build_policy
from curvenav.models.policy import generate_candidate_bank, INFERENCE_CANDIDATES, INTEGRATION_STEPS
from curvenav.deployment.streaming_memory import encode_memory
from curvenav.data.six_frame_memory import SixFrameMemory, MEMORY_CONTRACT
from curvenav.encoders.temporal_geometry import FeatureGeometryDecoder

FORMAT = 'bspline_unified_spatial_omni_v4'
ACTOR_SEMANTICS = 'native16_euler2_unified_spatial_independent_free_v9'
CANDIDATES = INFERENCE_CANDIDATES

@dataclass
class BSplineMemoryBundle:
    config: object
    policy: nn.Module
    head: nn.Module
    architecture: dict
    provenance: dict

    def encode(self, memory, goal, goal_present):
        memory.write()
        plan,visual=memory.packet()
        return encode_memory(self.policy,self.head,plan,visual,goal,goal_present)[0]

    def native_bank(self, condition, *, checkpoint_steps=False):
        return generate_candidate_bank(self.policy, condition, checkpoint_steps=checkpoint_steps)

    def candidates(self, condition, goal):
        """Score the shared native bank and return its unchanged argmax curve."""
        if condition.tokens.shape[0] != 1:
            raise ValueError('native actor bank requires a single scene')
        goal = torch.as_tensor(goal, dtype=torch.float32, device=condition.tokens.device).reshape(1, 2)
        present = bool(condition.goal_present[0, 0])
        if not present:
            goal = torch.zeros_like(goal)
            condition = replace(condition, terminal_goal=torch.zeros_like(condition.terminal_goal))
        bank = self.native_bank(condition)
        scores = self.policy.score_candidate_paths(condition, goal, bank['paths'])[0]
        return dict(values=bank['values'][0], paths=bank['paths'][0], tangent=bank['tangent'][0],
            scores=scores, selected_index=int(scores.argmax()),
            source_ids=list(range(CANDIDATES)) if present else list(range(3*CANDIDATES, 4*CANDIDATES)),
            source_groups=['goal' if present else 'no_goal']*CANDIDATES,
            mode='native16', structural_proposals_applied=False)

    def new_memory(self, *, seed=0):
        return SixFrameMemory(self.policy, seed=seed, **self.architecture['memory'])


def create_bundle(config):
    """Initialize the single current graph directly; no checkpoint-dependent construction."""
    assert config.data.observation_frames==6 and config.trajectory_decoder.integration_steps==2
    policy=build_policy(config)
    head=FeatureGeometryDecoder(config.depth_encoder.model_dim+8,policy.planning_horizon_m,policy.depth_encoder.metric_projector.radius)
    architecture=dict(actor_semantics=ACTOR_SEMANTICS,memory_contract=MEMORY_CONTRACT,
        precision='bf16_neural_fp32_fusion',memory=dict(capacity=32768))
    return BSplineMemoryBundle(config,policy.eval(),head.eval(),architecture,dict(initialization='fresh'))


def load_bundle(path, device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['format']==FORMAT
    actor=create_bundle(config_from_mapping(saved['base_config']))
    assert saved['architecture']==actor.architecture, 'Checkpoint architecture differs'
    for name in ('policy','head'):
        module=getattr(actor,name);module.load_state_dict(saved[name],strict=True);module.to(device).eval()
    for name in ('control_increment_mean_xy_m','control_increment_std_xy_m'):
        assert torch.equal(getattr(actor.policy.curve_codec,name).cpu(),torch.tensor(getattr(actor.config.trajectory,name)))
    actor.provenance=saved['provenance']
    return actor


def save_bundle(path, *, base_config, policy, head, architecture, provenance):
    state=lambda module:{name:value.detach().cpu() for name,value in module.state_dict().items()}
    torch.save(dict(format=FORMAT,base_config=base_config,architecture=architecture,
        provenance=provenance,policy=state(policy),head=state(head)),path)


def main():
    import argparse
    from pathlib import Path
    import yaml
    parser=argparse.ArgumentParser(description='Initialize the six-frame spatial-memory B-spline model')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args();torch.manual_seed(args.seed)
    raw=yaml.safe_load(args.config.read_text());actor=create_bundle(config_from_mapping(raw))
    with args.output.open('xb') as stream:
        save_bundle(stream,base_config=raw,policy=actor.policy,head=actor.head,
            architecture=actor.architecture,provenance=dict(actor.provenance,initialization_seed=args.seed))


if __name__=='__main__':main()
