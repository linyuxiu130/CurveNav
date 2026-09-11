from io import BytesIO
import numpy as np
import pytest
from curvenav.deployment.interface import CurveNavNpzInterface, RESPONSE_FIELDS
from curvenav.deployment.runtime import RuntimePrediction
from test_depth_memory import condition


class _FakeRuntime:
    batch_size = 0

    def reset(self, batch_size):
        self.batch_size = batch_size

    def step(self, point_goals, context):
        self.context = context
        return RuntimePrediction(np.zeros((len(point_goals), 64, 3), np.float32))


def _payload(**updates):
    values = {name: value.numpy() for name, value in vars(condition(2)).items()}
    values["depth"] = values["depth"].astype(np.float16)
    values.update(updates)
    stream = BytesIO()
    np.savez(stream, **values)
    return stream.getvalue()


def test_npz_interface_uses_one_stateless_snapshot_contract():
    runtime = _FakeRuntime()
    response = CurveNavNpzInterface(runtime).predict(_payload())
    assert runtime.batch_size == 2
    np.testing.assert_array_equal(
        runtime.context["camera_intrinsics"], condition(2).camera_intrinsics.numpy()
    )
    with np.load(BytesIO(response), allow_pickle=False) as archive:
        assert frozenset(archive.files) == RESPONSE_FIELDS
        assert archive["path"].shape == (2, 64, 3)


def test_npz_interface_requires_exact_fields_and_dtypes():
    interface = CurveNavNpzInterface(_FakeRuntime())
    with pytest.raises(ValueError, match="request fields"):
        interface.predict(_payload(unexpected=np.zeros(1)))
    with pytest.raises(TypeError, match="point_goal"):
        interface.predict(_payload(point_goal=np.zeros((2, 2), np.float64)))
