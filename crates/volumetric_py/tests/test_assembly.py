import numpy as np
import pytest

import volumetric as v


def two_spheres():
    """Sphere `a` at the origin fixed to the world, sphere `b` at x = 3
    swivelling about z through the origin."""
    p = v.Project()
    sphere = v.model_bytes("simple_sphere_model")
    p.add_model(sphere, id="a")
    p.add_op("translate_operator", ["a", {"dx": 3.0, "dy": 0.0, "dz": 0.0}], output="b", export=False)
    mechanism = {
        "parts": ["a", "b"],
        "joints": [
            {"name": "mount", "child": "a"},
            {"name": "swivel", "kind": "revolute", "parent": "a", "child": "b",
             "axis": {"origin": [0.0, 0.0, 0.0], "direction": [0.0, 0.0, 1.0]},
             "min": -180.0, "max": 180.0},
        ],
    }
    p.add_op("mechanism_operator", [mechanism, None], output="mech", export=True)
    p.add_op("assemble_operator", ["mech", "a", "b", {"swivel": 90.0}], output="chair")
    return p


def test_mechanism_and_assembly_round_trip_with_kinematics():
    out = two_spheres().run()
    mech = out["mech"].mechanism()
    assert mech.parts == ["a", "b"]
    assert mech.state_keys == ["swivel"]
    assert mech.default_state == {"swivel": 0.0}
    assert mech.ranges[0]["max"] == 180.0
    assert [j["name"] for j in mech.joints] == ["mount", "swivel"]

    rest = mech.pose()
    assert rest.shape == (2, 3, 4)
    np.testing.assert_allclose(rest[1], np.eye(3, 4))
    turned = mech.pose({"swivel": 90.0})
    # (3, 0, 0) on b goes to (0, 3, 0).
    np.testing.assert_allclose(turned[1] @ [3.0, 0.0, 0.0, 1.0], [0.0, 3.0, 0.0], atol=1e-12)
    with pytest.raises(ValueError):
        mech.pose({"swivel": 400.0})

    # A point on b at (0, 3, 0) after the quarter turn moves along -x per
    # degree of swivel: |ω × r| = 3 * pi/180.
    vel = mech.velocity("b", [0.0, 3.0, 0.0], {"swivel": 90.0})
    assert vel.shape == (1, 3)
    np.testing.assert_allclose(vel[0], [-3.0 * np.pi / 180.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(mech.velocity("a", [1.0, 0.0, 0.0]), [[0.0, 0.0, 0.0]])

    asm = out["chair"].assembly()
    assert out["chair"].kind == "Assembly" and out["chair_model"].kind == "Model"
    assert asm.parts == ["a", "b"] and len(asm) == 2
    assert asm.state == {"swivel": 90.0}
    assert asm.part_model("b")[:4] == b"\0asm"
    np.testing.assert_allclose(asm.poses(), turned)
    again = v.Assembly.decode(asm.encode())
    assert again.mechanism.state_keys == ["swivel"]
    assert v.Mechanism.decode(mech.encode()).parts == ["a", "b"]
