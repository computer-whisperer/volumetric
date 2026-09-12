import numpy as np
import pytest

import volumetric as v


def cylinder_project():
    p = v.Project()
    p.add_op("cylinder_operator", [{"radius": 0.05}, [0.0, 0.0, 0.0], [0.0, 0.0, 0.2]], output="post")
    return p


def test_catalog_lists_operators_with_metadata():
    names = v.operators()
    assert "cylinder_operator" in names and "boolean_operator" in names
    info = v.operator_info("cylinder_operator")
    assert info["name"] == "cylinder_operator"
    assert [slot["type"] for slot in info["inputs"]] == ["CBOR configuration", "VecF64(3)", "VecF64(3)"]
    assert info["variadic"] is None
    assert "radius" in info["docs"] or "radius" in info["inputs"][0]["cddl"]
    assert v.operator_info("boolean_operator")["variadic"] == 0
    assert "simple_sphere_model" in v.models()


def test_build_run_and_mesh_a_cylinder():
    p = cylinder_project()
    assert p.exports == ["post"]
    (step,) = p.steps()
    assert step["operator"] == "cylinder_operator" and step["outputs"] == ["post"]
    assert step["inputs"][1:] == [24, 24]  # two VecF64(3) inline, the config's CBOR first
    assert p.validate() == []
    out = p.run()
    assert list(out) == ["post"]
    post = out["post"]
    assert post.kind == "Model" and post.value is None and post.bytes[:4] == b"\0asm"
    mesh = post.mesh(max_depth=3)
    assert mesh.kind == "Tri3"
    assert mesh.nodes.shape[1] == 3 and mesh.elements.shape[1] == 3
    assert mesh.elements.dtype == np.uint32
    lo, hi = mesh.nodes.min(axis=0), mesh.nodes.max(axis=0)
    assert np.allclose(lo, [-0.05, -0.05, 0.0], atol=0.004)
    assert np.allclose(hi, [0.05, 0.05, 0.2], atol=0.004)
    assert mesh.node_fields["normal"].shape == mesh.nodes.shape


def test_values_come_back_typed():
    p = cylinder_project()
    p.add_op("subspace_operator", [{"kind": "plane"}, [0.0, 0.0, 0.1], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], output="plane")
    vid = p.add_asset({"a": 1.5, "b": -2.0}, "f64map", id="values")
    assert vid == "values" and ("values", "F64Map") in p.asset_ids()
    p.add_op("f64_map_merge_operator", ["values"], output="merged")  # imports are not exports; a step's output is
    out = p.run()
    assert out["merged"].kind == "F64Map" and out["merged"].value == {"a": 1.5, "b": -2.0}
    plane = out["plane"].value
    assert plane["rank"] == 2 and plane["basis"].shape == (2, 3)
    assert np.allclose(plane["origin"], [0.0, 0.0, 0.1])


def test_boolean_is_variadic_and_patterns_follow():
    p = cylinder_project()
    p.add_op("cylinder_operator", [{"radius": 0.05}, [0.1, 0.0, 0.0], [0.1, 0.0, 0.2]], output="post2", export=False)
    (both,) = p.add_op("boolean_operator", ["post", "post2", {"op": "union"}], output="both")
    assert both == "both" and p.exports == ["post", "both"]
    out = p.run()
    mesh = out["both"].mesh(max_depth=3)
    assert mesh.nodes[:, 0].max() > 0.14


def test_wrong_inputs_are_refused_with_the_slot_named():
    p = v.Project()
    with pytest.raises(ValueError, match="expects 3 input"):
        p.add_op("cylinder_operator", [{"radius": 0.05}])
    with pytest.raises(ValueError, match=r"VecF64\(3\)"):
        p.add_op("cylinder_operator", [{"radius": 0.05}, [0.0, 0.0], [0.0, 0.0, 0.2]])
    with pytest.raises(ValueError, match="unknown operator"):
        p.add_op("no_such_operator", [])
    with pytest.raises(ValueError, match="unknown asset kind"):
        p.add_asset(b"x", "mesh")
    with pytest.raises(ValueError, match="WASM module"):
        p.add_asset(b"\0asm\1\0\0\0", "blob")


def test_save_and_open_round_trip(tmp_path):
    p = cylinder_project()
    path = tmp_path / "post.vproj"
    p.save(str(path))
    q = v.Project.open(str(path))
    assert q.exports == ["post"] and q.asset_ids() == p.asset_ids()
    r = v.Project.from_bytes(p.to_bytes())
    assert r.steps() == p.steps()
    assert r.run()["post"].bytes == p.run()["post"].bytes


def test_bundled_model_is_an_import():
    p = v.Project()
    mid = p.add_model(v.model_bytes("simple_sphere_model"), id="ball")
    assert mid == "ball"
    assert ("ball", "Model") in p.asset_ids()


def test_models_sample_in_process():
    p = cylinder_project()
    post = p.run()["post"]
    assert post.dimensions == 3
    lo, hi = post.bounds()                                   # the operator pads its bounds
    assert (lo <= [-0.05, -0.05, 0.0]).all() and (hi >= [0.05, 0.05, 0.2]).all()
    assert (hi - lo < 0.4).all()
    points = np.array([[0, 0, 0.1], [0.04, 0, 0.1], [0.06, 0, 0.1], [0, 0, 0.25]])
    assert post.occupied(points).tolist() == [True, True, False, False]
    assert (post.sample(points) > 0.5).tolist() == [True, True, False, False]
    with pytest.raises(ValueError, match=r"\(n,3\)"):
        post.sample(np.zeros((2, 2)))


def test_set_config_changes_a_step():
    p = cylinder_project()
    changes = p.set_config("cylinder", {"radius": 0.1})
    assert changes == {"radius": ("Float(0.05)", "Float(0.1)")}
    lo, hi = p.run()["post"].bounds()
    assert hi[0] - lo[0] >= 0.2
    assert p.set_config(0, {"radius": 0.02})["radius"][1] == "Float(0.02)"
    with pytest.raises(ValueError, match="no config field"):
        p.set_config(0, {"radii": 1.0})
    with pytest.raises(ValueError, match="out of range"):
        p.set_config(5, {"radius": 1.0})
    with pytest.raises(ValueError, match="expected a number"):
        p.set_config(0, {"radius": "big"})


def test_add_views_imports_a_view_set(chair_views):
    p = cylinder_project()
    assert p.add_views(chair_views.select(ids=["DSC00742"], embed="none"), id="photos") == "photos"
    assert ("photos", "ViewSet") in p.asset_ids()
    assert p.add_views(chair_views) == "views"
