"""Old standalone producers -> certified managed shared consumer.

This is a control-plane fixture, not a timing/QoR benchmark: OpenSTA uses the
deterministic protocol fixture, while partition, route and TDM run their real
implementations. EMUFLOW_TEST_LEGACY_SOURCE may select a frozen old checkout;
the default exercises the supported standalone contract in the current tree.
"""

import copy
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from emuflow.errors import ValidationError
from emuflow.experiment_dag import (
    EXPERIMENT_SPEC_V2_SCHEMA,
    plan_experiment,
    run_experiment_node,
    validate_experiment_checkpoint,
)
from emuflow.experiment_identity import build_implementation_closure
from emuflow.experiment_partition import validate_partition_checkpoint
from emuflow.experiment_stages import _managed_checkpoint
from emuflow.experiment_upstream import validate_frontend_checkpoint
from emuflow.io import read_json, write_json
from emuflow.opensta import DEFAULT_TIMING_MODEL
from tests.native_build import tlr_router


ROOT = Path(__file__).resolve().parents[1]
PLATFORM = ROOT / "platforms/virtual/xcvu3p_2fpga_p2p.json"


class ManagedCheckpointCompatibilityTest(unittest.TestCase):
    def _spec(self) -> dict:
        legacy = Path(os.environ.get("EMUFLOW_TEST_LEGACY_SOURCE", ROOT)).resolve()
        router = tlr_router()
        old_closure = build_implementation_closure(legacy, ["src/emuflow"])
        new_closure = build_implementation_closure(ROOT, ["src/emuflow"])
        fixture_inputs = {
            label: hashlib.sha256(path.read_bytes()).hexdigest()
            for label, path in {
                "board": PLATFORM,
                "yosys_json": ROOT / "examples/yosys/counter.json",
                "timing_model": DEFAULT_TIMING_MODEL,
                "opensta_protocol_fixture": ROOT / "tests/fixtures/fake_opensta_paths.py",
                "router": router,
                "python": Path(sys.executable),
            }.items()
        }
        prefix = [sys.executable, "-m", "emuflow", "experiment-stage"]
        board = ["--platform", str(PLATFORM)]
        sta = ["--opensta", str(ROOT / "tests/fixtures/fake_opensta_paths.py"),
               "--clock-period", "clk=10", "--timing-model", str(DEFAULT_TIMING_MODEL)]

        def deps(*names):
            return [arg for name in names for arg in (f"--{name}", f"{{dependency:{name}}}")]

        def node(name, dependencies, run, validate, artifacts, *, managed=False):
            source = ROOT if managed else legacy
            closure = new_closure if managed else old_closure
            return {
                "id": name, "stage": name, "dependencies": dependencies,
                "inputs": fixture_inputs,
                "configuration": {"qualification": "control-plane-protocol-fixture",
                                  "producer_contract": "managed" if managed else "standalone",
                                  "seed": 11, "workers": 1},
                "implementation": closure, "validator_implementation": closure,
                "command": [*prefix, *run, "--out", "{output_dir}"],
                "validator": [*prefix, *validate],
                "environment": {"PYTHONPATH": str(source / "src"),
                                "PATH": str(Path(sys.executable).parent) + os.pathsep + os.defpath},
                "artifacts": [{"path": path, "role": "consumer-checkpoint"} for path in artifacts],
                "storage_estimate": {"peak_bytes": 16 * 1024**2, "retained_bytes": 4 * 1024**2},
            }

        all_deps = ["frontend", "timing", "partition", "cut-timing", "route", "tdm"]
        return {
            "schema": EXPERIMENT_SPEC_V2_SCHEMA,
            "experiment_id": "standalone-to-managed-protocol-fixture",
            # Unit-test provenance only; implementation/input closures seal the
            # actual selected trees and tools (including an optional old tree).
            "source_commit": "0" * 40,
            "nodes": [
                node("frontend", [],
                     ["frontend-run", *board, "--yosys-json", str(ROOT / "examples/yosys/counter.json"),
                      "--top", "counter", "--clock", "clk"],
                     ["frontend-validate", "{artifact_root}", *board],
                     ["phase1", "sources", "synthesized.json", "experiment-frontend-report.json"]),
                node("timing", ["frontend"],
                     ["timing-run", *deps("frontend"), *sta],
                     ["timing-validate", "{artifact_root}", *deps("frontend")],
                     ["path-database.json", "partition-net-weights.json", "experiment-timing-report.json"]),
                node("partition", ["frontend", "timing"],
                     ["partition-run", *deps("frontend", "timing"), *board,
                      "--provider", "greedy", "--seed", "11", "--cut-mode", "sequential-only"],
                     ["partition-validate", "{artifact_root}", *deps("frontend", "timing"), *board,
                      "--provider", "greedy", "--seed", "11", "--cut-mode", "sequential-only"],
                     ["clusters.json", "assignment.json", "constraints.normalized.json",
                      "phase3_report.json", "experiment-partition-report.json"]),
                node("cut-timing", ["frontend", "timing", "partition"],
                     ["cut-timing-run", *deps("frontend", "timing", "partition"), *sta],
                     ["cut-timing-validate", "{artifact_root}", *deps("frontend", "timing", "partition"),
                      "--timing-model", str(DEFAULT_TIMING_MODEL)],
                     ["cut-timing-paths.json", "cut-segment-qualification.json",
                      "experiment-cut-timing-report.json"]),
                node("route", ["partition", "cut-timing"],
                     ["route-run", *deps("partition", "cut-timing"), *board,
                      "--provider", "timing-aware-load-balanced-v1", "--router", str(router)],
                     ["route-validate", "{artifact_root}", *deps("partition", "cut-timing"), *board],
                     ["routes.json", "phase4_report.json", "route_constraints.normalized.json",
                      "experiment-route-report.json"]),
                node("tdm", ["route"],
                     ["tdm-run", *deps("route"), *board,
                      "--provider", "deterministic-round-barrier-earliest-slot-v2"],
                     ["tdm-validate", "{artifact_root}", *deps("route"), *board],
                     ["schedule.json", "phase5_report.json", "experiment-tdm-report.json"]),
                node("shared", all_deps,
                     ["shared-materialize", *deps(*all_deps), *board, "--managed-dag-node"],
                     ["shared-validate", "--shared", "{artifact_root}", *board, "--managed-dag-node"],
                     ["frontend", "timing", "partition", "system-route", "tdm",
                      "experiment-shared-report.json"], managed=True),
            ],
        }

    def test_certified_standalone_ancestors_feed_managed_shared_without_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self._spec()
            spec_path, plan_path, cache = root / "spec.json", root / "plan.json", root / "cache"
            write_json(spec_path, spec)
            outputs, manifests, keys = {}, {}, {}
            # Independently validate and seal each old-format boundary before
            # the managed consumer becomes ready. No mocked certificates.
            for node in spec["nodes"]:
                plan = plan_experiment(spec_path, cache, plan_path)
                planned = next(item for item in plan["nodes"] if item["id"] == node["id"])
                self.assertEqual(planned["state"], "ready")
                logs = root / "attempts" / node["id"]
                if os.environ.get("EMUFLOW_TEST_LEGACY_SOURCE") and node["id"] != "shared":
                    # Exercise old publication/certificate code too, not just
                    # old producer bytes sealed by the current executor.
                    completed = subprocess.run(
                        [sys.executable, "-m", "emuflow", "experiment-cache", "run-node",
                         "--plan", str(plan_path), "--node", node["id"], "--run-dir", str(logs)],
                        env={**os.environ, **node["environment"]},
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    result = read_json(logs / "experiment-node-report.json")
                else:
                    result = run_experiment_node(plan_path, node["id"], logs)
                detail = "\n".join(path.read_text()[-3000:] for path in logs.glob("*.log"))
                self.assertEqual(result["status"], "pass", f"{node['id']}: {result}\n{detail}")
                outputs[node["id"]] = Path(result["checkpoint"]["output_dir"])
                manifests[node["id"]] = outputs[node["id"]].parent / "checkpoint.json"
                keys[node["id"]] = planned["key"]
                validate_experiment_checkpoint(manifests[node["id"]])
            report = read_json(outputs["shared"] / "experiment-shared-report.json")
            self.assertEqual(report["dependency_execution_keys"], {k: v for k, v in keys.items() if k != "shared"})
            self.assertEqual(report["validation_mode"], "managed-dependency-certificates")
            self.assertNotIn("validation_mode", read_json(outputs["frontend"] / "experiment-frontend-report.json"))
            self.assertEqual(read_json(outputs["partition"] / "assignment.json")["schema"],
                             "emuflow.partition-assignment/v1")
            self.assertGreater(len(read_json(outputs["partition"] / "assignment.json")["cut_nets"]), 0)
            for relative, record in report["artifacts"].items():
                self.assertGreater(record["bytes"], 0)
                self.assertTrue((outputs["shared"] / relative).is_file())
                ancestor_file = outputs[record["source_stage"]].joinpath(*Path(relative).parts[1:])
                self.assertEqual((outputs["shared"] / relative).read_bytes(), ancestor_file.read_bytes())

            # Adding/changing a consumer must retain all six original keys.
            changed = copy.deepcopy(spec)
            changed["nodes"][-1]["configuration"]["acceptance_revision"] = 2
            write_json(spec_path, changed)
            next_plan = plan_experiment(spec_path, cache, plan_path)
            self.assertEqual(next_plan["counts"], {"reuse": 6, "revalidate": 0, "ready": 1, "waiting": 0})
            self.assertEqual({n["id"]: n["key"] for n in next_plan["nodes"][:-1]},
                             {k: v for k, v in keys.items() if k != "shared"})
            original_manifests = {k: p.read_bytes() for k, p in manifests.items() if k != "shared"}

            # Old reports are not relabelled as new producer results.
            with self.assertRaisesRegex(ValidationError, "managed-validation contract"):
                validate_frontend_checkpoint(outputs["frontend"], PLATFORM, managed_dag_node=True)
            with self.assertRaisesRegex(ValidationError, "seed contract"):
                validate_partition_checkpoint(outputs["frontend"], outputs["timing"], PLATFORM,
                                              outputs["partition"], expected_seed=12)
            with self.assertRaisesRegex(ValidationError, "managed checkpoint contract"):
                _managed_checkpoint(outputs["partition"], expected_stage="frontend")

            # Missing/failed certificates reject even otherwise valid output.
            validation = next((outputs["partition"].parent / "validations").glob("*.json"))
            saved = validation.read_bytes()
            validation.rename(validation.with_suffix(".saved"))
            with self.assertRaisesRegex(ValidationError, "certificate"):
                _managed_checkpoint(outputs["partition"], expected_stage="partition")
            with self.assertRaisesRegex(ValidationError, "dependency partition is not validated"):
                run_experiment_node(plan_path, "shared", root / "attempts/shared-missing-certificate")
            self.assertFalse(Path(next_plan["nodes"][-1]["output_dir"]).exists())
            self.assertFalse(any((cache / "staging").iterdir()))
            self.assertFalse((root / "attempts/shared-missing-certificate").exists())
            validation.with_suffix(".saved").rename(validation)
            certificate = read_json(validation)
            certificate["status"] = "failed"
            write_json(validation, certificate)
            with self.assertRaisesRegex(ValidationError, "certificate"):
                _managed_checkpoint(outputs["partition"], expected_stage="partition")
            validation.write_bytes(saved)

            accepted = run_experiment_node(plan_path, "shared", root / "attempts/shared-retry")
            self.assertEqual(accepted["status"], "pass")
            self.assertEqual(original_manifests,
                             {k: p.read_bytes() for k, p in manifests.items() if k != "shared"})
            self.assertEqual(plan_experiment(spec_path, cache, plan_path)["counts"],
                             {"reuse": 7, "revalidate": 0, "ready": 0, "waiting": 0})

            # The explicit audit, not the metadata-only hot path, detects a
            # same-size mutation even if an owner restores read-only mode.
            assignment = outputs["partition"] / "assignment.json"
            original = assignment.read_bytes()
            self.assertIn(b"fpga0", original)
            assignment.chmod(0o644)
            assignment.write_bytes(original.replace(b"fpga0", b"fpgaX", 1))
            assignment.chmod(0o444)
            with self.assertRaisesRegex(ValidationError, "seal is broken"):
                validate_experiment_checkpoint(manifests["partition"])


if __name__ == "__main__":
    unittest.main()
