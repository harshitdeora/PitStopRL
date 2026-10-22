"""Differential checks against the real Python simulator and Torch network."""

import json
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from pitstoprl.dqn import DQNAgent
from pitstoprl.env import RaceConfig, RaceEnv
from pitstoprl.visualize import export_visualization

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


@pytest.fixture(scope="module")
def exported_model(tmp_path_factory):
    checkpoint = ROOT / "artifacts/dqn.pt"
    before = checkpoint.read_bytes()
    path = tmp_path_factory.mktemp("visualization") / "lab.html"
    export_visualization(checkpoint, path)
    html = path.read_text()
    match = re.search(r"window.PITSTOP_MODEL = (.*?);</script>", html, re.DOTALL)
    assert match is not None
    assert "<script src=" not in html
    assert '<link rel="stylesheet"' not in html
    assert checkpoint.read_bytes() == before
    return json.loads(match[1]), html


def node_run(script, payload):
    if NODE is None:
        pytest.skip("Node.js is required for cross-language engine checks.")
    result = subprocess.run(
        [NODE, "-e", script, str(ROOT / "visualization/engine.js")],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def test_export_is_self_contained_and_uses_real_checkpoint(exported_model):
    model, html = exported_model
    assert [len(layer["weight"]) for layer in model["layers"]] == [128, 128, 4]
    assert len(model["observation_names"]) == 27
    assert "Race state in." in html
    assert "weights stay fixed" in html


def test_javascript_hidden_activations_and_q_values_match_torch(exported_model):
    model, _ = exported_model
    agent = DQNAgent.load(ROOT / "artifacts/dqn.pt")
    observations = []
    for laps in (15, 57, 66, 90):
        env = RaceEnv(RaceConfig.for_laps(laps, wear_multiplier=1.2), seed=70)
        obs = env.reset(seed=70)
        for step in range(laps):
            if step % 5 == 0:
                observations.append(obs.tolist())
            obs, _, _, _, _ = env.step(agent.act(obs))
    script = """
      const fs = require('node:fs'), E = require(process.argv[1]);
      const data = JSON.parse(fs.readFileSync(0,'utf8'));
      process.stdout.write(JSON.stringify(data.observations.map(o => E.infer(data.model,o))));
    """
    actual = node_run(script, {"model": model, "observations": observations})
    for observation, prediction in zip(observations, actual):
        tensor = torch.tensor(observation, dtype=torch.float32)
        expected_layers = [tensor.numpy()]
        with torch.inference_mode():
            for index, layer in enumerate(agent.network.layers):
                tensor = layer(tensor)
                if index in (1, 3, 4):
                    expected_layers.append(tensor.numpy())
        for expected, calculated in zip(expected_layers, prediction["activations"]):
            np.testing.assert_allclose(calculated, expected, atol=2e-5, rtol=2e-5)
        assert prediction["action"] == int(np.argmax(expected_layers[-1]))


class RecordedRandom:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.draws = []

    def _get(self, name, *args):
        value = getattr(self.rng, name)(*args)
        self.draws.append({"kind": name, "value": float(value)})
        return value

    def normal(self, *args):
        return self._get("normal", *args)

    def uniform(self, *args):
        return self._get("uniform", *args)

    def random(self):
        return self._get("random")

    def integers(self, *args):
        return int(self._get("integers", *args))


@pytest.mark.parametrize("actions", [[0] * 12, [0, 1, 0, 2, 0, 3] * 2])
def test_browser_simulation_matches_python_for_identical_random_draws(exported_model, actions):
    model, _ = exported_model
    config = RaceConfig.for_laps(
        12, safety_car_probability=0.8, wear_multiplier=1.3, pit_loss_seconds=32
    )
    env = RaceEnv(config, seed=8)
    recorder = RecordedRandom(8)
    env.rng = recorder
    expected = []
    for action in actions:
        obs, reward, _, _, info = env.step(action)
        expected.append(
            {
                "obs": obs.tolist(),
                "time": env.total_time,
                "pits": env.pit_count,
                "changed": env.changed_compound,
                "lapTime": info["lap_time"],
                "rawReward": reward,
                "reward": -info["controllable_cost"] / 10,
            }
        )
    script = """
      const fs=require('node:fs'), E=require(process.argv[1]);
      const d=JSON.parse(fs.readFileSync(0,'utf8')); let offset=0;
      function draw(kind) {
        const item=d.draws[offset++];
        if(!item || item.kind!==kind) throw Error('Random draw order mismatch: '+kind);
        return item.value;
      }
      const rng={normal:()=>draw('normal'),uniform:()=>draw('uniform'),
        random:()=>draw('random'),integer:()=>draw('integers')};
      const race=new E.Race(d.model,d.config,8,rng);
      const records=d.actions.map(action=>{
        const record=race.step(action);
        return {obs:race.observation(),time:race.time,pits:race.pits,changed:race.changed,
          lapTime:record.lapTime,reward:record.reward,rawReward:record.rawReward};
      });
      if(offset!==d.draws.length) throw Error('Unused random draws');
      process.stdout.write(JSON.stringify(records));
    """
    actual = node_run(
        script,
        {"model": model, "config": asdict(config), "actions": actions, "draws": recorder.draws},
    )
    for python, browser in zip(expected, actual):
        np.testing.assert_allclose(browser["obs"], python["obs"], atol=1e-7)
        for key in ("time", "lapTime", "reward", "rawReward"):
            assert browser[key] == pytest.approx(python[key], abs=1e-8)
        assert browser["pits"] == python["pits"]
        assert browser["changed"] == python["changed"]


def test_browser_races_are_deterministic_and_use_own_config(exported_model):
    model, _ = exported_model
    script = """
      const fs=require('node:fs'),E=require(process.argv[1]);
      const model=JSON.parse(fs.readFileSync(0,'utf8'));
      const settings={laps:66,wear_multiplier:1.3,safety_car_probability:.07};
      const a=new E.Race(model,settings,99), b=new E.Race(model,settings,99);
      a.run(); b.run();
      let completedThrows=false;
      try {a.step();} catch {completedThrows=true;}
      process.stdout.write(JSON.stringify({equal:JSON.stringify(a.records)===JSON.stringify(b.records),
        laps:a.records.length, fuel:a.fuel,completedThrows,
        settings:E.makeConfig(settings)}));
    """
    result = node_run(script, model)
    assert result["equal"]
    assert result["laps"] == 66
    assert result["fuel"] == pytest.approx(0, abs=1e-8)
    assert result["completedThrows"]
    assert result["settings"]["initial_fuel_kg"] == pytest.approx(118.8)
