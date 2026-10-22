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
from pitstoprl.env import RaceConfig, RaceEnv, action_mask
from pitstoprl.policies import FixedStintPolicy, RuleBasedPolicy
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
        allowed = action_mask(np.array(observation))
        assert prediction["actionMask"] == allowed.tolist()
        assert prediction["action"] == int(np.argmax(np.where(allowed, expected_layers[-1], -np.inf)))


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


@pytest.mark.parametrize("laps", [1, 15, 66, 90])
@pytest.mark.parametrize("policy_name,policy", [
    ("fixedStintPolicy", FixedStintPolicy()),
    ("wearBasedPolicy", RuleBasedPolicy()),
])
def test_browser_baseline_rollouts_match_python(exported_model, laps, policy_name, policy):
    model, _ = exported_model
    config = RaceConfig.for_laps(
        laps, wear_multiplier=1.3, pit_loss_seconds=32, safety_car_probability=0.3
    )
    env = RaceEnv(config, seed=8)
    recorder = RecordedRandom(8)
    env.rng = recorder
    obs = env._observation()
    expected = []
    for _ in range(laps):
        action = policy(obs)
        obs, _, _, _, info = env.step(action)
        expected.append({
            "action": action, "time": env.total_time, "pits": env.pit_count,
            "compound": int(env.compound), "changed": env.changed_compound,
            "penalty": info["rule_penalty"],
        })
    script = """
      const fs=require('node:fs'), E=require(process.argv[1]);
      const d=JSON.parse(fs.readFileSync(0,'utf8')); let offset=0;
      function draw(kind) {
        const item=d.draws[offset++];
        if(!item || item.kind!==kind) throw Error('Random draw mismatch: '+kind);
        return item.value;
      }
      const rng={normal:()=>draw('normal'),uniform:()=>draw('uniform'),
        random:()=>draw('random'),integer:()=>draw('integers')};
      const race=new E.Race(d.model,d.config,8,rng);
      race.run(E[d.policy]);
      if(offset!==d.draws.length) throw Error('Unused random draws');
      process.stdout.write(JSON.stringify(race.records.map(r=>({action:r.action,
        time:r.after.time,pits:r.after.pits,compound:r.after.compound,
        changed:r.after.changed,penalty:r.penalty}))));
    """
    actual = node_run(script, {
        "model": model, "config": asdict(config), "policy": policy_name, "draws": recorder.draws,
    })
    for python, browser in zip(expected, actual, strict=True):
        assert browser["time"] == pytest.approx(python["time"], abs=1e-8)
        for key in ("action", "pits", "compound", "changed", "penalty"):
            assert browser[key] == python[key]


def test_custom_comparisons_use_paired_randomness_and_include_penalties(exported_model):
    model, _ = exported_model
    script = """
      const fs=require('node:fs'), E=require(process.argv[1]);
      const model=JSON.parse(fs.readFileSync(0,'utf8')), before=JSON.stringify(model);
      const settings=[{laps:1},{laps:15,wear_multiplier:.7},
        {laps:57,safety_car_probability:0},{laps:66,wear_multiplier:1.3,pit_loss_seconds:32},
        {laps:90,safety_car_probability:1,pit_loss_seconds:12},{laps:300}];
      const results=settings.map((config,index)=>{
        const seed=index%2 ? 4294967295 : 0;
        const runs=E.compareStrategies(model,config,seed);
        const repeat=E.compareStrategies(model,config,seed);
        return {laps:config.laps, paired:Object.values(runs).every(r=>
          r.records.every((lap,i)=>lap.safetyCar===runs.agent.records[i].safetyCar &&
            lap.uncontrollable===runs.agent.records[i].uncontrollable &&
            lap.wearNoise===runs.agent.records[i].wearNoise)),
          strategies:Object.fromEntries(Object.entries(runs).map(([name,race])=>[name,{
            laps:race.records.length,time:race.time,
            summed:race.records.reduce((s,r)=>s+r.lapTime+r.penalty,0),
            penalty:race.records.reduce((s,r)=>s+r.penalty,0),changed:race.changed,
            maxWear:Math.max(...race.records.map(r=>r.after.wear)),
            repeat:JSON.stringify(race.records)===JSON.stringify(repeat[name].records)
          }]))};
      });
      process.stdout.write(JSON.stringify({results,unchanged:before===JSON.stringify(model)}));
    """
    actual = node_run(script, model)
    assert actual["unchanged"]
    for scenario in actual["results"]:
        assert scenario["paired"]
        assert set(scenario["strategies"]) == {"agent", "fixed", "wear"}
        for strategy in scenario["strategies"].values():
            assert strategy["laps"] == scenario["laps"]
            assert strategy["time"] == pytest.approx(strategy["summed"], abs=1e-8)
            assert strategy["penalty"] == (0 if strategy["changed"] else 30)
            assert strategy["repeat"]
            assert strategy["maxWear"] <= 1


def test_browser_action_masks_match_python_near_wear_limit(exported_model):
    model, _ = exported_model
    observations = []
    for multiplier in (0.7, 1.3, 25.0):
        env = RaceEnv(RaceConfig.for_laps(66, wear_multiplier=multiplier))
        for compound in range(3):
            env.compound = compound
            for wear in (0.0, 0.90, 0.98, 0.9999999):
                env.wear = wear
                for sc in (0, 2):
                    env.safety_car_remaining = sc
                    observations.append(env._observation().tolist())
    script = """
      const fs=require('node:fs'),E=require(process.argv[1]);
      const d=JSON.parse(fs.readFileSync(0,'utf8'));
      process.stdout.write(JSON.stringify(d.observations.map(o=>({mask:E.actionMask(o),
        action:E.infer(d.model,o).action}))));
    """
    actual = node_run(script, {"model": model, "observations": observations})
    for obs, result in zip(observations, actual, strict=True):
        expected = action_mask(np.array(obs))
        assert result["mask"] == expected.tolist()
        assert expected[result["action"]]


def test_browser_rejects_unsafe_actions_and_survives_maximum_wear(exported_model):
    model, _ = exported_model
    script = """
      const fs=require('node:fs'),E=require(process.argv[1]);
      const model=JSON.parse(fs.readFileSync(0,'utf8')); let draws=0;
      const rng={normal:()=>{draws++;return 0},uniform:(a,b)=>{draws++;return b},
        random:()=>{draws++;return 1}};
      const race=new E.Race(model,{laps:300,wear_multiplier:1.3},8,rng);
      race.wear=.98;
      const before=JSON.stringify(race.state()); let rejected=false;
      try {race.step(0)} catch(error) {rejected=error.message.includes('100% tyre wear')}
      const unchanged=JSON.stringify(race.state())===before && draws===0;
      const maxima=[];
      for(const wear of [.7,1.3,3,25]) for(const policy of [null,E.fixedStintPolicy,E.wearBasedPolicy]) {
        const r=new E.Race(model,{laps:300,wear_multiplier:wear},8,rng);
        r.run(policy);
        maxima.push(Math.max(...r.records.map(lap=>lap.after.wear)));
      }
      let impossible=false;
      try {E.makeConfig({wear_multiplier:40})} catch {impossible=true}
      process.stdout.write(JSON.stringify({rejected,unchanged,maxima,impossible}));
    """
    result = node_run(script, model)
    assert result["rejected"] and result["unchanged"] and result["impossible"]
    assert max(result["maxima"]) <= 1
