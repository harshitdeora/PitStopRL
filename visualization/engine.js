/* Browser-side inference using exported, frozen PyTorch weights. No training. */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.PitStopEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD'];
  const ACTIONS = ['STAY OUT', 'PIT SOFT', 'PIT MEDIUM', 'PIT HARD'];
  const ENVIRONMENT_VERSION = 'wear_limit_v1';
  const WEAR_LIMIT = 1, MAX_WEAR_NOISE = 1.07, WEAR_GUARD = 1e-6;
  const WEAR_RATES = [.058, .039, .029];
  const DEFAULTS = {
    laps: 57, base_lap_time: 88, fuel_burn_kg: 1.8,
    fuel_penalty_per_kg: .030, pit_loss_seconds: 22,
    safety_car_pit_factor: .55, safety_car_probability: .035,
    safety_car_duration_min: 2, safety_car_duration_max: 4,
    safety_car_lap_delta: 25, lap_noise_std: .18,
    terminal_rule_penalty: 30, reward_reference_seconds: 100,
    reward_scale: 100, wear_multiplier: 1,
  };

  class Random {
    constructor(seed) { this.state = seed >>> 0; }
    random() {
      this.state = (this.state + 0x6D2B79F5) >>> 0;
      let value = this.state;
      value = Math.imul(value ^ value >>> 15, value | 1);
      value ^= value + Math.imul(value ^ value >>> 7, value | 61);
      return ((value ^ value >>> 14) >>> 0) / 4294967296;
    }
    normal(mean, std) {
      return mean + std * Math.sqrt(-2 * Math.log(Math.max(this.random(), 1e-12)))
        * Math.cos(2 * Math.PI * this.random());
    }
    uniform(low, high) { return low + (high - low) * this.random(); }
    integer(low, high) { return low + Math.floor(this.random() * (high - low)); }
  }

  function makeConfig(settings = {}) {
    const config = { ...DEFAULTS, ...settings };
    config.initial_fuel_kg = settings.initial_fuel_kg ?? config.laps * config.fuel_burn_kg;
    if (!Number.isInteger(config.laps) || config.laps < 1 || config.laps > 300)
      throw new Error('Choose a whole race length between 1 and 300 laps.');
    if (!Object.values(config).every(Number.isFinite)) throw new Error('Race settings must be finite.');
    if (config.pit_loss_seconds < 0 || config.wear_multiplier <= 0 || config.lap_noise_std < 0)
      throw new Error('Pit cost, tyre wear and noise settings are invalid.');
    if (config.safety_car_probability < 0 || config.safety_car_probability > 1)
      throw new Error('Safety-car probability must be between 0 and 1.');
    if (config.initial_fuel_kg <= 0 || config.initial_fuel_kg + 1e-8 < config.laps * config.fuel_burn_kg)
      throw new Error('Starting fuel must cover the race distance.');
    if (.029 * Math.max(config.wear_multiplier, Math.fround(config.wear_multiplier))
      * MAX_WEAR_NOISE > WEAR_LIMIT - WEAR_GUARD)
      throw new Error('Even fresh hard tyres must last one green-flag lap below 100% wear.');
    return config;
  }

  function actionMask(observation) {
    const obs = observation.map(Math.fround);
    const current = Math.round(obs[1] * 2);
    const factor = obs[11] * (obs[5] >= .5 ? .35 : 1);
    const fresh = WEAR_RATES.map(rate => factor * rate * MAX_WEAR_NOISE);
    return [obs[3] + fresh[current], ...fresh].map(wear => wear <= WEAR_LIMIT - WEAR_GUARD);
  }

  function infer(model, observation) {
    if (observation.length !== model.observation_names.length || !observation.every(Number.isFinite))
      throw new Error('Observation does not match the trained model.');
    let values = observation.map(Math.fround);
    const activations = [values];
    model.layers.forEach((layer, layerIndex) => {
      values = layer.weight.map((weights, neuron) => {
        let sum = layer.bias[neuron];
        for (let i = 0; i < weights.length; i++) sum += weights[i] * values[i];
        return Math.fround(layerIndex < model.layers.length - 1 ? Math.max(0, sum) : sum);
      });
      activations.push(values);
    });
    if (!values.every(Number.isFinite)) throw new Error('Non-finite network output.');
    const allowed = actionMask(observation);
    let action = -1;
    for (let i = 0; i < values.length; i++)
      if (allowed[i] && (action < 0 || values[i] > values[action])) action = i;
    if (action < 0) throw new Error('No action can keep tyre wear below 100%.');
    return { action, q: values, activations, actionMask: allowed };
  }

  // Evaluation-only policies matching pitstoprl/policies.py. They do not
  // supply labels, training rewards, or overrides for the DQN agent.
  function fixedStintPolicy(observation) {
    const allowed = actionMask(observation);
    const current = Math.round(observation[1] * 2);
    const total = Math.round(observation[9] * 100);
    const remaining = Math.round(observation[8] * 100);
    const trigger = Math.min(Math.max(Math.trunc(total * .4), 0), total - 1);
    const hardPhase = total - remaining >= trigger;
    const preferred = hardPhase ? 3 : 2;
    if ((hardPhase && current !== 2) || !allowed[0]) return allowed[preferred] ? preferred : 3;
    return 0;
  }

  function wearBasedPolicy(observation) {
    const age = observation[2] * 30;
    const remaining = Math.round(observation[8] * 100);
    const current = Math.round(observation[1] * 2);
    // The Python policy compares a float32 wear feature with this threshold.
    const allowed = actionMask(observation);
    let shouldPit = observation[3] > Math.fround(.8) || !allowed[0];
    if (!shouldPit && remaining < 14 && observation[7] < .5 && age >= 10) shouldPit = true;
    if (!shouldPit) return 0;
    let target = remaining <= 13 ? 0 : remaining <= 27 ? 1 : 2;
    if (target === current) target = current !== 2 ? 2 : 1;
    return allowed[target + 1] ? target + 1 : 3;
  }

  class Race {
    constructor(model, settings = {}, seed = 2000000, random = null) {
      if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295)
        throw new Error('Seed must be a whole number between 0 and 4294967295.');
      if (model.environment_version !== ENVIRONMENT_VERSION || model.tyre_wear_limit !== WEAR_LIMIT)
        throw new Error('Re-export the dashboard with a model trained for the 100% wear limit.');
      this.model = model;
      this.config = makeConfig(settings);
      this.random = random || new Random(seed);
      this.seed = seed;
      this.lap = 0; this.compound = 1; this.tyreAge = 0; this.wear = 0;
      this.fuel = this.config.initial_fuel_kg; this.scRemaining = 0;
      this.changed = false; this.pits = 0; this.time = 0;
      this.records = [];
    }
    observation() {
      const c = this.config;
      return [this.lap / c.laps, this.compound / 2, this.tyreAge / 30, this.wear,
        Math.max(this.fuel / c.initial_fuel_kg, 0), Number(this.scRemaining > 0),
        this.scRemaining / 4, Number(this.changed), (c.laps - this.lap) / 100,
        c.laps / 100, c.pit_loss_seconds / 40, c.wear_multiplier,
        c.safety_car_probability / .1, c.safety_car_pit_factor, c.base_lap_time / 100,
        this.wear ** 2, this.wear ** 3, c.initial_fuel_kg / 180, c.fuel_burn_kg / 3,
        c.fuel_penalty_per_kg / .05, c.safety_car_lap_delta / 40, c.lap_noise_std,
        c.terminal_rule_penalty / 100, c.reward_scale / 100,
        c.safety_car_duration_min / 4, c.safety_car_duration_max / 4,
        c.reward_reference_seconds / 100].map(Math.fround);
    }
    state() {
      return { lap: this.lap, compound: this.compound, tyreAge: this.tyreAge,
        wear: this.wear, fuel: this.fuel, scRemaining: this.scRemaining,
        changed: this.changed, pits: this.pits, time: this.time };
    }
    step(forcedAction = null) {
      const c = this.config;
      if (this.lap >= c.laps) throw new Error('Race is complete.');
      const before = this.state();
      const observation = this.observation();
      const prediction = infer(this.model, observation);
      const action = forcedAction ?? prediction.action;
      if (!Number.isInteger(action) || action < 0 || action > 3) throw new Error('Invalid action.');
      if (!prediction.actionMask[action])
        throw new Error('Action could exceed 100% tyre wear on this lap; choose a safe tyre change.');
      const safetyCar = this.scRemaining > 0;
      const pitLoss = action ? c.pit_loss_seconds * (safetyCar ? c.safety_car_pit_factor : 1) : 0;
      if (action) {
        const newCompound = action - 1;
        this.changed = this.changed || newCompound !== this.compound;
        this.compound = newCompound; this.tyreAge = 0; this.wear = 0; this.pits++;
      }
      const tyreCost = [-.5, 0, .48][this.compound] + 1.8 * this.wear + 11 * this.wear ** 3;
      const noise = this.random.normal(0, c.lap_noise_std);
      const uncontrollable = c.base_lap_time + c.fuel_penalty_per_kg * this.fuel
        + noise + (safetyCar ? c.safety_car_lap_delta : 0);
      const lapTime = uncontrollable + tyreCost + pitLoss;
      const wearNoise = this.random.uniform(.93, 1.07);
      this.wear += WEAR_RATES[this.compound] * c.wear_multiplier
        * (safetyCar ? .35 : 1) * wearNoise;
      if (this.wear > WEAR_LIMIT) throw new Error('Tyre wear constraint violated.');
      this.tyreAge++; this.fuel = Math.max(0, this.fuel - c.fuel_burn_kg);
      this.time += lapTime; this.lap++;
      if (this.scRemaining > 0) this.scRemaining--;
      if (this.scRemaining === 0 && this.lap < c.laps - 2
        && this.random.random() < c.safety_car_probability)
        this.scRemaining = this.random.integer(c.safety_car_duration_min, c.safety_car_duration_max + 1);
      const penalty = this.lap === c.laps && !this.changed ? c.terminal_rule_penalty : 0;
      this.time += penalty;
      const record = { lap: this.lap, before, after: this.state(), observation,
        action, q: prediction.q, activations: prediction.activations, actionMask: prediction.actionMask,
        safetyCar, pitLoss, lapTime, penalty, wearNoise,
        reward: -(tyreCost + pitLoss + penalty) / 10,
        rawReward: (c.reward_reference_seconds - lapTime - penalty) / c.reward_scale,
        uncontrollable };
      this.records.push(record);
      return record;
    }
    run(policy = null) {
      while (this.lap < this.config.laps) this.step(policy ? policy(this.observation()) : null);
      return this.records;
    }
  }

  function compareStrategies(model, settings = {}, seed = 2000000) {
    const races = {
      agent: new Race(model, settings, seed),
      fixed: new Race(model, settings, seed),
      wear: new Race(model, settings, seed),
    };
    races.agent.run();
    races.fixed.run(fixedStintPolicy);
    races.wear.run(wearBasedPolicy);
    // Each strategy has its own tyre state and decisions, but identical
    // exogenous randomness. Guard this invariant if the simulator changes.
    for (const baseline of [races.fixed, races.wear]) {
      if (baseline.records.some((record, i) => record.safetyCar !== races.agent.records[i].safetyCar
        || record.uncontrollable !== races.agent.records[i].uncontrollable
        || record.wearNoise !== races.agent.records[i].wearNoise))
        throw new Error('Race conditions did not match across strategies.');
    }
    return races;
  }

  return { Race, Random, infer, makeConfig, actionMask, fixedStintPolicy, wearBasedPolicy,
    compareStrategies, COMPOUNDS, ACTIONS, ENVIRONMENT_VERSION, WEAR_LIMIT };
});
