/* Browser-side inference using exported, frozen PyTorch weights. No training. */
(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.PitStopEngine = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD'];
  const ACTIONS = ['STAY OUT', 'PIT SOFT', 'PIT MEDIUM', 'PIT HARD'];
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
    return config;
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
    let action = 0;
    for (let i = 1; i < values.length; i++) if (values[i] > values[action]) action = i;
    return { action, q: values, activations };
  }

  class Race {
    constructor(model, settings = {}, seed = 2000000, random = null) {
      if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295)
        throw new Error('Seed must be a whole number between 0 and 4294967295.');
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
      this.wear += [.058, .039, .029][this.compound] * c.wear_multiplier
        * (safetyCar ? .35 : 1) * this.random.uniform(.93, 1.07);
      this.tyreAge++; this.fuel = Math.max(0, this.fuel - c.fuel_burn_kg);
      this.time += lapTime; this.lap++;
      if (this.scRemaining > 0) this.scRemaining--;
      if (this.scRemaining === 0 && this.lap < c.laps - 2
        && this.random.random() < c.safety_car_probability)
        this.scRemaining = this.random.integer(c.safety_car_duration_min, c.safety_car_duration_max + 1);
      const penalty = this.lap === c.laps && !this.changed ? c.terminal_rule_penalty : 0;
      this.time += penalty;
      const record = { lap: this.lap, before, after: this.state(), observation,
        action, q: prediction.q, activations: prediction.activations,
        safetyCar, pitLoss, lapTime, penalty,
        reward: -(tyreCost + pitLoss + penalty) / 10,
        rawReward: (c.reward_reference_seconds - lapTime - penalty) / c.reward_scale,
        uncontrollable };
      this.records.push(record);
      return record;
    }
    run() {
      while (this.lap < this.config.laps) this.step();
      return this.records;
    }
  }
  return { Race, Random, infer, makeConfig, COMPOUNDS, ACTIONS };
});
