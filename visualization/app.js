(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const model = window.PITSTOP_MODEL;
  const E = window.PitStopEngine;
  if (!model || !E) {
    document.querySelector('main').innerHTML = '<p class="load-error">The trained model is missing. Run <code>python -m pitstoprl.visualize</code> to export the preview.</p>';
    return;
  }
  const colors = ['#f17979', '#e9cf75', '#e8ede5'];
  const names = ['Soft', 'Medium', 'Hard'];
  const actionNames = ['Stay out', 'Pit for soft', 'Pit for medium', 'Pit for hard'];
  const presets = {
    reference: { laps: 57, wear: 1, pit: 22, sc: .035 },
    heldout: { laps: 66, wear: 1, pit: 22, sc: .035 },
    endurance: { laps: 90, wear: 1.3, pit: 22, sc: .035 },
    safety: { laps: 57, wear: 1, pit: 12, sc: .07 },
    expensive: { laps: 57, wear: 1, pit: 32, sc: .035 },
  };
  const track = $('track'), pitPath = $('pit-path');
  const length = track.getTotalLength(), pitLength = pitPath.getTotalLength();
  const pitExit = pitPath.getPointAtLength(pitLength);
  let rejoinFraction = 0, exitDistance = Infinity;
  for (let sample = 0; sample <= 1000; sample++) {
    const point = track.getPointAtLength(sample / 1000 * length);
    const distance = Math.hypot(point.x - pitExit.x, point.y - pitExit.y);
    if (distance < exitDistance) { exitDistance = distance; rejoinFraction = sample / 1000; }
  }
  const speedOptions = [.5, 1, 2, 4, 8];
  let speed = 1, records = [], race, position = 0, playing = false;
  let lastFrame = null, shownLap = -1, eventKey = '';
  const networkViews = [];
  const ns = 'http://www.w3.org/2000/svg';

  function svgElement(tag, attrs = {}) {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([name, value]) => node.setAttribute(name, value));
    return node;
  }
  function networkPosition(layer, index) {
    if (layer === 0) return [18 + index % 3 * 8, 20 + Math.floor(index / 3) * 24];
    if (layer === 1) return [119 + index % 8 * 7.5, 17 + Math.floor(index / 8) * 13.5];
    if (layer === 2) return [250 + index % 8 * 7.5, 17 + Math.floor(index / 8) * 13.5];
    return [395, 38 + index * 55];
  }
  function createNetwork(svg) {
    svg.replaceChildren();
    const edgeGroup = svgElement('g'), nodeGroup = svgElement('g');
    const edges = [], nodes = [];
    model.layers.forEach((layer, index) => {
      const maxWeight = Math.max(...layer.weight.flat().map(Math.abs));
      layer.weight.forEach((weights, target) => {
        const chosen = weights.map((weight, source) => ({ weight, source }))
          .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight)).slice(0, index === 2 ? 6 : 2);
        chosen.forEach(({ weight, source }) => {
          const from = networkPosition(index, source), to = networkPosition(index + 1, target);
          const bend = (to[0] - from[0]) * .46;
          const path = svgElement('path', {
            d: `M ${from[0]} ${from[1]} C ${from[0] + bend} ${from[1]}, ${to[0] - bend} ${to[1]}, ${to[0]} ${to[1]}`,
            fill: 'none', stroke: weight >= 0 ? '#b59ae8' : '#74bbb7',
            'stroke-width': .55, 'stroke-opacity': .08, class: 'network-edge',
          });
          edgeGroup.appendChild(path);
          edges.push({ path, sourceLayer: index, source, strength: Math.abs(weight) / maxWeight });
        });
      });
    });
    [27, 128, 128, 4].forEach((count, layer) => {
      for (let index = 0; index < count; index++) {
        const [cx, cy] = networkPosition(layer, index);
        const circle = svgElement('circle', { cx, cy, r: layer === 3 ? 5 : layer === 0 ? 2.5 : 2.05,
          fill: '#4b4a59', class: 'network-node' });
        const title = svgElement('title');
        circle.appendChild(title);
        nodeGroup.appendChild(circle);
        nodes.push({ circle, title, layer, index });
      }
    });
    ['OUT', 'S', 'M', 'H'].forEach((label, index) => {
      const text = svgElement('text', { x: 407, y: 41 + index * 55, fill: '#8d9b85',
        'font-family': 'monospace', 'font-size': 7 });
      text.textContent = label;
      nodeGroup.appendChild(text);
    });
    svg.append(edgeGroup, nodeGroup);
    const view = { svg, edges, nodes };
    networkViews.push(view);
    return view;
  }
  function updateNetwork(record) {
    const a = record.activations;
    const maxima = a.map(layer => Math.max(.001, ...layer.map(Math.abs)));
    networkViews.forEach(view => {
      view.nodes.forEach(({ circle, title, layer, index }) => {
        const value = a[layer][index];
        const intensity = Math.min(1, Math.abs(value) / maxima[layer]);
        if (layer === 3) {
          const selected = index === record.action;
          circle.setAttribute('fill', selected ? '#d5ff65' : '#707c68');
          circle.classList.toggle('selected', selected);
        } else {
          const alpha = .15 + .85 * Math.sqrt(intensity);
          circle.setAttribute('fill', layer === 0 ? `rgba(143,201,181,${alpha})` : `rgba(194,163,255,${alpha})`);
        }
        const label = layer === 0 ? model.observation_names[index] : layer === 3 ? E.ACTIONS[index] : `Hidden ${layer}, neuron ${index + 1}`;
        title.textContent = `${label}: ${value.toFixed(6)}`;
      });
      view.edges.forEach(({ path, sourceLayer, source, strength }) => {
        const activation = Math.abs(a[sourceLayer][source]) / maxima[sourceLayer];
        path.setAttribute('stroke-opacity', (.015 + .34 * Math.sqrt(activation) * strength).toFixed(3));
      });
    });
  }
  createNetwork($('network'));

  E.ACTIONS.forEach((name, index) => {
    const row = document.createElement('div');
    row.className = 'q-row'; row.id = `q-${index}`;
    row.innerHTML = `<span class="q-name"><i class="q-dot"></i>${name}</span><div class="q-track"><div class="q-fill"></div></div><span class="q-score"></span>`;
    $('q-values').appendChild(row);
  });

  function clock(seconds) {
    const ticks = Math.round(seconds * 100);
    const minutes = Math.floor(ticks / 6000), secs = (ticks % 6000) / 100;
    return `${String(minutes).padStart(2, '0')}:${secs.toFixed(2).padStart(5, '0')}`;
  }
  function setPlaying(value) {
    playing = value;
    lastFrame = null;
    $('play-icon').textContent = value ? 'Ⅱ' : position >= records.length ? '↺' : '▶';
    $('play-label').textContent = value ? 'Pause race' : position >= records.length ? 'Replay' : position === 0 ? 'Lights out' : 'Continue';
    $('play').setAttribute('aria-label', value ? 'Pause race' : position >= records.length ? 'Replay race' : 'Play race');
    $('network-play').textContent = value ? 'Ⅱ Pause network' : '▶ Play network';
    $('network-play').setAttribute('aria-label', value ? 'Pause network' : 'Play network');
  }
  function showNotice(message) {
    $('notice').textContent = message;
    $('notice').hidden = !message;
  }
  function settingsFromControls() {
    return { laps: Number($('laps').value), wear_multiplier: Number($('wear').value),
      pit_loss_seconds: Number($('pit-cost').value), safety_car_probability: Number($('sc-chance').value) };
  }
  function loadRace() {
    try {
      const comparison = E.compareStrategies(model, settingsFromControls(), Number($('seed').value));
      race = comparison.agent; records = race.records;
      updateComparison(comparison);
      position = 0; shownLap = -1; eventKey = '';
      setPlaying(false);
      $('new-race').innerHTML = 'Load race <span>↗</span>';
      $('new-race').classList.remove('dirty');
      $('lap-total').textContent = race.config.laps;
      $('scrub').max = race.config.laps; $('scrub').value = 0;
      $('scrub-end').textContent = `${race.config.laps} LAPS`;
      $('stint-axis-middle').textContent = `LAP ${Math.ceil(race.config.laps / 2)}`;
      $('stint-axis-end').textContent = `LAP ${race.config.laps}`;
      $('stint-chart').replaceChildren();
      records.forEach((record, index) => {
        const cell = document.createElement('button');
        cell.className = 'stint-cell';
        cell.setAttribute('aria-label', `Inspect lap ${index + 1}`);
        cell.title = `Inspect lap ${index + 1}`;
        cell.addEventListener('click', () => seek(index));
        $('stint-chart').appendChild(cell);
      });
      const outside = race.config.laps < 20 || race.config.laps > 90;
      showNotice(outside ? 'Outside the training range: this race is an extrapolation test. The agent may make weaker decisions.' : '');
      render();
    } catch (error) { showNotice(error.message); }
  }
  function updateComparison(comparison) {
    Object.entries(comparison).forEach(([name, result]) => {
      $(`${name}-total-time`).textContent = clock(result.time);
      $(`${name}-total-seconds`).textContent = `${result.time.toFixed(2)} seconds`;
      $(`${name}-result-detail`).textContent = `${result.pits} pit ${result.pits === 1 ? 'stop' : 'stops'} · ${result.changed ? 'Compound rule met' : `${result.config.terminal_rule_penalty}s compound penalty included`}`;
    });
    $('comparison-race').textContent = `${race.config.laps} laps · seed ${race.seed} · wear ${race.config.wear_multiplier.toFixed(1)}× · pit cost ${race.config.pit_loss_seconds}s · safety-car chance ${(race.config.safety_car_probability * 100).toFixed(1)}%`;
  }
  function seek(value) {
    position = Math.max(0, Math.min(Number(value), records.length));
    setPlaying(false);
    shownLap = -1;
    render();
  }
  function markDirty() {
    $('scenario').value = 'custom';
    $('new-race').innerHTML = 'Apply changes <span>↗</span>';
    showNotice('Settings changed. Load the race to apply them; current playback keeps its original conditions.');
  }
  function moveCar(element, path, fraction, visible = true) {
    element.setAttribute('opacity', visible ? '1' : '0');
    if (!visible) return;
    const pathLength = path === track ? length : pitLength;
    const distance = Math.max(0, Math.min(fraction, .99999)) * pathLength;
    const point = path.getPointAtLength(distance);
    const next = path.getPointAtLength(Math.min(pathLength, distance + 1));
    const previous = path.getPointAtLength(Math.max(0, distance - 1));
    const angle = Math.atan2(next.y - previous.y, next.x - previous.x) * 180 / Math.PI;
    element.setAttribute('transform', `translate(${point.x} ${point.y}) rotate(${angle})`);
  }
  function trackAnimation(record, fraction, complete) {
    let route = track, progress = complete ? 0 : fraction;
    // The pit lane branches off at the start of the lap. Allocate a visible
    // dwell interval, then rejoin at its measured position on the SVG circuit.
    if (record.action && !complete) {
      if (fraction < .38) {
        route = pitPath;
        if (fraction < .12) progress = fraction / .12 * .55;
        else if (fraction < .25) progress = .55;
        else progress = .55 + (fraction - .25) / .13 * .45;
      } else progress = rejoinFraction + (fraction - .38) / .62 * (1 - rejoinFraction);
    }
    moveCar($('race-car'), route, progress);
    moveCar($('car-trail'), route, progress);
    moveCar($('sc-car'), track, (progress + .055) % 1, record.safetyCar && !complete);
    $('pit-pulse').setAttribute('opacity', record.action && fraction >= .1 && fraction <= .29 && !complete ? '.85' : '0');
    $('circuit').classList.toggle('under-safety-car', record.safetyCar && !complete);
    const event = $('track-event');
    event.hidden = complete || (!record.action && !record.safetyCar);
    event.classList.toggle('safety', !record.action && record.safetyCar);
    if (record.action) {
      $('track-event-icon').textContent = '↳';
      $('track-event-title').textContent = fraction < .38 ? 'BOX, BOX' : 'BACK ON TRACK';
      $('track-event-detail').textContent = `Fresh ${names[record.after.compound].toLowerCase()} tyres · ${record.pitLoss.toFixed(1)}s pit loss`;
    } else if (record.safetyCar) {
      $('track-event-icon').textContent = '⚑';
      $('track-event-title').textContent = 'SAFETY CAR DEPLOYED';
      $('track-event-detail').textContent = 'Reduced pace. Cheaper pit stops.';
    }
  }
  function updateDecision(record) {
    updateNetwork(record);
    const minimum = Math.min(...record.q), maximum = Math.max(...record.q);
    const range = Math.max(.001, maximum - minimum);
    record.q.forEach((value, index) => {
      const row = $(`q-${index}`);
      row.classList.toggle('selected', index === record.action);
      row.querySelector('.q-fill').style.width = `${12 + 88 * (value - minimum) / range}%`;
      row.querySelector('.q-score').textContent = value.toFixed(2);
      row.title = `${E.ACTIONS[index]}: Q = ${value.toFixed(6)}. Bar length shows relative value, not probability.`;
    });
    $('decision-label').textContent = actionNames[record.action];
    $('decision-detail').textContent = record.action
      ? record.safetyCar ? 'A tyre change at the safety-car pit rate.' : `A fresh set of ${names[record.after.compound].toLowerCase()} tyres.`
      : `Continue on the current ${names[record.before.compound].toLowerCase()} tyres.`;
    $('track-status').classList.toggle('safety', record.safetyCar);
    $('track-status').querySelector('span').textContent = record.safetyCar ? 'SAFETY CAR' : 'GREEN FLAG';
    const list = $('input-list');
    list.replaceChildren();
    model.observation_names.forEach((name, index) => {
      const row = document.createElement('div'); row.className = 'input-row';
      const label = document.createElement('span'); label.textContent = name.replaceAll('_', ' ');
      const value = document.createElement('code'); value.textContent = record.observation[index].toFixed(4);
      row.append(label, value); list.appendChild(row);
    });
  }
  function updateTelemetry(state, remaining) {
    const compound = state.compound;
    document.querySelectorAll('.car-tyre').forEach(tyre => tyre.style.stroke = colors[compound]);
    $('tyre-icon').textContent = ['S', 'M', 'H'][compound];
    $('tyre-icon').style.borderColor = colors[compound]; $('tyre-icon').style.color = colors[compound];
    $('compound-name').textContent = names[compound];
    $('tyre-age').textContent = `${state.tyreAge} ${state.tyreAge === 1 ? 'lap' : 'laps'} old`;
    document.querySelectorAll('.compound-keys>span').forEach((node, index) => node.classList.toggle('active', index === compound));
    $('wear-value').innerHTML = `${Math.round(state.wear * 100)}<span>%</span>`;
    $('wear-description').textContent = state.wear < .15 ? 'Fresh rubber' : state.wear < .45 ? 'In the window' : state.wear < .8 ? 'Grip fading' : 'High degradation';
    $('wear-fill').style.width = `${Math.min(state.wear * 100, 100)}%`;
    $('wear-fill').style.background = state.wear < .45 ? 'var(--lime)' : state.wear < .8 ? 'var(--amber)' : 'var(--soft)';
    $('fuel-value').innerHTML = `${state.fuel.toFixed(1)}<span>kg</span>`;
    $('fuel-laps').textContent = remaining ? `${remaining} laps to go` : 'Race complete';
    $('fuel-fill').style.width = `${state.fuel / race.config.initial_fuel_kg * 100}%`;
    $('pit-count').innerHTML = `${state.pits}<span>${state.pits === 1 ? 'stop' : 'stops'}</span>`;
    $('rule-status').textContent = state.changed ? 'Compound rule met' : 'Second compound needed';
    $('rule-status').classList.toggle('complete', state.changed);
    $('rule-check').textContent = state.changed ? '✓' : '○';
    $('rule-check').classList.toggle('complete', state.changed);
  }
  function updateTimeline(index, complete) {
    const cells = $('stint-chart').children;
    records.forEach((record, i) => {
      const visible = i < index || complete;
      cells[i].style.background = visible ? colors[record.after.compound] : '';
      cells[i].classList.toggle('pit', visible && Boolean(record.action));
      cells[i].classList.toggle('sc', visible && record.safetyCar);
      cells[i].classList.toggle('current', i === index && !complete);
      cells[i].title = visible ? `Lap ${i + 1}: ${names[record.after.compound]}${record.action ? ' · pit stop' : ''}${record.safetyCar ? ' · safety car' : ''}` : `Inspect decision before lap ${i + 1}`;
    });
    const events = records.slice(0, complete ? records.length : index).flatMap(record => {
      const items = [];
      if (record.action) items.push({ type: 'pit', lap: record.lap, text: `Pit → ${names[record.after.compound]}` });
      if (record.safetyCar && (record.lap === 1 || !records[record.lap - 2].safetyCar))
        items.push({ type: 'safety', lap: record.lap, text: 'Safety car' });
      return items;
    });
    const key = events.map(event => `${event.type}${event.lap}`).join();
    if (key !== eventKey || index === 0) {
      eventKey = key; $('race-events').replaceChildren();
      if (!events.length) $('race-events').innerHTML = '<span class="event-empty">Your race story starts at lights out.</span>';
      events.slice(-8).forEach(event => {
        const chip = document.createElement('span'); chip.className = `event-chip ${event.type}`;
        chip.innerHTML = `<b>L${String(event.lap).padStart(2, '0')}</b> ${event.text}`;
        $('race-events').appendChild(chip);
      });
    }
  }
  function render() {
    if (!records.length) return;
    const complete = position >= records.length;
    const index = Math.min(Math.floor(position), records.length - 1);
    const record = records[index];
    const fraction = complete ? 1 : position - index;
    const postPit = record.action && fraction >= .25;
    const state = complete ? record.after : postPit ? {
      ...record.before, compound: record.after.compound, tyreAge: 0, wear: 0,
      pits: record.after.pits, changed: record.after.changed,
    } : record.before;
    trackAnimation(record, fraction, complete);
    const baseTime = record.before.time;
    $('race-time').textContent = clock(complete ? record.after.time : baseTime + fraction * record.lapTime);
    $('lap-number').textContent = String(complete ? records.length : index + 1).padStart(2, '0');
    $('lap-caption').textContent = complete ? 'The race is complete' : position === 0 ? 'Ready on the grid' : `Decision before lap ${index + 1}`;
    $('run-state').textContent = complete ? 'Chequered flag' : playing ? record.safetyCar ? 'Safety-car conditions' : 'Racing' : position === 0 ? 'Awaiting lights out' : 'Playback paused';
    $('scrub').value = position;
    $('scrub-current').textContent = `LAP ${String(complete ? records.length : index + 1).padStart(2, '0')}`;
    $('finish-overlay').hidden = !complete;
    $('step').disabled = complete;
    $('network-step').disabled = complete;
    $('network-lap').textContent = `LAP ${String(complete ? records.length : index + 1).padStart(2, '0')}`;
    if (complete) {
      $('finish-summary').textContent = `${clock(race.time)} · ${race.pits} pit ${race.pits === 1 ? 'stop' : 'stops'} · ${race.changed ? 'Compound rule met' : 'Compound penalty applied'}`;
      $('track-status').classList.remove('safety');
      $('track-status').querySelector('span').textContent = 'FINISHED';
    }
    updateTelemetry(state, complete ? 0 : records.length - index);
    if (shownLap !== index || complete) {
      shownLap = index; updateDecision(record); updateTimeline(index, complete);
      if (complete) $('track-status').querySelector('span').textContent = 'FINISHED';
    }
  }
  function animate(timestamp) {
    if (playing && records.length) {
      const elapsed = lastFrame === null ? 0 : Math.min(timestamp - lastFrame, 100);
      lastFrame = timestamp;
      const index = Math.min(Math.floor(position), records.length - 1);
      const record = records[index];
      const duration = 2400 * record.lapTime / race.config.base_lap_time;
      position = Math.min(records.length, position + elapsed * speed / duration);
      if (position >= records.length) setPlaying(false);
      render();
    } else lastFrame = null;
    requestAnimationFrame(animate);
  }

  $('play').addEventListener('click', () => {
    if (position >= records.length) { position = 0; shownLap = -1; eventKey = ''; }
    setPlaying(!playing); render();
  });
  $('step').addEventListener('click', () => seek(Math.min(records.length, Math.floor(position) + 1)));
  $('network-play').addEventListener('click', () => $('play').click());
  $('network-step').addEventListener('click', () => $('step').click());
  $('restart').addEventListener('click', () => seek(0));
  $('replay-race').addEventListener('click', () => { seek(0); setPlaying(true); });
  $('speed').addEventListener('click', () => {
    speed = speedOptions[(speedOptions.indexOf(speed) + 1) % speedOptions.length];
    $('speed').textContent = `${speed}×`;
  });
  $('scrub').addEventListener('input', event => seek(event.target.value));
  $('new-race').addEventListener('click', loadRace);
  $('scenario').addEventListener('change', () => {
    const preset = presets[$('scenario').value];
    if (preset) {
      $('laps').value = preset.laps; $('wear').value = preset.wear;
      $('pit-cost').value = preset.pit; $('sc-chance').value = preset.sc;
      loadRace();
    }
  });
  ['laps', 'wear', 'pit-cost', 'sc-chance', 'seed'].forEach(id => $(id).addEventListener('change', markDirty));
  $('about-button').addEventListener('click', () => { setPlaying(false); render(); $('about-dialog').showModal(); });
  $('inputs-button').addEventListener('click', () => { setPlaying(false); render(); $('inputs-dialog').showModal(); });
  $('expand-network').addEventListener('click', () => {
    setPlaying(false);
    if (!$('large-network-container').children.length) {
      const svg = svgElement('svg', { viewBox: '0 0 430 236', role: 'img', 'aria-label': 'Expanded actual neural network activations' });
      $('large-network-container').appendChild(svg); createNetwork(svg);
    }
    updateNetwork(records[Math.min(Math.floor(position), records.length - 1)]);
    $('network-dialog').showModal();
  });
  document.querySelectorAll('dialog').forEach(dialog => {
    dialog.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  });
  document.addEventListener('keydown', event => {
    if (event.code === 'Space' && !['INPUT', 'SELECT', 'BUTTON'].includes(event.target.tagName)
      && !document.querySelector('dialog[open]')) { event.preventDefault(); $('play').click(); }
  });
  $('provenance').textContent = `Checkpoint SHA-256: ${model.checkpoint_sha256}`;
  loadRace();
  requestAnimationFrame(animate);
})();
