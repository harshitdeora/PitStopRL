"""Export a standalone, offline race and neural-network visualization.

No web framework or online service is used. The trained weights are embedded in
the HTML and inference runs in the browser. The core checkpoint stays unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .dqn import DQNAgent
from .env import RaceEnv


def export_visualization(checkpoint: Path, output: Path) -> Path:
    source = Path(__file__).resolve().parents[1] / "visualization"
    if not source.is_dir():
        raise FileNotFoundError("Run the visualization exporter from the repository checkout.")
    agent = DQNAgent.load(checkpoint)
    layers = []
    for index in (0, 2, 4):
        layer = agent.network.layers[index]
        layers.append(
            {"weight": layer.weight.detach().tolist(), "bias": layer.bias.detach().tolist()}
        )
    model = {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "observation_names": list(RaceEnv.OBSERVATION_NAMES),
        "metadata": agent.metadata,
        "layers": layers,
    }
    model_js = "window.PITSTOP_MODEL = " + json.dumps(model, separators=(",", ":")) + ";"
    html = (source / "index.html").read_text()
    html = html.replace(
        '<link rel="stylesheet" href="style.css">',
        "<style>" + (source / "style.css").read_text() + "</style>",
    )
    for filename in ("model.js", "engine.js", "app.js"):
        content = model_js if filename == "model.js" else (source / filename).read_text()
        # Protect script boundaries if future model metadata includes HTML text.
        content = content.replace("</script", "<\\/script")
        html = html.replace(
            f'<script src="{filename}"></script>', "<script>" + content + "</script>"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an offline trained-agent visualization.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/dqn.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/PitStopRL_Visual_Lab.html"))
    args = parser.parse_args()
    path = export_visualization(args.checkpoint, args.output)
    print(f"Open {path.resolve()} in a browser. No server or network connection is required.")


if __name__ == "__main__":
    main()
