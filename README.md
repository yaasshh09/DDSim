# DDSim

[![CI](https://github.com/yaasshh09/DDSim/actions/workflows/ci.yml/badge.svg)](https://github.com/yaasshh09/DDSim/actions/workflows/ci.yml)

I built a semiconductor device simulator from scratch. You give it a device's
shape and doping, and it solves the drift-diffusion equations to get the I-V
and C-V curves.

Try it in the browser: [ddsim.fly.dev](https://ddsim.fly.dev)

<p align="center">
  <img src="docs/images/site/diode.png" alt="A PN diode solved in the browser: residual, I-V curve and band diagram" width="800">
</p>

## What it does

This project solves Poisson’s equation along with electron and hole continuity equations (the Van Roosbroeck system) in 1D and 2D. 

It can simulate devices such as PN diodes, MOS capacitors, NMOS transistors, and custom 2D devices.

The simulator uses Scharfetter-Gummel discretization, full Newton solving, bias continuation, and exact small-signal C-V analysis. It also includes SRH and Auger recombination, Arora, Lombardi and Caughey-Thomas mobility models, and Fermi-Dirac statistics.

Results have been checked with textbook solutions and DEVSIM 2.11 across multiple benchmarks, with a total of 2,680 tests. 
 
The project is built using Python, NumPy, and SciPy sparse, with FastAPI and plain JavaScript powering the web application without a build step.

<p align="center">
  <img src="docs/images/mosfet_rolloff.png" alt="NMOS gate length sweep, with DEVSIM overlaid" width="800">
</p>
<p align="center">
  <img src="docs/images/mos_cap_cv.png" alt="MOS capacitor C-V against DEVSIM" width="400">
  <img src="docs/images/pn_diode_iv.png" alt="PN diode I-V, with the ideality factor crossover" width="400">
</p>

## Web app

The browser front end runs the real solver on the server and streams progress
back over a web socket, so you watch the residual fall and the curve draw itself
point by point.

You can pick one of four starting devices, or follow a guided lesson.

<p align="center">
  <img src="docs/images/site/welcome.png" alt="Welcome screen with four starting devices" width="400">
  <img src="docs/images/site/lesson.png" alt="A guided lesson on the pn junction" width="400">
</p>

You can see where the current flows and draw a cutline through the device, or
sweep a MOS capacitor's C-V.

<p align="center">
  <img src="docs/images/site/nmos.png" alt="NMOS current streamlines and bands along a cutline" width="400">
  <img src="docs/images/site/mos-cap.png" alt="MOS capacitor C-V in the browser" width="400">
</p>

Every knob has an explainer. Each one starts with a plain
explanation, then goes into the equations and points to the docs it came from.

<p align="center">
  <img src="docs/images/site/explainer.png" alt="The band diagram explainer" width="800">
</p>

You can also draw your own device out of silicon, oxide, implants and
contacts, then solve it as you wish.

<p align="center">
  <img src="docs/images/site/editor.png" alt="The device editor" width="800">
</p>

## Running it locally

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/pytest            # the full suite
.venv/Scripts/ddsim serve       # then open http://127.0.0.1:8000
```

Or from Python:

```python
from ddsim.device.pn_diode import pn_diode
from ddsim.extract.iv import iv_sweep

device = pn_diode(Na=1e16, Nd=1e16, length=12e-4, junction=6e-4, n_nodes=201)
curve = iv_sweep(device, "anode", [0.1, 0.2, 0.3, 0.4, 0.5], step=0.05)
for point in curve.points:
    print(f"{point.voltage:.2f} V  {point.current:.4e} A/cm^2")
```

The browser smoke test needs Chromium once: `.venv/Scripts/python -m playwright install chromium`.

## License

CC BY-NC 4.0. You're free to use, share and adapt it for non-commercial work as
long as you credit me. See [LICENSE](LICENSE).
