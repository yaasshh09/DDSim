# DDSim

[![CI](https://github.com/yaasshh09/DDSim/actions/workflows/ci.yml/badge.svg)](https://github.com/yaasshh09/DDSim/actions/workflows/ci.yml)

I built a working semi-conductor device simulator completely from scratch. It has been benchmarked against the famous and public DEVSIM. It solves the drift diffusion equations in order to get the corressponding IV and CV curves

<p align="center">
  <img src="docs/images/site/diode.png" alt="A PN diode" width="800">
</p>

## What it does

This project is based on poisson’s equation along with electron and hole continuity equations (from the van roosbroeck system) in 1D and 2D. 

It can simulate devices such as PN diodes, MOS capacitors, NMOS transistors, and custom 2D devices.

<p align="center">
  <img src="docs/images/mosfet_rolloff.png" alt="NMOS gate length" width="800">
</p>
<p align="center">
  <img src="docs/images/mos_cap_cv.png" alt="MOS capacitor" width="400">
</p>

<p align="center">
  <img src="docs/images/pn_diode_iv.png" alt="PN diode I-V" width="400">
</p>
## Web app

The browser front end runs the real solver on the server and streams progress
back over a web socket, so you watch the residual fall and the curve draw point by point.


<p align="center">
  <img src="docs/images/site/welcome.png" alt="Welcome screen" width="400">
  <img src="docs/images/site/lesson.png" alt="A guided lesson" width="400">
</p>

You can see where the current flows and draw a cutline through the device, or
sweep a MOS capacitor's C-V.

<p align="center">
  <img src="docs/images/site/nmos.png" alt="NMOS current" width="400">
  <img src="docs/images/site/mos-cap.png" alt="MOS capacitor" width="400">
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

## License

CC BY-NC 4.0. You're free to use, share and adapt it for non-commercial work as
long as you credit me. See [LICENSE](LICENSE).
