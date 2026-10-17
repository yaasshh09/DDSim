## DDSim

[![CI](https://github.com/yaasshh09/DDSim/actions/workflows/ci.yml/badge.svg)](https://github.com/yaasshh09/DDSim/actions/workflows/ci.yml)

I built a working semiconductor device simulator completely from scratch. It has been benchmarked against DEVSIM, a public simulator. It solves the drift diffusion equations in order to get the coressponding IV and CV curves.

Built mainly using python and trace amounts of JS, CSS and HTML.

## What it does

The project is mainly based on poisson's equations along with a few electron and hole continuty equations from the van roosbroeck systems.

It can simulate a variety of devices like pn diodes, mos capacitors, nmos transistors and also has an option to create your own custom 2d devices

<p align="center">
  <img src="images\mosfet_rolloff.png" alt="mosfet_rolloff" width="800">
</p>


<p align="center">
  <img src="images\mos_cap_cv.png" alt="mos_cap_cv" width="300">
  <img src="images\mosfet_transfer_1um.png" alt="mos_transfer" width="300">
</p>


<p align="center">
  <img src="images\pn_diode_iv.png" alt="A PN diode" width="800">
</p>

## Web app

The browser front end runs the solver on the server and streams progress back over a web socket. This allows us to be able to watch the residual fall and the curve draw point by point in real-time.


<p align="center">
  <img src="images/site/welcome.png" alt="Welcome screen" width="400">
  <img src="images/site/lesson.png" alt="A guided lesson" width="400">
</p>

You can see where current flows and draw cutlines through the device, or sweep a mos capacitor's cv.

<p align="center">
  <img src="images/site/nmos.png" alt="NMOS current" width="400">
  <img src="images/site/mos-cap.png" alt="MOS capacitor" width="400">
</p>

Almost everything has an in-depth explainer. Each one starts+ with a simple explanation, then goes into the equations and points to the docs it came from (WIP).


<p align="center">
  <img src="images/site/explainer.png" alt="The band diagram explainer" width="800">
</p>

You can also draw your own device out of silicon, oxide, implants and contacts, then solve it as you wish. NOTE: THE FEATURE IS VERY CLUNKY, DO NOT RECOMMEND TO USE

<p align="center">
  <img src="images/site/editor.png" alt="The device editor" width="800">
</p>


## License

CC BY-NC 4.0. You're free to use, share and adapt it for non-commercial work as long as you credit me. See [LICENSE](LICENSE).
