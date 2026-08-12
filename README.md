# Flight To Autonomy

## Introduction
This system will allow a Tello drone to fly through a course and accomplish some object recognition tasks without a human in the loop. Replacing the pilot with a ground computer that will talk to the drone over wifi. The only feedback the computer will have will be the feed from the onboard camera and the IMU.

## Features

📡 Software can reliably retrieve and display to the computer screen camera feed and telemetry from the IMU and flight controller.

🛠️ We are currently working on the gate perception module.


## Technologies
Project is been built using:

- [Python 3.13.7](https://www.python.org/downloads/release/python-3137/)
- [OpenCV 5.0.0.93](https://docs.opencv.org/5.0/)
- [DJITelloPy 2.5.0](https://djitellopy.readthedocs.io/en/latest/tello/)

## Installation
Make sure you have a python version compatible with **Python 3.13.7**. First step is cloning the repo: 

```bash
git clone https://github.com/emryscv/FlightToAutonomy
```

Create and activate a virtual enviroment inside the repository folder (**recommended**):

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the project requierements:

```bash
pip install -r requirements.txt
```

### Enjoy!!!

## Development Setup

Once the repo was cloned and the requierements installed, there are two remaining steps to have it fully working.

1. Power on the Tello drone and wait the LED at the front to cycle between red and green lights
2. Find a Wifi with an SSID of the format TELLO-XXXXXX

After the Wifi is conneceted you should be able to run the project by excecuting, on of this 3 commands:

```bash
python main.py
```

```bash
python3 main.py
```

```bash
"venv/bin/python" "main.py"
```

depending on how your environment is configured and whether a virtual environment is being used or not. 

## License
This project is licensed under the MIT License - see the LICENSE.txt file for details.

## Contributors
- [Emrys Cruz Viera](https://github.com/emryscv)
- [Rebeca Carroll](https://github.com/FS-rcarroll) (Tutoring)

## Project Status
Project still in development and is not ready to test yet.