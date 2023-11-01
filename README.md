# AccidentDetection

A demo notebook to try the project can be foubd here <a href="https://colab.research.google.com/github/ShreyanshBardia/AccidentDetection/blob/master/crash_detection.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> 
</a> detecting car accidents involving two or more cars.

# Description

This project can be used to detect accidents on the go from the CCTV footage and then notify respective authorities so that the required help can be provided as soon as possible.

In brief, using DeepSort we assign unique ID to each car and track its movement. We calculate trajectory of all the cars and if two cars are in near proximity with their trajectories intersecting at a near point, we detect it as a crash. Various conditions have been kept in mind to make the project deployable and robust such as a still car, car overtaking each other etc. A detailed [report](https://github.com/ShreyanshBardia/AccidentDetection/blob/master/Accident%20Detection_git.pdf) can be found in the files section. We achieved an 81% precision and 81% recall. 

Below are demo examples

All cars are detected with a green mask, whereas cars involved in crash have been detected with an orange-red mask

## Example 1

![Example 1](https://github.com/ShreyanshBardia/AccidentDetection/blob/master/output.gif)

## Example 2

![Example 2](output_1.gif)
