# PhoenixPilot

A friendly fork of OpenPilot by some of the former OP developers.

## Goals
The goal of PhoenixPilot is to focus on writing high quality open source code for autopilots that can easily provide the basis for research projects or further development by anyone.  The project focuses on high quality code, robust testing, and ease of use.

In terms of platforms, PhoenixPilot will continue support for CopterControl as well as a new board in development: Freedom which is an advance on OP Revolution designed for autonomous navigation and highly calibrated flight.  In addition PhoenixPilot has a policy of openness and will welcome any new boards into our targets that are released under an open license.

## Getting involved
Click that big fork button on github and start coding!  We use pull requests as reviews so expect a lot of constructive feedback!

In addition check out https://groups.google.com/forum/#!forum/phoenixpilot for more discussion

## Code structure

flight - contains the firmware components of the code
flight/PiOS - contains the drivers
flight/Modules - the flight control logic, broken into modules that communicate via UAVObjects
ground - contains the GCS code

