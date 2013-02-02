# Tau Labs

A friendly fork of OpenPilot by some of the former OP developers.

## Goals
The goal of Tau Labs is to focus on writing high quality open source code for autopilots that can easily provide the basis for research projects or further development by anyone.  The project focuses on high quality code, robust testing, and ease of use. Our target audience is professionals, researchers, and students, but we want to make those more advanced techniques easy and accessible for anyone.

By “research”, we mean not only universities or institutions focused on research on UAVs, but any group who might have use for UAVs for their research purpose. Examples include UAVs used for agricultural surveys, air quality logging. By “students” we mean aiming the use of our project in the classroom, especially thanks to the availability of an entry-level reference platform (see below).

The PhoenixPilot software is released under the GPL and will be treated in that spirit.  Porting the software to new boards is encouraged and fun.  The project will also maintain a set of reference platforms which the code will be more frequently tested against and will be expected to perform optimally.  As it was put, these will receive “A+ development support.”  As Lilvinz put it, with open source you can only give and create and we want to continue doing that.

## Getting involved
Click that big fork button on github and start coding!  We use pull requests as reviews so expect a lot of constructive feedback!

In addition check out https://groups.google.com/forum/#!forum/phoenixpilot for more discussion

Chat on freenode.net #phoenixpilot

## Code Layout

Here is a quick breakdown of the main directories to get you oriented

* flight - contains the firmware components of the code
* flight/target - the location of the board targets (e.g. flight/targets/freedom)
* flight/PiOS - contains the drivers
* flight/Modules - the flight control logic, broken into modules that communicate via UAVObjects
* flight/tests - unit tests for some components of the flight code
* ground - contains the GCS code
* shared - contains UAV Object definitions shared between the GCS and the flight firmware
* androidgcs - contains the ground control software for android

