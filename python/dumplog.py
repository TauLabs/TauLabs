#!/usr/bin/python

if __name__ == "__main__":
    from taulabs import telemetry
    uavo_list = telemetry.GetUavoBasedOnArgs()

    for o in uavo_list: print o
