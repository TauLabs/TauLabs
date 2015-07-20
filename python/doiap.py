#!/usr/bin/python -B

import sys, time
from taulabs import telemetry

def main():
    t = telemetry.get_telemetry_by_args(service_in_iter=False, iter_blocks=False)
    t.start_thread()

    magic_value=1122

    from taulabs.uavo import UAVO_FirmwareIAPObj

    t.request_object(UAVO_FirmwareIAPObj)

    for i in range(3):
        print magic_value
        f=UAVO_FirmwareIAPObj._make_to_send(Command=magic_value,
                Description=(0,)*100,
                CPUSerial=(0,)*12,
                BoardRevision=0, 
                BoardType=0,
                ArmReset=0,
                crc=0)

        magic_value += 1111

        t.send_object(f)
        time.sleep(0.9)

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
