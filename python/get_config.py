#!/usr/bin/python -B

import argparse
import errno
import code
import struct
import time

# Insert the parent directory into the module import search path.
import os
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))

from taulabs import uavo, telemetry, uavo_collection

#-------------------------------------------------------------------------------
USAGE = "%(prog)s"
DESC  = """
  Retrieve the configuration of a flight controller.\
"""

#-------------------------------------------------------------------------------
def main():
    tStream = telemetry.get_telemetry_by_args(service_in_iter=False)
    tStream.start_thread()

    settings_objects = tStream.uavo_defs.get_settings_objects()

    pipeline = []

    tStream.wait_connection()

    # try to retrieve every object 3 times
    # this is just intended for robustness; never seen this get an object
    # on anything other than the first try.
    for retry in range(3):
        for s in settings_objects:
            if tStream.last_values.get(s):
                continue

            tStream.request_object(s)
            pipeline.append(s)

            # Wait for outstanding requests to complete
            # Ideally this would continue on a nak
            for wait in range(20):
                # Wait minimum 10ms per object
                time.sleep(0.01)

                if len(pipeline) < 3:
                    break

                for p in pipeline[:]:
                    if tStream.last_values.get(p):
                        pipeline.remove(p)
            else:
                # We hit the end of the above retry loop
                pipeline = []

    missing = []

    for s in settings_objects:
        val = tStream.last_values.get(s)
        if val is not None:
            print val
        else:
            missing.append(s._name)

    for s in missing:
        print "No instance of %s" % (s)
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
