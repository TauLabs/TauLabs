#!/usr/bin/python -B

# Insert the parent directory into the module import search path.
import os
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))

import argparse
import errno
import code
import struct
import time
from taulabs import uavo, telemetry, uavo_collection

#-------------------------------------------------------------------------------
USAGE = "%(prog)s"
DESC  = """
  Establish a minimal telemetry connection over TCP/IP.\
"""

#-------------------------------------------------------------------------------
def main():
    # Setup the command line arguments.
    parser = argparse.ArgumentParser(usage = USAGE, description = DESC)

    parser.add_argument("-g", "--githash",
                        action  = "store",
                        dest    = "githash",
                        help    = "override githash for UAVO XML definitions")

    # Parse the command-line.
    args = parser.parse_args()

    githash = args.githash

    tStream = telemetry.NetworkTelemetry(service_in_iter=False)
    tStream.start_thread()

    settings_objects = tStream.uavo_defs.get_settings_objects()

    # Need to actually control send rates and see when we're done.
    for s in settings_objects:
        tStream.request_object(s)
        time.sleep(0.15)

    time.sleep(2.5)

    for s in settings_objects:
        val = tStream.last_values.get(s)
        if val is not None:
            print val
        else:
            print "No instance of %s" % (s._name)

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
