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

    tStream = telemetry.NetworkTelemetry()

#    tStream = telemetry.NetworkTelemetry(serviceInIter=False)
#    tStream.start_thread()

#    print settingsObjs
#    print tStream.get_last_values()
#    print [v for (k,v) in tStream.get_last_values().iteritems() 
#                if k in settingsObjs]

    for obj in tStream:
        #print obj
        pass

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
