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
def normalize_path(path):
    return os.path.normpath(os.path.join(os.getcwd(), path))

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

    githash = None
    if args.githash is not None:
        githash = args.githash

    uavo_defs = uavo_collection.UAVOCollection()
    if githash:
        uavo_defs.from_git_hash(githash)
    else:
        uavo_defs.from_uavo_xml_path("../shared/uavobjectdefinition")

    print "Found %d unique UAVO definitions" % len(uavo_defs)

    tStream = telemetry.Telemetry(uavo_defs)
    tStream.open_network()

#    tStream = telemetry.Telemetry(uavo_defs, serviceInIter=False)
#    tStream.open_network()
#    tStream.start_thread()


#    print settingsObjs
#    print tStream.get_last_values()
#    print [v for (k,v) in tStream.get_last_values().iteritems() 
#                if k in settingsObjs]

    for obj in tStream:
        print obj
        pass

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
