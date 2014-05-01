#!/usr/bin/python -B

# Insert the parent directory into the module import search path.
import os
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))

import argparse
import errno
import code
import struct

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

    githash = "next"
    if args.githash is not None:
        githash = args.githash

    import taulabs
    uavo_defs = taulabs.uavo_collection.UAVOCollection()
    uavo_defs.from_git_hash(githash)
    uavo_list = taulabs.uavo_list.UAVOList(uavo_defs)

    # Expose the UAVO types to the local workspace
    uavo_classes = [(t[0], t[1]) for t in taulabs.uavo.__dict__.iteritems() if 'UAVO_' in t[0]]
    locals().update(uavo_classes)

    print "Found %d unique UAVO definitions" % len(uavo_defs)

    parser = taulabs.uavtalk.UavTalk(uavo_defs)
    telemetry = taulabs.telemetry.Telemetry(parser)

    telemetry.open_network()

    while True:
        telemetry.serviceConnection()

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
