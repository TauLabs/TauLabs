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
USAGE = "%(prog)s [logfile...]"
DESC  = """
  Load a Tau Labs log file into an ipython pylab (matplotlib + numpy) environment.\
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

    parser.add_argument("sources",
                        nargs = "+",
                        help  = "list of log files for processing")

    # Parse the command-line.
    args = parser.parse_args()

    # Process the source files.
    for src in args.sources:
        print "Reading Log file: %s" % src

        # Open the log file
        src = normalize_path(src)
        fd  = open(src, "rb")

        # Check the header signature
        #    First line is "Tau Labs git hash:"
        #    Second line is the actual git hash
        #    Third line is the UAVO hash
        #    Fourth line is "##"
        sig = fd.readline()
        if sig != 'Tau Labs git hash:\n':
            print "Source file does not have a recognized header signature"
            print '|' + sig + '|'
            sys.exit(2)

        # Determine the git hash that this log file is based on
        githash = fd.readline()[:-1]
        print "Log file is based on git hash: %s" % githash

        if args.githash is not None:
            print "Overriding git hash with '%s' instead of '%s' from file" % (args.githash, githash)
            githash = args.githash

        uavohash = fd.readline()
        divider = fd.readline()

        print "Exporting UAVO XML files from git repo"

        import taulabs
        uavo_defs = taulabs.uavo_collection.UAVOCollection()
        uavo_defs.from_git_hash(githash)
        uavo_list = taulabs.uavo_list.UAVOList(uavo_defs)

        print "Found %d unique UAVO definitions" % len(uavo_defs)

        parser = taulabs.uavtalk.UavTalk(uavo_defs)

        logFormat = True

        from collections import namedtuple
        LogHeader = namedtuple('LogHeader', 'time size')

        while fd:

            if logFormat and parser.state == taulabs.uavtalk.UavTalk.STATE_COMPLETE:
                # This logging format is somewhat of a hack and simply prepends additional
                # information in front of each UAVTalk packet.  We look for this information
                # whenever the parser has completed a packet. Note that there is no checksum
                # applied to this information so it can be totally messed up, especially if 
                # there is a frame shift error. The internal timestamping method of UAVTalk is
                # a much better idea.

                # Read the next log record header
                log_hdr_fmt = "<IQ"
                log_hdr_data = fd.read(struct.calcsize(log_hdr_fmt))

                # Check if we hit the end of the file
                if len(log_hdr_data) == 0:
                    # Normal End of File (EOF) at a record boundary
                    break;

                # Got a log record header.  Unpack it.
                log_hdr = LogHeader._make(struct.unpack(log_hdr_fmt, log_hdr_data))

            parser.processByte(ord(fd.read(1)))

            if parser.state == taulabs.uavtalk.UavTalk.STATE_COMPLETE:
                u  = parser.getLastReceivedObject(timestamp=log_hdr.time)
                uavo_list.append(u)

        fd.close()

        code.interact(local=locals())


#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
