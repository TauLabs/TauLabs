#!/usr/bin/python -B

# Insert the parent directory into the module import search path.
import os
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))

import argparse
import errno
import math

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
        statinfo = os.stat(src)

        if args.githash is not None:
            githash = args.githash
        else:
            print "githash must be specified on the line for overo files"
            sys.exit(2)

        print "Exporting UAVO XML files from git repo"

        import taulabs
        uavo_defs = taulabs.uavo_collection.UAVOCollection()
        uavo_defs.from_git_hash(githash)

        print "Found %d unique UAVO definitions" % len(uavo_defs)

        # Remaining data in the file is in this format:
        #   For GCS logging the format is as follows
        #   UAVTalk packet (always with timestamped packets)
        #   Sync val (0x3c)
        #   Message type (1 byte)
        #   Length (2 bytes)
        #   Object ID (4 bytes)
        #   Instance ID (optional, 2 bytes)
        #   Typestamp (optional, 2 bytes)
        #   Data (variable length)
        #   Checksum (1 byte)

        print "Processing Log File Records"

        from collections import namedtuple
        UAVOSync   = namedtuple('UAVOSync', 'sync')
        UAVOHeader = namedtuple('UAVOHeader', 'type length id')

        import struct
        base_time = None

        uavo_list = taulabs.uavo_list.UAVOList(uavo_defs)

        unknown = False

        i = 0
        while fd:

            # Print message for every MB processed
            if math.floor(fd.tell() / 1e6) > i:
                i = math.floor(fd.tell() / 1e6)
                print "Processed " + `fd.tell()` + " / " + `statinfo.st_size`


            # Check for valid sync byte
            uavo_sync_fmt = "<B"
            uavo_sync_data = fd.read(struct.calcsize(uavo_sync_fmt))
            uavo_sync = UAVOSync._make(struct.unpack(uavo_sync_fmt, uavo_sync_data))
            
            if uavo_sync.sync == 0xff:
                # Overo files have padding of 0xFF, this is fine
                continue
            elif unknown == True:
                # When unknown UAVO expected to lose sync
                continue
            elif uavo_sync.sync != 0x3c:
                print "Bad sync: " + `uavo_sync`
                continue

            # Read the UAVO message header
            uavo_hdr_fmt = "<BHI"
            uavo_hdr_data = fd.read(struct.calcsize(uavo_hdr_fmt))

            # Got a UAVO message header.  Unpack it.
            uavo_hdr = UAVOHeader._make(struct.unpack(uavo_hdr_fmt, uavo_hdr_data))

            #print uavo_hdr
            # Ignore meta objects entirely
            if uavo_hdr.id & 0x1:
                print "Found meta data. Strange in log"
                continue

            if (uavo_hdr.type & 0x80) == 0:
                print "Error, non timestamped packet"
                continue

            # Is this a known UAVO?
            uavo_key = '{0:08x}'.format(uavo_hdr.id)
            if not uavo_key in uavo_defs:
                # Unknown UAVO.  Discard the rest of the log entry.
                print "Unknown UAVO: %s" % uavo_key
                unknown = True
                continue

            unknown = False

            # Assumes a timestamped packet
            packet_length = uavo_hdr.length - 8

            # This is a known UAVO.
            # Use the UAVO definition to read and parse the remainder of the UAVO message.
            uavo_def = uavo_defs[uavo_key]
            data = fd.read(packet_length) #uavo_def.get_size_of_data())
            u = uavo_def.instance_from_bytes(data, timestamp_packet=True)
            uavo_list.append(u)

            # Check the lengths made sense
            if packet_length != (uavo_def.get_size_of_data() + 2):
                print "Packet length mismatch"
                continue

            # Read (and discard) the checksum
            fd.read(1)

        fd.close()

        print "Processed %d Log File Records" % len(uavo_list)

        # Build a new module that will make up the global namespace for the
        # interactive shell.  This allows us to restrict what the ipython shell sees.
        import imp
        user_module = imp.new_module('taulabs_env')
        user_module.__dict__.update({
                'operator'  : __import__('operator'),
                'base_time' : base_time,
                })

        # Build the "user" (ie. local) namespace that the interactive shell
        # will see.  These variables show up in the output of "%whos" in the shell.
        user_ns = {
            'base_time' : base_time,
            'log_file'  : src,
            'githash'   : githash,
            'uavo_defs' : uavo_defs,
            'uavo_list' : uavo_list,
            }

        # Extend the shell environment to include all of the uavo.UAVO_* classes that were
        # auto-created when the uavo xml files were processed.
        uavo_classes = [(t[0], t[1]) for t in taulabs.uavo.__dict__.iteritems() if 'UAVO_' in t[0]]
        user_module.__dict__.update(uavo_classes)

        # Instantiate an ipython shell to interact with the log data.
        import IPython
        from IPython.frontend.terminal.embed import InteractiveShellEmbed
        e = InteractiveShellEmbed(user_ns = user_ns, user_module = user_module)
        e.enable_pylab(import_all = True)
        e("Analyzing log file: %s" % src)

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
