#!/usr/bin/python -B

# Insert the parent directory into the module import search path.
import os
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))

import argparse
import errno

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

        print "Found %d unique UAVO definitions" % len(uavo_defs)

        # Remaining data in the file is in this format:
        #   For GCS logging the format is as follows
        #   4 bytes timestamp (milliseconds)
        #   8 bytes data size
        #   UAVTalk packet (always without timestamped packets)
        #   Sync val (0x3c)
        #   Message type (1 byte)
        #   Length (2 bytes)
        #   Object ID (4 bytes)
        #   Instance ID (optional, 2 bytes)
        #   Data (variable length)
        #   Checksum (1 byte)

        print "Processing Log File Records"

        from collections import namedtuple
        LogHeader = namedtuple('LogHeader', 'time size')
        UAVOHeader = namedtuple('UAVOHeader', 'sync type length id')

        import struct
        base_time = None

        uavo_list = taulabs.uavo_list.UAVOList(uavo_defs)
        while fd:
            # Read the next log record header
            log_hdr_fmt = "<IQ"
            log_hdr_data = fd.read(struct.calcsize(log_hdr_fmt))

            # Check if we hit the end of the file
            if len(log_hdr_data) == 0:
                # Normal End of File (EOF) at a record boundary
                break;

            # Got a log record header.  Unpack it.
            log_hdr = LogHeader._make(struct.unpack(log_hdr_fmt, log_hdr_data))

            # Set the baseline timestamp from the first record in the log file
            if base_time is None:
                base_time = log_hdr.time

            # Read the UAVO message header
            uavo_hdr_fmt = "<BBHI"
            uavo_hdr_data = fd.read(struct.calcsize(uavo_hdr_fmt))

            # Got a UAVO message header.  Unpack it.
            uavo_hdr = UAVOHeader._make(struct.unpack(uavo_hdr_fmt, uavo_hdr_data))

            # Ignore meta objects entirely
            if uavo_hdr.id & 0x1:
                # discard the rest of the log entry
                fd.read(min(log_hdr.size,256) - len(uavo_hdr_data))
                continue

            # Is this a known UAVO?
            uavo_key = '{0:08x}'.format(uavo_hdr.id)
            if not uavo_key in uavo_defs:
                # Unknown UAVO.  Discard the rest of the log entry.
                print "Unknown UAVO: %s" % uavo_key
                fd.read(min(log_hdr.size,256) - len(uavo_hdr_data))
                continue

            # This is a known UAVO.
            # Use the UAVO definition to read and parse the remainder of the UAVO message.
            uavo_def = uavo_defs[uavo_key]
            data = fd.read(uavo_def.get_size_of_data())
            u = uavo_def.instance_from_bytes(log_hdr.time, data)
            uavo_list.append(u)

            # Read (and discard) the checksum
            fd.read(1)

            # Make sure our sizes all matched up
            if log_hdr.size - len(uavo_hdr_data) - 1 != uavo_def.get_size_of_data():
                print "size mismatch"

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
