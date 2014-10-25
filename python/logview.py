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

    # Log format indicates this log is using the old file format which
    # embeds the timestamping information between the UAVTalk packet 
    # instead of as part of the packet
    parser.add_argument("-t", "--timestamped",
                        action  = 'store_false',
                        default = True,
                        help    = "indicate that this is an overo log file or some format that has timestamps")

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
        import cPickle as pickle
        try:
            pickle_name = src + '.pickle'
            pickle_fd = open(pickle_name, 'rb')
            githash = pickle.load(pickle_fd)
            uavo_parsed = pickle.load(pickle_fd)
            pickle_fd.close()
            print "Recovered %d log entries from git hash '%s' pickled log file '%s'" % (len(uavo_parsed), githash, pickle_name)
            pickle_data_loaded = True
        except:
            pickle_data_loaded = False

        if not pickle_data_loaded:
            fd  = open(src, "rb")

            if args.githash is not None:
                githash = args.githash
            else:
                # If we specify the log header no need to attempt to parse it

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

                uavohash = fd.readline()
                divider = fd.readline()

        print "Exporting UAVO XML files from git repo"

        import taulabs
        uavo_defs = taulabs.uavo_collection.UAVOCollection()
        uavo_defs.from_git_hash(githash)

        print "Found %d unique UAVO definitions" % len(uavo_defs)
        parser = taulabs.uavtalk.UavTalk(uavo_defs)

        base_time = None

        if not pickle_data_loaded:
            print "Parsing using the LogFormat: " + `args.timestamped`
            print "Reading log file..."
            uavo_parsed = []
            while fd:
                try:
                    if args.timestamped and parser.state == taulabs.uavtalk.UavTalk.STATE_COMPLETE:
                        # This logging format is somewhat of a hack and simply prepends additional
                        # information in front of each UAVTalk packet.  We look for this information
                        # whenever the parser has completed a packet. Note that there is no checksum
                        # applied to this information so it can be totally messed up, especially if 
                        # there is a frame shift error. The internal timestamping method of UAVTalk is
                        # a much better idea.

                        from collections import namedtuple
                        LogHeader = namedtuple('LogHeader', 'time size')

                        # Read the next log record header
                        log_hdr_fmt = "<IQ"
                        log_hdr_data = fd.read(struct.calcsize(log_hdr_fmt))

                        # Check if we hit the end of the file
                        if len(log_hdr_data) == 0:
                            # Normal End of File (EOF) at a record boundary
                            break

                        # Got a log record header.  Unpack it.
                        log_hdr = LogHeader._make(struct.unpack(log_hdr_fmt, log_hdr_data))

                        # Set the baseline timestamp from the first record in the log file
                        if base_time is None:
                            base_time = log_hdr.time


                    parser.processByte(ord(fd.read(1)))

                    if parser.state == taulabs.uavtalk.UavTalk.STATE_COMPLETE:
                        if args.timestamped:
                            u  = parser.getLastReceivedObject(timestamp=log_hdr.time)
                        else:
                            u  = parser.getLastReceivedObject()
                        if u is not None:
                            uavo_parsed.append(u)

                except TypeError:
                    print "End of file"
                    break

            fd.close()

            print "Processed %d Log File Records" % len(uavo_parsed)

            print "Writing pickled log file to '%s'" % pickle_name
            import cPickle as pickle
            pickle_fd = open(pickle_name, 'wb')
            pickle.dump(githash, pickle_fd)
            pickle.dump(uavo_parsed, pickle_fd)
            pickle_fd.close()

        print "Converting log records into python objects"
        uavo_list = taulabs.uavo_list.UAVOList(uavo_defs)
        for obj_id, data, timestamp in uavo_parsed:
            obj = uavo_defs[obj_id]
            u = obj.instance_from_bytes(data, timestamp)
            uavo_list.append(u)

        # We're done with this (potentially very large) variable, delete it.
        del uavo_parsed

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
