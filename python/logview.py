#!/usr/bin/python -B

# Insert the parent directory into the module import search path.
import os
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))

import argparse
import errno
import code
import struct

import taulabs.telemetry

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

    parser.add_argument("-v", "--viewer",
                        action  = 'store_true',
                        default = False,
                        dest    = "viewer",
                        help    = "launch the log viewer gui")

    parser.add_argument("-d", "--dumptext",
                        action = 'store_true',
                        default = False,
                        dest = "dumptext",
                        help = "dump the log file in text")

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

        parseHeader = False
        githash = None

        if args.githash is not None:
            # If we specify the log header no need to attempt to parse it
            githash = args.githash
        else:
            parseHeader = True

        print "Exporting UAVO XML files from git repo"

        from taulabs import telemetry

        uavo_list = telemetry.FileTelemetry(filename=src, parseHeader=parseHeader,
            weirdTimestamps=args.timestamped)

        print "Found %d unique UAVO definitions" % len(uavo_list.uavo_defs)

        print "Parsing using the LogFormat: " + `args.timestamped`

        # retrieve the time from the first object.. also guarantees we can parse
	base_time = next(iter(uavo_list)).time

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
            'uavo_defs' : uavo_list.uavo_defs,
            'uavo_list' : uavo_list,
            }

        # Extend the shell environment to include all of the uavo.UAVO_* classes that were
        # auto-created when the uavo xml files were processed.
        uavo_classes = [(t[0], t[1]) for t in taulabs.uavo.__dict__.iteritems() if 'UAVO_' in t[0]]
        user_module.__dict__.update(uavo_classes)

        if args.viewer:
            # Start the log viwer app
            from PyQt4 import QtGui
            from logviewer.gui import Window
            app = QtGui.QApplication(sys.argv)
            main = Window()
            main.show()
            main.plot(uavo_list, uavo_defs)
            sys.exit(app.exec_())
	elif args.dumptext:
            for obj in uavo_list:
                print obj
        else:
            # Instantiate an ipython shell to interact with the log data.
            import IPython
            from IPython.frontend.terminal.embed import InteractiveShellEmbed
            e = InteractiveShellEmbed(user_ns = user_ns, user_module = user_module)
            e.enable_pylab(import_all = True)
            e("Analyzing log file: %s" % src)

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    #import cProfile
    #cProfile.run('main()')
    main()
