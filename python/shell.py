#!/usr/bin/python -B

from taulabs import telemetry, uavo

#-------------------------------------------------------------------------------
def main():
    uavo_list = telemetry.GetUavoBasedOnArgs()

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
        'log_file'  : uavo_list.filename,
        'githash'   : uavo_list.githash,
        'uavo_defs' : uavo_list.uavo_defs,
        'uavo_list' : uavo_list,
        }

    # Extend the shell environment to include all of the uavo.UAVO_* classes that were
    # auto-created when the uavo xml files were processed.
    uavo_classes = [(t[0], t[1]) for t in uavo.__dict__.iteritems() if 'UAVO_' in t[0]]
    user_module.__dict__.update(uavo_classes)

    # Instantiate an ipython shell to interact with the log data.
    import IPython
    from IPython.frontend.terminal.embed import InteractiveShellEmbed
    e = InteractiveShellEmbed(user_ns = user_ns, user_module = user_module)
    e.enable_pylab(import_all = True)
    e("Analyzing log file: %s" % uavo_list.filename)

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
