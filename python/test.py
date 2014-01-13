#!/usr/bin/python -B

# Insert the parent directory into the module import search path.
import os
import sys
sys.path.insert(1, os.path.dirname(sys.path[0]))

def main():

    # Load the UAVO xml files in the workspace
    import taulabs
    uavo_defs = taulabs.uavo_collection.UAVOCollection()
    uavo_defs.from_uavo_xml_path('shared/uavobjectdefinition')

    # Build a new module that will make up the global namespace for the
    # interactive shell.  This allows us to restrict what the ipython shell sees.
    import imp
    user_module = imp.new_module('taulabs_env')
    user_module.__dict__.update({
            'operator'  : __import__('operator'),
            })

    # Extend the shell environment to include all of the uavo.UAVO_* classes that were
    # auto-created when the uavo xml files were processed.
    uavo_classes = [(t[0], t[1]) for t in taulabs.uavo.__dict__.iteritems() if 'UAVO_' in t[0]]
    user_module.__dict__.update(uavo_classes)

    # Build the "user" (ie. local) namespace that the interactive shell
    # will see.  These variables show up in the output of "%whos" in the shell.
    user_ns = {
        'uavo_defs' : uavo_defs,
        }

    import IPython
    from IPython.frontend.terminal.embed import InteractiveShellEmbed
    e = InteractiveShellEmbed(user_ns = user_ns, user_module = user_module)
    e.enable_pylab(import_all = True)
    e("Have a look around.  Your UAVO definitions are in 'uavo_defs'.")

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
