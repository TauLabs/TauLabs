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

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
