#!/usr/bin/python -B

def main():

    # Load the UAVO xml files in the workspace
    import taulabs
    uavo_defs = taulabs.uavo_collection.UAVOCollection()
    uavo_defs.from_uavo_xml_path('shared/uavobjectdefinition')

#-------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
