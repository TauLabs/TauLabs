"""
UAVO collection interface.

Copyright (C) 2014-2015 Tau Labs, http://taulabs.org
Licensed under the GNU LGPL version 2.1 or any later version (see COPYING.LESSER)
"""

import uavo

import operator
import os.path as op

class UAVOCollection(dict):
    def __init__(self):
        self.clear()

    def find_by_name(self, uavo_name):
        if uavo_name[0:5]=='UAVO_':
            uavo_name = uavo_name[5:]

        for u in self.itervalues():
            if u._name == uavo_name:
                return u

        return None

    def get_settings_objects(self):
        objs = [ u for u in self.itervalues() if u._is_settings ]
        objs.sort(key=operator.attrgetter('_name'))

        return objs

    def from_git_hash(self, githash):
        import subprocess
        import tarfile
        from cStringIO import StringIO
        # Get the directory where the Tau Labs code is located
        tl_dir = op.join(op.dirname(__file__), "..", "..")
        #
        # Grab the exact uavo definition files from the git repo using the header's git hash
        #
        p = subprocess.Popen(['git', 'archive', githash, '--', 'shared/uavobjectdefinition/'],
                             stdout=subprocess.PIPE, cwd=tl_dir)
        # grab the tar file data
        git_archive_data, git_archive_errors = p.communicate()

        # coerce the tar file data into a file object so that tarfile likes it
        fobj = StringIO(git_archive_data)

        # feed the tar file data to a tarfile object
        t = tarfile.open(fileobj=fobj)

        # Get the file members
        f_members = []
        for f_info in t.getmembers():
            if not f_info.isfile():
                continue
            # Make sure the shared definitions are parsed first
            if op.basename(f_info.name).lower().find('shared') > 0:
                f_members.insert(0, f_info)
            else:
                f_members.append(f_info)

        # Build up the UAV objects from the xml definitions
        for f_info in f_members:
            f = t.extractfile(f_info)

            u = uavo.make_class(f)

            # add this uavo definition to our dictionary
            self.update([('{0:08x}'.format(u._id), u)])

    def from_uavo_xml_path(self, path):
        import os
        import glob

        # Make sure the shared definitions are parsed first
        file_names = []
        for file_name in glob.glob(os.path.join(path, '*.xml')):
            if op.basename(file_name).lower().find('shared') > 0:
                file_names.insert(0, file_name)
            else:
                file_names.append(file_name)

        for file_name in file_names:
            with open(file_name, 'rU') as f:
                u = uavo.make_class(f)

                # add this uavo definition to our dictionary
                self.update([('{0:08x}'.format(u._id), u)])
