import uavo

import numpy as np

class UAVOList(list):
    def __init__(self, uavo_defs, init_list = []):
        self.uavo_defs = uavo_defs
        self = init_list

    def as_numpy_array(self, match_class):
        # Find the subset of this list that is of the requested class
        filtered_list = filter(lambda x: isinstance(x, match_class), self)

        # Check for an empty list
        if filtered_list == []:
            return np.array([])

        # Find the uavo definition associated with this UAVO type
        if not "{0:08x}".format(filtered_list[0].uavo_id) in self.uavo_defs:
            dtype = None
        else:
            uavo_def = self.uavo_defs["{0:08x}".format(filtered_list[0].uavo_id)]
            dtype  = [('name', 'S20'), ('time', 'double'), ('uavo_id', 'uint')]

            for f in uavo_def.fields:
                dtype += [(f['name'], '(' + `f['elements']` + ",)" + uavo_def.type_numpy_map[f['type']])]

        return np.array(filtered_list, dtype=dtype)
