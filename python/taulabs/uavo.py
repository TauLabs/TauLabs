import struct

def flatten(lst):
    result = []
    for element in lst: 
        if hasattr(element, '__iter__'):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result

class UAVTupleClass():
    def bytes(self):
        return self.packstruct.pack(*flatten(self[3:]))

    @classmethod
    def from_bytes(cls, data, timestamp, startOffs=0):
        import struct

        uavo = cls.uavometa

        #
        # add the values
        #

        # unpack each field separately
        offset = startOffs

        if not cls.flat:
            unpack_field_values = []
            for fmt in cls.formats:
                val = fmt.unpack_from(data, offset)
                if len(val) == 1:
                    # elevate the value outside of the tuple if there is
                    # exactly one value
                    val = val[0]
                unpack_field_values.append(val)
                offset += fmt.size
        else:
            unpack_field_values = cls.packstruct.unpack_from(data, offset)

        field_values = []
        field_values.append(uavo.meta['name'])

        # This gets a bit awkward. The order of field_values must match the structure
        # which for the intro header is name, timestamp, and id and then optionally
        # instance ID. For the timestamped packets we must parse the instance ID and
        # then the timestamp, so we will pop that out and shuffle the order. We also
        # convert from ms to seconds here.

        if timestamp != None:
            field_values.append(timestamp / 1000.0) 
        else:
            if uavo.meta['is_single_inst']:
                offset = 0
            else:
                offset = 1
            field_values.append(unpack_field_values.pop(offset) / 1000.0)
        field_values.append(uavo.id)

        # add the remaining fields
        field_values = tuple(field_values) + tuple(unpack_field_values)

        return cls._make(field_values)


class UAVO():
    type_enum_map = {
        'int8'    : 0,
        'int16'   : 1,
        'int32'   : 2,
        'uint8'   : 3,
        'uint16'  : 4,
        'uint32'  : 5,
        'float'   : 6,
        'enum'    : 7,
        }

    type_numpy_map = {
        'int8'    : 'int8',
        'int16'   : 'int16',
        'int32'   : 'int32',
        'uint8'   : 'uint8',
        'uint16'  : 'uint16',
        'uint32'  : 'uint32',
        'float'   : 'float',
        'enum'    : 'uint8',
        }

    type_size_map = {
        'int8'    : 1,
        'int16'   : 2,
        'int32'   : 4,
        'uint8'   : 1,
        'uint16'  : 2,
        'uint32'  : 4,
        'float'   : 4,
        'enum'    : 1,
        }

    struct_element_map = {
        'int8'    : 'b',
        'int16'   : 'h',
        'int32'   : 'i',
        'uint8'   : 'B',
        'uint16'  : 'H',
        'uint32'  : 'I',
        'float'   : 'f',
        'enum'    : 'B',
        }

    def __init__(self, xml_file):
        self.fields = []
        self.meta = {}
        self.id = 0

        self.hash = 0

        from lxml import etree

        tree = etree.parse(xml_file)

        subs = {
            'tree'            : tree,
            'object'          : tree.find('object'),
            'fields'          : tree.findall('object/field'),
            'description'     : tree.find('object/description'),
            'access'          : tree.find('object/access'),
            'logging'         : tree.find('object/logging'),
            'telemetrygcs'    : tree.find('object/telemetrygcs'),
            'telemetryflight' : tree.find('object/telemetryflight'),
            }

        self.tree = tree
        self.subs = subs

        self.meta['name']           = subs['object'].get('name')
        self.meta['is_single_inst'] = int((subs['object'].get('singleinstance') == 'true'))
        self.meta['is_settings']    = int((subs['object'].get('settings') == 'true'))

        self.meta['description']    = subs['description'].text

        import copy
        import re
        for field in subs['fields']:
            info = {}
            # process typical attributes
            for attr in ['name', 'units', 'type', 'elements', 'elementnames']:
                info[attr] = field.get(attr)
            if field.get('cloneof'):
                # this is a clone of another field, find its data
                cloneof_name = field.get('cloneof')
                for i, field in enumerate(self.fields):
                    if field['name'] == cloneof_name:
                        clone_info = copy.deepcopy(field)
                        break

                # replace it with the new name
                clone_info['name'] = info['name']
                # use the expanded/substituted info instead of the stub
                info = clone_info
            else:
                if info['elements'] != None:
                    # we've got an inline "elements" attribute
                    info['elementnames'] = []
                    info['elements'] = int(field.get('elements'))
                elif info['elementnames'] != None:
                    # we've got an inline "elementnames" attribute
                    info['elementnames'] = []
                    info['elements'] = 0
                    for elementname in field.get('elementnames').split(','):
                        info['elementnames'].append(elementname.strip(' '))
                        info['elements'] += 1
                else:
                    # we must have one or more elementnames/elementname elements in this sub-tree
                    info['elementnames'] = []
                    info['elements'] = 0
                    for elementname_text in [elementname.text for elementname in field.findall('elementnames/elementname')]:
                        info['elementnames'].append(elementname_text)
                        info['elements'] += 1

                if info['type'] == 'enum':
                    info['options'] = []
                    if field.get('options'):
                        # we've got an inline "options" attribute
                        for option_text in field.get('options').split(','):
                            info['options'].append(option_text.strip(' '))
                    else:
                        # we must have some 'option' elements in this sub-tree
                        for option_text in [option.text for option in field.findall('options/option')]:
                            info['options'].append(option_text)

                # convert type string to an int
                info['type_val'] = self.type_enum_map[info['type']]
            self.fields.append(info)

        # Sort fields by size (bigger to smaller) to ensure alignment when packed
        self.fields.sort(key=lambda x: self.type_size_map[x['type']], reverse = True)

        self.id = self._calculate_id()

        self.__build_class_of()

    def __build_class_of(self):
        from collections import namedtuple
        fields = ['name', 'time', 'uavo_id']
        if not self.meta['is_single_inst']:
            fields.append("inst_id")

        fields.extend([f['name'] for f in self.fields])

        name = 'UAVO_' + self.meta['name']

        self.form_packformat()

        class tmpClass(namedtuple(name, fields), UAVTupleClass):
            packstruct = self.fmt
            formats = self.formats
            flat = self.flat
            uavometa = self

        # This is magic for two reasons.  First, we create the class to have
        # the proper dynamic name.  Second, we override __slots__, so that
        # child classes don't get a dict / keep all their namedtuple goodness
        self.tuple_class = type(name, (tmpClass,), { "__slots__" : () })

        # Make sure this new class is exposed in the module globals so that it can be pickled
        globals()[self.tuple_class.__name__] = self.tuple_class

    def get_size_of_data(self):
        size = 0

        if not self.meta['is_single_inst']:
            # this is multi-instance so the optional instance-id is present
            size += 2

        for f in self.fields:
            size += self.type_size_map[f['type']] * f['elements']

        return size

    def form_packformat(self):
        formats = []

        # add format for instance-id IFF this is a multi-instance UAVO
        if not self.meta['is_single_inst']:
            # this is multi-instance so the optional instance-id is present
            formats.append('H')

        flat = True

        # add formats for each field
        for f in self.fields:
            if f['elements'] != 1:
                flat = False

            formats.append('' + f['elements'].__str__() + self.struct_element_map[f['type']])

        self.flat = flat

        self.fmt = struct.Struct('<' + ''.join(formats))

        self.formats = [ struct.Struct('<' + f) for f in formats ]

    def instance_from_bytes(self, *args, **kwargs):
        return self.tuple_class.from_bytes(*args, **kwargs)

    def __str__(self):
        return "%s(id='%08x') %s" % (self.meta['name'], self.id, " ".join([f['name'] for f in self.fields]))

    def __repr__(self):
        return "%s(id='%08x', name=%r)" % (self.__class__, self.id, self.meta['name'])

    def _update_hash_byte(self, value):
        self.hash = (self.hash ^ ((self.hash << 5) + (self.hash >> 2) + value)) & 0x0FFFFFFFF

    def _update_hash_string(self, string):
        for c in string:
            self._update_hash_byte(ord(c))

    def _calculate_id(self):
        self.hash = 0

        self._update_hash_string(self.meta['name'])
        self._update_hash_byte(self.meta['is_settings'])
        self._update_hash_byte(self.meta['is_single_inst'])

        for field in self.fields:
            self._update_hash_string(field['name'])
            self._update_hash_byte(int(field['elements']))
            self._update_hash_byte(field['type_val'])
            if field['type'] == 'enum':
                for option in field['options']:
                    self._update_hash_string(option)

        return self.hash & 0x0FFFFFFFE

