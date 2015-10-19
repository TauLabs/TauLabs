"""
Implementation of UAV Object Types

Copyright (C) 2014-2015 Tau Labs, http://taulabs.org
Licensed under the GNU LGPL version 2.1 or any later version (see COPYING.LESSER)
"""

import struct
import re
import warnings
import copy
from collections import namedtuple, OrderedDict

RE_SPECIAL_CHARS = re.compile('[\\.\\-\\s\\+/\\(\\)]')

def flatten(lst):
    result = []
    for element in lst:
        if hasattr(element, '__iter__'):
            result.extend(flatten(element))
        else:
            result.append(element)
    return result


class UAVTupleClass():
    """ This is the prototype for a class that contains a uav object. """

    @classmethod
    def _make_to_send(cls, *args, **kwargs):
        """ Accepts all the uavo fields and creates an object.

        The object is dated as of the current time, and has the name and id
        fields set properly.
        """

        import time
        return cls(cls._name, round(time.time() * 1000), cls._id, *args, **kwargs)

    def to_bytes(self):
        """ Serializes this object into a byte stream. """
        return self._packstruct.pack(*flatten(self[3:]))

    @classmethod
    def get_size_of_data(cls):
        return cls._packstruct.size

    @classmethod
    def from_bytes(cls, data, timestamp, instance_id, offset=0):
        """ Deserializes and creates an instance of this object.

         - data: the data to deserialize
         - timestamp: the timestamp to put on the object instance
         - offset: an optional index into data where to begin deserialization
        """
        import struct

        unpack_field_values = cls._packstruct.unpack_from(data, offset)

        field_values = []
        field_values.append(cls._name)

        if timestamp is not None:
            field_values.append(timestamp / 1000.0)

        field_values.append(cls._id)

        if instance_id is not None:
            field_values.append(instance_id)

        # add the remaining fields.  If the thing should be nested, construct
        # an appropriate tuple.
        if not cls._flat:
            pos = 0

            for n in cls._num_subelems:
                if n == 1:
                    field_values.append(unpack_field_values[pos])
                else:
                    field_values.append(tuple(unpack_field_values[pos:pos+n]))
                pos += n

            field_values = tuple(field_values)
        else:
            # Short cut; nothing is nested
            field_values = tuple(field_values) + tuple(unpack_field_values)

        return cls._make(field_values)

    def __repr__(self):
        """ String representation of the contents """
        rep = self.__class__.__name__ + '('
        for field in self._fields:
            field_value = getattr(self, field)

            # Show ENUMs in a fancy way (string and value)
            if hasattr(self, 'ENUMR_' + field):
                if not isinstance(field_value, tuple):
                    field_value = (field_value,)
                value_str = ''
                this_enum = getattr(self, 'ENUMR_' + field)
                for ii, v in enumerate(field_value):
                    value_str += '%s(%d)' % (this_enum.get(v, "UNKNOWN"), v)
                    if ii < len(field_value) - 1:
                        value_str += ', '
                if len(field_value) > 1:
                    value_str = '(%s)' % value_str
            elif field == 'uavo_id':
                value_str = '0x%X' % field_value
            else:
                value_str = str(field_value)
            rep += '%s=%s, ' % (field, value_str)
        rep = rep[:-2] + ')'
        return rep

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

# This is a very long, scary method.  It parses an XML file describing
# a UAVO and builds an implementation class.
def make_class(collection, xml_file, update_globals=True):
    fields = []

    ##### PARSE THE XML FILE INTO INTERNAL REPRESENTATIONS #####

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

    name = subs['object'].get('name')
    is_single_inst = int((subs['object'].get('singleinstance') == 'true'))
    is_settings = int((subs['object'].get('settings') == 'true'))

    description = subs['description'].text

    ##### CONSTRUCT PROPER INTERNAL REPRESENTATION OF FIELD DATA #####
    for field in subs['fields']:
        info = {}
        # process typical attributes
        for attr in ['name', 'units', 'type', 'elements', 'elementnames',
                     'parent', 'defaultvalue']:
            info[attr] = field.get(attr)
        if field.get('cloneof'):
            # this is a clone of another field, find its data
            cloneof_name = field.get('cloneof')
            for i, field in enumerate(fields):
                if field['name'] == cloneof_name:
                    clone_info = copy.deepcopy(field)
                    break

            # replace it with the new name
            clone_info['name'] = info['name']
            # use the expanded/substituted info instead of the stub
            info = clone_info
        else:
            if info['elements'] is not None:
                # we've got an inline "elements" attribute
                info['elementnames'] = []
                info['elements'] = int(field.get('elements'))
            elif info['elementnames'] is not None:
                # we've got an inline "elementnames" attribute
                info['elementnames'] = []
                info['elements'] = 0
                for elementname in field.get('elementnames').split(','):
                    info['elementnames'].append(RE_SPECIAL_CHARS.sub('', elementname))
                    info['elements'] += 1
            else:
                # we must have one or more elementnames/elementname elements in this sub-tree
                info['elementnames'] = []
                info['elements'] = 0
                for elementname_text in [elementname.text for elementname in field.findall('elementnames/elementname')]:
                    info['elementnames'].append(elementname_text)
                    info['elements'] += 1

            if info['type'] == 'enum':
                info['options'] = OrderedDict()
                if field.get('options'):
                    # we've got an inline "options" attribute
                    for ii, option_text in enumerate(field.get('options').split(',')):
                        info['options'][option_text.strip()] = ii
                else:
                    # we must have some 'option' elements in this sub-tree
                    for ii, option_text in enumerate([option.text
                            for option in field.findall('options/option')]):
                        info['options'][option_text.strip()] = ii

            # convert type string to an int
            info['type_val'] = type_enum_map[info['type']]

            # Get parent
            if info['parent'] is not None:
                parent_name, field_name = info['parent'].split('.')
                parent_class = collection.find_by_name(parent_name)
                if len(info['options']) == 0:
                    parent_options = getattr(parent_class, 'ENUMR_' + field_name)
                    for k, v in sorted(parent_options.iteritems(), key=lambda x: x[0]):
                        info['options'][v] = k
                else:
                    parent_options = getattr(parent_class, 'ENUM_' + field_name)
                    for k in info['options'].iterkeys():
                        info['options'][k] = parent_options[k]

            # Add default values
            if info['defaultvalue'] is not None:
                if info['type'] == 'enum':
                    try:
                        values = tuple(info['options'][v.strip()]
                                       for v in info['defaultvalue'].split(','))
                    except KeyError:
                        warnings.warn('Invalid default value: %s.%s has no option %s'
                                      % (name, info['name'], info['defaultvalue']))
                        values = (0,)
                else:  # float or int
                    values = tuple(float(v) for v in info['defaultvalue'].split(','))
                    if info['type'] != float:
                        values = tuple(int(v) for v in values)

                if len(values) == 1:
                    values = values[0]
                info['defaultvalue'] = values
            else:
                # Use 0 as the default
                info['defaultvalue'] = 0

            if info['elements'] > 1 and not isinstance(info['defaultvalue'], tuple):
                info['defaultvalue'] = (info['defaultvalue'],) * info['elements']

        fields.append(info)

    # Sort fields by size (bigger to smaller) to ensure alignment when packed
    fields.sort(key=lambda x: struct.calcsize(struct_element_map[x['type']]), reverse = True)

    ##### CALCULATE THE APPROPRIATE UAVO ID #####
    hash_calc = UAVOHash()

    hash_calc.update_hash_string(name)
    hash_calc.update_hash_byte(is_settings)
    hash_calc.update_hash_byte(is_single_inst)

    for field in fields:
        hash_calc.update_hash_string(field['name'])
        hash_calc.update_hash_byte(int(field['elements']))
        hash_calc.update_hash_byte(field['type_val'])
        if field['type'] == 'enum':
            next_idx = 0
            for option, idx in field['options'].iteritems():
                if idx != next_idx:
                    hash_calc.update_hash_byte(idx)
                hash_calc.update_hash_string(option)
                next_idx = idx + 1

    uavo_id = hash_calc.get_hash()

    ##### FORM A STRUCT TO PACK/UNPACK THIS UAVO'S CONTENT #####
    formats = []
    num_subelems = []

    is_flat = True

    # add formats for each field
    for f in fields:
        if f['elements'] != 1:
            is_flat = False

        num_subelems.append(f['elements'])

        formats.append('' + f['elements'].__str__() + struct_element_map[f['type']])

    fmt = struct.Struct('<' + ''.join(formats))

    ##### CALCULATE THE NUMPY TYPE ASSOCIATED WITH THIS CLASS #####
    dtype  = [('name', 'S20'), ('time', 'double'), ('uavo_id', 'uint')]

    if not is_single_inst:
        dtype += ('inst_id', 'uint'),

    for f in fields:
        dtype += [(f['name'], '(' + `f['elements']` + ",)" + type_numpy_map[f['type']])]


    ##### DYNAMICALLY CREATE A CLASS TO CONTAIN THIS OBJECT #####
    tuple_fields = ['name', 'time', 'uavo_id']
    if not is_single_inst:
        tuple_fields.append("inst_id")

    tuple_fields.extend([f['name'] for f in fields])

    name = 'UAVO_' + name

    class tmpClass(UAVTupleClass, namedtuple(name, tuple_fields)):
        _packstruct = fmt
        _flat = is_flat
        _name = name
        _id = uavo_id
        _single = is_single_inst
        _num_subelems = num_subelems
        _dtype = dtype
        _is_settings = is_settings

    # This is magic for two reasons.  First, we create the class to have
    # the proper dynamic name.  Second, we override __slots__, so that
    # child classes don't get a dict / keep all their namedtuple goodness
    tuple_class = type(name, (tmpClass,), { "__slots__" : () })

    # Set the default values for the constructor
    defaults = [name, 0, uavo_id]
    defaults.extend(f['defaultvalue'] for f in fields)
    tuple_class.__new__.__defaults__ = tuple(defaults)

    # Add enums
    for field in fields:
        if field['type'] == 'enum':
            enum = field['options']
            mapping = dict((key, value) for key, value in enum.iteritems())
            reverse_mapping = dict((value, key) for key, value in enum.iteritems())
            setattr(tuple_class, 'ENUM_' + field['name'], mapping)
            setattr(tuple_class, 'ENUMR_' + field['name'], reverse_mapping)

    if update_globals:
        globals()[tuple_class.__name__] = tuple_class


    return tuple_class


class UAVOHash():
    def __init__(self):
        self.hval = 0

    def update_hash_byte(self, value):
        self.hval = (self.hval ^ ((self.hval << 5) + (self.hval >> 2) + value)) & 0x0FFFFFFFF

    def update_hash_string(self, string):
        for c in string:
            self.update_hash_byte(ord(c))

    def get_hash(self):
        return self.hval & 0x0FFFFFFFE
