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
    """ This is the prototype for a class that contains a uav object. """

    @classmethod
    def _make_to_send(cls, *args):
        """ Accepts all the uavo fields and creates an object.
        
        The object is dated as of the current time, and has the name and id
        fields set properly.
        """

        import time
        return cls(cls._name, round(time.time() * 1000), cls._id, *args)

    def bytes(self):
        """ Serializes this object into a byte stream. """
        return self.packstruct.pack(*flatten(self[3:]))

    @classmethod
    def get_size_of_data(cls):
        return cls.packstruct.size

    @classmethod
    def from_bytes(cls, data, timestamp, offset=0):
        """ Deserializes and creates an instance of this object.
        
         - data: the data to deserialize
         - timestamp: the timestamp to put on the object instance
         - offset: an optional index into data where to begin deserialization
        """
        import struct

        unpack_field_values = cls.packstruct.unpack_from(data, offset)

        field_values = []
        field_values.append(cls._name)

        # This gets a bit awkward. The order of field_values must match the structure
        # which for the intro header is name, timestamp, and id and then optionally
        # instance ID. For the timestamped packets we must parse the instance ID and
        # then the timestamp, so we will pop that out and shuffle the order. We also
        # convert from ms to seconds here.

        if timestamp != None:
            field_values.append(timestamp / 1000.0) 
        else:
            if cls._single:
                raw_ts = unpack_field_values.pop(0)
            else:
                raw_ts = unpack_field_values.pop(1)

            field_values.append(raw_ts / 1000.0)

        field_values.append(cls._id)

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
def make_class(xml_file):
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
            for i, field in enumerate(fields):
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
            info['type_val'] = type_enum_map[info['type']]
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
            for option in field['options']:
                hash_calc.update_hash_string(option)
    uavo_id = hash_calc.get_hash()

    ##### FORM A STRUCT TO PACK/UNPACK THIS UAVO'S CONTENT #####
    formats = []
    num_subelems = []

    # add format for instance-id IFF this is a multi-instance UAVO
    if not is_single_inst:
        # this is multi-instance so the optional instance-id is present
        formats.append('H')
        num_subelems.append(1)

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

    for f in fields:
        dtype += [(f['name'], '(' + `f['elements']` + ",)" + type_numpy_map[f['type']])]


    ##### DYNAMICALLY CREATE A CLASS TO CONTAIN THIS OBJECT #####

    from collections import namedtuple
    tuple_fields = ['name', 'time', 'uavo_id']
    if not is_single_inst:
        tuple_fields.append("inst_id")

    tuple_fields.extend([f['name'] for f in fields])

    name = 'UAVO_' + name

    class tmpClass(namedtuple(name, tuple_fields), UAVTupleClass):
        packstruct = fmt
        _flat = is_flat
        _name = name
        _id = uavo_id
        _single = is_single_inst
        _num_subelems = num_subelems
        _dtype = dtype

    # This is magic for two reasons.  First, we create the class to have
    # the proper dynamic name.  Second, we override __slots__, so that
    # child classes don't get a dict / keep all their namedtuple goodness
    tuple_class = type(name, (tmpClass,), { "__slots__" : () })

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
