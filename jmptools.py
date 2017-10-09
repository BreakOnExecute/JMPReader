# JMP 11 binary file reader module by Thomas K. Reynolds

# MIT License
#
# Copyright (c) 2016 Thomas K. Reynolds
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import pandas as pd
import struct
import datetime
import math

import os

# for display purposes
pd.set_option('expand_frame_repr', False)

# module global variables
version = 1.0  # reader version
_data = None  # pandas dataframe that will hold all the JMP file data for module internal use (no JMP formatting info)
_abs_column_addresses = []  # Every JMP file holds an absolute file offset for where each set of column info is held
_jmpfile = None   # file object for the JMP file for reading
_n_rows = 0  # number of rows in the JMP data table
_n_columns = 0  # number of columns in the JMP data table

# column_format_type splits into these 4 categories for date/time, time only, date only, duration formatting
# Internally in the JMP file, the data for such a column will be stored as a double.  But information below
# is used to figure out from the formatting how I should extract the value.
_column_format_type_dt_vals = [0x69, 0x6A, 0x73, 0x74, 0x7D, 0x7E, 0x77, 0x78, 0x86, 0x87, 0x7B, 0x7C,
                               0x80, 0x81, 0x89, 0x8A]
_column_format_type_t_vals = [0x79, 0x82]
_column_format_type_d_vals = [0x65, 0x6E, 0x6F, 0x8B, 0x70, 0x71, 0x72, 0x7A, 0x75, 0x76, 0x7F, 0x66, 0x67, 0x88]
_column_format_type_dur_vals = [0x6C, 0x6D, 0x83, 0x84, 0x85]
_column_format_type_bucket = {"datetime": 0, "time": 1, "date": 2, "duration": 3}


# for debug work:
# take a byte array as input and give formatted hex string as output
# num_per_line is further formatting for output
def _bytetohex(bytearray_in, num_per_line=16):
    byte_array_length = len(bytearray_in)
    s = ""
    for idx, x in enumerate(bytearray_in):
        if (idx % num_per_line == 0) & (idx != 0):
            s += '\r\n'
        if idx == byte_array_length - 1:
            s += '{:02X}'.format(x)
        else:
            s += '{:02X} '.format(x)
    return s

# for debug work:
# take a list of byte arrays, and output a string of bytes
def _list_of_byte_array_to_hex(list_of_byte_array):
    s = ""
    list_length = len(list_of_byte_array)
    for idx, x in enumerate(list_of_byte_array):
        s += _bytetohex(x)
        if idx < list_length - 1:
            s += "\r\n"
    return s


# ***main routine to call***
# pass a JMP filename/path this this function
# return an error code, an error message, and copy of the _data (if no error) as a pandas dataframe
def readjmp(filename):
    global _jmpfile
    global _data
    global _abs_column_addresses
    _data = None
    _abs_column_addresses = []

    _jmpfile = open(filename, "rb")
    try:
        _read_header()
    except ValueError, e:
        return -1, e.message, None
    try:
        _decode_all_columns()
    except ValueError, e:
        return -2, e.message, None

    _jmpfile.close()
    return 0, "No error", _data.copy(deep=True)


# helper function: read n_bytes worth of data from the JMP file
def _read_bytes(n_bytes):
    global _jmpfile
    return bytes(_jmpfile.read(n_bytes))


# read the JMP header - this is the part before you get to the column definitions that contain the actual data
# It has a lot of interesting stuff in it (scripts, row state, etc.), but I just read over that to get to the data.
def _read_header():
    global _n_rows
    global _n_columns
    global _abs_column_addresses
    temp_data = _read_bytes(8)  # first 8 bytes should be FF FF 00 00 03 00 00 00 for JMP 11 _jmpfile
    if temp_data != bytearray.fromhex("FF FF 00 00 03 00 00 00"):
        raise ValueError('Error while reading header - _jmpfile is most likely not a JMP 11 version _jmpfile')
    _n_rows = struct.unpack("I", _read_bytes(4))[0]  # next 4 bytes are # of rows
    _n_columns = struct.unpack("I", _read_bytes(4))[0]  # next 4 bytes are # of columns
    _read_bytes(12)  # unknown what these bytes are
    _read_bytes(2)  # should be 06 00 (encoding type)
    n_bytes_to_read = struct.unpack("I", _read_bytes(4))[0]
    _read_bytes(n_bytes_to_read)  # 'utf-8' string in my example files
    _read_bytes(2)  # should be 07 00 always?
    n_bytes_to_read = struct.unpack("I", _read_bytes(4))[0]
    _read_bytes(n_bytes_to_read)  # likely _jmpfile time stamp

    # read through a large chunk of the remaining header and should arrive
    # where column structures start to be described
    # temp_bytes info:
    # 04 00, 05 00, 0F 00 = unknown
    # 03 00 = all the scripts in the table
    # 02 00 = row state color information, most likely
    # 10 00 = likely more row state information, each row 8 bytes of info
    # 12 00 = string with version info
    # FF FF = when I read this, I am at the column info
    done = False
    while not done:
        temp_bytes = _read_bytes(2)  # describe what type of information follows (see above list)
        n_bytes_to_read = struct.unpack("I", _read_bytes(4))[0]  # number of bytes associated with this section
        # this is the temp_data associated with this section, but don't need to understand for now
        _read_bytes(n_bytes_to_read)  # read and discard
        if temp_bytes == bytearray.fromhex("FF FF"):
            done = True

    _read_bytes(2)  # should tell # of bytes used to give absolute offsets of column info
    count = 0
    while count < _n_columns:
        _abs_column_addresses.append(_read_bytes(4))  # may need to revisit on extremely large files - won't be 4 bytes?
        count += 1


# export _data (pandas dataframe) to CSV format
# input = file_name...what to write to
def to_csv(file_name):
    global _data
    _data.to_csv(file_name, index=False, encoding='utf_8_sig')


# decode all columns
# go through each column offset we figured out above, and then go and decode the data for every column
# then concatenate it all together into a final dataframe (_data)
def _decode_all_columns():
    global _data
    global _abs_column_addresses
    df_all = []
    for idx, i in enumerate(_abs_column_addresses):
        pd_series = _decode_column(i)
        df_all.append(pd.concat(pd_series, axis=1))

    # construct a concatenated final pandas dataframe
    _data = pd.concat(df_all, axis=1)


# - JMP stores date/time as a double, which is # of seconds since 1/1/1904 12:00:00 AM
# - have to handle blank entries specially - these show up as double value nan, which
# - is not handled well in python datetime functions
# - also handles duration
def _double_to_datetime(column_type, double_value):
    if math.isnan(double_value):  # special case where it's a nan
        dt_value = float('nan')
    # date/time of some type:
    elif column_type in _column_format_type_t_vals + _column_format_type_d_vals + \
            _column_format_type_dt_vals:
        dt_value = datetime.datetime(1904, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=double_value)
    else:  # it's a duration
        dt_value = datetime.timedelta(seconds=double_value)

    if column_type in _column_format_type_t_vals:
        return_type = _column_format_type_bucket["time"]
    elif column_type in _column_format_type_d_vals:
        return_type = _column_format_type_bucket["date"]
    elif column_type in _column_format_type_dur_vals:
        return_type = _column_format_type_bucket["duration"]
    else:
        return_type = _column_format_type_bucket["datetime"]  # full date/time formatting or NaN

    return dt_value, return_type  # return datetime/nan, and then what type (date, time, datetime, duration)


# given the byte in file related to the column format, return if column is a date/time/duration column
def _is_datetime_column(column_format):
    if column_format in _column_format_type_dt_vals + _column_format_type_t_vals + \
            _column_format_type_d_vals + _column_format_type_dur_vals:
        return True  # it is a datetime column
    else:
        return False  # it is not a datetime column


# decode a single column
# input is the file address that starts describing the column
def _decode_column(file_address):
    global _jmpfile
    int_offset = struct.unpack("I", file_address)[0]  # get _jmpfile position to seek as integer
    _jmpfile.seek(int_offset)  # seek to start of column name, first byte is the length of the string
    column_name_length = struct.unpack("B", _read_bytes(1))[0]  # length of column name
    column_name = _read_bytes(column_name_length).decode("utf-8")  # actual column name
    if column_name_length < 32:
        _read_bytes(31 - column_name_length)

    # data_type_dict = {1: "Numeric", 2: "Char", 3: "Row State", 4: "Large String?", 0xFF: "1-byte Integer",
    # 0xFE: "2-byte Integer", 0xFC: "4-byte Integer"}
    data_type = struct.unpack("B", _read_bytes(1))[0]  # numeric, char, row state, etc.

    # modeling_type_dict = {0: "Continuous", 1: "Ordinal", 2: "Nominal"}
    _read_bytes(1)  # modeling_type - read, but don't actually need to use

    # print column_name +" ["+data_type_dict[data_type]+", "+modeling_type_dict[modeling_type]+"]"
    # column_format_width:
    _read_bytes(1)  # for display purposes in JMP...don't need
    column_format_type = struct.unpack("B", _read_bytes(1))[0]
    n_data_bytes_per_row = struct.unpack("H", _read_bytes(2))[0]  # each row value takes up this many bytes
    # is_column_locked:
    _read_bytes(2)  # I think this is 2 bytes, but not 100% sure
    skip_number = struct.unpack("H", _read_bytes(2))[0]  # to help figure out how to skip to the actual row data
    _read_bytes(12)
    done = False
    skip_count = 1
    list_check = []
    is_list_check = False
    while (not done) and (skip_count <= skip_number - 1):
        skip_count += 1
        temp_bytes_val = struct.unpack("H", _read_bytes(2))[0]
        # 0x06 has to do with column hidden/exclude state, others I haven't delved into
        if (temp_bytes_val == 0x0C) or (temp_bytes_val == 0x0B) or (temp_bytes_val == 0x09) or \
                (temp_bytes_val == 0x13) or (temp_bytes_val == 0x06):
            field_length = struct.unpack("I", _read_bytes(4))[0]
            _read_bytes(field_length)
        # formula field, treat specially - basically read past it, but have to determine how much to read
        elif temp_bytes_val == 0x07:
            field_length = struct.unpack("I", _read_bytes(4))[0]
            _read_bytes(field_length)
        elif temp_bytes_val == 0x08:  # related to list check - enumerates possible values in terms of byte value
            num_vals = struct.unpack("I", _read_bytes(4))[0]
            _read_bytes(num_vals)  # finish reading past the field
        # range check is stored as 2 doubles (lower, upper of range), followed by 2 bytes
        # indicating range check rule
        elif temp_bytes_val == 0x05:  # indicates there is a range check on this column
            num_vals = struct.unpack("I", _read_bytes(4))[0]  # next 4 bytes indicate length of bytes to read
            _read_bytes(num_vals)  # finish reading past the field, do nothing with it
        elif temp_bytes_val == 0x10:  # related to a row-state column
            num_vals = struct.unpack("I", _read_bytes(4))[0]  # next 4 bytes indicate length of bytes to read
            _read_bytes(num_vals)  # finish reading past the field, do nothing with it
        elif temp_bytes_val == 0x01:  # related to notes
            num_vals = struct.unpack("I", _read_bytes(4))[0]  # next 4 bytes indicate length of bytes to read
            _read_bytes(num_vals)  # finish reading past the field, do nothing with it
        # specifically handle list check type attributes on columns
        elif temp_bytes_val == 0x04:  # List check field - lists out all options and then references them
            is_list_check = True
            field_length = struct.unpack("I", _read_bytes(4))[0]
            num_list_items = struct.unpack("H", _read_bytes(2))[0]
            record_length = (field_length - 2) / num_list_items
            if data_type == 1:  # numeric, each is stored as 8 bytes
                for i in range(num_list_items):
                    list_check.append(struct.unpack("d", _read_bytes(8))[0])
            elif (data_type == 2) or (data_type == 4):  # char
                for i in range(num_list_items):
                    if record_length - 1 < 256:  # this should normally be the case
                        str_length = struct.unpack("B", _read_bytes(1))[0]
                        list_check.append(_read_bytes(str_length))
                        # read out any excess bytes that aren't part of the string
                        _read_bytes(record_length - str_length - 1)
                    else:  # there are strings >=256 bytes, JMP actually has a bug and does not handle this!
                        _read_bytes(1)  # this string length is not correct - read & ignore
                        list_check.append(_read_bytes(record_length - 1).partition('\x00')[0].decode("utf-8"))
        elif temp_bytes_val == 0x0F:  # given column name is actually >255 bytes, so get correct column long name
                                      # overwrite value read above
            column_name_length = struct.unpack("I", _read_bytes(4))[0]
            column_name = _read_bytes(column_name_length).decode("utf-8")
        else:
            # didn't plan for this, raise an exception
            raise ValueError('Unhandled column field type = ' + "0x%0.2X" % temp_bytes_val)

    # time to read the actual data for every row!
    row_values = []
    dt_formatting = None  # None means it's not date/time, which is the default
    if data_type == 1:  # numeric, actually I just assume it is stored as 8-byte double
        if not is_list_check:
            for i in range(_n_rows):
                # handle date/time, time, date columns
                if _is_datetime_column(column_format_type):
                    double_value = struct.unpack("d", _read_bytes(n_data_bytes_per_row))[0]
                    dt_value, dt_formatting = _double_to_datetime(column_format_type, double_value)

                    row_values.append(dt_value)
                else:  # just treat as a number
                    row_values.append(struct.unpack("d", _read_bytes(n_data_bytes_per_row))[0])
        else:  # handle list check case
            for i in range(_n_rows):
                list_index = struct.unpack("B", _read_bytes(1))[0]
                if list_index != 0xFF:  # FF means the row is empty
                    if _is_datetime_column(column_format_type):  # handle datetime
                        dt_value, dt_formatting = _double_to_datetime(column_format_type, list_check[list_index])
                        row_values.append(dt_value)
                    else:  # treat as number
                        row_values.append(list_check[list_index])
                else:
                    row_values.append(float('nan'))
    elif (data_type == 2) or (data_type == 4):  # chars
        if not is_list_check:
            for i in range(_n_rows):
                # different formatting of strings - lead byte gives length of string 0x0100 is max you can have
                # here - 0xFF lead byte then 255 character string = 0x0100 field length
                if n_data_bytes_per_row <= 0x0100:
                    n_chars = struct.unpack("B", _read_bytes(1))[0]
                    row_values.append(_read_bytes(n_chars).decode("utf-8"))
                    # read excess characters that are in the buffer after the string, if present
                    _read_bytes(n_data_bytes_per_row - n_chars - 1)
                # takes long form where there is no byte before saying how long the string is, and instead string
                # is null terminated and buffered bytes up to full length of field are there
                else:
                    # read and then get rid of the null terminated string and excess bytes
                    row_values.append(_read_bytes(n_data_bytes_per_row).partition('\x00')[0].decode("utf-8"))
        else:  # this is a list check column, so has bytes refering to item in list
            for i in range(_n_rows):
                list_index = struct.unpack("B", _read_bytes(1))[0]
                if list_index != 0xFF:  # FF means the row is empty
                    row_values.append(list_check[list_index].decode("utf-8"))
                else:
                    row_values.append("")
    # row state column, should be 2 bytes per row, but don't check
    # for this purpose, I will just decode to 2-byte integer - not sure what else to do at this point,
    # could ultimately break out as some sort of string saying what the state actually is
    elif data_type == 0x03:
        for i in range(_n_rows):
            row_values.append(struct.unpack("H", _read_bytes(n_data_bytes_per_row))[0])
    # 1-byte signed integer, interestingly JMP does not use traditional range for 8-bit signed integer...
    # -126-127 is the range, with -127 representing "blank"
    elif data_type == 0xFF:
        for i in range(_n_rows):
            row_values.append(struct.unpack("b", _read_bytes(n_data_bytes_per_row))[0])
            if row_values[i] == -127:
                row_values[i] = ""
    elif data_type == 0xFE:  # 2-byte signed integer,-32767 represents blank entry
        for i in range(_n_rows):
            row_values.append(struct.unpack("h", _read_bytes(n_data_bytes_per_row))[0])
            if row_values[i] == -32767:
                row_values[i] = ""
    elif data_type == 0xFC:  # 4-byte signed integer, -2147483647 represents blank entry
        for i in range(_n_rows):
            row_values.append(struct.unpack("i", _read_bytes(n_data_bytes_per_row))[0])
            if row_values[i] == -2147483647:
                row_values[i] = ""

    # create pandas series with appropriate dtype if a date/datetime/time/duration
    if dt_formatting == _column_format_type_bucket["time"]:
        pd_series = [pd.Series(row_values, name=column_name, dtype=np.dtype('datetime64[ns]')).dt.time]
    elif dt_formatting == _column_format_type_bucket["date"]:
        pd_series = [pd.Series(row_values, name=column_name, dtype=np.dtype('datetime64[ns]')).dt.date]
    elif dt_formatting == _column_format_type_bucket["duration"]:
        pd_series = [pd.Series(row_values, name=column_name, dtype=np.dtype('timedelta64[ns]'))]
    elif dt_formatting == _column_format_type_bucket["datetime"]:
        pd_series = [pd.Series(row_values, name=column_name, dtype=np.dtype('datetime64[ns]'))]
    else:
        pd_series = [pd.Series(row_values, name=column_name)]

    return pd_series  # return the column as a pandas series

if __name__ == "__main__":
    '''

    # get all the files in the testing directory & create the list of where to write the CSV for each file
    file_list = []
    csv_out_list = []
    for infile in os.listdir('E:/All JMP Testing/'):
        file_list.append('E:/All JMP Testing/'+infile)
        csv_out_list.append(os.path.splitext('E:/All JMP Testing/CSV/'+infile)[0]+'.csv')

    # stats will hold the number of JMP files that had no error, or each of the 2 different error codes
    stats = {0: 0, -1: 0, -2: 0}

    # loop through every file, and if it ends in JMP, read it and write CSV
    # die instantly out of the loop if there is an unhandled error
    counter = 0
    for i in range(0, len(file_list)):
        if file_list[i].endswith('.jmp'):
            if counter % 50 == 0:
                print counter
            try:
                error_code, error_message, df = readjmp(file_list[i])
                if error_code == 0:
                    stats[0] += 1
                    to_csv(csv_out_list[i])
                    # print "No Error", counter, file_list[i], error_code, counter
                elif error_code == -1:
                    stats[-1] += 1
                    # print "***Info:", counter, file_list[i], error_code, counter
                else:
                    stats[-2] += 1
                    print "***Unknown:", counter, file_list[i], error_code, counter
            except:
                print "totally broken", file_list[i], counter
                break
        counter += 1

    print stats
    '''
