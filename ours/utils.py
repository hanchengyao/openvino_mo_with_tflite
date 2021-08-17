# [Eason] file naming cannot contain invalid char
def valid_file_name_for_const_node(const_node_name):
    invalid_chars_in_name = ['/', ':', '*', '?', '<', '>']

    for invalid_char in invalid_chars_in_name:
        if invalid_char in const_node_name:
            const_node_name = '_'.join(const_node_name.split(invalid_char))

    return const_node_name

    