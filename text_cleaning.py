PHP_TOKEN = '[XMLPHP_TOKEN]'
CODE_TOKEN = '[CODE_TOKEN]'

def replace_php_xml_code_to_tag(text):
    if '&lt;?' in text and '&gt;\n' in text:
        begin_idx = text.index('&lt;?')
        end_idx = text.rindex('&gt;\n')
        return text[:begin_idx] + ' ' + PHP_TOKEN + ' \n' + text[end_idx+4:]
    else:
        return text

def clear_newline_in_code_block(row_info_l):
    prev_indent_no = 0
    new_l = []
    for idx, (row, len_row, indent_nr, row_type) in enumerate(row_info_l):
        if prev_indent_no > 0 and indent_nr == 0 and len_row == 0:
            continue
        new_l.append([row, len_row, indent_nr, row_type])
        prev_indent_no = indent_nr
        
    return new_l
    
def replace_code_block(text):
    row_l = text.split('\n')
    row_info_l = [[row, len(row), len(row) - len(row.lstrip()),'NO_CODE'] for row in row_l]
    
    #row_info_l = clear_newline_in_code_block(row_info_l)

    prev_indent_no = 0
    code_start_flag = False
    for idx, (row, len_row, indent_nr, row_type) in enumerate(row_info_l):    
        if prev_indent_no == 0 and indent_nr > 0 and row_info_l[idx-1][3] == 'NO_CODE':
            row_info_l[idx-1][3] = 'CODE_BEGIN'
            row_info_l[idx][3] = 'CODE'
            code_start_flag = True

        # current row and next row are two \n\n
        if row_info_l[idx-1][3] == 'CODE' and indent_nr == 0 and idx < len(row_info_l)-1 and row_info_l[idx][1] == 0 and row_info_l[idx+1][1] == 0:
            row_info_l[idx-1][3] = 'CODE_END'
            code_start_flag = False

        if code_start_flag:
            row_info_l[idx][3] = 'CODE'
            if indent_nr == 0 and row_info_l[idx][1] == 0:
                row_info_l[idx][3] = 'CODE_NEWLINE'

        prev_indent_no = indent_nr

    proc_text = ''
    for row, len_row, indent_nr, row_type in row_info_l: 
        if row_type == 'CODE_BEGIN' and any(e in row for e in '=()[]{}&#@/_:'):
            proc_text += (' ' + CODE_TOKEN + ' \n')
            continue

        if (row_type == 'CODE_END' or row_type == 'CODE' or row_type == 'CODE_NEWLINE') and any(e in row for e in '=()[]{}&#@/_:'):
            continue 
            
        proc_text += row
        proc_text += '\n'        
            
    return proc_text