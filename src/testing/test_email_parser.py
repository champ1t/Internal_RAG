import re
import html

def extract_joomla_emails(content):
    """
    Decodes Joomla email obfuscation:
    var addy123 = 'part1' + 'part2';
    addy123 = addy123 + 'part3';
    """
    emails = []
    
    # regex to find the variable name used for the address
    # var addy99195 = '...';
    # patterns can vary, but generally:
    # var addyXXXX = '...';
    # addyXXXX = addyXXXX + '...';
    
    # Line-based parser for Joomla Cloaking
    # Logic:
    # 1. Look for 'var addy\d+ = ...'
    # 2. Look for 'addy\d+ = addy\d+ + ...'
    # 3. Stop when 'document.write' or end of script
    
    lines = content.splitlines()
    var_map = {} # val_name -> current_expr_string
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Match Init: var addy123 = '...
        m_init = re.search(r"var\s+(addy\d+)\s*=\s*(.*);", line)
        if m_init:
            var_name = m_init.group(1)
            expr = m_init.group(2)
            var_map[var_name] = expr
            continue
            
        # Match Add: addy123 = addy123 + ...
        # Regex: (addy\d+) \s*=\s* \1 \s*\+\s* (.*);
        m_add = re.search(r"(addy\d+)\s*=\s*\1\s*\+\s*(.*);", line)
        if m_add:
            var_name = m_add.group(1)
            added_expr = m_add.group(2)
            if var_name in var_map:
                var_map[var_name] += " + " + added_expr
                
    # Resolve all found variables
    final_emails = []
    for var, full_expr in var_map.items():
        # Extract quoted strings: 'str1' + 'str2'
        # Simple extraction of '...' or "..."
        parts = re.findall(r"['\"](.*?)['\"]", full_expr)
        combined = "".join(parts)
        decoded = html.unescape(combined)
        
        # Cleanup: sometimes it includes 'mailto:' prefix if the script puts it in the var
        # In this sample: document.write( '<a ' + path + '\'' + prefix + addy99195 ...
        # The var 'addy' itself usually holds the email.
        
        if "@" in decoded and "." in decoded:
            final_emails.append(decoded)
            
    return list(set(final_emails))

def test_parser():
    # Real example from file
    raw_sample = """
    <p style="text-align: center; ">e-mail : 
 <script language='JavaScript' type='text/javascript'>
 <!--
 var prefix = 'm&#97;&#105;lt&#111;:';
 var suffix = '';
 var attribs = '';
 var path = 'hr' + 'ef' + '=';
 var addy99195 = 's&#111;mb&#111;&#111;nc' + '&#64;';
 addy99195 = addy99195 + 'ntplc' + '&#46;' + 'c&#111;' + '&#46;' + 'th';
 document.write( '<a ' + path + '\'' + prefix + addy99195 + suffix + '\'' + attribs + '>' );
 document.write( addy99195 );
 document.write( '<\/a>' );
 //-->
 </script>
    """
    
    print("Parsing sample...")
    emails = extract_joomla_emails(raw_sample)
    print(f"Found: {emails}")
    
    if "<REDACTED_EMAIL>" in emails:
        print("SUCCESS: Decoded correctly.")
    else:
        print("FAILURE: Did not decode correctly.")

if __name__ == "__main__":
    test_parser()
