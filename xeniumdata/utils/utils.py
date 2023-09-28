class textformat:
    '''
    Helper class to format printed text.
    e.g. print(color.RED + color.BOLD + 'Hello, World!' + color.END)
    '''
    # colors and formats
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    # signs
    TSIGN = "\u251c"
    LSIGN = "\u2514"
    HLINE = "\u2500"
    RARROWHEAD = "\u27A4"
    
    # spacer
    SPACER = "    "

def remove_last_line_from_csv(filename):
    with open(filename) as myFile:
        lines = myFile.readlines()
        last_line = lines[len(lines)-1]
        lines[len(lines)-1] = last_line.rstrip()
    with open(filename, 'w') as myFile:    
        myFile.writelines(lines)
        
def decode_robust(s):
    try:
        return s.decode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        return s