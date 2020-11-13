import re
IS_VECTOR = re.compile(r'(?<=/)[^/]*?(?=_)')
IS_DIMS = re.compile(r'\d\d(\d)?(?=\.json)')

def insert_row(df, row):
    df.loc[-1] = row
    df.index = df.index + 1
    df.sort_index()
    return df

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
