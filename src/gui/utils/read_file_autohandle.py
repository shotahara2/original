#fileの形式別の読み込み関数
#ファイルパスを入力するだけで拡張子を自動判定し、適切な方法（pd.read_csvなど）でデータフレームの型として読み込む
#csvデータの読み込み
def read_file_autohandle(input_path):
    '''
    入力ファイルから拡張子を自動判定してpandasに読み込む
    '''
    root, ext = os.path.splitext(input_path)  #拡張子の取得

    #拡張子毎に読み込みを実施
    if ext == ".csv":
        result = read_csv_data(input_path)
    elif ext == ".tsv":
        result = read_tsv_data(input_path)
    elif (ext == ".xls") or (ext == ".xlsx"):
        result = read_xls_data_firstseat(input_path)
    elif ext == ".json":
        result = read_json_data(input_path)
    return result

def read_csv_data(filename):
    '''
    csvデータを読みとる
    '''
    result = pd.read_csv(filename)
    result = result.drop('Unnamed: 0', axis=1)
    result.reset_index(drop=True, inplace=True)
    return result

def read_tsv_data(filename):
    '''
    tsvデータを読みとる
    '''
    result = pd.read_table(filename)
    return result

def read_xls_data_firstseat(filename):
    '''
    xlsx or xlsファイルの1枚目のシートを読みとる
    '''
    result = pd.read_excel(filename)
    return result

def read_xls_data_allsheat(filename):
    '''
    xlsx or xlsファイルの全てのシートを読み込む
    '''
    result = pd.read_excel(filename)
    return result

def read_json_data(filename):
    '''
    jsonファイルを読み込む
    '''
    result = pd.read_json(filename)
    return result
