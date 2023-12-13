from lib_general import *

def _init_ml():
    '''import required module for machine learning'''
    from lib_ml import tf, TimeseriesGenerator, logging, h5py, tfio, Image
    global tf, TimeseriesGenerator, logging, h5py, tfio, Image
    tf.get_logger().setLevel(logging.ERROR)

def _init_calc():
    '''import required module for data fetching indicators calculation'''
    from lib_calc import talib, math, r2_score, Path, yf
    global talib, math, r2_score, Path, yf
    
def _init_candlestick_creation():
    '''import required module for candlestick dataset creation'''
    from lib_candlestick_creation import io, mpf, Image, ImageOps
    global io, mpf, Image, ImageOps
    
def remove_list_duplicate(x):
    if not isinstance(x, list):
        x = list(x)
    elif isinstance(x, list):
        pass
    return list(dict.fromkeys(x))
    
def update_world_stocks_database(api_key='2772aa38-3d67-4d12-9886-2349ac20fc3a', ROOT_PATH='./', target_filename='world_stocks_database.json', sleep=5):
    '''
    ## RUN WITH base.env (ipykernel) ##
    Fetch world stocks database ticker list
    based on StockSymbol database.
    
    More info about StockSymbol: https://github.com/yongghongg/stock-symbol
    
    This function save ss.market_list data with additional
    entry: symbolList.
    
    Result saved into save_path every loop.'''
    from stocksymbol import StockSymbol
    ss = StockSymbol(api_key)

    world_stocks = []
    market_lists = ss.market_list
    
    save_path = os.path.join(ROOT_PATH, target_filename)

    for i, market_list in enumerate(market_lists):
        market_location = market_list['market']
        print(f'{i}/{len(market_lists)}: {market_location}')
        symbol_list = ss.get_symbol_list(market=market_list['abbreviation'])
        market_list['symbolList'] = symbol_list
        world_stocks.append(market_list)

        # Save current report
        with open(save_path, 'w') as f:
            json.dump(world_stocks, f)

        time.sleep(sleep)
        
def get_abbreviation_list_from_world_stocks_database(ROOT_PATH='./', source_filename='world_stocks_database.json'):
    source_path = os.path.join(ROOT_PATH, source_filename)
    # Get world stocks database
    with open(source_path, 'r') as f:
        world_stocks_database = json.load(f)
    abbreviation_list = []
    for i, item in enumerate(world_stocks_database):
        abbreviation_list.append(item['abbreviation'])
    return abbreviation_list
        
def get_tickers_from_world_stocks_database(abbr, ROOT_PATH = './', source_filename='world_stocks_database.json', without_exclusions=False, remove_duplicate=True):
    '''
    Available abbreviations:
    ve venezuela
    za southafrica
    sg singapore
    th thailand
    tr turkey
    pt portugal
    qa qatar
    ru russia
    tw taiwan
    sa saudiarabia
    nz newzealand
    se sweden
    lv latvia
    pl poland
    nl netherlands
    my malaysia
    mx mexico
    no norway
    jp japan
    kr korea
    lt lithuania
    ie ireland
    lk srilanka
    il israel
    is iceland
    in india
    it italy
    gr greece
    gb unitedkingdom
    hu hungary
    hk hongkong
    es spain
    id indonesia
    fi finland
    eg egypt
    ch switzerland
    cz czechrepublic
    cn china
    dk denmark
    be belgium
    br brazil
    de germany
    ar argentina
    ca canada
    at austria
    au australia
    fr france
    us america
    '''
    def abbr_index(world_stocks_database):
        '''Create dict translation for
        abbreviation input to list index'''
        index_translation = {}
        for i, item in enumerate(world_stocks_database):
            index_translation[item['abbreviation']] = i
        return index_translation
    
    def get_tickers(world_stocks_database, index):
        tickers = []
        for ticker_entry in world_stocks_database[index]['symbolList']:
            tickers.append(ticker_entry['symbol'])
        return tickers

    source_path = os.path.join(ROOT_PATH, source_filename)
    # Get world stocks database
    with open(source_path, 'r') as f:
        world_stocks_database = json.load(f)

    index_translation = abbr_index(world_stocks_database)
    
    # Get tickers
    index = index_translation[abbr]
    tickers = get_tickers(world_stocks_database, index)
    tickers = remove_list_duplicate(tickers)
    if not without_exclusions:
        exclusions = get_exclusions(abbr, ROOT_PATH)
        tickers = [ticker for ticker in tickers if ticker not in exclusions]
        return tickers
    elif without_exclusions:
        return tickers

def get_exclusions(abbr, ROOT_PATH='./', folder='exclusions_list/'):
    '''Get tickers that caused error in the process,
    either already delisted, too few data, causing NaN
    during production, or any cause that might be updated
    future.'''
    base_path = os.path.join(ROOT_PATH, folder)
    path = os.path.join(base_path, f'{abbr}.json')
    if not os.path.exists(path):
        exclusions = []
        store_exclusions(abbr, exclusions, ROOT_PATH=ROOT_PATH, folder=folder)
    elif os.path.exists(path):
        with open(path, 'r') as f:
            exclusions = json.load(f)
    return exclusions

def store_exclusions(abbr, exclusions, ROOT_PATH='./', folder='exclusions_list/'):
    '''Store tickers that caused error in the process,
    either already delisted, too few data, causing NaN
    during production, or any cause that might be updated
    future.'''
    base_path = os.path.join(ROOT_PATH, folder)
    path = os.path.join(base_path, f'{abbr}.json')
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    elif os.path.exists(base_path):
        pass
    with open(path, 'w') as f:
        json.dump(exclusions, f)
    
def update_exclusions(abbr, additional_exclusion, ROOT_PATH='./', folder='exclusions_list/'):
    '''Update tickers that caused error in the process,
    either already delisted, too few data, causing NaN
    during production, or any cause that might be updated
    future.'''
    current_exclusions = get_exclusions(abbr, ROOT_PATH=ROOT_PATH, folder=folder)
    current_exclusions.append(additional_exclusion)
    store_exclusions(abbr, current_exclusions, ROOT_PATH=ROOT_PATH, folder=folder)
    
def clean_tickers_list(tickers, exclusions):
    '''Clean tickers list with passed
    exclusions.
    '''
    return [ticker for ticker in tickers if ticker not in exclusions]

def stock_list_100_highestrank_and_availability():
    '''Return a list of idx stock that have
    highest availability and most popular.
    Algorithm:
    - Get 150 stock with highest data availability
    - From stock filtered above, get 100 most popular
        stock (by relative idx volume rank)
    '''
    tickers = ['BBRI', 'BUMI', 'ELTY', 'TLKM', 'BRPT', 'LPKR', 'BKSL', 'BMRI', 'KLBF', 'BTEK', 'ASRI', 'KIJA', 'FREN', 'ANTM', 'ASII', 'ADRO', 'MEDC', 'BHIT', 'PWON', 'PNLF', 'BNII', 'PTBA', 'TINS', 'ELSA', 'MLPL', 'DOID', 'KREN', 'CNKO', 'CTRA', 'META', 'ENRG', 'SMRA', 'MDLN', 'BBNI', 'APIC', 'BMTR', 'LSIP', 'MAPI', 'BSDE', 'INDF', 'BBKP', 'MNCN', 'CPIN', 'WIKA', 'BNBR', 'SSIA', 'BNGA', 'BEKS', 'ADHI', 'RAJA', 'DILD', 'BBCA', 'PBRX', 'DGIK', 'PNBN', 'INDY', 'ACES', 'MYOR', 'INCO', 'AKRA', 'TBLA', 'KPIG', 'GZCO', 'TOTL', 'UNVR', 'INKP', 'SMCB', 'MPPA', 'GJTL', 'INTA', 'JSMR', 'HMSP', 'CMNP', 'MASA', 'SRSN', 'BNLI', 'INAF', 'RALS', 'ADMG', 'UNTR', 'SCMA', 'ISAT', 'DSFI', 'BDMN', 'MTDL', 'SULI', 'TURI', 'SMGR', 'TMAS', 'MAIN', 'KAEF', 'EXCL', 'SMSM', 'LPPS', 'POLY', 'KBLI', 'UNSP', 'PKPK', 'BUDI', 'CSAP']
    return tickers

def excluded_stock():
    '''Stocks that causes error in the process.
    '''
    # New list (use for inference & db update)
    tickers = ['HDTX','TRIO','CMPP','NIPS','SUGI','TRIL','APIC','MREI','PLIN','APII']
    # Old list (use for db_ver='4' dataset creation)
    # Add OKAS (caused error during yfinance fetch)
    tickers = ['HDTX','TRIO','CMPP','NIPS','SUGI','TRIL','APIC']
    return tickers

def us_excluded_stock(ROOT_PATH='./', db='us_raw.db', additional_exclusions=['ACLX','ADRT','AHRN','ALOR','ALSA','AOGO','APCA','APXI','ATEK','BFAC','BIOS','BLEU','BOCN','BPAC','BRCC','BRD','BRKH','CEG','CFFS','CFSB','CISO','CMCA','CNGL','CRDO','DAOO','DRCT','EMLD','EVE','FEXD','FGI','FRBN','GAQ','GNDR','GEEX','GFGD','GGAA','HAIA','HILS','HMA','HNRA','HORI','HTCR','IFIN','IGTA','IVCB','IVCP','KACL','KNSW','KSCP','LFAC','LGF.B','LSPR','MAAQ','MDV','MNTN','MTEK','MTVC','NFNT','NSTS','NVAC','NVCT','NVX','OLIT','PACI','PORT','PRLH','RCAC','RJAC','SAGA','SCUA','SHAP','SHEL','SKYH','SKYX','SSIC','STET','SUAC','SZZL','TCBP','TCOA','TGAA','TKLF','TLGY','TPG','UTAA','VHNA','VZLA','WEL','WTMA','XPDB','ZING','IDX']):
    '''Stocks that failed to fetch/delisted'''
    valid_tickers = us_db_stocklist(ROOT_PATH=ROOT_PATH, db=db)
    complete_tickers = us_stocklist()
    us_excluded_stock = [x for x in complete_tickers if x not in valid_tickers]
    for ae in additional_exclusions:
        if ae not in us_excluded_stock:
            us_excluded_stock.append(ae)
    return us_excluded_stock

def stock_metadata(ROOT_PATH='./'):
    '''Fetch available (traded)
    stock reference.
    '''
    # Read investing metadata
    metadata_conn = sqlite3.connect(f'{ROOT_PATH}investing_data.db', timeout=10)

    sql = f'select * from metadata'
    investing_metadata = pd.read_sql(sql, metadata_conn)
    return investing_metadata

def fetch_yfinance(ticker, suffix='.JK', period='max'):
    '''valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max'''
    stock_object = yf.Ticker(f'{ticker}{suffix}')
    df = stock_object.history(period=period)
    df = df.reset_index()
    df['Date'] = df['Date'].apply(lambda x: x.value)
    df = df.drop(['Dividends', 'Stock Splits'], axis='columns')
    df = df.rename(columns={'Date':'time','Open':'open','High':'high','Low':'low','Close':'close'})
    return df

## UNCOMMENT to run outside colab ##
# tickers=stock_metadata()['ticker'].values
def yfinance_db(stock_db='idx_raw.db', if_exists='replace', selected_stock_only=False, ROOT_PATH='./', tickers=[], selected_stock=stock_list_100_highestrank_and_availability(), suffix='.JK', period='max', sleep=1.5):
    '''Fetch data from yfinance and store
    it into defined stock_db.
    
    valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    '''
    stocks_db_conn = sqlite3.connect(f'{ROOT_PATH}{stock_db}')
    
    if selected_stock_only:
        tickers = selected_stock
    
    excluded_stock = []
    tick = datetime.datetime.now()
    for i, ticker in enumerate(tickers):           
        try:
            # Fetch and store data
            df = fetch_yfinance(ticker, suffix, period=period)
            df.to_sql(name=ticker, con=stocks_db_conn, index=False, if_exists=if_exists)
        except KeyError:
            excluded_stock.append(ticker)
        
        if i % 50 == 0:
            tock = datetime.datetime.now()
            print(f'{tock-tick} {i} {ticker}')
        else:
            tick = datetime.datetime.now()
        time.sleep(sleep)
    return excluded_stock

def fetch_yfinance_v2(ticker, period='max', timeout=25):
    '''valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max'''
    success = False
    attempt = 1
    while not success:
        try:
            stock_object = yf.Ticker(ticker)
            df = stock_object.history(period=period, timeout=timeout)
            success = True
        except requests.exceptions.Timeout:
            print(f'Trying to fetch yfinance data... Attempt {attempt}')
            time.sleep(10)
            attempt+=1
    df = df.reset_index()
    df['Date'] = df['Date'].apply(lambda x: x.value)
    df = df.drop(['Dividends', 'Stock Splits'], axis='columns')
    df = df.rename(columns={'Date':'time','Open':'open','High':'high','Low':'low','Close':'close'})
    return df

def yfinance_db_v2(abbr, if_exists='replace', ROOT_PATH='./', period='max', sleep=1.5, timeout=15):
    '''Fetch data from yfinance and store
    it into defined stock_db.
    
    valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    
    Update and streamlining exclusions to handle
    world stock database
    '''
    db_conn = sqlite3.connect(f'{ROOT_PATH}raw_db/{abbr}_raw.db')
    tickers = get_tickers_from_world_stocks_database(abbr, without_exclusions=True, ROOT_PATH=ROOT_PATH)
    tick = datetime.datetime.now()
    for i, ticker in enumerate(tickers):           
        try:
            # Fetch and store data
            df = fetch_yfinance_v2(ticker, period=period, timeout=timeout)
            df.to_sql(name=ticker, con=db_conn, index=False, if_exists=if_exists)
        except KeyError:
            update_exclusions(abbr, ticker, ROOT_PATH=ROOT_PATH)
        
        if i % 50 == 0:
            tock = datetime.datetime.now()
            print(f'Elapsed time for 50 requests: {tock-tick} {i}')
            tick = datetime.datetime.now()
        elif i % 50 != 0:
            pass
        time.sleep(sleep)
        
def update_world_stocks_database_rawdata(ROOT_PATH='./', period='max', sleep=1.5):
    '''Fetch all tickers data from all countries
    listed in world stocks database.
    
    Env: talib'''
    abbreviation_list = get_abbreviation_list_from_world_stocks_database()
    # Get data from world stock database
    for abbr in abbreviation_list:
        tick = datetime.datetime.now()
        print(f'Fetching {abbr} data from yfinance...')

        yfinance_db_v2(abbr, ROOT_PATH=ROOT_PATH, sleep=sleep, period=period)

        tock = datetime.datetime.now()
        print(f'Elapsed time: {tock-tick}')

def update_us_stocklist(tickers_path = './us_stocklist/us_stocklist.html', outfile_path='./us_stocklist.json'):
    '''Mine tickers list from https://stockanalysis.com/stocks/.
    Step to update list:
    1. Open "https://stockanalysis.com/stocks/"
    2. Save as html
    3. Locate html file and pass into `tickers_path` param
    4. Overwrite output file name if necessary `us_stocklist.json`
    5. Run this function
    '''
    with open(tickers_path, 'r') as f:
        html = f.read()

    soup = bs4.BeautifulSoup(html)
    symbol_table = soup.find_all(attrs={'class':'symbol-table'})[0]
    lists = symbol_table.tbody.find_all('tr')
    tickers = [entry.td.a.text for entry in lists]

    with open(outfile_path, 'w') as f:
        json.dump(tickers, f)
        
def us_stocklist(stocklist_path='./us_stocklist.json'):
    with open(stocklist_path, 'r') as f:
        tickers_from_json = json.load(f)
    return tickers_from_json

def us_db_stocklist(ROOT_PATH='./', db='us_raw.db'):
    '''Return us tickers tickers list
    that already fetched into db'''
    us_db_path = os.path.join(ROOT_PATH, db)
    us_db_conn = sqlite3.connect(us_db_path)
    fetched_us_stocks = tablename_list(us_db_conn)
    return fetched_us_stocks

def update_shenzhen_stocklist(outfile_path='./shenzhen_stocklist.json'):
    '''Fetch newest shenzen stocklist from
    https://en.m.wikipedia.org/wiki/List_of_companies_listed_on_the_Shenzhen_Stock_Exchange
    and store it to `shenzhen_stocklist.json`.
    '''
    def filter_tickers(table_rows):
        tickers = []
        for i, table_row in enumerate(table_rows):
            try:
                # Try to get td text attribute.
                # If not found and `AttributeError`
                # has thrown, its indicated that current
                # row is header/nothing/blank.
                ticker = table_row.td.text
            except AttributeError:
                continue

            try:
                # Try to convert value to 
                # make sure its not heading / blank row
                ticker_int = int(ticker)
                tickers.append(ticker)
            except ValueError:
                continue
        return tickers

    def wikitables_tickers(soup):
        wikitables = soup.find_all(attrs={'class':'wikitable'})
        tickers = []
        for i, wikitable in enumerate(wikitables):
            all_trs = wikitable.find_all('tr')
            filtered_tickers = filter_tickers(all_trs)
            tickers = tickers + filtered_tickers
        return tickers

    def last_table_tickers(soup):
        last_table = soup.find_all(attrs={'class':'mf-section-8'})
        last_table_rows = last_table[0].table.find_all('tr')
        tickers = filter_tickers(last_table_rows)    
        return tickers

    url = 'https://en.m.wikipedia.org/wiki/List_of_companies_listed_on_the_Shenzhen_Stock_Exchange'
    res = requests.get(url)
    soup = bs4.BeautifulSoup(res.text)
    
    wikitables_tickers_list = wikitables_tickers(soup)
    last_table_tickers_list = last_table_tickers(soup)
    tickers = wikitables_tickers_list + last_table_tickers_list
    
    # Save to json
    with open(outfile_path, 'w') as f:
        json.dump(tickers, f)
    
def shenzhen_stocklist(stocklist_path='./shenzhen_stocklist.json'):
    with open(stocklist_path, 'r') as f:
        tickers_from_json = json.load(f)
    return tickers_from_json
      
# stock_metadata()['ticker'].values
def calculate_stock_volume_contribution(origin_db, target_db, excluded_stock, selected_stock_only=False, ROOT_PATH='./', tickers=[], selected_stock=stock_list_100_highestrank_and_availability()):
    '''Calculate ratio of traded stock volume
    for each day compared to IDX total volume
    in each day.
    
    Store calculated ratio in new table 'IDX'
    '''
    origin_db_conn = sqlite3.connect(os.path.join(ROOT_PATH, origin_db))
    target_db_conn = sqlite3.connect(os.path.join(ROOT_PATH, target_db))
    
    # Delete all df variables that exists
    try:
        del df
    except NameError:
        pass
    
    if selected_stock_only:
        tickers = selected_stock
    
    # Clean ticker list
    tickers = [x for x in tickers if x not in excluded_stock]
        
    # Create cache db to speed up the process
    cache_db_conn = sqlite3.connect(':memory:')

    for i, ticker in enumerate(tickers):
        print(f'{i}/{len(tickers)}: {ticker}. Calculate total volume.')
        df1 = pd.read_sql(f'select time, Volume from `{ticker}`', origin_db_conn)

        try:
            df = df.merge(df1, how='outer')
            df = df.rename(columns={'Volume':ticker})
        except NameError:
            df = df1.copy(deep=True)
            df = df.rename(columns={'Volume':ticker})
            
        # Store current df to disk cache every some
        # period to avoid slowdown
        if i % 200 == 0:
            store_splitdf(df, primary_index='time', conn=cache_db_conn, max_column=75, append_existing_table=True)
            df = df[['time']]
        elif (i + 1) == len(tickers):
            store_splitdf(df, primary_index='time', conn=cache_db_conn, max_column=75, append_existing_table=True)
            df = df[['time']]
        else:
            pass
        
    # Restore cached dataframe
    df = restore_splitdf(cache_db_conn)

    # Calculate stock volume
    df_volume = df.copy(deep=True)
    df_volume = df.drop('time', axis='columns')
    df_volume = df_volume.fillna(0)
    df_volume['IDX'] = df_volume.sum(axis=1)

    # df_volume
    df['IDX'] = df_volume['IDX']
    df = df.sort_values(by='time')
    
    # Fill 0 in df['IDX'] with 1
    # to avoid zero division error
    df.loc[df['IDX'] < 1, 'IDX'] = 1

    # Calculate the stock volume relative to total volume
    for i, ticker in enumerate(tickers):
        print(f'{i}/{len(tickers)}: {ticker}. Calculate relative volume.')
        df[f'{ticker}_'] = df[ticker] / df['IDX']

    # Drop stock traded volume
    df = df.drop(tickers, axis='columns')
    
    # Drop row that has zero sum of volumes
    df = df[df.IDX != 0.0]

    # Save dataframe into database
    # If total columns more than 1000, store dataframe to
    # separate database as batches
    if len(df.columns) <= 1000:
        df.to_sql(name='IDX', con=target_db_conn, index=False, if_exists='replace')
    elif len(df.columns) > 1000:
        target_db_split = os.path.join(ROOT_PATH, f'{target_db}_volume.db')
        target_db_split_conn = sqlite3.connect(target_db_split)
        store_splitdf(df, primary_index='time', conn=target_db_split_conn, max_column=500)
        
def calculate_stock_volume_contribution_v2(tickers, origin_db_conn, target_db_conn, target_db_volume_conn, ROOT_PATH='./'):
    '''Calculate ratio of traded stock volume
    for each day compared to IDX total volume
    in each day.
    
    Store calculated ratio in new table 'IDX'
    '''
    # Create cache db to speed up the process
    cache_db_conn = sqlite3.connect(':memory:')

    for i, ticker in enumerate(tickers):
        print(f'{i}/{len(tickers)}: {ticker}. Calculate total volume.')
        df1 = pd.read_sql(f'select time, Volume from `{ticker}`', origin_db_conn)
        # REVISION: Change volume dtype to float32 to reduce memory usage
        df1['Volume'] = df1['Volume'].astype('float32')

        try:
            df = df.merge(df1, how='outer')
            df = df.rename(columns={'Volume':ticker})
        except NameError:
            df = df1.copy(deep=True)
            df = df.rename(columns={'Volume':ticker})
            
        # Store current df to disk cache every some
        # period to avoid slowdown
        if i % 128 == 0:
            store_splitdf(df, primary_index='time', conn=cache_db_conn, max_column=75, append_existing_table=True)
            df = df[['time']]
        elif (i + 1) == len(tickers):
            store_splitdf(df, primary_index='time', conn=cache_db_conn, max_column=75, append_existing_table=True)
            df = df[['time']]
        else:
            pass
        
    # Restore cached dataframe
    df = restore_splitdf(cache_db_conn)

    # Calculate stock volume
    df_volume = df.copy(deep=True)
    df_volume = df.drop('time', axis='columns')
    df_volume = df_volume.fillna(0)
    df_volume['IDX'] = df_volume.sum(axis=1)

    # df_volume
    df['IDX'] = df_volume['IDX']
    # REVISION: Change volume dtype to float32 to reduce memory usage
    df['IDX'] = df['IDX'].astype('float32')
    df = df.sort_values(by='time')
    
    # Fill 0 in df['IDX'] with 1
    # to avoid zero division error
    df.loc[df['IDX'] < 1, 'IDX'] = 1

    # Calculate the stock volume relative to total volume
    for i, ticker in enumerate(tickers):
        print(f'{i}/{len(tickers)}: {ticker}. Calculate relative volume.')
        df[f'{ticker}_'] = df[ticker] / df['IDX']
        # REVISION: Change volume dtype to float32 to reduce memory usage
        df[f'{ticker}_'] = df[f'{ticker}_'].astype('float32')

    # Drop stock traded volume
    df = df.drop(tickers, axis='columns')
    
    # Drop row that has zero sum of volumes
    df = df[df.IDX != 0.0]

    # Save dataframe into database
    # If total columns more than 1000, store dataframe to
    # separate database as batches
    if len(df.columns) <= 1000:
        df.to_sql(name='IDX', con=target_db_conn, index=False, if_exists='replace')
    elif len(df.columns) > 1000:
        store_splitdf(df, primary_index='time', conn=target_db_volume_conn, max_column=500)
        
def calculate_EMA(df, column, emas=(3,10,30,200)):
    for ema in emas:
        df[f'{column}_EMA{ema}'] = df[column].ewm(span=ema, adjust=False).mean()
    return df

# Old version
# calculate gradient from 4 samples
def calculate_EMA_gradient(df, column, emas=(3,10,30,200)):
    for ema in emas:
        multiplier = [1, 1/2, 3/4, 7/8]
        samples = [int(m * ema) for m in multiplier]
        sample_columns = []
        for i, sample in enumerate(samples):
            sample_column_name = f'{column}_EMA{ema}_G{i}'
            sample_columns.append(sample_column_name)
            df[sample_column_name] = (df[f'{column}_EMA{ema}'] - df[f'{column}_EMA{ema}'].shift(sample)) / ema
        df[f'{column}_EMA{ema}_G'] = df.loc[:, sample_columns].sum(axis=1) / len(samples)
        df = df.drop(sample_columns, axis='columns')
    return df

# Revised version
# Calculate gradient from tn-1 only
def calculate_EMA_gradient(df, column, emas=(3,10,30,200)):
    '''This versin have been revisited. 
    Using normalized gradient instead of sol difference between
    now and preceeding value.'''
    for ema in emas:
        df[f'{column}_EMA{ema}_G'] = (df[f'{column}_EMA{ema}'] - df[f'{column}_EMA{ema}'].shift(1)) / df[f'{column}_EMA{ema}'].shift(1)
    return df

def calculate_signal_EMA_offset(df, column, signal=3, emas=(10,30,200)):
    signal_column = f'{column}_EMA{signal}'
    for ema in emas:
        source_column = f'{column}_EMA{ema}'
        target_column = f'{column}_EMA{signal}_EMA{ema}_offset'
        df[target_column] = (df[signal_column] - df[source_column]) / df[source_column]
    return df

def calculate_candle_score(df, columns=('open','high','low','close','change')):
    open_c, high_c, low_c, close_c, change_c = columns
    candle_I0, candle_I1, candle_I2, candle_I3, candle_I4, candle_I5 = ('candle_I0', 'candle_I1', 'candle_I2', 'candle_I3', 'candle_I4', 'candle_I5')
    candle_S1, candle_S2, candle_S3, candle_S4 = ('candle_S1', 'candle_S2', 'candle_S3', 'candle_S4')
    
    # Candle body
    df[candle_I0] = df[open_c] - df[close_c]
    
    # Identify red / green candle status
    df[candle_I1] = np.select([df[candle_I0] < 0, df[candle_I0] > 0, df[candle_I0].isna()],
                              [-1, 1, np.nan], default=0)
    
    # High-low range, relative to close price
    df[candle_I2] = (df[high_c] - df[low_c]) / df[close_c]
    
    # Absolute relative body length to close price
    df[candle_I3] = (df[candle_I0] / df[close_c]).abs()
    
    # Body length / high-low range ratio
    df[candle_I4] = df[candle_I3] / df[candle_I2]
    
    # Candle body offset relative to high-low mean
    df[candle_I5] = ((((df[high_c] - df[low_c]) / 2) + 
                      ((df[close_c] - df[open_c]) / 2)) / 
                     ((df[high_c] - df[low_c]) / 2))
        
    # Score1: product of I1 * sum of I2-I5
    df[candle_S1] = df[candle_I1] * (df[candle_I2] + df[candle_I3] + df[candle_I4] + df[candle_I5])
    
    # Score2: product of I1 * average if I2-I5
    df[candle_S2] = df[candle_I1] * (df[candle_I2] + df[candle_I3] + df[candle_I4] + df[candle_I5]) / 4
    
    # Score3: product of I1-I5
    df[candle_S3] = df[candle_I1] * df[candle_I2] * df[candle_I3] * df[candle_I4] * df[candle_I5]
    
    # Score4: product of I1 * absolute of I2-I5 product
    df[candle_S4] = df[candle_I1] * (df[candle_I2] * df[candle_I3] * df[candle_I4] * df[candle_I5]).abs()
    
    return df
    
def calculate_favorite_stock(FAVORITE_STOCK, DB_PATH, threshold=20, column_groups=('day', 'month', 'year'), ROOT_PATH='./'):
    '''day-month-year favorite
    FAVORITE_STOCK: string, path to save the results.
    '''
    stocks_db_conn = sqlite3.connect(f'{ROOT_PATH}{DB_PATH}')
    df_IDX = pd.read_sql('select * from `IDX`', stocks_db_conn)

    # Convert integer timestamp into date
    df_IDX['time'] = pd.to_datetime(df_IDX['time'])

    # Make new column for year/month/day
    df_IDX.loc[:,'year'] = df_IDX['time'].dt.year
    df_IDX.loc[:,'month'] = df_IDX['time'].dt.month
    df_IDX.loc[:,'day'] = df_IDX['time'].dt.day

    # Drop time column to avoid interference to the rank
    columns_to_drop = ['time','IDX','day','month','year']
    for column_to_drop in columns_to_drop:
        if column_to_drop in df_IDX.columns:
            df_IDX_rank = df_IDX.drop([column_to_drop], axis='columns')
        elif column_to_drop not in df_IDX.columns:
            pass

    # Calculate rank
    df_IDX_rank = df_IDX_rank.rank(axis=1, ascending=True)

    # Normalize rank
    df_IDX_rank = df_IDX_rank.apply(lambda x: x / df_IDX_rank.count(axis=1))

    all_results = {}
    for column_group in column_groups:
        groups_result = {}
        for group_value in df_IDX[column_group].unique():
            tick = datetime.datetime.now()

            # Calculate rank for every specified range
            df_IDX_filtered = df_IDX.loc[df_IDX[column_group] == group_value]
            record_length = len(df_IDX_filtered)
            
            group_result = {}
            weight = np.arange(1, threshold+1)[::-1] / record_length

            # Loop through selected index
            for index in df_IDX_filtered.index:
                top_tickers = df_IDX_rank.iloc[index].sort_values(ascending=False)[:threshold].index


                for i, top_ticker in enumerate(top_tickers):
                    if top_ticker in group_result:
                        group_result[top_ticker] = group_result[top_ticker] + weight[i]
                    else:
                        group_result[top_ticker] = weight[i]

            groups_result[str(group_value)] = group_result

            tock = datetime.datetime.now()
            print(f'{str(column_group)} - {group_value}')

        all_results[column_group] = groups_result
        with open(FAVORITE_STOCK, 'w') as f:
            json.dump(all_results, f)
            
def calculate_favorite_stockv2(FAVORITE_STOCK, DB_PATH, threshold=20, main_group='year', sub_group='day', ROOT_PATH='./'):
    '''day-year / month-year
    FAVORITE_STOCK: string, path to save the results.
    '''
    stocks_db_conn = sqlite3.connect(f'{ROOT_PATH}{DB_PATH}')
    df_IDX = pd.read_sql('select * from `IDX`', stocks_db_conn)

    # Convert integer timestamp into date
    df_IDX['time'] = pd.to_datetime(df_IDX['time'])

    # Make new column for year/month/day
    df_IDX.loc[:,'year'] = df_IDX['time'].dt.year
    df_IDX.loc[:,'month'] = df_IDX['time'].dt.month
    df_IDX.loc[:,'day'] = df_IDX['time'].dt.day

    # Drop time column to avoid interference to the rank
    columns_to_drop = ['time','IDX','day','month','year']
    for column_to_drop in columns_to_drop:
        if column_to_drop in df_IDX.columns:
            df_IDX_rank = df_IDX.drop([column_to_drop], axis='columns')
        elif column_to_drop not in df_IDX.columns:
            pass

    # Calculate rank
    df_IDX_rank = df_IDX_rank.rank(axis=1, ascending=True)

    # Normalize rank
    df_IDX_rank = df_IDX_rank.apply(lambda x: x / df_IDX_rank.count(axis=1))

    groups_result = {}
    for main_value in df_IDX[main_group].unique():
        for sub_value in df_IDX[sub_group].unique():
            tick = datetime.datetime.now()

            # Calculate rank for every specified range
            df_IDX_filtered = df_IDX.loc[(df_IDX[main_group] == main_value) & (df_IDX[sub_group] == sub_value)]
            record_length = len(df_IDX_filtered)

            group_result = {}
            weight = np.arange(1, threshold+1)[::-1] / record_length

            # Loop through selected index
            for index in df_IDX_filtered.index:
                top_tickers = df_IDX_rank.iloc[index].sort_values(ascending=False)[:threshold].index

                for i, top_ticker in enumerate(top_tickers):
                    if top_ticker in group_result:
                        group_result[top_ticker] = group_result[top_ticker] + weight[i]
                    else:
                        group_result[top_ticker] = weight[i]

            groups_result[f"{main_value}_{f'0{sub_value}' if sub_value < 10 else f'{sub_value}'}"] = group_result

            tock = datetime.datetime.now()
            print(f'{main_value} - {sub_value}')

            with open(FAVORITE_STOCK, 'w') as f:
                json.dump(groups_result, f)
                
def calculate_horizontal_support_resistance(df, ticker, indicators):
    '''
    ticker: string
    indicator: dict
    '''
    dfhs = df.copy(deep=True)
    
    sr_lines = {}
    # Calculate horizontal support/resistance lines
    for indicator in indicators:
        # Stack multiple column to single column
        dfhs_sliced = dfhs[indicators[indicator]]
        try:
            bins = int(len(dfhs_sliced.stack()) / 10)

            # Calculate histogram intervals
            x = pd.cut(dfhs_sliced.stack(), bins).value_counts()
        except ValueError:
            bins = len(dfhs_sliced.stack())
            print('Value error: ', ticker, 'with data length: ', bins)

            # Calculate histogram intervals
            x = pd.cut(dfhs_sliced.stack(), bins).value_counts()

        # Calculate middle value of interval and store to new df
        mid_values = []
        for mid in x.index:
            mid_values.append((mid.mid, x[mid]))
        hist_pd = pd.DataFrame(mid_values)

        # Filter horizontal s/r lines that has less than mean frequency
        sr_line_df = hist_pd.loc[hist_pd[1] <= hist_pd[1].mean()].sort_values(by=0)
        sr_line_df = sr_line_df.reset_index(drop=True)

        # Get 10 s/r lines with same range between them
        sr_thresholds = np.linspace(0,1,10)
        calculated_sr = []
        for sr_threshold in sr_thresholds:
            calculated_sr.append(sr_line_df.iloc[int((len(sr_line_df) - 1) * sr_threshold)][0])
        sr_lines[indicator] = calculated_sr
    return sr_lines

def calculate_horizontal_support_resistance_v2(df, indicators):
    '''
    indicator: dict
    '''
    dfhs = df.copy(deep=True)
    
    sr_lines = {}
    # Calculate horizontal support/resistance lines
    for indicator in indicators:
        # Stack multiple column to single column
        dfhs_sliced = dfhs[indicators[indicator]]
        try:
            bins = int(len(dfhs_sliced.stack()) / 10)

            # Calculate histogram intervals
            x = pd.cut(dfhs_sliced.stack(), bins).value_counts()
        except ValueError:
            bins = len(dfhs_sliced.stack())
            print('Value error: with data length: ', bins)

            # Calculate histogram intervals
            x = pd.cut(dfhs_sliced.stack(), bins).value_counts()

        # Calculate middle value of interval and store to new df
        mid_values = []
        for mid in x.index:
            mid_values.append((mid.mid, x[mid]))
        hist_pd = pd.DataFrame(mid_values)

        # Filter horizontal s/r lines that has less than mean frequency
        sr_line_df = hist_pd.loc[hist_pd[1] <= hist_pd[1].mean()].sort_values(by=0)
        sr_line_df = sr_line_df.reset_index(drop=True)

        # Get 10 s/r lines with same range between them
        sr_thresholds = np.linspace(0,1,10)
        calculated_sr = []
        for sr_threshold in sr_thresholds:
            calculated_sr.append(sr_line_df.iloc[int((len(sr_line_df) - 1) * sr_threshold)][0])
        sr_lines[indicator] = calculated_sr
    return sr_lines

def __comp__calculate_stock_change_ratio(df, shift=1):
    '''Calculate stock change in %.
    Default shift value is 1, meaning that
    it's calculating daily stock price change.    
    '''
    df['change'] = (df['close'] - df['close'].shift(shift)) / df['close'].shift(shift)
    return df

def __comp__calculate_stock_volume_rank(df, df_IDX_rank, ticker):
    '''Calculate stock volume rank from `IDX` table
    and insert it into individual stock table
    '''
    df = df.merge(df_IDX_rank[['time', f'{ticker}_']], how='inner')
    df = df.rename(columns={f'{ticker}_':'Volume_rank'})
    return df

def __comp__calculate_stock_indicator(df):
    source_columns = ['close', 'rsi14', 'Volume', 'Volume_rank', 'change']
    df['rsi14'] = talib.RSI(df['close'], timeperiod=14)
    df = calculate_candle_score(df)
    for source_column in source_columns:
        df = calculate_EMA(df, source_column)
        df = calculate_EMA_gradient(df, source_column)
        df = calculate_signal_EMA_offset(df, source_column)
    return df

def __comp__calculate_oscillation_between_sr(df, origin_db_conn, ticker):
    indicators = {'close':['open','high','low','close'],'rsi14':['rsi14'],'Volume':['Volume'],'Volume_rank':['Volume_rank'],'change':['change']}
    sr_lines = calculate_horizontal_support_resistance(df, ticker, indicators)

    # Calculate indicator progress between interval
    for indicator in indicators:
        # Define condition and choice list
        condlist = [df[indicator] <= sr for sr in sr_lines[indicator][1:]]
        choicelist_t = [sr for sr in sr_lines[indicator][1:]]
        choicelist_b = [sr for sr in sr_lines[indicator][:len(sr_lines[indicator]) - 1]]
        df[f'{indicator}_b'] = np.select(condlist, choicelist_b, default=choicelist_b[-1])
        df[f'{indicator}_t'] = np.select(condlist, choicelist_t, default=choicelist_t[-1])

        # Calculate progress between interval.
        # *basically, just min/max norm between bottom/top interval
        df[f'{indicator}_srp'] = (df[indicator] - df[f'{indicator}_b']) / (df[f'{indicator}_t'] - df[f'{indicator}_b'])
        df = df.drop([f'{indicator}_b',f'{indicator}_t'], axis='columns')

    # Add close price relative to all time low / high
    df['close_rel'] = (df['close'] - df['close'].min()) / df['close'].max()
    return df

def __comp__calculate_oscillation_between_sr_v2(df):
    indicators = {'close':['open','high','low','close'],'rsi14':['rsi14'],'Volume':['Volume'],'Volume_rank':['Volume_rank'],'change':['change']}
    sr_lines = calculate_horizontal_support_resistance_v2(df, indicators)

    # Calculate indicator progress between interval
    for indicator in indicators:
        # Define condition and choice list
        condlist = [df[indicator] <= sr for sr in sr_lines[indicator][1:]]
        choicelist_t = [sr for sr in sr_lines[indicator][1:]]
        choicelist_b = [sr for sr in sr_lines[indicator][:len(sr_lines[indicator]) - 1]]
        df[f'{indicator}_b'] = np.select(condlist, choicelist_b, default=choicelist_b[-1])
        df[f'{indicator}_t'] = np.select(condlist, choicelist_t, default=choicelist_t[-1])

        # Calculate progress between interval.
        # *basically, just min/max norm between bottom/top interval
        df[f'{indicator}_srp'] = (df[indicator] - df[f'{indicator}_b']) / (df[f'{indicator}_t'] - df[f'{indicator}_b'])
        df = df.drop([f'{indicator}_b',f'{indicator}_t'], axis='columns')

    # Add close price relative to all time low / high
    df['close_rel'] = (df['close'] - df['close'].min()) / df['close'].max()
    return df

def __comp__calculate_cumulative_change(df):
    '''Calculate price change f_shift ahead from
    previous day.
    '''
    f_shifts=(3,5,7,10)
    b_shifts = [1 for _ in range(len(f_shifts))]
    for i, f_shift in enumerate(f_shifts):
        b_shift = b_shifts[i]
        df[f'change_b{b_shift}f{f_shift}'] = (df['close'].shift(-f_shift) - df['close'].shift(b_shift)) / df['close'].shift(b_shift)
    return df

def __comp__calculate_cumulative_change_v2(df, signal, successive):
    '''Calculate price change that satisfies passed params.
    Compared to v1:
    - One custom calculation per one function call
    - Follow algorithm in `gradient_difference_signal_indicator`
        for `signal` and `successive` parameters.
    - `signal` and `successive` abbreviated into `sig` and `suc`
    '''
    df[f'change_sig{signal}_suc{successive}'] = (df['close'].shift(-signal) - df['close'].shift(-successive)) / df['close'].shift(-successive)
    return df
    

def __comp__calculate_forecast_column(df):
    '''Calculate close_EMA3_G and close_EMA10_G
    at +1 and +2 forecast
    '''
    columns_to_forecast = ('close_EMA3_G','close_EMA10_G')
    forecast_lengths = (1,2)
    for column_to_forecast in columns_to_forecast:
        for forecast_length in forecast_lengths:
            df[f'{column_to_forecast}_s{forecast_length}'] = df[column_to_forecast].shift(-forecast_length)
    return df

def _calculate_stock_volume_rank(target_db):
    '''Component of calculate_stock_volume_rank    
    '''
    # Fetch portion to IHSG
    stocks_db_conn = sqlite3.connect(target_db)
    # Check where the whole stocks volume are stored
    standard_db_tablename_list = tablename_list(stocks_db_conn)
    if 'IDX' in standard_db_tablename_list:
        df_IDX = pd.read_sql(f'select * from `IDX`', stocks_db_conn)
    elif 'IDX' not in standard_db_tablename_list:
        extended_db_conn = sqlite3.connect(f'{target_db}_volume.db')
        df_IDX = restore_splitdf(extended_db_conn)        

    # Drop time column to avoid interference to the rank
    df_IDX_rank = df_IDX.drop(['time','IDX'], axis='columns')

    # Calculate rank
    df_IDX_rank = df_IDX_rank.rank(axis=1, ascending=True)

    # Normalize rank
    df_IDX_rank = df_IDX_rank.apply(lambda x: x / df_IDX_rank.count(axis=1))

    # Bring back time column
    df_IDX_rank.insert(0, 'time', df_IDX['time'])
    return df_IDX_rank

def _calculate_stock_volume_rank_v2(target_db_conn, target_db_volume_conn):
    '''Component of calculate_stock_volume_rank    
    '''
    # Check where the whole stocks volume are stored
    standard_db_tablename_list = tablename_list(target_db_conn)
    if 'IDX' in standard_db_tablename_list:
        df_IDX = pd.read_sql(f'select * from `IDX`', target_db_conn)
    elif 'IDX' not in standard_db_tablename_list:
        df_IDX = restore_splitdf(target_db_volume_conn)        

    # Drop time column to avoid interference to the rank
    df_IDX_rank = df_IDX.drop(['time','IDX'], axis='columns')

    # Calculate rank
    df_IDX_rank = df_IDX_rank.rank(axis=1, ascending=True)

    # Normalize rank
    df_IDX_rank = df_IDX_rank.apply(lambda x: x / df_IDX_rank.count(axis=1))

    # Bring back time column
    df_IDX_rank.insert(0, 'time', df_IDX['time'])
    return df_IDX_rank

# stock_metadata()['ticker'].values
def calculate_all_indicator(origin_db, target_db, excluded_stock, verbose=1, selected_stock_only=False, ROOT_PATH='./', tickers=[], selected_stock=stock_list_100_highestrank_and_availability()):
    origin_db_conn = sqlite3.connect(origin_db)
    target_db_conn = sqlite3.connect(target_db)
    
    # calculate_stock_volume_contribution
    # Calculate ratio of traded stock volume for
    # each day compared to IDX total volume each day
    calculate_stock_volume_contribution(origin_db, target_db, excluded_stock, selected_stock_only=selected_stock_only, ROOT_PATH=ROOT_PATH, tickers=tickers, selected_stock=selected_stock)
    
    # Fetch calculated IDX rank data frame
    df_IDX_rank = _calculate_stock_volume_rank(target_db)
    
    if selected_stock_only:
        tickers = selected_stock
        
    # Clean ticker list
    tickers = [x for x in tickers if x not in excluded_stock]
    
    for i, ticker in enumerate(tickers):
        # Read origin stock data
        df = pd.read_sql(f'select * from `{ticker}`', origin_db_conn)
        
        # Script below can be uncomment if future
        # error arise durint this function call.
        # # If df length less than 100, continue
        # if len(df) < 100:
        #     print(f'{ticker} has less than 20 records. Continue...')
        #     continue
        
        # Some indicator calculation #
        # calculate_stock_change_ratio
        df = __comp__calculate_stock_change_ratio(df, shift=1)
        
        # calculate_stock_volume_rank
        # Merge volume rank into individual stock df
        df = __comp__calculate_stock_volume_rank(df, df_IDX_rank, ticker)
        
        # calculate_stock_indicator
        # Rare error: `Exception: inputs are all NaN`
        # Happen in us stocks calculation. at 3000/6204.
        # Known no specific tickers. 
        # Try to catch thiss error instead
        try:
            df = __comp__calculate_stock_indicator(df)
        except Exception:
            print(f'Exception occured for ticker: {ticker}')
            continue
            
        # calculate_oscillation_between_sr
        # To some extent current error
        # handler failed to catch 
        # 'ValueError: `bins` should be a positive integer.'.
        # Try to catch from this function call instead.
        try:
            df = __comp__calculate_oscillation_between_sr_v2(df, origin_db_conn, ticker)
        except ValueError:
            print(f'Value error for ticker: {ticker}')
            continue
        
        # calculate_cumulative_change
        df = __comp__calculate_cumulative_change(df)
            
        # calculate_forecast_column
        df = __comp__calculate_forecast_column(df)     
        ##############################
        
        # Write back into table
        df.to_sql(name=ticker, con=target_db_conn, index=False, if_exists='replace')
        
        if verbose and i%50 == 0:
            print(f'Current progress: {i}/{len(tickers)}')
            
def calculate_all_indicators_v2(abbr, verbose=50, ROOT_PATH='./'):
    '''Calculate additional technical indicator
    and store result to /indicators_db folder.
    
    env: talib'''
    target_path = os.path.join(ROOT_PATH, 'indicators_db/')
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    elif os.path.exists(target_path):
        pass
    
    origin_db_path = os.path.join(ROOT_PATH, f'raw_db/{abbr}_raw.db')
    target_db_path = os.path.join(ROOT_PATH, f'indicators_db/{abbr}_indicators.db')
    target_db_volume_path = os.path.join(ROOT_PATH, f'indicators_db/{abbr}_indicators.db_volume.db')
    origin_db_conn = sqlite3.connect(origin_db_path)
    target_db_conn = sqlite3.connect(target_db_path)
    target_db_volume_conn = sqlite3.connect(target_db_volume_path)
    
    tickers = get_tickers_from_world_stocks_database(abbr, without_exclusions=False, ROOT_PATH=ROOT_PATH)
    
    # calculate_stock_volume_contribution
    # Calculate ratio of traded stock volume for
    # each day compared to IDX total volume each day
    calculate_stock_volume_contribution_v2(tickers, origin_db_conn, target_db_conn, target_db_volume_conn, ROOT_PATH=ROOT_PATH)
    
    # Fetch calculated IDX rank data frame
    df_IDX_rank = _calculate_stock_volume_rank_v2(target_db_conn, target_db_volume_conn)
    
    for i, ticker in enumerate(tickers):
        # Read origin stock data
        df = pd.read_sql(f'select * from `{ticker}`', origin_db_conn)
        
        # Script below can be uncomment if future
        # error arise durint this function call.
        # # If df length less than 100, continue
        # if len(df) < 100:
        #     print(f'{ticker} has less than 20 records. Continue...')
        #     continue
        
        # Some indicator calculation #
        # calculate_stock_change_ratio
        df = __comp__calculate_stock_change_ratio(df, shift=1)
        
        # calculate_stock_volume_rank
        # Merge volume rank into individual stock df
        df = __comp__calculate_stock_volume_rank(df, df_IDX_rank, ticker)
        
        # calculate_stock_indicator
        # Rare error: `Exception: inputs are all NaN`
        # Happen in us stocks calculation. at 3000/6204.
        # Known no specific tickers. 
        # Try to catch thiss error instead
        try:
            df = __comp__calculate_stock_indicator(df)
        except Exception:
            print(f'Exception occured for ticker: {ticker} during calculate_stock_indicator.')
            update_exclusions(abbr, ticker, ROOT_PATH=ROOT_PATH)
            continue
            
        # calculate_oscillation_between_sr
        # To some extent current error
        # handler failed to catch 
        # 'ValueError: `bins` should be a positive integer.'.
        # Try to catch from this function call instead.
        try:
            df = __comp__calculate_oscillation_between_sr_v2(df)
        except ValueError:
            print(f'Value error for ticker: {ticker} during calculate_oscillation_between_sr.')
            update_exclusions(abbr, ticker, ROOT_PATH=ROOT_PATH)
            continue
        
        # calculate_cumulative_change
        df = __comp__calculate_cumulative_change(df)
            
        # calculate_forecast_column
        df = __comp__calculate_forecast_column(df)     
        ##############################
        
        # Write back into table
        df.to_sql(name=ticker, con=target_db_conn, index=False, if_exists='replace')
        
        if verbose and i % verbose == 0:
            print(f'Current progress: {i}/{len(tickers)}')
            
# Additional indicators
# Excluded indicators (unclear variable definition)
# MAVP
# Failed with error code2: Bad Parameter (TA_BAD_PARAM): MAMA (f'{target_columns}_mama', f'{target_columns}_fama')

def _universal_indicators(df, target_columns, source_columns):
    close = df[target_columns]
    add_source_columns = [f'{target_columns}_bbands_upper', f'{target_columns}_bbands_middle', f'{target_columns}_bbands_lower', f'{target_columns}_dema', f'{target_columns}_ema', f'{target_columns}_ht_trendline', f'{target_columns}_kama', f'{target_columns}_real', f'{target_columns}_midpoint', f'{target_columns}_sma', f'{target_columns}_t3', f'{target_columns}_tema', f'{target_columns}_trima', f'{target_columns}_wma', f'{target_columns}_apo', f'{target_columns}_cmo', f'{target_columns}_macd', f'{target_columns}_macdsignal', f'{target_columns}_macdhist', f'{target_columns}_macd_ext', f'{target_columns}_macdsignal_ext', f'{target_columns}_macdhist_ext', f'{target_columns}_macd_fix', f'{target_columns}_macdsignal_fix', f'{target_columns}_macdhist_fix', f'{target_columns}_mom', f'{target_columns}_ppo', f'{target_columns}_roc', f'{target_columns}_rocp', f'{target_columns}_rocr', f'{target_columns}_rocr100', f'{target_columns}_rsi', f'{target_columns}_stochrsi_fastk', f'{target_columns}_stochrsi_fastd', f'{target_columns}_trix', f'{target_columns}_ht_dcperiod', f'{target_columns}_ht_dcphase', f'{target_columns}_ht_phasor_inphase', f'{target_columns}_ht_phasor_quadrature', f'{target_columns}_ht_sine_sine', f'{target_columns}_ht_sine_leadsine', f'{target_columns}_ht_trendmode']
    for i in add_source_columns: source_columns.append(i)
    # Overlap Studies Functions
    df[f'{target_columns}_bbands_upper'], df[f'{target_columns}_bbands_middle'], df[f'{target_columns}_bbands_lower'] = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df[f'{target_columns}_dema'] = talib.DEMA(close, timeperiod=30)
    df[f'{target_columns}_ema'] = talib.EMA(close, timeperiod=30)
    df[f'{target_columns}_ht_trendline'] = talib.HT_TRENDLINE(close)
    df[f'{target_columns}_kama'] = talib.KAMA(close, timeperiod=30)
    df[f'{target_columns}_real'] = talib.MA(close, timeperiod=30, matype=0)
    # df[f'{target_columns}_mama'], df[f'{target_columns}_fama'] = talib.MAMA(close, fastlimit=0, slowlimit=0)
    df[f'{target_columns}_midpoint'] = talib.MIDPOINT(close, timeperiod=14)
    df[f'{target_columns}_sma'] = talib.SMA(close, timeperiod=30)
    df[f'{target_columns}_t3'] = talib.T3(close, timeperiod=5, vfactor=0)
    df[f'{target_columns}_tema'] = talib.TEMA(close, timeperiod=30)
    df[f'{target_columns}_trima'] = talib.TRIMA(close, timeperiod=30)
    df[f'{target_columns}_wma'] = talib.WMA(close, timeperiod=30)
    
    # Momentum indicator functions
    df[f'{target_columns}_apo'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    df[f'{target_columns}_cmo'] = talib.CMO(close, timeperiod=14)
    df[f'{target_columns}_macd'], df[f'{target_columns}_macdsignal'], df[f'{target_columns}_macdhist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df[f'{target_columns}_macd_ext'], df[f'{target_columns}_macdsignal_ext'], df[f'{target_columns}_macdhist_ext'] = talib.MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    df[f'{target_columns}_macd_fix'], df[f'{target_columns}_macdsignal_fix'], df[f'{target_columns}_macdhist_fix'] = talib.MACDFIX(close, signalperiod=9)
    df[f'{target_columns}_mom'] = talib.MOM(close, timeperiod=10)
    df[f'{target_columns}_ppo'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
    df[f'{target_columns}_roc'] = talib.ROC(close, timeperiod=10)
    df[f'{target_columns}_rocp'] = talib.ROCP(close, timeperiod=10)
    df[f'{target_columns}_rocr'] = talib.ROCR(close, timeperiod=10)
    df[f'{target_columns}_rocr100'] = talib.ROCR100(close, timeperiod=10)
    df[f'{target_columns}_rsi'] = talib.RSI(close, timeperiod=14)
    df[f'{target_columns}_stochrsi_fastk'], df[f'{target_columns}_stochrsi_fastd'] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    df[f'{target_columns}_trix'] = talib.TRIX(close, timeperiod=30)
    
    # Cycle indicator functions
    df[f'{target_columns}_ht_dcperiod'] = talib.HT_DCPERIOD(close)
    df[f'{target_columns}_ht_dcphase'] = talib.HT_DCPHASE(close)
    df[f'{target_columns}_ht_phasor_inphase'], df[f'{target_columns}_ht_phasor_quadrature'] = talib.HT_PHASOR(close)
    df[f'{target_columns}_ht_sine_sine'], df[f'{target_columns}_ht_sine_leadsine'] = talib.HT_SINE(close)
    df[f'{target_columns}_ht_trendmode'] = talib.HT_TRENDMODE(close)
    return df, source_columns

def _price_indicators(df, source_columns):
    open = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['Volume']
    add_source_columns = ['midprice', 'sar', 'sarext', 'adx', 'adxr', 'aroondown', 'aroonup', 'aroonosc', 'bop', 'cci', 'dx', 'mfi', 'minus_di', 'minus_dm', 'plus_di', 'plus_dm', 'stoch_slowk', 'stoch_slowd', 'stochf_fastk', 'stochf_fastd', 'ultosc', 'willr', 'ad', 'adosc', 'obv', 'avgprice', 'medprice', 'typprice', 'wclprice', 'atr', 'natr', 'trange']
    for i in add_source_columns: source_columns.append(i)
    # Overlap Studies Functions
    df['midprice'] = talib.MIDPRICE(high, low, timeperiod=14)
    df['sar'] = talib.SAR(high, low, acceleration=0, maximum=0)
    df['sarext'] = talib.SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
    
    # Momentum indicator functions
    df['adx'] = talib.ADX(high, low, close, timeperiod=14)
    df['adxr'] = talib.ADXR(high, low, close, timeperiod=14)
    df['aroondown'], df['aroonup'] = talib.AROON(high, low, timeperiod=14)
    df['aroonosc'] = talib.AROONOSC(high, low, timeperiod=14)
    df['bop'] = talib.BOP(open, high, low, close)
    df['cci'] = talib.CCI(high, low, close, timeperiod=14)
    df['dx'] = talib.DX(high, low, close, timeperiod=14)
    df['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
    df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
    df['minus_dm'] = talib.MINUS_DM(high, low, timeperiod=14)
    df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    df['plus_dm'] = talib.PLUS_DM(high, low, timeperiod=14)
    df['stoch_slowk'], df['stoch_slowd'] = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['stochf_fastk'], df['stochf_fastd'] = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    df['ultosc'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df['willr'] = talib.WILLR(high, low, close, timeperiod=14)
    
    # Volume indicator functions
    df['ad'] = talib.AD(high, low, close, volume)
    df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    df['obv'] = talib.OBV(close, volume)
    
    # Price transform functions
    df['avgprice'] = talib.AVGPRICE(open, high, low, close)
    df['medprice'] = talib.MEDPRICE(high, low)
    df['typprice'] = talib.TYPPRICE(high, low, close)
    df['wclprice'] = talib.WCLPRICE(high, low, close)
    
    # Volatility indicator functions
    df['atr'] = talib.ATR(high, low, close, timeperiod=14)
    df['natr'] = talib.NATR(high, low, close, timeperiod=14)
    df['trange'] = talib.TRANGE(high, low, close)
    return df, source_columns

# Pattern recognition functions
def _pattern_recognition(df, source_columns):
    open = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    add_source_columns = ['cdl2crows', 'cdl3blackrows', 'cdl3inside', 'cdl3linestrike', 'cdl3outside', 'cdl3starsinsouth', 'cdl3whitesoldiers', 'cdlabandonedbaby', 'cdladvanceblock', 'cdlbelthold', 'cdlbreakaway', 'cdlclosingmarubozu', 'cdlconcealbabyswall', 'cdlcounterattack', 'cdldarkcloudcover', 'cdldoji', 'cdldojistar', 'cdldragonflydoji', 'cdlengulfing', 'cdleveningdojistar', 'cdleveningstar', 'cdlgapinsidewhite', 'cdlgravestonedoji', 'cdlhammer', 'cdlhangingman', 'cdlharami', 'cdlharamicross', 'cdlhighwave', 'cdlhikkake', 'cdlhikkakemod', 'cdlhomingpigeon', 'cdlidentical3crows', 'cdlinneck', 'cdlinvertedhammer', 'cdlkicking', 'cdlkickingbylength', 'cdlladderbottom', 'cdllongleggeddoji', 'cdllongline', 'cdlmarubozu', 'cdlmatchinglow', 'cdlmathold', 'cdlmorningdojistar', 'cdlmorningstar', 'cdlonneck', 'cdlpiercing', 'cdlrickshawman', 'cdlrisefall3methods', 'cdlseparatinglines', 'cdlshootingstar', 'cdlshortline', 'cdlspinningtop', 'cdlstalledpattern', 'cdlsticksandwich', 'cdltakuri', 'cdltasukigap', 'cdlthrusting', 'cdltristar', 'cdlunique3river', 'cdlupsidegap2crows', 'cdlxsidegap3methods']
    for i in add_source_columns: source_columns.append(i)
    df['cdl2crows'] = talib.CDL2CROWS(open, high, low, close)
    df['cdl3blackrows'] = talib.CDL3BLACKCROWS(open, high, low, close)
    df['cdl3inside'] = talib.CDL3INSIDE(open, high, low, close)
    df['cdl3linestrike'] = talib.CDL3LINESTRIKE(open, high, low, close)
    df['cdl3outside'] = talib.CDL3OUTSIDE(open, high, low, close)
    df['cdl3starsinsouth'] = talib.CDL3STARSINSOUTH(open, high, low, close)
    df['cdl3whitesoldiers'] = talib.CDL3WHITESOLDIERS(open, high, low, close)
    df['cdlabandonedbaby'] = talib.CDLABANDONEDBABY(open, high, low, close, penetration=0)
    df['cdladvanceblock'] = talib.CDLADVANCEBLOCK(open, high, low, close)
    df['cdlbelthold'] = talib.CDLBELTHOLD(open, high, low, close)
    df['cdlbreakaway'] = talib.CDLBREAKAWAY(open, high, low, close)
    df['cdlclosingmarubozu'] = talib.CDLCLOSINGMARUBOZU(open, high, low, close)
    df['cdlconcealbabyswall'] = talib.CDLCONCEALBABYSWALL(open, high, low, close)
    df['cdlcounterattack'] = talib.CDLCOUNTERATTACK(open, high, low, close)
    df['cdldarkcloudcover'] = talib.CDLDARKCLOUDCOVER(open, high, low, close, penetration=0)
    df['cdldoji'] = talib.CDLDOJI(open, high, low, close)
    df['cdldojistar'] = talib.CDLDOJISTAR(open, high, low, close)
    df['cdldragonflydoji'] = talib.CDLDRAGONFLYDOJI(open, high, low, close)
    df['cdlengulfing'] = talib.CDLENGULFING(open, high, low, close)
    df['cdleveningdojistar'] = talib.CDLEVENINGDOJISTAR(open, high, low, close, penetration=0)
    df['cdleveningstar'] = talib.CDLEVENINGSTAR(open, high, low, close, penetration=0)
    df['cdlgapinsidewhite'] = talib.CDLGAPSIDESIDEWHITE(open, high, low, close)
    df['cdlgravestonedoji'] = talib.CDLGRAVESTONEDOJI(open, high, low, close)
    df['cdlhammer'] = talib.CDLHAMMER(open, high, low, close)
    df['cdlhangingman'] = talib.CDLHANGINGMAN(open, high, low, close)
    df['cdlharami'] = talib.CDLHARAMI(open, high, low, close)
    df['cdlharamicross'] = talib.CDLHARAMICROSS(open, high, low, close)
    df['cdlhighwave'] = talib.CDLHIGHWAVE(open, high, low, close)
    df['cdlhikkake'] = talib.CDLHIKKAKE(open, high, low, close)
    df['cdlhikkakemod'] = talib.CDLHIKKAKEMOD(open, high, low, close)
    df['cdlhomingpigeon'] = talib.CDLHOMINGPIGEON(open, high, low, close)
    df['cdlidentical3crows'] = talib.CDLIDENTICAL3CROWS(open, high, low, close)
    df['cdlinneck'] = talib.CDLINNECK(open, high, low, close)
    df['cdlinvertedhammer'] = talib.CDLINVERTEDHAMMER(open, high, low, close)
    df['cdlkicking'] = talib.CDLKICKING(open, high, low, close)
    df['cdlkickingbylength'] = talib.CDLKICKINGBYLENGTH(open, high, low, close)
    df['cdlladderbottom'] = talib.CDLLADDERBOTTOM(open, high, low, close)
    df['cdllongleggeddoji'] = talib.CDLLONGLEGGEDDOJI(open, high, low, close)
    df['cdllongline'] = talib.CDLLONGLINE(open, high, low, close)
    df['cdlmarubozu'] = talib.CDLMARUBOZU(open, high, low, close)
    df['cdlmatchinglow'] = talib.CDLMATCHINGLOW(open, high, low, close)
    df['cdlmathold'] = talib.CDLMATHOLD(open, high, low, close, penetration=0)
    df['cdlmorningdojistar'] = talib.CDLMORNINGDOJISTAR(open, high, low, close, penetration=0)
    df['cdlmorningstar'] = talib.CDLMORNINGSTAR(open, high, low, close, penetration=0)
    df['cdlonneck'] = talib.CDLONNECK(open, high, low, close)
    df['cdlpiercing'] = talib.CDLPIERCING(open, high, low, close)
    df['cdlrickshawman'] = talib.CDLRICKSHAWMAN(open, high, low, close)
    df['cdlrisefall3methods'] = talib.CDLRISEFALL3METHODS(open, high, low, close)
    df['cdlseparatinglines'] = talib.CDLSEPARATINGLINES(open, high, low, close)
    df['cdlshootingstar'] = talib.CDLSHOOTINGSTAR(open, high, low, close)
    df['cdlshortline'] = talib.CDLSHORTLINE(open, high, low, close)
    df['cdlspinningtop'] = talib.CDLSPINNINGTOP(open, high, low, close)
    df['cdlstalledpattern'] = talib.CDLSTALLEDPATTERN(open, high, low, close)
    df['cdlsticksandwich'] = talib.CDLSTICKSANDWICH(open, high, low, close)
    df['cdltakuri'] = talib.CDLTAKURI(open, high, low, close)
    df['cdltasukigap'] = talib.CDLTASUKIGAP(open, high, low, close)
    df['cdlthrusting'] = talib.CDLTHRUSTING(open, high, low, close)
    df['cdltristar'] = talib.CDLTRISTAR(open, high, low, close)
    df['cdlunique3river'] = talib.CDLUNIQUE3RIVER(open, high, low, close)
    df['cdlupsidegap2crows'] = talib.CDLUPSIDEGAP2CROWS(open, high, low, close)
    df['cdlxsidegap3methods'] = talib.CDLXSIDEGAP3METHODS(open, high, low, close)
    return df, source_columns

def _indicator_derivatives(df, source_columns):
    # Calculate EMA, their gradient, and signal offset for every indicator
    for source_column in source_columns:
        df = calculate_EMA(df, source_column)
        df = calculate_EMA_gradient(df, source_column)
        df = calculate_signal_EMA_offset(df, source_column)
    return df

def calculate_talib_indicators_primary(df):
    '''Without gradient and their derivatives
    without candle pattern
    '''
    source_columns = []
    # Calculate universal_indicators for every target columns
    target_columns = ['open','high','low','close','Volume','change','Volume_rank']

    for target_column in target_columns:
        df, source_columns = _universal_indicators(df, target_column, source_columns)
    # Calculate price_indicators for price
    df, source_columns = _price_indicators(df, source_columns)
    return df, source_columns

def calculate_talib_indicators(df):
    source_columns = []
    # Calculate universal_indicators for every target columns
    target_columns = ['open','high','low','close','Volume','change','Volume_rank']

    for target_column in target_columns:
        df, source_columns = _universal_indicators(df, target_column, source_columns)
    # Calculate price_indicators for price
    df, source_columns = _price_indicators(df, source_columns)
    # Recognize candle pattern
    df, source_columns = _pattern_recognition(df, source_columns)

    # Calculate EMA, their gradient, and signal offset for every indicator
    df = _indicator_derivatives(df, source_columns)
    return df

def store_splitdf(df, primary_index, conn, max_column=500, append_existing_table=False):
    if append_existing_table:
        # Read current tablename order
        current_tablename_list = tablename_list(conn)
        if current_tablename_list != []:
            current_tablename_list = [int(x) for x in current_tablename_list]
            start_tablename_from = max(current_tablename_list) + 1
        elif current_tablename_list == []:
            start_tablename_from = 0
    elif not append_existing_table:
        start_tablename_from = 0
    
    column_length = len(df.columns)
    splits = math.ceil(column_length / max_column)
    for split in range(splits):
        columns = df.columns[max_column * split:max_column * (1 + split)]
        columns = np.insert(columns, 0, primary_index) if primary_index not in columns else columns
        temp_df = df[columns]
        temp_df.to_sql(str(start_tablename_from + split), conn, if_exists='replace', index=False)
    
    
def tablename_list(conn):
    cursor = conn.cursor()
    cursor.execute(f"select name from sqlite_master where type='table'")
    return [x[0] for x in cursor.fetchall()]

def restore_splitdf(conn):
    table_names = tablename_list(conn)
    first = True
    for table_name in table_names:
        temp_df = pd.read_sql(f'select * from `{table_name}`', conn)
        if first:
            df = temp_df.copy(deep=True)
            first = False
        elif not first:
            df = df.merge(temp_df)
    return df

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), 'df needs to be a pd.DataFrame'
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep]

def calculate_indicator_correlation(df):
    primary_index = 'time'
    # Standard score normalization
    df = (df - df.mean()) / df.std()

    columns = df.columns
    results = {}
    combinations = []
    count = 0
    tick = datetime.datetime.now()
    for i in columns:
        for j in columns:
            combination = [i,j]
            if (i == primary_index) or (j == primary_index) or (i == j) or (combination in combinations) or (combination[::-1] in combinations):
                continue
            combinations.append(combination)

            slice_df = clean_dataset(df[[i,j]])

            try:
                results[f'{i},{j}'] = r2_score(slice_df[i], slice_df[j])
            except ValueError:
                continue
        count+=1
    return results

def uncorrelated_indicators(results, minimum=20, maximum=30, interval=0.001, increment=0.00005):
    rdf = pd.DataFrame({'keys':results.keys(), 'values':results.values()})
    len_unique_columns = 0
    while (len_unique_columns < minimum) or (len_unique_columns > maximum):
        # Conditions
        if len_unique_columns < minimum:
            interval+=increment
        elif len_unique_columns > maximum:
            interval-=increment

        # Get slice
        zero_corr_df = rdf.loc[(rdf['values'] <= +interval) & (rdf['values'] >= -interval)]

        # Put value and avoid duplicate
        no_corr_combs = zero_corr_df['keys'].values
        unique_columns = []
        for no_corr_comb in no_corr_combs:
            a, b = no_corr_comb.split(',')
            if a not in unique_columns:
                unique_columns.append(a) 
            if b not in unique_columns:
                unique_columns.append(b) 
        len_unique_columns = len(unique_columns)
    return unique_columns

def calculate_uncorrelated_indicators_db_100stocks(ROOT_PATH='./', db_ver='v1', calculate_v1_indicators=True):
    '''Each DB contains 1 ticker data.
    - v1 indicators + additional primary indicators filtered using correlation coefficient
    - 20-30 most uncorrelated indicators and their derivatives stored into .db
    - close_EMA3 / close_EMA10 derivatives as required columns
    - Max column per table: {selected_stock_only}
    '''
    excluded_stock = excluded_stock()
    required_columns = ['close_EMA3_G', 'close_EMA10_G', 'close_EMA3_G_s1', 'close_EMA3_G_s2', 'close_EMA10_G_s1', 'close_EMA10_G_s2']
    primary_index = 'time'
    max_column_per_table = 250
    selected_stock_only = False

    DB_ROOT_PATH = f'{ROOT_PATH}db/{db_ver}/'
    Path(DB_ROOT_PATH).mkdir(parents=True, exist_ok=True)
    db_readme = f'''DB {db_ver}

    Each DB contains 1 ticker data.
    - v1 indicators + additional primary indicators filtered using correlation coefficient
    - 20-30 most uncorrelated indicators and their derivatives stored into .db
    - close_EMA3 / close_EMA10 derivatives as required columns
    - Max column per table: {selected_stock_only}
    '''
    with open(f'{DB_ROOT_PATH}readme.txt', 'w') as f:
        f.write(db_readme)

    # New flow
    origin_db = f'{ROOT_PATH}idx_raw.db'
    v1_db = f'{ROOT_PATH}idx_indicators.db'
    v2_db_placeholder = '{}idx_{}.db'

    origin_db_conn = sqlite3.connect(origin_db)
    v1_db_conn = sqlite3.connect(v1_db)

    # Calculate v1 indicators
    if calculate_v1_indicators:
        calculate_all_indicator(origin_db, v1_db, excluded_stock, verbose=1, selected_stock_only=selected_stock_only, ROOT_PATH=ROOT_PATH)
    elif not calculate_v1_indicators:
        pass

    tickers = stock_list_100_highestrank_and_availability()

    # Try to resume from previous progress if any
    try:
        with open(f'{DB_ROOT_PATH}progress.cache', 'r') as f:
            hotstart = int(f.read())
    except FileNotFoundError:
        hotstart = 0

    count = 0
    for ticker in tickers:
        if count < hotstart:
            count+=1
            continue
        if ticker in excluded_stock:
            continue
        tick = datetime.datetime.now()
        v2_db = v2_db_placeholder.format(DB_ROOT_PATH, ticker)
        v2_db_conn = sqlite3.connect(v2_db)

        v1_df = pd.read_sql(f'select * from `{ticker}`', v1_db_conn)

        # Calculate primary indicators
        prim_df, source_columns = calculate_talib_indicators_primary(v1_df)

        # Calculate indicator correlation
        results = calculate_indicator_correlation(prim_df)

        # Eliminate most uncorrelated indicators (20-30 indicators)
        unique_columns = uncorrelated_indicators(results)

        # Check if required_columns are in the unique_columns list, else, append those required_columns
        # close_EMA3_G, close_EMA10_G, close_EMA3_G_s1, close_EMA3_G_s2, close_EMA10_G_s1, close_EMA10_G_s2 are in the list
        for required_column in required_columns:
            if required_column not in unique_columns:
                unique_columns.append(required_column)

        # Add primary_index column at the beginning
        unique_columns.insert(0, primary_index)

        # Add pattern recognition
        prim_df, unique_columns = _pattern_recognition(prim_df, unique_columns)

        # Slice dataframe by unique_columns
        unique_df = prim_df[unique_columns]

        # Calculate indicator derivatives for additional calculated indicator in source_columns
        derivatives_columns = []
        for unique_column in unique_columns:
            if unique_column in source_columns:
                derivatives_columns.append(unique_column)
        derivatives_df = _indicator_derivatives(unique_df, derivatives_columns)

        # Store calculated df as separated .db for each stock
        # with 0->inf. as table name for specified split value
        store_splitdf(derivatives_df, primary_index, v2_db_conn, max_column=max_column_per_table)

        count+=1
        tock = datetime.datetime.now()
        print(f'{count}/{len(tickers)}: {ticker} -- Elapsed time: {tock-tick}')
        with open(f'{DB_ROOT_PATH}progress.cache', 'w') as f:
            f.write(str(count))
            
def load_ml_database(DB_PATH, columns_to_drop=['time']):
    '''Instead of regular read_sql,
    the ml-specialized df need to fill
    null value and drop unnecessary column
    '''
    db_conn = sqlite3.connect(DB_PATH)
    df = restore_splitdf(db_conn)
    df = df.fillna(0)
    for column_to_drop in columns_to_drop:
        try:
            df = df.drop(column_to_drop, axis='columns')
        except ValueError:
            pass
    return df

def input_c_from_sa_v2(ROOT_PATH, model_version, ticker, output_c):
    '''Fetch input_c list from
    sensitivity analysis file
    
    Revision from v1:
    - Revising path to grouping based on 
        simulation version.
    '''
    filename = f'{ROOT_PATH}statistics/v{model_version}/sa_{ticker}_{output_c}.json'

    with open(filename, 'r') as f:
        sa_dict = json.load(f)

    final_column = sorted(sa_dict.items(), key=lambda x: x[1])[0][0]

    input_c = []
    for sa_key in sa_dict.keys():
        input_c.append(sa_key)
        if sa_key == final_column:
            break
    return input_c

def split_traintest(df, split=0.8, normalized=True):
    '''Splif df with train fraction.
    Normalize data using train set mean-std normalization.
    Return normalized train-test set, mean, and std.
    '''
    # Train/test split
    train_df = df[:int(split*len(df))]
    test_df = df[int(split*len(df)):]
    
    # reset index if and only if part of pd.DataFrame
    # instance
    if isinstance(test_df, pd.DataFrame):
        test_df.reset_index(inplace=True, drop=True)    
    
    if not normalized:
        return train_df, test_df
    elif normalized:
        train_mean = train_df.mean()
        train_std = train_df.std()

        # Normalize dataset
        train_df = (train_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std
        return train_df, test_df, train_mean, train_std

def model_compiler(model):
    '''Adam optimizer, mse loss, RMSE metrics
    '''
    model.compile(optimizer='adam',
                  loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def simpleRNN_1layer(units, return_sequences=False, create_input_layer=False, shape=None):
    if create_input_layer:
        model = tf.keras.models.Sequential([
            tf.keras.Input(shape),
            tf.keras.layers.SimpleRNN(units, return_sequences=return_sequences),
            tf.keras.layers.Dense(1),
        ])
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.SimpleRNN(units, return_sequences=return_sequences),
            tf.keras.layers.Dense(1),
        ])
    return model_compiler(model)

def GRU_1layer(units, return_sequences=False, create_input_layer=False, shape=None):
    if create_input_layer:
        model = tf.keras.models.Sequential([
            tf.keras.Input(shape),
            tf.keras.layers.GRU(units, return_sequences=return_sequences),
            tf.keras.layers.Dense(1),
        ])
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.GRU(units, return_sequences=return_sequences),
            tf.keras.layers.Dense(1),
        ])
    return model_compiler(model)

def LSTM_1layer(units, return_sequences=False, create_input_layer=False, shape=None):
    if create_input_layer:
        model = tf.keras.models.Sequential([
            tf.keras.Input(shape),
            tf.keras.layers.LSTM(units, return_sequences=return_sequences),
            tf.keras.layers.Dense(1),
        ])
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(units, return_sequences=return_sequences),
            tf.keras.layers.Dense(1),
        ])
    return model_compiler(model)

def read_optimal_model_config(statistics_file_path, indicator='test_rmse'):
    '''Read most optimal architecture, recurrent
    lstmu, and epoch from optimization simulation.
    '''
    columns = ('ticker','input_c','output_c','architecture','recurrent','lstmu','epoch','train_mse','train_rmse','test_mse','test_rmse')
    sa_DF = pd.read_csv(statistics_file_path, header=0, names=columns)
    best_config = sa_DF.sort_values(by='test_rmse').values[0]
    return best_config[3], best_config[4], best_config[5], best_config[6]

def make_required_dir(ROOT_PATH, model_version):
    paths = [f'{ROOT_PATH}statistics/v{model_version}/', f'{ROOT_PATH}progress_cache/v{model_version}/', f'{ROOT_PATH}statistics/v{model_version}/backtest/', f'{ROOT_PATH}statistics/v{model_version}/production/']
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

def execute_input_c_v2(ticker, ROOT_PATH, output_c='close_EMA3_G_s1', model_version='2', input_c_limit=25, epochs=(15,), rnnus=(14,), recurrents=(7,), train=0.8):
    '''Performing sensitivity input_c sensitivity analysis
    that satisfies passed parameters
    
    Revision from v1:
    - fixing results/temporary directory confusion
        when running a new version. Grouping results by
        `model_version` params.
    - Fixing output_c temporary fixing that previously
        think that the original EMAx_G is already a +1
        offset value.
    - Replace data loading with `restore_splitdf` function
    
    Important directories
    ./db -> source stock database
    ./statistics -> model performance histories
    '''
    DB_PATH = f'{ROOT_PATH}db/v{model_version}/idx_{ticker}.db'
    STATISTICS_FILE = f'{ROOT_PATH}statistics/v{model_version}/sa_{ticker}_{output_c}.csv'
    
    # Excluded columns
    excluded_columns_res = ('change_b.f.', '_s[0-9]*')

    # Load data
    df = load_ml_database(DB_PATH)
    
    # Split train-test
    train_df, test_df, train_mean, train_std = split_traintest(df, split=train)

    # Convert all possible input column into dictionary
    # as `additional_c`
    additional_c = [x for x in df.columns]

    # Delete columns that satisfy `excluded_columns_re`
    for excluded_columns_re in excluded_columns_res:
        additional_c = [x for x in additional_c if not re.search(excluded_columns_re, x)]

    # Replace csv with plain statistics file name
    sa_statistics = STATISTICS_FILE.replace('.csv', '')

    try:
        # Fetch `input_s` as hotstart
        # Loop through `additional_c`, and delete keys that have been simulated before
        # in `additional_c`. Using condition in `input_s.keys()`
        with open(f'{sa_statistics}.json', 'r') as f:
            input_s = json.load(f)

        # Delete `additional_c` that have been simulated before
        additional_c = [x for x in additional_c if x not in input_s.keys()]

    except FileNotFoundError:
        input_s = {}

    # Do a simulation as long as there are more `additional_c` to simulate
    # Loop through `additional_c` everytime, until no more left.
    # Better executed with while
    while len(additional_c) > 0 and len(input_s.keys()) < input_c_limit:
        # Use `additional_c_results` dictionary as `additional_c` sim results control. 
        # the format: {add_c: RMSE_performance}
        additional_c_results = {}

        current_ac_loop = 1
        for ac in additional_c:
            # Add additional simulated column into previous input order
            # input_c = [x for x in input_s.keys()]
            # input_c.append(ac)
            # This `input_c` resetted in every `additional_c` loop

            # Configure input column by adding +1 column to a previous
            # configuration to see how much the performance increasing
            input_c = [x for x in input_s.keys()]
            input_c.append(ac)

            for recurrent in recurrents:
                for rnnu in rnnus:
                    for epoch in epochs:
                        tick = datetime.datetime.now()

                        # Make dataset
                        train_dataset = TimeseriesGenerator(train_df[input_c], train_df[output_c], length=recurrent, shuffle=True)
                        test_dataset = TimeseriesGenerator(test_df[input_c], test_df[output_c], length=recurrent)

                        # RNN Model
                        model = GRU_1layer(rnnu, return_sequences=False)
                        
                        # Feed and train the model
                        model.fit(train_dataset, epochs=epoch, verbose=0)

                        # Evaluate model
                        train_eval = model.evaluate(train_dataset, verbose=0)
                        test_eval = model.evaluate(test_dataset, verbose=0)

                        input_c_joined = '|'.join(input_c)
                        statistics = f'''{ticker},{input_c_joined},{output_c},{recurrent},{rnnu},{epoch},{train_eval[0]},{train_eval[1]},{test_eval[0]},{test_eval[1]}\n'''
                        with open(STATISTICS_FILE, 'a') as f:
                            f.write(statistics)

                        tock = datetime.datetime.now()

                        print(f'''{ticker} i[{len(input_c)}] o[{output_c}] acleft[{current_ac_loop}/{len(additional_c)}] aic[{ac}] R[{recurrent}] RU[{rnnu}] E[{epoch}] trainRMSE[{round(train_eval[1], 4)}] testRMSE[{round(test_eval[1], 4)}] time: {tock-tick}''')
                        current_ac_loop+=1

                        # Add testRMSE into `additional_c_results`
                        additional_c_results[ac] = test_eval[1]

        # Everytime the loop through `additional_c` finish,
        # sort the `additional_c_results` using
        # max_res = sorted(additional_c_results.items(), keys=lambda x: x[1])
        max_res = sorted(additional_c_results.items(), key=lambda x: x[1])

        # Fetch max_res[0] to `input_s`
        # input_s[max_res[0][0]] = max_res[0][1]
        input_s[max_res[0][0]] = max_res[0][1]

        # Everytime the SA period finish, the most sensitive
        # additional column deleted from `additional_c`
        additional_c.remove(max_res[0][0])

        print(f'=== MOST SENSITIVE: {max_res[0][0]} with combined RMSE: {max_res[0][1]} ===')

        # Save current `input_s` to file
        with open(f'{sa_statistics}.json', 'w') as f:
            json.dump(input_s, f)
            
def execute_configuration_optimization_v2(ticker, ROOT_PATH, output_c='close_EMA3_G_s1', model_version='2', epochs=(10,20,40), rnnus=(4,8,12,16), recurrents=(5,7,10,15,20), train=0.8):
    '''
    Differences with v1:
    - remove offset parameter, integerating it with 'correct'
        output_c instead of manipulating it on the go. 
    '''
    DB_PATH = f'{ROOT_PATH}db/v{model_version}/idx_{ticker}.db'
    STATISTICS_FILE = f'{ROOT_PATH}statistics/v{model_version}/sa_{ticker}_{output_c}.csv'
    input_c = input_c_from_sa_v2(ROOT_PATH, model_version, ticker, output_c)
    architectures = ('SimpleRNN', 'GRU', 'LSTM')

    # Load data
    df = load_ml_database(DB_PATH)
    
    # Split and normalize train-test set
    train_df, test_df, train_mean, train_std = split_traintest(df, split=train)
    
    # Create progress cache
    progress_cache_file = f'{ROOT_PATH}progress_cache/v{model_version}/opt_{ticker}_{output_c}.txt'
    try:
        with open(progress_cache_file, 'r') as f:
            HOTSTART = int(f.read())
    except FileNotFoundError:
        HOTSTART = 0

    loops = 0
    for architecture in architectures:
        for recurrent in recurrents:
            for rnnu in rnnus:
                for epoch in epochs:
                    tick = datetime.datetime.now()
                    # Hotstart
                    if loops < HOTSTART:
                        loops+=1
                        continue

                    # Make dataset
                    train_dataset = TimeseriesGenerator(train_df[input_c], train_df[output_c], length=recurrent, shuffle=True)
                    test_dataset = TimeseriesGenerator(test_df[input_c], test_df[output_c], length=recurrent)

                    # RNN Model
                    if architecture == 'SimpleRNN':
                        model = simpleRNN_1layer(rnnu, return_sequences=False)
                    elif architecture == 'GRU':
                        model = GRU_1layer(rnnu, return_sequences=False)
                    elif architecture == 'LSTM':
                        model = LSTM_1layer(rnnu, return_sequences=False)

                    # Feed and train the model
                    model.fit(train_dataset, epochs=epoch, verbose=0)

                    # Evaluate model
                    train_eval = model.evaluate(train_dataset, verbose=0)
                    test_eval = model.evaluate(test_dataset, verbose=0)

                    input_c_joined = '|'.join(input_c)
                    statistics = f'''{ticker},{input_c_joined},{output_c},{architecture},{recurrent},{rnnu},{epoch},{train_eval[0]},{train_eval[1]},{test_eval[0]},{test_eval[1]}\n'''
                    with open(STATISTICS_FILE, 'a') as f:
                        f.write(statistics)

                    tock = datetime.datetime.now()

                    print(f'''{loops} {ticker} {len(input_c)} {architecture} o[{output_c}] R[{recurrent}] RU[{rnnu}] E[{epoch}] trainRMSE[{round(train_eval[1], 5)}] testRMSE[{round(test_eval[1], 5)}] time: {tock-tick}''')

                    loops+=1
                    # Save progress to cache
                    with open(progress_cache_file, 'w') as f:
                        f.write(str(loops))
            
def execute_batch_v2(simtype, offset, instance_no, ROOT_PATH, split=5, output_cs=('close_EMA3_G_s1','close_EMA10_G_s1'), model_version='2', input_c_limit=25, epochs=(10,), rnnus=(14,), recurrents=(5,), train=0.8):
    '''
    type: string
        input_c
        opt
        
    Revision compared to v1:
    - add excluded stock
    - small revision to adapt to newer `input_c` and `opt` version
    '''
    make_required_dir(ROOT_PATH, model_version)
    # Create progress cache
    progress_cache_file = f'{ROOT_PATH}progress_cache/v{model_version}/{simtype}_{offset}_{instance_no}.txt'
    try:
        with open(progress_cache_file, 'r') as f:
            HOTSTART = int(f.read())
    except FileNotFoundError:
        HOTSTART = 0
    
    # Fetch stock metadata
    # investing_metadata = stock_metadata(ROOT_PATH)
    excluded_stocks = excluded_stock()
    tickers = stock_list_100_highestrank_and_availability()
    tickers = [ticker for ticker in tickers if ticker not in excluded_stocks]
    
    # Total batch per instances
    batch = int(len(tickers) / split)
    
    batch_metadata = tickers[instance_no * batch:(instance_no + 1) * batch]
    
    loops = 0
    for ticker in batch_metadata:
        for output_c in output_cs:
            # Hotstart
            if loops < HOTSTART:
                loops+=1
                continue
            if simtype == 'input_c':
                execute_input_c_v2(ticker, ROOT_PATH, output_c=output_c, model_version=model_version, input_c_limit=input_c_limit, epochs=epochs, rnnus=rnnus, recurrents=recurrents, train=train)
            elif simtype == 'opt':
                execute_configuration_optimization_v2(ticker, ROOT_PATH, output_c=output_c, model_version=model_version, epochs=epochs, rnnus=rnnus, recurrents=recurrents, train=train)
            loops+=1
            
            # Save progress to cache
            with open(progress_cache_file, 'w') as f:
                f.write(str(loops))  
                
def execute_retrain_model_v2(ROOT_PATH='./', output_cs=('close_EMA3_G_s1','close_EMA10_G_s1','close_EMA3_G_s2','close_EMA10_G_s2'), model_version='2', iteration_version ='1', backtest=True):
    '''
    source_db = 'idx_data_v1.3.db'
    For backtest:
        backtest = True
        train = 0.9
    For production:
        backtest = False
        train = 0.99
        
    Revision in v2
    - Create new directory path for easier
        model version grouping
    '''
    make_required_dir(ROOT_PATH, model_version)
    if backtest:
        train = 0.9
        STATISTICS_FILE = f'{ROOT_PATH}statistics/v{model_version}/backtest/{iteration_version}.csv'
    else:
        train = 0.99
        STATISTICS_FILE = f'{ROOT_PATH}statistics/v{model_version}/production/{iteration_version}.csv'

    excluded_stocks = excluded_stock()
    tickers = stock_list_100_highestrank_and_availability()
    tickers = [ticker for ticker in tickers if ticker not in excluded_stocks]
    
    for ticker in tickers:
        DB_PATH = f'{ROOT_PATH}db/v{model_version}/idx_{ticker}.db'
        # Load data
        df = load_ml_database(DB_PATH)
        train_df, test_df, train_mean, train_std = split_traintest(df, split=train)

        loops = 0
        for output_c in output_cs:
            tick = datetime.datetime.now()

            # Read optimal input_c configuration
            input_c = input_c_from_sa_v2(ROOT_PATH, model_version, ticker, output_c)

            # Read optimal model configuration
            optimization_statistics_path = f'{ROOT_PATH}statistics/v{model_version}/sa_{ticker}_{output_c}.csv'
            architecture, recurrent, rnnu, epoch = read_optimal_model_config(optimization_statistics_path, indicator='test_rmse')

            # Make dataset
            train_dataset = TimeseriesGenerator(train_df[input_c], train_df[output_c], length=recurrent, shuffle=True)
            test_dataset = TimeseriesGenerator(test_df[input_c], test_df[output_c], length=recurrent)

            # RNN Model
            if architecture == 'SimpleRNN':
                model = simpleRNN_1layer(rnnu, return_sequences=False)
            elif architecture == 'GRU':
                model = GRU_1layer(rnnu, return_sequences=False)
            elif architecture == 'LSTM':
                model = LSTM_1layer(rnnu, return_sequences=False)

            # Feed and train the model
            model.fit(train_dataset, epochs=epoch, verbose=0)

            # Save model weights
            weights_save_path = f'{ROOT_PATH}models/v{model_version}/{iteration_version}/{ticker}_{output_c}/'
            model.save_weights(weights_save_path)

            # Delete current model and re-load weights to make sure that weight is recoverable
            del model
            # RNN Model
            if architecture == 'SimpleRNN':
                model = simpleRNN_1layer(rnnu, return_sequences=False)
            elif architecture == 'GRU':
                model = GRU_1layer(rnnu, return_sequences=False)
            elif architecture == 'LSTM':
                model = LSTM_1layer(rnnu, return_sequences=False)
            model.load_weights(weights_save_path)

            # Evaluate model
            train_eval = model.evaluate(train_dataset, verbose=0)
            test_eval = model.evaluate(test_dataset, verbose=0)

            input_c_joined = '|'.join(input_c)
            train_mean_joined = '|'.join([str(x) for x in list(train_mean[input_c])])
            train_std_joined = '|'.join([str(x) for x in list(train_std[input_c])])
            statistics = f'''{ticker},{input_c_joined},{offset},{output_c},{architecture},{recurrent},{rnnu},{epoch},{train_eval[0]},{train_eval[1]},{test_eval[0]},{test_eval[1]},{train_mean_joined},{train_std_joined},{train_mean[output_c]},{train_std[output_c]}\n'''
            with open(STATISTICS_FILE, 'a') as f:
                f.write(statistics)

            tock = datetime.datetime.now()

            print(f'''{loops} {ticker} {len(input_c)} {offset} {output_c} {architecture} R[{recurrent}] RU[{rnnu}] E[{epoch}] trainRMSE[{round(train_eval[1], 5)}] testRMSE[{round(test_eval[1], 5)}] time: {tock-tick}''')
            loops+=1
            
def columns_to_drop():
    '''Contains columns that:
    - behave linearly over time
    - provide sight to future values
    - contains all nan values'''
    columns_to_drop = ['time','index','gd','signal', 'change_EMA3_G', 'change_EMA10_G', 'change_EMA30_G', 'change_EMA200_G', 'cdl2crows', 'cdl3starsinsouth', 'cdlabandonedbaby', 'cdlbreakaway', 'cdlconcealbabyswall', 'cdlidentical3crows', 'cdlmathold', 'cdlrisefall3methods', 'cdlupsidegap2crows', 'open_ht_trendmode_EMA3_G', 'open_ht_trendmode_EMA10_G', 'open_ht_trendmode_EMA30_G', 'open_ht_trendmode_EMA200_G', 'high_ht_trendmode_EMA3_G', 'high_ht_trendmode_EMA10_G', 'high_ht_trendmode_EMA30_G', 'high_ht_trendmode_EMA200_G', 'low_ht_trendmode_EMA3_G', 'low_ht_trendmode_EMA10_G', 'low_ht_trendmode_EMA30_G', 'low_ht_trendmode_EMA200_G', 'close_ht_trendmode_EMA3_G', 'close_ht_trendmode_EMA10_G', 'close_ht_trendmode_EMA30_G', 'close_ht_trendmode_EMA200_G', 'Volume_ht_trendmode_EMA3_G', 'Volume_ht_trendmode_EMA10_G', 'Volume_ht_trendmode_EMA30_G', 'Volume_ht_trendmode_EMA200_G', 'change_roc_EMA3_G', 'change_roc_EMA10_G', 'change_roc_EMA30_G', 'change_roc_EMA200_G', 'change_rocp_EMA3_G', 'change_rocp_EMA10_G', 'change_rocp_EMA30_G', 'change_rocp_EMA200_G', 'change_rocr_EMA3_G', 'change_rocr_EMA10_G', 'change_rocr_EMA30_G', 'change_rocr_EMA200_G', 'change_rocr100_EMA3_G', 'change_rocr100_EMA10_G', 'change_rocr100_EMA30_G', 'change_rocr100_EMA200_G', 'change_ht_trendmode_EMA3_G', 'change_ht_trendmode_EMA10_G', 'change_ht_trendmode_EMA30_G', 'change_ht_trendmode_EMA200_G', 'Volume_rank_ht_trendmode_EMA3_G', 'Volume_rank_ht_trendmode_EMA10_G', 'Volume_rank_ht_trendmode_EMA30_G', 'Volume_rank_ht_trendmode_EMA200_G', 'willr_EMA3_G', 'willr_EMA10_G', 'willr_EMA30_G', 'willr_EMA200_G', 'cdl2crows_EMA3', 'cdl2crows_EMA10', 'cdl2crows_EMA30', 'cdl2crows_EMA200', 'cdl2crows_EMA3_G', 'cdl2crows_EMA10_G', 'cdl2crows_EMA30_G', 'cdl2crows_EMA200_G', 'cdl2crows_EMA3_EMA10_offset', 'cdl2crows_EMA3_EMA30_offset', 'cdl2crows_EMA3_EMA200_offset', 'cdl3blackrows_EMA3_G', 'cdl3blackrows_EMA10_G', 'cdl3blackrows_EMA30_G', 'cdl3blackrows_EMA200_G', 'cdl3inside_EMA3_G', 'cdl3inside_EMA10_G', 'cdl3inside_EMA30_G', 'cdl3inside_EMA200_G', 'cdl3linestrike_EMA3_G', 'cdl3linestrike_EMA10_G', 'cdl3linestrike_EMA30_G', 'cdl3linestrike_EMA200_G', 'cdl3outside_EMA3_G', 'cdl3outside_EMA10_G', 'cdl3outside_EMA30_G', 'cdl3outside_EMA200_G', 'cdl3starsinsouth_EMA3', 'cdl3starsinsouth_EMA10', 'cdl3starsinsouth_EMA30', 'cdl3starsinsouth_EMA200', 'cdl3starsinsouth_EMA3_G', 'cdl3starsinsouth_EMA10_G', 'cdl3starsinsouth_EMA30_G', 'cdl3starsinsouth_EMA200_G', 'cdl3starsinsouth_EMA3_EMA10_offset', 'cdl3starsinsouth_EMA3_EMA30_offset', 'cdl3starsinsouth_EMA3_EMA200_offset', 'cdl3whitesoldiers_EMA3_G', 'cdl3whitesoldiers_EMA10_G', 'cdl3whitesoldiers_EMA30_G', 'cdl3whitesoldiers_EMA200_G', 'cdlabandonedbaby_EMA3', 'cdlabandonedbaby_EMA10', 'cdlabandonedbaby_EMA30', 'cdlabandonedbaby_EMA200', 'cdlabandonedbaby_EMA3_G', 'cdlabandonedbaby_EMA10_G', 'cdlabandonedbaby_EMA30_G', 'cdlabandonedbaby_EMA200_G', 'cdlabandonedbaby_EMA3_EMA10_offset', 'cdlabandonedbaby_EMA3_EMA30_offset', 'cdlabandonedbaby_EMA3_EMA200_offset', 'cdladvanceblock_EMA3_G', 'cdladvanceblock_EMA10_G', 'cdladvanceblock_EMA30_G', 'cdladvanceblock_EMA200_G', 'cdlbelthold_EMA3_G', 'cdlbelthold_EMA10_G', 'cdlbelthold_EMA30_G', 'cdlbelthold_EMA200_G', 'cdlbreakaway_EMA3', 'cdlbreakaway_EMA10', 'cdlbreakaway_EMA30', 'cdlbreakaway_EMA200', 'cdlbreakaway_EMA3_G', 'cdlbreakaway_EMA10_G', 'cdlbreakaway_EMA30_G', 'cdlbreakaway_EMA200_G', 'cdlbreakaway_EMA3_EMA10_offset', 'cdlbreakaway_EMA3_EMA30_offset', 'cdlbreakaway_EMA3_EMA200_offset', 'cdlclosingmarubozu_EMA3_G', 'cdlclosingmarubozu_EMA10_G', 'cdlclosingmarubozu_EMA30_G', 'cdlclosingmarubozu_EMA200_G', 'cdlconcealbabyswall_EMA3', 'cdlconcealbabyswall_EMA10', 'cdlconcealbabyswall_EMA30', 'cdlconcealbabyswall_EMA200', 'cdlconcealbabyswall_EMA3_G', 'cdlconcealbabyswall_EMA10_G', 'cdlconcealbabyswall_EMA30_G', 'cdlconcealbabyswall_EMA200_G', 'cdlconcealbabyswall_EMA3_EMA10_offset', 'cdlconcealbabyswall_EMA3_EMA30_offset', 'cdlconcealbabyswall_EMA3_EMA200_offset', 'cdlcounterattack_EMA3_G', 'cdlcounterattack_EMA10_G', 'cdlcounterattack_EMA30_G', 'cdlcounterattack_EMA200_G', 'cdldarkcloudcover_EMA3_G', 'cdldarkcloudcover_EMA10_G', 'cdldarkcloudcover_EMA30_G', 'cdldarkcloudcover_EMA200_G', 'cdldoji_EMA3_G', 'cdldoji_EMA10_G', 'cdldoji_EMA30_G', 'cdldoji_EMA200_G', 'cdldojistar_EMA3_G', 'cdldojistar_EMA10_G', 'cdldojistar_EMA30_G', 'cdldojistar_EMA200_G', 'cdldragonflydoji_EMA3_G', 'cdldragonflydoji_EMA10_G', 'cdldragonflydoji_EMA30_G', 'cdldragonflydoji_EMA200_G', 'cdlengulfing_EMA3_G', 'cdlengulfing_EMA10_G', 'cdlengulfing_EMA30_G', 'cdlengulfing_EMA200_G', 'cdleveningdojistar_EMA3_G', 'cdleveningdojistar_EMA10_G', 'cdleveningdojistar_EMA30_G', 'cdleveningdojistar_EMA200_G', 'cdleveningstar_EMA3_G', 'cdleveningstar_EMA10_G', 'cdleveningstar_EMA30_G', 'cdleveningstar_EMA200_G', 'cdlgapinsidewhite_EMA3_G', 'cdlgapinsidewhite_EMA10_G', 'cdlgapinsidewhite_EMA30_G', 'cdlgapinsidewhite_EMA200_G', 'cdlgravestonedoji_EMA3_G', 'cdlgravestonedoji_EMA10_G', 'cdlgravestonedoji_EMA30_G', 'cdlgravestonedoji_EMA200_G', 'cdlhammer_EMA3_G', 'cdlhammer_EMA10_G', 'cdlhammer_EMA30_G', 'cdlhammer_EMA200_G', 'cdlhangingman_EMA3_G', 'cdlhangingman_EMA10_G', 'cdlhangingman_EMA30_G', 'cdlhangingman_EMA200_G', 'cdlharami_EMA3_G', 'cdlharami_EMA10_G', 'cdlharami_EMA30_G', 'cdlharami_EMA200_G', 'cdlharamicross_EMA3_G', 'cdlharamicross_EMA10_G', 'cdlharamicross_EMA30_G', 'cdlharamicross_EMA200_G', 'cdlhighwave_EMA3_G', 'cdlhighwave_EMA10_G', 'cdlhighwave_EMA30_G', 'cdlhighwave_EMA200_G', 'cdlhikkake_EMA3_G', 'cdlhikkake_EMA10_G', 'cdlhikkake_EMA30_G', 'cdlhikkake_EMA200_G', 'cdlhikkakemod_EMA3_G', 'cdlhikkakemod_EMA10_G', 'cdlhikkakemod_EMA30_G', 'cdlhikkakemod_EMA200_G', 'cdlhomingpigeon_EMA3_G', 'cdlhomingpigeon_EMA10_G', 'cdlhomingpigeon_EMA30_G', 'cdlhomingpigeon_EMA200_G', 'cdlidentical3crows_EMA3', 'cdlidentical3crows_EMA10', 'cdlidentical3crows_EMA30', 'cdlidentical3crows_EMA200', 'cdlidentical3crows_EMA3_G', 'cdlidentical3crows_EMA10_G', 'cdlidentical3crows_EMA30_G', 'cdlidentical3crows_EMA200_G', 'cdlidentical3crows_EMA3_EMA10_offset', 'cdlidentical3crows_EMA3_EMA30_offset', 'cdlidentical3crows_EMA3_EMA200_offset', 'cdlinneck_EMA3_G', 'cdlinneck_EMA10_G', 'cdlinneck_EMA30_G', 'cdlinneck_EMA200_G', 'cdlinvertedhammer_EMA3_G', 'cdlinvertedhammer_EMA10_G', 'cdlinvertedhammer_EMA30_G', 'cdlinvertedhammer_EMA200_G', 'cdlkicking_EMA3_G', 'cdlkicking_EMA10_G', 'cdlkicking_EMA30_G', 'cdlkicking_EMA200_G', 'cdlkickingbylength_EMA3_G', 'cdlkickingbylength_EMA10_G', 'cdlkickingbylength_EMA30_G', 'cdlkickingbylength_EMA200_G', 'cdlladderbottom_EMA3_G', 'cdlladderbottom_EMA10_G', 'cdlladderbottom_EMA30_G', 'cdlladderbottom_EMA200_G', 'cdllongleggeddoji_EMA3_G', 'cdllongleggeddoji_EMA10_G', 'cdllongleggeddoji_EMA30_G', 'cdllongleggeddoji_EMA200_G', 'cdllongline_EMA3_G', 'cdllongline_EMA10_G', 'cdllongline_EMA30_G', 'cdllongline_EMA200_G', 'cdlmarubozu_EMA3_G', 'cdlmarubozu_EMA10_G', 'cdlmarubozu_EMA30_G', 'cdlmarubozu_EMA200_G', 'cdlmatchinglow_EMA3_G', 'cdlmatchinglow_EMA10_G', 'cdlmatchinglow_EMA30_G', 'cdlmatchinglow_EMA200_G', 'cdlmathold_EMA3', 'cdlmathold_EMA10', 'cdlmathold_EMA30', 'cdlmathold_EMA200', 'cdlmathold_EMA3_G', 'cdlmathold_EMA10_G', 'cdlmathold_EMA30_G', 'cdlmathold_EMA200_G', 'cdlmathold_EMA3_EMA10_offset', 'cdlmathold_EMA3_EMA30_offset', 'cdlmathold_EMA3_EMA200_offset', 'cdlmorningdojistar_EMA3_G', 'cdlmorningdojistar_EMA10_G', 'cdlmorningdojistar_EMA30_G', 'cdlmorningdojistar_EMA200_G', 'cdlmorningstar_EMA3_G', 'cdlmorningstar_EMA10_G', 'cdlmorningstar_EMA30_G', 'cdlmorningstar_EMA200_G', 'cdlonneck_EMA3_G', 'cdlonneck_EMA10_G', 'cdlonneck_EMA30_G', 'cdlonneck_EMA200_G', 'cdlpiercing_EMA3_G', 'cdlpiercing_EMA10_G', 'cdlpiercing_EMA30_G', 'cdlpiercing_EMA200_G', 'cdlrickshawman_EMA3_G', 'cdlrickshawman_EMA10_G', 'cdlrickshawman_EMA30_G', 'cdlrickshawman_EMA200_G', 'cdlrisefall3methods_EMA3', 'cdlrisefall3methods_EMA10', 'cdlrisefall3methods_EMA30', 'cdlrisefall3methods_EMA200', 'cdlrisefall3methods_EMA3_G', 'cdlrisefall3methods_EMA10_G', 'cdlrisefall3methods_EMA30_G', 'cdlrisefall3methods_EMA200_G', 'cdlrisefall3methods_EMA3_EMA10_offset', 'cdlrisefall3methods_EMA3_EMA30_offset', 'cdlrisefall3methods_EMA3_EMA200_offset', 'cdlseparatinglines_EMA3_G', 'cdlseparatinglines_EMA10_G', 'cdlseparatinglines_EMA30_G', 'cdlseparatinglines_EMA200_G', 'cdlshootingstar_EMA3_G', 'cdlshootingstar_EMA10_G', 'cdlshootingstar_EMA30_G', 'cdlshootingstar_EMA200_G', 'cdlshortline_EMA3_G', 'cdlshortline_EMA10_G', 'cdlshortline_EMA30_G', 'cdlshortline_EMA200_G', 'cdlspinningtop_EMA3_G', 'cdlspinningtop_EMA10_G', 'cdlspinningtop_EMA30_G', 'cdlspinningtop_EMA200_G', 'cdlstalledpattern_EMA3_G', 'cdlstalledpattern_EMA10_G', 'cdlstalledpattern_EMA30_G', 'cdlstalledpattern_EMA200_G', 'cdlsticksandwich_EMA3_G', 'cdlsticksandwich_EMA10_G', 'cdlsticksandwich_EMA30_G', 'cdlsticksandwich_EMA200_G', 'cdltakuri_EMA3_G', 'cdltakuri_EMA10_G', 'cdltakuri_EMA30_G', 'cdltakuri_EMA200_G', 'cdltasukigap_EMA3_G', 'cdltasukigap_EMA10_G', 'cdltasukigap_EMA30_G', 'cdltasukigap_EMA200_G', 'cdlthrusting_EMA3_G', 'cdlthrusting_EMA10_G', 'cdlthrusting_EMA30_G', 'cdlthrusting_EMA200_G', 'cdltristar_EMA3_G', 'cdltristar_EMA10_G', 'cdltristar_EMA30_G', 'cdltristar_EMA200_G', 'cdlunique3river_EMA3_G', 'cdlunique3river_EMA10_G', 'cdlunique3river_EMA30_G', 'cdlunique3river_EMA200_G', 'cdlupsidegap2crows_EMA3', 'cdlupsidegap2crows_EMA10', 'cdlupsidegap2crows_EMA30', 'cdlupsidegap2crows_EMA200', 'cdlupsidegap2crows_EMA3_G', 'cdlupsidegap2crows_EMA10_G', 'cdlupsidegap2crows_EMA30_G', 'cdlupsidegap2crows_EMA200_G', 'cdlupsidegap2crows_EMA3_EMA10_offset', 'cdlupsidegap2crows_EMA3_EMA30_offset', 'cdlupsidegap2crows_EMA3_EMA200_offset', 'cdlxsidegap3methods_EMA3_G', 'cdlxsidegap3methods_EMA10_G', 'cdlxsidegap3methods_EMA30_G', 'cdlxsidegap3methods_EMA200_G']
    return columns_to_drop
            
def drop_unallowed_columns(df, columns_to_drop=columns_to_drop(), re_rules=['change_b.f.', '_s[0-9]*', 'candle*']):
    # Drop column in `columns_to_drop`
    for column_to_drop in columns_to_drop:
        try:
            df = df.drop(column_to_drop, axis='columns')
        except (ValueError, KeyError):
            pass    
    
    # Drop column using `re_rule`
    columns = [x for x in df.columns]
    for re_rule in re_rules:
        columns = [x for x in columns if not re.search(re_rule, x)]
    df = df[columns]
    return df

def model_132(recurrent, train_df_input):
    # MODEL 132
    # kernel_regularizers = tf.keras.regularizers.l1_l2()
    kernel_regularizers = None
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=4, kernel_size=1, strides=1, padding='same', kernel_initializer=tf.keras.initializers.HeUniform(), input_shape=(recurrent, train_df_input.shape[1]), kernel_regularizer=kernel_regularizers),
        tf.keras.layers.Conv1D(filters=8, kernel_size=9, strides=1, padding='same', kernel_initializer=tf.keras.initializers.HeUniform(), kernel_regularizer=kernel_regularizers),
        tf.keras.layers.Conv1D(filters=64, kernel_size=7, strides=10, padding='same', kernel_initializer=tf.keras.initializers.HeUniform(), kernel_regularizer=kernel_regularizers),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer=tf.keras.initializers.HeUniform(), kernel_regularizer=kernel_regularizers),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True, kernel_regularizer=kernel_regularizers)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=8, return_sequences=True, kernel_regularizer=kernel_regularizers)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=32, return_sequences=False, kernel_regularizer=kernel_regularizers)),
        tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=kernel_regularizers),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=kernel_regularizers),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.Huber()
    model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()])
    return model

def model_188(recurrent, train_df_input):
    heuniform = tf.keras.initializers.HeUniform()
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(recurrent, train_df_input.shape[1])),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer=heuniform),
        tf.keras.layers.LSTM(units=1000, return_sequences=False),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999)
    loss = tf.keras.losses.Huber()
    model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.RootMeanSquaredError()])
    return model

def merge_all_stock(ROOT_PATH):
    db_path = f'{ROOT_PATH}idx_indicators.db'
    db_conn = sqlite3.connect(db_path)
    tickers = stock_metadata(ROOT_PATH)
    excluded_stocks = excluded_stock()
    tickers = [ticker for ticker in tickers['ticker'].values if ticker not in excluded_stocks]
    
    frames = []
    first = True
    count = 0
    for ticker in tickers:
        if first:
            first = False
            pass
        elif not first:
            del df

        df = pd.read_sql(f'select * from `{ticker}`', db_conn)
        df = (df - df.mean()) / df.std()
        frames.append(df)
        count+=1
        if count % 50 == 0:
            print(f'{count}/{len(tickers)}: {ticker}')
    result = pd.concat(frames)
    result = result.reset_index(drop=True)
    return result

def load_all_data(ROOT_PATH):
    '''Load all stock inside idx_indicators.db
    into one giant dataframe. Before concatinating,
    the data is scaled using standard scaler.
    '''
    # Using whole idx data to perform universal training
    # After the train has finished, freeze all layers
    # up to RNN, and replace the dense. 
    # Retrain using last 5 years data every stock in idx
    ROOT_PATH = './gdrive/MyDrive/#PROJECT/idx/'

    # Above code dont work with colab. It may because the size
    # is to big for the compute to handle.
    # DB_PATH = f'{ROOT_PATH}idx_merges.db'
    # db_conn = sqlite3.connect(DB_PATH)
    # df = pd.read_sql(f'select * from idx', db_conn)
    # df = df.fillna(0)

    df = merge_all_stock(ROOT_PATH)
    df = df.fillna(0)
    return df

def model_switcher(model_no, recurrent, train_df_input):
    '''Return compiled model based on model identifier'''
    if model_no == '132':
        return model_132(recurrent, train_df_input)
    elif model_no == '188':
        return model_188(recurrent, train_df_input)
    
def model_switcher_kt_build(model_no, hp, train_inputs, train_labels):
    if model_no == '263':
        return model_263_kt_build(hp, train_inputs, train_labels)
    elif model_no == '300':
        return model_300_kt_build(hp, train_inputs, train_labels)
    elif model_no == '301':
        return model_301_kt_build(hp, train_inputs, train_labels)
    elif model_no == '307':
        return model_307_kt_build(hp, train_inputs, train_labels)
    elif model_no == '308':
        return model_308_kt_build(hp, train_inputs, train_labels)
    elif model_no == '309':
        return model_309_kt_build(hp, train_inputs, train_labels)
    elif model_no == '310':
        return model_310_kt_build(hp, train_inputs, train_labels)
    elif model_no == '311':
        return model_311_kt_build(hp, train_inputs, train_labels)
    elif model_no == '312':
        return model_312_kt_build(hp, train_inputs, train_labels)
    elif model_no == '313':
        return model_313_kt_build(hp, train_inputs, train_labels)
    elif model_no == '314':
        return model_314_kt_build(hp, train_inputs, train_labels)
    elif model_no == '315':
        return model_315_kt_build(hp, train_inputs, train_labels)
    elif model_no == '316':
        return model_316_kt_build(hp, train_inputs, train_labels)
    elif model_no == '317':
        return model_317_kt_build(hp, train_inputs, train_labels)
    elif model_no == '318':
        return model_318_kt_build(hp, train_inputs, train_labels)
    elif model_no == '319':
        return model_319_kt_build(hp, train_inputs, train_labels)
    elif model_no == '320':
        return model_320_kt_build(hp, train_inputs, train_labels)
    else:
        print(f'Invalid model number. Check again. Passed value: {model_no}')
        pass
    
def resume_from_checkpoint(checkpoint_dir, model_no, recurrent, train_df_input, label_columns):
    '''return model with latest weight, latest_epoch, 
    latest_label_column, and latest_label_index from checkpoint_dir.
    Return consecutively untrained model, 0, None, 0 if
    checkpoint not found in path.'''
    latest_weight_path = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_weight_path != None:
        print(latest_weight_path)
        model = model_switcher(model_no, recurrent, train_df_input)
        model.load_weights(latest_weight_path)
        latest_epoch = int(latest_weight_path.split('-')[1].replace('.ckpt',''))
        latest_label_column = latest_weight_path.split('/')[-1].split('cp')[0]
        latest_label_index = label_columns.index(latest_label_column)
        label_columns = label_columns[latest_label_index:]
        print(f'Resuming training from {label_columns} at {latest_epoch} epochs.')
    elif latest_weight_path == None:
        model = model_188(recurrent, train_df_input)
        latest_epoch = 0
        latest_label_column = None
        latest_label_index = 0
    return model, latest_epoch, latest_label_column, latest_label_index, label_columns

def prepare_newest_data_v1(origin_db='idx_raw.db', target_db='idx_indicators.db', selected_stock_only=False, period='max', sleep=10):
    '''This version:
    - fetch newest data from yfinance
    - store newest data to `idx_raw.db`
    - calculate v1 indicator (base)
    - store calculated indicators to
        `idx_indicators.db`
    '''
    excluded_stock = yfinance_db(stock_db=origin_db, if_exists='replace', selected_stock_only=selected_stock_only, period=period, sleep=sleep)
    calculate_all_indicator(origin_db, target_db, excluded_stock, verbose=1, selected_stock_only=selected_stock_only)
    
def fix_dataset_length(origin_db='idx_raw.db', target_db='idx_indicators.db', validity_threshold=0.95, excluded_stock=excluded_stock(), selected_stock_only=False):
    '''Identifyand fix different end-date 
    data for stocks in idx that may cause
    mis-interpretation in the model.
    
    Drop latest value in `idx_raw.db` if its
    found that the null value < validity_threshold.
    
    Recalculate `idx_indicators.db` if the last row
    availability lower than validity_threshold.       
    '''
    origin_db_conn = sqlite3.connect(origin_db)
    target_db_conn = sqlite3.connect(target_db)
    df = pd.read_sql('select * from `IDX`', target_db_conn)

    last_row_stats = df.iloc[-1].describe()
    second_last_row_stats = df.iloc[-2].describe()
    validity_ratio = last_row_stats['count'] / second_last_row_stats['count']
    
    if validity_ratio < validity_threshold:
        print('Found incomplete date, fixing...')
        # Get date value
        date_to_eliminate = df.iloc[-1].time

        # Run to idx raw and delete
        tickers = tablename_list(origin_db_conn)
        for ticker in tickers:
            df_ticker = pd.read_sql(f'select * from `{ticker}`', origin_db_conn)
            df_ticker = df_ticker.loc[df_ticker['time'] < date_to_eliminate]
            df_ticker.to_sql(ticker, origin_db_conn, if_exists='replace', index=False)
        print('Done fixing, recalculate indicators...')
        calculate_all_indicator(origin_db, target_db, excluded_stock, verbose=1, selected_stock_only=selected_stock_only)
        print('Done indicator calculation.')
    else:
        print('No anomaly detected, all data was complete.')
        
def get_recommended_stocks(benchmark_column='close_EMA200_G', tickers=stock_list_100_highestrank_and_availability(), ma=True):
    '''Compare all stocks in tickers based on `benchmark_column`
    and `tickers`. Return `selected_tickers` and
    `selected_tickers_ema`.'''
    db_path = 'idx_indicators.db'
    db_conn = sqlite3.connect(db_path)
    selected_tickers = []
    selected_tickers_ema = []
    for ticker in tickers:
        df = pd.read_sql(f'select * from `{ticker}` order by time desc limit 1', db_conn)
        if selected_tickers_ema == []:
            benchmark = 0
        else:
            benchmark = np.array(selected_tickers_ema).mean() if ma else 0
        if df[benchmark_column].values[0] > benchmark:
            selected_tickers.append(ticker)
            selected_tickers_ema.append(df[benchmark_column].values[0])
    return selected_tickers, selected_tickers_ema

def gradient_difference_signal_indicator(df, signal=2, successive=1):
    '''
    This would add 2 columns to DataFrame: `gd` and `signal`
    `gd` is based on difference between EMA3_G at signal and EMA3_G 
    at successive period.
    '''
    df['gd'] = df['close_EMA3_G'].shift(-signal) - df['close_EMA3_G'].shift(-successive)
    df.loc[df['gd'] > 0, 'signal'] = 1
    df.loc[df['gd'] <= 0, 'signal'] = 0
    return df   
    

def random_group(list_items, size, seed=0, same_size=True, use_itertools=False, num_of_samples=1000):
    '''Yield successive sized chunks from lsit
    Params:
        same_size: eliminate underrequirements group
            Eliminate list with different size
    '''
    random.seed(seed)
    random.shuffle(list_items)
    if not use_itertools:
        results = []
        for i in range(0, len(list_items), size):
            results.append(list_items[i:i + size])
        if same_size:
            return results[:len(list_items) // size]
        elif not same_size:
            return results
    elif use_itertools:
        combinations_list = list(itertools.combinations(list_items,size))
        return random.sample(combinations_list, num_of_samples)
    
def time_slice_list_from_group(group, ROOT_PATH='./', db_type='1', db_ver='3', db='idx_indicators.db'):
    first = True
    for ticker in group:
        if db_type == '1':
            DB_PATH = os.path.join(ROOT_PATH, db)
            db_conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql(f'select time from `{ticker}`', db_conn)
        elif db_type == '2':
            DB_PATH = os.path.join(ROOT_PATH, f'db/v{db_ver}/idx_{ticker}.db')
            db_conn = sqlite3.connect(DB_PATH)
            table_names = tablename_list(db_conn)
            df = pd.read_sql(f'select time from `{table_names[0]}`', db_conn)
            
        if first:
            combined_db = df.copy(deep=True)
            first = False
        elif not first:
            combined_db = combined_db.merge(df, how='inner', on='time')
    return combined_db

def make_multiple_shot_input_v2(df, recurrent):
    '''Make multiple shot input dataset
    Algorithm: data until time t-1 used to represent t
    '''
    first = True
    for i, index in enumerate(df[recurrent - 1:].index):
        # Structure and normalize 1-shot input data
        input_data = df[i: i + recurrent]
        input_data_np = input_data.to_numpy()
        input_data_np = input_data_np.reshape(1, input_data_np.shape[0], input_data_np.shape[1])

        # Cut-off process when the df len remainder not enough
        # to fullfill the recurrent requirement
        if input_data_np.shape[1] != recurrent:
            continue

        if first:
            multiple_shot_input_data = copy.deepcopy(input_data_np)
            first = False
        else:
            multiple_shot_input_data = np.vstack((multiple_shot_input_data, input_data_np))
    return multiple_shot_input_data

# @njit
def make_multiple_shot_input_v3(npdf, recurrent):
    '''Make multiple shot input dataset
    Algorithm: data until time t-1 used to represent t
    
    Difference with v2:
    - remove pandas representation and convert to pure numpy.'''
    for i in range(npdf[recurrent - 1:].shape[0]):
        # Structure and normalize 1-shot input data
        input_data = npdf[i: i + recurrent].copy()
        input_data = input_data.reshape((1, input_data.shape[0], input_data.shape[1]))

        # Cut-off process when the df len remainder not enough
        # to fullfill the recurrent requirement
        if input_data.shape[1] != recurrent:
            continue

        if i == 0:
            multiple_shot_input_data = input_data.copy()
        elif i > 0:
            multiple_shot_input_data = np.vstack((multiple_shot_input_data, input_data))
    return multiple_shot_input_data

def adjust_dataset_shape(inputs, labels, changes, shift=0):
    '''Due to shifting and recurrent slicing, the
    shape of inputs and labels may differ. This function
    slice the labels that satisfy time shift and adjust
    their dimensions.
    
    Params:
        shift: time shift from t (t+shift)
    '''
    labels = labels[inputs.shape[1] + shift:]
    changes = changes[inputs.shape[1] + shift:]
    if labels.shape[0] > inputs.shape[0]:
        labels = labels[:inputs.shape[0] - labels.shape[0]]
        changes = changes[:inputs.shape[0] - changes.shape[0]]
    elif labels.shape[0] < inputs.shape[0]:
        inputs = inputs[:labels.shape[0] - inputs.shape[0]]
    return inputs, labels, changes

def backtest_v1(model, train_inputs, test_inputs, train_labels, test_labels, train_changes, test_changes, threshold=0.5):
    '''This function valid for recommendation algorithm v1
    that output one two output node (recommended & not_recommended)
    '''
    train_predictions = model.predict(train_inputs)
    test_predictions = model.predict(test_inputs)

    train_predictions_one_hot = copy.deepcopy(train_predictions)
    train_predictions_one_hot[train_predictions_one_hot > threshold] = 1
    train_predictions_one_hot[train_predictions_one_hot <= threshold] = 0

    test_predictions_one_hot = copy.deepcopy(test_predictions)
    test_predictions_one_hot[test_predictions_one_hot > threshold] = 1
    test_predictions_one_hot[test_predictions_one_hot <= threshold] = 0

    train_ideal = np.dot(train_changes.T, train_labels if isinstance(train_labels, np.ndarray) else train_labels.numpy())
    train_real = np.dot(train_changes.T, train_predictions_one_hot)
    test_ideal = np.dot(test_changes.T, test_labels if isinstance(test_labels, np.ndarray) else test_labels.numpy())
    test_real = np.dot(test_changes.T, test_predictions_one_hot)

    return train_ideal, train_real, test_ideal, test_real

def performance_ratio(train_ideal, train_real, test_ideal, test_real):
    train_r = train_real[0][0] / train_ideal[0][0]
    train_nr = train_real[0][1] / train_ideal[0][1]
    test_r = test_real[0][0] / test_ideal[0][0]
    test_nr = test_real[0][1] / test_ideal[0][1]
    return train_r, train_nr, test_r, test_nr

def make_one_shot_input(ticker, conn, recurrent, shift_from_last=0):
    '''Make input for tomorrow prediction'''
    df = pd.read_sql(f'select * from `{ticker}`', conn)
    input_slice = df[-recurrent-shift_from_last:-shift_from_last if shift_from_last > 0 else None]
    input_cleaned = drop_unallowed_columns(input_slice)
    return input_cleaned

def model_263(train_inputs, train_labels):
    # MODEL 263: 0.6745-0.6149 #
    # tf.keras.layers.Lambda(lambda x: tf.math.multiply(x, np.array([0.3,0.7])))
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2])),
        tf.keras.layers.Conv1D(filters=16, kernel_size=1, strides=1, padding='same'),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=2, padding='same'),
        tf.keras.layers.Conv1D(filters=16, kernel_size=1, strides=1, padding='same'),
        tf.keras.layers.GRU(units=6, return_sequences=False),
        tf.keras.layers.Dense(units=12, activation='relu'),
        tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005) # 
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) # , run_eagerly=True
    ############
    return model

def model_263_kt(hp):
    '''model_263 kt version: 
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    first_conv_filters = hp.Int('first_conv_filters', min_value=8, max_value=64, step=8)
    second_conv_filters = hp.Int('second_conv_filters', min_value=16, max_value=128, step=16)
    second_conv_kernel_size = hp.Int('second_conv_kernel_size', min_value=1, max_value=15, step=2)
    second_conv_strides = hp.Int('second_conv_strides', min_value=1, max_value=5, step=1)
    third_conv_filters = hp.Int('third_conv_filters', min_value=16, max_value=128, step=16)
    third_conv_kernel_size = hp.Int('third_conv_kernel_size', min_value=1, max_value=20, step=3)
    third_conv_strides = hp.Int('third_conv_strides', min_value=1, max_value=5, step=1)
    recurrent_unit = hp.Int('recurrent_unit', min_value=6, max_value=60, step=6)
    dense_unit = hp.Int('dense_unit', min_value=12, max_value=60, step=12)
    lr = hp.Float('lr', min_value=0.000001, max_value=0.00009, sampling='log')
    
    input_shape = tf.keras.Input(shape=(120,60))
    X = tf.keras.layers.Conv1D(filters=first_conv_filters, kernel_size=1, strides=1, padding='causal')(input_shape)
    X = tf.keras.layers.Conv1D(filters=second_conv_filters, kernel_size=second_conv_kernel_size, strides=second_conv_strides, padding='causal')(X)
    X = tf.keras.layers.Conv1D(filters=third_conv_filters, kernel_size=third_conv_kernel_size, strides=third_conv_strides, padding='causal')(X)
    X = tf.keras.layers.GRU(units=recurrent_unit, return_sequences=False)(X)
    X = tf.keras.layers.Dense(units=dense_unit, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_300_kt(hp):
    '''developed from model_263: 
    - additional C1D and recurrent layer
    - 2x more variable search range
    - widening lr range
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    c_filters_1 = hp.Int('c_filters_1', min_value=8, max_value=256, step=8)
    c_filters_2 = hp.Int('c_filters_2', min_value=16, max_value=256, step=16)
    c_kernel_2 = hp.Int('c_kernel_2', min_value=1, max_value=48, step=2)
    c_strides_2 = hp.Int('c_strides_2', min_value=1, max_value=10, step=1)
    c_filters_3 = hp.Int('c_filters_3', min_value=16, max_value=256, step=16)
    c_kernel_3 = hp.Int('c_kernel_3', min_value=1, max_value=48, step=3)
    c_strides_3 = hp.Int('c_strides_3', min_value=1, max_value=10, step=1)
    c_filters_4 = hp.Int('c_filters_4', min_value=16, max_value=256, step=16)
    c_kernel_4 = hp.Int('c_kernel_4', min_value=1, max_value=48, step=3)
    c_strides_4 = hp.Int('c_strides_4', min_value=1, max_value=10, step=1)
    r_unit_1 = hp.Int('r_unit_1', min_value=6, max_value=120, step=6)
    r_unit_2 = hp.Int('r_unit_2', min_value=6, max_value=120, step=6)
    d_unit_1 = hp.Int('d_unit_1', min_value=12, max_value=120, step=12)
    lr = hp.Float('lr', min_value=0.0000005, max_value=0.0001, sampling='log')
    
    input_shape = tf.keras.Input(shape=(120,60))
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=1, strides=1, padding='causal')(input_shape)
    X = tf.keras.layers.Conv1D(filters=c_filters_2, kernel_size=c_kernel_2, strides=c_strides_2, padding='causal')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_3, kernel_size=c_kernel_3, strides=c_strides_3, padding='causal')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_4, kernel_size=c_kernel_4, strides=c_strides_4, padding='causal')(X)
    X = tf.keras.layers.GRU(units=r_unit_1, return_sequences=True)(X)
    X = tf.keras.layers.GRU(units=r_unit_2, return_sequences=False)(X)
    X = tf.keras.layers.Dense(units=d_unit_1, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_301_kt(hp):
    '''developed from model_300: 
    - add batch normalization and relu activation between convolutional layer
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    c_filters_1 = hp.Int('c_filters_1', min_value=8, max_value=256, step=8)
    c_filters_2 = hp.Int('c_filters_2', min_value=16, max_value=256, step=16)
    c_kernel_2 = hp.Int('c_kernel_2', min_value=1, max_value=48, step=2)
    c_strides_2 = hp.Int('c_strides_2', min_value=1, max_value=10, step=1)
    c_filters_3 = hp.Int('c_filters_3', min_value=16, max_value=256, step=16)
    c_kernel_3 = hp.Int('c_kernel_3', min_value=1, max_value=48, step=3)
    c_strides_3 = hp.Int('c_strides_3', min_value=1, max_value=10, step=1)
    c_filters_4 = hp.Int('c_filters_4', min_value=16, max_value=256, step=16)
    c_kernel_4 = hp.Int('c_kernel_4', min_value=1, max_value=48, step=3)
    c_strides_4 = hp.Int('c_strides_4', min_value=1, max_value=10, step=1)
    r_unit_1 = hp.Int('r_unit_1', min_value=6, max_value=120, step=6)
    r_unit_2 = hp.Int('r_unit_2', min_value=6, max_value=120, step=6)
    d_unit_1 = hp.Int('d_unit_1', min_value=12, max_value=120, step=12)
    lr = hp.Float('lr', min_value=0.0000005, max_value=0.0001, sampling='log')
    
    input_shape = tf.keras.Input(shape=(120,60))
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=1, strides=1, padding='causal')(input_shape)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_2, kernel_size=c_kernel_2, strides=c_strides_2, padding='causal')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_3, kernel_size=c_kernel_3, strides=c_strides_3, padding='causal')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_4, kernel_size=c_kernel_4, strides=c_strides_4, padding='causal')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.GRU(units=r_unit_1, return_sequences=True)(X)
    X = tf.keras.layers.GRU(units=r_unit_2, return_sequences=False)(X)
    X = tf.keras.layers.Dense(units=d_unit_1, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_302_kt(hp):
    '''developed from model_301: 
    - add bypass and residual layer to increase generalization
        between layers
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    def bypass(X, X_bypass, operations):
        '''A conditional layer that let the optimizer
        choose to bypass, add, or concatenate'''
        # assert X.shape == X_bypass.shape
        if operations == 0:
            return X
        elif operations == 1:
            return tf.keras.layers.Add()([X, X_bypass])
        elif operations == 2:
            return tf.keras.layers.concatenate([X, X_bypass])
        
    bp_1 = hp.Choice('bp_1', [0,1])
    bp_2 = hp.Choice('bp_2', [0,1])
    bp_3 = hp.Choice('bp_3', [0,1])
    
    
    c_filters_1 = hp.Int('c_filters_1', min_value=8, max_value=256, step=8)
    c_kernel_1 = hp.Int('c_kernel_1', min_value=1, max_value=48, step=2)
    c_strides_1 = hp.Int('c_strides_1', min_value=1, max_value=4, step=1)
    c_kernel_2 = hp.Int('c_kernel_2', min_value=1, max_value=48, step=2)
    c_kernel_3 = hp.Int('c_kernel_3', min_value=1, max_value=48, step=3)
    c_kernel_4 = hp.Int('c_kernel_4', min_value=1, max_value=48, step=4)
    r_unit_1 = hp.Int('r_unit_1', min_value=6, max_value=120, step=6)
    r_unit_2 = hp.Int('r_unit_2', min_value=6, max_value=120, step=6)
    d_unit_1 = hp.Int('d_unit_1', min_value=12, max_value=120, step=12)
    lr = hp.Float('lr', min_value=0.0000005, max_value=0.0001, sampling='log')
    
    input_shape = tf.keras.Input(shape=(120,60))
    
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=c_kernel_1, strides=c_strides_1, padding='causal')(input_shape)
    X1 = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X1)
    
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=c_kernel_2, strides=1, padding='causal')(X)
    X2 = tf.keras.layers.BatchNormalization()(X)
    X = bypass(X2, X1, operations=bp_1)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=c_kernel_3, strides=1, padding='causal')(X)
    X3 = tf.keras.layers.BatchNormalization()(X)
    X = bypass(X3, X2, operations=bp_2)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=c_kernel_4, strides=1, padding='causal')(X)
    X4 = tf.keras.layers.BatchNormalization()(X)
    X = bypass(X4, X3, operations=bp_3)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.GRU(units=r_unit_1, return_sequences=True)(X)
    X = tf.keras.layers.GRU(units=r_unit_2, return_sequences=False)(X)
    X = tf.keras.layers.Dense(units=d_unit_1, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_303_kt(hp):
    '''developed from model_301: 
    - add bypass and residual layer to increase generalization
        between layers
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    def bypass(X, X_bypass, operations):
        '''A conditional layer that let the optimizer
        choose to bypass, add, or concatenate'''
        # assert X.shape == X_bypass.shape
        if operations == 0:
            return X
        elif operations == 1:
            return tf.keras.layers.Add()([X, X_bypass])
        elif operations == 2:
            print(X.shape, X_bypass.shape)
            return tf.keras.layers.Concatenate(axis=1)([X, X_bypass])
        
    bp_1 = hp.Choice('bp_1', [0,2])
    bp_2 = hp.Choice('bp_2', [0,2])
    bp_3 = hp.Choice('bp_3', [0,2])
    
    
    c_filters_1 = hp.Int('c_filters_1', min_value=8, max_value=256, step=8)
    c_kernel_2 = hp.Int('c_kernel_2', min_value=1, max_value=48, step=2)
    c_strides_2 = hp.Int('c_strides_2', min_value=1, max_value=4, step=1)
    c_kernel_3 = hp.Int('c_kernel_3', min_value=1, max_value=48, step=3)
    c_strides_3 = hp.Int('c_strides_3', min_value=1, max_value=4, step=1)
    c_kernel_4 = hp.Int('c_kernel_4', min_value=1, max_value=48, step=4)
    c_strides_4 = hp.Int('c_strides_4', min_value=1, max_value=4, step=1)
    r_unit_1 = hp.Int('r_unit_1', min_value=6, max_value=120, step=6)
    r_unit_2 = hp.Int('r_unit_2', min_value=6, max_value=120, step=6)
    d_unit_1 = hp.Int('d_unit_1', min_value=12, max_value=120, step=12)
    lr = hp.Float('lr', min_value=0.0000005, max_value=0.0001, sampling='log')
    
    input_shape = tf.keras.Input(shape=(120,60))
    
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=1, strides=1, padding='causal')(input_shape)
    X1 = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X1)
    
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=c_kernel_2, strides=c_strides_2, padding='causal')(X)
    X2 = tf.keras.layers.BatchNormalization()(X)
    X = bypass(X2, X1, operations=bp_1)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=c_kernel_3, strides=c_strides_3, padding='causal')(X)
    X3 = tf.keras.layers.BatchNormalization()(X)
    X = bypass(X3, X2, operations=bp_2)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=c_kernel_4, strides=c_strides_4, padding='causal')(X)
    X4 = tf.keras.layers.BatchNormalization()(X)
    X = bypass(X4, X3, operations=bp_3)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.GRU(units=r_unit_1, return_sequences=True)(X)
    X = tf.keras.layers.GRU(units=r_unit_2, return_sequences=False)(X)
    X = tf.keras.layers.Dense(units=d_unit_1, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_304_kt(hp):
    '''developed from model_301: 
    - add bypass and residual layer to increase generalization
        between layers
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''       
    def block(X, filters, recunits):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    b1_filters = hp.Int('b1_filters', min_value=4, max_value=16, step=4)
    b2_filters = hp.Int('b2_filters', min_value=16, max_value=64, step=16)
    b3_filters = hp.Int('b3_filters', min_value=4, max_value=32, step=8)
    b1b2_recunits = hp.Int('b1b2_recunits', min_value=8, max_value=64, step=8)
    b3_recunits = hp.Int('b3_recunits', min_value=4, max_value=32, step=8)
    final_recunits = hp.Int('final_recunits', min_value=4, max_value=16, step=4)
    final_dunits = hp.Int('final_dunits', min_value=4, max_value=16, step=4)
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    
    input_shape = tf.keras.Input(shape=(120,60))
    # Block 1
    X_bn_1, X_act_1 = block(input_shape, b1_filters, b1b2_recunits)
    # Block 2
    X_bn_2, X_act_2 = block(X_act_1, b2_filters, b1b2_recunits)
    # Addition block
    X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
    X = tf.keras.layers.Activation('relu')(X)
    # Block 3
    X_bn_3, X_act_3 = block(X, b3_filters, b3_recunits)
    # Final layer
    X = tf.keras.layers.GRU(units=final_recunits, return_sequences=False)(X_act_3)
    X = tf.keras.layers.Dense(units=final_dunits, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_305_kt(hp):
    '''developed from model_304: 
    - split data sequence into sequence using prime number strides
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''       
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits):
        b1_filters = hp.Int(f'comp{component_num}_b1_filters', min_value=4, max_value=16, step=4)
        b2_filters = hp.Int(f'comp{component_num}_b2_filters', min_value=16, max_value=64, step=16)
        b3_filters = hp.Int(f'comp{component_num}_b3_filters', min_value=4, max_value=32, step=8)
        b1b2_recunits = hp.Int(f'comp{component_num}_1b2_recunits', min_value=8, max_value=64, step=8)
        b3_recunits = hp.Int(f'comp{component_num}_b3_recunits', min_value=4, max_value=32, step=8)
        final_recunits = hp.Int(f'comp{component_num}_final_recunits', min_value=4, max_value=16, step=4)
        # final_dunits = hp.Int(f'comp{component_num}_final_dunits', min_value=4, max_value=16, step=4)
        # Block 1
        X_bn_1, X_act_1 = block(input_shape, b1_filters, b1b2_recunits, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, b2_filters, b1b2_recunits)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, b3_filters, b3_recunits)
        # Final layer
        X = tf.keras.layers.GRU(units=final_recunits, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits, activation='relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    superblock_final_dunits = hp.Int('superblock_final_dunits', min_value=4, max_value=16, step=4)
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    input_shape = tf.keras.Input(shape=(recurrents,60))
    
    n_components = generate_prime(recurrents)
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_306_kt(hp):
    '''developed from model_305: 
    - add component_strides to reduce number of components,
        thus reducing model complexity (the 305 ver take 5 minutes
        to compile)
    - add batch normalization in the end of superblock
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''       
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits):
        b1_filters = hp.Int(f'comp{component_num}_b1_filters', min_value=4, max_value=16, step=4)
        b2_filters = hp.Int(f'comp{component_num}_b2_filters', min_value=16, max_value=64, step=16)
        b3_filters = hp.Int(f'comp{component_num}_b3_filters', min_value=4, max_value=32, step=8)
        b1b2_recunits = hp.Int(f'comp{component_num}_1b2_recunits', min_value=8, max_value=64, step=8)
        b3_recunits = hp.Int(f'comp{component_num}_b3_recunits', min_value=4, max_value=32, step=8)
        final_recunits = hp.Int(f'comp{component_num}_final_recunits', min_value=4, max_value=16, step=4)
        # final_dunits = hp.Int(f'comp{component_num}_final_dunits', min_value=4, max_value=16, step=4)
        # Block 1
        X_bn_1, X_act_1 = block(input_shape, b1_filters, b1b2_recunits, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, b2_filters, b1b2_recunits)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, b3_filters, b3_recunits)
        # Final layer
        X = tf.keras.layers.GRU(units=final_recunits, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 3
    superblock_final_dunits = hp.Int('superblock_final_dunits', min_value=4, max_value=16, step=4)
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    input_shape = tf.keras.Input(shape=(recurrents,60))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_307_kt(hp):
    '''developed from model_306: 
    - Keep component_strides=1, but maintaining complexity of
        each superblock by considering their total number of
        constituent.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hp.Int(f'comp{component_num}', min_value=minv, max_value=maxv, step=step)
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hp.Int('superblock_final_dunits', min_value=2, max_value=6, step=2)
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    input_shape = tf.keras.Input(shape=(recurrents,60))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_308_kt(hp):
    '''developed from model_307: 
    - Stacking superblock until the number of recurrent
        sequence before the final node reach threshold.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(super_component_num, input_shape, component_num, final_dunits, recurrents, fraction=0.75, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hp.Int(f'{super_component_num}_comp{component_num}', min_value=minv, max_value=maxv, step=step)
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    def superblock_component(input_shape, super_component_num):
        component_strides = 1
        superblock_final_dunits = hp.Int(f'{super_component_num}_superblock_final_dunits', min_value=2, max_value=6, step=2)
        
        n_components = generate_prime(int(input_shape.shape[1] // component_strides))
        component_results = []
        for component_num in n_components:
            superblock_result = superblock(super_component_num, input_shape, component_num, superblock_final_dunits, input_shape.shape[1])
            component_results.append(superblock_result)

        # reverse component_result so the highest stride come first
        component_results = component_results[::-1]
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    def stack_superblock(input_shape, threshold=10):
        # Define initial state
        # This initial state would be re-weitten in each
        # loop
        current_recurrent = input_shape.shape[1]
        X = input_shape
        count_component = 0
        while current_recurrent > threshold:
            X = superblock_component(X, count_component)
            current_recurrent = X.shape[1]
            count_component+=1
        return X
            
        
    recurrents = 120
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    
    input_shape = tf.keras.Input(shape=(recurrents,60))
    X = stack_superblock(input_shape, threshold=10)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_309_kt(hp):
    '''developed from model_307, optimized from model_308: 
    - Stacking superblock until the number of recurrent
        sequence before the final node reach threshold.
    - using threshold=2 and fraction=1. Compared to
        threshold=10 and fraction=0.75 in model_308.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(super_component_num, input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hp.Int(f'{super_component_num}_comp{component_num}', min_value=minv, max_value=maxv, step=step)
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    def superblock_component(input_shape, super_component_num):
        component_strides = 1
        superblock_final_dunits = hp.Int(f'{super_component_num}_superblock_final_dunits', min_value=2, max_value=6, step=2)
        
        n_components = generate_prime(int(input_shape.shape[1] // component_strides))
        component_results = []
        for component_num in n_components:
            superblock_result = superblock(super_component_num, input_shape, component_num, superblock_final_dunits, input_shape.shape[1])
            component_results.append(superblock_result)

        # reverse component_result so the highest stride come first
        component_results = component_results[::-1]
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    def stack_superblock(input_shape, threshold=2):
        # Define initial state
        # This initial state would be re-weitten in each
        # loop
        current_recurrent = input_shape.shape[1]
        X = input_shape
        count_component = 0
        while current_recurrent > threshold:
            X = superblock_component(X, count_component)
            current_recurrent = X.shape[1]
            count_component+=1
        return X
        
    recurrents = 120
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    
    input_shape = tf.keras.Input(shape=(recurrents,60))
    X = stack_superblock(input_shape, threshold=2)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_310_kt(hp):
    '''developed from model_307: 
    - 2-3x more layer units (with hope to increase performance)
        fraction 1 -> 3
        minimum_value 4 -> 8
        superblock_final_dunits 2/6/2 -> 8/24/4
        bdRNN 8 -> 32
        dense 8 -> 32
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=3, floor_fraction=0.25, num_steps=4, minimum_value=8):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hp.Int(f'comp{component_num}', min_value=minv, max_value=maxv, step=step)
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hp.Int('superblock_final_dunits', min_value=8, max_value=24, step=4)
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    input_shape = tf.keras.Input(shape=(recurrents,60))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=32))(X)
    X = tf.keras.layers.Dense(units=32, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_311_kt(hp):
    '''Recurrent preprocessor model.
    - Every features from index 0 -> len(features) - 1
        processed individually, and then concatenated with
        other proprocessed features.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    def recurrent_preprocessor_component(X, feature_no, rnnu_comp):
        '''Version 1: using single LSTM layer to preprocess'''
        X = tf.keras.layers.LSTM(rnnu_comp, return_sequences=False)(X[:,:,feature_no:feature_no+1])
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    def recurrent_preprocessor(input_shape, features, min_value=4, max_value=10, step=2):
        rnnu_comp = hp.Int(f'rnnu_comp', min_value=min_value, max_value=max_value, step=step)
        component_results = []
        for feature_no in range(features):
            X = recurrent_preprocessor_component(input_shape, feature_no, rnnu_comp)
            component_results.append(X)
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    final_rnnu = hp.Int('final_rnnu', min_value=8, max_value=48, step=8)
    final_denseu = hp.Int('final_denseu', min_value=8, max_value=48, step=8)
    
    recurrents = 120
    features = 60
    input_shape = tf.keras.Input(shape=(recurrents, features))
    X = recurrent_preprocessor(input_shape, features)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=final_rnnu))(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model 

def model_312_kt(hp):
    '''Recurrent preprocessor model.
    - Every features from index 0 -> len(features) - 1
        processed individually, and then concatenated with
        other proprocessed features.
    
    Rev from model_311:
    - Remove final rnn layers. Early training show shortcoming in acc
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    def recurrent_preprocessor_component(X, feature_no, rnnu_comp):
        '''Version 1: using single LSTM layer to preprocess'''
        X = tf.keras.layers.LSTM(rnnu_comp, return_sequences=False)(X[:,:,feature_no:feature_no+1])
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    def recurrent_preprocessor(input_shape, features, min_value=4, max_value=10, step=2):
        rnnu_comp = hp.Int(f'rnnu_comp', min_value=min_value, max_value=max_value, step=step)
        component_results = []
        for feature_no in range(features):
            X = recurrent_preprocessor_component(input_shape, feature_no, rnnu_comp)
            component_results.append(X)
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    final_denseu = hp.Int('final_denseu', min_value=8, max_value=48, step=8)
    
    recurrents = 120
    features = 60
    input_shape = tf.keras.Input(shape=(recurrents, features))
    X = recurrent_preprocessor(input_shape, features)
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_313_kt(hp):
    '''Recurrent preprocessor model.
    - Every features from index 0 -> len(features) - 1
        processed individually, and then concatenated with
        other proprocessed features.
    
    Rev from model_311:
    - Remove final rnn layers. Early training show shortcoming in acc
    
    Rev from model_312:
    - Early training show even worse loss: 0.9.
    - Try to stack conv1d 1 kernel
    - add model.summary()
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    def recurrent_preprocessor_component(X, feature_no, rnnu_comp):
        '''Version 1: using single LSTM layer to preprocess'''
        X = tf.keras.layers.LSTM(rnnu_comp, return_sequences=False)(X[:,:,feature_no:feature_no+1])
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    def recurrent_preprocessor(input_shape, features, min_value=4, max_value=10, step=2):
        rnnu_comp = hp.Int(f'rnnu_comp', min_value=min_value, max_value=max_value, step=step)
        component_results = []
        for feature_no in range(features):
            X = recurrent_preprocessor_component(input_shape, feature_no, rnnu_comp)
            component_results.append(X)
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    final_denseu = hp.Int('final_denseu', min_value=8, max_value=48, step=8)
    final_c1d_1_filters = hp.Int('final_c1d_1_filters', min_value=4, max_value=8, step=2)
    final_c1d_2_filters = hp.Int('final_c1d_2_filters', min_value=4, max_value=16, step=4)
    final_c1d_2_kernels = hp.Int('final_c1d_2_kernels', min_value=1, max_value=7, step=1)
    final_c1d_2_strides = hp.Int('final_c1d_2_strides', min_value=1, max_value=3, step=1)
    final_c1d_3_filters = hp.Int('final_c1d_3_filters', min_value=4, max_value=16, step=4)
    final_c1d_3_kernels = hp.Int('final_c1d_3_kernels', min_value=1, max_value=7, step=1)
    final_c1d_3_strides = hp.Int('final_c1d_3_strides', min_value=1, max_value=3, step=1)
    
    recurrents = 120
    features = 60
    input_shape = tf.keras.Input(shape=(recurrents, features))
    X = recurrent_preprocessor(input_shape, features)
    # Stack multiple conv1d
    X = tf.keras.layers.Conv1D(filters=final_c1d_1_filters, kernel_size=1, strides=1)(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=final_c1d_2_filters, kernel_size=final_c1d_2_kernels, strides=final_c1d_2_strides)(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=final_c1d_3_filters, kernel_size=final_c1d_3_kernels, strides=final_c1d_3_strides)(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_314_kt(hp):
    '''developed from model_307: 
    - Stacking superblock component together like block that stacked together.
    - Stacking difference with 308 & 309:
        - This 314 stack component before concatenation
            of superblock
        - The final concatenation is the same as 307, but
            with added component before that concatenation.
        - Desired effect:
            - Its possible that concatenation in 307 already
                abstract enough that additional abstraction
                just made the performance and learning curve
                going down.
            - Abstraction in prime component may give boost in
                performance by outputting more distinguishable
                value to later layer without shuffling the
                prime components even more.
            - This model also verify if stacking conv+rnn
                is possible. If not, try to stack more
                conv layer only to the component (next model).
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock_component(X, superblock_hyp, component_num, first_node=True, final_node=False):
        X_bn_1, X_act_1 = block(X, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num) if first_node else 1)
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=not final_node)(X_act_3)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)        
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hp.Int(f'comp{component_num}', min_value=minv, max_value=maxv, step=step)
        
        X_bn_1, X_act_1 = superblock_component(input_shape, superblock_hyp, component_num, first_node=True, final_node=False)
        X_bn_2, X_act_2 = superblock_component(X_act_1, superblock_hyp, component_num, first_node=False, final_node=False)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        X_bn_3, X_act_3 = superblock_component(X, superblock_hyp, component_num, first_node=False, final_node=True)
        
        X = tf.keras.layers.Dense(units=final_dunits)(X_act_3)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hp.Int('superblock_final_dunits', min_value=2, max_value=6, step=2)
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    input_shape = tf.keras.Input(shape=(recurrents,60))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_315_kt(hp):
    '''developed from model_307: 
    - Difference with 307:
        - Add conv stack and additive layer in the superblock
        - Why?
            - Early kt search with 314 show very long model
                compilation although the network not complex enough
                in terms of parameters.
            - Need faster iteration between model
            - even if the model 314 is better compared to 307 & 310,
                comparable model with faster training time is needed.
            - It's still unknown wheter 314 complexity is enough to
                drive lower val_loss. Using this model variation as
                comparison can lower iteration time.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hp.Int(f'comp{component_num}', min_value=minv, max_value=maxv, step=step)
        
        # Block 1
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Block 1 + 2: Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Block 4
        X_bn_4, X_act_4 = block(X_act_3, superblock_hyp, superblock_hyp)
        # Block 3 + 4: Addition block
        X = tf.keras.layers.Add()([X_bn_3, X_bn_4])
        X = tf.keras.layers.Activation('relu')(X)
        
        # Block 5
        X_bn_5, X_act_5 = block(X, superblock_hyp, superblock_hyp)
        
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hp.Int('superblock_final_dunits', min_value=2, max_value=6, step=2)
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    input_shape = tf.keras.Input(shape=(recurrents,60))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_316_kt(hp):
    '''developed from model_314: 
    - Evaluation after ~2 epochs of training:
        - The gradient descent really slow, although
            the accuracy is superb with those high losses.
        - It's possible that the networks are to deep
            to significantly update earlier params.
        - Correction:
            - Remove third layer in superblock
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock_component(X, superblock_hyp, component_num, first_node=True, final_node=False):
        X_bn_1, X_act_1 = block(X, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num) if first_node else 1)
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=not final_node)(X_act_3)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)        
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hp.Int(f'comp{component_num}', min_value=minv, max_value=maxv, step=step)
        
        X_bn_1, X_act_1 = superblock_component(input_shape, superblock_hyp, component_num, first_node=True, final_node=False)
        X_bn_2, X_act_2 = superblock_component(X_act_1, superblock_hyp, component_num, first_node=False, final_node=False)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        
        X = tf.keras.layers.Flatten()(X)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hp.Int('superblock_final_dunits', min_value=2, max_value=6, step=2)
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    input_shape = tf.keras.Input(shape=(recurrents,60))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=2, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_317_kt(hp):
    '''Benchmark model for multiple group size.
    Consist of:
    - recurrent + dense + concatenate + dense
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    # Keras tuner parameters
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    final_denseu = hp.Int('final_denseu', min_value=8, max_value=32, step=8)
    recu = hp.Int('recu', min_value=16, max_value=48, step=16)
    denseu = hp.Int('denseu', min_value=8, max_value=16, step=8)
    
    recurrents = 120
    base_features = 60
    group_size = 2
    
    input_shape = tf.keras.Input(shape=(recurrents,base_features*group_size))
    component_results = []
    for i in range(group_size):
        
        # Do recurrent + Dense
        input_slice = input_shape[:,:,base_features*i:base_features*(i+1)+1]
        X = tf.keras.layers.LSTM(recu, return_sequences=False)(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dense(units=denseu)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.expand_dims(X, axis=1)
        component_results.append(X)
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dense(units=final_denseu)(X)
    X = tf.keras.layers.Flatten()(X)
    
    outputs = tf.keras.layers.Dense(units=group_size+1, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_317_kt_build(hp, a, b):
    '''Benchmark model for multiple group size.
    Consist of:
    - recurrent + dense + concatenate + dense
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    keys = ['lr', 'final_denseu', 'recu', 'denseu']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    # Keras tuner parameters
    lr = hyperparameters['lr']
    final_denseu = hyperparameters['final_denseu']
    recu = hyperparameters['recu']
    denseu = hyperparameters['denseu']
    
    recurrents = 120
    base_features = 60
    group_size = 2
    
    input_shape = tf.keras.Input(shape=(recurrents,base_features*group_size))
    component_results = []
    for i in range(group_size):
        
        # Do recurrent + Dense
        input_slice = input_shape[:,:,base_features*i:base_features*(i+1)+1]
        X = tf.keras.layers.LSTM(recu, return_sequences=False)(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dense(units=denseu)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.expand_dims(X, axis=1)
        component_results.append(X)
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dense(units=final_denseu)(X)
    X = tf.keras.layers.Flatten()(X)
    
    outputs = tf.keras.layers.Dense(units=group_size+1, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_318_kt(hp):
    '''Deeper version of 317:
    - Add 2 conv layer before and after recurrent
    - add activation in final dense unit
    - Recurrent return_sequence=True
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    # Keras tuner parameters
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    final_denseu = hp.Int('final_denseu', min_value=8, max_value=32, step=8)
    recu = hp.Int('recu', min_value=16, max_value=48, step=16)
    denseu = hp.Int('denseu', min_value=8, max_value=16, step=8)
    comp_filters = hp.Int('comp_filters', min_value=4, max_value=32, step=4)
    
    recurrents = 120
    base_features = 60
    group_size = 2
    
    input_shape = tf.keras.Input(shape=(recurrents,base_features*group_size))
    component_results = []
    for i in range(group_size):
        
        # Do recurrent + Dense
        input_slice = input_shape[:,:,base_features*i:base_features*(i+1)+1]
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dense(units=denseu)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.expand_dims(X, axis=1)
        component_results.append(X)
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    X = tf.keras.layers.Flatten()(X)
    
    outputs = tf.keras.layers.Dense(units=group_size+1, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_318_kt_build(hp, a, b):
    '''Deeper version of 317:
    - Add 2 conv layer before and after recurrent
    - add activation in final dense unit
    - Recurrent return_sequence=True
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    keys = ['lr', 'final_denseu', 'recu', 'denseu','comp_filters']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    # Keras tuner parameters
    lr = hyperparameters['lr']
    final_denseu = hyperparameters['final_denseu']
    recu = hyperparameters['recu']
    denseu = hyperparameters['denseu']
    comp_filters = hyperparameters['comp_filters']
    
    recurrents = 120
    base_features = 60
    group_size = 2
    
    input_shape = tf.keras.Input(shape=(recurrents,base_features*group_size))
    component_results = []
    for i in range(group_size):
        
        # Do recurrent + Dense
        input_slice = input_shape[:,:,base_features*i:base_features*(i+1)+1]
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Dense(units=denseu)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.expand_dims(X, axis=1)
        component_results.append(X)
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    X = tf.keras.layers.Flatten()(X)
    
    outputs = tf.keras.layers.Dense(units=group_size+1, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_319_kt(hp):
    '''Modified from model 318
    - 3 recurrent component
    - addition block before final recurrent component
    - remove batch normalization after component dense unit
    - add activation after component dense unit
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    # Keras tuner parameters
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    final_denseu = hp.Int('final_denseu', min_value=8, max_value=32, step=8)
    recu = hp.Int('recu', min_value=16, max_value=48, step=16)
    denseu = hp.Int('denseu', min_value=8, max_value=16, step=8)
    comp_filters = hp.Int('comp_filters', min_value=4, max_value=32, step=4)
    
    recurrents = 120
    base_features = 60
    group_size = 2
    
    input_shape = tf.keras.Input(shape=(recurrents,base_features*group_size))
    component_results = []
    for i in range(group_size):
        
        # Do recurrent + Dense
        input_slice = input_shape[:,:,base_features*i:base_features*(i+1)+1]
        # 1st component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn1 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn1)
        
        # 2nd component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn2 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn2)
        
        # 3rd component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
    
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn3 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn3)  
        
        # Addition layer
        X = tf.keras.layers.Add()([X_bn1, X_bn2, X_bn3])
        X = tf.keras.layers.Activation('relu')(X)
        
        # Final recurrent in component
        X = tf.keras.layers.LSTM(recu, return_sequences=False)(X)
        
        X = tf.keras.layers.Dense(units=denseu)(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.expand_dims(X, axis=1)
        component_results.append(X)
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    X = tf.keras.layers.Flatten()(X)
    
    outputs = tf.keras.layers.Dense(units=group_size+1, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_319_kt_build(hp, a, b):
    '''Modified from model 318
    - 3 recurrent component
    - addition block before final recurrent component
    - remove batch normalization after component dense unit
    - add activation after component dense unit
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    keys = ['lr', 'final_denseu', 'recu', 'denseu','comp_filters']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    # Keras tuner parameters
    lr = hyperparameters['lr']
    final_denseu = hyperparameters['final_denseu']
    recu = hyperparameters['recu']
    denseu = hyperparameters['denseu']
    comp_filters = hyperparameters['comp_filters']
    
    recurrents = 120
    base_features = 60
    group_size = 2
    
    input_shape = tf.keras.Input(shape=(recurrents,base_features*group_size))
    component_results = []
    for i in range(group_size):
        
        # Do recurrent + Dense
        input_slice = input_shape[:,:,base_features*i:base_features*(i+1)+1]
        # 1st component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn1 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn1)
        
        # 2nd component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn2 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn2)
        
        # 3rd component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_slice)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
    
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn3 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn3)  
        
        # Addition layer
        X = tf.keras.layers.Add()([X_bn1, X_bn2, X_bn3])
        X = tf.keras.layers.Activation('relu')(X)
        
        # Final recurrent in component
        X = tf.keras.layers.LSTM(recu, return_sequences=False)(X)
        
        X = tf.keras.layers.Dense(units=denseu)(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.expand_dims(X, axis=1)
        component_results.append(X)
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    X = tf.keras.layers.Flatten()(X)
    
    outputs = tf.keras.layers.Dense(units=group_size+1, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_320_kt(hp):
    '''Modified from model 319
    - !! FIX 
        - flaw in input slice algorithm: remove `+1` in slicing formula.
        - Negligence that all Conv1D layer receive `input_slice` as
            previous layer. 
    - Add 3 recurrent preprocessor for input,
        and add them up as second layer component input.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    # Keras tuner parameters
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    final_denseu = hp.Int('final_denseu', min_value=8, max_value=32, step=8)
    recu = hp.Int('recu', min_value=16, max_value=48, step=16)
    denseu = hp.Int('denseu', min_value=8, max_value=16, step=8)
    comp_filters = hp.Int('comp_filters', min_value=4, max_value=32, step=4)
    
    recurrents = 120
    base_features = 60
    group_size = 2
    input_preprocessors = 3
    processor_features = base_features / input_preprocessors
    
    input_shape = tf.keras.Input(shape=(recurrents,base_features*group_size))
    component_results = []
    for i in range(group_size):
        input_preprocessor_results = []
        for j in range(input_preprocessors):
            input_preprocessor = input_shape[:,:,int(base_features*i + j*processor_features):int(base_features*i + (j+1)*processor_features)]
            # 1st component layer
            X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_preprocessor)
            X = tf.keras.layers.BatchNormalization()(X)
            X = tf.keras.layers.Activation('relu')(X)
            X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
            X = tf.keras.layers.BatchNormalization()(X)
            X = tf.keras.layers.Activation('relu')(X)

            X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
            X = tf.keras.layers.BatchNormalization()(X)
            input_preprocessor_results.append(X)
        
        X_bn1 = tf.keras.layers.Add()(input_preprocessor_results)        
        X = tf.keras.layers.Activation('relu')(X_bn1)
        
        # 2nd component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn2 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn2)
        
        # 3rd component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
    
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn3 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn3)  
        
        # Addition layer
        X = tf.keras.layers.Add()([X_bn1, X_bn2, X_bn3])
        X = tf.keras.layers.Activation('relu')(X)
        
        # Final recurrent in component
        X = tf.keras.layers.LSTM(recu, return_sequences=False)(X)
        
        X = tf.keras.layers.Dense(units=denseu)(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.expand_dims(X, axis=1)
        component_results.append(X)
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    X = tf.keras.layers.Flatten()(X)
    
    outputs = tf.keras.layers.Dense(units=group_size+1, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_320_kt_build(hp, a, b):
    '''Modified from model 319
    - !! FIX 
        - flaw in input slice algorithm: remove `+1` in slicing formula.
        - Negligence that all Conv1D layer receive `input_slice` as
            previous layer. 
    - Add 3 recurrent preprocessor for input,
        and add them up as second layer component input.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    keys = ['lr', 'final_denseu', 'recu', 'denseu','comp_filters']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    # Keras tuner parameters
    lr = hyperparameters['lr']
    final_denseu = hyperparameters['final_denseu']
    recu = hyperparameters['recu']
    denseu = hyperparameters['denseu']
    comp_filters = hyperparameters['comp_filters']
    
    recurrents = 120
    base_features = 60
    group_size = 2
    input_preprocessors = 3
    processor_features = base_features / input_preprocessors
    
    input_shape = tf.keras.Input(shape=(recurrents,base_features*group_size))
    component_results = []
    for i in range(group_size):
        input_preprocessor_results = []
        for j in range(input_preprocessors):
            input_preprocessor = input_shape[:,:,int(base_features*i + j*processor_features):int(base_features*i + (j+1)*processor_features)]
            # 1st component layer
            X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_preprocessor)
            X = tf.keras.layers.BatchNormalization()(X)
            X = tf.keras.layers.Activation('relu')(X)
            X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
            X = tf.keras.layers.BatchNormalization()(X)
            X = tf.keras.layers.Activation('relu')(X)

            X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
            X = tf.keras.layers.BatchNormalization()(X)
            input_preprocessor_results.append(X)
        
        X_bn1 = tf.keras.layers.Add()(input_preprocessor_results)        
        X = tf.keras.layers.Activation('relu')(X_bn1)
        
        # 2nd component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn2 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn2)
        
        # 3rd component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
    
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn3 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn3)  
        
        # Addition layer
        X = tf.keras.layers.Add()([X_bn1, X_bn2, X_bn3])
        X = tf.keras.layers.Activation('relu')(X)
        
        # Final recurrent in component
        X = tf.keras.layers.LSTM(recu, return_sequences=False)(X)
        
        X = tf.keras.layers.Dense(units=denseu)(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.expand_dims(X, axis=1)
        component_results.append(X)
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    X = tf.keras.layers.Flatten()(X)
    
    outputs = tf.keras.layers.Dense(units=group_size+1, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_321_kt(hp):
    '''Modified from model 320
    - Randomly change ticker processor A-B / B-A
        to increase model generalization.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    # Keras tuner parameters
    lr = hp.Float('lr', min_value=0.000001, max_value=0.0001, sampling='log')
    final_denseu = hp.Int('final_denseu', min_value=8, max_value=32, step=8)
    recu = hp.Int('recu', min_value=16, max_value=48, step=16)
    denseu = hp.Int('denseu', min_value=8, max_value=16, step=8)
    comp_filters = hp.Int('comp_filters', min_value=4, max_value=32, step=4)
    
    recurrents = 120
    base_features = 60
    group_size = 2
    input_preprocessors = 3
    processor_features = base_features / input_preprocessors
    
    position = [[0,1],[1,0]]
    choice = [0,1]
    np.random.choice(choice)
    
    input_shape = tf.keras.Input(shape=(recurrents,base_features*group_size))
    component_results = []
    for i in range(group_size):
        preprocessor_choice = np.random.choice(choice)
        assigned_processor = position[preprocessor_choice]
        input_preprocessor_results = []
        for j in range(input_preprocessors):
            input_preprocessor = input_shape[:,:,int(base_features*i + j*processor_features):int(base_features*i + (j+1)*processor_features)]
            # 1st component layer
            X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(input_preprocessor)
            X = tf.keras.layers.BatchNormalization()(X)
            X = tf.keras.layers.Activation('relu')(X)
            X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
            X = tf.keras.layers.BatchNormalization()(X)
            X = tf.keras.layers.Activation('relu')(X)

            X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
            X = tf.keras.layers.BatchNormalization()(X)
            input_preprocessor_results.append(X)
        
        X_bn1 = tf.keras.layers.Add()(input_preprocessor_results)        
        X = tf.keras.layers.Activation('relu')(X_bn1)
        
        # 2nd component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn2 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn2)
        
        # 3rd component layer
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.Conv1D(filters=comp_filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
    
        X = tf.keras.layers.LSTM(recu, return_sequences=True)(X)
        X_bn3 = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X_bn3)  
        
        # Addition layer
        X = tf.keras.layers.Add()([X_bn1, X_bn2, X_bn3])
        X = tf.keras.layers.Activation('relu')(X)
        
        # Final recurrent in component
        X = tf.keras.layers.LSTM(recu, return_sequences=False)(X)
        
        X = tf.keras.layers.Dense(units=denseu)(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.expand_dims(X, axis=1)
        component_results.append(X)
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    X = tf.keras.layers.Flatten()(X)
    
    outputs = tf.keras.layers.Dense(units=group_size+1, activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_263_kt_build(hp, train_inputs, train_labels):
    first_conv_filters, second_conv_filters, second_conv_kernel_size, second_conv_strides, third_conv_filters, third_conv_kernel_size, third_conv_strides, recurrent_unit, dense_unit, lr = hp
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    X = tf.keras.layers.Conv1D(filters=first_conv_filters, kernel_size=1, strides=1, padding='causal')(input_shape)
    X = tf.keras.layers.Conv1D(filters=second_conv_filters, kernel_size=second_conv_kernel_size, strides=second_conv_strides, padding='causal')(X)
    X = tf.keras.layers.Conv1D(filters=third_conv_filters, kernel_size=third_conv_kernel_size, strides=third_conv_strides, padding='causal')(X)
    X = tf.keras.layers.GRU(units=recurrent_unit, return_sequences=False)(X)
    X = tf.keras.layers.Dense(units=dense_unit, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_300_kt_build(hp, train_inputs, train_labels):
    '''developed from model_263: 
    - additional C1D and recurrent layer
    - 2x more variable search range
    - widening lr range
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    c_filters_1, c_filters_2, c_kernel_2, c_strides_2, c_filters_3, c_kernel_3, c_strides_3, c_filters_4, c_kernel_4, c_strides_4, r_unit_1, r_unit_2, d_unit_1, lr = hp
    
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=1, strides=1, padding='causal')(input_shape)
    X = tf.keras.layers.Conv1D(filters=c_filters_2, kernel_size=c_kernel_2, strides=c_strides_2, padding='causal')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_3, kernel_size=c_kernel_3, strides=c_strides_3, padding='causal')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_4, kernel_size=c_kernel_4, strides=c_strides_4, padding='causal')(X)
    X = tf.keras.layers.GRU(units=r_unit_1, return_sequences=True)(X)
    X = tf.keras.layers.GRU(units=r_unit_2, return_sequences=False)(X)
    X = tf.keras.layers.Dense(units=d_unit_1, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_301_kt_build(hp, train_inputs, train_labels):
    '''developed from model_300: 
    - add batch normalization and relu activation between convolutional layer
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    c_filters_1, c_filters_2, c_kernel_2, c_strides_2, c_filters_3, c_kernel_3, c_strides_3, c_filters_4, c_kernel_4, c_strides_4, r_unit_1, r_unit_2, d_unit_1, lr = hp
    
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    X = tf.keras.layers.Conv1D(filters=c_filters_1, kernel_size=1, strides=1, padding='causal')(input_shape)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_2, kernel_size=c_kernel_2, strides=c_strides_2, padding='causal')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_3, kernel_size=c_kernel_3, strides=c_strides_3, padding='causal')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=c_filters_4, kernel_size=c_kernel_4, strides=c_strides_4, padding='causal')(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.GRU(units=r_unit_1, return_sequences=True)(X)
    X = tf.keras.layers.GRU(units=r_unit_2, return_sequences=False)(X)
    X = tf.keras.layers.Dense(units=d_unit_1, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_307_kt_build(hp, train_inputs, train_labels):
    '''developed from model_306: 
    - Keep component_strides=1, but maintaining complexity of
        each superblock by considering their total number of
        constituent.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    
    keys = ['superblock_final_dunits', 'lr', 'comp2', 'comp3', 'comp5', 'comp7', 'comp11', 'comp13', 'comp17', 'comp19', 'comp23', 'comp29', 'comp31', 'comp37', 'comp41', 'comp43', 'comp47', 'comp53', 'comp59', 'comp61', 'comp67', 'comp71', 'comp73', 'comp79', 'comp83', 'comp89', 'comp97', 'comp101', 'comp103', 'comp107', 'comp109', 'comp113']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hyperparameters[f'comp{component_num}']
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hyperparameters['superblock_final_dunits']
    lr = hyperparameters['lr']
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_308_kt_build(hp, train_inputs, train_labels):
    '''developed from model_307: 
    - Stacking superblock until the number of recurrent
        sequence before the final node reach threshold.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    keys = ['lr', '0_superblock_final_dunits', '0_comp2', '0_comp3', '0_comp5', '0_comp7', '0_comp11', '0_comp13', '0_comp17', '0_comp19', '0_comp23', '0_comp29', '0_comp31', '0_comp37', '0_comp41', '0_comp43', '0_comp47', '0_comp53', '0_comp59', '0_comp61', '0_comp67', '0_comp71', '0_comp73', '0_comp79', '0_comp83', '0_comp89', '0_comp97', '0_comp101', '0_comp103', '0_comp107', '0_comp109', '0_comp113', '1_superblock_final_dunits', '1_comp2', '1_comp3', '1_comp5', '1_comp7', '1_comp11', '1_comp13', '1_comp17', '1_comp19', '1_comp23', '1_comp29']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(super_component_num, input_shape, component_num, final_dunits, recurrents, fraction=0.75, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hyperparameters[f'{super_component_num}_comp{component_num}']
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    def superblock_component(input_shape, super_component_num):
        component_strides = 1
        superblock_final_dunits = hyperparameters[f'{super_component_num}_superblock_final_dunits']
        
        n_components = generate_prime(int(input_shape.shape[1] // component_strides))
        component_results = []
        for component_num in n_components:
            superblock_result = superblock(super_component_num, input_shape, component_num, superblock_final_dunits, input_shape.shape[1])
            component_results.append(superblock_result)

        # reverse component_result so the highest stride come first
        component_results = component_results[::-1]
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    def stack_superblock(input_shape, threshold=10):
        # Define initial state
        # This initial state would be re-weitten in each
        # loop
        current_recurrent = input_shape.shape[1]
        X = input_shape
        count_component = 0
        while current_recurrent > threshold:
            X = superblock_component(X, count_component)
            current_recurrent = X.shape[1]
            count_component+=1
        return X
        
    recurrents = 120
    lr = hyperparameters['lr']
    
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    X = stack_superblock(input_shape, threshold=10)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_309_kt_build(hp, train_inputs, train_labels):
    '''developed from model_307, optimized from model_308: 
    - Stacking superblock until the number of recurrent
        sequence before the final node reach threshold.
    - using threshold=2 and fraction=1. Compared to
        threshold=10 and fraction=0.75 in model_308.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)''' 
    
    keys = ['lr', '0_superblock_final_dunits', '0_comp2', '0_comp3', '0_comp5', '0_comp7', '0_comp11', '0_comp13', '0_comp17', '0_comp19', '0_comp23', '0_comp29', '0_comp31', '0_comp37', '0_comp41', '0_comp43', '0_comp47', '0_comp53', '0_comp59', '0_comp61', '0_comp67', '0_comp71', '0_comp73', '0_comp79', '0_comp83', '0_comp89', '0_comp97', '0_comp101', '0_comp103', '0_comp107', '0_comp109', '0_comp113', '1_superblock_final_dunits', '1_comp2', '1_comp3', '1_comp5', '1_comp7', '1_comp11', '1_comp13', '1_comp17', '1_comp19', '1_comp23', '1_comp29', '2_superblock_final_dunits', '2_comp2', '2_comp3', '2_comp5', '2_comp7', '3_superblock_final_dunits', '3_comp2', '3_comp3']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(super_component_num, input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hyperparameters[f'{super_component_num}_comp{component_num}']
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    def superblock_component(input_shape, super_component_num):
        component_strides = 1
        superblock_final_dunits = hyperparameters[f'{super_component_num}_superblock_final_dunits']
        
        n_components = generate_prime(int(input_shape.shape[1] // component_strides))
        component_results = []
        for component_num in n_components:
            superblock_result = superblock(super_component_num, input_shape, component_num, superblock_final_dunits, input_shape.shape[1])
            component_results.append(superblock_result)

        # reverse component_result so the highest stride come first
        component_results = component_results[::-1]
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    def stack_superblock(input_shape, threshold=2):
        # Define initial state
        # This initial state would be re-weitten in each
        # loop
        current_recurrent = input_shape.shape[1]
        X = input_shape
        count_component = 0
        while current_recurrent > threshold:
            X = superblock_component(X, count_component)
            current_recurrent = X.shape[1]
            count_component+=1
        return X
        
    recurrents = 120
    lr = hyperparameters['lr']
    
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    X = stack_superblock(input_shape, threshold=2)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_310_kt_build(hp, train_inputs, train_labels):
    '''developed from model_307: 
    - 2-3x more layer units (with hope to increase performance)
        fraction 1 -> 3
        minimum_value 4 -> 8
        superblock_final_dunits 2/6/2 -> 8/24/4
        bdRNN 8 -> 32
        dense 8 -> 32
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    
    keys = ['superblock_final_dunits', 'lr', 'comp2', 'comp3', 'comp5', 'comp7', 'comp11', 'comp13', 'comp17', 'comp19', 'comp23', 'comp29', 'comp31', 'comp37', 'comp41', 'comp43', 'comp47', 'comp53', 'comp59', 'comp61', 'comp67', 'comp71', 'comp73', 'comp79', 'comp83', 'comp89', 'comp97', 'comp101', 'comp103', 'comp107', 'comp109', 'comp113']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hyperparameters[f'comp{component_num}']
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hyperparameters['superblock_final_dunits']
    lr = hyperparameters['lr']
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=32))(X)
    X = tf.keras.layers.Dense(units=32, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_311_kt_build(hp, train_inputs, train_labels):
    '''Recurrent preprocessor model.
    - Every features from index 0 -> len(features) - 1
        processed individually, and then concatenated with
        other proprocessed features.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    
    keys = ['lr', 'final_rnnu', 'final_denseu', 'rnnu_comp']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def recurrent_preprocessor_component(X, feature_no, rnnu_comp):
        '''Version 1: using single LSTM layer to preprocess'''
        X = tf.keras.layers.LSTM(rnnu_comp, return_sequences=False)(X[:,:,feature_no:feature_no+1])
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    def recurrent_preprocessor(input_shape, features, min_value=4, max_value=10, step=2):
        rnnu_comp = hyperparameters['rnnu_comp']
        component_results = []
        for feature_no in range(features):
            X = recurrent_preprocessor_component(input_shape, feature_no, rnnu_comp)
            component_results.append(X)
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    lr = hyperparameters['lr']
    final_rnnu = hyperparameters['final_rnnu']
    final_denseu = hyperparameters['final_denseu']
    
    recurrents = 120
    features = 60
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    X = recurrent_preprocessor(input_shape, features)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=final_rnnu))(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_312_kt_build(hp, train_inputs, train_labels):
    '''Recurrent preprocessor model.
    - Every features from index 0 -> len(features) - 1
        processed individually, and then concatenated with
        other proprocessed features.
    
    Rev from model_311:
    - Remove final rnn layers. Early training show shortcoming in acc
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    
    keys = ['lr', 'final_denseu', 'rnnu_comp']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def recurrent_preprocessor_component(X, feature_no, rnnu_comp):
        '''Version 1: using single LSTM layer to preprocess'''
        X = tf.keras.layers.LSTM(rnnu_comp, return_sequences=False)(X[:,:,feature_no:feature_no+1])
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    def recurrent_preprocessor(input_shape, features, min_value=4, max_value=10, step=2):
        rnnu_comp = hyperparameters['rnnu_comp']
        component_results = []
        for feature_no in range(features):
            X = recurrent_preprocessor_component(input_shape, feature_no, rnnu_comp)
            component_results.append(X)
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    lr = hyperparameters['lr']
    final_denseu = hyperparameters['final_denseu']
    
    recurrents = 120
    features = 60
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    X = recurrent_preprocessor(input_shape, features)
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_313_kt_build(hp, train_inputs, train_labels):
    '''Recurrent preprocessor model.
    - Every features from index 0 -> len(features) - 1
        processed individually, and then concatenated with
        other proprocessed features.
    
    Rev from model_311:
    - Remove final rnn layers. Early training show shortcoming in acc
    
    Rev from model_312:
    - Early training show even worse loss: 0.9.
    - Try to stack conv1d 1 kernel
    - add model.summary()
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''
    
    keys = ['lr', 'final_denseu', 'final_c1d_1_filters', 'final_c1d_2_filters', 'final_c1d_2_kernels', 'final_c1d_2_strides', 'final_c1d_3_filters', 'final_c1d_3_kernels', 'final_c1d_3_strides', 'rnnu_comp']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def recurrent_preprocessor_component(X, feature_no, rnnu_comp):
        '''Version 1: using single LSTM layer to preprocess'''
        X = tf.keras.layers.LSTM(rnnu_comp, return_sequences=False)(X[:,:,feature_no:feature_no+1])
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    def recurrent_preprocessor(input_shape, features, min_value=4, max_value=10, step=2):
        rnnu_comp = hyperparameters['rnnu_comp']
        component_results = []
        for feature_no in range(features):
            X = recurrent_preprocessor_component(input_shape, feature_no, rnnu_comp)
            component_results.append(X)
        X = tf.keras.layers.Concatenate(axis=1)(component_results)
        return X
    
    lr = hyperparameters['lr']
    final_denseu = hyperparameters['final_denseu']
    final_c1d_1_filters = hyperparameters['final_c1d_1_filters']
    final_c1d_2_filters = hyperparameters['final_c1d_2_filters']
    final_c1d_2_kernels = hyperparameters['final_c1d_2_kernels']
    final_c1d_2_strides = hyperparameters['final_c1d_2_strides']
    final_c1d_3_filters = hyperparameters['final_c1d_3_filters']
    final_c1d_3_kernels = hyperparameters['final_c1d_3_kernels']
    final_c1d_3_strides = hyperparameters['final_c1d_3_strides']
    
    recurrents = 120
    features = 60
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    X = recurrent_preprocessor(input_shape, features)
    # Stack multiple conv1d
    X = tf.keras.layers.Conv1D(filters=final_c1d_1_filters, kernel_size=1, strides=1)(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=final_c1d_2_filters, kernel_size=final_c1d_2_kernels, strides=final_c1d_2_strides)(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Conv1D(filters=final_c1d_3_filters, kernel_size=final_c1d_3_kernels, strides=final_c1d_3_strides)(X)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(units=final_denseu, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_314_kt_build(hp, train_inputs, train_labels):
    '''developed from model_307: 
    - Stacking superblock component together like block that stacked together.
    - Stacking difference with 308 & 309:
        - This 314 stack component before concatenation
            of superblock
        - The final concatenation is the same as 307, but
            with added component before that concatenation.
        - Desired effect:
            - Its possible that concatenation in 307 already
                abstract enough that additional abstraction
                just made the performance and learning curve
                going down.
            - Abstraction in prime component may give boost in
                performance by outputting more distinguishable
                value to later layer without shuffling the
                prime components even more.
            - This model also verify if stacking conv+rnn
                is possible. If not, try to stack more
                conv layer only to the component (next model).
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    keys = ['superblock_final_dunits', 'lr', 'comp2', 'comp3', 'comp5', 'comp7', 'comp11', 'comp13', 'comp17', 'comp19', 'comp23', 'comp29', 'comp31', 'comp37', 'comp41', 'comp43', 'comp47', 'comp53', 'comp59', 'comp61', 'comp67', 'comp71', 'comp73', 'comp79', 'comp83', 'comp89', 'comp97', 'comp101', 'comp103', 'comp107', 'comp109', 'comp113']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock_component(X, superblock_hyp, component_num, first_node=True, final_node=False):
        X_bn_1, X_act_1 = block(X, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num) if first_node else 1)
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=not final_node)(X_act_3)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)        
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hyperparameters[f'comp{component_num}']
        
        X_bn_1, X_act_1 = superblock_component(input_shape, superblock_hyp, component_num, first_node=True, final_node=False)
        X_bn_2, X_act_2 = superblock_component(X_act_1, superblock_hyp, component_num, first_node=False, final_node=False)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        X_bn_3, X_act_3 = superblock_component(X, superblock_hyp, component_num, first_node=False, final_node=True)
        
        X = tf.keras.layers.Dense(units=final_dunits)(X_act_3)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hyperparameters['superblock_final_dunits']
    lr = hyperparameters['lr']
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def model_315_kt_build(hp, train_inputs, train_labels):
    '''developed from model_307: 
    - Difference with 307:
        - Add conv stack and additive layer in the superblock
        - Why?
            - Early kt search with 314 show very long model
                compilation although the network not complex enough
                in terms of parameters.
            - Need faster iteration between model
            - even if the model 314 is better compared to 307 & 310,
                comparable model with faster training time is needed.
            - It's still unknown wheter 314 complexity is enough to
                drive lower val_loss. Using this model variation as
                comparison can lower iteration time.
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    keys = ['superblock_final_dunits', 'lr', 'comp2', 'comp3', 'comp5', 'comp7', 'comp11', 'comp13', 'comp17', 'comp19', 'comp23', 'comp29', 'comp31', 'comp37', 'comp41', 'comp43', 'comp47', 'comp53', 'comp59', 'comp61', 'comp67', 'comp71', 'comp73', 'comp79', 'comp83', 'comp89', 'comp97', 'comp101', 'comp103', 'comp107', 'comp109', 'comp113']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hyperparameters[f'comp{component_num}']
        
        # Block 1
        X_bn_1, X_act_1 = block(input_shape, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num))
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Block 1 + 2: Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Block 4
        X_bn_4, X_act_4 = block(X_act_3, superblock_hyp, superblock_hyp)
        # Block 3 + 4: Addition block
        X = tf.keras.layers.Add()([X_bn_3, X_bn_4])
        X = tf.keras.layers.Activation('relu')(X)
        
        # Block 5
        X_bn_5, X_act_5 = block(X, superblock_hyp, superblock_hyp)
        
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=False)(X_act_3)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hyperparameters['superblock_final_dunits']
    lr = hyperparameters['lr']
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def model_316_kt_build(hp, train_inputs, train_labels):
    '''developed from model_314: 
    - Evaluation after ~2 epochs of training:
        - The gradient descent really slow, although
            the accuracy is superb with those high losses.
        - It's possible that the networks are to deep
            to significantly update earlier params.
        - Correction:
            - Remove third layer in superblock
    
    - 120 recurrent, 60 filtered features input
    - 2 outputs (1 true / 1 false)'''  
    
    keys = ['superblock_final_dunits', 'lr', 'comp2', 'comp3', 'comp5', 'comp7', 'comp11', 'comp13', 'comp17', 'comp19', 'comp23', 'comp29', 'comp31', 'comp37', 'comp41', 'comp43', 'comp47', 'comp53', 'comp59', 'comp61', 'comp67', 'comp71', 'comp73', 'comp79', 'comp83', 'comp89', 'comp97', 'comp101', 'comp103', 'comp107', 'comp109', 'comp113']
    values = hp
    hyperparameters = {keys[i]:values[i] for i in range(len(keys))}
    
    def min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value):
        recommended_value = int((recurrents // component_num) * fraction)
        recommended_value = recommended_value if recommended_value >= minimum_value else minimum_value
        floor_value = int(floor_fraction * recommended_value)
        floor_value = floor_value if floor_value >= minimum_value else minimum_value
        step_value = int((recommended_value - floor_value) // num_steps)
        step_value = step_value if step_value >= 1 else 1
        return floor_value, recommended_value, step_value
    
    def block(X, filters, recunits, first_block=False, strides_val=1):
        # Conv1
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1 if not first_block else strides_val, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv2
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Conv3
        X = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, strides=1, padding='causal')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        # Recurrent unit
        X = tf.keras.layers.GRU(units=recunits, return_sequences=True)(X)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)
        return X_bn, X_act
    
    def superblock_component(X, superblock_hyp, component_num, first_node=True, final_node=False):
        X_bn_1, X_act_1 = block(X, superblock_hyp, superblock_hyp, first_block=True, strides_val=int(component_num) if first_node else 1)
        # Block 2
        X_bn_2, X_act_2 = block(X_act_1, superblock_hyp, superblock_hyp)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        # Block 3
        X_bn_3, X_act_3 = block(X, superblock_hyp, superblock_hyp)
        # Final layer
        X = tf.keras.layers.GRU(units=superblock_hyp, return_sequences=not final_node)(X_act_3)
        X_bn = tf.keras.layers.BatchNormalization()(X)
        X_act = tf.keras.layers.Activation('relu')(X_bn)        
        return X_bn, X_act
    
    def superblock(input_shape, component_num, final_dunits, recurrents, fraction=1, floor_fraction=0.25, num_steps=4, minimum_value=4):
        recommended_value = int((recurrents // component_num) * fraction)
        minv, maxv, step = min_max_step(component_num, recurrents, fraction, floor_fraction, num_steps, minimum_value)
        superblock_hyp = hyperparameters[f'comp{component_num}']
        
        X_bn_1, X_act_1 = superblock_component(input_shape, superblock_hyp, component_num, first_node=True, final_node=False)
        X_bn_2, X_act_2 = superblock_component(X_act_1, superblock_hyp, component_num, first_node=False, final_node=False)
        # Addition block
        X = tf.keras.layers.Add()([X_bn_1, X_bn_2])
        X = tf.keras.layers.Activation('relu')(X)
        
        X = tf.keras.layers.Flatten()(X)
        X = tf.keras.layers.Dense(units=final_dunits)(X)
        X = tf.keras.layers.BatchNormalization()(X)
        X = tf.keras.layers.Activation('relu')(X)
        return tf.expand_dims(X, axis=1)
    
    recurrents = 120
    component_strides = 1
    superblock_final_dunits = hyperparameters['superblock_final_dunits']
    lr = hyperparameters['lr']
    input_shape = tf.keras.Input(shape=(train_inputs.shape[1], train_inputs.shape[2]))
    
    n_components = generate_prime(int(recurrents // component_strides))
    component_results = []
    for component_num in n_components:
        superblock_result = superblock(input_shape, component_num, superblock_final_dunits, recurrents)
        component_results.append(superblock_result)
        
    # reverse component_result so the highest stride come first
    component_results = component_results[::-1]
    
    X = tf.keras.layers.Concatenate(axis=1)(component_results)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=8))(X)
    X = tf.keras.layers.Dense(units=8, activation='relu')(X)
    outputs = tf.keras.layers.Dense(units=train_labels.shape[1], activation='softmax')(X)
    
    model = tf.keras.models.Model(inputs=input_shape, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    print(f'Total params: {model.count_params()}')
    return model

def calculate_full_features_db(tickers=stock_list_100_highestrank_and_availability(), excluded_stock=excluded_stock(), ROOT_PATH='./', db_ver='3', calculate_v1_indicators=False, primary_index='time'):
    '''Perform complete 4000s features calculation
    '''
    origin_db = os.path.join(ROOT_PATH, 'idx_raw.db')
    v1_db = os.path.join(ROOT_PATH, 'idx_indicators.db')
    DB_ROOT_PATH = os.path.join(ROOT_PATH, f'db/v{db_ver}')
    origin_db_conn = sqlite3.connect(origin_db)
    v1_db_conn = sqlite3.connect(v1_db)
    
    # Calculate v1_db
    if calculate_v1_indicators:
        calculate_all_indicator(origin_db, v1_db, excluded_stock, verbose=1, selected_stock_only=False, ROOT_PATH=ROOT_PATH)
    
    count=1
    for ticker in tickers:
        tick = datetime.datetime.now()
        Path(DB_ROOT_PATH).mkdir(parents=True, exist_ok=True)
        v2_db = os.path.join(DB_ROOT_PATH, f'idx_{ticker}.db')
        v2_db_conn = sqlite3.connect(v2_db)

        # Read data back from version 1 db
        df = pd.read_sql(f'select * from `{ticker}`', v1_db_conn)
        df = calculate_talib_indicators(df)

        # Store df into db
        store_splitdf(df, primary_index, v2_db_conn, max_column=250)
        tock = datetime.datetime.now()
        print(f'{count}/{len(tickers)} {ticker} time: {tock-tick}')
        count+=1

def save_dataset(train_inputs, train_labels, train_changes, test_inputs, test_labels, test_changes, shift, interval, recurrent, ticker_group, split, data_version, group_size, dataset_ver='1', db_ver='2', ROOT_PATH='/'):
    '''Save dataset and their metadata
    to drive.'''
    dataset_path = os.path.join(ROOT_PATH, f'db/v{db_ver}/dataset/v{dataset_ver}/')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, mode=0o777)
    ticker_group_joined = '_'.join(ticker_group)
    dataset_file = os.path.join(dataset_path, f'{ticker_group_joined}_shift{shift}_interval{interval}_recurrent{recurrent}_split{split}.hdf5')
    with h5py.File(dataset_file, 'w') as f:
        f.create_dataset('train_inputs', data=train_inputs, compression='gzip', compression_opts=9)
        f.create_dataset('train_labels', data=train_labels, compression='gzip', compression_opts=9)
        f.create_dataset('train_changes', data=train_changes, compression='gzip', compression_opts=9)
        f.create_dataset('test_inputs', data=test_inputs, compression='gzip', compression_opts=9)
        f.create_dataset('test_labels', data=test_labels, compression='gzip', compression_opts=9)
        f.create_dataset('test_changes', data=test_changes, compression='gzip', compression_opts=9)
        
        f.attrs['shift'] = shift
        f.attrs['interval'] = interval
        f.attrs['recurrent'] = recurrent
        f.attrs['tickers'] = ticker_group
        f.attrs['split'] = split
        f.attrs['data_version'] = data_version
        f.attrs['group_size'] = group_size
        f.attrs['dataset_ver'] = dataset_ver
        f.attrs['db_ver'] = db_ver
        
    print('Verifying dataset metadata...')
    with h5py.File(dataset_file, 'r') as f:
        print(f'Found {len(f.attrs.keys())} metadata with value:\n')
        for key in f.attrs.keys():
            print(f'{key} => {f.attrs[key]}')
            
        
        print('Check data validity...')
        containers = ('train_inputs','train_labels','train_changes','test_inputs','test_labels','test_changes')
        sources = (train_inputs, train_labels, train_changes, test_inputs, test_labels, test_changes)
        for i, container in enumerate(containers):
            read_result = f[container][()]
            if type(sources[i]) == np.ndarray:
                source_shape = sources[i].shape
            elif type(sources[i]) == tuple:
                error_occured = False
                source_shape = []
                # Loop until no more child index found (IndexError)
                # or the last value is not array-like (TypeError)
                index_level = 0
                current_level = sources[i]
                source_shape.append(len(current_level))
                while not error_occured:
                    try:
                        current_level = current_level[0]
                        source_shape.append(len(current_level))
                    except (IndexError, TypeError):
                        error_occured = True
                source_shape = (*source_shape,)
            
            if read_result.shape == source_shape:
                print(f'{container} shape is valid')
            else:
                print(f'{container} shape is invalid: {source_shape} in source and {read_result.shape} in stored dataset.')
        # Clear memory after checking
        del sources, read_result, source_shape
            
def create_dataset_one_shot(ticker_group, ROOT_PATH='./', db='idx_indicators.db', recurrent=1, split=0.8, shift=0, db_type='1', db_ver='3', dataset_ver='1', interval=5, path_include_root=False):
    
    time_df = time_slice_list_from_group(ticker_group, db_type=db_type, db_ver=db_ver, db=db)
    first = True
    labels = []
    changes = []
    first = True
    for ticker in ticker_group:
        if db_type == '1':
            if path_include_root:
                db_conn = sqlite3.connect(db)
            elif not path_include_root:
                DB_PATH = os.path.join(ROOT_PATH, db)
                db_conn = sqlite3.connect(DB_PATH)            
            df = pd.read_sql(f'select * from `{ticker}`', db_conn)
        elif db_type == '2':
            ticker_db_path = os.path.join(ROOT_PATH, f'db/v{db_ver}/idx_{ticker}.db')
            ticker_db_conn = sqlite3.connect(ticker_db_path)
            df = restore_splitdf(ticker_db_conn)

        df = time_df.merge(df, how='inner', on='time')

        # Use EMA3_G from 1 & 2 days ahead as signal (EMA3_G gradient difference)
        df = gradient_difference_signal_indicator(df, signal=shift + interval, successive=shift)
        df = df.fillna(0)

        # Fetch labels
        df_label = df[['gd']]
        df_label = df_label.rename(columns={'gd':ticker})
        labels.append(df_label)

        # Fetch real change
        df_change = df.copy(deep=True)
        df_change = __comp__calculate_cumulative_change_v2(df_change, signal=shift + interval, successive=shift)
        df_change = df_change[[f'change_sig{shift + interval}_suc{shift}']]
        df_change = df_change.rename(columns={f'change_sig{shift + interval}_suc{shift}':ticker})
        df_change = df_change.fillna(0)
        changes.append(df_change)

        # Split to train and test
        df_cleaned = drop_unallowed_columns(df)
        train_df, test_df, train_mean, train_std = split_traintest(df_cleaned, split=split)

        # Sanity check: the train_df and test_df data length
        # need to be longer than recurrent + interval + shift
        train_cond_check = len(train_df) > recurrent + interval + shift
        test_cond_check = len(test_df) > recurrent + interval + shift
        if not (train_cond_check and test_cond_check):
            return 0, 0, 0, 0, 0, 0, 0
        elif train_cond_check and test_cond_check:
            pass
        
        # Version3: please convert df to numpy first
        # Convert to float32 for more efficient memory management
        train_input = make_multiple_shot_input_v3(train_df.to_numpy().astype('float32'), recurrent)
        test_input = make_multiple_shot_input_v3(test_df.to_numpy().astype('float32'), recurrent)

        # Concatenate the last axis (merging data across tickers)
        if first:
            first = False
            train_inputs = copy.deepcopy(train_input)
            test_inputs = copy.deepcopy(test_input)
        elif not first:
            train_inputs = np.concatenate((train_inputs, train_input), axis=-1)
            test_inputs = np.concatenate((test_inputs, test_input), axis=-1)

    # Labels engineering
    df_labels = pd.concat(labels, axis=1)
    df_labels.loc[df_labels.max(axis=1) < 0, '_'] = 0
    one_hot_labels = tf.one_hot(tf.math.argmax(df_labels, axis=1), depth=df_labels.shape[1])
    train_one_hot_labels, test_one_hot_labels = split_traintest(one_hot_labels, split=split, normalized=False)

    # Changes engineering
    df_changes = pd.concat(changes, axis=1)
    df_changes['_'] = 0
    train_changes, test_changes = split_traintest(df_changes, split=split, normalized=False)

    train_inputs, train_one_hot_labels, train_changes = adjust_dataset_shape(train_inputs, train_one_hot_labels, train_changes, shift=shift)
    test_inputs, test_one_hot_labels, test_changes = adjust_dataset_shape(test_inputs, test_one_hot_labels, test_changes, shift=shift)
    
    return train_inputs, train_one_hot_labels, train_changes, test_inputs, test_one_hot_labels, test_changes, df.iloc[-1].time
    
            
def create_dataset(ROOT_PATH='./', db='idx_indicators.db', tickers=stock_list_100_highestrank_and_availability(), excluded_stock=excluded_stock(), group_size=1, recurrent=1, split=0.8, shift=0, db_type='1', db_ver='3', dataset_ver='1', ticker_group_start_slice=None, ticker_group_end_slice=None, interval=5, use_itertools=False, num_of_samples=1000, start_from=1, path_include_root=False):
    '''Make dataset from `db` for easier prototyping.
    Each created dataset would stored into drive and their
    respective metadata.
    
    db_type:
        1: standard indicators
            reproduced from calculate_all_indicators
        2: extended indicators
            reproduced from calculate_full_features_db
    '''
    print(f'Start from ticker_group number: {start_from}')
    tickers = [ticker for ticker in tickers if ticker not in excluded_stock]
    ticker_groups = random_group(tickers, group_size, use_itertools=use_itertools, num_of_samples=num_of_samples)
    count = 1
    for ticker_group in ticker_groups[ticker_group_start_slice:ticker_group_end_slice]:
        if count < start_from:
            count+=1
            continue
        tick = datetime.datetime.now()
        print(f'Processing...\n {count}/{len(ticker_groups[ticker_group_start_slice:ticker_group_end_slice])+1} {ticker_group}')
        count+=1
        
        train_inputs, train_one_hot_labels, train_changes, test_inputs, test_one_hot_labels, test_changes, data_time = create_dataset_one_shot(ticker_group, ROOT_PATH=ROOT_PATH, db=db, recurrent=recurrent, split=split, shift=shift, db_type=db_type, db_ver=db_ver, dataset_ver=dataset_ver, interval=interval, path_include_root=path_include_root)
        
        # Check if overall records is enough for dataset creation
        if type(train_inputs) == int and type(train_one_hot_labels) == int and type(train_changes) == int and type(test_inputs) == int and type(test_one_hot_labels) == int and type(test_changes) == int and type(data_time) == int:
            print(f'Not enough data to process {ticker_group}. Skip to next ticker.')
            continue
        else:
            pass

        # Store dataset to drive
        save_dataset(train_inputs, train_one_hot_labels, train_changes, test_inputs, test_one_hot_labels, test_changes, shift, interval, recurrent, ticker_group, split, data_time, group_size, dataset_ver=dataset_ver, db_ver=db_ver, ROOT_PATH=ROOT_PATH)

        tock = datetime.datetime.now()
        print('Finished. Elapsed time: ', tock-tick)
        
def create_dataset_v2(abbr, ROOT_PATH='./', recurrent=120, split=0.8, shift=0, db_ver='8', ticker_start_slice=None, ticker_end_slice=None, interval=5, start_from=0, path_include_root=True):
    '''Make dataset from `db` for easier prototyping.
    Each created dataset would stored into drive and their
    respective metadata.
    
    env: tensorflow
    
    db_type:
        1: standard indicators
            reproduced from calculate_all_indicators
        2: extended indicators
            reproduced from calculate_full_features_db            
    
    v2 rev: 
        - integrate with world stocks database protocol.
        - remove group
        - use 1 as default group
        - use abbr as dataset_ver
        - use '1' as default db_type
    '''
    target_path = os.path.join(ROOT_PATH, 'indicators_db/')
    target_db_path = os.path.join(target_path, f'{abbr}_indicators.db')    
    tickers = get_tickers_from_world_stocks_database(abbr, without_exclusions=False, ROOT_PATH=ROOT_PATH)
    
    print(f'Start from ticker number: {start_from}. Entry slice: {ticker_start_slice}, end slice: {ticker_end_slice}')
    for i, ticker in enumerate(tickers[ticker_start_slice:ticker_end_slice]):
        if i < start_from:
            continue
        tick = datetime.datetime.now()
        ticker_expanded_dims = [ticker]
        print(f'Processing...\n {i}/{len(tickers[ticker_start_slice:ticker_end_slice])} {ticker_expanded_dims}')
        
        train_inputs, train_one_hot_labels, train_changes, test_inputs, test_one_hot_labels, test_changes, data_time = create_dataset_one_shot(ticker_expanded_dims, ROOT_PATH=ROOT_PATH, db=target_db_path, recurrent=recurrent, split=split, shift=shift, db_type='1', db_ver=db_ver, dataset_ver=abbr, interval=interval, path_include_root=path_include_root)
        
        # Check if overall records is enough for dataset creation
        if type(train_inputs) == int and type(train_one_hot_labels) == int and type(train_changes) == int and type(test_inputs) == int and type(test_one_hot_labels) == int and type(test_changes) == int and type(data_time) == int:
            print(f'Not enough data to process {ticker_expanded_dims}. Skip to next ticker.')
            update_exclusions(abbr, ticker, ROOT_PATH=ROOT_PATH)
            continue
        else:
            pass

        # Store dataset to drive
        save_dataset(train_inputs, train_one_hot_labels, train_changes, test_inputs, test_one_hot_labels, test_changes, shift, interval, recurrent, ticker_expanded_dims, split, data_time, group_size=1, dataset_ver=abbr, db_ver=db_ver, ROOT_PATH=ROOT_PATH)

        tock = datetime.datetime.now()
        print('Finished. Elapsed time: ', tock-tick)
        
def random_alphanumeric(k=32):
    random.seed(None)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=32))

def create_candlestick_dataset_report(ticker, recurrent, successive, signal, target_size, total_records, ROOT_PATH, db_ver):
    
    # Create required directory
    if not os.path.exists(os.path.join(ROOT_PATH, f'db/report/v{db_ver}/')):
        os.makedirs(os.path.join(ROOT_PATH,  f'db/report/v{db_ver}/'))
    else:
        pass  
    
    # Store information about created dataset
    metadata = {
        'ticker': ticker,
        'recurrent': recurrent,
        'successive': successive,
        'signal': signal,
        'target_size': target_size,
        'total_records': total_records,
        'created_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    }
    with open(os.path.join(ROOT_PATH, f'db/report/v{db_ver}/{random_alphanumeric()}.json'), 'w') as f:
        json.dump(metadata, f)    

def create_candlestick_dataset_one_ticker(ticker, db_conn, recurrent=10, db_ver='5', successive=1, signal=2, target_size=(75,75), ROOT_PATH='./'):
    df = pd.read_sql(f'select * from `{ticker}`', db_conn)
    df['time'] = df['time'].astype('datetime64[ns]')
    df = df.set_index('time', drop=True)

    difference_of = 'close'
    df['difference'] = df[difference_of].shift(-signal) - df[difference_of].shift(-successive)
    df.loc[df['difference'] > 0, 'signal'] = 1
    df.loc[df['difference'] <= 0, 'signal'] = 0

    mc = mpf.make_marketcolors(up='whitesmoke', down='black')
    s = mpf.make_mpf_style(marketcolors=mc)

    for i, index in enumerate(df.index):
        img_buffer = io.BytesIO()
        df_slice = df.iloc[i:i+recurrent]

        # Check if remaining data is enough to make dataset
        if len(df_slice) < recurrent:
            break
        elif len(df_slice) == recurrent:
            pass

        # Check if signal is not NaN
        # to find end of data records.
        current_signal = df.iloc[i+recurrent].signal
        if np.isnan(current_signal):
            break
        elif not np.isnan(current_signal):
            current_signal = int(current_signal)

        mpf.plot(df[i:i+recurrent], type='candle', savefig=img_buffer, figratio=(1,1), ylabel='', style=s, axisoff=True, scale_padding=0.1)
        img_buffer.seek(0)

        # Manipulate image
        im = Image.open(img_buffer)
        im_resized = im.resize(target_size)
        im_gray = ImageOps.grayscale(im_resized)
        assert im_gray.getbands()[0] == 'L', 'Grayscale conversion is needed'

        # Store image to folder based on label
        im_gray.save(os.path.join(ROOT_PATH, f'db/v{db_ver}/{current_signal}/{random_alphanumeric()}.jpg'))
        
    # Create report
    create_candlestick_dataset_report(ticker, recurrent, successive, signal, target_size, i, ROOT_PATH, db_ver)
    
def create_candlestick_dataset(recurrent=10, db_ver='5', successive=1, signal=2, target_size=(75,75), ROOT_PATH='./', db='idx_raw.db', parallel=False, batches=100, run_no=0, hotstart=0):
    db_path = os.path.join(ROOT_PATH, db)
    db_conn = sqlite3.connect(db_path)
    
    # Make required directory
    categories = ['0','1']
    ROOT_PATH = './'
    for category in categories:
        if not os.path.exists(os.path.join(ROOT_PATH, f'db/v{db_ver}/{category}/')):
            os.makedirs(os.path.join(ROOT_PATH,  f'db/v{db_ver}/{category}/'))
        else:
            pass  
        
    tickers = tablename_list(db_conn)
    if parallel:
        tickers = tickers[run_no*batches:(run_no+1)*batches]        
    
    for i, ticker in enumerate(tickers):
        if i < hotstart:
            print(f'{i}/{len(tickers)} {ticker}. Skipping...')
            continue
        tick = datetime.datetime.now()
        print(f'{i}/{len(tickers)} {ticker}. Processing...')
        
        create_candlestick_dataset_one_ticker(ticker, db_conn, recurrent=recurrent, db_ver=db_ver, successive=successive, signal=signal, target_size=target_size, ROOT_PATH=ROOT_PATH)
        
        tock = datetime.datetime.now()
        print(f'{i}/{len(tickers)} {ticker}. Elapsed time: {tock-tick}')
        
def filename_list_from_directory(path, search_condition):
    '''Return list of multiple filename that
    satisfies search_condition'''
    dirlist = os.listdir(path)
    search_results = [fn for fn in dirlist if re.search(search_condition, fn)]
    return search_results

def random_positioning(generator, features=60, groups=2):
    '''Randomly switch ticker features position'''
    # Seed value set before function call t
    # np.random.seed(seed)
    position_components = np.arange(groups)
    position = list(itertools.permutations(position_components))
    choice = np.arange(len(position))
    restacked = []
    position_choice = position[np.random.choice(choice)]
    for i in position_choice:
        restacked.append(generator[:,:,i*base_features:(i+1)*base_features])
    np_restacked = np.hstack(restacked)
    return np_restacked

def load_dataset(ticker_group, shift, interval, recurrent, db_ver, dataset_ver, split, ROOT_PATH, generator=False, batch_size=32, load_training_essentials=False, allow_different_dataset_version=True, shuffle=False, buffer_size=1024, random_features_positioning=False):
    dataset_path = os.path.join(ROOT_PATH, f'db/v{db_ver}/dataset/v{dataset_ver}/')
    assert not shuffle or generator, 'Shuffle only works with generator.'
    # assert not shuffle or random_features_positioning, 'Random features positioning only works with generator.'
    if not os.path.exists(dataset_path):
        print(f'Dataset path not exists. Please check again. db_ver: {db_ver}, dataset_ver: {dataset_ver}')
        # os.makedirs(dataset_path, mode=0o777)
    ticker_group_joined = '_'.join(ticker_group)
    dataset_filename = f'{ticker_group_joined}_shift{shift}_interval{interval}_recurrent{recurrent}_split{split}.hdf5'
    filenames = filename_list_from_directory(dataset_path, dataset_filename)
    # Check if filename found in the folder
    if len(filenames) < 1:
        if load_training_essentials:
            return 0, 0
        elif not load_training_essentials:
            return 0, 0, 0, 0, 0, 0, 0
    elif len(filenames) >= 1:
        pass
        
    for i, filename in enumerate(filenames):
        dataset_file = os.path.join(dataset_path, filename)
        if generator:
            train_inputs = tfio.IODataset.from_hdf5(dataset_file, '/train_inputs')
            train_labels = tfio.IODataset.from_hdf5(dataset_file, '/train_labels')
            test_inputs = tfio.IODataset.from_hdf5(dataset_file, '/test_inputs')
            test_labels = tfio.IODataset.from_hdf5(dataset_file, '/test_labels')

            # Eliminate nan
            train_inputs = np.nan_to_num(train_inputs, posinf=0.0, neginf=0.0)
            test_inputs = np.nan_to_num(test_inputs, posinf=0.0, neginf=0.0)

            # Zip input and labels
            train_gen = tf.data.Dataset.zip((train_inputs, train_labels))
            test_gen = tf.data.Dataset.zip((test_inputs, test_labels))

            # Apply batches
            train_gen = train_gen.batch(batch_size)
            test_gen = test_gen.batch(batch_size)
            
            # Apply shuffle
            if shuffle:
                train_gen = train_gen.shuffle(buffer_size=buffer_size)
                test_gen = test_gen.shuffle(buffer_size=buffer_size)
                
            # Apply random positioning
            # !! INCOMPLETE DEVELOPMENT !!
            if random_features_positioning:
                current_seed = np.random.randint(0,100)
                np.random.seed(current_seed)
                train_gen = random_positioning(generator, features=60, groups=2, seed=0)
                
        elif not generator:
            with h5py.File(dataset_file, 'r') as f:
                train_inputs = f['train_inputs'][()]
                test_inputs = f['test_inputs'][()]

        if not load_training_essentials:
            with h5py.File(dataset_file, 'r') as f:
                train_labels = f['train_labels'][()]
                train_changes = f['train_changes'][()]
                test_labels = f['test_labels'][()]
                test_changes = f['test_changes'][()]
                data_version = f.attrs['data_version']
        elif load_training_essentials:
            with h5py.File(dataset_file, 'r') as f:
                data_version = f.attrs['data_version']
        
        # Chain generator
        # Append train_labels, train_changes, test_labels, test_changes
        # Verify data_version; make sure to match one another.
        if i == 0:
            combined_data_version = data_version
            if generator:
                combined_train_gen = [train_gen]
                combined_test_gen = [test_gen]
            elif not generator:
                combined_train_inputs = [train_inputs]
                combined_test_inputs = [test_inputs]
            
            if not load_training_essentials:
                combined_train_labels = [train_labels]
                combined_train_changes = [train_changes]
                combined_test_labels = [test_labels]
                combined_test_changes = [test_changes]
        elif i > 0:
            if (data_version != combined_data_version) and allow_different_dataset_version:
                # print(f'Found different data version between dataset slice. Continue. Ticker group: {ticker_group}')
                pass
            elif (data_version != combined_data_version) and not allow_different_dataset_version:
                raise ValueError('Found different data version between dataset slice. To allow different data version, please specify `allow_different_dataset_version` param.')
            # assert data_version == combined_data_version, 'Found different data version between dataset slice.'
            if generator:
                combined_train_gen.append(train_gen)
                combined_test_gen.append(test_gen)
            elif not generator:
                combined_train_inputs.append(train_inputs)
                combined_test_inputs.append(test_inputs)
            if not load_training_essentials:
                combined_train_labels.append(train_labels)
                combined_train_changes.append(train_changes)
                combined_test_labels.append(test_labels)
                combined_test_changes.append(test_changes)  
            
    # Merge/stack/chain multiple dataset together
    if generator:
        if len(combined_train_gen) == 1 and len(combined_test_gen) == 1:
            merged_train_gen = combined_train_gen[0]
            merged_test_gen = combined_test_gen[0]
        elif len(combined_train_gen) > 1 and len(combined_test_gen) > 1:
            merged_train_gen = itertools.chain(*combined_train_gen)
            merged_test_gen = itertools.chain(*combined_test_gen)
    elif not generator:
        merged_train_inputs = np.vstack(combined_train_inputs)
        merged_test_inputs = np.vstack(combined_test_inputs)
    if not load_training_essentials:
        merged_train_labels = np.vstack(combined_train_labels)
        merged_train_changes = np.vstack(combined_train_changes)
        merged_test_labels = np.vstack(combined_test_labels)
        merged_test_changes = np.vstack(combined_test_changes)
    elif load_training_essentials:
        pass
    
    if generator and not load_training_essentials:
        return merged_train_gen, merged_train_labels, merged_train_changes, merged_test_gen, merged_test_labels, merged_test_changes, combined_data_version
    elif not generator and not load_training_essentials:
        return merged_train_inputs, merged_train_labels, merged_train_changes, merged_test_inputs, merged_test_labels, merged_test_changes, combined_data_version
    elif generator and load_training_essentials:
        return merged_train_gen, merged_test_gen
    elif not generator and load_training_essentials:
        return merged_train_inputs, merged_test_inputs

def generate_prime(upper_bound):
    '''Generate prime number between 0 and upper bound'''
    def check(number):
        divide = 0
        for i in range(number):
            if number % (i + 1) == 0:
                divide+=1
        return True if divide == 2 else False
    def update_exceptions(prime, upper_bound, exceptions):
        new = np.arange(prime, upper_bound + 1, prime)
        for n in new: exceptions.append(n)
        
    possibilities = list(np.arange(1, upper_bound + 1, 1))
    exceptions = []   
    prime_nums = []
    for possibility in possibilities:
        if possibility in exceptions:
            continue
        prime = check(possibility)
        if prime:
            prime_nums.append(possibility)
            update_exceptions(possibility, upper_bound, exceptions)
    return prime_nums     
    
def check_trial_json_validity(trial_metadata, metrics):
    '''Valid for Hyperband optimizer'''
    valid = True
    # Check if contains required metric metadata
    for metric in metrics:
        try:
            trial_metadata['metrics']['metrics'][metric]['observations'][0]['value'][0]
        except KeyError:
            valid = False
            
    # Check if contains valid hyperparameter value
    try:
        trial_metadata['hyperparameters']['values']
    except KeyError:
        valid = False
    return valid 

def best_model_metadata(models_path, metrics=['val_loss', 'val_accuracy'], primary_metrics='val_loss', ascending=True, return_best_model_only=True):
    identifier = 'trial'
    trials_list = os.listdir(models_path)
    trials_list = [trial for trial in trials_list if re.search(identifier, trial)]

    metric_results = [[] for _ in metrics]
    model_configs = []
    invalid_paths = []
    trials_list_valid = []
    for trial in trials_list:
        trial_metadata_path = os.path.join(models_path, f'{trial}/trial.json')
        if os.path.exists(trial_metadata_path):
            with open(trial_metadata_path, 'r') as f:
                trial_metadata = json.load(f)
            # Check validity before appending 'broken' data
            valid = check_trial_json_validity(trial_metadata, metrics)
            if not valid:
                continue
            
            for i, metric in enumerate(metrics):
                metric_value = trial_metadata['metrics']['metrics'][metric]['observations'][0]['value'][0]
                metric_results[i].append(metric_value)
            model_config = trial_metadata['hyperparameters']['values']
            model_configs.append(model_config)
            trials_list_valid.append(trial)
        else:
            invalid_paths.append(trial_metadata_path)

    # MAKE DF BUILD AUTOMATIC!!
    df = pd.DataFrame({'id':trials_list_valid,metrics[0]:metric_results[0], metrics[1]:metric_results[1], 'model_config':model_configs})
    if not return_best_model_only:
        return df
    elif return_best_model_only:
        best_model_metadata = df.sort_values(by=primary_metrics, ascending=ascending).reset_index(drop=True).iloc[0]
        return best_model_metadata

def hp_params(best_model):
    '''This is valid for Hyperband optimizer'''
    param_list = list(best_model.model_config.keys())
    if param_list[-1] == 'tuner/trial_id':
        hp_params = param_list[:-5]
    elif param_list[-1] == 'tuner/round':
        hp_params = param_list[:-4]
    hp = [best_model.model_config[hp_param] for hp_param in hp_params]
    return hp

def last_epochs_from_directory(path, model_ticker_target_join):
    '''Return the last state progress by epochs
    information about model retraining.
    
    First check for model_ticker_target value.
    If no last epochs found, use model_source, 
    indicated by 0 value returned.
    '''
    search_results = filename_list_from_directory(path, model_ticker_target_join)
    last_epochs = []
    for search_result in search_results:
        sliced = search_result.split('retrain-epochs')
        # If the model haven't been retrained before
        if len(sliced) == 1:
            last_epochs.append(0)
        elif len(sliced) > 1:
            last_epochs.append(int(sliced[1]))
    return max(last_epochs) if len(last_epochs) > 0 else 0

def read_best_model(model_base_id, model_source, model_ticker_target, shift, interval, recurrent, db_ver, dataset_ver, kt_iter, split, ROOT_PATH='./', retrain=False, metrics=['val_loss', 'val_accuracy'], primary_metrics='val_loss', ascending=True, verbose=0, retrain_epochs=2, generator=False, batch_size=32, backtest=True, save_model=True, allow_different_dataset_version=True, shuffle=False, buffer_size=1024):
    '''Valid for Hyperband optimizer'''
    model_source_join = '_'.join(model_source)
    model_ticker_target_join = '_'.join(model_ticker_target)
    base_path = os.path.join(ROOT_PATH, 'models/kt/', f'v{kt_iter}/')
    
    train_inputs, train_labels, train_changes, test_inputs, test_labels, test_changes, data_version = load_dataset(ticker_group=model_ticker_target, shift=shift, interval=interval, recurrent=recurrent, db_ver=db_ver, dataset_ver=dataset_ver, split=split, ROOT_PATH=ROOT_PATH, generator=generator, batch_size=batch_size, allow_different_dataset_version=allow_different_dataset_version, shuffle=shuffle, buffer_size=buffer_size)
    
    last_epochs = last_epochs_from_directory(base_path, model_ticker_target_join)
    # Load model backbone
    model_source_path = os.path.join(base_path, f'{model_source_join}/')
    best_model = best_model_metadata(model_source_path, metrics=metrics, primary_metrics=primary_metrics, ascending=ascending)

    if not generator:
        # Eliminate nan
        train_inputs = np.nan_to_num(train_inputs, posinf=0.0, neginf=0.0)
        test_inputs = np.nan_to_num(test_inputs, posinf=0.0, neginf=0.0)

        model = model_switcher_kt_build(model_base_id, hp_params(best_model), train_inputs, train_labels)
    elif generator:
        # Eliminate nan
        train_inputs = np.nan_to_num(train_inputs, posinf=0.0, neginf=0.0)
        test_inputs = np.nan_to_num(test_inputs, posinf=0.0, neginf=0.0)
        
        # print(train_inputs)
        for tg in train_inputs:
            x, y = tg
            break
        model = model_switcher_kt_build(model_base_id, hp_params(best_model), x, y)
    # If the model haven't been retrained before, read weight from keras_tuner output folder
    if last_epochs == 0:
        print('Load model from keras_tuner')
        # Load weights from keras_tuner directory
        model.load_weights(os.path.join(model_source_path, f'{best_model.id}/', 'checkpoints/epoch_0/checkpoint'))
    # If have been trained before, continue training for last epoch
    elif last_epochs > 0:
        print(f'Load weights from epoch {last_epochs}')
        # Load weights from previously trained model
        model.load_weights(os.path.join(base_path, f'{model_ticker_target_join}_retrain-epochs{last_epochs}/weights/checkpoint'))
    
    # Retrain model
    if retrain:
        if not generator:
            model.fit(train_inputs, train_labels, validation_data=(test_inputs, test_labels), epochs=retrain_epochs, verbose=verbose)
        elif generator:
            model.fit(np.nan_to_num(train_inputs, posinf=0.0, neginf=0.0), validation_data=np.nan_to_num(test_inputs, posinf=0.0, neginf=0.0), epochs=retrain_epochs, verbose=verbose)
        
    if verbose and backtest:
        print(f'Best model metadata:\n', best_model)
        train_ideal, train_real, test_ideal, test_real = backtest_v1(model, train_inputs if generator else train_inputs, test_inputs if generator else test_inputs, train_labels, test_labels, train_changes, test_changes)
        print('Train ideal: ', train_ideal[0], '\nTrain real: ', train_real[0], '\nTest ideal: ', test_ideal[0], '\nTest real: ', test_real[0])
        train_r, train_nr, test_r, test_nr = performance_ratio(train_ideal, train_real, test_ideal, test_real)
        print(f'\nModel backtest performance: tr {train_r:.2f}, tnr {train_nr:.2f}, vr {test_r:.2f}, vnr {test_nr:.2f}')
        print('above results is in performance fraction between ideal (observed) and real condition produced from the model\n')
        
    if retrain and save_model:
        save_path = os.path.join(ROOT_PATH, f'models/kt/v{kt_iter}/{model_ticker_target_join}_retrain-epochs{last_epochs + retrain_epochs}/')
        # Disabling save model due to exploding memory usage issue
        # model.save(f'{save_path}/model/')
        model.save_weights(f'{save_path}/weights/checkpoint')
    return model, data_version

def standard_scaler_one_shot(ticker_group, data_version, ROOT_PATH='./', db='idx_indicators.db', split=0.8):
    DB_PATH = os.path.join(ROOT_PATH, db)
    db_conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f'select * from `{ticker_group[0]}`', db_conn)
    df = df.loc[df['time'] <= data_version]
    df_cleaned = drop_unallowed_columns(df)
    train_df, test_df, train_mean, train_std = split_traintest(df_cleaned, split=split)
    return train_mean, train_std

def merge_dataset(ROOT_PATH='./', tickers=stock_list_100_highestrank_and_availability(), excluded_stock=excluded_stock(), group_size=1, recurrent=120, split=0.8, shift=0, db_ver='3', dataset_ver='4', interval=1, shuffle=False, seed=0, save_in_batches=False, save_batches=16, use_itertools=False, num_of_samples=1000, allow_different_dataset_version=True):
    tickers = [ticker for ticker in tickers if ticker not in excluded_stock]
    ticker_groups = random_group(tickers, group_size, use_itertools=use_itertools, num_of_samples=num_of_samples)

    save_batches_number = 0
    i_reducer = 0
    for i, ticker_group in enumerate(ticker_groups):
        i = i - i_reducer
        print(f'{i}/{len(ticker_groups)-1} Current progress: {ticker_group}')
        train_inputs, train_labels, train_changes, test_inputs, test_labels, test_changes, data_version = load_dataset(ticker_group, shift, interval, recurrent, db_ver, dataset_ver, split, ROOT_PATH, allow_different_dataset_version=allow_different_dataset_version)
        
        if type(train_inputs) == int and type(train_labels) == int and type(train_changes) == int and type(test_inputs) == int and type(test_labels) == int and type(test_changes) == int and type(data_version) == int:
            print(f'No data found for {ticker_group}. Skip to next ticker group.')
            
            # Reduce i by 1 to make sure every batches have homogenous
            # length. 
            # Another reason: below condition would fail and error raised
            # if not.
            i_reducer+=1
            continue
        
        # Eliminate nan
        train_inputs = np.nan_to_num(train_inputs, posinf=0.0, neginf=0.0)
        test_inputs = np.nan_to_num(test_inputs, posinf=0.0, neginf=0.0)
        
        # Merge multiple dataset into one
        if i % save_batches == 0:
            glob_data_version = data_version
            glob_train_inputs = train_inputs.copy()
            glob_train_labels = train_labels.copy()
            glob_train_changes = train_changes.copy()
            glob_test_inputs = test_inputs.copy()
            glob_test_labels = test_labels.copy()
            glob_test_changes = test_changes.copy()
        elif i % save_batches != 0:
            if (data_version != glob_data_version) and allow_different_dataset_version:
                # print(f'Found different data version between dataset slice. Continue. Ticker group: {ticker_group}')
                pass
            elif (data_version != glob_data_version) and not allow_different_dataset_version:
                raise ValueError('Found different data version between dataset slice. To allow different data version, please specify `allow_different_dataset_version` param.')
            glob_train_inputs = np.vstack((glob_train_inputs, train_inputs))
            glob_train_labels = np.vstack((glob_train_labels, train_labels))
            glob_train_changes = np.vstack((glob_train_changes, train_changes))
            glob_test_inputs = np.vstack((glob_test_inputs, test_inputs))
            glob_test_labels = np.vstack((glob_test_labels, test_labels))
            glob_test_changes = np.vstack((glob_test_changes, test_changes))

        # Shuffle dataset
        if (shuffle and (i == (len(ticker_groups) - 1)) and (not save_in_batches)) or (shuffle and ((i + 1) % save_batches == 0) and save_in_batches):
            train_shuffle = list(zip(glob_train_inputs, glob_train_labels, glob_train_changes))
            test_shuffle = list(zip(glob_test_inputs, glob_test_labels, glob_test_changes))
            random.seed(seed)
            random.shuffle(train_shuffle)
            random.shuffle(test_shuffle)
            glob_train_inputs, glob_train_labels, glob_train_changes = zip(*train_shuffle)
            glob_test_inputs, glob_test_labels, glob_test_changes = zip(*test_shuffle)
            target_name = [f'{save_batches_number}MERGED-shuffled']
            # Clear memory
            train_shuffle, test_shuffle
        elif ((not shuffle) and (i == (len(ticker_groups) - 1)) and (not save_in_batches)) or ((not shuffle) and (((i + 1) % save_batches) == 0) and save_in_batches):
            target_name = [f'{save_batches_number}MERGED-unshuffled']

        # Save dataset
        if (((i + 1) % save_batches == 0) and save_in_batches) or ((i == len(ticker_groups) - 1) and (not save_in_batches)):
            print(f'Saving for save_batches_number: {save_batches_number}...')
            save_dataset(glob_train_inputs, glob_train_labels, glob_train_changes, glob_test_inputs, glob_test_labels, glob_test_changes, shift, interval, recurrent, target_name, split, glob_data_version, len(target_name), dataset_ver, db_ver, ROOT_PATH)
            save_batches_number+=1
            
            # Clear memory after saving
            del train_inputs, train_labels, train_changes, test_inputs, test_labels, test_changes, data_version
            del glob_train_inputs, glob_train_labels, glob_train_changes, glob_test_inputs, glob_test_labels, glob_test_changes, glob_data_version
        else:
            pass
        
def read_best_model_hyperparameters_name(ticker_group, kt_iter, ROOT_PATH='./'):
    '''Return hyperparameters name for best
    model from kt search.'''
    ticker_group_join = '_'.join(ticker_group)
    models_path = os.path.join(ROOT_PATH, f'./models/kt/v{kt_iter}/{ticker_group_join}/')
    best_model_m = best_model_metadata(models_path, return_best_model_only=True)
    return list(best_model_m.model_config.keys())

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def copy_dir(source_item, destination_item):
    if os.path.isdir(source_item):
        make_dir(destination_item)
        sub_items = glob.glob(source_item + '/*')
        for sub_item in sub_items:
            copy_dir(sub_item, destination_item + '/' + sub_item.split('/')[-1])
    else:
        shutil.copy(source_item, destination_item)

def cleanup_machine_resources(target_folders=['models'], DRIVE_PATH='gdrive/MyDrive/#PROJECT/idx/'):
    '''Transfer results from local machine
    to Google Drive after process finish'''
    import shutil
    for target_folder in target_folders:
        source_path = f'./{target_folder}'
        destination_path = f'{DRIVE_PATH}{target_folder}'
        copy_dir(source_path, destination_path)
        
def clean_candlestick_dataset(ROOT_PATH, db_ver, image_source_size=(75,75), verbose_every=10000):
    '''Clean candlestick data from error.
    
    File that caused error when read move to
    exclusions.
    
    Use env:tensorflow'''
    def move_exclusion(ROOT_PATH, db_ver, class_, filename):
        db_path_source = os.path.join(ROOT_PATH, f'db/v{db_ver}/')
        db_path_target = os.path.join(ROOT_PATH, f'db/v{db_ver}_exclusions/')
        class_path_source = os.path.join(db_path_source, f'{class_}/')
        class_path_target = os.path.join(db_path_target, f'{class_}/')
        filename_path_source = os.path.join(class_path_source, filename)
        filename_path_target = os.path.join(class_path_target, filename)

        if not os.path.exists(class_path_target): os.makedirs(class_path_target)
        if os.path.exists(filename_path_source): 
            shutil.move(filename_path_source, filename_path_target) 
        else: 
            print(f'Source file or path not exists: {filename_path_source}')
        
    from PIL import Image

    db_path = os.path.join(ROOT_PATH, f'db/v{db_ver}/')
    classes = os.listdir(db_path)

    for class_ in classes:
        print(f'Scanning for class {class_}...')
        class_path = os.path.join(db_path, f'{class_}/')
        listdir = os.listdir(class_path)

        exclusions = []
        error_count = 0
        error = False
        tick = datetime.datetime.now()
        for i, filename in enumerate(listdir):
            try:
                im = Image.open(os.path.join(class_path, filename))
            except Image.UnidentifiedImageError as e:
                print(f'Error no {error_count}. {e}')
                error_count+=1
                exclusions.append(filename)
                error = True

            if im.size != image_source_size:
                exclusions.append(filename)
                error = True

            if error: 
                move_exclusion(ROOT_PATH, db_ver, class_, filename)
                error = False            

            if i % verbose_every == 0:
                tock = datetime.datetime.now()
                print(f'Current progress: {i}/{len(listdir)}. Elapsed time: {tock-tick}')
                tick = datetime.datetime.now()
        
        

