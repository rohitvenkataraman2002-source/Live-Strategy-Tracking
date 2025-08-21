import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
import os
import pickle
import json
from pathlib import Path
import pymysql
from scipy.optimize import newton, brentq
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import asyncio
import aiohttp
# import asyncpg
from functools import lru_cache, wraps
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple
import time
from numba import jit, prange
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CACHE_DIR = "stock_price_cache"
CACHE_FILE = os.path.join(CACHE_DIR, "stock_prices.pkl")
CACHE_METADATA_FILE = os.path.join(CACHE_DIR, "cache_metadata.json")
LOCAL_CACHE_FILE = os.path.join(CACHE_DIR, "local_cache.pkl")

# Local cache configuration (replacing Redis)
LOCAL_CACHE_TTL = 86400  # 24 hours in seconds

# Initialize local cache
LOCAL_CACHE = {}
LOCAL_CACHE_TIMESTAMPS = {}

def is_cache_valid(key, ttl=LOCAL_CACHE_TTL):
    """Check if local cache entry is still valid"""
    if key not in LOCAL_CACHE_TIMESTAMPS:
        return False
    return (time.time() - LOCAL_CACHE_TIMESTAMPS[key]) < ttl

def save_local_cache():
    """Save local cache to disk"""
    try:
        create_cache_directory()
        cache_data = {
            'cache': LOCAL_CACHE,
            'timestamps': LOCAL_CACHE_TIMESTAMPS
        }
        with open(LOCAL_CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.warning(f"Failed to save local cache: {e}")

def load_local_cache():
    """Load local cache from disk"""
    global LOCAL_CACHE, LOCAL_CACHE_TIMESTAMPS
    try:
        if os.path.exists(LOCAL_CACHE_FILE):
            with open(LOCAL_CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
            LOCAL_CACHE = cache_data.get('cache', {})
            LOCAL_CACHE_TIMESTAMPS = cache_data.get('timestamps', {})
            
            # Clean expired entries
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in LOCAL_CACHE_TIMESTAMPS.items()
                if (current_time - timestamp) > LOCAL_CACHE_TTL
            ]
            for key in expired_keys:
                LOCAL_CACHE.pop(key, None)
                LOCAL_CACHE_TIMESTAMPS.pop(key, None)
            
            logger.info(f"📋 Loaded local cache with {len(LOCAL_CACHE)} entries")
    except Exception as e:
        logger.warning(f"Failed to load local cache: {e}")
        LOCAL_CACHE = {}
        LOCAL_CACHE_TIMESTAMPS = {}

# Load local cache on startup
load_local_cache()

# Performance monitoring decorator
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"⏱️  {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

# Optimized database connection with connection pooling
class DatabaseManager:
    def __init__(self):
        self.connection_params = {
            'host': '10.100.20.190',
            'user': 'raghunath',
            'password': 'Raghunath@123',
            'db': 'StratDB',
            'cursorclass': pymysql.cursors.DictCursor,
            'autocommit': True
        }
        self._asset_isin_mapping = None
        
    @lru_cache(maxsize=1)
    def get_asset_isin_mapping(self):
        """Cache the asset ISIN mapping"""
        if self._asset_isin_mapping is None:
            connection = pymysql.connect(**self.connection_params)
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT nse_code, isin FROM market_cap;")
                    rows = cursor.fetchall()
                    self._asset_isin_mapping = {row['nse_code']: row['isin'] for row in rows}
            finally:
                connection.close()
        return self._asset_isin_mapping

    def get_transactions_from_db(self, strategy_id):    
        # Create a DB connection
        connection = pymysql.connect(**self.connection_params)
        
        try:
            query = """
                SELECT 
                    strategy_id,
                    asset_type,
                    Date as transaction_date,
                    name,
                    price,
                    net_amount,
                    transaction_type,
                    quantity                
                FROM strategy_backtests
                WHERE strategy_id = %s
                ORDER BY transaction_date
            """
                         
            with connection.cursor() as cursor:
              cursor.execute(query, strategy_id)
              rows = cursor.fetchall()
                    
            df = pd.DataFrame(rows)
            
            return df

        finally:
            connection.close()

db_manager = DatabaseManager()
ASSET_ISIN_MAPPING = db_manager.get_asset_isin_mapping()

# Optimized cache management (Local version)
class OptimizedCacheManager:
    def __init__(self):
        self.file_cache = {}
        
    def _get_cache_key(self, key_data):
        """Generate cache key from data"""
        return hashlib.md5(str(key_data).encode()).hexdigest()
    
    def get_from_local_cache(self, key):
        """Get data from local cache"""
        try:
            if key in LOCAL_CACHE and is_cache_valid(key):
                return LOCAL_CACHE[key]
            return None
        except Exception as e:
            logger.warning(f"Local cache get error: {e}")
            return None
    
    def set_to_local_cache(self, key, data, ttl=LOCAL_CACHE_TTL):
        """Set data to local cache"""
        try:
            LOCAL_CACHE[key] = data
            LOCAL_CACHE_TIMESTAMPS[key] = time.time()
            
            # Periodically save to disk
            if len(LOCAL_CACHE) % 100 == 0:  # Save every 100 entries
                save_local_cache()
            return True
        except Exception as e:
            logger.warning(f"Local cache set error: {e}")
            return False
    
    def get_stock_prices(self, company_name, start_date, end_date):
        """Get stock prices with multi-level caching"""
        cache_key = self._get_cache_key(f"stock_prices_{company_name}_{start_date}_{end_date}")
        
        # Try local cache first
        data = self.get_from_local_cache(cache_key)
        if data:
            return data
        
        # Try file cache
        if cache_key in self.file_cache:
            return self.file_cache[cache_key]
        
        return None
    
    def set_stock_prices(self, company_name, start_date, end_date, data):
        """Set stock prices to both caches"""
        cache_key = self._get_cache_key(f"stock_prices_{company_name}_{start_date}_{end_date}")
        
        # Set to local cache
        self.set_to_local_cache(cache_key, data)
        
        # Set to file cache
        self.file_cache[cache_key] = data

cache_manager = OptimizedCacheManager()

def create_cache_directory():
    """Create cache directory if it doesn't exist"""
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

@timing_decorator
def load_cache():
    """Load cached stock price data with optimization"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
            logger.info(f"📦 Loaded cached data for {len(cache_data)} stocks")
            return cache_data
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return {}
    return {}

@timing_decorator
def save_cache(cache_data):
    """Save stock price data to cache with optimization"""
    create_cache_directory()
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'stocks_count': len(cache_data),
            'total_data_points': sum(len(stock_data) for stock_data in cache_data.values())
        }
        
        with open(CACHE_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Saved cache with {len(cache_data)} stocks")
        
        # Also save local cache
        save_local_cache()
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

# Optimized API fetching with async support
class OptimizedAPIFetcher:
    def __init__(self, max_workers=10):
        self.max_workers = max_workers
        self.session = None
    
    async def fetch_upstox_candles_async(self, session, instrument_key, from_date, to_date, interval):
        """Async version of fetch_upstox_candles"""
        url = f"https://api.upstox.com/v2/historical-candle/{instrument_key.replace('|', '%7C')}/{interval}/{to_date}/{from_date}"
        headers = {'Accept': 'application/json'}
        
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"API Error {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching data for {instrument_key}: {e}")
            return None
    
    def fetch_upstox_candles(self, instrument_key, from_date, to_date, interval):
        """Synchronous version for backward compatibility"""
        url = f"https://api.upstox.com/v2/historical-candle/{instrument_key.replace('|', '%7C')}/{interval}/{to_date}/{from_date}"
        headers = {'Accept': 'application/json'}
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"API Error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error fetching data for {instrument_key}: {e}")
            return None

api_fetcher = OptimizedAPIFetcher()

# Optimized price fetching with parallel processing
@timing_decorator
def get_historical_stock_prices_optimized(company_name, start_date, end_date, cache_data):
    """Optimized historical stock price fetching"""
    
    if company_name not in ASSET_ISIN_MAPPING:
        logger.error(f"❌ Company {company_name} not found in ASSET_ISIN_MAPPING")
        return None
    
    # Check multi-level cache first
    cached_data = cache_manager.get_stock_prices(company_name, start_date, end_date)
    if cached_data:
        logger.info(f"💰 Using cached data for {company_name}")
        return cached_data
    
    # Check if we have cached data in the traditional cache
    if company_name in cache_data:
        cached_prices = cache_data[company_name]
        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date()
        
        cached_dates = set(cached_prices.keys())
        required_dates = set(pd.date_range(start=start_date, end=end_date, freq='D').date)
        missing_dates = required_dates - cached_dates
        
        if not missing_dates:
            filtered_data = {date: price for date, price in cached_prices.items() 
                           if start_dt <= date <= end_dt}
            cache_manager.set_stock_prices(company_name, start_date, end_date, filtered_data)
            return filtered_data
    else:
        cache_data[company_name] = {}
    
    # Fetch from API
    isin_code = ASSET_ISIN_MAPPING[company_name]
    instrument_key = f"NSE_EQ|{isin_code}"
    
    logger.info(f"📡 Fetching fresh data for {company_name} from {start_date} to {end_date}")
    
    data = api_fetcher.fetch_upstox_candles(instrument_key, start_date, end_date, "day")
    
    if data and data.get('status') == 'success' and data.get('data'):
        candles = data['data'].get('candles', [])
        if len(candles) > 0:
            fresh_data = {}
            for candle in candles:
                date = pd.to_datetime(candle[0]).date()
                price = float(candle[4])
                fresh_data[date] = price
            
            cache_data[company_name].update(fresh_data)
            
            # Forward fill missing dates using vectorized operations
            all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            filled_data = {}
            sorted_dates = sorted(fresh_data.keys())
            
            for date in all_dates:
                date_obj = date.date()
                if date_obj in fresh_data:
                    filled_data[date_obj] = fresh_data[date_obj]
                else:
                    previous_dates = [d for d in sorted_dates if d < date_obj]
                    if previous_dates:
                        last_trading_date = max(previous_dates)
                        filled_data[date_obj] = fresh_data[last_trading_date]
            
            cache_data[company_name].update(filled_data)
            cache_manager.set_stock_prices(company_name, start_date, end_date, filled_data)
            
            logger.info(f"✅ Fetched {len(candles)} trading days, filled to {len(filled_data)} total days for {company_name}")
            return filled_data
    
    logger.error(f"❌ Failed to fetch data for {company_name} - SKIPPING")
    return None

# Parallel processing for stock price matrix building
@timing_decorator
def build_stock_price_matrix_parallel(transactions_df, start_date, end_date, max_workers=None):
    """Build stock price matrix with parallel processing"""
    logger.info("\n=== BUILDING STOCK PRICE MATRIX (PARALLEL) ===")
    
    cache_data = load_cache()
    unique_stocks = transactions_df['name'].unique()
    logger.info(f"🔄 Processing {len(unique_stocks)} unique stocks")
    
    if max_workers is None:
        max_workers = min(len(unique_stocks), mp.cpu_count() * 2)
    
    price_matrix = {}
    
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date).date()
    
    batch_size_days = (end_date - start_date).days // 3
    
    def process_stock(stock):
        """Process a single stock"""
        try:
            combined_prices = {}
            batch_start = start_date
            
            while batch_start <= end_date:
                batch_end = min(batch_start + timedelta(days=batch_size_days), end_date)
                batch_prices = get_historical_stock_prices_optimized(
                    stock, batch_start, batch_end, cache_data
                )
                
                if batch_prices:
                    combined_prices.update(batch_prices)
                batch_start = batch_end + timedelta(days=1)
            
            return stock, combined_prices if combined_prices else None
        except Exception as e:
            logger.error(f"Error processing {stock}: {e}")
            return stock, None
    
    # Use ThreadPoolExecutor for I/O bound operations
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_stock = {executor.submit(process_stock, stock): stock for stock in unique_stocks}
        
        for future in as_completed(future_to_stock):
            stock_name, stock_data = future.result()
            if stock_data:
                price_matrix[stock_name] = stock_data
                logger.info(f"✅ Processed {stock_name}")
            else:
                logger.warning(f"⚠️ Failed to process {stock_name}")
    
    save_cache(cache_data)
    
    logger.info(f"\n✅ Stock price matrix built with {len(price_matrix)} stocks")
    return price_matrix

# Optimized holdings calculation using vectorized operations
@jit(nopython=True)
def calculate_holdings_fast(quantities, transaction_types, stocks_array, unique_stocks):
    """Fast holdings calculation using Numba"""
    holdings = np.zeros(len(unique_stocks))
    stock_to_idx = {stock: i for i, stock in enumerate(unique_stocks)}
    
    for i in range(len(quantities)):
        stock_idx = stock_to_idx.get(stocks_array[i], -1)
        if stock_idx >= 0:
            if transaction_types[i] == 1:  # Buy = 1
                holdings[stock_idx] += quantities[i]
            else:  # Sell = 0
                holdings[stock_idx] -= quantities[i]
    
    return holdings

def calculate_daily_holdings_optimized(transactions_df, date):
    """Optimized holdings calculation"""
    relevant_txns = transactions_df[transactions_df['transaction_date'].dt.date <= date]
    
    if len(relevant_txns) == 0:
        return {}
    
    # Use groupby for faster aggregation
    holdings_summary = relevant_txns.groupby(['name', 'transaction_type']).agg({
        'quantity': 'sum',
        'net_amount': 'sum'
    }).reset_index()
    
    # Calculate net positions
    holdings_dict = {}
    for _, row in holdings_summary.iterrows():
        stock = row['name']
        if stock not in holdings_dict:
            holdings_dict[stock] = {'quantity': 0, 'net_amount': 0}
        
        if row['transaction_type'] == 'Buy':
            holdings_dict[stock]['quantity'] += row['quantity']
            holdings_dict[stock]['net_amount'] += row['net_amount']
        else:
            holdings_dict[stock]['quantity'] -= row['quantity']
            holdings_dict[stock]['net_amount'] -= row['net_amount']
    
    return {stock: data['quantity'] for stock, data in holdings_dict.items() if data['quantity'] != 0}

# Transaction filtering logic integration
def get_price_for_date(symbol, date):
    """Get price for symbol on specific date"""
    isin_code = ASSET_ISIN_MAPPING.get(symbol)
    if not isin_code:
        return None
    
    instrument_key = f"NSE_EQ|{isin_code}"
    data = api_fetcher.fetch_upstox_candles(instrument_key, date, date, "day")
    
    if data and "data" in data and "candles" in data["data"] and data["data"]["candles"]:
        return data["data"]["candles"][0][1]  # Open price
    else:
        return None

# Global price cache to avoid repeated fetches (Local version)
PRICE_CACHE = {}

class UltraFastPriceFetcher:
    def __init__(self):
        self.session = None
        
    async def fetch_price_async(self, session, symbol, date):
        """Ultra-fast async price fetching"""
        cache_key = f"{symbol}_{date}"
        
        # Check global cache first
        if cache_key in PRICE_CACHE:
            return symbol, PRICE_CACHE[cache_key]
        
        # Check local cache
        local_cache_key = f"ultrafast_price_{symbol}_{date}"
        cached_price = cache_manager.get_from_local_cache(local_cache_key)
        if cached_price is not None:
            PRICE_CACHE[cache_key] = cached_price
            return symbol, cached_price
        
        isin_code = ASSET_ISIN_MAPPING.get(symbol)
        if not isin_code:
            return symbol, None
            
        instrument_key = f"NSE_EQ|{isin_code}"
        url = f"https://api.upstox.com/v2/historical-candle/{instrument_key.replace('|', '%7C')}/day/{date}/{date}"
        headers = {'Accept': 'application/json'}
        
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and "data" in data and "candles" in data["data"] and data["data"]["candles"]:
                        price = float(data["data"]["candles"][0][1])  # Open price
                        PRICE_CACHE[cache_key] = price
                        # Cache to local cache
                        cache_manager.set_to_local_cache(local_cache_key, price, 3600)  # 1 hour cache
                        return symbol, price
                return symbol, None
        except Exception:
            return symbol, None
    
    async def batch_fetch_prices_ultra_fast(self, symbols, date):
        """Ultra-fast batch price fetching with async"""
        if not symbols:
            return {}
            
        # Check local cache in batch
        cached_prices = {}
        uncached_symbols = []
        
        for symbol in symbols:
            local_cache_key = f"ultrafast_price_{symbol}_{date}"
            cached_price = cache_manager.get_from_local_cache(local_cache_key)
            if cached_price is not None:
                cached_prices[symbol] = cached_price
            else:
                uncached_symbols.append(symbol)
        
        if not uncached_symbols:
            return cached_prices
        
        # Async fetch for uncached symbols
        connector = aiohttp.TCPConnector(limit=50, limit_per_host=25)
        timeout = aiohttp.ClientTimeout(total=3)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [self.fetch_price_async(session, symbol, date) for symbol in uncached_symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and cache
        final_prices = cached_prices.copy()
        
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                symbol, price = result
                if price is not None:
                    final_prices[symbol] = price
                    # Cache locally
                    local_cache_key = f"ultrafast_price_{symbol}_{date}"
                    cache_manager.set_to_local_cache(local_cache_key, price, 3600)  # 1 hour cache
        
        return final_prices

# Initialize ultra-fast fetcher
ultra_fetcher = UltraFastPriceFetcher()

def calculate_holdings_vectorized(before_start_df):
    """Ultra-fast vectorized holdings calculation"""
    if len(before_start_df) == 0:
        return {}
    
    # Vectorized calculation using pandas
    before_start_df = before_start_df.copy()
    before_start_df['qty_signed'] = before_start_df.apply(
        lambda x: x['quantity'] if x['transaction_type'].lower() == 'buy' else -x['quantity'], 
        axis=1
    )
    
    # Group and sum in one operation
    holdings_series = before_start_df.groupby('name')['qty_signed'].sum()
    
    # Filter positive holdings
    return holdings_series[holdings_series > 0].to_dict()

def build_position_tracker_vectorized(combined_df):
    """Ultra-fast position tracking using vectorized operations"""
    if len(combined_df) == 0:
        return [], {}
    
    # Sort once
    combined_sorted = combined_df.sort_values('transaction_date').reset_index(drop=True)
    
    # Vectorized position calculation
    combined_sorted['qty_signed'] = combined_sorted.apply(
        lambda x: x['quantity'] if x['transaction_type'].lower() == 'buy' else -x['quantity'], 
        axis=1
    )
    
    # Calculate cumulative positions per symbol
    final_rows = combined_sorted.to_dict('records')
    
    # Fast position tracking
    positions = {}
    open_positions = {}
    
    for row in final_rows:
        symbol = row['name']
        change = row['qty_signed']
        
        positions[symbol] = positions.get(symbol, 0) + change
        if positions[symbol] > 0:
            open_positions[symbol] = positions[symbol]
        elif symbol in open_positions:
            del open_positions[symbol]
    
    return final_rows, open_positions

@timing_decorator  
def filter_portfolio_transactions_ultra_fast(df, start_date, end_date):
    """Ultra-optimized filter transactions - target 2 seconds"""
    logger.info("🚀 Starting ultra-fast transaction filtering...")
    
    start_time = time.time()
    
    # Convert dates once
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    checkpoint1 = time.time()
    logger.info(f"⏱️  Date conversion: {checkpoint1 - start_time:.3f}s")
    
    # Split transactions with vectorized operations
    before_start = df[df['transaction_date'] < start_date]
    within_period = df[(df['transaction_date'] >= start_date) & (df['transaction_date'] <= end_date)]
    
    checkpoint2 = time.time()
    logger.info(f"⏱️  Data splitting: {checkpoint2 - checkpoint1:.3f}s")
    
    # Ultra-fast holdings calculation
    holdings = calculate_holdings_vectorized(before_start)
    
    checkpoint3 = time.time()
    logger.info(f"⏱️  Holdings calculation: {checkpoint3 - checkpoint2:.3f}s")
    
    # Batch fetch start prices if needed
    synthetic_buys = []
    if holdings:
        logger.info(f"📡 Batch fetching start prices for {len(holdings)} symbols...")
        start_date_str = start_date.strftime('%Y-%m-%d')
        
        # Run async price fetching
        start_prices = asyncio.run(
            ultra_fetcher.batch_fetch_prices_ultra_fast(list(holdings.keys()), start_date_str)
        )
        
        # Create synthetic buys
        for symbol, qty in holdings.items():
            price = start_prices.get(symbol)
            if price:
                synthetic_buys.append({
                    'transaction_date': start_date,
                    'name': symbol,
                    'transaction_type': 'Buy',
                    'quantity': qty,
                    'price': price,
                    'net_amount': qty * price,
                    'type': 'synthetic'
                })
    
    checkpoint4 = time.time()
    logger.info(f"⏱️  Start price fetching: {checkpoint4 - checkpoint3:.3f}s")
    
    # Combine transactions
    if synthetic_buys:
        combined = pd.concat([pd.DataFrame(synthetic_buys), within_period], ignore_index=True)
    else:
        combined = within_period.copy()
    
    # Ultra-fast position tracking
    final_rows, open_positions = build_position_tracker_vectorized(combined)
    
    checkpoint5 = time.time()
    logger.info(f"⏱️  Position tracking: {checkpoint5 - checkpoint4:.3f}s")
    
    # Batch fetch end prices
    remaining_symbols = [symbol for symbol, qty in open_positions.items() if qty > 0]
    if remaining_symbols:
        logger.info(f"📡 Batch fetching end prices for {len(remaining_symbols)} symbols...")
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Run async price fetching
        end_prices = asyncio.run(
            ultra_fetcher.batch_fetch_prices_ultra_fast(remaining_symbols, end_date_str)
        )
        
        # Add synthetic closes
        for symbol, qty in open_positions.items():
            if qty > 0:
                price = end_prices.get(symbol)
                if price:
                    final_rows.append({
                        'transaction_date': end_date,
                        'name': symbol,
                        'transaction_type': 'Sell',
                        'quantity': qty,
                        'price': price,
                        'net_amount': qty * price,
                        'type': 'synthetic_close'
                    })
    
    checkpoint6 = time.time()
    logger.info(f"⏱️  End price fetching: {checkpoint6 - checkpoint5:.3f}s")
    
    # Create final DataFrame
    final_df = pd.DataFrame(final_rows).sort_values('transaction_date').reset_index(drop=True)
    #final_df.to_csv('filtereda11.csv', index=False)
    
    total_time = time.time() - start_time
    logger.info(f"🚀 Ultra-fast filtering completed in {total_time:.3f}s")
    
    return final_df

# Ultra-fast single price function for backward compatibility
def get_price_for_date_ultra_fast(symbol, date):
    """Ultra-fast single price fetch with aggressive caching"""
    cache_key = f"{symbol}_{date}"
    
    # Check memory cache first
    if cache_key in PRICE_CACHE:
        return PRICE_CACHE[cache_key]
    
    # Check local cache
    local_cache_key = f"ultrafast_price_{symbol}_{date}"
    cached_price = cache_manager.get_from_local_cache(local_cache_key)
    if cached_price:
        price = float(cached_price)
        PRICE_CACHE[cache_key] = price
        return price
    
    # Fetch using async (convert to sync)
    result = asyncio.run(ultra_fetcher.batch_fetch_prices_ultra_fast([symbol], date))
    return result.get(symbol)

def get_price_for_date(symbol, date):
    """Ultra-fast wrapper for the original function"""
    return get_price_for_date_ultra_fast(symbol, date)

def adjust_to_trading_day(date_str, trading_days, direction="backward"):
    """Adjust date to nearest trading day"""
    date = datetime.strptime(date_str, "%Y-%m-%d")
    trading_set = set(trading_days)
    
    if direction == "backward":
        while date.strftime("%Y-%m-%d") not in trading_set:
            date -= timedelta(days=1)
    else:
        while date.strftime("%Y-%m-%d") not in trading_set:
            date += timedelta(days=1)
    
    return date.strftime("%Y-%m-%d")

# Optimized NAV calculation with vectorized operations
@timing_decorator
def calculate_nav_series_optimized(transactions_df, current_portfolio_value, calculation_end_date=None):
    """Optimized NAV series calculation with date-aware end date"""
    logger.info("\n=== CALCULATING NAV SERIES (OPTIMIZED) ===")
    
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    transactions_df = transactions_df.sort_values('transaction_date').copy()
    
    start_date = transactions_df['transaction_date'].min()
    
    # KEY CHANGE: Use calculation_end_date if provided, otherwise current date
    if calculation_end_date:
        end_date = pd.to_datetime(calculation_end_date)
        logger.info(f"📅 Using calculation end date: {calculation_end_date}")
    else:
        end_date = pd.Timestamp.now()
        logger.info(f"📅 Using current date as end date: {end_date.strftime('%Y-%m-%d')}")
    
    # Use parallel processing for price matrix
    price_matrix = build_stock_price_matrix_parallel(
        transactions_df,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Pre-process transactions for faster lookup
    daily_transactions = transactions_df.groupby('transaction_date')
    transaction_dates = set(transactions_df['transaction_date'].dt.date)
    
    # Vectorized NAV calculation
    nav_data = []
    stock_units = {}
    stock_wise_data = []
    cumulative_max = 0
    cumulative_max_nav = 0
    
    logger.info(f"\n🔄 Calculating NAV for {len(daily_dates)} days...")
    
    for i, current_date in enumerate(daily_dates):
        if i % 200 == 0:  # Reduced logging frequency
            logger.info(f"  Progress: {i}/{len(daily_dates)} days ({i/len(daily_dates)*100:.1f}%)")
        
        current_date_only = current_date.date()
        net_cash_flow = 0
        buy_amount = 0
        sell_amount = 0
        
        daily_stock_transactions = {}
        prev_nav = nav_data[-1]['nav'] if nav_data else 10.0
        
        # Process transactions (optimized)
        if current_date_only in transaction_dates:
            day_transactions = daily_transactions.get_group(current_date)
            
            # Vectorized transaction processing
            buy_mask = day_transactions['transaction_type'] == 'Buy'
            sell_mask = day_transactions['transaction_type'] == 'Sell'
            
            buy_amount = day_transactions[buy_mask]['net_amount'].sum()
            sell_amount = day_transactions[sell_mask]['net_amount'].sum()
            net_cash_flow = buy_amount - sell_amount
            
            # Process stock-level transactions
            for _, txn in day_transactions.iterrows():
                stock_symbol = txn['name']
                quantity = abs(txn['quantity'])
                amount = abs(txn['net_amount'])
                
                if stock_symbol not in daily_stock_transactions:
                    daily_stock_transactions[stock_symbol] = {
                        'buy_amount': 0, 'sell_amount': 0, 
                        'buy_qty': 0, 'sell_qty': 0, 'net_cash_flow': 0
                    }
                
                if txn['transaction_type'] == 'Buy':
                    daily_stock_transactions[stock_symbol]['buy_amount'] += amount
                    daily_stock_transactions[stock_symbol]['buy_qty'] += quantity
                    daily_stock_transactions[stock_symbol]['net_cash_flow'] += amount
                    
                    if stock_symbol not in stock_units:
                        stock_units[stock_symbol] = 0
                    units = amount / prev_nav if prev_nav > 0 else 0
                    stock_units[stock_symbol] += units
                    
                elif txn['transaction_type'] == 'Sell':
                    daily_stock_transactions[stock_symbol]['sell_amount'] += amount
                    daily_stock_transactions[stock_symbol]['sell_qty'] += quantity
                    daily_stock_transactions[stock_symbol]['net_cash_flow'] -= amount
                    
                    if stock_symbol in stock_units:
                        del stock_units[stock_symbol]
        
        # Calculate holdings and portfolio value (optimized)
        current_holdings = calculate_daily_holdings_optimized(transactions_df, current_date_only)
        
        portfolio_value = 0
        for stock, quantity in current_holdings.items():
            stock_price = None
            if stock in price_matrix and current_date_only in price_matrix[stock]:
                stock_price = price_matrix[stock][current_date_only]
            else:
                if stock in price_matrix:
                    available_dates = [d for d in price_matrix[stock].keys() if d <= current_date_only]
                    if available_dates:
                        nearest_date = max(available_dates)
                        stock_price = price_matrix[stock][nearest_date]
            
            if stock_price is not None:
                portfolio_value += quantity * stock_price
        
        # NAV calculation (same formula as original)
        total_units = sum(stock_units.values())
        
        if total_units > 0:
            nav = portfolio_value / total_units
        else:
            nav = prev_nav if prev_nav > 0 else 10.0
        
        if portfolio_value == 0 and sell_amount > 0 and total_units == 0:
            nav = prev_nav
        
        # MDD calculations (same formulas)
        if sell_amount > 0:
            cumulative_max = portfolio_value
        else:
            cumulative_max = max(cumulative_max, portfolio_value)
        mdd = cumulative_max - portfolio_value
        
        if nav_data and nav_data[-1]['nav'] == 0 and nav > 0:
            cumulative_max_nav = nav
        elif nav > 0 and nav > cumulative_max_nav:
            cumulative_max_nav = nav
        
        if len(current_holdings) == 0 and nav_data and portfolio_value == 0:
            drawdown = nav_data[-1]['drawdown']
        elif nav == 0:
            drawdown = 0
        else:
            drawdown = nav - cumulative_max_nav
        
        # Store data
        nav_data.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'units': total_units,
            'nav': nav,
            'net_cash_flow': net_cash_flow,
            'buy_amount': buy_amount,
            'sell_amount': sell_amount,
            'holdings_count': len(current_holdings),
            'stock_units': dict(stock_units),
            'max_portfolio_value': cumulative_max,
            'mdd': mdd,
            'cumulative_max': cumulative_max_nav,
            'drawdown': drawdown
        })
        
        # Stock-wise data collection (optimized)
        all_stocks = set(list(current_holdings.keys()) + list(stock_units.keys()) + list(daily_stock_transactions.keys()))
        for stock in all_stocks:
            stock_price = None
            if stock in price_matrix and current_date_only in price_matrix[stock]:
                stock_price = price_matrix[stock][current_date_only]
            else:
                if stock in price_matrix:
                    available_dates = [d for d in price_matrix[stock].keys() if d <= current_date_only]
                    if available_dates:
                        nearest_date = max(available_dates)
                        stock_price = price_matrix[stock][nearest_date]
            
            stock_quantity = current_holdings.get(stock, 0)
            stock_units_held = stock_units.get(stock, 0)
            stock_value = (stock_quantity * stock_price) if stock_price else 0
            
            default_txn = {
                'buy_amount': 0, 'sell_amount': 0, 
                'buy_qty': 0, 'sell_qty': 0, 'net_cash_flow': 0
            }
            stock_txn = daily_stock_transactions.get(stock, default_txn)
            
            stock_wise_data.append({
                'date': current_date,
                'stock_symbol': stock,
                'stock_price': stock_price,
                'quantity_held': stock_quantity,
                'units_tracked': stock_units_held,
                'stock_value': stock_value,
                'buy_amount': stock_txn['buy_amount'],
                'sell_amount': stock_txn['sell_amount'],
                'buy_quantity': stock_txn['buy_qty'],
                'sell_quantity': stock_txn['sell_qty'],
                'net_cash_flow': stock_txn['net_cash_flow'],
                'portfolio_nav': nav,
                'portfolio_total_units': total_units,
                'portfolio_value': portfolio_value
            })
    
    logger.info(f"✅ NAV calculation completed for {len(nav_data)} days")
    
    nav_df = pd.DataFrame(nav_data)
    stock_wise_df = pd.DataFrame(stock_wise_data)
    
    max_drawdown = nav_df['mdd'].max()
    max_nav_drawdown = nav_df['drawdown'].min()
    
    # Get final values
    final_nav = 10.0
    total_units = 0
    for i in range(len(nav_df) - 1, -1, -1):
        if nav_df.iloc[i]['holdings_count'] > 0:
            final_nav = nav_df.iloc[i]['nav']
            total_units = nav_df.iloc[i]['units']
            break
    
    if total_units == 0 and len(nav_df) > 0:
        final_nav = nav_df.iloc[-1]['nav']
        total_units = nav_df.iloc[-1]['units']
    
    logger.info(f"📊 Final NAV: ₹{final_nav:.4f}")
    logger.info(f"📊 Total Units: {total_units:.0f}")
    logger.info(f"📊 Maximum Drawdown (Portfolio): ₹{max_drawdown:,.2f}")
    logger.info(f"📊 Maximum Drawdown (NAV drop): ₹{abs(max_nav_drawdown):.4f}")
    
    return nav_df, stock_wise_df, max_nav_drawdown

@timing_decorator
def get_current_price_optimized(company_name, price_date=None):
    """Optimized current price fetching with caching - DATE AWARE VERSION"""
    if company_name not in ASSET_ISIN_MAPPING:
        logger.error(f"❌ Company {company_name} not found in ASSET_ISIN_MAPPING")
        return None
    
    # KEY CHANGE: Use price_date if provided, otherwise current date
    if price_date:
        target_date = pd.to_datetime(price_date).strftime('%Y-%m-%d')
        logger.info(f"📅 Fetching price for {company_name} as of {target_date}")
    else:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    # Check local cache first
    cache_key = f"current_price_{company_name}_{target_date}"
    cached_price = cache_manager.get_from_local_cache(cache_key)
    if cached_price:
        logger.info(f"💰 Using cached price for {company_name} as of {target_date}: ₹{cached_price}")
        return cached_price
    
    isin_code = ASSET_ISIN_MAPPING[company_name]
    instrument_key = f"BSE_EQ|{isin_code}" if isin_code=='INE765D01022' else f"NSE_EQ|{isin_code}"
    
    # Try up to 7 days back from target date to find trading day
    for days_back in range(7):
        try_date = (pd.to_datetime(target_date) - timedelta(days=days_back)).strftime('%Y-%m-%d')
        data = api_fetcher.fetch_upstox_candles(instrument_key, try_date, try_date, "day")
        
        if data and data.get('status') == 'success' and data.get('data'):
            candles = data['data'].get('candles', [])
            if len(candles) > 0:
                price = float(candles[0][4])
                # Cache the price
                cache_manager.set_to_local_cache(cache_key, price, 3600)  # 1 hour TTL
                logger.info(f"💰 Price for {company_name} as of {target_date}: ₹{price}")
                return price
    
    logger.error(f"❌ Could not fetch price for {company_name} as of {target_date}")
    return None

@timing_decorator
def calculate_benchmark_nav_optimized(nav_df_with_market_data, transactions_df):
    """Optimized benchmark NAV calculation"""
    logger.info("\n=== CALCULATING BENCHMARK NAV (OPTIMIZED) ===")
    
    df = nav_df_with_market_data.copy()
    transactions_df = transactions_df.copy()
    
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    transactions_df = transactions_df.sort_values('transaction_date')
    
    stock_benchmark_units = {}
    stock_benchmark_quantity = {}
    benchmark_cumulative_max = 0
    benchmark_cumulative_max_nav = 0
    benchmark_data = []
    
    # Optimize transaction grouping
    daily_transactions = transactions_df.groupby('transaction_date')
    transaction_dates = set(transactions_df['transaction_date'].dt.date)
    
    logger.info(f"🔄 Calculating Benchmark NAV for {len(df)} days...")
    
    # Vectorized processing where possible
    for i, row in df.iterrows():
        if i % 200 == 0:
            logger.info(f"  Progress: {i}/{len(df)} days ({i/len(df)*100:.1f}%)")
        
        current_date = row['date']
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date).date()
        elif hasattr(current_date, 'date'):
            current_date = current_date.date()
        
        buy_amount = row['buy_amount']
        sell_amount = row['sell_amount']
        market_open = row.get('market_open', 0)
        market_close = row.get('market_close', 0)
        
        # Get all portfolio metrics from the row
        portfolio_value = row.get('portfolio_value', 0)
        portfolio_units = row.get('units', 0)
        portfolio_nav = row.get('nav', 10.0)
        net_cash_flow = row.get('net_cash_flow', 0)
        holdings_count = row.get('holdings_count', 0)
        portfolio_mdd = row.get('mdd', 0)
        portfolio_drawdown = row.get('drawdown', 0)
        portfolio_cumulative_max = row.get('cumulative_max', 0)
        
        prev_benchmark_nav = benchmark_data[-1]['benchmark_nav'] if benchmark_data else 10.0
        
        benchmark_buy_amount = 0
        benchmark_sell_amount = 0
        benchmark_net_cash_flow = 0
        
        # Process stock-level transactions
        if current_date in transaction_dates:
            current_date_transactions = daily_transactions.get_group(pd.Timestamp(current_date))
            
            for _, txn in current_date_transactions.iterrows():
                stock_symbol = txn['name']
                transaction_type = txn['transaction_type']
                amount = abs(txn['net_amount'])
                
                if transaction_type == 'Buy' and market_open > 0:
                    benchmark_buy_amount += amount
                    benchmark_net_cash_flow += amount
                    
                    ben_units = amount / prev_benchmark_nav if prev_benchmark_nav > 0 else 0
                    ben_quantity = amount / market_open
                    
                    if stock_symbol not in stock_benchmark_units:
                        stock_benchmark_units[stock_symbol] = 0
                        stock_benchmark_quantity[stock_symbol] = 0
                    
                    stock_benchmark_units[stock_symbol] += ben_units
                    stock_benchmark_quantity[stock_symbol] += ben_quantity
                
                elif transaction_type == 'Sell':
                    benchmark_sell_amount += amount
                    benchmark_net_cash_flow -= amount
                    
                    if stock_symbol in stock_benchmark_units:
                        del stock_benchmark_units[stock_symbol]
                        del stock_benchmark_quantity[stock_symbol]
        
        # Calculate benchmark metrics
        total_benchmark_units = sum(stock_benchmark_units.values())
        total_benchmark_quantity = sum(stock_benchmark_quantity.values())
        
        benchmark_value = 0
        benchmark_nav = prev_benchmark_nav
        
        if total_benchmark_quantity > 0 and total_benchmark_units > 0 and market_close > 0:
            benchmark_value = total_benchmark_quantity * market_close
            benchmark_nav = benchmark_value / total_benchmark_units
        elif total_benchmark_units == 0 and total_benchmark_quantity == 0:
            benchmark_value = 0
            benchmark_nav = prev_benchmark_nav
        else:
            benchmark_nav = prev_benchmark_nav
        
        if benchmark_value == 0 and benchmark_sell_amount > 0 and total_benchmark_units == 0:
            benchmark_nav = prev_benchmark_nav
        
        # Benchmark MDD calculations
        benchmark_cumulative_max = max(benchmark_cumulative_max, benchmark_value)
        benchmark_mdd = benchmark_cumulative_max - benchmark_value
        
        if benchmark_data and benchmark_data[-1]['benchmark_nav'] == 0 and benchmark_nav > 0:
            benchmark_cumulative_max_nav = benchmark_nav
        elif benchmark_nav > 0 and benchmark_nav > benchmark_cumulative_max_nav:
            benchmark_cumulative_max_nav = benchmark_nav
        
        if total_benchmark_units == 0 and benchmark_data and benchmark_value == 0:
            benchmark_drawdown = benchmark_data[-1]['benchmark_drawdown']
        elif benchmark_nav == 0:
            benchmark_drawdown = 0
        else:
            benchmark_drawdown = benchmark_nav - benchmark_cumulative_max_nav
        
        benchmark_data.append({
            'date': pd.Timestamp(current_date),
            'portfolio_value': portfolio_value,
            'portfolio_units': portfolio_units,
            'portfolio_nav': portfolio_nav,
            'portfolio_net_cash_flow': net_cash_flow,
            'portfolio_buy_amount': buy_amount,
            'portfolio_sell_amount': sell_amount,
            'holdings_count': holdings_count,
            'portfolio_mdd': portfolio_mdd,
            'portfolio_drawdown': portfolio_drawdown,
            'portfolio_cumulative_max': portfolio_cumulative_max,
            'benchmark_units': total_benchmark_units,
            'benchmark_quantity': total_benchmark_quantity,
            'benchmark_value': benchmark_value,
            'benchmark_nav': benchmark_nav,
            'benchmark_buy_amount': benchmark_buy_amount,
            'benchmark_sell_amount': benchmark_sell_amount,
            'benchmark_net_cash_flow': benchmark_net_cash_flow,
            'market_open': market_open,
            'market_close': market_close,
            'benchmark_max_value': benchmark_cumulative_max,
            'benchmark_mdd': benchmark_mdd,
            'benchmark_cumulative_max': benchmark_cumulative_max_nav,
            'benchmark_drawdown': benchmark_drawdown,
            'stock_benchmark_units': dict(stock_benchmark_units),
            'stock_benchmark_quantity': dict(stock_benchmark_quantity)
        })
    
    logger.info(f"✅ Benchmark NAV calculation completed for {len(benchmark_data)} days")
    
    benchmark_df = pd.DataFrame(benchmark_data)
    max_benchmark_drawdown = benchmark_df['benchmark_mdd'].max()
    max_benchmark_nav_drawdown = benchmark_df['benchmark_drawdown'].min()
    
    final_benchmark_nav = benchmark_data[-1]['benchmark_nav'] if benchmark_data else 10.0
    final_benchmark_units = benchmark_data[-1]['benchmark_units'] if benchmark_data else 0
    final_benchmark_quantity = benchmark_data[-1]['benchmark_quantity'] if benchmark_data else 0
    
    logger.info(f"📊 Final Benchmark NAV: ₹{final_benchmark_nav:.4f}")
    logger.info(f"📊 Final Benchmark Units: {final_benchmark_units:.0f}")
    logger.info(f"📊 Maximum Benchmark Drawdown: ₹{abs(max_benchmark_nav_drawdown):.4f}")
    
    return benchmark_df, max_benchmark_nav_drawdown

@timing_decorator
def get_nifty500_data_optimized(start_date, end_date):
    """Optimized Nifty 500 data fetching with caching"""
    cache_key = f"nifty50_{start_date}_{end_date}"
    cached_data = cache_manager.get_from_local_cache(cache_key)
    
    if cached_data:
        logger.info("💰 Using cached Nifty 50 data")
        return pd.DataFrame(cached_data)
    
    instrument_key = "NSE_INDEX|Nifty 50"
    logger.info(f"Fetching Nifty 50 data from {start_date} to {end_date}")
    
    data = api_fetcher.fetch_upstox_candles(instrument_key, start_date, end_date, "day")
    
    if data and data.get('status') == 'success' and data.get('data'):
        candles = data['data'].get('candles', [])
        if len(candles) > 0:
            nifty_df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'open_interest'])
            nifty_df['date'] = pd.to_datetime(nifty_df['timestamp']).dt.date
            nifty_df['close'] = nifty_df['close'].astype(float)
            nifty_df = nifty_df.sort_values('date').reset_index(drop=True)
            result_df = nifty_df[['date', 'open', 'close']].rename(columns={'close': 'market_close', 'open': 'market_open'})
            
            # Cache the result
            cache_manager.set_to_local_cache(cache_key, result_df.to_dict('records'), 3600)
            return result_df
    
    logger.error("Could not fetch Nifty 500 data")
    return None

# Optimized XIRR calculation
@jit(nopython=True)
def npv_fast(rate, cash_flows, days_from_start):
    """Fast NPV calculation using Numba"""
    npv_value = 0.0
    for i in range(len(cash_flows)):
        npv_value += cash_flows[i] / ((1 + rate) ** (days_from_start[i] / 365.0))
    return npv_value

def calculate_xirr_optimized(cash_flows, dates):
    """Optimized XIRR calculation"""
    if len(cash_flows) != len(dates):
        return None
    
    # Convert to numpy arrays for faster computation
    cash_flows_array = np.array(cash_flows, dtype=np.float64)
    start_date = dates[0]
    days_from_start = np.array([(date - start_date).days for date in dates], dtype=np.float64)
    
    def npv(rate):
        return npv_fast(rate, cash_flows_array, days_from_start)
    
    try:
        return newton(npv, 0.1, maxiter=100)
    except Exception as e1:
        try:
            return brentq(npv, -0.99, 10)
        except Exception as e2:
            logger.error(f"❌ XIRR calculation failed: {e1}, {e2}")
            return None

@timing_decorator
def market_xirr_optimized(transactions_df):
    """Optimized market XIRR calculation"""
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date']).dt.date
    
    start_date = transactions_df['transaction_date'].min()
    end_date = pd.to_datetime('today').date()
    
    # Optimized market data fetching
    market_df = get_nifty500_data_optimized(str(start_date), str(end_date))
    
    market_data = {}
    if market_df is not None and not market_df.empty:
        market_df['date'] = pd.to_datetime(market_df['date']).dt.date
        grouped = market_df.groupby('date').first()
        for date, row in grouped.iterrows():
            market_data[date] = (row['market_open'], row['market_close'])
    
    # Vectorized market data mapping
    transactions_df['market_open'] = transactions_df['transaction_date'].map(lambda d: market_data.get(d, (None, None))[0])
    transactions_df['market_close'] = transactions_df['transaction_date'].map(lambda d: market_data.get(d, (None, None))[0])
    
    transactions_df['market_open'] = transactions_df['market_open'].ffill()
    transactions_df['market_close'] = transactions_df['market_close'].ffill()
    
    # Optimized quantity calculation
    m_qty = []
    prev_qty_per_stock = {}
    
    for idx, row in transactions_df.iterrows():
        stock = row['name']
        if row['transaction_type'] == 'Buy':
            qty = row['net_amount'] / row['market_open']
        else:
            qty = prev_qty_per_stock.get(stock, 0)
        
        prev_qty_per_stock[stock] = qty
        m_qty.append(qty)
    
    transactions_df['Mkt_qty'] = m_qty
    transactions_df['Mkt_value'] = transactions_df['Mkt_qty'] * transactions_df['market_open']
    
    transactions_df['Mkt_Totalamount'] = transactions_df.apply(
        lambda x: -x['Mkt_value'] if x['transaction_type'] == 'Buy' else x['Mkt_value'],
        axis=1
    )
    
    cash_flows_mkt = transactions_df['Mkt_Totalamount'].tolist()
    dates_mkt = transactions_df['transaction_date'].tolist()
    
    return calculate_xirr_optimized(cash_flows_mkt, dates_mkt)

@timing_decorator
def calculate_portfolio_metrics_optimized(transactions_df, calculation_end_date=None):
    """Optimized comprehensive portfolio metrics calculation - DATE AWARE VERSION"""
    logger.info("\n=== CALCULATING PORTFOLIO METRICS (OPTIMIZED) ===")
    
    # KEY CHANGE: Set effective_end_date for all calculations
    if calculation_end_date:
        effective_end_date = pd.to_datetime(calculation_end_date)
        logger.info(f"📅 Calculating metrics as of: {calculation_end_date}")
    else:
        effective_end_date = pd.Timestamp.now()
        logger.info(f"📅 Calculating metrics as of current date: {effective_end_date.strftime('%Y-%m-%d')}")
    
    # 1. Current holdings calculation with optimization
    logger.info("\n1. Current Holdings Calculation")
    
    current_holdings = transactions_df.copy()
    
    
    logger.info(f"Current holdings: {len(current_holdings)} stocks")
    
    # 2. Parallel current price fetching - DATE AWARE
    logger.info(f"\n2. Fetching Prices as of {effective_end_date.strftime('%Y-%m-%d')} (Parallel)")
    current_prices = {}
    failed_prices = []
    
    def fetch_price(company_name):
        # KEY CHANGE: Pass calculation date to price function
        price = get_current_price_optimized(company_name, effective_end_date.strftime('%Y-%m-%d'))
        return company_name, price
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_stock = {executor.submit(fetch_price, name): name for name in current_holdings['name'].unique()}
        
        for future in as_completed(future_to_stock):
            company_name, price = future.result()
            if price:
                current_prices[company_name] = price
            else:
                failed_prices.append(company_name)
    
    if failed_prices:
        logger.warning(f"⚠️ Failed to get prices for: {failed_prices}")
    
    # 3. Portfolio value calculations (vectorized)
    logger.info("\n3. Portfolio Value Calculations")
    current_holdings = current_holdings[current_holdings['name'].isin(current_prices.keys())].copy()
    
    if len(current_holdings) == 0:
        logger.error("❌ No stocks with valid prices found!")
        return None
    
    # Vectorized calculations
    current_holdings['current_price'] = current_holdings['name'].map(current_prices)    
    current_holdings['current_value'] = current_holdings['quantity'] * current_holdings['current_price']    
    # current_holdings.to_csv('current_holdings_raw.csv', index=False)
    
    
    # 4. Optimized XIRR calculation - DATE AWARE
    logger.info("\n4. XIRR Calculation (Optimized)")
    current_holdings['totalamount'] = current_holdings.apply(
        lambda row: -abs(row['net_amount']) if row['transaction_type'] == 'Buy' else abs(row['net_amount']),
        axis=1
    )
    
    transactions_df['totalamount'] = transactions_df.apply(
        lambda row: -abs(row['net_amount']) if row['transaction_type'] == 'Buy' else abs(row['net_amount']),
        axis=1
    )

    total_investment_value = current_holdings.loc[current_holdings['transaction_type'] == 'Buy', 'net_amount'].sum()
    total_current_value = current_holdings.loc[current_holdings['transaction_type'] == 'Buy', 'current_value'].sum() 
    total_absolute_gain = current_holdings['totalamount'].sum()
    total_gain_percent = (total_absolute_gain / total_investment_value) * 100

    # Portfolio totals
    # total_investment_value = current_holdings['investment_value'].sum()
    
    # total_absolute_gain = current_holdings['absolute_gain'].sum()
    # total_gain_percent = (total_absolute_gain / total_investment_value) * 100
    
    logger.info(f"Total Investment Value: ₹{total_investment_value:,.2f}")
    logger.info(f"Total Current Value: ₹{total_current_value:,.2f}")
    logger.info(f"Total Absolute Gain: ₹{total_absolute_gain:,.2f}")
    logger.info(f"Total Gain %: {total_gain_percent:.2f}%")


    #transactions_df.to_csv('trinside.csv', index=False)
    transactions_df = transactions_df.sort_values(by='transaction_date')
    cash_flows = transactions_df['totalamount'].tolist()
    dates = transactions_df['transaction_date'].tolist()
    
    xirr = calculate_xirr_optimized(cash_flows, dates)
    portfolio_xirr = xirr
    xirr = xirr * 100 
    logger.info(f"XIRR: {xirr:.2f}%")
    
    # Market XIRR
    mkt_xirr = market_xirr_optimized(transactions_df)
    bnmk_xirr = mkt_xirr
    mkt_xirr = mkt_xirr * 100 
    logger.info(f"Market XIRR: {mkt_xirr}")
    
    # 5. Time-based returns - DATE AWARE
    logger.info("\n5. Time-based Returns")
    first_transaction_date = transactions_df['transaction_date'].min()
    first_transaction_date = pd.to_datetime(first_transaction_date)
    
    # KEY CHANGE: Calculate days from first transaction to calculation end date (not current date)
    days_invested = (effective_end_date - first_transaction_date).days
    
    if days_invested > 0:        
        annualized_return = (((total_absolute_gain / total_investment_value) * (365.0 / days_invested)) ) * 100
    else:
        annualized_return = 0
    
    logger.info(f"Days Invested: {days_invested}")
    logger.info(f"Annualized Return: {annualized_return:.2f}%")
    
    # 6. Optimized NAV and Drawdown Calculation - DATE AWARE
    logger.info("\n6. NAV and Drawdown Calculation (Optimized)")
    # KEY CHANGE: Pass calculation end date to NAV function
    nav_df, stock_wise_df, max_nav_drawdown = calculate_nav_series_optimized(
        transactions_df, 
        total_current_value, 
        calculation_end_date
    )
    
    # Save results
    #nav_df.to_csv('NAV_optimized.csv', index=False)
    
    # 7. Market data and benchmark calculations - DATE AWARE
    start_date = first_transaction_date.strftime('%Y-%m-%d')
    end_date = effective_end_date.strftime('%Y-%m-%d')  # KEY CHANGE: Use calculation end date
    
    nifty_data = get_nifty500_data_optimized(start_date, end_date)
    
    if nifty_data is not None:
        nifty_data['date'] = pd.to_datetime(nifty_data['date'])
        nav_df['date'] = nav_df['date'].dt.date
        nifty_data['date'] = nifty_data['date'].dt.date
        
        merged_data = pd.merge(nav_df, nifty_data, on='date', how='inner')
        benchmark_df, max_benchmark_drawdown = calculate_benchmark_nav_optimized(merged_data, transactions_df)
        #benchmark_df.to_csv('Benchmark_optimized3.csv', index=False)
        #stock_wise_df.to_csv('Stock_Wise_optimized.csv', index=False)
    else:
        benchmark_df = nav_df
        max_benchmark_drawdown = 0
        
    # 8. Risk metrics calculation (optimized) - DATE AWARE
    logger.info("\n7. Risk Metrics Calculation (Optimized)")
    ben_nav_df = benchmark_df.copy()
    
    # Vectorized return calculations
    ben_nav_df['prev_nav'] = ben_nav_df['portfolio_nav'].shift(1)
    ben_nav_df['portfolio_return'] = np.where(
        (ben_nav_df['prev_nav'] == 0) | (ben_nav_df['portfolio_nav'] == 0) | (ben_nav_df['holdings_count'] == 0),
        0,
        ((ben_nav_df['portfolio_nav'] - ben_nav_df['prev_nav']) / ben_nav_df['prev_nav']) * 100
    )
    
    ben_nav_df['market_return'] = ((ben_nav_df['benchmark_nav'] - ben_nav_df['benchmark_nav'].shift(1)) / ben_nav_df['benchmark_nav'].shift(1)) * 100
    ben_nav_df = ben_nav_df.dropna(subset=['market_return'])
    
    # Vectorized statistical calculations
    port_avg = ben_nav_df['portfolio_return'].mean()
    market_avg = ben_nav_df['market_return'].mean()
    
    ben_nav_df['D1'] = ben_nav_df['portfolio_return'] - port_avg
    ben_nav_df['D2'] = ben_nav_df['market_return'] - market_avg
    ben_nav_df['D1_D2'] = ben_nav_df['D1'] * ben_nav_df['D2']
    
    N = len(ben_nav_df)
    covariance = ben_nav_df['D1_D2'].sum() / (N - 1)
    market_variance = (ben_nav_df['D2'] ** 2).sum() / (N - 1)
    portfolio_variance = (ben_nav_df['D1'] ** 2).sum() / (N - 1)
    
    market_std = np.sqrt(market_variance)
    portfolio_std = np.sqrt(portfolio_variance)
    beta = covariance / market_variance
    correlation = ben_nav_df['portfolio_return'].corr(ben_nav_df['market_return'])
    print('portfolio_variance', portfolio_variance)
    print('portfolio_std', portfolio_std)
    print('correlation',correlation)
    # Portfolio and market returns
    df_with_holdings = ben_nav_df[ben_nav_df['holdings_count'] > 0]
    first_portfolio_return = ben_nav_df['portfolio_nav'].iloc[0]
    first_market_return = ben_nav_df['benchmark_nav'].iloc[0]
    last_portfolio_return = df_with_holdings['portfolio_nav'].iloc[-1]
    last_market_return = df_with_holdings['benchmark_nav'].iloc[-1]
    
    portfolio_return = (last_portfolio_return - first_portfolio_return) / first_portfolio_return
    nifty_return = (last_market_return - first_market_return) / first_market_return
    
    risk_free_rate = 0.06

    sharpe_ratio = (portfolio_xirr - risk_free_rate) / portfolio_std
    bnmk_sharpe_ratio = (bnmk_xirr - risk_free_rate) / market_std

    alpha = portfolio_xirr - (risk_free_rate + beta * (bnmk_xirr - risk_free_rate))
    bnmk_alpha = bnmk_xirr - (risk_free_rate + beta * (bnmk_xirr - risk_free_rate))
    volatility = portfolio_std
    bnmk_volatility = market_std
    #ben_nav_df.to_csv('Beta_Calculations_Optimized.csv', index=False)
    logger.info(f"Market XIRR: {mkt_xirr}")
    logger.info(f"Beta: {beta:.5f}")
    logger.info(f"Alpha: {alpha:.5f}%")
    logger.info(f"Benchmark Alpha: {bnmk_alpha:.5f}%")
    logger.info(f"Volatility: {volatility:.5f}%")
    logger.info(f"Benchmark Volatility: {bnmk_volatility:.5f}%")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.5f}")
    logger.info(f"Benchmark Sharpe Ratio: {bnmk_sharpe_ratio:.5f}")
    logger.info(f"Maximum NAV Drawdown: {max_nav_drawdown:.2f}%")
    logger.info(f'Maximum Benchmark Drawdown: {max_benchmark_drawdown:.2f}%')
    logger.info(f"Correlation: {correlation:.2f}%")
    # 9. Portfolio composition
    logger.info("\n8. Portfolio Composition")
    current_holdings['weight'] = (current_holdings['current_value'] / total_current_value) * 100
    
    # 10. Comprehensive summary - DATE AWARE
    metrics_summary = {
        'Calculation Date': effective_end_date.strftime('%Y-%m-%d'),
        'Total Portfolio Value (Current)': f"₹{total_current_value:,.2f}",
        'Total Investment Value (Cost)': f"₹{total_investment_value:,.2f}",
        'Total Gains': f"₹{total_absolute_gain:,.2f}",
        'Total Gains %': f"{total_gain_percent:.2f}%",
        'XIRR': f"{xirr:.4f}%",
        'Benchmark XIRR': f"{mkt_xirr:.4f}%",
        'Annualized Return': f"{annualized_return:.2f}%",
        'Beta': f"{beta:.2f}",
        'Alpha': f"{alpha:.2f}%",
        'Volatility': f"{volatility:.2f}%",
        'Benchmark Volatility': f"{bnmk_volatility:.2f}%",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Benchmark Sharpe Ratio': f"{bnmk_sharpe_ratio:.2f}",
        'Maximum NAV Drawdown': f"{max_nav_drawdown:.2f}%",
        'Maximum BEN Drawdown': f"{max_benchmark_drawdown:.2f}%",
        'Days Invested': f"{days_invested}",
        'Active Instruments': f"{len(current_holdings)}",
        'Current NAV': f"{df_with_holdings['portfolio_nav'].iloc[-1]:.4f}",
        'benchmark_nav': f"{df_with_holdings['benchmark_nav'].iloc[-1]:.4f}",
        'Stocks with Missing Prices': f"{len(failed_prices)}"
    }
    
    return {
        'summary_metrics': metrics_summary,
        'current_holdings': current_holdings,
        'nav_data': nav_df,        
        'portfolio_composition': current_holdings[['name', 'current_value', 'weight']],#, 'gain_percent']],
        'failed_prices': failed_prices
    }

def clear_cache():
    """Clear all cached data"""
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        if os.path.exists(CACHE_METADATA_FILE):
            os.remove(CACHE_METADATA_FILE)
        if os.path.exists(LOCAL_CACHE_FILE):
            os.remove(LOCAL_CACHE_FILE)
        
        # Clear local cache
        global LOCAL_CACHE, LOCAL_CACHE_TIMESTAMPS
        LOCAL_CACHE.clear()
        LOCAL_CACHE_TIMESTAMPS.clear()
        
        logger.info("🧹 Cache cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")

def get_cache_info():
    """Get information about the cache"""
    if os.path.exists(CACHE_METADATA_FILE):
        try:
            with open(CACHE_METADATA_FILE, 'r') as f:
                metadata = json.load(f)
            return metadata
        except:
            return None
    return None

def show_cache_stats():
    """Show detailed cache statistics"""
    cache_info = get_cache_info()
    if cache_info:
        logger.info(f"Cache Statistics:")
        logger.info(f"  Stocks: {cache_info['stocks_count']}")
        logger.info(f"  Data Points: {cache_info['total_data_points']}")
        logger.info(f"  Last Updated: {cache_info['last_updated']}")
        
        if os.path.exists(CACHE_FILE):
            cache_size = os.path.getsize(CACHE_FILE) / (1024 * 1024)
            logger.info(f"  Cache Size: {cache_size:.2f} MB")
        
        if os.path.exists(LOCAL_CACHE_FILE):
            local_cache_size = os.path.getsize(LOCAL_CACHE_FILE) / (1024 * 1024)
            logger.info(f"  Local Cache Size: {local_cache_size:.2f} MB")
            logger.info(f"  Local Cache Entries: {len(LOCAL_CACHE)}")
    else:
        logger.info("No cache found")

@timing_decorator
def update_cache_for_recent_data():
    """Update cache with recent data for all stocks"""
    logger.info("Updating cache with recent data...")
    
    cache_data = load_cache()
    today = datetime.now().strftime('%Y-%m-%d')
    week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    updated_stocks = 0
    
    def update_stock(stock_name):
        recent_data = get_historical_stock_prices_optimized(
            stock_name, week_ago, today, cache_data
        )
        return recent_data is not None
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_stock = {executor.submit(update_stock, stock): stock for stock in ASSET_ISIN_MAPPING.keys()}
        
        for future in as_completed(future_to_stock):
            if future.result():
                updated_stocks += 1
    
    if updated_stocks > 0:
        save_cache(cache_data)
        logger.info(f"✅ Updated cache for {updated_stocks} stocks")

@timing_decorator
def main_optimized(start_date=None, end_date=None):
    """Main optimized function with optional date filtering - DATE AWARE VERSION"""
    logger.info("=== OPTIMIZED PORTFOLIO METRICS DASHBOARD ===\n")
    
    # Load transaction data
    logger.info("Loading transaction data...")
    transactions_df = pd.read_csv("Transactions_sample.csv")
    #strategy_id = 2
    #transactions_df = db_manager.get_transactions_from_db(strategy_id)
    transactions_df['price'] = transactions_df['price'].astype(float)
    transactions_df['net_amount'] = transactions_df['net_amount'].astype(float)
    transactions_df['quantity'] = transactions_df['quantity'].astype(float)
    print('#####################$$$$$$$$$$$$$$$$$transactions_df \n', transactions_df)
    #time.sleep(10000)
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    logger.info(f"Loaded {len(transactions_df)} transactions")
    
    calculation_end_date = None  # Default to current date
    
    # Apply date filtering if provided
    if start_date and end_date:
        logger.info(f"Filtering transactions from {start_date} to {end_date}")
        
        # KEY CHANGE: Set calculation_end_date for all subsequent calculations
        calculation_end_date = end_date
        
        # Generate trading days for date adjustment
        trading_days = pd.date_range("2000-01-01", "2025-12-31", freq="B").strftime("%Y-%m-%d").tolist()
        
        # Adjust dates to trading days
        adjusted_start = adjust_to_trading_day(start_date, trading_days, "backward")
        adjusted_end = adjust_to_trading_day(end_date, trading_days, "backward")
        
        logger.info(f"Adjusted dates: {adjusted_start} to {adjusted_end}")
        transactions_df['quantity'] = transactions_df['quantity'].astype(float)
        transactions_df['price'] = transactions_df['price'].astype(float)
        transactions_df['net_amount'] = transactions_df['net_amount'].astype(float)
        
        print(transactions_df.info())
        # Filter transactions
        transactions_df = filter_portfolio_transactions_ultra_fast(transactions_df, adjusted_start, adjusted_end)  
        print('transactions_df after filter', transactions_df)      
        logger.info(f"Filtered to {len(transactions_df)} transactions")
        # time.sleep(100000)
    
    # Show cache info
    cache_info = get_cache_info()
    if cache_info:
        logger.info(f"📊 Cache Status: {cache_info['stocks_count']} stocks, {cache_info['total_data_points']} data points")
    else:
        logger.info("📊 No cache found - will build fresh cache")
    
    # Calculate all metrics with optimization - DATE AWARE
    start_time = time.time()
    # KEY CHANGE: Pass calculation_end_date to metrics calculation
    results = calculate_portfolio_metrics_optimized(transactions_df, calculation_end_date)
    end_time = time.time()
    
    if results is None:
        logger.error("❌ Portfolio calculation failed - no valid stock prices found")
        return None
    
    logger.info(f"\n⏱️  Total calculation time: {end_time - start_time:.2f} seconds")
    
    # Display results
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZED PORTFOLIO DASHBOARD SUMMARY")
    logger.info("="*80)
    
    for metric, value in results['summary_metrics'].items():
        logger.info(f"{metric:.<40} {value}")
    
    # Show failed stocks if any
    if results['failed_prices']:
        logger.warning(f"\n⚠️ WARNING: Failed to get prices for {len(results['failed_prices'])} stocks:")
        for stock in results['failed_prices']:
            logger.info(f"   - {stock}")
    
    # Save results to CSV files with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info("\n" + "="*80)
    logger.info("SAVING OPTIMIZED RESULTS")
    logger.info("="*80)
    
    # 1. Summary metrics
    summary_df = pd.DataFrame(list(results['summary_metrics'].items()), 
                             columns=['Metric', 'Value'])
    #summary_df.to_csv(f'portfolio_summary_optimized_{timestamp}.csv', index=False)
    logger.info("✅ portfolio_summary_optimized.csv")
    
    # 2. Current holdings
    #results['current_holdings'].to_csv(f'current_holdings_optimized_{timestamp}.csv', index=False)
    logger.info("✅ current_holdings_optimized.csv")
    
    # 3. NAV data
    #results['nav_data'].to_csv(f'nav_series_optimized_{timestamp}.csv', index=False)
    logger.info("✅ nav_series_optimized.csv")
    
    # 4. Portfolio composition
    #results['portfolio_composition'].to_csv(f'portfolio_composition_optimized_{timestamp}.csv', index=False)
    logger.info("✅ portfolio_composition_optimized.csv")
    
    # 5. Failed stocks
    if results['failed_prices']:
        failed_df = pd.DataFrame({'Failed_Stocks': results['failed_prices']})
        #failed_df.to_csv(f'failed_stocks_optimized_{timestamp}.csv', index=False)
        logger.info("✅ failed_stocks_optimized.csv")
    
    # Show final cache info
    final_cache_info = get_cache_info()
    if final_cache_info:
        logger.info(f"\n📊 Final Cache Status: {final_cache_info['stocks_count']} stocks, {final_cache_info['total_data_points']} data points")
    
    logger.info(f"\n📊 Local Cache Status: {len(LOCAL_CACHE)} entries")
    
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION COMPLETE!")
    logger.info("="*80)
    
    return results

# Enhanced memory management
def optimize_memory():
    """Optimize memory usage"""
    import gc
    gc.collect()
    
    # Clear large objects from memory
    if 'cache_manager' in globals():
        cache_manager.file_cache.clear()
    
    # Save and clear local cache periodically
    save_local_cache()
    
    logger.info("🧹 Memory optimized")

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.checkpoints = {}
    
    def start(self):
        self.start_time = time.time()
        logger.info("⏱️ Performance monitoring started")
    
    def checkpoint(self, name):
        if self.start_time is None:
            self.start()
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        self.checkpoints[name] = elapsed
        logger.info(f"⏱️  Checkpoint '{name}': {elapsed:.2f}s")
    
    def summary(self):
        if not self.checkpoints:
            return
        
        logger.info("\n⏱️ Performance Summary:")
        for name, elapsed in self.checkpoints.items():
            logger.info(f"  {name}: {elapsed:.2f}s")
        
        total_time = max(self.checkpoints.values())
        logger.info(f"  Total Time: {total_time:.2f}s")

# Configuration for different environments
class Config:
    DEVELOPMENT = {
        'local_cache_ttl': 3600,  # 1 hour
        'max_workers': 5,
        'batch_size': 100,
        'log_level': logging.DEBUG
    }
    
    PRODUCTION = {
        'local_cache_ttl': 86400,  # 24 hours
        'max_workers': 20,
        'batch_size': 500,
        'log_level': logging.INFO
    }
    
    @classmethod
    def get_config(cls, env='DEVELOPMENT'):
        return getattr(cls, env, cls.DEVELOPMENT)

# Usage examples and testing functions
def run_performance_test():
    """Run performance test with monitoring"""
    monitor = PerformanceMonitor()
    monitor.start()
    
    # Test with small dataset
    logger.info("Running performance test...")
    
    try:
        # Load sample data
        monitor.checkpoint("Data Loading")
        
        # Run optimization
        results = main_optimized()
        monitor.checkpoint("Portfolio Calculation")
        
        # Memory cleanup
        optimize_memory()
        monitor.checkpoint("Memory Cleanup")
        
        monitor.summary()
        
        if results:
            logger.info("✅ Performance test completed successfully")
            return True
        else:
            logger.error("❌ Performance test failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Performance test error: {e}")
        return False

def run_with_date_filter(start_date, end_date):
    """Run calculation with date filtering"""
    logger.info(f"Running portfolio calculation from {start_date} to {end_date}")
    return main_optimized(start_date, end_date)

# Main execution
if __name__ == "__main__":
    # Set configuration
    config = Config.get_config('PRODUCTION')
    logging.getLogger().setLevel(config['log_level'])
    
    # Update local cache TTL from config
    LOCAL_CACHE_TTL = config['local_cache_ttl']
    
    # Show cache statistics
    show_cache_stats()
    
    # Optional: Clear cache
    # clear_cache()
    
    # Optional: Update cache with recent data
    # update_cache_for_recent_data()
    
    # Run main calculation
    try:
        # Example 1: Run without date filtering
        results = main_optimized()
        
        # Example 2: Run with date filtering (NOW DATE-AWARE)
        #results = run_with_date_filter("2024-06-01", "2024-06-30")
        
        # Example 3: Run performance test
        # run_performance_test()
        pass
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        # Cleanup and save local cache
        save_local_cache()
        optimize_memory()
        logger.info("Process completed")