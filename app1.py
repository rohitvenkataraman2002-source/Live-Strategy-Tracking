import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
import traceback
from werkzeug.utils import secure_filename
import logging
import tempfile
import shutil

# Import your existing script
import live_track.NRED1 as NRED1

# Configure Streamlit page
st.set_page_config(
    page_title="Portfolio Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    .metrics-table {
        font-size: 12px;
    }
    
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
    
    .neutral {
        color: #6c757d;
    }
    
    .overall-row {
        background-color: #e8f5e8;
        font-weight: bold;
    }
    
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        color: rgb(30, 103, 119);
        overflow-wrap: break-word;
    }
    
    div[data-testid="metric-container"] > label[data-testid="metric-label"] > div {
        overflow-wrap: break-word;
        white-space: break-spaces;
        color: red;
    }
</style>
""", unsafe_allow_html=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_required_columns():
    """Get the required columns for portfolio analysis"""
    return {
        'transaction_date': {
            'description': 'Date of transaction (YYYY-MM-DD format)',
            'example': '2024-01-15',
            'required': True
        },
        'name': {
            'description': 'Stock/Asset name or symbol',
            'example': 'RELIANCE',
            'required': True
        },
        'price': {
            'description': 'Price per unit',
            'example': '2450.50',
            'required': True
        },
        'net_amount': {
            'description': 'Total transaction amount',
            'example': '24505.00',
            'required': True
        },
        'transaction_type': {
            'description': 'Type of transaction (Buy/Sell)',
            'example': 'Buy',
            'required': True
        },
        'quantity': {
            'description': 'Number of units',
            'example': '10',
            'required': True
        }
    }

def validate_dataframe(df, column_mapping=None):
    """Validate that the uploaded file has required columns"""
    required_columns_info = get_required_columns()
    required_columns = list(required_columns_info.keys())
    
    if column_mapping:
        # Check if mapped columns exist in original file
        missing_columns = []
        for required_col, mapped_col in column_mapping.items():
            if mapped_col not in df.columns:
                missing_columns.append(f"{required_col} -> {mapped_col}")
        
        if missing_columns:
            return False, f"Mapped columns not found in file: {', '.join(missing_columns)}"
    else:
        # Check if required columns exist directly
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}", df.columns.tolist()
    
    if len(df) == 0:
        return False, "File is empty"
    
    return True, "File validation successful"

def prepare_data_for_processing(df, column_mapping=None):
    """Prepare data with column mapping applied"""
    if column_mapping:
        # Create mapping dictionary from source columns to required columns
        rename_dict = {}
        for required_col, source_col in column_mapping.items():
            if source_col in df.columns:
                rename_dict[source_col] = required_col
        
        # Apply renaming
        df = df.rename(columns=rename_dict)
        
        # Select only required columns
        required_columns = list(get_required_columns().keys())
        available_required = [col for col in required_columns if col in df.columns]
        df = df[available_required]
        
        logger.info(f"Applied column mapping: {column_mapping}")
    
    return df

def extract_metrics_from_results(results):
    """Extract specific metrics from your script results"""
    if not results or 'summary_metrics' not in results:
        return None
    
    metrics = results['summary_metrics']
    logger.info(f"Available metrics keys: {list(metrics.keys())}")
    
    return {
        'portfolio_xirr': metrics.get('XIRR', 'N/A'),
        'benchmark_xirr': metrics.get('Benchmark XIRR', 'N/A'),
        'portfolio_nav': metrics.get('Current NAV', 'N/A'),
        'benchmark_nav': metrics.get('benchmark_nav', 'N/A'),
        'portfolio_drawdown': metrics.get('Maximum NAV Drawdown', 'N/A'),
        'benchmark_drawdown': metrics.get('Maximum BEN Drawdown', 'N/A'),
        'portfolio_sharpe': metrics.get('Sharpe Ratio', 'N/A'),
        'benchmark_sharpe': metrics.get('Benchmark Sharpe Ratio', 'N/A'),
        'portfolio_annual_return': metrics.get('Annualized Return', 'N/A'),
        'beta': metrics.get('Beta', 'N/A'),
        'alpha': metrics.get('Alpha', 'N/A'),
        'volatility': metrics.get('Volatility', 'N/A'),
        'benchmark_volatility': metrics.get('Benchmark Volatility', 'N/A')
    }

def format_metric_value(value, metric_type='number'):
    """Format metric values for display"""
    if value == 'N/A' or value is None:
        return 'N/A'
    
    try:
        if isinstance(value, str):
            # Remove percentage signs and convert
            clean_value = value.replace('%', '').replace('₹', '').replace(',', '')
            numeric_value = float(clean_value)
        else:
            numeric_value = float(value)
        
        if metric_type == 'percentage':
            return f"{numeric_value:.2f}%"
        elif metric_type == 'currency':
            return f"₹{numeric_value:,.2f}"
        elif metric_type == 'ratio':
            return f"{numeric_value:.4f}"
        else:
            return f"{numeric_value:.2f}"
    except:
        return str(value)

def create_metrics_dataframe(results_data):
    """Create a formatted dataframe for display"""
    if not results_data:
        return pd.DataFrame()
    
    df_data = []
    for row in results_data:
        df_data.append({
            'Year': row.get('year', 'N/A'),
            'Portfolio XIRR': format_metric_value(row.get('portfolio_xirr'), 'percentage'),
            'Benchmark XIRR': format_metric_value(row.get('benchmark_xirr'), 'percentage'),
            'Portfolio NAV': format_metric_value(row.get('portfolio_nav'), 'ratio'),
            'Benchmark NAV': format_metric_value(row.get('benchmark_nav'), 'ratio'),
            'Portfolio Drawdown': format_metric_value(row.get('portfolio_drawdown'), 'percentage'),
            'Benchmark Drawdown': format_metric_value(row.get('benchmark_drawdown'), 'percentage'),
            'Portfolio Sharpe': format_metric_value(row.get('portfolio_sharpe'), 'ratio'),
            'Benchmark Sharpe': format_metric_value(row.get('benchmark_sharpe'), 'ratio'),
            'Portfolio Annual Return': format_metric_value(row.get('portfolio_annual_return'), 'percentage'),
            'Beta': format_metric_value(row.get('beta'), 'ratio'),
            'Alpha': format_metric_value(row.get('alpha'), 'percentage'),
            'Portfolio Volatility': format_metric_value(row.get('volatility'), 'percentage'),
            'Benchmark Volatility': format_metric_value(row.get('benchmark_volatility'), 'percentage')
        })
    
    return pd.DataFrame(df_data)

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'results_data' not in st.session_state:
    st.session_state.results_data = None
if 'column_mapping_required' not in st.session_state:
    st.session_state.column_mapping_required = False
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None

def main():
    # Header
    st.title("📊 Portfolio Analytics Dashboard")
    st.markdown("Upload your transaction data and analyze portfolio performance across years")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 File Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose transaction file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel file with transaction data"
        )
        
        # File type selection
        file_type = st.selectbox(
            "Select file type",
            options=['csv', 'xlsx'],
            help="Select the type of file you're uploading"
        )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            if file_type == 'csv':
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"File uploaded successfully! Found {len(df)} rows.")
            
            # Store in session state
            st.session_state.uploaded_df = df
            
            # Validate the dataframe
            validation_result = validate_dataframe(df)
            
            if validation_result[0]:
                # File is valid, proceed with processing
                st.session_state.column_mapping_required = False
                
                # Show data preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head(10))
                
                # Process button
                if st.button("🚀 Analyze Portfolio", type="primary"):
                    process_portfolio_data(df)
            
            else:
                # Column mapping required
                st.session_state.column_mapping_required = True
                st.warning("Column mapping required. Please map your columns below.")
                
                # Show column mapping interface
                show_column_mapping_interface(df)
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Show results if processing is complete
    if st.session_state.processing_complete and st.session_state.results_data:
        show_results()

def show_column_mapping_interface(df):
    """Show interface for column mapping"""
    st.subheader("🔗 Column Mapping")
    st.info("Map your file columns to the required fields:")
    
    # Show file info
    with st.expander("📊 File Information"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total Rows:** {len(df)}")
            st.write(f"**Available Columns:** {', '.join(df.columns.tolist())}")
        with col2:
            st.write("**Required Columns:**")
            required_cols = get_required_columns()
            for col, info in required_cols.items():
                st.write(f"- {col}: {info['description']}")
    
    # Show data preview
    with st.expander("👀 Data Preview"):
        st.dataframe(df.head(3))
    
    # Column mapping form
    st.write("**Map your columns:**")
    
    required_columns = get_required_columns()
    column_mapping = {}
    
    for required_col, info in required_columns.items():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write(f"**{required_col}***")
            st.caption(info['description'])
            st.caption(f"Example: {info['example']}")
        
        with col2:
            # Try to find matching column automatically
            auto_match = None
            for file_col in df.columns:
                if required_col.lower() in file_col.lower():
                    auto_match = file_col
                    break
            
            selected_col = st.selectbox(
                f"Select column for {required_col}",
                options=[''] + df.columns.tolist(),
                index=df.columns.tolist().index(auto_match) + 1 if auto_match else 0,
                key=f"mapping_{required_col}"
            )
            
            if selected_col:
                column_mapping[required_col] = selected_col
                # Show sample value
                sample_value = df[selected_col].iloc[0] if len(df) > 0 else 'N/A'
                st.caption(f"Sample value: {sample_value}")
    
    # Validate mapping
    required_fields = list(required_columns.keys())
    missing_fields = [field for field in required_fields if field not in column_mapping]
    
    if missing_fields:
        st.warning(f"Please map the following required fields: {', '.join(missing_fields)}")
    else:
        st.success("All required fields mapped!")
        
        if st.button("✅ Confirm Mapping & Analyze", type="primary"):
            # Apply mapping and process
            mapped_df = prepare_data_for_processing(df, column_mapping)
            process_portfolio_data(mapped_df)

def process_portfolio_data(df):
    """Process the portfolio data using NRED1 script"""
    with st.spinner("Processing portfolio metrics... This may take a few minutes..."):
        try:
            # Save dataframe to temporary CSV file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
                df.to_csv(tmp_file.name, index=False)
                temp_file_path = tmp_file.name
            
            # Copy to expected filename for NRED1
            processing_file = "Transactions_sample.csv"
            shutil.copy(temp_file_path, processing_file)
            
            # Process overall metrics
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Processing overall metrics...")
            progress_bar.progress(20)
            
            overall_results = NRED1.main_optimized()
            
            results_data = []
            
            if overall_results:
                overall_metrics = extract_metrics_from_results(overall_results)
                if overall_metrics:
                    overall_metrics['year'] = 'Overall'
                    results_data.append(overall_metrics)
            
            # Process yearly metrics
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            min_year = df['transaction_date'].min().year
            max_year = df['transaction_date'].max().year
            
            total_years = max_year - min_year + 1
            
            for i, year in enumerate(range(min_year, max_year + 1)):
                status_text.text(f"Processing year {year}...")
                progress_bar.progress(20 + int(60 * (i + 1) / total_years))
                
                try:
                    start_date = f"{year}-01-01"
                    end_date = f"{year}-12-31"
                    
                    if year == min_year:
                        start_date = df['transaction_date'].min().strftime('%Y-%m-%d')
                    if year == max_year:
                        end_date = df['transaction_date'].max().strftime('%Y-%m-%d')
                    
                    # Filter data for the year
                    start_date_dt = pd.to_datetime(start_date)
                    end_date_dt = pd.to_datetime(end_date)
                    
                    year_df = df[
                        (df['transaction_date'] >= start_date_dt) & 
                        (df['transaction_date'] <= end_date_dt)
                    ].copy()
                    
                    if len(year_df) == 0:
                        continue
                    
                    # Use date filter function if available
                    if hasattr(NRED1, 'run_with_date_filter'):
                        year_results = NRED1.run_with_date_filter(start_date, end_date)
                    else:
                        # Fallback to main function
                        year_results = NRED1.main_optimized(start_date, end_date)
                    
                    if year_results:
                        year_metrics = extract_metrics_from_results(year_results)
                        if year_metrics:
                            year_metrics['year'] = str(year)
                            results_data.append(year_metrics)
                
                except Exception as e:
                    st.warning(f"Error processing year {year}: {str(e)}")
                    continue
            
            progress_bar.progress(100)
            status_text.text("Processing complete!")
            
            # Clean up temporary files
            try:
                os.unlink(temp_file_path)
                if os.path.exists(processing_file):
                    os.unlink(processing_file)
            except:
                pass
            
            if results_data:
                st.session_state.results_data = results_data
                st.session_state.processing_complete = True
                st.success(f"Analysis completed successfully! Generated metrics for {len(results_data)} periods.")
                st.rerun()
            else:
                st.error("No results could be generated from the data")
        
        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
            st.exception(e)

def show_results():
    """Display the analysis results"""
    st.header("📈 Portfolio Performance Analysis")
    st.subheader("Comprehensive metrics comparison across years")
    
    # Summary info
    results_data = st.session_state.results_data
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Periods Analyzed", len(results_data))
    with col2:
        st.metric("Data Processing", "Complete")
    with col3:
        st.metric("Portfolio vs Benchmark", "Available")
    
    # Create and display metrics table
    metrics_df = create_metrics_dataframe(results_data)
    
    if not metrics_df.empty:
        st.subheader("📊 Performance Metrics Table")
        
        # Style the dataframe
        def highlight_overall_row(row):
            if row['Year'] == 'Overall':
                return ['background-color: #e8f5e8; font-weight: bold'] * len(row)
            return [''] * len(row)
        
        def color_values(val):
            """Color positive/negative values"""
            if isinstance(val, str) and '%' in val:
                try:
                    num_val = float(val.replace('%', ''))
                    if num_val > 0:
                        return 'color: #28a745; font-weight: bold'
                    elif num_val < 0:
                        return 'color: #dc3545; font-weight: bold'
                except:
                    pass
            return ''
        
        # Apply styling
        styled_df = metrics_df.style.apply(highlight_overall_row, axis=1)
        
        # Apply color formatting to specific columns
        percentage_cols = ['Portfolio XIRR', 'Benchmark XIRR', 'Portfolio Drawdown', 
                          'Benchmark Drawdown', 'Portfolio Annual Return', 'Alpha',
                          'Portfolio Volatility', 'Benchmark Volatility']
        
        for col in percentage_cols:
            if col in metrics_df.columns:
                styled_df = styled_df.applymap(color_values, subset=[col])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Download results
        csv = metrics_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Additional insights
        if len(results_data) > 1:
            st.subheader("🔍 Key Insights")
            
            overall_row = next((row for row in results_data if row.get('year') == 'Overall'), None)
            if overall_row:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    xirr_val = overall_row.get('portfolio_xirr', 'N/A')
                    st.metric("Overall Portfolio XIRR", format_metric_value(xirr_val, 'percentage'))
                
                with col2:
                    benchmark_xirr = overall_row.get('benchmark_xirr', 'N/A')
                    st.metric("Overall Benchmark XIRR", format_metric_value(benchmark_xirr, 'percentage'))
                
                with col3:
                    sharpe_val = overall_row.get('portfolio_sharpe', 'N/A')
                    st.metric("Portfolio Sharpe Ratio", format_metric_value(sharpe_val, 'ratio'))
                
                with col4:
                    beta_val = overall_row.get('beta', 'N/A')
                    st.metric("Portfolio Beta", format_metric_value(beta_val, 'ratio'))
    
    # Reset button
    if st.button("🔄 Analyze Another Portfolio", type="secondary"):
        # Clear session state
        for key in ['processing_complete', 'results_data', 'column_mapping_required', 'uploaded_df']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()