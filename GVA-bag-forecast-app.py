import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Helper functions for calculations
def BIAS(T, P):
    """Calculate bias between true and predicted values."""
    return 100 * (P - T) / T

def APE(T, P):
    """Calculate absolute percentage error."""
    return abs(T-P)/T

def WMAPE(Ts, Ps):
    """Calculate weighted mean absolute percentage error."""
    WAPE = [T*APE(T,P) for T,P in zip(Ts,Ps)]
    return sum(WAPE)/sum(Ts)

def CV(Ts, Ps):
    """Calculate coefficient of variation."""
    APEs = [1- APE(T,P) for T,P in zip(Ts,Ps)]
    std = np.std(APEs)
    mean_ = np.mean(APEs)
    return 100*std/mean_

def SUP_error(realized, projected):
    """Calculate error between realized and projected SUPs."""
    error = 0
    for r, p in zip(realized, projected):
        error += abs(r-p)
    return error

class PerformanceAnalyzer:
    def __init__(self):
        self.SHOWUP_COLUMNS = [str(x) for x in range(0, 181, 5)]  # 0 to 180 minutes in 5-minute intervals
        
    def load_data(self, sup_file, bagfactor_file, realized_data_file):
        """Load and prepare all necessary data files."""
        # Load projected SUPs
        self.projected_sups = pd.read_csv(sup_file)
        self.projected_sups['Local Schedule Time'] = pd.to_datetime(self.projected_sups['Local Schedule Time'])
        
        # Load forecasted Bag Factor
        self.projected_bagfactor = pd.read_csv(bagfactor_file)
        self.projected_bagfactor['Local Schedule Time'] = pd.to_datetime(self.projected_bagfactor['Local Schedule Time'])
        
        # Load realized data
        self.realized_data = pd.read_csv(realized_data_file)
        self.realized_data['Local Schedule Time'] = pd.to_datetime(self.realized_data['Local Schedule Time'])
        
        # Merge datasets
        self.merged_data = self.merge_datasets()
        return self.merged_data
    
    def merge_datasets(self):
        """Merge all datasets and prepare for analysis."""
        # Merge logic here similar to your original notebook
        # This is a simplified version - you'll need to adapt based on your exact data structure
        merged = self.projected_sups.merge(
            self.realized_data,
            on=['Local Schedule Time', 'Airline IATA Code', 'Flight Number'],
            suffixes=('_projected', '_realized')
        )
        merged = merged.merge(
            self.projected_bagfactor,
            on=['Local Schedule Time', 'Airline IATA Code', 'Flight Number']
        )
        return merged

    def plot_daily_performance(self, date, data_type='combined'):
        """Create interactive plot for daily performance."""
        selected = self.merged_data[self.merged_data['Date of Flight'] == date].copy()
        
        # Create time range for the day
        time_range = pd.date_range(start=f'{date} 00:00', end=f'{date} 23:55', freq='5min')
        realized = pd.Series(0, index=time_range)
        projected = pd.Series(0, index=time_range)
        
        # Calculate arrivals based on selected data type
        for _, flight in selected.iterrows():
            self._add_flight_to_series(flight, realized, projected, data_type)
            
        # Create interactive plot using plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=realized.index, y=realized, name='Realized',
                                line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=projected.index, y=projected, name='Projected',
                                line=dict(color='red')))
        
        fig.update_layout(
            title=f'Bag Check-in Pattern for {date}',
            xaxis_title='Time of Day',
            yaxis_title='Number of Bags',
            height=600
        )
        
        return fig
    
    def _add_flight_to_series(self, flight, realized, projected, data_type):
        """Helper method to add flight data to time series."""
        for col in self.SHOWUP_COLUMNS:
            minutes = int(col)
            arrival_time = flight['Local Schedule Time'] - pd.Timedelta(minutes=minutes)
            
            if arrival_time in realized.index:
                if data_type == 'SUPs':
                    realized[arrival_time] += flight[f'{col}_realized'] * flight['Nb Bags Local']
                    projected[arrival_time] += flight[f'{col}_projected'] * flight['Nb Bags Local']
                elif data_type == 'BagFactor':
                    realized[arrival_time] += flight[f'{col}_realized'] * flight['Nb Bags Local']
                    projected[arrival_time] += flight[f'{col}_realized'] * flight['Pax'] * flight['Bag Factor Forecast']
                else:  # combined
                    realized[arrival_time] += flight[f'{col}_realized'] * flight['Nb Bags Local']
                    projected[arrival_time] += flight[f'{col}_projected'] * flight['Pax'] * flight['Bag Factor Forecast']

def main():
    st.title("Baggage Forecast Performance Analysis")
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    
    # File uploader widgets
    sup_file = st.sidebar.file_uploader("Upload SUP Forecast CSV", type='csv')
    bagfactor_file = st.sidebar.file_uploader("Upload Bag Factor Forecast CSV", type='csv')
    realized_file = st.sidebar.file_uploader("Upload Realized Data CSV", type='csv')
    
    if all([sup_file, bagfactor_file, realized_file]):
        # Initialize analyzer
        analyzer = PerformanceAnalyzer()
        
        # Load data
        data = analyzer.load_data(sup_file, bagfactor_file, realized_file)
        
        # Date selection
        available_dates = sorted(data['Date of Flight'].unique())
        selected_date = st.sidebar.selectbox("Select Date", available_dates)
        
        # Analysis type selection
        analysis_type = st.sidebar.selectbox(
            "Select Analysis Type",
            ['SUPs Only', 'Bag Factor Only', 'Combined Analysis']
        )
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Daily Pattern", "Performance Metrics", "Airline Analysis"])
        
        with tab1:
            st.plotly_chart(analyzer.plot_daily_performance(
                selected_date,
                analysis_type.lower().replace(' only', '')
            ))
            
        with tab2:
            # Add performance metrics visualization
            st.write("Performance Metrics Coming Soon")
            
        with tab3:
            # Add airline-specific analysis
            st.write("Airline Analysis Coming Soon")

if __name__ == "__main__":
    main()
