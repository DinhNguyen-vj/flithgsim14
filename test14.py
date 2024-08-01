import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from collections import deque

# Custom CSS to adjust the width of the Streamlit app
st.markdown(
    """
    <style>
    .css-1d391kg.e1fqkh3o1 {
        max-width: 1600px;
        margin: 0 auto;
    }

    .flight-block h3 {
        font-size: 2rem;
        margin-bottom: 0px;
    }
    .flight-block .stSelectbox, .flight-block .stMultiselect {
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Function to convert portion of day to hours and minutes
def convert_to_time(portion):
    day = int(portion)
    time_fraction = portion - day
    hours = int(time_fraction * 24)
    minutes = ((int((time_fraction * 24 - hours) * 60)+4)//5)*5

    
    if minutes == 60:
        hours += 1
        minutes = 0
        if hours == 24:
            day += 1
            hours = 0
    
    return day, hours, minutes


# Function to add flight times to the figure
def add_flight_times(fig, flights, idx):
    flights = flights.sort_values(by='STD').reset_index(drop=True)
    y_base = idx  # Base y-coordinate for the aircraft
    y_offset_increment = 0.1  # Increment for y-offset in case of overlap

    for i, row in flights.iterrows():
        std_day, std_hour, std_minute = convert_to_time(row['STD'])
        sta_day, sta_hour, sta_minute = convert_to_time(row['STA'])

        start_time = std_day * 24 + std_hour + std_minute / 60
        end_time = sta_day * 24 + sta_hour + sta_minute / 60

        # Determine the y-offset for the current flight
        y_offset = y_base

        # Determine the color
        color = 'blue'
        if i > 0:
            prev_row = flights.iloc[i - 1]
            prev_sta_day, prev_sta_hour, prev_sta_minute = convert_to_time(prev_row['STA'])
            prev_end_time = prev_sta_day * 24 + prev_sta_hour + prev_sta_minute / 60

            if start_time < prev_end_time and row['AC'] == prev_row['AC']:
                y_offset += y_offset_increment  # Move the flight up slightly to avoid overlap
                color = 'red'
            elif row['DEP'] != prev_row['ARR']:
                color = 'orange'

        hovertext = f"Dep: {row['DEP']}<br>Flight No: {row['FLIGHT NO']}<br>Arr: {row['ARR']}"

        fig.add_trace(go.Scatter(
            x=[start_time, end_time],
            y=[y_offset, y_offset],
            mode='lines',
            line=dict(color=color, width=10),
            name=row['AC'],
            hoverinfo='text',
            hovertext=hovertext
        ))

# Function to plot flights
def plot_flights(df, title_suffix):
    figures = {}
    for type_name, type_group in df.groupby('Type'):
        # Create a Plotly figure for this type
        fig = go.Figure()

        # Loop through each aircraft
        for idx, (aircraft, flights) in enumerate(type_group.groupby('AC')):
            add_flight_times(fig, flights, idx)

        # Set labels and title
        fig.update_layout(
            title=f'Aircraft FLIGHT SCHEDULE for Type: {type_name} ({title_suffix})',
            xaxis=dict(
                title='Time (hours)',
                tickvals=np.arange(0, 8 * 24 + 1, 24),  # Major gridlines every 24 hours, including the last Sunday
                ticktext=['0_Sunday', '1_Monday', '2_Tuesday', '3_Wednesday', '4_Thursday', '5_Friday', '6_Saturday', '7_Sunday', '8_Monday'],
                tickangle=90,
                showgrid=True,
                gridwidth=0.5,
                gridcolor='lightgrey',
                minor=dict(
                    dtick=6,  # Minor gridlines every 6 hours
                    gridwidth=0.2,
                    gridcolor='lightgrey'  # Color for minor gridlines
                )
            ),
            yaxis=dict(
                title='Aircraft',
                tickmode='array',
                tickvals=np.arange(len(type_group['AC'].unique())),
                ticktext=type_group['AC'].unique(),
                showgrid=True,
                gridwidth=1,
                gridcolor='black'
            ),
            xaxis_range=[0, 8 * 24],  # Ensure the x-axis covers the full 8 days
            showlegend=False,
            height=1800,          #Set height of the figure
            width=1600            #Set the width of the figure
        )

        # Add major gridlines
        fig.update_xaxes(
            tickvals=np.arange(0, 8 * 24 + 1, 24),  # Major gridlines every 24 hours, including the last Sunday
            showgrid=True,
            gridwidth=2,  # Major gridline width
            gridcolor='black',
            layer='above traces'  # Make sure major gridlines are on top
        )

        # Add zoom option
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=12, label="12h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="linear"
            )
        )

        figures[type_name] = fig

    return figures

# Function to format duration in hh:mm
def format_duration(duration_minutes):
    hours = int(duration_minutes // 60)
    minutes = int(duration_minutes % 60)
    return f"{hours:02d}:{minutes:02d}"


# Function to plot ground times
def plot_ground_times(df, title_suffix, min_ground_time, max_ground_time, next_departure_stations, include_overlap):
    figures = {}
    for type_name, type_group in df.groupby('Type'):
        # Create a Plotly figure for this type
        fig = go.Figure()

        # Loop through each aircraft
        for idx, (aircraft, flights) in enumerate(type_group.groupby('AC')):
            # Sort flights by STD
            flights = flights.sort_values(by='STD').reset_index(drop=True)
            ground_times = []

            # Check for station mismatches
            station_mismatch = any(flights.iloc[i]['DEP'] != flights.iloc[i - 1]['ARR'] for i in range(1, len(flights)))
            
            if station_mismatch:
                continue  # Ignore the entire aircraft if there is a station mismatch

            # Check for overlaps and calculate ground times
            overlap = False
            for i in range(1, len(flights)):
                flight_overlap = flights.iloc[i]['STD'] < flights.iloc[i - 1]['STA']

                if flight_overlap:
                    overlap = True
                    if not include_overlap:
                        break  # Skip the entire aircraft if overlaps are not to be included

                if not flight_overlap:
                    ground_time = ((flights.iloc[i]['STD'] - flights.iloc[i - 1]['STA']) * 24 * 60+1)//5*5  # Convert to minutes
                    if min_ground_time <= ground_time <= max_ground_time and flights.iloc[i]['DEP'] in next_departure_stations:
                        ground_times.append((flights.iloc[i - 1]['STA'] * 24, flights.iloc[i]['STD'] * 24, ground_time, flights.iloc[i]['DEP']))

            if overlap and not include_overlap:
                continue  # Skip the entire aircraft if overlaps are not to be included

            if ground_times:
                for gt in ground_times:
                    hovertext = f"Ground Time: {format_duration(gt[2])}<br>Next Dep: {gt[3]}"
                    fig.add_trace(go.Scatter(
                        x=[gt[0], gt[1]],
                        y=[idx, idx],
                        mode='lines',
                        line=dict(color='green', width=10),
                        name=aircraft,
                        hoverinfo='text',
                        hovertext=hovertext
                    ))

        # Set labels and title
        fig.update_layout(
            title=f'Aircraft Ground Times for Type: {type_name} ({title_suffix})',
            xaxis=dict(
                title='Time (hours)',
                tickvals=np.arange(0, 8 * 24 + 1, 24),  # Major gridlines every 24 hours, including the last Sunday
                ticktext=['0_Sunday', '1_Monday', '2_Tuesday', '3_Wednesday', '4_Thursday', '5_Friday', '6_Saturday', '7_Sunday', '8_Monday'],
                tickangle=90,
                showgrid=True,
                gridwidth=0.5,
                gridcolor='lightgrey',
                minor=dict(
                    dtick=6,  # Minor gridlines every 6 hours
                    gridwidth=0.2,
                    gridcolor='lightgrey'  # Color for minor gridlines
                )
            ),
            yaxis=dict(
                title='Aircraft',
                tickmode='array',
                tickvals=np.arange(len(type_group['AC'].unique())),
                ticktext=type_group['AC'].unique(),
                showgrid=True,
                gridwidth=1,
                gridcolor='black'
            ),
            xaxis_range=[0, 8 * 24],  # Ensure the x-axis covers the full 8 days
            showlegend=False,
            height=1800,  # Set height of the figure
            width=1600   # Set the width of the figure
        )

        # Add major gridlines
        fig.update_xaxes(
            tickvals=np.arange(0, 8 * 24 + 1, 24),  # Major gridlines every 24 hours, including the last Sunday
            showgrid=True,
            gridwidth=2,  # Major gridline width
            gridcolor='black',
            layer='above traces'  # Make sure major gridlines are on top
        )

        # Add zoom option
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=12, label="12h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="linear"
            )
        )

        figures[type_name] = fig

    return figures



# Streamlit app
st.title("Aircraft Flight Schedule Visualization")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

# Initialize session state for undo functionality
if 'history' not in st.session_state:
    st.session_state.history = deque()
if 'update_count' not in st.session_state:
    st.session_state.update_count = 0

if uploaded_file is not None:
    # Read the Excel file
    if 'original_df' not in st.session_state:
        st.session_state.original_df = pd.read_excel(uploaded_file,sheet_name='streamlit')
        st.session_state.modified_df = st.session_state.original_df.copy()

    original_df = st.session_state.original_df
    modified_df = st.session_state.modified_df

    # Display the dataframe for selection
    st.write("Select flights to modify:")

    days_of_the_week = ['1_Monday', '2_Tuesday', '3_Wednesday', '4_Thursday', '5_Friday', '6_Saturday', '7_Sunday']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="flight-block"><h3>First Flight Block</h3>', unsafe_allow_html=True)
        with st.container():
            cols = st.columns(3)
            selected_day_label1 = cols[0].selectbox("Choose DAY", options=days_of_the_week, key='day1')
            selected_day1 = int(selected_day_label1.split('_')[0])
            type_options1 = original_df['Type'].unique()
            selected_type1 = cols[1].selectbox("Choose Type", options=type_options1, key='type1')
            ac_options1 = original_df[original_df['Type'] == selected_type1]['AC'].unique()
            selected_ac1 = cols[2].selectbox("Choose AC ", options=ac_options1, key='ac1')
        with st.container():
            cols = st.columns(3)
            flights_to_modify1 = modified_df[(modified_df['DAY'] == selected_day1) & (modified_df['AC'] == selected_ac1)]

            if not flights_to_modify1.empty:
                selected_flights1 = cols[0].multiselect(
                    "FLIGHT NO(s) to modify",
                    flights_to_modify1['FLIGHT NO'].unique(),
                    key='flights1'
                )
                new_type1 = cols[1].selectbox("New Type", options=type_options1, key='new_type1')
                new_ac1 = cols[2].selectbox("New AC ", options=original_df[original_df['Type'] == new_type1]['AC'].unique(), key='new_ac1')
            else:
                st.error("No flights found for the selected DAY and AC in Block 1")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="flight-block"><h3>Second Flight Block</h3>', unsafe_allow_html=True)
        with st.container():
            cols = st.columns(3)
            selected_day_label2 = cols[0].selectbox("Choose DAY", options=days_of_the_week, key='day2')
            selected_day2 = int(selected_day_label2.split('_')[0])
            type_options2 = original_df['Type'].unique()
            selected_type2 = cols[1].selectbox("Choose Type", options=type_options2, key='type2')
            ac_options2 = original_df[original_df['Type'] == selected_type2]['AC'].unique()
            selected_ac2 = cols[2].selectbox("Choose AC", options=ac_options2, key='ac2')
        with st.container():
            cols = st.columns(3)
            flights_to_modify2 = modified_df[(modified_df['DAY'] == selected_day2) & (modified_df['AC'] == selected_ac2)]

            if not flights_to_modify2.empty:
                selected_flights2 = cols[0].multiselect(
                    "FLIGHT NO(s) to modify",
                    flights_to_modify2['FLIGHT NO'].unique(),
                    key='flights2'
                )
                new_type2 = cols[1].selectbox("New Type", options=type_options2, key='new_type2')
                new_ac2 = cols[2].selectbox("New AC", options=original_df[original_df['Type'] == new_type2]['AC'].unique(), key='new_ac2')
            else:
                st.error("No flights found for the selected DAY and AC in Block 2")
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4, col5 = st.columns(3)

    with col3:
        if st.button("Update Flights", key="update_flights", help="Click to update the selected flights"):
            st.session_state.history.append(modified_df.copy())
            if selected_flights1:
                modified_df.loc[
                    (modified_df['DAY'] == selected_day1) & (modified_df['FLIGHT NO'].isin(selected_flights1)),
                    ['Type', 'AC']
                ] = [new_type1, new_ac1]
                modified_df.loc[
                    (modified_df['DAY'] == selected_day1) & (modified_df['FLIGHT NO'].isin(selected_flights1)),
                    'highlight'
                ] = True

            if selected_flights2:
                modified_df.loc[
                    (modified_df['DAY'] == selected_day2) & (modified_df['FLIGHT NO'].isin(selected_flights2)),
                    ['Type', 'AC']
                ] = [new_type2, new_ac2]
                modified_df.loc[
                    (modified_df['DAY'] == selected_day2) & (modified_df['FLIGHT NO'].isin(selected_flights2)),
                    'highlight'
                ] = True

            # Sort the modified dataframe by Type, AC, and DAY
            modified_df = modified_df.sort_values(by=['Type', 'AC', 'DAY']).reset_index(drop=True)
            st.session_state.modified_df = modified_df
            st.session_state.update_count += 1
            st.success("Flights updated!")

    with col4:
        if st.button("Undo Last Modification", key="undo_last") and st.session_state.history:
            st.session_state.modified_df = st.session_state.history.pop()
            modified_df = st.session_state.modified_df
            st.session_state.update_count -= 1
            st.success("Last modification undone!")

    with col5:
        if st.button("Reset", key="reset"):
            while st.session_state.history:
                st.session_state.modified_df = st.session_state.history.pop()
            modified_df = st.session_state.modified_df
            st.session_state.update_count = 0
            st.success("Data and chart reset to original state!")

    st.write(f"Number of Updates: {st.session_state.update_count}")

    # Highlight modified flights in the dataframe display
    modified_display_df = modified_df.copy()
    modified_display_df.style.applymap(lambda x: 'background-color: yellow' if x else '', subset=['highlight'])

    st.write(modified_display_df)  # Display the modified dataframe

    if st.button("Done", key="done"):
        st.session_state['show_ground_time'] = True
        st.success("Modifications finalized!")

    # Plot original flights
    original_figures = plot_flights(original_df, 'Original Data')

    # Plot modified flights
    modified_figures = plot_flights(modified_df, 'Modified Data')

    # Display charts side by side for each type
    for type_name in original_figures.keys():
        st.subheader(f"Type: {type_name}")

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(original_figures[type_name], use_container_width=True)

        with col2:
            st.plotly_chart(modified_figures[type_name], use_container_width=True)

if 'show_ground_time' in st.session_state:
    st.title("Ground Time Visualization")

    col1, col2,col3, col4 = st.columns(4)
    with col1:
        min_ground_time = st.number_input("Minimum ground time (minutes)", min_value=0, value=5)
    with col2:
        max_ground_time = st.number_input("Maximum ground time (minutes)", min_value=0, value=2400)
    with col3:
        next_departure_stations = st.multiselect("Select Next Departure Stations", options=modified_df['DEP'].unique())
    with col4:
        include_overlap = st.checkbox("Include overlapping flights")

    if st.button("Filter Ground Times", key="filter_ground_times"):
        # Plot ground times
        ground_time_figures = plot_ground_times(modified_df, 'Ground Time Data', min_ground_time, max_ground_time, next_departure_stations, include_overlap)

        # Display ground time charts for each type
        for type_name in ground_time_figures.keys():
            st.subheader(f"Type: {type_name}")
            st.plotly_chart(ground_time_figures[type_name], use_container_width=True)

