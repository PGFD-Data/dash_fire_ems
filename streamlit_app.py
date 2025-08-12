import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
from datetime import datetime
import io

st.set_page_config(page_title=" 🔥 Incident Data Analytics County Fire & EMS Department 🚑 🚒", layout="wide")
st.info('This is a swift data analytics app for the fire & EMS department of the PG county')

# ---------- Helper functions ----------
def try_parse_datetime_col(df):
    # Try to detect a datetime column: priority by name, then dtype
    datetime_cols = []
    for col in df.columns:
        low = col.lower()
        if 'date' in low or 'time' in low or 'datetime' in low or 'alarm' in low or 'dispatch' in low:
            datetime_cols.append(col)
    if datetime_cols:
        # prefer exact 'incident date' if present
        for prefer in ['incident date', 'call date', 'dispatch date', 'alarm date', 'date/time', 'datetime', 'incident_datetime']:
            for col in df.columns:
                if prefer in col.lower():
                    return col
        return datetime_cols[0]

    # fallback: pick first datetime dtype column
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            return col
    return None

def to_minsec_str(seconds):
    # seconds -> "MM:SS"
    if pd.isna(seconds):
        return ""
    seconds = int(round(seconds))
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}:{s:02d}"

def parse_response_time(value):
    """
    Accepts strings like "mm:ss", "m:ss", "HH:MM:SS", numeric seconds, or pandas Timedelta.
    Returns seconds (float) or np.nan.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    # Timedelta
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    # String parsing
    try:
        s = str(value).strip()
        if ':' in s:
            parts = s.split(':')
            parts = [p.strip() for p in parts if p.strip() != '']
            # support MM:SS OR HH:MM:SS
            if len(parts) == 2:
                m, sec = parts
                return float(int(m) * 60 + float(sec))
            elif len(parts) == 3:
                h, m, sec = parts
                return float(int(h) * 3600 + int(m) * 60 + float(sec))
            else:
                return float(s)
        else:
            # no colon, maybe seconds numeric
            return float(s)
    except Exception:
        return np.nan

def format_percent_change(prev, curr):
    # compute percent change safely; returns formatted string like "+12.3%" or "-4.5%" or "â€”"
    if prev is None or curr is None:
        return "â€”"
    try:
        prev = float(prev)
        curr = float(curr)
        if prev == 0:
            if curr == 0:
                return "0.0%"
            else:
                return "âˆž"
        change = (curr - prev) / abs(prev) * 100.0
        sign = "+" if change >= 0 else ""
        return f"{sign}{change:.1f}%"
    except Exception:
        return "â€”"

# ---------- UI - Title/Header ----------
st.title("Incident Data analytics for PG County Fire & EMS Department ðŸš’")
st.markdown("Highly interactive dashboard â€” upload CSV / XLSX / TXT or use default dataset.")

# ---------- File uploader and load data ----------
uploaded_file = st.sidebar.file_uploader("Upload data (CSV, XLSX, TXT)", type=['csv','xlsx','txt'])
use_default = False
if uploaded_file is None:
    # Try to load default file in codespace
    default_path = "/mnt/data/FY2025.xlsx"
    try:
        df = pd.read_excel(default_path)
        use_default = True
        st.sidebar.caption(f"Using default data: `{default_path}`")
    except Exception:
        df = pd.DataFrame()
        st.sidebar.info("No default file found. Please upload a CSV/XLSX/TXT.")
else:
    # read uploaded file
    try:
        if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt'):
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, low_memory=False)
        else:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        df = pd.DataFrame()

if df.empty:
    st.warning("No data loaded yet. Upload a file or make sure `/mnt/data/FY2025.xlsx` exists.")
    st.stop()

# Standardize column names trim whitespace
df.columns = [c.strip() for c in df.columns]

# ---------- Detect key columns and prepare columns ----------
# Detect datetime column
date_col = try_parse_datetime_col(df)

if date_col:
    # Try parsing to datetime
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    except Exception:
        pass

# Ensure Incident # column detection
incident_col = None
for c in df.columns:
    if c.lower().replace(' ', '') in ['incident#','incidentnum','incidentnumber','incident_no','incident']:
        incident_col = c
        break
if incident_col is None:
    # fallback to first column name with '#' in it or "incident"
    for c in df.columns:
        if '#' in c or 'incident' in c.lower():
            incident_col = c
            break

if incident_col is None:
    st.error("Could not find an 'Incident #' column in the dataset. Rename your incident number column to include 'Incident #' or 'Incident'.")
    st.stop()

# Ensure 'Hour of Day' column exists or derive from datetime
if 'Hour of Day' not in df.columns:
    if date_col:
        df['Hour of Day'] = df[date_col].dt.hour
    else:
        # Try to find a column with 'hour' in name
        for c in df.columns:
            if 'hour' in c.lower():
                df['Hour of Day'] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
                break
        if 'Hour of Day' not in df.columns:
            df['Hour of Day'] = 0

# Ensure 'Day of Week' column
if 'Day of Week' not in df.columns:
    if date_col:
        df['Day of Week'] = df[date_col].dt.day_name()
    else:
        for c in df.columns:
            if 'day' in c.lower() and 'week' in c.lower():
                df['Day of Week'] = df[c]
                break
        if 'Day of Week' not in df.columns:
            # If there's a weekday integer, map it
            mapped = None
            for c in df.columns:
                if 'weekday' in c.lower() or ('day' in c.lower() and df[c].dtype in [np.int64, np.int32, np.float64]):
                    try:
                        df['Day of Week'] = df[c].map(lambda x: ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][int(x)]) 
                        mapped = True
                        break
                    except Exception:
                        pass
            if 'Day of Week' not in df.columns:
                df['Day of Week'] = 'Unknown'

# Capitalize Disposition first letters if present
if 'Disposition' in df.columns:
    df['Disposition'] = df['Disposition'].astype(str).str.title()

# Ensure Hour of Day integer
df['Hour of Day'] = pd.to_numeric(df['Hour of Day'], errors='coerce').fillna(0).astype(int)

# Fill placeholders for other filter columns if not present
for col in ['City', 'Zip Code', 'First Due', 'Incident Call Type Final', 'Call Type Category', 'Unit']:
    if col not in df.columns:
        df[col] = np.nan

# Create a cleaned copy for filtering (do not modify original column names inadvertently)
df_clean = df.copy()

# ---------- Sidebar : Data filters ----------
st.sidebar.header("Data Input & Filters")

# Date range widget (calendar limited to 2010-2025)
min_date = pd.Timestamp("2010-01-01")
max_date = pd.Timestamp("2025-12-31")
if date_col and df_clean[date_col].notna().any():
    earliest = max(df_clean[date_col].min(), min_date)
    latest = min(df_clean[date_col].max(), max_date)
else:
    earliest = min_date
    latest = max_date

start_date = st.sidebar.date_input("Start date", value=earliest.date(), min_value=min_date.date(), max_value=max_date.date())
end_date = st.sidebar.date_input("End date", value=latest.date(), min_value=min_date.date(), max_value=max_date.date())

# ensure start <= end
if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# Multi-select filters
city_sel = st.sidebar.multiselect("City", options=sorted(df_clean['City'].dropna().unique()[:500]), default=None)
zip_sel = st.sidebar.multiselect("Zip Code", options=sorted(df_clean['Zip Code'].dropna().astype(str).unique()[:500]), default=None)
firstdue_sel = st.sidebar.multiselect("First Due", options=sorted(df_clean['First Due'].dropna().unique()[:500]), default=None)
calltype_final_sel = st.sidebar.multiselect("Incident Call Type Final", options=sorted(df_clean['Incident Call Type Final'].dropna().unique()[:500]), default=None)
callcat_sel = st.sidebar.multiselect("Call Type Category", options=sorted(df_clean['Call Type Category'].dropna().unique()[:500]), default=None)
unit_sel = st.sidebar.multiselect("Unit", options=sorted(df_clean['Unit'].dropna().unique()[:500]), default=None)
disp_sel = st.sidebar.multiselect("Disposition", options=sorted(df_clean['Disposition'].dropna().unique()[:500]), default=None)
hour_sel = st.sidebar.multiselect("Hour of Day", options=sorted(df_clean['Hour of Day'].dropna().unique()), default=None)
day_sel = st.sidebar.multiselect("Day of Week", options=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'] + list(df_clean['Day of Week'].dropna().unique()), default=None)

# Option to choose period for comparison
st.sidebar.markdown("---")
period_comp = st.sidebar.selectbox("Comparison period", options=['Day', 'Week', 'Month', 'Year'], index=2)
period_n = st.sidebar.number_input(f"Number of {period_comp}s to compare (N)", min_value=1, max_value=365, value=1)

# Allow user to choose time window for computing current and previous periods
st.sidebar.markdown("**Comparison reference**")
end_ref = st.sidebar.date_input("Reference end date (for current period)", value=end_date, key="ref_end")
start_ref = st.sidebar.date_input("Reference start date (for current period)", value=start_date, key="ref_start")
# Note: we will compute previous period as immediately preceding the current period of same length

# ---------- Apply filters ----------
filtered = df_clean.copy()

# Apply date window if we have a date column
if date_col:
    # Convert to timestamps for comparison
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    filtered = filtered[(filtered[date_col] >= start_dt) & (filtered[date_col] <= end_dt)]

# Apply side filters if selected
if city_sel:
    filtered = filtered[filtered['City'].isin(city_sel)]
if zip_sel:
    filtered = filtered[filtered['Zip Code'].astype(str).isin(zip_sel)]
if firstdue_sel:
    filtered = filtered[filtered['First Due'].isin(firstdue_sel)]
if calltype_final_sel:
    filtered = filtered[filtered['Incident Call Type Final'].isin(calltype_final_sel)]
if callcat_sel:
    filtered = filtered[filtered['Call Type Category'].isin(callcat_sel)]
if unit_sel:
    filtered = filtered[filtered['Unit'].isin(unit_sel)]
if disp_sel:
    filtered = filtered[filtered['Disposition'].isin(disp_sel)]
if hour_sel:
    filtered = filtered[filtered['Hour of Day'].isin([int(h) for h in hour_sel])]
if day_sel:
    filtered = filtered[filtered['Day of Week'].isin(day_sel)]

# If incident id has NaNs, drop for unique counts but keep for responses
filtered = filtered.copy()

# ---------- Metrics: Incident Calls (unique) and Unit Responses (repeating) ----------
# Unique incidents count (dropna)
unique_incidents = filtered.dropna(subset=[incident_col]).drop_duplicates(subset=[incident_col])
incident_count_current = unique_incidents[incident_col].nunique()

# Unit responses count (allow repeats)
unit_responses_current = filtered[incident_col].count()

# Now compute previous period counts for comparison
def shift_period(start, end, period, n=1):
    # shift back by n periods
    if period == 'Day':
        delta = pd.Timedelta(days=n * (end - start).days + n)  # best-effort
        return (start - pd.Timedelta(days=n*(end-start).days + n), end - pd.Timedelta(days=n*(end-start).days + n))
    elif period == 'Week':
        return (start - pd.Timedelta(weeks=n), end - pd.Timedelta(weeks=n))
    elif period == 'Month':
        # approximate by subtracting 30*n days
        return (start - pd.Timedelta(days=30*n), end - pd.Timedelta(days=30*n))
    elif period == 'Year':
        return (start - pd.Timedelta(days=365*n), end - pd.Timedelta(days=365*n))
    else:
        return (start, end)

# Compute the "current" reference window from start_ref and end_ref; then compute previous window
cur_start_ref = pd.to_datetime(start_ref)
cur_end_ref = pd.to_datetime(end_ref) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
prev_start_ref, prev_end_ref = shift_period(cur_start_ref, cur_end_ref, period_comp, period_n)

# Subset dataframes for the reference windows (for percent change calculations)
if date_col:
    df_prev = df_clean[(df_clean[date_col] >= prev_start_ref) & (df_clean[date_col] <= prev_end_ref)]
    df_cur = df_clean[(df_clean[date_col] >= cur_start_ref) & (df_clean[date_col] <= cur_end_ref)]
else:
    # if no date column, use full
    df_prev = df_clean.copy()
    df_cur = df_clean.copy()

# Compute prev metrics
unique_prev = df_prev.dropna(subset=[incident_col]).drop_duplicates(subset=[incident_col])
incident_count_prev = unique_prev[incident_col].nunique()
unit_responses_prev = df_prev[incident_col].count()

# percent changes
incident_pct = format_percent_change(incident_count_prev, incident_count_current)
unit_pct = format_percent_change(unit_responses_prev, unit_responses_current)

# ---------- Top level metric cards ----------
metric_cols = st.columns(3)
with metric_cols[0]:
    # col1 Incident Calls (unique)
    st.metric(label="Incident Calls (Unique)", value=f"{incident_count_current:,}", delta=incident_pct)

with metric_cols[1]:
    # col2 Unit Responses (repeating)
    st.metric(label="Unit Responses (Total)", value=f"{unit_responses_current:,}", delta=unit_pct)

with metric_cols[2]:
    # small summary: Loaded rows & unique incidents in dataset
    total_rows = len(filtered)
    total_unique_dataset = df_clean.dropna(subset=[incident_col])[incident_col].nunique()
    st.markdown("**Filtered rows:**")
    st.write(f"{total_rows:,} rows")
    st.markdown("**Unique Incident IDs in dataset:**")
    st.write(f"{total_unique_dataset:,}")

st.markdown("---")

# ---------- Charts Row: Bar charts for Call Type Category ----------
# Shared color mapping
color_map = px.colors.qualitative.Set2

col3, col4 = st.columns(2)

# col3: Unique Incidents by Call Type Category
with col3:
    if "Call Type Category" in df_clean.columns and incident_col in df_clean.columns:
        unique_inc = filtered.dropna(subset=[incident_col]).drop_duplicates(subset=[incident_col])
        call_type_incidents = unique_inc['Call Type Category'].value_counts().reset_index()
        call_type_incidents.columns = ['Call Type Category', 'Unique Incidents']

        fig_cat_incidents = px.bar(
            call_type_incidents.sort_values(by='Unique Incidents', ascending=False),
            x='Call Type Category',
            y='Unique Incidents',
            title='ðŸš¨ Unique Incidents by Call Type Category',
            color='Call Type Category',
            color_discrete_sequence=color_map,
            labels={"Unique Incidents": "Total Unique Incidents"},
            template="plotly_white"
        )
        fig_cat_incidents.update_layout(xaxis_tickangle=-45, showlegend=False)
        fig_cat_incidents.update_traces(hovertemplate='%{x}<br>Unique Incidents: %{y}')
        st.plotly_chart(fig_cat_incidents, use_container_width=True)
    else:
        st.info("Call Type Category or Incident column not found for Unique Incidents chart.")

# col4: Unit Responses by Call Type Category
with col4:
    if "Call Type Category" in df_clean.columns and incident_col in df_clean.columns:
        call_type_responses = filtered.groupby('Call Type Category')[incident_col].count().reset_index()
        call_type_responses.columns = ['Call Type Category', 'Unit Responses']

        fig_cat_responses = px.bar(
            call_type_responses.sort_values(by='Unit Responses', ascending=False),
            x='Call Type Category',
            y='Unit Responses',
            title='ðŸš’ Unit Responses by Call Type Category',
            color='Call Type Category',
            color_discrete_sequence=color_map,
            labels={"Unit Responses": "Number of Unit Responses"},
            template="plotly_white"
        )
        fig_cat_responses.update_layout(xaxis_tickangle=-45, showlegend=False)
        fig_cat_responses.update_traces(hovertemplate='%{x}<br>Unit Responses: %{y}')
        st.plotly_chart(fig_cat_responses, use_container_width=True)
    else:
        st.info("Call Type Category or Incident column not found for Unit Responses chart.")

st.markdown("---")

# ---------- Col5: Frequency table for Unique Incidents with Response Time stats ----------
st.header("Response Time Summary â€” Unique Incidents (one event per Incident #)")

# Prepare response time column detection
response_col = None
for c in df_clean.columns:
    if 'response time' in c.lower() or ('response' in c.lower() and 'time' in c.lower()):
        response_col = c
        break

if response_col is None:
    # try other likely names
    for c in df_clean.columns:
        if 'response' in c.lower():
            response_col = c
            break

if response_col is None:
    st.warning("No 'Response Time' column detected. Frequency tables will show counts but cannot compute response time stats.")
else:
    st.write(f"Detected response time column: **{response_col}**")

# Build unique incidents table
unique_incidents = filtered.dropna(subset=[incident_col]).drop_duplicates(subset=[incident_col])
if len(unique_incidents) == 0:
    st.info("No unique incidents in filtered data to summarize.")
else:
    # Convert response time to seconds
    if response_col:
        unique_incidents['_response_seconds'] = unique_incidents[response_col].apply(parse_response_time)
    else:
        unique_incidents['_response_seconds'] = np.nan

    grp_cols = ['Call Type Category', 'Incident Call Type Final']
    # ensure columns exist
    for c in grp_cols:
        if c not in unique_incidents.columns:
            unique_incidents[c] = "Unknown"

    summary_unique = (
        unique_incidents
        .groupby(grp_cols)
        .agg(
            Count_Incidents=(incident_col, 'nunique'),
            Avg_Response_s=('_response_seconds', 'mean'),
            P90_Response_s=('_response_seconds', lambda x: np.nanpercentile(x.dropna(), 90) if x.dropna().size>0 else np.nan)
        )
        .reset_index()
    )

    # format Avg and P90 as mm:ss
    summary_unique['Avg_Response'] = summary_unique['Avg_Response_s'].apply(lambda x: to_minsec_str(x) if not pd.isna(x) else "")
    summary_unique['P90_Response'] = summary_unique['P90_Response_s'].apply(lambda x: to_minsec_str(x) if not pd.isna(x) else "")

    display_cols = ['Call Type Category', 'Incident Call Type Final', 'Count_Incidents', 'Avg_Response', 'P90_Response']
    st.dataframe(summary_unique[display_cols].sort_values(by='Count_Incidents', ascending=False).reset_index(drop=True), height=350)

    # allow download
    csv_buf = summary_unique.to_csv(index=False).encode('utf-8')
    st.download_button("â¬‡ï¸ Download Unique Incidents Response Summary (CSV)", data=csv_buf, file_name="unique_incidents_response_summary.csv", mime="text/csv")

st.markdown("---")

# ---------- Col6: Frequency table for Unit Responses (all counts of Incident Call Type Finals) ----------
st.header("Response Time Summary â€” Unit Responses (all incident rows counted)")

if len(filtered) == 0:
    st.info("No unit response data in filtered dataset.")
else:
    df_responses = filtered.copy()
    if response_col:
        df_responses['_response_seconds'] = df_responses[response_col].apply(parse_response_time)
    else:
        df_responses['_response_seconds'] = np.nan

    for c in ['Call Type Category','Incident Call Type Final']:
        if c not in df_responses.columns:
            df_responses[c] = "Unknown"

    summary_responses = (
        df_responses
        .groupby(['Call Type Category', 'Incident Call Type Final'])
        .agg(
            Count_Responses=(incident_col, 'count'),
            Avg_Response_s=('_response_seconds', 'mean'),
            P90_Response_s=('_response_seconds', lambda x: np.nanpercentile(x.dropna(), 90) if x.dropna().size>0 else np.nan)
        )
        .reset_index()
    )

    summary_responses['Avg_Response'] = summary_responses['Avg_Response_s'].apply(lambda x: to_minsec_str(x) if not pd.isna(x) else "")
    summary_responses['P90_Response'] = summary_responses['P90_Response_s'].apply(lambda x: to_minsec_str(x) if not pd.isna(x) else "")

    display_cols_resp = ['Call Type Category', 'Incident Call Type Final', 'Count_Responses', 'Avg_Response', 'P90_Response']
    st.dataframe(summary_responses[display_cols_resp].sort_values(by='Count_Responses', ascending=False).reset_index(drop=True), height=350)

    # download
    csv_buf2 = summary_responses.to_csv(index=False).encode('utf-8')
    st.download_button("â¬‡ï¸ Download Unit Responses Summary (CSV)", data=csv_buf2, file_name="unit_responses_summary.csv", mime="text/csv")

st.markdown("---")

# ---------- Col7: Interactive Heatmap: Incidents by Day & Hour ----------
with st.expander("Interactive Heatmap: Incidents by Day & Hour", expanded=True):
    filtered_df = filtered.copy()

    # Rename 'Incident #' to 'Incident_Num' for consistency
    if incident_col in filtered_df.columns:
        filtered_df = filtered_df.rename(columns={incident_col: 'Incident_Num'})

    # Ensure correct ordering of days
    day_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    filtered_df['Day of Week'] = pd.Categorical(filtered_df['Day of Week'], categories=day_order, ordered=True)

    # Slider for Hour filter
    min_hour = int(filtered_df['Hour of Day'].min()) if not filtered_df['Hour of Day'].empty else 0
    max_hour = int(filtered_df['Hour of Day'].max()) if not filtered_df['Hour of Day'].empty else 23
    selected_hours = st.slider("Select Hour Range", min_hour, max_hour, (min_hour, max_hour))

    df_filtered_hours = filtered_df[(filtered_df['Hour of Day'] >= selected_hours[0]) & 
                                   (filtered_df['Hour of Day'] <= selected_hours[1])]

    # Aggregate data for heatmap (unique incidents)
    heatmap_data = (
        df_filtered_hours.groupby(['Day of Week', 'Hour of Day'])['Incident_Num']
        .nunique()
        .reset_index()
        .rename(columns={'Incident_Num': 'Total_Incidents'})
    )

    if heatmap_data.empty:
        st.info("No data available for the selected hours/days to build heatmap.")
    else:
        heatmap = alt.Chart(heatmap_data).mark_rect().encode(
            x=alt.X('Hour of Day:O', title='Hour of Day'),
            y=alt.Y('Day of Week:O', sort=day_order, title='Day of Week'),
            color=alt.Color('Total_Incidents:Q', scale=alt.Scale(scheme='redyellowgreen', reverse=True), title='Total Incidents'),
            tooltip=['Day of Week', 'Hour of Day', 'Total_Incidents']
        ).properties(
            width=900,
            height=420,
            title="ðŸ“Š Heatmap of Unique Incidents"
        ).interactive()

        st.altair_chart(heatmap, use_container_width=True)

        # Download button
        tmp_csv = heatmap_data.to_csv(index=False).encode('utf-8')
        st.download_button(label="â¬‡ï¸ Download Heatmap Data (CSV)", data=tmp_csv, file_name="heatmap_data.csv", mime="text/csv")

st.markdown("---")

# ---------- Additional exploratory visuals (optional) ----------
st.header("Additional Quick Visuals")

col_a, col_b = st.columns(2)

with col_a:
    # Top Units (bar)
    unit_counts = filtered['Unit'].value_counts().reset_index().head(20)
    unit_counts.columns = ['Unit','Count']
    if len(unit_counts) > 0:
        fig_units = px.bar(unit_counts, x='Unit', y='Count', title="Top Units by Response Count", template="plotly_white")
        fig_units.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig_units, use_container_width=True)
    else:
        st.info("No Unit data to show.")

with col_b:
    # Hourly distribution
    hourly = filtered.groupby('Hour of Day')[incident_col].nunique().reset_index()
    hourly.columns = ['Hour of Day','Unique Incidents']
    if len(hourly) > 0:
        fig_hour = px.bar(hourly, x='Hour of Day', y='Unique Incidents', title="Incidents by Hour of Day", template="plotly_white")
        st.plotly_chart(fig_hour, use_container_width=True)
    else:
        st.info("No hour data to show.")

st.markdown("---")

st.write("App built by a data analyst helper. Use the side panel to refine filters and date ranges. If you have custom column names, ensure they include 'Incident #' and a 'Response Time' column name or update the dataset headers accordingly.")
