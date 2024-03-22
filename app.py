
# import library
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st
from pandas.tseries.offsets import DateOffset


# define functions

# load data
@st.cache_data # loads this function once and doesn't reload everytime we reload our page
def load_data():

    # load data
    df_vid = pd.read_csv('resource/Video_Performance_Over_Time.csv')

    # dropping row Total to avoid issue with calculations
    df_agg = pd.read_csv('resource/Aggregated_Metrics_By_Video.csv').iloc[1:,:]

    df_agg_sub = pd.read_csv('resource/Aggregated_Metrics_By_Country_And_Subscriber_Status.csv')

    df_com = pd.read_csv('resource/All_Comments_Final.csv')

    # converting all columns to uppercase
    for df in [df_agg,df_vid,df_agg_sub, df_com]:
        df.columns = df.columns.str.upper()

    # remove \xad
    df_agg.columns = df_agg.columns.str.replace('\xad','')

    # cleaning  val
    df_agg = df_agg.replace([np.inf, -np.inf], np.nan).dropna()

    # convert date to datetype

    df_agg['VIDEO PUBLISH TIME'] = pd.to_datetime(df_agg['VIDEO PUBLISH TIME'], format='mixed')

    # rename column

    df_agg.rename(columns={'VIDEO PUBLISH TIME': 'PUBLISH DATE', 'COMMENTS ADDED' : 'COMMENTS'}, inplace=True)
        
    df_agg['AVERAGE VIEW DURATION'] = df_agg['AVERAGE VIEW DURATION'].apply(lambda x: datetime.strptime(x, '%H:%M:%S'))

    # create new column for df_agg['AVERAGE VIEW SECONDS'
    df_agg['AVERAGE VIEW SECONDS'] = df_agg['AVERAGE VIEW DURATION'].apply(lambda x: x.second + x.minute * 60 + x.hour * 60 * 60)

    # engagement ration, every engagement a view could do divided by the number of viewrs
    df_agg['ENGAGEMENT RATIO'] = (df_agg['SHARES'] + df_agg['LIKES'] + df_agg['DISLIKES'] + df_agg['COMMENTS']) / df_agg['VIEWS'] 

    # ratio of views to subscribers gained
    df_agg['VIEW TO SUBSCRIBER RATIO'] = df_agg['VIEWS'] / df_agg['SUBSCRIBERS GAINED'] # how views does it take to gain a subscriber

    # ratio of views to subscribers lost, 
    df_agg['VIEW TO SUBSCRIBER LOST RATIO'] = df_agg['VIEWS'] / df_agg['SUBSCRIBERS LOST'] # how views does it take to lose a subscriber

    # sort data by 'VIDEO PUBLISH TIME'
    df_agg.sort_values(by = 'PUBLISH DATE', ascending=False, inplace=True)

    # CONVERTING DATE to datetime
    df_vid['DATE'] = pd.to_datetime(df_vid['DATE'], format='mixed')

    df_com['DATE'] = pd.to_datetime(df_com['DATE'])
    df_com.rename(columns={'VIDID':'VIDEO'}, inplace=True)

    # removing '_'
    df_com.columns = df_com.columns.str.replace('_', ' ')

    # create dataframe
    return df_vid, df_agg, df_agg_sub, df_com

df_vid, df_agg, df_agg_sub, df_com = load_data()

# country code naming
def audience_sample(country):
    # top countries
    if country == 'US':
        return 'USA'
    elif country == 'IN':
        return 'INDIA'
    elif country == 'CI':
        return 'CHINA'
    else:
        return 'OTHERS'

# engineer data

# aggregated differential 

# create a copy of our dataframe
df_agg_diff = df_agg.copy()

def duration_month():
    # duration of data

    max_month = (df_agg_diff['PUBLISH DATE'] - df_agg_diff['PUBLISH DATE']) 

    data_duration = st.sidebar.slider(
        'Range of Data in months',
        3,56,3
        )
    return data_duration


# build dashboard

# sidebar
add_sidebar = st.sidebar.selectbox("Aggregate or Individual Video", ("Aggregate Metrics", "Individual Video Analysis"))

# styling dataframe
def styling_positive(value, props):
    try:
        return props if value >= 0 else None
    except:
        pass


def styling_negative(value, props):
    try:
        return props if value < 0 else None
    except:
        pass

## local picture
if add_sidebar == "Aggregate Metrics":

    metric_agg = df_agg[[
        'PUBLISH DATE',
        'COMMENTS', 
        'SHARES', 
        'DISLIKES', 
        'LIKES', 
        'SUBSCRIBERS GAINED', 
        'RPM (USD)', 
        'VIEWS', 
        'YOUR ESTIMATED REVENUE (USD)',
        'AVERAGE VIEW SECONDS', 
        'ENGAGEMENT RATIO', 
        'VIEW TO SUBSCRIBER RATIO',
    ]]  

    # duration
    n_month = duration_month()

    st.header(f'Aggregated data ')

    def metric_median(n_month):
        # date range
        metric_date_n = metric_agg['PUBLISH DATE'].max() - DateOffset(months=n_month)
        median_date_n = metric_agg[metric_agg['PUBLISH DATE'] >= metric_date_n].median()
        
        return metric_date_n,median_date_n

    metric_nmo, median_nmo = metric_median(n_month)

    col1, col2, col3,col4,col5,col6 = st.columns(6)
    columns = [col1, col2, col3,col4,col5,col6]

    count = 0
    for i in median_nmo.index:
        with columns[count]:
            if i != 'PUBLISH DATE':
                delta = (median_nmo[i] - median_nmo[i])/median_nmo[i]
                st.metric(label = i, value =round(median_nmo[i]), delta="{:.4%}".format(delta))
            else:
                delta = median_nmo[i] - median_nmo[i]
                st.metric(label = 'Months range', value = n_month, delta=f"{round((n_month/12),2)} years")
            count += 1
            if count >= 6:
                count = 0


    df_agg_diff_final = df_agg_diff.loc[:,[
    'VIDEO',
    'VIDEO TITLE',
    'PUBLISH DATE',
    'COMMENTS',
    'SHARES',
    'DISLIKES',
    'LIKES',
    'SUBSCRIBERS LOST',
    'SUBSCRIBERS GAINED',
    'VIEWS',
    'SUBSCRIBERS',
    'YOUR ESTIMATED REVENUE (USD)',
    'IMPRESSIONS',
    'IMPRESSIONS CLICK-THROUGH RATE (%)',
    'AVERAGE VIEW DURATION',
    'AVERAGE VIEW SECONDS',
    'ENGAGEMENT RATIO',
    'VIEW TO SUBSCRIBER RATIO',
    'VIEW TO SUBSCRIBER LOST RATIO']
    ]
    
    # extract only date
    df_agg_diff_final['PUBLISH DATE'] = df_agg_diff_final['PUBLISH DATE'].dt.date

    # extracting time
    df_agg_diff_final['AVERAGE VIEW DURATION'] = df_agg_diff_final['AVERAGE VIEW DURATION'].dt.time

    # table
    table_sidebar = st.sidebar.selectbox("Count or percentage", ("Count values", "Percentage value"))

    if table_sidebar != 'Count values':

        # Select only the numeric columns
        numeric_columns = df_agg_diff_final.select_dtypes(include=['number'])

        df_agg_numeric_lst = df_agg_diff_final[df_agg_diff_final.columns[2:]].columns.tolist()
        
        #df_to_percent

        # Calculate column sums
        column_sums = numeric_columns.sum()

        # Divide each element by its column sum and multiply by 100 to get percentages and round

        df_to_percent = {}
        for i in numeric_columns:
            df_to_percent[i] = '{:.1%}'.format

        st.title('Percentage Value')
        st.dataframe(df_agg_diff_final.style.hide().map(styling_positive, props = 'color:green;').map(styling_negative, props = 'color:red;').format(df_to_percent))
   
    else:
        st.title('Count value')
        st.dataframe(df_agg_diff_final.style.hide().map(styling_positive, props = 'color:green;').map(styling_negative, props = 'color:red;'))


## individual video

elif add_sidebar =="Individual Video Analysis":

    # videos title list
    video_title = tuple(df_agg['VIDEO TITLE'])
    # select videos
    selected_video = st.selectbox("Select Video", video_title)
    # filtered by title
    filtered_video = df_agg[df_agg['VIDEO TITLE'] == selected_video]

    # subscribers filter
    filtered_agg_sub = df_agg_sub[df_agg_sub['VIDEO TITLE'] == selected_video]
        
    filtered_agg_sub['COUNTRY'] = filtered_agg_sub['COUNTRY CODE'].apply(audience_sample)

    filtered_agg_sub.sort_values(by='IS SUBSCRIBED', inplace=True)

    # vid data
    filtered_vid = filtered_agg_sub['EXTERNAL VIDEO ID'].unique()[0]
    df_com_filtered = df_com[df_com['VIDEO'] == filtered_vid ]

    # converting boolean to int
    #filtered_agg_sub['IS SUBSCRIBED'] = filtered_agg_sub['IS SUBSCRIBED'].astype('int')

    # metric column
    ind_col = filtered_agg_sub.loc[:,[
        'VIDEO LENGTH', 
        'VIEWS', 
        'VIDEO LIKES ADDED', 
        'VIDEO DISLIKES ADDED', 
        'USER SUBSCRIPTIONS ADDED',
        'AVERAGE WATCH TIME', 
        'USER COMMENTS ADDED']
    ]

    # sidebar to select columns to plot
    x_chart = st.sidebar.selectbox(
        'Select plot values against Subscribers',
        ind_col.columns[1:]
    )

    st.title(f'Plot of Subscribers and non-subscribers and their interaction on Video: {x_chart}')
    # plot in plotly
    fig = px.bar(filtered_agg_sub, x=x_chart, y = 'IS SUBSCRIBED', color = 'COUNTRY', orientation = 'h')
    st.plotly_chart(fig)

    st.title('Summary of Video Engagement')

    # metric column renaming
    ind_col.rename(columns={
        'VIDEO LIKES ADDED' : 'LIKES', 
        'VIDEO DISLIKES ADDED' : 'DISLIKES', 
        'USER SUBSCRIPTIONS ADDED' : 'SUBSCRIPTIONS',
        'USER COMMENTS ADDED' : 'COMMENTS'
        }, inplace = True)

    col1, col2, col3,col4,col5,col6,col7 = st.columns(7)
    columns = [col1, col2, col3,col4,col5,col6,col7]

    count = 0
    for i in ind_col.columns:
        with columns[count]:
            if i == 'COMMENTS':
                delta = df_com_filtered[i].count()
                st.metric(label = i, value = delta)
            elif i not in ['VIDEO LENGTH', 'AVERAGE WATCH TIME']:        
                delta = round(ind_col[i].sum())       
                st.metric(label = i, value = delta)
            else:
                delta = ind_col[i].mean()
                st.metric(label = i, value = delta )
            count += 1

    st.title('Video comprehensive interactions')
    st.dataframe(df_com_filtered)


#### support

with st.sidebar:
    st.markdown('''
    
    
    🌟  If you've enjoyed my work and would like to support me, 
    consider buying me a coffee!    
    Your generosity keeps me inspired and fuels my creativity. 
    Thank you for being a part of my journey. ☕️💫
    ''')

    st.markdown('[Support me on Ko-fi](https://ko-fi.com/onscript)')
