import streamlit as st 
from functions import *
import yfinance as yf

st.set_page_config(page_title="Exploring Networks Centrality Over Time", layout="wide")

def plot_ts(df, nome, units, chart):
    fig = go.Figure()
    
    colors = ['#15616D','#001524','#FF7d00','#78290F','#FFECD1','#808080','#8a8a8a','#949494','#9d9d9d','#a7a7a7','#b1b1b1','#bbbbbb','#c5c5c5','#cecece','#d8d8d8','#e2e2e2','#ececec','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6','#f6f6f6']

    if chart == 'Hist':
        for i in range(len(df.columns)):
            fig.add_trace(go.Scatter(
                    x=df.index, y=df.iloc[:, i], line=dict(color=colors[i], width=3), name=str(df.columns[i])))

    elif chart == 'Hist pct.':
        #fig = px.ecdf(df, x="total_bill", color="sex", markers=True, lines=False, marginal="histogram")
        fig = px.histogram(df, x='Change',color_discrete_sequence=['#195385'],
                   marginal="box", # or violin, rug
                   hover_data={'Value':':.2f',
                               'Change': ':.2f',# customize hover for column of y attribute
                             'Z-score':':.2f'})

        fig.add_trace(go.Scatter(x=[df['Change'].iloc[-1].round(2)],
                         text=['Last Value'],
                         mode='markers+text',
                         marker=dict(color='#FF5003', size=10),
                         textfont=dict(color='#FF5003', size=12),
                         textposition='top center',
                         showlegend=False
                         ))

    elif chart=='Dist':

        fig = ff.create_distplot(df, group_labels,show_curve=True, bin_size=.2,show_hist=False,colors =colors)#colors=['#C44601','#F57600','#FAAF90','#D9E4FF','#8BABF1','#0073E6','#054FB9'])
    
    fig.add_annotation(
    text = (f"Source: Yahoo, Unicamp Institute of Economics.")
    , showarrow=False
    , x = 0
    , y = -0.19
    , xref='paper'
    , yref='paper' 
    , xanchor='left'
    , yanchor='bottom'
    , xshift=-1
    , yshift=-5
    , font=dict(size=10, color="grey")
    , font_family= "Verdana"
    , align="left"
    )
    
    fig.update_layout(title={ 'text': '<b>'+ nome+'<b>','y':0.95,'x':0.5,'xanchor': 'center','yanchor': 'top'},
                            paper_bgcolor='rgba(0,0,0,0)', #added the backround collor to the plot 
                            plot_bgcolor='rgba(0,0,0,0)',
                             title_font_size=20,
                             font_color = '#0D1018',
                             #xaxis_title=f"{source}",
                             yaxis_title=units, 
                             template='plotly_white',
                             font_family="Verdana",
                             images=[dict(source='https://raw.githubusercontent.com/caiquecober/Research/master/logo_ie.png',
                                 xref="paper", yref="paper",
                                 x=0.92, y=-0.17,
                                 sizex=0.15, sizey=0.15,
                                 opacity=0.8,
                                 xanchor="center",
                                 yanchor="middle",
                                 sizing="contain",
                                 visible=True,
                                 layer="below")],
                             legend=dict(
                                 orientation="v",
                                 yanchor="bottom",
                                 y=0,
                                 xanchor="left",
                                 x=1.1,
                                 font_family='Verdana'),
                                 autosize=True,
                                 height=500,
                                 )
    
    fig.update_xaxes(showgrid=False,showline=True, linewidth=1.5, linecolor='black')
    fig.update_yaxes(showgrid=False,showline=True, linewidth=1.5, linecolor='black')
    fig.update_layout(
        yaxis=dict(linecolor="black", showgrid=False, tickwidth=1, tickcolor="black", ticks="inside"), 
        xaxis=dict(linecolor="black", showgrid=False, tickwidth=1, tickcolor="black", ticks="inside")
    )
    

    # if chart =='Hist pct.':
    #     #test if asset histogram should be in percentages or not.
    #     if lst_pct.__contains__(asset1)== True:
    #         fig.update_layout(xaxis= { 'tickformat': ',.2%'})
                    
    return fig



@st.cache_resource()
def load_data(slc_title,var_name):
    raw_df = pd.read_pickle('hist (2).pkl')
    raw_df = raw_df[raw_df[var_name]== slc_title]
    raw_df = raw_df.pivot(columns='variable_3',values='value',index='Date')
    return raw_df


@st.cache_resource()
def  calculate_network(raw_df):

    window_size = 100  # Set the rolling window size
    central_nodes_over_time = pipeline_graph_with_fixed_rolling_window(raw_df, 0.5,1, window_size)

    # You can then visualize or analyze central_nodes_over_time as needed.
    # Flatten the data and calculate inverse ranks for each day
    flattened_data = []
    for time, asset_dict in central_nodes_over_time:
        centrality_values = asset_dict
        sorted_assets = sorted(centrality_values, key=centrality_values.get, reverse=True)
        rank = {asset: 3 - sorted_assets.index(asset) for asset in sorted_assets}
        for asset, degree_centrality in centrality_values.items():
            flattened_data.append((time, asset, degree_centrality, rank[asset]))

    # Create a DataFrame
    df = pd.DataFrame(flattened_data, columns=['Timestamp', 'Asset', 'Eigenvector Centrality', 'Rank'])

    return df

############################################ Start of the application code

############################### Header Work


st.markdown('# Exploring Newtorks Centrality Over Time')
tab1, tab2 = st.tabs(['Static','Build Your Own'])


css = '''
<style>
    .stTabs [data-baseweb="tab-highlight"] {
        background-color:#e97510;
    }
    
    .stTabs [data-testid="stMarkdownContainer"] {
        color:black;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)
st.markdown(""" <style> 
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

with tab1:

    col1,col2,_ = st.columns([1,1,5])
    slc_mj_mi = col1.selectbox('Choose Categories',['Major','Minor'])

    if slc_mj_mi == 'Major':
        slc_title = col2.selectbox('Chose Asset Class',['Commodities','Currencies','Equities','Fixed Income','Crypto'])
        var_name = 'variable_0'
    if slc_mj_mi ==  'Minor':
        slc_title = col2.selectbox('Chose Asset Class',['Energy','Agg','Metals','USD/DM','USD/EM','EUR/ALL','Others',
                                                        'US Factors and Sectors','EU Factors and Sectors','Countries',
                                                        'US Fixed Income','EU Fixed Income','Smart Contract',
                                                        'Metaverse','DeFi','Privacy Coins'])
        var_name = 'variable_1'

    ################################ Data Work
    raw_df= load_data(slc_title, var_name)
    st.write(raw_df.head())

    df = calculate_network(raw_df)
    st.write(df.head())

    rolling_avg_data  = df.pivot(index='Timestamp',columns='Asset',values='Rank').fillna(0).rolling(90).mean()

    # Calculate the average of each column
    column_averages = rolling_avg_data.mean()

    # Sort columns based on average values in descending order
    sorted_columns = column_averages.sort_values(ascending=False)

    # Reorder the DataFrame based on the sorted column order
    rolling_avg_data = rolling_avg_data[sorted_columns.index]


    fig =  plot_ts(rolling_avg_data,'90-Day Rolling Mean of Rankings Over Time','Mean Rank', 'Hist')


    st.plotly_chart(fig, use_container_width=True)


with tab2:
    import re
    col1,col2,col3 = st.columns([3,3,1])
    tickers = col1.text_input('Write Tickers','BIL,SHY,EDV,FXY,RWM,EMLC,FXI')
    lst_tickers = re.split(',',tickers)
    def get_etf(lst_tickers):

        df = []

        for i in lst_tickers:
            ts = yf.Ticker(i)
            historical = ts.history(period="5y")['Close']
            #print(historical)
            df.append(historical)

        data =  pd.concat(df, axis=1)
        #data.index = data.index.tz_convert(None)
        data.columns = lst_tickers
        return data 
    
    df_personal = get_etf(lst_tickers)

    df = calculate_network(df_personal)
    st.write(df.head())

    rolling_avg_data  = df.pivot(index='Timestamp',columns='Asset',values='Rank').fillna(0).rolling(90).mean()

    # Calculate the average of each column
    column_averages = rolling_avg_data.mean()

    # Sort columns based on average values in descending order
    sorted_columns = column_averages.sort_values(ascending=False)

    # Reorder the DataFrame based on the sorted column order
    rolling_avg_data = rolling_avg_data[sorted_columns.index]


    fig =  plot_ts(rolling_avg_data,'90-Day Rolling Mean of Rankings Over Time','Mean Rank', 'Hist')


    st.plotly_chart(fig, use_container_width=True)


