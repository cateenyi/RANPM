# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 22:46:24 2021

@author: echasek
"""
import pandas as pd
import altair as alt
import plotly.express as px
import streamlit as st
#import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import linear_model
#from sklearn.metrics import accuracy_score
#import time

st.set_page_config(layout='wide')
st.cache()

#------------------------------#
## Functions
## Read file
                
def get_df(file):
  # get extension and read file
  extension = file.name.split('.')[1]
  if extension.upper() == 'CSV':
    df = pd.read_csv(file)
    df['Time'] = df['Time'].dt.date
  elif extension.upper() == 'XLS':
    df = pd.read_excel(file)
    df['Time'] = df['Time'].dt.date    
  elif extension.upper() == 'XLSX':
    df = pd.read_excel(file, engine='openpyxl')
    df['Time'] = df['Time'].dt.date
    
  return df

def get_df2(file):
  # get extension and read file
  extension = file.name.split('.')[1]
  if extension.upper() == 'CSV':
    df = pd.read_csv(file)
  elif extension.upper() == 'XLS':
    df = pd.read_excel(file)
  elif extension.upper() == 'XLSX':
    df = pd.read_excel(file, engine='openpyxl')

  return df

## Get inputs

def collect_inputs():
    inputs = dict()
    
    hour = st.sidebar.slider('Hour', 0,23)
    users = st.sidebar.slider('Users', 0,80000)
    inputs = {'hour': hour, 'users': users}
    return inputs

def main():
    st.title('RAN Performance Analysis')
    #Menu
    menu = ['Home', 'RNC Analysis', 'BSC Analysis', 'LTE Analysis', 'About App']

    choice = st.sidebar.selectbox('Select Option', menu)
    
    if choice == 'RNC Analysis':
        menu = ['KPI Trends', 'RNC Load Prediction']
        selection = st.sidebar.selectbox('Analysis', menu)
        
        if selection == 'KPI Trends':
            st.subheader('RNC Performance Analysis')
            expander1 = st.beta_expander('Expand to view required MAPS data')
            with expander1:
                st.write("""
                         Upload Busy Hour MAPS data with columns for
                         * Time
                         * 3G RNC
                         * 3G Data Volume (PS)(TB)_MTN(#)
                         * 3G Erlang (CS)_MTN(Erl)
                         * RNC Capacity_FachDchUsers(Number)
                         * RNC Capacity_IubThroughput(Number)
                         * RNC_DC_Load(%)
                         * RNC_MP LOAD(%)
                         
                         """)
            file = False
            file = st.file_uploader('Upload MAPS Report', type=['csv', 'xls', 'xlsx'])
            if not file:
                st.write("Upload a .csv, xls or .xlsx file to get started")
                return
            df = get_df(file)

## Aggregate voice and data
    
            rncvoice_total = df[['Time', '3G RNC', '3G Erlang (CS)_MTN(Erl)', 'RNC Capacity_FachDchUsers(Number)']].groupby(['Time']).sum()
            rncdata_total = df[['Time', '3G RNC', '3G Data Volume (PS)_MTN(MB)', 'RNC Capacity_IubThroughput(Number)']].groupby(['Time']).sum()
# Remove zero sum days
            rncvoice_total = rncvoice_total[rncvoice_total['3G Erlang (CS)_MTN(Erl)'] > 0]
            rncdata_total = rncdata_total[rncdata_total['3G Data Volume (PS)_MTN(MB)'] > 0]
            
#---------------------------------#
## Network Level Trends
            expander = st.beta_expander("View Network Level Trends")
            with expander:
            
                st.subheader("Trend Analysis")
                st.success('Network level')
                col1, col2 = st.beta_columns([1,1])
                chart2 = px.line(rncvoice_total, x=rncvoice_total.index, y='RNC Capacity_FachDchUsers(Number)', template='plotly_dark', labels={'RNC Capacity_FachDchUsers(Number)': 'Users', 'Time': ''}, title='Total FACH Users')
                col1.write(chart2)
                
                chart3 = px.line(rncdata_total, x=rncdata_total.index, y='RNC Capacity_IubThroughput(Number)', template='plotly_dark', labels={'RNC Capacity_IubThroughput(Number)': 'Iub Throughput', 'Time': ''}, title='Total Iub Throughput')
                col2.write(chart3)
    
#---------------------------------#
## RNC Level
    
            rnc = st.sidebar.multiselect('Select RNC', list(df['3G RNC'].unique()))
            df.set_index('Time', inplace=True)
            if rnc:
                cols = st.sidebar.selectbox('Select Fields to View', list(df.columns)[1:])
                columns = ['3G RNC'] + [cols]
                df2 = df[columns][df['3G RNC'].isin(rnc)]
                st.write('---')
                st.subheader("Trend Analysis")
                st.success('RNC level')

## Find max values per column in last 30 days

                expander = st.beta_expander("View Max Values per RNC (30 days)")
                with expander:
                        st.success('RNC Peak Values')
                    
# st.write(df2.tail(30).max())
                        data = {key:df[df['3G RNC']==key].tail(30)[cols].max() for key in rnc}
                        df3 = pd.DataFrame(data, index=[cols])
                    
# convert df to wide-format                    
                        df4 = df3.melt(var_name='3G RNC', value_name=cols)
                        
                        col1, col2 = st.beta_columns([1,1])
                        col1.write(df4)
                        chart = alt.Chart(df4).mark_bar(
                            cornerRadiusTopLeft=3,
                            cornerRadiusTopRight=3
                            ).encode(
                        x=cols,
                        y='3G RNC',
                        color='3G RNC'
                        )
                        col2.write(chart)
                        
    
    
                chart1 = px.line(df2, y=cols, color='3G RNC', template='plotly_dark', title=cols)
                st.write(chart1)
                
        elif selection == 'RNC Load Prediction':
            st.subheader('EvoC RNC 8300 DC Load Prediction')
            st.success('DC Load prediction based on hour of day and number of FACH users')
            st.write('The prediction is based on a Linear Model that takes time and users as the key features influencing DC load')
            
            expander = st.beta_expander('Expand to view required MAPS data')
            with expander:
                st.write("""
                         Upload hourly MAPS data with columns for
                         * Time
                         * 3G RNC
                         * 3G Data Volume (PS)(TB)_MTN(#)
                         * 3G Erlang (CS)_MTN(Erl)
                         * RNC Capacity_FachDchUsers(Number)
                         * RNC Capacity_IubThroughput(Number)
                         * RNC_DC_Load(%)
                         * RNC_MP LOAD(%)
                         
                         """)
            file = False 
            file = st.file_uploader('Upload MAPS Report', type=['csv', 'xls', 'xlsx'])
            if not file:
                st.write("Upload a .csv, xls or .xlsx file to get started")
                return
            
            df = get_df2(file)
            df.dropna(inplace=True)
            df['Time'] = df.Time.dt.hour

            X = df[['Time', 'RNC Capacity_FachDchUsers(Number)']]
            y = df[['RNC_DC_Load(%)']]
            lr = linear_model.LinearRegression()
            models = [lr]
            # train model
           
            for model in models:
                try:
                    lr.predict((X,y))
                except:
                    with st.spinner('Training Model...'):
                        lr.fit(X,y)
                        #time.sleep(5)
                        st.success("""
                                 The $R^2 %6.4f$ of your model is:
                                     """)
                        st.write(f'### {lr.score(X,y):.4f}')
                

            menu = ['Plot Actual vs Predicted DC Load', 'Make DC Load Prediction', 'Load out of Sample Test Data']
            option = st.sidebar.radio('Select Option', menu)
           
            if option == 'Plot Actual vs Predicted DC Load':
                
                st.success('Actual vs Predicted DC Load')
                y_pred = lr.predict(X)
                y_pred = pd.DataFrame(y_pred,columns=['Predicted DC Load(%)'])
    
                Y = pd.concat([y, y_pred], axis=1)
                Y['Index'] = Y.index
    
                chart = px.line(pd.melt(Y, ['Index']),x='Index', y='value',color='variable',
                                labels={'variable':'', 'value': 'DC Load', 'Index':''},
                                template='plotly_dark', width=800)
                st.write(chart)
            
            elif option == 'Make DC Load Prediction':
                    
                st.sidebar.success("Select Inputs to Predict RNC Load")
                
                inputdict = collect_inputs()
                inputdf = pd.DataFrame([inputdict])
                st.success('Predicted DC Load')                
                # make prediction
                
                X_test = inputdf[['hour', 'users']]
                
                st.write(f'### {lr.predict(X_test)[0][0]:.2f}%')
            
            else:
                # st.write('Upload test data: column1-hour, column2-users')
                file = st.sidebar.file_uploader('Upload test data (hour & users)', type=['csv', 'xls', 'xlsx'])
                if not file:
                    st.sidebar.write("Upload a .csv, xls or .xlsx file to get started")
                    return
                st.success('DC Load Prediction')
                X_test = get_df2(file)
                y_pred = pd.DataFrame(lr.predict((X_test)), columns=['Predicted RNC DC Load (%)'])
                y_pred = pd.concat(([X_test, y_pred]), axis=1)
                st.write(y_pred)
                
                
    
#---------------------------------#
        
    elif choice == 'BSC Analysis':
        st.subheader('BSC Performance Analysis')
        file = False 
        file = st.file_uploader('Upload MAPS Report', type=['csv', 'xls', 'xlsx'])
        if not file:
            st.write("Upload a .csv, xls or .xlsx file to get started")
            return
        df = get_df(file)
        df.set_index('Time', inplace=True)
        colsgph = ['BSCGPRS2_EPB1GPH0040LOAD(Number)', 'BSCGPRS2_EPB1GPH4160LOAD(Number)', 'BSCGPRS2_EPB1GPH6180LOAD(Number)',	'BSCGPRS2_EPB1GPH8190LOAD(Number)', 'BSCGPRS2_EPB1GPH9100LOAD(Number)']
        colscth = ['EPB1CTH0040LOAD(Times)', 'EPB1CTH4160LOAD(Times)', 'EPB1CTH6180LOAD(Times)', 'EPB1CTH8190LOAD(Times)', 'EPB1CTH9100LOAD(Times)']
        
        col1, col2 = st.beta_columns([1,1])
        bsc = st.sidebar.multiselect('Choose BSC', list(df['2G BSC'].unique()))
        if bsc:
            
            col1.success('BSC GPH Stats')
            cols1 = col1.selectbox('Select KPI to View', colsgph)
            columns1 = ['2G BSC'] + [cols1]
            df2 = df[columns1][df['2G BSC'].isin(bsc)]
                
            chart1 = alt.Chart(data=df2.reset_index(), height=200, width=800).mark_line( ).encode(
                x='Time:T',
                y=cols1,
                color='2G BSC'
                )
                
            if columns1:
                st.write(chart1)
                    
            col2.success('BSC CTH Stats')
            cols2 = col2.selectbox('Select KPI to View', colscth)
            columns2 = ['2G BSC'] + [cols2]
            df2 = df[columns2][df['2G BSC'].isin(bsc)]
            # chart2 = px.line(df2, y=columns2, color='2G BSC', template='plotly_dark',
            #                      labels={'value':cols2})
            chart2 = alt.Chart(data=df2.reset_index(), height=200, width=800).mark_line( ).encode(
                x='Time:T',
                y=cols2,
                color='2G BSC'
                )
                
            if columns2:
                st.write(chart2)
                
        # menu = ['GPH', 'CTH']
        # option = st.sidebar.radio('Select Option', menu)
        
        # if option == 'GPH':
        #     st.success('BSC GPH Load Trend')
                            
        #     cols = st.sidebar.selectbox('Select Field to View', colsgph)
        #     bsc = st.sidebar.multiselect('Choose BSC', list(df['2G BSC'].unique()))
        #     columns = ['2G BSC'] + [cols]
        #     df2 = df[columns][df['2G BSC'].isin(bsc)]
            
        #     chart = px.line(df2, y=columns, color='2G BSC', template='plotly_dark')
            
        #     if chart:
        #         st.write(chart)            
                
        # elif option == 'CTH':
        #     st.success('BSC CTH Load Trend')

        #     cols = st.sidebar.selectbox('Select Field to View', colscth)                            
        #     bsc = st.sidebar.multiselect('Choose BSC', list(df['2G BSC'].unique()))
        #     columns = ['2G BSC'] + [cols]
        #     df2 = df[columns][df['2G BSC'].isin(bsc)]
            
        #     chart = px.line(df2, y=columns, color='2G BSC', height=400, width=850)
            
        #     if chart:
        #         st.write(chart)            
       
            
    elif choice == 'LTE Analysis':
        st.subheader('LTE Performance Analysis')
                
        menu = ['PRB Analysis', 'KPI Trends']
        selection = st.sidebar.selectbox('Analysis', menu)
        file = False 
        file = st.file_uploader('Upload MAPS Report', type=['csv', 'xls', 'xlsx'])
        if not file:
            st.write("Upload a .csv, xls or .xlsx file to get started")
            return


        if selection == 'KPI Trends':
            df = get_df(file)            
            df.set_index('Time', inplace=True)            
            counter = st.sidebar.multiselect('Select KPI', list(df.columns))
    
                
            if len(counter)==1:
    
                st.write('---')
                st.subheader("Trend Analysis")
                st.success('Greater Accra Area')
##-----------------------------#
# Single variable chart
                # chart = px.line(df, x=df.index, y=counter[0], title=counter[0],
                                # template='seaborn', width=800)
                chart = alt.Chart(data=df.reset_index(), width=800, height=200).mark_area(opacity=0.3, color='#57A44C').encode(
                    x='Time:T',
                    y=counter[0]
                    )
                
                st.write(chart)
                
            elif len(counter)==2:
##-----------------------------#
# Line and Area Chart
                st.write('---')
                st.subheader("Trend Analysis")
                st.success('Greater Accra Area')    
                
                base = alt.Chart(df.reset_index()).encode(
                    alt.X('Time:T', axis=alt.Axis(title='Date'))
                ).properties(
                    width=800,
                    height=300)
                
                area = base.mark_area(opacity=0.3, color='#57A44C').encode(
                    alt.Y(counter[0],
                          axis=alt.Axis(title=counter[0], titleColor='#57A44C'))
    
                )
                
                line = base.mark_line(stroke='#5276A7', interpolate='monotone').encode(
                    alt.Y(counter[1],
                          axis=alt.Axis(title=counter[1], titleColor='#5276A7'))
                )
                
                chart = alt.layer(line, area).resolve_scale(
                    y = 'independent'
                )
                st.write(chart)
            elif len(counter)==3:
                st.sidebar.write('Select only 2 KPIs for comparison')
                
        elif selection == 'PRB Analysis':
            st.subheader("PRB Analysis")
            st.success('Greater Accra Area')
            df = get_df2(file)
            counter = st.sidebar.multiselect('Category', list(df.columns[1:]))
            
            if len(counter)==1:
                
                base = alt.Chart(df).encode(
                    alt.X('Year-Week:O', axis=alt.Axis(title='Year-Week'))
                    ).properties(
                        width=800,
                        height=300)
                        
                chart = alt.layer(
                    base.mark_area(color='#57A44C', opacity=0.3).encode(
                        y=alt.Y(counter[0], axis=alt.Axis(title=f'#Cell {counter[0]}', titleColor='#57A44C'))),
                    base.mark_area(color='#5276A7', opacity=0.5).encode(y=counter[0]),                   
                    )
                #y=alt.Y('y', axis=alt.Axis(format='$', title='dollar amount'))
                    
                st.write(chart)
                
            if len(counter)==2:
                
                base = alt.Chart(df).encode(
                    alt.X('Year-Week:O', axis=alt.Axis(title='Year-Week'))
                    ).properties(
                        width=800,
                        height=300)
                        
                chart = alt.layer(
                    base.mark_area(color='#57A44C', opacity=0.3).encode(
                        y=alt.Y(counter[0], axis=alt.Axis(title=f'#Cell {counter[0]}, {counter[1]}', titleColor='#57A44C'))),
                    base.mark_area(color='#5276A7', opacity=0.5).encode(y=counter[0]),                   
                    base.mark_area(color='red', opacity=0.8).encode(y=counter[1])
                    )
                #y=alt.Y('y', axis=alt.Axis(format='$', title='dollar amount'))
                    
                st.write(chart)
            
            if len(counter)==3:
                
                base = alt.Chart(df).encode(
                    alt.X('Year-Week:O', axis=alt.Axis(title='Year-Week'))
                    ).properties(
                        width=800,
                        height=300)
                        
                chart = alt.layer(
                    base.mark_area(color='#57A44C', opacity=0.3).encode(
                        y=alt.Y(counter[0], axis=alt.Axis(title=f'#Cell {counter[0]}, {counter[1]}, {counter[2]}', titleColor='#57A44C'))),
                    base.mark_area(color='#5276A7', opacity=0.5).encode(y=counter[1]),                   
                    base.mark_area(color='red', opacity=0.4).encode(y=counter[2])
                    )
                #y=alt.Y('y', axis=alt.Axis(format='$', title='dollar amount'))
                    
                st.write(chart)
            
    elif choice == 'About App':
        st.subheader('A web app for quick RAN performance analysis')
        
    else:
        st.subheader('Home')
        
    return

if __name__ == '__main__':
    main()
