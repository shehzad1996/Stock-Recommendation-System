
'''
author - Shehzad Darbar
'''
import tkinter as tk                    
from tkinter import ttk
from PIL import ImageTk, Image
import requests
from io import BytesIO
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pylab import rcParams
import yfinance as yf
from tkinter import messagebox
from tkinter import filedialog
import traceback
import tkinter.messagebox

#arima forecasting for the stock value
def mosttrending():
    
    import pytrends
    from pytrends.request import TrendReq
    import pandas as pd
    
    
    # import the TrendReq method from the pytrends request module
    from pytrends.request import TrendReq
    
    folder_selected = filedialog.askdirectory(title = 'Please select output directory')
    
    folder_selected =folder_selected +"/"
    
    # execute the TrendReq method by passing the host language (hl) and timezone (tz) parameters
    #LANGUAGE AS en-US and tz=360 RELATED TO USA TIMING
    pytrends = TrendReq(hl='en-US', tz=360)
    
    kw_list = ['share price','stock price']
    
    #https://www.premiumleads.com/en/blog/seo/how-to-get-google-trends-data-with-pytrends-and-python/
    
    #geographic location is set as US which can be modified
    #today 3-m is from today and past 3 months of data
    pytrends.build_payload(kw_list, cat=0, timeframe='today 3-m', geo='US')
    related_queries = pytrends.related_queries()#pytrends.interest_over_time()
    related_queries
    
    
    # Code for cleaning related queries to find top searched queries and rising queries
    top_queries=[]
    rising_queries=[]
    for key, value in related_queries.items():
        for k1, v1 in value.items():
            print(k1)
            if(k1=="top"):
                top_queries.append(v1)
            elif(k1=="rising"):
                rising_queries.append(v1)
                
    top_searched=pd.DataFrame(top_queries[1])
    top_searched
    rising_searched=pd.DataFrame(rising_queries[1])
    
    rising_searched.to_csv(folder_selected +"Stocks_search_volume.csv",index =None)
    tkinter.messagebox.showinfo("Process Completed", "Trending stocks search volume csv file is saved in :: "+folder_selected)
   
    
    


def recommendationengine():
    import pandas as pd
    import ta
    # input from the tkinter app
    
    from tkinter import filedialog as fd
    
    yahoo_tickers = filedialog.askopenfile(title = 'Select yahoo stock file')
    
    #yahoo_tickers = 'yahoo_tickers.csv'
    df = pd.read_csv(yahoo_tickers)
    
    #folder_selected = '/Users/user/Desktop/upwork/Client-0057/'
    #please select output directory
    folder_selected = filedialog.askdirectory(title = 'Please select output directory')
    
    folder_selected =folder_selected +"/"
    #folder to store the values of the recommendations
    #make a new folder if not exist
    
    newpath = folder_selected +'Recommendations/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    df.columns
    dfc = df.groupby('Country').size()
      
    #filter to only keep USA
    df = df[df['Country']=='USA']
        
        
    # now collect the data for each company using the yahoo finance
    # we will only collect data for the last 12 months 
    names = df.Ticker.to_list()
    
    #only keep first 100 stocks,
    #this can be increased but may take time#
    #if you are looking for only the specific stock only load those stocks in to input file
    names = names[0:100]
    
    len(names)
    
    # loop through our data and get all the data  in one file
     
    framelist = pd.DataFrame()
    for name in names:
        stock = yf.Ticker(name)
        prices = stock.history(period='max')
        prices['name'] = stock
        prices = prices[['name','Close']]
        framelist=framelist.append(prices)
     
  
    # proper format for name column
    
    framelist1 = framelist.copy()
    framelist1.dtypes
    
    framelist1['name']= framelist1['name'].astype('str')
    framelist1['name']= framelist1['name'].str.replace("yfinance.Ticker object <", '', regex=True)
    framelist1['name']= framelist1['name'].str.replace(">", '', regex=True)
        
      
    def MACDdecision(df):
        #df = test.copy()
        #df = df.squeeze()
        df.Close
        
        df['MACD_diff'] = ta.trend.macd_diff(df.Close)
        df.loc[(df.MACD_diff>0)&(df.MACD_diff.shift(1)<0),'Decision MACD']='Buy'
        
        
    def RSI_SMAdecision(df):
        df['RSI'] = ta.momentum.rsi(df.Close,window = 10)
        df['SMA200'] = ta.trend.sma_indicator(df.Close, window = 200)
        df.loc[(df.Close>df.SMA200)&(df.RSI<30),'Decision RSI/SMA']='Buy'
        
    framelist1.name
      
    #get the unique names
    unames = framelist1['name'].unique()
    
    #for each stock in the dataframe find the MACD and RSI
    framelist1.dtypes
    
    final = pd.DataFrame()
    for name in unames:
        #print(name)
        test = framelist1[framelist1['name']==str(name)]
        test.columns
        test = test[['Close']]
        MACDdecision(test)
        RSI_SMAdecision(test)
        test['name'] = name
        final = final.append(test)
        
    
    #from the below loop we will get to know if the macd and the RSI signals showing to buy the stock
    #at the last row of each stock
    #when we find the signals we save the records in to buying_list
    
    buying_list = []
    name_only = []
    
    
    for name in unames:
        test = final[final['name']==name]
        if test['Decision MACD'].iloc[-1] =='Buy':
            buyme = 'Buying signal MACD - '+name
            buying_list.append(buyme)
            name_only.append(name)
            
        if test['Decision RSI/SMA'].iloc[-1] =='Buy':
            buyme = 'Buying signal RSI/SMA - '+name
            buying_list.append(buyme)
            name_only.append(name)
            
            
    #buying list is now saved on the disk as a csv file
    
    buying_listdf = pd.DataFrame(buying_list)
    buying_listdf.columns =['Recommendations']
    buying_listdf.to_csv(newpath+'Recommendations.csv')
    
    tkinter.messagebox.showinfo("Process Completed", "Recommendations are saved in :: "+newpath,index=None)
   
  
    
            

def arimaforecast():
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    import numpy as np, pandas as pd
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    
    
     #folder_selected = '/Users/user/Desktop/upwork/Client-0057/'
    #please select output directory
    output_folder = filedialog.askdirectory(title = 'Please select output directory')
    output_folder = output_folder+'/'
    
    stock_nm =stock_ticker.get(1.0, "end-1c")# 
    
    stock = yf.Ticker(stock_nm)
    
    # WE TAKE THE DATA FOR THE LAST 12 MONTHS AND CHANGE THE FREQUENCY TO WEEKLY
    
    prices = stock.history(period='24mo')
    prices.head()
    
    prices.index
    
    #change the frequency of the data
    prices = prices.asfreq('W-Mon')
    
    # apply interpolation on the data frame for missing values
    prices = prices.interpolate()
    
    
    # now we will make the model which will predict the closing price
    #plot close price
    fig = plt.figure(figsize=(10,6))
    plt.plot(prices['Close'])
    plt.title(stock_nm+' GROUP closing price')
    plt.show()
    fig.savefig(output_folder+'Closing.png')
    
    
    #Distribution of the dataset
    df_close = prices['Close']
    df_close.plot(kind='kde')
    
    
    
    #Test for staionarity
    def test_stationarity(timeseries):
        #Determing rolling statistics
        rolmean = timeseries.rolling(3).mean()
        rolstd = timeseries.rolling(3).std()
        #Plot rolling statistics:
        plt.plot(timeseries, color='blue',label='Original')
        plt.plot(rolmean, color='red', label='Rolling Mean')
        plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean and Standard Deviation')
        plt.show(block=False)
        #plt.savefig('Closing_price.png')
        print("Results of dickey fuller test")
        adft = adfuller(timeseries,autolag='AIC')
        # output for dft will give us without defining what the values are.
        #hence we manually write what values does it explains using a for loop
        output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
        for key,values in adft[4].items():
            output['critical value (%s)'%key] =  values
        print(output)
        
    test_stationarity(df_close)
    
    
    #We can’t rule out the Null hypothesis because the p-value is bigger than 0.05. Additionally, the test statistics exceed the critical values. As a result, the data is nonlinear.
    
    #To separate the trend and the seasonality from a time series, 
    # we can decompose the series using the following code.
    
    
    
    result = seasonal_decompose(df_close, model='additive')
    fig = plt.figure()  
    fig = result.plot()  
    fig.savefig(output_folder+'seasonal_decompose.png')
    fig.set_size_inches(16, 9)

    
    #if not stationary then eliminate trend
    #Eliminate trend
    
    rcParams['figure.figsize'] = 10, 6
    df_log = np.log(df_close)
    
    
    #split data into train and training set
    train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
    fig = plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Closing Prices')
    plt.plot(df_log, 'green', label='Train data')
    plt.plot(test_data, 'blue', label='Test data')
    plt.legend()
    fig.savefig(output_folder+'training_testset.png')
    
    
    model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                          test='adf',       # use adftest to find optimal 'd'
                          max_p=3, max_q=3, # maximum p and q
                          m=1,              # frequency of series
                          d=None,           # let model determine 'd'
                          seasonal=False,   # No Seasonality
                          start_P=0, 
                          D=0, 
                          trace=True,
                          error_action='ignore',  
                          suppress_warnings=True, 
                          stepwise=True)
    print(model_autoARIMA.summary())
    model_autoARIMA.plot_diagnostics(figsize=(15,8))
    plt.show()
    
    
    
    #model_order from the model
    # Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
    results_as_html = model_autoARIMA.summary().tables[0].as_html()
    model_df = pd.read_html(results_as_html, header=0)
    model_df = model_df[0]
    model_df.columns   
    
    p1 = int(model_df.iloc[:,1]  [0]   [8])
    
    p2 = int(model_df.iloc[:,1]  [0]   [11])
    
    p3 = int(model_df.iloc[:,1]  [0]   [14])
    
    #Modeling
    # Build Model
    model = sm.tsa.arima.ARIMA(train_data, order=(p1,p2,p3))  
    fitted = model.fit()  
    print(fitted.summary())
    
    
    
    #Let’s now begin forecasting stock prices on the test dataset with a 95% confidence level.
    conf = fitted.forecast(11, alpha=0.05)  # 95% conf
    
    #make array to check the accuracy of the model
    pred = np.array(conf)
    
    actual = np.array(test_data)
    
    
    # Accuracy metrics
    def forecast_accuracy(forecast, actual):
        mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
           
        return mape
    
    
    #check the accuracy of the model
    model_accuracy = 100-(forecast_accuracy(conf.values, test_data.values))*100
    
    #pass this to tkinter app to show the accuracy of the model going foreward
    model_accuracy
    
    
    
    #plot the accuracy of the mode along with the real data
    #plot close price
    plt.figure(figsize=(10,6))
    plt.plot(conf,label='Forecast')
    plt.plot(test_data,label='test data')
    plt.plot(train_data,label='train data')
    plt.title(stock_nm+' GROUP closing price')
    plt.show()
    
    
    
    #now implement the model on the whole dataset and predict the future
    #Modeling
    # Build Model
    final_train = train_data.append(test_data)
    model = sm.tsa.arima.ARIMA(final_train, order=(p1,p2,p3))  
    fitted = model.fit()  
    print(fitted.summary())
    
    
    #Let’s now begin forecasting stock prices on the test dataset with a 95% confidence level.
    forecast = fitted.forecast(5, alpha=0.05)  # 95% conf
    

    #plot close price
    plt.figure(figsize=(10,6))
    plt.plot(forecast,label='Forecast')
    plt.plot(final_train,label='train data')
    plt.title(stock_nm+' GROUP closing price')
    plt.show()
    
    
    # save the forecast result
    forecast = forecast.reset_index()
    forecast.columns = ['date','forecast_values']
    forecast.to_csv(output_folder+'forecast.csv',index=None)
    
    tkinter.messagebox.showinfo("Process Completed", "Forecast is saved in :: "+output_folder)
   
    

#Design the tkinter app from here
  
root = tk.Tk()
root.title("Stock Recomendation - Master")
root.geometry("800x700")

# add bg to tkinter app
# collect image from this url
url= "https://beststocks.com/wp-content/uploads/2021/08/Best-growth-stocks-ideas-to-buy-today-for-long-term-investment.jpg"

response = requests.get(url)
bgimg = Image.open(BytesIO(response.content))
bgimg = bgimg.resize((800,700))

image2 =  ImageTk.PhotoImage(bgimg)

#add a tab control to contol the tabs and different tasks
tabControl = ttk.Notebook(root)
 
#create tab for arima forecasting of the stock
tab1 = ttk.Frame(tabControl)

#name tab 1
tabControl.add(tab1, text ='Tab 1')

#add image to the tab1
image_label = ttk.Label(tab1 , image =image2)
image_label.place(x = 0 , y = 120)


#user text box in tab-1

stock_ticker = tk.Text(tab1, height = 1 ,width = 25)
stock_ticker.place(x=400, y=60)

#Label for input box
stock_tickerlabel = tk.Label(tab1, text = "Input the Ticker value ::")
stock_tickerlabel.place(x=200, y=60)


#button to execute the function
# Button Creation
stock_ticker_btn = tk.Button(tab1,text = "Weekly-Forecast", command = arimaforecast)
stock_ticker_btn.place(x=400, y=85)

#Title for tab-1
stock_tickerlabel = tk.Label(tab1, text = "ARIMA Forecasting",font=('Helvetica bold',18))
stock_tickerlabel.place(x=300, y=10)




##create tab for recommendation using MACD and RSI
tab2 = ttk.Frame(tabControl)
tabControl.add(tab2, text ='Tab 2')

#add image to the tab2
image_label = ttk.Label(tab2 , image =image2)
image_label.place(x = 0 , y = 120)

#button to execute the function of recommendation
rec_btn = tk.Button(tab2,text = "Stock-Recommendation",command =recommendationengine )
rec_btn.place(x=400, y=70)

#button to get rising stock prices
res_btn = tk.Button(tab2,text = "Trending-Stock",command =mosttrending )
res_btn.place(x=200, y=70)


#expand both tabs
tabControl.pack(expand = 1, fill ="both")



import traceback

# You would normally put that on the App class
def show_error(self, *args):
    err = traceback.format_exception(*args)
    tkinter.messagebox.showerror('Exception',err)
# but this works too
tk.Tk.report_callback_exception = show_error
    
    
root.mainloop()  