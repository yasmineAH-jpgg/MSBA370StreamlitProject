import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import SessionState
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import Dashboard
import datetime
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from types import FunctionType

marketing_df=pd.read_csv('data/marketing_data.csv')
marketing_df=Dashboard.clean_data(marketing_df)
st.sidebar.title("Menu")
options= st.sidebar.radio("Navigate",["Interactive Analysis","Data Analysis","Statistical Analysis","Predicting Purchases"])
st.sidebar.title("About")
st.sidebar.info("This is an easy interactive analytics tool that can guide you through your marketing decisions!")
inData=False

def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

img = load_image('data/audience-analytics-header.jpg')
st.image(img)
st.title("Exploratory Data Analysis for Purchasing Store")
#options= st.selectbox('Please Select',['PowerBI'])

df_read=pd.DataFrame()
#session_state=SessionState.get(df=df_read)

#session_state=SessionState.get(df=df)

if options=='Interactive Analysis':
	#st.markdown("Here we will share the dataframe")
	st.subheader("Dataset")
	if inData==False:
		data_file=st.file_uploader("Upload CSV",type=['csv'])
	
	#if inData==True:
	if data_file is None:
		st.markdown('Kindly Upload your CSV file')
	if data_file is not None:
		inData=True
		file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
		st.write(file_details)
		df_read = pd.read_csv(data_file)
		session_state = SessionState.get(df=df_read)
		st.dataframe(session_state.df)
	#if st.button("Upload different csv file"):
	#	inData=False

	st.subheader("Visualizations")
	st.markdown("Here we will put the analytics graphs and charts")
	options=st.multiselect('Dataset Variables',session_state.df.columns)
	print(options)
	select = st.selectbox('Visualization type', ['Bar Chart', 'Pie Chart'], key='1')
	if select=="Pie Chart":
		if len(options)==0:
			st.write('Nothing Selected Yet')
		#count=session_state.df.groupby(options[1])
		elif len(options)==1:
			data1_pie=session_state.df[options[0]].value_counts()
			fig = px.pie(session_state.df, values=data1_pie,names=data1_pie.index.tolist())
		elif len(options)==2:
			data2_pie= session_state.df[[options[0],options[1]]]
			data2_pie=data2_pie.groupby([options[0]]).sum()
			#print(data2_pie)
			#print(data2.index.tolist())
			fig = px.pie(session_state.df,values=data2_pie[options[1]].tolist(),names=data2_pie.index.tolist())
		st.plotly_chart(fig)
	if select=="Bar Chart":
		if len(options)==0:
			st.write('Nothing Selected Yet')
		elif len(options)==1:
			data1_bar=session_state.df[options[0]].value_counts()
			fig=px.bar(x=data1_bar.index.tolist(),y=data1_bar)
			st.plotly_chart(fig)
		else:
			data2_bar= session_state.df[[options[0],options[1]]]
			data2_bar=data2_bar.groupby([options[0]]).sum()
			#print(data2_pie)
			#print(data2.index.tolist())
			fig = px.bar(session_state.df, x=data2_bar.index.tolist(), y=data2_bar[options[1]].tolist())
			st.plotly_chart(fig)
	
if options=='Data Analysis':
	#heatmap=Dashboard.null_heatmap(marketing_df)
	st.subheader("Original Data Insights")
	#st.write(heatmap,"Null Values in Dataset",width=1)
	#st.markdown("As we can see from the plot, there are some null values in Income column so we drop them in Analysis")
	
	marketing_df=Dashboard.delete_nulls(marketing_df)
	num_plots=Dashboard.num_plots(marketing_df)
	num_plots=num_plots.tolist()
	st.write(num_plots[0].figure, "Num_Plots")

	numeric_dist=Dashboard.numeric_dist(marketing_df)
	st.write(numeric_dist[0][0].figure,"Numeric Distribution of Data")

	marketing_df=Dashboard.handle_anomalies(marketing_df)

	#Total kids
	marketing_df['Totalkids'] = marketing_df['Kidhome'] + marketing_df['Teenhome']
	marketing_df['YearCustomer'] = pd.DatetimeIndex(marketing_df['Dt_Customer']).year


	# total amount spent
	mnt_cols = [col for col in marketing_df.columns if 'Mnt' in col]
	marketing_df['TotalMnt'] = marketing_df[mnt_cols].sum(axis=1)

	# Total Purchases
	purchases_cols = [col for col in marketing_df.columns if 'Purchases' in col]
	marketing_df['TotalPurchases'] = marketing_df[purchases_cols].sum(axis=1)

	# Total Campaigns Accepted
	campaigns_cols = [col for col in marketing_df.columns if 'Cmp' in col] + ['Response'] 
	marketing_df['TotalCampaignsAcc'] = marketing_df[campaigns_cols].sum(axis=1)

	#age
	year=datetime.datetime.today().year
	marketing_df['Age']=year-marketing_df['Year_Birth']

	#Age_group
	bins= [18,39,59,90]
	labels = ['Adult','Middle Age Adult','Senior Adult']
	marketing_df['AgeGroup'] = pd.cut(marketing_df['Age'], bins=bins, labels=labels, right=False)
	marketing_df['AgeGroup'] = marketing_df['AgeGroup'].astype('object')

	data1_pie=marketing_df['Education'].value_counts()
	fig = px.pie(marketing_df, values=data1_pie,names=data1_pie.index.tolist(),color_discrete_sequence=px.colors.sequential.Agsunset)
	st.write(fig,'Level of Education')

	data1_pie=marketing_df['Marital_Status'].value_counts()
	fig = px.pie(marketing_df, values=data1_pie,names=data1_pie.index.tolist(),color_discrete_sequence=px.colors.sequential.Agsunset)
	st.write(fig,'Marital Status')

	data1_pie=marketing_df['Country'].value_counts()
	fig = px.pie(marketing_df, values=data1_pie,names=data1_pie.index.tolist(),color_discrete_sequence=px.colors.sequential.Agsunset)
	st.write(fig,'Countries')

	data1_pie=marketing_df['AgeGroup'].value_counts()
	fig = px.pie(marketing_df, values=data1_pie,names=data1_pie.index.tolist(),color_discrete_sequence=px.colors.sequential.Agsunset)
	st.write(fig,'Age Group')

	st.subheader("Findings")
	#st.write("Almost 50% of clients' education level is graduate, and few customers have an primary level of education \n"
	#		  "The number of married clients is more than widowed and divorced \n"
	#		  "There is a remarkably high percentage of customers in Spain while the percentage of clients in the United States and Montenegro is very small \n"
	#		  "There is a very high percentage of clients between 39 to 59 years old compared to other age groups.")
	st.markdown("Almost 50% of clients' education level is graduate, and few customers have an primary level of education.")
	st.markdown("The number of married clients is more than widow and divorce.")
	st.markdown("There is a remarkably high percentage of customers in Spain while the percentage of clients in the United States and Montenegro is very small")
	st.markdown("There is a very high percentage of clients between 39 to 59 years old compared to other age groups.")

	st.subheader("Total Spending")
	data2_bar= marketing_df[['Education','TotalMnt']]
	data2_bar=data2_bar.groupby(['Education']).sum()
	fig = px.bar(marketing_df, x=data2_bar.index.tolist(), y=data2_bar['TotalMnt'].tolist(),color_discrete_sequence =['indigo']*len(marketing_df))
	st.write("Education vs Total Spending")
	st.plotly_chart(fig)
	
	data2_bar= marketing_df[['Marital_Status','TotalMnt']]
	data2_bar=data2_bar.groupby(['Marital_Status']).sum()
	fig = px.bar(marketing_df, x=data2_bar.index.tolist(), y=data2_bar['TotalMnt'].tolist(),color_discrete_sequence =['indigo']*len(marketing_df))
	st.write("Marital Status vs Total Spending")
	st.plotly_chart(fig)
	
	data2_bar= marketing_df[['Country','TotalMnt']]
	data2_bar=data2_bar.groupby(['Country']).sum()
	fig = px.bar(marketing_df, x=data2_bar.index.tolist(), y=data2_bar['TotalMnt'].tolist(),color_discrete_sequence =['indigo']*len(marketing_df))
	st.write("Countries vs Total Spending")
	st.plotly_chart(fig)
	
	data2_bar= marketing_df[['AgeGroup','TotalMnt']]
	data2_bar=data2_bar.groupby(['AgeGroup']).sum()
	fig = px.bar(marketing_df, x=data2_bar.index.tolist(), y=data2_bar['TotalMnt'].tolist(),color_discrete_sequence =['indigo']*len(marketing_df))
	st.write("Age Group vs Total Spending")
	st.plotly_chart(fig)
	
	st.subheader("Findings")
	st.markdown("People with PhDs used to spend more than other group of people")
	st.markdown("Total spending of divorced, single, and married group members is roughly equal while widows spending is slightly higher than these individuals.")
	st.markdown("Montenegro spends significantly more than other countries.")

	st.subheader("Total Purchases")
	channels = ['NumWebPurchases', 'NumCatalogPurchases',  'NumStorePurchases']
	data = marketing_df[channels].sum()
	data=data.tolist()
	fig = px.bar(x=['NumWebPurchases', 'NumCatalogPurchases',  'NumStorePurchases'], y=data,color_discrete_sequence =['palevioletred']*len(marketing_df))
	#x=sns.barplot(x=channels,y=data.values, palette= sns.color_palette("rocket", as_cmap=False))
	st.write("Purchases per Category")
	st.plotly_chart(fig)

	st.subheader("Total Amount  Spent on each Product")
	col_products = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
	data = marketing_df[col_products].sum().tolist()
	fig = px.bar(x=col_products, y=data,color_discrete_sequence =['rebeccapurple']*len(marketing_df))
	#x=sns.barplot(x=channels,y=data.values, palette= sns.color_palette("rocket", as_cmap=False))
	st.write("Purchases per Category")
	st.plotly_chart(fig)

	st.subheader("Income per Age Group")
	fig=px.bar(marketing_df,x="AgeGroup",y="Income",color_discrete_sequence =['palevioletred']*len(marketing_df))
	st.plotly_chart(fig)

	st.subheader("Purchases per Age Group")
	Purchases = ['NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases']
	dataset = marketing_df.groupby('AgeGroup')[Purchases].mean()
	score_label = np.arange(0, 10, 1)
	Adult_mean  = list(dataset.T['Adult'])
	Middleage_mean  = list(dataset.T['Middle Age Adult'])
	SeniorAdult_mean  = list(dataset.T['Senior Adult'])
	# set width of bar
	barWidth = 0.35

	fig, ax = plt.subplots(figsize=(19,8))

	# Set position of bar on X axis
	r1 = np.arange(0,len(Purchases)*2,2)
	r2 = [x + barWidth for x in r1]
	r3 = [x + barWidth for x in r2]

	# Make the plot

	Adult = ax.bar(r1, Adult_mean, width=barWidth, label='Adult',color='purple')
	Middleage = ax.bar(r2,Middleage_mean, width=barWidth, label='Middelage',color='indigo')
	SeniorAdult= ax.bar(r3, SeniorAdult_mean,width=barWidth, label='Senior Adult',color='navy')

	# inserting x axis label
	plt.xticks([r + barWidth for r in range(0,len(Purchases)*2,2)], dataset)
	ax.set_xticklabels(Purchases)

	# inserting y axis label
	ax.set_yticks(score_label)
	ax.set_yticklabels(score_label)

	# inserting legend
	ax.legend()

	st.write(ax.figure)

	st.subheader("Products Amount Purchased per Age Group")

	Products = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
	dataset = marketing_df.groupby('AgeGroup')[Products].mean()
	score_label = np.arange(0, 500, 50)
	Adult_mean  = list(dataset.T['Adult'])
	Middleage_mean  = list(dataset.T['Middle Age Adult'])
	SeniorAdult_mean  = list(dataset.T['Senior Adult'])
	# set width of bar
	barWidth = 0.35

	fig, ax = plt.subplots(figsize=(19,8))

	# Set position of bar on X axis
	r1 = np.arange(0,len(Products)*2,2)
	r2 = [x + barWidth for x in r1]
	r3 = [x + barWidth for x in r2]
	# Make the plot

	Adult = ax.bar(r1, Adult_mean, width=barWidth, label='Adult',color='purple')
	Middleage = ax.bar(r2,Middleage_mean, width=barWidth, label='Middelage',color='indigo')
	SeniorAdult= ax.bar(r3, SeniorAdult_mean,width=barWidth, label='Senior Adult',color='navy')


	# inserting x axis label
	plt.xticks([r + barWidth for r in range(0,len(Products)*2,2)], dataset)
	ax.set_xticklabels(Products)

	# inserting y axis label
	ax.set_yticks(score_label)
	ax.set_yticklabels(score_label)

	# inserting legend
	ax.legend()
	st.write(ax.figure)

	st.subheader("Total Purchases vs Income")
	fig,ax=plt.subplots(figsize=(18,8))
	fig=sns.scatterplot(data=marketing_df,x='Income', y='TotalPurchases',ax=ax,hue='AgeGroup',style="AgeGroup",palette='rocket')
	st.write(fig.figure)

	st.subheader("Income vs Quantity of Products Purchased")
	f,ax=plt.subplots(3,2,figsize=(18,17))
	#f=sns.scatterplot(data=marketing_df, x='Income', y='MntWines', hue='AgeGroup',markers=["o", "s", "D"],ax=ax[0][0],palette='rocket')
	f=sns.scatterplot(data=marketing_df, x='Income', y='MntWines', hue='AgeGroup',style="AgeGroup",ax=ax[0][0],palette="rocket")
	f=sns.scatterplot(data=marketing_df, x='Income', y='MntFruits', hue='AgeGroup',style="AgeGroup",ax=ax[0][1],palette="rocket")
	f=sns.scatterplot(data=marketing_df, x='Income', y='MntMeatProducts', hue='AgeGroup',style="AgeGroup",ax=ax[1][0],palette="rocket")
	f=sns.scatterplot(data=marketing_df, x='Income', y='MntSweetProducts', hue='AgeGroup',style="AgeGroup",ax=ax[1][1],palette="rocket")
	f=sns.scatterplot(data=marketing_df, x='Income', y='MntGoldProds', hue='AgeGroup',style="AgeGroup",ax=ax[2][0], palette="rocket")
	f=sns.scatterplot(data=marketing_df, x='Income', y='MntFishProducts', hue='AgeGroup',style="AgeGroup",ax=ax[2][1], palette="rocket")
	st.write(f.figure)
if options=="Statistical Analysis":

	#Total kids
	marketing_df['Totalkids'] = marketing_df['Kidhome'] + marketing_df['Teenhome']
	marketing_df['YearCustomer'] = pd.DatetimeIndex(marketing_df['Dt_Customer']).year


	# total amount spent
	mnt_cols = [col for col in marketing_df.columns if 'Mnt' in col]
	marketing_df['TotalMnt'] = marketing_df[mnt_cols].sum(axis=1)

	# Total Purchases
	purchases_cols = [col for col in marketing_df.columns if 'Purchases' in col]
	marketing_df['TotalPurchases'] = marketing_df[purchases_cols].sum(axis=1)

	# Total Campaigns Accepted
	campaigns_cols = [col for col in marketing_df.columns if 'Cmp' in col] + ['Response'] 
	marketing_df['TotalCampaignsAcc'] = marketing_df[campaigns_cols].sum(axis=1)

	#age
	year=datetime.datetime.today().year
	marketing_df['Age']=year-marketing_df['Year_Birth']

	#Age_group
	bins= [18,39,59,90]
	labels = ['Adult','Middle Age Adult','Senior Adult']
	marketing_df['AgeGroup'] = pd.cut(marketing_df['Age'], bins=bins, labels=labels, right=False)
	marketing_df['AgeGroup'] = marketing_df['AgeGroup'].astype('object')


	st.subheader("Correlation between numeric variables on the store purchases")
	df_num = marketing_df.drop(columns=['ID']).select_dtypes(include = ['float64', 'int64'])
	plt.figure(figsize=(25,14))
	mask = np.triu(np.ones_like(df_num.corr(), dtype=np.bool))
	heatmap = sns.heatmap(df_num.corr(), mask=mask, vmin=-1, vmax=1, annot=True)
	heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=16);
	st.write(heatmap.figure)

	st.subheader("Correlation with NumStorePurchases")
	corr_with_SalePrice = df_num.corr()
	plot_data = corr_with_SalePrice["NumStorePurchases"].sort_values(ascending=True)
	plt.figure(figsize=(12,6))
	plot=plot_data.plot.bar(color='red')
	plt.title("Correlations with the  NumStorePurchases")
	#plt.show(plot.figure())
	st.write(plot.figure)
	st.markdown('We can see the correlation of numerical columns/decorations on NumStorePurchases. The columns that have clear correlation (high positive or high negative) are important for the prediction model, but few of those with small (about zero) correlation will not have much effect on the SalePrice')

	Data=marketing_df.drop(columns=['Response', 'Complain','Recency','Teenhome'])
	#Droping uninformative features
	Data=marketing_df.drop(columns=['ID','Dt_Customer'])

	st.subheader('NumStorePurchases variation on different categories of categorical variables/columns.')
	few_cat_variables = ['Education','Marital_Status','Country','AgeGroup']

	for i in range(len(few_cat_variables)):
		boxplot=sns.boxplot(x=few_cat_variables[i], y='NumStorePurchases', data=marketing_df,color='orange')
		st.write(boxplot.figure)

if options=='Predicting Purchases':

	st.subheader("Select Features of Client")
	#age
	year=datetime.datetime.today().year
	marketing_df['Age']=year-marketing_df['Year_Birth']

	#Age_group
	bins= [18,39,59,90]
	labels = ['Adult','Middle Age Adult','Senior Adult']
	marketing_df['AgeGroup'] = pd.cut(marketing_df['Age'], bins=bins, labels=labels, right=False)
	marketing_df['AgeGroup'] = marketing_df['AgeGroup'].astype('object')

	ms=marketing_df["Marital_Status"].value_counts()
	msnames=ms.index.tolist()
	msnames.remove('Alone')
	msnames.remove('YOLO')
	msnames.remove('Absurd')
	X_to_predict=[]
	marital_status=st.multiselect('Marital Status',msnames)
	
	ag=marketing_df["AgeGroup"].value_counts()
	agnames=ag.index.tolist()
	age_group= st.multiselect('Age Group',agnames)
	
	ed=marketing_df["Education"].value_counts()
	ed=ed.index.tolist()
	education=st.multiselect('Education',ed)

	income=st.slider("Income",1000,700000)
	
	print(marital_status,age_group,education,income)

	if marital_status[0]=='Married':
		X_to_predict.append(0)
	elif marital_status[0]=='Together':
		X_to_predict.append(1)
	elif marital_status[0]=='Single':
		X_to_predict.append(2)
	elif marital_status[0]=='Divorced':
		X_to_predict.append(3)
	elif marital_status[0]=='Widow':
		X_to_predict.append(4)

	if age_group[0]== 'Middle Age Adult':
		X_to_predict.append(0)
	elif age_group[0]=='Senior Adult':
		X_to_predict.append(1)
	elif age_group[0]=='Adult':
		X_to_predict.append(2)	

	if education[0] == 'Graduation':
		X_to_predict.append(0)
	elif education[0]=='PhD':
		X_to_predict.append(1)
	elif education[0]=='Master':
		X_to_predict.append(2)
	elif education[0]=='2n Cycle':
		X_to_predict.append(3)
	elif education[0]=='Basic':
		X_to_predict.append(4)
	X_to_predict.append(income/1000)
	print(X_to_predict)

	
	#Getting training data for regression model

	#1: NumStorePurchases
	Data=marketing_df
	X = Data[['Marital_Status','AgeGroup','Education','Income','NumStorePurchases']]
	#y = Data['NumStorePurchases']
	X =X[X.Marital_Status != "YOLO"]
	X =X[X.Marital_Status != "Alone"]
	X =X[X.Marital_Status != "Absurd"]
	X=X.dropna()
	y=X['NumStorePurchases']
	X=X[['Marital_Status','AgeGroup','Education','Income']]

	#replacing string by floats
	X['Marital_Status']=X['Marital_Status'].replace(['Married','Together','Single','Divorced','Widow'],[0,1,2,3,4])
	X['AgeGroup']=X['AgeGroup'].replace(['Middle Age Adult','Senior Adult','Adult'],[0,1,2])
	X['Education']=X['Education'].replace(['Graduation','PhD','Master','2n Cycle','Basic'],[0,1,2,3,4])
	X['Income']=X['Income']/1000
	#print(X.isnull().values.any())
	

	#Train, test split
	x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
	
	x_train = [x_train.iloc[i].tolist() for i in range(len(x_train))]
	X_test = [X_test.iloc[i].tolist() for i in range(len(X_test))]
	y_train = y_train.tolist()
	y_test = y_test.tolist()

	#Model: Gradient Boost Regressor
	gbr = GradientBoostingRegressor(n_estimators=175, learning_rate=0.08, max_depth=3, random_state=0, loss='ls')			
	
	gbr.fit(x_train,y_train)

	if st.button("Predict Number of Store Purchases") and len(X_to_predict)==4:
		st.write(gbr.predict(np.array(X_to_predict).reshape(1,-1)))
		X_to_predict=[]

	
	#1: NumStorePurchases
	Data=marketing_df
	X = Data[['Marital_Status','AgeGroup','Education','Income','NumWebPurchases']]
	#y = Data['NumStorePurchases']
	X =X[X.Marital_Status != "YOLO"]
	X =X[X.Marital_Status != "Alone"]
	X =X[X.Marital_Status != "Absurd"]
	X=X.dropna()
	y=X['NumWebPurchases']
	X=X[['Marital_Status','AgeGroup','Education','Income']]

	#replacing string by floats
	X['Marital_Status']=X['Marital_Status'].replace(['Married','Together','Single','Divorced','Widow'],[0,1,2,3,4])
	X['AgeGroup']=X['AgeGroup'].replace(['Middle Age Adult','Senior Adult','Adult'],[0,1,2])
	X['Education']=X['Education'].replace(['Graduation','PhD','Master','2n Cycle','Basic'],[0,1,2,3,4])
	X['Income']=X['Income']/1000
	#print(X.isnull().values.any())
	

	#Train, test split
	x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
	
	x_train = [x_train.iloc[i].tolist() for i in range(len(x_train))]
	X_test = [X_test.iloc[i].tolist() for i in range(len(X_test))]
	y_train = y_train.tolist()
	y_test = y_test.tolist()
	gbr = GradientBoostingRegressor(n_estimators=175, learning_rate=0.08, max_depth=3, random_state=0, loss='ls')			
	
	gbr.fit(x_train,y_train)

	if st.button("Predict Number of Web Purchases") and len(X_to_predict)==4:
		st.write(gbr.predict(np.array(X_to_predict).reshape(1,-1)))
		X_to_predict=[]
