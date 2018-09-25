import dash
import dash_core_components as dcc
import dash_html_components as html
from dash import Dash
from dash.dependencies import Input, Output

import plotly.plotly as py
from plotly.graph_objs import *

from sklearn import datasets
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd

from config import user, api_key

py.sign_in(user, api_key)

#intialize the dash app

app = Dash(__name__)

application = app.server

# load the iris dataset
iris = datasets.load_iris()

# values for our dropdown
dropdown_val = list(iris.feature_names)

# define color palette
colors = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"]

###############################################################################
# build out the HTML

# begin making html page
app.layout = html.Div(className="container", style={"padding": "10px"}, children=[

				html.Div(className="jumbotron text-center", children=[
					html.H1("Iris Analysis"),
					html.P("Select the X and Y to visualize the data"),
					html.P("Use the slider to choose the number of clusters")

				]),

				dcc.Dropdown(className="col-md-4", style={"margin-bottom": "10px"}, id="dropdown_x",
					options=[
					{'label': val, 'value': val} for val in dropdown_val
					],
					value=dropdown_val[0]
				),

				dcc.Dropdown(className="col-md-4", id="dropdown_y",
					options=[
					{'label': val, 'value': val} for val in dropdown_val
					],
					value=dropdown_val[1]
				),

				html.Br(),

				dcc.Slider(id="slider_n",
				    min=1,
				    max=9,
				    marks={i: '{}'.format(i) for i in range(1, 10)},
				    value=3,
				),
				
				html.Br(),

				html.Div(style={"padding": "20px"} ,children=[
					dcc.Graph(id="cluster")
				])
			])

# import external css
app.css.append_css({"external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css"})

# import external javascript
app.scripts.append_script({"external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"})

##################################################
#                                                #
#       Callback Functions to update plots       #
#                                                #
##################################################
@app.callback(Output('cluster', 'figure'), [Input('dropdown_x', 'value'), Input('dropdown_y', 'value'), Input('slider_n', 'value')])
def update_graph(x_val, y_val, n):
	# build our dataframe
	df = pd.DataFrame(iris.data, columns=iris.feature_names)
	df['target'] = iris.target

	###############################################################################
	# prepare for modeling
	# assign your x and y values
	X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]

	# initialize kmeans algorithm
	kmeans = KMeans(n_clusters = n)

	# fit data to X
	kmeans.fit(X)

	# build our resutls dataframe
	df["predicted_classes"] = kmeans.labels_

	# count number of clusters
	num_of_clusters = df["predicted_classes"].nunique()
	# create empty data list to store traces
	data = []
	# plot the actual labels
	for i in range(num_of_clusters):
	    # split up the clusters to visualize and extract sepal length and width
	    cluster_df = df[df["predicted_classes"] == i]
	    data.append({
	                "x": cluster_df[x_val],
	                "y": cluster_df[y_val],
	                "type": "scatter",
	                "name": f"class_{i}",
	                "mode": "markers",
	                "marker": dict(
	                    color = colors[i],
	                    size = 10
	                )
	            })
	    
	layout = {
	  "hovermode": "closest", 
	  "margin": {
	    "r": 10, 
	    "t": 25, 
	    "b": 40, 
	    "l": 60
	  }, 
	  "title": f"Iris Dataset - {x_val} vs {y_val}", 
	  "xaxis": {
	    "domain": [0, 1], 
	    "title": f"{x_val}"
	  }, 
	  "yaxis": {
	    "domain": [0, 1], 
	    "title": f"{y_val}"
	  }
	}

	fig = {"data":data, "layout": layout}
	return fig

if __name__ == '__main__':
    application.run()