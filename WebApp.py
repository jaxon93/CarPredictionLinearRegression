import gradio as gr
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

#Dictionaries to use for the data
asp = {
    'Standard':'std',
   'Turbo':'turbo'
}

drivew = {
    'Rear wheel drive': 'rwd',
    'Front wheel drive': 'fwd', 
    '4 wheel drive': '4wd'
}

cylnum = {
    2: 'two',
    3: 'three', 
    4: 'four',
    5: 'five', 
    6: 'six', 
    8: 'eight',
    12: 'twelve'
}

#Columns of the df
df = pd.read_csv('df_columns')
df.drop(['Unnamed: 0','price'], axis = 1, inplace=True)
cols = df.columns

#Dummy columns of the dummy df
dummy = pd.read_csv('dummy_df')
dummy.drop('Unnamed: 0', axis = 1, inplace=True)
cols_to_use = dummy.columns

#Create the values in the app
# Capitalizing first letter of cars
cars = df['CarName'].unique().tolist()
carNameCap = []
for col in cars:
    carNameCap.append(col.capitalize())

#For fuel
fuel = df['fueltype'].unique().tolist()
fuelCap = []
for fu in fuel:
    fuelCap.append(fu.capitalize())

#For carbod, engine type, fuel systme
carb = df['carbody'].unique().tolist() 
engtype = df['enginetype'].unique().tolist()
fuelsys = df['fuelsystem'].unique().tolist()

#Function to model data to fit the model
def transform(data):
    #Scale the data
    sc= StandardScaler()

    #Import the model
    lasso_reg = pickle.load(open('model.pkl','rb'))

    #Dataframe with the new data
    new_df = pd.DataFrame([data],columns = cols)

    #Splitting categorical vs numerical columns
    cat = []
    num = []
    for col in new_df.columns: 
        if new_df[col].dtypes == 'object': 
            cat.append(col)
        else: 
            num.append(col)

    #Creating the values to feed the model
    x1_new = pd.get_dummies(new_df[cat], drop_first = False)
    x2_new = new_df[num]
    X_new = pd.concat([x2_new,x1_new], axis = 1)
    
    final_df = pd.DataFrame(columns = cols_to_use)
    final_df = pd.concat([final_df, X_new])
    final_df = final_df.fillna(0)
    final_df = pd.concat([final_df,dummy])

    X_new = final_df.values
    X_new[:, :(len(x1_new.columns))]= sc.fit_transform(X_new[:, :(len(x1_new.columns))])
    print(X_new[-1].reshape(-1, 1))
    output = lasso_reg.predict(X_new[-1].reshape(1, -1))
    return "The price of the car " + str(round(np.exp(output)[0],2)) + "$"

#Main function to predict price
def predict_price(car, fueltype, aspiration, doornumber, carbody, drivewheel, enginelocation, wheelbase, carlength, carwidth, 
                carheight, curbweight, enginetype, cylindernumber, enginesize, fuelsystem, boreratio, horsepower, citympg, highwaympg): 

    new_data = [car.lower(), fueltype.lower(), asp[aspiration], doornumber.lower(), carbody, drivew[drivewheel], enginelocation.lower(),
                wheelbase, carlength, carwidth, carheight, curbweight, enginetype, cylnum[cylindernumber], enginesize, fuelsystem, 
                boreratio, horsepower, citympg, highwaympg]
    
    return transform(new_data) 


car = gr.Dropdown(label = "Car brand", choices=carNameCap)

fueltype = gr.Radio(label = "Fuel Type", choices = fuelCap)

aspiration = gr.Radio(label = "Aspiration type", choices = ["Standard", "Turbo"])

doornumber = gr.Radio(label = "Number of doors", choices = ["Two", "Four"])

carbody = gr.Dropdown(label ="Car body type", choices = carb)

drivewheel = gr.Radio(label = "Drive wheel", choices = ['Rear wheel drive', 'Front wheel drive', '4 wheel drive'])

enginelocation = gr.Radio(label = "Engine location", choices = ['Front', 'Rear'])

wheelbase = gr.Slider(label = "Distance between the wheels on the side of the car (in inches)", minimum = 50, maximum = 300)

carlength = gr.Slider(label = "Length of the car (in inches)", minimum = 50, maximum = 300)

carwidth = gr.Slider(label = "Width of the car (in inches)", minimum = 50, maximum = 300)

carheight = gr.Slider(label = "Height of the car (in inches)", minimum = 50, maximum = 300)

curbweight = gr.Slider(label = "Weight of the car (in pounds)", minimum = 500, maximum = 6000)

enginetype = gr.Dropdown(label = "Engine type", choices = engtype)

cylindernumber = gr.Radio(label = "Cylinder number", choices = [2, 3, 4, 5, 6, 8, 12])

enginesize = gr.Slider(label = "Engine size (swept volume of all the pistons inside the cylinders)", minimum = 50, maximum = 500)

fuelsystem = gr.Dropdown(label = "Fuel system (link to ressource: ", choices = fuelsys)

boreratio = gr.Slider(label = "Bore ratio (ratio between cylinder bore diameter and piston stroke)", minimum = 1, maximum = 6)

horsepower = gr.Slider(label = "Horse power of the car", minimum = 25, maximum = 400)

citympg = gr.Slider(label = "Mileage in city (in km)", minimum = 0, maximum = 100)

highwaympg = gr.Slider(label = "Mileage on highway (in km)", minimum = 0, maximum = 100)

Output = gr.Textbox()

app = gr.Interface(title="Predict the price of a car based on its specs", 
                    fn=predict_price,
                    inputs=[car,
                            fueltype,
                            aspiration,
                            doornumber,
                            carbody,
                            drivewheel, 
                            enginelocation, 
                            wheelbase,
                            carlength, 
                            carwidth, 
                            carheight, 
                            curbweight,
                            enginetype, 
                            cylindernumber, 
                            enginesize,
                            fuelsystem,
                            boreratio,
                            horsepower, 
                            citympg, 
                            highwaympg
                            ],
                    outputs=Output)

app.launch()