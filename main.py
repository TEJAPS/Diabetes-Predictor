from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

@app.route('/')
def home():
	# return "hello"
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	# df= pd.read_csv("YoutubeSpamMergedData.csv")
	# df_data = df[["CONTENT","CLASS"]]
	# # Features and Labels
	# df_x = df_data['CONTENT']
	# df_y = df_data.CLASS
 #    # Extract Feature With CountVectorizer
	# corpus = df_x
	# cv = CountVectorizer()
	# X = cv.fit_transform(corpus) # Fit the Data
	# from sklearn.model_selection import train_test_split
	# X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
	# #Naive Bayes Classifier
	# from sklearn.naive_bayes import MultinomialNB
	# clf = MultinomialNB()
	# clf.fit(X_train,y_train)
	# clf.score(X_test,y_test)

	#Alternative Usage of Saved Model
	ytb_model = open("class_tree.pkl","rb")
	clf = joblib.load(ytb_model)
	#OR
	#ytb_model="https://github.com/TEJAPS/Type_Of_Diabeties_Prediction/blob/master/class_tree.pkl"
	#clf= pd.read_csv(ytb_model)

	if request.method == 'POST':
		age = request.form['age']
		gender = request.form['gender']
		weight = request.form['weight']
		height = request.form['weight']
		# bp = request.form['bp']
		bp = 0
		bmi = request.form['bmi']
		fbs = request.form['fbs']
		ppbs = request.form['ppbs']
		hba1c = request.form['hba1c']
		tsh = request.form['tsh']
		screat = request.form['screat']
		data = [age,gender,weight,height,bp,bmi,fbs,ppbs,hba1c,tsh,screat]
		# vect = cv.transform(data).toarray()
		my_prediction = clf.predict([data])
	return render_template('result.html',prediction = my_prediction[0])



if __name__ == '__main__':
	app.run(debug=True)
	#app.run(host = '0.0.0.0')
