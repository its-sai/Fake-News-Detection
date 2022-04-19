
from glob import escape
from flask import Flask ,escape, request ,render_template
import pickle


feature_extraction = pickle.load(open("tfidf.pkl",'rb'))

vector=pickle.load(open("vectorizer.pkl",'rb'))

model=pickle.load(open("finalized_model.pkl",'rb'))

app = Flask(__name__ , template_folder='templates')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction",methods=['GET','POST'])
def prediction():
    author=""
    title=""
    res=""
    if request.method=="POST":
        author=str(request.form['author'])
        title=str(request.form['title'])
         
        prediction = model.predict(vector.transform([author,title]))
        print (prediction)
        if prediction[0]==0:
            res="Real"
        else:
            res="fake"
        
        return render_template("prediction.html",prediction_text=res)
    else:
        return render_template("prediction.html")

if __name__=='__main__':
  app.run()