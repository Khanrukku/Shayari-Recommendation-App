from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Sample data for shayari
shayari_data = {
    "romantic": [
        {"writer": "Mirza Ghalib", "text": "Hazaron khwahishen aisi ke har khwahish pe dam nikle, Bohat nikle mere armaan lekin phir bhi kam nikle."},
        {"writer": "Faiz Ahmed Faiz", "text": "Mujh se pehli si mohabbat mere mehboob na maang, Maine samjha tha ke tu hai to dard aur gham hain."},
    ],
    "sad": [
        {"writer": "Mirza Ghalib", "text": "Dil hi to hai na sang-o-khisht dard se bhar na aaye kyun, Royenge hum hazaar baar koi humein sataye kyun."},
        {"writer": "Jaun Elia", "text": "Shayad mujhe koi yaad karta hai, Magar yeh kaun hai jo saans bhi nahi leta."},
    ],
    "inspirational": [
        {"writer": "Allama Iqbal", "text": "Khudi ko kar buland itna ke har taqdeer se pehle, Khuda bande se khud poochhe bata teri raza kya hai."},
        {"writer": "Kaifi Azmi", "text": "Raat bhar ka hai mehman andhera, kiske roke ruka hai savera."},
    ]
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_shayari', methods=['POST'])
def get_shayari():
    category = request.form.get('category').lower()
    shayari_list = shayari_data.get(category, [])
    return jsonify(shayari_list)

if __name__ == '__main__':
    app.run(debug=True)
