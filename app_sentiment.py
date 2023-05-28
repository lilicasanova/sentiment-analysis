from flask import Flask, request, jsonify
import os
import pickle
import pandas as pd
import sqlite3


os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API de sentiment analysis"

#1 - Endpoint para mostrar la tabla de tweets de mi tuiterdb.db

@app.route('/tweets', methods=['GET'])
def get_tweets():
    conn = sqlite3.connect('tuiterdb.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM tweets')
    tweets = cursor.fetchall()
    conn.close()
    return jsonify(tweets)

#2 - Endpoint para mostrar la tabla de users de mi tuiterdb.db
@app.route('/usuarios', methods=['GET'])
def get_users():
    conn = sqlite3.connect('tuiterdb.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM usuarios')
    users = cursor.fetchall()
    conn.close()
    return jsonify(users)

#3 - Endpoint para predecir con mi modelo sobre las bbdd
@app.route('/predict', methods=['GET'])
def analyze_tweets():
    conn = sqlite3.connect('tuiterdb.db')
    cursor = conn.cursor()
    cursor.execute('SELECT contenido FROM tweets')
    tweets = cursor.fetchall()
    conn.close()

    #cargamos el modelo y aplicamos predicciones
    model_path = 'sentiment_model.pkl'
    model = pickle.load(open(model_path, 'rb'))
    predictions = model.predict(tweets)

    #hacemos balance de sentimiento negativo vs positivo
    positive_count = sum(predictions == 1)
    negative_count = sum(predictions == 0)

    return jsonify({
        'tweets positivos': positive_count,
        'tweets negativos': negative_count
    })

#4 - Endpoint para aplicar el modelo a tweets proporcionados por el usuario
@app.route('/analyze-tweets', methods=['POST'])
def analyze_user_tweets():
    data = request.get_json()
    account = data['nombre_usuario']
    date = data['fecha_publicacion']

    #obtenemos los tweets del usuario desde nuestra base de datos
    conn = sqlite3.connect('tuiterdb.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT t.contenido 
        FROM tweets AS t 
        JOIN usuarios AS u ON t.usuario_id = u.id 
        WHERE u.account = ? AND t.date BETWEEN ? AND ?
    """, (account, date))
    user_tweets = cursor.fetchall()
    conn.close()

     #verificamos que haya tweets disponibles
    if len(user_tweets) == 0:
        return jsonify({
            'error': 'No hay tweets disponibles para el usuario en la fecha proporcionada.'
        })

    #cargamos el modelo
    model_path = 'sentiment_model.pkl' 
    model = pickle.load(open(model_path, 'rb'))

   #aplicamos nuestro modelo al tweet proporcionado
    prediction = model.predict([user_tweets[0]])

    return jsonify({
        'El sentimiento del tweet proporcionado es': 'positivo' if prediction[0] == 1 else 'negativo'
    })

if __name__ == '__main__':
    app.run()