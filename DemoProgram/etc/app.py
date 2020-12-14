from flask import Flask, jsonify
import json
import sys
# global variables
num = 0
last_choice = 'empty'
questionnaire_key = ''
user_choice = []
data = {}
app = Flask(__name__)
with open('static/data.json') as f:
    data = json.load(f)
print(data, file=sys.stdout)
@app.route('/<int:index>/Start')
def StartQuestionnaire(index):
    global num, last_choice, questionnaire_key, user_choice
    num = 0
    last_choice = 'empty'
    user_choice.clear()
    questionnaire_key = 'questionnaire_' + str(index)
    user_choice.append(data[questionnaire_key][0]['question'])
    print(user_choice, file=sys.stdout)
    return jsonify(data[questionnaire_key][0])
# last selected option will be passed as keyword argument
@app.route('/<int:index>/<string:option>')
def GetQuestion(index, option):
    global num, last_choice, questionnaire_key
    num = num + 1
    response = {}
    user_choice.append(option)
    if last_choice != 'empty':
        response = data[questionnaire_key][num][last_choice][option]
    else:
        if option != 'Yes' and option != 'No':
            last_choice = option
        response = data[questionnaire_key][num][option]
    if option == 'No' or num == len(data[questionnaire_key]) - 1:
        for elem in user_choice:
            print(elem, file=sys.stdout)
    return jsonify(response)
if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)