import flask
from flask import request
from predictor_api import get_class_info, get_class, make_prediction, feature_names, all_poses

# initialize the app
app = flask.Flask(__name__)


@app.route("/")
def build_yoga_class():
    return flask.render_template("predictor.html",
                                 feature_names=feature_names,
                                 all_poses=all_poses)


@app.route("/predict", methods=["POST", "GET"])
def predict():

    class_info = get_class_info(request.form)
    yoga_class = get_class(request.form)

    x_input, predictions = make_prediction(class_info)
    if request.method == 'POST':
        result = yoga_class
        result2 = class_info
        return flask.render_template("results.html",
                                     result=result,
                                     result2=result2,
                                     prediction=predictions,
                                     x_input=x_input)


# start the server, continuously listen to requests

# for local development:
if __name__ == "__main__":
    app.run(debug=True)

# for public web serving:
app.run(host="0.0.0.0")
