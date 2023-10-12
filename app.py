from flask import Flask
from views import views

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)
app.register_blueprint(views, url_prefix="/")

# main driver function
if __name__ == '__main__':
	app.run(debug=True, port=5000)
