from src.deployment import app
from src import utils

if __name__ == '__main__':
    utils.set_seeds()
    app.flask_app.run(debug=True)