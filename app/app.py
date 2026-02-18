from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fake News Detector</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                background-color: #f5f5f0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            }
            nav {
                background-color: #333;
                padding: 1rem 2rem;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            nav h1 {
                color: white;
                font-size: 1.5rem;
                font-weight: 600;
            }
        </style>
    </head>
    <body>
        <nav>
            <h1>Fake News Detector</h1>
        </nav>
        <div style="padding: 2rem;">
            <h2>Welcome to the Fake News Detector!</h2>
            <p>This application uses machine learning to analyze news articles and determine their authenticity. Stay informed and be cautious of misinformation.</p>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)