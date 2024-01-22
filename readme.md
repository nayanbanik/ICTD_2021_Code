# Run Locally 

1. Download and install Miniconda (Version 4.10.3 Python 3.8) from the link: https://docs.conda.io/en/latest/miniconda.html
2. Open Anaconda Prompt (miniconda3) from Start Menu
3. Change the directory to current folder. Type: `cd /to/current/directory/`
4. Create a new virtual environment with required packages and dependencies. Type: `conda env create --file environment.yml`
5. Activate the newly created virtual environment. Type: `activate ictd`
6. Download the 'model.pt' file from here: https://drive.google.com/file/d/1-MM02HiM7E68_RAo1rPw1INXSxbve37x/view?usp=sharing and put it in *./static/assets* folder
7. Set the flask server in development mode. Type: `set FLASK_ENV=development`
8. Start the flask server. Type: `flask run`
9. Keep the window running and open a browser. Visit: http://127.0.0.1:5000/
10. The site will run. Input a news body on the box and click Generate Headline button

# Run Codespaces

1. Go to https://github.com/codespaces and create codespaces with this repository.
2. In the terminal type `pip install flask torchtext==0.6.0`
3. Upload the downloaded 'model.pt' file in the *./static/assets* folder
4. Set the flask server in development mode. Type: `set FLASK_ENV=development`
5. Start the flask server. Type: `flask run`
6. From the terminal end, click http://127.0.0.1:5000/ and it will launch the app in new window
7. Input a news body on the box and click Generate Headline button


