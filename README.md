<!-- One time thing -->
<!-- Open terminal by ctrl + ` or through the menu option(Three lines on the top left)-->

<!-- in the terminal enter this note do this once only when u just unzipped it into a new pc -->

cd backend
.venv\Scripts\Activate
<!-- You will  .venv is working when the is a (.venv) at the start-->
pip install -r requirements.txt
streamlit run app.py


<!-- To run it again after closing = -->
cd backend
.venv\Scripts\Activate
streamlit run app.py