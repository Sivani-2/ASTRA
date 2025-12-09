To run:
cd backend
then, uvicorn main:app --reload --port 8000  
Open index.html to see the website
Make sure you have uvicorn installed.
pip install uvicorn (or) python -m pip install uvicorn fastapi
We also need to create a .env and put the groq api key.

Requirements: 
fastapi
uvicorn[standard]
groq
PyPDF2
python-multipart