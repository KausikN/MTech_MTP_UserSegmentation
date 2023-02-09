#Base Image to use
FROM python:3.9.16-slim

#Expose port
EXPOSE 7860

#Optional - install git to fetch packages directly from github
RUN apt-get update && apt-get install -y git

#Copy Requirements.txt file into app directory
COPY requirements.txt app/requirements.txt

#install all requirements in requirements.txt
RUN pip install -r app/requirements.txt

#Copy all files in UserSegmentation directory into app directory
COPY . /app

#Change Working Directory to app directory
WORKDIR /app

#Run the application on port 7860
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]