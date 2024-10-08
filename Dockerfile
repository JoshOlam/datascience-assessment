# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

CMD python app.py & streamlit run streamlit_app.py --server.port=8501

# # Assuming you've copied the necessary files already
# COPY start.sh /start.sh
# RUN chmod +x /start.sh
# CMD ["/start.sh"]


# # Run app.py when the container launches
# CMD ["python", "app.py"]

# # Run app.py when the container launches
# CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501"]
