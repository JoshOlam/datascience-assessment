# Titanic Data Science Project

## Project Overview

This project analyzes the Titanic dataset to predict survival outcomes using a machine learning model. The project includes preprocessing of the dataset, model training using a Random Forest classifier, and a web application using Streamlit for visualization and interaction.

## Project Structure

Here’s a brief overview of the project's directory structure:

- `assets/imgs/`: Contains image files used in the app.
- `config/`: JSON configuration files defining model parameters and mappings.
- `data/`: Contains the Titanic dataset.
- `models/`: Serialized version of the trained model.
- `src/`: Python scripts for data preprocessing.
- `venv/`: Python virtual environment for dependencies.
- `app.py`: Python script to run the flask app.
- `streamlit_app.py`: Main Python script to run the Streamlit app.
- `Dockerfile`, `docker-compose.yml`: Docker files for containerization.
- `requirements.txt`: List of dependencies.

## Prerequisites

Before setting up the project, ensure you have the following installed:
- Python 3.10+
- Numpy 2.0.0
- Docker (for containerization)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/JoshOlam/skaletek-datascience-assessment.git
cd skaletek-datascience-assessment
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the `.env.example` file to a new file named `.env`, and adjust the variables as necessary.

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

#### Web UI
![Alt text](assets/imgs/image.png)

## Using Docker (Optional)

To deploy using Docker, ensure Docker and Docker Compose are installed and use the following commands:

### Build and Run Container

```bash
docker-compose up build
```

Or

```bash
docker compose up build
```

## Deployment

The app can be deployed using Docker or directly on a cloud platform like Heroku or AWS. Ensure environmental variables are set in the production environment.

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
