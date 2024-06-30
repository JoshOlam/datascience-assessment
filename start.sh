#!/bin/bash
python app.py &
streamlit run streamlit_app.py --server.port=8501
