# Car Sales Forecasting and Optimization

## Project Overview
This project aims to build a robust forecasting model that helps businesses predict future car sales trends. The project is divided into four milestones, each focusing on a different aspect of the data science pipeline.

## Team Members
- Ahmed Gamal
- Begad
- Youssef Bassiouny
- Mostafa Nasser
- Mohammed Amr

## Project Structure
```
├── Milestone1/                    # Data Collection and Cleaning
│   ├── Car_sales_CleanData.csv    # Cleaned dataset
│   ├── Report exploration.doc     # Data exploration report
│   └── CarSales Data collection ,exploration ,cleaning.ipynb
│
├── MileStone2/                    # Data Analysis and Visualization
│   ├── Data Analysis and Visualization.ipynb
│   ├── DashBoard.ipynb
│   ├── Cars-sales-visual.pbix
│   └── milestone2.pdf
│
├── Milestone3/                    # Model Development
│   ├── Forecasting Model Development and Optimization.ipynb
│   ├── Random_Forest_Regressor.pkl
│   ├── income-model.ipynb
│   └── Forecasting Model Performance Report.docx
│
└── Milestone4/                    # Deployment
    ├── deployment/
    │   ├── backend/              # FastAPI backend
    │   │   ├── main.py
    │   │   ├── config.py
    │   │   ├── models/
    │   │   ├── services/
    │   │   └── routers/
    │   ├── frontend/            # Streamlit frontend
    │   │   ├── app.py
    │   │   ├── config.py
    │   │   ├── components/
    │   │   └── services/
    │   └── requirements.txt
    └── Deployments.ipynb
```

## Milestones

### Milestone 1: Data Collection and Cleaning
- Collected car sales data with features including:
  - Car details (make, model, engine, transmission, etc.)
  - Customer information (gender, annual income)
  - Sales information (price, date, dealer details)
- Performed data cleaning and preprocessing
- Created a clean dataset for analysis

### Milestone 2: Data Analysis and Visualization
- Conducted exploratory data analysis
- Created interactive dashboards
- Visualized sales trends and patterns
- Analyzed relationships between features
- Generated insights for business decisions

### Milestone 3: Model Development and Optimization
- Developed multiple forecasting models:
  - Random Forest Regressor
  - XGBoost
  - Decision Tree
- Performed feature engineering
- Optimized model parameters
- Evaluated model performance using:
  - R² (R-squared)
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Square Error)
- Selected Random Forest as the best performing model

### Milestone 4: Deployment
- Created a FastAPI backend with:
  - Model prediction endpoint
  - Input validation
  - Error handling
  - API documentation
- Developed a Streamlit frontend with:
  - User-friendly interface
  - Real-time predictions
  - Visual feedback
  - Error handling

## Features
- Car price prediction based on multiple features
- Interactive web interface
- Real-time predictions
- Comprehensive data analysis
- Model performance monitoring
- API documentation

## Technologies Used
- Python
- FastAPI
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Power BI

## Setup and Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r Milestone4/deployment/requirements.txt
```

4. Run the application:
```bash
# Start the backend
cd Milestone4/deployment/backend
python main.py

# Start the frontend (in a new terminal)
cd Milestone4/deployment/frontend
streamlit run app.py
```

## API Documentation
Once the backend is running, access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Frontend
The web interface is available at: http://localhost:8501

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
