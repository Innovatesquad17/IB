from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load the CSV files
internships_df = pd.read_csv(r'C:\Users\Admin\Desktop\internship_data.csv', encoding='ISO-8859-1')
student_data_path = r'C:\Users\Admin\Desktop\Student_data.csv'

# Vectorizer for converting skills and location into numeric features
vectorizer = TfidfVectorizer()

# Train the model with internship data
def train_model():
    # Combine skills and location columns as features for the model
    internships_df['combined_features'] = internships_df['Skills Required'].fillna('') + ' ' + internships_df['Location'].fillna('')
    
    # Convert the combined text features into numerical data using TF-IDF
    X = vectorizer.fit_transform(internships_df['combined_features'])
    
    # Use KNN for finding nearest neighbors based on combined skills and location
    knn_model = NearestNeighbors(n_neighbors=5, metric='cosine').fit(X)
    
    return knn_model

# Initialize the model when the app starts
knn_model = train_model()

# Route for home (login page)
@app.route('/')
def home():
    return render_template('login.html')

# Route for register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get user input from form
        student_data = {
            'Name': request.form['name'],
            'Institution': request.form['institution'],
            'Category': request.form['category'],
            'Email': request.form['email'],
            'Contact Number': request.form['contact number'],
            'Gender': request.form['gender'],
            'Area of Interest': request.form['area of interest'],
            'Nationality': request.form['nationality'],
            'Physically Handicapped': request.form['physically handicapped'],
            'Academic Qualifications': request.form['academic qualifications'],
            'Preferred  Locations': request.form['preferred locations'],
            'Skills': request.form['skills'],
            'Languages Known': request.form['languages known'],
            'GitHub Link': request.form['github link'],
            'LinkedIn Link': request.form['linkedin link']
        }
        # Save the data to CSV
        df = pd.DataFrame([student_data])
        df.to_csv(student_data_path, mode='a', header=False, index=False)
        return redirect(url_for('home'))
    return render_template('register.html')

# Route for login page
@app.route('/login', methods=['POST'])
def login():
    name = request.form['name']
    # Here, you could add user authentication (e.g., check if name exists in CSV)
    return redirect(url_for('recommend'))

# Route for recommendation
@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        # Get user input for skills and location
        skills = request.form['skills'].lower()
        location = request.form['location'].lower()

        # Combine skills and location as features
        user_input = skills + ' ' + location
        
        # Transform the user input using the same vectorizer used for training
        user_input_vector = vectorizer.transform([user_input])
        
        # Find nearest internships using the KNN model
        distances, indices = knn_model.kneighbors(user_input_vector)

        # Retrieve the recommended internships
        recommended_internships = internships_df.iloc[indices[0]].to_dict(orient='records')
        
        return render_template('recommendation.html', internships=recommended_internships)
    
    return render_template('recommendation.html')

if __name__ == '__main__':
    app.run(debug=True)
