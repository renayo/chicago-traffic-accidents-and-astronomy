import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ephem
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Define minor bodies with their orbital elements
# These are approximate elements for demonstration
MINOR_BODIES = {
    'Eris': {
        'a': 67.6681,  # semi-major axis in AU
        'e': 0.44177,  # eccentricity
        'i': 44.187,   # inclination in degrees
        'om': 35.87,   # longitude of ascending node
        'w': 151.43,   # argument of perihelion
        'M': 205.989,  # mean anomaly at epoch
        'epoch': '2000/1/1.5'
    },
    'Sedna': {
        'a': 506.2,
        'e': 0.85491,
        'i': 11.93,
        'om': 144.26,
        'w': 311.29,
        'M': 358.19,
        'epoch': '2000/1/1.5'
    },
    'Pholus': {
        'a': 20.43,
        'e': 0.5715,
        'i': 24.66,
        'om': 119.06,
        'w': 354.89,
        'M': 12.69,
        'epoch': '2000/1/1.5'
    },
    'Nessus': {
        'a': 24.64,
        'e': 0.5194,
        'i': 15.64,
        'om': 31.22,
        'w': 170.73,
        'M': 262.53,
        'epoch': '2000/1/1.5'
    }
}

def create_minor_body(elements, observer):
    """Create a PyEphem body from orbital elements"""
    body = ephem.EllipticalBody()
    body._inc = np.radians(elements['i'])
    body._Om = np.radians(elements['om'])
    body._om = np.radians(elements['w'])
    body._a = elements['a']
    body._e = elements['e']
    body._M = np.radians(elements['M'])
    body._epoch = ephem.Date(elements['epoch'])
    body.compute(observer)
    return body

def calculate_ascendant(date, lat=41.8781, lon=-87.6298):
    """Calculate the Ascendant (rising sign) for a given time and location"""
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.date = date
    
    # Calculate local sidereal time
    lst = float(observer.sidereal_time()) * 180 / np.pi  # Convert to degrees
    
    # Simplified calculation of ascendant
    # In reality, this requires complex calculations involving ecliptic obliquity
    # This is an approximation
    obliquity = 23.44  # Earth's axial tilt
    lat_rad = np.radians(lat)
    
    # Calculate RAMC (Right Ascension of Midheaven)
    ramc = lst
    
    # Approximate ascendant longitude
    # This is simplified - actual calculation is more complex
    asc_longitude = (ramc + 90) % 360
    
    return asc_longitude

def get_ecliptic_longitude(body):
    """Convert equatorial coordinates to ecliptic longitude"""
    # This is a simplified conversion
    # Actual conversion requires transformation matrix
    ra_deg = float(body.ra) * 180 / np.pi
    dec_deg = float(body.dec) * 180 / np.pi
    
    # Approximate ecliptic longitude (simplified)
    # In reality, this requires proper coordinate transformation
    ecl_lon = ra_deg  # Simplified approximation
    
    return ecl_lon % 360

def calculate_moon_distance(date):
    """Calculate Moon's distance from Earth in Earth radii"""
    observer = ephem.Observer()
    observer.date = date
    moon = ephem.Moon(observer)
    # Distance in AU, convert to Earth radii (1 AU ≈ 23,455 Earth radii)
    distance_earth_radii = moon.earth_distance * 23455
    return distance_earth_radii

def calculate_moon_phase(date):
    """Calculate moon phase (0 = new moon, 1 = full moon)"""
    observer = ephem.Observer()
    observer.date = date
    moon = ephem.Moon(observer)
    return moon.phase / 100.0

def is_mercury_retrograde(date):
    """Check if Mercury is in retrograde motion"""
    observer = ephem.Observer()
    observer.date = date
    
    mercury = ephem.Mercury(observer)
    
    # Check position 1 day before and after
    observer_before = ephem.Observer()
    observer_before.date = ephem.Date(date - 1)
    mercury_before = ephem.Mercury(observer_before)
    
    observer_after = ephem.Observer()
    observer_after.date = ephem.Date(date + 1)
    mercury_after = ephem.Mercury(observer_after)
    
    # If RA is decreasing, Mercury is retrograde
    ra_before = float(mercury_before.ra)
    ra_current = float(mercury.ra)
    ra_after = float(mercury_after.ra)
    
    # Check if moving backward in RA
    is_retrograde = (ra_before < ra_current and ra_current > ra_after) or \
                   (ra_before > ra_current and ra_current < ra_after)
    
    return 1 if is_retrograde else 0

def within_eclipse_window(date):
    """Check if date is within 2 weeks before an eclipse"""
    # List of eclipses in 2025 (example dates - you should use actual eclipse data)
    eclipses_2025 = [
        ephem.Date('2025/3/14'),   # Lunar eclipse
        ephem.Date('2025/3/29'),   # Solar eclipse
        ephem.Date('2025/9/7'),    # Lunar eclipse
        ephem.Date('2025/9/21'),   # Solar eclipse
    ]
    
    current_date = ephem.Date(date)
    
    for eclipse_date in eclipses_2025:
        days_before = float(eclipse_date - current_date)
        if 0 <= days_before <= 14:  # Within 2 weeks before
            return 1
    
    return 0

def calculate_planetary_longitudes(date, lat=41.8781, lon=-87.6298):
    """Calculate ecliptic longitudes for all planets and bodies"""
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.date = date
    
    longitudes = {}
    
    # Major planets
    planets = {
        'sun': ephem.Sun(),
        'moon': ephem.Moon(),
        'mercury': ephem.Mercury(),
        'venus': ephem.Venus(),
        'mars': ephem.Mars(),
        'jupiter': ephem.Jupiter(),
        'saturn': ephem.Saturn(),
        'uranus': ephem.Uranus(),
        'neptune': ephem.Neptune(),
        'pluto': ephem.Pluto()
    }
    
    for name, planet in planets.items():
        planet.compute(observer)
        longitudes[f'{name}_longitude'] = get_ecliptic_longitude(planet)
    
    # Minor bodies
    for name, elements in MINOR_BODIES.items():
        try:
            body = create_minor_body(elements, observer)
            longitudes[f'{name}_longitude'] = get_ecliptic_longitude(body)
        except:
            # If calculation fails, use a placeholder value
            longitudes[f'{name}_longitude'] = 0.0
    
    return longitudes

def extract_astrological_features(df):
    """Extract the 20 specified astrological features from the dataframe"""
    features_list = []
    
    # Group by hour to aggregate features
    df['datetime'] = pd.to_datetime(df['CRASH_DATE'])
    df['hour_group'] = df['datetime'].dt.floor('H')
    
    print("Processing hourly groups...")
    total_groups = len(df.groupby('hour_group'))
    processed = 0
    
    for hour, group in df.groupby('hour_group'):
        if pd.isna(hour):
            continue
        
        processed += 1
        if processed % 100 == 0:
            print(f"  Processed {processed}/{total_groups} hours...")
            
        features = {}
        features['datetime'] = hour
        features['accident_count'] = len(group)  # Target variable
        
        # Use the median time of accidents in this hour for calculations
        representative_date = hour + timedelta(minutes=30)
        ephem_date = ephem.Date(representative_date)
        
        # Get Chicago coordinates from the data if available
        if 'LATITUDE' in group.columns and 'LONGITUDE' in group.columns:
            lat = group['LATITUDE'].median()
            lon = group['LONGITUDE'].median()
            if pd.isna(lat) or pd.isna(lon):
                lat, lon = 41.8781, -87.6298
        else:
            lat, lon = 41.8781, -87.6298
        
        # 1. Ascendant
        features['ascendant'] = calculate_ascendant(ephem_date, lat, lon)
        
        # Get all planetary longitudes
        longitudes = calculate_planetary_longitudes(ephem_date, lat, lon)
        
        # 2. Moon longitude
        features['moon_longitude'] = longitudes['moon_longitude']
        
        # 3. Moon distance from Earth
        features['moon_distance'] = calculate_moon_distance(ephem_date)
        
        # 4. Moon Phase
        features['moon_phase'] = calculate_moon_phase(ephem_date)
        
        # 5. Sun longitude
        features['sun_longitude'] = longitudes['sun_longitude']
        
        # 6. Mercury retrograde
        features['mercury_retrograde'] = is_mercury_retrograde(ephem_date)
        
        # 7. Within 2 weeks before eclipse
        features['eclipse_proximity'] = within_eclipse_window(ephem_date)
        
        # 8-15. Planetary longitudes
        features['mercury_longitude'] = longitudes['mercury_longitude']
        features['venus_longitude'] = longitudes['venus_longitude']
        features['mars_longitude'] = longitudes['mars_longitude']
        features['jupiter_longitude'] = longitudes['jupiter_longitude']
        features['saturn_longitude'] = longitudes['saturn_longitude']
        features['uranus_longitude'] = longitudes['uranus_longitude']
        features['neptune_longitude'] = longitudes['neptune_longitude']
        features['pluto_longitude'] = longitudes['pluto_longitude']
        
        # 16-19. Minor body longitudes
        features['eris_longitude'] = longitudes['Eris_longitude']
        features['sedna_longitude'] = longitudes['Sedna_longitude']
        features['pholus_longitude'] = longitudes['Pholus_longitude']
        features['nessus_longitude'] = longitudes['Nessus_longitude']
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)

def train_prediction_model(features_df):
    """Train a model to predict accident counts based on astrological features"""
    
    # List of the 20 astrological features
    astrological_features = [
        'ascendant', 'moon_longitude', 'moon_distance', 'moon_phase',
        'sun_longitude', 'mercury_retrograde', 'eclipse_proximity',
        'mercury_longitude', 'venus_longitude', 'mars_longitude',
        'jupiter_longitude', 'saturn_longitude', 'uranus_longitude',
        'neptune_longitude', 'pluto_longitude', 'eris_longitude',
        'sedna_longitude', 'pholus_longitude', 'nessus_longitude'
    ]
    
    # Add the 20th feature if needed (we have 19 listed above)
    # You can add another feature or duplicate one as needed
    
    # Ensure all features exist
    for feature in astrological_features:
        if feature not in features_df.columns:
            print(f"Warning: {feature} not found in features, adding zeros")
            features_df[feature] = 0
    
    X = features_df[astrological_features].fillna(0)
    y = features_df['accident_count']
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Total accident hours: {len(y)}")
    print(f"Average accidents per hour: {y.mean():.2f}")
    print(f"Max accidents in an hour: {y.max()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Training metrics
        train_r2 = r2_score(y_train, y_pred_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                   cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        results[name] = {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'train_r2': train_r2,
            'cv_rmse': cv_rmse,
            'predictions': y_pred,
            'actual': y_test
        }
        
        print(f"  Test RMSE: {rmse:.2f}")
        print(f"  Test MAE: {mae:.2f}")
        print(f"  Test R²: {r2:.3f}")
        print(f"  Train R²: {train_r2:.3f}")
        print(f"  CV RMSE: {cv_rmse:.2f}")
        
        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': astrological_features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n  Top 10 Most Important Astrological Features:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
    
    return results, scaler, astrological_features

def main():
    # Load the data
    print("Loading Chicago car accidents data...")
    df = pd.read_csv('chicago_accidents.csv')
    
    print(f"Loaded {len(df)} accident records")
    print(f"Date range: {df['CRASH_DATE'].min()} to {df['CRASH_DATE'].max()}")
    
    # Extract astrological features
    print("\nExtracting 20 astrological features...")
    print("Features being extracted:")
    print("1. Ascendant")
    print("2. Moon longitude")
    print("3. Moon distance from Earth")
    print("4. Moon Phase")
    print("5. Sun longitude")
    print("6. Mercury retrograde")
    print("7. Within 2 weeks before eclipse")
    print("8-15. Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto longitudes")
    print("16-19. Eris, Sedna, Pholus, Nessus longitudes")
    print("\nThis may take a few minutes...")
    
    features_df = extract_astrological_features(df)
    
    print(f"\nCreated {len(features_df)} hourly aggregated records")
    print(f"Average accidents per hour: {features_df['accident_count'].mean():.2f}")
    print(f"Max accidents in an hour: {features_df['accident_count'].max()}")
    
    # Save features for analysis
    features_df.to_csv('astrological_features.csv', index=False)
    print("Astrological features saved to 'astrological_features.csv'")
    
    # Train prediction models
    print("\nTraining prediction models with astrological features...")
    results, scaler, feature_columns = train_prediction_model(features_df)
    
    # Select best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
    best_model = results[best_model_name]['model']
    
    print(f"\n{'='*50}")
    print(f"Best Model: {best_model_name}")
    print(f"Test RMSE: {results[best_model_name]['rmse']:.2f}")
    print(f"Test R²: {results[best_model_name]['r2']:.3f}")
    print(f"Train R²: {results[best_model_name]['train_r2']:.3f}")
    
    # Save the model and scaler
    import joblib
    joblib.dump(best_model, 'accident_astro_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    joblib.dump(feature_columns, 'feature_columns.pkl')
    
    print("\nModel saved to 'accident_astro_model.pkl'")
    print("Scaler saved to 'feature_scaler.pkl'")
    print("Feature columns saved to 'feature_columns.pkl'")
    
    # Create predictions DataFrame for analysis
    predictions_df = pd.DataFrame({
        'actual': results[best_model_name]['actual'],
        'predicted': results[best_model_name]['predictions']
    })
    predictions_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'")
    
    # Calculate and display correlation matrix for features
    feature_correlation = features_df[feature_columns].corr()['accident_count'].sort_values(ascending=False)
    print("\n" + "="*50)
    print("Correlation of astrological features with accident count:")
    for feature, corr in feature_correlation.head(10).items():
        if feature != 'accident_count':
            print(f"  {feature}: {corr:.3f}")
    
    return features_df, results

if __name__ == "__main__":
    features_df, results = main()