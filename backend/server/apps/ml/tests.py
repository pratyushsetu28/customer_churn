from django.test import TestCase

from apps.ml.churn_classifier.random_forest import RandomForestClassifier
import inspect
from apps.ml.registry import MLRegistry

class MLTests(TestCase):
    def test_rf_algorithm(self):
   				
   			
   		
   			
   			
						
					
				
	

        input_data = {
            
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": 29.85
        }
        my_alg = RandomForestClassifier()
        response = my_alg.compute_prediction(input_data);
        self.assertEqual('OK', response['status'])
    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "income_classifier"
        algorithm_object = RandomForestClassifier()
        algorithm_name = "random forest"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Piotr"
        algorithm_description = "Random Forest with simple pre- and post-processing"
        algorithm_code = inspect.getsource(RandomForestClassifier)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)
    
