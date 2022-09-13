import sagemaker
import boto3
from sagemaker.pytorch import PyTorch

# sagemaker_session = sagemaker.LocalSession()
# sagemaker_session.config = {'local': {'local_code': True}}

sagemaker_session = sagemaker.Session()

role = "arn:aws:iam::456181194192:role/service-role/AmazonSageMakerServiceCatalogProductsExecutionRole"

metric_definitions = [
    {"Name": "EncoderLoss", "Regex": "EncoderLoss.*:\D*(.*?)$"},
    {"Name": "ViewLoss", "Regex": "ViewLoss.*:\D*(.*?)$"},
]

estimator = PyTorch(entry_point='main.py',
                    # source_dir= r'C://Users/hello/Projects/viewmaker_physiological',
                    source_dir= '/home/hy29/rdf/viewmaker_physiological',
                    base_job_name = 'ViewMaker-ptbxl',
                    role=role,
                    py_version='py3',
                    framework_version='1.8.0',
                    instance_count=1,
                    instance_type='ml.p3.2xlarge',
                    metric_definitions=metric_definitions
                    )

estimator.fit({'training': 's3://compwell-databucket/processed_DA/ptbxl/100hz/'})