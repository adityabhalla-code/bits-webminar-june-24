
import pandas as pd
import json
import boto3
import pathlib
import io
import sagemaker
from time import gmtime, strftime, sleep
from sagemaker.deserializers import CSVDeserializer
from sagemaker.serializers import CSVSerializer

from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import (
    ProcessingInput, 
    ProcessingOutput, 
    ScriptProcessor
)
from sagemaker.inputs import TrainingInput

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep, 
    TrainingStep, 
    CreateModelStep,
    CacheConfig
)
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.parameters import (
    ParameterInteger, 
    ParameterFloat, 
    ParameterString, 
    ParameterBoolean
)
from sagemaker.workflow.clarify_check_step import (
    ModelBiasCheckConfig, 
    ClarifyCheckStep, 
    ModelExplainabilityCheckConfig
)
from sagemaker import Model
from sagemaker.inputs import CreateModelInput
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.conditions import (
    ConditionGreaterThan,
    ConditionGreaterThanOrEqualTo
)
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import (
    Join,
    JsonGet
)
from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)
from sagemaker.lambda_helper import Lambda

from sagemaker.model_metrics import (
    MetricsSource, 
    ModelMetrics, 
    FileSource
)
from sagemaker.drift_check_baselines import DriftCheckBaselines

from sagemaker.image_uris import retrieve
import os

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    sagemaker_project_id=None,
    role=None,
    default_bucket=None,
    input_data_url=None,
    bucket_prefix="siemens-poc/xgboost",
    model_package_group_name="siemens-poc-xgboost-model-group",
    pipeline_name="siemens-poc-pipeline",
    base_job_prefix="siemens-poc-pipeline",
    processing_instance_type="ml.t3.medium",
    training_instance_type="ml.m5.xlarge",
    test_score_threshold=0.75,
):
    """Gets a SageMaker ML Pipeline instance.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    session = get_pipeline_session(region, default_bucket)
    sm = session.sagemaker_client

    # Set S3 urls for processed data
    train_s3_url = f"s3://{default_bucket}/{bucket_prefix}/train"
    validation_s3_url = f"s3://{default_bucket}/{bucket_prefix}/validation"
    test_s3_url = f"s3://{default_bucket}/{bucket_prefix}/test"
    baseline_s3_url = f"s3://{default_bucket}/{bucket_prefix}/baseline"
    evaluation_s3_url = f"s3://{default_bucket}/{bucket_prefix}/evaluation"
    prediction_baseline_s3_url = f"s3://{default_bucket}/{bucket_prefix}/prediction_baseline"
    
    # Set S3 url for model artifact
    output_s3_url = f"s3://{default_bucket}/{bucket_prefix}/output"

    # Parameters for pipeline execution
    # Set processing instance type
    process_instance_type_param = ParameterString(
        name="ProcessingInstanceType",
        default_value=processing_instance_type,
    )

    # Set training instance type
    train_instance_type_param = ParameterString(
        name="TrainingInstanceType",
        default_value=training_instance_type,
    )

    # Set training instance count
    train_instance_count_param = ParameterInteger(
        name="TrainingInstanceCount",
        default_value=1
    )

    # Set model approval param
    model_approval_status_param = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    # Minimal threshold for model performance on the test dataset
    test_score_threshold_param = ParameterFloat(
        name="TestScoreThreshold", 
        default_value=100.0
    )

    # Set S3 url for input dataset
    input_s3_url_param = ParameterString(
        name="InputDataUrl",
        default_value=input_data_url,
    )
    
    # Define step cache config
    cache_config = CacheConfig(
        enable_caching=True,
        expire_after="P30d" # 30-day
    )
    
    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type=process_instance_type_param,
        instance_count=1,
        base_job_name=f"{pipeline_name}/preprocess",
        sagemaker_session=session,
    )
    
    processing_inputs=[
        ProcessingInput(source=input_s3_url_param, destination="/opt/ml/processing/input")
    ]

    processing_outputs=[
        ProcessingOutput(output_name="train_data", source="/opt/ml/processing/output/train", 
                         destination=train_s3_url),
        ProcessingOutput(output_name="validation_data", source="/opt/ml/processing/output/validation",
                         destination=validation_s3_url),
        ProcessingOutput(output_name="test_data", source="/opt/ml/processing/output/test",
                         destination=test_s3_url),
        ProcessingOutput(output_name="baseline_data", source="/opt/ml/processing/output/baseline", 
                         destination=baseline_s3_url),
    ]

    processor_args = sklearn_processor.run(
        inputs=processing_inputs,
        outputs=processing_outputs,
        code=os.path.join(BASE_DIR, "preprocessing.py"),
        # arguments = ['arg1', 'arg2'],
    )

    # Define processing step
    step_process = ProcessingStep(
        name=f"{pipeline_name}-preprocess-data",
        step_args=processor_args,
        cache_config = cache_config
    )

    # Training step for generating model artifacts
    xgboost_image_uri = sagemaker.image_uris.retrieve(
        "xgboost",
        region=region, 
        version="1.5-1")

    # Instantiate an XGBoost estimator object
    estimator = sagemaker.estimator.Estimator(
        image_uri=xgboost_image_uri,
        role=role, 
        instance_type=train_instance_type_param,
        instance_count=train_instance_count_param,
        output_path=output_s3_url,
        sagemaker_session=session,
        base_job_name=f"{pipeline_name}/train",
    )

    # Define algorithm hyperparameters
    estimator.set_hyperparameters(
        num_round=150, # the number of rounds to run the training
        max_depth=5, # maximum depth of a tree
        eta=0.5, # step size shrinkage used in updates to prevent overfitting
        alpha=2.5, # L1 regularization term on weights
        objective="reg:squarederror",
        eval_metric="rmse", # evaluation metrics for validation data
        subsample=0.8, # subsample ratio of the training instance
        colsample_bytree=0.8, # subsample ratio of columns when constructing each tree
        min_child_weight=3, # minimum sum of instance weight (hessian) needed in a child
        early_stopping_rounds=10, # the model trains until the validation score stops improving
        verbosity=1, # verbosity of printing messages
    )

    training_inputs = {
        "train": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "train_data"
            ].S3Output.S3Uri,
            content_type="text/csv",
        ),
        "validation": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "validation_data"
            ].S3Output.S3Uri,
            content_type="text/csv",
        ),
    }

    training_args = estimator.fit(training_inputs)

    # Define training step
    step_train = TrainingStep(
        name=f"{pipeline_name}-train",
        step_args=training_args,
        cache_config = cache_config
    )
    
    # Evaluation step
    script_processor = ScriptProcessor(
        image_uri=xgboost_image_uri,
        role=role,
        command=["python3"],
        instance_type=process_instance_type_param,
        instance_count=1,
        base_job_name=f"{pipeline_name}/evaluate",
        sagemaker_session=session,
    )

    eval_inputs=[
        ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts, 
                        destination="/opt/ml/processing/model"),
        ProcessingInput(source=step_process.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri, 
                        destination="/opt/ml/processing/test"),
    ]

    eval_outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation", 
                         destination=evaluation_s3_url),
        ProcessingOutput(output_name="prediction_baseline_data", source="/opt/ml/processing/output/prediction_baseline", 
                         destination=prediction_baseline_s3_url),
    ]

    eval_args = script_processor.run(
        inputs=eval_inputs,
        outputs=eval_outputs,
        code=os.path.join(BASE_DIR, "evaluation.py"),
    )

    evaluation_report = PropertyFile(
        name="ModelEvaluationReport", output_name="evaluation", path="evaluation.json"
    )

    step_eval = ProcessingStep(
        name=f"{pipeline_name}-evaluate-model",
        step_args=eval_args,
        property_files=[evaluation_report],
        cache_config = cache_config
    )
    
    # Define register step
    model = Model(
        image_uri=xgboost_image_uri,        
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=session,
        role=role,
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )

    register_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge", "ml.m5.large"],
        transform_instances=["ml.m5.xlarge", "ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status_param,
        model_metrics=model_metrics,
    )

    step_register = ModelStep(
        name=f"{pipeline_name}-register",
        step_args=register_args
    )

    # Fail step
    step_fail = FailStep(
        name=f"{pipeline_name}-fail",
        error_message=Join(on=" ", values=["Execution failed due to RMSE Score >", test_score_threshold_param]),
    )
    
    # Condition step
    cond_lte = ConditionGreaterThan(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metric.test_rmse.value",
        ),
        right=test_score_threshold_param,
    )

    step_cond = ConditionStep(
        name=f"{pipeline_name}-check-test-score",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[step_fail],
    )
    
    # Pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            process_instance_type_param,
            train_instance_type_param,
            train_instance_count_param,
            model_approval_status_param,
            test_score_threshold_param,
            input_s3_url_param,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=session,
    )
    
    return pipeline
