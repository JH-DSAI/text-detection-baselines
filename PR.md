# PR: Azure ML Model Integration

## Plan

We have a text detection model hosted on Azure ML as a batch inference endpoint.  We want to expose this endpoint as a model that can be evaluated using this package.

## Resources

* ML endpoint invocation from app: https://github.com/JH-DSAI/text-detect-batch/blob/543c48480d292d2cc619a1c4d9da576233d42bfe/backend/app/services/azure_batch.py#L169
* Batch scoring function, including verdict creation: https://github.com/noa/dsai_detection/blob/9cb5069301dd6682d6ab2fb612b454896f7dfdcf/src/hopdetect/deploy/driver.py#L402
* SubmissionResult model definition: https://github.com/noa/dsai_detection/blob/9cb5069301dd6682d6ab2fb612b454896f7dfdcf/src/hopdetect/pipeline/data_model.py#L192
* SubmissionResult instantiation: https://github.com/noa/dsai_detection/blob/1c87c292c3f1c298d02def1ec622de30792204cc/src/hopdetect/deploy/score.py#L387
* ML result download from app: https://github.com/JH-DSAI/text-detect-batch/blob/543c48480d292d2cc619a1c4d9da576233d42bfe/backend/app/services/azure_batch.py#L272
