select
  *
from
  "ResponseArea","EvaluationFunction" 
WHERE "ResponseArea"."evaluationFunctionId"="EvaluationFunction".id 
AND   "EvaluationFunction".name = 'shortTextAnswer';