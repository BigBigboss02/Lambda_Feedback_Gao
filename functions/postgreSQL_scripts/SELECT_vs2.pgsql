SELECT
  "ResponseArea".*,
  "EvaluationFunction".*,
  "Part".*,
  "QuestionVersion".*,
  "QuestionAlgorithmFunction".*
FROM
  "ResponseArea",
  (
    SELECT * FROM "EvaluationFunction"
    WHERE "name" = 'shortTextAnswer'
  ) AS "EvaluationFunction",
  (
    SELECT * FROM "Part"
    WHERE "Part"."id" = "ResponseArea"."partId"
  ) AS "Part",
  (
    SELECT * FROM "QuestionVersion"
    WHERE "QuestionVersion"."id" = "Part"."questionVersionId"
  ) AS "QuestionVersion",
  (
    SELECT * FROM "Question"
    WHERE "Question"."publishedVersionId" = "QuestionVersion"."id"
  ) AS "Question",
  (
    SELECT * FROM "QuestionAlgorithmFunction"
    WHERE "QuestionAlgorithmFunction"."id" = "Part"."questionAlgorithmFunctionId"
  ) AS "QuestionAlgorithmFunction"
WHERE
  "ResponseArea"."evaluationFunctionId" = "EvaluationFunction".id
  AND "Part"."questionVersionId" = "QuestionVersion".id
  AND "QuestionVersion"."questionId" = "Question".id;
