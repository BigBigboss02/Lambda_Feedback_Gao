SELECT
  *
FROM
  "ResponseArea"
JOIN
  "EvaluationFunction" ON "ResponseArea"."evaluationFunctionId" = "EvaluationFunction".id
JOIN
  "Part" ON "ResponseArea"."partId" = "Part".id
WHERE
  "EvaluationFunction".name = 'shortTextAnswer';

  -- add questionversion and questionAlgorithmFunctionId--from part to question version from question version to quetio from question find  the published version IDENTITY
  -- publsihed version ID is  the version ID in the response Area
  -- v from question, each question has a published version ID, to question version, search to the publishhed version IDENTITY
  -- no too much feedback now
  -- look short text answer

  -- question version to module to moduleinstance to set_bit--in module look for surg70005, 2024-25 for moduleinstannce,question1
  -- like surg70005
  -- feedback oriented, boolean validaty is the one
  -- use datq from database to fine promt lamma 3