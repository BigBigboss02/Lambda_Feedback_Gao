SELECT
  p."partContent",
  p."partAnswerContent",
  qv."masterContent"

FROM
  "ResponseArea" AS ra
JOIN
  "EvaluationFunction" AS ef ON ra."evaluationFunctionId" = ef.id
JOIN
  "Part" AS p ON ra."partId" = p.id
JOIN
  "QuestionVersion" AS qv ON p."questionVersionId" = qv.id
JOIN
  "Question" AS q ON qv."questionId" = q.id
WHERE
  ef."name" = 'shortTextAnswer'
  AND q."publishedVersionId" = qv.id;
