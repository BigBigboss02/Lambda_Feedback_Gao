SELECT id, "createdAt", "updatedAt", "hiddenAt", "isTemplate", 
"questionVersionId", "setId", number, "draftId", "publishedVersionId", "linkQuestionId", 
"displayFinalAnswer", "displayStructuredTutorial", "displayWorkedSolution"
FROM public."Question"
WHERE id = '9b5b1a69-381e-45d6-9da7-a47d172340ac';

SELECT id, "createdAt", "updatedAt", "deletedAt", "responseAreaId", "responseType", "config", "answer"
FROM public."Response"
WHERE id = '9b5b1a69-381e-45d6-9da7-a47d172340ac';
