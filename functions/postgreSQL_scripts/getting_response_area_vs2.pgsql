-- 1. Get all Module Instances for a specific Module
SELECT "ModuleInstance".*
FROM "ModuleInstance"
JOIN "Module" ON "ModuleInstance"."moduleId" = "Module".id
WHERE "Module"."name" = 'Module Name';

-- 2. Fetch all sets for a given Module Instance
SELECT "Set".*
FROM "Set"
JOIN "ModuleInstance" ON "Set"."moduleInstanceId" = "ModuleInstance".id
JOIN "Module" ON "ModuleInstance"."moduleId" = "Module".id
WHERE "Module"."name" = 'Module Name';

-- 3. Retrieve published Question IDs for a given Set
SELECT "Question".*
FROM "Question"
JOIN "QuestionVersion" ON "Question"."publishedVersionId" = "QuestionVersion".id
JOIN "Set" ON "QuestionVersion"."setId" = "Set".id
JOIN "ModuleInstance" ON "Set"."moduleInstanceId" = "ModuleInstance".id
JOIN "Module" ON "ModuleInstance"."moduleId" = "Module".id
WHERE "Module"."name" = 'Module Name';

-- 4. Retrieve details of Question Version using publishedVersionId
SELECT "QuestionVersion".*
FROM "QuestionVersion"
JOIN "Question" ON "QuestionVersion".id = "Question"."publishedVersionId"
WHERE "Question"."publishedVersionId" IS NOT NULL;

-- 5. Fetch all Parts associated with a Question Version
SELECT "Part".*
FROM "Part"
JOIN "QuestionVersion" ON "Part"."questionVersionId" = "QuestionVersion".id
JOIN "Question" ON "QuestionVersion".id = "Question"."publishedVersionId";

-- 6. Retrieve Response ID and Master Content
SELECT r.id AS "responseId", qv."masterContent"
FROM "ResponseArea" r
JOIN "Part" p ON r."partId" = p.id
JOIN "QuestionVersion" qv ON p."questionVersionId" = qv.id
JOIN "Question" ON qv.id = "Question"."publishedVersionId";
