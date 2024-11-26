-- 1. Get all Module Instances for a specific Module
SELECT *
FROM "ModuleInstance"
WHERE "moduleId" = (SELECT id FROM "Module" WHERE "name" = 'Module Name');

-- 2. Fetch all sets for a given Module Instance
SELECT *
FROM "Set"
WHERE "moduleInstanceId" IN (
    SELECT id FROM "ModuleInstance"
    WHERE "moduleId" = (SELECT id FROM "Module" WHERE "name" = 'Module Name')
);

-- 3. Retrieve published Question IDs for a given Set
SELECT *
FROM "Question"
WHERE "publishedVersionId" IN (
    SELECT id FROM "QuestionVersion"
    WHERE "setId" IN (
        SELECT id FROM "Set"
        WHERE "moduleInstanceId" IN (
            SELECT id FROM "ModuleInstance"
            WHERE "moduleId" = (SELECT id FROM "Module" WHERE "name" = 'Module Name')
        )
    )
);

-- 4. Retrieve details of Question Version using publishedVersionId
SELECT *
FROM "QuestionVersion"
WHERE "id" IN (
    SELECT "publishedVersionId"
    FROM "Question"
    WHERE "publishedVersionId" IS NOT NULL
);

-- 5. Fetch all Parts associated with a Question Version
SELECT *
FROM "Part"
WHERE "questionVersionId" IN (
    SELECT id FROM "QuestionVersion"
    WHERE "id" IN (
        SELECT "publishedVersionId"
        FROM "Question"
    )
);

-- 6. Retrieve Response ID and Master Content
SELECT r.id AS "responseId", qv."masterContent"
FROM "ResponseArea" r
JOIN "Part" p ON r."partId" = p.id
JOIN "QuestionVersion" qv ON p."questionVersionId" = qv.id
WHERE p."questionVersionId" IN (
    SELECT id FROM "QuestionVersion"
    WHERE "id" IN (
        SELECT "publishedVersionId"
        FROM "Question"
    )
);
