
SELECT * FROM public."Response"
ORDER BY "responseType" DESC
LIMIT 10;


            with "ModuleOfInterest" as (
            select * from "PartInfo1" 
            where "{moduleName}"=%s and "{moduleInstance}"=%s
            ),
            "StudentsOnly" as (
            select "id" from "User" 
            where "role"='STUDENT'
            ),
            "PartsAccessed" as (
            select * from "PartAccessEvent" 
            where "universalPartId" in (select "universalPartId" from "ModuleOfInterest")
            and "userId" in (select "id" from "StudentsOnly")
            ), 
            "data" as (
            select "PartsAccessed".*, "ModuleOfInterest".*
            from "PartsAccessed" 
            join "ModuleOfInterest" on "PartsAccessed"."universalPartId" = "ModuleOfInterest"."universalPartId"
            )
            select "createdAt", "userId", "moduleName", "setName", "setNumber", "questionNumber", "partPosition"  from "data" 
