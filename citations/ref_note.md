# Ref Note

Gao, Zhuangfei
Hi doc.johnson, can you have a look when u get some free time and make some suggestions for me[File: research_plan_suggestion.docx]
I think you need a problem definition before a plan. 
 
Define a function which has 
 
input: response <string>, answer <string>, (params <JSON>)
output: {is_correct <bool>, feedback <string>}
 
The response is the student response (one candidate, not a list), the answer is a list of acceptable answers, the params are optional - you may or may not need them. 
 
The function checks if the response is in the answer, and returns a Boolean and feedback in a string. 
 
Then define another higher level function that splits a student list into individual items, and calls the above function for each member of the list.
 
Work on the first function first. Create a list of tests it should pass, then create the tests, and show that the function fails. Then work on the function until it passes. Then move onto the higher level function and work in the same way. 
 
Then your plan is related to the above work.
 