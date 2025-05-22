/*
Database Teaching Team
applied9_sql_basic.sql

student id: 27030768
student name: Adrian Leong Tat Wei
last modified date: 8/5/2025

*/

/* Part A - Retrieving data from a single table */

-- A1 List all units and their details. Order the output by unit code.
SELECT 
    *
FROM
    uni.unit
ORDER BY
    unitcode;

-- A2 List the full student details for those students who live in Caulfield. Order the output by student id.
-- SELECT

-- A3 List the full student details for those students who have a surname starting with the letter M. 
-- In the display, rename the columns stufname and stulname to firstname and lastname. Order the output by student id.


-- A4 List the student's id, surname, first name and address for those students who have a surname starting with the 
-- letter S and first name which contains the letter i. Order the output by student id.


-- A5 Assuming that a unit code is created based on the following rules:
    -- The first three letters represent the faculty abbreviation, eg. FIT for the Faculty of Information Technology.
    -- The first digit of the number following the letter represents the year level. For example, FIT2094 is a unit code 
    -- from the Faculty of IT (FIT) and the number 2 refers to a second year unit.
-- List the unit details for all first-year units in the Faculty of Information Technology. Order the output by unit code.


-- A6 List the unit code and semester of all units that are offered in 2021. Order the output by unit code, and within a 
-- given unit code order by semester. To complete this question, you need to use the Oracle function to_char to convert the 
-- data type for the year component of the offering date into text. For example, to_char(ofyear,'yyyy') - here we are only 
-- using the year part of the date.


-- A7 List the year and unit code for all units that were offered in either semester 2 of 2019 or semester 2 of 2020. 
-- Order the output by year and then by unit code. To display the offering year correctly in Oracle, you need to use the 
-- to_char function. For example, to_char(ofyear,'yyyy'). 


-- A8 List the student id, unit code and mark for those students who have failed any unit in semester 2 of 2021 
-- (a fail is where the mark < 50). Order the output by student id then order by unit code.


-- A9 List the student id for all students who have no mark and no grade in FIT3176 in semester 1 of 2020. Order the output by student id.



/* Part B - Retrieving data from multiple tables */

-- B1


-- B2


-- B3


-- B4


-- B5


-- B6


-- B7


-- B8


-- B9


-- B10
