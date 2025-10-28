/*
Database Teaching Team
applied11_sql_advanced.sql

student id: 
student name:
last modified date:

*/
--1  Assuming that the student's name is unique, display Claudette Serman’s academic record. Include the unit code, unit name, year, semester, mark and explained_grade in the listing. The Explained Grade column must show Fail for N, Pass for P, Credit for C, Distinction for D and High Distinction for HD. Order by year, within the same year order the list by semester, and within the same semester order the list by the unit code.

SELECT
    unitcode,
    ofyear,
    ofsemester,
    enrolmark,
    CASE enrolgrade        
        WHEN 'N' THEN 'Fail'
        WHEN 'P' THEN 'Pass'
        WHEN 'C' THEN 'Credit'
        WHEN 'D' THEN 'Distinction'
        WHEN 'HD' THEN 'High Distinction'
        ELSE 'Unknown'
    END AS explained_grade
FROM
    uni.enrolment
WHERE
    stuid IN (
    SELECT
        stuid
    FROM
        uni.student
    WHERE
            stufname == "Claudette"
        AND
            stulname == "Serman")

--2 Find the number of scheduled classes assigned to each staff member for each semester in 2019. If the number of classes is 2 then this should be labelled as a correct load, more than 2 as an overload and less than 2 as an underload. Include the staff id, staff first name, staff last name, semester, number of scheduled classes and load in the listing. Sort the list by decreasing order of the number of scheduled classes and when the number of classes is the same, sort by the staff id then by the semester.



--3 Find the total number of prerequisite units for all units. Include in the list the unit code of units that do not have a prerequisite. Order the list in descending order of the number of prerequisite units and where several units have the same number of prerequisites order then by unit code.


--4

/* Using outer join */



/* Using set operator MINUS */



/* Using subquery */



--5



--6



--7



--8


    
--9


    
--10







