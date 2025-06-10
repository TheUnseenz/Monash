/*****PLEASE ENTER YOUR DETAILS BELOW*****/
--T5-rm-select.sql

--Student ID:
--Student Name:


/* Comments for your marker:




*/


/* (a) */
-- PLEASE PLACE REQUIRED SQL SELECT STATEMENT FOR THIS PART HERE
-- ENSURE that your query is formatted and has a semicolon
-- (;) at the end of this answer
SELECT
    t.team_name AS "TEAM_NAME",
    TO_CHAR(c.carn_date, 'DD-Mon-YYYY') AS "CARNIVAL_DATE",
    comp_leader.comp_fname || ' ' || comp_leader.comp_lname AS "TEAMLEADER",
    (
        SELECT COUNT(e_member.entry_no)
        FROM ENTRY e_member
        WHERE e_member.team_id = t.team_id
        AND e_member.event_id IN ( -- Ensure team members are from the same carnival as the team itself
            SELECT event_id
            FROM EVENT
            WHERE carn_date = t.carn_date
        )
    ) AS "TEAM_NO_MEMBERS"
FROM
    TEAM t
JOIN
    CARNIVAL c ON t.carn_date = c.carn_date
JOIN
    ENTRY e_leader ON t.event_id = e_leader.event_id AND t.entry_no = e_leader.entry_no
JOIN
    COMPETITOR comp_leader ON e_leader.comp_no = comp_leader.comp_no
WHERE
    t.team_name IN (
        SELECT team_name
        FROM TEAM
        WHERE carn_date < TO_DATE('10-JUN-2025', 'DD-MON-YYYY') -- Completed carnivals
        GROUP BY team_name
        HAVING COUNT(*) = (
            SELECT MAX(name_count)
            FROM (
                SELECT COUNT(*) AS name_count
                FROM TEAM
                WHERE carn_date < TO_DATE('10-JUN-2025', 'DD-MON-YYYY') -- Completed carnivals
                GROUP BY team_name
            )
        )
    )
    AND c.carn_date < TO_DATE('10-JUN-2025', 'DD-MON-YYYY') -- Ensure only completed carnivals
ORDER BY
    t.team_name,
    c.carn_date;





/* (b) */
-- PLEASE PLACE REQUIRED SQL SELECT STATEMENT FOR THIS PART HERE
-- ENSURE that your query is formatted and has a semicolon
-- (;) at the end of this answer
SELECT
    et.eventtype_desc AS "EVENT_TYPE",
    c.carn_name || ', ' || TO_CHAR(c.carn_date, 'Day') || ' ' || TO_CHAR(c.carn_date, 'DD-Mon-YYYY') AS "CARNIVAL_DETAILS",
    TO_CHAR(e.entry_elapsedtime, 'HH24:MI:SS') AS "RECORD_TIME",
    comp.comp_no || ' - ' || comp.comp_fname || ' ' || comp.comp_lname AS "COMPETITOR_DETAILS",
    TRUNC(MONTHS_BETWEEN(c.carn_date, comp.comp_dob) / 12) AS "COMPETITOR_AGE"
FROM
    ENTRY e
JOIN
    EVENT ev ON e.event_id = ev.event_id
JOIN
    EVENTTYPE et ON ev.eventtype_code = et.eventtype_code
JOIN
    CARNIVAL c ON ev.carn_date = c.carn_date
JOIN
    COMPETITOR comp ON e.comp_no = comp.comp_no
WHERE
    e.entry_elapsedtime IS NOT NULL
    AND (et.eventtype_code, e.entry_elapsedtime) IN (
        SELECT
            ev2.eventtype_code,
            MIN(e2.entry_elapsedtime)
        FROM
            ENTRY e2
        JOIN
            EVENT ev2 ON e2.event_id = ev2.event_id
        WHERE
            e2.entry_elapsedtime IS NOT NULL
        GROUP BY
            ev2.eventtype_code
    )
ORDER BY
    et.eventtype_desc,
    comp.comp_no;





/* (c) */
-- PLEASE PLACE REQUIRED SQL SELECT STATEMENT FOR THIS PART HERE
-- ENSURE that your query is formatted and has a semicolon
-- (;) at the end of this answer
SELECT
    c.carn_name AS "CARNIVAL_NAME",
    TO_CHAR(c.carn_date, 'DD-Mon-YYYY') AS "CARNIVAL_DATE",
    et.eventtype_desc AS "EVENT_DESCRIPTION",
    CASE
        WHEN COUNT(e.entry_no) = 0 THEN RPAD('Not offered', 11, ' ') -- Left-aligned 'Not offered'
        ELSE TO_CHAR(COUNT(e.entry_no), 'FM999') -- Right-aligned number
    END AS "NUM_ENTRIES",
    CASE
        WHEN SUM(COUNT(e.entry_no)) OVER (PARTITION BY c.carn_date) = 0 THEN TO_CHAR(0, 'FM999')
        ELSE TO_CHAR(TRUNC(COUNT(e.entry_no) * 100 / SUM(COUNT(e.entry_no)) OVER (PARTITION BY c.carn_date)), 'FM999')
    END AS "PERCENTAGE_ENTRIES"
FROM
    CARNIVAL c
CROSS JOIN
    EVENTTYPE et
LEFT JOIN
    EVENT ev ON c.carn_date = ev.carn_date AND et.eventtype_code = ev.eventtype_code
LEFT JOIN
    ENTRY e ON ev.event_id = e.event_id
GROUP BY
    c.carn_name,
    c.carn_date,
    et.eventtype_desc
ORDER BY
    c.carn_date,
    COUNT(e.entry_no) DESC,
    et.eventtype_desc;

